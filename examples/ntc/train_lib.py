"""Runs an NTC training loop."""

import collections
import math
import os
from absl import logging
import equinox as eqx
import jax
import optax
import wandb
from tqdm import tqdm

# import ntc
from examples.ntc import ntc


@eqx.filter_jit
def evaluate(model, x):
    return model(x, None, None)


def save_state(path, model, epoch, opt_state, config):
    state = (model, epoch, opt_state)
    fn_state = f"{path}/state_{config.lmbda}.eqx"
    with open(fn_state, "wb") as f:
        eqx.tree_serialise_leaves(f, state)


def load_state(path, model, opt_state=None):
    if opt_state is None:
        state = (model, 0)
    else:
        state = (model, 0, opt_state)
    fn_state = f"{path}/state.eqx"
    with open(fn_state, "rb") as f:
        return eqx.tree_deserialise_leaves(f, state)


def instantiate_model(rng, config):
    cls = getattr(ntc, config.model_cls)
    kwargs = config.model_kwargs[config.model_cls]
    return cls(rng, **kwargs)


def checkify(fn):
    error_set = jax.experimental.checkify.all_checks
    error_set -= jax.experimental.checkify.div_checks
    checkified = jax.experimental.checkify.checkify(fn, errors=error_set)

    def new_fn(*args, **kwargs):
        err, result = checkified(*args, **kwargs)
        err.throw()
        return result

    return new_fn


def train(config, checkpoint_path, train_iterator, rng, start_path=None):
    """The main training loop."""
    if start_path is None:
        start_path = checkpoint_path

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=f"{config.model_cls}_lambda{config.lmbda}_lr{config.learning_rate}_adam",
        config={
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "patch_size": config.patch_size,
            "model": config.model_cls,
            "lambda": config.lmbda,
            "temperature": config.temperature,
            "optimizer": "Adam",
            "lr_schedule": "piecewise_constant",
        }
    )

    # Create temperature schedule based on dynamic_t configuration
    if hasattr(config, 'dynamic_t') and config.dynamic_t:
        # Dynamic temperature schedule: linear decrease from max_temp to min_temp over first 70 epochs
        # Using piecewise_interpolate_schedule for JAX-compatible conditional logic
        temperature_schedule = optax.schedules.piecewise_interpolate_schedule(
            init_value=config.max_temp,
            boundaries_and_scales={
                config.bound_epoch * config.num_steps_per_epoch: config.min_temp / config.max_temp,  # At bound_epoch, scale to min_temp
            },
            interpolate_type='linear'  # Linear interpolation between boundaries
        )
        logging.info(f"Using dynamic temperature schedule: {config.max_temp} -> {config.min_temp} over first {config.bound_epoch} epochs, then fixed at {config.min_temp}")
    else:
        # Fixed temperature schedule using config.temperature
        temperature_schedule = optax.schedules.piecewise_constant_schedule(config.temperature, {})
        logging.info(f"Using fixed temperature schedule: {config.temperature}")
    
    # Initialize optimizer with initial learning rate
    optimizer = optax.adam(learning_rate=config.learning_rate)

    os.makedirs(checkpoint_path, exist_ok=True)

    # Save configuration parameters to config.txt file
    config_file_path = os.path.join(checkpoint_path, "config.txt")
    with open(config_file_path, "w") as f:
        f.write("Configuration Parameters:\n")
        f.write("=" * 50 + "\n\n")

        # Save all config parameters except lmbda
        for key, value in config.items():
            if key != "lmbda":  # Exclude lmbda as requested
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for sub_key, sub_value in value.items():
                        f.write(f"  {sub_key}: {sub_value}\n")
                else:
                    f.write(f"{key}: {value}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Configuration saved at: {config_file_path}\n")

    logging.info(f"Configuration saved to: {config_file_path}")

    rng, init_rng = jax.random.split(rng)
    model = instantiate_model(init_rng, config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    try:
        model, start_epoch, opt_state = load_state(start_path, model, opt_state)
    except IOError:
        start_epoch = 0

    # Initialize learning rate reduction tracking
    lr_reduction_counter = 0
    previous_training_loss = None
    current_learning_rate = config.learning_rate
    
    # Initialize global step counter for temperature scheduling (not tied to optimizer state)
    global_step = 0
    
    # Initialize current temperature (will be updated each epoch)
    current_temperature = config.max_temp if hasattr(config, 'dynamic_t') and config.dynamic_t else config.temperature
    
    # Initialize best validation distortion tracking
    best_val_distortion = float('inf')

    @eqx.filter_jit
    def train_step(model, opt_state, x, rng, temperature):
        logging.info("Compiling train_step.")
        grad_fn = eqx.filter_grad(ntc.batched_loss_fn, has_aux=True)
        rng = jax.random.split(rng, x.shape[0])
        grads, metrics = grad_fn(model, x, config.lmbda, rng, temperature)
        update, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, update)
        metrics.update(lr=current_learning_rate, t=temperature)
        return model, opt_state, metrics

    @eqx.filter_jit
    def eval_step(model, x):
        logging.info("Compiling eval_step.")
        _, metrics = ntc.batched_loss_fn(model, x, config.lmbda, None, None)
        return {f"val_{k}": v for k, v in metrics.items()}

    if config.checkify:
        train_step = checkify(train_step)
        eval_step = checkify(eval_step)

    # Create progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, config.num_epochs),
                     desc="Epochs",
                     position=0,
                     initial=start_epoch)

    for i, epoch in enumerate(epoch_pbar):
        logging.info("Starting epoch %d.", epoch)

        # Update temperature for this epoch (only for dynamic temperature)
        if hasattr(config, 'dynamic_t') and config.dynamic_t:
            if epoch >= config.bound_epoch:
                current_temperature = config.min_temp
            else:
                # Calculate temperature based on current epoch progress
                epoch_progress = epoch / config.bound_epoch  # Progress from 0 to 1 over bound_epoch epochs
                current_temperature = config.max_temp - (config.max_temp - config.min_temp) * epoch_progress

        # Log current learning rate and temperature at the start of epoch
        logging.info(f"Epoch {epoch} - Current learning rate: {current_learning_rate:.2e}, temperature: {current_temperature:.4f}")

        metrics = collections.defaultdict(lambda: 0.0)
        step_metrics = dict()
        
        # Create progress bar for training steps
        train_pbar = tqdm(range(config.num_steps_per_epoch), desc="Training", position=1, leave=False)
        for step in train_pbar:
            rng, train_rng = jax.random.split(rng)
            model, opt_state, step_metrics = train_step(
                model, opt_state, next(train_iterator), train_rng, current_temperature
            )
            for k in step_metrics:
                metrics[k] += float(step_metrics[k])
            # Update training progress bar with current loss, learning rate, and temperature
            train_pbar.set_postfix({"loss": float(step_metrics["loss"]), "lr": f"{current_learning_rate:.2e}", "temp": f"{current_temperature:.3f}"})
            # Debug: Check if step_metrics["t"] matches current_temperature
            if step == 0:  # Only log once per epoch to avoid spam
                logging.info(f"Debug - current_temperature: {current_temperature:.4f}, step_metrics['t']: {float(step_metrics['t']):.4f}")
            global_step += 1  # Increment global step counter
        for k in step_metrics:
            metrics[k] /= config.num_steps_per_epoch

        # Create progress bar for evaluation steps
        eval_pbar = tqdm(range(config.num_eval_steps), desc="Evaluation", position=1, leave=False)
        for _ in eval_pbar:
            step_metrics = eval_step(model, next(train_iterator))
            for k in step_metrics:
                metrics[k] += float(step_metrics[k])
            # Update evaluation progress bar with current validation loss
            eval_pbar.set_postfix({"val_loss": float(step_metrics["val_loss"])})
        for k in step_metrics:
            metrics[k] /= config.num_eval_steps

        # Learning rate reduction logic (after bound_epoch)
        if epoch >= config.bound_epoch:
            current_training_loss = metrics["loss"]
            
            if previous_training_loss is not None:
                loss_difference = abs(current_training_loss - previous_training_loss)
                
                if loss_difference < 0.07:
                    lr_reduction_counter += 1
                    logging.info(f"Epoch {epoch}: Loss difference {loss_difference:.6f} < 0.01. Counter: {lr_reduction_counter}/10")
                else:
                    lr_reduction_counter = 0
                    logging.info(f"Epoch {epoch}: Loss difference {loss_difference:.6f} >= 0.01. Resetting counter.")
                
                # Reduce learning rate if counter reaches 10
                if lr_reduction_counter >= 30:
                    new_lr = current_learning_rate / 10.0
                    if new_lr >= 1e-7:  # Check minimum learning rate
                        current_learning_rate = new_lr
                        # Recreate optimizer with new learning rate
                        optimizer = optax.adam(learning_rate=current_learning_rate)
                        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
                        logging.info(f"Epoch {epoch}: Reducing learning rate to {current_learning_rate:.2e} (global_step: {global_step})")
                    else:
                        logging.info(f"Epoch {epoch}: Learning rate would be {new_lr:.2e} < 1e-7, keeping at {current_learning_rate:.2e}")
                    lr_reduction_counter = 0  # Reset counter
            
            previous_training_loss = current_training_loss

        # Check if current validation distortion is the best so far
        current_val_distortion = metrics["val_distortion"]

        # Check if we have a new best validation distortion
        if current_val_distortion < best_val_distortion: # TODO use loss compare
            # We have a new best, so save the checkpoint
            logging.info(f"New best validation distortion: {current_val_distortion:.6f} < {best_val_distortion:.6f}")
            save_state(checkpoint_path, model, epoch, opt_state, config)
            best_val_distortion = current_val_distortion
        else:
            logging.info(f"No improvement: val_distortion {current_val_distortion:.6f} >= best_val_distortion {best_val_distortion:.6f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "loss": metrics["loss"],
            "rate": metrics["rate"],
            "distortion": metrics["distortion"],
            "learning_rate": current_learning_rate,
            "temperature": metrics["t"],
            "val_loss": metrics["val_loss"],
            "val_rate": metrics["val_rate"],
            "val_distortion": metrics["val_distortion"],
            "best_val_distortion": best_val_distortion,
        })

        # For HyperPriorModel, log additional metrics if they exist
        if "rate_y" in metrics:
            wandb.log({
                "rate_y": metrics["rate_y"],
                "rate_z": metrics["rate_z"],
                "val_rate_y": metrics["val_rate_y"],
                "val_rate_z": metrics["val_rate_z"],
            })

        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix({
            "loss": metrics["loss"],
            "val_loss": metrics["val_loss"],
            "rate": metrics["rate"],
            "distortion": metrics["distortion"],
            "lr": f"{current_learning_rate:.2e}",
            "temp": f"{current_temperature:.3f}"
        })

        # Print metrics on separate lines
        logging.info("Epoch %d metrics:", epoch)
        for metric_name, metric_value in metrics.items():
            logging.info("  %s: %f", metric_name, metric_value)

        nan_metrics = [k for k, v in metrics.items() if math.isnan(v)]
        if nan_metrics:
            raise RuntimeError(
                f"Encountered NaN in metrics: {nan_metrics}. Stopping training."
            )

    # Save final checkpoint
    save_state(checkpoint_path, model, config.num_epochs, opt_state)
    wandb.finish()

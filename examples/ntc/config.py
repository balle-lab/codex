"""Base configuration for NTC experiments."""

import ml_collections


def get_config():
    """Base configuration."""
    return ml_collections.ConfigDict(dict(
        label="base configuration",
        run=0,
        wandb_project="Image Compression",  # Wandb project name

        debug_nans=False,
        checkify=False,

        lmbda=8.0,
        log_sigma=4.0,
        learning_rate=1e-4,
        temperature=float("inf"),  # Initial temperature for dynamic schedule
        max_temp=1.0,  # Maximum temperature for dynamic schedule
        min_temp=0.2,  # Minimum temperature for dynamic schedule
        bound_epoch=200,  # Epoch boundary for temperature and learning rate reduction
        dynamic_t=False,  # Whether to use dynamic temperature schedule

        num_epochs=1000,
        num_steps_per_epoch=1000,
        num_eval_steps=100,
        patch_size=384,
        batch_size=4,
        shuffle_size=128,

        model_cls="FactorizedPriorModel",
        model_kwargs=dict(
            FactorizedPriorModel=dict(
                x_channels=3,
                y_channels=192,
                em_y="fourier",
            ),
            HyperPriorModel=dict(
                x_channels=3,
                y_channels=192,
                z_channels=128,
                em_z="fourier",
            ),
        ),
    ))

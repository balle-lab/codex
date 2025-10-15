"""Runs an NTC training loop."""

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)


from examples.ntc import train_lib
from codex.loss import pretrained_features

# pyright: reportUnusedCallResult=false
config_flags.DEFINE_config_file("config")
flags.DEFINE_string(
    "checkpoint_path",
    "./run/train/",
    "Directory where to write checkpoints.",
)
flags.DEFINE_string(
    "start_path",
    None,
    "Directory to read initial checkpoint from.",
)
flags.DEFINE_float(
    "lmbda",
    None,
    "Override lambda value from config. If not set, uses config.lmbda.",
)
flags.DEFINE_float(
    "lr",
    None,
    "Override learning rate from config. If not set, uses config.learning_rate.",
)
flags.DEFINE_integer(
    "batch_size",
    None,
    "Override batch size from config. If not set, uses config.batch_size.",
)
flags.DEFINE_float(
    "temperature",
    None,
    "Override temperature from config. If not set, uses config.temperature.",
)
flags.DEFINE_integer(
    "y_channels",
    None,
    "Override y_channels from config. If not set, uses config.model_kwargs[model_cls].y_channels.",
)
flags.DEFINE_boolean(
    "dynamic_t",
    None,
    "Override dynamic_t from config. If not set, uses config.dynamic_t.",
)
flags.DEFINE_string(
    "wandb_project",
    None,
    "Override wandb_project from config. If not set, uses config.wandb_project.",
)
flags.DEFINE_float(
    "max_temp",
    None,
    "Override max_temp from config. If not set, uses config.max_temp.",
)
flags.DEFINE_float(
    "min_temp",
    None,
    "Override min_temp from config. If not set, uses config.min_temp.",
)
flags.DEFINE_integer(
    "bound_epoch",
    None,
    "Override bound_epoch from config. If not set, uses config.bound_epoch.",
)

FLAGS = flags.FLAGS


def load_training_set(patch_size, batch_size, shuffle_size):
    """Returns a tf.Dataset with training images."""

    def image_filter(item):
        shape = tf.shape(item["image"])
        return (shape[0] >= patch_size) and (shape[1] >= patch_size) and (shape[2] == 3)

    def image_preprocess(item):
        """Preprocesses an image from the CLIC dataset."""
        image = item["image"]
        shape = tf.cast(tf.shape(image), dtype=tf.float32)
        min_factor = float(patch_size) / tf.math.minimum(shape[0], shape[1])
        scale_factor = tf.random.uniform((), minval=min_factor, maxval=1.0)
        shape = scale_factor * shape[:2]
        shape = tf.math.minimum(tf.cast(tf.round(shape), tf.int32), patch_size)
        image = tf.image.resize(image, shape, method="bilinear", antialias=True)
        image = tf.image.random_crop(image, (patch_size, patch_size, 3))
        image = tf.transpose(image, (2, 0, 1)) / 255
        return image

    ds = tfds.load("clic", split="train", shuffle_files=True)
    ds = ds.repeat()
    ds = ds.filter(image_filter)
    ds = ds.map(image_preprocess)
    ds = ds.shuffle(shuffle_size)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(2)
    return ds


def main(_):
    tf.config.experimental.set_visible_devices([], "GPU")
    logging.info(
        "JAX devices: %s, TF devices: %s",
        jax.devices(),
        tf.config.get_visible_devices(),
    )

    jax.config.update("jax_debug_nans", FLAGS.config.debug_nans)

    # Override lambda value if provided via command line
    if FLAGS.lmbda is not None:
        original_lmbda = FLAGS.config.lmbda
        FLAGS.config.lmbda = FLAGS.lmbda
        logging.info(f"Overriding lambda: {original_lmbda} -> {FLAGS.lmbda}")
    else:
        logging.info(f"Using lambda from config: {FLAGS.config.lmbda}")

    # Override learning rate if provided via command line
    if FLAGS.lr is not None:
        original_lr = FLAGS.config.learning_rate
        FLAGS.config.learning_rate = FLAGS.lr
        logging.info(f"Overriding learning rate: {original_lr} -> {FLAGS.lr}")
    else:
        logging.info(f"Using learning rate from config: {FLAGS.config.learning_rate}")
        
    if FLAGS.batch_size is not None:
        original_batch_size = FLAGS.config.batch_size
        FLAGS.config.batch_size = FLAGS.batch_size
        logging.info(f"Overriding batch size: {original_batch_size} -> {FLAGS.batch_size}")
    else:
        logging.info(f"Using batch size from config: {FLAGS.config.batch_size}")
        
    if FLAGS.temperature is not None:
        original_temperature = FLAGS.config.temperature
        FLAGS.config.temperature = FLAGS.temperature
        logging.info(f"Overriding temperature: {original_temperature} -> {FLAGS.temperature}")
    else:
        logging.info(f"Using temperature from config: {FLAGS.config.temperature}")
        
    if FLAGS.y_channels is not None:
        model_cls = FLAGS.config.model_cls
        original_y_channels = FLAGS.config.model_kwargs[model_cls]["y_channels"]
        FLAGS.config.model_kwargs[model_cls]["y_channels"] = FLAGS.y_channels
        logging.info(f"Overriding y_channels for {model_cls}: {original_y_channels} -> {FLAGS.y_channels}")
    else:
        model_cls = FLAGS.config.model_cls
        logging.info(f"Using y_channels from config for {model_cls}: {FLAGS.config.model_kwargs[model_cls]['y_channels']}")

    # Override dynamic_t if provided via command line
    if FLAGS.dynamic_t is not None:
        original_dynamic_t = FLAGS.config.dynamic_t
        FLAGS.config.dynamic_t = FLAGS.dynamic_t
        logging.info(f"Overriding dynamic_t: {original_dynamic_t} -> {FLAGS.dynamic_t}")
    else:
        logging.info(f"Using dynamic_t from config: {FLAGS.config.dynamic_t}")

    # Override wandb_project if provided via command line
    if FLAGS.wandb_project is not None:
        original_wandb_project = FLAGS.config.wandb_project
        FLAGS.config.wandb_project = FLAGS.wandb_project
        logging.info(f"Overriding wandb_project: {original_wandb_project} -> {FLAGS.wandb_project}")
    else:
        logging.info(f"Using wandb_project from config: {FLAGS.config.wandb_project}")

    # Override max_temp if provided via command line
    if FLAGS.max_temp is not None:
        original_max_temp = FLAGS.config.max_temp
        FLAGS.config.max_temp = FLAGS.max_temp
        logging.info(f"Overriding max_temp: {original_max_temp} -> {FLAGS.max_temp}")
    else:
        logging.info(f"Using max_temp from config: {FLAGS.config.max_temp}")

    # Override min_temp if provided via command line
    if FLAGS.min_temp is not None:
        original_min_temp = FLAGS.config.min_temp
        FLAGS.config.min_temp = FLAGS.min_temp
        logging.info(f"Overriding min_temp: {original_min_temp} -> {FLAGS.min_temp}")
    else:
        logging.info(f"Using min_temp from config: {FLAGS.config.min_temp}")

    # Override bound_epoch if provided via command line
    if FLAGS.bound_epoch is not None:
        original_bound_epoch = FLAGS.config.bound_epoch
        FLAGS.config.bound_epoch = FLAGS.bound_epoch
        logging.info(f"Overriding bound_epoch: {original_bound_epoch} -> {FLAGS.bound_epoch}")
    else:
        logging.info(f"Using bound_epoch from config: {FLAGS.config.bound_epoch}")


    host_count = jax.process_count()
    # host_id = jax.process_index()
    local_device_count = jax.local_device_count()
    logging.info(
        "Device count: %d, host count: %d, local device count: %d",
        jax.device_count(),
        host_count,
        local_device_count,
    )

    (seed,) = np.frombuffer(os.getrandom(8), dtype=np.int64)  # pylint: disable=no-member
    rng = jax.random.key(seed)

    train_set = load_training_set(
        FLAGS.config.patch_size, FLAGS.config.batch_size, FLAGS.config.shuffle_size
    )
    train_iterator = train_set.as_numpy_iterator()

    # Load VGG16 model for Wasserstein distortion calculation
    logging.info("Loading VGG16 model for perceptual loss...")
    pretrained_features.load_vgg16_model()
    logging.info("VGG16 model loaded successfully.")

    train_lib.train(
        FLAGS.config,
        FLAGS.checkpoint_path,
        train_iterator,
        rng,
        start_path=FLAGS.start_path,
    )


if __name__ == "__main__":
    app.run(main)

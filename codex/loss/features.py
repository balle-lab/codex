from PIL import Image
import jax
import jax.numpy as jnp
import flaxmodels as fm


def load_vgg_model():
    """Loads VGG16 model with pretrained ImageNet weights."""
    vgg = fm.VGG16(output='activations', pretrained='imagenet')
    init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
    dummy_input = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
    params = vgg.init(init_rngs, dummy_input)
    return vgg, params


def preprocess_image(image_path):
    """Loads and preprocesses image to (1, 224, 224, 3) JAX array in [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    x = jnp.array(img, dtype=jnp.float32) / 255.0
    x = jnp.expand_dims(x, axis=0)  # Add batch dimension
    return x


def extract_vgg_features(model, params, x):
    """Extracts VGG features from image tensor x."""
    activations = model.apply(params, x, train=False)
    selected_keys = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
    return [activations[key].squeeze(0) for key in selected_keys]
  
import requests
import json
from PIL import Image
import jax
import jax.numpy as jnp
import flaxmodels as fm

def vgg16(image):
  """
  Extract specific layer's feature maps of an image passing VGG
  """
  # Load image
  img = Image.open(image)
  # display(img.resize((480, 360)))

  # Image must be 224x224 if classification head is included
  img = img.resize((224, 224))
  # Image should be in range [0, 1]
  x = jnp.array(img, dtype=jnp.float32) / 255.0
  # Add batch dimension
  x = jnp.expand_dims(x, axis=0)

  vgg16 = fm.VGG16(output='activations', pretrained='imagenet')
  init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)}
  params = vgg16.init(init_rngs, x)
  out = vgg16.apply(params, x, train=False)

  selected_keys = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]

  features = [out[key].squeeze(0) for key in selected_keys]

  # for idx, f in enumerate(features):
  #     print(f"Feature {selected_keys[idx]} shape: {f.shape}")

  return features

if __name__ == '__main__':
    vgg16("eleph1.jpg")

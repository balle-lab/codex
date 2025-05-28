import jax
import jax.numpy as jnp
import numpy as np
import pytest
from codex.loss import wasserstein
import math


def test_lowpass_behavior():
  """
  Target:
  To verify that lowpass() correctly performs spatial downsampling
  without introducing invalid values.

  It checks that:
  The spatial resolution is halved when using stride=2.
  The value range stays between 0 and 1, the same as the inp
  """
  # Create random noise [0, 1]
  noise = jnp.array(np.random.rand(1, 64, 64), dtype=jnp.float32)
  # Apply lowpass with stride=2
  filtered = wasserstein.lowpass(noise, stride=2)

  # Check shape halved
  assert filtered.shape == (1, 32, 32), f"Expected shape (1, 32, 32), got {filtered.shape}"

  # Check values still in [0, 1]
  assert jnp.all(filtered >= 0) and jnp.all(filtered <= 1), \
    "Filtered values out of [0, 1] range"


def test_compute_multiscale_stats_behavior():
  """
  Target:
  To verify that compute_multiscale_stats():
  Correctly generates a pyramid of decreasing resolution.
  Preserves spatial uniformity (low variance) when given a constant input.

  It checks that:
  Variance is near-zero at all pyramid levels.
  The output stays spatially uniform, regardless of any global mean shift.
  The pyramid resolution halves at each level as expected.
  """
  # Constant input
  constant_value = 5.0
  constant_array = jnp.full((1, 64, 64), constant_value, dtype=jnp.float32)

  means, variances = wasserstein.compute_multiscale_stats(constant_array, num_levels=3)

  for i, (ms, vs) in enumerate(zip(means, variances)):
    # Check shape is halved at each level
    expected_shape = (1, 64 // (2 ** i), 64 // (2 ** i))
    assert ms.shape == expected_shape, f"Expected shape {expected_shape}, got {m.shape}"

    # Variance should be zero
    for i in range(1, len(vs)-1):
      for j in range(1, len(vs[0]-1)):
        assert jnp.allclose(vs[i][j], 0, atol=1e-6), f"Variance at level {i} not zero"

    # Mean should equal the constant value with tolerance level
    for i in range(1, len(ms)-1):
      for j in range(1, len(ms[0]-1)):
        assert jnp.allclose(ms[i][j], constant_value, atol=1e-6), \
          f"Mean at level {i} not equal to {constant_value}"


def test_wasserstein_distortion_behavior():
  """
  Target:
  To validate the correct behavior of the wasserstein_distortion() function
  when comparing single feature arrays.

  It checks that:
  The distance is zero when comparing identical features.
  The distance is positive when comparing different features.
  The function raises an error if the feature shapes don't match.
  """
  # Create two identical feature arrays
  feature_a = jnp.ones((1, 32, 32))
  feature_b = jnp.ones((1, 32, 32))

  log2_sigma = jnp.zeros((32, 32))

  # Should return 0 for identical features
  result = wasserstein.wasserstein_distortion(feature_a, feature_b, log2_sigma)
  assert jnp.isclose(result, 0.0), f"Expected 0.0, got {result}"

  # Create different features
  feature_b_different = jnp.ones((1, 32, 32)) * 2.0

  # Should return positive value
  result = wasserstein.wasserstein_distortion(feature_a, feature_b_different, log2_sigma)
  assert result > 0, f"Expected positive value, got {result}"

  # Should raise ValueError for shape mismatch
  with pytest.raises(ValueError):
      wasserstein.wasserstein_distortion(feature_a, jnp.ones((1, 16, 16)), log2_sigma)


def test_multi_wasserstein_distortion_behavior():
  """
  Target:
  To validate the correct behavior of the multi_wasserstein_distortion() function
  when comparing lists of feature arrays.

  It checks that:
  The distance is non-negative (ideally zero) when features are identical.
  The distance is positive when features differ.
  The function raises an error if:
    The list lengths differ.
    Any feature shape in the lists doesn't match.
  """
  # Matching feature lists
  features_a = [jnp.ones((1, 32, 32)), jnp.ones((1, 16, 16))]
  features_b_same = [jnp.ones((1, 32, 32)), jnp.ones((1, 16, 16))]

  log2_sigma = jnp.zeros((64, 64))

  # Should return 0 for identical lists
  result = wasserstein.multi_wasserstein_distortion(features_a, features_b_same, log2_sigma)
  assert result >= 0, f"Expected non-negative value, got {result}"

  # Different features
  features_b_different = [jnp.ones((1, 32, 32)) * 2.0, jnp.ones((1, 16, 16)) * 2.0]
  result = wasserstein.multi_wasserstein_distortion(features_a, features_b_different, log2_sigma)
  assert result > 0, f"Expected positive value, got {result}"

  # Should raise ValueError for list length mismatch
  with pytest.raises(ValueError):
      wasserstein.multi_wasserstein_distortion(features_a, [jnp.ones((1, 32, 32))], log2_sigma)

  # Should raise ValueError for shape mismatch in one of the pairs
  with pytest.raises(ValueError):
      wasserstein.multi_wasserstein_distortion(features_a, [jnp.ones((1, 32, 32)), \
        jnp.ones((1, 8, 8))], log2_sigma)


def test_compute_multiscale_stats_non_constant():
  """
  Target:
  Verify variance computation on non-uniform input (checkerboard pattern).
  This is a high-variance pattern with sharp spatial contrast, perfect for testing
  whether compute_multiscale_stats detects local variance.

  It checks that:
  Variance Positivity: Ensures variance is non-negative at all pyramid levels.
  Shape Halving: Validates spatial resolution reduction at each level
  (e.g., 64x64 → 32x32 → 16x16).
  """
  # Create a checkerboard pattern
  input_arr = jnp.zeros((1, 64, 64))
  input_arr = input_arr.at[:, ::2, ::2].set(1.0)
  input_arr = input_arr.at[:, 1::2, 1::2].set(1.0)

  means, variances = wasserstein.compute_multiscale_stats(input_arr, num_levels=3)

  for level in range(3):
      # Check variance is positive and reasonable
      assert jnp.all(variances[level] >= -1e-5), f"Negative variance at level {level}"
      # Check shape halving
      expected_shape = (1, 64 // (2 ** level), 64 // (2 ** level))
      assert means[level].shape == expected_shape, f"Shape mismatch at level {level}"


def test_wasserstein_distortion_intermediates():
  """
  Target:
  Test if intermediate results are returned when return_intermediates=True.

  It checks that:
  Dictionary Structure: Ensures intermediates include wd_maps (Wasserstein distortion maps).
  Distortion Value: Checks that identical features yield zero distortion.
  """
  features = jnp.ones((1, 32, 32))
  log2_sigma = jnp.zeros((32, 32))

  dist, intermediates = wasserstein.wasserstein_distortion(
      features, features, log2_sigma, return_intermediates=True
  )

  assert isinstance(intermediates, dict), "Intermediates should be a dict"
  assert "wd_maps" in intermediates, "Missing wd_maps in intermediates"
  assert len(intermediates["wd_maps"]) == 5 + 1, "Incorrect number of wd_maps"
  assert jnp.isclose(dist, 0.0), "Distortion should be zero for identical features"


# def test_wasserstein_distortion_jit_compilable():
#     """
#     Target:
#     Verify that wasserstein_distortion can be compiled with jax.jit
#     without errors due to jnp.max(log2_sigma) check.
#     """
#     features = jnp.ones((1, 32, 32))
#     log2_sigma = jnp.zeros((32, 32))  # max = 0, should not raise ValueError

#     # JIT compile
#     compiled_fn = jax.jit(wasserstein_distortion)

#     # Should not raise any error
#     result = compiled_fn(features, features, log2_sigma, num_levels=3)
#     assert jnp.isclose(result, 0.0), "Expected zero distortion for identical features"


# def test_num_levels_handling():
#   """
#   Target:
#   Validate error handling for insufficient num_levels.

#   It checks that:
#   Valid Case: num_levels=3 works when log2_sigma has a max value of 2.
#   Invalid Case: num_levels=1 triggers an error when log2_sigma exceeds it.
#   """
#   features = jnp.ones((1, 32, 32))
#   log2_sigma = jnp.ones((32, 32)) * 2  # Max level 2

#   # num_levels=3 should handle this without error
#   dist = wasserstein.wasserstein_distortion(features, features, log2_sigma, num_levels=3)
#   assert jnp.isclose(dist, 0.0), "Distortion should be zero"

#   # Check error if num_levels < max(log2_sigma)
#   with pytest.raises(ValueError):
#       wasserstein.wasserstein_distortion(features, features, log2_sigma, num_levels=1)


def test_multi_wasserstein_sigma_scaling(monkeypatch):
    """
    Target:
    Validate sigma map scaling for feature arrays with different resolutions.

    It checks that:
    Scaling Logic: If a feature array is smaller (e.g., 32x32 vs. original 64x64 sigma map),
    log2_sigma is adjusted by subtracting log2(size_ratio).
    """

    # Original sigma map (64x64)
    log2_sigma = jnp.full((64, 64), 4.0)
    # Feature with smaller spatial dim (32x32)
    features = [jnp.ones((1, 32, 32))]
    # Expected adjusted sigma: 4 - log2(64/32) = 3
    expected_ls = jnp.full((32, 32), 3.0)

    def check_sigma(fa, fb, ls, **kwargs):
      assert jnp.allclose(ls, expected_ls, atol=0.1), "Sigma not scaled correctly"
      return jnp.zeros(()), {"dummy": []}  # Return a dummy intermediates dict


    # Monkeypatch to intercept call to wasserstein_distortion
    monkeypatch.setattr(wasserstein, "wasserstein_distortion", check_sigma)

    # Run function under test
    wasserstein.multi_wasserstein_distortion(features, features, log2_sigma, return_intermediates=False)

    
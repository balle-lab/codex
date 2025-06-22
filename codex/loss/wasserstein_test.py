# Copyright 2025 CoDeX authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language governing
# permissions and limitations under the License.
# ========================================================================================
"""Tests of Wasserstein Distortion metric."""

import jax
import jax.numpy as jnp
import pytest
from codex.loss import pretrained_features
from codex.loss import wasserstein


def test_lowpass_behavior():
    """Verifies that lowpass() correctly performs spatial downsampling.

    Checks
    ------
    - The spatial resolution is halved when using stride=2.
    - The value range stays between 0 and 1, the same as the input.
    """
    # Create random noise [0, 1]
    noise = jax.random.uniform(jax.random.key(0), (1, 64, 64))
    # Apply lowpass with stride=2
    filtered = wasserstein.lowpass(noise, stride=2)

    # Check shape halved
    assert filtered.shape == (1, 32, 32)

    # Check values still in [0, 1]
    assert jnp.all(filtered >= 0) and jnp.all(filtered <= 1)


def test_compute_multiscale_stats_behavior():
    """Verifies that compute_multiscale_stats() behaves correctly.

    Verifies that compute_multiscale_stats(): Correctly generates a pyramid of decreasing
    resolution. Preserves spatial uniformity (low variance) when given a constant input.

    Checks
    ------
    - Variance is near-zero at all pyramid levels.
    - The output stays spatially uniform, regardless of any global mean shift.
    - The pyramid resolution halves at each level as expected.
    """
    # Constant input
    constant_value = 5.0
    constant_array = jnp.full((1, 64, 64), constant_value, dtype=jnp.float32)

    means, variances = wasserstein.compute_multiscale_stats(constant_array, num_levels=3)

    for i, (ms, vs) in enumerate(zip(means, variances)):
        # Check shape is halved at each level
        expected_shape = (1, 64 // (2**i), 64 // (2**i))
        assert ms.shape == expected_shape

        # Variance should be zero
        for i in range(1, len(vs) - 1):
            for j in range(1, len(vs[0] - 1)):
                assert jnp.allclose(vs[i][j], 0, atol=1e-6)

        # Mean should equal the constant value with tolerance level
        for i in range(1, len(ms) - 1):
            for j in range(1, len(ms[0] - 1)):
                assert jnp.allclose(ms[i][j], constant_value, atol=1e-6)


def test_wasserstein_distortion_behavior():
    """Verifies the correct behavior of wasserstein_distortion().

    Checks
    ------
    - The distance is zero when comparing identical features.
    - The distance is positive when comparing different features.
    - The function raises an error if the feature shapes don't match.
    """
    # Create two identical feature arrays
    feature_a = jnp.ones((1, 32, 32))
    feature_b = jnp.ones((1, 32, 32))

    log2_sigma = jnp.zeros((32, 32))

    # Should return 0 for identical features
    result = wasserstein.wasserstein_distortion(feature_a, feature_b, log2_sigma)
    assert jnp.isclose(result, 0.0)

    # Create different features
    feature_b_different = jnp.ones((1, 32, 32)) * 2.0

    # Should return positive value
    result = wasserstein.wasserstein_distortion(
        feature_a, feature_b_different, log2_sigma
    )
    assert result > 0


def test_multi_wasserstein_distortion_behavior():
    """Verifies the correct behavior of the multi_wasserstein_distortion().

    Checks
    ------
    - The distance is non-negative (ideally zero) when features are identical.
    - The distance is positive when features differ.
    - The function raises an error if:
        - The list lengths differ.
        - Any feature shape in the lists doesn't match.
    """
    # Matching feature lists
    features_a = [jnp.ones((1, 32, 32)), jnp.ones((1, 16, 16))]
    features_b_same = [jnp.ones((1, 32, 32)), jnp.ones((1, 16, 16))]

    log2_sigma = jnp.zeros((64, 64))

    # Should return 0 for identical lists
    result = wasserstein.multi_wasserstein_distortion(
        features_a, features_b_same, log2_sigma
    )
    assert result >= 0

    # Different features
    features_b_different = [jnp.ones((1, 32, 32)) * 2.0, jnp.ones((1, 16, 16)) * 2.0]
    result = wasserstein.multi_wasserstein_distortion(
        features_a, features_b_different, log2_sigma
    )
    assert result > 0

    # Should raise ValueError for list length mismatch
    with pytest.raises(ValueError):
        _ = wasserstein.multi_wasserstein_distortion(
            features_a, [jnp.ones((1, 32, 32))], log2_sigma
        )

    # Should raise ValueError for shape mismatch in one of the pairs
    with pytest.raises(ValueError):
        _ = wasserstein.multi_wasserstein_distortion(
            features_a, [jnp.ones((1, 32, 32)), jnp.ones((1, 8, 8))], log2_sigma
        )


def test_compute_multiscale_stats_non_constant():
    """Verifies variance computation on non-uniform input (checkerboard pattern).

    This is a high-variance pattern with sharp spatial contrast, perfect for testing
    whether compute_multiscale_stats detects local variance.

    Checks
    ------
    - Variance Positivity: Ensures variance is non-negative at all pyramid levels.
    - Shape Halving: Validates spatial resolution reduction at each level (e.g., 64x64 →
      32x32 → 16x16).
    """
    # Create a checkerboard pattern
    input_arr = jnp.zeros((1, 64, 64))
    input_arr = input_arr.at[:, 0::2, 0::2].set(1.0)
    input_arr = input_arr.at[:, 1::2, 1::2].set(1.0)

    means, variances = wasserstein.compute_multiscale_stats(input_arr, num_levels=3)

    for level in range(3):
        # Check variance is positive and reasonable
        assert jnp.all(variances[level] >= -1e-5)
        # Check shape halving
        expected_shape = (1, 64 // (2**level), 64 // (2**level))
        assert means[level].shape == expected_shape


def test_wasserstein_distortion_intermediates():
    """Test if intermediate results are returned when return_intermediates=True.

    Checks
    ------
    - Dictionary Structure: Ensures intermediates include wd_maps (Wasserstein distortion
      maps).
    - Distortion Value: Checks that identical features yield zero distortion.
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


def test_wasserstein_distortion_jit_compilable():
    """Verify that wasserstein_distortion can be compiled with jax.jit.

    No errors should occur due to jnp.max(log2_sigma) check.
    """
    features = jnp.ones((1, 32, 32))
    log2_sigma = jnp.zeros((32, 32))  # max = 0, should not raise ValueError

    # JIT compile
    compiled_fn = jax.jit(
        wasserstein.wasserstein_distortion, static_argnames=["num_levels"]
    )

    # Should not raise any error
    result = compiled_fn(features, features, log2_sigma, num_levels=3)  # pylint: disable=not-callable
    assert jnp.isclose(result, 0.0)


def test_num_levels_handling():
    """Validates error handling for insufficient num_levels.

    Checks
    ------
    - Valid Case: num_levels=3 works when log2_sigma has a max value of 2.
    - Invalid Case: num_levels=1 triggers an error when log2_sigma exceeds it.
    """
    features = jnp.ones((1, 32, 32))
    log2_sigma = jnp.full((32, 32), 3.0)

    # num_levels=3 should handle this without error
    dist = wasserstein.wasserstein_distortion(
        features, features, log2_sigma, num_levels=3
    )
    assert jnp.isclose(dist, 0.0)

    # num_levels=2 should fail
    with pytest.raises(ValueError):
        _ = wasserstein.wasserstein_distortion(
            features, features, log2_sigma, num_levels=2
        )


def test_multi_wasserstein_sigma_scaling(monkeypatch):
    """Validates sigma map scaling for feature arrays with different resolutions.

    Checks
    ------
    - Scaling Logic: If a feature array is smaller (e.g., 32x32 vs. original 64x64 sigma
      map), log2_sigma is adjusted by subtracting log2(size_ratio).
    """

    # Original sigma map (64x64)
    log2_sigma = jnp.full((64, 64), 4.0)
    # Feature with smaller spatial dim (32x32)
    features = [jnp.ones((1, 32, 32))]
    # Expected adjusted sigma: 4 - log2(64/32) = 3
    expected_ls = jnp.full((32, 32), 3.0)

    def check_sigma(fa, fb, ls, **kwargs):
        del fa, fb, kwargs
        assert jnp.allclose(ls, expected_ls, atol=0.1), "Sigma not scaled correctly"
        return jnp.zeros(()), {"dummy": []}  # Return a dummy intermediates dict

    # Monkeypatch to intercept call to wasserstein_distortion
    monkeypatch.setattr(wasserstein, "wasserstein_distortion", check_sigma)

    # Run function under test
    _ = wasserstein.multi_wasserstein_distortion(
        features, features, log2_sigma, return_intermediates=False
    )


def test_vgg16_wasserstein_distortion_can_be_called():
    """Tests that VGG16 Wasserstein Distortion can be computed without shape errors."""
    pretrained_features.load_vgg16_model(mock=True)

    image = jax.random.uniform(jax.random.key(0), (3, 224, 224))
    dist = wasserstein.vgg16_wasserstein_distortion(image, image, jnp.full((224, 224), 3))
    assert jnp.isclose(dist, 0.0)

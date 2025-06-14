# Copyright 2022 CoDeX authors.
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
"""Tests of deep factorized entropy model."""

import chex
import distrax
import jax
import jax.numpy as jnp
from codex.ems import equinox

# TODO(jonaballe): Improve unit tests, e.g. check that the distribution is normalized,
# that expected value of `bin_bits` with noise is an upper bound on it without noise, ...


def test_logistic_is_special_case():
    # With no hidden units, the density should collapse to a logistic distribution
    # convolved with a standard uniform distribution.
    em = equinox.DeepFactorizedEntropyModel(
        jax.random.key(0), num_pdfs=1, num_units=(), init_scale=1
    )
    x = jnp.linspace(-5.0, 5.0, 30)[:, None]
    prob_em = em.bin_prob(x)
    logistic = distrax.Logistic(loc=-em.cdf_logits.biases[0][0, 0], scale=1.0)
    prob_log = logistic.cdf(x + 0.5) - logistic.cdf(x - 0.5)  # type: ignore
    chex.assert_trees_all_close(prob_em, prob_log, atol=1e-7)


def test_uniform_is_special_case():
    # With the scale parameter going to zero, the density should approach a unit-width
    # uniform distribution.
    em = equinox.DeepFactorizedEntropyModel(
        jax.random.key(0), num_pdfs=1, init_scale=1e-6
    )
    x = jnp.linspace(-1.0, 1.0, 10)[:, None]
    prob = em.bin_prob(x)
    chex.assert_trees_all_close(prob[:, 0], jnp.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0]))


def test_bin_prob_and_bits_are_consistent():
    em = equinox.DeepFactorizedEntropyModel(
        jax.random.key(0), num_pdfs=1, num_units=(2, 3)
    )
    x = jnp.linspace(-5.0, 5.0, 30)[:, None]
    prob = em.bin_prob(x)
    bits = em.bin_bits(x)
    chex.assert_trees_all_close(prob, 2**-bits, atol=1e-7)

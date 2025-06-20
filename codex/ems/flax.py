# Copyright 2024 CoDeX authors.
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
"""Entropy models as Flax modules."""

from typing import override
from flax import linen as _nn
import jax as _jax
from codex.ems import deep_factorized as _deep_factorized
from codex.ems import fourier as _fourier
from codex.ems import *  # pylint: disable=wildcard-import,unused-wildcard-import


class MonotonicMLP(_deep_factorized.MonotonicMLPBase, _nn.Module):
    """MLP that implements monotonically increasing functions by construction.

    Attributes
    ----------
    num_mlps
        Number of independent MLPs.
    num_units
        The number of filters for each of the hidden layers. The first and last layer of
        the network implementing the cumulative distribution are not included (they are
        assumed to be 1).
    init_scale
        Scale factor for the density at initialization. It is recommended to choose a
        large enough scale factor such that most values initially lie within a region of
        high likelihood. This improves training.
    """

    num_mlps: int
    num_units: tuple[int, ...]
    init_scale: float

    @override
    def setup(self):
        scale = self.init_scale ** (1 / (1 + len(self.num_units)))
        num_units = (1,) + self.num_units + (1,)
        matrices = []
        biases = []
        factors = []
        for k, shape in enumerate(zip(num_units[1:], num_units[:-1])):
            # shape == (out_dims, in_dims)
            shape = (self.num_mlps,) + shape
            matrices.append(
                self.param(
                    f"matrix_{k}",
                    lambda _, sh, sc: _deep_factorized.matrix_init(sh, sc),
                    shape,
                    scale,
                )
            )
            biases.append(
                self.param(
                    f"bias_{k}",
                    lambda r, s: _jax.random.uniform(r, shape=s, minval=-0.5, maxval=0.5),
                    shape[:2],
                )
            )
            if k < len(num_units) - 2:
                factors.append(
                    self.param(f"factor_{k}", _nn.initializers.zeros, shape[:2])
                )
        self.matrices = matrices
        self.biases = biases
        self.factors = factors


class DeepFactorizedEntropyModel(
    _deep_factorized.DeepFactorizedEntropyModelBase, _nn.Module
):
    r"""Fully factorized entropy model based on neural network cumulative.

    This is a flexible, nonparametric entropy model convolved with a unit-width uniform
    density. The model learns a factorized distribution. For example, if the input has 64
    channels (the last dimension size), then the distribution has the form::

       CDF(x) = \prod_{i=1}^64 CDF_i(x_i)

    where each function ``CDF_i`` is modeled by MLPs of different parameters.

    Attributes
    ----------
    num_pdfs
        The number of distinct scalar PDFs on the right of the input array. These are
        treated as independent, but non-identically distributed. The remaining array
        elements on the left are treated as i.i.d. (like in a batch dimension).
    num_units
        The number of filters for each of the hidden layers. The first and last layer of
        the network implementing the cumulative distribution are not included (they are
        assumed to be 1).
    init_scale
        Scale factor for the density at initialization. It is recommended to choose a
        large enough scale factor such that most values initially lie within a region of
        high likelihood. This improves training.

    Notes
    -----
    This model is further described in appendix 6.1 of [1]_. Convolution with a unit-width
    uniform distribution is described in appendix 6.2, ibid. Please cite the paper if you
    use this code for scientific or research work.

    .. [1] J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston: "Variational image
       compression with a scale hyperprior," 6th Int. Conf. on Learning Representations
       (ICLR), 2018. https://openreview.net/forum?id=rkcQFMZRb
    """

    num_pdfs: int
    num_units: tuple[int, ...] = (3, 3, 3)
    init_scale: float = 10.0

    @override
    def setup(self):
        self.cdf_logits = MonotonicMLP(self.num_pdfs, self.num_units, self.init_scale)


class PeriodicFourierEntropyModel(_fourier.PeriodicFourierEntropyModelBase, _nn.Module):
    """Fourier basis entropy model mapped to the real line.

    Attributes
    ----------
    period
        Length of interval on `x` over which entropy model is periodic.
    num_pdfs
        The number of distinct scalar PDFs on the right of the input array. These are
        treated as independent, but non-identically distributed. The remaining array
        elements on the left are treated as i.i.d. (like in a batch dimension).
    num_freqs
        Number of frequency components of the Fourier series.
    init_scale
        Scale of normal distribution for random initialization of coefficients.
    """

    period: float
    num_pdfs: int
    num_freqs: int = 10
    init_scale: float = 1e-3

    @override
    def setup(self):
        def init_coef(rng):
            return self.init_scale * _jax.random.normal(
                rng, (self.num_pdfs, self.num_freqs)
            )

        self.real = self.param("real", init_coef)
        self.imag = self.param("imag", init_coef)


class RealMappedFourierEntropyModel(
    _fourier.RealMappedFourierEntropyModelBase, _nn.Module
):
    """Fourier basis entropy model mapped to the real line.

    Attributes
    ----------
    num_pdfs
        The number of distinct scalar PDFs on the right of the input array. These are
        treated as independent, but non-identically distributed. The remaining array
        elements on the left are treated as i.i.d. (like in a batch dimension).
    num_freqs
        Number of frequency components of the Fourier series.
    init_scale
        Scale of normal distribution for random initialization of coefficients.
    """

    num_pdfs: int
    num_freqs: int = 10
    init_scale: float = 1e-3

    @override
    def setup(self):
        def init_coef(rng):
            return self.init_scale * _jax.random.normal(
                rng, (self.num_pdfs, self.num_freqs)
            )

        self.real = self.param("real", init_coef)
        self.imag = self.param("imag", init_coef)
        self.scale = self.param("scale", lambda _: _jax.numpy.ones((self.num_pdfs,)))
        self.offset = self.param("offset", lambda _: _jax.numpy.zeros((self.num_pdfs,)))

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
"""Base class for entropy models of continuous distributions."""

from typing import ClassVar
import jax
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike
DTypeLike = jax.typing.DTypeLike

# TODO(jonaballe): Think through shape contracts and broadcasting for all methods of this
# interface.


def logsum_expbig_minus_expsmall(big: ArrayLike, small: ArrayLike) -> Array:
    """Numerically stable evaluation of ``log(exp(big) - exp(small))``."""
    # Have to special case `inf` and `-inf` since otherwise we get a NaN out of the exp
    # (if both small and big are -inf).
    return jnp.where(jnp.isinf(big), big, jnp.log1p(-jnp.exp(small - big)) + big)  # type: ignore


class ContinuousEntropyModel:
    """Entropy model for continuous random variables.

    Attributes
    ----------
    min_dtype
        Minimum data type for probability computations. Inputs will be type promoted
        against this to ensure numerical accuracy.
    """

    min_dtype: ClassVar[DTypeLike] = jnp.float32

    def _maybe_upcast(self, pytree):
        dtype = jax.dtypes.result_type(
            self.min_dtype,
            *(a for a in jax.tree_util.tree_leaves(pytree) if a is not None),
        )
        return jax.tree.map(
            lambda a: None if a is None else jnp.asarray(a, dtype=dtype), pytree
        )

    def bin_prob(self, center: ArrayLike, temperature: ArrayLike = None) -> Array:
        """Computes probability mass of bins, see `bin_bits` for explanation."""
        # Default implementation may work in most cases, but may be overridden for
        # performance/stability reasons.
        return 2 ** -self.bin_bits(center, temperature)

    def bin_bits(self, center: ArrayLike, temperature: ArrayLike = None) -> Array:
        """Computes information content of unit-width quantization bins in bits.

        This function models the distribution of the bottleneck tensor **after** it has
        been subjected to (soft or hard) quantization.

        .. note:: The function does not quantize the `center`, this is the responsibility
           of the caller.

        Use cases::

            em = ConditionalEntropyModel(...)
            y = ...  # Some continuous valued data.

            # Case A: caller uses noise during training.
            u = jax.random.uniform(rng, y.shape, minval=-0.5, maxval=0.5)
            y_bits = cdx.ops.perturb_and_apply(em.bin_bits, y, u)

            # Case B: caller uses soft round.
            temperature = ...
            y_hat = cdx.ops.soft_round(y, temperature)
            y_bits = em.bin_bits(y_hat_inv, temperature)

        For training a compression model, it is recommended to either

        * Train without soft rounding (``temperature = None``), and then switch to hard
          quantization for inference. Since the loss function is then agnostic to the
          quantization offset, a suitable quantization offset should be determined after
          training using a heuristic, grid search, etc.
        * Anneal `temperature` towards zero during the course of training, and then switch
          to hard quantization for inference. Since the loss function is then aware of the
          quantization offset, it can simply be set to zero during inference.

        Parameters
        ----------
        center
            Locations of quantization bin centers.
        temperature
            Scalar temperature parameter for soft quantization. Should be ``None`` or
            ``jnp.inf`` if no soft quantization was used.

        Returns
        -------
        Array
            ``-log_2 E_u p(Q(center, temperature) + u)``, where ``Q`` is the soft rounding
            function, ``u`` is additive uniform noise, ``p`` is `self.distribution`, and
            ``E_u`` is the expectation with respect to ``u``.
        """
        raise NotImplementedError()

    def quantization_offset(self) -> Array:
        """Determines a quantization offset using a heuristic.

        Returns a good heuristic value of ``o`` to use for ``jnp.around(x - o) + o`` when
        quantizing.

        Notes
        -----
        According to [1]_, the optimal quantizer for Laplacian sources centers one
        reconstruction value at the mode of the distribution. This can also be a good
        heuristic for other distributions.

        .. [1] G. J. Sullivan: "Efficient Scalar Quantization of Exponential and Laplacian
           Random Variables," IEEE Trans. on Information Theory, vol. 42, issue 5, 1996.
           https://doi.org/10.1109/18.532878
        """
        raise NotImplementedError()

    def tail_locations(self, tail_mass: Array) -> tuple[Array, Array]:
        """Determines approximate tail quantiles.

        Parameters
        ----------
        tail_mass
            Scalar between 0 and 1. The total probability mass of the tails excluding the
            central interval between ``left`` and ``right``.

        Returns
        -------
        Array
            Approximate location of the left distribution tail.
        Array
            Approximate location of the right distribution tail.
        """
        raise NotImplementedError()

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
"""Quantization operations."""

import functools
import jax
from jax import numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


def soft_round(x: ArrayLike, temperature: ArrayLike | None) -> Array:
    """Differentiable approximation to `jnp.round`.

    Lower temperatures correspond to closer approximations of the round function. For
    temperatures approaching infinity, this function resembles the identity.

    Parameters
    ----------
    x
        Input to the function.
    temperature
        Non-negative number. Controls smoothness of the approximation. For convenience, we
        support ``temperature = None``, which is the same as ``temperature = inf``, for
        which `soft_round` is the same as identity.

    Returns
    -------
    Array
        Soft-rounded input having the same shape as `x`.

    Notes
    -----
    This function is further described in Sec. 4.1. of [1]_. The temperature argument is
    the reciprocal of `alpha` in the paper. Please cite the paper if you use this code for
    scientific or research work.

    .. [1] E. Agustsson, L. Theis: "Universally Quantized Neural Compression," Adv. in
       Neural Information Processing Systems, vol. 33, 2020.
       https://arxiv.org/abs/2006.09952
    """
    if temperature is None:
        temperature = jnp.inf

    def _soft_round(x, t):
        m = jnp.floor(x) + 0.5
        z = 2 * jnp.tanh(0.5 / t)
        r = jnp.tanh((x - m) / t) / z
        return m + r

    return jnp.where(
        temperature < 1e-4,  # type: ignore
        jnp.round(x),
        jnp.where(
            temperature > 1e4,  # type: ignore
            x,
            _soft_round(x, jnp.clip(temperature, 1e-4, 1e4)),
        ),
    )


def soft_round_inverse(x: ArrayLike, temperature: ArrayLike | None) -> Array:
    """Inverse of `soft_round`.

    Lower temperatures correspond to closer approximations of the round function. For
    temperatures approaching infinity, this function resembles the identity.

    Parameters
    ----------
    x
        Input to the function.
    temperature
        Non-negative number. Controls smoothness of the approximation. For convenience, we
        support ``temperature = None``, which is the same as ``temperature = inf``, for
        which `soft_round` is the same as identity.

    Returns
    -------
    Array
        Return value of the same shape as `x`.

    Notes
    -----
    This function is further described in Sec. 4.1. of [1]_. The temperature argument is
    the reciprocal of `alpha` in the paper. Please cite the paper if you use this code for
    scientific or research work.

    .. [1] E. Agustsson, L. Theis: "Universally Quantized Neural Compression," Adv. in
       Neural Information Processing Systems, vol. 33, 2020.
       https://arxiv.org/abs/2006.09952
    """
    if temperature is None:
        temperature = jnp.inf

    def _sr_inverse(x, t):
        m = jnp.floor(x) + 0.5
        z = 2 * jnp.tanh(0.5 / t)
        r = jnp.arctanh((x - m) * z) * t
        return m + r

    return jnp.where(
        temperature < 1e-4,  # type: ignore
        jnp.ceil(x) - 0.5,
        jnp.where(
            temperature > 1e4,  # type: ignore
            x,
            _sr_inverse(x, jnp.clip(temperature, 1e-4, 1e4)),
        ),
    )


def soft_round_conditional_mean(x: ArrayLike, temperature: ArrayLike | None) -> Array:
    """Conditional mean of inputs given noisy soft rounded values.

    Computes ``g(z) = E[X | Q(X) + U = z]`` where ``Q`` is the soft-rounding function,
    ``U`` is uniform between -0.5 and 0.5 and ``X`` is considered uniform when truncated
    to the interval ``[z - 0.5, z + 0.5]``.

    Parameters
    ----------
    x
        Input to the function.
    temperature
        Non-negative number. Controls smoothness of the approximation. For convenience, we
        support ``temperature = None``, which is the same as ``temperature = inf``, for
        which `soft_round` is the same as identity.

    Returns
    -------
    Array
        Return value of the same shape as `x`.

    Notes
    -----
    This function is further described in Sec. 4.1. of [1]_. The temperature argument is
    the reciprocal of `alpha` in the paper. Please cite the paper if you use this code for
    scientific or research work.

    .. [1] E. Agustsson, L. Theis: "Universally Quantized Neural Compression," Adv. in
       Neural Information Processing Systems, vol. 33, 2020.
       https://arxiv.org/abs/2006.09952
    """
    return soft_round_inverse(x - 0.5, temperature) + 0.5  # pytype: ignore


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def ste_argmax(logits: ArrayLike, temperature: ArrayLike, axis=-1) -> Array:
    """`jnp.argmax` with straight-through gradient estimation.

    The gradient of this function is overridden to be the gradient of::

        jax.nn.softmax(logits / temperature, axis)

    Args:
      logits: Inputs to the function.
      temperature: Scalar temperature parameter for the custom gradient.
      axis: The dimension index of `logits` to take the argmax over.

    Returns:
      One-hot representation of the ``argmax`` of `logits`.
    """
    del temperature  # unused in forward pass
    index = jnp.argmax(logits, axis=axis)
    return jax.nn.one_hot(index, logits.shape[axis], axis=axis, dtype=logits.dtype)


@ste_argmax.defjvp
def ste_argmax_jvp(axis, primals, tangents):
    """Straight-through gradient estimator for `ste_argmax`."""
    logits, temperature = primals
    logits_dot, _ = tangents
    _, argmax_dot = jax.jvp(
        lambda l: jax.nn.softmax(l / temperature, axis=axis), (logits,), (logits_dot,)
    )
    return ste_argmax(logits, temperature), argmax_dot


@jax.custom_jvp
def ste_round(x: ArrayLike) -> Array:
    """`jnp.round` with straight-through gradient estimation."""
    return jnp.round(x)


@ste_round.defjvp
def ste_round_jvp(primals, tangents):
    """Straight-through gradient estimator for `ste_round`."""
    (x,) = primals
    (x_dot,) = tangents
    return ste_round(x), x_dot

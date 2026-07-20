from math import prod
from typing import Callable, List

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


def grad(
    fun: Callable[[e3nn.IrrepsArray], e3nn.IrrepsArray],
    argnums: int = 0,
    has_aux: bool = False,
    regroup_output: bool = True,
) -> e3nn.IrrepsArray:
    r"""Take the gradient of an equivariant function and reduce it into irreps.

    Args:
        fun: An equivariant function.
        argnums: The argument number to differentiate with respect to.
        has_aux: If True, the function returns a tuple of the output and an auxiliary value.
        regroup_output (bool, optional): Regroup the outputs into irreps. Defaults to True.

    Returns:
        The gradient of the function. Also an equivariant function.

    Examples:
        >>> jnp.set_printoptions(precision=3, suppress=True)
        >>> f = grad(lambda x: 0.5 * e3nn.norm(x, squared=True))
        >>> x = e3nn.IrrepsArray("1o", jnp.array([1.0, 2, 3]))
        >>> f(x)
        1x1o [1. 2. 3.]
    """
    if not isinstance(argnums, int):
        raise ValueError("argnums must be an int.")

    def _grad(*args, **kwargs) -> e3nn.IrrepsArray:
        pass

    return _grad

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def einsum(eq, *xs):
    return jnp.einsum(eq, *xs, optimize="optimal")


@einsum.defjvp
def einsum_jvp(eq: str, xs: Tuple[jax.Array], x_dots: Tuple[jax.Array]) -> jax.Array:
    pass

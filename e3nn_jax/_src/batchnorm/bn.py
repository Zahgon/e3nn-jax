from math import prod
from typing import Optional

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


def batch_norm(
    input: e3nn.IrrepsArray,
    ra_mean: Optional[jax.Array],
    ra_var: Optional[jax.Array],
    weight: Optional[jax.Array],
    bias: Optional[jax.Array],
    normalization: str,
    reduce: str,
    is_instance: bool,
    use_running_average: bool,
    use_affine: bool,
    momentum: float,
    epsilon: float,
    mask: Optional[jax.Array] = None,
):
    pass

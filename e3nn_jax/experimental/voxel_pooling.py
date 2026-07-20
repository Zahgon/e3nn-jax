from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def interpolate_trilinear(input: jax.Array, x: float, y: float, z: float) -> jax.Array:
    pass


def interpolate_nearest(input: jax.Array, x: float, y: float, z: float) -> jax.Array:
    pass


@partial(jax.jit, static_argnums=(1, 2))
def _zoom(
    input: jax.Array,
    output_size: Tuple[int, int, int],
    interpolation="linear",
) -> jax.Array:
    nx, ny, nz = input.shape[-3:]

    def f(n_src, n_dst):
        a = n_src / n_dst * jnp.arange(n_dst)
        delta = 0.5 * (n_src / n_dst - 1)
        return delta + a

    xi = f(nx, output_size[0])
    yi = f(ny, output_size[1])
    zi = f(nz, output_size[2])

    xg, yg, zg = jnp.meshgrid(xi, yi, zi, indexing="ij")

    if interpolation == "linear":
        interp = interpolate_trilinear
    if interpolation == "nearest":
        interp = interpolate_nearest

    output = jax.vmap(interp, (None, 0, 0, 0), -1)(
        input, xg.flatten(), yg.flatten(), zg.flatten()
    )
    output = output.reshape(*input.shape[:-3], *output_size)
    return output


def zoom(
    input: jax.Array,
    *,
    resize_rate: Optional[Tuple[float, float, float]] = None,
    output_size: Optional[Tuple[int, int, int]] = None,
    interpolation: str = "linear",
) -> jax.Array:
    r"""Rescale a 3D image by bilinear interpolation.

    Args:
        input: array of shape ``[..., x, y, z]``
        resize_rate: tuple of 3 floats
        output_size: tuple of 3 ints

    Returns:
        3D image of size output_size
    """
    nx, ny, nz = input.shape[-3:]

    if resize_rate is not None:
        assert output_size is None

        if isinstance(resize_rate, (float, int)):
            resize_rate = (resize_rate,) * 3

        output_size = (
            round(nx * resize_rate[0]),
            round(ny * resize_rate[1]),
            round(nz * resize_rate[2]),
        )

    assert isinstance(output_size, tuple)

    return _zoom(input, output_size, interpolation)

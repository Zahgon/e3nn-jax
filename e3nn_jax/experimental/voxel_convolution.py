import math
from typing import Dict, Optional, Tuple, Union

import flax
import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax

import e3nn_jax as e3nn
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct


class ConvolutionHaiku(hk.Module):

    def __init__(
        self,
        irreps_out: e3nn.Irreps,
        irreps_sh: e3nn.Irreps,
        diameter: float,
        num_radial_basis: Union[int, Dict[int, int]],
        steps: Tuple[float, float, float],
        *,
        relative_starts: Union[float, Dict[int, float]] = 0.0,
        padding: str = "SAME",
    ):
        super().__init__()

        self.num_radial_basis = num_radial_basis
        self.relative_starts = relative_starts
        self.irreps_out = irreps_out
        self.irreps_sh = irreps_sh
        self.diameter = diameter
        self.padding = padding
        self.steps = steps

    def kernel(
        self,
        irreps_in: e3nn.Irreps,
        irreps_out: e3nn.Irreps,
        steps: Optional[jax.Array] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> jax.Array:
        pass

    def __call__(
        self, input: e3nn.IrrepsArray, steps: Optional[jax.Array] = None
    ) -> e3nn.IrrepsArray:
        r"""Evaluate the convolution.

        Args:
            input: Input data of shape ``[batch, x, y, z, irreps_in.dim]``
            steps: dynamic steps, if None use the static steps

        Returns:
            Output data of shape ``[batch, x, y, z, irreps_out.dim]``
        """
        return _call(self, input, steps)


class ConvolutionFlax(flax.linen.Module):
    irreps_out: e3nn.Irreps
    irreps_sh: e3nn.Irreps
    diameter: float
    num_radial_basis: Union[int, Dict[int, int]]
    steps: Tuple[float, float, float]
    relative_starts: Union[float, Dict[int, float]] = 0.0
    padding: str = "SAME"

    def kernel(
        self,
        irreps_in: e3nn.Irreps,
        irreps_out: e3nn.Irreps,
        steps: Optional[jax.Array] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> jax.Array:
        pass

    @flax.linen.compact
    def __call__(
        self, input: e3nn.IrrepsArray, steps: Optional[jax.Array] = None
    ) -> e3nn.IrrepsArray:
        return _call(self, input, steps)


ConvolutionFlax.__doc__ = ConvolutionHaiku.__doc__
ConvolutionFlax.kernel.__doc__ = ConvolutionHaiku.kernel.__doc__
ConvolutionFlax.__call__.__doc__ = ConvolutionHaiku.__call__.__doc__


def _tp_weight(
    self: Union[ConvolutionHaiku, ConvolutionFlax],
    lattice: jax.Array,
    i_in: int,
    i_sh: int,
    i_out: int,
    mul_ir_in: e3nn.MulIrrep,
    ir_sh: e3nn.Irrep,
    mul_ir_out: e3nn.MulIrrep,
    path_shape: Tuple[int, ...],
    weight_std: float,
    get_parameter,
) -> jax.Array:
    pass


def _kernel(
    self: Union[ConvolutionHaiku, ConvolutionFlax],
    irreps_in: e3nn.Irreps,
    irreps_out: e3nn.Irreps,
    steps: Optional[jax.Array],
    get_parameter,
    dtype,
) -> jax.Array:
    pass


def _call(
    self: Union[ConvolutionHaiku, ConvolutionFlax],
    input: e3nn.IrrepsArray,
    steps: Optional[jax.Array] = None,
) -> e3nn.IrrepsArray:
    pass

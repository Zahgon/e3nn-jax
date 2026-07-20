from typing import Optional, Union, Tuple, Dict

import equinox as eqx
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn
from e3nn_jax._src.utils.dtype import get_pytree_dtype

from .linear import (
    FunctionalLinear,
    linear_indexed,
    linear_mixed,
    linear_mixed_per_channel,
    linear_vanilla,
)


def _get_gradient_normalization(
    gradient_normalization: Optional[Union[float, str]],
) -> float:
    pass


class Linear(eqx.Module):

    irreps_out: e3nn.Irreps = eqx.field(static=True)
    irreps_in: e3nn.Irreps = eqx.field(static=True)
    channel_out: int = eqx.field(static=True)
    channel_in: int = eqx.field(static=True)
    gradient_normalization: Optional[Union[float, str]] = eqx.field(static=True)
    path_normalization: Optional[Union[float, str]] = eqx.field(static=True)
    biases: bool = eqx.field(static=True)
    num_indexed_weights: Optional[int] = eqx.field(static=True)
    weights_per_channel: bool = eqx.field(static=True)
    force_irreps_out: bool = eqx.field(static=True)
    weights_dim: Optional[int] = eqx.field(static=True)
    linear_type: str = eqx.field(static=True)

    _linear: FunctionalLinear = eqx.field(static=True)
    _weights: Dict[str, jax.Array]
    _input_dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        *,
        irreps_out: e3nn.Irreps,
        irreps_in: e3nn.Irreps,
        channel_out: Optional[int] = None,
        channel_in: Optional[int] = None,
        biases: bool = False,
        path_normalization: Optional[Union[str, float]] = None,
        gradient_normalization: Optional[Union[str, float]] = None,
        num_indexed_weights: Optional[int] = None,
        weights_per_channel: bool = False,
        force_irreps_out: bool = False,
        weights_dim: Optional[int] = None,
        input_dtype: jnp.dtype = jnp.float32,
        linear_type: str = "vanilla",
        key: jax.Array,
    ):
        irreps_in_regrouped = e3nn.Irreps(irreps_in).regroup()
        irreps_out = e3nn.Irreps(irreps_out)

        self.irreps_in = irreps_in_regrouped
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.biases = biases
        self.path_normalization = path_normalization
        self.num_indexed_weights = num_indexed_weights
        self.weights_per_channel = weights_per_channel
        self.force_irreps_out = force_irreps_out
        self.linear_type = linear_type
        self.weights_dim = weights_dim
        self._input_dtype = input_dtype

        self.gradient_normalization = _get_gradient_normalization(
            gradient_normalization
        )

        channel_irrep_multiplier = 1
        if self.channel_out is not None:
            assert not self.weights_per_channel
            channel_irrep_multiplier = self.channel_out

        if not self.force_irreps_out:
            irreps_out = irreps_out.filter(keep=irreps_in_regrouped)
            irreps_out = irreps_out.simplify()
        self.irreps_out = irreps_out

        self._linear = FunctionalLinear(
            irreps_in_regrouped,
            channel_irrep_multiplier * irreps_out,
            biases=self.biases,
            path_normalization=self.path_normalization,
            gradient_normalization=self.gradient_normalization,
        )
        self._weights = self._get_weights(key)

    def _get_weights(self, key: jax.Array):
        pass

    def __call__(self, weights_or_input, input_or_none=None) -> e3nn.IrrepsArray:
        """Apply the linear operator.

        Args:
            weights (optional IrrepsArray or jax.Array): scalar weights that are contracted with free parameters.
                An array of shape ``(..., contracted_axis)``. Broadcasting with `input` is supported.
            input (IrrepsArray): input irreps-array of shape ``(..., [channel_in,] irreps_in.dim)``.
                Broadcasting with `weights` is supported.

        Returns:
            IrrepsArray: output irreps-array of shape ``(..., [channel_out,] irreps_out.dim)``.
                Properly normalized assuming that the weights and input are properly normalized.
        """
        if input_or_none is None:
            weights = None
            input: e3nn.IrrepsArray = weights_or_input
        else:
            weights: jax.Array = weights_or_input
            input: e3nn.IrrepsArray = input_or_none
        del weights_or_input, input_or_none

        input = e3nn.as_irreps_array(input)

        dtype = get_pytree_dtype(weights, input)
        if dtype.kind == "i":
            dtype = jnp.float32
        input = input.astype(dtype)

        if self.irreps_in != input.irreps.regroup():
            raise ValueError(
                f"e3nn.equinox.Linear: The input irreps ({input.irreps}) "
                f"do not match the expected irreps ({self.irreps_in})."
            )

        if self.channel_in is not None:
            if self.channel_in != input.shape[-2]:
                raise ValueError(
                    f"e3nn.equinox.Linear: The input channel ({input.shape[-2]}) "
                    f"does not match the expected channel ({self.channel_in})."
                )

        input = input.remove_zero_chunks().regroup()

        def get_parameter(
            name: str,
            path_shape: Tuple[int, ...],
            weight_std: float,
            dtype: jnp.dtype = jnp.float32,
        ):
            pass

        assertion_message = (
            "Weights cannot be provided when 'linear_type' is 'vanilla'."
            "Otherwise, weights must be provided."
            "If weights are provided, they must be either: \n"
            "* integers and num_indexed_weights must be provided, or \n"
            "* floats and num_indexed_weights must not be provided.\n"
            f"weights.dtype={weights.dtype if weights is not None else None}, "
            f"num_indexed_weights={self.num_indexed_weights}"
        )

        if self.linear_type == "vanilla":
            assert weights is None, assertion_message
            output = linear_vanilla(input, self._linear, get_parameter)

        if self.linear_type in ["indexed", "mixed", "mixed_per_channel"]:
            assert weights is not None, assertion_message
            if isinstance(weights, e3nn.IrrepsArray):
                if not weights.irreps.is_scalar():
                    raise ValueError("weights must be scalar")
                weights = weights.array

        if self.linear_type == "indexed":
            assert weights.dtype.kind == "i", assertion_message
            if self.weights_per_channel:
                raise NotImplementedError(
                    "weights_per_channel not implemented for indexed weights"
                )

            output = linear_indexed(
                input, self._linear, get_parameter, weights, self.num_indexed_weights
            )

        if self.linear_type in ["mixed", "mixed_per_channel"]:
            assert weights.dtype.kind in "fc", assertion_message
            assert self.num_indexed_weights is None, assertion_message

        if self.linear_type == "mixed":
            output = linear_mixed(
                input,
                self._linear,
                get_parameter,
                weights,
                self.gradient_normalization,
            )

        if self.linear_type == "mixed_per_channel":
            output = linear_mixed_per_channel(
                input,
                self._linear,
                get_parameter,
                weights,
                self.gradient_normalization,
            )

        if self.channel_out is not None:
            output = output.mul_to_axis(self.channel_out)

        return output.rechunk(self.irreps_out)

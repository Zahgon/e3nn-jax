from math import sqrt
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

import e3nn_jax as e3nn
from e3nn_jax import Irreps, IrrepsArray, config
from e3nn_jax._src.utils.sum_tensors import sum_tensors
from e3nn_jax._src.utils.dtype import get_pytree_dtype


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float
    weight_std: float


class FunctionalLinear:
    irreps_in: Irreps
    irreps_out: Irreps
    instructions: List[Instruction]
    output_mask: jax.Array

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        instructions: Optional[List[Tuple[int, int]]] = None,
        biases: Optional[Union[List[bool], bool]] = None,
        path_normalization: Union[str, float] = None,
        gradient_normalization: Union[str, float] = None,
    ):
        if path_normalization is None:
            path_normalization = config("path_normalization")
        if isinstance(path_normalization, str):
            path_normalization = {"element": 0.0, "path": 1.0}[path_normalization]

        if gradient_normalization is None:
            gradient_normalization = config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]

        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)

        if instructions is None:
            instructions = [
                (i_in, i_out)
                for i_in, (_, ir_in) in enumerate(irreps_in)
                for i_out, (_, ir_out) in enumerate(irreps_out)
                if ir_in == ir_out
            ]

        instructions = [
            Instruction(
                i_in=i_in,
                i_out=i_out,
                path_shape=(irreps_in[i_in].mul, irreps_out[i_out].mul),
                path_weight=1,
                weight_std=1,
            )
            for i_in, i_out in instructions
        ]

        def alpha(this):
            pass

        instructions = [
            Instruction(
                i_in=ins.i_in,
                i_out=ins.i_out,
                path_shape=ins.path_shape,
                path_weight=sqrt(alpha(ins)) ** gradient_normalization,
                weight_std=sqrt(alpha(ins)) ** (1.0 - gradient_normalization),
            )
            for ins in instructions
        ]

        if biases is None:
            biases = len(irreps_out) * (False,)
        if isinstance(biases, bool):
            biases = [biases and ir.is_scalar() for _, ir in irreps_out]

        assert len(biases) == len(irreps_out)
        assert all(ir.is_scalar() or (not b) for b, (_, ir) in zip(biases, irreps_out))

        instructions += [
            Instruction(
                i_in=-1,
                i_out=i_out,
                path_shape=(mul_ir.dim,),
                path_weight=1.0,
                weight_std=0.0,
            )
            for i_out, (bias, mul_ir) in enumerate(zip(biases, irreps_out))
            if bias
        ]

        with jax.ensure_compile_time_eval():
            if irreps_out.dim > 0:
                output_mask = jnp.concatenate(
                    [
                        (
                            jnp.ones(mul_ir.dim, bool)
                            if any(
                                (ins.i_out == i_out) and (0 not in ins.path_shape)
                                for ins in instructions
                            )
                            else jnp.zeros(mul_ir.dim, bool)
                        )
                        for i_out, mul_ir in enumerate(irreps_out)
                    ]
                )
            else:
                output_mask = jnp.ones(0, bool)

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.instructions = instructions
        self.output_mask = output_mask

    @property
    def num_weights(self) -> int:
        pass

    def aggregate_paths(self, paths, output_shape, output_dtype) -> IrrepsArray:
        pass

    def split_weights(self, weights: jax.Array) -> List[jax.Array]:
        pass

    def __call__(
        self, ws: Union[List[jax.Array], jax.Array], input: IrrepsArray
    ) -> IrrepsArray:
        input = input.rechunk(self.irreps_in)
        if input.ndim != 1:
            raise ValueError(
                f"FunctionalLinear does not support broadcasting, input shape is {input.shape}"
            )

        if not isinstance(ws, list):
            ws = self.split_weights(ws)

        paths = [
            (
                ins.path_weight * w
                if ins.i_in == -1
                else (
                    None
                    if input.chunks[ins.i_in] is None
                    else ins.path_weight
                    * jnp.einsum("uw,ui->wi", w, input.chunks[ins.i_in])
                )
            )
            for ins, w in zip(self.instructions, ws)
        ]
        return self.aggregate_paths(paths, input.shape[:-1], input.dtype)

    def matrix(self, ws: List[jax.Array]) -> jax.Array:
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out}, "
            f"{len(self.instructions)} instructions, {self.num_weights} weights)"
        )


def linear_vanilla(
    input: IrrepsArray,
    linear: FunctionalLinear,
    get_parameter: Callable[[str, Tuple[int, ...], float, Any], jax.Array],
) -> IrrepsArray:
    pass


def linear_indexed(
    input: IrrepsArray,
    lin: FunctionalLinear,
    get_parameter: Callable[[str, Tuple[int, ...], float, Any], jax.Array],
    indices: jax.Array,
    num_indexed_weights: int,
) -> IrrepsArray:
    pass


def linear_mixed(
    input: IrrepsArray,
    lin: FunctionalLinear,
    get_parameter: Callable[[str, Tuple[int, ...], float, Any], jax.Array],
    weights: jax.Array,
    gradient_normalization: float,
) -> IrrepsArray:
    pass


def linear_mixed_per_channel(
    input: IrrepsArray,
    lin: FunctionalLinear,
    get_parameter: Callable[[str, Tuple[int, ...], float, Any], jax.Array],
    weights: jax.Array,
    gradient_normalization: float,
) -> IrrepsArray:
    pass


def validate_inputs_for_instructions(
    input: IrrepsArray,
    instructions: Optional[List[Tuple[int, int]]],
    simplify_irreps_internally: bool,
    channel_out: Optional[int],
    irreps_in: Optional[Irreps],
) -> None:
    pass


def parse_gradient_normalization(gradient_normalization: Optional[str]) -> float:
    pass

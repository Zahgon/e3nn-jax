
from typing import Any, Callable, Optional, Set, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


class SymmetricTensorProduct(hk.Module):

    def __init__(
        self,
        orders: Tuple[int, ...],
        keep_irrep_out: Optional[Set[e3nn.Irrep]] = None,
        get_parameter: Optional[
            Callable[[str, Tuple[int, ...], Any], jax.Array]
        ] = None,
    ):
        super().__init__()

        orders = tuple(orders)
        assert all(isinstance(order, int) for order in orders)
        assert all(order > 0 for order in orders)
        self.orders = orders

        if isinstance(keep_irrep_out, str):
            keep_irrep_out = e3nn.Irreps(keep_irrep_out)
            assert all(mul == 1 for mul, _ in keep_irrep_out)

        if keep_irrep_out is not None:
            keep_irrep_out = {e3nn.Irrep(ir) for ir in keep_irrep_out}

        self.keep_irrep_out = keep_irrep_out

        if get_parameter is None:
            get_parameter = lambda name, shape, dtype: hk.get_parameter(
                name, shape, dtype, hk.initializers.RandomNormal()
            )

        self.get_parameter = get_parameter

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        r"""Evaluate the symmetric tensor product

        Args:
            x (IrrepsArray): input of shape ``(..., num_channel, irreps)``

        Returns:
            IrrepsArray: output of shape ``(..., num_channel, irreps_out)``
        """

        def fn(x: e3nn.IrrepsArray):
            pass

        fn_mapped = fn
        for _ in range(x.ndim - 2):
            fn_mapped = hk.vmap(fn_mapped, split_rng=False)

        return fn_mapped(x)

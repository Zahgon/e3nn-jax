from functools import partial
from typing import Any, List

import jax
import numpy as np
from jax import linear_util as lu
from jax.core import Atom, ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var, jaxpr_as_fun


def curry(f):
    pass


@curry
@curry
def closed_jaxpr_transform_to_fn_transform(
    closed_jaxpr_transform, fn, *args
):  # pragma: no cover
    f = lu.wrap_init(fn)

    in_flat, in_tree = jax.tree_util.tree_flatten(args)
    f, out_tree = jax.flatten_fun_nokwargs(f, in_tree)
    closed_jaxpr = jax.make_jaxpr(f.call_wrapped)(*in_flat)
    closed_jaxpr, input_indices = closed_jaxpr_transform(closed_jaxpr)
    out_flat = jaxpr_as_fun(closed_jaxpr)(*[in_flat[i] for i in input_indices])

    return jax.tree_util.tree_unflatten(out_tree(), out_flat)


def replace_var(jaxpr: Jaxpr, old: Var, new: Var) -> List[JaxprEqn]:  # pragma: no cover
    pass


def remove_deadcode(
    jaxpr: Jaxpr, output_indices=None
) -> ClosedJaxpr:  # pragma: no cover
    pass


def remove_duplicate_constants(
    closed_jaxpr: ClosedJaxpr,
) -> ClosedJaxpr:  # pragma: no cover
    pass


def remove_duplicate_equations(
    jaxpr: Jaxpr, skip_first=0
) -> ClosedJaxpr:  # pragma: no cover
    pass


def optimize_jaxpr(closed_jaxpr: ClosedJaxpr) -> ClosedJaxpr:  # pragma: no cover
    pass


reduce_compile_time = closed_jaxpr_transform_to_fn_transform(optimize_jaxpr)



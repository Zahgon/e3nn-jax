import inspect
from functools import wraps

import jax
import e3nn_jax as e3nn


def overload_for_irreps_without_array(
    irrepsarray_argnums=None, irrepsarray_argnames=None, shape=()
):
    def decorator(func):
        pass

    return decorator

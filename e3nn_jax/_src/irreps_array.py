import functools
import math
import operator
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np
from attr import attrib, attrs
from jax.tree_util import tree_map

import e3nn_jax as e3nn
from e3nn_jax import Irreps
from e3nn_jax._src.irreps import IntoIrreps


def _infer_backend(pytree):
    any_numpy = any(
        isinstance(x, np.ndarray) for x in jax.tree_util.tree_leaves(pytree)
    )
    any_jax = any(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(pytree))
    if any_numpy and any_jax:
        raise ValueError("Cannot mix numpy and jax arrays")
    if any_numpy:
        return np
    if any_jax:
        return jnp
    return jnp


def _is_ellipse(x):
    return type(x) == type(Ellipsis)


def _is_none_slice(x):
    return isinstance(x, slice) and x == slice(None)


@attrs(frozen=True, init=True, repr=False, cmp=False)
class IrrepsArray:

    irreps: Irreps = attrib(converter=Irreps)
    array: jax.Array = attrib()
    _zero_flags: Optional[Tuple[bool, ...]] = attrib(
        default=None, kw_only=True, converter=lambda x: None if x is None else tuple(x)
    )
    _chunks: Optional[List[Optional[jax.Array]]] = attrib(default=None, kw_only=True)

    def __attrs_post_init__(self):
        if (
            hasattr(self.array, "shape")
            and isinstance(self.array.shape, tuple)
            and len(self.array.shape) > 0
        ):
            if self.array.shape[-1] != self.irreps.dim:
                raise ValueError(
                    f"IrrepsArray: Array shape {self.array.shape} incompatible with irreps {self.irreps}. "
                    f"{self.array.shape[-1]} != {self.irreps.dim}"
                )
            if self._chunks is not None:
                for (mul, ir), chunk in zip(self.irreps, self._chunks):
                    if chunk is not None and chunk.shape[-2:] != (mul, ir.dim):
                        raise ValueError(
                            f"IrrepsArray: chunk shape {chunk.shape} incompatible with mul={mul} and ir.dim={ir.dim}"
                        )

        if self._zero_flags is not None:
            if len(self._zero_flags) != len(self.irreps):
                raise ValueError(
                    f"IrrepsArray: len(zero_flags) != len(irreps), {len(self._zero_flags)} != {len(self.irreps)}"
                )

        if self._chunks is not None:
            if len(self._chunks) != len(self.irreps):
                raise ValueError(
                    f"IrrepsArray: len(chunks) != len(irreps), {len(self._chunks)} != {len(self.irreps)}"
                )

    @staticmethod
    def from_list(
        irreps: IntoIrreps,
        chunks: List[Optional[jax.Array]],
        leading_shape: Tuple[int, ...],
        dtype=None,
        *,
        backend=None,
    ):
        pass

    @staticmethod
    def as_irreps_array(array: Union[jax.Array, "IrrepsArray"], *, backend=None):
        warnings.warn(
            "IrrepsArray.as_irreps_array is deprecated, use e3nn.as_irreps_array instead.",
            DeprecationWarning,
        )
        return e3nn.as_irreps_array(array)

    @staticmethod
    def zeros(irreps: IntoIrreps, leading_shape, dtype=None) -> "IrrepsArray":
        warnings.warn(
            "IrrepsArray.zeros is deprecated, use e3nn.zeros instead.",
            DeprecationWarning,
        )
        return e3nn.zeros(irreps, leading_shape, dtype)

    @staticmethod
    def zeros_like(irreps_array: "IrrepsArray") -> "IrrepsArray":
        warnings.warn(
            "IrrepsArray.zeros_like is deprecated, use e3nn.zeros_like instead.",
            DeprecationWarning,
        )
        return e3nn.zeros_like(irreps_array)

    @property
    def list(self) -> List[Optional[jax.Array]]:
        pass

    @property
    def chunks(self) -> List[Optional[jax.Array]]:
        pass

    @property
    def zero_flags(self):
        pass

    @property
    def shape(self):
        pass

    @property
    def dtype(self):
        r"""dtype. Equivalent to ``self.array.dtype``."""
        return self.array.dtype

    @property
    def ndim(self):
        pass


    def __repr__(self):  # noqa: D105
        r = str(self.array)
        if "\n" in r:
            return f"{self.irreps}\n{r}"
        return f"{self.irreps} {r}"

    def __len__(self):  # noqa: D105
        return len(self.array)

    def __eq__(
        self: "IrrepsArray", other: Union["IrrepsArray", jax.Array]
    ) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend(self.array)

        if isinstance(other, IrrepsArray):
            if self.irreps != other.irreps:
                raise ValueError(
                    "IrrepsArray({self.irreps}, shape={self.shape}) == IrrepsArray({other.irreps}) is not equivariant."
                )

            leading_shape = jnp.broadcast_shapes(self.shape[:-1], other.shape[:-1])

            def eq(mul: int, x: jax.Array, y: jax.Array) -> jax.Array:
                pass

            chunks = [
                eq(mul, x, y)[..., None]
                for (mul, ir), x, y in zip(self.irreps, self.chunks, other.chunks)
            ]
            return e3nn.from_chunks(
                [(mul, "0e") for mul, _ in self.irreps], chunks, leading_shape, bool
            )

        other = jnp.asarray(other)
        if self.irreps.lmax > 0 or (other.ndim > 0 and other.shape[-1] != 1):
            raise ValueError(
                f"IrrepsArray({self.irreps}, shape={self.shape}) == scalar(shape={other.shape}) is not equivariant."
            )
        return IrrepsArray(self.irreps, self.array == other)

    def __neg__(self: "IrrepsArray") -> "IrrepsArray":
        return IrrepsArray(
            self.irreps,
            -self.array,
            zero_flags=self.zero_flags,
            chunks=tree_map(lambda x: -x, self._chunks),
        )

    def __add__(
        self: "IrrepsArray", other: Union["IrrepsArray", jax.Array, float, int]
    ) -> "IrrepsArray":  # noqa: D105
        if isinstance(other, (float, int)) and other == 0:
            return self

        jnp = _infer_backend(self.array)

        if not isinstance(other, IrrepsArray):
            if all(ir == "0e" for _, ir in self.irreps):
                other = jnp.asarray(other)
                return IrrepsArray(self.irreps, self.array + other)
            raise ValueError(
                f"IrrepsArray({self.irreps}, shape={self.shape}) + scalar is not equivariant."
            )

        if self.irreps != other.irreps:
            raise ValueError(
                f"IrrepsArray({self.irreps}, shape={self.shape}) + IrrepsArray({other.irreps}) is not equivariant."
            )

        zero_flags = tuple(x and y for x, y in zip(self.zero_flags, other.zero_flags))
        chunks = None
        if self._chunks is not None and other._chunks is not None:
            chunks = [
                y if x is None else x if y is None else x + y
                for x, y in zip(self._chunks, other._chunks)
            ]

        return IrrepsArray(
            self.irreps, self.array + other.array, zero_flags=zero_flags, chunks=chunks
        )

    def __radd__(
        self: "IrrepsArray", other: Union[jax.Array, float, int]
    ) -> "IrrepsArray":
        return self + other

    def __sub__(
        self: "IrrepsArray", other: Union["IrrepsArray", jax.Array, float, int]
    ) -> "IrrepsArray":  # noqa: D105
        if isinstance(other, (float, int)) and other == 0:
            return self

        jnp = _infer_backend(self.array)

        if not isinstance(other, IrrepsArray):
            if all(ir == "0e" for _, ir in self.irreps):
                other = jnp.asarray(other)
                return IrrepsArray(irreps=self.irreps, array=self.array - other)
            raise ValueError(
                f"IrrepsArray({self.irreps}, shape={self.shape}) - scalar is not equivariant."
            )

        if self.irreps != other.irreps:
            raise ValueError(
                f"IrrepsArray({self.irreps}, shape={self.shape}) - IrrepsArray({other.irreps}) is not equivariant."
            )

        zero_flags = tuple(x and y for x, y in zip(self.zero_flags, other.zero_flags))
        chunks = None
        if self._chunks is not None and other._chunks is not None:
            chunks = [
                x if y is None else -y if x is None else x - y
                for x, y in zip(self._chunks, other._chunks)
            ]

        return IrrepsArray(
            self.irreps, self.array - other.array, zero_flags=zero_flags, chunks=chunks
        )

    def __rsub__(
        self: "IrrepsArray", other: Union[jax.Array, float, int]
    ) -> "IrrepsArray":
        return -self + other

    def __mul__(
        self: "IrrepsArray", other: Union["IrrepsArray", jax.Array]
    ) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend(self.array)

        if isinstance(other, IrrepsArray):
            if self.irreps.num_irreps != other.irreps.num_irreps:
                raise ValueError(
                    f"IrrepsArray({self.irreps}, shape={self.shape}) * IrrepsArray({other.irreps}) "
                    "only works if the number of irreps is the same."
                )
            irreps_out = e3nn.elementwise_tensor_product(self.irreps, other.irreps)
            if irreps_out.num_irreps != self.irreps.num_irreps:
                raise ValueError(
                    f"IrrepsArray({self.irreps}, shape={self.shape}) * IrrepsArray({other.irreps}) "
                    "is only supported for scalar * irreps and irreps * scalar. "
                    "To perform irreps * irreps use e3nn.elementwise_tensor_product or e3nn.tensor_product."
                )
            return e3nn.elementwise_tensor_product(self, other)

        other = jnp.asarray(other)
        if other.ndim > 0 and other.shape[-1] == self.irreps.num_irreps:
            other = IrrepsArray(f"{other.shape[-1]}x0e", other)
            return e3nn.elementwise_tensor_product(self, other)

        if self.irreps.lmax > 0 and other.ndim > 0 and other.shape[-1] != 1:
            raise ValueError(
                f"IrrepsArray({self.irreps}, shape={self.shape}) * scalar(shape={other.shape}) is not equivariant."
            )

        return IrrepsArray(
            self.irreps,
            self.array * other,
            zero_flags=self.zero_flags,
            chunks=tree_map(lambda x: x * other[..., None], self._chunks),
        )

    def __rmul__(self: "IrrepsArray", other: jax.Array) -> "IrrepsArray":  # noqa: D105
        return self * other

    def __truediv__(
        self: "IrrepsArray", other: Union["IrrepsArray", jax.Array]
    ) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend(self.array)

        if isinstance(other, IrrepsArray):
            if (
                len(other.irreps) == 0
                or other.irreps.lmax > 0
                or self.irreps.num_irreps != other.irreps.num_irreps
            ):
                raise ValueError(
                    f"IrrepsArray({self.irreps}, shape={self.shape}) / IrrepsArray({other.irreps}) is not equivariant."
                )

            if any(x is None for x in other.chunks):
                raise ValueError(
                    "There are deterministic Zeros in the array of the lhs. Cannot divide by Zero."
                )
            other = 1.0 / other
            return e3nn.elementwise_tensor_product(self, other)

        other = jnp.asarray(other)
        if other.ndim > 0 and other.shape[-1] == self.irreps.num_irreps:
            other = IrrepsArray(f"{other.shape[-1]}x0e", 1.0 / other)
            return e3nn.elementwise_tensor_product(self, other)

        if self.irreps.lmax > 0 and other.ndim > 0 and other.shape[-1] != 1:
            raise ValueError(
                f"IrrepsArray({self.irreps}, shape={self.shape}) / scalar(shape={other.shape}) is not equivariant."
            )

        return IrrepsArray(
            self.irreps,
            self.array / other,
            zero_flags=self.zero_flags,
            chunks=tree_map(lambda x: x / other[..., None], self._chunks),
        )

    def __rtruediv__(
        self: "IrrepsArray", other: jax.Array
    ) -> "IrrepsArray":  # noqa: D105
        jnp = _infer_backend((self.array, other))

        other = jnp.asarray(other)
        if self.irreps.lmax > 0:
            raise ValueError(
                f"scalar(shape={other.shape}) / IrrepsArray({self.irreps}, shape={self.shape}) is not equivariant."
            )
        if any(x is None for x in self.chunks):
            raise ValueError(
                "There are deterministic Zeros in the array of the lhs. Cannot divide by Zero."
            )

        return IrrepsArray(self.irreps, other / self.array)

    def __pow__(self, exponent) -> "IrrepsArray":  # noqa: D105
        if all(ir == "0e" for _, ir in self.irreps):
            return IrrepsArray(
                self.irreps,
                self.array**exponent,
                chunks=tree_map(lambda x: x**exponent, self._chunks),
            )

        if exponent % 1.0 == 0.0 and self.irreps.lmax == 0:
            irreps = self.irreps
            if exponent % 2.0 == 0.0:
                irreps = [(mul, "0e") for mul, ir in self.irreps]
            return IrrepsArray(
                irreps,
                array=self.array**exponent,
                chunks=tree_map(lambda x: x**exponent, self._chunks),
            )

        raise ValueError(
            f"IrrepsArray({self.irreps}, shape={self.shape}) ** scalar is not equivariant."
        )

    def __iter__(self):  # noqa: D105
        if self.ndim <= 1:
            raise ValueError("Can't iterate over IrrepsArray with ndim <= 1")
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index) -> "IrrepsArray":  # noqa: D105
        if not isinstance(index, tuple):
            index = (index,)

        if isinstance(index[-1], (e3nn.Irrep, e3nn.MulIrrep, Irreps, str)):
            if not (any(map(_is_ellipse, index[:-1])) or len(index) == self.ndim):
                raise IndexError(
                    f"Error in IrrepsArray.__getitem__, Irreps index must be the last index, try x[..., {index[-1]}]."
                )

            irreps = Irreps(index[-1])

            ii = [
                i
                for i in range(len(self.irreps))
                if self.irreps[i : i + len(irreps)] == irreps
            ]
            if len(ii) != 1:
                raise IndexError(
                    f"Error in IrrepsArray.__getitem__, Can't slice with {irreps} "
                    f"because it doesn't appear exactly once in {self.irreps}."
                )
            i = ii[0]

            return IrrepsArray(
                irreps,
                self.array[
                    ..., self.irreps[:i].dim : self.irreps[: i + len(irreps)].dim
                ],
                zero_flags=self.zero_flags[i : i + len(irreps)],
                chunks=self.chunks[i : i + len(irreps)],
            )[index[:-1] + (slice(None),)]

        if (
            (any(map(_is_ellipse, index[:-1])) or len(index) == self.ndim)
            and isinstance(index[-1], slice)
            and isinstance(index[-1].start, (int, type(None)))
            and isinstance(index[-1].stop, (int, type(None)))
            and index[-1].step is None
            and (index[-1].start is not None or index[-1].stop is not None)
        ):
            start, stop, _ = index[-1].indices(self.shape[-1])

            irreps_start = None
            irreps_stop = None

            for i in range(len(self.irreps) + 1):
                if self.irreps[:i].dim == start:
                    irreps_start = i

                if irreps_start is None and start < self.irreps[:i].dim:
                    mul, ir = self.irreps[i - 1]
                    if (start - self.irreps[: i - 1].dim) % ir.dim == 0:
                        mul1 = (start - self.irreps[: i - 1].dim) // ir.dim
                        return self.rechunk(
                            self.irreps[: i - 1]
                            + e3nn.Irreps([(mul1, ir), (mul - mul1, ir)])
                            + self.irreps[i:]
                        )[index]

                if self.irreps[:i].dim == stop:
                    irreps_stop = i
                    break

                if irreps_stop is None and stop < self.irreps[:i].dim:
                    mul, ir = self.irreps[i - 1]
                    if (stop - self.irreps[: i - 1].dim) % ir.dim == 0:
                        mul1 = (stop - self.irreps[: i - 1].dim) // ir.dim
                        return self.rechunk(
                            self.irreps[: i - 1]
                            + e3nn.Irreps([(mul1, ir), (mul - mul1, ir)])
                            + self.irreps[i:]
                        )[index]

            if irreps_start is None or irreps_stop is None:
                raise IndexError(
                    f"Error in IrrepsArray.__getitem__, unable to slice {self.irreps} with {start}:{stop}."
                )

            return IrrepsArray(
                self.irreps[irreps_start:irreps_stop],
                self.array[..., start:stop],
                zero_flags=self.zero_flags[irreps_start:irreps_stop],
                chunks=self.chunks[irreps_start:irreps_stop],
            )[index[:-1] + (slice(None),)]

        if (
            len(index[:-1]) == self.ndim or any(map(_is_ellipse, index[:-1]))
        ) and index[-1] is None:
            raise IndexError(
                "Error in IrrepsArray.__getitem__, cannot add a new dimension at the end."
            )

        if (len(index) == self.ndim or any(map(_is_ellipse, index[:-1]))) and not (
            _is_ellipse(index[-1]) or _is_none_slice(index[-1]) or index[-1] is None
        ):
            if isinstance(index[-1], int):
                raise IndexError(
                    f"Error in IrrepsArray.__getitem__, integer index in the irreps dimension is not supported, "
                    f"try x[..., {index[-1]}:{index[-1] + 1}] instead."
                )
            raise IndexError(
                f"Error in IrrepsArray.__getitem__, indexing the irreps dimension with [..., {index[-1]}] "
                "is not supported."
            )

        return IrrepsArray(
            self.irreps,
            self.array[index],
            zero_flags=self.zero_flags,
            chunks=tree_map(lambda x: x[index + (slice(None),)], self._chunks),
        )

    @property
    def at(self):
        pass

    def reshape(self, shape) -> "IrrepsArray":
        r"""Reshape the array.

        Args:
            shape (tuple): new shape

        Returns:
            IrrepsArray: new IrrepsArray

        Examples:
            >>> IrrepsArray("2x0e + 1o", jnp.ones((6, 5))).reshape((2, 3, 5))
            2x0e+1x1o
            [[[1. 1. 1. 1. 1.]
              [1. 1. 1. 1. 1.]
              [1. 1. 1. 1. 1.]]
            <BLANKLINE>
             [[1. 1. 1. 1. 1.]
              [1. 1. 1. 1. 1.]
              [1. 1. 1. 1. 1.]]]
        """
        assert shape[-1] == self.irreps.dim or shape[-1] == -1
        return IrrepsArray(
            self.irreps,
            self.array.reshape(shape[:-1] + (self.irreps.dim,)),
            zero_flags=self.zero_flags,
            chunks=tree_map(
                lambda x: x.reshape(shape[:-1] + x.shape[-2:]), self._chunks
            ),
        )

    def astype(self, dtype) -> "IrrepsArray":
        r"""Change the dtype of the array.

        Args:
            dtype (dtype): new dtype

        Returns:
            IrrepsArray: new IrrepsArray
        """
        return IrrepsArray(
            irreps=self.irreps,
            array=self.array.astype(dtype),
            zero_flags=self.zero_flags,
            chunks=tree_map(lambda x: x.astype(dtype), self._chunks),
        )

    def remove_nones(self) -> "IrrepsArray":
        pass

    def remove_zero_chunks(self) -> "IrrepsArray":
        pass

    def simplify(self) -> "IrrepsArray":
        r"""Simplify the irreps.

        Examples:
            >>> IrrepsArray("0e + 0e + 0e", jnp.ones(3)).simplify()
            3x0e [1. 1. 1.]

            >>> IrrepsArray("0e + 0x1e + 0e", jnp.ones(2)).simplify()
            2x0e [1. 1.]
        """
        return self.rechunk(self.irreps.simplify())

    def unify(self) -> "IrrepsArray":
        r"""Unify the irreps.

        Examples:
            >>> IrrepsArray("0e + 0x1e + 0e", jnp.ones(2)).unify()
            1x0e+0x1e+1x0e [1. 1.]
        """
        return self.rechunk(self.irreps.unify())

    def sort(self) -> "IrrepsArray":
        r"""Sort the irreps.

        Examples:
            >>> IrrepsArray("0e + 1o + 2x0e", jnp.arange(6)).sort()
            1x0e+2x0e+1x1o [0 4 5 1 2 3]
        """
        irreps, p, inv = self.irreps.sort()
        return e3nn.from_chunks(
            irreps,
            [self.chunks[i] for i in inv],
            self.shape[:-1],
            self.dtype,
            backend=_infer_backend(self.array),
        )

    def sorted(self) -> "IrrepsArray":
        pass

    def regroup(self) -> "IrrepsArray":
        r"""Regroup the same irreps together.

        Equivalent to :meth:`sorted` followed by :meth:`simplify`.

        Examples:
            >>> IrrepsArray("0e + 1o + 2x0e", jnp.arange(6)).regroup()
            3x0e+1x1o [0 4 5 1 2 3]
        """
        return self.sort().simplify()

    def filter(
        self,
        keep: Union[
            e3nn.Irreps, List[e3nn.Irrep], Callable[[e3nn.MulIrrep], bool]
        ] = None,
        *,
        drop: Union[
            e3nn.Irreps, List[e3nn.Irrep], Callable[[e3nn.MulIrrep], bool]
        ] = None,
        lmax: int = None,
    ) -> "IrrepsArray":
        pass

    def filtered(self, *args, **kwargs) -> "IrrepsArray":
        pass

    def extend_with_zeros(self, new_irreps: Irreps) -> "IrrepsArray":
        pass

    @property
    def slice_by_mul(self):
        pass

    @property
    def slice_by_dim(self):
        pass

    @property
    def slice_by_chunk(self):
        pass

    def axis_to_irreps(self, axis: int = -2) -> "IrrepsArray":
        pass

    repeat_irreps_by_last_axis = axis_to_irreps

    def irreps_to_axis(self) -> "IrrepsArray":  # noqa: D102
        raise NotImplementedError


    def mul_to_axis(
        self, factor: Optional[int] = None, axis: int = -2
    ) -> "IrrepsArray":
        pass

    def factor_mul_to_last_axis(self, axis: int = -2) -> "IrrepsArray":
        pass

    def axis_to_mul(self, axis: int = -2) -> "IrrepsArray":
        pass

    def repeat_mul_by_last_axis(self, axis: int = -2) -> "IrrepsArray":
        pass

    def transform_by_log_coordinates(
        self, log_coordinates: jax.Array, k: int = 0
    ) -> "IrrepsArray":
        pass

    def transform_by_angles(
        self, alpha: float, beta: float, gamma: float, k: int = 0, inverse: bool = False
    ) -> "IrrepsArray":
        r"""Rotate the data by angles according to the irreps.

        Args:
            alpha (float): third rotation angle around the second axis (in radians)
            beta (float): second rotation angle around the first axis (in radians)
            gamma (float): first rotation angle around the second axis (in radians)
            k (int): parity operation
            inverse (bool): if True, apply the inverse rotation

        Returns:
            `IrrepsArray`: rotated data

        Examples:
            >>> np.set_printoptions(precision=3, suppress=True)
            >>> x = IrrepsArray("2e", jnp.array([0.1, 2, 1.0, 1, 1]))
            >>> x.transform_by_angles(jnp.pi, 0, 0)
            1x2e [ 0.1 -2.   1.  -1.   1. ]
        """
        alpha = (
            alpha
            if isinstance(alpha, (int, float))
            else jnp.asarray(alpha, dtype=self.dtype)
        )
        beta = (
            beta
            if isinstance(beta, (int, float))
            else jnp.asarray(beta, dtype=self.dtype)
        )
        gamma = (
            gamma
            if isinstance(gamma, (int, float))
            else jnp.asarray(gamma, dtype=self.dtype)
        )
        D = {
            ir: ir.D_from_angles(alpha, beta, gamma, k)
            for ir in {ir for _, ir in self.irreps}
        }
        if inverse:
            D = {ir: jnp.swapaxes(D[ir], -2, -1) for ir in D}
        new_chunks = [
            (
                jnp.reshape(
                    jnp.einsum("ij,...uj->...ui", D[ir], x),
                    self.shape[:-1] + (mul, ir.dim),
                )
                if x is not None
                else None
            )
            for (mul, ir), x in zip(self.irreps, self.chunks)
        ]
        return e3nn.from_chunks(self.irreps, new_chunks, self.shape[:-1], self.dtype)

    def transform_by_quaternion(self, q: jax.Array, k: int = 0) -> "IrrepsArray":
        pass

    def transform_by_axis_angle(
        self, axis: jax.Array, angle: float, k: int = 0
    ) -> "IrrepsArray":
        pass

    def transform_by_matrix(self, R: jax.Array) -> "IrrepsArray":
        r"""Rotate data by a rotation given by a matrix.

        Args:
            R (`jax.Array`): rotation matrix

        Returns:
            `IrrepsArray`: rotated data
        """
        d = jnp.sign(jnp.linalg.det(R))
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.transform_by_angles(*e3nn.matrix_to_angles(R), k)

    def rechunk(self, irreps: IntoIrreps) -> "IrrepsArray":
        r"""Rechunk the array with new (equivalent) irreps.

        Args:
            irreps (Irreps): new irreps

        Returns:
            `IrrepsArray`: new IrrepsArray

        Examples:
            >>> x = e3nn.from_chunks("6x0e + 4x0e", [None, jnp.ones((4, 1))], ())
            >>> x.rechunk("5x0e + 5x0e").chunks
            [None, Array([[0.],
                   [1.],
                   [1.],
                   [1.],
                   [1.]], dtype=float32)]
        """
        irreps = Irreps(irreps)
        assert self.irreps.simplify() == irreps.simplify(), (self.irreps, irreps)

        if self.irreps == irreps:
            return self

        if len(self.irreps) == 0:
            zero_flags = np.empty((0,), dtype=bool)
        else:
            zero_flags = np.concatenate(
                [
                    z * np.ones(mul * ir.dim, dtype=bool)
                    for z, (mul, ir) in zip(self.zero_flags, self.irreps)
                ]
            )
        zero_flags = [bool(np.all(zero_flags[s])) for s in irreps.slices()]

        new_chunks = None
        if self._chunks is not None:
            jnp = _infer_backend(self.array)
            leading_shape = self.shape[:-1]

            new_chunks = []
            current_array = 0

            while len(new_chunks) < len(irreps) and irreps[len(new_chunks)].mul == 0:
                new_chunks.append(None)

            for mul_ir, y in zip(self.irreps, self.chunks):
                mul, _ = mul_ir

                while mul > 0:
                    if isinstance(current_array, int):
                        current_mul = current_array
                    else:
                        current_mul = current_array.shape[-2]

                    needed_mul = irreps[len(new_chunks)].mul - current_mul

                    if mul <= needed_mul:
                        x = y
                        m = mul
                        mul = 0
                    elif mul > needed_mul:
                        if y is None:
                            x = None
                        else:
                            x, y = jnp.split(y, [needed_mul], axis=-2)
                        m = needed_mul
                        mul -= needed_mul

                    if x is None:
                        if isinstance(current_array, int):
                            current_array += m
                        else:
                            current_array = jnp.concatenate(
                                [
                                    current_array,
                                    jnp.zeros(
                                        leading_shape + (m, mul_ir.ir.dim), self.dtype
                                    ),
                                ],
                                axis=-2,
                            )
                    else:
                        if isinstance(current_array, int):
                            if current_array == 0:
                                current_array = x
                            else:
                                current_array = jnp.concatenate(
                                    [
                                        jnp.zeros(
                                            leading_shape
                                            + (current_array, mul_ir.ir.dim),
                                            self.dtype,
                                        ),
                                        x,
                                    ],
                                    axis=-2,
                                )
                        else:
                            current_array = jnp.concatenate([current_array, x], axis=-2)

                    if isinstance(current_array, int):
                        if current_array == irreps[len(new_chunks)].mul:
                            new_chunks.append(None)
                            current_array = 0
                    else:
                        if current_array.shape[-2] == irreps[len(new_chunks)].mul:
                            new_chunks.append(current_array)
                            current_array = 0

                    while (
                        len(new_chunks) < len(irreps)
                        and irreps[len(new_chunks)].mul == 0
                    ):
                        new_chunks.append(None)

            assert current_array == 0

            assert len(new_chunks) == len(irreps)
            for (mul, ir), x, z in zip(irreps, new_chunks, zero_flags):
                if z:
                    assert x is None
                else:
                    assert x.shape[-2:] == (mul, ir.dim)

        return IrrepsArray(irreps, self.array, zero_flags=zero_flags, chunks=new_chunks)

    def broadcast_to(self, shape) -> "IrrepsArray":
        """Broadcast the array to a new shape."""
        jnp = _infer_backend(self.array)

        assert isinstance(shape, tuple)
        assert shape[-1] == self.irreps.dim or shape[-1] == -1
        leading_shape = shape[:-1]
        array = jnp.broadcast_to(self.array, leading_shape + (self.irreps.dim,))
        chunks = [
            None if x is None else jnp.broadcast_to(x, leading_shape + x.shape[-2:])
            for x in self.chunks
        ]
        return IrrepsArray(
            self.irreps, array, zero_flags=self.zero_flags, chunks=chunks
        )


jax.tree_util.register_pytree_node(
    IrrepsArray,
    lambda x: ((x.array,), x.irreps),
    lambda irreps, data: IrrepsArray(irreps, data[0]),
)


def _standardize_axis(
    axis: Union[None, int, Tuple[int, ...]], result_ndim: int
) -> Tuple[int, ...]:
    if axis is None:
        return tuple(range(result_ndim))
    try:
        axis = (operator.index(axis),)
    except TypeError:
        axis = tuple(operator.index(i) for i in axis)

    if not all(-result_ndim <= i < result_ndim for i in axis):
        raise ValueError("axis out of range")
    axis = tuple(i % result_ndim for i in axis)

    return tuple(sorted(set(axis)))


class _IndexUpdateHelper:
    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index):
        return _IndexUpdateRef(self.irreps_array, index)


class _IndexUpdateRef:
    def __init__(self, irreps_array, index) -> None:
        self.irreps_array = irreps_array
        self.index = index

    def set(self, values: Any) -> IrrepsArray:
        pass

    def add(self, values: Any) -> IrrepsArray:
        index = self.index
        self = self.irreps_array

        if not isinstance(index, tuple):
            index = (index,)

        if isinstance(index[-1], (e3nn.Irrep, e3nn.MulIrrep, Irreps, str)):
            raise NotImplementedError('x.at[..., "1e + 2e"] is not implemented')

        if (
            (any(map(_is_ellipse, index[:-1])) or len(index) == self.ndim)
            and isinstance(index[-1], slice)
            and index[-1].step is None
            and isinstance(index[-1].start, (int, type(None)))
            and isinstance(index[-1].stop, (int, type(None)))
            and (index[-1].start is not None or index[-1].stop is not None)
        ):
            raise NotImplementedError("x.at[..., 3:32] is not implemented")

        if len(index) == self.ndim or any(map(_is_ellipse, index)):
            if not (_is_ellipse(index[-1]) or _is_none_slice(index[-1])):
                raise IndexError(
                    f"Indexing with {index[-1]} in the irreps dimension is not supported."
                )

        if isinstance(values, IrrepsArray):
            if self.irreps.simplify() != values.irreps.simplify():
                raise ValueError(
                    "The irreps of the array and the values to add must be the same."
                )

            values = values.rechunk(self.irreps)

            zero_flags = tuple(
                x and y for x, y in zip(self.zero_flags, values.zero_flags)
            )
            return IrrepsArray(
                self.irreps,
                self.array.at[index].add(values.array),
                zero_flags=zero_flags,
            )

        raise NotImplementedError(
            f"x.at[i].add(v) with v={type(values)} is not implemented."
        )


class _MulIndexSliceHelper:
    irreps_array: IrrepsArray

    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError(
                "IrrepsArray.slice_by_mul only supports one slices (like IrrepsArray.slice_by_mul[2:4])."
            )
        start, stop, stride = index.indices(self.irreps_array.irreps.num_irreps)
        if stride != 1:
            raise NotImplementedError(
                "IrrepsArray.slice_by_mul does not support strides."
            )

        irreps = []
        list = []
        i = 0
        for (mul, ir), x in zip(self.irreps_array.irreps, self.irreps_array.chunks):
            if start <= i and i + mul <= stop:
                irreps.append((mul, ir))
                list.append(x)
            elif start < i + mul and i < stop:
                irreps.append((min(stop, i + mul) - max(start, i), ir))
                list.append(x[..., max(start, i) - i : min(stop, i + mul) - i, :])

            i += mul
        return e3nn.from_chunks(
            irreps,
            list,
            self.irreps_array.shape[:-1],
            self.irreps_array.dtype,
            backend=_infer_backend(self.irreps_array.array),
        )


class _DimIndexSliceHelper:
    irreps_array: IrrepsArray

    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError(
                "IrrepsArray.slice_by_dim only supports slices (like IrrepsArray.slice_by_dim[2:4])."
            )
        return self.irreps_array[..., index]


class _ChunkIndexSliceHelper:
    irreps_array: IrrepsArray

    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError(
                "IrrepsArray.slice_by_chunk only supports slices (like IrrepsArray.slice_by_chunk[2:4])."
            )
        start, stop, stride = index.indices(len(self.irreps_array.irreps))

        return e3nn.from_chunks(
            self.irreps_array.irreps[start:stop:stride],
            self.irreps_array.chunks[start:stop:stride],
            self.irreps_array.shape[:-1],
            self.irreps_array.dtype,
            backend=_infer_backend(self.irreps_array.array),
        )

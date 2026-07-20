import collections
import dataclasses
import itertools
import math
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy

from e3nn_jax import (
    axis_angle_to_log_coordinates,
    generators,
    matrix_to_angles,
    perm,
    quaternion_to_angles,
)

from .J import Jd

IntoIrrep = Union[int, "Irrep", "MulIrrep", Tuple[int, int]]


@dataclasses.dataclass(init=False, frozen=True)
class Irrep:

    l: int
    p: int

    def __init__(self, l: IntoIrrep, p=None):
        """Initialize an Irrep."""
        if p is None:
            if isinstance(l, Irrep):
                p = l.p
                l = l.l

            if isinstance(l, MulIrrep):
                p = l.ir.p
                l = l.ir.l

            if isinstance(l, str):
                try:
                    name = l.strip()
                    l = int(name[:-1])
                    assert l >= 0
                    p = {
                        "e": 1,
                        "o": -1,
                        "y": (-1) ** l,
                    }[name[-1]]
                except Exception:
                    raise ValueError(f'unable to convert string "{name}" into an Irrep')
            elif isinstance(l, tuple):
                l, p = l

        assert isinstance(l, int) and l >= 0, l
        assert p in [-1, 1], p
        object.__setattr__(self, "l", l)
        object.__setattr__(self, "p", p)

    def __repr__(self):
        """Representation of the Irrep."""
        p = {+1: "e", -1: "o"}[self.p]
        return f"{self.l}{p}"

    @classmethod
    def iterator(cls, lmax=None):
        pass

    def D_from_log_coordinates(self, log_coordinates, k=0):
        pass

    def D_from_angles(self, alpha, beta, gamma, k=0):
        r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`.

        (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`.

        Args:
            alpha (`jax.Array`): of shape :math:`(...)`
                Rotation :math:`\alpha` around Y axis, applied third.
            beta (`jax.Array`): of shape :math:`(...)`
                Rotation :math:`\beta` around X axis, applied second.
            gamma (`jax.Array`): of shape :math:`(...)`
                Rotation :math:`\gamma` around Y axis, applied first.
            k (optional `jax.Array`): of shape :math:`(...)`
                How many times the parity is applied.

        Returns:
            `jax.Array`: of shape :math:`(..., 2l+1, 2l+1)`

        See Also:
            Irreps.D_from_angles
        """
        k = jnp.asarray(k)
        if isinstance(alpha, (int, float)) and alpha == 0:
            alpha = None
        else:
            alpha = jnp.asarray(alpha)

        if isinstance(beta, (int, float)) and beta == 0:
            beta = None
        else:
            beta = jnp.asarray(beta)

        if isinstance(gamma, (int, float)) and gamma == 0:
            gamma = None
        else:
            gamma = jnp.asarray(gamma)

        shape = jnp.broadcast_shapes(
            *[a.shape for a in [alpha, beta, gamma] if a is not None], k.shape
        )

        if alpha is not None:
            alpha = jnp.broadcast_to(alpha, shape)
        if beta is not None:
            beta = jnp.broadcast_to(beta, shape)
        if gamma is not None:
            gamma = jnp.broadcast_to(gamma, shape)
        k = jnp.broadcast_to(k, shape)

        return (
            _wigner_D_from_angles(self.l, alpha, beta, gamma)
            * self.p ** k[..., None, None]
        )

    def D_from_quaternion(self, q, k=0):
        pass

    def D_from_matrix(self, R):
        pass

    def D_from_axis_angle(self, axis, angle, k=0):
        pass

    def generators(self):
        r"""Generators of the representation of :math:`SO(3)`.

        Returns:
            `jax.Array`: array of shape :math:`(3, 2l+1, 2l+1)`

        See Also:
            `generators`
        """
        return generators(self.l)

    @property
    def dim(self) -> int:
        pass

    def is_scalar(self) -> bool:
        """Equivalent to ``l == 0 and p == 1``."""
        return self.l == 0 and self.p == 1

    def __mul__(self, other):
        r"""Generate the irreps from the product of two irreps.

        Returns:
            generator of `Irrep`
        """
        other = Irrep(other)
        p = self.p * other.p
        lmin = abs(self.l - other.l)
        lmax = self.l + other.l
        for l in range(lmin, lmax + 1):
            yield Irrep(l, p)

    def __rmul__(self, other):
        r"""Integer times the irrep.

        >>> 3 * Irrep('1e')
        3x1e
        """
        assert isinstance(other, int)
        return Irreps([(other, self)])

    def __add__(self, other):
        r"""Sum of two irreps."""
        return Irreps(self) + Irreps(other)

    def __radd__(self, other):
        r"""Sum of two irreps."""
        return Irreps(other) + Irreps(self)

    def __iter__(self):
        r"""Deconstruct the irrep into ``l`` and ``p``."""
        yield self.l
        yield self.p

    def __lt__(self, other):
        r"""Compare the order of two irreps."""
        return (self.l, -self.p * (-1) ** self.l) < (
            other.l,
            -other.p * (-1) ** other.l,
        )

    def __eq__(self, other: object) -> bool:
        """Compare two irreps."""
        other = Irrep(other)
        return (self.l, self.p) == (other.l, other.p)


jax.tree_util.register_pytree_node(Irrep, lambda ir: ((), ir), lambda ir, _: ir)


@dataclasses.dataclass(init=False, frozen=True)
class MulIrrep:

    mul: int
    ir: Irrep

    def __init__(self, mul, ir=None):
        r"""An irrep with a multiplicity."""
        if ir is None:
            mul, ir = mul

        object.__setattr__(self, "mul", mul)
        object.__setattr__(self, "ir", ir)

    @property
    def dim(self) -> int:
        pass

    def __repr__(self):
        """Representation of the irrep."""
        return f"{self.mul}x{self.ir}"

    def __iter__(self):
        """Deconstruct the mulirrep into ``mul`` and ``ir``."""
        yield self.mul
        yield self.ir

    def __lt__(self, other):
        """Compare the order of two mulirreps."""
        return (self.ir, self.mul) < (other.ir, other.mul)


jax.tree_util.register_pytree_node(
    MulIrrep, lambda mulir: ((), mulir), lambda mulir, _: mulir
)

IntoIrreps = Union[
    None,
    Irrep,
    MulIrrep,
    str,
    "Irreps",
    List[
        Union[
            str,
            Irrep,
            MulIrrep,
            Tuple[int, IntoIrrep],
        ]
    ],
]


class Irreps(tuple):

    def __new__(cls, irreps: IntoIrreps = None):
        r"""Create a new Irreps object."""
        if isinstance(irreps, Irreps):
            return super().__new__(cls, irreps)

        out: List[MulIrrep] = []
        if isinstance(irreps, Irrep):
            out.append(MulIrrep(1, Irrep(irreps)))
        elif irreps is None:
            pass
        elif isinstance(irreps, MulIrrep):
            out.append(irreps)
        elif isinstance(irreps, str):
            try:
                if irreps.strip() != "":
                    for mul_ir in irreps.split("+"):
                        if "x" in mul_ir:
                            mul, ir = mul_ir.split("x")
                            mul = int(mul)
                            ir = Irrep(ir)
                        else:
                            mul = 1
                            ir = Irrep(mul_ir)

                        assert isinstance(mul, int) and mul >= 0
                        out.append(MulIrrep(mul, ir))
            except Exception:
                raise ValueError(f'Unable to convert string "{irreps}" into an Irreps')
        else:
            for mul_ir in irreps:
                if isinstance(mul_ir, str):
                    mul = 1
                    ir = Irrep(mul_ir)
                elif isinstance(mul_ir, Irrep):
                    mul = 1
                    ir = mul_ir
                elif isinstance(mul_ir, MulIrrep):
                    mul, ir = mul_ir
                elif isinstance(mul_ir, int):
                    mul, ir = 1, Irrep(l=mul_ir, p=1)
                elif len(mul_ir) == 2:
                    mul, ir = mul_ir
                    ir = Irrep(ir)
                else:
                    mul = None
                    ir = None

                if not (isinstance(mul, int) and mul >= 0 and ir is not None):
                    raise ValueError(f'Unable to interpret "{mul_ir}" as an irrep.')

                out.append(MulIrrep(mul, ir))
        return super().__new__(cls, out)

    @staticmethod
    def spherical_harmonics(lmax, p=-1):
        r"""Representation of the spherical harmonics.

        Args:
            lmax (int): maximum :math:`l`
            p (optional {1, -1}): the parity of the representation

        Returns:
            `Irreps`: representation of :math:`(Y^0, Y^1, \dots, Y^{\mathrm{lmax}})`

        Examples:
            >>> Irreps.spherical_harmonics(3)
            1x0e+1x1o+1x2e+1x3o
            >>> Irreps.spherical_harmonics(4, p=1)
            1x0e+1x1e+1x2e+1x3e+1x4e
        """
        return Irreps([(1, (l, p**l)) for l in range(lmax + 1)])

    def slices(self) -> List[slice]:
        r"""List of slices corresponding to indices for each irrep.

        Examples:
            >>> Irreps('2x0e + 1e').slices()
            [slice(0, 2, None), slice(2, 5, None)]
        """
        s = []
        i = 0
        for mul_ir in self:
            s.append(slice(i, i + mul_ir.dim))
            i += mul_ir.dim
        return s

    def __getitem__(self, i) -> Union[MulIrrep, "Irreps"]:
        r"""Indexing."""
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return Irreps(x)
        return x

    def __contains__(self, ir) -> bool:
        r"""Check if an irrep is in the representation."""
        ir = Irrep(ir)
        return ir in (irrep for _, irrep in self)

    def count(self, ir: IntoIrrep) -> int:
        r"""Multiplicity of ``ir``.

        Args:
            ir (`Irrep`):

        Returns:
            `int`: total multiplicity of ``ir``

        Examples:
            >>> Irreps("2x0e + 3x1o").count("1o")
            3
        """
        ir = Irrep(ir)
        return sum(mul for mul, irrep in self if ir == irrep)

    def is_scalar(self) -> bool:
        r"""Check if the representation is scalar.

        Returns:
            `bool`: ``True`` if the representation is scalar

        Examples:
            >>> Irreps("2x0e + 3x1o").is_scalar()
            False
            >>> Irreps("2x0e + 2x0e").is_scalar()
            True
            >>> Irreps("0o").is_scalar()
            False
        """
        return {ir for _, ir in self} == {Irrep("0e")}

    def index(self, _object):  # noqa: D102
        raise NotImplementedError

    def __add__(self, irreps):
        r"""Add two representations."""
        irreps = Irreps(irreps)
        return Irreps(super().__add__(irreps))

    def __radd__(self, irreps):
        r"""Add two representations."""
        return Irreps(irreps) + self

    def __mul__(self, other):
        r"""Multiply the multiplicities of the irreps.

        Examples:
            >>> Irreps('0e + 1e') * 2
            2x0e+2x1e
        """
        if isinstance(other, Irreps):
            raise NotImplementedError(
                "Use e3nn.tensor_product for this, see the documentation"
            )
        return Irreps([(mul * other, ir) for mul, ir in self])

    def __rmul__(self, other):
        r"""Multiply the multiplicities of the irreps.

        Examples:
            >>> 2 * Irreps('0e + 1e')
            2x0e+2x1e
        """
        return self * other

    def __floordiv__(self, other):
        r"""Divide the multiplicities of the irreps.

        Examples:
            >>> Irreps('12x0e + 14x1e') // 2
            6x0e+7x1e
        """
        return Irreps([(mul // other, ir) for mul, ir in self])

    def __eq__(self, other: object) -> bool:
        r"""Check if two representations are equal."""
        if isinstance(other, str):
            try:
                other = Irreps(other)
            except ValueError:
                return False
        return super().__eq__(other)

    def __hash__(self) -> int:
        r"""Hash of the representation."""
        return super().__hash__()

    def repeat(self, n: int) -> "Irreps":
        r"""Repeat the representation ``n`` times.

        Examples:
            >>> Irreps('0e + 1e').repeat(2)
            1x0e+1x1e+1x0e+1x1e
        """
        return Irreps([(mul, ir) for mul, ir in self] * n)

    def unify(self) -> "Irreps":
        r"""Regroup same irrep together.

        Returns:
            `Irreps`: new `Irreps` object

        Examples:
            >>> Irreps('0e + 1e').unify()
            1x0e+1x1e

            >>> Irreps('0e + 1e + 1e').unify()
            1x0e+2x1e

            >>> Irreps('0e + 0x1e + 0e').unify()
            1x0e+0x1e+1x0e
        """
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            else:
                out.append((mul, ir))
        return Irreps(out)

    def remove_zero_multiplicities(self) -> "Irreps":
        """Remove any irreps with multiplicities of zero.

        Examples:
            >>> Irreps("4x0e + 0x1o + 2x3e").remove_zero_multiplicities()
            4x0e+2x3e
        """
        return Irreps([(mul, ir) for mul, ir in self if mul > 0])

    def simplify(self) -> "Irreps":
        """Simplify the representations.

        Examples:
            Note that simplify does not sort the representations.

            >>> Irreps("1e + 1e + 0e").simplify()
            2x1e+1x0e

            Equivalent representations which are separated from each other are not combined.

            >>> Irreps("1e + 1e + 0e + 1e").simplify()
            2x1e+1x0e+1x1e

            Except if they are separated by an irrep with multiplicity of zero.

            >>> Irreps("1e + 0x0e + 1e").simplify().simplify()
            2x1e
        """
        return self.remove_zero_multiplicities().unify()

    def sort(
        self,
    ) -> NamedTuple("Sort", irreps="Irreps", p=Tuple[int, ...], inv=Tuple[int, ...]):
        r"""Sort the representations.

        Returns:
            (tuple): tuple containing:

                irreps (`Irreps`): sorted irreps
                p (tuple of int): permutation of the indices
                inv (tuple of int): inverse permutation of the indices

        Examples:
            >>> Irreps("1e + 0e + 1e").sort().irreps
            1x0e+1x1e+1x1e
            >>> Irreps("2o + 1e + 0e + 1e").sort().p
            (3, 1, 0, 2)
            >>> Irreps("2o + 1e + 0e + 1e").sort().inv
            (2, 1, 3, 0)
        """
        Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
        out = [(ir, i, mul) for i, (mul, ir) in enumerate(self)]
        out = sorted(out)
        inv = tuple(i for _, i, _ in out)
        p = perm.inverse(inv)
        irreps = Irreps([(mul, ir) for ir, _, mul in out])
        return Ret(irreps, p, inv)

    def regroup(self) -> "Irreps":
        r"""Regroup the same irreps together.

        Equivalent to :meth:`sort` followed by :meth:`simplify`.

        Returns:
            `Irreps`: regrouped irreps

        Examples:
            >>> Irreps("1e + 0e + 1e + 0x2e").regroup()
            1x0e+2x1e
        """
        return self.sort().irreps.simplify()

    def set_mul(self, mul: int) -> "Irreps":
        pass

    def filter(
        self,
        keep: Union["Irreps", List[Irrep], Callable[[MulIrrep], bool]] = None,
        *,
        drop: Union["Irreps", List[Irrep], Callable[[MulIrrep], bool]] = None,
        lmax: int = None,
    ) -> "Irreps":
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

    @property
    def dim(self) -> int:
        pass

    @property
    def num_irreps(self) -> int:
        pass

    @property
    def mul_gcd(self) -> int:
        pass

    @property
    def ls(self) -> List[int]:
        pass

    @property
    def lmax(self) -> int:
        pass

    def __repr__(self):
        """Representation of the irreps."""
        if len(self) == 0:
            return "Irreps()"
        return "+".join(f"{mul_ir}" for mul_ir in self)

    def D_from_log_coordinates(self, log_coordinates, k=0):
        pass

    def D_from_angles(self, alpha, beta, gamma, k=0):
        r"""Compute the D matrix from the angles.

        Args:
            alpha (float): third rotation angle around the second axis (in radians)
            beta (float): second rotation angle around the first axis (in radians)
            gamma (float): first rotation angle around the second axis (in radians)
            k (int): parity operation

        Returns:
            `jax.Array`: array of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return jax.scipy.linalg.block_diag(
            *[
                ir.D_from_angles(alpha, beta, gamma, k)
                for mul, ir in self
                for _ in range(mul)
            ]
        )

    def D_from_quaternion(self, q, k=0):
        pass

    def D_from_matrix(self, R):
        pass

    def D_from_axis_angle(self, axis, angle, k=0):
        pass

    def generators(self) -> jax.Array:
        r"""Generators of the representation.

        Returns:
            `jax.Array`: array of shape :math:`(3, \mathrm{dim}, \mathrm{dim})`
        """
        return jax.vmap(jax.scipy.linalg.block_diag)(
            *[ir.generators() for mul, ir in self for _ in range(mul)]
        )


jax.tree_util.register_pytree_node(
    Irreps, lambda irreps: ((), irreps), lambda irreps, _: irreps
)


class _MulIndexSliceHelper:
    irreps: Irreps

    def __init__(self, irreps) -> None:
        self.irreps = irreps

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError("Irreps.slice_by_mul only supports slices.")

        start, stop, stride = index.indices(self.irreps.num_irreps)
        if stride != 1:
            raise NotImplementedError("Irreps.slice_by_mul does not support strides.")

        out = []
        i = 0
        for mul, ir in self.irreps:
            if start <= i and i + mul <= stop:
                out.append((mul, ir))
            elif start < i + mul and i < stop:
                out.append((min(stop, i + mul) - max(start, i), ir))
            i += mul
        return Irreps(out)


class _DimIndexSliceHelper:
    irreps: Irreps

    def __init__(self, irreps) -> None:
        self.irreps = irreps

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError("Irreps.slice_by_dim only supports slices.")

        start, stop, stride = index.indices(self.irreps.dim)
        if stride != 1:
            raise NotImplementedError("Irreps.slice_by_dim does not support strides.")

        out = []
        i = 0
        for mul, ir in self.irreps:
            if start <= i and i + mul * ir.dim <= stop:
                out.append((mul, ir))
            elif start < i + mul * ir.dim and i < stop:
                dim = min(stop, i + mul * ir.dim) - max(start, i)
                if dim % ir.dim != 0:
                    raise ValueError(
                        f"Error in Irreps.slice_by_dim: {start}:{stop} is not a valid slice for irreps {self.irreps} "
                        f"because it does not split {mul}x{ir} in an equivariant way."
                    )
                out.append((dim // ir.dim, ir))
            i += mul * ir.dim
        return Irreps(out)


class _ChunkIndexSliceHelper:
    irreps: Irreps

    def __init__(self, irreps) -> None:
        self.irreps = irreps

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError("Irreps.slice_by_chunk only supports slices.")

        return Irreps(self.irreps[index])


def _wigner_D_from_angles(
    l: int,
    alpha: Optional[jax.Array],
    beta: Optional[jax.Array],
    gamma: Optional[jax.Array],
) -> jax.Array:
    r"""The Wigner-D matrix of the real irreducible representations of :math:`SO(3)`.

    Args:
        l (int): the representation order of the irrep
        alpha (jax.Array): the first Euler angle
        beta (jax.Array): the second Euler angle
        gamma (jax.Array): the third Euler angle

    Returns:
        jax.Array: the Wigner-D matrix
    """
    shape = ()
    if alpha is not None:
        alpha = alpha % (2 * jnp.pi)
        shape = alpha.shape
    if beta is not None:
        beta = beta % (2 * jnp.pi)
        shape = beta.shape
    if gamma is not None:
        gamma = gamma % (2 * jnp.pi)
        shape = gamma.shape

    def f(a, b, c):
        def rot_y(phi):
            M = jnp.zeros((2 * l + 1, 2 * l + 1), dtype=phi.dtype)
            inds = jnp.arange(0, 2 * l + 1, 1)
            reversed_inds = jnp.arange(2 * l, -1, -1)
            frequencies = jnp.arange(l, -l - 1, -1.0, dtype=phi.dtype)
            M = M.at[inds, reversed_inds].set(jnp.sin(frequencies * phi))
            M = M.at[inds, inds].set(jnp.cos(frequencies * phi))
            return M

        R = []
        if a is not None:
            R += [rot_y(a)]

        if b is not None:
            if l < len(Jd):
                J = Jd[l].astype(b.dtype)
                R += [J @ rot_y(b) @ J]
            else:
                X = generators(l)
                R += [jax.scipy.linalg.expm(b.astype(X.dtype) * X[0]).astype(b.dtype)]

        if c is not None:
            R += [rot_y(c)]

        if len(R) == 0:
            return jnp.eye(2 * l + 1)

        r = R[0]
        for r_ in R[1:]:
            r = r @ r_
        return r

    f_vec = f
    for _ in range(len(shape)):
        f_vec = jax.vmap(f_vec)

    return f_vec(alpha, beta, gamma)


def _wigner_D_from_log_coordinates(l: int, log_coordinates: jax.Array) -> jax.Array:
    pass

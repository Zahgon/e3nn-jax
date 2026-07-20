import math
from typing import Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal
import scipy.spatial

import e3nn_jax as e3nn

from .activation import parity_function
from .so3 import change_basis_real_to_complex
from .spherical_harmonics.legendre import _sh_alpha, _sh_beta


class SphericalSignal:

    grid_values: jax.Array
    quadrature: str
    p_val: int
    p_arg: int

    def __init__(
        self,
        grid_values: jax.Array,
        quadrature: str,
        *,
        p_val: int = 1,
        p_arg: int = -1,
        _perform_checks: bool = True,
    ) -> None:
        if _perform_checks:
            if len(grid_values.shape) < 2:
                raise ValueError(
                    f"Grid values should have atleast 2 axes. Got grid_values of shape {grid_values.shape}."
                )

            if quadrature not in ["soft", "gausslegendre"]:
                raise ValueError(
                    f"Invalid quadrature for SphericalSignal: {quadrature}"
                )

            if p_val not in (-1, 1):
                raise ValueError(
                    f"Parity p_val must be either +1 or -1. Received: {p_val}"
                )

            if p_arg not in (-1, 1):
                raise ValueError(
                    f"Parity p_arg must be either +1 or -1. Received: {p_arg}"
                )

        self.grid_values = grid_values
        self.quadrature = quadrature
        self.p_val = p_val
        self.p_arg = p_arg

    @staticmethod
    def from_function(
        func: Callable[[jax.Array], float],
        res_beta: int,
        res_alpha: int,
        quadrature: str,
        *,
        p_val: int = 1,
        p_arg: int = -1,
        dtype: jnp.dtype = jnp.float32,
    ) -> "SphericalSignal":
        pass

    @staticmethod
    def zeros(
        res_beta: int,
        res_alpha: int,
        quadrature: str,
        *,
        p_val: int = 1,
        p_arg: int = -1,
        dtype: jnp.dtype = jnp.float32,
    ) -> "SphericalSignal":
        """Create a null signal on a grid."""
        return SphericalSignal(
            jnp.zeros((res_beta, res_alpha), dtype),
            quadrature,
            p_val=p_val,
            p_arg=p_arg,
        )

    def __repr__(self) -> str:
        if hasattr(self.grid_values, "ndim") and self.ndim >= 2:
            return (
                "SphericalSignal("
                f"shape={self.shape}, "
                f"res_beta={self.res_beta}, res_alpha={self.res_alpha}, "
                f"quadrature={self.quadrature}, p_val={self.p_val}, p_arg={self.p_arg})\n"
                f"{self.grid_values}"
            )
        else:
            return f"SphericalSignal({self.grid_values})"

    def __mul__(self, other: Union[float, "SphericalSignal"]) -> "SphericalSignal":
        """Multiply SphericalSignal by a scalar."""
        if isinstance(other, SphericalSignal):
            if self.quadrature != other.quadrature:
                raise ValueError(
                    "Multiplication of SphericalSignals with different quadrature is not supported."
                )
            if self.grid_resolution != other.grid_resolution:
                raise ValueError(
                    "Multiplication of SphericalSignals with different grid resolution is not supported."
                )
            if self.p_arg != other.p_arg:
                raise ValueError(
                    "Multiplication of SphericalSignals with different p_arg is not equivariant."
                )

            return SphericalSignal(
                self.grid_values * other.grid_values,
                self.quadrature,
                p_val=self.p_val * other.p_val,
                p_arg=self.p_arg,
            )

        if isinstance(other, e3nn.IrrepsArray):
            if other.irreps != e3nn.Irreps("0e"):
                raise ValueError("Scalar must be a 0e IrrepsArray.")
            other = other.array[..., 0]

        other = jnp.asarray(other)[..., None, None]
        return SphericalSignal(
            self.grid_values * other,
            self.quadrature,
            p_val=self.p_val,
            p_arg=self.p_arg,
        )

    def __rmul__(self, other: Union[float, "SphericalSignal"]) -> "SphericalSignal":
        """Multiply SphericalSignal by a compatible SphericalSignal or scalar."""
        return self * other

    def __truediv__(self, scalar: float) -> "SphericalSignal":
        """Divide SphericalSignal by a scalar."""
        return self * (1 / scalar)

    def __add__(self, other: "SphericalSignal") -> "SphericalSignal":
        """Add to another SphericalSignal."""
        if self.grid_resolution != other.grid_resolution:
            raise ValueError(
                "Grid resolutions for both signals must be identical. "
                "Use .resample() to change one of the grid resolutions."
            )
        if (self.p_val, self.p_arg) != (other.p_val, other.p_arg):
            raise ValueError("Parity for both signals must be identical.")
        if self.quadrature != other.quadrature:
            raise ValueError("Quadrature for both signals must be identical.")

        return SphericalSignal(
            self.grid_values + other.grid_values,
            self.quadrature,
            p_val=self.p_val,
            p_arg=self.p_arg,
        )

    def __sub__(self, other: "SphericalSignal") -> "SphericalSignal":
        """Subtract another SphericalSignal."""
        return self + (-other)

    def __neg__(self) -> "SphericalSignal":
        """Negate SphericalSignal."""
        return SphericalSignal(
            -self.grid_values, self.quadrature, p_val=self.p_val, p_arg=self.p_arg
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        pass

    @property
    def dtype(self) -> jnp.dtype:
        """Returns the dtype of this signal."""
        return self.grid_values.dtype

    @property
    def ndim(self) -> int:
        pass

    @property
    def grid_y(self) -> jax.Array:
        pass

    @property
    def grid_alpha(self) -> jax.Array:
        pass

    @property
    def grid_vectors(self) -> jax.Array:
        pass

    @property
    def quadrature_weights(self) -> jax.Array:
        pass

    @property
    def res_beta(self) -> int:
        pass

    @property
    def res_alpha(self) -> int:
        pass

    @property
    def grid_resolution(self) -> Tuple[int, int]:
        pass

    def resample(
        self, res_beta: int, res_alpha: int, lmax: int, quadrature: Optional[str] = None
    ) -> "SphericalSignal":
        pass

    def _transform_by(
        self,
        transform_type: str,
        transform_kwargs: Tuple[Union[float, int], ...],
        lmax: int,
    ) -> "SphericalSignal":
        """A wrapper for different transform_by functions."""
        coeffs = e3nn.from_s2grid(self, s2_irreps(lmax, self.p_val, self.p_arg))
        transforms = {
            "angles": coeffs.transform_by_angles,
            "matrix": coeffs.transform_by_matrix,
            "axis_angle": coeffs.transform_by_axis_angle,
            "quaternion": coeffs.transform_by_quaternion,
        }
        transformed_coeffs = transforms[transform_type](**transform_kwargs)
        return e3nn.to_s2grid(
            transformed_coeffs,
            *self.grid_resolution,
            quadrature=self.quadrature,
            p_val=self.p_val,
            p_arg=self.p_arg,
        )

    def transform_by_angles(
        self, alpha: float, beta: float, gamma: float, lmax: int
    ) -> "SphericalSignal":
        """Rotate the signal by the given Euler angles."""
        return self._transform_by(
            "angles",
            transform_kwargs=dict(alpha=alpha, beta=beta, gamma=gamma),
            lmax=lmax,
        )

    def transform_by_matrix(self, R: jax.Array, lmax: int) -> "SphericalSignal":
        """Rotate the signal by the given rotation matrix."""
        return self._transform_by("matrix", transform_kwargs=dict(R=R), lmax=lmax)

    def transform_by_axis_angle(
        self, axis: jax.Array, angle: float, lmax: int
    ) -> "SphericalSignal":
        pass

    def transform_by_quaternion(self, q: jax.Array, lmax: int) -> "SphericalSignal":
        pass

    def apply(self, func: Callable[[jax.Array], jax.Array]) -> "SphericalSignal":
        """Applies a function pointwise on the grid."""
        new_p_val = parity_function(func) if self.p_val == -1 else self.p_val
        if new_p_val == 0:
            raise ValueError(
                "Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd."
            )
        return self.replace_values(grid_values=func(self.grid_values))

    def replace_values(self, grid_values: jax.Array) -> "SphericalSignal":
        """Replace the grid values of the signal."""
        return SphericalSignal(
            grid_values, self.quadrature, p_val=self.p_val, p_arg=self.p_arg
        )

    @staticmethod
    def _find_peaks_2d(x: np.ndarray) -> List[Tuple[int, int]]:
        pass

    def find_peaks(self, lmax: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def pad_to_plot(
        self,
        *,
        translation: Optional[jax.Array] = None,
        radius: float = 1.0,
        scale_radius_by_amplitude: bool = False,
        normalize_radius_by_max_amplitude: bool = False,
    ) -> Tuple[jax.Array, jax.Array]:
        pass

    def plotly_surface(
        self,
        translation: Optional[jax.Array] = None,
        radius: float = 1.0,
        scale_radius_by_amplitude: bool = False,
        normalize_radius_by_max_amplitude: bool = False,
    ):
        pass

    def integrate(self) -> e3nn.IrrepsArray:
        pass

    def sample(self, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        pass

    def __getitem__(self, index) -> "SphericalSignal":
        grid_values = self.grid_values[index]

        if grid_values.ndim < 2:
            raise ValueError(
                "This indexing does not produce something that can be interpreted as a signal on the sphere. "
                "Consider using `SphericalSignal.grid_values` instead."
            )

        return SphericalSignal(
            grid_values=grid_values,
            quadrature=self.quadrature,
            p_val=self.p_val,
            p_arg=self.p_arg,
            _perform_checks=False,
        )


jax.tree_util.register_pytree_node(
    SphericalSignal,
    lambda x: ((x.grid_values,), (x.quadrature, x.p_val, x.p_arg)),
    lambda aux, grid_values: SphericalSignal(
        grid_values=grid_values[0],
        quadrature=aux[0],
        p_val=aux[1],
        p_arg=aux[2],
        _perform_checks=False,
    ),
)


def s2_dirac(
    position: Union[jax.Array, e3nn.IrrepsArray],
    lmax: int,
    *,
    p_val: int = 1,
    p_arg: int = -1,
) -> e3nn.IrrepsArray:
    r"""Spherical harmonics expansion of a Dirac delta on the sphere.

    The integral of the Dirac delta is 1.

    Args:
        position (`jax.Array` or `IrrepsArray`): position of the delta, shape ``(3,)``.
            It will be normalized to have a norm of 1.

        lmax (int): maximum degree of the spherical harmonics expansion
        p_val (int): parity of the value of the signal on the sphere (1 or -1)
        p_arg (int): parity of the argument of the signal on the sphere (1 or -1)

    Returns:
        `IrrepsArray`: Spherical harmonics coefficients

    Examples:

    .. jupyter-execute::
        :hide-code:

        import jax.numpy as jnp
        import e3nn_jax as e3nn
        import plotly.graph_objects as go

    .. jupyter-execute::

        position = jnp.array([0.0, 0.0, 1.0])

        coeffs_3 = e3nn.s2_dirac(position, 3, p_val=1, p_arg=-1)
        coeffs_6 = e3nn.s2_dirac(position, 6, p_val=1, p_arg=-1)
        coeffs_9 = e3nn.s2_dirac(position, 9, p_val=1, p_arg=-1)

    .. jupyter-execute::
        :hide-code:

        signal_3 = e3nn.to_s2grid(coeffs_3, 50, 69, quadrature="gausslegendre")
        signal_6 = e3nn.to_s2grid(coeffs_6, 50, 69, quadrature="gausslegendre")
        signal_9 = e3nn.to_s2grid(coeffs_9, 50, 69, quadrature="gausslegendre")

        axis = dict(
            # showbackground=False,
            # showgrid=False,
            # showline=False,
            showticklabels=False,
            # ticks="",
            title="",
        )
        go.Figure(
            [
                go.Surface(dict(**signal_3.plotly_surface(jnp.array([-2.1, 0, 0])), showscale=False)),
                go.Surface(dict(**signal_6.plotly_surface(), showscale=False)),
                go.Surface(dict(**signal_9.plotly_surface(jnp.array([2.1, 0, 0])), showscale=False)),
            ],
            layout=go.Layout(
                scene=dict(
                    xaxis=dict(range=[-3.1, 3.1], **axis),
                    yaxis=dict(range=[-1, 1], **axis),
                    zaxis=dict(range=[-1, 1], **axis),
                    camera=dict(
                        eye=dict(x=0.0, y=1.0, z=3.0),
                        up=dict(x=0.0, y=1.0, z=0.0),
                    ),
                    aspectratio=dict(x=3.1, y=1, z=1),
                ),
            ),
        )

    Note:

        To compute a sum of weighted Dirac deltas, use:

        .. jupyter-execute::

            positions = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
            weights = jnp.array([1, 1, -1, -1.0])

            e3nn.sum(e3nn.s2_dirac(positions, 4, p_val=1, p_arg=-1) * weights[:, None], axis=0)
    """
    irreps = s2_irreps(lmax, p_val, p_arg)
    coeffs = e3nn.spherical_harmonics(
        irreps, position, normalize=True, normalization="integral"
    )  # [dim]
    return coeffs / jnp.sqrt(4 * jnp.pi)


def s2_irreps(lmax: int, p_val: int = 1, p_arg: int = -1) -> e3nn.Irreps:
    r"""The Irreps of coefficients of a spherical harmonics expansion.

    .. math::

        f(\vec x) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_l^m Y_{l,m}(\vec x)

    When the inversion operator is applied to the signal, the new function :math:`I f` is given by

    .. math::

        [I f](\vec x) = p_{\text{val}} f(p_{\text{arg}} \vec x)

    Args:
        lmax (int): maximum degree of the expansion
        p_val (int): parity of the value of the signal on the sphere (1 or -1)
        p_arg (int): parity of the argument of the signal on the sphere (1 or -1)
    """
    return e3nn.Irreps([(1, (l, p_val * p_arg**l)) for l in range(lmax + 1)])


def get_s2fft_grid_resolution(lmax: int) -> Tuple[int, int]:
    """Returns the grid resolution for S2FFT."""
    return (2 * lmax + 2, 2 * lmax + 1)


def _check_compatibility_with_s2fft(
    lmax: int, res_beta: int, res_alpha: int, quadrature: str, fft: bool
) -> None:
    """Check if the inputs are compatible with S2FFT."""

    if quadrature != "soft":
        raise ValueError("Please supply quadrature='soft' to use S2FFT.")

    expected_grid_resolution = get_s2fft_grid_resolution(lmax)
    if (res_beta, res_alpha) != expected_grid_resolution:
        raise ValueError(
            f"Invalid shape {(res_beta, res_alpha)} for the signal. Expected shape: {expected_grid_resolution}."
        )

    if not fft:
        raise ValueError("Please supply fft=True to use S2FFT.")

    return True


def from_s2grid(
    x: SphericalSignal,
    irreps: e3nn.Irreps,
    *,
    normalization: str = "integral",
    lmax_in: Optional[int] = None,
    fft: bool = True,
    use_s2fft: bool = False,
) -> e3nn.IrrepsArray:
    r"""Transform signal on the sphere into spherical harmonics coefficients.

    The output has degree :math:`l` between 0 and lmax, and parity :math:`p = p_{val}p_{arg}^l`

    The inverse transformation of :func:`to_s2grid`

    Args:
        x (`SphericalSignal`): signal on the sphere of shape ``(..., y/beta, alpha)``
        irreps (`Irreps`): irreps of the coefficients
        normalization ({'norm', 'component', 'integral'}): normalization of the spherical harmonics basis
        lmax_in (int, optional): maximum degree of the input signal, only used for normalization purposes
        fft (bool): True if we use FFT, False if we use the naive implementation

    Returns:
        `IrrepsArray`: coefficient array of shape ``(..., (lmax+1)^2)``
    """
    res_beta, res_alpha = x.grid_resolution

    irreps = e3nn.Irreps(irreps)

    if not all(mul == 1 for mul, _ in irreps.regroup()):
        raise ValueError("Multiplicities of all irreps should be ones.")

    _check_parities(irreps, x.p_val, x.p_arg)

    lmax = max(irreps.ls)

    if lmax_in is None:
        lmax_in = lmax

    if use_s2fft:
        _check_compatibility_with_s2fft(
            lmax, x.res_beta, x.res_alpha, x.quadrature, fft
        )
        return _from_s2grid_s2fft(x, irreps, normalization=normalization)

    with jax.ensure_compile_time_eval():
        _, _, sh_y, sha, qw = _spherical_harmonics_s2grid(
            lmax, res_beta, res_alpha, quadrature=x.quadrature, dtype=x.dtype
        )

        n = _normalization(lmax, normalization, x.dtype, "from_s2", lmax_in)

        m_in = jnp.asarray(_expand_matrix(range(lmax + 1)), x.dtype)  # [l, m, j]
        m_out = jnp.asarray(_expand_matrix(irreps.ls), x.dtype)  # [l, m, i]
        sh_y = _rollout_sh(sh_y, lmax)
        sh_y = jnp.einsum("lmj,bj,lmi,l,b->mbi", m_in, sh_y, m_out, n, qw)  # [m, b, i]

    if fft:
        int_a = _rfft(x.grid_values, lmax) / res_alpha  # [..., res_beta, 2*l+1]
    else:
        int_a = (
            jnp.einsum("...ba,am->...bm", x.grid_values, sha) / res_alpha
        )  # [..., res_beta, 2*l+1]

    int_b = jnp.einsum("mbi,...bm->...i", sh_y.astype(x.dtype), int_a)  # [..., irreps]

    return e3nn.IrrepsArray(irreps, int_b)


def _from_s2grid_s2fft(
    sig: SphericalSignal,
    irreps: e3nn.Irreps,
    *,
    normalization: str = "integral",
) -> e3nn.IrrepsArray:
    """An S2FFT powered version of e3nn_jax.from_s2grid."""
    import s2fft

    lmax = irreps.lmax
    expected_grid_resolution = get_s2fft_grid_resolution(lmax)
    if sig.grid_resolution != expected_grid_resolution:
        raise ValueError(
            f"Input signal resolution {sig.grid_resolution} does not match the required resolution {expected_grid_resolution}."
        )

    with jax.ensure_compile_time_eval():
        precomps = s2fft.generate_precomputes_jax(
            L=lmax + 1, forward=True, sampling="dh"
        )

    def _from_s2grid_s2fft_single_dim(sig: SphericalSignal) -> e3nn.IrrepsArray:
        pass

    _from_s2grid_s2fft_func = _from_s2grid_s2fft_single_dim
    for _ in range(sig.ndim - 2):
        _from_s2grid_s2fft_func = jax.vmap(_from_s2grid_s2fft_func)
    return _from_s2grid_s2fft_func(sig)


def to_s2grid(
    coeffs: e3nn.IrrepsArray,
    res_beta: int,
    res_alpha: int,
    *,
    quadrature: str,
    normalization: str = "integral",
    fft: bool = True,
    p_val: Optional[int] = None,
    p_arg: Optional[int] = None,
    use_s2fft: bool = False,
) -> SphericalSignal:
    r"""Sample a signal on the sphere given by the coefficient in the spherical harmonics basis.

    The inverse transformation of :func:`from_s2grid`

    Args:
        coeffs (`IrrepsArray`): coefficient array
        res_beta (int): number of points on the sphere in the :math:`\theta` direction
        res_alpha (int): number of points on the sphere in the :math:`\phi` direction
        normalization ({'norm', 'component', 'integral'}): normalization of the basis
        quadrature (str): "soft" or "gausslegendre"
        fft (bool): True if we use FFT, False if we use the naive implementation
        p_val (int, optional): parity of the value of the signal
        p_arg (int, optional): parity of the argument of the signal

    Returns:
        `SphericalSignal`: signal on the sphere of shape ``(..., y/beta, alpha)``

    Note:

        We use a rectangular grid for the :math:`\beta` and :math:`\alpha` angles.
        The grid is uniform in the :math:`\alpha` angle while for :math:`\beta`, two different quadratures are available:

        * The `soft <https://link.springer.com/article/10.1007/s00041-008-9013-5>`_
          quadrature is a uniform sampling of the beta angle.
        * The `gausslegendre <https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature>`_
          quadrature is a quadrature rule that is exact for polynomials of degree ``2 res_beta - 1``.

        The gausslegendre quadrature allows exact integration of spherical harmonics upto a degree l,
        if res_beta is atleast (l + 3)/2 and res_alpha is atleast l + 1.
        See https://cbeentjes.github.io/files/Ramblings/QuadratureSphere.pdf for more information on
        quadrature rules for the sphere. In their notation, alpha is theta and beta is phi.

        .. jupyter-execute::
            :hide-code:

            import jax.numpy as jnp
            import e3nn_jax as e3nn
            import plotly.graph_objects as go

            soft = e3nn.SphericalSignal.zeros(10, 19, "soft")
            gauss = e3nn.SphericalSignal.zeros(10, 19, "gausslegendre")

            axis = dict(showticklabels=False, title="", range=[-1.1, 1.1])

            go.Figure(
                data=[
                    go.Scatter3d(
                        x=s.grid_vectors[:, :, 0].reshape(-1),
                        y=s.grid_vectors[:, :, 1].reshape(-1),
                        z=s.grid_vectors[:, :, 2].reshape(-1),
                        mode="markers",
                        marker=dict(
                            color=c,
                            size=100 * jnp.broadcast_to(s.quadrature_weights[:, None], s.grid_resolution).reshape(-1)
                        ),
                        name=s.quadrature,
                    )
                    for s, c in zip([soft, gauss], ["blue", "red"])
                ],
                layout=go.Layout(
                    scene=dict(
                        xaxis=axis,
                        yaxis=axis,
                        zaxis=axis,
                        camera=dict(
                            eye=dict(x=0.4, y=0.0, z=1.3),
                            up=dict(x=0.0, y=1.0, z=0.0),
                        ),
                    ),
                ),
            )
    """
    coeffs = coeffs.regroup()
    lmax = coeffs.irreps.ls[-1]

    if not all(mul == 1 for mul, _ in coeffs.irreps):
        raise ValueError(f"Multiplicities should be ones. Got {coeffs.irreps}.")

    if (p_val is not None) != (p_arg is not None):
        raise ValueError("p_val and p_arg should be both None or both not None.")

    p_val, p_arg = _check_parities(coeffs.irreps, p_val, p_arg)

    if p_val is None or p_arg is None:
        raise ValueError(
            f"p_val and p_arg cannot be determined from the irreps {coeffs.irreps}, please specify them."
        )

    if use_s2fft:
        _check_compatibility_with_s2fft(lmax, res_beta, res_alpha, quadrature, fft)
        return _to_s2grid_s2fft(
            coeffs,
            res_beta,
            res_alpha,
            normalization=normalization,
            p_val=p_val,
            p_arg=p_arg,
        )

    with jax.ensure_compile_time_eval():
        _, _, sh_y, sha, _ = _spherical_harmonics_s2grid(
            lmax, res_beta, res_alpha, quadrature=quadrature, dtype=coeffs.dtype
        )

        n = _normalization(lmax, normalization, coeffs.dtype, "to_s2")

        m_in = jnp.asarray(_expand_matrix(range(lmax + 1)), coeffs.dtype)  # [l, m, j]
        m_out = jnp.asarray(_expand_matrix(coeffs.irreps.ls), coeffs.dtype)  # [l, m, i]
        sh_y = _rollout_sh(sh_y, lmax)
        sh_y = jnp.einsum("lmj,bj,lmi,l->mbi", m_in, sh_y, m_out, n)  # [m, b, i]

    signal_b = jnp.einsum(
        "mbi,...i->...bm", sh_y.astype(coeffs.dtype), coeffs.array
    )  # [batch, beta, m]

    if fft:
        if res_alpha % 2 == 0:
            raise ValueError("res_alpha must be odd for fft")

        signal = _irfft(signal_b, res_alpha) * res_alpha  # [..., res_beta, res_alpha]
    else:
        signal = jnp.einsum(
            "...bm,am->...ba", signal_b, sha
        )  # [..., res_beta, res_alpha]

    return SphericalSignal(signal, quadrature=quadrature, p_val=p_val, p_arg=p_arg)


def _to_s2grid_s2fft(
    coeffs: e3nn.IrrepsArray,
    res_beta: int,
    res_alpha: int,
    *,
    normalization: str = "integral",
    p_val: Optional[int] = None,
    p_arg: Optional[int] = None,
) -> SphericalSignal:
    """An S2FFT powered version of e3nn_jax.to_s2grid."""
    import s2fft

    lmax = coeffs.irreps.lmax
    expected_grid_resolution = get_s2fft_grid_resolution(lmax)
    if (res_beta, res_alpha) != expected_grid_resolution:
        raise ValueError(
            f"To use S2FFT, the input grid must have res_beta={expected_grid_resolution[0]}, "
            f"res_alpha={expected_grid_resolution[1]}."
        )

    def _to_s2grid_s2fft_single_dim(coeffs: e3nn.IrrepsArray) -> SphericalSignal:
        """An S2FFT powered version of e3nn_jax.to_s2grid for a single signal."""
        coeffs_reshaped = jnp.zeros((lmax + 1, 2 * lmax + 1), dtype=complex)
        normalization_factors = _normalization(
            lmax, normalization, coeffs.dtype, "to_s2"
        )

        for l in range(lmax + 1):
            r = coeffs.array[l**2 : (l + 1) ** 2]
            r = r * normalization_factors[l]
            m = jnp.arange(-l, l + 1)
            r = r * (-1) ** jnp.where(m < 0, m + 1, m)
            A = change_basis_real_to_complex(l)
            c = 1j ** (-l) * A @ r
            coeffs_reshaped = coeffs_reshaped.at[l, lmax - l : lmax + l + 1].set(c)

        with jax.ensure_compile_time_eval():
            precomps = s2fft.generate_precomputes_jax(
                L=lmax + 1, forward=False, sampling="dh"
            )

        f = s2fft.transforms.spherical.inverse_jax(
            coeffs_reshaped, L=lmax + 1, sampling="dh", reality=True, precomps=precomps
        )
        return e3nn.SphericalSignal(f, quadrature="soft", p_val=p_val, p_arg=p_arg)

    _to_s2grid_s2fft_single_dim = _to_s2grid_s2fft_single_dim
    for _ in range(coeffs.ndim - 1):
        _to_s2grid_s2fft_single_dim = jax.vmap(_to_s2grid_s2fft_single_dim)
    return _to_s2grid_s2fft_single_dim(coeffs)


def legendre_transform_to_s2grid(
    coeffs: jax.Array,
    res_beta: int,
    *,
    quadrature: str,
    normalization: str = "integral",
) -> jax.Array:
    r"""Sample a signal along `beta` on the sphere given by the m=0 coefficients in the spherical harmonics basis.

    The inverse transformation of :func:`legendre_transform_from_s2grid`

    Args:
        coeffs (`jax.Array`): coefficient array of shape ``(lmax+1,)``
        res_beta (int): number of points on the sphere in the :math:`\theta` direction
        normalization ({'norm', 'component', 'integral'}): normalization of the basis
        quadrature (str): "soft" or "gausslegendre"

    Returns:
        `jax.Array`: signal on the sphere of shape ``(y/beta,)``
    """
    lmax = coeffs.shape[-1] - 1

    y, _ = _quadrature_weights(res_beta, quadrature=quadrature)
    sh_y = _sh_beta(lmax, y)

    n = _normalization(lmax, normalization, coeffs.dtype, "to_s2")
    sh_y_m0 = sh_y[:, :, 0]
    sh_y_qw_m0 = jnp.einsum("bi,i->bi", sh_y_m0, n)

    return jnp.einsum("bi,...i->...b", sh_y_qw_m0, coeffs)


def legendre_transform_from_s2grid(
    x_beta: jax.Array,
    lmax: int,
    res_beta: int,
    *,
    quadrature: str,
    normalization: str = "integral",
) -> jax.Array:
    r"""
    Transform signal on the sphere, and return the m=0 spherical harmonic components.
    Args:
        x_beta (`jax.Array`): signal on the sphere along beta; shape ``(y/beta,)``
        lmax (int): maximum l of the resulting irreps
        res_beta (int): number of points on the sphere in the :math:`\theta` direction
        quadrature (str): "soft" or "gausslegendre"
        normalization ({'norm', 'component', 'integral'}): normalization of the spherical harmonics basis

    Returns:
        `jax.Array`: coefficient array of shape ``(..., lmax+1)``
    """
    assert res_beta == x_beta.shape[-1]

    y, qw = _quadrature_weights(res_beta, quadrature=quadrature)
    sh_y = _sh_beta(lmax, y)

    n = _normalization(lmax, normalization, x_beta.dtype, "from_s2", lmax)

    sh_y_m0 = sh_y[:, :, 0]
    sh_y_qw_m0 = jnp.einsum("bi,i,b->bi", sh_y_m0, n, qw)

    x_prime = jnp.einsum("bi,...b->...i", sh_y_qw_m0, x_beta)
    return x_prime


def betas_to_spherical_signal(x_beta, res_alpha, *, quadrature) -> SphericalSignal:
    """
    Convert signals along beta to a SphericalSignal symmetric about alpha.

    Args:
        x_beta (`jax.Array`): signal on the sphere along beta; shape ``(...,y/beta)``
        res_alpha (int)
        quadrature (str): either "soft" or "gausslegendre"
    Returns:
        `SphericalSignal`: signal on the sphere of shape ``(y/beta, alpha)``
    """
    return SphericalSignal(
        jnp.repeat(jnp.expand_dims(x_beta, axis=-1), res_alpha, axis=-1), quadrature
    )


def m0_values_to_irrepsarray(m0_values, lmax, p_val, p_arg) -> e3nn.IrrepsArray:
    """
    Convert m=0 spherical harmonic components to an IrrepsArray.

    Args:
        m0_values (`jax.Array`): values along m=0, shape (..., lmax+1)
        lmax (int): maximum l of the resulting irreps
        p_val (int): parity of the value of the signal
        p_arg (int): parity of the argument of the signal
    Returns:
        `e3nn.IrrepsArray`: IrrepsArray with `irreps` and values `m0_values`
    """
    m0_indices = jnp.cumsum(jnp.repeat(jnp.arange(lmax + 1), 2))[::2] + jnp.arange(
        lmax + 1
    )
    irreps = s2_irreps(lmax, p_val, p_arg)
    m0 = jnp.zeros((*m0_values.shape[:-1], (lmax + 1) ** 2))
    m0 = m0.at[:, m0_indices].set(m0_values)
    return e3nn.IrrepsArray(irreps, m0)


def to_s2point(
    coeffs: e3nn.IrrepsArray,
    point: e3nn.IrrepsArray,
    *,
    normalization: str = "integral",
) -> e3nn.IrrepsArray:
    """Evaluate a signal on the sphere given by the coefficient in the spherical harmonics basis.

    It computes the same thing as :func:`to_s2grid` but at a single point.

    Args:
        coeffs (`IrrepsArray`): coefficient array of shape ``(*shape1, irreps)``
        point (`jax.Array`): point on the sphere of shape ``(*shape2, 3)``
        normalization ({'norm', 'component', 'integral'}): normalization of the basis

    Returns:
        `IrrepsArray`: signal on the sphere of shape ``(*shape1, *shape2, irreps)``
    """
    coeffs = coeffs.regroup()

    if not all(mul == 1 for mul, _ in coeffs.irreps):
        raise ValueError(f"Multiplicities should be ones. Got {coeffs.irreps}.")

    if not isinstance(point, e3nn.IrrepsArray):
        raise TypeError(f"point should be an e3nn.IrrepsArray, got {type(point)}.")

    if point.irreps not in ["1e", "1o"]:
        raise ValueError(f"point should be of irreps '1e' or '1o', got {point.irreps}.")

    p_arg = point.irreps[0].ir.p
    p_val, _ = _check_parities(coeffs.irreps, None, p_arg)

    sh = e3nn.spherical_harmonics(
        coeffs.irreps.ls, point, True, "integral"
    )  # [*shape2, irreps]
    n = _normalization(sh.irreps.lmax, normalization, coeffs.dtype, "to_s2")[
        jnp.array(sh.irreps.ls)
    ]  # [num_irreps]
    sh = sh * n

    shape1 = coeffs.shape[:-1]
    coeffs = coeffs.reshape((-1, coeffs.shape[-1]))
    shape2 = point.shape[:-1]
    sh = sh.reshape((-1, sh.shape[-1]))

    irreps = {1: "0e", -1: "0o"}[p_val]
    return e3nn.IrrepsArray(
        irreps,
        jnp.einsum("ai,bi->ab", coeffs.array, sh.array).reshape(shape1 + shape2 + (1,)),
    )


def _s2grid_vectors(y: jax.Array, alpha: jax.Array) -> jax.Array:
    pass


def _quadrature_weights_soft(b: int) -> np.ndarray:
    r"""function copied from ``lie_learn.spaces.S3``
    Compute quadrature weights for the grid used by Kostelec & Rockmore [1, 2].
    """
    assert (
        b % 2 == 0
    ), "res_beta needs to be even for soft quadrature weights to be computed properly"
    k = np.arange(b // 2)
    return np.array(
        [
            (
                (4.0 / b)
                * np.sin(np.pi * (2.0 * j + 1.0) / (2.0 * b))
                * (
                    (1.0 / (2 * k + 1))
                    * np.sin((2 * j + 1) * (2 * k + 1) * np.pi / (2.0 * b))
                ).sum()
            )
            for j in np.arange(b)
        ],
    )


def _s2grid(
    res_beta: int, res_alpha: int, quadrature: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Returns arrays describing the grid on the sphere.

    Args:
        res_beta (int): :math:`N`
        res_alpha (int): :math:`M`
        quadrature (str): "soft" or "gausslegendre"

    Returns:
        (tuple): tuple containing:
            y (`numpy.ndarray`): array of shape ``(res_beta)``
            alpha (`numpy.ndarray`): array of shape ``(res_alpha)``
            qw (`numpy.ndarray`): array of shape ``(res_beta)``, ``sum(qw) = 1``
    """

    y, qw = _quadrature_weights(res_beta, quadrature=quadrature)

    i = np.arange(res_alpha)
    alpha = i / res_alpha * 2 * np.pi
    return y, alpha, qw


def _quadrature_weights(
    res_beta: int, *, quadrature: str
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Returns quadrature weights for the grid on the sphere.

    Args:
        res_beta (int): :math:`N`
        quadrature (str): "soft" or "gausslegendre"

    Returns:
        (tuple): tuple containing:
            y (`numpy.ndarray`): array of shape ``(res_beta)``
            qw (`numpy.ndarray`): array of shape ``(res_beta)``, ``sum(qw) = 1``
    """
    if quadrature == "soft":
        i = np.arange(res_beta)
        betas = (i + 0.5) / res_beta * np.pi
        y = -np.cos(betas)  # minus sign is here to go from -1 to 1 in both quadratures
        qw = _quadrature_weights_soft(res_beta)
    elif quadrature == "gausslegendre":
        y, qw = np.polynomial.legendre.leggauss(res_beta)
    else:
        raise Exception("quadrature needs to be 'soft' or 'gausslegendre'")
    qw /= 2.0
    return y, qw


def _spherical_harmonics_s2grid(
    lmax: int,
    res_beta: int,
    res_alpha: int,
    *,
    quadrature: str,
    dtype: np.dtype = np.float32,
):
    r"""spherical harmonics evaluated on the grid on the sphere
    .. math::
        f(x) = \sum_{l=0}^{l_{\mathit{max}}} F^l \cdot Y^l(x)
        f(\beta, \alpha) = \sum_{l=0}^{l_{\mathit{max}}} F^l \cdot S^l(\alpha) P^l(\cos(\beta))
    Args:
        lmax (int): :math:`l_{\mathit{max}}`
        res_beta (int): :math:`N`
        res_alpha (int): :math:`M`
        quadrature (str): "soft" or "gausslegendre"

    Returns:
        (tuple): tuple containing:
            y (`jax.Array`): array of shape ``(res_beta)``
            alphas (`jax.Array`): array of shape ``(res_alpha)``
            sh_y (`jax.Array`): array of shape ``(res_beta, lmax + 1, lmax + 1)``
            sh_alpha (`jax.Array`): array of shape ``(res_alpha, 2 * lmax + 1)``
            qw (`jax.Array`): array of shape ``(res_beta)``
    """
    y, alphas, qw = _s2grid(res_beta, res_alpha, quadrature)
    y, alphas, qw = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype), (y, alphas, qw)
    )
    sh_alpha = _sh_alpha(lmax, alphas)  # [..., 2 * l + 1]
    sh_y = _sh_beta(lmax, y)  # [..., l, m]
    return y, alphas, sh_y, sh_alpha, qw


def _check_parities(
    irreps: e3nn.Irreps, p_val: Optional[int] = None, p_arg: Optional[int] = None
) -> Tuple[int, int]:
    p_even = {ir.p for mul, ir in irreps if ir.l % 2 == 0}
    p_odd = {ir.p for mul, ir in irreps if ir.l % 2 == 1}
    if not (p_even in [{1}, {-1}, set()] and p_odd in [{1}, {-1}, set()]):
        raise ValueError(
            "irrep parities should be of the form (p_val * p_arg**l) for all l, where p_val and p_arg are ±1"
        )

    p_even = p_even.pop() if p_even else None
    p_odd = p_odd.pop() if p_odd else None

    if p_val is not None and p_arg is not None:
        if not (p_even in [p_val, None] and p_odd in [p_val * p_arg, None]):
            raise ValueError(
                f"irrep ({irreps}) parities are not compatible with the given p_val ({p_val}) and p_arg ({p_arg})."
            )
        return p_val, p_arg

    if p_val is not None:
        if p_even is None:
            p_even = p_val
        if p_even != p_val:
            raise ValueError(
                f"irrep ({irreps}) parities are not compatible with the given p_val ({p_val})."
            )

    if p_arg is not None:
        if p_odd is None and p_even is not None:
            p_odd = p_even * p_arg
        elif p_odd is not None and p_even is None:
            p_even = p_odd * p_arg
        elif p_odd is not None and p_even is not None:
            if p_odd != p_even * p_arg:
                raise ValueError(
                    f"irrep ({irreps}) parities are not compatible with the given p_arg ({p_arg})."
                )

    if p_even is not None and p_odd is not None:
        return p_even, p_even * p_odd

    return p_even, None


def _normalization(
    lmax: int, normalization: str, dtype, direction: str, lmax_in: Optional[int] = None
) -> jax.Array:
    """Handles normalization of different components of IrrepsArrays."""
    assert direction in ["to_s2", "from_s2"]

    if normalization == "component":
        if direction == "to_s2":
            return jnp.sqrt(4 * jnp.pi) / (
                (jnp.sqrt(2 * jnp.arange(lmax + 1) + 1)).astype(dtype)
                * jnp.sqrt(lmax + 1)
            )
        else:
            return jnp.sqrt(4 * jnp.pi) * (
                (jnp.sqrt(2 * jnp.arange(lmax + 1) + 1)).astype(dtype)
                * jnp.sqrt(lmax + 1)
            )
    if normalization == "norm":
        if direction == "to_s2":
            return jnp.sqrt(4 * jnp.pi) * jnp.ones(lmax + 1, dtype) / jnp.sqrt(lmax + 1)
        else:
            return (
                jnp.sqrt(4 * jnp.pi) * jnp.ones(lmax + 1, dtype) * jnp.sqrt(lmax_in + 1)
            )
    if normalization == "integral":
        return jnp.ones(lmax + 1, dtype) * jnp.sqrt(4 * jnp.pi)

    raise Exception("normalization needs to be 'norm', 'component' or 'integral'")


def _rfft(x: jax.Array, l: int) -> jax.Array:
    r"""Real fourier transform
    Args:
        x (`jax.Array`): input array of shape ``(..., res_beta, res_alpha)``
        l (int): value of `l` for which the transform is being run
    Returns:
        `jax.Array`: transformed values - array of shape ``(..., res_beta, 2*l+1)``
    """
    x_reshaped = x.reshape((-1, x.shape[-1]))
    x_transformed_c = jnp.fft.rfft(x_reshaped)  # (..., 2*l+1)
    x_transformed = jnp.concatenate(
        [
            jnp.flip(jnp.imag(x_transformed_c[..., 1 : l + 1]), -1) * -jnp.sqrt(2),
            jnp.real(x_transformed_c[..., :1]),
            jnp.real(x_transformed_c[..., 1 : l + 1]) * jnp.sqrt(2),
        ],
        axis=-1,
    )
    return x_transformed.reshape((*x.shape[:-1], 2 * l + 1))


def _irfft(x: jax.Array, res: int) -> jax.Array:
    r"""Inverse of the real fourier transform
    Args:
        x (`jax.Array`): array of shape ``(..., 2*l + 1)``
        res (int): output resolution, has to be an odd number
    Returns:
        `jax.Array`: positions on the sphere, array of shape ``(..., res)``
    """
    assert res % 2 == 1

    l = (x.shape[-1] - 1) // 2
    x_reshaped = jnp.concatenate(
        [
            x[..., l : l + 1],
            (x[..., l + 1 :] + jnp.flip(x[..., :l], -1) * -1j) / jnp.sqrt(2),
            jnp.zeros((*x.shape[:-1], l), x.dtype),
        ],
        axis=-1,
    ).reshape((-1, x.shape[-1]))
    x_transformed = jnp.fft.irfft(x_reshaped, res)
    return x_transformed.reshape((*x.shape[:-1], x_transformed.shape[-1]))


def _expand_matrix(ls: List[int]) -> np.ndarray:
    """
    conversion matrix between a flatten vector (L, m) like that
    (0, 0) (1, -1) (1, 0) (1, 1) (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)
    and a bidimensional matrix representation like that
                    (0, 0)
            (1, -1) (1, 0) (1, 1)
    (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)

    Args:
        ls: list of l values
    Returns:
        array of shape ``[l, m, l * m]``
    """
    lmax = max(ls)
    m = np.zeros((lmax + 1, 2 * lmax + 1, sum(2 * l + 1 for l in ls)), np.float64)
    i = 0
    for l in ls:
        m[l, lmax - l : lmax + l + 1, i : i + 2 * l + 1] = np.eye(
            2 * l + 1, dtype=np.float64
        )
        i += 2 * l + 1
    return m


def _rollout_sh(input: jax.Array, lmax: int) -> jax.Array:
    """
    Input:
        [[(0,0)            ]       l=0
         [(1,0) (1,1)      ]       l=1
         [(2,0) (2,1) (2,2)]]      l=2
    Output:
        [(0,0) (1,1) (1,0) (1,1) (2,2) (2,1) (2,0) (2,1) (2,2)]
    """
    assert input.shape[-2] == lmax + 1  # l
    assert input.shape[-1] == lmax + 1  # abs(m)
    ls = []
    ms = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            ls.append(l)
            ms.append(abs(m))
    ls = jnp.asarray(ls)
    ms = jnp.asarray(ms)
    return input[..., ls, ms]

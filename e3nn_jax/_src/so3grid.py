from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn

from .s2grid import SphericalSignal


class SO3Signal:

    def __init__(
        self,
        s2_signals: SphericalSignal,
        *,
        _perform_checks: bool = True,
    ) -> None:
        if _perform_checks:
            if len(s2_signals.shape) < 3:
                raise ValueError(
                    f"s2_signals should have atleast 3 axes. Got {s2_signals.shape}."
                )

        self.s2_signals = s2_signals

    @property
    def batch_dims(self) -> Tuple[int, ...]:
        pass

    @property
    def shape(self) -> Tuple[int, int, int]:
        pass

    @property
    def res_beta(self) -> int:
        pass

    @property
    def res_alpha(self) -> int:
        pass

    @property
    def grid_values(self) -> jnp.ndarray:
        pass

    @property
    def res_theta(self) -> int:
        pass

    @property
    def grid_theta(self) -> jnp.ndarray:
        pass

    @property
    def grid_resolution(self) -> str:
        pass

    @staticmethod
    def from_function(
        func: Callable[[jax.Array], float],
        res_beta: int,
        res_alpha: int,
        res_theta: int,
        quadrature: str,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> "SO3Signal":
        pass

    def __mul__(self, other: Union[float, "SO3Signal"]) -> "SO3Signal":
        if isinstance(other, SO3Signal):
            if self.shape != other.shape:
                raise ValueError(
                    f"Shapes of the two signals do not match: {self.shape} != {other.shape}"
                )
            return SO3Signal(self.s2_signals * other.s2_signals)

        return SO3Signal(self.s2_signals * other)

    def __rmul__(self, other: float) -> "SO3Signal":
        return self * other

    def __neg__(self) -> "SO3Signal":
        return self * -1

    def __truediv__(self, other: Union[float, "SO3Signal"]) -> "SO3Signal":
        if isinstance(other, SO3Signal):
            if self.shape != other.shape:
                raise ValueError(
                    f"Shapes of the two signals do not match: {self.shape} != {other.shape}"
                )

            return self.replace_values(self.grid_values / other.grid_values)

        return self * (1 / other)

    def apply(self, func: Callable[..., jnp.ndarray]) -> "SO3Signal":
        """Apply a pointwise function to the signal."""
        return SO3Signal(self.s2_signals.apply(func))

    def vmap_over_batch_dims(
        self, func: Callable[..., jnp.ndarray]
    ) -> Callable[..., jnp.ndarray]:
        """Apply a function to the signal while preserving the batch dimensions."""
        for _ in range(len(self.batch_dims)):
            func = jax.vmap(func)
        return func

    def argmax(
        self,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Find the rotation (and corresponding grid indices) with the maximum value of the signal."""
        flat_index = jnp.argmax(self.grid_values.reshape(*self.shape[:-3], -1), axis=-1)

        theta_idx, beta_idx, alpha_idx = jnp.unravel_index(flat_index, self.shape[-3:])

        axis = self.s2_signals.grid_vectors[..., beta_idx, alpha_idx, :]
        assert axis.shape == (*self.batch_dims, 3)

        angle = self.grid_theta[theta_idx]
        assert angle.shape == (*self.batch_dims,)

        Rs = self.vmap_over_batch_dims(e3nn.axis_angle_to_matrix)(axis, angle)
        assert Rs.shape == (*self.batch_dims, 3, 3)

        return Rs, (theta_idx, beta_idx, alpha_idx)

    def replace_values(self, grid_values: jnp.ndarray) -> "SO3Signal":
        """Replace the values of the signal with the given grid_values."""
        return SO3Signal(self.s2_signals.replace_values(grid_values))

    def integrate_over_angles(self) -> SphericalSignal:
        pass

    def integrate(self) -> float:
        pass

    def sample(self, rng: jax.random.PRNGKey) -> jnp.ndarray:
        pass

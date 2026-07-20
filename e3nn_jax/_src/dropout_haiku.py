import haiku as hk
import jax
import jax.numpy as jnp

from e3nn_jax import Irreps, IrrepsArray


class Dropout(hk.Module):

    def __init__(self, p, *, irreps=None):
        super().__init__()
        self.irreps = Irreps(irreps) if irreps is not None else None
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__} (p={self.p})"

    def __call__(self, rng, x: IrrepsArray, is_training=True) -> IrrepsArray:
        """Evaluate equivariant dropout.

        Args:
            rng (`jax.random.PRNGKey`): the random number generator
            x (IrrepsArray): the input
            is_training (bool): whether to perform dropout

        Returns:
            IrrepsArray: the output
        """
        if not is_training:
            return x

        if self.irreps is not None:
            x = x.rechunk(self.irreps)
        if not isinstance(x, IrrepsArray):
            raise TypeError(f"{self.__class__.__name__} only supports IrrepsArray")

        noises = []
        zero_flags = []
        for (mul, ir), a in zip(x.irreps, x.chunks):
            if self.p >= 1:
                zero_flags.append(True)
                noises.append(jnp.zeros((mul * ir.dim,), x.dtype))
            elif self.p <= 0:
                zero_flags.append(False)
                noises.append(jnp.ones((mul * ir.dim,), x.dtype))
            elif a is None:
                zero_flags.append(True)
                noises.append(jnp.zeros((mul * ir.dim,), x.dtype))
            else:
                noise = jax.random.bernoulli(rng, p=1 - self.p, shape=(mul, 1)) / (
                    1 - self.p
                )
                zero_flags.append(False)
                noises.append(jnp.repeat(noise, ir.dim, axis=1).flatten())

        noises = jnp.concatenate(noises)
        return IrrepsArray(x.irreps, x.array * noises, zero_flags=zero_flags)

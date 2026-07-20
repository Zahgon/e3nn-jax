import e3nn_jax as e3nn
import jax.numpy as jnp
import jax


def tensor_product_with_spherical_harmonics(
    input: e3nn.IrrepsArray, vector: e3nn.IrrepsArray, degree: int
) -> e3nn.IrrepsArray:
    """Tensor product of something with the spherical harmonics of a vector.

    The idea of this optimization comes from the paper::

        Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs

    Args:
        input (IrrepsArray): input
        vector (IrrepsArray): vector, irreps must be "1o" or "1e"
        degree (int): the maximum degree of the spherical harmonics

    Returns:
        IrrepsArray: tensor product

    Notes:
        This function is equivalent to::

            tensor_product(input, spherical_harmonics(range(degree + 1), vector, True))

    Examples:
        >>> input = e3nn.normal("3x0e + 2x1o", jax.random.PRNGKey(0))
        >>> vector = e3nn.normal("1e", jax.random.PRNGKey(1))
        >>> degree = 2
        >>> output1 = tensor_product_with_spherical_harmonics(input, vector, degree)
        >>> output2 = e3nn.tensor_product(input, e3nn.spherical_harmonics(range(degree + 1), vector, True))
        >>> assert output1.irreps == output2.irreps
        >>> assert jnp.allclose(output1.array, output2.array, atol=1e-6)
    """
    input = e3nn.as_irreps_array(input)

    if not (vector.irreps == "1o" or vector.irreps == "1e"):
        raise ValueError(
            "tensor_product_with_spherical_harmonics: vector must be a vector."
        )

    leading_shape = jnp.broadcast_shapes(input.shape[:-1], vector.shape[:-1])
    input = input.broadcast_to(leading_shape + (-1,))
    vector = vector.broadcast_to(leading_shape + (-1,))

    f = impl
    for _ in range(len(leading_shape)):
        f = e3nn.utils.vmap(f, in_axes=(0, 0, None), out_axes=0)

    return f(input, vector, degree)


def impl(
    input: e3nn.IrrepsArray, vector: e3nn.IrrepsArray, degree: int
) -> e3nn.IrrepsArray:
    pass


def sl(lout: int, lin: int) -> slice:
    return slice(lout - lin, lout + lin + 1)


def is_diag(x: jax.Array) -> bool:
    pass


def normalize(x):
    n2 = jnp.sum(x**2, axis=-1, keepdims=True)
    n2 = jnp.where(n2 > 0.0, n2, 1.0)
    return x / jnp.sqrt(n2)

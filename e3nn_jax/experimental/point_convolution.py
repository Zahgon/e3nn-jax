from typing import Callable, Sequence, Tuple

import flax
import haiku as hk
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn

_docstring_class = r"""Message passing convolution

Args:
    target_irreps (e3nn.Irreps): irreps of the output
    radial_basis (Callable[[jax.Array], jax.Array]): radial basis functions
    avg_num_neighbors (float): average number of neighbors
    sh_lmax (int): maximum spherical harmonics degree
    num_radial_basis (int): number of radial basis functions
    mlp_neurons (List[int]): number of neurons in each layer of the MLP
    mlp_activation (Callable[[jax.Array], jax.Array]): activation function of the MLP
"""

_docstring_call = r"""Compute the message passing convolution

Args:
    positions (e3nn.IrrepsArray): positions of the nodes
    node_feats (e3nn.IrrepsArray): features of the nodes
    senders (jax.Array): indices of the sender nodes
    receivers (jax.Array): indices of the receiver nodes

Returns:
    e3nn.IrrepsArray: features of the nodes
"""


def radial_basis(r, cutoff, num_radial_basis):
    """Radial basis functions

    This can be used as the `radial_basis` argument of `MessagePassingConvolution`::

        lambda r: radial_basis(r, 6.0, 8)

    Args:
        r (jax.Array): distances
        cutoff (float): cutoff radius
        num_radial_basis (int): number of radial basis functions

    Returns:
        jax.Array: radial basis functions
    """
    r = r / cutoff
    return e3nn.bessel(r, num_radial_basis) * e3nn.soft_envelope(r)[:, None]


def _call(
    self, positions, node_feats, senders, receivers, Linear, MultiLayerPerceptron
):
    pass


class MessagePassingConvolutionHaiku(hk.Module):
    def __init__(
        self,
        target_irreps: e3nn.Irreps,
        radial_basis: Callable[[jax.Array], jax.Array],
        *,
        avg_num_neighbors: float,
        sh_lmax: int = 3,
        num_radial_basis: int = 8,
        mlp_neurons: Sequence[int] = (64,),
        mlp_activation: Callable[[jax.Array], jax.Array] = jax.nn.gelu,
        name: str = None,
    ):
        super().__init__(name=name)
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.radial_basis = radial_basis
        self.avg_num_neighbors = avg_num_neighbors
        self.sh_lmax = sh_lmax
        self.num_radial_basis = num_radial_basis
        self.mlp_neurons = tuple(mlp_neurons)
        self.mlp_activation = mlp_activation

    def __call__(
        self,
        positions: e3nn.IrrepsArray,  # [n_edges, 1o or 1e]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        senders: jax.Array,  # [n_edges, ]
        receivers: jax.Array,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:  # [n_nodes, irreps]
        return _call(
            self,
            positions,
            node_feats,
            senders,
            receivers,
            e3nn.haiku.Linear,
            e3nn.haiku.MultiLayerPerceptron,
        )


MessagePassingConvolutionHaiku.__doc__ = _docstring_class
MessagePassingConvolutionHaiku.__call__.__doc__ = _docstring_call


class MessagePassingConvolutionFlax(flax.linen.Module):
    target_irreps: e3nn.Irreps
    radial_basis: Callable[[jax.Array], jax.Array]
    avg_num_neighbors: float
    sh_lmax: int = 3
    num_radial_basis: int = 8
    mlp_neurons: Tuple[int, ...] = (64,)
    mlp_activation: Callable[[jax.Array], jax.Array] = jax.nn.gelu

    @flax.linen.compact
    def __call__(
        self,
        positions: e3nn.IrrepsArray,  # [n_edges, 1o or 1e]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        senders: jax.Array,  # [n_edges, ]
        receivers: jax.Array,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:  # [n_nodes, irreps]
        return _call(
            self,
            positions,
            node_feats,
            senders,
            receivers,
            e3nn.flax.Linear,
            e3nn.flax.MultiLayerPerceptron,
        )


MessagePassingConvolutionFlax.__doc__ = _docstring_class
MessagePassingConvolutionFlax.__call__.__doc__ = _docstring_call

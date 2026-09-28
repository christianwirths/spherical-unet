"""Spherical U-Net for fields on regular latitude-longitude grids."""

from .model import (
    TOPOLOGIES,
    AvgPool2dGraph,
    AvgUnpool2dGraph,
    DirectNeighConv,
    GraphResNetBlock,
    GraphSelfAttention,
    SinusoidalTimeEmbedding,
    SphericalDecoder,
    SphericalEncoder,
    SphericalUNet,
    SphericalUNetCore,
    SphericalUNetOutput,
    SphericalUNetWrapper,
    UpsampleConv2d,
    build_equiangular_graph,
    build_equiangular_neighbours,
    spherical_pad,
)
from .checkpoint import verify_widening, widen_input_channels

__version__ = "0.1.0"

__all__ = [
    "TOPOLOGIES",
    "AvgPool2dGraph",
    "AvgUnpool2dGraph",
    "DirectNeighConv",
    "GraphResNetBlock",
    "GraphSelfAttention",
    "SinusoidalTimeEmbedding",
    "SphericalDecoder",
    "SphericalEncoder",
    "SphericalUNet",
    "SphericalUNetCore",
    "SphericalUNetOutput",
    "SphericalUNetWrapper",
    "UpsampleConv2d",
    "build_equiangular_graph",
    "build_equiangular_neighbours",
    "spherical_pad",
    "verify_widening",
    "widen_input_channels",
]

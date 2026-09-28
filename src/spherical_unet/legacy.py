"""
Chebyshev graph convolutions (DeepSphere style), kept for reference.

These were the convolution of earlier versions of the network and are
superseded by :class:`spherical_unet.model.DirectNeighConv`: with polynomial
order ``K = 3`` a Chebyshev filter has only 3 isotropic spectral weights,
against 9 anisotropic weights for the direct 3x3 neighbourhood. Nothing in
:mod:`spherical_unet.model` uses this module.

Extra dependencies: ``scipy`` for everything here, and the DeepSphere fork of
PyGSP for the Laplacian builders::

    pip install "spherical-unet[legacy]"
    pip install git+https://github.com/epfl-lts2/pygsp.git@39a0665f637191152605911cf209fc16a36e5ae9#egg=PyGSP
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import _gn_num_groups

__all__ = [
    "build_equiangular_laplacians",
    "build_healpix_laplacians",
    "cheb_conv",
    "ChebConvLayer",
    "SphericalChebBN",
]

_PYGSP_HINT = (
    "You need the DeepSphere fork of PyGSP:\n"
    "  pip install git+https://github.com/epfl-lts2/pygsp.git"
    "@39a0665f637191152605911cf209fc16a36e5ae9#egg=PyGSP"
)


def _scipy_csr_to_sparse_tensor(csr_mat) -> torch.Tensor:
    coo = csr_mat.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def _prepare_laplacian(laplacian) -> torch.Tensor:
    """Rescale the Laplacian spectrum to [-1, 1] for Chebyshev filters."""
    from scipy import sparse
    from scipy.sparse import linalg as splinalg

    tol = 5e-3
    lmax = splinalg.eigsh(laplacian, k=1, tol=tol, ncv=min(laplacian.shape[0], 10),
                          return_eigenvectors=False)[0] * (1 + 2 * tol)
    identity = sparse.identity(laplacian.shape[0], format=laplacian.format,
                               dtype=laplacian.dtype)
    return _scipy_csr_to_sparse_tensor(laplacian * (2.0 / lmax) - identity)


def build_equiangular_laplacians(n_lat: int, n_lon: int, depth: int,
                                 laplacian_type: str = "combinatorial"):
    """Rescaled Laplacians of PyGSP ``SphereEquiangular`` graphs, one per level.

    Returns ``(laps, grid_dims)``, both ordered coarsest first.
    """
    try:
        from pygsp.graphs.sphereequiangular import SphereEquiangular
    except ImportError as err:
        raise ImportError("Could not import SphereEquiangular. " + _PYGSP_HINT) from err

    factor = 2 ** (depth - 1)
    if n_lat % factor or n_lon % factor:
        raise ValueError(f"Grid {n_lat}x{n_lon} is not divisible by 2^(depth-1) = {factor}.")

    laps, dims = [], []
    lat, lon = n_lat, n_lon
    for _ in range(depth):
        G = SphereEquiangular(size=(lat, lon))
        G.compute_laplacian(laplacian_type)
        laps.append(_prepare_laplacian(G.L))
        dims.append((lat, lon))
        lat //= 2
        lon //= 2
    return laps[::-1], dims[::-1]


def build_healpix_laplacians(n_pixels: int, depth: int,
                             laplacian_type: str = "combinatorial"):
    """Rescaled Laplacians of nested HEALPix graphs, one per level.

    Returns ``(laps, grid_dims)`` with ``grid_dims`` as ``(V,)`` tuples, both
    ordered coarsest first.
    """
    try:
        from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
    except ImportError as err:
        raise ImportError("Could not import SphereHealpix. " + _PYGSP_HINT) from err

    nside = int(round(math.sqrt(n_pixels / 12)))
    if 12 * nside * nside != n_pixels:
        raise ValueError(f"n_pixels={n_pixels} is not 12 * nside^2")
    laps, dims = [], []
    for i in range(depth):
        subdiv = nside // (2 ** i)
        G = SphereHealpix(subdiv, nest=True, k=20)
        G.compute_laplacian(laplacian_type)
        laps.append(_prepare_laplacian(G.L))
        dims.append((12 * subdiv * subdiv,))
    return laps[::-1], dims[::-1]


def cheb_conv(laplacian: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Chebyshev polynomial graph convolution.

    Args:
        laplacian: ``[V, V]`` sparse Laplacian rescaled to [-1, 1].
        x:         ``[B, V, F_in]``.
        weight:    ``[K, F_in, F_out]``.

    Returns:
        ``[B, V, F_out]``.
    """
    B, V, Fin = x.shape
    K, _, Fout = weight.shape

    x0 = x.permute(1, 2, 0).reshape(V, Fin * B)
    polynomials = [x0]
    if K >= 2:
        x1 = torch.sparse.mm(laplacian, x0)
        polynomials.append(x1)
        for _ in range(2, K):
            x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
            polynomials.append(x2)
            x0, x1 = x1, x2

    out = torch.stack(polynomials, dim=0).view(K, V, Fin, B)
    out = out.permute(3, 1, 2, 0).reshape(B * V, Fin * K)
    # As in DeepSphere: `out` is flattened (F_in, K) but `weight` (K, F_in).
    # This is only a fixed relabelling of learned parameters; it is kept so
    # weights trained with the original code keep their meaning.
    out = out.matmul(weight.reshape(Fin * K, Fout))
    return out.view(B, V, Fout)


class ChebConvLayer(nn.Module):
    """Chebyshev graph convolution with bias."""

    def __init__(self, in_ch: int, out_ch: int, K: int):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.K = K
        self.weight = nn.Parameter(torch.empty(K, in_ch, out_ch))
        self.bias = nn.Parameter(torch.empty(out_ch))
        nn.init.normal_(self.weight, 0.0, math.sqrt(2.0 / (in_ch * K)))
        nn.init.constant_(self.bias, 0.01)

    def forward(self, lap: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return cheb_conv(lap, x, self.weight) + self.bias


class SphericalChebBN(nn.Module):
    """ChebConv -> GroupNorm -> optional SiLU."""

    def __init__(self, in_ch: int, out_ch: int, lap: torch.Tensor, K: int,
                 activation: bool = True):
        super().__init__()
        self.register_buffer("lap", lap)
        self.conv = ChebConvLayer(in_ch, out_ch, K)
        self.norm = nn.GroupNorm(_gn_num_groups(out_ch), out_ch)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.lap, x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return F.silu(x) if self.activation else x

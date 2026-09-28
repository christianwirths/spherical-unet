"""
Spherical U-Net on equiangular (regular latitude-longitude) grids.

This file is self-contained (it depends only on ``torch`` and ``numpy``) so it
can be copied into another project as a single module.

The network is a graph U-Net whose external interface is image-like: it takes
``[B, C, H, W]`` tensors on a regular lat-lon grid and returns ``[B, C_out, H,
W]``. Grid-to-graph and graph-to-grid conversions happen inside the wrapper, so
it is a drop-in replacement for a diffusers ``UNet2DModel`` in diffusion,
consistency or flow-matching training loops (``model(x, t).sample``).

Architecture (structurally matches diffusers ``UNet2DModel``):
  - **Encoder**: 2 x ``GraphResNetBlock`` per level, 2x2 average pooling
    between levels.
  - **Mid-block**: ``GraphResNetBlock`` -> ``GraphSelfAttention`` ->
    ``GraphResNetBlock`` at the coarsest resolution (global receptive field).
  - **Decoder**: bilinear upsampling -> 3x3 smoothing conv -> skip concat ->
    2 x ``GraphResNetBlock`` per level.
  - **Time conditioning**: sinusoidal embedding -> MLP, added after the first
    conv of every ResNet block (diffusers ``"default"`` time_embedding_norm).
  - **Normalisation**: GroupNorm everywhere (identical in train and eval).
  - **Activation**: SiLU throughout.

Convolution operator, ``DirectNeighConv``:
  A 9-tap spatial gather (self + 8 surrounding vertices) followed by
  ``nn.Linear(9 * F_in, F_out)``. It is a 3x3 convolution with 9 anisotropic
  weights per filter, but with spherical topology:
    - **Longitude**: circular (the East and West edges are stitched).
    - **Poles**: crossing a pole lands on the same row at the antipodal
      longitude (``j + W // 2``).

Topology modes (``topology`` argument of the wrapper):
  - ``"legacy"`` (default): bit-compatible with checkpoints trained with the
    original implementation. Two known geometric quirks are kept on purpose:
    the diagonal pole neighbours are mirrored in longitude (NE across the pole
    maps to ``j_refl - 1`` instead of ``j_refl + 1``), and the post-upsample
    ``Conv2d`` uses circular padding in *both* axes, so the first and last
    latitude rows see each other. The bilinear upsampling is not periodic in
    longitude either (it clamps at the seam).
  - ``"spherical"``: geometrically correct pole neighbours, post-upsample conv
    padded spherically (circular in longitude, pole-reflected in latitude) and
    bilinear upsampling that is periodic in longitude and pole-aware. The
    ``DirectNeighConv`` layers then run as a ``Conv2d`` on the spherically
    padded field (same operator, 2-3x faster, far less activation memory).
    Use this for new models trained from scratch.

The grid is assumed to be cell-centred (no row exactly on a pole): the row
beyond a pole row is that pole row itself shifted by 180 degrees. Latitude may
run north-to-south or south-to-north; the topology is symmetric.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

__all__ = [
    "SphericalUNetOutput",
    "SinusoidalTimeEmbedding",
    "build_equiangular_neighbours",
    "build_equiangular_graph",
    "spherical_pad",
    "DirectNeighConv",
    "GraphResNetBlock",
    "GraphSelfAttention",
    "AvgPool2dGraph",
    "AvgUnpool2dGraph",
    "UpsampleConv2d",
    "SphericalEncoder",
    "SphericalDecoder",
    "SphericalUNetCore",
    "SphericalUNetWrapper",
    "SphericalUNet",
    "TOPOLOGIES",
]

TOPOLOGIES = ("legacy", "spherical")
N_NEIGHBOURS = 9


def _check_topology(topology: str) -> None:
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {TOPOLOGIES}, got {topology!r}")


# ---------------------------------------------------------------------------
# Output container (matches the diffusers UNet2DOutput interface)
# ---------------------------------------------------------------------------
@dataclass
class SphericalUNetOutput:
    """Drop-in replacement for ``diffusers.models.unet_2d.UNet2DOutput``."""
    sample: torch.Tensor


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (as in the DDPM U-Net)
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding of scalar time values.

    Maps ``t`` of shape ``[B]`` to ``[B, dim]``: ``dim / 2`` sines followed by
    ``dim / 2`` cosines with frequencies ``10000^(-k / (dim/2))``, as in the
    Transformer positional encoding. The periods span ``2 pi`` to roughly
    ``2 pi * 10^4``, which suits integer-like diffusion steps (0..1000) or
    EDM noise levels. For a time variable in ``[0, 1]`` (flow matching),
    scale it up (for example by 1000) before calling the network, otherwise
    most embedding dimensions are nearly constant.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim < 2 or dim % 2:
            raise ValueError(f"time embedding dim must be even and >= 2, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Follow the input dtype so the module composes with a float64 model.
        dtype = t.dtype if t.is_floating_point() else torch.get_default_dtype()
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=dtype) / half
        )
        args = t[:, None].to(dtype) * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Spherical neighbourhood on equiangular grids
# ---------------------------------------------------------------------------

def build_equiangular_neighbours(H: int, W: int, topology: str = "legacy") -> np.ndarray:
    """Neighbour index array for an equiangular grid with spherical topology.

    Vertices are the row-major flattened grid points, ``v = i * W + j``. Each
    vertex gets 9 neighbours in the fixed order::

        [self, N, NE, E, SE, S, SW, W, NW]

    where "N" means row ``i - 1`` and "E" means column ``j + 1``.

    Boundary handling:
      - Longitude: circular.
      - Poles (``i = 0`` or ``i = H - 1``): the row beyond the pole is the pole
        row itself shifted by ``W // 2`` columns (antipodal longitude). With
        ``topology="legacy"`` the diagonal neighbours across the pole are
        mirrored (NE -> ``j_refl - 1``), reproducing the original
        implementation. With ``topology="spherical"`` they are placed
        geometrically (NE -> ``j_refl + 1``), which equals zero-free
        "pole padding" of a 3x3 convolution.

    For odd ``W`` the antipode ``j + W // 2`` is half a cell off; this only
    happens at coarse U-Net levels (for example 360 -> 45 columns at depth 4).

    Returns:
        ``[H * W, 9]`` int64 array of flat vertex indices.
    """
    _check_topology(topology)
    i = np.repeat(np.arange(H), W)                  # row of each vertex
    j = np.tile(np.arange(W), H)                    # column of each vertex
    je, jw = (j + 1) % W, (j - 1) % W
    j_refl = (j + W // 2) % W
    # Across-pole column offsets for the (E-side, W-side) diagonals.
    d_e, d_w = (-1, 1) if topology == "legacy" else (1, -1)
    refl_e, refl_w = (j_refl + d_e) % W, (j_refl + d_w) % W

    north = i > 0
    south = i < H - 1
    row_n = np.where(north, i - 1, 0) * W           # pole row 0 reflects onto itself
    row_s = np.where(south, i + 1, H - 1) * W       # pole row H-1 reflects onto itself

    neigh = np.empty((H * W, N_NEIGHBOURS), dtype=np.int64)
    neigh[:, 0] = i * W + j                                      # self
    neigh[:, 1] = row_n + np.where(north, j, j_refl)             # N
    neigh[:, 2] = row_n + np.where(north, je, refl_e)            # NE
    neigh[:, 3] = i * W + je                                     # E
    neigh[:, 4] = row_s + np.where(south, je, refl_e)            # SE
    neigh[:, 5] = row_s + np.where(south, j, j_refl)             # S
    neigh[:, 6] = row_s + np.where(south, jw, refl_w)            # SW
    neigh[:, 7] = i * W + jw                                     # W
    neigh[:, 8] = row_n + np.where(north, jw, refl_w)            # NW
    return neigh


def build_equiangular_graph(n_lat: int, n_lon: int, depth: int,
                            topology: str = "legacy"):
    """Neighbour arrays and grid shapes for every U-Net level.

    Each level halves both grid dimensions (2x2 average pooling).

    Returns:
        ``(neigh_orders_list, grid_dims)``: a list of ``[V, 9]`` int64 tensors
        and a list of ``(H, W)`` tuples, both ordered *coarsest first*.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    factor = 2 ** (depth - 1)
    if n_lat % factor or n_lon % factor:
        raise ValueError(
            f"Grid {n_lat}x{n_lon} is not divisible by 2^(depth-1) = {factor}. "
            f"Reduce the depth or pad the grid."
        )
    all_neigh, dims = [], []
    lat, lon = n_lat, n_lon
    for _ in range(depth):
        all_neigh.append(torch.from_numpy(build_equiangular_neighbours(lat, lon, topology)))
        dims.append((lat, lon))
        lat //= 2
        lon //= 2
    return all_neigh[::-1], dims[::-1]


def spherical_pad(x: torch.Tensor, pad: int = 1) -> torch.Tensor:
    """Pad a ``[B, C, H, W]`` lat-lon field with spherical topology.

    Longitude is padded circularly. Latitude is padded by reflecting across
    each pole: padded row ``-k`` is row ``k - 1`` shifted by ``W // 2`` columns
    (cell-centred grid): padded column ``j`` holds column ``(j + W // 2) % W``,
    exactly as in :func:`build_equiangular_neighbours`. For odd ``W`` that
    is half a cell short of the antipode.
    """
    if pad == 0:
        return x
    H, W = x.shape[-2:]
    if pad > H:
        raise ValueError(f"pad ({pad}) larger than the number of rows ({H})")
    top = x[..., :pad, :].flip(-2).roll(-(W // 2), dims=-1)
    bottom = x[..., -pad:, :].flip(-2).roll(-(W // 2), dims=-1)
    x = torch.cat([top, x, bottom], dim=-2)
    return F.pad(x, (pad, pad, 0, 0), mode="circular")


def _graph_to_image(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """``[B, V, C]`` -> ``[B, C, H, W]`` (row-major vertices)."""
    B, _, C = x.shape
    return x.view(B, h, w, C).permute(0, 3, 1, 2)


def _image_to_graph(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` -> ``[B, V, C]``."""
    B, C = x.shape[:2]
    return x.permute(0, 2, 3, 1).reshape(B, -1, C)


# ---------------------------------------------------------------------------
# Direct spatial neighbourhood convolution
# ---------------------------------------------------------------------------

class DirectNeighConv(nn.Module):
    """9-tap spatial convolution on a graph given by a neighbour index array.

    Gathers ``[self, N, NE, E, SE, S, SW, W, NW]`` for every vertex and applies
    ``nn.Linear(9 * in_ch, out_ch)``. The flattened input is neighbour-major:
    column ``n * in_ch + c`` multiplies channel ``c`` of neighbour ``n``.

    The parameter is stored as ``self.weight`` (an ``nn.Linear``), so the
    state-dict keys are ``<name>.weight.weight`` and ``<name>.weight.bias``.

    Two execution paths compute the same operator:

    - **gather** (``grid_hw=None``): index with ``neigh_orders``, then the
      Linear. Works for any neighbour table (used for ``"legacy"``) and is
      bit-compatible with the original implementation. It materialises a
      ``[B, V, 9, F_in]`` tensor; set ``recompute=True`` to recompute it in
      the backward pass instead of storing it (about 9x less activation
      memory per conv for ~10% more time, forward bitwise unchanged).
    - **padded conv2d** (``grid_hw=(H, W)``): the weight is reshaped to a
      ``[F_out, F_in, 3, 3]`` kernel and applied to the
      :func:`spherical_pad`-ded field. Equal (up to round-off) to the gather
      on the ``"spherical"`` table, 2-3x faster and much lighter in memory.
    """

    # Neighbour slot of each 3x3 kernel position in row-major order:
    # NW, N, NE / W, self, E / SW, S, SE.
    KERNEL_SLOTS = (8, 1, 2, 7, 0, 3, 6, 5, 4)

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.weight = nn.Linear(N_NEIGHBOURS * in_ch, out_ch)
        self.recompute = False

    def conv_kernel(self) -> torch.Tensor:
        """The weight as a ``[F_out, F_in, 3, 3]`` Conv2d kernel ("N" = row above)."""
        w = self.weight.weight.view(self.out_ch, N_NEIGHBOURS, self.in_ch)
        w = w[:, list(self.KERNEL_SLOTS)]                   # [F_out, 9, F_in]
        return w.transpose(1, 2).reshape(self.out_ch, self.in_ch, 3, 3)

    def _gather_linear(self, neigh_orders: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, V, _ = x.shape
        mat = x[:, neigh_orders]                            # [B, V, 9, F_in]
        return self.weight(mat.reshape(B, V, N_NEIGHBOURS * self.in_ch))

    def forward(self, neigh_orders: torch.Tensor, x: torch.Tensor,
                grid_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """``neigh_orders`` ``[V, 9]``, ``x`` ``[B, V, F_in]`` -> ``[B, V, F_out]``."""
        if grid_hw is not None:
            img = spherical_pad(_graph_to_image(x, *grid_hw), 1)
            return _image_to_graph(F.conv2d(img, self.conv_kernel(), self.weight.bias))
        if self.recompute and torch.is_grad_enabled():
            return _checkpoint(self._gather_linear, neigh_orders, x, use_reentrant=False)
        return self._gather_linear(neigh_orders, x)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _drop_keys(state_dict, prefix: str, names) -> None:
    """Remove ``prefix + name`` entries in place (legacy persistent buffers)."""
    for name in names:
        state_dict.pop(prefix + name, None)


def _gn_num_groups(num_channels: int, preferred: int = 32) -> int:
    """Largest group count in (preferred, 16, 8, 4, 2, 1) dividing num_channels.

    Falls back to 1 (LayerNorm over channels and vertices) for odd widths.
    """
    for g in (preferred, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1  # pragma: no cover


def _group_norm_graph(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    """Apply a GroupNorm to a ``[B, V, C]`` graph signal."""
    return norm(x.transpose(1, 2)).transpose(1, 2)


class GraphResNetBlock(nn.Module):
    """Graph ResNet block mirroring diffusers ``ResnetBlock2D``.

    Main path::

        GN -> SiLU -> DirectNeighConv -> + time_proj(SiLU(t_emb))
           -> GN -> SiLU -> DirectNeighConv
    Residual::

        x -> (Linear shortcut if in_ch != out_ch) -> add

    Input normalisation bypass: the last ``n_unnormed`` input channels skip
    ``norm1`` and the input SiLU and enter ``conv1`` unchanged. This is meant
    for the first encoder block, for inputs on a fixed physical scale (for
    example static orography or masks) that must not be renormalised per
    sample. It also keeps the GroupNorm statistics (and the group split) of the
    remaining channels identical to a model without those extra channels,
    which is what makes zero-initialised input widening exact (see
    ``spherical_unet.checkpoint.widen_input_channels``).

    With ``n_unnormed == in_ch`` there is no ``norm1`` at all and ``conv1``
    sees the raw input. Use this when the absolute level of the inputs is
    signal: a per-sample GroupNorm subtracts each sample's mean and divides by
    its scale, so ``GN(x + c) == GN(x)`` and a uniform shift of the input
    would otherwise only be visible through the residual shortcut.

    Args:
        in_ch:        Input channels.
        out_ch:       Output channels.
        neigh_orders: ``[V, 9]`` int64 neighbour indices at this level.
        time_dim:     Width of the time embedding.
        n_unnormed:   Trailing input channels that bypass ``norm1``.
        grid_hw:      ``(H, W)`` of this level to run the convolutions as a
                      spherically padded ``Conv2d`` (``"spherical"`` topology);
                      ``None`` uses the neighbour gather.
    """

    def __init__(self, in_ch: int, out_ch: int, neigh_orders: torch.Tensor,
                 time_dim: int, n_unnormed: int = 0,
                 grid_hw: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.grid_hw = tuple(grid_hw) if grid_hw is not None else None
        if not 0 <= n_unnormed <= in_ch:
            raise ValueError(f"n_unnormed must lie in [0, {in_ch}], got {n_unnormed}")
        self.n_unnormed = n_unnormed
        self.n_normed = in_ch - n_unnormed

        self.norm1 = (nn.GroupNorm(_gn_num_groups(self.n_normed), self.n_normed)
                      if self.n_normed > 0 else None)
        self.conv1 = DirectNeighConv(in_ch, out_ch)
        # Rebuilt from the grid at construction, so not stored in checkpoints.
        self.register_buffer("neigh_orders", neigh_orders, persistent=False)

        self.time_emb_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(_gn_num_groups(out_ch), out_ch)
        self.conv2 = DirectNeighConv(out_ch, out_ch)

        self.shortcut = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints of the original implementation stored the table.
        _drop_keys(state_dict, prefix, ("neigh_orders",))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """``x`` ``[B, V, F_in]``, ``t_emb`` ``[B, D_time]`` -> ``[B, V, F_out]``."""
        residual = x

        if self.norm1 is None:
            h = x
        elif self.n_unnormed:
            main, extra = x[..., :self.n_normed], x[..., self.n_normed:]
            h = F.silu(_group_norm_graph(self.norm1, main))
            h = torch.cat([h, extra], dim=-1)
        else:
            h = F.silu(_group_norm_graph(self.norm1, x))
        h = self.conv1(self.neigh_orders, h, self.grid_hw)

        h = h + self.time_emb_proj(F.silu(t_emb)).unsqueeze(1)

        h = F.silu(_group_norm_graph(self.norm2, h))
        h = self.conv2(self.neigh_orders, h, self.grid_hw)

        return h + self.shortcut(residual)


class GraphSelfAttention(nn.Module):
    """Multi-head self-attention over all vertices (diffusers ``AttentionBlock``).

    GN -> Q, K, V projections -> scaled dot-product attention -> output
    projection -> residual add. Used at the coarsest level only, where the
    vertex count is small (for example 22 x 45 = 990 for a 176 x 360 grid at
    depth 4).
    """

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        if channels % num_heads:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.group_norm = nn.GroupNorm(_gn_num_groups(channels), channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, V, C]`` -> ``[B, V, C]``."""
        residual = x
        B, V, C = x.shape
        x = _group_norm_graph(self.group_norm, x)

        def heads(t):
            return t.view(B, V, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(heads(self.to_q(x)), heads(self.to_k(x)),
                                              heads(self.to_v(x)))
        attn = attn.transpose(1, 2).reshape(B, V, C)
        return self.to_out(attn) + residual


class AvgPool2dGraph(nn.Module):
    """2x2 average pooling of a row-major lat-lon graph signal.

    ``[B, H*W, C]`` -> ``[B, H*W/4, C]``. With even ``W`` no pooling window
    crosses the longitude seam, so no special handling is needed.
    """

    def __init__(self, h: int, w: int):
        super().__init__()
        self.h = h
        self.w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _image_to_graph(F.avg_pool2d(_graph_to_image(x, self.h, self.w), kernel_size=2))


class AvgUnpool2dGraph(nn.Module):
    """2x bilinear upsampling of a row-major lat-lon graph signal.

    ``[B, H*W, C]`` -> ``[B, 4*H*W, C]`` (``h``, ``w`` are the *coarse* dims).
    Bilinear rather than nearest avoids 2x2 block artefacts. With
    ``topology="spherical"`` the field is first padded spherically by one
    cell, so the interpolation is periodic in longitude and pole-aware in
    latitude; ``"legacy"`` clamps at all four edges.
    """

    def __init__(self, h: int, w: int, topology: str = "legacy"):
        super().__init__()
        _check_topology(topology)
        self.h = h
        self.w = w
        self.topology = topology

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _graph_to_image(x, self.h, self.w)
        if self.topology == "spherical":
            x = F.interpolate(spherical_pad(x, 1), scale_factor=2, mode="bilinear",
                              align_corners=False)
            x = x[..., 2:-2, 2:-2]
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return _image_to_graph(x)


class UpsampleConv2d(nn.Module):
    """Post-upsample 3x3 Conv2d -> GroupNorm -> SiLU, in the image domain.

    Smooths bilinear-upsampling artefacts with 9 direction-specific weights,
    like the conv in diffusers ``UpBlock2D``. ``topology="legacy"`` pads
    circularly in both axes (the first and last latitude rows see each
    other); ``"spherical"`` pads with :func:`spherical_pad`.

    Args:
        channels: Feature channels.
        h, w:     Grid shape *after* upsampling.
    """

    def __init__(self, channels: int, h: int, w: int, topology: str = "legacy"):
        super().__init__()
        _check_topology(topology)
        self.h = h
        self.w = w
        self.topology = topology
        if topology == "legacy":
            self.conv = nn.Conv2d(channels, channels, kernel_size=3,
                                  padding=1, padding_mode="circular")
        else:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.norm = nn.GroupNorm(_gn_num_groups(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, V, C]`` -> ``[B, V, C]`` with ``V = h * w``."""
        x = _graph_to_image(x, self.h, self.w)
        if self.topology == "spherical":
            x = spherical_pad(x, 1)
        x = F.silu(self.norm(self.conv(x)))
        return _image_to_graph(x)


# ---------------------------------------------------------------------------
# Encoder / decoder
# ---------------------------------------------------------------------------

class SphericalEncoder(nn.Module):
    """Encoder with 2 ResNet blocks per level (diffusers ``layers_per_block=2``).

    Args:
        channel_list:      ``[in_features, c_0, c_1, ...]``: input width, then the
                           width of every level (finest first).
        neigh_orders_list: ``[V, 9]`` neighbour tensors, *coarsest first*.
        grid_dims:         ``(H, W)`` per level, *coarsest first*.
        time_dim:          Width of the time embedding.
        n_unnormed_first:  Trailing input channels of the very first block
                           that bypass its input GroupNorm.
        topology:          ``"legacy"`` or ``"spherical"``.
    """

    def __init__(self, channel_list: List[int], neigh_orders_list: List[torch.Tensor],
                 grid_dims: List[Tuple[int, int]], time_dim: int = 16,
                 n_unnormed_first: int = 0, topology: str = "legacy"):
        super().__init__()
        _check_topology(topology)
        self.depth = len(channel_list) - 1
        if len(neigh_orders_list) < self.depth or len(grid_dims) < self.depth:
            raise ValueError(f"need at least {self.depth} grid levels, got "
                             f"{len(neigh_orders_list)} neighbour arrays and "
                             f"{len(grid_dims)} grid dims")

        # Pooling between levels (not before the first level).
        self.pools = nn.ModuleList(AvgPool2dGraph(*grid_dims[-i]) for i in range(1, self.depth))

        self.level_blocks = nn.ModuleList()
        for i in range(self.depth):
            neigh = neigh_orders_list[-(i + 1)]          # finest -> coarser
            hw = grid_dims[-(i + 1)] if topology == "spherical" else None
            b0 = GraphResNetBlock(channel_list[i], channel_list[i + 1], neigh, time_dim,
                                  n_unnormed=(n_unnormed_first if i == 0 else 0), grid_hw=hw)
            b1 = GraphResNetBlock(channel_list[i + 1], channel_list[i + 1], neigh, time_dim,
                                  grid_hw=hw)
            self.level_blocks.append(nn.ModuleList([b0, b1]))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> List[torch.Tensor]:
        """Returns the output of every level, *deepest first*."""
        enc_outputs = []
        for i, blocks in enumerate(self.level_blocks):
            if i > 0:
                x = self.pools[i - 1](x)
            for blk in blocks:
                x = blk(x, t_emb)
            enc_outputs.append(x)
        return enc_outputs[::-1]


class SphericalDecoder(nn.Module):
    """Decoder mirroring diffusers ``UpBlock2D``.

    Per level: bilinear upsample -> ``UpsampleConv2d`` -> concat skip ->
    2 x ``GraphResNetBlock``. Ends with GN -> SiLU -> ``DirectNeighConv``
    (diffusers ``conv_norm_out`` + ``conv_out``).

    Args:
        channel_list:      Encoder level widths *reversed* (deepest first).
        out_channels:      Output features.
        neigh_orders_list: ``[V, 9]`` neighbour tensors, *coarsest first*.
        grid_dims:         ``(H, W)`` per level, *coarsest first*.
        time_dim:          Width of the time embedding.
        topology:          ``"legacy"`` or ``"spherical"``.
    """

    def __init__(self, channel_list: List[int], out_channels: int,
                 neigh_orders_list: List[torch.Tensor],
                 grid_dims: List[Tuple[int, int]], time_dim: int = 16,
                 topology: str = "legacy"):
        super().__init__()
        self.depth = len(channel_list) - 1

        self.unpools = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()
        self.level_blocks = nn.ModuleList()
        for i in range(self.depth):
            coarse_hw = grid_dims[-(self.depth - i + 1)]
            fine_hw = grid_dims[-(self.depth - i)]
            neigh = neigh_orders_list[-(self.depth - i)]
            hw = fine_hw if topology == "spherical" else None
            self.unpools.append(AvgUnpool2dGraph(*coarse_hw, topology=topology))
            self.upsample_convs.append(UpsampleConv2d(channel_list[i], *fine_hw,
                                                      topology=topology))
            in_ch = channel_list[i] + channel_list[i + 1]   # upsampled + skip
            out_ch = channel_list[i + 1]
            self.level_blocks.append(nn.ModuleList([
                GraphResNetBlock(in_ch, out_ch, neigh, time_dim, grid_hw=hw),
                GraphResNetBlock(out_ch, out_ch, neigh, time_dim, grid_hw=hw),
            ]))

        self.final_norm = nn.GroupNorm(_gn_num_groups(channel_list[-1]), channel_list[-1])
        self.final = DirectNeighConv(channel_list[-1], out_channels)
        self.register_buffer("final_neigh", neigh_orders_list[-1], persistent=False)
        self.final_hw = tuple(grid_dims[-1]) if topology == "spherical" else None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        _drop_keys(state_dict, prefix, ("final_neigh",))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, enc_outputs: List[torch.Tensor], t_emb: torch.Tensor) -> torch.Tensor:
        """``enc_outputs`` deepest first -> ``[B, V_finest, out_channels]``."""
        x = enc_outputs[0]
        for i, blocks in enumerate(self.level_blocks):
            x = self.upsample_convs[i](self.unpools[i](x))
            x = torch.cat([x, enc_outputs[i + 1]], dim=2)
            for blk in blocks:
                x = blk(x, t_emb)
        x = F.silu(_group_norm_graph(self.final_norm, x))
        return self.final(self.final_neigh, x, self.final_hw)


class SphericalUNetCore(nn.Module):
    """Graph U-Net operating on ``[B, V, F]`` signals.

    Args:
        in_features:       Input features per vertex.
        out_features:      Output features per vertex.
        channel_list:      Width of every level, finest first, for example
                           ``[128, 128, 256, 256]``.
        neigh_orders_list: ``[V, 9]`` neighbour tensors, *coarsest first*.
        grid_dims:         ``(H, W)`` per level, *coarsest first*.
        time_dim:          Width of the time embedding.
        attn_heads:        Heads of the mid-block self-attention.
        n_static:          Trailing input features that bypass the first
                           block's input GroupNorm.
        topology:          ``"legacy"`` or ``"spherical"``.
    """

    def __init__(self, in_features: int, out_features: int,
                 channel_list: List[int], neigh_orders_list: List[torch.Tensor],
                 grid_dims: List[Tuple[int, int]], time_dim: int = 16,
                 attn_heads: int = 1, n_static: int = 0, topology: str = "legacy"):
        super().__init__()
        self.encoder = SphericalEncoder(
            [in_features] + list(channel_list), neigh_orders_list, grid_dims,
            time_dim=time_dim, n_unnormed_first=n_static, topology=topology,
        )

        deepest_ch = channel_list[-1]
        coarsest = neigh_orders_list[0]
        hw = grid_dims[0] if topology == "spherical" else None
        self.mid_resnet1 = GraphResNetBlock(deepest_ch, deepest_ch, coarsest, time_dim,
                                            grid_hw=hw)
        self.mid_attention = GraphSelfAttention(deepest_ch, num_heads=attn_heads)
        self.mid_resnet2 = GraphResNetBlock(deepest_ch, deepest_ch, coarsest, time_dim,
                                            grid_hw=hw)

        self.decoder = SphericalDecoder(
            list(reversed(channel_list)), out_features, neigh_orders_list, grid_dims,
            time_dim=time_dim, topology=topology,
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(x, t_emb)
        h = self.mid_resnet1(enc_out[0], t_emb)
        h = self.mid_attention(h)
        enc_out[0] = self.mid_resnet2(h, t_emb)
        return self.decoder(enc_out, t_emb)


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class SphericalUNetWrapper(nn.Module):
    """Image-interface spherical U-Net: ``model(images, times).sample``.

    Takes ``[B, C, H, W]`` fields on a regular lat-lon grid and a time value
    per sample, and returns a :class:`SphericalUNetOutput` whose ``.sample``
    is ``[B, out_channels, H, W]``.

    Optional inputs, appended inside the wrapper in this channel order::

        [data (in_channels)] [coordinates (n_coord)] [static (n_static_channels)]

    - **Coordinate channels** (``use_coord_channels=True``): fixed sin/cos
      encodings of position, see ``coord_mode``. Longitude channels break the
      longitudinal equivariance of the convolution so the network can learn
      geographically pinned features; latitude channels carry physically real
      information (insolation, Coriolis) that is not recoverable over open
      ocean from static fields alone.
    - **Static channels** (``n_static_channels > 0``): fixed fields such as
      orography, land fraction or ice masks, passed per call (``static=``) or
      cached once with :meth:`set_static_fields`. They bypass the first
      block's GroupNorm so they keep their physical scale.
    - **Source token** (``use_source_token=True``): a learned embedding per
      data source (for example reanalysis vs. model output) added to the time
      embedding. Zero-initialised, so a fresh model is identical to one
      without it; ``source < 0`` or ``source=None`` adds nothing (a null token
      for classifier-free-guidance style dropout).

    Args:
        in_channels:        Data channels of ``images``.
        out_channels:       Output channels.
        image_height:       Grid rows ``H`` (latitudes).
        image_width:        Grid columns ``W`` (longitudes).
        channel_list:       Width per level, finest first. Only the first
                            ``spherical_depth`` entries are used.
        spherical_depth:    Number of U-Net levels. ``H`` and ``W`` must be
                            divisible by ``2 ** (spherical_depth - 1)``.
        time_emb_dim:       Width of the (even) sinusoidal time embedding.
        use_coord_channels: Append coordinate channels.
        coord_mode:         ``"latlon"`` (sin/cos lat and lon, 4 channels),
                            ``"lat"`` (sin/cos lat, 2) or ``"none"``.
        latitudes:          Optional row latitudes in degrees (length ``H``).
                            Default: north-first cell centres for
                            ``topology="spherical"``; ``linspace(90, -90, H)``
                            (end points on the poles, as in the original
                            code) for ``"legacy"``.
        longitudes:         Optional column longitudes in degrees (length
                            ``W``). Default: ``0 .. 360 * (1 - 1/W)``.
        n_static_channels:  Number of static channels.
        use_source_token:   Enable the source embedding.
        n_sources:          Number of distinct source ids.
        bypass_input_norm:  If True, *all* inputs of the first block bypass
                            its GroupNorm (the absolute input level is kept),
                            overriding the static-only split.
        attn_heads:         Heads of the mid-block self-attention.
        topology:           ``"legacy"`` (checkpoint compatible, default) or
                            ``"spherical"`` (geometrically correct, faster),
                            see the module docstring.
        memory_efficient:   Recompute the neighbour gather of every
                            ``DirectNeighConv`` in the backward pass instead
                            of storing it (gather path only, i.e.
                            ``"legacy"``). Forward is bitwise unchanged.
    """

    COORD_CHANNELS = {"latlon": 4, "lat": 2, "none": 0}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_height: int,
        image_width: int,
        channel_list: Sequence[int] = (128, 128, 256, 256),
        spherical_depth: int = 4,
        time_emb_dim: int = 64,
        use_coord_channels: bool = False,
        coord_mode: str = "latlon",
        latitudes: Optional[Sequence[float]] = None,
        longitudes: Optional[Sequence[float]] = None,
        n_static_channels: int = 0,
        use_source_token: bool = False,
        n_sources: int = 2,
        bypass_input_norm: bool = False,
        attn_heads: int = 1,
        topology: str = "legacy",
        memory_efficient: bool = False,
    ):
        super().__init__()
        if coord_mode not in self.COORD_CHANNELS:
            raise ValueError(f"coord_mode must be one of "
                             f"{sorted(self.COORD_CHANNELS)}, got {coord_mode!r}")
        _check_topology(topology)
        if len(channel_list) < spherical_depth:
            raise ValueError(f"channel_list has {len(channel_list)} entries, "
                             f"need at least spherical_depth={spherical_depth}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.H = image_height
        self.W = image_width
        self.time_emb_dim = time_emb_dim
        self.use_coord_channels = use_coord_channels
        self.coord_mode = coord_mode
        self.n_static_channels = n_static_channels
        self.use_source_token = use_source_token
        self.n_sources = n_sources
        self.bypass_input_norm = bypass_input_norm
        self.topology = topology

        neigh_orders_list, grid_dims = build_equiangular_graph(
            image_height, image_width, spherical_depth, topology)
        self.n_levels = len(neigh_orders_list)
        self.grid_dims = grid_dims

        # ---- time embedding ----
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ---- source token ----
        if use_source_token:
            self.source_embed = nn.Embedding(n_sources, time_emb_dim)
            nn.init.zeros_(self.source_embed.weight)

        # ---- coordinate channels ----
        n_coord = self.COORD_CHANNELS[coord_mode] if use_coord_channels else 0
        self.n_coord_channels = n_coord
        if n_coord:
            self._build_coord_features(image_height, image_width, latitudes, longitudes)

        # ---- U-Net ----
        n_in_total = in_channels + n_coord + n_static_channels
        self.unet = SphericalUNetCore(
            in_features=n_in_total,
            out_features=out_channels,
            channel_list=list(channel_list[:spherical_depth]),
            neigh_orders_list=neigh_orders_list,
            grid_dims=grid_dims,
            time_dim=time_emb_dim,
            attn_heads=attn_heads,
            n_static=(n_in_total if bypass_input_norm else n_static_channels),
            topology=topology,
        )

        # Optional cached static field [n_static, H, W] (see set_static_fields).
        self.register_buffer("_static_cache", None, persistent=False)

        self.memory_efficient = memory_efficient
        for m in self.modules():
            if isinstance(m, DirectNeighConv):
                m.recompute = memory_efficient

    # Neighbour arrays are rebuilt from the grid at construction and are not
    # part of the state dict. Checkpoints of the original implementation
    # stored them (``_neigh_<i>`` here, ``neigh_orders`` / ``final_neigh`` in
    # the blocks, which drop their own); drop them so such checkpoints still
    # load with ``strict=True``. The reverse (a new checkpoint into the
    # original code) needs ``strict=False``.
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        _drop_keys(state_dict, prefix, [f"_neigh_{i}" for i in range(self.n_levels)])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _build_coord_features(self, H: int, W: int,
                              latitudes: Optional[Sequence[float]],
                              longitudes: Optional[Sequence[float]]):
        """Register the ``[V, n_coord]`` coordinate buffer.

        Channels, in order: sin(lat), cos(lat), sin(lon), cos(lon); ``"lat"``
        keeps the first two, so it is a prefix of ``"latlon"``. All values lie
        in [-1, 1].
        """
        if latitudes is None and self.topology == "spherical":
            lat_rad = torch.deg2rad(90.0 - 180.0 * (torch.arange(H) + 0.5) / H)
        elif latitudes is None:
            lat_rad = torch.linspace(math.pi / 2, -math.pi / 2, H)
        else:
            lat_rad = torch.deg2rad(torch.as_tensor(latitudes, dtype=torch.float32))
        if longitudes is None:
            lon_rad = torch.linspace(0, 2 * math.pi * (1 - 1 / W), W)
        else:
            lon_rad = torch.deg2rad(torch.as_tensor(longitudes, dtype=torch.float32))
        if lat_rad.shape != (H,) or lon_rad.shape != (W,):
            raise ValueError(f"latitudes/longitudes must have lengths ({H}, {W}), got "
                             f"({tuple(lat_rad.shape)}, {tuple(lon_rad.shape)})")

        lat_grid, lon_grid = torch.meshgrid(lat_rad, lon_rad, indexing="ij")
        coords = torch.stack([
            torch.sin(lat_grid).reshape(-1),
            torch.cos(lat_grid).reshape(-1),
            torch.sin(lon_grid).reshape(-1),
            torch.cos(lon_grid).reshape(-1),
        ], dim=1)[:, :self.COORD_CHANNELS[self.coord_mode]]
        self.register_buffer("_coord_features", coords)       # [V, n_coord]

    def set_static_fields(self, static: Optional[torch.Tensor]):
        """Cache static channels so they need not be passed on every call.

        Useful at inference, when the static geometry changes rarely (for
        example once per coupling interval with a host model). Pass ``None``
        to clear the cache.

        Args:
            static: ``[n_static, H, W]`` or ``[1, n_static, H, W]``.
        """
        if static is None:
            self._static_cache = None
            return
        if self.n_static_channels == 0:
            raise RuntimeError("model was built with n_static_channels=0")
        s = torch.as_tensor(static)
        if s.dim() == 4:
            if s.shape[0] != 1:
                raise ValueError(f"expected a single static field, got batch {s.shape[0]}")
            s = s[0]
        if s.shape != (self.n_static_channels, self.H, self.W):
            raise ValueError(
                f"static field must be [{self.n_static_channels}, {self.H}, {self.W}], "
                f"got {tuple(s.shape)}")
        ref = next(self.parameters())
        self._static_cache = s.to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _broadcast_batch(t: torch.Tensor, B: int, name: str) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B, *t.shape[1:])
        if t.shape[0] != B:
            raise ValueError(f"{name} has batch size {t.shape[0]}, expected {B} or 1")
        return t

    def forward(self, images: torch.Tensor, times: torch.Tensor,
                static: Optional[torch.Tensor] = None,
                source: Optional[torch.Tensor] = None) -> SphericalUNetOutput:
        """
        Args:
            images: ``[B, in_channels, H, W]``.
            times:  ``[B]``, ``[1]`` or scalar time / noise level.
            static: ``[B, n_static, H, W]``, ``[1, ...]`` or ``[n_static, H, W]``;
                    ``None`` uses the cache from :meth:`set_static_fields`.
                    Ignored when ``n_static_channels == 0``.
            source: ``[B]`` (or scalar) integer source ids; entries ``< 0`` and
                    ``None`` add no token. Ignored unless ``use_source_token``.

        Returns:
            :class:`SphericalUNetOutput` with ``.sample`` ``[B, out_channels, H, W]``.
        """
        B, C, H, W = images.shape
        if (C, H, W) != (self.in_channels, self.H, self.W):
            raise ValueError(f"expected images [B, {self.in_channels}, {self.H}, {self.W}], "
                             f"got {tuple(images.shape)}")
        V = H * W
        x = _image_to_graph(images)                                   # [B, V, C]

        if self.n_coord_channels:
            coords = self._coord_features.to(x.dtype).unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([x, coords], dim=2)

        if self.n_static_channels:
            if static is None:
                if self._static_cache is None:
                    raise RuntimeError(
                        "model expects static channels but none were given. Pass "
                        "`static=` to forward, or call set_static_fields() first.")
                static = self._static_cache
            if static.dim() == 3:
                static = static.unsqueeze(0)
            if static.dim() != 4 or static.shape[1:] != (self.n_static_channels, H, W):
                raise ValueError(f"static must be [B, {self.n_static_channels}, {H}, {W}], "
                                 f"got {tuple(static.shape)}")
            static = self._broadcast_batch(static, B, "static")
            s = _image_to_graph(static.to(device=x.device, dtype=x.dtype))
            x = torch.cat([x, s], dim=2)

        times = torch.as_tensor(times, device=images.device)
        if times.dim() > 1:
            if times.numel() != times.shape[0]:
                raise ValueError(f"times must be scalar, [1] or [B], got {tuple(times.shape)}")
            times = times.reshape(times.shape[0])
        times = self._broadcast_batch(times, B, "times")
        # The sinusoid is computed in float32 (float64 for a double model) and
        # only then cast to the parameter dtype: fp16/bf16 cannot resolve the
        # high frequencies at t ~ 1000, and integer steps must work too.
        param_dtype = self.time_mlp[0].weight.dtype
        emb_dtype = torch.float64 if param_dtype == torch.float64 else torch.float32
        t_emb = self.time_embed(times.to(emb_dtype)).to(param_dtype)
        t_emb = self.time_mlp(t_emb)                                  # [B, D_time]

        # Added after the MLP, so a zero-initialised embedding leaves t_emb
        # bit-identical to the token-free model.
        if self.use_source_token and source is not None:
            src = torch.as_tensor(source, device=t_emb.device)
            if src.is_floating_point():
                raise TypeError("source ids must be integers")
            src = self._broadcast_batch(src.long(), B, "source")
            if bool((src >= self.n_sources).any()):
                raise ValueError(f"source ids must be < n_sources={self.n_sources}")
            keep = (src >= 0).to(t_emb.dtype).unsqueeze(1)
            t_emb = t_emb + self.source_embed(src.clamp(min=0)) * keep

        x = self.unet(x, t_emb)                                       # [B, V, C_out]
        x = x.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)
        return SphericalUNetOutput(sample=x)


#: Short alias.
SphericalUNet = SphericalUNetWrapper

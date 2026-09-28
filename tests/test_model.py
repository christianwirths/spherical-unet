"""Tests for spherical_unet.model."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from spherical_unet import (
    AvgUnpool2dGraph,
    DirectNeighConv,
    GraphResNetBlock,
    SinusoidalTimeEmbedding,
    SphericalUNetWrapper,
    build_equiangular_graph,
    build_equiangular_neighbours,
    spherical_pad,
)

H, W = 16, 32
SMALL = dict(channel_list=(8, 8, 16), spherical_depth=3, time_emb_dim=16)
# Kernel position (row, col) of each neighbour slot [self, N, NE, E, SE, S, SW, W, NW].
KERNEL_POS = [(1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0)]


def _reference_neighbours(H, W):
    """Loop implementation of the original (legacy) neighbour table."""
    neigh = np.zeros((H * W, 9), dtype=np.int64)
    for i in range(H):
        for j in range(W):
            v = i * W + j
            je, jw = (j + 1) % W, (j - 1) % W
            neigh[v, 0] = v
            neigh[v, 3] = i * W + je
            neigh[v, 7] = i * W + jw
            if i > 0:
                neigh[v, 1] = (i - 1) * W + j
                neigh[v, 2] = (i - 1) * W + je
                neigh[v, 8] = (i - 1) * W + jw
            else:
                jr = (j + W // 2) % W
                neigh[v, 1] = jr
                neigh[v, 2] = (jr - 1) % W
                neigh[v, 8] = (jr + 1) % W
            if i < H - 1:
                neigh[v, 5] = (i + 1) * W + j
                neigh[v, 4] = (i + 1) * W + je
                neigh[v, 6] = (i + 1) * W + jw
            else:
                jr = (j + W // 2) % W
                last = (H - 1) * W
                neigh[v, 5] = last + jr
                neigh[v, 4] = last + (jr - 1) % W
                neigh[v, 6] = last + (jr + 1) % W
    return neigh


def _model(**kw):
    torch.manual_seed(0)
    cfg = dict(in_channels=2, out_channels=3, image_height=H, image_width=W, **SMALL)
    cfg.update(kw)
    return SphericalUNetWrapper(**cfg)


def _perturb_zero_init(model, scale=0.1):
    """Give the zero-initialised source token non-trivial values."""
    if model.use_source_token:
        with torch.no_grad():
            model.source_embed.weight.normal_(0, scale)


# --------------------------------------------------------------------- graph

@pytest.mark.parametrize("hw", [(4, 8), (6, 12), (5, 7), (22, 45)])
def test_legacy_neighbours_match_reference(hw):
    np.testing.assert_array_equal(build_equiangular_neighbours(*hw, "legacy"),
                                  _reference_neighbours(*hw))


@pytest.mark.parametrize("topology", ["legacy", "spherical"])
def test_neighbours_in_range(topology):
    neigh = build_equiangular_neighbours(H, W, topology)
    assert neigh.shape == (H * W, 9) and neigh.dtype == np.int64
    assert neigh.min() >= 0 and neigh.max() < H * W
    np.testing.assert_array_equal(neigh[:, 0], np.arange(H * W))


@pytest.mark.parametrize("topology", ["legacy", "spherical"])
def test_neighbour_relation_is_symmetric(topology):
    # u in neigh(v)  <=>  v in neigh(u), for even W.
    neigh = build_equiangular_neighbours(H, W, topology)
    for v in range(H * W):
        for u in neigh[v]:
            assert v in neigh[u], (topology, v, u)


def test_spherical_neighbour_slots_away_from_poles():
    # Off the pole rows, going in direction k and then in the opposite
    # direction returns to the start.
    opposite = [0, 5, 6, 7, 8, 1, 2, 3, 4]
    neigh = build_equiangular_neighbours(H, W, "spherical")
    interior = np.arange(W, (H - 1) * W)
    for k in range(1, 9):
        u = neigh[interior, k]
        np.testing.assert_array_equal(neigh[u, opposite[k]], interior)


def test_pole_diagonals_by_topology():
    # Vertex (0, 0): its NE neighbour across the pole is at the antipodal
    # longitude + 1 column geometrically, - 1 column in the legacy table.
    legacy = build_equiangular_neighbours(H, W, "legacy")
    sph = build_equiangular_neighbours(H, W, "spherical")
    assert sph[0, 2] == W // 2 + 1 and legacy[0, 2] == W // 2 - 1
    assert sph[0, 1] == legacy[0, 1] == W // 2


def test_graph_levels_and_divisibility():
    neigh, dims = build_equiangular_graph(176, 360, 4)
    assert dims == [(22, 45), (44, 90), (88, 180), (176, 360)]
    assert [n.shape[0] for n in neigh] == [h * w for h, w in dims]
    with pytest.raises(ValueError):
        build_equiangular_graph(170, 360, 4)


def test_spherical_pad():
    x = torch.arange(4 * 8, dtype=torch.float32).view(1, 1, 4, 8)
    p = spherical_pad(x, 1)
    assert p.shape == (1, 1, 6, 10)
    torch.testing.assert_close(p[0, 0, 1:-1, 1:-1], x[0, 0])
    torch.testing.assert_close(p[0, 0, 0, 1:-1], x[0, 0, 0].roll(4))      # across north pole
    torch.testing.assert_close(p[0, 0, -1, 1:-1], x[0, 0, -1].roll(4))    # across south pole
    torch.testing.assert_close(p[0, 0, 1:-1, 0], x[0, 0, :, -1])          # circular lon
    torch.testing.assert_close(p[0, 0, 1:-1, -1], x[0, 0, :, 0])


def test_spherical_topology_equals_padded_conv2d():
    # DirectNeighConv on the spherical graph is exactly a 3x3 conv of the
    # spherically padded field, with the neighbour-major weight layout.
    torch.manual_seed(0)
    cin, cout = 3, 5
    conv = DirectNeighConv(cin, cout).double()
    neigh = torch.from_numpy(build_equiangular_neighbours(H, W, "spherical"))
    x = torch.randn(2, cin, H, W, dtype=torch.float64)
    got = conv(neigh, x.permute(0, 2, 3, 1).reshape(2, H * W, cin))

    w = conv.weight.weight.view(cout, 9, cin)
    kernel = torch.zeros(cout, cin, 3, 3, dtype=torch.float64)
    for n, (r, c) in enumerate(KERNEL_POS):
        kernel[:, :, r, c] = w[:, n]
    ref = F.conv2d(spherical_pad(x, 1), kernel, conv.weight.bias)
    torch.testing.assert_close(got, ref.permute(0, 2, 3, 1).reshape(2, H * W, cout))


def test_spherical_unpool_is_longitude_periodic():
    x = torch.randn(1, 4, 8, 16)
    up = AvgUnpool2dGraph(8, 16, topology="spherical")
    g = lambda t: t.permute(0, 2, 3, 1).reshape(1, -1, 4)   # noqa: E731
    y = up(g(x)).view(1, 16, 32, 4)
    y_rolled = up(g(x.roll(3, dims=-1))).view(1, 16, 32, 4)
    torch.testing.assert_close(y_rolled, y.roll(6, dims=2))


# --------------------------------------------------------------------- blocks

def test_time_embedding():
    emb = SinusoidalTimeEmbedding(16)(torch.tensor([0.0, 1.0, 500.0]))
    assert emb.shape == (3, 16)
    torch.testing.assert_close(emb[0], torch.cat([torch.zeros(8), torch.ones(8)]))
    assert SinusoidalTimeEmbedding(8)(torch.tensor([1.0], dtype=torch.float64)).dtype \
        == torch.float64
    with pytest.raises(ValueError):
        SinusoidalTimeEmbedding(15)


def test_resnet_block_norm_bypass():
    neigh = torch.from_numpy(build_equiangular_neighbours(8, 16))
    blk = GraphResNetBlock(4, 8, neigh, time_dim=8, n_unnormed=4)
    assert blk.norm1 is None and blk.n_normed == 0
    with pytest.raises(ValueError):
        GraphResNetBlock(4, 8, neigh, time_dim=8, n_unnormed=5)
    # GroupNorm removes a uniform shift, which is why the bypass exists.
    blk = GraphResNetBlock(4, 8, neigh, time_dim=8)
    x = torch.randn(1, 8 * 16, 4)
    torch.testing.assert_close(blk.norm1(x.transpose(1, 2)),
                               blk.norm1((x + 3.0).transpose(1, 2)), atol=1e-5, rtol=0)


# --------------------------------------------------------------------- wrapper

@pytest.mark.parametrize("kw", [
    {},
    {"topology": "spherical"},
    {"use_coord_channels": True, "coord_mode": "latlon"},
    {"use_coord_channels": True, "coord_mode": "lat", "n_static_channels": 2},
    {"use_source_token": True, "n_sources": 3},
    {"bypass_input_norm": True, "attn_heads": 2},
])
def test_forward_backward(kw):
    model = _model(**kw)
    _perturb_zero_init(model)
    B = 2
    x = torch.randn(B, 2, H, W)
    static = torch.rand(B, 2, H, W) if kw.get("n_static_channels") else None
    source = torch.tensor([0, 2]) if kw.get("use_source_token") else None
    out = model(x, torch.rand(B) * 100, static=static, source=source).sample
    assert out.shape == (B, 3, H, W)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters without gradient: {missing}"


def test_scalar_and_broadcast_time():
    model = _model().eval()
    x = torch.randn(3, 2, H, W)
    with torch.no_grad():
        a = model(x, torch.tensor(5.0)).sample
        b = model(x, torch.tensor([5.0])).sample
        c = model(x, torch.full((3,), 5.0)).sample
    torch.testing.assert_close(a, c)
    torch.testing.assert_close(b, c)
    with pytest.raises(ValueError):
        model(x, torch.ones(2))


def test_input_shape_is_checked():
    model = _model()
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, H, W), torch.ones(1))


def test_spherical_topology_is_longitude_equivariant():
    # Without coordinate channels, a longitude roll by a multiple of the
    # coarsest cell (2^(depth-1) columns) commutes with the whole network.
    model = _model(topology="spherical").double().eval()
    x = torch.randn(1, 2, H, W, dtype=torch.float64)
    t = torch.tensor([3.0], dtype=torch.float64)
    shift = 2 ** (SMALL["spherical_depth"] - 1) * 3
    with torch.no_grad():
        y = model(x, t).sample
        y_rolled = model(x.roll(shift, dims=-1), t).sample
    torch.testing.assert_close(y_rolled, y.roll(shift, dims=-1))


def test_source_token_zero_init_and_null():
    base = _model()
    tok = _model(use_source_token=True)
    tok.load_state_dict(base.state_dict(), strict=False)
    x, t = torch.randn(2, 2, H, W), torch.ones(2)
    with torch.no_grad():
        ref = base(x, t).sample
        torch.testing.assert_close(tok(x, t, source=torch.tensor([0, 1])).sample, ref)
        _perturb_zero_init(tok)
        torch.testing.assert_close(tok(x, t, source=torch.tensor([-1, -1])).sample, ref)
        torch.testing.assert_close(tok(x, t).sample, ref)
        assert not torch.allclose(tok(x, t, source=torch.tensor([1, 1])).sample, ref)


def test_static_cache_matches_explicit():
    model = _model(n_static_channels=2).eval()
    x, t = torch.randn(2, 2, H, W), torch.ones(2)
    s = torch.rand(2, H, W)
    with torch.no_grad():
        explicit = model(x, t, static=s.expand(2, -1, -1, -1)).sample
        model.set_static_fields(s)
        cached = model(x, t).sample
        torch.testing.assert_close(cached, explicit)
        model.set_static_fields(None)
        with pytest.raises(RuntimeError):
            model(x, t)
    with pytest.raises(ValueError):
        model.set_static_fields(torch.rand(3, H, W))


def test_absolute_level_visible_with_bypass():
    model = _model(bypass_input_norm=True).eval()
    x, t = torch.randn(1, 2, H, W), torch.ones(1)
    with torch.no_grad():
        d = (model(x + 0.5, t).sample - model(x, t).sample).abs().mean()
    assert d > 1e-4
    assert model.unet.encoder.level_blocks[0][0].norm1 is None
    assert model.unet.encoder.level_blocks[0][1].norm1 is not None


def test_coordinate_features():
    lats = np.linspace(-87.5, 87.5, H)          # south first, cell centres
    model = _model(use_coord_channels=True, coord_mode="latlon", latitudes=lats)
    c = model._coord_features.view(H, W, 4)
    assert math.isclose(c[0, 0, 0].item(), math.sin(math.radians(-87.5)), rel_tol=1e-6)
    assert math.isclose(c[-1, 0, 1].item(), math.cos(math.radians(87.5)), rel_tol=1e-5)
    torch.testing.assert_close(c[:, 0, :2], c[:, 7, :2])
    default = _model(use_coord_channels=True, coord_mode="lat")._coord_features
    assert default.shape == (H * W, 2)
    assert math.isclose(default[0, 0].item(), 1.0, abs_tol=1e-6)   # north pole first


def test_state_dict_has_no_neighbour_buffers_and_loads_legacy_keys():
    model = _model(n_static_channels=1)
    sd = model.state_dict()
    assert not any("neigh" in k for k in sd)
    # Checkpoints from the original implementation carried the neighbour
    # tables as persistent buffers; they must still load strictly.
    legacy = dict(sd)
    for name, buf in model.named_buffers():
        if name.endswith(("neigh_orders", "final_neigh")):
            legacy[name] = buf.clone()
    for i in range(model.n_levels):
        legacy[f"_neigh_{i}"] = torch.zeros(1, dtype=torch.long)
    fresh = _model(n_static_channels=1)
    fresh.load_state_dict(legacy, strict=True)
    # Nested under a parent module, with a prefix.
    parent = torch.nn.Module()
    parent.backbone = _model(n_static_channels=1)
    parent.load_state_dict({f"backbone.{k}": v for k, v in legacy.items()}, strict=True)


def test_dtype_and_device_moves():
    model = _model(use_coord_channels=True, n_static_channels=1).double()
    model.set_static_fields(torch.rand(1, H, W))
    assert model._static_cache.dtype == torch.float64
    out = model(torch.randn(1, 2, H, W, dtype=torch.float64),
                torch.ones(1, dtype=torch.float64)).sample
    assert out.dtype == torch.float64
    blk = model.unet.encoder.level_blocks[0][0]
    assert blk.neigh_orders.dtype == torch.int64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_forward():
    model = _model(topology="spherical", n_static_channels=1).cuda()
    out = model(torch.randn(2, 2, H, W, device="cuda"), torch.ones(2, device="cuda"),
                static=torch.rand(2, 1, H, W, device="cuda")).sample
    assert out.is_cuda and torch.isfinite(out).all()


# --------------------------------------------------------------------- execution paths, dtypes, validation

@pytest.mark.parametrize("hw", [(6, 8), (6, 5), (4, 9)])
def test_padded_conv2d_path_matches_gather_including_odd_width(hw):
    h, w = hw
    torch.manual_seed(0)
    conv = DirectNeighConv(3, 4).double()
    neigh = torch.from_numpy(build_equiangular_neighbours(h, w, "spherical"))
    x = torch.randn(2, h * w, 3, dtype=torch.float64)
    torch.testing.assert_close(conv(neigh, x, grid_hw=(h, w)), conv(neigh, x))


def test_spherical_model_fast_path_matches_gather():
    # 16 x 40 at depth 4: coarsest level 2 x 5 (odd width).
    kw = dict(image_height=16, image_width=40, channel_list=(8, 8, 16, 16), spherical_depth=4)
    model = _model(topology="spherical", **kw).double().eval()
    x = torch.randn(2, 2, 16, 40, dtype=torch.float64)
    t = torch.tensor([1.0, 50.0], dtype=torch.float64)
    with torch.no_grad():
        fast = model(x, t).sample
        # Force the gather path on the spherical neighbour tables.
        def table(hw):
            return torch.from_numpy(build_equiangular_neighbours(*hw, "spherical"))
        for m in model.modules():
            if isinstance(m, GraphResNetBlock):
                assert m.neigh_orders is None       # the fast path keeps no table
                m.neigh_orders, m.grid_hw = table(m.grid_hw), None
        dec = model.unet.decoder
        dec.final_neigh, dec.final_hw = table(dec.final_hw), None
        slow = model(x, t).sample
    torch.testing.assert_close(fast, slow, rtol=0, atol=1e-12)
    # Longitude equivariance with an odd coarse width (roll by one coarse cell).
    with torch.no_grad():
        torch.testing.assert_close(model(x.roll(8, dims=-1), t).sample, fast.roll(8, dims=-1))


def test_memory_efficient_flag_is_live():
    model = _model(memory_efficient=True)
    convs = [m for m in model.modules() if isinstance(m, DirectNeighConv)]
    assert all(c.recompute for c in convs)
    model.memory_efficient = False
    assert not any(c.recompute for c in convs)
    with pytest.warns(UserWarning):
        _model(topology="spherical", memory_efficient=True)


def test_memory_efficient_gives_same_gradients():
    x, t = torch.randn(2, 2, H, W), torch.rand(2) * 10
    grads = []
    for flag in (False, True):
        model = _model(memory_efficient=flag)
        model(x, t).sample.square().mean().backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    torch.testing.assert_close(grads[0], grads[1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("times", [torch.tensor([1.0, 999.0]), torch.tensor([1, 999]), 7])
def test_time_dtypes(dtype, times):
    major, minor = (int(v) for v in torch.__version__.split(".")[:2])
    if dtype == torch.float16 and (major, minor) < (2, 1):
        pytest.skip("no fp16 matmul on CPU before torch 2.1")
    model = _model().to(dtype)
    out = model(torch.randn(2, 2, H, W, dtype=dtype), times).sample
    assert out.dtype == dtype and torch.isfinite(out.float()).all()


def test_times_column_vector_and_bad_shapes():
    model = _model().eval()
    x = torch.randn(2, 2, H, W)
    with torch.no_grad():
        a = model(x, torch.tensor([[3.0], [4.0]])).sample
        b = model(x, torch.tensor([3.0, 4.0])).sample
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError):
        model(x, torch.ones(2, 2))


def test_static_and_source_validation():
    model = _model(n_static_channels=2, use_source_token=True, n_sources=2)
    x, t = torch.randn(2, 2, H, W), torch.ones(2)
    with pytest.raises(ValueError):
        model(x, t, static=torch.rand(2, 3, H, W))
    with pytest.raises(ValueError):
        model(x, t, static=torch.rand(2, 2, H, W + 2))
    s = torch.rand(2, 2, H, W)
    with pytest.raises(ValueError):
        model(x, t, static=s, source=torch.tensor([0, 2]))
    with pytest.raises(TypeError):
        model(x, t, static=s, source=torch.tensor([0.0, 1.0]))


def test_spherical_default_latitudes_are_cell_centres():
    model = _model(topology="spherical", use_coord_channels=True, coord_mode="lat")
    sin_lat = model._coord_features.view(H, W, 2)[:, 0, 0]
    expected = torch.sin(torch.deg2rad(90 - 180 * (torch.arange(H) + 0.5) / H))
    torch.testing.assert_close(sin_lat, expected)

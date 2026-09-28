"""Tests for spherical_unet.checkpoint and spherical_unet.legacy."""

import pytest
import torch

from spherical_unet import SphericalUNetWrapper, verify_widening, widen_input_channels
from spherical_unet.legacy import ChebConvLayer, cheb_conv

H, W = 16, 32


def _model(n_static, width=12, **kw):
    torch.manual_seed(1)
    return SphericalUNetWrapper(in_channels=3, out_channels=2, image_height=H, image_width=W,
                                channel_list=(width, 16), spherical_depth=2, time_emb_dim=8,
                                use_coord_channels=True, coord_mode="lat",
                                n_static_channels=n_static, **kw)


@pytest.mark.parametrize("n_old_static,n_new_static", [(0, 3), (1, 3)])
def test_widening_is_exact(n_old_static, n_new_static):
    old = _model(n_old_static).double()
    new = _model(n_new_static).double()
    n_old = 3 + 2 + n_old_static
    n_new = 3 + 2 + n_new_static
    new.load_state_dict(widen_input_channels(old.state_dict(), n_old, n_new), strict=True)

    x = torch.randn(2, 3, H, W, dtype=torch.float64)
    t = torch.tensor([1.0, 7.0], dtype=torch.float64)
    # Old static channels (if any) are the prefix of the new ones.
    extra = 10 * torch.randn(2, n_new_static, H, W, dtype=torch.float64)
    old_static = extra[:, :n_old_static] if n_old_static else None
    with torch.no_grad():
        ref = old(x, t, static=old_static).sample
        got = new(x, t, static=extra).sample
    assert (ref - got).abs().max().item() < 1e-12


def test_widening_from_identity_shortcut():
    # 3 data + 2 coord + 3 static = 8 = first width: the old block has an
    # Identity shortcut, the widened one needs [I | 0].
    old = _model(3, width=8).double()
    new = _model(5, width=8).double()
    new.load_state_dict(widen_input_channels(old.state_dict(), 8, 10), strict=True)
    x = torch.randn(1, 3, H, W, dtype=torch.float64)
    t = torch.ones(1, dtype=torch.float64)
    extra = 10 * torch.randn(1, 5, H, W, dtype=torch.float64)
    with torch.no_grad():
        ref = old(x, t, static=extra[:, :3]).sample
        got = new(x, t, static=extra).sample
    assert (ref - got).abs().max().item() < 1e-12


def test_widening_to_identity_shortcut_is_refused():
    old = _model(1, width=8)
    with pytest.raises(ValueError, match="Identity shortcut"):
        widen_input_channels(old.state_dict(), 6, 8)


def test_verify_widening_helper():
    old, new = _model(0).double(), _model(2).double()
    new.load_state_dict(widen_input_channels(old.state_dict(), 5, 7), strict=True)
    x = torch.randn(1, 3, H, W, dtype=torch.float64)
    t = torch.ones(1, dtype=torch.float64)
    assert verify_widening(old, new, x, t, atol=1e-12) < 1e-12
    # A wrong (prefix-copy) remapping must be caught.
    bad = widen_input_channels(old.state_dict(), 5, 7)
    key = "unet.encoder.level_blocks.0.0.conv1.weight.weight"
    w = old.state_dict()[key]
    bad[key] = torch.cat([w, torch.zeros(w.shape[0], 18, dtype=w.dtype)], dim=1)
    new.load_state_dict(bad, strict=True)
    with pytest.raises(AssertionError):
        verify_widening(old, new, x, t, atol=1e-12)


def test_verify_widening_with_existing_static():
    old, new = _model(1).double(), _model(3).double()
    new.load_state_dict(widen_input_channels(old.state_dict(), 6, 8), strict=True)
    x = torch.randn(2, 3, H, W, dtype=torch.float64)
    t = torch.ones(2, dtype=torch.float64)
    assert verify_widening(old, new, x, t, atol=1e-12) < 1e-12
    with pytest.raises(TypeError):
        verify_widening(old, new, x, t, static=None)


def test_widening_data_channel_with_insert_at():
    # bypass_input_norm: no norm1, so a new *data* channel is possible; it sits
    # before the coordinate channels, at index n_old_data = 3.
    def m(n_data):
        torch.manual_seed(1)
        return SphericalUNetWrapper(in_channels=n_data, out_channels=2, image_height=H,
                                    image_width=W, channel_list=(12, 16), spherical_depth=2,
                                    time_emb_dim=8, use_coord_channels=True, coord_mode="lat",
                                    bypass_input_norm=True).double().eval()
    old, new = m(3), m(4)
    new.load_state_dict(widen_input_channels(old.state_dict(), 5, 6, insert_at=3), strict=True)
    x = torch.randn(2, 3, H, W, dtype=torch.float64)
    x_new = torch.cat([x, 10 * torch.randn(2, 1, H, W, dtype=torch.float64)], dim=1)
    t = torch.ones(2, dtype=torch.float64)
    with torch.no_grad():
        diff = (old(x, t).sample - new(x_new, t).sample).abs().max().item()
    assert diff < 1e-12
    # Appending instead (the default) mis-wires the coordinate weights.
    new.load_state_dict(widen_input_channels(old.state_dict(), 5, 6), strict=True)
    with torch.no_grad():
        assert (old(x, t).sample - new(x_new, t).sample).abs().max().item() > 1e-3
    with pytest.raises(ValueError):
        widen_input_channels(old.state_dict(), 5, 6, insert_at=6)


def test_widening_with_prefix_and_errors():
    old = _model(0)
    sd = {f"backbone.{k}": v for k, v in old.state_dict().items()}
    out = widen_input_channels(sd, 5, 6, prefix="backbone.")
    assert out["backbone.unet.encoder.level_blocks.0.0.conv1.weight.weight"].shape[1] == 54
    with pytest.raises(KeyError):
        widen_input_channels(sd, 5, 6)
    with pytest.raises(ValueError):
        widen_input_channels(old.state_dict(), 5, 4)
    with pytest.raises(ValueError):
        widen_input_channels(old.state_dict(), 4, 6)


def test_cheb_conv_order_one_is_linear():
    torch.manual_seed(0)
    V, Fin, Fout = 10, 3, 4
    lap = torch.eye(V).to_sparse()
    x = torch.randn(2, V, Fin)
    weight = torch.randn(1, Fin, Fout)
    torch.testing.assert_close(cheb_conv(lap, x, weight), x @ weight[0])
    layer = ChebConvLayer(Fin, Fout, K=3)
    assert layer(lap, x).shape == (2, V, Fout)

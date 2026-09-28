"""
Checkpoint surgery for :class:`spherical_unet.SphericalUNetWrapper`.

``widen_input_channels`` grows the input width of a trained model by appending
new, zero-initialised input channels (for example extra static fields), so the
widened model reproduces the original exactly at initialisation and can be
fine-tuned from there.

Only the first encoder block sees the raw input, so only its conv1 and
residual shortcut change (plus a new ``shortcut.bias`` when the old block had
an Identity shortcut)::

    unet.encoder.level_blocks.0.0.conv1.weight.weight   [out, 9 * n_old] -> [out, 9 * n_new]
    unet.encoder.level_blocks.0.0.shortcut.weight       [out, n_old]     -> [out, n_new]

(the doubled ``.weight`` is because ``DirectNeighConv.weight`` is itself an
``nn.Linear``).

The conv1 columns are *neighbour-major*: ``DirectNeighConv`` gathers
``[B, V, 9, C]`` and flattens to ``[B, V, 9 * C]``, so column ``n * C + c``
multiplies channel ``c`` of neighbour ``n``. Growing ``C`` is therefore not a
prefix copy; each of the 9 neighbour blocks is relocated::

    old column  n * n_old + c   ->   new column  n * n_new + c'
    (c' = c for c < insert_at, c' = c + n_added otherwise; new channels zero)

Getting this wrong would not crash; it would silently scramble which weight
multiplies which neighbour's channel. :func:`verify_widening` is the guard.

Requirements:

- The new channels must bypass the first block's ``norm1``, i.e. they must
  lie in its trailing ``n_unnormed`` channels (static channels, or
  ``bypass_input_norm=True``). ``norm1`` is left untouched; GroupNorm
  statistics over the old channels are then unchanged, which is what makes
  the widening exact (up to float round-off).
- ``insert_at`` must be where the new channels sit in the new model's input.
  The wrapper's order is ``[data][coords][static]``: new static channels are
  appended (``insert_at=None``), while new *data* channels go at
  ``insert_at=n_old_data`` (only possible with ``bypass_input_norm=True``).
  A wrong position is not detected by ``load_state_dict``; use
  :func:`verify_widening`.
"""

from typing import Dict, Optional

import torch

from .model import N_NEIGHBOURS

FIRST_CONV = "unet.encoder.level_blocks.0.0.conv1.weight.weight"
FIRST_SHORTCUT = "unet.encoder.level_blocks.0.0.shortcut.weight"

__all__ = ["widen_input_channels", "verify_widening", "FIRST_CONV", "FIRST_SHORTCUT"]


def widen_input_channels(state: Dict[str, torch.Tensor], n_old_in: int, n_new_in: int,
                         prefix: str = "", insert_at: Optional[int] = None
                         ) -> Dict[str, torch.Tensor]:
    """Return a copy of ``state`` with the first block's input width grown.

    Args:
        state:    State dict of a ``SphericalUNetWrapper`` (or of a module that
                  contains one under ``prefix``).
        n_old_in: Total input width of the first block in ``state``.
        n_new_in: Target width (``>= n_old_in``); new channels are
                  zero-initialised.
        prefix:   Key prefix of the wrapper inside ``state``, for example
                  ``"backbone."``.
        insert_at: Input index of the first new channel (``0 .. n_old_in``);
                  default ``None`` appends after all old channels.

    Every other tensor is passed through unchanged.
    """
    if n_new_in < n_old_in:
        raise ValueError(f"cannot shrink input width {n_old_in} -> {n_new_in}")
    out = dict(state)
    if n_new_in == n_old_in:
        return out
    if insert_at is None:
        insert_at = n_old_in
    if not 0 <= insert_at <= n_old_in:
        raise ValueError(f"insert_at must lie in [0, {n_old_in}], got {insert_at}")
    n_add = n_new_in - n_old_in
    # Positions of the old channels inside the new input.
    old_pos = list(range(insert_at)) + list(range(insert_at + n_add, n_new_in))

    conv_key, sc_key = prefix + FIRST_CONV, prefix + FIRST_SHORTCUT
    if conv_key not in out:
        raise KeyError(f"{conv_key!r} not in state dict; wrong prefix?")

    w = out[conv_key]                                   # [out_ch, 9 * n_old_in]
    out_ch = w.shape[0]
    if w.shape[1] != N_NEIGHBOURS * n_old_in:
        raise ValueError(f"{conv_key} has width {w.shape[1]}, expected "
                         f"{N_NEIGHBOURS * n_old_in} for n_old_in={n_old_in}")
    new = w.new_zeros((out_ch, N_NEIGHBOURS, n_new_in))
    new[:, :, old_pos] = w.view(out_ch, N_NEIGHBOURS, n_old_in)
    out[conv_key] = new.view(out_ch, N_NEIGHBOURS * n_new_in)

    # The residual shortcut is a Linear when in_ch != out_ch, else Identity.
    if n_new_in == out_ch:
        raise ValueError(
            f"n_new_in == first level width ({out_ch}): the widened block would use an "
            f"Identity shortcut, which cannot reproduce the old learned shortcut. Choose "
            f"a different number of new channels.")
    if sc_key in out:
        s = out[sc_key]                                 # [out_ch, n_old_in]
        new_s = s.new_zeros((out_ch, n_new_in))
        new_s[:, old_pos] = s
    else:
        # Old block had in_ch == out_ch (Identity): the equivalent Linear is
        # [I | 0] with zero bias.
        new_s = w.new_zeros((out_ch, n_new_in))
        new_s[:, old_pos] = torch.eye(out_ch, dtype=w.dtype, device=w.device)
        out[prefix + FIRST_SHORTCUT.replace(".weight", ".bias")] = w.new_zeros(out_ch)
    out[sc_key] = new_s
    return out


@torch.no_grad()
def verify_widening(model_old, model_new, images: torch.Tensor, times: torch.Tensor,
                    extra: Optional[torch.Tensor] = None, atol: float = 1e-5,
                    **forward_kwargs) -> float:
    """Check that a model widened with *static* channels reproduces the original.

    Because the new input columns are zero, the output must match the original
    for *any* values of the new channels, so the check feeds large random
    ones: passing with zeros would prove much less. ``extra`` is the new
    model's full static field ``[B, n_static_new, H, W]``; the old model gets
    its leading ``n_static_old`` channels. By default a random field (times
    10) is used.

    The tolerance cannot be zero: the widened ``nn.Linear`` sums over more
    (zero) terms in a different order, and float addition is not associative.
    To separate round-off from a real wiring bug, run both models in float64
    with a tight ``atol``; round-off then shrinks by ~1e-9, a bug does not.

    Other keyword arguments (for example ``source=``) go to both models.
    Returns the maximum absolute difference; raises ``AssertionError`` above
    ``atol``.
    """
    if "static" in forward_kwargs:
        raise TypeError("pass the new model's static field as `extra`, not `static`")
    n_old, n_new = model_old.n_static_channels, model_new.n_static_channels
    if n_new <= n_old:
        raise ValueError(f"model_new must have more static channels than model_old "
                         f"({n_new} vs {n_old})")
    model_old.eval()
    model_new.eval()
    B, _, H, W = images.shape
    if extra is None:
        extra = 10.0 * torch.randn(B, n_new, H, W, device=images.device, dtype=images.dtype)
    ref = model_old(images, times, static=(extra[:, :n_old] if n_old else None),
                    **forward_kwargs).sample
    got = model_new(images, times, static=extra, **forward_kwargs).sample
    diff = (ref - got).abs().max().item()
    if diff > atol:
        raise AssertionError(f"widened model does not reproduce the original: max |diff| "
                             f"= {diff:.3e} > {atol:.3e}")
    return diff

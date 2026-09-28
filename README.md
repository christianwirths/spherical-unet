# spherical-unet

Graph U-Net for regular lat-lon grids with spherical topology (circular
longitude, pole crossing to the antipodal meridian). diffusers-style
interface: `model(x, t).sample`, `[B, C, H, W]` in and out.

`src/spherical_unet/model.py` is self-contained (torch + numpy) and can be
copied into a project as is.

```bash
pip install git+https://github.com/christianwirths/spherical-unet.git
```

```python
from spherical_unet import SphericalUNetWrapper

model = SphericalUNetWrapper(
    in_channels=3, out_channels=3, image_height=176, image_width=360,
    channel_list=(128, 128, 256, 256), spherical_depth=4, time_emb_dim=64,
    use_coord_channels=True, coord_mode="lat",
    n_static_channels=3,            # bypass the first GroupNorm
    use_source_token=True,          # zero-init embedding, source < 0 = null
    topology="spherical",
)
y = model(x, t, static=static, source=source).sample
```

## Notes

- `topology="legacy"` (default) is bit-compatible with the original
  implementation, including its quirks (mirrored pole diagonals, circular
  latitude padding in the post-upsample conv). Old checkpoints load with
  `strict=True`. Use `"spherical"` for new models: geometrically correct and
  about 2x faster, since the convs run as a spherically padded `Conv2d`. Do
  not switch the mode of a trained model.
- `bypass_input_norm=True`: no GroupNorm on the raw input, so the absolute
  input level stays visible (`GN(x + c) == GN(x)`).
- Time embedding periods span 2 pi to 2 pi x 10^4; scale flow-matching times in
  [0, 1] by about 1000.
- `memory_efficient=True` recomputes the neighbour gathers in backward (legacy
  path only, not with the Inductor backend).
- H and W must be divisible by `2 ** (spherical_depth - 1)`; the grid is
  assumed cell-centred.
- `widen_input_channels` / `verify_widening` add zero-initialised input
  channels to a trained model. Use `insert_at` for new data channels.
- `spherical_unet.legacy` holds the old Chebyshev layers (needs scipy).

## Tests

```bash
pip install -e ".[test,legacy]" && OMP_NUM_THREADS=4 pytest -q
```

MIT license.

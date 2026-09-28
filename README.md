# spherical-unet

A U-Net for gridded fields on the sphere. It takes regular latitude-longitude
(equiangular) fields as `[B, C, H, W]` tensors and returns `[B, C_out, H, W]`.
Internally it is a graph U-Net whose convolutions respect the sphere's
topology: longitude wraps around, and crossing a pole lands on the antipodal
meridian.

The interface matches diffusers' `UNet2DModel` (`model(x, t).sample`), so it
can replace that model as the backbone of diffusion, consistency or
flow-matching models. With a constant time it also works as a deterministic
regression network (for example climate downscaling).

- One self-contained file, [`src/spherical_unet/model.py`](src/spherical_unet/model.py),
  that depends only on `torch` and `numpy`. Copy it into a project, or install
  the package.
- Optional inputs: coordinate channels, static fields (orography, masks) that
  bypass normalisation, a learned per-source token, and a full input-norm
  bypass that keeps the absolute level of the inputs.
- Checkpoint utilities: zero-initialised input widening that is exact.

## Install

```bash
pip install git+https://github.com/<user>/spherical-unet.git
# or, from a checkout
pip install -e ".[test]"
```

Requires Python >= 3.9, PyTorch >= 2.0 and NumPy. `scipy` is only needed for the
optional legacy Chebyshev layers (`pip install -e ".[legacy]"`).

### Vendoring

`model.py` imports nothing from the rest of the package. To use the network
without a dependency, copy that file into your project:

```python
from mypackage.spherical_unet import SphericalUNetWrapper
```

`checkpoint.py` needs only `model.py` next to it (it imports `N_NEIGHBOURS`
through a relative import).

## Quick start

```python
import torch
from spherical_unet import SphericalUNetWrapper

model = SphericalUNetWrapper(
    in_channels=3, out_channels=3,
    image_height=176, image_width=360,          # H, W divisible by 2**(depth-1)
    channel_list=(128, 128, 256, 256), spherical_depth=4,
    time_emb_dim=64,
    topology="spherical",                       # recommended for new models
)

x = torch.randn(2, 3, 176, 360)
t = torch.tensor([10.0, 500.0])                 # diffusion step / noise level per sample
y = model(x, t).sample                          # [2, 3, 176, 360]
```

With static fields and a source token:

```python
model = SphericalUNetWrapper(
    in_channels=3, out_channels=3, image_height=176, image_width=360,
    use_coord_channels=True, coord_mode="lat", latitudes=lats_deg,
    n_static_channels=3,                        # e.g. orography, land fraction, ice mask
    use_source_token=True, n_sources=2,
    topology="spherical",
)
y = model(x, t, static=static, source=torch.tensor([0, 1])).sample
# or cache the static fields once, e.g. for inference:
model.set_static_fields(static[0])              # [n_static, H, W]
y = model(x, t).sample
```

[`examples/minimal_denoiser.py`](examples/minimal_denoiser.py) trains a small
denoiser on synthetic fields on CPU in about a minute.

## Architecture

The structure follows diffusers' `UNet2DModel` with `layers_per_block=2`:

```
input [B, C, H, W] -> graph signal [B, V, C]  (V = H*W, row-major)
  (+ coordinate channels) (+ static channels)

encoder, per level (finest first):
    [2x2 avg pool]  ->  GraphResNetBlock  ->  GraphResNetBlock   --skip-->
mid block (coarsest level):
    GraphResNetBlock  ->  GraphSelfAttention (global)  ->  GraphResNetBlock
decoder, per level:
    bilinear x2  ->  3x3 Conv2d + GN + SiLU  ->  concat skip
    ->  GraphResNetBlock  ->  GraphResNetBlock
output:
    GN -> SiLU -> DirectNeighConv  ->  [B, C_out, H, W]

GraphResNetBlock:
    GN -> SiLU -> DirectNeighConv -> + Linear(SiLU(t_emb)) -> GN -> SiLU -> DirectNeighConv
    + residual (Linear if the width changes)

time: sinusoidal(t) -> Linear -> SiLU -> Linear  (+ source embedding)
```

`DirectNeighConv` gathers 9 neighbours per vertex, `[self, N, NE, E, SE, S, SW,
W, NW]`, and applies `nn.Linear(9 * C_in, C_out)`. That is a 3x3 convolution with
9 anisotropic weights per filter, on the sphere's topology. GroupNorm is used
everywhere, so train and eval mode behave the same.

For a 176 x 360 grid at depth 4 the levels are 176x360, 88x180, 44x90 and
22x45 (990 vertices for the attention). Default widths `(128, 128, 256, 256)`
with `time_emb_dim=64` give about 14 M parameters for 16 input channels.

## Topology: `"legacy"` vs `"spherical"`

| Operation | `topology="legacy"` (default) | `topology="spherical"` |
|---|---|---|
| Pole neighbours N/S | pole row shifted by `W//2` | same |
| Pole diagonals | mirrored (NE -> `j + W//2 - 1`) | geometric (NE -> `j + W//2 + 1`) |
| Post-upsample 3x3 conv padding | circular in lon **and lat** (the two pole rows see each other) | circular in lon, pole-reflected in lat |
| Bilinear upsampling | clamps at every edge | periodic in lon, pole-aware |

Table 1: How the two topology modes treat the grid boundaries. `j` is the
column index and `W` the number of columns.

`"legacy"` reproduces the original implementation **bit for bit**, so weights
trained with it load and run unchanged. `"spherical"` is geometrically
correct: `DirectNeighConv` on its graph equals a 3x3 `Conv2d` of the
spherically padded field, and without coordinate channels the whole network
is exactly equivariant to longitude rolls by multiples of the coarsest cell
(both properties are tested). **Use `"spherical"` for new models.** The two
modes share the same parameters, but the outputs differ everywhere once the
receptive field has mixed in pole and seam values, so do not switch the mode
of a trained model.

Assumptions: the grid is cell-centred (no row lies exactly on a pole), and
latitude may run in either direction. When `W` is odd, which happens only at
coarse levels (360 -> 45 at depth 4), the antipode `j + W//2` is half a cell
off.

## Wrapper options

| Argument | Default | Meaning |
|---|---|---|
| `in_channels`, `out_channels` | - | data channels in and out |
| `image_height`, `image_width` | - | grid size `H` (lat), `W` (lon); divisible by `2**(depth-1)` |
| `channel_list` | `(128, 128, 256, 256)` | width per level, finest first |
| `spherical_depth` | `4` | number of levels |
| `time_emb_dim` | `64` | sinusoidal embedding width (even) |
| `use_coord_channels`, `coord_mode` | `False`, `"latlon"` | append sin/cos lat (`"lat"`, 2 ch) or lat+lon (`"latlon"`, 4 ch) |
| `latitudes`, `longitudes` | `None` | row / column coordinates in degrees for the coordinate channels |
| `n_static_channels` | `0` | static channels that bypass the first GroupNorm |
| `use_source_token`, `n_sources` | `False`, `2` | zero-initialised learned embedding per source id |
| `bypass_input_norm` | `False` | all inputs bypass the first GroupNorm |
| `attn_heads` | `1` | heads of the mid-block attention |
| `topology` | `"legacy"` | see above |

Table 2: Constructor arguments of `SphericalUNetWrapper`. Channel counts are
per grid cell; coordinates are in degrees.

Notes:

- **Time scale.** The sinusoidal embedding has periods from 2 pi to about
  2 pi x 10^4. That suits integer diffusion steps (0 to 1000) or EDM noise
  levels. For a flow-matching time in [0, 1], multiply it by about 1000 first,
  otherwise most embedding dimensions barely change.
- **Default coordinates.** Without `latitudes`, the latitude channels use
  `linspace(90, -90, H)`: north first, with the end points on the poles. This
  matches the original implementation. For a cell-centred or south-first grid,
  pass the real row latitudes.
- **Input normalisation.** GroupNorm is invariant to a per-sample shift,
  `GN(x + c) = GN(x)`. With a normalised first block, the network sees a
  uniform offset of the input (for example a global-mean temperature change)
  only through the residual shortcut. Static channels therefore bypass it,
  and `bypass_input_norm=True` makes all inputs bypass it. The inputs should
  then be on a fixed, roughly unit scale.
- **Source token.** Ids `>= 0` select an embedding. `source=None` or ids `< 0`
  add nothing (a null token for dropout or guidance). Because the embedding
  starts at zero, a freshly built token model equals the token-free one.

## Checkpoints

- The neighbour tables are rebuilt from the grid, so they are not stored in the
  state dict. Checkpoints from the original implementation, which did store
  them, still load with `strict=True`; the extra keys are dropped.
- `widen_input_channels(state, n_old_in, n_new_in, prefix="")` adds new input
  channels, zero-initialised, after all the old ones (for example extra static
  fields). It relocates the neighbour-major `conv1` columns and extends the
  residual shortcut, so the widened model reproduces the old one **exactly**,
  whatever the new channels contain. `verify_widening` checks this. The new
  channels must bypass the first GroupNorm (static channels, or
  `bypass_input_norm=True`).

```python
from spherical_unet import widen_input_channels, verify_widening

new_model.load_state_dict(widen_input_channels(old_model.state_dict(), 5, 7), strict=True)
verify_widening(old_model.double(), new_model.double(), x.double(), t.double(), atol=1e-12)
```

## Legacy Chebyshev layers

`spherical_unet.legacy` holds the DeepSphere-style Chebyshev graph convolutions
and Laplacian builders from earlier versions, for reference. The main model
does not use them. With `K = 3` a Chebyshev filter has 3 isotropic weights,
against 9 anisotropic ones for `DirectNeighConv`.

## Tests

```bash
pip install -e ".[test,legacy]"
pytest -q
```

On shared many-core machines, set `OMP_NUM_THREADS` to a small number; thread
oversubscription can slow the CPU tests by orders of magnitude.

## License

MIT, see [LICENSE](LICENSE).

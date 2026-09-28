"""Minimal example: train the spherical U-Net as a denoiser on synthetic fields.

Runs on CPU in about a minute. The target fields are random smooth
combinations of low-order spherical patterns, so the task only illustrates the
calling convention (``model(x, t, static=...).sample``), not a real use case.

    python examples/minimal_denoiser.py
"""

import math

import torch

from spherical_unet import SphericalUNetWrapper

H, W = 32, 64                                     # 5.625 degree cell-centred grid
LAT = torch.linspace(90 - 90 / H, -90 + 90 / H, H)   # north first, cell centres
LON = torch.arange(W) * 360.0 / W


def synthetic_fields(batch: int) -> torch.Tensor:
    """[batch, 1, H, W] smooth fields: random mix of a few zonal/wave patterns."""
    lat = torch.deg2rad(LAT)[:, None]
    lon = torch.deg2rad(LON)[None, :]
    basis = torch.stack([
        torch.sin(lat).expand(H, W),
        (torch.cos(lat) * torch.cos(lon)),
        (torch.cos(lat) * torch.sin(2 * lon)),
        (torch.cos(lat) ** 2 * torch.cos(3 * lon)),
    ])                                            # [4, H, W]
    coef = torch.randn(batch, 4, 1, 1)
    return (coef * basis).sum(1, keepdim=True)


def main():
    torch.manual_seed(0)
    # One static channel: a fake "orography" that is fixed for all samples.
    orography = torch.relu(torch.cos(torch.deg2rad(LAT))[:, None]
                           * torch.cos(torch.deg2rad(LON))[None, :]).expand(1, H, W)

    model = SphericalUNetWrapper(
        in_channels=1, out_channels=1, image_height=H, image_width=W,
        channel_list=(16, 32, 64), spherical_depth=3, time_emb_dim=32,
        use_coord_channels=True, coord_mode="lat", latitudes=LAT.tolist(),
        n_static_channels=1, topology="spherical",
    )
    model.set_static_fields(orography)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(201):
        x0 = synthetic_fields(8)
        sigma = torch.exp(torch.empty(8).uniform_(math.log(0.05), math.log(2.0)))
        noisy = x0 + sigma[:, None, None, None] * torch.randn_like(x0)
        # Condition on log-noise scaled to the time embedding's range.
        pred = model(noisy, 250 * torch.log(sigma)).sample
        loss = ((pred - x0) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  denoising MSE {loss.item():.4f}")


if __name__ == "__main__":
    main()

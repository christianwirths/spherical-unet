"""Compatibility with the original implementation.

``tests/data/legacy_golden.pt`` was generated once with the original code:
float64 state dicts in the original checkpoint format (including the
neighbour-table buffers it used to store), inputs and outputs, for a 16 x 40
grid (coarsest level 2 x 5, odd width), plus one Chebyshev convolution with
K = 3.

``topology="legacy"`` computes exactly the same operations as the original, so
on one machine and torch build the outputs are bit-identical (verified when
the fixture was made). Across BLAS builds and CPU instruction sets float64
rounding differs by ~1e-14, so the stored outputs are compared with an
absolute tolerance of 1e-12; any wiring or topology change gives errors of
order 1e-2 or larger.
"""

from pathlib import Path

import pytest
import torch

from spherical_unet import SphericalUNetCore, SphericalUNetWrapper
from spherical_unet.legacy import cheb_conv

GOLDEN = torch.load(Path(__file__).parent / "data" / "legacy_golden.pt", weights_only=True)
TOL = dict(rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", sorted(GOLDEN["cases"]))
@pytest.mark.parametrize("memory_efficient", [False, True])
def test_legacy_outputs_match_golden(name, memory_efficient):
    case = GOLDEN["cases"][name]
    model = SphericalUNetWrapper(image_height=GOLDEN["H"], image_width=GOLDEN["W"],
                                 memory_efficient=memory_efficient,
                                 **case["config"]).double().eval()
    model.load_state_dict(case["state_dict"], strict=True)
    # Grad enabled, so memory_efficient actually runs the recompute path.
    out = model(case["images"], case["times"], **case["extra"]).sample
    torch.testing.assert_close(out.detach(), case["output"], **TOL)
    out.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_legacy_core_subdict_loads_strictly():
    case = GOLDEN["cases"]["static_source"]
    model = SphericalUNetWrapper(image_height=GOLDEN["H"], image_width=GOLDEN["W"],
                                 **case["config"])
    sub = {k[len("unet."):]: v for k, v in case["state_dict"].items() if k.startswith("unet.")}
    assert any(k.endswith("neigh_orders") for k in sub)
    assert isinstance(model.unet, SphericalUNetCore)
    model.unet.load_state_dict(sub, strict=True)


def test_legacy_cheb_conv_matches_golden():
    c = GOLDEN["cheb"]
    out = cheb_conv(c["lap"].to_sparse(), c["x"], c["weight"])
    torch.testing.assert_close(out, c["output"], **TOL)

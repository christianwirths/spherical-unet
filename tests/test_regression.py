"""Bit-compatibility with the original implementation.

``tests/data/legacy_golden.pt`` was generated once with the original code:
float64 state dicts in the original checkpoint format (including the
neighbour-table buffers it used to store), inputs and outputs, for a 16 x 40
grid (coarsest level 2 x 5, odd width), plus one Chebyshev convolution with
K = 3. ``topology="legacy"`` must reproduce the outputs bit for bit.
"""

from pathlib import Path

import pytest
import torch

from spherical_unet import SphericalUNetCore, SphericalUNetWrapper
from spherical_unet.legacy import cheb_conv

GOLDEN = torch.load(Path(__file__).parent / "data" / "legacy_golden.pt", weights_only=False)


@pytest.mark.parametrize("name", sorted(GOLDEN["cases"]))
@pytest.mark.parametrize("memory_efficient", [False, True])
def test_legacy_outputs_are_bit_identical(name, memory_efficient):
    case = GOLDEN["cases"][name]
    model = SphericalUNetWrapper(image_height=GOLDEN["H"], image_width=GOLDEN["W"],
                                 memory_efficient=memory_efficient,
                                 **case["config"]).double().eval()
    model.load_state_dict(case["state_dict"], strict=True)
    with torch.no_grad():
        out = model(case["images"], case["times"], **case["extra"]).sample
    assert torch.equal(out, case["output"])


def test_legacy_core_subdict_loads_strictly():
    case = GOLDEN["cases"]["static_source"]
    model = SphericalUNetWrapper(image_height=GOLDEN["H"], image_width=GOLDEN["W"],
                                 **case["config"])
    sub = {k[len("unet."):]: v for k, v in case["state_dict"].items() if k.startswith("unet.")}
    assert any(k.endswith("neigh_orders") for k in sub)
    assert isinstance(model.unet, SphericalUNetCore)
    model.unet.load_state_dict(sub, strict=True)


def test_legacy_cheb_conv_is_bit_identical():
    c = GOLDEN["cheb"]
    out = cheb_conv(c["lap"].to_sparse(), c["x"], c["weight"])
    assert torch.equal(out, c["output"])

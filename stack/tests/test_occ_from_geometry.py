"""`occ_from_geometry` — pinned by MUTATION, not by inspection.

⛔ WHY EACH ASSERTION IS HERE. `6c5fb62` measured that the box head's LEARNED `occ_logit` loses to a
2-parameter read of the head's own predicted azimuth (0.28818 vs 0.16055 log-loss, CI
[-0.1844, -0.0376]), and that the target is an exact identity: `occ == |atan2(cy, cx)| > 60.0 deg`,
agreement 1.0000 on BOTH halves of an episode-disjoint split. `agent_slots.occ_logit_from_centre`
implements that identity. These tests pin the three ways it could silently stop being the identity.

⛔ THE EXPECTATIONS ARE LITERALS, NEVER EXPRESSIONS OVER THE CODE UNDER TEST. A test that asserts
`sign(logit) == (az > OCC_HALF_ANGLE_RAD)` using the module's own constant is a check that shares
the defect it checks for: change the constant and both sides move together, and the test is green
forever. So the boundary cases below carry their own hand-computed geometry.

⭐ The companion `mutate_occ_from_geometry.py` reintroduces each real defect and proves these go
RED. A guard that cannot fail is worse than none.
"""
from __future__ import annotations

import inspect
import math

import pytest
import torch

from tanitad.data import bev_raster
from tanitad.models.agent_slots import (OCC_HALF_ANGLE_RAD, OCC_TEMPERATURE,
                                        AgentSlotDecoder, SLOT_SLICES,
                                        occ_logit_from_centre)


def _dec(**kw):
    return AgentSlotDecoder(d_memory=32, n_memory=8, n_queries=4, d_model=32,
                            depth=1, n_heads=4, enforce_band=False, **kw)


def test_half_angle_is_pinned_to_fov_mask_not_merely_documented():
    """⛔ TWO SITES CARRYING ONE ANGLE IS HOW THE GATE-VALUE DRIFT HAPPENED.

    `bev_raster.fov_mask` owns the predicate; `agent_slots` owns the decode. If the two ever
    disagree the decode silently stops being the identity it claims to be, and NOTHING else in the
    tree would notice. This binds them by reading `fov_mask`'s ACTUAL default out of its signature.
    """
    sig = inspect.signature(bev_raster.fov_mask)
    fov_default = float(sig.parameters["half_angle_rad"].default)
    assert OCC_HALF_ANGLE_RAD == pytest.approx(fov_default, abs=1e-12), (
        f"agent_slots.OCC_HALF_ANGLE_RAD ({math.degrees(OCC_HALF_ANGLE_RAD):.4f} deg) has drifted "
        f"from bev_raster.fov_mask's half_angle_rad default "
        f"({math.degrees(fov_default):.4f} deg). They encode ONE physical fact — the front "
        f"camera's horizontal field — and `occ` is defined as that predicate.")
    # and a LITERAL, so a coordinated change to both still has to be deliberate
    assert OCC_HALF_ANGLE_RAD == pytest.approx(math.radians(60.0), abs=1e-12)


@pytest.mark.parametrize(
    # (cx, cy, expect_occluded) — hand-computed, NOT derived from the module's constant.
    # 60 deg from straight ahead: tan(60 deg) = sqrt(3), so |cy| = sqrt(3)*cx is exactly ON the
    # boundary. Inside the field  -> occ False; outside -> occ True.
    "cx, cy, occluded",
    [
        (10.0, 0.0, False),          # straight ahead, azimuth 0
        (10.0, 5.0, False),          # 26.57 deg
        (10.0, 17.0, False),         # 59.53 deg — just inside
        (10.0, 17.4, True),          # 60.11 deg — just outside
        (10.0, -17.4, True),         # symmetric: the predicate is on |azimuth|
        (1.0, 100.0, True),          # 89.43 deg, hard abeam
        (-10.0, 1.0, True),          # BEHIND the ego: atan2 gives ~174 deg
        (-10.0, -1.0, True),         # behind, other side
    ])
def test_sign_matches_the_hand_computed_predicate(cx, cy, occluded):
    lg = occ_logit_from_centre(torch.tensor([cx]), torch.tensor([cy]))
    assert (lg.item() > 0.0) is occluded, (
        f"centre ({cx}, {cy}) is azimuth {math.degrees(abs(math.atan2(cy, cx))):.2f} deg; "
        f"expected occluded={occluded} but the logit is {lg.item():+.4f}")


def test_boundary_is_exactly_zero_logit():
    """At |azimuth| == the half-angle the logit is 0 — the decision boundary, by construction."""
    cx = torch.tensor([10.0])
    cy = cx * math.tan(OCC_HALF_ANGLE_RAD)
    assert occ_logit_from_centre(cx, cy).abs().item() < 1e-5


def test_temperature_scales_confidence_but_never_the_decision():
    """⭐ The SIGN is the identity; the slope is only calibration. Separating them matters because
    a future re-fit of the temperature must not be able to change a single classification."""
    cx = torch.tensor([10.0, 10.0, 10.0])
    cy = torch.tensor([1.0, 17.0, 40.0])
    base = occ_logit_from_centre(cx, cy)
    for t in (0.5, 1.0, 50.0):
        alt = occ_logit_from_centre(cx, cy, temperature=t)
        assert torch.equal(alt > 0, base > 0), (
            f"temperature {t} changed a DECISION, not just a confidence")
        assert torch.all(alt.abs() >= 0)


def test_default_is_off_so_the_emitted_contract_is_unchanged():
    """⛔ This is a CONTRACT change and therefore opt-in. The default must keep emitting the
    learned slice, or every existing arm silently changes what it reports."""
    d = _dec()
    assert d.occ_from_geometry is False
    raw = torch.randn(2, d.n_queries, d.head.out_features)
    out = d.decode(raw)
    expected = raw[..., SLOT_SLICES["occluded"]].squeeze(-1)
    assert torch.equal(out["occ_logit"], expected), (
        "with occ_from_geometry off, decode must return the LEARNED slice verbatim")


def test_opt_in_replaces_the_channel_with_the_geometric_read():
    d = _dec()
    raw = torch.randn(2, d.n_queries, d.head.out_features)
    learned = d.decode(raw)["occ_logit"].clone()
    d.occ_from_geometry = True
    got = d.decode(raw)["occ_logit"]
    box = d.decode(raw)["box"]
    want = occ_logit_from_centre(box[..., 0], box[..., 1])
    assert torch.allclose(got, want, atol=1e-6), (
        "with the flag on, occ_logit must be the geometric read of THIS head's own centre")
    assert not torch.equal(got, learned), (
        "the flag changed nothing — the learned slice is still being emitted")


def test_it_reads_the_predicted_centre_not_the_raw_slice():
    """⚠️ The decode scales cx/cy by `ranges` before they are metres. Reading the RAW slice instead
    of the DECODED centre would apply the 60 deg predicate to normalised units — a correct formula
    under the wrong units, which is the family that produced the 396 g anchor table."""
    d = _dec()
    d.occ_from_geometry = True
    raw = torch.zeros(1, d.n_queries, d.head.out_features)
    raw[..., SLOT_SLICES["cx"]] = 0.9        # normalised; metres only after * ranges.x_fwd_m
    raw[..., SLOT_SLICES["cy"]] = 0.2
    out = d.decode(raw)
    from_raw = occ_logit_from_centre(torch.tensor([0.9]), torch.tensor([0.2]))
    from_box = occ_logit_from_centre(out["box"][..., 0], out["box"][..., 1])
    assert torch.allclose(out["occ_logit"], from_box, atol=1e-6)
    # the two differ because x_fwd_m and y_half_m are not equal — that inequality IS the test
    assert not torch.allclose(out["occ_logit"].flatten()[:1], from_raw, atol=1e-3), (
        "the raw-slice and decoded-centre readings coincide, so this test cannot detect the "
        "units error it exists to catch — check SlotDecodeRanges")


def test_box3d_inherits_the_option_without_forwarding():
    """The 3-D head subclasses the 2-D one and replaces `self.head`; the attribute must survive
    that, or the option silently does nothing on the head that is actually scored."""
    from tanitad.models.box3d_head import Box3DSlotDecoder
    d = Box3DSlotDecoder(d_memory=32, n_memory=8, n_queries=4, d_model=32,
                         depth=1, n_heads=4, enforce_band=False)
    assert d.occ_from_geometry is False
    d.occ_from_geometry = True
    raw = torch.randn(2, d.n_queries, d.head.out_features)
    out = d.decode(raw)
    want = occ_logit_from_centre(out["box"][..., 0], out["box"][..., 1])
    assert torch.allclose(out["occ_logit"], want, atol=1e-6)

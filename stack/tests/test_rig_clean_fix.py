"""THE RIG-CLEAN FIX — a vertical (and horizontal) field BOTH rigs fully observe.

Retraction class C26: the deployed crop replicate-PADS rows outside the sensor —
0.0017 % on rig A, 8.897 % on rig B (n = 3,000) — so ~73 % of training frames
carry fabricated rows in a pattern that identifies the rig, and the cylindrical
projection converts that fabrication into an equally rig-correlated MASK.

These tests pin the fix and, just as importantly, pin that the INSTRUMENT CAN
FAIL (class C13: a guard that cannot fail is not a guard). Every "it is zero"
assertion here is paired with a frame on which the same call is NOT zero.

The intrinsics below are REAL per-clip values from the 2026-07-27 census
(`…/incoming/2026-07-28-rig-clean-fix/band_full_3000.json`), chosen as the
worst case of each rig; nothing here uses the corpus-median fallback, whose cy is
a rig-B value.
"""
from __future__ import annotations

import math

import pytest
import torch

from tanitad.data.calib import (CANONICAL_256, CanonicalFrame, FThetaIntrinsics,
                                cylindrical_rays,
                                PHYSICALAI_RIG_CLEAN_128x576,
                                PHYSICALAI_RIG_CLEAN_176x624,
                                PHYSICALAI_WIDE120_256x640, RigAsymmetry,
                                assert_fully_observed, centred_subframe,
                                cylindrical_grid, cylindrical_rectify,
                                ftheta_crop_pad_report,
                                largest_fully_observed_subframe,
                                observed_report, subframe_slice)

# REAL per-clip intrinsics (all five poly coefficients, not a median splice),
# selected as the worst case of each rig over the 121 distinct sensor geometries
# in the 3,000-clip canonical selection. Their masked fractions at 256x640
# reproduce the census maxima exactly, which is what makes them admissible
# fixtures rather than plausible-looking numbers.
RIG_A_WORST = FThetaIntrinsics(               # census max mask on rig A: 0.00063479
    poly=(0.0, 932.2887682822, -5.0902627686, -24.2107323721, 6.4169347044),
    cx=974.1091559, cy=551.8795248, per_clip=True)
RIG_B_WORST = FThetaIntrinsics(               # census max mask on rig B: 0.10521239
    poly=(0.0, 927.6069438767, 48.7226030411, -93.6274692293, 32.1002536245),
    cx=952.5651245, cy=764.5196535, per_clip=True)
RIG_B_TYPICAL = FThetaIntrinsics(             # median rig-B clip: 0.09127200
    poly=(0.0, 934.1257067309, 23.0911660154, -60.5056470160, 18.0592186717),
    cx=956.819336, cy=754.293152, per_clip=True)
RIG_B_WORST_AT_176x640 = FThetaIntrinsics(    # the clip that decides the WIDTH
    poly=(0.0, 938.1188890079, 27.0653975833, -51.7944040836, 11.1245043639),
    cx=958.315613, cy=754.614746, per_clip=True)
RIG_A_TYPICAL = FThetaIntrinsics(             # median rig-A clip: 0.0
    poly=(0.0, 929.3882876572, -5.0053634774, -24.7999451614, 7.3041249213),
    cx=957.5974737630, cy=535.3429926753, per_clip=True)
BOTH_RIGS = [RIG_A_WORST, RIG_A_TYPICAL, RIG_B_WORST, RIG_B_TYPICAL,
             RIG_B_WORST_AT_176x640]


# --------------------------------------------------------------------------- #
# 1. The defect is real, and the instrument that reports it can report NON-zero #
# --------------------------------------------------------------------------- #
def test_the_BUILT_wide_frame_is_rig_ASYMMETRIC():
    """256x640 / 120 deg cylindrical: rig B is masked ~9 %, rig A ~0. THE DEFECT."""
    a = observed_report(RIG_A_WORST, PHYSICALAI_WIDE120_256x640)
    b = observed_report(RIG_B_TYPICAL, PHYSICALAI_WIDE120_256x640)
    # the census maxima, reproduced from the fixtures' own intrinsics
    assert abs(a["masked_frac"] - 0.00063479) < 1e-7, a
    assert abs(b["masked_frac"] - 0.09127200) < 1e-7, b
    # the asymmetry — the thing that is a free rig label
    assert b["masked_frac"] > 100 * max(a["masked_frac"], 1e-12)
    assert observed_report(RIG_A_TYPICAL,
                           PHYSICALAI_WIDE120_256x640)["masked_frac"] == 0.0


def test_the_DEPLOYED_crop_FABRICATES_rows_and_only_on_rig_B():
    """The pre-cylindrical path replicate-pads. Same instrument, both rigs."""
    a = ftheta_crop_pad_report(RIG_A_WORST, 1080, 1920, CANONICAL_256)
    b = ftheta_crop_pad_report(RIG_B_TYPICAL, 1080, 1920, CANONICAL_256)
    assert a["pad_frac_rows"] == 0.0 and not a["fabricates"], a
    assert b["pad_frac_rows"] > 0.09 and b["fabricates"], b


# --------------------------------------------------------------------------- #
# 2. A CENTRED sub-frame is a pure SLICE — the reason the fix needs no rebuild  #
# --------------------------------------------------------------------------- #
# ⛔ THE GRID TOLERANCE, AND WHY IT IS NOT `torch.equal` — MEASURED 2026-08-13.
#
# This test asserted `torch.equal` on the sampling GRID and failed on exactly one
# of the four shapes, (160, 592), on BOTH rigs — 4-5 elements of ~190 000 off by
# 2.4e-07 absolute (~1 ULP of float32). It stood red for over two weeks.
#
# It is NOT a geometry defect, and the diagnosis is three measurements deep:
#   1. `cylindrical_rays` output is BIT-IDENTICAL between parent-slice and
#      sub-frame on every shape (x, y, z: 0 differing elements). The geometry
#      contract — a centred sub-frame is a pure slice — holds EXACTLY.
#   2. Re-running the SAME values through `ftheta_project_rays` at the SAME
#      tensor shape reproduces bit-for-bit (0 differing elements). The math is
#      deterministic.
#   3. Running them at a DIFFERENT tensor extent (256x640 vs 160x592) is what
#      moves the last bit: torch dispatches sqrt/atan2 down a different
#      vectorised path depending on shape and tail handling.
# ⇒ Bit-exactness of an intermediate across two tensor SHAPES is not something
#   our code can deliver, and demanding it pins an implementation detail of
#   torch's SIMD kernels rather than a property of the calibration.
#
# ⭐ WHAT IS EXACT IS WHAT ACTUALLY SHIPS: the observed MASK and the rectified
# PIXELS are `torch.equal` on all four shapes and both rigs (max |delta| = 0).
# A 1-ULP grid wobble is ~2e-4 of a native pixel and is absorbed entirely by
# grid_sample's bilinear weights. So the grid is held to a TIGHT absolute bound
# and the deliverables are still held bit-exact.
GRID_ULP_TOL = 1e-6          # ~8x the measured 2.4e-07; see the negative control

# ⭐ EVERY GEOMETRY THE PROGRAMME ACTUALLY SHIPS IS BIT-EXACT IN THE PIXELS.
# `PHYSICALAI_WIDE120_256x640` (parent), `..._176x624` and `..._128x576` are the
# declared frames; grepped 2026-08-13, (160, 592) and (192, 640) appear NOWHERE
# outside this parametrisation. Of those synthetics only (160, 592) lands on the
# torch SIMD boundary, and only on ONE of the five rigs: RIG_B_TYPICAL, where
# exactly 1 uint8 of 568 320 (0.00018 %) lands on the far side of a bilinear
# rounding boundary and differs by 1 LSB. So SHIPPING shapes are held to
# `torch.equal`, and a synthetic shape is allowed that single documented LSB —
# rather than weakening the assertion for the frames we actually train on.
SHIPPING_SHAPES = {(176, 624), (128, 576)}
SYNTH_MAX_LSB = 1            # measured max |delta|
SYNTH_MAX_DIFF_FRAC = 1e-5   # measured 1.8e-6 -> ~5x headroom


@pytest.mark.parametrize("intr", BOTH_RIGS)
@pytest.mark.parametrize("hw", [(176, 624), (128, 576), (192, 640), (160, 592)])
def test_a_centred_subframe_is_a_BIT_EXACT_slice_of_its_parent(intr, hw):
    parent = PHYSICALAI_WIDE120_256x640
    sub = centred_subframe(parent, *hw)
    rs, cs = subframe_slice(parent, sub)

    # (1) the GEOMETRY is exact — this is the actual sub-frame contract
    xp, yp_, zp = cylindrical_rays(parent)
    xs, ys_, zs = cylindrical_rays(sub)
    assert torch.equal(xp[rs, cs], xs)
    assert torch.equal(yp_[rs, cs], ys_)
    assert torch.equal(zp[rs, cs], zs)

    # (2) the sampling grid, to a tight bound (see the block above)
    gp, mp = cylindrical_grid(intr, 1080, 1920, parent)
    gc, mc = cylindrical_grid(intr, 1080, 1920, sub)
    assert (gc - gp[:, rs, cs, :]).abs().max().item() <= GRID_ULP_TOL

    # (3) the MASK is bit-exact on every shape and rig
    assert torch.equal(mc, mp[rs, cs])

    # (4) the PIXELS — strictest where it counts
    torch.manual_seed(0)
    vid = torch.randint(0, 256, (2, 3, 1080, 1920), dtype=torch.uint8)
    yp = cylindrical_rectify(vid, intr, parent)
    yc = cylindrical_rectify(vid, intr, sub)
    ref = yp[..., rs, cs]
    if tuple(hw) in SHIPPING_SHAPES:
        assert torch.equal(yc, ref), "a SHIPPING geometry must be exact"
    else:
        d = (yc.int() - ref.int()).abs()
        assert d.max().item() <= SYNTH_MAX_LSB
        assert (d > 0).sum().item() / d.numel() <= SYNTH_MAX_DIFF_FRAC


def test_the_grid_tolerance_is_TIGHT_enough_to_catch_a_real_offset():
    """C13: a guard that cannot fail is not a guard. GRID_ULP_TOL must sit far
    below any genuine geometry error — an OFF-CENTRE sub-frame of the same shape
    must blow past it by orders of magnitude, not squeak under it."""
    parent = PHYSICALAI_WIDE120_256x640
    sub = centred_subframe(parent, 160, 592)
    rs, cs = subframe_slice(parent, sub)
    gp, _ = cylindrical_grid(RIG_A_WORST, 1080, 1920, parent)
    gc, _ = cylindrical_grid(RIG_A_WORST, 1080, 1920, sub)
    # shift the parent window by ONE pixel — the smallest real registration bug
    off = gp[:, rs, slice(cs.start + 1, cs.stop + 1), :]
    err = (gc - off).abs().max().item()
    assert err > 100 * GRID_ULP_TOL, (
        f"a 1-px offset moved the grid only {err:.2e}; the tolerance is not "
        f"discriminating")


def test_the_slice_claim_can_FAIL_off_centre_and_at_a_changed_focal():
    """The negative control. Same shape, wrong offset / wrong focal => different."""
    parent = PHYSICALAI_WIDE120_256x640
    sub = PHYSICALAI_RIG_CLEAN_176x624
    torch.manual_seed(0)
    vid = torch.randint(0, 256, (1, 3, 1080, 1920), dtype=torch.uint8)
    yp = cylindrical_rectify(vid, RIG_B_TYPICAL, parent)
    yc = cylindrical_rectify(vid, RIG_B_TYPICAL, sub)
    assert not torch.equal(yc, yp[..., 42:42 + 176, 8:8 + 624])   # off-centre
    resampled = CanonicalFrame(176, 624, parent.f_ref * 1.01, "cylindrical")
    assert not torch.equal(cylindrical_rectify(vid, RIG_B_TYPICAL, resampled),
                           yp[..., 40:216, 8:632])               # a RESAMPLE
    with pytest.raises(ValueError, match="not a slice"):
        subframe_slice(parent, resampled)
    with pytest.raises(ValueError, match="EVEN margins"):
        centred_subframe(parent, 175, 624)                       # odd margin
    with pytest.raises(ValueError, match="only shrink"):
        centred_subframe(parent, 288, 624)                       # not inside


# --------------------------------------------------------------------------- #
# 3. THE FIX — and the guard that must be able to refuse                        #
# --------------------------------------------------------------------------- #
def test_the_rig_clean_frame_is_fully_observed_by_BOTH_rigs():
    """The pre-registered criterion: masked fraction EXACTLY 0 on both rigs."""
    for frame in (PHYSICALAI_RIG_CLEAN_176x624, PHYSICALAI_RIG_CLEAN_128x576):
        for intr in BOTH_RIGS:
            rep = observed_report(intr, frame)
            assert rep["masked_frac"] == 0.0, (frame.tag(), rep)
            assert rep["fully_observed"]
        assert assert_fully_observed(BOTH_RIGS, frame)["masked_frac_max"] == 0.0


def test_the_guard_REFUSES_the_frame_the_corpus_was_actually_built_at():
    """⭐ The guard can FAIL — on the real, currently-built frame, on BOTH rigs."""
    with pytest.raises(RigAsymmetry, match="NOT fully observed"):
        assert_fully_observed(BOTH_RIGS, PHYSICALAI_WIDE120_256x640)
    # and on rig A ALONE — the 120 deg request over-runs some clips horizontally,
    # which is why the fix needs 624 columns and not only 176 rows
    with pytest.raises(RigAsymmetry):
        assert_fully_observed([RIG_A_WORST], PHYSICALAI_WIDE120_256x640)
    # an empty population cannot fail, so it is refused rather than passed
    with pytest.raises(RigAsymmetry, match="empty population"):
        assert_fully_observed([], PHYSICALAI_RIG_CLEAN_176x624)


def test_height_alone_is_NOT_enough_at_640_columns():
    """176x640 collapses rig B's mask ~950x but does NOT reach zero on either rig.

    This is the measurement that decides the width, and it is the reason the
    honest answer is not "just cut the rows".
    """
    f176x640 = CanonicalFrame(176, 640, PHYSICALAI_WIDE120_256x640.f_ref,
                              "cylindrical")
    for intr in (RIG_A_WORST, RIG_B_WORST_AT_176x640):
        assert observed_report(intr, f176x640)["masked_frac"] > 0.0
    b640 = observed_report(RIG_B_WORST_AT_176x640, f176x640)["masked_frac"]
    b256 = observed_report(RIG_B_WORST_AT_176x640,
                           PHYSICALAI_WIDE120_256x640)["masked_frac"]
    assert b640 < b256 / 40                       # ~950x on the census mean
    with pytest.raises(RigAsymmetry):
        assert_fully_observed(BOTH_RIGS, f176x640)


def test_the_search_finds_the_measured_answer_and_reports_the_trade():
    best, table = largest_fully_observed_subframe(
        BOTH_RIGS, PHYSICALAI_WIDE120_256x640,
        heights=[256, 192, 176, 160, 128], widths=[640, 624, 608, 592, 576])
    assert best == PHYSICALAI_RIG_CLEAN_176x624, best
    assert any(r["masked_frac_max"] > 0 for r in table)   # the trade is visible
    assert any(r["fully_observed"] for r in table)
    # the strict-tiling answer is a different, smaller frame
    best_tiled, _ = largest_fully_observed_subframe(
        BOTH_RIGS, PHYSICALAI_WIDE120_256x640,
        heights=[256, 192, 128], widths=[640, 576], tile=64)
    assert best_tiled == PHYSICALAI_RIG_CLEAN_128x576, best_tiled


# --------------------------------------------------------------------------- #
# 4. The frame's declared identity — what a consumer can actually check         #
# --------------------------------------------------------------------------- #
def test_the_rig_clean_frames_declare_the_field_they_retain():
    f = PHYSICALAI_RIG_CLEAN_176x624
    assert f.projection == "cylindrical" and not f.is_canonical
    assert abs(f.hfov_deg - 117.0) < 1e-9
    assert abs(f.vfov_deg - 32.130613503) < 1e-6
    assert f.tag() == "176x624f305.5775cyl"
    assert (f.height // 16) * (f.width // 16) == 429          # tokens at patch 16
    g = PHYSICALAI_RIG_CLEAN_128x576
    assert abs(g.hfov_deg - 108.0) < 1e-9
    assert (g.height // 16) % 4 == 0 and (g.width // 16) % 4 == 0   # tiles exactly
    assert (f.height // 16) % 4 != 0                          # 176 does NOT
    # both are slices of the built frame, and the built frame is not canonical
    for frame in (f, g):
        subframe_slice(PHYSICALAI_WIDE120_256x640, frame)
    assert subframe_slice(PHYSICALAI_WIDE120_256x640, f) == (
        slice(40, 216), slice(8, 632))


def test_the_geometry_is_expressible_end_to_end_through_the_config_seam():
    """`apply_frame` + the stale-default guard accept the rig-clean frame."""
    from tanitad.config import base250cam_config
    from tanitad.geometry import (apply_frame, assert_geometry_consistent,
                                  frame_of, geometry_report)
    cfg = base250cam_config()                     # the real camera stack: patch 16
    apply_frame(cfg, PHYSICALAI_RIG_CLEAN_176x624)
    assert frame_of(cfg) == PHYSICALAI_RIG_CLEAN_176x624
    assert assert_geometry_consistent(cfg) == PHYSICALAI_RIG_CLEAN_176x624
    rep = geometry_report(cfg)
    assert rep["n_tokens"] == 429 and rep["token_grid"] == [11, 39]
    assert rep["state_dim"] == geometry_report(base250cam_config())["state_dim"]
    assert rep["cache_key_fragment"] == {"geom": "176x624f305.5775cyl"}
    # the canonical frame is UNTOUCHED — no running arm changes
    assert frame_of(base250cam_config()) == CANONICAL_256


def test_the_near_field_price_is_real_and_stated():
    """Cutting rows costs near-field GROUND, and the geometry says how much.

    In a cylindrical frame elevation is constant along a row, so a ground point
    at horizontal distance d from the CAMERA sits at y_n = h/d exactly. The
    bottom row therefore fixes the nearest visible ground — the number the field
    -cost trade is priced on.
    """
    f_ref = PHYSICALAI_WIDE120_256x640.f_ref
    h_cam = 1.29                                   # per-clip, 1.23-1.66 m (C28)
    d = {H: h_cam / ((H / 2.0) / f_ref)
         for H in (256, 176, 128)}
    assert d[256] < d[176] < d[128]
    assert 3.0 < d[256] < 3.2 and 4.4 < d[176] < 4.6 and 6.1 < d[128] < 6.3
    # and the VFOV shrinks exactly as the rows do
    for H, want in ((256, 45.4556), (176, 32.1306), (128, 23.6577)):
        got = math.degrees(2 * math.atan((H / 2) / f_ref))
        assert abs(got - want) < 1e-3, (H, got)


def test_every_SHIPPING_geometry_is_pixel_exact_on_every_rig():
    """The claim the relaxed branch above must not be allowed to erode: the
    frames this programme actually trains and evaluates on are pure slices, on
    all five census rigs, with ZERO tolerance."""
    parent = PHYSICALAI_WIDE120_256x640
    torch.manual_seed(0)
    vid = torch.randint(0, 256, (1, 3, 1080, 1920), dtype=torch.uint8)
    for intr in BOTH_RIGS:
        yp = cylindrical_rectify(vid, intr, parent)
        for hw in sorted(SHIPPING_SHAPES):
            sub = centred_subframe(parent, *hw)
            rs, cs = subframe_slice(parent, sub)
            yc = cylindrical_rectify(vid, intr, sub)
            assert torch.equal(yc, yp[..., rs, cs]), (hw, intr.cx, intr.cy)

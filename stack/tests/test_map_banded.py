"""refcv9 C1 -- the banded map head's TARGETS and LOSSES (``tanitad/models/map_banded.py``). CPU. Every expectation is a LITERAL or an
analytic identity authored here; every claim carries a DELIBERATE-REGRESSION arm that must go RED (monkeypatching the one function
that carries the claim):

  1. geometry: 100 m x +-30 m -> 100x600 / 120x240 / 120x120 = 103,200 cells (vs 600,000); refusals;
  2. E0e analytic controls: a straight kerb gives an SDF LINEAR in lateral distance with slope 1 (at 0.10 and 0.25 m cells); a 1-cell
     lane line dilates to EXACTLY 0.5 m (5 cells at 0.10 m); the lane offset of a cell is the exact vector to the line;
  3. GT-as-prediction: perfect per-channel IoU in every band where the definitions agree; a one-cell shift degrades the thin
     channels monotonically;
  4. REAL data (135 banked SAM3 clips): non-zero positives per channel per band, fractions PRINTED;
  5. losses: the perfect-prediction value, the unobs invariance (loss AND gradient), the band weights, the max-entropy term;
  RED arms: no unobs masking, undilated thin targets, drv from the 8-way argmax, band weights 1/1/1.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.models import map_banded as MB  # noqa: E402

REAL = sorted(glob.glob("E:/Projects/TanitAD-artifacts/sam3-maps-eval/*.sam3mapgt.npz"))
#: the CI extent (the only extent the banked /2 rasters carry): 60 m x +-16 m -> 600 x 320 at 0.10 m
X60, Y16 = 60.0, 16.0
CODES = {"nocls": 0, "drivable": 1, "lane": 2, "xwalk": 3, "arrow": 4, "edge": 5, "hatch": 6, "sidewalk": 7}


def _road(b=1, kerb_col=160):
    """drivable on the left (cols >= kerb_col), sidewalk on the right: a straight axis-aligned kerb between cols kerb_col-1 / kerb_col."""
    c = torch.full((b, 600, 320), 7, dtype=torch.uint8)
    c[:, :, kerb_col:] = 1
    return c


# --------------------------------------------------------------------------- #
# 1. geometry                                                                  #
# --------------------------------------------------------------------------- #
def test_geometry_refcv8_extent_is_103200_cells_against_600000():
    g = MB.band_geometry(100.0, 30.0)
    assert [(b.name, b.rows, b.cols, b.cell) for b in g] == [("near", 100, 600, 0.10), ("mid", 120, 240, 0.25), ("far", 120, 120, 0.50)]
    assert sum(b.n_cells for b in g) == 103_200 and 1000 * 600 == 600_000
    assert [b.fine_rows for b in g] == [(0, 100), (100, 400), (400, 1000)]
    assert MB.BAND_WEIGHTS == (1.0, 1.0, 0.5)
    assert [(b.rows, b.cols) for b in MB.band_geometry(X60, Y16)] == [(100, 320), (120, 128), (40, 64)]


def test_geometry_refusals():
    with pytest.raises(ValueError, match="no far band"):
        MB.band_geometry(40.0, 30.0)
    with pytest.raises(ValueError, match="does not tile"):
        MB.band_geometry(100.0, 30.1)
    with pytest.raises(ValueError, match="does not tile"):
        MB.band_geometry(100.2, 30.0)


def test_the_class_codes_are_read_from_the_label_names():
    assert MB.DRV_CODES == (1, 2, 3, 4, 6)
    assert (MB.C_LANE, MB.C_XWALK, MB.C_ARROW, MB.C_EDGE, MB.C_HATCH, MB.C_SIDEWALK) == (2, 3, 4, 5, 6, 7)
    assert MB.CHANNELS == ("drv", "drv_sdf", "sdw", "lane", "lane_ox", "lane_oy", "edge_h", "xwalk", "arrow", "hatch")


# --------------------------------------------------------------------------- #
# 2. E0e analytic controls                                                     #
# --------------------------------------------------------------------------- #
def test_E0e_a_straight_kerb_gives_an_SDF_linear_in_lateral_distance_with_slope_1():
    tg = MB.build_banded_targets(_road(), x_max_m=X60, y_half_m=Y16)
    # near band, 0.10 m: inside cell j >= 160 is at distance (j - 160 + 0.5) * 0.1 from the boundary, outside (160 - j - 0.5) * 0.1
    near = tg[0]["sdf"][0, 50]                                   # any row
    ok = tg[0]["sdf_sup"][0, 50]
    j = torch.arange(320)
    want = torch.where(j >= 160, (j - 160 + 0.5) * 0.1, -(160 - j - 0.5) * 0.1)
    inside_2m = (want.abs() < 2.0 - 1e-6)
    assert torch.equal(ok, inside_2m)                           # supervised exactly within 2 m of the kerb
    assert torch.allclose(near[ok], want[ok], atol=1e-5)
    d = near[ok][1:] - near[ok][:-1]
    assert torch.allclose(d, torch.full_like(d, 0.1), atol=1e-5)  # slope 1: one 10 cm cell = 0.1 m
    # mid band, 0.25 m: fine col 160 = mid col 64 (128 cols)
    mid = tg[1]["sdf"][0, 20]
    jm = torch.arange(128)
    want_m = torch.where(jm >= 64, (jm - 64 + 0.5) * 0.25, -(64 - jm - 0.5) * 0.25)
    okm = tg[1]["sdf_sup"][0, 20]
    assert torch.allclose(mid[okm], want_m[okm], atol=1e-5) and int(okm.sum()) == 16   # 8 cells each side within 2 m
    # everywhere the clip holds
    for t in tg:
        assert float(t["sdf"].abs().max()) <= MB.SDF_CLIP_M + 1e-6


def test_E0e_a_one_cell_lane_line_dilates_to_exactly_half_a_metre():
    c = _road(kerb_col=0)                                       # all drivable
    c[:, :, 100] = 2                                            # a 1-cell (0.1 m) lane line
    tg = MB.build_banded_targets(c, x_max_m=X60, y_half_m=Y16)
    row = tg[0]["lane"][0, 30]
    assert [int(v) for v in row.nonzero().flatten()] == [98, 99, 100, 101, 102]          # 5 cells = 0.5 m
    assert int(row.sum()) * 0.1 == pytest.approx(0.5)
    # the coarser bands: at least 0.5 m wide (>= 2 cells at 0.25 m, >= 1 at 0.50 m) and still contiguous
    assert int(tg[1]["lane"][0, 5].sum()) in (2, 3) and int(tg[2]["lane"][0, 5].sum()) in (1, 2)
    # and the line itself is drivable surface (markings lie ON the road): drv is 1 on the line cell
    assert float(tg[0]["drv"][0, 30, 100]) == 1.0 and float(tg[0]["drv"][0, 30].min()) == 1.0


def test_E0e_the_lane_offset_is_the_exact_vector_to_the_line_and_carries_the_sub_cell_position():
    c = _road(kerb_col=0)
    c[:, :, 100] = 2
    tg = MB.build_banded_targets(c, x_max_m=X60, y_half_m=Y16)
    t = tg[0]
    # near band: cell col 95 centre y = -16 + 95.5 * 0.1 = -6.45; the line cell centre is -5.95: oy = +0.5, ox = 0
    assert float(t["off"][0, 1, 30, 95]) == pytest.approx(0.5, abs=1e-5) and float(t["off"][0, 0, 30, 95]) == pytest.approx(0.0, abs=1e-5)
    assert float(t["off"][0, 1, 30, 106]) == pytest.approx(-0.6, abs=1e-5)
    # supervised iff the line is within 2 m: cols 80..120 (|dy| <= 2.0), nowhere else
    sup = t["off_sup"][0, 30].nonzero().flatten()
    assert int(sup.min()) == 100 - 20 and int(sup.max()) == 100 + 20
    # mid band (0.25 m): fine col 100 lies in mid col 40 (centre y = -16 + 40.5 * 0.25 = -5.875); the line POSITION is -5.95, so the
    # cell holding it reads oy = -5.95 - (-5.875) = -0.075 -- the sub-cell position a heatmap cell cannot carry
    assert float(tg[1]["off"][0, 1, 5, 40]) == pytest.approx(-0.075, abs=1e-5)


# --------------------------------------------------------------------------- #
# 3. GT as prediction                                                          #
# --------------------------------------------------------------------------- #
def _real_codes(n_clips=6, frames=(10, 60, 120, 180)):
    if len(REAL) < 6:
        pytest.skip("the banked SAM3 clips (E:/Projects/TanitAD-artifacts/sam3-maps-eval) are not on this box")
    out = []
    for f in REAL[::22][:n_clips]:
        fc = np.load(f)["fine_codes"]
        out += [fc[i] for i in frames]
    return torch.from_numpy(np.stack(out))


def _iou(pred, tgt, seen):
    p, t = (pred > 0.5) & seen, (tgt > 0.5) & seen
    u = int((p | t).sum())
    return float((p & t).sum()) / u if u else None


def test_GT_as_prediction_scores_perfect_IoU_in_every_band_and_a_shift_degrades_the_thin_channels_monotonically():
    codes = _real_codes()
    tg = MB.build_banded_targets(codes, x_max_m=X60, y_half_m=Y16)
    for b, t in enumerate(tg):
        for ch in MB.PROB_CHANNELS:
            assert _iou(t[ch], t[ch], t["seen"]) in (1.0, None), (b, ch)
    # where the definitions AGREE the near band IS the raster: drv = the code set, sdw / xwalk / arrow / hatch = their code
    c = codes[:, :100, :].long()
    t0 = tg[0]
    seen = c != 255
    assert torch.equal(t0["drv"] > 0.5, sum((c == k) for k in (1, 2, 3, 4, 6)).bool())
    for ch, k in (("sdw", 7), ("xwalk", 3), ("arrow", 4), ("hatch", 6)):
        assert torch.equal(t0[ch] > 0.5, c == k), ch
    assert torch.equal(t0["seen"], seen)
    # a one-cell column shift of a perfect lane / edge prediction: IoU strictly falls with the shift (1, 2, 3, 4 cells)
    for ch in ("lane", "edge_h"):
        ious = [_iou(torch.roll(t0[ch], s, dims=2), t0[ch], t0["seen"]) for s in (0, 1, 2, 3, 4)]
        assert ious[0] == 1.0 and all(a > b for a, b in zip(ious, ious[1:])), (ch, ious)


# --------------------------------------------------------------------------- #
# 4. REAL data content                                                         #
# --------------------------------------------------------------------------- #
def test_REAL_data_every_channel_has_positives_in_every_band_and_the_fractions_are_printed(capsys):
    codes = _real_codes()
    tg = MB.build_banded_targets(codes, x_max_m=X60, y_half_m=Y16)
    fr = MB.target_fractions(tg)
    table = MB.format_fractions(fr)
    with capsys.disabled():
        print("\n[map-banded] REAL-data target fractions (%d frames, 6 banked SAM3 clips, 60 m x +-16 m):\n%s" % (codes.shape[0], table))
    MB.assert_target_content(fr)                                 # raises with the table if any channel x band is empty
    with pytest.raises(AssertionError, match="ZERO positives"):  # the assertion has teeth: an all-sidewalk raster has no arrows
        MB.assert_target_content(MB.target_fractions(MB.build_banded_targets(_road(kerb_col=0), x_max_m=X60, y_half_m=Y16)))


# --------------------------------------------------------------------------- #
# 5. losses                                                                    #
# --------------------------------------------------------------------------- #
def _perfect(tg, amp=30.0):
    """Logits that reproduce the targets exactly: +-amp on the sigmoid channels, the SDF / offsets themselves."""
    zs = []
    for t in tg:
        B, h, w = t["seen"].shape
        z = torch.zeros(B, MB.N_CHANNELS, h, w)
        for i, ch in enumerate(MB.CHANNELS):
            if ch in MB.PROB_CHANNELS:
                z[:, i] = (t[ch] * 2.0 - 1.0) * amp
        z[:, 1], z[:, 4], z[:, 5] = t["sdf"], t["off"][:, 0], t["off"][:, 1]
        zs.append(z)
    return zs


def _fixture(b=2, seed=0):
    codes = _real_codes(n_clips=1, frames=(20, 90))[:b]
    tg = MB.build_banded_targets(codes, x_max_m=X60, y_half_m=Y16)
    return codes, tg


def test_LOSS_a_perfect_prediction_costs_nothing_but_the_maxent_term_on_unobs_cells():
    _, tg = _fixture()
    un = [torch.zeros_like(t["seen"]) for t in tg]                # everything observed
    row = MB.banded_map_loss({"logits": _perfect(tg), "unobs": un}, tg)
    for k, v in row.items():
        if k.startswith("map_banded_loss_"):
            assert float(v) == pytest.approx(0.0, abs=1e-6), k
    assert float(row["loss"]) == pytest.approx(0.0, abs=1e-5)
    assert set(MB.telemetry_keys()) <= set(row) and len([k for k in row if k.startswith("map_banded_loss_")]) == 30


def test_LOSS_unobs_cells_contribute_exactly_zero_to_the_value_and_the_gradient():
    _, tg = _fixture()
    g = torch.Generator().manual_seed(0)
    un = [torch.rand(t["seen"].shape, generator=g) < 0.4 for t in tg]
    z = [torch.randn(t["seen"].shape[0], MB.N_CHANNELS, *t["seen"].shape[1:], generator=g).requires_grad_() for t in tg]
    old = MB.MAXENT_W
    MB.MAXENT_W = 0.0
    try:
        row = MB.banded_map_loss({"logits": z, "unobs": un}, tg, with_metrics=False)
        row["loss"].backward()
        for zb, u in zip(z, un):
            assert float(zb.grad.abs()[u[:, None].expand_as(zb)].max()) == 0.0          # EXACTLY 0.0 on every unobs cell
            assert float(zb.grad.abs().max()) > 0.0
        # changing the logits ONLY on unobs cells leaves every loss term bit-identical
        z2 = [zb.detach().clone() for zb in z]
        for zb, u in zip(z2, un):
            zb[u[:, None].expand_as(zb)] += 5.0
        row2 = MB.banded_map_loss({"logits": z2, "unobs": un}, tg, with_metrics=False)
        for k in row:
            if k.startswith("map_banded_loss_"):
                assert float(row[k]) == float(row2[k]), k
        assert float(row["loss"]) == float(row2["loss"])
    finally:
        MB.MAXENT_W = old


def test_RED_a_head_that_drops_unobs_masking_turns_the_invariance_check_red(monkeypatch):
    monkeypatch.setattr(MB, "supervised_mask", lambda seen, unobs: seen)
    with pytest.raises(AssertionError):
        test_LOSS_unobs_cells_contribute_exactly_zero_to_the_value_and_the_gradient()


def test_LOSS_maxent_pushes_unobs_cells_toward_one_half_and_only_there():
    _, tg = _fixture()
    un = [torch.ones_like(t["seen"]) for t in tg]
    z0 = [torch.zeros(t["seen"].shape[0], MB.N_CHANNELS, *t["seen"].shape[1:]) for t in tg]
    r0 = MB.banded_map_loss({"logits": z0, "unobs": un}, tg, with_metrics=False)
    # at p = 0.5 the binary entropy is 1 bit per channel: neg-entropy = -1 each, 7 channels -> maxent = -7 per band
    for b in MB.BAND_NAMES:
        assert float(r0["map_banded_maxent_%s" % b]) == pytest.approx(-7.0, abs=1e-5)
    z1 = [z + 4.0 for z in z0]
    r1 = MB.banded_map_loss({"logits": z1, "unobs": un}, tg, with_metrics=False)
    assert float(r1["map_banded_maxent_near"]) > float(r0["map_banded_maxent_near"])      # confident = higher (less negative) cost
    assert MB.MAXENT_W == 1e-3 and MB.FOCAL_GAMMA == 2.0


def test_LOSS_band_weights_are_1_1_half_and_the_RED_arm():
    _, tg = _fixture()
    un = [torch.zeros_like(t["seen"]) for t in tg]
    z = _perfect(tg)
    z[2][:, 0] = -z[2][:, 0]                                      # only the FAR band is wrong (the drv channel inverted)
    row = MB.banded_map_loss({"logits": z, "unobs": un}, tg, with_metrics=False)
    far_sum = sum(float(row["map_banded_loss_%s_far" % ch]) for ch in MB.CHANNELS)
    assert far_sum > 1.0
    assert float(row["loss"]) == pytest.approx(0.5 * far_sum, rel=1e-4)                  # band weight 0.5
    z = _perfect(tg)
    z[0][:, 0] = -z[0][:, 0]                                      # only the NEAR band is wrong
    row = MB.banded_map_loss({"logits": z, "unobs": un}, tg, with_metrics=False)
    assert float(row["loss"]) == pytest.approx(sum(float(row["map_banded_loss_%s_near" % ch]) for ch in MB.CHANNELS), rel=1e-4)


def test_RED_band_weights_1_1_1_turn_the_far_band_check_red(monkeypatch):
    monkeypatch.setattr(MB, "BAND_WEIGHTS", (1.0, 1.0, 1.0))
    with pytest.raises(AssertionError):
        test_LOSS_band_weights_are_1_1_half_and_the_RED_arm()


def test_RED_thin_targets_undilated_turn_the_dilation_test_red(monkeypatch):
    monkeypatch.setattr(MB, "dilate_thin", lambda mask, radius=2: mask)
    with pytest.raises(AssertionError):
        test_E0e_a_one_cell_lane_line_dilates_to_exactly_half_a_metre()


def test_RED_drv_from_the_8way_argmax_instead_of_the_code_set_turns_the_content_test_red(monkeypatch):
    monkeypatch.setattr(MB, "_drv_mask", lambda s: s == MB.C_DRIVABLE)       # 'class 1 only': markings fall OUT of the surface
    with pytest.raises(AssertionError):
        test_E0e_a_one_cell_lane_line_dilates_to_exactly_half_a_metre()
    with pytest.raises(AssertionError):
        test_GT_as_prediction_scores_perfect_IoU_in_every_band_and_a_shift_degrades_the_thin_channels_monotonically()


def test_LOSS_a_band_with_nothing_supervised_is_an_exact_zero_attached_to_the_graph():
    _, tg = _fixture()
    un = [torch.ones_like(t["seen"]) for t in tg]
    z = [torch.randn(t["seen"].shape[0], MB.N_CHANNELS, *t["seen"].shape[1:]).requires_grad_() for t in tg]
    old = MB.MAXENT_W
    MB.MAXENT_W = 0.0
    try:
        row = MB.banded_map_loss({"logits": z, "unobs": un}, tg, with_metrics=False)
        assert float(row["loss"]) == 0.0 and row["loss"].requires_grad
        row["loss"].backward()
        assert all(float(zb.grad.abs().max()) == 0.0 for zb in z)                       # the control: EXACTLY 0.0
    finally:
        MB.MAXENT_W = old


def test_unobs_masks_follow_the_lift_validity_per_band():
    g = MB.band_geometry(100.0, 30.0)
    v = torch.ones(1, 400, 240, dtype=torch.bool)
    v[0, 10:12, 5] = False                         # near: coarse rows 10-11 -> fine rows 25..29 (nearest-exact, x 2.5)
    v[0, 100, 7] = False                           # mid: coarse row 100 -> mid row 60
    v[0, 200, 40] = False                          # far: one of four lift cells of a 0.5 m cell -> the far cell is UNOBSERVED
    n, m, f = MB.unobs_masks(v, g)
    assert n.shape == (1, 100, 600) and m.shape == (1, 120, 240) and f.shape == (1, 120, 120)
    assert bool(n[0, 25:30, 12:15].all()) and not bool(n[0, 24, 12]) and not bool(n[0, 30, 12])
    assert bool(m[0, 60, 7]) and int(m.sum()) == 1
    assert bool(f[0, 20, 20]) and int(f.sum()) == 1

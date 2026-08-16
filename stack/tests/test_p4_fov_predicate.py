"""P4 permanence vs the FOV mask — the predicate identity, and the guard.

⭐ THIS IS THE CASE WHERE THE OBVIOUS FIX DESTROYS THE FINDING.

Every other `bev_raster` consumer got an ``_infov`` twin on 2026-08-16 (the P8 v6
port + the consumer audit), because scoring cells no camera could observe is a
correctness defect. **The P4 visible/occluded split is the exception**: its
``occluded`` arm IS the complement of that mask, because
``build_obstacle_join.visibility_occ`` and ``bev_raster.fov_mask`` are the SAME
PREDICATE. Masking it empties the finding rather than correcting it.

These tests do three jobs:
  1. **MEASURE the identity** (so the stamp is a measurement, not a quotation);
  2. **MEASURE what a twin would do** (so "it would empty it" is a number);
  3. **FAIL IF A TWIN IS EVER ADDED** — the guard, both at the API surface and in
     the source of the accumulation block itself.

CPU only. No corpus, no checkpoint, no GPU, no join file.
"""
from __future__ import annotations

import inspect
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_STACK = Path(__file__).resolve().parents[1]
_REPO = _STACK.parent
for _p in (str(_STACK), str(_STACK / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_obstacle_join as boj                                    # noqa: E402
from tanitad.data.bev_raster import (GRID_DEFAULT, cell_azimuth_rad,  # noqa: E402
                                     cell_centers_xy, fov_mask, rasterize)

torch = pytest.importorskip("torch")
tp8 = pytest.importorskip("train_p8_occupancy")

P4_DIR = (_REPO / "TanitAD Research Hub" / "Architecture & Inference"
          / "Implementation" / "incoming" / "2026-08-16-p4-fov-predicate")
BANKED = (_REPO / "TanitAD Research Hub" / "Architecture & Inference"
          / "Implementation" / "incoming" / "2026-08-07-hierarchical-wm-redesign")


# =========================================================================== 1
# the identity itself — MEASURED, not quoted
# =========================================================================== 1
@pytest.mark.parametrize("hfov", [30.0, 60.0, 90.0, 117.0, 120.0, 150.0, 179.0])
def test_occ_and_fov_mask_are_the_same_predicate(hfov):
    """`occ == 0` and `fov_mask == True` select the identical cell set."""
    X, Y = cell_centers_xy(GRID_DEFAULT)
    pts = np.stack([X.ravel(), Y.ravel()] + [np.zeros(X.size)] * 3, axis=1)
    occ = boj.visibility_occ(pts, hfov_deg=hfov).reshape(X.shape)
    msk = fov_mask(GRID_DEFAULT, math.radians(hfov) / 2.0)
    assert np.array_equal(occ == 0, msk)


def test_the_shipped_self_check_agrees_and_refuses_when_falsified(monkeypatch):
    got = boj.assert_occ_matches_fov_mask(120.0)
    assert got == {"hfov_deg": 120.0, "n_cells": 7680, "n_disagree": 0,
                   "identical": True, "n_out_of_field": 590}
    # and it is a REAL check, not a tautology: break the predicate and it fires.
    # "everything is visible" disagrees on exactly the 590 out-of-field cells.
    monkeypatch.setattr(boj, "visibility_occ",
                        lambda ag, hfov_deg=120.0: np.zeros(len(ag),
                                                            dtype=np.int64))
    with pytest.raises(AssertionError, match="590/7680 cells.*FALSIFIED"):
        boj.assert_occ_matches_fov_mask(120.0)


def test_the_two_DEFAULTS_are_bit_identical_floats():
    """Not "equal to within rounding" — the same IEEE double."""
    a = inspect.signature(fov_mask).parameters["half_angle_rad"].default
    b = math.radians(boj.HFOV_DEG_DEFAULT) / 2.0
    assert a == b
    assert a.hex() == b.hex() == "0x1.0c152382d7365p+0"


def test_the_only_difference_is_granularity():
    """`occ` grades the AGENT CENTRE; `fov_mask` grades the CELL CENTRE.

    A footprint straddling the boundary is the entire discrepancy: the agent is
    flagged occluded while some of its cells are in-field.
    """
    half = math.radians(60.0)
    # a long vehicle whose CENTRE is just outside the field
    az = half + math.radians(2.0)
    r = 6.0
    ag = [{"cx": r * math.cos(az), "cy": r * math.sin(az), "yaw": az,
           "l": 12.0, "w": 2.6, "occ": 1}]
    assert boj.visibility_occ(np.array([[ag[0]["cx"], ag[0]["cy"], 0, 0, 0]]),
                              hfov_deg=120.0).tolist() == [1]
    cells = rasterize(ag) > 0.5
    msk = fov_mask(GRID_DEFAULT, half)
    assert cells.any()
    assert (cells & msk).any(), "granularity gap should be demonstrable"
    assert (cells & ~msk).any()


# =========================================================================== 2
# what an `_infov` twin would DO — the number behind "it would empty it"
# =========================================================================== 2
def test_masking_the_occluded_arm_all_but_empties_a_sub_cell_agent():
    """A sub-cell agent keeps ~1.5 % of its cells; 98 %+ are emptied outright.

    ⚠️ NOT exactly zero, and that residual matters: the raster tests CELL
    CENTRES, so a hit sits up to half a cell diagonal (0.354 m) from the agent
    centre, which near the ego origin crosses the azimuth boundary. That IS the
    agent-centre vs cell-centre granularity gap named in the stamp. A 45-sample
    sweep reported 0 here and was an over-claim; this pins the measured value.
    """
    half = math.radians(60.0)
    msk = fov_mask(GRID_DEFAULT, half)
    rng = np.random.default_rng(11)
    n = kept = 0
    for _ in range(4000):
        # sample inside the out-of-field wedge (x < 9.2376 m), where every
        # occluded agent on this grid necessarily lives
        cx = float(rng.uniform(0.0, 9.2376))
        cy = float(rng.uniform(-16.0, 16.0))
        if abs(math.atan2(cy, cx)) <= half:
            continue
        # footprint strictly smaller than one cell => at most one cell centre
        r = rasterize([{"cx": cx, "cy": cy, "yaw": 0.0, "l": 0.45, "w": 0.45,
                        "occ": 1}]) > 0.5
        if not r.any():
            continue
        n += 1
        assert int(r.sum()) == 1
        kept += int((r & msk).any())
    assert n > 1000, f"only {n} occluded sub-cell agents sampled"
    assert 0.0 < kept / n < 0.05, (
        f"sub-cell survival {kept}/{n} — the twin should all but empty this arm")


def test_masking_the_occluded_arm_reselects_it_by_vehicle_LENGTH():
    """The survivors are boundary slivers — so a longer vehicle survives more.

    That is the point: an `_infov` twin does not produce "the occluded arm,
    corrected". It produces a different population, ordered by extent.
    """
    half = math.radians(60.0)
    msk = fov_mask(GRID_DEFAULT, half)
    rng = np.random.default_rng(12)
    surv = {}
    for tag, (ln, wd) in (("car", (4.5, 2.0)), ("truck", (12.0, 2.6))):
        tot = keep = 0
        rng2 = np.random.default_rng(rng.integers(1 << 30))
        for _ in range(1200):
            cx = float(rng2.uniform(0.5, 20.0))
            cy = float(rng2.uniform(-16.0, 16.0))
            if abs(math.atan2(cy, cx)) <= half:
                continue
            r = rasterize([{"cx": cx, "cy": cy, "yaw": 0.0, "l": ln, "w": wd,
                            "occ": 1}]) > 0.5
            tot += int(r.sum())
            keep += int((r & msk).sum())
        surv[tag] = keep / tot if tot else 0.0
    assert 0.0 < surv["car"] < 0.5
    assert surv["truck"] > surv["car"]


def test_cell_recall_returns_NaN_on_an_emptied_subset():
    """The mechanism by which a masked occluded arm silently loses its n."""
    logits = torch.full((2, 120, 64), 5.0)
    empty = torch.zeros((2, 120, 64))
    got = tp8.cell_recall(logits, empty, tau=0.5)
    assert torch.isnan(got).all()
    assert tp8._mean_n(got.tolist()) == (None, 0)


# =========================================================================== 3
# THE GUARD — fails the moment somebody "tidies" this
# =========================================================================== 3
def test_cell_recall_has_NO_mask_parameter():
    """⛔ The twin would arrive as a `mask=`/`fov=` kwarg here. It must not exist."""
    params = set(inspect.signature(tp8.cell_recall).parameters)
    assert params == {"logits", "subset_target", "tau"}, (
        "cell_recall gained a parameter. If it is a field mask, STOP: the P4 "
        "occluded arm IS the masked-out set and masking it empties the finding. "
        "See P4_SPLIT_STAMP and …/incoming/2026-08-16-p4-fov-predicate/.")


def test_the_P4_accumulation_block_never_mentions_a_field_mask():
    """Source-level guard on the exact block that builds `occ_acc`.

    A parameter check alone can be routed around (e.g. by multiplying the
    subset raster by `fov` before the call), so the block itself is scanned.
    """
    src = (_STACK / "scripts" / "train_p8_occupancy.py").read_text(
        encoding="utf-8")
    m = re.search(r"if occ_acc is not None:\n(.*?)\n    head\.train\(\)",
                  src, re.S)
    assert m, "the P4 accumulation block moved — re-anchor this guard"
    block = m.group(1)
    for bad in ("_infov", "fov", "mask"):
        assert bad not in block, (
            f"the P4 accumulation block now mentions {bad!r}. ⛔ The occluded "
            f"arm IS the complement of the camera-field mask — masking it "
            f"EMPTIES the finding rather than correcting it. The remedy is "
            f"--p4-region-control, not a twin.")


def test_the_split_output_carries_the_stamp_and_no_infov_key():
    """Whatever else changes, the emitted JSON must say what the split is."""
    stamp = tp8.P4_SPLIT_STAMP
    assert stamp["cell_set"] == tp8.P4_SPLIT_CELL_SET == "all"
    assert stamp["cell_set_is_binding"] is True
    assert "DO_NOT_ADD_AN_INFOV_TWIN" in stamp
    assert "_evidence_class" in stamp
    for k in stamp:
        assert not k.endswith("_infov")


def test_the_fov_gate_suffix_map_is_not_wired_into_the_P4_split():
    """`FOV_GATE_SUFFIX` is for the IoU arms only; it must not reach the split."""
    src = (_STACK / "scripts" / "train_p8_occupancy.py").read_text(
        encoding="utf-8")
    m = re.search(r'split\["predicate_identity"\]', src)
    assert m, "the stamp write vanished from the split"
    seg = src[src.index("if occ_acc is not None:\n        split = "):m.start()]
    assert "FOV_GATE_SUFFIX" not in seg and "_infov" not in seg


# =========================================================================== 4
# the region-matched control (the discriminator this split actually needs)
# =========================================================================== 4
def test_out_of_field_wedge_ceiling_matches_the_census():
    x = tp8.out_of_field_x_ceiling_m(GRID_DEFAULT, math.radians(60.0))
    assert x == pytest.approx(9.2376, abs=1e-4)
    X, _Y = cell_centers_xy(GRID_DEFAULT)
    out = ~fov_mask(GRID_DEFAULT, math.radians(60.0))
    assert float(X[out].max()) < x
    assert int(out.sum()) == 590


def test_select_subset_splits_the_visible_arm_at_that_boundary():
    x_ceil = tp8.out_of_field_x_ceiling_m()
    ag = np.array([
        [3.0, 0.0, 0.0, 4.5, 2.0, 0.0],       # visible, near
        [40.0, 1.0, 0.0, 4.5, 2.0, 0.0],      # visible, far
        [2.0, 8.0, 0.0, 4.5, 2.0, 1.0],       # occluded (|az| 76 deg)
    ])
    assert tp8.select_subset(ag, "visible").shape[0] == 2
    assert tp8.select_subset(ag, "occluded").shape[0] == 1
    near = tp8.select_subset(ag, "visible_near")
    far = tp8.select_subset(ag, "visible_far")
    assert near.shape[0] == 1 and near[0, 0] < x_ceil
    assert far.shape[0] == 1 and far[0, 0] >= x_ceil
    assert near.shape[0] + far.shape[0] == 2
    with pytest.raises(ValueError):
        tp8.select_subset(ag, "in_fov")


def test_every_occluded_agent_ON_THE_GRID_is_in_the_near_band():
    """Why `occluded_near` is not a separate subset: it is `occluded`."""
    half = math.radians(60.0)
    x_ceil = tp8.out_of_field_x_ceiling_m(GRID_DEFAULT, half)
    rng = np.random.default_rng(7)
    for _ in range(3000):
        cx = float(rng.uniform(0.25, 60.0))
        cy = float(rng.uniform(-16.0, 16.0))
        if abs(math.atan2(cy, cx)) > half:
            assert cx < x_ceil


def test_the_control_is_pre_registered_with_both_outcomes():
    doc = tp8.select_subset.__doc__
    assert "REGIONAL" in doc and "permanence" in doc
    assert "Both outcomes are committed here" in doc


def test_the_control_costs_no_extra_forward_pass():
    """It must reuse `log_enc`/`log_pred`, or it is not the cheap check."""
    src = (_STACK / "scripts" / "train_p8_occupancy.py").read_text(
        encoding="utf-8")
    m = re.search(r"if occ_acc is not None:\n(.*?)\n    head\.train\(\)",
                  src, re.S)
    block = m.group(1)
    assert "head(" not in block and "p8_latents" not in block, (
        "the P4 subsets must be scored against the already-computed logits")
    assert "log_enc[sub]" in block and "log_pred[sub]" in block


# =========================================================================== 4b
# the re-score path — what makes the discriminator 4 minutes instead of 6.5 h
# =========================================================================== 4b
def test_head_ckpt_roundtrips_in_both_saved_shapes(tmp_path):
    head = tp8.BEVOccupancyHead(64, grid=GRID_DEFAULT, ch0=32, ch1=16,
                                enforce_band=False)
    ref = {k: v.clone() for k, v in head.state_dict().items()}
    for name, blob in (("raw.pt", head.state_dict()),
                       ("wrapped.pt", {"head": head.state_dict(),
                                       "step": 3000})):
        p = tmp_path / name
        torch.save(blob, p)
        fresh = tp8.BEVOccupancyHead(64, grid=GRID_DEFAULT, ch0=32, ch1=16,
                                     enforce_band=False)
        tp8.load_head_ckpt(fresh, str(p), "cpu")
        for k, v in fresh.state_dict().items():
            assert torch.equal(v, ref[k]), f"{name}: {k} did not load"


def test_head_ckpt_refuses_a_mismatched_state_dict(tmp_path):
    a = tp8.BEVOccupancyHead(64, grid=GRID_DEFAULT, ch0=32, ch1=16,
                             enforce_band=False)
    p = tmp_path / "wrong.pt"
    sd = {k: v for k, v in a.state_dict().items()}
    sd.pop(next(iter(sd)))                       # drop one tensor
    sd["not_a_layer.weight"] = torch.zeros(3)
    torch.save({"head": sd}, p)
    fresh = tp8.BEVOccupancyHead(64, grid=GRID_DEFAULT, ch0=32, ch1=16,
                                 enforce_band=False)
    with pytest.raises(SystemExit, match="does not match this head"):
        tp8.load_head_ckpt(fresh, str(p), "cpu")


def test_steps_zero_is_a_valid_rescore_configuration():
    """`--steps 0` must not divide by zero anywhere on the way to mini_eval."""
    a = tp8.build_args(["--ckpt", "x.pt", "--v2-cache", "c",
                        "--v2-val-cache", "v", "--out", "o",
                        "--steps", "0", "--head-ckpt", "h.pt",
                        "--p4-region-control"])
    assert a.steps == 0 and a.p4_region_control and a.head_ckpt == "h.pt"
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        torch.optim.AdamW([torch.zeros(1, requires_grad=True)]),
        T_max=max(1, a.steps))
    assert sch.get_last_lr()
    assert list(range(1, a.steps + 1)) == []


# =========================================================================== 5
# the stamps are actually written where they must be found
# =========================================================================== 5
def test_the_join_builder_carries_the_machine_readable_stamp():
    st = boj.P4_PREDICATE_IDENTITY
    assert st["stamp"] == "P4_OCCLUDED_IS_THE_FOV_MASK_COMPLEMENT"
    assert st["occ_is_fov_mask"] is True
    assert "DO_NOT_ADD_AN_INFOV_TWIN" in st
    assert "encoder_frame_rule" in st
    assert st["_evidence_class"].startswith("MEASURED")
    assert st["stamp"] == tp8.P4_SPLIT_STAMP["stamp"], (
        "the producer's and the consumer's stamps must be the same token so a "
        "grep finds both")


def test_the_banked_P8_artifacts_are_annotated_in_place():
    for name in ("p8_gate_attempt1.json", "p8_gate_attempt2.json"):
        p = BANKED / name
        if not p.exists():
            pytest.skip(f"{name} not present")
        d = json.loads(p.read_text(encoding="utf-8"))
        ann = d.get("_p4_predicate_identity_2026_08_16")
        assert ann, f"{name} lost its P4 predicate annotation"
        assert ann["stamp"] == "P4_OCCLUDED_IS_THE_FOV_MASK_COMPLEMENT"
        # annotation ONLY — the banked numbers must be untouched
        assert d["mini_eval"]["visible_occluded_split"]["available"] is True


def test_the_census_artifact_reproduces_and_pins_the_numbers():
    p = P4_DIR / "raw" / "p4_predicate_identity.json"
    if not p.exists():
        pytest.skip("census artifact not present")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["A_predicate_identity"]["all_identical"] is True
    assert d["B_default_half_angle"]["bit_identical"] is True
    assert d["C_out_of_field_geometry"]["n_out_of_field"] == 590
    tw = d["D_infov_twin_would"]
    surv = [tw[t]["cell_survival_frac"] for t in
            ("subcell_agent_0p05", "subcell_agent_0p45",
             "automobile_4p5x2p0", "heavy_truck_12x2p6")]
    assert surv == sorted(surv), "survival must rise with vehicle extent"
    assert surv[0] == 0.0 and surv[-1] > 0.25
    assert tw["automobile_4p5x2p0"]["agent_emptied_frac"] > 0.5
    assert d["E_sensor_vs_encoder_frame"]["n_cells_disagreeing"] == 36
    assert d["E_sensor_vs_encoder_frame"]["n_out_117"] == 626
    lf0 = d["E_sensor_vs_encoder_frame"]["lf0_corridor_overlap"]
    assert lf0["pm1.5m_minrow2"]["n_disagreeing_in_corridor"] == 0
    f = d["F_banked_p4_reread"]["attempt2"]
    assert f["enc_occ_gt_vis_at_all_k"] is True
    assert f["pred_occ_gt_vis_at_all_k"] is False
    assert f["pred_k_with_positive_gap"] == ["10"]


# =========================================================================== 6
# the 120 / 117 disagreement
# =========================================================================== 6
def test_the_join_default_is_the_SENSOR_field_not_the_v5f_encoder_frame():
    assert boj.HFOV_DEG_DEFAULT == 120.0
    n120 = int((~fov_mask(GRID_DEFAULT, math.radians(120.0) / 2.0)).sum())
    n117 = int((~fov_mask(GRID_DEFAULT, math.radians(117.0) / 2.0)).sum())
    assert (n120, n117, n117 - n120) == (590, 626, 36)


def test_the_disagreeing_cells_are_exactly_the_1p5_degree_annulus():
    az = np.degrees(np.abs(cell_azimuth_rad(GRID_DEFAULT)))
    dis = (fov_mask(GRID_DEFAULT, math.radians(120.0) / 2.0)
           & ~fov_mask(GRID_DEFAULT, math.radians(117.0) / 2.0))
    assert int(dis.sum()) == 36
    assert az[dis].min() > 58.5 and az[dis].max() <= 60.0


def test_the_120_117_gap_cannot_move_the_LF0_verdict():
    """0 disagreeing cells inside LF0's scanned corridor at its run config."""
    _X, Y = cell_centers_xy(GRID_DEFAULT)
    rows = np.arange(GRID_DEFAULT.shape[0])[:, None]
    dis = (fov_mask(GRID_DEFAULT, math.radians(120.0) / 2.0)
           & ~fov_mask(GRID_DEFAULT, math.radians(117.0) / 2.0))
    corridor = (np.abs(Y) <= 1.75) & (rows >= 2)          # --min-row 2 default
    assert int((corridor & dis).sum()) == 0
    assert int(((np.abs(Y) <= 1.75) & (rows >= 0) & dis).sum()) == 2

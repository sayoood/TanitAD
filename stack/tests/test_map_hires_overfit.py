"""refcv7 NEW-2 -- the G-MAP-OVERFIT harness (``stack/scripts/map_hires_overfit.py``),
conformed to the map-signal audit's ``raw/PREREG_G_MAP_OVERFIT.md`` §9a.

Mechanics only (the real run needs the 16 TRAIN frames on Thor):

* the spec: the audit's REAL ``gmo_spec.json`` loads once its two placeholders are
  filled (class weights, decision rule) and is refused while they are not; a missing
  presence floor, a stray ``must_fail_all`` and an unknown arm are refused;
* the arms are what they say: ``healthy`` moves the trunk's stride-8 stage,
  ``s8_zeros`` / ``s8_detached`` / ``frozen_trunk`` leave it EXACTLY unchanged,
  ``lane_w0`` zeroes the lane weight in the LOSS only (the decision keeps the launch
  weights);
* the controls C1-C3 reproduce their KNOWN values on real cells;
* the verdict is reachable in every direction: PASS; FAIL when a gated regression arm
  passes; ``must_fail_all`` needs EVERY thin class to fail; INCONCLUSIVE (=> FAIL) under
  the presence floor; FAIL when a control does not reproduce; FAIL when the 1 ms time
  guard (PREREG §2 / §9a.6) did not run on every clip;
* ⭐ the audit's ``gmo_spec.json`` AS WRITTEN (md5 pinned): R3 ``refcv6_head_05m`` is
  accepted as an informative arm and recorded NOT RUN with its reason (A6 removed the
  0.5 m head); the statistics carry every band of the A7 extent.

⛔ Lands WITH the shared-file edits (the trunk's stride-8 tap).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import map_hires_overfit as O                                   # noqa: E402
from tanitad.data.rig_projection import RigCamera               # noqa: E402
from tanitad.models import bev_lift as L                        # noqa: E402
from tanitad.models import map_head_hires as H                  # noqa: E402
from tanitad.models import timm_trunk as TT                     # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width         # noqa: E402

AUDIT_SPEC = (ROOT.parent / "TanitAD Research Lab" / "Architecture & Inference" /
              "Research" / "2026-09-26-map-signal-audit" / "raw" / "gmo_spec.json")
ALL8 = {"nocls": 0.85, "drivable": 0.85, "sidewalk": 0.85, "lane": 0.5,
        "crosswalk": 0.5, "arrow": 0.5, "edge": 0.5, "hatched": 0.5}


def _spec(tmp_path, **over):
    s = {"frames": [{"clip_sha12": "x" * 12, "raw_frame": 5}], "steps": 4,
         "lr": 1e-2, "batch": 2, "eval_every": 4, "seed": 0, "weight_decay": 0.0,
         "class_weights": "uniform",
         "thresholds": {"iou": dict(ALL8), "band": "0_20", "min_cells": 10,
                        "decision_rule": "prior_corrected",
                        "per_class_ce_final_over_step0_max": 0.5},
         "must_fail": {"lane_w0": ["lane"], "s8_zeros": list(O.THIN)},
         "must_fail_all": {"s8_zeros": True},
         "informative_arms": ["s8_detached", "frozen_trunk", "refcv6_head_05m"]}
    s.update(over)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(s), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# the spec                                                                     #
# --------------------------------------------------------------------------- #
#: md5 of the audit's gmo_spec.json AS LANDED (12953d2): the harness is tested against
#: that file byte for byte, so a later edit to the spec cannot pass silently.
AUDIT_SPEC_MD5 = "d3a41b92918d9623f4719fe2ae548815"


@pytest.mark.skipif(not AUDIT_SPEC.is_file(), reason="the audit's gmo_spec.json is "
                    "not in this checkout")
def test_the_audits_real_spec_loads_as_written_once_its_placeholders_are_filled():
    import hashlib
    assert hashlib.md5(AUDIT_SPEC.read_bytes()).hexdigest() == AUDIT_SPEC_MD5
    with pytest.raises(SystemExit, match="decision rule"):
        O.load_spec(AUDIT_SPEC)
    with pytest.raises(SystemExit, match="class weights"):
        O.load_spec(AUDIT_SPEC, decision_rule="prior_corrected")
    from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7
    s = O.load_spec(AUDIT_SPEC, decision_rule="prior_corrected", class_weights="w.json",
                    band_keys=EXTENT_REFCV7.band_keys)
    assert len(s["frames"]) == 16 and s["steps"] == 1000 and s["batch"] == 4
    assert s["thresholds"]["min_cells"] == 1000 and s["weight_decay"] == 0.0
    assert s["must_fail"]["s8_zeros"] == list(O.THIN)
    assert s["must_fail_all"] == {"s8_zeros": True}
    assert s["informative_arms"] == ["s8_detached", "frozen_trunk", "refcv6_head_05m"]
    assert s["frameset_md5"] == "4eafa03c2b6a6e6d6336be1d78acb91d"
    # R3 is informative and NOT RUN under A6: recorded with its reason, never gated
    v = O.verdict(_results(), s, _OK_CTRL, _TIME_OK)
    assert v["informative"]["refcv6_head_05m"]["ran"] is False
    assert "A6" in v["informative"]["refcv6_head_05m"]["why"]
    assert v["G_MAP_OVERFIT"] == "PASS"


def test_spec_refusals(tmp_path):
    O.load_spec(_spec(tmp_path))
    th = {"iou": dict(ALL8), "band": "0_20", "decision_rule": "raw"}
    with pytest.raises(SystemExit, match="min_cells"):
        O.load_spec(_spec(tmp_path, thresholds=th))
    with pytest.raises(SystemExit, match="must_fail_all"):
        O.load_spec(_spec(tmp_path, must_fail_all={"lane_w0": True},
                          must_fail={"s8_zeros": ["lane"]}))
    with pytest.raises(SystemExit, match="informative"):
        O.load_spec(_spec(tmp_path, informative_arms=["s8_random"]))
    with pytest.raises(SystemExit, match="not in"):
        O.load_spec(_spec(tmp_path, thresholds=dict(th, iou={"lanes": 0.5},
                                                      min_cells=1)))
    with pytest.raises(SystemExit, match="raw_frame"):
        O.load_spec(_spec(tmp_path, frames=[{"clip_sha12": "c" * 12}]))


def test_the_audit_frameset_format_expands(tmp_path):
    spec = O.load_spec(_spec(tmp_path, frames=[
        {"clip_sha12": "aaaaaaaaaaaa", "raw_v2ep_frames": [10, 110]},
        {"clip_sha12": "bbbbbbbbbbbb", "raw_frame": 50}]))
    assert spec["frames"] == [{"clip_sha12": "aaaaaaaaaaaa", "raw_frame": 10},
                              {"clip_sha12": "aaaaaaaaaaaa", "raw_frame": 110},
                              {"clip_sha12": "bbbbbbbbbbbb", "raw_frame": 50}]


# --------------------------------------------------------------------------- #
# the arms and the controls                                                    #
# --------------------------------------------------------------------------- #
def _tiny():
    torch.manual_seed(0)
    trunk = TT.TimmResNetTrunk(TT.TimmTrunkConfig(
        model_name="resnet18.a1_in1k", pretrained=False, verify_imagenet_stats=False,
        frames=3, image_hw=(64, 128), frozen_bn=True))
    trunk.enable_s8_tap()
    cfg = H.MapHiresConfig(w_map_hires=1.0, d_lift=16, d_model=16, d_up=16,
                           dilations=(1,))
    branch = H.MapHiresBranch(cfg, d_image=trunk.s8_dim, image_hw=trunk.s8_shape)
    frame = frame_for_width(128, 64)
    g = L.build_lift_geometry(RigCamera.nominal(frame, height_m=1.5, x_m=1.5),
                              frame=frame, stride=8, grid=cfg.lift_grid)
    codes = torch.full((2, 600, 320), 1, dtype=torch.uint8)
    codes[:, :, 160:] = 7
    codes[:, :, 100:120] = 2
    codes[:, 50:60, 140:150] = 3
    data = {"x": torch.randint(0, 255, (2, 9, 64, 128), dtype=torch.uint8),
            "codes": codes, "grid": g.grid.unsqueeze(0).repeat(2, 1, 1, 1, 1),
            "valid": g.valid.unsqueeze(0).repeat(2, 1, 1, 1),
            "sha12": ["a" * 12, "b" * 12], "raw_frame": [5, 6]}
    return trunk, branch, data


def test_the_arms_are_what_they_say(tmp_path):
    trunk, branch, data = _tiny()
    spec = O.load_spec(_spec(tmp_path))
    # near_zeros / near_block_zeros need a near lift / block (NEW-2 R2 / R3): see test_R2_*,
    # test_R3_* below
    res = {a: O.run_arm(a, trunk, branch, data, spec, torch.ones(8), "cpu")
           for a in O.ARMS if a not in ("near_zeros", "near_block_zeros")}
    # NEW-2 R4: edge_w0 zeroes edge's LOSS weight only (the decision keeps the launch weights)
    assert res["edge_w0"]["w_loss"][O.EDGE] == 0.0 and res["edge_w0"]["w_decision"][O.EDGE] == 1.0
    assert res["edge_w0"]["w_loss"][O.LANE] == 1.0 and res["edge_w0"]["trunk_s8_abs_change"] > 0.0
    assert res["healthy"]["trunk_s8_abs_change"] > 0.0
    for a in ("s8_zeros", "s8_detached", "frozen_trunk"):
        assert res[a]["trunk_s8_abs_change"] == 0.0, a
    assert res["lane_w0"]["w_loss"][2] == 0.0 and res["lane_w0"]["w_decision"][2] == 1.0
    for a in res:
        assert res[a]["loss_finite_every_step"] and res[a]["final"]["n"]["lane"] > 0
    assert trunk.s8_tap and all(p.grad is None for p in branch.parameters())
    ctrl = O.controls(data, res["healthy"]["_logits"], torch.ones(8),
                      "prior_corrected", "0_20")
    assert all(v["reproduced"] for v in ctrl.values()), ctrl
    c1 = ctrl["C1_constant_drivable"]
    assert c1["iou"]["drivable"] == pytest.approx(c1["n_drivable"] / c1["n_scored"],
                                                  abs=1e-12)
    assert c1["iou"]["sidewalk"] == 0.0 and c1["iou"]["arrow"] is None


def test_C2_reproduces_under_the_prior_corrected_rule_with_real_weights():
    _t, _b, data = _tiny()
    lv = torch.ones(2, 600, 320, dtype=torch.bool)
    w = torch.tensor([0.16, 0.07, 0.97, 2.6, 13.6, 4.6, 1.0, 0.06])
    ctrl = O.controls(data, [(torch.zeros(2, 8, 600, 320), lv)], w,
                      "prior_corrected", "0_20")
    assert ctrl["C2_gt_as_logits"]["reproduced"] is True
    assert ctrl["C2_gt_as_logits"]["iou"]["lane"] == 1.0


# --------------------------------------------------------------------------- #
# the verdict                                                                  #
# --------------------------------------------------------------------------- #
def _final(iou: dict, n: int = 5000, ce0: float = 2.0, ce1: float = 0.5) -> dict:
    return {"step0": {"ce_mean": {c: ce0 for c in H.CLASS_KEYS}},
            "final": {"iou": {c: iou.get(c, 0.9) for c in H.CLASS_KEYS},
                      "n": {c: n for c in H.CLASS_KEYS},
                      "ce_mean": {c: ce1 for c in H.CLASS_KEYS}},
            "loss_finite_every_step": True}


_OK_CTRL = {"C1_constant_drivable": {"reproduced": True,
                                     "iou": {c: (0.4 if c == "drivable" else 0.0)
                                             for c in H.CLASS_KEYS}},
            "C2_gt_as_logits": {"reproduced": True},
            "C3_rule_identity_w_ones": {"reproduced": True}}


def _results(**over):
    r = {"healthy": _final({}),
         "lane_w0": _final({"lane": 0.01}),
         "s8_zeros": _final({t: 0.05 for t in O.THIN})}
    r.update(over)
    return r


_TIME_OK = {"a" * 12: {"kind": "time_1ms"}, "b" * 12: {"kind": "time_1ms"}}


def test_the_verdict_is_reachable_in_every_direction(tmp_path):
    spec = O.load_spec(_spec(tmp_path))
    assert O.verdict(_results(), spec, _OK_CTRL, _TIME_OK)["G_MAP_OVERFIT"] == "PASS"
    # ⛔ PREREG §2 / §9a.6: only the pose-content guard ran on a clip -> FAIL; no
    # guard record at all -> FAIL
    weak = {"a" * 12: {"kind": "time_1ms"}, "b" * 12: {"kind": "pose_content_5cm"}}
    v = O.verdict(_results(), spec, _OK_CTRL, weak)
    assert v["time_guard"]["time_1ms_on_every_clip"] is False
    assert v["G_MAP_OVERFIT"] == "FAIL"
    assert O.verdict(_results(), spec, _OK_CTRL)["G_MAP_OVERFIT"] == "FAIL"
    # R1 passes (a non-discriminating arm): FAIL
    v = O.verdict(_results(lane_w0=_final({})), spec, _OK_CTRL, _TIME_OK)
    assert v["G_MAP_OVERFIT"] == "FAIL"
    assert v["regression_arms"]["lane_w0"]["failed_as_required"] is False
    # R2 fails only FOUR of the five thin classes: must_fail_all => FAIL
    four = {t: 0.05 for t in O.THIN[:4]}
    v = O.verdict(_results(s8_zeros=_final(four)), spec, _OK_CTRL, _TIME_OK)
    assert v["regression_arms"]["s8_zeros"]["rule"] == "all"
    assert v["G_MAP_OVERFIT"] == "FAIL"
    # MAIN misses a thin bar: FAIL; the CE did not halve: FAIL
    assert O.verdict(_results(healthy=_final({"edge": 0.3})), spec,
                     _OK_CTRL, _TIME_OK)["G_MAP_OVERFIT"] == "FAIL"
    assert O.verdict(_results(healthy=_final({}, ce1=1.5)), spec,
                     _OK_CTRL, _TIME_OK)["G_MAP_OVERFIT"] == "FAIL"
    # the presence floor: a class short of min_cells is INCONCLUSIVE => FAIL
    v = O.verdict(_results(healthy=_final({}, n=5)), spec, _OK_CTRL, _TIME_OK)
    assert v["MAIN"]["verdict"] == "INCONCLUSIVE" and v["G_MAP_OVERFIT"] == "FAIL"
    # a control that does not reproduce: FAIL
    bad = dict(_OK_CTRL, C3_rule_identity_w_ones={"reproduced": False})
    assert O.verdict(_results(), spec, bad, _TIME_OK)["G_MAP_OVERFIT"] == "FAIL"
    # a gated arm that did not run cannot certify anything
    only = {"healthy": _final({})}
    assert O.verdict(only, spec, _OK_CTRL, _TIME_OK)["G_MAP_OVERFIT"] == "FAIL"
    # informative arms are REPORTED, never gated
    v = O.verdict(_results(s8_detached=_final({})), spec, _OK_CTRL, _TIME_OK)
    assert v["G_MAP_OVERFIT"] == "PASS" and v["informative"]["s8_detached"]["ran"]
    assert v["informative"]["refcv6_head_05m"]["ran"] is False


def test_the_statistics_carry_every_band_of_the_A7_extent():
    """The gated band stays the prereg's 0_20; every 20 m band of the 100 m x +-30 m
    extent is reported (SPEC_REFCV7 §11.2 item 1)."""
    from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7
    bk = EXTENT_REFCV7.band_keys
    tot = {k: torch.ones(8, 5) for k in ("n", "inter", "union", "interraw",
                                          "unionraw", "ce")}
    tot["inter"][:, 4] = 0.0
    sm = O.summarise(tot, "0_20", bk)
    assert sm["iou"]["lane"] == 1.0 and sm["n"]["lane"] == 1
    assert list(sm["by_band"]["lane"]) == list(bk)
    assert sm["by_band"]["lane"]["80_100"]["iou"] == 0.0
    with pytest.raises(SystemExit, match="band"):
        O.load_spec(_spec(Path(__import__("tempfile").mkdtemp()),
                          thresholds={"iou": dict(ALL8), "band": "60_80",
                                      "min_cells": 1, "decision_rule": "raw"}))
    O.load_spec(_spec(Path(__import__("tempfile").mkdtemp()),
                      thresholds={"iou": dict(ALL8), "band": "60_80", "min_cells": 1,
                                  "decision_rule": "raw"}), band_keys=bk)



# =========================================================================== #
# NEW-2 R2 (SPEC_REFCV7 A12): the near-lift arm and its must-fail arm          #
# =========================================================================== #
A12_SPEC = (ROOT.parent / "TanitAD Research Lab" / "Architecture & Inference" /
            "Research" / "2026-09-26-refcv7-map-hires" / "raw" / "gmo_spec_A12.json")
#: md5 of the A12 spec AS WRITTEN (registered by SPEC_REFCV7 §17, ab1fb45)
A12_SPEC_MD5 = "f0aae8f10087194ebdada897a435e606"


@pytest.mark.skipif(not A12_SPEC.is_file(), reason="the A12 spec is not in this checkout")
def test_R2_the_A12_spec_loads_as_written_and_amends_only_what_A12_says():
    import hashlib
    assert hashlib.md5(A12_SPEC.read_bytes()).hexdigest() == A12_SPEC_MD5
    from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7
    s = O.load_spec(A12_SPEC, decision_rule="prior_corrected", class_weights="w.json",
                    band_keys=EXTENT_REFCV7.band_keys)
    assert s["near_lift_m"] == 20.0
    assert s["must_fail"] == {"lane_w0": ["lane"], "s8_zeros": list(O.THIN),
                              "near_zeros": ["edge"]}
    assert s["must_fail_all"] == {"s8_zeros": True}
    assert len(s["frames"]) == 16 and s["steps"] == 1000 and s["batch"] == 4
    assert s["frameset_md5"] == "4eafa03c2b6a6e6d6336be1d78acb91d"
    assert s["amends"]["spec_md5"] == AUDIT_SPEC_MD5                  # the prereg, as landed
    if AUDIT_SPEC.is_file():
        base = O.load_spec(AUDIT_SPEC, decision_rule="prior_corrected",
                           class_weights="w.json", band_keys=EXTENT_REFCV7.band_keys)
        diff = sorted(k for k in set(base) | set(s)
                      if base.get(k) != s.get(k))
        assert diff == ["amends", "must_fail", "near_lift_m", "registered"], diff


def _near_tiny(near=10.0):
    """The trunk and data of :func:`_tiny`, and TWO branches from ONE seed: NEW-2 as landed
    and with the near lift."""
    trunk, _branch, data = _tiny()
    kw = dict(w_map_hires=1.0, d_lift=16, d_model=16, d_up=16, dilations=(1,))
    torch.manual_seed(0)
    b0 = H.MapHiresBranch(H.MapHiresConfig(**kw), d_image=trunk.s8_dim,
                          image_hw=trunk.s8_shape)
    torch.manual_seed(0)
    nb = H.MapHiresBranch(H.MapHiresConfig(near_lift_x_m=near, **kw), d_image=trunk.s8_dim,
                          image_hw=trunk.s8_shape)
    return trunk, b0, nb, data


def test_R2_the_near_arms_run_and_near_zeros_feeds_the_near_lift_zeros(tmp_path):
    trunk, _b, nb, data = _near_tiny()
    spec = O.load_spec(_spec(tmp_path, must_fail={"lane_w0": ["lane"],
                                                  "s8_zeros": list(O.THIN),
                                                  "near_zeros": ["edge"]}))
    seen = []
    orig = type(nb).forward

    def spy(self, f, g, v, *, near_source=None):
        seen.append(None if near_source is None else float(near_source.abs().sum()))
        return orig(self, f, g, v, near_source=near_source)
    type(nb).forward = spy
    try:
        res = {a: O.run_arm(a, trunk, nb, data, spec, torch.ones(8), "cpu")
               for a in ("healthy", "near_zeros")}
    finally:
        type(nb).forward = orig
    assert res["near_zeros"]["trunk_s8_abs_change"] > 0.0      # the trunk trains as in MAIN
    assert res["near_zeros"]["loss_finite_every_step"]
    # healthy never passes a source; near_zeros passes zeros on EVERY forward
    assert None in seen and 0.0 in seen and all(x in (None, 0.0) for x in seen)


def test_R2_the_harness_refuses_near_zeros_without_a_near_lift_and_a_spec_mismatch(tmp_path):
    base = ["--spec", str(_spec(tmp_path)), "--v2-cache", "V", "--gt-root", "G",
            "--extrinsics", "E", "--out", str(tmp_path / "o"), "--device", "cpu"]
    with pytest.raises(SystemExit, match="near_zeros arm needs --near-lift-m"):
        O.main(base + ["--arms", "healthy,lane_w0,s8_zeros,near_zeros"])
    sp = _spec(tmp_path, near_lift_m=20.0)
    with pytest.raises(SystemExit, match="registers near_lift_m"):
        O.main(["--spec", str(sp), "--v2-cache", "V", "--gt-root", "G", "--extrinsics", "E",
                "--out", str(tmp_path / "o"), "--device", "cpu", "--near-lift-m", "10"])


def test_R2_the_record_proves_the_shared_init_with_the_NEW2_MAIN():
    """``branch_init_sha256_without_near`` of a near branch == ``branch_init_sha256`` of the
    NEW-2 branch from the same seed; the full fingerprints differ (the zero skip)."""
    trunk, branch, nb, _d = _near_tiny()
    assert O._fingerprint_without(nb, "near.") == O._fingerprint(branch)
    assert O._fingerprint(nb) != O._fingerprint(branch)
    assert O._fingerprint_without(branch, "near.") == O._fingerprint(branch)



# =========================================================================== #
# NEW-2 R3 (SPEC_REFCV7 §20, A15): the decoder arm and its must-fail arm       #
# =========================================================================== #
A15_SPEC = (ROOT.parent / "TanitAD Research Lab" / "Architecture & Inference" /
            "Research" / "2026-09-26-refcv7-map-hires" / "raw" / "gmo_spec_A15.json")
#: md5 of the A15 spec AS WRITTEN (registered by SPEC_REFCV7 §20, c1ed8d9)
A15_SPEC_MD5 = "20929e21a577a6374f31b3f754ddd4d4"


@pytest.mark.skipif(not A15_SPEC.is_file(), reason="the A15 spec is not in this checkout")
def test_R3_the_A15_spec_loads_as_written_and_amends_only_what_A15_says():
    import hashlib
    assert hashlib.md5(A15_SPEC.read_bytes()).hexdigest() == A15_SPEC_MD5
    from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7
    s = O.load_spec(A15_SPEC, decision_rule="prior_corrected", class_weights="w.json",
                    band_keys=EXTENT_REFCV7.band_keys)
    assert s["near_lift_m"] == 20.0 and s["near_refine_blocks"] == 1
    assert s["must_fail"] == {"lane_w0": ["lane"], "s8_zeros": list(O.THIN),
                              "near_block_zeros": ["edge"]}
    assert s["must_fail_all"] == {"s8_zeros": True}
    assert len(s["frames"]) == 16 and s["steps"] == 1000 and s["batch"] == 4
    assert s["amends"]["spec_md5"] == A12_SPEC_MD5                    # the A12 spec, as landed
    if A12_SPEC.is_file():
        base = O.load_spec(A12_SPEC, decision_rule="prior_corrected",
                           class_weights="w.json", band_keys=EXTENT_REFCV7.band_keys)
        diff = sorted(k for k in set(base) | set(s) if base.get(k) != s.get(k))
        assert diff == ["amends", "must_fail", "near_refine_blocks", "registered"], diff


def _r3_tiny():
    trunk, _b, data = _tiny()
    kw = dict(w_map_hires=1.0, d_lift=16, d_model=16, d_up=16, dilations=(1,),
              near_lift_x_m=10.0)
    torch.manual_seed(0)
    r2 = H.MapHiresBranch(H.MapHiresConfig(**kw), d_image=trunk.s8_dim,
                          image_hw=trunk.s8_shape)
    torch.manual_seed(0)
    r3 = H.MapHiresBranch(H.MapHiresConfig(near_refine_blocks=1, **kw),
                          d_image=trunk.s8_dim, image_hw=trunk.s8_shape)
    return trunk, r2, r3, data


def test_R3_the_block_arms_run_and_near_block_zeros_feeds_the_block_zeros(tmp_path):
    trunk, _r2, r3, data = _r3_tiny()
    spec = O.load_spec(_spec(tmp_path, must_fail={"lane_w0": ["lane"],
                                                  "s8_zeros": list(O.THIN),
                                                  "near_block_zeros": ["edge"]}))
    seen = []
    orig = type(r3).forward

    def spy(self, f, g, v, *, near_source=None, near_block_zeros=False):
        seen.append(bool(near_block_zeros))
        return orig(self, f, g, v, near_source=near_source, near_block_zeros=near_block_zeros)
    type(r3).forward = spy
    try:
        res = {a: O.run_arm(a, trunk, r3, data, spec, torch.ones(8), "cpu")
               for a in ("healthy", "near_block_zeros")}
    finally:
        type(r3).forward = orig
    assert res["near_block_zeros"]["trunk_s8_abs_change"] > 0.0
    assert res["near_block_zeros"]["loss_finite_every_step"]
    assert True in seen and False in seen


def test_R3_the_harness_refuses_near_block_zeros_without_a_block_and_a_spec_mismatch(tmp_path):
    base = ["--spec", str(_spec(tmp_path)), "--v2-cache", "V", "--gt-root", "G",
            "--extrinsics", "E", "--out", str(tmp_path / "o"), "--device", "cpu"]
    with pytest.raises(SystemExit, match="near_block_zeros arm needs --near-refine-blocks"):
        O.main(base + ["--near-lift-m", "20",
                       "--arms", "healthy,lane_w0,s8_zeros,near_block_zeros"])
    sp = _spec(tmp_path, near_refine_blocks=1)
    with pytest.raises(SystemExit, match="registers near_refine_blocks"):
        O.main(["--spec", str(sp), "--v2-cache", "V", "--gt-root", "G", "--extrinsics", "E",
                "--out", str(tmp_path / "o"), "--device", "cpu", "--near-lift-m", "20"])


def test_R3_the_record_proves_the_shared_init_with_R2_and_NEW2():
    trunk, r2, r3, _d = _r3_tiny()
    assert O._fingerprint_without(r3, "near_refine.") == O._fingerprint(r2)
    assert O._fingerprint_without(r3, "near.", "near_refine.") == \
        O._fingerprint_without(r2, "near.")
    assert O._fingerprint(r3) != O._fingerprint(r2)



# =========================================================================== #
# NEW-2 R4 (§9 lever 4, the weights): edge_w0 and the registered definition    #
# =========================================================================== #
def test_R4_the_class_indices_the_harness_zeroes_are_the_named_classes():
    assert H.CLASS_KEYS[O.LANE] == "lane" and H.CLASS_KEYS[O.EDGE] == "edge"   # literals


def test_R4_a_spec_registering_a_weights_definition_refuses_another_file(tmp_path):
    """The lever-4 arm registers WHICH weights (e.g. ``mf``); a file of another definition
    (the A8 ``sqrt_mf`` launch file) is refused before any trunk is built."""
    from tanitad.data.semantic_map_gt_fine import FINE_CLASSES
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"schema": H.CLASS_WEIGHT_SCHEMA, "classes": list(FINE_CLASSES),
                             "weights": [1.0] * 8, "dry_run": False,
                             "definition_id": "sqrt_mf", "pre_registered": True,
                             "extent": {"x_max_m": 100.0, "y_half_m": 30.0}}),
                 encoding="utf-8")
    sp = _spec(tmp_path, class_weights_definition="mf")
    with pytest.raises(SystemExit, match="registers class weights 'mf'"):
        O.main(["--spec", str(sp), "--class-weights", str(w), "--v2-cache", "V",
                "--gt-root", "G", "--extrinsics", "E", "--out", str(tmp_path / "o"),
                "--device", "cpu"])


def test_R4_a_spec_registering_a_weights_definition_accepts_a_file_of_that_definition(
        tmp_path, monkeypatch):
    """The refusal test's red arm: a file OF the registered definition passes the check and
    main() proceeds to build the trunk (stubbed to raise a sentinel there). A check that
    refused every file -- or read the wrong key -- passes the refusal test and FAILS this."""
    from tanitad.data.semantic_map_gt_fine import FINE_CLASSES
    from tanitad.models import timm_trunk as TT

    class ReachedTheTrunkBuild(Exception):
        pass

    def _stop(*_a, **_k):
        raise ReachedTheTrunkBuild()

    monkeypatch.setattr(TT, "TimmResNetTrunk", _stop)
    for definition in ("mf", "sqrt_mf"):                        # the spec names it; literals
        w = tmp_path / f"w_{definition}.json"
        w.write_text(json.dumps({"schema": H.CLASS_WEIGHT_SCHEMA,
                                 "classes": list(FINE_CLASSES), "weights": [1.0] * 8,
                                 "dry_run": False, "definition_id": definition,
                                 "pre_registered": definition == "sqrt_mf",
                                 "extent": {"x_max_m": 100.0, "y_half_m": 30.0}}),
                     encoding="utf-8")
        sp = _spec(tmp_path, class_weights_definition=definition)
        with pytest.raises(ReachedTheTrunkBuild):
            O.main(["--spec", str(sp), "--class-weights", str(w), "--v2-cache", "V",
                    "--gt-root", "G", "--extrinsics", "E", "--out", str(tmp_path / "o"),
                    "--device", "cpu"])



# =========================================================================== #
# NEW-2 R4 (SPEC_REFCV7 §21, A16): the weights arm's registered spec          #
# =========================================================================== #
A16_SPEC = (ROOT.parent / "TanitAD Research Lab" / "Architecture & Inference" /
            "Research" / "2026-09-26-refcv7-map-hires" / "raw" / "gmo_spec_A16.json")
#: md5 of the A16 spec AS WRITTEN (registered by SPEC_REFCV7 §21, 879673c)
A16_SPEC_MD5 = "a4ef45d0d6da9383ee356a52eaf6dde7"


@pytest.mark.skipif(not A16_SPEC.is_file(), reason="the A16 spec is not in this checkout")
def test_R4_the_A16_spec_loads_as_written_and_amends_only_what_A16_says():
    import hashlib
    assert hashlib.md5(A16_SPEC.read_bytes()).hexdigest() == A16_SPEC_MD5
    from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7
    s = O.load_spec(A16_SPEC, decision_rule="prior_corrected", class_weights="w.json",
                    band_keys=EXTENT_REFCV7.band_keys)
    assert s["near_lift_m"] == 20.0 and s["near_refine_blocks"] == 1         # A15's, stacked
    assert s["class_weights_definition"] == "mf"                               # the lever
    assert s["must_fail"] == {"lane_w0": ["lane"],
                              "s8_zeros": ["lane", "crosswalk", "arrow", "edge", "hatched"],
                              "edge_w0": ["edge"]}
    assert s["must_fail_all"] == {"s8_zeros": True}
    assert len(s["frames"]) == 16 and s["steps"] == 1000 and s["batch"] == 4
    assert s["lr"] == 0.001 and s["seed"] == 0 and s["eval_every"] == 100
    assert s["amends"]["spec_md5"] == "20929e21a577a6374f31b3f754ddd4d4"      # A15, as landed
    if A15_SPEC.is_file():
        base = O.load_spec(A15_SPEC, decision_rule="prior_corrected",
                           class_weights="w.json", band_keys=EXTENT_REFCV7.band_keys)
        diff = sorted(k for k in set(base) | set(s) if base.get(k) != s.get(k))
        assert diff == ["amends", "class_weights_definition", "must_fail", "registered"], diff



# =========================================================================== #
# NEW-2 R5 (candidate; the map twin of A17): the lr decays over the final 10 % #
# =========================================================================== #
def test_R5_the_lr_multiplier_is_the_registered_cosine_literals():
    m = O.lr_multiplier({"steps": 1000,
                         "lr_decay": {"kind": "cosine_to_zero", "start_step": 900}})
    assert m(1) == 1.0 and m(899) == 1.0 and m(900) == 1.0            # held at peak
    assert m(950) == pytest.approx(0.5, abs=1e-12)                      # half-way
    assert m(1000) == pytest.approx(0.0, abs=1e-12)                     # zero at the end
    assert m(925) == pytest.approx(0.8535533905932737, abs=1e-12)       # (1 + cos(pi/4)) / 2
    assert O.lr_multiplier({"steps": 1000}) is None                     # no key: constant


@pytest.mark.parametrize("d", [{"kind": "linear", "start_step": 900},
                               {"kind": "cosine_to_zero", "start_step": 0},
                               {"kind": "cosine_to_zero", "start_step": 1000},
                               {"kind": "cosine_to_zero", "start_step": 900.0},
                               {"kind": "cosine_to_zero", "start_step": True},
                               {"kind": "cosine_to_zero"}])
def test_R5_a_malformed_lr_decay_is_refused(d):
    with pytest.raises(SystemExit, match="lr_decay"):
        O.lr_multiplier({"steps": 1000, "lr_decay": d})


def _lrs_applied(tmp_path, monkeypatch, **spec_over):
    """The lr each opt.step() actually applied, on the tiny rig (4 steps, base lr 1e-2)."""
    trunk, branch, data = _tiny()
    seen = []
    orig = torch.optim.AdamW.step

    def spy(self, *a, **k):
        seen.append(float(self.param_groups[0]["lr"]))
        return orig(self, *a, **k)
    monkeypatch.setattr(torch.optim.AdamW, "step", spy)
    spec = O.load_spec(_spec(tmp_path, **spec_over))
    res = O.run_arm("healthy", trunk, branch, data, spec, torch.ones(8), "cpu")
    return seen, res


def test_R5_the_decay_is_applied_step_by_step_and_off_is_constant(tmp_path, monkeypatch):
    on, res_on = _lrs_applied(tmp_path, monkeypatch,
                              lr_decay={"kind": "cosine_to_zero", "start_step": 2})
    assert on == pytest.approx([0.01, 0.01, 0.005, 0.0], abs=1e-12)    # literals
    assert res_on["curve"][-1]["lr"] == pytest.approx(0.0, abs=1e-12)
    off, res_off = _lrs_applied(tmp_path, monkeypatch)
    assert off == [0.01, 0.01, 0.01, 0.01]                               # untouched
    assert "lr" not in res_off["curve"][-1]                              # the record is as before


def test_R5_a_decay_that_bites_changes_the_result_and_the_constant_path_is_deterministic(
        tmp_path):
    """The decay is not inert (its final differs from the constant path's), and the constant
    path reproduces itself exactly, so the difference is the decay's and not noise. (That the
    constant path never touches the lr is pinned step by step in the test above.)"""
    trunk, branch, data = _tiny()
    a = O.run_arm("healthy", trunk, branch, data, O.load_spec(_spec(tmp_path)), torch.ones(8),
                  "cpu")
    trunk, branch, data = _tiny()
    b = O.run_arm("healthy", trunk, branch, data,
                  O.load_spec(_spec(tmp_path, steps=4, lr_decay={"kind": "cosine_to_zero",
                                                                 "start_step": 2})),
                  torch.ones(8), "cpu")
    assert a["final"]["ce_mean"] != b["final"]["ce_mean"]               # the decay bites
    trunk, branch, data = _tiny()
    c = O.run_arm("healthy", trunk, branch, data, O.load_spec(_spec(tmp_path)), torch.ones(8),
                  "cpu")
    assert a["final"]["ce_mean"] == c["final"]["ce_mean"]               # determinism control



# =========================================================================== #
# NEW-2 R5 (SPEC_REFCV7 §22.1, A17.1): the lr-decay arm's registered spec     #
# =========================================================================== #
A171_SPEC = (ROOT.parent / "TanitAD Research Lab" / "Architecture & Inference" /
             "Research" / "2026-09-26-refcv7-map-hires" / "raw" / "gmo_spec_A171.json")
#: md5 of the A17.1 arm spec AS WRITTEN (registered by SPEC_REFCV7 §22.1, 2ac0bfb)
A171_SPEC_MD5 = "5abd5b907738fc735a0def0c1e7fbb64"


@pytest.mark.skipif(not A171_SPEC.is_file(), reason="the A17.1 arm spec is not in this checkout")
def test_R5_the_A171_spec_loads_as_written_and_amends_only_what_A17_1_says():
    import hashlib
    assert hashlib.md5(A171_SPEC.read_bytes()).hexdigest() == A171_SPEC_MD5
    from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7
    s = O.load_spec(A171_SPEC, decision_rule="prior_corrected", class_weights="w.json",
                    band_keys=EXTENT_REFCV7.band_keys)
    assert s["lr_decay"] == {"kind": "cosine_to_zero", "start_step": 900}      # A17.1
    m = O.lr_multiplier(s)
    assert m(900) == 1.0 and m(950) == pytest.approx(0.5, abs=1e-12)
    assert m(1000) == pytest.approx(0.0, abs=1e-12)
    assert s["near_lift_m"] == 20.0 and s["near_refine_blocks"] == 1          # A15's config
    assert "class_weights_definition" not in s                                 # sqrt_mf (A8)
    assert s["must_fail"] == {"lane_w0": ["lane"],
                              "s8_zeros": ["lane", "crosswalk", "arrow", "edge", "hatched"],
                              "near_block_zeros": ["edge"]}
    assert s["must_fail_all"] == {"s8_zeros": True}
    assert len(s["frames"]) == 16 and s["steps"] == 1000 and s["batch"] == 4
    assert s["lr"] == 0.001 and s["seed"] == 0 and s["eval_every"] == 100
    assert s["amends"]["spec_md5"] == "20929e21a577a6374f31b3f754ddd4d4"      # A15, as landed
    if A15_SPEC.is_file():
        base = O.load_spec(A15_SPEC, decision_rule="prior_corrected",
                           class_weights="w.json", band_keys=EXTENT_REFCV7.band_keys)
        diff = sorted(k for k in set(base) | set(s) if base.get(k) != s.get(k))
        assert diff == ["amends", "lr_decay", "registered"], diff

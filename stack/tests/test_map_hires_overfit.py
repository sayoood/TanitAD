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
    res = {a: O.run_arm(a, trunk, branch, data, spec, torch.ones(8), "cpu")
           for a in O.ARMS}
    assert res["healthy"]["trunk_s8_abs_change"] > 0.0
    for a in ("s8_zeros", "s8_detached", "frozen_trunk"):
        assert res[a]["trunk_s8_abs_change"] == 0.0, a
    assert res["lane_w0"]["w_loss"][2] == 0.0 and res["lane_w0"]["w_decision"][2] == 1.0
    for a in O.ARMS:
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

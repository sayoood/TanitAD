"""refcv9 M-GATE -- ``map_hires_overfit.py --banded``: the banded map head's OWN overfit record (schema /2-banded).

Mechanics on CPU (the real run needs the 16 TRAIN frames and a GPU):

* OFF is the tip's harness, bit for bit: every top-level definition of the tip's ``map_hires_overfit.py`` has the SAME AST digest,
  and ``main`` is the tip's once its two additions (the ``--banded`` argument, the ``if a.banded`` dispatch) are removed;
* the channel <-> inherited-class table is the ONLY new mapping, and every bar it reaches is the A18 spec's literal;
* R1 is the lane TERM weight 0 in the LOSS: an analytic identity on the trainer's own telemetry and a gradient proof;
* the gated slices: near all of 0-10 m, mid only its 10-20 m rows, far none (INFORMATIVE);
* the verdict is reachable in every direction, and a control that does not reproduce fails it.
"""
from __future__ import annotations

import ast
import copy
import hashlib
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
from tanitad.models import map_banded as MB                     # noqa: E402
from tanitad.models import map_head_hires as H                  # noqa: E402
from tanitad.models import timm_trunk as TT                     # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width         # noqa: E402

A18 = (ROOT.parent / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-09-26-refcv7-map-hires" /
       "raw" / "gmo_spec_A18.json")
A18_SHA = "5cb4f6fc4a8c15660237f541f077bfb334337dd287f674f8de02b7d414077c58"
#: the tip's (535d981) top-level definitions of map_hires_overfit.py: (name-or-kind, sha256(ast.dump)[:20]), in file order
TIP_DIGESTS = [
    ('Expr', '1ecd3c13601182459a50'),
    ('ImportFrom', '890bede517510aea64e3'),
    ('Import', '356e6775028bc09e699d'),
    ('Import', '33e554a09588607f47c3'),
    ('Import', '092cb77ae0770f1627b2'),
    ('Import', '211ee42757705c7e350e'),
    ('Import', 'f329650e2c61a07a1bab'),
    ('Import', 'f7252e3c1dbe1d5edf1f'),
    ('Import', 'ff05bad0fd9a7212d3cd'),
    ('ImportFrom', '59051b17634cdcb5a576'),
    ('Import', '4d5e05f5856c1a5a1012'),
    ('HERE', '9b47380b6ea02e93182e'),
    ('Expr', '4159451e5200b30cd2c1'),
    ('ImportFrom', 'e3229d6d1d5f55d72b36'),
    ('ImportFrom', '123b1b82861adcab68b9'),
    ('ImportFrom', '02f3e26812342787d93c'),
    ('ARMS', 'f4a6bf32d5d8d05db703'),
    ('INFORMATIVE_KNOWN', '409265aa780364761c19'),
    ('LANE', '0c7034bb1d1db3f066ae'),
    ('DRIVABLE', '62c4f2f9b35cc575cc05'),
    ('EDGE', '7083cc03e82fb12b9176'),
    ('THIN', 'fbd633543306a742d9a2'),
    ('NOT_RUN_WHY', '4239a71cb670f511b456'),
    ('_expand_frames', '8cfb1886e8f024d21429'),
    ('_placeholder', '064c134d36a7f09a6fdc'),
    ('load_spec', 'b269150da7967b1d90e0'),
    ('LR_DECAY_KINDS', 'd7267db4381e38fd1c26'),
    ('lr_multiplier', '4b89507a3cccfc892aaf'),
    ('class_weights_of', 'fbb887eaa31de3e52899'),
    ('load_frames', 'c3f204a85819691792a9'),
    ('_axis_guard', 'a22c5409d3c1d28735ff'),
    ('_band_slice', 'afb155f5fa4aeb369154'),
    ('summarise', 'dee4e2e1ff5c1611ffd8'),
    ('_signal_totals', '4568af7baab9b20cbbc7'),
    ('_s8_params', '09e58e1307e5da5ea045'),
    ('_forward', '2984eb16153631c83030'),
    ('evaluate', '3fb9108c7044a27e26ed'),
    ('run_arm', 'f747608db5bc25b34c0a'),
    ('controls', 'e27ffcb880960768d1b3'),
    ('_passes_bars', '1134fadf95687055d6bd'),
    ('verdict', '9d97170e3ef510f29ac3'),
    ('_fingerprint', '5f7496a752fc81f46376'),
    ('_fingerprint_without', '56d5414e7e165408da0d'),
    ('main', 'e98acc1bb7a25f8a2cfe'),
    ('If', 'f0ea0be627ebd2887d7f'),
]
BANDED_NEW = {"BANDED_SCHEMA", "BANDED_CHANNEL_CLASS", "BANDED_CHANNELS", "BANDED_THIN", "BANDED_MUST_FAIL", "BANDED_MUST_FAIL_ALL",
              "BANDED_GATED_BANDS", "BANDED_DEFAULT_ARMS", "BANDED_INFORMATIVE_ARMS", "BANDED_INFORMATIVE_CHANNELS",
              "banded_gate_rows", "banded_hi_m", "_banded_forward", "_band_targets", "banded_loss", "_new_acc", "_acc_add",
              "_acc_summary", "evaluate_banded", "run_arm_banded", "controls_banded", "banded_bar", "banded_clears",
              "verdict_banded", "main_banded"}


def _key(n):
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        return n.name
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
        return n.targets[0].id
    return type(n).__name__


def _is_banded_addition(n):
    """The two statements ``main`` gained: ``ap.add_argument("--banded", ...)`` and ``if a.banded: return main_banded(a)``."""
    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and n.value.args \
            and isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value == "--banded":
        return True
    return isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(ast.parse("a.banded").body[0].value)


def test_OFF_is_the_tips_harness_bit_for_bit():
    src = (ROOT / "scripts" / "map_hires_overfit.py").read_text(encoding="utf-8")
    body = [n for n in ast.parse(src).body if _key(n) not in BANDED_NEW]
    got = []
    for n in body:
        if isinstance(n, ast.FunctionDef) and n.name == "main":
            n = copy.deepcopy(n)
            n.body = [x for x in n.body if not _is_banded_addition(x)]
        got.append((_key(n), hashlib.sha256(ast.dump(n).encode()).hexdigest()[:20]))
    assert [tuple(x) for x in TIP_DIGESTS] == got                 # includes `main` (minus the two additions) and the order
    main = next(n for n in ast.parse(src).body if _key(n) == "main")
    assert sum(_is_banded_addition(x) for x in main.body) == 2   # the pin is not satisfied by deleting the flag


def test_OFF_imports_nothing_new_at_module_level():
    tree = ast.parse((ROOT / "scripts" / "map_hires_overfit.py").read_text(encoding="utf-8"))
    names = " ".join(ast.dump(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    assert "map_banded" not in names                              # the /1 closures' eager-import set is unchanged


def test_the_channel_class_table_reaches_the_A18_bars_and_the_codes_of_the_design():
    spec = O.load_spec(A18, class_weights="uniform", decision_rule="prior_corrected")
    assert hashlib.sha256(A18.read_bytes()).hexdigest() == A18_SHA
    assert set(O.BANDED_CHANNEL_CLASS) == set(MB.PROB_CHANNELS)   # every probability channel, none invented
    assert set(O.BANDED_CHANNEL_CLASS.values()) <= set(spec["thresholds"]["iou"])
    assert {ch: O.banded_bar(spec, ch) for ch in O.BANDED_CHANNELS} == {
        "drv": 0.85, "sdw": 0.85, "lane": 0.5, "edge_h": 0.5, "xwalk": 0.5, "arrow": 0.5, "hatch": 0.5}
    assert MB.DRV_CODES == (1, 2, 3, 4, 6) and MB.C_SIDEWALK == 7 and MB.C_LANE == 2 and MB.C_EDGE == 5
    assert [H.CLASS_KEYS[i] for i in (MB.C_DRIVABLE, MB.C_SIDEWALK, MB.C_LANE, MB.C_EDGE, MB.C_XWALK, MB.C_ARROW, MB.C_HATCH)] == \
        ["drivable", "sidewalk", "lane", "edge", "crosswalk", "arrow", "hatched"]
    assert O.BANDED_THIN == ("lane", "xwalk", "arrow", "edge_h", "hatch")
    assert O.BANDED_MUST_FAIL == {"lane_w0": ("lane",), "s8_zeros": O.BANDED_THIN} and O.BANDED_MUST_FAIL_ALL == ("s8_zeros",)


def test_gated_slices_near_all_mid_only_10_to_20_m_far_none():
    geom = MB.band_geometry(100.0, 30.0)
    hi = O.banded_hi_m("0_20")
    assert hi == 20.0 and O.banded_hi_m("all") == float("inf")
    assert [O.banded_gate_rows(g, hi) for g in geom] == [100, 40, 0]
    assert [O.banded_gate_rows(g, float("inf")) for g in geom] == [100, 120, 120]
    with pytest.raises(SystemExit):
        O.banded_hi_m("0_999")


# ---------------------------------------------------------------------------------------------- the toy rig ==
def _tiny():
    torch.manual_seed(0)
    trunk = TT.TimmResNetTrunk(TT.TimmTrunkConfig(model_name="resnet18.a1_in1k", pretrained=False, verify_imagenet_stats=False,
                                                  frames=3, image_hw=(64, 128), frozen_bn=True))
    trunk.enable_s8_tap()
    cfg = H.MapHiresConfig(w_map_hires=1.0, d_lift=16, d_model=16, d_up=16, dilations=(1,), near_lift_x_m=10.0, banded=True)
    branch = H.MapHiresBranch(cfg, d_image=trunk.s8_dim, image_hw=trunk.s8_shape)
    frame = frame_for_width(128, 64)
    g = L.build_lift_geometry(RigCamera.nominal(frame, height_m=1.5, x_m=1.5), frame=frame, stride=8, grid=cfg.lift_grid)
    codes = torch.full((2, 600, 320), 1, dtype=torch.uint8)
    codes[:, :, 160:] = 7
    codes[:, :, 100:120] = 2
    codes[:, 50:60, 140:150] = 3
    codes[:, 300:310, 140:150] = 4
    codes[:, 100:110, 60:70] = 6
    codes[:, :, 159:160] = 5
    data = {"x": torch.randint(0, 255, (2, 9, 64, 128), dtype=torch.uint8), "codes": codes,
            "grid": g.grid.unsqueeze(0).repeat(2, 1, 1, 1, 1), "valid": g.valid.unsqueeze(0).repeat(2, 1, 1, 1),
            "sha12": ["a" * 12, "b" * 12], "raw_frame": [5, 6]}
    return trunk, branch, data, cfg.extent


def _spec(**over):
    s = O.load_spec(A18, class_weights="uniform", decision_rule="prior_corrected")
    s.update(steps=3, lr=1e-2, batch=2, eval_every=3, seed=0, lr_decay=None)
    s.update(over)
    return s


def test_the_arms_are_what_they_say_on_the_banded_head():
    trunk, branch, data, ext = _tiny()
    geom, hi, spec = MB.band_geometry(ext.x_max_m, ext.y_half_m), 20.0, _spec()
    res = {a: O.run_arm_banded(a, trunk, branch, data, spec, "cpu", ext, geom, hi) for a in ("healthy", "lane_w0", "s8_zeros")}
    assert res["healthy"]["trunk_s8_abs_change"] > 0.0 and res["lane_w0"]["trunk_s8_abs_change"] > 0.0
    assert res["s8_zeros"]["trunk_s8_abs_change"] == 0.0           # R2: the trunk gets no map gradient
    assert res["lane_w0"]["zero_channels"] == ["lane"] and res["healthy"]["zero_channels"] == []
    for r in res.values():
        assert r["loss_finite_every_step"] and set(r["final"]["cells"]) == set(O.BANDED_CHANNELS)
    assert all(p.grad is None for p in branch.parameters())        # the shared init is never trained in place
    ctrl = O.controls_banded(data, res["healthy"].pop("_kept"), ext, geom, hi, "cpu")
    assert ctrl["C1_constant_drv"]["reproduced"] and ctrl["C2_gt_as_logits"]["reproduced"], ctrl
    d = ctrl["C1_constant_drv"]["cells"]["drv"]["near"]["gate"]
    assert d["iou"] == pytest.approx(d["n_pos"] / d["n_scored"], abs=1e-12)
    assert ctrl["C1_constant_drv"]["cells"]["sdw"]["near"]["gate"]["iou"] in (0.0, None)
    assert ctrl["C3_rule_identity_w_ones"]["reproduced"] is None   # no analogue; recorded, not faked


def test_R1_is_the_lane_term_weight_zero_an_analytic_identity_on_the_trainers_own_telemetry():
    trunk, branch, data, ext = _tiny()
    lg, un = O._banded_forward(trunk, branch, data, torch.arange(2), "cpu", "healthy")
    tg = O._band_targets(data, torch.arange(2), "cpu", ext)
    full, row = O.banded_loss(lg, un, tg)
    r1, _ = O.banded_loss(lg, un, tg, zero_channels=("lane",))
    assert torch.equal(full, MB.banded_map_loss({"logits": lg, "unobs": un}, tg, with_metrics=False)["loss"])  # () = the trainer's
    lane = sum(MB.BAND_WEIGHTS[b] * float(row["map_banded_loss_lane_" + bn]) for b, bn in enumerate(MB.BAND_NAMES))
    assert float(full) - float(r1) == pytest.approx(lane, rel=1e-4, abs=1e-5) and lane > 0.0
    rest = sum(MB.BAND_WEIGHTS[b] * (sum(float(row["map_banded_loss_%s_%s" % (c, bn)]) for c in MB.CHANNELS if c != "lane")
                                     + MB.MAXENT_W * float(row["map_banded_maxent_" + bn])) for b, bn in enumerate(MB.BAND_NAMES))
    assert float(r1) == pytest.approx(rest, rel=1e-4, abs=1e-5)    # it is every OTHER term, exactly


def test_R1_the_lane_channel_receives_no_supervision_gradient_and_the_others_do():
    """With no unobserved cell the max-entropy term is exactly 0, so the lane row of the head gets ~0 gradient under R1 and a clearly
    non-zero one under the full loss: the mutation check that the arm is a real change."""
    trunk, branch, data, ext = _tiny()
    data = dict(data, valid=torch.ones_like(data["valid"]))
    ch, sdw = MB.CHANNELS.index("lane"), MB.CHANNELS.index("sdw")
    grads = {}
    for name, zero in (("full", ()), ("r1", ("lane",))):
        b = copy.deepcopy(branch)
        lg, un = O._banded_forward(trunk, b, data, torch.arange(2), "cpu", "healthy")
        assert not any(bool(u.any()) for u in un)
        loss, _ = O.banded_loss(lg, un, O._band_targets(data, torch.arange(2), "cpu", ext), zero_channels=zero)
        loss.backward()
        w = b.banded.head_near.weight.grad
        grads[name] = (float(w[ch].abs().sum()), float(w[sdw].abs().sum()))
    assert grads["full"][0] > 1e-3 and grads["full"][1] > 1e-3
    assert grads["r1"][0] <= 1e-6 * grads["full"][0]                # the lane row: cancelled (float error only)
    assert grads["r1"][1] == pytest.approx(grads["full"][1], rel=1e-4)      # a neighbour channel: untouched


# ---------------------------------------------------------------------------------------------- the verdict ==
def _cells(iou_by_ch, n=5000.0):
    cells = {}
    for ch in O.BANDED_CHANNELS:
        cells[ch] = {}
        for g in MB.band_geometry(100.0, 30.0):
            v = iou_by_ch(ch, g.name)
            d = {"gated": O.banded_gate_rows(g, 20.0) > 0}
            for key in ("gate", "band"):
                d[key] = {"iou": v, "n_pos": int(n), "n_scored": int(n * 4)}
            cells[ch][g.name] = d
    return cells


def _final(iou, l0=2.0, l1=0.5, finite=True):
    terms = {f"{c}_{b}": l0 for c in MB.CHANNELS for b in MB.BAND_NAMES}
    return {"step0": {"loss": dict(terms)}, "final": {"cells": _cells(iou), "loss": {k: l1 for k in terms}},
            "loss_finite_every_step": finite}


def _good(ch, band):
    return 0.95


def _ctrl(ok=True):
    return {"C1_constant_drv": {"cells": _cells(lambda ch, b: 0.4 if ch == "drv" else 0.0), "reproduced": ok},
            "C2_gt_as_logits": {"cells": _cells(lambda ch, b: 1.0), "reproduced": True},
            "C3_rule_identity_w_ones": {"reproduced": None}}


GUARD = {"c": {"kind": "time_1ms"}}


def _results(**over):
    r = {"healthy": _final(_good), "lane_w0": _final(lambda ch, b: 0.0 if ch == "lane" else 0.95),
         "s8_zeros": _final(lambda ch, b: 0.0 if ch in O.BANDED_THIN else 0.9)}
    r.update(over)
    return r


def _verdict(results=None, ctrl=None, guard=GUARD):
    spec = O.load_spec(A18, class_weights="uniform", decision_rule="prior_corrected")
    return O.verdict_banded(results or _results(), spec, ctrl or _ctrl(), guard, MB.band_geometry(100.0, 30.0), 20.0)


def test_the_verdict_is_reachable_in_every_direction():
    v = _verdict()
    assert v["G_MAP_OVERFIT_BANDED"] == "PASS", v
    assert v["MAIN"]["verdict"] == "PASS" and all(r["failed_as_required"] for r in v["regression_arms"].values())
    assert set(v["MAIN"]["bars"]) == {f"{c}_{b}" for c in O.BANDED_CHANNELS for b in O.BANDED_GATED_BANDS}   # far is NOT gated
    assert _verdict(_results(lane_w0=_final(_good)))["G_MAP_OVERFIT_BANDED"] == "FAIL"        # R1 did not fail lane
    one_passes = _final(lambda ch, b: 0.9 if ch == "arrow" else 0.0)
    assert _verdict(_results(s8_zeros=one_passes))["G_MAP_OVERFIT_BANDED"] == "FAIL"           # R2 must fail EVERY thin channel
    low_mid_lane = _final(lambda ch, b: 0.4 if (ch, b) == ("lane", "mid") else 0.95)
    assert _verdict(_results(healthy=low_mid_lane))["MAIN"]["verdict"] == "FAIL"               # a gated cell under its bar
    far_bad = _final(lambda ch, b: 0.0 if b == "far" else 0.95)
    assert _verdict(_results(healthy=far_bad))["G_MAP_OVERFIT_BANDED"] == "PASS"               # far is informative
    assert _verdict(_results(healthy=_final(_good, l1=1.5)))["MAIN"]["verdict"] == "FAIL"      # loss did not fall to <= 0.5x
    assert _verdict(_results(healthy=_final(_good, finite=False)))["MAIN"]["verdict"] == "FAIL"
    assert _verdict(ctrl=_ctrl(ok=False))["G_MAP_OVERFIT_BANDED"] == "FAIL"                    # a control that does not reproduce
    assert _verdict(guard={"c": {"kind": "pose_content_5cm"}})["G_MAP_OVERFIT_BANDED"] == "FAIL"   # no 1 ms guard
    r = _results()
    r.pop("s8_zeros")                                              # an arm that did not run is NOT excused (no A19 here)
    v = _verdict(r)
    assert v["regression_arms"]["s8_zeros"]["ran"] is False and v["G_MAP_OVERFIT_BANDED"] == "FAIL"


def test_presence_is_the_inherited_floor_in_area_inconclusive_is_fail():
    res = _results()
    assert _verdict(res)["MAIN"]["min_area_m2"] == pytest.approx(10.0)       # 1,000 cells x 0.1 m x 0.1 m
    res["healthy"]["final"]["cells"]["arrow"]["mid"]["gate"]["n_pos"] = 100   # 100 x 0.0625 m^2 = 6.25 m^2 < 10
    v = _verdict(res)
    assert v["MAIN"]["inconclusive"] and "arrow_mid" in v["MAIN"]["presence_short"] and v["G_MAP_OVERFIT_BANDED"] == "FAIL"
    res["healthy"]["final"]["cells"]["arrow"]["mid"]["gate"]["n_pos"] = 160   # 160 x 0.0625 = 10 m^2: the floor, met
    assert _verdict(res)["MAIN"]["verdict"] == "PASS"


def test_the_flag_is_wired_and_refuses_the_wrong_near_lift_and_a_missing_gated_arm(tmp_path):
    src = (ROOT / "scripts" / "map_hires_overfit.py").read_text(encoding="utf-8")
    assert "main_banded(a)" in src and '"--banded"' in src
    base = ["--banded", "--spec", str(A18), "--v2-cache", str(tmp_path), "--gt-root", str(tmp_path),
            "--extrinsics", str(tmp_path / "e.json"), "--out", str(tmp_path / "o"), "--decision-rule", "prior_corrected",
            "--near-refine-blocks", "1"]
    with pytest.raises(SystemExit, match="near-lift-m 10"):
        O.main(base + ["--near-lift-m", "20", "--arms", "healthy,lane_w0,s8_zeros"])
    with pytest.raises(SystemExit, match="gated arm"):
        O.main(base + ["--near-lift-m", "10", "--arms", "healthy"])

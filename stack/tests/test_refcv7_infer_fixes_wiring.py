"""refcv7 diagnostics F1 / F2 / F4 (2026-10-04) -- the TRAINER wiring of the three OPT-IN inference-time fixes.

Source of the fixes: ``TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-map-box-diagnostics``.
The functions themselves are pinned with literals in ``test_refcv7_map_class_threshold.py`` and
``test_refcv7_det_presence_gates.py``; this file pins that the trainer REACHES them, and that a run launched with the
OLD argv is exactly what it was:

1. ``--map-hires-decision-rule`` now accepts ``class_threshold`` (default still ``prior_corrected``) and
   ``--map-hires-class-thresholds`` / ``--det-presence-gates`` exist and default to ``None``.
2. every refusal, each with a GREEN control: ``class_threshold`` with no file; a thresholds file with ``--map-hires off``
   (READ BY NOTHING, M18); a file that does not exist; gates without VIS-1.
3. THROUGH THE REAL ``train()`` (the launch gate's G-EVAL capture, ``launch_gate.run_trainer_until``): the opt-in argv builds
   a model whose ``state_dict`` is BIT-IDENTICAL to the default argv's (the flags add no parameter and consume no RNG),
   carries the thresholds on ``model._map_hires_class_thresholds``, and a file fitted under OTHER class weights is REFUSED
   at build time; the EVAL LOADER (``refcv7_loader``) rebuilds the same model with the thresholds and G-DVB clean.
4. the trainer's own ``_map_hires_loss`` / ``_eval_row_from_acc``: with thresholds the row ADDS the thresholded and 2-cell
   tolerant keys and changes no existing one; without them there is none; a declared ``class_threshold`` rule with no
   thresholds REFUSES at the first metrics row.
5. the F4 call site in ``train()`` (a SOURCE pin: the in-run eval loop cannot be executed in a unit test).
"""
from __future__ import annotations

import dataclasses as dc
import hashlib
import json
from pathlib import Path

import pytest
import torch

from test_map_hires_wiring import (ON, T, _attach, _eval_row, _extras, _model,         # noqa: E402,F401
                                   _pin)
from test_refcv7_eval_loader import _record, rig                                       # noqa: E402,F401  (the real-train() rig)
import launch_gate as LG                                                              # noqa: E402  (scripts/ is on sys.path by now)
from tanitad.models import map_head_hires as H                                         # noqa: E402

STACK = Path(__file__).resolve().parents[1]
CFG = STACK / "tanitad" / "configs" / "refcv7_map_hires_class_thresholds_train.json"
GATES = STACK / "tanitad" / "configs" / "refcv7_det_presence_gates_train.json"
TAU_LOGIT = [-0.41000000000000014, -0.5199999999999996, -1.8999999999999986, -2.129999999999999,
             -2.8599999999999994, -3.1400000000000006, -3.09, -0.8099999999999987]
#: the rig's synthetic class weights (test_refcv7_eval_loader._synth_inputs)
RIG_WEIGHTS = [1.0, 1.5, 3.0, 2.0, 4.0, 1.2, 2.5, 1.1]
CLASSES = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")


# =========================================================================== #
# 1. the flags                                                                 #
# =========================================================================== #
def _action(parser, flag):
    return [a for a in parser._actions if flag in a.option_strings][0]


def test_the_flags_exist_and_default_to_the_old_behaviour():
    p = T.build_parser()
    a = _action(p, "--map-hires-decision-rule")
    assert tuple(a.choices) == ("prior_corrected", "raw", "class_threshold") and a.default == "prior_corrected"
    assert _action(p, "--map-hires-class-thresholds").default is None
    assert _action(p, "--det-presence-gates").default is None
    ns = p.parse_args(ON)
    assert (ns.map_hires_decision_rule, ns.map_hires_class_thresholds, ns.det_presence_gates) == (
        "prior_corrected", None, None)
    for flag in ("--map-hires-class-thresholds", "--det-presence-gates"):
        assert "%" not in (_action(p, flag).help or "").replace("%%", "")                   # argparse help hygiene


# =========================================================================== #
# 2. the pins                                                                  #
# =========================================================================== #
def test_GREEN_the_opt_in_combinations_and_the_default_do_not_refuse():
    _pin(ON)                                                                  # the old argv: unchanged
    _pin(ON + ["--map-hires-class-thresholds", str(CFG)])                     # monitor-only: rule untouched
    _pin(ON + ["--map-hires-decision-rule", "class_threshold", "--map-hires-class-thresholds", str(CFG)])


@pytest.mark.parametrize("argv,needle", [
    (ON + ["--map-hires-decision-rule", "class_threshold"], "without --map-hires-class-thresholds"),
    (ON + ["--map-hires-class-thresholds", "no_such_thresholds.json"], "does not exist"),
    (["--arm", "hier", "--out", "X", "--map-hires-class-thresholds", str(CFG)], "READ BY NOTHING"),
    (["--arm", "hier", "--out", "X", "--map-hires-decision-rule", "class_threshold"], "READ BY NOTHING"),
])
def test_every_dead_or_unreadable_thresholds_combination_refuses(argv, needle):
    with pytest.raises(SystemExit) as e:
        _pin(argv)
    assert needle in str(e.value), str(e.value)[:300]


def test_a_malformed_thresholds_file_refuses_at_the_pin(tmp_path):
    d = json.loads(CFG.read_text(encoding="utf-8"))
    d["schema"] = "other/1"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        _pin(ON + ["--map-hires-class-thresholds", str(p)])
    assert "schema" in str(e.value)


def test_the_trainer_helper_reads_nothing_by_default_and_binds_the_file_to_the_weights():
    ns = T.build_parser().parse_args(ON)
    assert T._map_hires_class_thresholds(ns, torch.ones(8)) == (None, None)             # a default run reads nothing
    ns = T.build_parser().parse_args(ON + ["--map-hires-class-thresholds", str(CFG)])
    with pytest.raises(SystemExit) as e:                                                  # unit weights != the fit's
        T._map_hires_class_thresholds(ns, torch.ones(8))
    assert "do not transfer" in str(e.value) and CFG.name in str(e.value)               # the file is NAMED
    w = torch.tensor(json.loads(CFG.read_text(encoding="utf-8"))["class_weight_values"], dtype=torch.float32)
    th, st = T._map_hires_class_thresholds(ns, w)
    assert th.tolist() == pytest.approx(TAU_LOGIT, abs=1e-6)
    assert st["sha256"] == hashlib.sha256(CFG.read_bytes()).hexdigest()


def test_det_gates_without_vis1_refuse_and_with_it_pass_the_pin(rig, tmp_path):
    a = LG.set_flag(rig.argv, "--trunk-compile", None)
    # the rig's argv has --slot-vis1; strip it (and the sidecar flag the pin chain also reads) -> READ BY NOTHING
    no_vis = LG.set_flag(LG.set_flag(a, "--slot-vis1", None), "--vis1-sidecar", None)
    with pytest.raises(SystemExit) as e:
        LG.run_trainer_until(rig.T, LG.set_flag(no_vis, "--det-presence-gates", [str(GATES)]), "config")
    assert "without --slot-vis1" in str(e.value)
    with pytest.raises(SystemExit) as e:
        LG.run_trainer_until(rig.T, LG.set_flag(a, "--det-presence-gates", [str(tmp_path / "nope.json")]), "config")
    assert "does not exist" in str(e.value)
    cap = LG.run_trainer_until(rig.T, LG.set_flag(a, "--det-presence-gates", [str(GATES)]), "config")
    assert cap["cfg"] is not None


# =========================================================================== #
# 3. through the REAL train() and the eval loader                              #
# =========================================================================== #
def _opt_in_argv(rig, tau_path, *, rule="class_threshold", drop_compile=True):
    a = LG.set_flag(rig.argv, "--trunk-compile", None) if drop_compile else list(rig.argv)
    a = LG.set_flag(a, "--map-hires-decision-rule", [rule])
    a = LG.set_flag(a, "--map-hires-class-thresholds", [str(tau_path)])
    return LG.set_flag(a, "--det-presence-gates", [str(GATES)])


def _rig_thresholds(tmp_path):
    """The shipped thresholds with `class_weight_values` set to the RIG's weights (so the binding passes)."""
    d = json.loads(CFG.read_text(encoding="utf-8"))
    d["class_weight_values"] = RIG_WEIGHTS
    p = tmp_path / "rig_thresholds.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_the_default_reference_model_carries_no_thresholds(rig):
    assert rig.args_t.map_hires_class_thresholds is None and rig.args_t.det_presence_gates is None
    assert rig.model_t._map_hires_class_thresholds is None
    assert rig.model_t._map_hires_class_thresholds_stamp is None
    assert rig.model_t._map_hires.cfg.decision_rule == "prior_corrected"


def test_the_opt_in_argv_builds_the_SAME_model_plus_the_thresholds_through_the_real_train(rig, tmp_path):
    tau = _rig_thresholds(tmp_path)
    cap = LG.run_trainer_until(rig.T, _opt_in_argv(rig, tau), "model")
    m = cap["model"]
    assert m._map_hires.cfg.decision_rule == "class_threshold"
    assert m._map_hires_class_thresholds.tolist() == pytest.approx(TAU_LOGIT, abs=1e-6)
    assert m._map_hires_class_thresholds_stamp["sha256"] == hashlib.sha256(tau.read_bytes()).hexdigest()
    assert cap["args"].det_presence_gates == str(GATES)
    ref = rig.model_t.state_dict()
    got = m.state_dict()
    assert list(ref) == list(got)                                       # the flags add no parameter / buffer
    bad = [k for k in ref if torch.is_tensor(ref[k]) and not torch.equal(ref[k], got[k])]
    assert not bad, bad[:5]                                             # ... and consume no RNG: bit-identical init


def test_through_the_real_train_thresholds_fitted_under_other_weights_are_REFUSED(rig):
    with pytest.raises(SystemExit) as e:                                # the shipped file vs the rig's weights
        LG.run_trainer_until(rig.T, _opt_in_argv(rig, CFG), "model")
    assert "do not transfer" in str(e.value)
    a = LG.set_flag(LG.set_flag(rig.argv, "--trunk-compile", None), "--map-hires-decision-rule", ["class_threshold"])
    with pytest.raises(SystemExit) as e:                                # the rule without its file
        LG.run_trainer_until(rig.T, a, "model")
    assert "without --map-hires-class-thresholds" in str(e.value)


def test_the_eval_loader_rebuilds_the_opt_in_model_strictly_with_gdvb_clean(rig, tmp_path):
    tau = _rig_thresholds(tmp_path)
    argv = _opt_in_argv(rig, tau, drop_compile=False)
    model, cfg, args, rec = rig.L.build_model(_record(rig.T, argv), str(rig.ck), device="cpu", strict=True,
                                              remap={})
    assert rec["declared_vs_built"]["mismatches"] == []
    assert model._map_hires.cfg.decision_rule == "class_threshold"
    assert model._map_hires_class_thresholds.tolist() == pytest.approx(TAU_LOGIT, abs=1e-6)
    assert args.map_hires_class_thresholds == str(tau)
    # the default loader path is untouched: the default argv still builds with NO thresholds
    m0, _c, a0, r0 = rig.L.build_model(_record(rig.T, rig.argv), str(rig.ck), device="cpu", strict=True, remap={})
    assert m0._map_hires_class_thresholds is None and r0["declared_vs_built"]["mismatches"] == []


def test_gdvb_reads_the_class_threshold_rule_off_the_built_config():
    from test_map_head_hires import _dvb_args, _dvb_model
    m = _dvb_model()
    m._map_hires.cfg = dc.replace(m._map_hires.cfg, decision_rule="class_threshold")
    assert H.dvb_check_decision_rule(m, _dvb_args(map_hires_decision_rule="class_threshold")) == []
    got = H.dvb_check_decision_rule(m, _dvb_args(map_hires_decision_rule="prior_corrected"))
    assert [x.read_from for x in got] == ["model._map_hires.cfg.decision_rule"]


# =========================================================================== #
# 4. the trainer's own loss row and eval row                                   #
# =========================================================================== #
THR_STATS = ("interthr", "unionthr", "predthr", "tppthr2", "tpgthr2")


def _thr():
    return H.load_class_thresholds(CFG)[0]


def test_the_trainers_row_ADDS_the_thresholded_and_tolerant_keys_and_changes_no_existing_one():
    model, _ = _model()
    _attach(model)
    model.eval()
    base = _extras(model, 1)
    assert not [k for k in base if any(("_%s_" % s) in k for s in THR_STATS)]            # default: none
    model._map_hires_class_thresholds = _thr()
    with_thr = _extras(model, 1)
    new = sorted(set(with_thr) - set(base))
    assert set(base) <= set(with_thr) and len(new) == 5 * 8 * 3                         # 5 stats x 8 classes x 3 bands
    assert all(k.split("_")[2] in THR_STATS for k in new)
    for k, v in base.items():
        assert (torch.equal(v, with_thr[k]) if torch.is_tensor(v) else v == with_thr[k]), k


def test_the_eval_row_carries_iouthr_and_iou2thr_for_all_8_classes_x_bands_pooled():
    model, _ = _model()
    _attach(model)
    model.eval()
    erow0, _acc0 = _eval_row(model)
    assert not [k for k in erow0 if "iouthr" in k or "iou2thr" in k]
    model._map_hires_class_thresholds = _thr()
    erow, acc = _eval_row(model)
    bands = ("0_20", "20_40", "40_60")
    for c in CLASSES:
        for b in bands:
            assert "eval_map_hires_iouthr_%s_%s" % (c, b) in erow and "eval_map_hires_iou2thr_%s_%s" % (c, b) in erow
    assert len([k for k in erow if "_iouthr_" in k or "_iou2thr_" in k]) == 48
    i, u = acc["map_hires_interthr_lane_0_20"], acc["map_hires_unionthr_lane_0_20"]
    assert erow["eval_map_hires_iouthr_lane_0_20"] == pytest.approx(i / u, rel=1e-12)  # pooled sum/sum, as the declared one
    for k, v in erow0.items():                                                            # every old key: same value
        assert erow[k] == v, k


def test_the_train_row_carries_the_thresholded_keys_exact_and_a_default_train_row_does_not():
    model, _ = _model()
    _attach(model)
    model.eval()
    losses = _extras(model, 1)
    losses["loss"] = losses["map_hires"]                                    # a scalar the row builder keeps
    base = T._train_row_scalars(losses, model)
    assert not [k for k in base if "iouthr" in k or "iou2thr" in k or "thr_" in k]
    model._map_hires_class_thresholds = _thr()
    losses2 = _extras(model, 1)
    losses2["loss"] = losses2["map_hires"]
    row = T._train_row_scalars(losses2, model)
    assert len([k for k in row if k.startswith("map_hires_iouthr_") or k.startswith("map_hires_iou2thr_")]) == 48
    for k, v in base.items():
        assert row[k] == v, k                                              # no existing train-row key moves
    # the raw counts are kept EXACT (not rounded to 5 dp): a 1-in-1e6 class would otherwise read 0.0
    assert row["map_hires_tppthr2_lane_0_20"] == float(losses2["map_hires_tppthr2_lane_0_20"])


def test_a_declared_class_threshold_rule_scores_inter_union_on_the_thresholded_masks_and_refuses_without_them():
    model, _ = _model()
    _attach(model)
    model.eval()
    model._map_hires.cfg = dc.replace(model._map_hires.cfg, decision_rule="class_threshold")
    with pytest.raises(ValueError, match="needs class_thresholds"):
        _extras(model, 1)
    model._map_hires_class_thresholds = _thr()
    e = _extras(model, 1)
    for c in CLASSES:
        for b in ("0_20", "20_40", "40_60"):
            assert e["map_hires_inter_%s_%s" % (c, b)] == e["map_hires_interthr_%s_%s" % (c, b)]
            assert e["map_hires_union_%s_%s" % (c, b)] == e["map_hires_unionthr_%s_%s" % (c, b)]


# =========================================================================== #
# 5. the F4 call sites (SOURCE pins: the in-run eval loop cannot run in a unit test)  #
# =========================================================================== #
def _f4_pin(src: str) -> None:
    a = src.find("_det_metrics.summarise(_pk, _hd)")                   # the DECLARED summary call, unchanged
    b = src.find("_det_metrics.gated_census_keys(_pk, _hd, det_gates)")
    assert a > 0 and b > 0, (a, b)
    assert 0 < b - a < 700                                            # the new census follows it, same loop
    assert "det_gates, _dg_stamp = _det_metrics.load_head_gates(args.det_presence_gates)" in src
    assert 'calib_stamp["presence_gates"] = _dg_stamp' in src
    assert "det_gates = None" in src                                  # a default run: gates None -> gated keys `{}`
    # the train-row census and the planner path never see the gates
    assert "train_row_keys(_pk_b3, \"box3d\", " not in src and "train_row_keys(_pk_ag, \"agent\", " not in src


def _trainer_src() -> str:
    return (STACK / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8").replace("\r\n", "\n")


def test_F4_the_eval_loop_reads_the_gates_and_calls_the_gated_census_after_the_declared_summary():
    _f4_pin(_trainer_src())


def test_F4_DELIBERATE_REGRESSION_gating_the_DECLARED_summary_or_the_train_census_goes_RED():
    """If the eval loop passed the gate to the DECLARED summary (instead of the new gated keys), the declared
    `eval_<head>_conf_ratio` would silently change meaning; if the train-row census took it, the planner-side logging
    would. The source pin must catch both edits."""
    src = _trainer_src()
    for old, new in (("_det_metrics.summarise(_pk, _hd)", "_det_metrics.summarise(_pk, _hd, gate=det_gates[_hd])"),
                     ("_det_metrics.train_row_keys(_pk_b3, \"box3d\")",
                      "_det_metrics.train_row_keys(_pk_b3, \"box3d\", gate=0.25)")):
        assert old in src
        with pytest.raises(AssertionError):
            _f4_pin(src.replace(old, new))

"""PREREG_V8F_ROLLHIST label lever RL, the trainer wiring: argv -> namespace -> make_dagger -> load_dagger_bank (label_bank at load) -> DaggerBank.attach_recovery -> DaggerSwap.apply -> swap_rows (gt, s_h) -> model.loss.
Run the WHOLE file. Reuses the pack / helpers of test_v8f_dagger_train.py (the synthetic trainer world)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parents[1] / "stack", HERE.parents[1] / "taniteval", HERE.parents[1] / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from test_v8f_dagger_train import pack, DAG, GATE_ARGV, TR, DG, D, M, dvb, _woken, TINY  # noqa: E402,F401
from tanitad.data import v8f_recovery as RV  # noqa: E402


def gate_parts_rl(pk, extra=()):
    """gate_parts with the trainer's clip STORE handed to make_dagger (what main does)."""
    a = TR.build_parser().parse_args(["--out", str(pk.init_out), *GATE_ARGV, *extra])
    cfg = TR.make_config(a)
    store, J, wins, clips, info = TR.build_data(a, cfg)
    info["ds_window_start"] = D.effective_window_start(cfg.hist, cfg.window_start)
    mu, sd = M.cons_stats(J)
    model = M.V8FMini(cfg, M.P10Prior.synthetic(a.seed), mu, sd)
    if a.init_ckpt:
        info["init_ckpt"] = TR.load_init_ckpt(a.init_ckpt, model, cfg, apply=False)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=a.wd)
    dag = TR.make_dagger(a, J, wins, clips, info, store)
    return a, cfg, model, opt, info, dag, J, wins, clips, store


def test_flag_exists_defaults_to_human_and_has_a_declared_vs_built_entry():
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.dagger_label == "human" == DR_default()
    assert TR.FLAG_CHECKS["dagger_label"] == "dagger:dagger_label"
    dests = {act.dest for act in TR.build_parser()._actions if act.dest != "help"}
    assert dests == set(TR.FLAG_CHECKS)
    with pytest.raises(SystemExit):
        TR.build_parser().parse_args(["--out", "x", "--dagger-label", "v2"])


def DR_default():
    return RV.DAGGER_LABELS[0]


def test_human_default_arm_is_unchanged_and_speed_arm_passes_g_dvb_with_the_label_built(pack):
    h = gate_parts_rl(pack, DAG(pack, 0.5, 3))
    assert dvb(h[:9]) == [] and h[5].bank.label_kind == "human" and h[5].built_levers()["dagger_label"] == "human" and h[5].bank.rl is None
    r = gate_parts_rl(pack, [*DAG(pack, 0.5, 3), "--dagger-label", "speed_v1"])
    assert dvb(r[:9]) == [], dvb(r[:9])
    assert r[5].bank.label_kind == "speed_v1" and r[5].built_levers()["dagger_label"] == "speed_v1"
    ctl = r[4]["dagger"]["speed_label"]
    assert ctl["n_labelled"] > 0 and ctl["n_emitted"] > 0 and ctl["label"] == "speed_v1"
    assert ctl["n_eligible_after"] == ctl["n_eligible"] and set(ctl["census"]) >= {"per_tick", "abs_v_own_minus_u_h1", "abs_delta30", "relabelled_share_of_ok"}   # the eligible index is unchanged


def test_g_dvb_refuses_a_label_declared_but_not_built_and_the_reverse_and_a_declaration_at_p_zero(pack):
    r = gate_parts_rl(pack, [*DAG(pack, 0.5, 3), "--dagger-label", "speed_v1"])
    h = gate_parts_rl(pack, DAG(pack, 0.5, 3))
    bad = dvb(r[:9], dag=h[5])                                           # declared speed_v1, swap built human
    assert any("lever dagger_label" in m and "speed_v1" in m for m in bad), bad
    bad = dvb(h[:9], dag=r[5])                                           # declared human, swap built speed_v1
    assert any("lever dagger_label" in m for m in bad), bad
    a0 = gate_parts_rl(pack, ["--dagger-label", "speed_v1"])          # p = 0: a lever that is declared and does nothing
    bad = dvb(a0[:9])
    assert any("dagger-label" in m and "declared and does nothing" in m for m in bad), bad


def test_reachability_the_speed_label_reaches_the_batch_and_the_loss_and_human_reaches_neither(pack):
    r = gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", "speed_v1"])
    h = gate_parts_rl(pack, DAG(pack, 1.0, 3))
    a, cfg, _, _, info, dag_r, J, wins, clips, store = r
    dag_h = h[5]
    ds = D.V8FMiniWindows(store, J, wins, hmax=16)
    el = [w for w in wins if dag_r.bank.candidates([w[0]], [w[1]])[0][0] > 0][:24]
    b = D.collate([ds.item(w[0], w[1], 1, 3) for w in el])
    cfg16 = M.V8FMiniConfig(**{**TINY, "hmax": 16, "wm": "b"})
    mu, sd = M.cons_stats(J)
    model = _woken(M.V8FMini(cfg16, M.P10Prior.synthetic(0), mu, sd))
    out_r, out_h = dag_r.apply(b, 1), dag_h.apply(b, 1)
    n_r = int((out_r["gt"] != b["gt"]).reshape(len(el), -1).any(1).sum())
    n_h = int((out_h["gt"] != b["gt"]).reshape(len(el), -1).any(1).sum())
    assert n_r > 0 and dag_r.last["n_label_replaced"] > 0, "speed_v1: rows whose label differs from the human label"
    assert n_h == 0 and dag_h.last["n_label_replaced"] == 0 and out_h["gt"] is b["gt"], "human: exactly 0"
    ade = lambda bb: float(model.loss(bb, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]["ade"])
    assert ade(out_r) != ade(out_h), "the swapped label reaches the loss term (ade differs between the two label kinds on the same swapped states)"
    # the launch gate's own G-LIVE probe accepts the speed_v1 arm and records the label reach
    res = TR.dagger_probe(model, b, cfg16, dag_r)
    assert res.get("rl_label_rows_changed", 0) > 0

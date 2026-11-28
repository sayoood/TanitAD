"""Literal tests for the DRAFT SPEC AMENDMENT A6 in `g0_refcv7.py` (no GPU, no data).

    PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval" python -m pytest -q test_g0_a6.py

Every expectation is a LITERAL or an INDEPENDENT derivation (the identity softplus(c) - softplus(-c) = c,
a brute-force re-evaluation of every flip, the trainer's own `tactical_behaviour_losses`). Each guard
carries a deliberate-regression arm that must go RED -- including the flip-SIGN defect this module's
author actually wrote first and caught only with the literal below.
"""
import itertools
import math
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

# the hand example (values computed once, written as literals):
#   batch 0: cells (l, c, y, w) = (0.05, 1.0, 1, 1), (2.0, 3.0, 1, 1); arm logits (-0.01, 2.02)
#   batch 1: cells (-1.0, 0.5, 0, 1), (0.5, -2.0, 0, 0 [unsupervised]); arm logits (-1.01, 0.4)
#   delta = max |l_s0 - l_arm| over SUPERVISED cells = 0.06; k = 3 -> threshold 0.18
#   batch values: (softplus(-1) + softplus(-3)) / 2 = 0.18092451954598232; softplus(-0.5) = 0.4740769841801067
#   replay mean 0.3275007518630445; the one undecidable cell (b0, c0) is CORRECT -> flipping adds
#   +c * w / sum(w) / nb = +1 * 1 / 2 / 2 = +0.25 -> hi 0.5775007518630445, lo = replay
T_REPLAY = 0.3275007518630445
HI = 0.5775007518630445


def _call(logits, conf, y, w, mask=None, val=None):
    return {"goal_logits": logits, "goal_conf": conf, "goal_y": y, "goal_w": w, "class_mask": mask,
            "tac_goal_conf_bce": val}


def _example():
    s0 = [_call([[0.05, 2.0]], [[1.0, 3.0]], [[1.0, 1.0]], [[1.0, 1.0]], val=0.18092451954598232),
          _call([[-1.0, 0.5]], [[0.5, -2.0]], [[0.0, 0.0]], [[1.0, 0.0]], val=0.4740769841801067)]
    alt = [_call([[-0.01, 2.02]], [[1.0, 3.0]], [[1.0, 1.0]], [[1.0, 1.0]]),
           _call([[-1.01, 0.4]], [[0.5, -2.0]], [[0.0, 0.0]], [[1.0, 0.0]])]
    return s0, alt


def test_threshold_interval_literals():
    import g0_refcv7 as G
    s0, alt = _example()
    iv = G.threshold_interval(s0, alt, k=3.0)
    assert abs(iv["t_replay"] - T_REPLAY) < 1e-12
    assert abs(iv["lo"] - T_REPLAY) < 1e-12
    assert abs(iv["hi"] - HI) < 1e-12
    assert abs(iv["delta_logit_max"] - 0.06) < 1e-12
    assert iv["n_supervised"] == 3
    assert iv["n_undecidable"] == 1
    assert iv["n_flipped_by_arm"] == 1
    # k = 0 -> nothing is undecidable -> a point interval at the replay
    iv0 = G.threshold_interval(s0, alt, k=0.0)
    assert iv0["n_undecidable"] == 0 and iv0["lo"] == iv0["hi"] == iv0["t_replay"]


def _brute_force_reachable(s0, und):
    """INDEPENDENT of the delta formula: actually negate every subset of the undecidable logits and
    recompute the 8-batch mean from scratch."""
    import copy
    import g0_refcv7 as G
    vals = []
    for r in range(len(und) + 1):
        for sub in itertools.combinations(und, r):
            calls = copy.deepcopy(s0)
            for u in sub:
                row, col = divmod(u["cell"], len(calls[u["batch"]]["goal_logits"][0]))
                calls[u["batch"]]["goal_logits"][row][col] = -calls[u["batch"]]["goal_logits"][row][col]
            vals.append(sum(G.conf_bce_from_cells(c) for c in calls) / len(calls))
    return min(vals), max(vals)


def test_interval_endpoints_match_brute_force_flips():
    import g0_refcv7 as G
    s0, alt = _example()
    iv = G.threshold_interval(s0, alt, k=3.0)
    lo, hi = _brute_force_reachable(s0, iv["undecidable"])
    assert abs(lo - iv["lo"]) < 1e-12 and abs(hi - iv["hi"]) < 1e-12


def test_REGRESSION_flip_sign_is_caught_by_brute_force():
    """DELIBERATE REGRESSION: the sign this module's author first wrote, (1 - 2*correct). The
    brute-force endpoints must DISAGREE with it -- i.e. the test above has power against it."""
    import g0_refcv7 as G
    s0, alt = _example()
    iv = G.threshold_interval(s0, alt, k=3.0)
    lo_bf, hi_bf = _brute_force_reachable(s0, iv["undecidable"])
    wrong_lo, wrong_hi = iv["t_replay"], iv["t_replay"]
    for u in iv["undecidable"]:
        d_wrong = -u["delta_mean"]                     # (1 - 2c) = -(2c - 1)
        wrong_lo += min(0.0, d_wrong)
        wrong_hi += max(0.0, d_wrong)
    assert abs(wrong_hi - hi_bf) > 0.2 and abs(wrong_lo - lo_bf) > 0.2


def test_conf_bce_from_cells_equals_the_trainers_function_and_one_flip_moves_it_by_c():
    """CONTROL against the REAL trainer code (refcv6_tactical.tactical_behaviour_losses), and the
    ANALYTIC identity: flipping one correct cell's validity decision changes the batch value by
    exactly +c * w / sum(w)."""
    import torch
    import g0_refcv7 as G  # noqa: F401  (bootstraps the loader's sys.path)
    from tanitad.refs import refcv6_tactical as v6
    g = torch.Generator().manual_seed(7)
    B, K = 3, 22
    logits = torch.randn(B, K, generator=g) * 2.0
    logits[1, 4] = 0.37                                  # the cell we will flip: decided valid
    conf = torch.randn(B, K, generator=g) * 2.0
    conf[1, 4] = 1.25
    y = (torch.rand(B, K, generator=g) > 0.6).float()
    y[1, 4] = 1.0                                        # -> correct
    w = (torch.rand(B, K, generator=g) > 0.3).float()
    w[1, 4] = 1.0
    mask = torch.ones(K)
    mask[3] = 0.0

    def real(lg):
        out = {"goal_logits": lg, "goal_conf": conf, "lat_logits": torch.zeros(B, 8),
               "lon_logits": torch.zeros(B, 8)}
        _t, tele = v6.tactical_behaviour_losses(
            out, goal_y=y, goal_w=w, lat_target=torch.full((B,), -100),
            lon_target=torch.full((B,), -100), goal_class_mask=mask, ignore_index=-100)
        return float(tele["tac_goal_conf_bce"])
    call = {"goal_logits": logits.tolist(), "goal_conf": conf.tolist(), "goal_y": y.tolist(),
            "goal_w": w.tolist(), "class_mask": mask.tolist()}
    assert abs(G.conf_bce_from_cells(call) - real(logits)) < 1e-6
    flipped = logits.clone()
    flipped[1, 4] = -0.37
    sw = float((w * mask.view(1, -1)).sum())
    assert abs((real(flipped) - real(logits)) - 1.25 * 1.0 / sw) < 1e-6


# ------------------------------------------------------------------------------------------- #
# judge(..., amend="A6")                                                                        #
# ------------------------------------------------------------------------------------------- #
def _rec(s0_calls=None, alt_calls=None, arm_row=None, m1_row=None):
    rec = {"model": {"state_dict": {"missing": [], "unexpected": []},
                     "param_breakdown": {"equal": True}, "anchor_file_vs_ckpt_buffers": {},
                     "declared_vs_built": {"mismatches": []}},
           "wrapper_control": {"clause": "PASS"},
           "mutations": {"m1": {"row": m1_row or {"eval_lat": 0.5}}},
           "a6": {"cells": {}}}
    if s0_calls is not None:
        rec["a6"]["cells"]["s0"] = s0_calls
    if alt_calls is not None:
        rec["a6"]["cells"]["fp32_s0"] = alt_calls
    if arm_row is not None:
        rec["a6"]["fp32_s0"] = {"row": arm_row}
    return rec


def _by_seed(row, n=24):
    return {s: {"row": dict(row), "buffers_unchanged": True} for s in range(n)}


def test_A6_threshold_term_in_and_out_of_the_interval():
    import g0_refcv7 as G
    s0, alt = _example()
    row = {"eval_tacv6_goal_conf_bce": round(T_REPLAY, 5), "eval_lat": 0.5}
    rec = _rec(s0, alt, arm_row=dict(row), m1_row={"eval_lat": 0.6})
    # 0.55 is inside [0.32750 - 1 %, 0.57750 + 1 %]; it is 68 % off the replay, i.e. OUT as SMOOTH
    v = G.judge({"eval_tacv6_goal_conf_bce": 0.55, "eval_lat": 0.5}, _by_seed(row), rec, amend="A6")
    t = v["terms"]["eval_tacv6_goal_conf_bce"]
    assert t["cls"] == "THRESHOLD_TARGET" and t["verdict"] == "OK"
    assert v["G0"] == "PASS"
    v5 = G.judge({"eval_tacv6_goal_conf_bce": 0.55, "eval_lat": 0.5}, _by_seed(row), rec, amend="A5")
    assert v5["terms"]["eval_tacv6_goal_conf_bce"]["cls"] == "SMOOTH"
    assert v5["terms"]["eval_tacv6_goal_conf_bce"]["verdict"] == "OUT"   # A6 never leaks into A5
    # 0.60 is above hi + 1 % (0.5775 + 0.006 = 0.5835): OUT under A6 as well
    v2 = G.judge({"eval_tacv6_goal_conf_bce": 0.60, "eval_lat": 0.5}, _by_seed(row), rec, amend="A6")
    assert v2["terms"]["eval_tacv6_goal_conf_bce"]["verdict"] == "OUT" and v2["G0"] == "FAIL"
    # BELOW the replay: lo is the replay itself (the only undecidable cell can only add)
    v3 = G.judge({"eval_tacv6_goal_conf_bce": 0.30, "eval_lat": 0.5}, _by_seed(row), rec, amend="A6")
    assert v3["terms"]["eval_tacv6_goal_conf_bce"]["verdict"] == "OUT"


def test_A6_fails_closed_without_cells_or_with_a_failed_control():
    import copy
    import g0_refcv7 as G
    s0, alt = _example()
    row = {"eval_tacv6_goal_conf_bce": round(T_REPLAY, 5), "eval_lat": 0.5}
    v = G.judge({"eval_tacv6_goal_conf_bce": round(T_REPLAY, 5), "eval_lat": 0.5}, _by_seed(row),
                _rec(None, None, arm_row=dict(row), m1_row={"eval_lat": 0.6}), amend="A6")
    assert v["terms"]["eval_tacv6_goal_conf_bce"]["verdict"] == "OUT"
    bad = copy.deepcopy(s0)
    bad[0]["tac_goal_conf_bce"] = 0.9                   # the cells no longer reproduce the trainer
    v2 = G.judge({"eval_tacv6_goal_conf_bce": round(T_REPLAY, 5), "eval_lat": 0.5}, _by_seed(row),
                 _rec(bad, alt, arm_row=dict(row), m1_row={"eval_lat": 0.6}), amend="A6")
    assert v2["terms"]["eval_tacv6_goal_conf_bce"]["verdict"] == "OUT"
    # no fp32 arm at all -> A6 NOT EVALUABLE -> FAIL, never PASS
    v3 = G.judge({"eval_tacv6_goal_conf_bce": round(T_REPLAY, 5), "eval_lat": 0.5}, _by_seed(row),
                 _rec(s0, alt, arm_row=None, m1_row={"eval_lat": 0.6}), amend="A6")
    assert v3["G0"] == "FAIL" and any("NOT EVALUABLE" in r for r in v3["reasons"])


def test_A6_smooth_floor_rescues_only_within_k_phi_and_names_it():
    import g0_refcv7 as G
    s0, alt = _example()
    row = {"eval_tacv6_goal_conf_bce": round(T_REPLAY, 5), "eval_lat": 0.500}
    inrun = {"eval_tacv6_goal_conf_bce": round(T_REPLAY, 5), "eval_lat": 0.508}   # 1.6 % > 1 %
    arm_ok = {**row, "eval_lat": 0.503}                 # phi 0.003 -> 3 phi 0.009 >= 0.008
    v = G.judge(inrun, _by_seed(row), _rec(s0, alt, arm_row=arm_ok, m1_row={"eval_lat": 0.6}),
                amend="A6")
    t = v["terms"]["eval_lat"]
    assert t["verdict"] == "OK" and t.get("a6_rescued") is True and abs(t["a6_phi"] - 0.003) < 1e-12
    assert [r["term"] for r in v["a6_rescued"]] == ["eval_lat"]
    arm_small = {**row, "eval_lat": 0.502}              # 3 phi = 0.006 < 0.008 -> OUT
    v2 = G.judge(inrun, _by_seed(row), _rec(s0, alt, arm_row=arm_small, m1_row={"eval_lat": 0.6}),
                 amend="A6")
    assert v2["terms"]["eval_lat"]["verdict"] == "OUT" and v2["G0"] == "FAIL"


def test_A6_mutation_detection_uses_the_same_floor_and_VOIDs_without_power():
    import g0_refcv7 as G
    s0, alt = _example()
    row = {"eval_tacv6_goal_conf_bce": round(T_REPLAY, 5), "eval_lat": 0.500}
    inrun = dict(row)
    arm = {**row, "eval_lat": 0.505}                    # phi 0.005 -> 3 phi = 0.015 (3 %)
    # M1 moves lat by 4 % -> beyond max(1 %, 3 %) -> detected -> PASS
    v = G.judge(inrun, _by_seed(row), _rec(s0, alt, arm_row=arm, m1_row={"eval_lat": 0.520}), amend="A6")
    assert v["mutation_detection"]["m1"]["detected"] is True and v["G0"] == "PASS"
    # M1 moves lat by 2 %: beyond the registered 1 % but INSIDE 3 phi -> not detected -> VOID
    v2 = G.judge(inrun, _by_seed(row), _rec(s0, alt, arm_row=arm, m1_row={"eval_lat": 0.510}), amend="A6")
    assert v2["mutation_detection"]["m1"]["detected"] is False and v2["G0"] == "VOID"
    # the same M1 row IS detected under A5 (registered 1 %): A6 can only lose power, never invent it
    v5 = G.judge(inrun, _by_seed(row), _rec(s0, alt, arm_row=arm, m1_row={"eval_lat": 0.510}), amend="A5")
    assert v5["mutation_detection"]["m1"]["detected"] is True


def test_A6_smooth_median_bar_literal():
    import g0_refcv7 as G
    s0, alt = _example()
    keys = [f"eval_t{i}" for i in range(5)]
    row = {"eval_tacv6_goal_conf_bce": round(T_REPLAY, 5), **{k: 1.0 for k in keys}}
    inrun = {"eval_tacv6_goal_conf_bce": round(T_REPLAY, 5), **{k: 1.003 for k in keys}}  # 0.3 % each
    arm = {**row, **{k: 1.0015 for k in keys}}          # phi 0.0015, |x| = in-run 1.003 (as rel_dev)
    v = G.judge(inrun, _by_seed(row), _rec(s0, alt, arm_row=arm, m1_row={k: 1.2 for k in keys}),
                amend="A6")
    # bar = max(0.002, 3 * 0.0015 / 1.003) = 0.004486540378863410 (computed once, a literal)
    assert abs(v["medians"]["SMOOTH_bar"] - 0.004486540378863410) < 1e-12
    assert not any("SMOOTH median" in r for r in v["reasons"])
    v5 = G.judge(inrun, _by_seed(row), _rec(s0, alt, arm_row=arm, m1_row={k: 1.2 for k in keys}),
                 amend="A5")
    assert any("SMOOTH median" in r for r in v5["reasons"])


def test_a6_registration_is_time_ordered(tmp_path, monkeypatch):
    import g0_refcv7 as G
    p = tmp_path / "SPEC_SHA256_AMENDMENT_A6.txt"
    monkeypatch.setattr(G, "A6_REGISTRATION", p)
    assert G.a6_registration("2026-10-04T12:00:00")["registered"] is False
    p.write_text("sha256 ...", encoding="utf-8")
    t_reg = time.mktime(time.strptime("2026-10-04T10:00:00", "%Y-%m-%dT%H:%M:%S"))
    os.utime(p, (t_reg, t_reg))
    assert G.a6_registration("2026-10-04T12:00:00")["registered"] is True
    r = G.a6_registration("2026-10-04T09:00:00")
    assert r["registered"] is False and "POST HOC" in r["why"]


def test_capture_hook_on_the_real_tactical_module_and_it_never_breaks_the_forward():
    """The hook patches the REAL `refcv6_tactical.tactical_behaviour_losses` (the attribute
    `compute_losses_v3` resolves at call time, refc_v3_train.py:5263), records exactly what it was
    called with, restores the original on remove(), and a RECORDING error never reaches the forward."""
    import types
    import torch
    import g0_refcv7 as G
    from tanitad.refs import refcv6_tactical as v6
    orig = v6.tactical_behaviour_losses
    tr = types.SimpleNamespace(v6tac=v6)
    B, K = 2, 22
    out = {"goal_logits": torch.zeros(B, K), "goal_conf": torch.zeros(B, K),
           "lat_logits": torch.zeros(B, 8), "lon_logits": torch.zeros(B, 8)}
    kw = dict(goal_y=torch.ones(B, K), goal_w=torch.ones(B, K), lat_target=torch.full((B,), -100),
              lon_target=torch.full((B,), -100), goal_class_mask=torch.ones(K), ignore_index=-100)
    cap = G.TacCellCapture(tr).install()
    try:
        assert v6.tactical_behaviour_losses is not orig
        _t, tele = v6.tactical_behaviour_losses(out, **kw)
    finally:
        cap.remove()
    assert v6.tactical_behaviour_losses is orig
    assert len(cap.calls) == 1 and cap.errors == []
    c = cap.calls[0]
    assert len(c["goal_logits"]) == 2 and len(c["goal_logits"][0]) == 22
    assert abs(c["tac_goal_conf_bce"] - float(tele["tac_goal_conf_bce"])) == 0.0
    assert abs(G.conf_bce_from_cells(c) - c["tac_goal_conf_bce"]) < 1e-6
    # a RECORDING failure is logged, never raised: an original whose telemetry lacks the key the
    # recorder reads -> the forward's result still comes back unchanged, and the error is named
    sentinel = (torch.tensor(1.5), {"something_else": 0.0})
    tr2 = types.SimpleNamespace(v6tac=types.SimpleNamespace(
        tactical_behaviour_losses=lambda out, **k: sentinel))
    cap2 = G.TacCellCapture(tr2).install()
    try:
        got = tr2.v6tac.tactical_behaviour_losses(out, **kw)
    finally:
        cap2.remove()
    assert got[0] is sentinel[0] and got[1] is sentinel[1]         # the forward's own objects
    assert cap2.calls == [] and len(cap2.errors) == 1 and cap2.errors[0].startswith("KeyError")


def test_batch_cache_lever_ram_vs_disk(tmp_path, monkeypatch):
    """The 2026-10-04 lever: RAM batches are the SAME tensors as the disk/mmap ones; the default is
    unchanged (disk at the default root); only `ram` (arg or env) selects RAM."""
    import torch
    import g0_refcv7 as G
    g = torch.Generator().manual_seed(1)
    eb = {"frames": torch.randint(0, 255, (2, 3, 4), generator=g, dtype=torch.uint8),
          "x": torch.randn(2, 3, generator=g)}
    monkeypatch.delenv("REFCV7_G0_BATCH_CACHE", raising=False)
    d = G.make_batches(None, tmp_path / "default", [lambda: eb])
    assert isinstance(d, G.MmapBatches) and Path(d.root) == tmp_path / "default"
    r = G.make_batches("ram", tmp_path / "unused", [lambda: eb])
    assert isinstance(r, G.RamBatches) and not (tmp_path / "unused").exists()
    assert torch.equal(r[0]["frames"], d[0]["frames"]) and torch.equal(r[0]["x"], d[0]["x"])
    monkeypatch.setenv("REFCV7_G0_BATCH_CACHE", "ram")
    assert isinstance(G.make_batches(None, tmp_path / "x2", [lambda: eb]), G.RamBatches)
    # an explicit path wins over the env
    assert isinstance(G.make_batches(str(tmp_path / "explicit"), tmp_path / "x3", [lambda: eb]),
                      G.MmapBatches)
    r.remove()
    assert len(r) == 0

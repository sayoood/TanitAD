#!/usr/bin/env python3
"""Tests for eval/instruments/strategic_family.py. Synthetic ego-frame trajectories only (no navtest files), CPU, no scipy anywhere.

    set REFE_EVAL_DIR=<dir holding command_adherence.py + proxy_eval.py>; set TANITAD_REPO=<repo with stack/ and taniteval/>
    python test_strategic_family.py        # prints ZZSTRAT PASS n/n; pytest also collects the test_* functions

Every check has a deliberate-regression (RED) arm that MUTATES the thing under test and requires the check to fail.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.modules.setdefault("scipy", None)                                         # scipy BLOCKED: any import of it raises ImportError
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strategic_family as SF  # noqa: E402

CA = SF.CA


def curve(yaw_rate: float, speed: float = 6.0, n: int = 8, dt: float = 0.5) -> np.ndarray:
    """ego-frame [n,3] (x fwd, y left, heading): constant speed and yaw rate from the origin, heading 0"""
    x = y = h = 0.0
    out = []
    for _ in range(n):
        h += yaw_rate * dt
        x += speed * dt * np.cos(h)
        y += speed * dt * np.sin(h)
        out.append((x, y, h))
    return np.array(out)


def make(n_logs=12, per_log=9, seed=0):
    """tokens cycling LEFT / STRAIGHT / RIGHT; GT = the matching curve with a per-token speed; plan = GT + deterministic noise, so plan != GT"""
    rng = np.random.default_rng(seed)
    cmds, logs, gts = [], [], []
    for g in range(n_logs):
        for k in range(per_log):
            c = (g + k) % 3                                                    # 0 LEFT, 1 STRAIGHT, 2 RIGHT
            yr = {0: 0.25, 1: 0.0, 2: -0.25}[c]
            gts.append(curve(yr, speed=4.0 + rng.random() * 4))
            cmds.append(c)
            logs.append(f"log{g:02d}")
    gt = np.array(gts)
    plan = gt + rng.normal(0, 0.05, gt.shape) * np.array([1, 1, 0.1])
    return np.array(cmds), np.array(logs), plan, gt


NB = 300
RESULTS: list = []


def check(fn):
    def run():
        try:
            fn()
            RESULTS.append((fn.__name__, True, ""))
        except Exception as e:                                                # noqa: BLE001
            RESULTS.append((fn.__name__, False, f"{type(e).__name__}: {str(e)[:300]}"))
    run.__name__ = fn.__name__
    return run


def red(fn, *a, **k):
    """the deliberate-regression arm: the call must raise AssertionError (the check went RED)"""
    try:
        fn(*a, **k)
    except AssertionError:
        return
    raise AssertionError("the deliberate regression was NOT caught: the check cannot fail")


def test_fixture_classes():
    cmd, logs, plan, gt = make()
    cls, _ = CA.plan_classes(gt)
    assert (cls == cmd).all(), "fixture curves must be classed as their command by the canonical labeller"


def test_logged_future_is_exact_zero():
    cmd, logs, plan, gt = make()
    c = SF.control_logged_future(cmd, logs, gt, NB)
    assert c["ok"], c
    # RED arm: a plan that differs from the logged future must NOT read zero
    bad = SF.family_arm(cmd, logs, gt + np.array([0.0, 0.5, 0.0]), gt, NB)
    assert bad["S_GOAL"]["LEFT"]["endpoint_error_m"]["point"] > 0.4
    # and the control itself reads ok=False when fed a shifted "future" as the arm (mutation of the arm, same GT)
    a = SF.family_arm(cmd, logs, gt + 0.1, gt, NB)
    assert a["S_GOAL"]["LEFT"]["endpoint_error_m"]["point"] != 0.0


def test_metric_values_known():
    cmd, logs, plan, gt = make()
    p = gt.copy()
    p[:, SF.END, 0] += 3.0                                                     # 3 m long at the endpoint
    p[:, SF.END, 2] += 0.2                                                     # 0.2 rad heading error
    t = SF.goal_token_metrics(p, gt, cmd)
    assert np.allclose(t["endpoint_error_m"], 3.0) and np.allclose(t["endpoint_heading_err_rad"], 0.2)
    q = gt.copy()
    q[:, SF.END, 1] += 1.0                                                     # 1 m further LEFT
    t = SF.goal_token_metrics(q, gt, cmd)
    assert np.allclose(t["lateral_offset_m"][cmd == CA.LEFT], 1.0) and np.allclose(t["lateral_offset_m"][cmd == CA.RIGHT], -1.0), "sign must be toward the COMMANDED side"
    assert np.allclose(t["lateral_offset_m"][cmd == CA.STRAIGHT], 1.0)
    s = gt.copy()
    s[..., :2] *= 0.5                                                          # half the route: arc ratio exactly 0.5
    t = SF.goal_token_metrics(s, gt, cmd)
    assert np.allclose(t["plan_arc_m"] / t["gt_arc_m"], 0.5)
    # RED arm: sign mutation (RIGHT not flipped) must be detected by the same assertion
    orig = SF.side_sign
    SF.side_sign = lambda c: np.ones(len(c))
    try:
        red(lambda: _assert_right_flip(q, gt, cmd))
    finally:
        SF.side_sign = orig


def _assert_right_flip(q, gt, cmd):
    t = SF.goal_token_metrics(q, gt, cmd)
    assert np.allclose(t["lateral_offset_m"][cmd == CA.RIGHT], -1.0)


def test_mirror_lowers_adherence_and_goes_red_when_noop():
    cmd, logs, plan, gt = make()
    c = SF.control_mirror(cmd, logs, plan)
    assert c["ok"] and c["turn_active_adherence_mirrored"] < c["turn_active_adherence_real"], c
    assert c["turn_active_adherence_real"] > 0.9 and c["turn_active_adherence_mirrored"] < 0.1, c
    orig = SF.mirror_lateral
    SF.mirror_lateral = lambda p: np.array(p, dtype=np.float64, copy=True)     # a no-op mirror
    try:
        noop = SF.control_mirror(cmd, logs, plan)
        assert noop["ok"] is False, "the mirror control must go RED when mirroring is a no-op"
    finally:
        SF.mirror_lateral = orig


def test_self_vs_self_exactly_zero():
    cmd, logs, plan, gt = make()
    A = SF.family_arm(cmd, logs, plan, gt, NB)
    c = SF.control_self(cmd, logs, A, NB)
    assert c["ok"], c
    P = SF.family_paired(cmd, logs, A, A, plan, plan, NB)
    d = P["S_GOAL"]["LEFT"]["endpoint_error_m"]
    assert d["point"] == 0.0 and d["ci95"] == [0.0, 0.0] and d["separated"] is False
    # RED arm: against a different arm the delta is NOT zero and the self-check would flag it
    B = SF.family_arm(cmd, logs, plan + np.array([0.0, 0.3, 0.0]), gt, NB)
    Q = SF.family_paired(cmd, logs, B, A, None, None, NB)
    dd = Q["S_GOAL"]["LEFT"]["endpoint_error_m"]
    assert dd["point"] != 0.0 and dd["separated"] is True
    red(lambda: _must_be_zero(dd))


def _must_be_zero(d):
    assert d["point"] == 0.0


def test_bootstrap_stream_matches_registered_stream():
    """the per-token mean of a subset under SF.goal_block equals proxy_eval.boot (the registered function) on the same tokens, bit for bit"""
    import proxy_eval as PE
    cmd, logs, plan, gt = make()
    tok = SF.goal_token_metrics(plan, gt, cmd)
    m = SF.subset_mask(cmd, "LEFT")
    _, S = SF.per_log_sums(tok, logs, m)
    blk = SF.goal_block(S, 500, SF.SEED)["endpoint_error_m"]
    idx = np.flatnonzero(m)
    toks = [f"t{i:05d}" for i in idx]
    mu, lo, hi = PE.boot({t: float(tok["endpoint_error_m"][i]) for t, i in zip(toks, idx)}, {t: str(logs[i]) for t, i in zip(toks, idx)}, n=500, seed=SF.SEED)
    assert abs(blk["point"] - mu) < 1e-12 and abs(blk["ci95"][0] - lo) < 1e-12 and abs(blk["ci95"][1] - hi) < 1e-12, (blk, mu, lo, hi)
    # RED arm: a different seed must give a different interval
    other = SF.goal_block(S, 500, SF.SEED + 1)["endpoint_error_m"]
    red(lambda: _eq(other["ci95"], [lo, hi]))


def _eq(a, b):
    assert abs(a[0] - b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12


def test_sdec_per_class_and_counts():
    cmd, logs, plan, gt = make()
    cls, _ = CA.plan_classes(plan)
    d = SF.sdec_arm(cmd, cls, logs, NB)
    for nm in SF.CLASSES:
        assert d["per_class"][nm]["n"] == int((cmd == CA.CMD_NAMES.index(nm)).sum())
        assert d["per_class"][nm]["adherence"]["point"] == 1.0
    assert d["turn_active"]["n"] == int(((cmd == CA.LEFT) | (cmd == CA.RIGHT)).sum())
    # RED arm: a plan that ignores the command (constant STRAIGHT) must drop turn-active adherence to 0
    const = np.full(len(cmd), CA.STRAIGHT)
    d2 = SF.sdec_arm(cmd, const, logs, NB)
    assert d2["turn_active"]["accuracy"]["point"] == 0.0
    red(lambda: _must_be_one(d2["turn_active"]["accuracy"]["point"]))


def _must_be_one(x):
    assert x == 1.0


def _must_raise(f):
    try:
        f()
    except (ValueError, CA.PlanFormatError):
        return
    raise AssertionError("did not raise")


def test_shape_refusals():
    cmd, logs, plan, gt = make()
    _must_raise(lambda: SF.goal_token_metrics(plan[:, :6], gt, cmd))               # wrong pose count refused
    red(lambda: _must_raise(lambda: SF.goal_token_metrics(plan, gt, cmd)))         # RED arm: a valid input does not raise, so the check goes red


def test_unknown_class_absent_is_reported_not_crashed():
    cmd, logs, plan, gt = make()
    A = SF.family_arm(cmd, logs, plan, gt, NB)
    assert "UNKNOWN" not in A["S_GOAL"] and set(A["S_GOAL"]) == set(SF.SUBSETS)
    keep = cmd != CA.RIGHT
    B = SF.family_arm(cmd[keep], logs[keep], plan[keep], gt[keep], NB)
    assert B["S_GOAL"]["RIGHT"]["n"] == 0


ALL = [test_fixture_classes, test_logged_future_is_exact_zero, test_metric_values_known, test_mirror_lowers_adherence_and_goes_red_when_noop,
       test_self_vs_self_exactly_zero, test_bootstrap_stream_matches_registered_stream, test_sdec_per_class_and_counts, test_shape_refusals,
       test_unknown_class_absent_is_reported_not_crashed]


def main() -> int:
    for t in ALL:
        check(t)()
    bad = [r for r in RESULTS if not r[1]]
    for n, ok, msg in RESULTS:
        print(("PASS " if ok else "FAIL ") + n + (" :: " + msg if msg else ""))
    print(f"ZZSTRAT {'PASS' if not bad else 'FAIL'} {len(RESULTS) - len(bad)}/{len(RESULTS)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""Unit tests for the L7 scorer (scorer9).  Every expectation is a LITERAL or an ANALYTIC target, never an expression
over the code under test.  Each guard also has a deliberate-regression MUTANT that must turn its check RED.

    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' python -m pytest -q tests/test_refc_scorer.py
    python tests/test_refc_scorer.py            # standalone: prints ZZL7TEST_PASS <n> or the failures
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import refc_scorer as RS  # noqa: E402
import l7_targets_pai as TP  # noqa: E402

FROZEN_MBR = ("C:/Users/Admin/qland/landed_hold_x8/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
              "2026-09-28-refcv7-standard-tests/battery/code/fan_mbr_x8.py")


# ------------------------------------------------------------------------------------------------ fixtures (literal)
def toy_fan():
    """B=3 windows, N=5 candidates, 8 slots.  GT = straight line at 10 m/s.  Candidate 3 IS the GT; candidates 0,1,2,4
    are offset laterally by 2.0, 1.0, 0.5, 0.25 m at every slot.  Oracle index = 3 on every window (literal)."""
    t = np.asarray(RS.SLOT_T_REFC)
    gt = np.stack([10.0 * t, np.zeros_like(t)], -1)                  # [8,2]
    offs = [2.0, 1.0, 0.5, 0.0, 0.25]
    fan = np.stack([gt + np.array([0.0, o]) for o in offs])           # [5,8,2]
    fan = np.repeat(fan[None], 3, 0)
    gtb = np.repeat(gt[None], 3, 0)
    return fan, gtb


# ------------------------------------------------------------------------------------------------ check functions
def check_perfect_scorer_selects_oracle(select_fn) -> list:
    fan, gt = toy_fan()
    neg_ade = -np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)   # the test's own ADE (numpy), not the builder's
    p = select_fn(torch.from_numpy(neg_ade))
    return [] if p.tolist() == [3, 3, 3] else [f"perfect scorer picked {p.tolist()}, expected [3, 3, 3]"]


def check_constant_scorer_ties(select_fn) -> list:
    fails = []
    z = torch.zeros(2, 6)
    reach = torch.tensor([[False, False, True, True, True, True]] * 2)
    if select_fn(z, reach).tolist() != [2, 2]:
        fails.append(f"constant + reach: {select_fn(z, reach).tolist()} != [2, 2]")
    ceil = torch.tensor([[True, True, False, True, True, True]] * 2)
    if select_fn(z, reach, ceil).tolist() != [3, 3]:
        fails.append(f"constant + reach + ceiling: {select_fn(z, reach, ceil).tolist()} != [3, 3]")
    dead_ceil = torch.zeros(2, 6, dtype=torch.bool)                   # empty ceiling row -> reach-only fallback
    if select_fn(z, reach, dead_ceil).tolist() != [2, 2]:
        fails.append(f"empty ceiling row fallback: {select_fn(z, reach, dead_ceil).tolist()} != [2, 2]")
    emit = torch.tensor([[True, True, True, False, True, True]] * 2)
    if select_fn(z, reach, ceil, emit).tolist() != [4, 4]:
        fails.append(f"constant + reach + ceiling + emit: {select_fn(z, reach, ceil, emit).tolist()} != [4, 4]")
    if select_fn(torch.zeros(1, 4)).tolist() != [0]:
        fails.append("unmasked constant row must pick index 0")
    return fails


def run_selection_checks(select_fn) -> list:
    return check_perfect_scorer_selects_oracle(select_fn) + check_constant_scorer_ties(select_fn)


def check_guard(assemble_fn) -> list:
    rec = {"anchor_traj": torch.zeros(1, 4, 8, 2), "v0": torch.zeros(1), "v_max_ms": torch.zeros(1)}
    try:
        assemble_fn(rec, "E")
    except RS.InadmissibleInput:
        pass
    else:
        return ["v_max_ms (the future-derived ceiling) was ACCEPTED as a scorer input"]
    rec2 = {"anchor_traj": torch.zeros(1, 4, 8, 2), "v0": torch.zeros(1), "nav_cmd": torch.zeros(1)}
    try:
        assemble_fn(rec2, "E")
    except RS.InadmissibleInput:
        return []
    return ["nav_cmd (route-derived) was ACCEPTED as a scorer input"]


# ------------------------------------------------------------------------------------------------ tests
def test_selection_contract():
    assert run_selection_checks(RS.select) == []


def test_selection_mutants_go_red():
    # M1: argmin instead of argmax (a sign defect)
    m1 = lambda s, r=None, c=None, e=None: RS.e9_masked(s, r, c, e).argmin(dim=1)
    # M2: the real historical defect class -- rank the UNMASKED score (the reach / ceiling masks dropped)
    m2 = lambda s, r=None, c=None, e=None: s.argmax(dim=1)
    # M3: the ceiling applied WITHOUT the empty-row fallback
    def m3(s, r=None, c=None, e=None):
        rank = s if r is None else s.masked_fill(~r, float("-inf"))
        if c is not None:
            rank = rank.masked_fill(~c, float("-inf"))
        return rank.argmax(dim=1)
    for name, m in (("argmin", m1), ("unmasked", m2), ("no-fallback", m3)):
        assert run_selection_checks(m) != [], f"mutant {name} was NOT detected"


def test_guard_refuses_forbidden_inputs():
    assert check_guard(RS.assemble_inputs) == []
    mutant = lambda rec, mode: {"traj": rec["anchor_traj"], "v0": rec["v0"]}        # a guard that checks nothing
    assert check_guard(mutant) != []


def test_unknown_input_refused():
    try:
        RS.assert_admissible(["anchor_traj", "v0", "some_new_tensor"])
    except RS.InadmissibleInput:
        return
    raise AssertionError("an undeclared input key was accepted")


def test_listwise_loss_analytic():
    T = 0.5
    cost = torch.tensor([[0.0, T * math.log(3.0)]])                  # p* = [3/4, 1/4]
    l_equal = RS.listwise_imitation_loss(torch.tensor([[0.0, 0.0]]), cost, None, T)
    assert abs(float(l_equal) - 0.6931471805599453) < 1e-6          # ln 2
    l_match = RS.listwise_imitation_loss(torch.tensor([[math.log(3.0), 0.0]]), cost, None, T)
    assert abs(float(l_match) - 0.5623351446188083) < 1e-6          # H(3/4, 1/4)
    l_mask = RS.listwise_imitation_loss(torch.tensor([[0.0, 5.0]]), torch.tensor([[0.0, 1.0]]),
                                        torch.tensor([[True, False]]), T)
    assert abs(float(l_mask) - 0.0) < 1e-7                           # one kept candidate: loss 0


def test_navsim_v1_rule_literal():
    half = {k: torch.zeros(1, 1) for k in ("nc", "dac", "ep", "ttc", "c")}          # sigmoid(0) = 0.5
    assert abs(float(RS.rule_navsim_v1(half)) - 0.125) < 1e-7       # 0.5 * 0.5 * (12 * 0.5) / 12
    dead = dict(half, nc=torch.full((1, 1), -1e4))
    assert float(RS.rule_navsim_v1(dead)) == 0.0                    # a failed multiplier zeroes the candidate
    sure = {k: torch.full((1, 1), 1e4) for k in ("nc", "dac", "ep", "ttc", "c")}
    assert abs(float(RS.rule_navsim_v1(sure)) - 1.0) < 1e-7


def test_along_cross_analytic():
    t = np.asarray(RS.SLOT_T_REFC)
    gt = np.stack([10.0 * t, np.zeros_like(t)], -1)[None]           # straight along +x
    fan = (gt + np.array([0.3, -0.2]))[:, None]                      # +0.3 m along, 0.2 m across, every slot
    al, cr = TP.along_cross(fan, gt)
    assert np.allclose(al, 0.3, atol=1e-9) and np.allclose(cr, 0.2, atol=1e-9)
    # a left circle R = 20 m at 8 m/s: a candidate 1 m OUTSIDE along the normal is pure cross-track error
    th = 8.0 * t / 20.0
    g2 = np.stack([20 * np.sin(th), 20 * (1 - np.cos(th))], -1)[None]
    ch = 8.0 * np.diff(np.concatenate([[0.0], t])) / 20.0
    mid = np.concatenate([[0.0], th])[:-1] + ch / 2.0                # chord direction of segment j
    n = np.stack([np.sin(mid), -np.cos(mid)], -1)                    # right-hand normal of each chord
    al2, cr2 = TP.along_cross((g2 + n)[:, None], g2)
    assert np.allclose(cr2, 1.0, atol=1e-9) and np.allclose(al2, 0.0, atol=1e-9)


def test_forward_refuses_grad_and_wrong_mode_inputs():
    m = RS.L7Scorer(RS.L7Config(mode="E"))
    tr = torch.zeros(1, 4, 8, 2, requires_grad=True)
    try:
        m(tr, torch.zeros(1))
    except ValueError:
        pass
    else:
        raise AssertionError("a grad-carrying candidate tensor was accepted")
    try:
        m(torch.zeros(1, 4, 8, 2), torch.zeros(1), model_feats=torch.zeros(1, 4, len(RS.MODEL_FEATURES)))
    except ValueError:
        return
    raise AssertionError("the E floor accepted model features")


def test_param_counts_literal():
    assert RS.n_params(RS.L7Scorer(RS.L7Config(mode="E"))) == 154504
    assert RS.n_params(RS.L7Scorer(RS.L7Config(mode="F"))) == 155400
    assert RS.n_params(RS.L7Scorer(RS.L7Config(mode="S", source_dims={"bev": 96, "agent": 256}))) == 464520


def test_mbr_risk_reproduces_frozen_picker():
    """Independently authored reference: the frozen fan_mbr_x8.mbr_pick (sha256 94cef175...).  argmin of mbr_risk over
    the top-32 set must equal its pick on every window."""
    if not os.path.exists(FROZEN_MBR):
        import pytest
        pytest.skip("frozen fan_mbr_x8.py not on this box")
    import importlib.util
    spec = importlib.util.spec_from_file_location("fan_mbr_x8_ref", FROZEN_MBR)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    rng = np.random.default_rng(7)
    fan = rng.normal(size=(40, 149, 4, 2)).astype(np.float32) * 3.0
    sc = rng.normal(size=(40, 149)).astype(np.float32)
    want = ref.mbr_pick(fan, sc, 32, 1.0)
    risk, top = RS.mbr_risk(torch.from_numpy(fan), torch.from_numpy(sc), None, 32, 1.0)
    got = risk.masked_fill(~top, float("inf")).argmin(1).numpy()
    assert (got == want).all(), f"{int((got != want).sum())} of 40 windows differ from the frozen MBR picker"


def test_oracle_scorer_on_real_bank_targets():
    """K2 on the banked eval139 s1 targets if present: imit = -cost selects the masked argmin on EVERY window, and its
    2 s ADE equals FANCEIL_R8_s1.json's oracle-in-fan 0.0841 (literal, 4 dp)."""
    p = os.path.join(os.path.dirname(os.path.dirname(HERE)), "smoke", "targets_pai_s1_full.npz")
    if not os.path.exists(p):
        import pytest
        pytest.skip("eval139 s1 targets not built on this box")
    t = np.load(p, allow_pickle=True)
    a = torch.from_numpy(t["ade_s2"].astype(np.float64))
    pick = RS.select(-a)
    assert int((pick != a.argmin(1)).sum()) == 0
    assert round(float(a[torch.arange(len(pick)), pick].mean()), 4) == 0.0841


if __name__ == "__main__":
    fails = []
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "Skipped":
                print(f"SKIP {fn.__name__}: {e}")
                continue
            fails.append(f"{fn.__name__}: {type(e).__name__}: {e}")
    if fails:
        print("ZZL7TEST_FAIL " + " | ".join(fails))
        raise SystemExit(1)
    print(f"ZZL7TEST_PASS {len(tests)}")

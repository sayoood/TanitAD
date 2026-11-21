"""Tests for the N7 generator levers. Run with the navsim-crun venv python, CUDA_VISIBLE_DEVICES=-1:
  C:/Users/Admin/navsim-crun/venv/Scripts/python.exe -B -m pytest -q test_n7_gen.py
Pure tests need no scene; REAL tests (marked real) build one navtrain scene (nuPlan DB) for a residual token and need F:/ and E:/ data."""
import json
import os

import n7_env  # noqa: F401
import numpy as np
import pytest

import n7_lib as L

HOLD = os.path.dirname(os.path.abspath(__file__))
RES17 = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/rj5_navtrain/repair_nt/navtrain_residual17.json"
PLANS = "F:/tanitad_scratch/rj5nt/repair_nt/nt_plans.npz"
ROLLS = "F:/tanitad_scratch/rj5nt/repair_nt/nt_rolls.jsonl"
REAL_TOKENS = ["7d4286e68aee5c8b", "85e35e87946d502d", "36c4507970805f49", "fc61cadc28715436", "69b54b1998f05c13"]   # residual tokens (navtrain); 36c4/fc61 are rescued by the slow-down lever, 69b5 by L1


def _plan():
    t = L.T20
    return np.stack([8.0 * t + 0.3 * t ** 2, 0.1 * np.sin(t), 0.02 * t], 1)


# ------------------------------------------------------------------------------------------------ identity
def test_identity_stretch_and_offset_bit_identical():
    p = _plan()
    assert np.array_equal(L.time_stretch(p, 1.0), p)
    assert np.array_equal(L.lateral_offset(p, 0.0, 1.0), p)
    assert np.array_equal(L.lateral_offset(p, 0.0, 0.2), p)


def test_stretch_semantics_slower_is_shorter():
    p = _plan()
    for s in (0.95, 0.9, 0.8):
        q = L.time_stretch(p, s)
        assert q[-1, 0] < p[-1, 0]                           # same path, covered more slowly
        assert np.all(np.diff(q[:, 0]) > 0)
    # the interpolant reproduces the plan at s*t when s*t is a grid time (0.5 -> t = 0.4, 0.8, ... of the plan)
    q = L.time_stretch(p, 0.5)
    assert np.allclose(q[1::2], p[:10], atol=1e-9)


def test_stretch_faster_is_the_regression_direction():
    p = _plan()
    assert L.time_stretch(p, 1.1)[-1, 0] > p[-1, 0]


# ------------------------------------------------------------------------------------------------ determinism
def test_determinism_pure():
    p = _plan()
    assert np.array_equal(L.time_stretch(p, 0.9), L.time_stretch(p, 0.9))
    assert np.array_equal(L.lateral_offset(p, 0.3, 1.0), L.lateral_offset(p, 0.3, 1.0))


def test_levers_frozen_dict_complete():
    """every lever has its parameters in LEVERS and the grid is the one the card freezes"""
    assert L.LEVERS["STR_label"]["s"] == (0.95, 0.90, 0.85, 0.80)
    assert L.LEVERS["OFF_label"]["ramp"] == (0.2, 1.0, 2.0)
    for c in (5, 10, 20):
        for m in (5, 10, 20):
            assert L.LEVERS[f"D_c{c}_m{m:02d}"] == {"kind": "D", "cap": c, "margin": m / 100}
            assert L.LEVERS[f"L1_c{c}_m{m:02d}"] == {"kind": "L1", "cap": c, "margin": m / 100}


# ------------------------------------------------------------------------------------------------ the certificate refuses an NC drop (and the other clauses)
REF = {"nc": 1.0, "dac": 0.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0, "ep": 0.0, "pdms": 0.0}
GOOD = {"nc": 1.0, "dac": 1.0, "ddc": 1.0, "ttc": 0.0, "c": 0.0, "ep": 0.5, "pdms": 0.4}


def test_certificate_accepts_good_and_ttc_c_free():
    assert L.cert_ok(GOOD, REF)
    assert not L.cert_ok(GOOD, REF, L.CERT_FULL)           # the FULL certificate also requires TTC / C not worse


@pytest.mark.parametrize("change", [{"nc": 0.0}, {"nc": 0.5}, {"dac": 0.0}, {"ddc": 0.5}, {"pdms": 0.0}])
def test_certificate_refuses(change):
    assert not L.cert_ok(dict(GOOD, **change), REF)


def test_certificate_nc_below_one_base():
    ref = dict(REF, nc=0.5)
    assert L.cert_ok(dict(GOOD, nc=0.5), ref)               # equal to a base NC of 0.5 passes
    assert not L.cert_ok(dict(GOOD, nc=0.0), ref)


def test_certificate_equals_n6_card():
    """the local certificate is the TF-N6 N6T card: DAC 1, NC and DDC clauses on, TTC/C off, PDMS strict, over a clause grid"""
    import itertools
    N6T = {"dac": 1.0, "nc": True, "ddc": True, "ttc": False, "c": False, "pdms_gt": True}      # copied from n6_nt_build.CARD6["N6T"]
    assert L.CERT_TOL == N6T
    for nc, ddc, dac, pd in itertools.product((0.0, 0.5, 1.0), (0.5, 1.0), (0.0, 1.0), (0.0, 0.3)):
        m = dict(GOOD, nc=nc, ddc=ddc, dac=dac, pdms=pd)
        assert L.cert_ok(m, REF) == (dac == 1.0 and nc >= 1.0 and ddc >= 1.0 and pd > 0.0)


# ------------------------------------------------------------------------------------------------ REAL scene: reachability + regression arms
@pytest.fixture(scope="module")
def scenes():
    import n4_geom as G
    z = np.load(PLANS, allow_pickle=False)
    ix = {str(t): i for i, t in enumerate(z["tokens"])}
    rolls = {}
    for ln in open(ROLLS, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            if r.get("token") in REAL_TOKENS and "traj20" in r:
                rolls[r["token"]] = np.asarray(r["traj20"], float)
    by = {}
    for t in REAL_TOKENS:
        by.setdefault(str(z["logs"][ix[t]]), []).append(t)
    out = {}
    for lg, tok, cx in G.contexts(by):
        label = np.asarray(z["plans20"][ix[tok]], float)
        out[tok] = (cx, rolls[tok], label, cx.score([label])[0][0])
    return out


@pytest.mark.real
def test_real_base_is_dac0(scenes):
    for tok, (cx, t20, label, ref) in scenes.items():
        assert ref["dac"] == 0.0


@pytest.mark.real
def test_real_zero_lever_is_the_base_plan(scenes):
    """a zero offset / s = 1 stretch scores to exactly the base label's metrics (identity through the real scorer)"""
    for tok, (cx, t20, label, ref) in scenes.items():
        for name in ("STR_label", "OFF_label"):
            params = dict(L.LEVERS[name])
            params["s" if name == "STR_label" else "dy"] = (1.0,) if name == "STR_label" else (0.0,)
            c = L.run_lever(name, params, cx, t20, label, ref)
            assert c and all(x["m"] == ref for x in c)
            assert all(np.array_equal(x["plan"], label) for x in c)


@pytest.mark.real
def test_real_determinism_two_runs_bit_identical(scenes):
    cx, t20, label, ref = scenes[REAL_TOKENS[0]]
    for name in ("L1_c5_m05", "STR_target", "OFF_label"):
        a = L.run_lever(name, L.LEVERS[name], cx, t20, label, ref)
        b = L.run_lever(name, L.LEVERS[name], cx, t20, label, ref)
        assert len(a) == len(b) > 0
        assert all(np.array_equal(x["plan"], y["plan"]) and x["m"] == y["m"] for x, y in zip(a, b))


@pytest.mark.real
def test_real_cap_prefix_property(scenes):
    """cap 5 candidates are exactly the first candidates of the cap 20 run (so the analysis may read cap 5 / 10 as a prefix of cap 20)"""
    cx, t20, label, ref = scenes[REAL_TOKENS[0]]
    for kind in ("D", "L1"):
        a = L.run_lever("x", L.LEVERS[f"{kind}_c5_m05"], cx, t20, label, ref)
        b = L.run_lever("x", L.LEVERS[f"{kind}_c20_m05"], cx, t20, label, ref)
        assert len(a) <= len(b)
        assert all(np.array_equal(x["plan"], y["plan"]) and x["m"] == y["m"] for x, y in zip(a, b))


@pytest.mark.real
def test_real_regression_arms_rescue_less(scenes):
    """each lever with its effect INVERTED (shifts sign-flipped; speed s -> 2 - s, i.e. faster) rescues fewer tokens than the true lever on these real tokens.
    (MEASURED, see DEV_N7.md: a faster plan is NOT impossible to certify on every token - 85e3 certifies at s > 1 and at no s < 1 - so the arm is aggregate, not per token.)"""
    for name in ("STR_label", "D_c5_m20", "L1_c5_m05"):
        n_true = n_bad = 0
        for tok, (cx, t20, label, ref) in scenes.items():
            n_true += int(any(L.cert_ok(x["m"], ref) for x in L.run_lever(name, L.LEVERS[name], cx, t20, label, ref, sign=1.0)))
            n_bad += int(any(L.cert_ok(x["m"], ref) for x in L.run_lever(name, L.LEVERS[name], cx, t20, label, ref, sign=-1.0)))
        print(name, "tokens rescued: true", n_true, "inverted", n_bad)
        if name != "L1_c5_m05":       # L1 inverted rescues the same 1 token (MEASURED, see DEV_N7.md): reported, not asserted; the pinned n5_l1 test covers its 0-of-129 sign flip
            assert n_true >= 1 and n_bad < n_true


@pytest.mark.real
def test_real_regression_arm_faster_does_not_rescue_slowdown_tokens(scenes):
    """on the two tokens the mild slow-down rescues (36c4 at s .85, fc61 at s .90) the faster plans s = 1.05 .. 1.2 are refused by the certificate"""
    for tok in ("36c4507970805f49", "fc61cadc28715436"):
        cx, t20, label, ref = scenes[tok]
        slow, _ = cx.score([L.time_stretch(label, s) for s in L.LEVERS["STR_label"]["s"]])
        assert any(L.cert_ok(m, ref) for m in slow)
        fast, _ = cx.score([L.time_stretch(label, s) for s in (1.05, 1.1, 1.2)])
        assert not any(L.cert_ok(m, ref) for m in fast)


@pytest.mark.real
def test_real_offset_grid_is_two_sided_so_signflip_is_vacuous_and_zero_offset_rescues_nothing(scenes):
    """OFF_label's grid contains +dy and -dy, so the sign-flipped arm scores the SAME plans (it is no regression arm: said plainly).
    The regression arm for this lever is the zero-magnitude grid (dy = 0): it must rescue 0 tokens (it re-scores the DAC = 0 base label)."""
    zero = dict(L.LEVERS["OFF_label"], dy=(0.0,))
    for tok, (cx, t20, label, ref) in scenes.items():
        a = L.run_lever("OFF_label", L.LEVERS["OFF_label"], cx, t20, label, ref, sign=1.0)
        b = L.run_lever("OFF_label", L.LEVERS["OFF_label"], cx, t20, label, ref, sign=-1.0)
        key = lambda c: sorted(x["plan"].tobytes() for x in c)           # noqa: E731
        assert key(a) == key(b)
        z = L.run_lever("OFF_label", zero, cx, t20, label, ref)
        assert z and not any(L.cert_ok(x["m"], ref) for x in z)

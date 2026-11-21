"""Known-answer and mutation tests of the EXEC-STANDSTILL reader's pure layer (exec_read.py) and of the gate checks (gates_exec.py).
Run:  python -I tests/test_exec_read.py      (plain asserts; also collected by pytest)
"""
import importlib.util
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
CODE = os.path.join(os.path.dirname(HERE), "code").replace("\\", "/")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


X = _load("exec_read", f"{CODE}/exec_read.py")
G = _load("gates_exec", f"{CODE}/gates_exec.py")
GD = _load("gapdecomp", "E:/Projects/TanitAD-artifacts/dagland/gapdecomp/gapdecomp.py")


def ps(e0, d0, lo0, hi0, e1, d1, lo1, hi1):
    return {0: {"E_base": e0, "dE": d0, "lo": lo0, "hi": hi0}, 1: {"E_base": e1, "dE": d1, "lo": lo1, "hi": hi1}}


# ------------------------------------------------------------------------------------------------ the committed outcome table
def test_outcome_table():
    assert X.outcome(ps(0.05, -0.04, -0.05, -0.03, 0.06, -0.05, -0.06, -0.04))["outcome"] == 1                 # both separated-negative, both >= half
    assert X.outcome(ps(0.05, -0.025, -0.03, -0.02, 0.06, -0.03, -0.04, -0.02))["outcome"] == 1                # exactly half counts (>=)
    assert X.outcome(ps(0.05, -0.02, -0.03, -0.01, 0.06, -0.05, -0.06, -0.04))["outcome"] == 2                 # both separated-negative, one < half
    assert X.outcome(ps(0.05, -0.04, -0.05, -0.03, 0.06, -0.01, -0.03, 0.01))["outcome"] == 2                  # separated at one seed only
    assert X.outcome(ps(0.05, -0.04, -0.05, 0.01, 0.06, -0.01, -0.03, 0.01))["outcome"] == 3                   # separated at neither
    assert X.outcome(ps(0.05, 0.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0))["outcome"] == 3                             # degenerate zero interval is NOT separated
    o = X.outcome(ps(0.05, 0.03, 0.02, 0.04, 0.06, -0.05, -0.06, -0.04))                                        # a separated-positive dE: outside the table
    assert o["outcome"] is None and o["label"].startswith("UNCOMMITTED")
    o = X.outcome(ps(-0.01, -0.04, -0.05, -0.03, 0.06, -0.05, -0.06, -0.04))                                    # E_base <= 0: 0.5 E_base undefined
    assert o["outcome"] is None and o["label"].startswith("UNCOMMITTED")


def test_outcome_mutations_are_detected():
    """Deliberate regressions of outcome(): each mutant must disagree with the real function on at least one vector of the table."""
    vecs = [ps(0.05, -0.04, -0.05, -0.03, 0.06, -0.05, -0.06, -0.04), ps(0.05, -0.02, -0.03, -0.01, 0.06, -0.05, -0.06, -0.04),
            ps(0.05, -0.04, -0.05, -0.03, 0.06, -0.01, -0.03, 0.01), ps(0.05, -0.04, -0.05, 0.01, 0.06, -0.01, -0.03, 0.01), ps(0.05, 0.03, 0.02, 0.04, 0.06, -0.05, -0.06, -0.04)]

    def mutant(kind):
        def f(p):
            q = {s: dict(v) for s, v in p.items()}
            if kind == "sign":                      # reads lo < 0 as separated-negative
                for v in q.values():
                    v["hi"], v["lo"] = -v["lo"], -v["hi"]
            if kind == "half_to_full":              # bar 1.0 E_base instead of 0.5
                for v in q.values():
                    v["dE"] = v["dE"] / 2
            return X.outcome(q)["outcome"]
        return f
    real = [X.outcome(v)["outcome"] for v in vecs]
    for kind in ("sign", "half_to_full"):
        assert [mutant(kind)(v) for v in vecs] != real, kind


# ------------------------------------------------------------------------------------------------ the estimator
def _toy(n_ep=40, per=6, seed=1):
    rng = np.random.default_rng(seed)
    eid = np.repeat(np.arange(n_ep), per).astype(str)
    cls = rng.integers(0, 4, n_ep * per).astype(np.int8)
    a = rng.normal(0.05, 0.1, n_ep * per)
    return eid, cls, a


def test_delta_e_known_answers():
    eid, cls, a = _toy()
    r0 = X.delta_e(GD, a, a, eid, cls, 200)
    assert r0["delta"] == 0.0 and r0["lo"] == 0.0 and r0["hi"] == 0.0 and not r0["separated"]                    # identical excess: dE is exactly 0 with CI [0, 0]
    r1 = X.delta_e(GD, a - 0.03, a, eid, cls, 200)
    assert abs(r1["delta"] + 0.03) < 1e-12 and r1["separated"] and r1["hi"] < 0                                   # a planted constant shift is recovered and is separated-negative
    b = a + np.random.default_rng(2).normal(0, 0.05, a.shape)
    r2 = X.delta_e(GD, b, a, eid, cls, 200)
    assert abs(r2["delta"] - float(np.mean(b - a))) < 1e-12                                                       # the point estimate is the plain mean of the paired difference
    na = a.copy()
    na[5] = np.nan
    r3 = X.delta_e(GD, na - 0.03, a, eid, cls, 200)
    assert r3["n_windows"] == len(a) - 1 and abs(r3["delta"] + 0.03) < 1e-12                                      # a NaN window is dropped from the pair, not zero-filled


def test_delta_e_equals_difference_of_E_when_valid_sets_coincide():
    eid, cls, a = _toy()
    b = a + np.random.default_rng(3).normal(0, 0.05, a.shape)
    p_base = a * 0.5
    p_hold = b * 0.25
    e_base = GD.decompose(a, p_base, eid, cls, "masked", 200)["delta"]
    e_hold = GD.decompose(b, p_hold, eid, cls, "masked", 200)["delta"]
    de = X.delta_e(GD, b - p_hold, a - p_base, eid, cls, 200)["delta"]
    assert abs(de - (e_hold - e_base)) < 1e-12


def test_estimator_mutation_detected():
    """A reader that pooled the windows across executors without the pairing (mean of hold minus mean of base on shuffled windows) gives another interval."""
    eid, cls, a = _toy()
    rng = np.random.default_rng(4)
    b = a - 0.02 + rng.normal(0, 0.001, a.shape)
    paired = X.delta_e(GD, b, a, eid, cls, 400)
    perm = rng.permutation(len(a))
    unpaired = X.delta_e(GD, b, a[perm], eid, cls, 400)                    # the pairing destroyed
    assert (paired["hi"] - paired["lo"]) < 0.5 * (unpaired["hi"] - unpaired["lo"])


# ------------------------------------------------------------------------------------------------ the diagnostic
def test_diag_share_known_answers():
    n = 10
    ch = np.zeros((n, 30))
    ch[0, 3] = np.radians(100.0)               # flagged (row 4)
    ch[1, 24] = np.radians(100.0)              # a spike at row 25: outside rows 1..20, inside 1..30
    ch[2, 5] = np.radians(59.0)                # below the threshold
    ch[3, 5] = -np.radians(61.0)               # negative yaw rate counts by magnitude
    mask = np.ones(n, bool)
    s20, s30 = X.diag_share(ch, mask, 20), X.diag_share(ch, mask, 30)
    assert s20["n_flagged"] == 2 and s30["n_flagged"] == 3
    half = X.diag_share(ch, np.arange(n) < 2, 20)
    assert half["n"] == 2 and half["n_flagged"] == 1 and half["share"] == 0.5
    assert X.diag_share(ch, np.zeros(n, bool), 20)["share"] is None


def test_launch_mask():
    v0 = np.array([0.0, 0.05, 0.2, 0.0])
    g = np.zeros((4, 30, 2))
    g[:, 29, 0] = [12.0, 12.0, 12.0, 9.0]
    assert X.launch_mask(v0, g).tolist() == [True, True, False, False]


# ------------------------------------------------------------------------------------------------ the gate checks (pure layer)
def _track(steps, psis, n=60):
    t = np.zeros((1, n, 4))
    t[0, :, 3] = np.asarray(steps) / G.DT
    t[0, :, 2] = psis
    return t


def test_phys_violations_and_red_arm():
    steps = np.r_[np.full(10, 0.01), np.full(50, 0.5)]
    held = _track(steps, np.zeros(60))                                       # heading held through the slow frames
    moving_tail = held.copy()
    moving_tail[0, 10:, 2] = 0.3
    assert G.phys_violations(held, np.zeros(1), 0.1)["n_violations"] == 0
    assert G.phys_violations(moving_tail, np.zeros(1), 0.1)["n_violations"] == 0                      # a heading change on FAST frames is allowed
    swing = held.copy()
    swing[0, 4, 2] = 1.0                                                     # the base executor's behaviour: the heading follows a 1 cm tangent
    v = G.phys_violations(swing, np.zeros(1), 0.1)
    assert v["n_violations"] >= 1 and v["n_below"] == 10                                              # RED: the check sees it (and is not vacuous)
    assert G.phys_violations(held, np.array([0.7]), 0.1)["n_violations"] == 1                         # the first frame is compared with psi0, not with itself


def test_all_steps_above_and_tick0_identity():
    t = np.concatenate([_track(np.full(60, 0.5), np.zeros(60)), _track(np.r_[np.full(59, 0.5), 0.05], np.zeros(60))])
    assert G.all_steps_above(t, 0.1).tolist() == [True, False]
    fl = np.random.default_rng(0).normal(size=(5, 30, 2))
    fl2 = fl.copy()
    fl2[:, 10:] += 1.0                                                       # later knots differ (the executed history differs), tick 0 does not
    assert G.tick0_identity(fl, fl2).all()
    fl3 = fl.copy()
    fl3[2, 0, 0] += 1e-9                                                     # a changed prior output at one window
    assert (~G.tick0_identity(fl, fl3)).tolist() == [False, False, True, False, False]


def test_eq_is_bitwise_and_nan_aware():
    a = np.array([1.0, np.nan])
    assert G.eq(a, a.copy()) and not G.eq(a, np.array([1.0 + 1e-16 * 3, np.nan])) and not G.eq(a, a[:1])


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            n += 1
            print("PASS", k)
    print(f"{n} passed")

"""pg_policy_test.py -- literal-expectation unit test of pg_policy (SYNTHETIC scores only; no probe is fitted or scored here).
Expectations are hand-derived literals, never an expression over the code under test.  Includes two deliberate-regression mutants that must go RED."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pg_policy as P  # noqa: E402

# columns = v7 ids 0..7 ; LON6 = v7 (0,1,3,4,5,7) ; columns 2 and 6 are distractors that must be ignored (col 6 carries a huge value)
RAW = np.array([
    [0.1, 0.9, 0.0, 2.0, 1.0, 0.5, 9.0, -1.0],   # w0: argmax BRAKE_TO(v3) -> group D ; best CRUISE-group class = CRUISE (v1)
    [0.2, 3.0, 0.0, 1.0, 1.5, 0.3, 9.0, -2.0],   # w1: argmax CRUISE(v1)   -> group C ; best DECEL-group class  = CREEP  (v4)
    [2.0, 1.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0],    # w2: argmax FOLLOW(v0)   -> group C
    [0.0, 0.0, 0.0, 0.2, 0.1, 0.1, 0.0, 5.0],    # w3: argmax ACCEL(v7)    -> group A ; best DECEL-group class = BRAKE_TO (v3)
    [0.0, 0.5, 0.0, 0.2, 3.0, 0.1, 0.0, 0.0],    # w4: argmax CREEP(v4)    -> group D
    [0.3, 0.4, 0.0, 0.1, 0.1, 0.1, 0.0, 0.2],    # w5: argmax CRUISE(v1)   -> group C
])
SCORE = np.array([-3.0, 5.0, 1.0, 0.5, 4.0, -1.0])      # top-2 = w1 (5.0), w4 (4.0)


def test_probegate_literal():
    lab, up, down, info = P.probegate(SCORE, RAW)
    assert info["k"] == 2 and info["tau"] == 4.0 and info["n"] == 6
    assert up.tolist() == [False, True, False, False, False, False]           # probe decel {w1,w4}; head D {w0,w4}  -> up = w1
    assert down.tolist() == [True, False, False, False, False, False]         # -> down = w0
    assert lab.tolist() == [1, 4, -100, -100, -100, -100]                     # w0 -> CRUISE(v1); w1 -> CREEP(v4); everything else NATIVE


def test_group_a_up_flip_picks_decel_group_best():
    s = np.array([-3.0, -9.0, -9.0, 7.0, 4.0, -9.0])                          # top-2 = w3 (A, 7.0), w4 (D, 4.0): w3 is an up-flip, w0 (D) a down-flip
    lab, up, down, _ = P.probegate(s, RAW)
    assert up.tolist() == [False, False, False, True, False, False] and down.tolist() == [True, False, False, False, False, False]
    assert lab.tolist() == [1, -100, -100, 3, -100, -100]                     # w3 best DECEL-group class = BRAKE_TO (v3), distractor col 6 ignored


def test_tie_break_is_by_window_index():
    s = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])                              # all tied: top-2 = windows 0,1 (index order)
    lab, up, down, info = P.probegate(s, RAW)
    assert up.tolist() == [False, True, False, False, False, False]           # probe decel {w0,w1}; head D {w0,w4}: w1 up, w4 down
    assert down.tolist() == [False, False, False, False, True, False]
    assert lab.tolist() == [-100, 4, -100, -100, 1, -100]                     # w4 best CRUISE-group class = CRUISE (v1, 0.5 > 0.0)


def test_equal_marginal_means_equal_flip_counts():
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(500, 8))
    s = rng.normal(size=500)
    _, up, down, info = P.probegate(s, raw)
    assert int(up.sum()) == int(down.sum())


def test_randgate_properties():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(400, 8))
    s = rng.normal(size=400)
    labp, upp, dnp, _ = P.probegate(s, raw)
    labr, upr, dnr, info = P.randgate(raw, int(upp.sum()), int(dnp.sum()), seed=0)
    grp, dpick, cpick = P.head_view(raw)
    assert int(upr.sum()) == int(upp.sum()) and int(dnr.sum()) == int(dnp.sum())
    assert (grp[upr] != "D").all() and (grp[dnr] == "D").all() and not (upr & dnr).any()
    assert (labr[upr] == dpick[upr]).all() and (labr[dnr] == cpick[dnr]).all()
    assert (labr[~(upr | dnr)] == -100).all()
    labr2, upr2, _, _ = P.randgate(raw, int(upp.sum()), int(dnp.sum()), seed=0)
    assert (labr2 == labr).all()                                              # seed 0 is deterministic
    labr3, upr3, _, _ = P.randgate(raw, int(upp.sum()), int(dnp.sum()), seed=1)
    assert not (upr3 == upr).all()                                            # and the seed matters


def _expect_red(fn):
    try:
        fn()
    except AssertionError:
        return True
    return False


def test_mutants_go_red():
    # M1: a policy that forces the WRONG group on up-flips (cruise instead of decel) must fail the literal test
    orig = P.labels_from_flips
    P.labels_from_flips = lambda grp, d, c, up, down: orig(grp, c, d, up, down)
    try:
        assert _expect_red(test_probegate_literal), "mutant M1 (swapped groups) stayed GREEN"
    finally:
        P.labels_from_flips = orig
    # M2: a head view that does not ignore distractor columns (argmax over all 8) must fail the literal test
    oh = P.head_view

    def bad_head_view(raw8):
        a = np.asarray(raw8, np.float64)
        grp = np.where(a.argmax(1) == 6, "A", oh(raw8)[0])
        return grp, oh(raw8)[1], oh(raw8)[2]
    P.head_view = bad_head_view
    try:
        assert _expect_red(test_probegate_literal), "mutant M2 (distractor column read) stayed GREEN"
    finally:
        P.head_view = oh


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print("ZZPGPOLICYTESTOKZZ", len(fns))

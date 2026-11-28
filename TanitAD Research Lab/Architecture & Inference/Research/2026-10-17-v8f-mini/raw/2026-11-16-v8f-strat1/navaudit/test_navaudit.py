"""test_navaudit.py -- a mutation per classification rule goes RED; the wide wrapper changes only the token; the future-read check catches a future read.
Run: C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_navaudit.py   (CPU)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navaudit as A  # noqa: E402

B = A.B


def case(**kw):
    c = dict(flag=0, unusable=False, side=0, lab=1, d=float("inf"), dann=50.0, later_k=None)
    c.update(kw)
    return c


EXPECT = [
    (case(flag=2), "MATCH-FAILED"),
    (case(unusable=True, side=1), "MATCH-FAILED"),
    (case(side=1, lab=1, d=70.0, dann=50.0), "ANNOUNCED-LATE-CUT"),
    (case(side=2, lab=2, d=70.0, dann=50.0), "ANNOUNCED-LATE-CUT"),
    (case(side=0, later_k=12), "ANNOUNCED-LATE-ROW"),
    (case(side=0, later_k=None), "ROUTE-LACKS-TURN"),
    (case(side=2, lab=1, d=70.0, dann=50.0), "OTHER"),
    (case(side=1, lab=1, d=40.0, dann=50.0), "OTHER"),            # unreachable on real data (the text would name the turn); pins the cut rule
]


def mismatches(rules):
    return [(c, e) for c, e in EXPECT if A.classify(c, rules) != e]


def mutant(**over):
    r = dict(A.DEFAULT_RULES)
    r.update(over)
    return r


def test_default_rules_match_expectations():
    assert mismatches(A.DEFAULT_RULES) == []


def test_mutation_failed_ignores_flag():
    assert mismatches(mutant(failed=lambda c: c["unusable"]))


def test_mutation_failed_ignores_unusable():
    assert mismatches(mutant(failed=lambda c: c["flag"] != 0))


def test_mutation_cut_side_not_checked():
    assert mismatches(mutant(cut=lambda c: c["d"] > c["dann"]))


def test_mutation_cut_inside_cut_counts():
    assert mismatches(mutant(cut=lambda c: c["side"] == c["lab"]))


def test_mutation_row_later_ignores_later_k():
    assert mismatches(mutant(row_later=lambda c: c["side"] == 0))


def test_mutation_row_later_never_fires():
    assert mismatches(mutant(row_later=lambda c: False))


def test_mutation_lacks_never_fires():
    assert mismatches(mutant(lacks=lambda c: False))


def test_mutation_lacks_swallows_other_side():
    assert mismatches(mutant(lacks=lambda c: c["later_k"] is None))


# ---------------------------------------------------------------------------------------------- builder wrapper on a synthetic clip
def synth(dist=70.0, n=60, v=10.0):
    vocab = np.array(["UNPARSED", "CONTINUE", "TURN_L", "TURN_R"])
    tr = np.zeros((2 * n, 4))
    tr[:, 0] = np.arange(2 * n) * v * 0.1
    tr[:, 3] = v
    tab = {"vocab": vocab, "eval_cls_id": np.full(n, 2), "eval_dist_m": np.full(n, dist), "eval_flag": np.zeros(n, np.uint8), "eval_age_s": np.zeros(n)}
    J = {"clip": np.array(["c0"]), "off": np.array([0]), "n": np.array([n]), "now_s": np.arange(n) * 0.2}
    return J, tab, {"c0": tr}


def test_wide_changes_only_the_token():
    J, tab, tracks = synth(70.0)
    n30 = A.build_variant(J, tab, "eval", tracks, *A.NARROW)
    w60 = A.build_variant(J, tab, "eval", tracks, *A.WIDE)
    assert (A.B.D_ANN_MIN, A.B.D_ANN_PER_MS) == (30.0, 6.0)            # restored
    assert n30[0][5, 0] == 1 and w60[0][5, 1] == 1                      # FOLLOW -> TURN_L at d 70 between the cuts 60 and 80
    side = np.full(60, 1)
    v = np.full(60, 10.0)
    p = A.diff_proof(n30[0], w60[0], n30[1], w60[1], np.zeros(60), side, np.maximum(30, 6 * v), np.maximum(60, 8 * v), n30[1].astype(float))
    assert p["changed_equals_expected_set"] and p["ch3_ch4_nav_d_identical"] and p["every_change_is_FOLLOW_to_side"]
    assert p["n_steps_changed"] == 60


def test_mutation_diff_proof_catches_a_distance_change():
    J, tab, tracks = synth(70.0)
    n30 = A.build_variant(J, tab, "eval", tracks, *A.NARROW)
    w60 = A.build_variant(J, tab, "eval", tracks, *A.WIDE)
    bad = w60[0].copy()
    bad[:, 3] += 0.01
    side = np.full(60, 1)
    v = np.full(60, 10.0)
    p = A.diff_proof(n30[0], bad, n30[1], w60[1], np.zeros(60), side, np.maximum(30, 6 * v), np.maximum(60, 8 * v), n30[1].astype(float))
    assert not p["ch3_ch4_nav_d_identical"]


def test_mutation_diff_proof_catches_a_wrong_step_set():
    J, tab, tracks = synth(70.0)
    n30 = A.build_variant(J, tab, "eval", tracks, *A.NARROW)
    w60 = A.build_variant(J, tab, "eval", tracks, *A.WIDE)
    bad = w60[0].copy()
    bad[0] = n30[0][0]                                                 # one step left unchanged
    side = np.full(60, 1)
    v = np.full(60, 10.0)
    p = A.diff_proof(n30[0], bad, n30[1], w60[1], np.zeros(60), side, np.maximum(30, 6 * v), np.maximum(60, 8 * v), n30[1].astype(float))
    assert not p["changed_equals_expected_set"]


def test_builder_reads_no_future_track_rows():
    J, tab, tracks = synth(40.0)
    r = A.future_independence(J, tab, "eval", tracks, n_samples=40)
    assert r["n_differ_when_future_track_rows_are_garbage"] == 0 and r["n_with_side_token"] > 0


def test_mutation_future_read_goes_red():
    J, tab, tracks = synth(40.0)

    def peeking(cls, dist, flag, age, now_s, tr, side):
        tr2 = np.concatenate([np.asarray(tr)[10:], np.asarray(tr)[-10:]])      # step j now reads track rows 10 ahead
        return B.clip_nav(cls, dist, flag, age, now_s, tr2, side)
    r = A.future_independence(J, tab, "eval", tracks, n_samples=40, clip_nav=peeking)
    assert r["n_differ_when_future_track_rows_are_garbage"] > 0

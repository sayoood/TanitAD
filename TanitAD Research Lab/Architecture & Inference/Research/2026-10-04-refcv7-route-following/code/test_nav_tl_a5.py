"""Known-value tests for SPEC_ADDENDUM_A5's nav_tl (code/nav_tl_a5.py).

Every expectation is a LITERAL derived by hand from A5's definition (CLAUDE.md: a cross-check must be derived
independently of the value it checks), never an expression over the code under test. Each rule that matters also has a
deliberate-regression ("mutation") arm that must change the answer.

Run:  python test_nav_tl_a5.py        (or pytest)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nav_tl_a5 as N  # noqa: E402
import route_metrics as rm  # noqa: E402

LEFT, RIGHT, FOLLOW = 1, -1, 0
# the A5 brief's synthetic record: ONE NAV_TURN_L entry, t_start 3.0 s, t_end 6.0 s (anchor-relative)
ONE_TURN_L = [{"token": "NAV_TURN_L", "t_start_s": 3.0, "t_end_s": 6.0}]


# ---------------------------------------------------------------------------------------------- nav_tl definition
def test_one_turn_left_known_values():
    assert N.nav_tl_side(ONE_TURN_L, 0.0) == LEFT          # starts 3 s ahead (<= 6)
    assert N.nav_tl_side(ONE_TURN_L, 4.0) == LEFT          # under way (3 <= 4 < 6)
    assert N.nav_tl_side(ONE_TURN_L, -4.0) == FOLLOW       # starts 7 s ahead (> 6)
    assert N.nav_tl_side(ONE_TURN_L, 7.0) == FOLLOW        # finished (t_end 6 <= 7)


def test_reasons_are_the_literal_cases():
    assert N.nav_tl_state(ONE_TURN_L, 0.0) == (LEFT, "ahead_within_H")
    assert N.nav_tl_state(ONE_TURN_L, 4.0) == (LEFT, "under_way")
    assert N.nav_tl_state(ONE_TURN_L, -4.0) == (FOLLOW, "beyond_H")
    assert N.nav_tl_state(ONE_TURN_L, 7.0) == (FOLLOW, "no_entry_left")


def test_horizon_boundary_is_inclusive_and_end_boundary_is_exclusive():
    # start - t_rel = 6.0 exactly -> included (<= H); 6.0001 -> excluded
    assert N.nav_tl_side(ONE_TURN_L, -3.0) == LEFT
    assert N.nav_tl_side(ONE_TURN_L, -3.0001) == FOLLOW
    # t_end > t_rel is strict: at t_rel == t_end the turn is finished
    assert N.nav_tl_side(ONE_TURN_L, 5.9999) == LEFT
    assert N.nav_tl_side(ONE_TURN_L, 6.0) == FOLLOW


def test_mutation_H_1e9_changes_the_answer_at_minus_4():
    """The horizon is load-bearing: with H = 1e9 the 7-s-ahead turn is no longer filtered out."""
    assert N.nav_tl_side(ONE_TURN_L, -4.0, H=6.0) == FOLLOW
    assert N.nav_tl_side(ONE_TURN_L, -4.0, H=1e9) == LEFT
    assert N.nav_tl_side(ONE_TURN_L, -4.0, H=1e9) != N.nav_tl_side(ONE_TURN_L, -4.0, H=6.0)
    # ... and H does NOT resurrect a FINISHED turn (the end rule is independent of the horizon)
    assert N.nav_tl_side(ONE_TURN_L, 7.0, H=1e9) == FOLLOW


def test_sensitivity_horizon_8_includes_a_7s_lead_but_not_a_9s_lead():
    assert N.nav_tl_side(ONE_TURN_L, -4.0, H=N.H_SENS) == LEFT       # 7 s ahead <= 8
    assert N.nav_tl_side(ONE_TURN_L, -6.0, H=N.H_SENS) == FOLLOW     # 9 s ahead > 8


def test_right_turn_maps_to_minus_one_and_follow_token_never_activates():
    right = [{"token": "NAV_TURN_R", "t_start_s": 3.0, "t_end_s": 6.0}]
    assert N.nav_tl_side(right, 0.0) == RIGHT
    follow = [{"token": "NAV_FOLLOW_ROAD", "t_start_s": 0.0, "t_end_s": 30.0}]
    for t_rel in (-6.0, -0.5, 0.0, 4.0, 12.0):
        assert N.nav_tl_state(follow, t_rel) == (FOLLOW, "follow_token")


def test_first_unfinished_entry_rules_and_a_far_first_entry_does_not_skip_to_the_next():
    seq = [{"token": "NAV_TURN_R", "t_start_s": 2.0, "t_end_s": 4.0},
           {"token": "NAV_TURN_L", "t_start_s": 9.0, "t_end_s": 12.0}]
    assert N.nav_tl_side(seq, 1.0) == RIGHT        # first entry 1 s ahead
    assert N.nav_tl_side(seq, 2.5) == RIGHT        # first entry under way
    assert N.nav_tl_side(seq, 5.0) == LEFT         # first finished -> the second is 4 s ahead
    assert N.nav_tl_side(seq, 0.0) == RIGHT        # first entry 2 s ahead
    # the first UNFINISHED entry is 7 s ahead: nav_tl is follow, it does not look past it
    far = [{"token": "NAV_TURN_R", "t_start_s": 8.0, "t_end_s": 10.0},
           {"token": "NAV_TURN_L", "t_start_s": 11.0, "t_end_s": 12.0}]
    assert N.nav_tl_side(far, 1.0) == FOLLOW


# ---------------------------------------------------------------------------------------------- the clock
def test_clock_known_values():
    # t_now = g0 + (t + W - 1 + n_stack - 1) * dt with W = 8, n_stack - 1 = 2
    assert abs(N.t_now_raw(0, 0.1, 0.1) - 1.0) < 1e-12              # 0.1 + 9 * 0.1
    assert abs(N.t_now_raw(10, 0.0, 0.1) - 1.9) < 1e-12             # (10 + 7 + 2) * 0.1
    assert abs(N.t_now_raw(0, 0.0, 0.1006666) - 0.9059994) < 1e-12  # 9 * 0.1006666
    # the trainer's fallback for a clip with no sidecar row: grid_start 0.0, nominal dt 0.1
    assert N.clock_for(123456789, {}) == (0.0, 0.1, "nominal_dt")
    assert N.clock_for(7, {7: (0.25, 0.1007)}) == (0.25, 0.1007, "sidecar")


def test_mutation_clock_without_the_raw_offset_is_off_by_exactly_two_rows():
    ok = N.t_now_raw(40, 0.113, 0.1006666)
    no_off = N.t_now_raw(40, 0.113, 0.1006666, raw_offset=0)
    assert abs((ok - no_off) - 2 * 0.1006666) < 1e-12
    assert abs(N.t_now_raw(40, 0.113, 0.1006666, window=1) - ok) > 0.7        # W is load-bearing too


def test_sha12_is_the_sha256_prefix():
    assert N.sha12("abc") == "ba7816bf8f01"          # the FIPS-180 "abc" test vector, first 12 hex chars
    assert 0 <= N.stable_sid("abc") < 2 ** 63


# ---------------------------------------------------------------------------------------------- T2c derangement
def test_derangement_is_a_fixed_point_free_deterministic_bijection():
    for n in (2, 3, 5, 139):
        keys = [f"k{i:03d}" for i in range(n)]
        d1, d2 = N.derangement(keys, seed=0), N.derangement(list(reversed(keys)), seed=0)
        assert d1 == d2                                             # input order does not matter
        assert sorted(d1.values()) == sorted(keys)                  # a bijection
        assert all(k != v for k, v in d1.items())                   # no fixed point
    assert N.derangement([f"k{i}" for i in range(20)], seed=0) != N.derangement([f"k{i}" for i in range(20)], seed=1)


# ---------------------------------------------------------------------------------------------- the arm rules
def _arc(theta_end, v=6.0):
    """8-slot path whose slot-50 -> slot-60 chord heads at theta_end (rad) and which moves v m/s."""
    t = np.array(rm.HORIZONS) * 0.1
    P = np.zeros((8, 2))
    P[:, 0] = v * t * np.cos(theta_end)
    P[:, 1] = v * t * np.sin(theta_end)
    return P


def _synthetic():
    # candidates: 0 straight, 1 left (+0.6 rad), 2 right (-0.6 rad), 3 straight
    fan = np.stack([_arc(0.0), _arc(0.6), _arc(-0.6), _arc(0.0)])[None]
    th = rm.terminal_heading(fan)
    dirc = rm.dir_class(th)
    assert list(dirc[0]) == [0, 1, -1, 0]
    s_e9 = np.array([[0.1, 0.0, -0.2, 0.9]])                          # E9 pick = candidate 3 (a straight one)
    z = {"s_e9": s_e9, "s_core": s_e9.copy(), "e9_graft": np.zeros_like(s_e9)}
    d = {"th_c": th, "dir_c": dirc, "reach": np.ones((1, 4), bool), "navc_term": np.zeros((1, 4))}
    return z, d


def test_T2_filters_to_the_nav_tl_side_and_falls_back_to_V0():
    z, d = _synthetic()
    assert int(z["s_e9"].argmax()) == 3                                # V0
    assert int(N.pick_T2(z, d, np.array([LEFT]))[0]) == 1              # the only left candidate
    assert int(N.pick_T2(z, d, np.array([RIGHT]))[0]) == 2             # the only right candidate
    assert int(N.pick_T2(z, d, np.array([FOLLOW]))[0]) == 3            # follow -> untouched V0
    # empty survivor set -> the unrestricted E9 argmax (= V0): no left candidate exists
    z2, d2 = _synthetic()
    d2["th_c"] = np.array([[0.0, 0.0, -0.6, 0.0]])
    assert int(N.pick_T2(z2, d2, np.array([LEFT]))[0]) == 3


def test_T2_mutation_a_clip_constant_token_would_damage_a_straight_part():
    """The reason A5 exists: the SAME rule fed a (wrong-in-time) 'left' on a window whose best candidate is straight
    moves the pick off it. nav_tl = follow keeps it."""
    z, d = _synthetic()
    assert int(N.pick_T2(z, d, np.array([LEFT]))[0]) != int(N.pick_T2(z, d, np.array([FOLLOW]))[0])


def test_T3_is_V0_when_follow_and_k_is_load_bearing():
    z, d = _synthetic()
    gate = 0.1625
    assert int(N.pick_T3(z, d, np.array([FOLLOW]), gate)[0]) == 3
    # nav_tl = left: the left candidate gains 10 * 0.1625 = 1.625 > the 0.9 - 0.0 margin -> picked
    assert int(N.pick_T3(z, d, np.array([LEFT]), gate)[0]) == 1
    # mutation: k = 0 removes the term entirely -> V0 again
    assert int(N.pick_T3(z, d, np.array([LEFT]), gate, k=0)[0]) == 3
    # mutation: k = 1 (gain 0.1625) is below the margin -> V0
    assert int(N.pick_T3(z, d, np.array([LEFT]), gate, k=1)[0]) == 3


def test_T3_replaces_the_clip_token_term_T3b_keeps_it():
    """Reading (a) vs (b) of 'V3 with the predicate recomputed': the shipped term built from a RIGHT clip token is
    removed under (a), kept under (b)."""
    z, d = _synthetic()
    gate = 0.1625
    clip_term = N.navc_term_from(d, z, np.array([RIGHT]), gate)          # what the model added for a right clip token
    assert np.allclose(clip_term, [[0, 0, gate, 0]])
    z["s_core"] = z["s_core"] + clip_term                                 # the shipped score includes it
    z["s_e9"] = z["s_core"].copy()
    d["navc_term"] = clip_term
    s_a = z["s_core"] - d["navc_term"] + 10 * N.navc_term_from(d, z, np.array([LEFT]), gate)
    s_b = z["s_core"] + 9 * N.navc_term_from(d, z, np.array([LEFT]), gate)
    assert abs(s_a[0, 2] - (-0.2)) < 1e-12 and abs(s_b[0, 2] - (-0.2 + gate)) < 1e-12   # cand 2: with / without the 1x term
    assert abs(s_a[0, 1] - 1.625) < 1e-12 and abs(s_b[0, 1] - (9 * gate)) < 1e-12       # cand 1


def test_navc_term_is_gate_times_predicate_and_zero_on_follow():
    z, d = _synthetic()
    t = N.navc_term_from(d, z, np.array([LEFT]), 0.5)
    assert np.allclose(t, [[0.0, 0.5, 0.0, 0.0]])
    assert np.allclose(N.navc_term_from(d, z, np.array([FOLLOW]), 0.5), 0.0)


# ---------------------------------------------------------------------------------------------- a real-data known value
def test_K1_row_from_the_replay_bank_if_the_data_is_present():
    """bank row (clip 693335b810c3, t_start_row 0) has t_label_s 1.1288 (4 dp). Skipped when the inputs are absent."""
    lab = Path("D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz")
    sc = Path("D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl")
    if not (lab.exists() and sc.exists()):
        return
    recs = N.load_records(lab)
    table = N.read_sidecar(sc)
    g0, dt, src = N.clock_for(recs["693335b810c3"]["sid"], table)
    assert src == "sidecar"
    assert abs(N.t_now_raw(0, g0, dt) - 1.1288) <= 5e-5 + 1e-9
    assert abs(N.t_now_raw(1, g0, dt) - 1.2293) <= 5e-5 + 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)}/{len(fns)} passed")

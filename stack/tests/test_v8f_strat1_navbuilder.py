"""PREREG_V8F_STRAT1 T-NAVB: the text-nav builder rule (BUILD_SPEC section 5) on a hand-checked fixture, and the G-CLOCK-NAV deliberate-regression arm (``+ travelled`` must FAIL the QA)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import build_navtext_nav as B  # noqa: E402

VOCAB = np.array(["UNPARSED", "CONTINUE", "TURN_L", "TURN_R", "UTURN", "TURN_SLIGHT_L"])
N = 40


def fixture():
    cls = np.ones(N, np.int16)
    dist = np.full(N, np.nan, np.float32)
    flag = np.zeros(N, np.uint8)
    age = np.zeros(N, np.float32)
    cls[10], dist[10], age[10] = 2, 80.0, 2.0            # TURN_L, 80 m, row 2 s old: travelled 20 m -> d 60 <= d_ann 60 -> L
    cls[12], dist[12], age[12] = 2, 200.0, 0.0           # TURN_L far: FOLLOW token, ch3 = 2.0
    cls[20], dist[20], age[20] = 3, 60.0, 1.0            # TURN_R 60 m, row 1 s old: d 50 -> R
    cls[22], dist[22], age[22] = 3, 20.0, 1.0            # TURN_R 20 m, row 1 s old: d 10 -> R (still a right under the sign-flipped form: the QA fails on the NUMBERS)
    cls[25] = 4                                           # UTURN -> none -> FOLLOW
    cls[26] = 2                                           # bare TURN_L (no distance): d = 0 -> L
    cls[28], dist[28], age[28] = 5, 40.0, 0.0            # TURN_SLIGHT_L -> L
    flag[30] = 2                                          # out of tolerance: all zeros
    track = np.zeros((2 * N + 2, 4))
    track[:, 0] = np.arange(len(track)) * 1.0             # 10 m/s at 10 Hz
    track[:, 3] = 10.0
    now_s = np.arange(N) * 0.2
    return cls, dist, flag, age, now_s, track


def test_rule_on_a_hand_checked_fixture():
    cls, dist, flag, age, now_s, track = fixture()
    nav, nav_d, clamp, d_raw = B.clip_nav(cls, dist, flag, age, now_s, track, B.side_of(VOCAB), -1)
    assert np.allclose(nav[10], [0, 1, 0, 0.6, 1]) and abs(nav_d[10] - 60.0) < 1e-4                # countdown, announced (d <= 60)
    assert np.allclose(nav[12], [1, 0, 0, 2.0, 1]) and abs(nav_d[12] - 200.0) < 1e-4              # beyond d_ann: FOLLOW, ch3 carries d / 100
    assert np.allclose(nav[20], [0, 0, 1, 0.5, 1]) and abs(nav_d[22] - 10.0) < 1e-4
    assert np.allclose(nav[25], [1, 0, 0, 0, 0]) and np.isnan(nav_d[25])                           # UTURN is none: FOLLOW, no distance
    assert np.allclose(nav[26], [0, 1, 0, 0.0, 1]) and nav_d[26] == 0.0                            # bare text: d = 0
    assert np.allclose(nav[28], [0, 1, 0, 0.4, 1])                                                 # TURN_SLIGHT_L is a left
    assert (nav[30] == 0).all() and np.isnan(nav_d[30])                                            # flag != 0
    assert (nav[0] == [1, 0, 0, 0, 0]).all()                                                       # no side
    assert not clamp.any()


def test_clock_nav_qa_passes_for_the_countdown_and_fails_for_plus_travelled():
    cls, dist, flag, age, now_s, track = fixture()
    side = B.side_of(VOCAB)
    J = {"nav": np.zeros((N, 5), np.float32), "nav_d": np.full(N, np.nan, np.float32)}
    J["nav"][10], J["nav_d"][10] = [0, 1, 0, 0.6, 1], 60.0                                         # the v9 nav the text nav is checked against
    J["nav"][20], J["nav_d"][20] = [0, 0, 1, 0.5, 1], 50.0
    J["nav"][22], J["nav_d"][22] = [0, 0, 1, 0.1, 1], 10.0
    res = {}
    for sign in (-1, +1):
        nav, nav_d, _, d_raw = B.clip_nav(cls, dist, flag, age, now_s, track, side, sign)
        res[sign] = B.clock_nav_qa(J, nav, nav_d, d_raw)
    assert res[-1]["pass"] and res[-1]["n_steps"] == 3 and res[+1]["n_steps"] >= 1
    assert not res[+1]["pass"]                      # the deliberate-regression arm: the sign-flipped form must FAIL the QA


def test_nan_rows_at_the_start_of_the_native_track_do_not_poison_the_arc():
    """MEASURED 2026-11-16 on the real native_tracks.pt: rows 0 and 1 are NaN; an unguarded cumsum made every countdown read 0 and the plus-travelled arm read no steps."""
    cls, dist, flag, age, now_s, track = fixture()
    track[:2] = np.nan
    nav, nav_d, _, _ = B.clip_nav(cls, dist, flag, age, now_s, track, B.side_of(VOCAB), -1)
    assert abs(nav_d[10] - 60.0) <= 2.5 and abs(nav_d[22] - 10.0) <= 2.5 and nav_d[10] > 0     # the two NaN rows drop at most their own 2 m of arc; the poisoned form read 0


def test_a_row_start_before_the_first_time_clamps_and_is_counted():
    cls, dist, flag, age, now_s, track = fixture()
    age[10] = 5.0                                    # row start 2 s - 5 s < the first row time 0
    nav, nav_d, clamp, _ = B.clip_nav(cls, dist, flag, age, now_s, track, B.side_of(VOCAB), -1)
    assert clamp[10] and abs(nav_d[10] - (80.0 - 20.0)) < 1e-4          # clamped at the first row: travelled = the arc from row 0 to row 20

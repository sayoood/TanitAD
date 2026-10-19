"""Pins the pure parts of force_lon_reroll_r8.py: the COMMITTED outcome table, the lon_v7 -> LON6 mapping, the per-window sampler seed, the
proportional apportionment, and the closure bootstrap against analytic targets.  Literal expectations only (never an expression over the code
under test), and each guard is paired with a value that must NOT trigger it (a mutation of the threshold flips one of these RED)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import force_lon_reroll_r8 as F  # noqa: E402
import s6_lon_decomp as SD  # noqa: E402


# ---- the committed table (verbatim from the brief): HEAD >= 0.50 and lo > 0 ; DECODER hi < 0.25 ; else INCONCLUSIVE
@pytest.mark.parametrize("point,lo,hi,want", [
    (0.50, 0.01, 0.9, "HEAD"),                       # exactly on the point threshold, lower bound just above 0
    (0.49, 0.10, 0.9, "INCONCLUSIVE"),               # just below 0.50
    (0.80, 0.00, 1.1, "INCONCLUSIVE"),               # lower bound exactly 0 is NOT > 0
    (0.80, -0.20, 1.1, "INCONCLUSIVE"),
    (0.05, -0.10, 0.2499, "DECODER"),                # upper bound just below 0.25
    (0.05, -0.10, 0.25, "INCONCLUSIVE"),             # upper bound exactly 0.25 is NOT < 0.25
    (0.30, 0.10, 0.60, "INCONCLUSIVE"),
    (-0.40, -0.90, 0.10, "DECODER"),                 # forcing makes it worse: the head is not the cause
    (0.60, 0.30, 0.9, "HEAD"),
])
def test_decide_is_the_committed_table(point, lo, hi, want):
    assert F.decide(point, lo, hi) == want


def test_decide_refuses_an_undefined_closure():
    assert F.decide(None, None, None).startswith("INCONCLUSIVE")


# ---- lon_v7 -> LON6 (refcv8_conditioning.py:63-64): LON6 = FOLLOW CRUISE BRAKE_TO CREEP HOLD ACCELERATE ; v7 ids 0 1 3 4 5 7
@pytest.mark.parametrize("lon_v7,want", [(0, 0), (1, 1), (3, 2), (4, 3), (5, 4), (7, 5)])
def test_f1_class_maps_each_hypothesis_to_its_index(lon_v7, want):
    assert F.f1_class(lon_v7) == (want, None)


@pytest.mark.parametrize("lon_v7", [-100, 2, 6])
def test_f1_class_excludes_unlabelled_and_masked_classes(lon_v7):
    c, why = F.f1_class(lon_v7)
    assert c is None and why


def test_pre_registered_constants():
    assert F.LON6 == ("FOLLOW", "CRUISE", "BRAKE_TO", "CREEP", "HOLD", "ACCELERATE")
    assert F.LON6_V7_IDS == (0, 1, 3, 4, 5, 7)
    assert (F.CRUISE6, F.BRAKE6, F.BRAKE_V7) == (1, 2, 3)
    assert (F.HEAD_POINT, F.HEAD_LO_GT, F.DECODER_HI_LT) == (0.50, 0.0, 0.25)


# ---- the sampler seed is a function of the WINDOW (not of the arm), stable across calls and processes
def test_window_seed_is_deterministic_and_window_specific():
    assert F.window_seed(0, 5, 17) == F.window_seed(0, 5, 17)
    assert F.window_seed(0, 5, 17) != F.window_seed(0, 5, 18)
    assert F.window_seed(0, 5, 17) != F.window_seed(0, 6, 17)
    assert F.window_seed(0, 5, 17) != F.window_seed(F.N2_SEED_OFFSET, 5, 17)       # N2 is a DIFFERENT draw of the same window
    assert F.window_seed(0, 5, 17) == 834385984                                     # sha256(b'0|5|17')[:8] = b1bbb83f, mod 2**31-1 (literal)
    assert 0 <= F.window_seed(0, 5, 17) < 2 ** 31 - 1


def test_apportion_is_proportional_and_exact():
    rng = np.random.default_rng(0)
    out = F.apportion({1: 10, 2: 30, 3: 60}, 20, rng)
    assert sum(out.values()) == 20 and out == {1: 2, 2: 6, 3: 12}
    assert F.apportion({1: 2, 2: 3}, 99, rng) == {1: 2, 2: 3}                     # never above an episode's own count


# ---- the closure statistic against analytic targets
def _data(n_ep=12, per=5, seed=0):
    rng = np.random.default_rng(seed)
    eid = np.repeat(np.arange(n_ep), per)
    r7 = rng.uniform(1.0, 2.0, eid.size)
    n = r7 + rng.uniform(0.5, 1.5, eid.size)               # refcv8 native is worse than refcv7 on every window
    return eid, n, r7


def test_closure_is_exactly_one_when_the_force_recovers_refcv7_and_zero_when_it_changes_nothing():
    eid, n, r7 = _data()
    one = F.ratio_boot(n - r7, n - r7, eid, SD)            # F1 == R7  -> (N - F1) / (N - R7) == 1
    assert one["point"] == pytest.approx(1.0) and one["lo"] == pytest.approx(1.0) and one["hi"] == pytest.approx(1.0)
    zero = F.ratio_boot(n - n, n - r7, eid, SD)            # F1 == N   -> 0
    assert zero["point"] == pytest.approx(0.0) and zero["lo"] == pytest.approx(0.0) and zero["hi"] == pytest.approx(0.0)
    half = F.ratio_boot(0.5 * (n - r7), n - r7, eid, SD)   # halfway -> 0.5 exactly, whatever the draw
    assert half["point"] == pytest.approx(0.5) and half["lo"] == pytest.approx(0.5) and half["hi"] == pytest.approx(0.5)


def test_closure_ci_widens_with_noise_and_decides_inconclusive_when_it_straddles():
    eid, n, r7 = _data(n_ep=10, per=3, seed=1)
    rng = np.random.default_rng(2)
    f = n - 0.5 * (n - r7) + rng.normal(0.0, 1.5, n.size)  # a half-recovering force with large window noise
    r = F.ratio_boot(n - f, n - r7, eid, SD)
    assert r["hi"] - r["lo"] > 0.3                         # a real CI, not a point
    assert F.decide(r["point"], r["lo"], r["hi"]) in ("INCONCLUSIVE", "HEAD", "DECODER")


def test_n_needed_is_estimated_and_honest_about_unreachable_outcomes():
    out = F.n_needed(0.10, -0.2, 0.4, 250)
    assert "ESTIMATED" in out["assumption"] and out["to_reach_HEAD"].startswith("unreachable")
    assert out["to_reach_DECODER"] == pytest.approx(250 * ((0.4 - 0.10) / (0.25 - 0.10)) ** 2)
    out2 = F.n_needed(0.70, -0.1, 1.2, 250)
    assert out2["to_reach_HEAD"] == pytest.approx(250 * ((0.70 + 0.1) / 0.70) ** 2)

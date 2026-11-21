"""test_ship_r8_sub_cap8.py -- the Amendment 4b clamp against ANALYTIC literals (never against the code under test).
    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 test_ship_r8_sub_cap8.py
Every expectation is written as a literal from the amendment's formula  s_cap[k] = min(s8[k], s_cap[k-1] + lim*(t[k]-t[k-1])*(1-1e-6)),  s_cap[-1] = 0, t[-1] = 0,
T = (0.5, 1, 1.5, 2, 3, 4, 5, 6) s.  Includes the deliberate-regression check: lim*1.05 must BREAK the per-interval bound the same test asserts for lim."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ship_r8_sub_forward as FW  # noqa: E402

T = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
DT = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])


def test_saturated_window_is_lim_times_t():
    s8 = np.full((1, 8), 1e6)
    got = FW.cap_progress(s8, np.array([10.0], np.float32))
    want = np.array([4.999995, 9.99999, 14.999985, 19.99998, 29.99997, 39.99996, 49.99995, 59.99994])    # 10 * (1 - 1e-6) * T, literal
    assert np.allclose(got[0], want, rtol=0, atol=1e-9), got
    assert (np.diff(np.concatenate([[0.0], got[0]])) <= 10.0 * DT * (1 - 1e-6) + 1e-12).all()


def test_unclamped_when_below_cap():
    s8 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    assert FW.cap_progress(s8, np.array([100.0])).tobytes() == s8.tobytes()


def test_nonfinite_ceiling_is_unclamped_bit_equal():
    s8 = np.array([[3.0, 9.0, 20.0, 41.0, 77.0, 120.0, 190.0, 260.0]] * 4)
    lim = np.array([np.inf, np.nan, -np.inf, 1e30])
    assert FW.cap_progress(s8, lim).tobytes() == s8.tobytes()
    assert FW.cap_progress(s8, np.full(4, np.inf)).tobytes() == s8.tobytes()      # K-CAP-ID's recompute


def test_partial_clamp_literal():
    s8 = np.array([[1.0, 2.0, 3.0, 40.0, 41.0, 42.0, 43.0, 44.0]])
    got = FW.cap_progress(s8, np.array([10.0]))[0]
    # k<=2 untouched (caps 4.999995, 9.99999+..); k=3: min(40, 3 + 4.999995) = 7.999995; then +9.99999 per 1 s slot while s8 stays above;
    # k=7: the cap would be 37.999965 + 9.99999 = 47.999955 > s8[7] = 44, so the window is unclamped there
    want = np.array([1.0, 2.0, 3.0, 7.999995, 17.999985, 27.999975, 37.999965, 44.0])
    assert np.allclose(got, want, atol=1e-9), got
    assert (got <= s8[0] + 1e-12).all()


def test_zero_ceiling_stops_the_plan():
    assert (FW.cap_progress(np.full((1, 8), 5.0), np.array([0.0])) == 0.0).all()


def test_monotone_for_nonneg_lim_and_monotone_s8():
    rng = np.random.default_rng(0)
    s8 = np.maximum.accumulate(np.abs(rng.normal(8.0, 6.0, (500, 8))), axis=1)
    lim = rng.uniform(0.0, 40.0, 500)
    got = FW.cap_progress(s8, lim)
    assert (np.diff(np.concatenate([np.zeros((500, 1)), got], axis=1), axis=1) >= 0).all()
    assert (got <= s8).all()
    assert (np.diff(np.concatenate([np.zeros((500, 1)), got], axis=1), axis=1) <= lim[:, None] * DT[None] * (1 - 1e-6) + 1e-12).all()


def test_deliberate_regression_105_breaks_the_bound():
    s8 = np.full((1, 8), 1e6)
    lim = np.array([10.0])
    bad = FW.cap_progress(s8, lim, 1.05)[0]
    adv = np.diff(np.concatenate([[0.0], bad]))
    assert (adv > 10.0 * DT).all(), "lim*1.05 must advance MORE than lim*dt on every interval (so K-CAP-COMPLY can see it)"


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print("ok", k)
    print(f"ZZCAP8TESTSZZ {n} passed")

"""Sanity tests for LAL-v2 (gate G-B2): every case has an analytically known
answer, derived in-comment so a reviewer can recompute by hand. numpy-only,
no simulator, no trained model.

Runs standalone:  pytest "<this package>/tests"
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lal_v2 import (  # noqa: E402
    LAL_NO_REACTION,
    compute_lal_v2,
    decel_onset_index,
)


# --------------------------------------------------------------------------- #
# 1. Anticipating policy: smooth slowdown BEFORE line-of-sight -> LAL_v2 > 0   #
# --------------------------------------------------------------------------- #
def test_anticipating_positive_smooth_slowdown():
    # dt=0.1, T=200, cruise 16 m/s. v_ref = max(v[:60]) = 16.
    # threshold = 0.85*16 = 13.6 m/s.
    # Anticipating: hold cruise to idx 60, then linear ramp down to 4 m/s by idx 140.
    #   ramp slope = (4-16)/(140-60) = -0.15 m/s per step.
    #   v crosses 13.6 when 16 - 0.15*(i-60) <= 13.6  ->  i-60 >= 16 -> i = 76.
    #   -> t_decel_onset = 7.6 s.
    # LoS first True at idx 110 -> t_LoS = 11.0 s.  LAL_v2 = 11.0 - 7.6 = +3.4 s.
    T = 200
    v = np.full(T, 16.0)
    ramp = np.clip((np.arange(T) - 60) / 80.0, 0, 1)   # 0..1 over idx 60..140
    v = 16.0 - 12.0 * ramp                              # 16 -> 4
    los = np.zeros(T, dtype=bool); los[110:] = True
    assert decel_onset_index(v) == 76
    assert compute_lal_v2(v, los, dt=0.1) == pytest.approx(3.4, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. Reactive policy: cruise until AFTER LoS, then brake -> LAL_v2 <= 0        #
#    (this is the SC-01 failure case LAL-v1 could not separate)                #
# --------------------------------------------------------------------------- #
def test_reactive_negative_brakes_after_los():
    # cruise 16 until idx 112, then ramp down. v_ref=16, threshold=13.6.
    #   ramp slope from idx 112: (2-16)/(160-112) = -0.2917 m/s per step.
    #   crosses 13.6 when 16 - 0.2917*(i-112) <= 13.6 -> i-112 >= 8.23 -> i = 121.
    #   -> t_decel_onset = 12.1 s.  LoS at idx 110 -> t_LoS = 11.0.
    #   LAL_v2 = 11.0 - 12.1 = -1.1 s  (reactive: braked 1.1 s AFTER line-of-sight).
    T = 200
    v = np.full(T, 16.0)
    ramp = np.clip((np.arange(T) - 112) / 48.0, 0, 1)
    v = 16.0 - 14.0 * ramp
    los = np.zeros(T, dtype=bool); los[110:] = True
    assert decel_onset_index(v) == 121
    assert compute_lal_v2(v, los, dt=0.1) == pytest.approx(-1.1, abs=1e-9)


# --------------------------------------------------------------------------- #
# 3. Discrimination: anticipating > reactive on the SAME scene (the point)     #
# --------------------------------------------------------------------------- #
def test_discriminates_where_lal_v1_collapsed():
    T = 200
    los = np.zeros(T, dtype=bool); los[110:] = True
    v_anti = 16.0 - 12.0 * np.clip((np.arange(T) - 60) / 80.0, 0, 1)
    v_react = 16.0 - 14.0 * np.clip((np.arange(T) - 112) / 48.0, 0, 1)
    lal_anti = compute_lal_v2(v_anti, los, dt=0.1)
    lal_react = compute_lal_v2(v_react, los, dt=0.1)
    assert lal_anti > 0 > lal_react          # opposite signs -> clean separation
    assert lal_anti - lal_react == pytest.approx(4.5, abs=1e-9)   # 3.4 - (-1.1)


# --------------------------------------------------------------------------- #
# 4. No line-of-sight ever -> 0.0 (nothing to anticipate; matches LAL-v1)      #
# --------------------------------------------------------------------------- #
def test_no_los_returns_zero():
    T = 50
    v = np.linspace(16.0, 4.0, T)            # decelerates, but hazard never seen
    los = np.zeros(T, dtype=bool)
    assert compute_lal_v2(v, los, dt=0.1) == 0.0


# --------------------------------------------------------------------------- #
# 5. Ego never decelerates -> sentinel (worst case, sortable)                  #
# --------------------------------------------------------------------------- #
def test_never_decelerates_sentinel():
    T = 60
    v = np.full(T, 16.0)                      # constant cruise, no drop
    los = np.zeros(T, dtype=bool); los[30:] = True
    assert compute_lal_v2(v, los, dt=0.1) == LAL_NO_REACTION


# --------------------------------------------------------------------------- #
# 6. Transient one-sample dip is NOT counted as an onset (sustained-hold)      #
# --------------------------------------------------------------------------- #
def test_transient_dip_rejected():
    # v_ref=16, threshold=13.6. A single-sample dip to 10 then back to 16 must
    # not register (hold=3 requires it to persist); the real sustained onset is
    # the later genuine ramp.
    T = 100
    v = np.full(T, 16.0)
    v[20] = 10.0                              # 1-sample glitch, recovers next step
    v[70:] = 8.0                              # genuine sustained drop from idx 70
    los = np.zeros(T, dtype=bool); los[80:] = True
    assert decel_onset_index(v) == 70         # not 20
    # t_LoS = 8.0 (idx 80), onset t = 7.0 (idx 70) -> LAL_v2 = +1.0
    assert compute_lal_v2(v, los, dt=0.1) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 7. Explicit timestamps honoured (non-uniform time base)                     #
# --------------------------------------------------------------------------- #
def test_explicit_timestamps():
    T = 40
    v = np.full(T, 20.0)
    v[25:] = 10.0                             # drop at idx 25 (10 <= 0.85*20=17)
    los = np.zeros(T, dtype=bool); los[30:] = True
    ts = np.arange(T) * 0.2                   # dt=0.2 s via explicit timestamps
    # onset idx 25 -> t=5.0 ; LoS idx 30 -> t=6.0 ; LAL_v2 = +1.0
    assert compute_lal_v2(v, los, timestamp_s=ts) == pytest.approx(1.0, abs=1e-9)

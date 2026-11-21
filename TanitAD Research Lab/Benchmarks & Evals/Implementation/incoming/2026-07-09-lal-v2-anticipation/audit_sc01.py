"""Independent-test audit of the SC-01 first-live-CARLA metric run (2026-07-08).

The Benchmarks & Eval "independent test" role (Mission Plan; agent duty #5):
recompute a live claim independently, with fresh seeds, and either confirm it or
expose its fragility. Target: `stack/experiments/p0-carla-workzone/suite_results_v1.json`.

Three measured results, all runnable on the 4060 / locally (numpy only, no CARLA):

  A. LAL-v1 discrimination collapse — reproduce the live "-0.7 / -0.7" null across
     a deceleration-smoothness sweep, locate the jerk-threshold cliff, and show
     LAL-v2 discriminates across the whole realistic range.
  B. LOPS headline recompute — Monte-Carlo the SC-01 tracking-noise model
     (N(0, 0.3) per axis, gamma=0.5) across many seeds -> mean +/- 95% CI and the
     analytic expectation; locate the committed 0.8338.
  C. OKRI >=3-seed power — seeds needed to separate the 32.4-vs-12.8 gap under a
     range of assumed per-seed SDs (my adopted CI-separation rule made concrete).

Usage:  python audit_sc01.py            (writes audit_results.json next to this file)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

# stack imports (the live LAL-v1 + telemetry container we are auditing)
_STACK = Path(__file__).resolve().parents[6] / "stack"
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lal_v2 import compute_lal_v2                                  # noqa: E402
from tanitad.eval.metrics import (                                # noqa: E402
    JERK_BRAKE_THRESHOLD, compute_lal, LOPS_GAMMA, ScenarioTelemetry,
)

DT = 0.1
T = 200
V_CRUISE = 16.0
LOS_IDX = 110          # line-of-sight at ~t=11 s (matches work_zone_phantom los_s)
TAPER_IDX = 90         # blind edge


def _telemetry_from_v(ego_v, los_idx=LOS_IDX):
    """Wrap a speed profile into ScenarioTelemetry (fields LAL-v1 reads)."""
    los = np.zeros(T, dtype=bool); los[los_idx:] = True
    jerk = np.gradient(np.gradient(ego_v, DT), DT)
    return ScenarioTelemetry(
        ego_v=ego_v, ego_jerk=jerk, steer_rate=np.zeros(T),
        latency_ms=np.full(T, 18.0), hazard_los_flag=los,
        dist_to_blind_spot=np.abs(TAPER_IDX * V_CRUISE * DT - np.cumsum(ego_v * DT)),
        is_occluded_flag=np.zeros(T, dtype=bool),
        wm_hazard_xy=np.full((T, 2), np.nan), gt_hazard_xy=np.zeros((T, 2)),
        dt=DT), jerk


def anticipatory_profile(smooth_s: float, brake_start_idx: int, v_floor=9.0):
    """World-model slowdown that begins BEFORE LoS, with a tunable smoothness.

    smooth_s = duration (s) of a cosine-eased deceleration ramp. Larger = gentler
    = lower peak |jerk| (physically what real vehicle dynamics produce, vs the
    design-oracle's sharp clipped-linear ramp that trips the jerk threshold).
    v_floor=9 m/s: a prudent *ease-off* to ~60% cruise on approach to the blind
    edge — NOT an emergency stop. This is comfort-bounded braking (ISO-2631-scale
    |jerk| ~ 1-2 m/s^3), the realistic anticipatory behaviour, and precisely the
    regime where LAL-v1's -1.5 m/s^3 hard-jerk trigger fails to fire.
    """
    n = max(2, int(round(smooth_s / DT)))
    v = np.full(T, V_CRUISE, dtype=float)
    for i in range(T):
        if i <= brake_start_idx:
            continue
        frac = min(1.0, (i - brake_start_idx) / n)
        ease = 0.5 - 0.5 * math.cos(math.pi * frac)     # cosine ease-in-out 0..1
        v[i] = V_CRUISE - (V_CRUISE - v_floor) * ease
    return v


def reactive_profile():
    """Hold cruise until just after LoS, then brake hard (the E2E baseline)."""
    v = np.full(T, V_CRUISE, dtype=float)
    n = 6
    for i in range(T):
        if i <= LOS_IDX + 2:
            continue
        frac = min(1.0, (i - (LOS_IDX + 2)) / n)
        v[i] = V_CRUISE * (1.0 - 0.9 * frac)
    return v


def result_A():
    """LAL-v1 collapse sweep + LAL-v2 robustness."""
    react_v = reactive_profile()
    tel_r, _ = _telemetry_from_v(react_v)
    lal1_react = compute_lal(tel_r)
    lal2_react = compute_lal_v2(react_v, tel_r.hazard_los_flag, dt=DT)

    rows = []
    # brake starts ~1.5 s before the blind edge (idx TAPER_IDX-15), before LoS
    brake_start = TAPER_IDX - 15
    for smooth_s in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        anti_v = anticipatory_profile(smooth_s, brake_start)
        tel_a, jerk = _telemetry_from_v(anti_v)
        lal1 = compute_lal(tel_a)
        lal2 = compute_lal_v2(anti_v, tel_a.hazard_los_flag, dt=DT)
        peak_decel_jerk = float(np.min(jerk))     # most-negative jerk
        rows.append({
            "smooth_s": smooth_s,
            "peak_decel_jerk": round(peak_decel_jerk, 3),
            "trips_v1_threshold": bool(peak_decel_jerk < JERK_BRAKE_THRESHOLD),
            "LAL_v1_anti": round(lal1, 3),
            "LAL_v1_react": round(lal1_react, 3),
            "LAL_v1_separates": bool(lal1 > lal1_react + 1e-9),
            "LAL_v2_anti": round(lal2, 3),
            "LAL_v2_react": round(lal2_react, 3),
            "LAL_v2_separates": bool(lal2 > lal2_react + 1e-9),
        })
    return {
        "jerk_threshold": JERK_BRAKE_THRESHOLD,
        "reactive_LAL_v1": round(lal1_react, 3),
        "reactive_LAL_v2": round(lal2_react, 3),
        "sweep": rows,
        "v1_collapse_cliff": "LAL_v1 stops separating once the anticipatory "
                             "slowdown's peak jerk stays above (less negative "
                             "than) the -1.5 m/s^3 threshold",
    }


def result_B(n_seeds=5000, n_occ_variants=(10, 20, 40, 80)):
    """Independent recompute of the SC-01 world_model LOPS = 0.8338.

    LOPS = mean_occ exp(-gamma * ||wm - gt||), wm = gt + N(0, sigma) per axis.
    ||wm-gt|| ~ Rayleigh(sigma); per-step kernel k = exp(-gamma * ||.||).
    """
    sigma = 0.3
    gamma = LOPS_GAMMA
    # analytic expectation of the per-step kernel via fine numeric integral
    r = np.linspace(0, 6 * sigma, 200_000)
    pdf = (r / sigma ** 2) * np.exp(-r ** 2 / (2 * sigma ** 2))
    E_kernel = float(np.trapezoid(np.exp(-gamma * r) * pdf, r))

    rng = np.random.default_rng(12345)
    out = {"sigma": sigma, "gamma": gamma,
           "analytic_E_per_step_kernel": round(E_kernel, 4),
           "committed_SC01_world_model_LOPS": 0.8338, "by_n_occluded": []}
    for n_occ in n_occ_variants:
        clip_means = np.empty(n_seeds)
        for s in range(n_seeds):
            d = rng.normal(0, sigma, size=(n_occ, 2))
            clip_means[s] = np.mean(np.exp(-gamma * np.linalg.norm(d, axis=1)))
        lo, hi = np.percentile(clip_means, [2.5, 97.5])
        out["by_n_occluded"].append({
            "n_occluded_steps": n_occ,
            "LOPS_mean": round(float(clip_means.mean()), 4),
            "LOPS_95ci": [round(float(lo), 4), round(float(hi), 4)],
            "committed_0.8338_inside_ci": bool(lo <= 0.8338 <= hi),
        })
    return out


def result_C(gap=32.3686 - 12.8285):
    """Seeds needed for CI-separation of the OKRI gap under assumed per-seed SD.

    Two same-scene means separate at 95% when gap > 2 * 1.96 * SD / sqrt(n)
    -> n > (2 * 1.96 * SD / gap)^2.  (My adopted rule, made numeric.)
    """
    rows = []
    for sd in (5.0, 10.0, 15.0, 20.0):
        n_req = (2 * 1.96 * sd / gap) ** 2
        rows.append({"assumed_per_seed_SD": sd,
                     "seeds_needed_for_CI_separation": max(1, math.ceil(n_req)),
                     "raw": round(n_req, 2)})
    return {"okri_gap": round(gap, 3),
            "rule": "n > (2*1.96*SD/gap)^2  (95% CI non-overlap, same scene)",
            "note": "per-seed SD must be MEASURED on the >=3-seed pod run; the "
                    "rows below are a sensitivity table, not a claim",
            "table": rows}


def main():
    res = {"audit": "SC-01 first-live-CARLA metric run (2026-07-08)",
           "source": "stack/experiments/p0-carla-workzone/suite_results_v1.json",
           "A_LAL_discrimination": result_A(),
           "B_LOPS_recompute": result_B(),
           "C_OKRI_seed_power": result_C()}
    out = Path(__file__).resolve().parent / "audit_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\n[audit] -> {out}")


if __name__ == "__main__":
    main()

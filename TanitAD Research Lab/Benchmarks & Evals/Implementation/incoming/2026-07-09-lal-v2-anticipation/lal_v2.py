"""LAL-v2 — anticipatory deceleration lead (Benchmarks & Eval, 2026-07-09).

WHY THIS EXISTS
---------------
The FIRST live CARLA run of the SC-01 Work-Zone Phantom scenario (2026-07-08,
`stack/experiments/p0-carla-workzone/suite_results_v1.json`) exposed a metric
defect: the reactive baseline AND the anticipating (world-model) policy both
scored **LAL_v1 = -0.7 s** — zero discrimination on real physics. Root cause
(confirmed by the run author's own commit note, 2d87acb): LAL-v1 detects
"braking onset" via a hard longitudinal-jerk threshold (ego_jerk < -1.5 m/s^3),
but a good anticipatory policy slows *smoothly* on approach to the blind edge —
a comfort-bounded deceleration (|jerk| typically < ~2 m/s^3 in AD) that never
trips the threshold. LAL-v1 therefore measures reaction *hardness*, not
anticipation, and is blind to exactly the behaviour TanitAD's world model is
supposed to produce.

LAL-v2 detects the onset of *sustained deceleration by speed drop* relative to
the free-cruise reference, so a gentle anticipatory slowdown registers. It is the
latent, pre-line-of-sight generalization of the recognized **Time-To-Brake (TTB)**
family (TTB/TTC, e.g. Euro-NCAP AEB): TTB/TTC require the hazard to be
*detectable*; LAL-v2 credits braking that begins before the hazard is in
line-of-sight at all — the object-permanence edge (H15).

    LAL_v2 = t_LoS - t_decel_onset        (s;  > 0  => slowed before line-of-sight)

    t_LoS          first time the hazard is in line-of-sight (LoS flag, first True)
    t_decel_onset  first time ego speed has dropped >= drop_frac below the
                   free-cruise reference v_ref and is still non-increasing
                   (sustained over `hold` steps).

Sign convention matches LAL-v1: > 0 proactive, <= 0 reactive.
Sentinels (shared with LAL-v1): 0.0 if the hazard never reaches LoS (nothing to
anticipate); LAL_NO_REACTION (-999.0) if the ego never decelerates at all.

numpy-only; operates on plain arrays (no ScenarioTelemetry import) so this ships
and tests standalone. The proposed integration (see INTAKE.md) adds a thin
`compute_lal_v2(tel)` wrapper to `stack/tanitad/eval/metrics.py` that reads the
same fields LAL-v1 already consumes (ego_v, hazard_los_flag, dt/timestamp_s).
"""

from __future__ import annotations

import numpy as np

# Shared sentinel with metrics.LAL_NO_REACTION (keep in sync on integration).
LAL_NO_REACTION = -999.0

# Deep-Think-14-consistent defaults (documented; no magic numbers in the body).
LAL2_DROP_FRAC = 0.15   # speed must fall >=15% below free-cruise to count as decel
LAL2_REF_FRAC = 0.30    # free-cruise reference = max speed over the first 30% of clip
LAL2_HOLD = 3           # steps the drop must persist (reject a one-sample dip)


def _times(n: int, dt: float, timestamp_s) -> np.ndarray:
    if timestamp_s is not None:
        return np.asarray(timestamp_s, dtype=float)
    return np.arange(n, dtype=float) * dt


def decel_onset_index(ego_v,
                      drop_frac: float = LAL2_DROP_FRAC,
                      ref_frac: float = LAL2_REF_FRAC,
                      hold: int = LAL2_HOLD):
    """Index of the first sustained deceleration onset, or None.

    v_ref is the max speed over the first ``ref_frac`` of the clip (the free-cruise
    speed before the work zone). Onset = first index i with
    ``ego_v[i] <= (1 - drop_frac) * v_ref`` that stays at/below that level for the
    next ``hold`` steps (sustained, not a transient dip).
    """
    v = np.asarray(ego_v, dtype=float)
    T = v.size
    if T == 0:
        return None
    ref_n = max(1, int(round(ref_frac * T)))
    v_ref = float(np.max(v[:ref_n]))
    if v_ref <= 0.0:
        return None
    thresh = (1.0 - drop_frac) * v_ref
    below = v <= thresh
    for i in np.flatnonzero(below):
        j = min(int(i) + hold, T - 1)
        # sustained: still at/below threshold `hold` steps later (allow tiny tol)
        if v[j] <= thresh + 1e-9:
            return int(i)
    return None


def compute_lal_v2(ego_v, hazard_los_flag, dt: float = 0.1,
                   timestamp_s=None,
                   drop_frac: float = LAL2_DROP_FRAC,
                   ref_frac: float = LAL2_REF_FRAC,
                   hold: int = LAL2_HOLD) -> float:
    """LAL-v2 anticipation lead in seconds (see module docstring)."""
    los = np.asarray(hazard_los_flag, dtype=bool)
    v = np.asarray(ego_v, dtype=float)
    t = _times(v.size, dt, timestamp_s)
    los_idx = np.flatnonzero(los)
    if los_idx.size == 0:
        return 0.0                      # nothing ever reaches LoS -> nothing to anticipate
    t_los = float(t[int(los_idx[0])])
    onset = decel_onset_index(v, drop_frac, ref_frac, hold)
    if onset is None:
        return LAL_NO_REACTION          # ego never decelerated
    return float(t_los - t[onset])

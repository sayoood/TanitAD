#!/usr/bin/env python3
"""Distance-keeping for the LONGITUDINAL metric family — headway / time-gap / min-TTC.

Closes the hole the binding four-family rule leaves open. `taniteval/taniteval/four_families.py`
currently returns::

    "distance_keeping": {"status": "UNAVAILABLE", "reason": "no lead-agent track in the episode
     cache — PhysicalAI-AV ships obstacle.offline ... Implementing it is a WORK ITEM, not a pass."}

Half of the family Sayed made binding could therefore not be computed at all. This module is that
work item.

⭐ WHAT IS NEW HERE, AND WHAT IS NOT.
`stack/scripts/lead_state_gate.py` already contains a **proven, strictly causal** `obstacle.offline`
reader (`ego_frame` / `lead_frame`) that answers *"where is the lead RIGHT NOW, relative to the
ego's TRUE pose"*. That is an **input feature**. It is NOT the metric: scoring an arm asks a
different question — *"where would the lead have been relative to the trajectory THIS ARM
PREDICTED"* — which needs the lead's future track expressed in the **window-origin ego frame**, so
that a predicted path and the real lead live in one coordinate system. `build_lead_tracks.py`
supplies that; this file is the pure, I/O-free metric over it.

⛔ CONVENTIONS — quote these with any number, they are not interchangeable across papers.
* frame: window-origin ego frame at t0. **x = forward, y = left**, metres, clip-local
  (`egomotion` carries no lat/lon — CLAUDE.md).
* ``gap`` is **rig-origin to lead REAR face**: ``along - size_x/2``. Identical to
  `lead_state_gate.lead_frame`, deliberately — two gap conventions in one programme is a
  retraction waiting to happen. It is NOT bumper-to-bumper: our ego origin is the rig, so the ego's
  own front overhang is not subtracted.
* ``time_gap`` (= time headway, THW) is ``gap / v_ego``, seconds, and is **undefined at standstill**
  (returned NaN below `MIN_SPEED_MPS`), not clamped to a large number.
* ``ttc`` is ``gap / closing_rate`` with ``closing_rate = -d(gap)/dt`` and is **capped at
  `TTC_CAP_S` when not closing** — again the `lead_frame` convention. A capped TTC is a
  *censored observation*; `n_closing` is reported so a mean is never read as if every window closed.
* the corridor gate uses the **predicted path's own local heading**, not the t0 axes — an arm that
  drifts laterally must lose its lead, or the metric would credit it with distance-keeping on a
  vehicle it is no longer behind.

Every aggregate carries its ``n`` and, when a family cannot be computed, a ``reason`` — the binding
rule requires per-family reasons and n, never a silent drop.
"""
from __future__ import annotations

import numpy as np

from taniteval.progress import PROGRESS_HUMAN_MIN_M      # numpy-only module; the shared 0.5 m

#: lateral half-corridor for "this agent is in my lane", metres. `lead_state_gate.LEAD_LAT_M`.
LEAD_LAT_M = 2.0
#: TTC ceiling, seconds. `lead_state_gate.TTC_CAP_S`.
TTC_CAP_S = 30.0
#: below this the ego is stopped and a time-gap is undefined rather than enormous, m/s.
MIN_SPEED_MPS = 0.5
#: closing rates beyond this are a track switch, not a vehicle. `lead_state_gate.CLOSING_CLIP_MS`.
CLOSING_CLIP_MS = 20.0

#: speed bands for the BINDING stratified read, m/s. ⛔ A pooled distance-keeping number hides the
#: regime that matters: MEASURED on this corpus, the tactical lossy rate runs 38.2 % at 1-3 m/s down
#: to 1.8 % at 10-15 m/s, and the nav-sentinel share 96.05 % down to 34.69 % at 15+ m/s. Boundaries
#: chosen to CONTAIN the bands those numbers are quoted on (1-3, 10-15, 15+) so they stay comparable.
SPEED_BANDS = ((0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 10.0),
               (10.0, 15.0), (15.0, float("inf")))
#: a stratum below this many lead-bearing windows is reported as UNPOWERED with its n, never as a
#: number. An episode-cluster bootstrap over a handful of windows in one or two clips is noise.
MIN_STRATUM_N = 30

__all__ = [
    "LEAD_LAT_M", "TTC_CAP_S", "MIN_SPEED_MPS", "CLOSING_CLIP_MS",
    "SPEED_BANDS", "MIN_STRATUM_N",
    "path_headings", "per_step_gap", "distance_keeping",
    "band_label", "assign_bands", "distance_keeping_by_speed",
    "paired_distance_keeping",
    "LEAD_VIOL_TG_S", "lead_viol", "paired_lead_viol", "lead_viol_vs_human",
    # 2026-10-11 additions (PI + Master Mind rulings): at-risk CI, progress control, verdict
    "LEAD_PROGRESS_MIN", "MIN_HUMAN_ARC_M", "LEAD_VIOL_BAR_KEY",
    "path_arc_length", "lead_progress", "lead_viol_verdict",
]


def path_headings(path: np.ndarray, t0_heading: float = 0.0) -> np.ndarray:
    """Local heading (rad) at each waypoint of ``path`` (K, 2), from successive differences.

    Step 0 has no predecessor inside the path, so it takes ``t0_heading`` (0.0 = the window's own
    forward axis). A zero-length step inherits the previous heading rather than producing a
    ``atan2(0, 0)`` artefact that would swing the corridor by a full turn.
    """
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"path must be (K, 2), got {path.shape}")
    k = path.shape[0]
    out = np.full(k, float(t0_heading))
    prev = np.vstack([np.zeros((1, 2)), path[:-1]])
    d = path - prev
    n = np.hypot(d[:, 0], d[:, 1])
    moved = n > 1e-6
    out[moved] = np.arctan2(d[moved, 1], d[moved, 0])
    # carry the last valid heading forward across stationary steps
    last = float(t0_heading)
    for i in range(k):
        if moved[i]:
            last = out[i]
        else:
            out[i] = last
    return out


def per_step_gap(path: np.ndarray, lead: np.ndarray, lead_len: float,
                 lat_m: float = LEAD_LAT_M) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (gap_m, lat_m_signed, in_corridor) per horizon step.

    ``path`` (K, 2) is the arm's predicted ego positions in the window-origin frame; ``lead``
    (K, 2) the lead agent's CENTRE in the same frame at the same times (NaN where the track has no
    sample). ``lead_len`` is the lead's ``size_x``.

    A step counts only when the lead is **ahead** (gap >= 0) and **inside the corridor**; both are
    evaluated in the predicted path's local frame, so an arm that leaves the lane loses its lead.
    """
    path = np.asarray(path, dtype=np.float64)
    lead = np.asarray(lead, dtype=np.float64)
    if path.shape != lead.shape:
        raise ValueError(f"path {path.shape} and lead {lead.shape} must match")
    hdg = path_headings(path)
    d = lead - path
    c, s = np.cos(hdg), np.sin(hdg)
    along = d[:, 0] * c + d[:, 1] * s
    lat = -d[:, 0] * s + d[:, 1] * c
    gap = along - float(lead_len) / 2.0
    ok = np.isfinite(gap) & np.isfinite(lat) & (gap >= 0.0) & (np.abs(lat) < lat_m)
    return gap, lat, ok


def distance_keeping(paths, leads, lead_lens, speeds, dt: float,
                     lat_m: float = LEAD_LAT_M) -> dict:
    """Headway / time-gap / min-TTC over a batch of windows. Pure — no I/O, no torch.

    Args:
        paths:     (W, K, 2) predicted ego positions in each window's origin frame, metres.
        leads:     (W, K, 2) lead-agent centres in the SAME frame, NaN where absent.
        lead_lens: (W,) lead ``size_x`` in metres (NaN/0 where absent).
        speeds:    (W,) ego speed at t0, m/s — the denominator of the time gap.
        dt:        spacing between consecutive waypoints, seconds. ⛔ TTC scales as 1/dt through
                   the closing rate; time-gap and headway are dt-invariant.

    Returns a dict whose per-window arrays are ``float('nan')`` where the window has no lead, plus
    the aggregate means, their ``n``, and — when ``n == 0`` — a ``status``/``reason`` pair in the
    same shape `four_families.longitudinal` already emits, so the consumer needs no special case.
    """
    paths = np.asarray(paths, dtype=np.float64)
    leads = np.asarray(leads, dtype=np.float64)
    lead_lens = np.asarray(lead_lens, dtype=np.float64).reshape(-1)
    speeds = np.asarray(speeds, dtype=np.float64).reshape(-1)
    if paths.ndim != 3 or paths.shape[2] != 2:
        raise ValueError(f"paths must be (W, K, 2), got {paths.shape}")
    if leads.shape != paths.shape:
        raise ValueError(f"leads {leads.shape} must match paths {paths.shape}")
    w = paths.shape[0]
    if lead_lens.shape[0] != w or speeds.shape[0] != w:
        raise ValueError("lead_lens and speeds must have one entry per window")
    if not dt > 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    headway = np.full(w, np.nan)
    time_gap = np.full(w, np.nan)
    ttc = np.full(w, np.nan)
    n_steps = np.zeros(w, dtype=int)
    closing = np.zeros(w, dtype=bool)

    for i in range(w):
        ln = lead_lens[i]
        if not np.isfinite(ln):
            continue
        gap, _lat, ok = per_step_gap(paths[i], leads[i], ln, lat_m=lat_m)
        n_steps[i] = int(ok.sum())
        if n_steps[i] == 0:
            continue
        g = np.where(ok, gap, np.nan)
        headway[i] = np.nanmin(g)                       # the tightest the arm ever gets
        if speeds[i] >= MIN_SPEED_MPS:
            time_gap[i] = headway[i] / speeds[i]
        # closing rate from the gap sequence the ARM would have produced
        idx = np.flatnonzero(ok)
        if idx.size >= 2:
            rate = -np.diff(g[idx]) / (np.diff(idx) * dt)
            rate = np.clip(rate, -CLOSING_CLIP_MS, CLOSING_CLIP_MS)
            # TTC at each closing step, from the gap standing at the START of that step
            with np.errstate(divide="ignore", invalid="ignore"):
                t_k = np.where(rate > 0.1, g[idx][:-1] / rate, TTC_CAP_S)
            t_k = np.clip(t_k, 0.0, TTC_CAP_S)
            ttc[i] = float(np.nanmin(t_k))
            closing[i] = bool(np.any(rate > 0.1))
        else:
            ttc[i] = TTC_CAP_S                          # single admissible step: no rate, censored

    have = np.isfinite(headway)
    n = int(have.sum())
    out = {
        "headway_min_m": headway,
        "time_gap_min_s": time_gap,
        "min_ttc_s": ttc,
        "n_steps_in_corridor": n_steps,
        "n": n,
        "n_windows": w,
        "dt_s": float(dt),
        "gap_convention": "rig-origin to lead rear face (along - size_x/2); NOT bumper-to-bumper",
        "ttc_cap_s": TTC_CAP_S,
    }
    if n == 0:
        out["status"] = "NOT-APPLICABLE"
        out["reason"] = (f"no lead agent inside the |lat| < {lat_m} m corridor ahead of the "
                         f"predicted path in any of the {w} windows — free-flow, not a failure")
        return out
    out["status"] = "OK"
    out["mean_headway_min_m"] = round(float(np.nanmean(headway[have])), 4)
    tg = np.isfinite(time_gap)
    out["mean_time_gap_min_s"] = (round(float(np.nanmean(time_gap[tg])), 4)
                                  if tg.any() else float("nan"))
    out["n_time_gap"] = int(tg.sum())
    out["mean_min_ttc_s"] = round(float(np.nanmean(ttc[have])), 4)
    out["n_closing"] = int(closing.sum())
    out["censoring_note"] = (f"{n - int(closing.sum())} of {n} windows never close on the lead and "
                             f"are censored at TTC_CAP_S={TTC_CAP_S}s. The mean is over censored "
                             f"data — quote n_closing beside it, never the mean alone.")
    return out


# --------------------------------------------------------------------------- #
# SPEED-STRATIFIED read + the paired estimator                                 #
#                                                                             #
# ⛔ Why this is not optional. This corpus's behaviour is strongly speed-       #
# dependent, so a pooled headway/time-gap/TTC is an average over regimes that  #
# do not resemble each other — and the crawling regime, which dominates the    #
# window count, is the one where a time gap is least meaningful. Reported      #
# per band, each with its own n and its own reason when it cannot be computed. #
# --------------------------------------------------------------------------- #
def band_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+"


def assign_bands(speeds, bands=SPEED_BANDS) -> np.ndarray:
    """-> [W] band index, ``-1`` where the speed is not finite. Half-open ``[lo, hi)``."""
    v = np.asarray(speeds, dtype=np.float64).reshape(-1)
    out = np.full(v.size, -1, dtype=np.int64)
    for i, (lo, hi) in enumerate(bands):
        out[np.isfinite(v) & (v >= lo) & (v < hi)] = i
    return out


def _agg(vals, eid, n_boot, seed):
    """Episode-cluster bootstrap over the finite entries, or the reason there are none."""
    from taniteval.ci import episode_cluster_bootstrap
    v = np.asarray(vals, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return {"n": 0, "status": "NOT-APPLICABLE", "reason": "no finite values in this stratum"}
    r = episode_cluster_bootstrap(v[ok], list(np.asarray(eid, dtype=object)[ok]),
                                  reduce="mean", n_boot=n_boot, seed=seed)
    r["n"] = int(ok.sum())
    r["n_total"] = int(v.size)
    return r


def distance_keeping_by_speed(dk: dict, speeds, eid, *, states=None,
                              bands=SPEED_BANDS, n_boot: int = 2000, seed: int = 0,
                              min_stratum_n: int = MIN_STRATUM_N) -> dict:
    """Stratify a :func:`distance_keeping` result by ego speed at t0, with CIs and denominators.

    Args:
        dk:     the dict :func:`distance_keeping` returned, INCLUDING its per-window arrays. When
                it came back from `four_families._distance_keeping` those live under
                ``dk["_per_window"]``; both shapes are accepted.
        speeds: ``[W]`` ego speed at each window's t0, m/s — the stratifier, and the same array
                that denominated the time gap.
        eid:    ``[W]`` episode / clip cluster id per window. ⛔ The bootstrap resamples THESE, not
                windows: consecutive windows of one clip are not independent.
        states: optional ``[W]`` of ``lead_source.LEAD`` / ``NO_LEAD`` / ``NO_LABEL``. Supplying it
                is what lets each stratum report **why** its denominator is what it is instead of
                a bare n — and it keeps NO_LABEL out of the free-flow count.

    Every stratum carries ``n`` and, when it cannot be computed, a ``status``/``reason``. A stratum
    thinner than ``min_stratum_n`` is UNPOWERED — reported, never quoted.
    """
    pw = dk.get("_per_window", dk)
    for key in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
        if key not in pw:
            raise KeyError(f"distance_keeping result is missing per-window {key!r}; pass the dict "
                           f"returned by distance_keeping() or one carrying '_per_window'")
    head = np.asarray(pw["headway_min_m"], dtype=np.float64)
    tg = np.asarray(pw["time_gap_min_s"], dtype=np.float64)
    ttc = np.asarray(pw["min_ttc_s"], dtype=np.float64)
    v = np.asarray(speeds, dtype=np.float64).reshape(-1)
    eid = np.asarray(eid, dtype=object).reshape(-1)
    if not (head.shape == tg.shape == ttc.shape == v.shape == eid.shape):
        raise ValueError(f"per-window arrays disagree: headway {head.shape}, speeds {v.shape}, "
                         f"eid {eid.shape}")
    st = None if states is None else np.asarray(states, dtype=object).reshape(-1)
    if st is not None and st.shape != head.shape:
        raise ValueError(f"states {st.shape} must match the window count {head.shape}")

    bi = assign_bands(v, bands)
    out = {
        "_what": "LONGITUDINAL distance-keeping, per speed band. NEVER pool these.",
        "_binding": ("Sayed 2026-08-02 clause 5: where a family cannot be computed, say so PER "
                     "STRATUM with the reason and the n, rather than silently dropping it."),
        "bands_mps": [band_label(lo, hi) for lo, hi in bands],
        "min_stratum_n": int(min_stratum_n),
        "estimator": "episode_cluster_bootstrap (taniteval.ci) — NEVER overlapping_holdout_se",
        "n_windows_total": int(head.size),
        "strata": {},
    }
    if st is not None:
        vals, cnts = np.unique(st, return_counts=True)
        out["window_states_total"] = {str(k): int(c) for k, c in zip(vals, cnts)}

    for i, (lo, hi) in enumerate(bands):
        m = bi == i
        lab = band_label(lo, hi)
        blk = {"speed_mps": [lo, (None if not np.isfinite(hi) else hi)],
               "n_windows": int(m.sum())}
        if st is not None:
            vals, cnts = np.unique(st[m], return_counts=True) if m.any() else ([], [])
            blk["window_states"] = {str(k): int(c) for k, c in zip(vals, cnts)}
        n_lead = int(np.isfinite(head[m]).sum())
        blk["n_with_lead"] = n_lead
        blk["lead_rate"] = (round(n_lead / int(m.sum()), 4) if m.any() else None)
        if not m.any():
            blk["status"] = "EMPTY"
            blk["reason"] = "no window fell in this speed band"
        elif n_lead == 0:
            blk["status"] = "NOT-APPLICABLE"
            blk["reason"] = (f"none of the {int(m.sum())} windows in this band had a lead agent "
                             f"inside the |lat| < {LEAD_LAT_M} m corridor ahead of the predicted "
                             f"path — free flow, not a failure")
        elif n_lead < int(min_stratum_n):
            blk["status"] = "UNPOWERED"
            blk["reason"] = (f"{n_lead} lead-bearing windows < min_stratum_n {min_stratum_n}; an "
                             f"episode-cluster bootstrap over this few is noise. Reported, not "
                             f"quoted.")
        else:
            blk["status"] = "OK"
            blk["headway_min_m"] = _agg(head[m], eid[m], n_boot, seed)
            blk["time_gap_min_s"] = _agg(tg[m], eid[m], n_boot, seed)
            blk["min_ttc_s"] = _agg(ttc[m], eid[m], n_boot, seed)
            if lo < MIN_SPEED_MPS:
                blk["time_gap_caveat"] = (
                    f"this band straddles MIN_SPEED_MPS={MIN_SPEED_MPS} m/s; a time gap is "
                    f"UNDEFINED at standstill and is NaN there, so n_time_gap < n_with_lead by "
                    f"construction. Never read the gap here as a following behaviour.")
        out["strata"][lab] = blk

    bad = int((bi < 0).sum())
    if bad:
        out["n_windows_unbanded"] = bad
        out["unbanded_reason"] = "ego speed at t0 was not finite"
    return out


def paired_distance_keeping(dk_a: dict, dk_b: dict, eid, *, names=("A", "B"),
                            n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired A-minus-B deltas on the three distance-keeping metrics, on JOINTLY-valid windows.

    ⛔ The paired estimator, never two independent intervals combined in quadrature, and never
    `overlapping_holdout_se` — which biases the POINT ESTIMATE (mean-of-split-means, not the
    full set) bidirectionally, up to a sign flip on paired deltas.

    Only windows where BOTH arms produced a finite value enter a metric's delta; a window where
    one arm drove out of the corridor and the other did not is not a paired observation, and
    ``n_used`` / ``n_a_only`` / ``n_b_only`` report exactly how many those were.
    """
    from taniteval.ci import paired_episode_cluster_bootstrap
    pa, pb = dk_a.get("_per_window", dk_a), dk_b.get("_per_window", dk_b)
    eid = np.asarray(eid, dtype=object).reshape(-1)
    out = {"_what": f"paired {names[0]} - {names[1]} distance-keeping deltas",
           "estimator": "paired_episode_cluster_bootstrap (taniteval.ci)",
           "arms": list(names), "metrics": {}}
    for key in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
        a = np.asarray(pa[key], dtype=np.float64)
        b = np.asarray(pb[key], dtype=np.float64)
        if a.shape != b.shape or a.shape != eid.shape:
            raise ValueError(f"{key}: shapes {a.shape} / {b.shape} / eid {eid.shape} disagree")
        ok = np.isfinite(a) & np.isfinite(b)
        blk = {"n_used": int(ok.sum()), "n_windows": int(a.size),
               "n_a_only": int((np.isfinite(a) & ~np.isfinite(b)).sum()),
               "n_b_only": int((np.isfinite(b) & ~np.isfinite(a)).sum())}
        if ok.sum() == 0:
            blk["status"] = "NOT-APPLICABLE"
            blk["reason"] = "no window has a finite value for BOTH arms"
        else:
            blk.update(paired_episode_cluster_bootstrap(
                a[ok], b[ok], list(eid[ok]), n_boot=n_boot, seed=seed))
            blk["status"] = "OK"
        out["metrics"][key] = blk
    return out


# --------------------------------------------------------------------------- #
# LEAD-VIOL  --  SPEC_REFCV8_DRAFT.md section 8.7 (X7), item (a)               #
#                                                                             #
# REGISTERED DEFINITION (verbatim, SPEC_REFCV8_DRAFT.md @ 8522e49 line 768):    #
#   "LEAD-VIOL = share of lead windows whose plan's minimum time gap is        #
#    < 1.0 s while the human's is >= 1.0 s."                                   #
# READING, implemented literally and flagged for the Master Mind in REPORT.md: #
#  * time gap = this module's own `time_gap_min_s` (min over the horizon of    #
#    gap / v_ego(t0)); the plan and the human path are scored against the     #
#    SAME logged lead future by the SAME `distance_keeping`, so the two time   #
#    gaps differ only in the path.                                            #
#  * strict `<` on the plan, inclusive `>=` on the human, both vs 1.0 s.       #
#  * DENOMINATOR = ALL lead windows (`lead_valid`), including those where the  #
#    human's own time gap is < 1.0 s or undefined: those can never be a        #
#    violation, and they are counted in the denominator because the spec says  #
#    "share of lead windows". The conditional reading (denominator = windows   #
#    where the human was >= 1.0 s) is emitted beside it as `share_of_at_risk`. #
#  * a plan whose own corridor gate LOSES the lead (veers out of the lane) has #
#    an undefined time gap and is NOT a violation under the literal text. That #
#    is a loophole a lane-leaving planner could use, so it is COUNTED            #
#    (`n_plan_lost_lead`) and a sensitivity share that treats it as a violation #
#    is emitted (`share_incl_lost_lead`). The registered bar is on `share`.     #
# The human path reads 0 BY CONSTRUCTION: plan == human cannot be both < 1.0   #
# and >= 1.0.                                                                 #
#                                                                             #
# RULINGS 2026-10-11 (these supersede the "DENOMINATOR = ALL lead windows"     #
# reading above for the BAR; the all-windows keys are kept, never repurposed): #
#  * PI: the bar's denominator is the HUMAN-GAP-CONDITIONED reading - a        #
#    violation counts only in lead windows where the human keeps >= 1.0 s.    #
#    That is `share_of_at_risk` (+ `share_of_at_risk_ci`) and, paired,         #
#    `x7_direction_met_at_risk`. It is barred on the 6 s lead block.           #
#  * Master Mind: LEAD-VIOL is paired with a PROGRESS CONTROL. On the SAME      #
#    at-risk windows, `lead_progress_ratio` = mean over windows of (the plan's  #
#    travelled arc length over the horizon / the human's). The bar is          #
#    admissible only if the arm's ratio is >= 0.90 (`LEAD_PROGRESS_MIN`): a     #
#    planner that meets LEAD-VIOL by crawling FAILS; STOP reads ratio 0.        #
#    `lead_viol_verdict` is the one place the two are combined.                #
# --------------------------------------------------------------------------- #
#: the registered threshold, seconds (SPEC_REFCV8_DRAFT.md section 8.7(a)).
LEAD_VIOL_TG_S = 1.0
#: Master Mind ruling 2026-10-11: LEAD-VIOL's bar is admissible only if the arm's
#: `lead_progress_ratio` (plan arc length / human arc length, mean over at-risk windows) >= this.
LEAD_PROGRESS_MIN = 0.90
#: the progress comparison is inclusive (>= 0.90) and runs on a float MEAN: twenty windows that each
#: read exactly 0.9 average to 0.8999999999999999. This guard (far below any measurable difference)
#: keeps a tie a tie; it cannot move a decision on real data.
_PROGRESS_EPS = 1e-9
#: at-risk windows whose HUMAN arc is shorter than this are excluded from the progress ratio and
#: COUNTED (`n_human_arc_degenerate`): a ratio over a crawling denominator explodes. This is the
#: constant `taniteval.progress` / `pseudosim` already use for the same degenerate case - reused,
#: not re-minted, so the two progress readings exclude the same standstill windows.
MIN_HUMAN_ARC_M = PROGRESS_HUMAN_MIN_M
#: the key of `paired_lead_viol(...)` the SPEC should cite for the X7 bar (human-gap-conditioned).
LEAD_VIOL_BAR_KEY = "x7_direction_met_at_risk"


def path_arc_length(paths) -> np.ndarray:
    """-> [W] polyline length (metres) of each ``(K, 2)`` path, the ego origin (0, 0) implicit.

    ``paths`` is ``(W, K, 2)`` in the window-origin frame (the same convention as
    :func:`path_headings`: step 0 runs from the origin to ``paths[:, 0]``). NaN anywhere in a
    path makes that window's length NaN - it is never silently skipped. The length is the sum of
    the chord lengths between the supplied waypoints, so a sparse grid under-reads a curved path
    by the same small factor for the plan and the human (the RATIO is what is used).
    """
    p = np.asarray(paths, dtype=np.float64)
    if p.ndim != 3 or p.shape[2] != 2:
        raise ValueError(f"paths must be (W, K, 2), got {p.shape}")
    seq = np.concatenate([np.zeros((p.shape[0], 1, 2)), p], axis=1)
    return np.linalg.norm(np.diff(seq, axis=1), axis=-1).sum(axis=1)


def _at_risk_mask(lead_win, human_tg, thr: float) -> np.ndarray:
    """-> bool [W]: the human-gap-CONDITIONED denominator (PI ruling 2026-10-11): lead windows
    whose HUMAN keeps a time gap >= ``thr``. Only these can ever be a violation."""
    h = np.asarray(human_tg, dtype=np.float64).reshape(-1)
    with np.errstate(invalid="ignore"):
        return np.asarray(lead_win, dtype=bool) & np.isfinite(h) & (h >= thr)


def _viol_flags(plan_tg, human_tg, thr: float = LEAD_VIOL_TG_S) -> np.ndarray:
    """-> bool [W]: plan time gap ``< thr`` AND human time gap ``>= thr``.

    NaN on either side compares False: an UNDEFINED time gap is not a violation (it is counted
    separately by :func:`lead_viol`). The boundary is asymmetric on purpose, exactly as registered:
    plan strictly below, human at-or-above.
    """
    p = np.asarray(plan_tg, dtype=np.float64).reshape(-1)
    h = np.asarray(human_tg, dtype=np.float64).reshape(-1)
    with np.errstate(invalid="ignore"):
        return (p < thr) & (h >= thr)


def _lead_window_mask(human_headway, lead_valid):
    """-> (bool [W], source). The DENOMINATOR: windows that count as 'lead windows'."""
    if lead_valid is not None:
        return np.asarray(lead_valid, dtype=bool).reshape(-1), "caller lead_valid"
    return (np.isfinite(np.asarray(human_headway, dtype=np.float64).reshape(-1)),
            "FALLBACK: lead inside the human path's corridor (finite human headway) - no "
            "lead_valid / state supplied")


def _per_window_dk(dk: dict, what: str) -> dict:
    pw = dk.get("_per_window", dk)
    for key in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
        if key not in pw:
            raise KeyError(f"{what} is missing per-window {key!r}; pass the dict returned by "
                           f"distance_keeping() or one carrying '_per_window'")
    return {k: np.asarray(pw[k], dtype=np.float64).reshape(-1)
            for k in ("headway_min_m", "time_gap_min_s", "min_ttc_s")}


def lead_progress(plan_arc, human_arc, at_risk, eid=None, *,
                  min_human_arc_m: float = MIN_HUMAN_ARC_M,
                  min_progress: float = LEAD_PROGRESS_MIN,
                  n_boot: int = 2000, seed: int = 0) -> dict:
    """LEAD-PROGRESS: the control that stops LEAD-VIOL being met by crawling.

    ``lead_progress_ratio`` = MEAN over the at-risk windows of ``plan_arc / human_arc``: the plan's
    travelled arc length over the horizon divided by the human's, on the SAME windows the
    human-gap-conditioned LEAD-VIOL denominator uses (Master Mind ruling 2026-10-11). The bar on
    LEAD-VIOL is admissible only if the ratio is >= ``min_progress`` (0.90). A STOP plan reads
    exactly 0; the human path reads exactly 1.

    Args:
        plan_arc / human_arc: ``[W]`` polyline length in metres, e.g. :func:`path_arc_length`.
        at_risk: ``[W]`` bool - the at-risk windows (lead window AND human time gap >= 1.0 s).
        eid: ``[W]`` episode id for the episode-cluster bootstrap. Without it the interval is
            UNAVAILABLE with a reason, never silent.

    A window whose human arc is below ``min_human_arc_m`` (or whose arcs are not finite) is
    EXCLUDED and COUNTED (``n_human_arc_degenerate`` / ``n_arc_nonfinite``) - a ratio over a
    crawling denominator explodes, and clamping it would turn "the human did not move" into a
    free pass. ``_per_window['progress_ratio']`` is NaN outside the used windows.

    Two sensitivity readings are emitted beside the RULED one (``lead_progress_ratio``), never
    instead of it: ``lead_progress_ratio_clipped`` (per-window ratio clipped at 1.0) and
    ``progress_ratio_of_means`` (sum of plan arcs / sum of human arcs). The ruled mean of ratios
    can be pulled up by windows where the plan out-runs the human; ``overshoot_masks_crawl`` is
    True exactly when the ruled mean clears the bar and the clipped reading does not.

    ⛔ DECISION READING (Master Mind ruling 2026-10-11, registered before any refcv8 number): the bar
    reads ``lead_progress_ratio_clipped`` - per-window ratios clipped at 1.0, then averaged - so
    out-running the human in some at-risk windows cannot buy back crawling in others.
    ``meets_min_progress`` is that decision; the unclipped mean's comparison is kept beside it as
    ``meets_min_progress_mean`` (sensitivity only).
    """
    pa = np.asarray(plan_arc, dtype=np.float64).reshape(-1)
    ha = np.asarray(human_arc, dtype=np.float64).reshape(-1)
    ar = np.asarray(at_risk, dtype=bool).reshape(-1)
    if not (pa.shape == ha.shape == ar.shape):
        raise ValueError(f"plan_arc {pa.shape}, human_arc {ha.shape} and at_risk {ar.shape} "
                         f"must have one entry per window")
    ids = None
    if eid is not None:
        ids = np.asarray(eid, dtype=object).reshape(-1)
        if ids.size != pa.size:
            raise ValueError(f"eid has {ids.size} entries for {pa.size} windows")
    fin = np.isfinite(pa) & np.isfinite(ha)
    with np.errstate(invalid="ignore"):
        usable = ar & fin & (ha >= float(min_human_arc_m))
        degenerate = ar & fin & (ha < float(min_human_arc_m))
    ratio = np.full(pa.size, np.nan)
    ratio[usable] = pa[usable] / ha[usable]
    n_use = int(usable.sum())
    out = {
        "metric": "LEAD-PROGRESS",
        "definition": ("mean over the at-risk windows of (plan arc length over the horizon / "
                       "human arc length); the LEAD-VIOL bar is admissible only if >= "
                       f"{min_progress:g} (Master Mind ruling 2026-10-11)"),
        "min_progress": float(min_progress),
        "min_human_arc_m": float(min_human_arc_m),
        "n_at_risk": int(ar.sum()),
        "n": n_use,
        "n_arc_nonfinite": int((ar & ~fin).sum()),
        "n_human_arc_degenerate": int(degenerate.sum()),
        "_per_window": {"progress_ratio": ratio},
    }
    if n_use == 0:
        out.update(status="NOT-APPLICABLE", lead_progress_ratio=None, meets_min_progress=None,
                   reason=(f"no usable at-risk window among the {int(ar.sum())} (arcs not finite "
                           f"or the human arc < {min_human_arc_m:g} m) - the progress control "
                           f"cannot be formed, so the LEAD-VIOL bar is not admissible"))
        return out
    r = ratio[usable]
    mean = float(r.mean())
    clipped = float(np.minimum(r, 1.0).mean())
    out.update(
        status="OK",
        lead_progress_ratio=mean,                      # the mean of ratios (reported; NOT the decision)
        meets_min_progress=bool(clipped >= float(min_progress) - _PROGRESS_EPS),   # MM 2026-10-11
        meets_min_progress_mean=bool(mean >= float(min_progress) - _PROGRESS_EPS),
        lead_progress_ratio_clipped=clipped,           # full precision: the decision reads THIS
        decision_reading="lead_progress_ratio_clipped",
        progress_ratio_of_means=float(pa[usable].sum() / ha[usable].sum()),
        overshoot_masks_crawl=bool(mean >= float(min_progress) - _PROGRESS_EPS
                                   and clipped < float(min_progress) - _PROGRESS_EPS),
        n_below_min=int((r < float(min_progress) - _PROGRESS_EPS).sum()),
        decision_rule=("point estimate of the CLIPPED mean (per-window ratio clipped at 1.0) >= "
                       "min_progress (MM ruling 2026-10-11; the SPEC states the bar on the point "
                       "estimate); the interval of the unclipped mean is reported beside it"),
    )
    if ids is None:
        out["ci"] = {"status": "UNAVAILABLE", "n": n_use,
                     "reason": "no per-window episode id, so the episode-cluster bootstrap "
                               "cannot be formed (pass eid). A pooled number is not a substitute."}
    else:
        from taniteval.ci import episode_cluster_bootstrap
        out["ci"] = episode_cluster_bootstrap(r, list(ids[usable]), reduce="mean",
                                              n_boot=n_boot, seed=seed)
        out["ci"]["question_answered"] = ("would another draw of EPISODES say this? It does NOT "
                                          "answer 'another training run' or 'another inference "
                                          "run' (H-ESTIM-SEED-1).")
    return out


def lead_viol(plan_dk: dict, human_dk: dict, eid=None, *, lead_valid=None,
              threshold_s: float = LEAD_VIOL_TG_S, n_boot: int = 2000, seed: int = 0,
              plan_arc=None, human_arc=None, min_progress: float = LEAD_PROGRESS_MIN) -> dict:
    """LEAD-VIOL: share of lead windows with plan time gap < 1.0 s while the human's is >= 1.0 s.

    Args:
        plan_dk / human_dk: the dicts :func:`distance_keeping` returned for the arm's predicted
            path and for the HUMAN (ground-truth) path, over the SAME windows, leads and speeds
            (or dicts carrying ``_per_window``). Both must be scored against the same logged
            lead future, or the comparison measures the lead, not the path.
        eid: ``[W]`` episode / clip cluster id - the bootstrap resamples THESE. Without it the
            share is still emitted but its interval is UNAVAILABLE with a reason, never silent.
        lead_valid: ``[W]`` bool - which windows are 'lead windows' (the denominator), e.g. the v9
            ``lead_valid`` AND the S2 / S-ROUTE subset, or ``lead_source.LEAD`` state. If omitted
            the fallback is 'the human's own path has a lead in its corridor'.
        plan_arc / human_arc: ``[W]`` arc lengths in metres (:func:`path_arc_length`) for the
            LEAD-PROGRESS control. Without them the control is UNAVAILABLE with a reason and
            :func:`lead_viol_verdict` withholds the bar.

    WHICH KEY IS THE BAR (PI + Master Mind rulings 2026-10-11): the denominator is the
    human-gap-CONDITIONED ``share_of_at_risk`` (with ``share_of_at_risk_ci``); ``share`` /
    ``ci`` (all lead windows) are kept for back-comparison and are NOT the bar. The bar is
    admissible only if ``lead_progress_ratio`` >= 0.90 - see :func:`lead_viol_verdict`.

    The per-window 0/1 flags are returned under ``_per_window["viol"]``; all per-window arrays sit under
    ``_per_window``: ``viol`` (0/1, NaN outside the lead
    windows), ``at_risk`` (1.0 at-risk / 0.0 lead window the human is already tight in / NaN
    outside the lead windows) and ``progress_ratio`` (NaN outside the used at-risk windows) -
    for :func:`paired_lead_viol`.
    """
    p = _per_window_dk(plan_dk, "plan_dk")
    h = _per_window_dk(human_dk, "human_dk")
    w = h["headway_min_m"].size
    if p["headway_min_m"].size != w:
        raise ValueError(f"plan_dk has {p['headway_min_m'].size} windows, human_dk has {w}")
    lead_win, src = _lead_window_mask(h["headway_min_m"], lead_valid)
    if lead_win.size != w:
        raise ValueError(f"lead_valid has {lead_win.size} entries for {w} windows")
    ids = None
    if eid is not None:
        ids = np.asarray(eid, dtype=object).reshape(-1)
        if ids.size != w:
            raise ValueError(f"eid has {ids.size} entries for {w} windows")
    thr = float(threshold_s)
    flags = _viol_flags(p["time_gap_min_s"], h["time_gap_min_s"], thr) & lead_win
    n_lead = int(lead_win.sum())
    out = {
        "metric": "LEAD-VIOL",
        "spec": "SPEC_REFCV8_DRAFT.md section 8.7 (X7) (a)",
        "definition": (f"share of lead windows whose plan's minimum time gap is < {thr:g} s while "
                       f"the human's is >= {thr:g} s"),
        "threshold_s": thr,
        "denominator_source": src,
        "n_windows": int(w),
        "n_lead_windows": n_lead,
    }
    viol = np.full(w, np.nan)
    viol[lead_win] = flags[lead_win].astype(np.float64)
    at_risk_f = np.full(w, np.nan)
    progress_ratio = np.full(w, np.nan)
    out["_per_window"] = {"viol": viol, "at_risk": at_risk_f, "progress_ratio": progress_ratio}
    if n_lead == 0:
        out["status"] = "NOT-APPLICABLE"
        out["reason"] = (f"no lead window among the {w} windows - nothing to violate; "
                         f"reported with n, never as a pass")
        out["n"] = 0
        return out
    htg = h["time_gap_min_s"]
    at_risk = _at_risk_mask(lead_win, htg, thr)
    at_risk_f[lead_win] = at_risk[lead_win].astype(np.float64)
    lost = lead_win & np.isfinite(htg) & ~np.isfinite(p["time_gap_min_s"])
    ttc_ok = lead_win & np.isfinite(p["min_ttc_s"])
    n_viol = int(flags.sum())
    n_risk = int(at_risk.sum())
    out.update({
        "status": "OK",
        "n": n_lead,
        "n_viol": n_viol,
        "share": round(n_viol / n_lead, 6),
        "n_at_risk": n_risk,
        "share_of_at_risk": (round(n_viol / n_risk, 6) if n_risk else None),
        "n_human_tg_undefined": int((lead_win & ~np.isfinite(htg)).sum()),
        "n_human_tg_below_thr": int((lead_win & np.isfinite(htg) & (htg < thr)).sum()),
        "n_plan_lost_lead": int(lost.sum()),
        "share_incl_lost_lead": round(
            (n_viol + int((lost & at_risk).sum())) / n_lead, 6),
        "n_plan_ttc_finite": int(ttc_ok.sum()),
        "n_plan_ttc_censored": int((ttc_ok & (p["min_ttc_s"] >= TTC_CAP_S - 1e-9)).sum()),
        "ttc_cap_s": TTC_CAP_S,
        "censoring_note": ("n_plan_ttc_censored = lead windows whose plan never closes on the lead "
                           "(TTC pinned at TTC_CAP_S). A STOP plan is censored by construction; "
                           "report it beside the share."),
        "lost_lead_note": ("a plan that leaves the lead's corridor has an UNDEFINED time gap and is "
                           "not a violation under the registered text; `share_incl_lost_lead` "
                           "counts those at-risk windows as violations (sensitivity, not the bar)."),
        "bar_denominator": ("share_of_at_risk (human-gap-conditioned, PI ruling 2026-10-11); "
                            "`share` is the all-lead-windows reading, kept for back-comparison"),
    })
    q = ("would another draw of EPISODES say this? It does NOT answer 'another training run' or "
         "'another inference run' (H-ESTIM-SEED-1): a replicate arm is still required.")
    if ids is None:
        why = {"status": "UNAVAILABLE", "n": n_lead,
               "reason": "no per-window episode id, so the episode-cluster bootstrap "
                         "cannot be formed (pass eid). A pooled number is not a substitute."}
        out["ci"] = dict(why)
        out["share_of_at_risk_ci"] = dict(why, n=n_risk)
    else:
        from taniteval.ci import episode_cluster_bootstrap
        out["ci"] = episode_cluster_bootstrap(
            viol[lead_win], list(ids[lead_win]), reduce="mean", n_boot=n_boot, seed=seed)
        out["ci"]["question_answered"] = q
        if n_risk == 0:
            out["share_of_at_risk_ci"] = {
                "status": "NOT-APPLICABLE", "n": 0,
                "reason": "no at-risk window (the human is below the threshold on every lead "
                          "window) - nothing can be a violation"}
        else:
            out["share_of_at_risk_ci"] = episode_cluster_bootstrap(
                viol[at_risk], list(ids[at_risk]), reduce="mean", n_boot=n_boot, seed=seed)
            out["share_of_at_risk_ci"]["question_answered"] = q
    # --- the progress control, on the SAME at-risk windows ---------------------------------- #
    if plan_arc is None or human_arc is None:
        prog = {"metric": "LEAD-PROGRESS", "status": "UNAVAILABLE", "n_at_risk": n_risk, "n": 0,
                "lead_progress_ratio": None, "meets_min_progress": None,
                "min_progress": float(min_progress),
                "reason": ("no plan_arc / human_arc supplied, so the progress control cannot be "
                           "formed and the LEAD-VIOL bar is NOT admissible (a planner that meets "
                           "it by crawling would pass). Pass path_arc_length(plan), "
                           "path_arc_length(human).")}
    else:
        prog = lead_progress(plan_arc, human_arc, at_risk, ids, min_progress=min_progress,
                             n_boot=n_boot, seed=seed)
        progress_ratio[:] = prog["_per_window"]["progress_ratio"]
    prog.pop("_per_window", None)
    out["progress"] = prog
    out["lead_progress_ratio"] = prog.get("lead_progress_ratio")
    out["lead_progress_ratio_clipped"] = prog.get("lead_progress_ratio_clipped")   # the decision reading
    return out


def _pw_get(r: dict, keys):
    pw = r.get("_per_window", {})
    for key in keys:
        if key in pw:
            return np.asarray(pw[key], dtype=np.float64)
    return None


def paired_lead_viol(a: dict, b: dict, eid, *, names=("A", "B"),
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired ``A - B`` LEAD-VIOL on the SAME lead windows (the X7 bar is ``refcv8 - refcv7 < 0``).

    ``a`` / ``b`` are :func:`lead_viol` results. Their lead-window sets must be identical - the
    denominator is a property of the corpus and the human path, not of the arm - or the delta
    compares different windows and this raises rather than silently intersecting. Uses
    ``ci.paired_episode_cluster_bootstrap`` (never two intervals in quadrature).

    Output layout (2026-10-11; every older key is unchanged):

    * the TOP-LEVEL ``delta`` / ``lo`` / ``hi`` / ``separated`` / ``x7_direction_met`` are the
      ALL-lead-windows reading, exactly as before - back-comparison only.
    * ``at_risk`` is the same paired bootstrap on the AT-RISK windows of both arms (human gap
      >= 1.0 s, PI ruling 2026-10-11), and ``x7_direction_met_at_risk`` mirrors its verdict.
      **That is the key the SPEC should bar** (``LEAD_VIOL_BAR_KEY``). The at-risk set must be
      identical in both arms (it is a property of the human path); if it is not, this raises.
    * ``progress`` is the paired LEAD-PROGRESS delta on the windows usable in BOTH arms, with each
      arm's own ratio. Admissibility of the bar is decided by :func:`lead_viol_verdict`, not here.
    """
    from taniteval.ci import paired_episode_cluster_bootstrap

    def _flags(nm, r):
        # a lead_viol() result keeps them at _per_window['viol']; the dict four_families
        # returns parks them at distance_keeping['_per_window']['lead_viol']
        v = _pw_get(r, ("viol", "lead_viol"))
        if v is None:
            raise KeyError(f"{nm}: no per-window LEAD-VIOL flags (_per_window.viol / .lead_viol)")
        return v

    va, vb = _flags(names[0], a), _flags(names[1], b)
    ids = np.asarray(eid, dtype=object).reshape(-1)
    if not (va.shape == vb.shape == ids.shape):
        raise ValueError(f"shapes disagree: {va.shape} / {vb.shape} / eid {ids.shape}")
    ok = np.isfinite(va)
    if not np.array_equal(ok, np.isfinite(vb)):
        raise ValueError("the two arms' lead-window sets differ; LEAD-VIOL deltas are only "
                         "defined on one shared denominator")
    out = {"_what": f"paired {names[0]} - {names[1]} LEAD-VIOL",
           "arms": list(names), "n_lead_windows": int(ok.sum()),
           "n_viol_a": int(np.nansum(va)), "n_viol_b": int(np.nansum(vb))}
    if not ok.any():
        out.update(status="NOT-APPLICABLE", reason="no shared lead window")
        out["x7_direction_met_at_risk"] = None
        return out
    r = paired_episode_cluster_bootstrap(va[ok], vb[ok], list(ids[ok]),
                                         n_boot=n_boot, seed=seed)
    out.update(r)
    out["status"] = "OK"
    out["x7_direction_met"] = bool(r["delta"] < 0 and r["separated"])
    out["question_answered"] = ("episode-resampling only (H-ESTIM-SEED-1): a separated interval "
                                "is necessary, not sufficient, for a lever effect")
    out["all_windows_note"] = ("top-level delta / x7_direction_met are the ALL-lead-windows "
                               "reading (kept for back-comparison); the bar reads "
                               f"`{LEAD_VIOL_BAR_KEY}` (at-risk denominator).")

    # --- the human-gap-conditioned (at-risk) denominator ------------------------------------ #
    ra, rb = _pw_get(a, ("at_risk", "lead_at_risk")), _pw_get(b, ("at_risk", "lead_at_risk"))
    if ra is None or rb is None:
        out["at_risk"] = {"status": "UNAVAILABLE",
                          "reason": ("an arm carries no per-window at-risk mask "
                                     "(_per_window.at_risk / .lead_at_risk) - it predates the "
                                     "2026-10-11 ruling; re-score it with this lead_metrics")}
        out[LEAD_VIOL_BAR_KEY] = None
    else:
        if ra.shape != ids.shape or rb.shape != ids.shape:
            raise ValueError(f"at-risk masks {ra.shape} / {rb.shape} disagree with eid {ids.shape}")
        sel_a, sel_b = (ra == 1.0), (rb == 1.0)
        if not np.array_equal(sel_a, sel_b):
            raise ValueError("the two arms' at-risk sets differ; the at-risk denominator is a "
                             "property of the human path and the lead windows, so a delta on "
                             "it is only defined when both arms were scored against the same "
                             "human path and the same lead windows")
        blk = {"n_at_risk": int(sel_a.sum()), "denominator": "lead windows where the human keeps "
               "a time gap >= the threshold (PI ruling 2026-10-11)",
               "n_viol_a": int(np.nansum(va[sel_a])), "n_viol_b": int(np.nansum(vb[sel_a]))}
        if not sel_a.any():
            blk.update(status="NOT-APPLICABLE", reason="no at-risk window shared by the two arms")
            out[LEAD_VIOL_BAR_KEY] = None
        else:
            blk.update(paired_episode_cluster_bootstrap(
                va[sel_a], vb[sel_a], list(ids[sel_a]), n_boot=n_boot, seed=seed))
            blk["status"] = "OK"
            blk[LEAD_VIOL_BAR_KEY] = bool(blk["delta"] < 0 and blk["separated"])
            out[LEAD_VIOL_BAR_KEY] = blk[LEAD_VIOL_BAR_KEY]
        out["at_risk"] = blk

    # --- the progress control, paired on the windows usable in both arms -------------------- #
    pa_, pb_ = (_pw_get(a, ("progress_ratio", "lead_progress_ratio")),
                _pw_get(b, ("progress_ratio", "lead_progress_ratio")))
    if pa_ is None or pb_ is None:
        out["progress"] = {"status": "UNAVAILABLE",
                           "reason": "an arm carries no per-window progress ratio "
                                     "(_per_window.progress_ratio / .lead_progress_ratio)"}
    else:
        if pa_.shape != ids.shape or pb_.shape != ids.shape:
            raise ValueError(f"progress ratios {pa_.shape} / {pb_.shape} disagree with eid "
                             f"{ids.shape}")
        both = np.isfinite(pa_) & np.isfinite(pb_)
        pblk = {"n_used": int(both.sum()),
                "n_a_only": int((np.isfinite(pa_) & ~np.isfinite(pb_)).sum()),
                "n_b_only": int((np.isfinite(pb_) & ~np.isfinite(pa_)).sum())}
        if not both.any():
            pblk.update(status="NOT-APPLICABLE",
                        reason="no at-risk window has a usable progress ratio in BOTH arms")
        else:
            pblk["lead_progress_ratio_a"] = float(pa_[both].mean())
            pblk["lead_progress_ratio_b"] = float(pb_[both].mean())
            pblk.update(paired_episode_cluster_bootstrap(
                pa_[both], pb_[both], list(ids[both]), n_boot=n_boot, seed=seed))
            pblk["status"] = "OK"
        out["progress"] = pblk
    return out


def lead_viol_verdict(lv: dict, *, paired: dict | None = None,
                      min_progress: float = LEAD_PROGRESS_MIN) -> dict:
    """The ONE place LEAD-VIOL and its progress control are combined (Master Mind 2026-10-11).

    ``lv`` is the arm's :func:`lead_viol` block (or a ``distance_keeping`` dict that carries it
    under ``lead_viol``); ``paired`` is :func:`paired_lead_viol` of the arm against its baseline.

    Returns ``verdict`` and ``bar_decision`` (True / False / None):

    * ``INADMISSIBLE-CRAWL`` - the arm's ``lead_progress_ratio_clipped`` < ``min_progress`` (the
      CLIPPED mean, MM ruling 2026-10-11: over-running some windows cannot mask crawling). The bar
      decision is WITHHELD (``bar_decision`` None): a planner that meets LEAD-VIOL by crawling
      fails. A STOP plan reads ratio 0 and lands here.
    * ``INADMISSIBLE-NO-PROGRESS-CONTROL`` - the ratio could not be formed (arcs not supplied, or
      no usable at-risk window). Absence of the control is not admissibility.
    * ``NOT-APPLICABLE`` - the arm has no lead window.
    * ``ADMISSIBLE-BAR-PENDING-BASELINE`` - progress clears the bar; no ``paired`` supplied, so
      the X7 comparison (arm - baseline < 0, separated) has not been evaluated.
    * ``UNAVAILABLE-PAIRED`` - progress clears the bar but the paired at-risk result is missing.
    * ``PASS`` / ``FAIL`` - progress clears the bar and ``paired[x7_direction_met_at_risk]`` is
      True / False.

    The decision reads the POINT ESTIMATE of the progress ratio (the SPEC states the bar on it);
    its interval is reported beside it.
    """
    blk = lv
    if isinstance(lv, dict) and "lead_viol" in lv and "metric" not in lv:
        blk = lv["lead_viol"]
    thr = float(min_progress)
    prog = blk.get("progress") or {}
    ratio = blk.get("lead_progress_ratio_clipped")         # the DECISION reading (MM 2026-10-11)
    out = {
        "metric": "LEAD-VIOL-VERDICT",
        "min_progress": thr,
        "lead_progress_ratio_clipped": ratio,
        "lead_progress_ratio": blk.get("lead_progress_ratio"),
        "lead_progress_ci": prog.get("ci"),
        "n_progress": prog.get("n"),
        "overshoot_masks_crawl": prog.get("overshoot_masks_crawl"),
        "bar": (f"arm - baseline LEAD-VIOL < 0, separated, on the at-risk denominator "
                f"(`{LEAD_VIOL_BAR_KEY}`), admissible only if lead_progress_ratio_clipped >= {thr:g}"),
        "bar_decision": None,
    }
    if blk.get("status") == "NOT-APPLICABLE":
        out["verdict"] = "NOT-APPLICABLE"
        out["reason"] = blk.get("reason", "no lead window")
        return out
    if ratio is None or not np.isfinite(float(ratio)):
        out["verdict"] = "INADMISSIBLE-NO-PROGRESS-CONTROL"
        out["reason"] = prog.get("reason", "lead_progress_ratio_clipped is not available")
        return out
    if float(ratio) < thr - _PROGRESS_EPS:
        out["verdict"] = "INADMISSIBLE-CRAWL"
        out["reason"] = (f"lead_progress_ratio_clipped {float(ratio):.4f} < {thr:g}: on the at-risk "
                         f"windows the plan covers too little of the human's distance, so a low "
                         f"LEAD-VIOL here is not evidence of distance keeping. Bar decision "
                         f"withheld.")
        return out
    if paired is None:
        out["verdict"] = "ADMISSIBLE-BAR-PENDING-BASELINE"
        out["reason"] = "progress control clears the bar; pass paired=paired_lead_viol(arm, base)"
        return out
    met = paired.get(LEAD_VIOL_BAR_KEY)
    if met is None:
        out["verdict"] = "UNAVAILABLE-PAIRED"
        out["reason"] = (paired.get("at_risk") or {}).get("reason", "paired result has no "
                                                          f"`{LEAD_VIOL_BAR_KEY}`")
        return out
    out["bar_decision"] = bool(met)
    out["verdict"] = "PASS" if met else "FAIL"
    pp = paired.get("progress") or {}
    out["baseline_lead_progress_ratio"] = pp.get("lead_progress_ratio_b")
    return out


def lead_viol_vs_human(plan_paths, human_paths, leads, lead_lens, speeds, dt: float, *,
                       eid=None, lead_valid=None, plan_dk: dict | None = None,
                       threshold_s: float = LEAD_VIOL_TG_S, n_boot: int = 2000,
                       seed: int = 0, plan_arc=None, human_arc=None,
                       min_progress: float = LEAD_PROGRESS_MIN) -> dict:
    """Convenience: score the HUMAN path with :func:`distance_keeping` against the same lead and
    return :func:`lead_viol`. ``plan_dk`` may be passed to avoid re-scoring the arm.

    The progress control uses ``plan_arc`` / ``human_arc`` when given, else the polyline length of
    ``plan_paths`` / ``human_paths`` (the steps the time gap is scored on)."""
    if plan_dk is None:
        plan_dk = distance_keeping(plan_paths, leads, lead_lens, speeds, dt)
    human_dk = distance_keeping(human_paths, leads, lead_lens, speeds, dt)
    if plan_arc is None and plan_paths is not None:
        plan_arc = path_arc_length(plan_paths)
    if human_arc is None and human_paths is not None:
        human_arc = path_arc_length(human_paths)
    return lead_viol(plan_dk, human_dk, eid, lead_valid=lead_valid,
                     threshold_s=threshold_s, n_boot=n_boot, seed=seed,
                     plan_arc=plan_arc, human_arc=human_arc, min_progress=min_progress)

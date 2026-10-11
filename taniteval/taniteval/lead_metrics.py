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
# --------------------------------------------------------------------------- #
#: the registered threshold, seconds (SPEC_REFCV8_DRAFT.md section 8.7(a)).
LEAD_VIOL_TG_S = 1.0


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


def lead_viol(plan_dk: dict, human_dk: dict, eid=None, *, lead_valid=None,
              threshold_s: float = LEAD_VIOL_TG_S, n_boot: int = 2000, seed: int = 0) -> dict:
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

    The per-window 0/1 flags are returned under ``_per_window["viol"]`` (NaN outside the lead
    windows) for :func:`paired_lead_viol`.
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
    out["_per_window"] = {"viol": viol}
    if n_lead == 0:
        out["status"] = "NOT-APPLICABLE"
        out["reason"] = (f"no lead window among the {w} windows - nothing to violate; "
                         f"reported with n, never as a pass")
        out["n"] = 0
        return out
    htg = h["time_gap_min_s"]
    at_risk = lead_win & np.isfinite(htg) & (htg >= thr)
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
    })
    if ids is None:
        out["ci"] = {"status": "UNAVAILABLE", "n": n_lead,
                     "reason": "no per-window episode id, so the episode-cluster bootstrap "
                               "cannot be formed (pass eid). A pooled number is not a substitute."}
    else:
        from taniteval.ci import episode_cluster_bootstrap
        out["ci"] = episode_cluster_bootstrap(
            viol[lead_win], list(ids[lead_win]), reduce="mean", n_boot=n_boot, seed=seed)
        out["ci"]["question_answered"] = ("would another draw of EPISODES say this? It does NOT "
                                          "answer 'another training run' or 'another inference run' "
                                          "(H-ESTIM-SEED-1): a replicate arm is still required.")
    return out


def paired_lead_viol(a: dict, b: dict, eid, *, names=("A", "B"),
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired ``A - B`` LEAD-VIOL on the SAME lead windows (the X7 bar is ``refcv8 - refcv7 < 0``).

    ``a`` / ``b`` are :func:`lead_viol` results. Their lead-window sets must be identical - the
    denominator is a property of the corpus and the human path, not of the arm - or the delta
    compares different windows and this raises rather than silently intersecting. Uses
    ``ci.paired_episode_cluster_bootstrap`` (never two intervals in quadrature).
    """
    from taniteval.ci import paired_episode_cluster_bootstrap

    def _flags(nm, r):
        # a lead_viol() result keeps them at _per_window['viol']; the dict four_families
        # returns parks them at distance_keeping['_per_window']['lead_viol']
        pw = r.get("_per_window", {})
        for key in ("viol", "lead_viol"):
            if key in pw:
                return np.asarray(pw[key], dtype=np.float64)
        raise KeyError(f"{nm}: no per-window LEAD-VIOL flags (_per_window.viol / .lead_viol)")

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
        return out
    r = paired_episode_cluster_bootstrap(va[ok], vb[ok], list(ids[ok]),
                                         n_boot=n_boot, seed=seed)
    out.update(r)
    out["status"] = "OK"
    out["x7_direction_met"] = bool(r["delta"] < 0 and r["separated"])
    out["question_answered"] = ("episode-resampling only (H-ESTIM-SEED-1): a separated interval "
                                "is necessary, not sufficient, for a lever effect")
    return out


def lead_viol_vs_human(plan_paths, human_paths, leads, lead_lens, speeds, dt: float, *,
                       eid=None, lead_valid=None, plan_dk: dict | None = None,
                       threshold_s: float = LEAD_VIOL_TG_S, n_boot: int = 2000,
                       seed: int = 0) -> dict:
    """Convenience: score the HUMAN path with :func:`distance_keeping` against the same lead and
    return :func:`lead_viol`. ``plan_dk`` may be passed to avoid re-scoring the arm."""
    if plan_dk is None:
        plan_dk = distance_keeping(plan_paths, leads, lead_lens, speeds, dt)
    human_dk = distance_keeping(human_paths, leads, lead_lens, speeds, dt)
    return lead_viol(plan_dk, human_dk, eid, lead_valid=lead_valid,
                     threshold_s=threshold_s, n_boot=n_boot, seed=seed)

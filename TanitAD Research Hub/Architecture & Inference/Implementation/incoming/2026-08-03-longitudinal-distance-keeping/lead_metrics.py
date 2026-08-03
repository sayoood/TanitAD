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

__all__ = [
    "LEAD_LAT_M", "TTC_CAP_S", "MIN_SPEED_MPS", "CLOSING_CLIP_MS",
    "path_headings", "per_step_gap", "distance_keeping",
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

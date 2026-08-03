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

#!/usr/bin/env python3
"""Turn `obstacle.offline` into the ``win["lead"]`` block `four_families` consumes.

⭐ WHAT THIS ADDS, AND WHAT IT DOES NOT.
`taniteval.lead_metrics` (2026-08-03) is the pure metric and is already admitted by the
pre-registered D-LEAD-1 control. The Architecture & Inference package
`2026-08-03-longitudinal-distance-keeping/build_lead_tracks.py` reads the label zips and builds
its OWN window grid. Neither can score an **eval** arm, because an eval window is not a point on
its own grid — it is window *j* of episode *e* of a cached split, and its ``t0`` is a position in
the CLIP's clock that nothing in the episode record states. That package's own INTAKE names this
as the open work item:

    "Wiring it into the *eval* path (val40 windows -> win["lead"]) ... Until that lands, arm evals
     will still report the family UNAVAILABLE."

This module is that wiring: the **registration** (episode pose index -> clip time) plus the
**per-window lead assembly**, both pure-numpy so they run on any host and are unit-testable.
I/O over the gated label zips stays with the caller.

⛔ THE REGISTRATION PROBLEM, and why it is solved by CONTENT and not by arithmetic.
`physicalai.build_episode` resamples the clip onto ``t_query = linspace(t_frames[0], t_frames[-1],
n_target)`` where ``t_frames`` is the **camera** timestamps parquet — a file that ships only inside
the ~2 GB camera chunk zip and is absent wherever the episode cache is not being rebuilt. Rebuilding
that arithmetic would also silently break the moment a build parameter changed.

:func:`register_poses_to_time` instead matches the episode's OWN ``poses`` (x, y) against the
egomotion track and returns the clip time of every pose index, with the fit residual reported so a
bad registration is loud rather than plausible. It is exact to the interpolation the cache was built
with, needs no camera clock, and works for the raw epcache, the v2 cache and any future format.

⛔ THREE WINDOW STATES, NEVER TWO. A window is ``LEAD`` (a causal in-corridor vehicle ahead),
``NO_LEAD`` (labels present, road genuinely clear) or ``NO_LABEL`` (no `obstacle.offline` for this
clip, or ``t0`` outside the ~20 s labelled span). ⚠️ Collapsing the third into the second is exactly
the bias this instrument exists to avoid: 2.44 % of the corpus has no `obstacle.offline` at all, and
`egomotion` runs 20-140 s while the labels stop at ~20 s, so **most** of a long clip is NO_LABEL.
Counting those as "no lead agent" would manufacture free-flow and flatter every arm.

⛔ CONVENTIONS — inherited unchanged from `lead_state_gate` / `lead_metrics` on purpose. Two gap
conventions in one programme is a retraction waiting to happen.
  * rig frame: **x forward, y left**, metres. ⭐ MEASURED, not assumed
    (`…/2026-08-03-obstacle-offline-join/raw/frame_convention.json`): of 2,778 tracks living >= 2 s,
    **1,756 (63.2 %) are world-static under x-fwd/y-left** vs 236 under the mirrored lateral and
    ~32 under either axis swap. A parked car is only parked under the right handedness.
  * ``gap`` = ``along - size_x/2`` (rig origin to the lead's REAR face), never bumper-to-bumper.
  * lead selection at t0 is **strictly causal** — cuboids timestamped <= t0 only.
    The lead's FUTURE positions are ground truth about the world, exactly like the ground-truth ego
    waypoints an ADE is measured against: a scoring input, never an arm input.
"""
from __future__ import annotations

import numpy as np

#: rollout.collect's window geometry. A window's origin is the pose at ``start + WINDOW - 1``.
WINDOW = 8
K_MAX = 20
STRIDE = 8

#: lateral half-corridor, metres — `lead_state_gate.LEAD_LAT_M` / `lead_metrics.LEAD_LAT_M`.
LEAD_LAT_M = 2.0
#: furthest admissible lead, metres — `lead_state_gate.LEAD_MAX_GAP_M`.
LEAD_MAX_GAP_M = 80.0
#: a cuboid older than this is not evidence about t0 — `lead_state_gate.MAX_STALE_S`.
MAX_STALE_S = 0.5
#: `obstacle.offline` is dense over roughly the first 20 s of a clip (E-GOAL-1).
OBS_SPAN_GUARD_S = 0.5

VEHICLE_CLASSES = ("automobile", "heavy_truck", "bus", "other_vehicle", "trailer")

#: window states. Exhaustive and mutually exclusive — see the module docstring.
LEAD = "LEAD"
NO_LEAD = "NO_LEAD"
NO_LABEL = "NO_LABEL"

#: registration is rejected above this median residual (metres). The episode poses ARE the
#: egomotion track resampled, so a correct match is sub-centimetre; anything near a metre means
#: the wrong clip, the wrong split or a changed build.
MAX_REGISTRATION_RESIDUAL_M = 0.25

__all__ = [
    "WINDOW", "K_MAX", "STRIDE", "LEAD_LAT_M", "LEAD_MAX_GAP_M", "MAX_STALE_S",
    "VEHICLE_CLASSES", "LEAD", "NO_LEAD", "NO_LABEL",
    "MAX_REGISTRATION_RESIDUAL_M", "RegistrationError",
    "window_last_indices", "register_poses_to_time", "select_lead_causal",
    "lead_track_in_window", "lead_block",
]


class RegistrationError(RuntimeError):
    """The episode could not be located on the clip's egomotion track."""


# --------------------------------------------------------------------------- #
# 1. which pose index each eval window sits on                                 #
# --------------------------------------------------------------------------- #
def window_last_indices(t_len: int, window: int = WINDOW, k_max: int = K_MAX,
                        stride: int = STRIDE) -> np.ndarray:
    """Pose indices that are the ORIGIN of each window `rollout.collect` emits, in its order.

    ``collect`` iterates ``starts = range(0, T - window - k_max, stride)`` and takes the ego pose
    at ``start + window - 1`` as the window origin, so this reproduces its window grid exactly.
    Returned in emission order, which is the order of every banked ``pred``/``gt`` row — that is
    what makes a lead block attachable to an already-scored dump with no re-inference.
    """
    if t_len < 0:
        raise ValueError(f"t_len must be >= 0, got {t_len}")
    starts = np.arange(0, max(int(t_len) - int(window) - int(k_max), 0), int(stride))
    return starts + int(window) - 1


# --------------------------------------------------------------------------- #
# 2. registration: episode pose index -> clip time                             #
# --------------------------------------------------------------------------- #
#: a pose whose neighbours are closer than this is inside a STOP, where position cannot identify
#: time at all. Excluded from the probe set — see :func:`register_poses_to_time`.
MIN_PROBE_MOVE_M = 0.30
#: fewer moving probes than this and the clip is too stationary to register by position.
MIN_MOVING_PROBES = 8


def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Median-of-pairwise-slopes fit ``y = a + b x``. ~29 % breakdown, which is what a clip with a
    long stop or a self-intersecting route needs; a least-squares fit is dragged by both."""
    n = x.size
    i, j = np.triu_indices(n, k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    b = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    return float(np.median(y - b * x)), b


def register_poses_to_time(poses_xy, ego_t, ego_x, ego_y, *,
                           max_residual_m: float = MAX_REGISTRATION_RESIDUAL_M,
                           n_probe: int = 96) -> dict:
    """Clip time (s) of every episode pose index, by matching the pose track to `egomotion`.

    ``poses_xy`` is ``[T, 2]`` — columns 0 and 1 of the cached episode's ``poses``, which
    `physicalai.signals_at` produced by interpolating egomotion ``x``/``y`` at the resampled
    query times. ``ego_*`` are the egomotion series in the clip's own clock (seconds).

    The episode grid is an AFFINE reparametrisation of the clip clock — ``build_episode`` uses
    ``linspace(t_frames[0], t_frames[-1], n_target)`` — so ``t = a + b * i`` holds exactly. ⚠️ ``b``
    is **not** 0.1 s: ``n_target = int(span_s * 10)`` truncates, so the realised spacing is
    ~0.1007 s (MEASURED across the R0 corpus: 0.10066–0.10084). Assuming 0.1 drifts ~0.13 s over a
    200-step episode — about 1.8 m of lead displacement at 13.6 m/s. Fitting it is not pedantry.

    ⛔ TWO THINGS MAKE A NAIVE FIT WRONG, and both occur in this corpus:
      * **stops** — while the ego is stationary, dozens of pose indices sit at one position, so
        nearest-point matching returns an arbitrary time inside the stop. Probes are therefore
        restricted to poses whose neighbours are more than ``MIN_PROBE_MOVE_M`` apart.
      * **route self-intersection** — a loop puts two different times at one place. Handled by a
        Theil-Sen (median-of-slopes) fit plus an inlier refit, rather than least squares, which a
        handful of such probes would drag.

    Returns ``{"t_s": [T], "a": .., "b": .., "residual_m": {...}, "n_probe": .., "n_inlier": ..}``.

    Raises :class:`RegistrationError` when the clip cannot be registered — a loud refusal, because
    a silently mis-registered window puts the lead in the wrong place and the metric still returns
    a perfectly plausible number.
    """
    p = np.asarray(poses_xy, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError(f"poses_xy must be [T, >=2], got {p.shape}")
    p = p[:, :2]
    t = np.asarray(ego_t, dtype=np.float64)
    ex = np.asarray(ego_x, dtype=np.float64)
    ey = np.asarray(ego_y, dtype=np.float64)
    if not (t.shape == ex.shape == ey.shape) or t.ndim != 1:
        raise ValueError("ego_t / ego_x / ego_y must be 1-D and the same length")
    if t.size < 2:
        raise RegistrationError("egomotion has fewer than 2 samples")
    o = np.argsort(t)
    t, ex, ey = t[o], ex[o], ey[o]

    tt = p.shape[0]
    if tt < 2:
        raise RegistrationError(f"episode has {tt} poses; cannot register")

    # local motion per pose index (central difference, clamped at the ends)
    lo = np.clip(np.arange(tt) - 1, 0, tt - 1)
    hi = np.clip(np.arange(tt) + 1, 0, tt - 1)
    move = np.hypot(p[hi, 0] - p[lo, 0], p[hi, 1] - p[lo, 1])
    moving = np.flatnonzero(move > MIN_PROBE_MOVE_M)
    if moving.size < MIN_MOVING_PROBES:
        raise RegistrationError(
            f"only {moving.size} of {tt} poses are moving (> {MIN_PROBE_MOVE_M} m between "
            f"neighbours); position cannot identify time on a stationary clip")
    pick = np.unique(np.linspace(0, moving.size - 1,
                                 min(int(n_probe), moving.size)).astype(int))
    idx = moving[pick]

    # nearest ego SAMPLE per probe. egomotion is 100 Hz, so a sample is within ~0.14 m at
    # 14 m/s; the fit averages that out.
    d2 = ((ex[None, :] - p[idx, 0:1]) ** 2 + (ey[None, :] - p[idx, 1:2]) ** 2)
    t_probe = t[np.argmin(d2, axis=1)]

    xi = idx.astype(np.float64)
    a, b = _theil_sen(xi, t_probe)
    # inlier refit: keep probes the robust fit already explains, then least-squares on those
    keep = np.abs(t_probe - (a + b * xi)) <= max(0.15, 3.0 * float(
        np.median(np.abs(t_probe - (a + b * xi)))))
    if keep.sum() >= max(MIN_MOVING_PROBES, 4):
        A = np.column_stack([np.ones(int(keep.sum())), xi[keep]])
        coef, *_ = np.linalg.lstsq(A, t_probe[keep], rcond=None)
        a, b = float(coef[0]), float(coef[1])
    t_s = a + b * np.arange(tt, dtype=np.float64)

    res = np.hypot(np.interp(t_s[idx], t, ex) - p[idx, 0],
                   np.interp(t_s[idx], t, ey) - p[idx, 1])
    med = float(np.median(res))
    out = {
        "t_s": t_s, "a": a, "b": b,
        "n_probe": int(idx.size), "n_inlier": int(keep.sum()),
        "n_moving_poses": int(moving.size), "n_poses": int(tt),
        "residual_m": {"median": round(med, 5), "p95": round(float(np.percentile(res, 95)), 5),
                       "max": round(float(res.max()), 5)},
        "grid_dt_s": round(b, 6),
    }
    if not np.isfinite(med) or med > float(max_residual_m):
        raise RegistrationError(
            f"episode does not lie on this egomotion track: median probe residual {med:.3f} m "
            f"> {max_residual_m} m (p95 {out['residual_m']['p95']} m, {out['n_inlier']}/"
            f"{out['n_probe']} inliers). Wrong clip, wrong split, or the cache was built from a "
            f"different source.")
    if not (0.05 <= b <= 0.5):
        raise RegistrationError(
            f"recovered grid spacing {b:.5f} s is outside the plausible 0.05-0.5 s band; the "
            f"fit did not lock on (residual {med:.3f} m, {out['n_inlier']}/{out['n_probe']} "
            f"inliers)")
    return out


# --------------------------------------------------------------------------- #
# 3. causal lead selection + the lead's future track in the window frame       #
# --------------------------------------------------------------------------- #
def select_lead_causal(obs_t, obs_track, obs_x, obs_y, obs_size_x, obs_is_vehicle,
                       t0: float, *, lat_m: float = LEAD_LAT_M,
                       max_gap_m: float = LEAD_MAX_GAP_M,
                       max_stale_s: float = MAX_STALE_S):
    """-> ``(track_id, gap_m, size_x)`` or ``(None, nan, nan)``. STRICTLY CAUSAL.

    Mirrors `lead_state_gate.lead_frame`'s rule exactly: among vehicles whose last cuboid at or
    before ``t0`` is no more than ``max_stale_s`` old, take the nearest one that is ahead
    (``gap >= 0``), within ``max_gap_m`` and inside the ``|lat| < lat_m`` corridor.
    """
    tt = np.asarray(obs_t, dtype=np.float64)
    if tt.size == 0:
        return None, float("nan"), float("nan")
    tid = np.asarray(obs_track)
    x = np.asarray(obs_x, dtype=np.float64)
    y = np.asarray(obs_y, dtype=np.float64)
    sx = np.asarray(obs_size_x, dtype=np.float64)
    veh = np.asarray(obs_is_vehicle, dtype=bool)
    past = (tt <= t0) & (tt >= t0 - float(max_stale_s)) & veh
    if not past.any():
        return None, float("nan"), float("nan")
    k = np.flatnonzero(past)
    # last sample per track inside the causal window
    order = k[np.argsort(tt[k], kind="stable")]
    last = {}
    for i in order:
        last[tid[i]] = i           # later samples overwrite earlier ones
    best, best_gap, best_sx = None, np.inf, float("nan")
    for trk, i in last.items():
        gap = float(x[i] - sx[i] / 2.0)
        if gap < 0.0 or gap > float(max_gap_m) or abs(float(y[i])) >= float(lat_m):
            continue
        if gap < best_gap:
            best, best_gap, best_sx = trk, gap, float(sx[i])
    if best is None:
        return None, float("nan"), float("nan")
    return best, float(best_gap), best_sx


def lead_track_in_window(obs_t, obs_track, obs_x, obs_y, track, t0: float,
                         ts: np.ndarray, ego_t, ego_x, ego_y, ego_yaw, *,
                         max_stale_s: float = MAX_STALE_S) -> np.ndarray:
    """The selected lead's centre at times ``ts``, in the WINDOW-ORIGIN ego frame at ``t0``.

    ``(K, 2)``, NaN where the track has no sample within ``max_stale_s``.

    Each cuboid is expressed in the rig frame **at its own timestamp**, so two samples 1 s apart
    live in two different frames. The composition is therefore mandatory::

        world: L_w(t) = ego_xy(t) + R(yaw(t)) @ [cx, cy]
        t0:    L_0(t) = R(-yaw(t0)) @ (L_w(t) - ego_xy(t0))

    Skipping it and using raw rig coordinates understates the gap by roughly the distance the ego
    covers over the horizon (~27 m at 13.6 m/s over 2 s) — it would invent tailgating everywhere.
    """
    tt = np.asarray(obs_t, dtype=np.float64)
    m = np.asarray(obs_track) == track
    ts = np.asarray(ts, dtype=np.float64)
    out = np.full((ts.size, 2), np.nan)
    if not m.any():
        return out
    k = np.flatnonzero(m)
    k = k[np.argsort(tt[k])]
    st, sx, sy = tt[k], np.asarray(obs_x, float)[k], np.asarray(obs_y, float)[k]
    j = np.searchsorted(st, ts, side="right") - 1
    jj = np.clip(j, 0, st.size - 1)
    ok = (j >= 0) & (np.abs(ts - st[jj]) <= float(max_stale_s))
    if not ok.any():
        return out
    et = np.asarray(ego_t, float)
    ex, ey, ey_yaw = (np.asarray(ego_x, float), np.asarray(ego_y, float),
                      np.asarray(ego_yaw, float))
    # ⚠️ the ego pose is taken at the CUBOID's own timestamp, not at the query time — that is
    # the frame the cuboid is expressed in. Using the query time here is a subtle ~metre error
    # whenever the track is stale.
    t_samp = st[jj]                       # [K] — the cuboid actually used for each query time
    cx_q, cy_q = sx[jj], sy[jj]           # [K] — its rig coordinates
    sxe, sye = np.interp(t_samp, et, ex), np.interp(t_samp, et, ey)
    yaw = np.interp(t_samp, et, ey_yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    wx = sxe + cx_q * c - cy_q * s
    wy = sye + cx_q * s + cy_q * c
    x0, y0 = float(np.interp(t0, et, ex)), float(np.interp(t0, et, ey))
    yaw0 = float(np.interp(t0, et, ey_yaw))
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    dx, dy = wx - x0, wy - y0
    out[ok, 0] = (dx * c0 + dy * s0)[ok]
    out[ok, 1] = (-dx * s0 + dy * c0)[ok]
    return out


# --------------------------------------------------------------------------- #
# 4. the block four_families consumes                                          #
# --------------------------------------------------------------------------- #
def lead_block(t0s, ts_rel, obs, ego, *, lat_m: float = LEAD_LAT_M,
               max_gap_m: float = LEAD_MAX_GAP_M,
               max_stale_s: float = MAX_STALE_S) -> dict:
    """Assemble ``win["lead"]`` for a list of window origins on ONE clip.

    Args:
        t0s:    ``[W]`` clip times (s) of each window's origin, from :func:`register_poses_to_time`.
        ts_rel: ``[K]`` horizon offsets (s) of the arm's waypoints — e.g. ``[0.5,1.0,1.5,2.0]`` for
                the sparse view, or ``arange(1,21)*0.1`` for the dense one. ⛔ These MUST be the
                same waypoint times the arm's ``pred`` is on, or headway is compared at the wrong
                instants.
        obs:    ``None`` when the clip has no `obstacle.offline`, else a dict of equal-length
                arrays ``t`` (s), ``track``, ``center_x``, ``center_y``, ``size_x``,
                ``is_vehicle``.
        ego:    dict of ``t`` (s), ``x``, ``y``, ``yaw`` (unwrapped), ``v`` — the egomotion track.

    Returns the ``lead=`` dict `four_families.longitudinal` takes (``leads``, ``lead_lens``,
    ``speeds``) plus the auditing fields this programme requires: a per-window ``state`` in
    {LEAD, NO_LEAD, NO_LABEL}, ``has_lead``, and the label span the states were decided against.
    """
    t0s = np.asarray(t0s, dtype=np.float64).reshape(-1)
    ts_rel = np.asarray(ts_rel, dtype=np.float64).reshape(-1)
    w, k = t0s.size, ts_rel.size
    leads = np.full((w, k, 2), np.nan)
    lead_lens = np.full(w, np.nan)
    speeds = np.full(w, np.nan)
    state = np.array([NO_LABEL] * w, dtype=object)
    gap0 = np.full(w, np.nan)

    et = np.asarray(ego["t"], float)
    speeds[:] = np.interp(t0s, et, np.asarray(ego["v"], float))

    if obs is None or np.asarray(obs["t"]).size == 0:
        span = None
    else:
        ot = np.asarray(obs["t"], float)
        span = (float(ot.min()), float(ot.max()))
        for i, t0 in enumerate(t0s):
            # NO_LABEL unless the whole window (origin AND horizon) is inside the labelled span.
            # A horizon that leaves the span looks exactly like an empty road, which is the bias.
            if not (span[0] - OBS_SPAN_GUARD_S <= t0
                    and t0 + ts_rel.max() <= span[1] + OBS_SPAN_GUARD_S):
                continue
            trk, g0, sx = select_lead_causal(
                ot, obs["track"], obs["center_x"], obs["center_y"], obs["size_x"],
                obs["is_vehicle"], float(t0), lat_m=lat_m, max_gap_m=max_gap_m,
                max_stale_s=max_stale_s)
            if trk is None:
                state[i] = NO_LEAD
                continue
            trk_xy = lead_track_in_window(
                ot, obs["track"], obs["center_x"], obs["center_y"], trk, float(t0),
                t0 + ts_rel, et, ego["x"], ego["y"], ego["yaw"], max_stale_s=max_stale_s)
            if not np.isfinite(trk_xy).any():
                # the lead was there at t0 but its track ends before the first waypoint:
                # labels present, no usable lead -> NO_LEAD, not NO_LABEL.
                state[i] = NO_LEAD
                continue
            state[i] = LEAD
            leads[i] = trk_xy
            lead_lens[i] = sx
            gap0[i] = g0

    has_lead = state == LEAD
    return {
        "leads": leads, "lead_lens": lead_lens, "speeds": speeds,
        "has_lead": has_lead, "state": state, "gap0_m": gap0,
        "ts_rel_s": ts_rel,
        "label_span_s": span,
        "counts": {LEAD: int((state == LEAD).sum()),
                   NO_LEAD: int((state == NO_LEAD).sum()),
                   NO_LABEL: int((state == NO_LABEL).sum())},
        "conventions": {
            "frame": "window-origin ego frame at t0; x forward, y left, metres, clip-local",
            "gap": "along - size_x/2 (rig origin to lead REAR face); NOT bumper-to-bumper",
            "selection": f"strictly causal, <= t0, staleness <= {max_stale_s}s, "
                         f"|lat| < {lat_m} m, gap <= {max_gap_m} m, classes {VEHICLE_CLASSES}",
            "NO_LABEL": "no obstacle.offline for the clip, or the window leaves the labelled "
                        "span. NEVER counted as free flow — that would manufacture empty road.",
        },
    }

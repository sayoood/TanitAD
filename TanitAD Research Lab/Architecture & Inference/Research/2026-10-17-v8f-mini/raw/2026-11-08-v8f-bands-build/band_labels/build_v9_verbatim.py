#!/usr/bin/env python3
"""build_v9_labels.py — the v9 per-frame label release (WP-A, refcv8 R8-1 / R8-2 / R8-3).

SPEC: ``FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-v9-labels/SPEC.md`` (accepted by the Master Mind
2026-10-04 with decisions D-WPA-1..8). Every literal below is the SPEC's, collected in :data:`LITERALS` and written
into the release manifest, so a release says what it was built with.

Layers (each a set of PURE functions; no module-level mutable state):

* the ego LOG (``egomotion_alpamayo/<clip>.parquet``): raw time = ``(timestamp - timestamp[0]) / 1e6``, read the way
  ``tanitad/data/egomotion_source.py`` reads it (unwrapped quaternion yaw, ``|v|``);
* the WHOLE-RECORDING 10 Hz track and its arc length (the builder's ``_arc_to`` rule: chords of the 10 Hz track);
* the TURN TABLE: the v7 builder's own ``manoeuvre_sequence`` / ``is_turn`` (``s2_geom_emit_v7.py``) over the whole
  recording, with the three builder defects fixed (SPEC §5.1);
* NAV per frame (SPEC §5.2) — SPATIAL: arc positions and the CURRENT speed only;
* the ROUTE CHECKPOINT (SPEC §6) — the driven path smoothed in arc length, read at a fixed arc ahead.

⛔ Clip ids never leave this module in the clear: outputs carry ``sid`` (``stable_episode_id``) and ``sha12``.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------------------------------------------- #
# LITERALS (SPEC.md; each with its section)                                                                         #
# --------------------------------------------------------------------------------------------------------------- #
LITERALS: dict = {
    "band_s": (2.0, 8.0),                 # §2.2  B = [NOW+2, NOW+8]
    "grid_dt_s": 0.1,                     # §2.3
    "g_max_s": 1.0,                       # §2.4  largest admissible log gap inside a time-domain interval
    # §2.4 shortest observed horizon for an ABSENCE claim. The SPEC's 6.0 s was GATED by the V7 truncation test, which
    # FAILED on eval139 (truncating full-band frames to 6 s changed 5.3 % lat-a / 7.3 % lat-b / 15.4 % lon labels,
    # bar <= 5 %, raw/s3_validation_eval139.json), so the pre-registered fallback applies: the full band.
    "h_abs_min_s": 8.0,
    "h_min_any_s": 3.0,                   # §2.4  below: every action/goal cell IGNORE
    "track_hz": 10.0,                     # §3.1  the builder's label timeline
    "turn_band_deg": 30.0,                # §3.2
    "nav_h_s": 6.0,                       # §5.2  D_ann = max(nav_d_min_m, nav_h_s * v_now)
    "nav_d_min_m": 30.0,                  # §5.2
    "suppress_window_raw_s": (8.0, 43.0),  # §5.1  A7 clip-level suppression within the builder's own 35-s view
    "rc_ds_m": 0.5,                       # §6.1
    "rc_sigma_m": 8.0,                    # §6.1  S_8, the INPUT path
    "rc_sigma_heavy_m": 25.0,             # §6.1  S_25, diagnostic reference only
    "rc_kernel_sigmas": 3.0,              # §6.1
    "rc_sagitta_max_m": 0.25,             # §6.1  chord interpolation fidelity
    "rc_a_lookahead_m": (30.0, 50.0, 80.0),  # §6.2  RC-A
    "rc_b_clamp_m": (20.0, 80.0),         # §6.2  RC-B
}


# --------------------------------------------------------------------------------------------------------------- #
# ids                                                                                                               #
# --------------------------------------------------------------------------------------------------------------- #
def stable_episode_id(clip_id: str) -> int:
    """``tanitad/data/v2_dataset.py:69`` restated (blake2b-8, >> 1) — the trainer's ``ep.episode_id``."""
    return int.from_bytes(hashlib.blake2b(clip_id.encode("utf-8"), digest_size=8).digest(), "big") >> 1


def sha12(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------------------------------------------- #
# the ego log                                                                                                       #
# --------------------------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EgoLog:
    """Native log samples (immutable arrays). ``ts`` raw seconds from the recording start."""
    ts: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray        # unwrapped quaternion yaw, rad (CCW +, left +)
    v: np.ndarray          # |(vx, vy, vz)|, m/s
    kappa: np.ndarray      # the provider's own curvature channel, 1/m (an INDEPENDENT channel for validation)
    rev: np.ndarray        # 1.0 where the ego moves BACKWARDS (world-frame velocity opposite the body heading), else 0.0

    @property
    def t_end(self) -> float:
        return float(self.ts[-1])


def _ro(a) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float64)
    a.setflags(write=False)
    return a


def yaw_from_quaternion(qx, qy, qz, qw):
    """ZYX yaw, as ``egomotion_source._yaw_from_quaternion``."""
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def load_log(path: str) -> EgoLog:
    import pandas as pd
    df = pd.read_parquet(path)
    ts = df["timestamp"].to_numpy(np.float64) / 1e6
    ts = ts - ts[0]
    yaw = np.unwrap(yaw_from_quaternion(*(df[c].to_numpy(np.float64) for c in ("qx", "qy", "qz", "qw"))))
    v = np.linalg.norm(df[["vx", "vy", "vz"]].to_numpy(np.float64), axis=1)
    vx, vy = df["vx"].to_numpy(np.float64), df["vy"].to_numpy(np.float64)
    # MEASURED 2026-10-04: the log velocity is WORLD-frame (atan2(vy, vx) tracks the quaternion yaw within 0.05-0.19 deg
    # forward; 180 deg on a reversing clip), so reversing = moving (> 0.3 m/s) against the body heading.
    rev = ((np.hypot(vx, vy) > 0.3) & (np.cos(np.arctan2(vy, vx) - yaw) < 0.0)).astype(np.float64)
    return log_from_arrays(ts, df["x"].to_numpy(np.float64), df["y"].to_numpy(np.float64), yaw, v,
                           df["curvature"].to_numpy(np.float64), rev)


def log_from_arrays(ts, x, y, yaw, v, kappa=None, rev=None) -> EgoLog:
    """Synthetic logs (analytic controls) enter through here too."""
    ts = np.asarray(ts, np.float64)
    if np.any(np.diff(ts) <= 0):
        keep = np.concatenate([[True], np.diff(ts) > 0])
        ts, x, y, yaw, v = ts[keep], np.asarray(x)[keep], np.asarray(y)[keep], np.asarray(yaw)[keep], np.asarray(v)[keep]
        kappa = None if kappa is None else np.asarray(kappa)[keep]
        rev = None if rev is None else np.asarray(rev)[keep]
    if kappa is None:
        kappa = np.full(len(ts), np.nan)
    if rev is None:
        rev = np.zeros(len(ts))
    return EgoLog(_ro(ts), _ro(x), _ro(y), _ro(yaw), _ro(v), _ro(kappa), _ro(rev))


def reversing_s(log: EgoLog, a: float, b: float) -> float:
    """Seconds of reversing motion inside [a, b] (time-weighted native samples)."""
    m = (log.ts >= a) & (log.ts <= b)
    i = np.nonzero(m)[0]
    if i.size < 2:
        return 0.0
    dt = np.diff(log.ts[i])
    return float((0.5 * (log.rev[i][1:] + log.rev[i][:-1]) * dt).sum())


def sample(log: EgoLog, t) -> tuple:
    """(x, y, yaw, v) at times ``t`` — linear interpolation of the native samples."""
    t = np.asarray(t, np.float64)
    return (np.interp(t, log.ts, log.x), np.interp(t, log.ts, log.y), np.interp(t, log.ts, log.yaw),
            np.interp(t, log.ts, log.v))


def max_gap(log: EgoLog, a: float, b: float) -> float:
    """Largest gap between consecutive native samples among the gaps that overlap [a, b] (inf if b > t_end)."""
    if b > log.t_end + 1e-9 or a < log.ts[0] - 1e-9:
        return math.inf
    lo = max(int(np.searchsorted(log.ts, a, "right")) - 1, 0)
    hi = min(int(np.searchsorted(log.ts, b, "left")), len(log.ts) - 1)
    if hi <= lo:
        return 0.0
    return float(np.diff(log.ts[lo:hi + 1]).max())


def h_obs(log: EgoLog, now: float, horizon: float = 8.0, g_max: float = LITERALS["g_max_s"]) -> float:
    """SPEC §2.4: min(horizon, start of the first gap > g_max after NOW, log end) - NOW."""
    i = max(int(np.searchsorted(log.ts, now, "right")) - 1, 0)
    g = np.diff(log.ts[i:])
    big = np.nonzero(g > g_max)[0]
    t_gap = log.ts[i + big[0]] if big.size else math.inf
    return float(max(0.0, min(horizon, t_gap - now, log.t_end - now)))


# --------------------------------------------------------------------------------------------------------------- #
# the whole-recording 10 Hz track and its arc length (the builder's own geometry)                                    #
# --------------------------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Track10:
    t: np.ndarray          # 0, 0.1, ... raw s
    poses: np.ndarray      # [T, 4] x, y, yaw (unwrapped), v  — as egomotion_source.load
    s: np.ndarray          # cumulative chord length of the 10 Hz track, m


def track10(log: EgoLog, hz: float = LITERALS["track_hz"]) -> Track10:
    grid = np.arange(0.0, log.t_end + 1e-9, 1.0 / hz)
    x, y, yaw, v = sample(log, grid)
    poses = np.stack([x, y, yaw, v], axis=1)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return Track10(_ro(grid), _ro(poses), _ro(s))


def s_at(tr: Track10, t):
    """Arc position (m) at raw time(s) ``t`` — linear in the 10 Hz arc."""
    return np.interp(np.asarray(t, np.float64), tr.t, tr.s)


def t_at_s(tr: Track10, s):
    """First raw time at which the arc reaches ``s`` (the arc is non-decreasing; stops are flat runs)."""
    s = np.asarray(s, np.float64)
    i = np.searchsorted(tr.s, s, "left")
    i = np.clip(i, 1, len(tr.s) - 1)
    s0, s1 = tr.s[i - 1], tr.s[i]
    f = np.where(s1 > s0, (s - s0) / np.maximum(s1 - s0, 1e-12), 0.0)
    return tr.t[i - 1] + np.clip(f, 0, 1) * (tr.t[i] - tr.t[i - 1])


# --------------------------------------------------------------------------------------------------------------- #
# the turn table (SPEC §3.1, §5.1)                                                                                 #
# --------------------------------------------------------------------------------------------------------------- #
def _builder():
    """The v7 builder module, imported from THIS tree's ``stack/scripts`` (its md5 is recorded by the caller)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import s2_geom_emit_v7 as S2      # noqa: E402
    return S2


@dataclass(frozen=True)
class Turn:
    t_start: float         # raw s
    t_end: float
    s_start: float         # arc m
    s_end: float
    dyaw_deg: float        # signed, left +
    r_arc_m: float
    v_min_ms: float
    is_turn: bool
    gap_affected: bool
    suppressed: bool

    @property
    def side(self) -> int:
        return 1 if self.dyaw_deg > 0 else -1

    @property
    def announced(self) -> bool:
        return self.is_turn and not self.suppressed


def turn_table(log: EgoLog, tr: Track10, suppression: dict | None = None,
               g_max: float = LITERALS["g_max_s"]) -> tuple:
    """Every sustained-yaw segment of the WHOLE recording (builder literals), as :class:`Turn`.

    Builder defects fixed (SPEC §5.1): (1) suppression read from ``turn_suppression.applied`` — the emitter's own key;
    (2) no horizon cap; (3) every segment is returned, not ``seq[0]`` only.  Suppression scope = A7's clip-level rule
    restricted to the builder's own 35-s view (raw [8, 43] s, D-WPA-3).
    """
    S2 = _builder()
    hz = LITERALS["track_hz"]
    segs = S2.manoeuvre_sequence(tr.poses, 0, hz=hz)
    sup_on = bool(suppression) and bool(suppression.get("applied"))
    w0, w1 = LITERALS["suppress_window_raw_s"]
    out = []
    for seg in segs:
        ta, tb = float(seg[0]), float(seg[1])
        is_t = bool(S2.is_turn(seg))
        gap = max_gap(log, max(ta - 0.5, 0.0), min(tb + 0.5, log.t_end)) > g_max
        sup = bool(sup_on and w0 <= ta <= w1)
        out.append(Turn(ta, tb, float(s_at(tr, ta)), float(s_at(tr, tb)), float(seg[2]), float(seg[3]),
                        float(seg[4]), is_t, bool(gap), sup))
    return tuple(out)


# --------------------------------------------------------------------------------------------------------------- #
# NAV per frame (SPEC §5.2) — spatial, current speed only                                                          #
# --------------------------------------------------------------------------------------------------------------- #
NAV_FOLLOW, NAV_TURN_L, NAV_TURN_R = 0, 1, 2


def nav_fields(turns: tuple, s_now: float, now: float, v_now: float, s_max: float) -> dict:
    ann = [u for u in turns if u.announced and u.s_end > s_now]
    d_ann = max(LITERALS["nav_d_min_m"], LITERALS["nav_h_s"] * max(v_now, 0.0))
    out = {"nav_token": NAV_FOLLOW, "nav_side_next": 0, "nav_d_next_m": np.nan, "nav_d_end_m": np.nan,
           "nav_dyaw_next_deg": np.nan, "nav_args_valid": 0, "nav_lookahead_m": float(max(s_max - s_now, 0.0)),
           "nav_t_next_s": np.nan, "nav_token_ttime": NAV_FOLLOW, "nav_gap_affected": 0, "nav_in_progress": 0,
           "nav_d_ann_m": d_ann}
    if not ann:
        return out
    u = min(ann, key=lambda q: q.s_start)
    in_prog = u.s_start <= s_now < u.s_end
    d_next = max(0.0, u.s_start - s_now)
    tok = (NAV_TURN_L if u.side > 0 else NAV_TURN_R)
    out.update({"nav_side_next": u.side, "nav_d_next_m": d_next, "nav_d_end_m": u.s_end - s_now,
                "nav_dyaw_next_deg": u.dyaw_deg, "nav_args_valid": 1, "nav_t_next_s": u.t_start - now,
                "nav_gap_affected": int(u.gap_affected), "nav_in_progress": int(in_prog)})
    if in_prog or d_next <= d_ann:
        out["nav_token"] = tok
    if in_prog or (u.t_start - now) <= LITERALS["nav_h_s"]:
        out["nav_token_ttime"] = tok
    return out


# --------------------------------------------------------------------------------------------------------------- #
# the ROUTE CHECKPOINT (SPEC §6)                                                                                   #
# --------------------------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SmoothPath:
    s: np.ndarray          # uniform arc grid, m (from the recording start)
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray        # unwrapped tangent heading, rad
    sigma: float
    s_hi_valid: float      # last arc with a complete forward kernel


def smooth_path(tr: Track10, sigma: float, ds: float = LITERALS["rc_ds_m"],
                n_sig: float = LITERALS["rc_kernel_sigmas"]) -> SmoothPath:
    """The 10 Hz track resampled at ``ds`` in arc length and Gaussian-smoothed in arc length (kernel truncated at
    ±n_sig·σ and renormalised; at the recording START the backward half is truncated, at the END the arc beyond
    ``s_hi_valid`` is not admitted)."""
    s_all, x_all, y_all = tr.s, tr.poses[:, 0], tr.poses[:, 1]
    keep = np.concatenate([[True], np.diff(s_all) > 1e-6])
    s_u, x_u, y_u = s_all[keep], x_all[keep], y_all[keep]
    grid = np.arange(0.0, s_u[-1] + 1e-9, ds)
    gx, gy = np.interp(grid, s_u, x_u), np.interp(grid, s_u, y_u)
    if sigma <= 0 or len(grid) < 3:
        xs, ys = gx, gy
    else:
        h = int(math.ceil(n_sig * sigma / ds))
        k = np.exp(-0.5 * (np.arange(-h, h + 1) * ds / sigma) ** 2)
        z = np.zeros(h)

        def conv(a):        # zero-padded 'valid' convolution: output length == len(a) for any len(a)
            return np.convolve(np.concatenate([z, a, z]), k, mode="valid")
        den = conv(np.ones_like(gx))
        xs, ys = conv(gx) / den, conv(gy) / den
    dx, dy = np.gradient(xs, ds), np.gradient(ys, ds)
    psi = np.unwrap(np.arctan2(dy, dx))
    s_hi = float(grid[-1] - n_sig * sigma) if sigma > 0 else float(grid[-1])
    return SmoothPath(_ro(grid), _ro(xs), _ro(ys), _ro(psi), float(sigma), s_hi)


def to_now_frame(px, py, x0, y0, psi0):
    c, s = math.cos(psi0), math.sin(psi0)
    dx, dy = np.asarray(px) - x0, np.asarray(py) - y0
    return c * dx + s * dy, -s * dx + c * dy


def chord_sagitta_max(tr: Track10, log: EgoLog, s_lo: float, s_hi: float, s_nat=None) -> float:
    """Largest ``c·|Δψ|/8`` over the NATIVE log chords whose arc lies in [s_lo, s_hi] (SPEC §6.1)."""
    if s_nat is None:
        s_nat = s_at(tr, log.ts)
    m = (s_nat >= s_lo) & (s_nat <= s_hi)
    i = np.nonzero(m)[0]
    if i.size < 2:
        return math.inf
    i = np.arange(max(i[0] - 1, 0), min(i[-1] + 2, len(log.ts)))
    c = np.hypot(np.diff(log.x[i]), np.diff(log.y[i]))
    dpsi = np.abs(np.diff(log.yaw[i]))
    return float((c * dpsi / 8.0).max())


def rc_point(path: SmoothPath, s_target: float, x0: float, y0: float, psi0: float) -> tuple:
    """(x, y, psi_rel_deg, ok) of the smoothed path at arc ``s_target`` in the NOW frame."""
    if s_target > path.s_hi_valid or s_target < path.s[0]:
        return np.nan, np.nan, np.nan, False
    px, py = np.interp(s_target, path.s, path.x), np.interp(s_target, path.s, path.y)
    pp = np.interp(s_target, path.s, path.psi)
    xl, yl = to_now_frame(px, py, x0, y0, psi0)
    rel = (pp - psi0 + math.pi) % (2 * math.pi) - math.pi
    return float(xl), float(yl), float(math.degrees(rel)), True


def rc_b_lookahead(turns: tuple, s_now: float) -> tuple:
    """SPEC §6.2: L_B = clip(s_end(next announced turn) - s_now, 20, 80); kind code."""
    lo, hi = LITERALS["rc_b_clamp_m"]
    ann = [u for u in turns if u.announced and u.s_end > s_now]
    if not ann:
        return hi, "no_turn"
    u = min(ann, key=lambda q: q.s_start)
    d = u.s_end - s_now
    if d > hi:
        return hi, "clamped_max"
    if d < lo:
        return lo, "clamped_min"
    return float(d), "turn_end"


@dataclass(frozen=True)
class ClipGeo:
    """Everything per clip that every frame reads: the log, the 10 Hz track, the arc of the native samples, the turn
    table and the smoothed paths. Built once per clip by :func:`clip_geo`; immutable."""
    log: EgoLog
    tr: Track10
    s_nat: np.ndarray
    turns: tuple
    paths: dict

    @property
    def s_max(self) -> float:
        return float(self.tr.s[-1])


def clip_geo(log: EgoLog, suppression: dict | None = None) -> ClipGeo:
    tr = track10(log)
    paths = {LITERALS["rc_sigma_m"]: smooth_path(tr, LITERALS["rc_sigma_m"]),
             LITERALS["rc_sigma_heavy_m"]: smooth_path(tr, LITERALS["rc_sigma_heavy_m"])}
    return ClipGeo(log, tr, _ro(s_at(tr, log.ts)), turn_table(log, tr, suppression), paths)


def rc_fields(tr: Track10, log: EgoLog, paths: dict, turns: tuple, now: float, s_nat=None,
              sigma: float | None = None, sigma_h: float | None = None) -> dict:
    """Every RC variant for one frame. ``paths`` = {sigma: SmoothPath} holding the input (8 m) and heavy (25 m)
    paths (or the ``sigma`` / ``sigma_h`` given, e.g. the Stage-2 σ = 15 m lever)."""
    x0, y0, psi0, _ = (float(a) for a in sample(log, now))
    s_now = float(s_at(tr, now))
    sig = LITERALS["rc_sigma_m"] if sigma is None else float(sigma)
    sig_h = LITERALS["rc_sigma_heavy_m"] if sigma_h is None else float(sigma_h)
    nk = LITERALS["rc_kernel_sigmas"]
    out = {"s_now": s_now}
    lb, kind = rc_b_lookahead(turns, s_now)
    out["rc_b_L_m"], out["rc_b_kind"] = lb, kind
    for name, L in [(f"A{int(a)}", a) for a in LITERALS["rc_a_lookahead_m"]] + [("B", lb)]:
        for tag, sg in (("", sig), ("H", sig_h)):
            x, y, p, ok = rc_point(paths[sg], s_now + L, x0, y0, psi0)
            if ok and tag == "":
                sag = chord_sagitta_max(tr, log, s_now - nk * sg, s_now + L + nk * sg, s_nat)
                ok = sag <= LITERALS["rc_sagitta_max_m"]
            out[f"rc{tag}_{name}_x"], out[f"rc{tag}_{name}_y"] = x, y
            out[f"rc{tag}_{name}_psi"], out[f"rc{tag}_{name}_valid"] = p, int(ok)
    return out


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def assert_no_g_imports() -> None:
    """⛔ The venv's editable ``tanitad`` maps to G:. Refuse if anything imported lives there."""
    bad = sorted({getattr(m, "__file__", "") or "" for m in list(sys.modules.values())
                  if (getattr(m, "__file__", "") or "").upper().startswith("G:")})
    if bad:
        raise SystemExit(f"[v9] ⛔ modules imported from G: {bad[:3]}")


# =============================================================================================================== #
# STAGE 3 — per-frame actions, goals, lead (FOLLOW), lane change (SAM3), speed proxy, release assembly             #
# =============================================================================================================== #
LITERALS.update({
    "v_stop_ms": 0.5, "v_creep_ms": 2.0, "dv_ms": 1.5,             # §3.5
    "vbar_window_s": 0.5, "plateau_tol_ms": 0.1,                   # §3.5 / S3-A1 item 5
    "lead_corridor_m": 1.75, "lead_range_m": 100.0, "lead_min_s_m": 2.0,   # §3.5 / S3-A1 item 3
    "lead_front_offset_m": 3.6, "lead_tg_max_s": 3.0, "lead_v_min_ms": 2.0,
    "lead_cov_min_s": 3.0, "lead_follow_frac": 0.5,
    "ego_footprint": (-1.5, 4.5, 1.2),                             # x_lo, x_hi, |y| (D3 ego-box defect)
    "lead_classes": ("automobile", "heavy_truck", "bus", "trailer", "other_vehicle", "train_or_tram_car", "rider"),
    "lc_rows": (50, 60, 70, 80), "lc_y_max_m": 4.5, "lc_width_m": (2.5, 4.5),   # §3.4 / S3-A1 item 2
    "lc_u_hi": 0.7, "lc_u_lo": 0.3, "lc_bridge_s": 1.0, "lc_turn_tol_deg": 10.0, "lc_persist_s": 3.0,
    "lc_measurable_frac": 0.5, "lc_partial_exc_m": 2.0,
    "reverse_mask_s": 0.3,           # post-validation fix (V2 diagnosis): >= 0.3 s of reversing in [NOW, NOW+8] masks actions
    "yield_stop_s": 0.5,                                           # §4.1 YIELD_FOR_TURN (builder definition)
    "vlm_half_window_s": 2.0,                                      # §4.3 |NOW - NOW*| <= 2
    "red_stop_min_s": 1.0, "red_app_v_ms": 5.0, "red_back_s": 8.0, "red_tail_s": 3.0,   # §4.3 RED propagation
    "vlm_lat_evidence_m": 0.3,
    "speed_past_s": 20.0, "speed_ladder_kmh": (20, 30, 50, 70, 80, 100, 120, 130), "speed_urban_floor_kmh": 50,
})

LAT9 = ("LANE_KEEP", "TURN_L", "TURN_R", "LANE_CHANGE_L", "LANE_CHANGE_R")
LON9 = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP")
#: the frozen v7 vocabulary (``tanitad/models/vocab_v7.py:320-331``), restated so the release is self-describing
LAT7 = ("LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC", "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R")
LON7 = ("FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD", "ADAPT_SPEED_FOR_CURVE", "ACCELERATE")
LAT9_TO_7 = {0: 0, 1: 6, 2: 7, 3: 1, 4: 2}
LON9_TO_7 = {0: 5, 1: 4, 2: 3, 3: 0, 4: 3, 5: 7, 6: 1}
GOAL22 = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "YIELD", "STOP_POINT",
          "SPEED_BAND", "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR", "OVERTAKE_VEHICLE", "MERGE", "GAP_TARGET",
          "REACT_ON_ONCOMING", "TAKE_EXIT_L", "TAKE_EXIT_R", "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED",
          "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN", "LANE_CHANGE_L", "LANE_CHANGE_R")
GOAL_PROV = {t: "geometry" for t in ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R",
                                     "STOP_POINT", "SPEED_BAND")}
GOAL_PROV.update({t: "vlm" for t in GOAL22 if t not in GOAL_PROV})
GOAL_PROV.update({"LANE_CHANGE_L": "sam3|vlm", "LANE_CHANGE_R": "sam3|vlm"})
VLM_TOKENS = tuple(t for t in GOAL22 if GOAL_PROV[t] == "vlm")
TL_TOKENS = ("TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN")
IGNORE = -100
COT_ABSENCE_POLICY_ID = "cot-absence-negative/2026-09-16"


def _bits(idx) -> int:
    out = 0
    for i in idx:
        out |= 1 << int(i)
    return out


def _gi(tok: str) -> int:
    return GOAL22.index(tok)


# ------------------------------------------------------------------------------------------- the trainer clock #
def clip_clock(sid: int, sidecar_table: dict, cache_poses) -> tuple:
    """``refc_v3_train.py::_clock_for`` restated: the sidecar when it holds the clip, else ``clip_clock.pose_dt`` with
    ``grid_start_s = 0.0``, else the nominal dt. Returns (g0, dt, src)."""
    if sid in sidecar_table:
        g0, dt = sidecar_table[sid]
        return float(g0), float(dt), "sidecar"
    import torch
    from tanitad.data import clip_clock as _cc
    dt = _cc.pose_dt(torch.as_tensor(np.asarray(cache_poses), dtype=torch.float64))
    return (0.0, float(dt), "pose_dt") if dt is not None else (0.0, float(_cc.NOMINAL_DT_S), "nominal_dt")


# ------------------------------------------------------------------------------------------- lead (FOLLOW) #
def lead_for_frame(log: EgoLog, tr: Track10, path0: SmoothPath, t: float, agents: list) -> tuple:
    """(lead?, gap_m, time_gap_s, s_along_m, v_ego) at raw time ``t`` for the agents of that frame (rig frame)."""
    x0, y0, psi0, v = (float(a) for a in sample(log, t))
    s_k = float(s_at(tr, t))
    rng = LITERALS["lead_range_m"]
    m = (path0.s >= s_k) & (path0.s <= s_k + rng)
    if m.sum() < 2:
        return False, np.nan, np.nan, np.nan, v
    px, py = to_now_frame(path0.x[m], path0.y[m], x0, y0, psi0)
    ps = path0.s[m] - s_k
    xlo, xhi, yab = LITERALS["ego_footprint"]
    cls_ok = LITERALS["lead_classes"]
    sel = [a for a in agents if a.get("cls") in cls_ok]
    if not sel:
        return False, np.nan, np.nan, np.nan, v
    cx = np.array([float(a["cx"]) for a in sel])
    cy = np.array([float(a["cy"]) for a in sel])
    ln = np.array([float(a.get("l", 4.5)) for a in sel])
    keep = ~((cx > xlo) & (cx < xhi) & (np.abs(cy) < yab))          # never the ego's own footprint
    if not keep.any():
        return False, np.nan, np.nan, np.nan, v
    cx, cy, ln = cx[keep], cy[keep], ln[keep]
    d2 = (cx[:, None] - px[None, :]) ** 2 + (cy[:, None] - py[None, :]) ** 2     # [A, P]
    j = d2.argmin(1)
    dmin = d2[np.arange(len(cx)), j]
    sa = ps[j]
    ok = (dmin <= LITERALS["lead_corridor_m"] ** 2) & (sa >= LITERALS["lead_min_s_m"]) & (sa <= rng)
    if not ok.any():
        return False, np.nan, np.nan, np.nan, v
    b = int(np.argmin(np.where(ok, sa, np.inf)))
    gap = float(sa[b]) - 0.5 * float(ln[b]) - LITERALS["lead_front_offset_m"]
    return True, gap, max(gap, 0.0) / max(v, 1.0), float(sa[b]), v


def iter_join_clips(join_path: str, wanted: set):
    """Stream the per-frame agent join (grouped by clip, MEASURED: 4,566 clips, 0 revisits) and yield
    ``(clip_id, {raw_frame: agents})`` for the wanted clips only."""
    import json as _json
    import lzma
    cur, frames = None, {}
    with lzma.open(join_path, "rt", encoding="utf-8") as f:
        for line in f:
            # ⛔ the file is COMPACT json ('"clip_id":"…"'); a find() that misses returns -1 and a fixed slice then
            # silently reads 36 wrong characters (MEASURED: the first lead pass matched 0 clips). Refuse instead.
            i = line.find('"clip_id":')
            if i < 0:
                raise ValueError("join line without a clip_id key")
            q0 = line.index('"', i + 10)
            cid = line[q0 + 1:line.index('"', q0 + 1)]
            if cid != cur:
                if cur is not None and cur in wanted:
                    yield cur, frames
                cur, frames = cid, {}
            if cid in wanted:
                r = _json.loads(line)
                frames[int(r["frame"])] = r["agents"]
    if cur is not None and cur in wanted:
        yield cur, frames


# ------------------------------------------------------------------------------------------- lane change (SAM3) #
def lc_offsets_from_codes(fine_codes) -> np.ndarray:
    """[F, 1000, 600] uint8 SAM3 fine codes -> [F, 3] (d_L, d_R, n_rows) per map frame (S3-A1 item 2).
    Columns: y = -30 + (j + 0.5) * 0.1 (``tanitad.sam3_map_gt/3`` fine anchoring)."""
    rows = list(LITERALS["lc_rows"])
    ymax = LITERALS["lc_y_max_m"]
    j0, j1 = int(round((30 - ymax) / 0.1)), int(round((30 + ymax) / 0.1))
    y = -30.0 + (np.arange(j0, j1) + 0.5) * 0.1
    sub = np.asarray(fine_codes[:, rows, j0:j1]) == 2                    # [F, R, J]
    yl = np.where(sub & (y > 0)[None, None, :], y[None, None, :], np.inf).min(-1)
    yr = np.where(sub & (y < 0)[None, None, :], -y[None, None, :], np.inf).min(-1)
    both = np.isfinite(yl) & np.isfinite(yr)
    n = both.sum(1)
    dl = np.where(n >= 2, np.nanmedian(np.where(both, yl, np.nan), axis=1), np.nan)
    dr = np.where(n >= 2, np.nanmedian(np.where(both, yr, np.nan), axis=1), np.nan)
    return np.stack([dl, dr, n.astype(np.float64)], 1)


def lc_crossings(k: np.ndarray, dl: np.ndarray, dr: np.ndarray, now_of_k, log: EgoLog) -> tuple:
    """Lane-line crossings from the per-frame offsets: (t_cross raw, side +1 L / -1 R) after the turn and
    persistence filters; also u per frame (NaN where invalid)."""
    w = dl + dr
    lo, hi = LITERALS["lc_width_m"]
    ok = np.isfinite(w) & (w >= lo) & (w <= hi)
    u = np.where(ok, dr / np.where(ok, w, 1.0), np.nan)
    t = np.asarray([now_of_k(int(kk)) for kk in k])
    vi = np.nonzero(ok)[0]
    raw = []
    for a, b in zip(vi[:-1], vi[1:]):
        if t[b] - t[a] > LITERALS["lc_bridge_s"] + 0.11:
            continue
        if u[a] >= LITERALS["lc_u_hi"] and u[b] <= LITERALS["lc_u_lo"]:
            raw.append((0.5 * (t[a] + t[b]), 1))
        elif u[a] <= LITERALS["lc_u_lo"] and u[b] >= LITERALS["lc_u_hi"]:
            raw.append((0.5 * (t[a] + t[b]), -1))
    out = []
    P = LITERALS["lc_persist_s"]
    for i, (tc, sd) in enumerate(raw):
        if any(abs(t2 - tc) <= P and s2 == -sd for j, (t2, s2) in enumerate(raw) if j != i):
            continue                                         # undone within 3 s: wobble, not a lane change
        if tc - 3.0 < log.ts[0] or tc + 3.0 > log.t_end:
            continue
        yaws = sample(log, np.array([tc - 3.0, tc + 3.0]))[2]
        if abs(math.degrees(yaws[1] - yaws[0])) > LITERALS["lc_turn_tol_deg"]:
            continue
        out.append((tc, sd))
    return tuple(out), u, t


# ------------------------------------------------------------------------------------------- helpers #
def _vbar(v, dt=0.1):
    n = max(1, int(round(LITERALS["vbar_window_s"] / dt)))
    k = np.ones(n) / n
    num = np.convolve(v, k, mode="same")
    den = np.convolve(np.ones_like(v), k, mode="same")
    return num / den


def _extremum_after(vb, j0, j_end, sign):
    """index of the first local extremum of vb at/after j0 (min for sign -1, max for +1), plateau tol; truncated?"""
    tol = LITERALS["plateau_tol_ms"]
    best = j0
    for j in range(j0 + 1, j_end + 1):
        if sign < 0:
            if vb[j] < vb[best]:
                best = j
            elif vb[j] > vb[best] + tol:
                return best, False
        else:
            if vb[j] > vb[best]:
                best = j
            elif vb[j] < vb[best] - tol:
                return best, False
    return best, True


def _extremum_before(vb, j0, sign):
    """last opposite extremum before j0 (a max before a decel crossing, sign -1; a min before an accel, +1)."""
    tol = LITERALS["plateau_tol_ms"]
    best = j0
    for j in range(j0 - 1, -1, -1):
        if sign < 0:
            if vb[j] > vb[best]:
                best = j
            elif vb[j] < vb[best] - tol:
                return best
        else:
            if vb[j] < vb[best]:
                best = j
            elif vb[j] > vb[best] + tol:
                return best
    return best


def d2_excursion_signed(x, y, psi):
    """D2's detrended lateral excursion (signed, left +) and net heading change (deg) of a path segment."""
    dpsi = np.degrees(psi - psi[0])
    c, s = math.cos(psi[0]), math.sin(psi[0])
    lat = -s * (x - x[0]) + c * (y - y[0])
    arcs = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    L = float(arcs[-1])
    kap = math.radians(float(dpsi[-1])) / L if L > 1e-6 else 0.0
    lat_arc = (1.0 - np.cos(kap * arcs)) / kap if abs(kap) > 1e-9 else np.zeros_like(arcs)
    res = lat - lat_arc
    j = int(np.argmax(np.abs(res)))
    return float(res[j]), float(dpsi[-1])


def cot_text_from_record(rec: dict) -> str:
    """``alpamayo_fusion.cot_text`` restated on the label record's own ``cot_source`` (no Alpamayo export needed):
    cot + chain_of_causation (dropped on a recorded direction conflict) + components + motion, de-duplicated."""
    cs = rec.get("cot_source") or {}
    chain = cs.get("chain_of_causation") or ""
    if (cs.get("conflict") or {}).get("conflict"):
        chain = ""
    seen = []
    for p in (cs.get("cot") or "", chain, cs.get("components_analysis") or "", cs.get("motion_analysis") or ""):
        p = (p or "").strip()
        if p and p not in seen:
            seen.append(p)
    return "  ".join(seen)


def vlm_tokens_for_record(rec: dict, geom_side: int) -> tuple:
    """(tokens set, has_cot) — the builder's CoT path WITHOUT its NUDGE lateral-evidence gate (D-WPA-6), keeping
    the side-conflict gate (geometry wins) and the both-exit-sides rule."""
    from tanitad.data import cot_tokens_v7 as COT
    text = cot_text_from_record(rec)
    if not text:
        return frozenset(), False
    g = COT.goals_from_cot(text)
    both_exit = "TAKE_EXIT_L" in g and "TAKE_EXIT_R" in g
    sided = {"TAKE_EXIT_L": 1, "TAKE_EXIT_R": -1, "LANE_CHANGE_L": 1, "LANE_CHANGE_R": -1}
    out = set()
    for t in g:
        if t.startswith("TAKE_EXIT_") and both_exit:
            continue
        if geom_side and t in sided and sided[t] != geom_side:
            continue
        out.add(t)
    return frozenset(out), True


def tl_probe_negative(rec: dict) -> bool:
    sc = rec.get("scene") or {}
    asked = str(sc.get("asked") or "").lower()
    return ("traffic light" in asked) and (sc.get("traffic_light_visible") is False)


def red_stop_episode(log: EgoLog, w0: float, w1: float):
    """The stop episode (v <= 0.5 for >= 1.0 s) overlapping [w0, w1]: (t_app, t_a, t_b) or None (§4.3)."""
    tg = np.arange(max(0.0, w0 - 10.0), min(log.t_end, w1 + 30.0), 0.1)
    v = sample(log, tg)[3]
    stop = v <= LITERALS["v_stop_ms"]
    i = 0
    while i < len(tg):
        if stop[i]:
            j = i
            while j + 1 < len(tg) and stop[j + 1]:
                j += 1
            ta, tb = tg[i], tg[j] + 0.1
            if tb - ta >= LITERALS["red_stop_min_s"] and ta <= w1 and tb >= w0:
                back = np.nonzero((tg < ta) & (tg >= ta - LITERALS["red_back_s"]) & (v >= LITERALS["red_app_v_ms"]))[0]
                t_app = float(tg[back[-1]]) if back.size else float(ta - LITERALS["red_back_s"])
                return t_app, float(ta), float(tb)
            i = j + 1
        else:
            i += 1
    return None


def speed_proxy(log: EgoLog, now: float) -> tuple:
    """D4's N2 / N3 on the log: past-20 s realised max (incl. NOW) snapped UP, urban floor 50 km/h."""
    lad = np.asarray(LITERALS["speed_ladder_kmh"], np.float64)
    a = max(0.0, now - LITERALS["speed_past_s"])
    sel = (log.ts >= a) & (log.ts <= now)
    vmax = float(max(log.v[sel].max() if sel.any() else 0.0, float(sample(log, now)[3])))
    kmh = max(vmax * 3.6, float(LITERALS["speed_urban_floor_kmh"]))
    n2 = lad[min(int(np.searchsorted(lad, kmh - 1e-9)), len(lad) - 1)]
    n3 = 0 if n2 <= 50 else (1 if n2 <= 100 else 2)
    return int(n2), int(n3)


# ------------------------------------------------------------------------------------------- per-clip labels #
def clip_labels(G: ClipGeo, now_k: np.ndarray, k: np.ndarray, rec: dict | None, lead: dict | None,
                lc: dict | None, h_cap: float | None = None, lc_labels: bool = False) -> dict:
    """Every per-frame field of one clip. ``now_k`` raw seconds of the frames ``k``.
    ``lead`` = {k: (has_data, lead?, gap, tg, s, v)}; ``lc`` = {"k": [...], "dl": [...], "dr": [...]} or None."""
    log, tr = G.log, G.tr
    n = len(k)
    dtk = float(np.median(np.diff(now_k))) if n > 1 else 0.1
    F: dict = {}

    def col(name, fill=np.nan, dtype=np.float64):
        if name not in F:
            F[name] = np.full(n, fill, dtype=dtype)
        return F[name]

    # ---- clip-level VLM context ------------------------------------------------------------------------------- #
    t0 = float(rec["t0_s"]) if rec else 8.0
    off = float(((rec or {}).get("alpamayo") or {}).get("anchor_offset_s", -2.9)) if rec else -2.9
    w0, w1 = t0 + off, t0 + off + 6.0
    now_star = w0 - LITERALS["band_s"][0]
    # LC from SAM3
    lc_cross, lc_u_by_k = (), {}
    if lc is not None and len(lc["k"]):
        lk = np.asarray(lc["k"], np.int64)
        kk2now = dict(zip(k.tolist(), now_k.tolist()))
        g0dt = (now_k[0] - k[0] * dtk)
        nowf = (lambda q: kk2now.get(q, g0dt + q * dtk))
        lc_cross, u, _ = lc_crossings(lk, np.asarray(lc["dl"]), np.asarray(lc["dr"]), nowf, log)
        lc_u_by_k = {int(a): float(b) for a, b in zip(lk, u)}
    # ---- per frame ---------------------------------------------------------------------------------------------- #
    turns = G.turns
    S2 = None
    for i in range(n):
        now = float(now_k[i])
        h = h_obs(log, now) if h_cap is None else min(h_obs(log, now), float(h_cap))   # h_cap: the V7 test only
        col("h_obs_s")[i] = h
        col("max_gap8_s")[i] = max_gap(log, now, min(now + 8.0, log.t_end))
        x0, y0, psi0, v_now = (float(a) for a in sample(log, now))
        col("v_now_ms")[i] = v_now
        n2, n3 = speed_proxy(log, now)
        col("speed_n2_kmh", -1, np.int16)[i] = n2
        col("speed_n3", -1, np.int8)[i] = n3
        # nav + rc (SPATIAL inputs)
        s_now = float(s_at(tr, now))
        for kk, vv in nav_fields(turns, s_now, now, v_now, G.s_max).items():
            col(kk)[i] = vv
        for kk, vv in rc_fields(tr, log, G.paths, turns, now, G.s_nat).items():
            if kk == "rc_b_kind":
                col("rc_b_kind", -1, np.int8)[i] = ("turn_end", "clamped_max", "clamped_min", "no_turn").index(vv)
            else:
                col(kk)[i] = vv
        # ---- action grid ------------------------------------------------------------------------------------- #
        lat_a = lat_b = IGNORE
        allow_a = allow_b = 0
        lon = IGNORE
        allow_lon = 0
        rev_s = reversing_s(log, now, now + min(h, 8.0))
        col("reversing", 0, np.int8)[i] = int(rev_s >= LITERALS["reverse_mask_s"])
        if h >= LITERALS["h_min_any_s"] and rev_s < LITERALS["reverse_mask_s"]:
            jh = int(math.floor(h / 0.1 + 1e-9))
            tg = now + np.arange(0, jh + 1) * 0.1
            gx, gy, gpsi, gv = sample(log, tg)
            # lateral
            dps = np.degrees(gpsi[20:] - gpsi[20])
            js = int(np.argmax(np.abs(dps)))
            theta = float(dps[js])
            col("lat_theta_deg")[i] = theta
            b0, b1 = now + 2.0, now + min(h, 8.0)
            dom, dom_mag = None, 0.0
            for u_ in turns:
                a_, c_ = max(u_.t_start, b0), min(u_.t_end, b1)
                if c_ <= a_:
                    continue
                yy = sample(log, np.array([a_, c_]))[2]
                cl = math.degrees(yy[1] - yy[0])
                if abs(theta) >= 1e-9 and np.sign(cl) == np.sign(theta) and abs(cl) > dom_mag:
                    dom, dom_mag = u_, abs(cl)
            col("lat_seg_found", 0, np.int8)[i] = int(dom is not None)
            big = abs(theta) >= LITERALS["turn_band_deg"]
            side_t = 1 if theta > 0 else 2
            is_junc = bool(big and dom is not None and dom.is_turn)
            col("curve", 0, np.int8)[i] = int(big and not is_junc)
            if dom is not None:
                col("turn_t_start_s")[i] = dom.t_start - now
                col("turn_t_end_s")[i] = dom.t_end - now
                col("turn_in_progress", 0, np.int8)[i] = int(dom.t_start - now < 2.0)
                col("turn_d_start_m")[i] = max(0.0, dom.s_start - s_now)
                col("turn_len_m")[i] = dom.s_end - dom.s_start
                col("turn_dyaw_deg")[i] = dom.dyaw_deg
                col("turn_r_arc_m")[i] = dom.r_arc_m
                col("turn_vmin_ms")[i] = dom.v_min_ms
                col("turn_is_turn", 0, np.int8)[i] = int(dom.is_turn)
                col("lat_gap_affected", 0, np.int8)[i] = int(dom.gap_affected)
                if dom.t_end <= log.t_end:
                    ex, ey, ep, _ = sample(log, dom.t_end)
                    lx, ly = to_now_frame(ex, ey, x0, y0, psi0)
                    col("turn_exit_x_m")[i], col("turn_exit_y_m")[i] = float(lx), float(ly)
                    col("turn_exit_psi_deg")[i] = math.degrees(float(ep) - psi0)
            # lane change (SAM3)
            cross = [(tc, sd) for tc, sd in lc_cross if b0 <= tc <= b1]
            band_k = [int(k[i]) + j for j in range(int(math.ceil(2.0 / dtk)), int(math.floor(min(h, 8.0) / dtk)) + 1)]
            if lc is None:
                meas, frac = -1, np.nan
            else:
                okc = sum(1 for q in band_k if np.isfinite(lc_u_by_k.get(q, np.nan)))
                frac = okc / max(len(band_k), 1)
                meas = int(frac >= LITERALS["lc_measurable_frac"])
            # ⛔ V5 FAILED (train: side-correct on 14 of 101 Alpamayo lane-change-text clips, bar 0.80; precision 14/15,
            # no-text firing 1.1 %): the pre-registered consequence is that LC ships NEVER-POSITIVE — the measurement
            # stays as a DIAGNOSTIC (lc_measurable_raw, lc_t_cross_s, lc_side) and the labels take the partial path.
            col("lc_measurable_raw", -1, np.int8)[i] = meas
            if not lc_labels and meas == 1:
                meas = 0
            col("lc_measurable", -1, np.int8)[i] = meas
            col("lc_frac")[i] = frac
            if cross:
                tc, sd = cross[0]
                col("lc_t_cross_s")[i] = tc - now
                col("lc_d_cross_m")[i] = float(s_at(tr, tc)) - s_now
                col("lc_side", 0, np.int8)[i] = sd
            exc, net = d2_excursion_signed(gx[20:], gy[20:], gpsi[20:])
            col("exc_lat_m")[i] = exc
            lc_cls = (3 if cross[0][1] > 0 else 4) if (cross and meas == 1) else None
            absence_ok = h >= LITERALS["h_abs_min_s"]
            for variant, is_t in (("a", is_junc), ("b", big)):
                if is_t:
                    cls_, allow = side_t, _bits([side_t])
                elif lc_cls is not None:
                    cls_, allow = lc_cls, _bits([lc_cls])
                elif not absence_ok:
                    cls_, allow = IGNORE, 0
                elif meas == 1:
                    cls_, allow = 0, _bits([0])
                else:                                   # LC not measurable: partial label (§3.6)
                    opts = [0]
                    if abs(net) < 15.0 and exc >= LITERALS["lc_partial_exc_m"]:
                        opts.append(3)
                    if abs(net) < 15.0 and exc <= -LITERALS["lc_partial_exc_m"]:
                        opts.append(4)
                    cls_, allow = (0 if len(opts) == 1 else IGNORE), _bits(opts)
                if variant == "a":
                    lat_a, allow_a = cls_, allow
                else:
                    lat_b, allow_b = cls_, allow
            # longitudinal
            vb = _vbar(gv)
            va = float(gv[20])
            band_v = gv[20:]
            col("v_a_ms")[i] = va
            col("v_lo_ms")[i], col("v_hi_ms")[i] = float(band_v.min()), float(band_v.max())
            if jh >= 80:
                col("v_end_ms")[i] = float(gv[80])
            jj = np.arange(20, jh + 1)
            acc = jj[gv[20:] >= va + LITERALS["dv_ms"]]
            dec = jj[gv[20:] <= va - LITERALS["dv_ms"]]
            stp = jj[gv[20:] <= LITERALS["v_stop_ms"]] if va > LITERALS["v_stop_ms"] else np.array([], int)
            t_acc = int(acc[0]) if acc.size else None
            t_dec = int(dec[0]) if dec.size else None
            t_stp = int(stp[0]) if stp.size else None
            vmax_b = float(band_v.max())
            # FOLLOW evidence over the band frames
            cov, fol_n, gaps, tgs = 0, 0, [], []
            if lead is not None:
                for q in band_k:
                    r = lead.get(q)
                    if r is None or not r[0]:
                        continue
                    cov += 1
                    if r[1]:
                        gaps.append(r[2])
                        tgs.append(r[3])
                        if r[3] <= LITERALS["lead_tg_max_s"] and r[5] >= LITERALS["lead_v_min_ms"]:
                            fol_n += 1
            cov_s = cov * dtk
            col("lead_cov_s")[i] = cov_s
            col("lead_follow_frac")[i] = fol_n / cov if cov else np.nan
            if gaps:
                col("lead_valid", 0, np.int8)[i] = 1
                col("lead_gap_m")[i], col("lead_tg_s")[i] = gaps[0], tgs[0]
                col("lead_gap_min_m")[i], col("lead_tg_min_s")[i] = float(min(gaps)), float(min(tgs))
            following = cov_s >= LITERALS["lead_cov_min_s"] and cov and fol_n / cov >= LITERALS["lead_follow_frac"]
            fol_known = cov_s >= LITERALS["lead_cov_min_s"]
            cls_l, vt, tr_, dr_, ts_, trunc = None, np.nan, np.nan, np.nan, np.nan, 0
            if absence_ok and vmax_b <= LITERALS["v_stop_ms"]:
                cls_l, vt = 0, 0.0
            elif absence_ok and vmax_b <= LITERALS["v_creep_ms"]:
                cls_l, vt = 1, vmax_b
            elif t_stp is not None and (t_acc is None or t_stp < t_acc):
                cls_l, vt, tr_ = 2, 0.0, t_stp * 0.1
                ts_ = _extremum_before(vb, t_stp, -1) * 0.1
                sx, sy, _, _ = sample(log, now + tr_)
                lx, ly = to_now_frame(sx, sy, x0, y0, psi0)
                col("stop_x_m")[i], col("stop_y_m")[i] = float(lx), float(ly)
                dr_ = float(s_at(tr, now + tr_)) - s_now
                after = np.nonzero((log.ts > now + tr_) & (log.v > LITERALS["v_stop_ms"]))[0]
                if after.size and max_gap(log, now + tr_, float(log.ts[after[0]])) <= LITERALS["g_max_s"]:
                    col("stop_dur_s")[i] = float(log.ts[after[0]]) - (now + tr_)
            elif following:
                cls_l, vt = 3, float(band_v.mean())
            elif t_dec is not None or t_acc is not None:
                first_dec = t_dec is not None and (t_acc is None or t_dec < t_acc)
                jc = t_dec if first_dec else t_acc
                sg = -1 if first_dec else 1
                je, trunc = _extremum_after(vb, jc, jh, sg)
                cls_l, vt, tr_ = (4 if first_dec else 5), float(vb[je]), je * 0.1
                dr_ = float(s_at(tr, now + tr_)) - s_now
                ts_ = _extremum_before(vb, jc, sg) * 0.1
            elif absence_ok:
                cls_l, vt = 6, float(band_v.mean())
            if cls_l is not None:
                partial = (not fol_known) and cls_l in (4, 5, 6)
                if partial:
                    lon, allow_lon = IGNORE, _bits([cls_l, 3])
                else:
                    lon, allow_lon = cls_l, _bits([cls_l])
                col("lon_v_target_ms")[i], col("lon_t_reach_s")[i] = vt, tr_
                col("lon_d_reach_m")[i], col("lon_t_start_s")[i] = dr_, ts_
                col("lon_truncated", 0, np.int8)[i] = int(trunc)
                col("lon_cls_computed", IGNORE, np.int16)[i] = cls_l
        col("lat_cls_a", IGNORE, np.int16)[i], col("lat_cls_b", IGNORE, np.int16)[i] = lat_a, lat_b
        col("lat_allowed_a", 0, np.uint8)[i], col("lat_allowed_b", 0, np.uint8)[i] = allow_a, allow_b
        col("lon_cls", IGNORE, np.int16)[i], col("lon_allowed", 0, np.uint8)[i] = lon, allow_lon
    # ---- v7-id encodings ------------------------------------------------------------------------------------- #
    for v in ("a", "b"):
        c9, a9 = F["lat_cls_" + v], F["lat_allowed_" + v]
        F[f"lat_v7id_{v}"] = np.array([LAT9_TO_7[int(c)] if c >= 0 else IGNORE for c in c9], np.int16)
        F[f"lat_allowed_v7_{v}"] = np.array([_bits([LAT9_TO_7[b] for b in range(5) if (int(a) >> b) & 1]) for a in a9], np.uint8)
    F["lon_v7id"] = np.array([LON9_TO_7[int(c)] if c >= 0 else IGNORE for c in F["lon_cls"]], np.int16)
    F["lon_allowed_v7"] = np.array([_bits([LON9_TO_7[b] for b in range(7) if (int(a) >> b) & 1]) for a in F["lon_allowed"]], np.uint8)
    # ---- goals ----------------------------------------------------------------------------------------------- #
    F.update(goal_bits(G, now_k, k, rec, F, now_star, w0, w1))
    return F


def goal_bits(G: ClipGeo, now_k, k, rec, F, now_star, w0, w1) -> dict:
    """22-token y / w bit-masks per frame (§4). Variant a is the default turn set (D-WPA-1)."""
    log = G.log
    n = len(now_k)
    # ⛔ a clip whose frames are ALL masked (reversing, h_obs < 3) never creates some columns: read with defaults
    # (caught by test_reversing_window_is_masked before any release was built on it).
    F = dict(F)
    for nm, fill, dt_ in (("v_a_ms", np.nan, np.float64), ("turn_t_start_s", np.nan, np.float64),
                          ("h_obs_s", np.nan, np.float64), ("lc_measurable", -1, np.int8),
                          ("lon_cls_computed", IGNORE, np.int16)):
        if nm not in F:
            F[nm] = np.full(n, fill, dtype=dt_)
    y = np.zeros(n, np.uint32)
    w = np.zeros(n, np.uint32)
    lc_prov = np.zeros(n, np.int8)            # 0 none, 1 sam3, 2 vlm
    vlm_frame = (np.abs(now_k - now_star) <= LITERALS["vlm_half_window_s"]).astype(np.int8)
    red_prop = np.zeros(n, np.int8)
    # clip-level VLM tokens (geometry side at NOW* = the frame nearest NOW*)
    i_star = int(np.argmin(np.abs(now_k - now_star)))
    c_star = int(F["lat_cls_a"][i_star])
    geom_side = 1 if c_star == 1 else (-1 if c_star == 2 else 0)
    toks, has_cot = (vlm_tokens_for_record(rec, geom_side) if rec else (frozenset(), False))
    probe_neg = tl_probe_negative(rec) if rec else False
    # VLM lateral evidence (D2-style detrended excursion >= 0.3 m within W_vlm)
    lat_ev = 0
    if w1 <= log.t_end and w0 >= 0:
        ex, ey, ep, _ = sample(log, np.arange(w0, w1 + 1e-9, 0.1))
        lat_ev = int(abs(d2_excursion_signed(ex, ey, ep)[0]) >= LITERALS["vlm_lat_evidence_m"])
    red_ep = red_stop_episode(log, w0, w1) if "TRAFFIC_LIGHT_REACT_RED" in toks else None
    for i in range(n):
        yy, ww = 0, 0
        la = int(F["lat_cls_a"][i])
        al = int(F["lat_allowed_a"][i])
        lac = int(F["lon_cls_computed"][i])
        lat_known = la >= 0
        # TURN_L / TURN_R (geometry): known whenever the allowed set decides turn-or-not (incl. LC partial labels)
        if al:
            for tok, c in (("TURN_L", 1), ("TURN_R", 2)):
                ww |= 1 << _gi(tok)
                if la == c:
                    yy |= 1 << _gi(tok)
        # FOLLOW_LANE: only when the lateral class is single-valued
        if lat_known:
            ww |= 1 << _gi("FOLLOW_LANE")
            if la == 0:
                yy |= 1 << _gi("FOLLOW_LANE")
        # STOP_POINT
        va = F["v_a_ms"][i]
        if np.isfinite(va) and va > LITERALS["v_stop_ms"]:
            if lac == 2:
                ww |= 1 << _gi("STOP_POINT")
                yy |= 1 << _gi("STOP_POINT")
            elif F["h_obs_s"][i] >= LITERALS["h_abs_min_s"] and lac >= 0:
                ww |= 1 << _gi("STOP_POINT")
        # YIELD_FOR_TURN_x (a stop of >= 0.5 s between NOW and the dominant turn's start)
        if lat_known:
            for tok, c in (("YIELD_FOR_TURN_L", 1), ("YIELD_FOR_TURN_R", 2)):
                ww |= 1 << _gi(tok)
                if la == c and np.isfinite(F["turn_t_start_s"][i]):
                    t_s = float(F["turn_t_start_s"][i])
                    if t_s > LITERALS["yield_stop_s"]:
                        tg = now_k[i] + np.arange(0, t_s + 1e-9, 0.1)
                        st = sample(log, tg)[3] <= LITERALS["v_stop_ms"]
                        run, best = 0, 0
                        for b in st:
                            run = run + 1 if b else 0
                            best = max(best, run)
                        if best * 0.1 >= LITERALS["yield_stop_s"]:
                            yy |= 1 << _gi(tok)
        # LANE_CHANGE_x: SAM3 where measurable, else VLM on the VLM frames
        if int(F["lc_measurable"][i]) == 1 and lat_known:
            lc_prov[i] = 1
            for tok, c in (("LANE_CHANGE_L", 3), ("LANE_CHANGE_R", 4)):
                ww |= 1 << _gi(tok)
                if la == c:
                    yy |= 1 << _gi(tok)
        elif vlm_frame[i] and has_cot:
            lc_prov[i] = 2
            for tok in ("LANE_CHANGE_L", "LANE_CHANGE_R"):
                ww |= 1 << _gi(tok)
                if tok in toks:
                    yy |= 1 << _gi(tok)
        # VLM tokens on the VLM frames (positives + the PI's cot-absence negatives + TL probe negatives)
        if vlm_frame[i] and has_cot:
            for tok in VLM_TOKENS:
                ww |= 1 << _gi(tok)
                if tok in toks:
                    yy |= 1 << _gi(tok)
        elif vlm_frame[i] and probe_neg:
            for tok in TL_TOKENS:
                ww |= 1 << _gi(tok)
        # RED propagation over the stop episode
        if red_ep is not None:
            t_app, t_a, t_b = red_ep
            if max(t_app, t_a - LITERALS["red_back_s"]) <= now_k[i] <= t_b - LITERALS["red_tail_s"]:
                ww |= 1 << _gi("TRAFFIC_LIGHT_REACT_RED")
                yy |= 1 << _gi("TRAFFIC_LIGHT_REACT_RED")
                red_prop[i] = 1
        y[i], w[i] = yy, ww
    return {"goal_y": y, "goal_w": w, "goal_lc_prov": lc_prov, "vlm_frame": vlm_frame, "red_propagated": red_prop,
            "vlm_lat_evidence": np.full(n, lat_ev, np.int8), "vlm_has_cot": np.full(n, int(has_cot), np.int8),
            "vlm_tl_probe_negative": np.full(n, int(probe_neg), np.int8)}


# =============================================================================================================== #
# CLI: lead-pass (dev box) · lc-pass (Thor for train: the SAM3 train GT lives only there) · build                 #
# =============================================================================================================== #
EGO_DEFAULT = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/{}.parquet"


def _read_sidecar(path: str) -> dict:
    import json as _json
    tab = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = _json.loads(line)
                tab[int(r["sid"])] = (float(r["grid_start_s"]), float(r["dt_s"]))
    return tab


def _manifest(path: str):
    import torch
    m = torch.load(path, map_location="cpu", weights_only=False)
    return [str(c) for c in m["clip_id"]], [p.double().numpy() for p in m["poses"]]


def cmd_lead_pass(a) -> None:
    import time as _t
    cids, poses = _manifest(a.manifest)
    side = _read_sidecar(a.sidecar)
    by_cid = {c: i for i, c in enumerate(cids)}
    out = {"sha12": [], "k": [], "has": [], "lead": [], "gap": [], "tg": [], "s": [], "v": []}
    t0, n = _t.time(), 0
    for cid, frames in iter_join_clips(a.join, set(cids)):
        i = by_cid[cid]
        sid = stable_episode_id(cid)
        g0, dt, _ = clip_clock(sid, side, poses[i])
        log = load_log(a.ego.format(cid))
        tr = track10(log)
        p0 = smooth_path(tr, 0.0)
        for kk, agents in sorted(frames.items()):
            t = g0 + kk * dt
            if t < 0 or t > log.t_end:
                continue
            L, gap, tg, s, v = lead_for_frame(log, tr, p0, t, agents)
            for key, val in (("sha12", sha12(cid)), ("k", kk), ("has", 1), ("lead", int(L)), ("gap", gap),
                             ("tg", tg), ("s", s), ("v", v)):
                out[key].append(val)
        n += 1
        if n % 200 == 0:
            print(f"[lead-pass] {n} clips {_t.time() - t0:.0f}s", flush=True)
    np.savez_compressed(a.out, **{k: np.asarray(v) for k, v in out.items()},
                        meta=np.asarray(json_dumps({"join_md5": file_md5(a.join), "n_clips": n,
                                                    "builder_md5": file_md5(__file__)})))
    print(f"[lead-pass] done {n} clips, {len(out['k'])} frames, {_t.time() - t0:.0f}s -> {a.out}")


def cmd_lc_pass(a) -> None:
    """Per map frame (d_L, d_R, n_rows) from SAM3 fine codes. Runs where the GT lives; numpy only; no clip ids."""
    import glob
    import time as _t
    files = sorted(glob.glob(os.path.join(a.gt_root, "*.sam3mapgt.npz")))
    if a.sha12_list:
        want = {l.strip() for l in open(a.sha12_list, encoding="utf-8") if l.strip()}
        files = [f for f in files if os.path.basename(f).split(".")[0] in want]
    out = {"sha12": [], "k": [], "dl": [], "dr": [], "nrows": []}
    t0 = _t.time()
    md5s = {}
    for n, f in enumerate(files):
        s12 = os.path.basename(f).split(".")[0]
        with np.load(f, allow_pickle=False) as z:
            codes = z["fine_codes"]
            T = codes.shape[0]
            off = lc_offsets_from_codes(codes)
        md5s[s12] = file_md5(f) if a.md5 else None
        out["sha12"] += [s12] * T
        out["k"] += list(range(T))
        out["dl"] += off[:, 0].tolist()
        out["dr"] += off[:, 1].tolist()
        out["nrows"] += off[:, 2].astype(int).tolist()
        if n % 100 == 0:
            print(f"[lc-pass] {n + 1}/{len(files)} {_t.time() - t0:.0f}s", flush=True)
    np.savez_compressed(a.out, **{k: np.asarray(v) for k, v in out.items()},
                        meta=np.asarray(json_dumps({"n_files": len(files), "gt_root": a.gt_root, "md5": md5s,
                                                    "builder_md5": file_md5(__file__),
                                                    "rows": list(LITERALS["lc_rows"])})))
    print(f"[lc-pass] done {len(files)} files {_t.time() - t0:.0f}s -> {a.out}")


def json_dumps(o) -> str:
    import json as _json
    return _json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


def cmd_build(a) -> None:
    import gzip
    import json as _json
    import time as _t
    assert_no_g_imports()
    cids, poses = _manifest(a.manifest)
    side = _read_sidecar(a.sidecar)
    recs = {}
    with gzip.open(a.labels, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = _json.loads(line)
                recs[r["clip_id"]] = r
    lead_by = {}
    if a.lead:
        z = np.load(a.lead, allow_pickle=True)
        for s12, kk, h, L, g, tg, s, v in zip(z["sha12"], z["k"], z["has"], z["lead"], z["gap"], z["tg"], z["s"], z["v"]):
            lead_by.setdefault(str(s12), {})[int(kk)] = (bool(h), bool(L), float(g), float(tg), float(s), float(v))
    lc_by = {}
    if a.lc:
        z = np.load(a.lc, allow_pickle=True)
        s12a = z["sha12"].astype(str)
        for s12 in np.unique(s12a):
            m = s12a == s12
            lc_by[str(s12)] = {"k": z["k"][m], "dl": z["dl"][m], "dr": z["dr"][m]}
    order = list(range(len(cids))) if not a.limit else list(range(min(a.limit, len(cids))))
    rows, clip_tab = [], {"sid": [], "sha12": [], "row0": [], "n_rows": [], "k0": [], "grid_start_s": [],
                          "dt_s": [], "clock_src": [], "has_record": [], "has_lead": [], "has_lc": [],
                          "n_turns": [], "n_announced": [], "suppressed": []}
    t0, r0 = _t.time(), 0
    for ci in order:
        cid = cids[ci]
        sid, s12 = stable_episode_id(cid), sha12(cid)
        g0, dt, src = clip_clock(sid, side, poses[ci])
        T = int(poses[ci].shape[0])
        k = np.arange(T) + 2
        now_k = g0 + k * dt
        rec = recs.get(cid)
        G = clip_geo(load_log(a.ego.format(cid)), (rec or {}).get("turn_suppression"))
        F = clip_labels(G, now_k, k, rec, lead_by.get(s12), lc_by.get(s12), lc_labels=(a.lc_labels == "on"))
        F["k"] = k.astype(np.int16)
        F["now_s"] = now_k
        rows.append(F)
        for key, val in (("sid", sid), ("sha12", s12), ("row0", r0), ("n_rows", T), ("k0", 2), ("grid_start_s", g0),
                         ("dt_s", dt), ("clock_src", ("sidecar", "pose_dt", "nominal_dt").index(src)),
                         ("has_record", int(rec is not None)), ("has_lead", int(s12 in lead_by)),
                         ("has_lc", int(s12 in lc_by)), ("n_turns", len(G.turns)),
                         ("n_announced", sum(u.announced for u in G.turns)),
                         ("suppressed", int(any(u.suppressed for u in G.turns)))):
            clip_tab[key].append(val)
        r0 += T
        if len(rows) % 100 == 0:
            print(f"[build {a.split}] {len(rows)}/{len(order)} clips {_t.time() - t0:.0f}s", flush=True)
    names = sorted(set().union(*[set(F) for F in rows]))
    out = {}
    for nm in names:
        parts = []
        for F, n in zip(rows, clip_tab["n_rows"]):
            if nm in F:
                parts.append(np.asarray(F[nm]))
            else:
                ref = next(G_[nm] for G_ in rows if nm in G_)
                fill = np.nan if np.issubdtype(np.asarray(ref).dtype, np.floating) else IGNORE if np.asarray(ref).dtype == np.int16 else 0
                parts.append(np.full(n, fill, dtype=np.asarray(ref).dtype))
        out["row__" + nm] = np.concatenate(parts)
    for key, val in clip_tab.items():
        out["clip__" + key] = np.asarray(val)
    os.makedirs(a.out_dir, exist_ok=True)
    npz = os.path.join(a.out_dir, f"v9_labels_{a.split}.npz")
    np.savez_compressed(npz, **out)
    S2 = _builder()
    man = {"schema": "tanitad.v9_labels/1", "split": a.split, "n_clips": len(order), "n_rows": int(r0),
           "row_key": "(sid, k): k = provider row + 2 = raw row; a dataset window with start t has NOW at k = t + 9",
           "clock_rule": "now_s = grid_start_s + k * dt_s (sidecar; else clip_clock.pose_dt with grid_start 0.0; else nominal)",
           "vocab": {"LAT9": LAT9, "LON9": LON9, "LAT7": LAT7, "LON7": LON7, "GOAL22": GOAL22,
                     "GOAL_PROV": GOAL_PROV, "LAT9_TO_7": LAT9_TO_7, "LON9_TO_7": LON9_TO_7},
           "literals": {k_: (list(v_) if isinstance(v_, tuple) else v_) for k_, v_ in LITERALS.items()},
           "lc_labels": a.lc_labels,
           "policies": {"goal_negatives_vlm": COT_ABSENCE_POLICY_ID, "nav": "allow_oracle_nav (ego-future)",
                        "rc": "ego-future path, spatial only; optimistic on PhysicalAI; privileged on NavSim"},
           "sources": {"manifest": [a.manifest, file_md5(a.manifest)], "sidecar": [a.sidecar, file_md5(a.sidecar)],
                       "labels": [os.path.basename(a.labels), file_md5(a.labels)],
                       "lead": [os.path.basename(a.lead), file_md5(a.lead)] if a.lead else None,
                       "lc": [os.path.basename(a.lc), file_md5(a.lc)] if a.lc else None,
                       "ego_store": a.ego.split("{")[0]},
           "builder": {"build_v9_labels.py": file_md5(__file__), "s2_geom_emit_v7.py": file_md5(S2.__file__),
                       "base_commit": a.base_commit},
           "fields": sorted(out), "npz_md5": file_md5(npz), "wall_s": round(_t.time() - t0, 1)}
    with open(os.path.join(a.out_dir, f"v9_labels_{a.split}.manifest.json"), "w", encoding="utf-8") as f:
        f.write(json_dumps(man))
    print(f"[build {a.split}] done {len(order)} clips {r0} rows {_t.time() - t0:.0f}s -> {npz}")


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lead-pass")
    p.add_argument("--manifest", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--join", required=True)
    p.add_argument("--ego", default=EGO_DEFAULT)
    p.add_argument("--out", required=True)
    p = sub.add_parser("lc-pass")
    p.add_argument("--gt-root", required=True)
    p.add_argument("--sha12-list", default=None)
    p.add_argument("--md5", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("build")
    p.add_argument("--split", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--lead", default=None)
    p.add_argument("--lc", default=None)
    p.add_argument("--ego", default=EGO_DEFAULT)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--base-commit", default="50efa52")
    p.add_argument("--lc-labels", choices=("on", "off"), default="off",
                   help="SAM3 lane-change CLASSES (default off: V5 failed, LC ships never-positive; diagnostics kept)")
    a = ap.parse_args(argv)
    {"lead-pass": cmd_lead_pass, "lc-pass": cmd_lc_pass, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()

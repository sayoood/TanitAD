"""X10 -- POSE-TO-IMAGE TIMING (SPEC_REFCV8 section 8.10; PI data audit D3).  OPT-IN, DEFAULT OFF.

WHAT THE DEFECT IS (MEASURED, see the X10 package RESULT.md)
------------------------------------------------------------
The episode cache is built by ``scripts/v2_compressed.py::_resampled`` (and its twin
``physicalai.build_episode``):

    t_query   = linspace(t_frames[0], t_frames[-1], int(span * 10))          # :119-120
    frame_idx = searchsorted(t_frames, t_query)                              # :121  (side='left')
    vid       = decode(mp4)[frame_idx]                                       # :122
    poses     = signals_at(ego, t_query)                                     # :124

Row ``k``'s POSE is the egomotion sampled at ``t_query[k]``.  Row ``k``'s IMAGE is the first camera
frame AT OR AFTER ``t_query[k]``.  The camera runs at ~30 Hz and the pose grid at ~9.934 Hz, so the
two clocks beat against each other and the image trails its pose by ``delta_k`` in ``[0, 33.4) ms``
(a sawtooth, period ~50 rows).  The cache stores ONLY the poses -- never ``t_frames`` -- so the
offset is not recoverable from the cache; it needs the camera ``*.timestamps.parquet``.

WHAT THE CORRECTION IS
----------------------
For a window whose NOW row is ``r``, every ego-state quantity is re-sampled ``delta_r`` LATER in
time, i.e. at the instant the NOW image was actually captured:

    pose'(row j) = pose(t_query[j] + delta_r)          (ONE shift per window, the NOW row's)

Rows keep their indices, so every ``(sid, k)`` join (v9 labels, agent boxes, SAM3 map) and the
label clock ``t_now = grid_start_s + (k + n_stack - 1) * dt_s`` are UNTOUCHED.  The re-sampling is
a linear interpolation of the cached 10 Hz track (no raw egomotion log is needed at train time);
its error against the 100 Hz log is MEASURED in the package RESULT.md.

WHY A WINDOW-LEVEL SHIFT AND NOT A PER-ROW ONE (the literal reading of "interpolate the pose track to
each camera frame's timestamp")
-------------------------------------------------------------------------------------------------
A trajectory target is a DISPLACEMENT over a fixed elapsed time, ``p(t0 + h*dt) - p(t0)``.  Shift
every row by its own ``delta_j`` and the elapsed time of waypoint ``h`` becomes
``h*dt + delta_{r+h} - delta_r`` -- a +-33 ms timing JITTER, ~0.5 m at 15 m/s.  That is a FIRST-order
error and it is larger than the second-order bias being removed.  One common shift keeps the elapsed
times exactly ``h*dt`` and re-anchors the whole window on the true image instant.  The per-row
variant (:func:`shift_rows_per_row`) exists ONLY so the measurement can price that jitter.

SIGN CONVENTION: a positive shift moves the sample LATER in time (toward the image).  The
deliberate-regression arm flips it and must go RED.

Nothing here adds an input at inference: ``delta`` rewrites TRAINING targets and the training-time
ego state, from timestamps that exist only in the recorded corpus.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

SCHEMA = "tanitad.pose_sync/1"
TARGET_HZ = 10.0            # == physicalai.TARGET_HZ, the cache's row rate
#: the camera is ~30 Hz, so a "next frame at or after" offset lives in [0, 1/30) s.  Anything past this
#: is a unit error (us read as ms) or a broken track, never an offset of this corpus.
DELTA_US_MAX = 50_000
#: the physically possible row step of the cache grid (same band as ``clip_clock.POSE_DT_BAND_S``)
DT_BAND_S = (0.098, 0.104)


class PoseSyncError(ValueError):
    """A sidecar that cannot be trusted is refused, never guessed around."""


# --------------------------------------------------------------------------------------------------
# 1. The grid: an EXACT replica of v2_compressed._resampled :111-121
# --------------------------------------------------------------------------------------------------
def resample_grid(t_frames, *, target_hz: float = TARGET_HZ, min_rows: int = 4):
    """Camera timestamps -> ``(t_query, frame_idx, unit)`` exactly as the cache builder derives them.

    ``unit`` is the clock's ticks-per-second (1e6 for the PhysicalAI microsecond clock).
    """
    t_frames = np.asarray(t_frames, dtype=np.float64)
    span = t_frames[-1] - t_frames[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):                       # ns / us / ms clocks -> seconds
        if span / cand > 1.0:
            unit = cand
            break
    n_target = max(int(span / unit * target_hz), min_rows)
    t_query = np.linspace(t_frames[0], t_frames[-1], n_target)
    frame_idx = np.searchsorted(t_frames, t_query).clip(0, len(t_frames) - 1)
    return t_query, frame_idx, unit


def grid_delta(t_frames, *, target_hz: float = TARGET_HZ, min_rows: int = 4):
    """Camera timestamps -> ``(delta_s [n_target], dt_s, n_target)``.

    ``delta_s[k] = t_image[k] - t_pose[k]`` in SECONDS: how much LATER the image of raw grid row ``k``
    was captured than the pose stored beside it.  ``dt_s`` is the grid's row step in seconds.
    """
    t_query, frame_idx, unit = resample_grid(t_frames, target_hz=target_hz, min_rows=min_rows)
    t_frames = np.asarray(t_frames, dtype=np.float64)
    delta_s = (t_frames[frame_idx] - t_query) / unit
    dt_s = float((t_query[-1] - t_query[0]) / (len(t_query) - 1) / unit)
    return delta_s, dt_s, int(len(t_query))


def analytic_delta_s(t0_us: float, tN_us: float, n_frames: int, n_target: int | None = None,
                     *, target_hz: float = TARGET_HZ) -> np.ndarray:
    """An INDEPENDENT derivation of ``delta`` from four numbers -- no per-frame timestamp is read.

    Assumes an IDEAL constant-period camera (``T = (tN - t0) / (n_frames - 1)``) and the builder's
    linspace grid.  ``delta_k = (ceil(k * dt_q / T) * T - k * dt_q)``.  Used only as a cross-check of
    :func:`grid_delta`: agreement says the sawtooth is a property of the two clocks, not of the data.
    """
    span = tN_us - t0_us
    if n_target is None:
        n_target = max(int(span / 1e6 * target_hz), 4)
    period = span / (n_frames - 1)
    dtq = span / (n_target - 1)
    k = np.arange(n_target, dtype=np.float64)
    m = np.ceil(k * dtq / period - 1e-9)
    return (m * period - k * dtq) / 1e6


# --------------------------------------------------------------------------------------------------
# 2. Re-sampling a cached track at a fractional row position
# --------------------------------------------------------------------------------------------------
def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def sample_rows(x, u, angle_cols=()):
    """Linearly sample ``x [T, C]`` at FRACTIONAL row positions ``u [M]``.

    * rows outside ``[0, T-1]`` are LINEARLY EXTRAPOLATED from the nearest segment (a <= 0.34-row tail,
      ~1 mm of position error at 2 m/s^2 -- priced in RESULT.md); the validity masks are therefore unchanged;
    * ``angle_cols`` interpolate on the SHORTEST ARC and re-wrap to (-pi, pi], mirroring
      ``physicalai.signals_at`` (which wraps its yaw), so no +-pi seam is smeared;
    * an integer ``u`` returns the stored row BIT-EXACTLY (the zero-offset control relies on this).
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    T = x.shape[0]
    if T < 2:
        raise PoseSyncError("need at least 2 rows to interpolate")
    i0 = np.floor(u).astype(np.int64).clip(0, T - 2)
    f = u - i0
    x0, x1 = x[i0], x[i0 + 1]
    out = x0 + f[:, None] * (x1 - x0)
    for c in angle_cols:
        out[:, c] = _wrap(x0[:, c] + f * _wrap(x1[:, c] - x0[:, c]))
    z, o = (f == 0.0), (f == 1.0)
    out[z] = x0[z]
    out[o] = x1[o]
    return out


def shift_rows(x, shift_rows_frac: float, angle_cols=()):
    """Every row of ``x [T, C]`` re-sampled ``shift_rows_frac`` ROWS later (negative = earlier)."""
    x = np.asarray(x)
    u = np.arange(x.shape[0], dtype=np.float64) + float(shift_rows_frac)
    return sample_rows(x, u, angle_cols)


def shift_rows_per_row(x, shift_rows_frac_vec, angle_cols=()):
    """Row ``j`` re-sampled ``shift_rows_frac_vec[j]`` rows later.  MEASUREMENT ONLY -- see module doc."""
    x = np.asarray(x)
    u = np.arange(x.shape[0], dtype=np.float64) + np.asarray(shift_rows_frac_vec, dtype=np.float64)
    return sample_rows(x, u, angle_cols)


# --------------------------------------------------------------------------------------------------
# 3. The sidecar
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SyncRow:
    delta_s: np.ndarray      # [n_raw] seconds, image time minus pose time, RAW grid rows
    dt_s: float              # the clip's row step, seconds


@dataclass
class PoseSync:
    """The loaded sidecar plus the sign switch.  ``sign=+1`` is the correction; ``-1`` exists ONLY for the
    deliberate-regression arm of the tests (it doubles the error instead of removing it)."""
    table: dict
    meta: dict = field(default_factory=dict)
    path: str = ""
    n_rows: int = 0
    sign: float = 1.0

    def lookup(self, sid: int):
        return self.table.get(int(sid))

    def shift_for(self, sid: int, raw_row: int):
        """Fractional ROW shift for a window whose NOW is raw grid row ``raw_row``; ``None`` if uncovered."""
        r = self.table.get(int(sid))
        if r is None or not (0 <= int(raw_row) < r.delta_s.shape[0]):
            return None
        return self.sign * float(r.delta_s[int(raw_row)]) / r.dt_s


def write_pose_sync_sidecar(path: str, rows: list, meta: dict) -> dict:
    """``rows``: dicts with ``sid`` (int), ``dt_s`` (float), ``delta_us`` (list[int]).  Writes JSONL plus
    ``<path>.meta.json``.  No clip id is written (the clip-id rule); rows are keyed on ``sid`` alone."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps({"sid": int(r["sid"]), "dt_s": float(r["dt_s"]),
                                 "n_rows": len(r["delta_us"]),
                                 "delta_us": [int(v) for v in r["delta_us"]]},
                                separators=(",", ":")) + "\n")
    m = dict(meta)
    m["schema"] = SCHEMA
    m["n_rows"] = len(rows)
    with open(path + ".meta.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)
    return m


def read_pose_sync_sidecar(path: str, *, sign: float = 1.0) -> PoseSync:
    """JSON-lines ``{"sid", "dt_s", "n_rows", "delta_us"}`` -> :class:`PoseSync`.

    REFUSES: a missing file; a row without ``sid`` / ``dt_s`` / ``delta_us``; ``len(delta_us) != n_rows``;
    a ``dt_s`` outside :data:`DT_BAND_S`; any ``delta_us`` outside ``[0, DELTA_US_MAX]`` (the builder's
    ``searchsorted(side='left')`` makes it non-negative by construction, so a negative value is a SIGN or
    UNIT error); a duplicated ``sid`` with different content; zero rows.
    """
    if not os.path.isfile(path):
        raise PoseSyncError(f"[pose-sync] REFUSING: {path!r} does not exist")
    meta: dict = {}
    mp = path + ".meta.json"
    if os.path.isfile(mp):
        with open(mp, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("schema") not in (None, SCHEMA):
            raise PoseSyncError(f"[pose-sync] REFUSING {mp}: schema {meta.get('schema')!r} is not {SCHEMA!r}")
    table: dict = {}
    n = 0
    lo, hi = DT_BAND_S
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            try:
                sid = int(r["sid"])
                dt = float(r["dt_s"])
                d = np.asarray(r["delta_us"], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as e:
                raise PoseSyncError(f"[pose-sync] REFUSING {path}:{i + 1}: needs sid / dt_s / delta_us "
                                    f"({type(e).__name__}: {e})") from e
            if d.ndim != 1 or d.shape[0] < 4 or ("n_rows" in r and int(r["n_rows"]) != d.shape[0]):
                raise PoseSyncError(f"[pose-sync] REFUSING {path}:{i + 1}: delta_us length {d.shape} does "
                                    f"not match n_rows {r.get('n_rows')}")
            if not (lo <= dt <= hi):
                raise PoseSyncError(f"[pose-sync] REFUSING {path}:{i + 1}: dt_s {dt} outside {DT_BAND_S}")
            if not (np.all(np.isfinite(d)) and d.min() >= 0.0 and d.max() <= DELTA_US_MAX):
                raise PoseSyncError(f"[pose-sync] REFUSING {path}:{i + 1}: delta_us range "
                                    f"[{d.min()}, {d.max()}] is outside [0, {DELTA_US_MAX}] us -- a sign or "
                                    f"unit error, not an image-after-pose offset")
            row = SyncRow(delta_s=d / 1e6, dt_s=dt)
            if sid in table and (table[sid].delta_s.shape != row.delta_s.shape
                                 or not np.array_equal(table[sid].delta_s, row.delta_s)
                                 or table[sid].dt_s != row.dt_s):
                raise PoseSyncError(f"[pose-sync] REFUSING {path}: sid {sid} appears twice with different "
                                    f"content")
            table[sid] = row
    if n == 0:
        raise PoseSyncError(f"[pose-sync] REFUSING {path!r}: ZERO rows")
    return PoseSync(table=table, meta=meta, path=str(path), n_rows=n, sign=float(sign))


# --------------------------------------------------------------------------------------------------
# 4. The window view the dataset applies
# --------------------------------------------------------------------------------------------------
def window_shifted_tracks(ps: PoseSync, sid: int, now: int, raw_offset: int, poses, actions=None):
    """The poses (and, if given, actions) of one episode re-sampled at the NOW row's image instant.

    ``now`` is the PROVIDER row of the window's last observed frame; ``raw_offset`` is ``n_stack - 1``
    (``V3Dataset._raw_offset``).  Returns ``(poses', actions' | None, shift_rows_frac)`` as float32 numpy,
    or ``None`` when the sidecar does not cover the clip / row (the caller counts and falls back to the
    unshifted track -- never a silent partial).
    """
    s = ps.shift_for(sid, int(now) + int(raw_offset))
    if s is None:
        return None
    P = shift_rows(np.asarray(poses), s, angle_cols=(2,)).astype(np.float32)
    A = None
    if actions is not None:
        A = shift_rows(np.asarray(actions), s).astype(np.float32)
    return P, A, s

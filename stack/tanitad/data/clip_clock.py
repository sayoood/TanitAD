"""THE LABEL CLOCK — the true time of a cache row on the v7/v8 label timeline.

⛔⛔ WHY THIS MODULE EXISTS (A16, 2026-09-26, MEASURED on the live refcv6 run).
``V3Dataset`` read every v7/v8 label at ``t_now = (t + w - 1) * 0.1`` — a PROVIDER row times a
literal. Both halves were wrong:

1. **The row is a PROVIDER row.** ``v2_dataset.py:36`` drops the first ``n_stack - 1`` raw frames
   (``poses[n_stack-1:]``); the labels live on the RAW clip timeline
   (``egomotion_source.py:56-57``). ``s2_labels.py:738-740`` adds the ``n_stack - 1`` offset; the
   refcv6 dataset did not (0.20 s).
2. **A row is not 0.1 s.** The cache grid is ``linspace(t0, tN, int(span * 10))``
   (``v2_compressed.py:119-120``), whose step is ``span / (n - 1)``: MEASURED median
   **0.1006666 s** over 4,357 train clips (row times recovered from the 100 Hz egomotion log), and
   the first camera row sits **+0.113 s** (median) after the log's origin, which IS the label
   timeline's zero.

Together the trainer read labels **0.369 s early** at the 8.0 s anchor (median); **19,044 of
179,129** tactical-supervised training windows (10.6 %) sat outside the true ±2 s band
(``…/2026-09-26-refcv6-frozen-trunk-audit/raw/q4d_label_offset_true_clock.json``).

⇒ the true time of provider row ``r`` is ``grid_start_s + (r + n_stack - 1) * dt_s``.

* ``dt_s`` is recoverable from the cache ITSELF: rows carry ``(x, y)`` and the log's own speed
  ``v = hypot(vx, vy)``, both interpolated at the same query times, so over moving rows
  ``dt = Σ|Δxy| / Σ v̄`` — an identity against a quantity that knows nothing about our cadence.
  :func:`pose_dt` implements it (MEASURED agreement with the log inversion: median 0.100667 vs
  0.1006666 s).
* ``grid_start_s`` is NOT recoverable from the cache: it is the camera grid's first timestamp
  relative to the egomotion log's first sample. It comes from a CLOCK SIDECAR built from the log
  (``scripts/build_clip_clock_sidecar.py``), keyed on ``sid = stable_episode_id(clip_id)``.
  Without one the dataset uses ``0.0`` and COUNTS every clip it had to (the corpus median is
  +0.113 s, so the residual is stated, not hidden).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch

#: the literal every consumer assumed; used ONLY where a clip carries too little motion for the
#: identity (a stationary clip has no displacement to divide).
NOMINAL_DT_S: float = 0.1
#: rows slower than this are excluded from the identity (GNSS/odometry noise dominates).
POSE_DT_V_MIN_MS: float = 2.0
#: fewer moving steps than this and the identity is not attempted.
POSE_DT_MIN_STEPS: int = 10
#: the physically possible band for this corpus's grid: n = int(span*10) samples over span s gives
#: dt in [0.1, 0.1 * (1 + 1/(n-1)) + 0.1/(n-1)); anything outside is a broken track, not a clock.
POSE_DT_BAND_S: tuple[float, float] = (0.098, 0.104)
SIDECAR_SCHEMA = "tanitad.clip_clock/1"


class ClipClockError(ValueError):
    """A clock sidecar that cannot be trusted is refused, never guessed around."""


def pose_dt(poses: torch.Tensor, *, v_min: float = POSE_DT_V_MIN_MS,
            min_steps: int = POSE_DT_MIN_STEPS) -> float | None:
    """``[T, >=4]`` provider poses ``(x, y, yaw, v)`` -> the row step in seconds, or ``None``.

    ``Σ|Δxy| / Σ ½(v_i + v_{i+1})`` over steps whose BOTH ends move faster than ``v_min``.
    ``None`` when fewer than ``min_steps`` steps qualify or the result leaves
    :data:`POSE_DT_BAND_S` — the caller then falls back to :data:`NOMINAL_DT_S` and counts it.
    """
    p = poses.detach().to(torch.float64)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 4:
        return None
    d = torch.hypot(p[1:, 0] - p[:-1, 0], p[1:, 1] - p[:-1, 1])
    vb = 0.5 * (p[1:, 3] + p[:-1, 3])
    m = (p[1:, 3] > v_min) & (p[:-1, 3] > v_min)
    if int(m.sum()) < int(min_steps):
        return None
    dt = float(d[m].sum() / vb[m].sum())
    lo, hi = POSE_DT_BAND_S
    if not (math.isfinite(dt) and lo <= dt <= hi):
        return None
    return dt


@dataclass(frozen=True)
class ClockSidecar:
    table: dict          # sid -> (grid_start_s, dt_s)
    meta: dict
    path: str
    n_rows: int


def read_clip_clock_sidecar(path: str) -> ClockSidecar:
    """JSON-lines ``{"sid", "grid_start_s", "dt_s", ...}`` -> :class:`ClockSidecar`.

    ⛔ REFUSES: a missing file; a row without ``sid`` / ``grid_start_s`` / ``dt_s``; a ``dt_s``
    outside :data:`POSE_DT_BAND_S`; a ``|grid_start_s|`` above 1 s (the corpus's measured range is
    -0.27 … +0.24 s — a larger value is a unit error, e.g. microseconds read as seconds); a
    duplicated ``sid`` with different values; zero rows.
    """
    if not os.path.isfile(path):
        raise ClipClockError(f"[clip-clock] ⛔ {path!r} does not exist")
    meta: dict = {}
    mp = path + ".meta.json"
    if os.path.isfile(mp):
        with open(mp, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("schema") not in (None, SIDECAR_SCHEMA):
            raise ClipClockError(f"[clip-clock] ⛔ {mp}: schema {meta.get('schema')!r} is not "
                                 f"{SIDECAR_SCHEMA!r}")
    table: dict = {}
    n = 0
    lo, hi = POSE_DT_BAND_S
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            try:
                sid = int(r["sid"])
                g0 = float(r["grid_start_s"])
                dt = float(r["dt_s"])
            except (KeyError, TypeError, ValueError) as e:
                raise ClipClockError(f"[clip-clock] ⛔ {path}:{i + 1}: needs sid / grid_start_s / "
                                     f"dt_s ({type(e).__name__}: {e})") from e
            if not (lo <= dt <= hi):
                raise ClipClockError(f"[clip-clock] ⛔ {path}:{i + 1}: dt_s {dt} outside "
                                     f"{POSE_DT_BAND_S} -- not a row step of this corpus")
            if not (abs(g0) <= 1.0):
                raise ClipClockError(f"[clip-clock] ⛔ {path}:{i + 1}: |grid_start_s| {g0} > 1 s "
                                     f"-- a unit error, not a clock offset")
            if sid in table and table[sid] != (g0, dt):
                raise ClipClockError(f"[clip-clock] ⛔ {path}: sid {sid} appears twice with "
                                     f"different clocks")
            table[sid] = (g0, dt)
    if n == 0:
        raise ClipClockError(f"[clip-clock] ⛔ {path!r} has ZERO rows")
    return ClockSidecar(table=table, meta=meta, path=str(path), n_rows=n)

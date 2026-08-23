#!/usr/bin/env python3
"""Build per-window LEAD TRACKS in the window-origin ego frame from `obstacle.offline`.

This is the ingest half of the distance-keeping work item. `lead_metrics.py` is the pure metric;
this file is the only part that touches bytes.

⭐ THE COORDINATE PROBLEM, which is the whole reason this file exists.
`obstacle.offline` cuboids carry ``reference_frame='rig'`` — each row is expressed in the ego's
frame **at that row's own timestamp**. Two rows 1 s apart are therefore in two different frames.
`lead_state_gate.lead_frame` never has to care, because it only ever reads the lead at the *current*
instant to build an input feature. Scoring a *predicted trajectory* does: the arm's waypoints all
live in the frame at t0, so the lead's future positions must be brought into that same frame::

    world:  L_w(t) = ego_xy(t) + R(yaw(t)) @ [center_x, center_y]
    t0:     L_0(t) = R(-yaw(t0)) @ (L_w(t) - ego_xy(t0))

``egomotion`` supplies ``x``/``y`` and the quaternion, so the composition is exact and needs no
approximation. Skipping it and using the raw rig coordinates would understate the gap by roughly the
distance the ego travels over the horizon (~27 m at 13.6 m/s over 2 s) — i.e. it would invent
tailgating everywhere.

⛔ CAUSALITY. The **lead SELECTION** at t0 is strictly causal — it uses only cuboids timestamped
<= t0, the same rule as `lead_state_gate.lead_frame`. The lead's **future positions** are ground
truth about the world, exactly like the ground-truth ego waypoints an ADE is measured against; they
are a scoring input and never an arm input. `assert_selection_causal` enforces the first half.

⛔ SPAN. `egomotion` runs 20–140 s while `obstacle.offline` stops at ~20 s (E-GOAL-1's finding). A
window whose horizon leaves the intersection yields "no lead", which is indistinguishable from an
empty road — so windows are only emitted inside the measured intersection, and the count dropped
for span is reported rather than silently absorbed.

PARITY: read-only over `labels/*.zip` and `r0/phase0_selection.parquet`. No clip is re-selected;
`physicalai-train-e438721ae894` / skip-hash `f09e44db` are untouched.
PRIVACY: PhysicalAI-AV is gated-confidential — artifacts carry `clip_<sha256[:8]>`, never the UUID.
"""
from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack" / "scripts"))

from lead_state_gate import (  # noqa: E402
    CLOSING_CLIP_MS, LEAD_LAT_M, LEAD_MAX_GAP_M, MAX_STALE_S,
    VEHICLE_CLASSES, quaternion_yaw,
)

ROOT = Path(r"C:/Users/Admin/tanitad-data/physicalai")
#: `obstacle.offline` is dense over roughly the first 20 s of a clip.
OBS_END_S = 20.0
#: ego history depth, so a window's t0 never precedes the clip.
HIST_S = 1.0


def anon(clip_id: str) -> str:
    return "clip_" + hashlib.sha256(str(clip_id).encode()).hexdigest()[:8]


def local_chunks() -> list[str]:
    """Chunks with obstacle.offline AND egomotion present on this disk."""
    def ch(pat):
        return {p.name.split("_")[-1].split(".")[0]
                for p in (ROOT / pat).glob("*.zip")}
    return sorted(ch("labels/obstacle.offline") & ch("labels/egomotion"))


def read_chunk(chunk: str) -> tuple[zipfile.ZipFile, zipfile.ZipFile]:
    return (zipfile.ZipFile(ROOT / "labels" / "obstacle.offline" /
                            f"obstacle.offline.chunk_{chunk}.zip"),
            zipfile.ZipFile(ROOT / "labels" / "egomotion" /
                            f"egomotion.chunk_{chunk}.zip"))


def _member(z: zipfile.ZipFile, clip: str) -> str | None:
    for n in z.namelist():
        if n.endswith(".parquet") and n.split("/")[-1].startswith(clip):
            return n
    return None


def ego_poses(ego: pd.DataFrame) -> dict:
    """Sorted (t_s, x, y, yaw_unwrapped, speed) — the frame chain everything else composes with."""
    t = ego["timestamp"].to_numpy(np.float64) / 1e6
    o = np.argsort(t)
    g = lambda c: ego[c].to_numpy(np.float64)[o]  # noqa: E731
    yaw = np.unwrap(quaternion_yaw(g("qx"), g("qy"), g("qz"), g("qw")))
    return {"t": t[o], "x": g("x"), "y": g("y"), "yaw": yaw,
            "v": np.hypot(g("vx"), g("vy"))}


def _at(p: dict, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.interp(ts, p["t"], p["x"]), np.interp(ts, p["t"], p["y"]),
            np.interp(ts, p["t"], p["yaw"]), np.interp(ts, p["t"], p["v"]))


def ego_path_in_window(p: dict, t0: float, ts: np.ndarray) -> np.ndarray:
    """The TRUE ego positions at ``ts`` expressed in the window-origin frame at ``t0``. (K, 2)."""
    x0, y0, yaw0, _ = _at(p, np.array([t0]))
    x, y, _, _ = _at(p, ts)
    dx, dy = x - x0[0], y - y0[0]
    c, s = np.cos(yaw0[0]), np.sin(yaw0[0])
    return np.stack([dx * c + dy * s, -dx * s + dy * c], axis=1)


def select_lead(obs: pd.DataFrame, t0: float) -> tuple[str | None, float, float]:
    """CAUSAL lead choice at ``t0`` -> (track_id, gap_m, size_x). Mirrors `lead_frame`'s rule.

    Nearest vehicle whose last sample at-or-before ``t0`` (within `MAX_STALE_S`) sits ahead,
    within `LEAD_MAX_GAP_M`, and inside the |lat| < `LEAD_LAT_M` corridor.
    """
    past = obs[obs["_t"] <= t0]
    if past.empty:
        return None, np.nan, np.nan
    last = past.sort_values("_t").groupby("track_id", sort=False).tail(1)
    last = last[(t0 - last["_t"]) <= MAX_STALE_S]
    last = last[last["label_class"].isin(VEHICLE_CLASSES)]
    if last.empty:
        return None, np.nan, np.nan
    gap = last["center_x"].to_numpy(np.float64) - last["size_x"].to_numpy(np.float64) / 2.0
    lat = last["center_y"].to_numpy(np.float64)
    cand = (gap >= 0.0) & (gap <= LEAD_MAX_GAP_M) & (np.abs(lat) < LEAD_LAT_M)
    if not cand.any():
        return None, np.nan, np.nan
    j = int(np.flatnonzero(cand)[np.argmin(gap[cand])])
    row = last.iloc[j]
    return str(row["track_id"]), float(gap[j]), float(row["size_x"])


def assert_selection_causal(obs: pd.DataFrame, t0: float, track: str | None) -> None:
    """The selected track must have a sample at or before t0 — the anti-oracle runtime check."""
    if track is None:
        return
    tt = obs.loc[obs["track_id"].astype(str) == track, "_t"]
    if not (tt <= t0).any():
        raise AssertionError(f"lead {track} selected at t0={t0:.3f}s with no causal sample "
                             f"(earliest {float(tt.min()):.3f}s) — ORACLE LEAK")


def lead_track_in_window(obs: pd.DataFrame, p: dict, t0: float, ts: np.ndarray,
                         track: str) -> np.ndarray:
    """The lead's centre at ``ts`` in the window-origin frame at ``t0``. (K, 2), NaN where absent.

    Rig -> world -> t0 frame. See the module docstring for why the composition is mandatory.
    """
    sub = obs[obs["track_id"].astype(str) == track].sort_values("_t")
    tt = sub["_t"].to_numpy(np.float64)
    cx = sub["center_x"].to_numpy(np.float64)
    cy = sub["center_y"].to_numpy(np.float64)
    out = np.full((ts.size, 2), np.nan)
    if tt.size == 0:
        return out
    j = np.searchsorted(tt, ts, side="right") - 1
    ok = (j >= 0) & (np.abs(ts - tt[np.clip(j, 0, tt.size - 1)]) <= MAX_STALE_S)
    if not ok.any():
        return out
    jj = np.clip(j, 0, tt.size - 1)
    ex, ey, eyaw, _ = _at(p, ts)                       # ego pose at each sample's own time
    ce, se = np.cos(eyaw), np.sin(eyaw)
    lw_x = ex + cx[jj] * ce - cy[jj] * se              # rig -> world
    lw_y = ey + cx[jj] * se + cy[jj] * ce
    x0, y0, yaw0, _ = _at(p, np.array([t0]))
    dx, dy = lw_x - x0[0], lw_y - y0[0]
    c0, s0 = np.cos(yaw0[0]), np.sin(yaw0[0])
    out[ok, 0] = (dx * c0 + dy * s0)[ok]               # world -> t0 frame
    out[ok, 1] = (-dx * s0 + dy * c0)[ok]
    return out


def cv_path(v0: float, ts_rel: np.ndarray) -> np.ndarray:
    """hold-v0 constant-velocity baseline in the window-origin frame: straight ahead at v0."""
    return np.stack([v0 * ts_rel, np.zeros_like(ts_rel)], axis=1)


def build_windows(clip: str, zo: zipfile.ZipFile, ze: zipfile.ZipFile,
                  horizon_s: float = 2.0, dt: float = 0.5,
                  stride_s: float = 1.0) -> tuple[list[dict], dict]:
    """-> (windows, diagnostics). Each window carries the GT path, the CV path and the lead track."""
    mo, me = _member(zo, clip), _member(ze, clip)
    diag = {"clip": anon(clip), "n_windows": 0, "n_lead": 0,
            "dropped_span": 0, "dropped_no_lead": 0}
    if mo is None or me is None:
        diag["skip"] = "missing obstacle.offline or egomotion member"
        return [], diag

    obs = pd.read_parquet(io.BytesIO(zo.read(mo)))
    obs["_t"] = obs["timestamp_us"].to_numpy(np.float64) / 1e6
    ego = pd.read_parquet(io.BytesIO(ze.read(me)))
    p = ego_poses(ego)

    # the INTERSECTION of the two spans — a horizon outside it fakes an empty road
    t_lo = max(float(p["t"][0]) + HIST_S, float(obs["_t"].min()))
    t_hi = min(float(p["t"][-1]), float(obs["_t"].max()), float(obs["_t"].min()) + OBS_END_S)
    diag["span_s"] = round(t_hi - t_lo, 3)
    if t_hi - t_lo <= horizon_s:
        diag["skip"] = f"usable span {t_hi - t_lo:.2f}s <= horizon {horizon_s}s"
        return [], diag

    k = int(round(horizon_s / dt))
    ts_rel = np.arange(1, k + 1) * dt
    out = []
    t0 = t_lo
    while t0 + horizon_s <= t_hi:
        diag["n_windows"] += 1
        ts = t0 + ts_rel
        track, gap0, size_x = select_lead(obs, t0)
        if track is None:
            diag["dropped_no_lead"] += 1
            t0 += stride_s
            continue
        assert_selection_causal(obs, t0, track)
        lead = lead_track_in_window(obs, p, t0, ts, track)
        if not np.isfinite(lead).any():
            diag["dropped_span"] += 1
            t0 += stride_s
            continue
        _, _, _, v0 = _at(p, np.array([t0]))
        diag["n_lead"] += 1
        out.append({
            "clip": anon(clip), "t0": round(float(t0), 3),
            "gt_path": ego_path_in_window(p, t0, ts),
            "cv_path": cv_path(float(v0[0]), ts_rel),
            "lead": lead, "lead_len": float(size_x), "v0": float(v0[0]),
            "gap0_m": float(gap0),
        })
        t0 += stride_s
    return out, diag

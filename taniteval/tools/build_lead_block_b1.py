#!/usr/bin/env python3
"""Build the PER-FRAME lead block for the v7.2 EVAL clips (B1 corpus) — backlog R1.

⛔ WHY. ``taniteval/tools/refav1_arm.py`` reported the LONGITUDINAL family's
distance-keeping half UNAVAILABLE ("no lead block on the 141-clip grid"). The binding
rule (Sayed 2026-08-02, clause 3) says a missing metric is a WORK ITEM, not an excuse.
The instrument exists for the parity corpus — ``taniteval.lead_source`` (selection +
frame chain + span guard), ``taniteval.lead_metrics`` (headway / time-gap / min-TTC),
``four_families._distance_keeping`` (the consumer) — but every builder so far was
POSITIONAL over ``rollout.collect``'s window grid. refav1's grid is different: a window
is (episode, cache step t), its origin is RAW v2ep frame ``2t`` (0.2 s tick, the
loader's ``cache j <-> frame 2j`` rule), and its horizon is ``--horizon-k`` × 0.2 s.
This builder therefore emits ONE ROW PER (clip, RAW 10 Hz frame) and the adapter
joins by ``(clip_id, frame)`` — never by position.

THE TIME BASE — RECONSTRUCTED, THEN CROSS-CHECKED, NEVER ASSUMED.
The B1 v2ep poses are ``physicalai.signals_at(egomotion, t_query)`` with
``t_query = linspace(t_cam[0], t_cam[-1], int(span_s * TARGET_HZ))`` (the builder's own
formula, ``v2_compressed._resampled`` / ``physicalai.build_episode``). Both inputs ship
with the corpus release (``egomotion/egomotion_alpamayo.tar``, ``timestamps/
timestamps.tar``), so the grid AND the poses of every clip are reconstructed here
exactly. ⭐ MEASURED 2026-09-02 on the 20 v2ep clips of the local eval slice: max
|Δxy| = 0.00000 m, max |Δyaw| = 0.00000°, max |Δv| = 0.00000 m/s (float32-identical),
and ``lead_source.register_poses_to_time`` (registration by CONTENT) recovers the
same affine grid to ≤ 1.8 ms (median 0.29 ms). The grid spacing is 0.1005–0.1008 s,
NOT 0.1 s, and frame 0 sits at −0.16 … −0.0002 s clip time — assuming ``t = 0.1·i``
would drift ~0.13 s by frame 200 (~1.8 m of lead displacement at 13.6 m/s).
``--v2ep-dir`` re-runs that cross-check on every clip it finds and REFUSES the build
above 1 cm / 1° / 1 cm/s.

FRAME INDEX SPACE: the RAW v2ep index ``i ∈ [0, n_target)`` — the refav1 loader reads
RAW poses (no n_stack trim; refav1_loader.py docstring), so frame ``2t`` is cache
step ``t``. This is NOT the post-trim provider index that the S2 label join and
``build_obstacle_join.py`` use (those are ``i − (n_stack − 1)``).

⭐ THE POSE-SYNC CLOCK (``--pose-sync-sidecar``, 2026-11-12). A run trained with the
pose-sync sidecar (``tanitad/data/pose_sync.py``, X10: refcv8 / refcv9) reads its ego state
``delta`` LATER than the pose row -- at the NOW image's capture instant, 0..35 ms -- so the
dump's ``v0`` and GT path are NOT ``poses[i, 3]`` at the pose clock. MEASURED on the first
real refcv9 E2 read (Thor, 2026-11-12): the distance-keeping join's label-free speed proof
REFUSED all 70 DEV episodes, max |dump v0 − block speed| 0.008–0.060 m/s per episode, while
the eval cache and this block agreed to ≤ 2e-5 m/s and every whole-frame offset was worse
(the frame map was right; the clock was not). With the sidecar, every row is rebuilt at the
instant the dataset reads: ``t0 = t_pose + s·dt_grid`` with ``s`` the TRAINER's own row
shift (``PoseSync.shift_for`` of that RAW row), and the ``speeds`` column becomes the
dataset's own ``v0`` -- computed by calling ``pose_sync.window_shifted_tracks`` (the function
``V3Dataset._pose_sync_apply`` calls), never a re-implementation. The join's 1e-3 m/s proof
is therefore UNCHANGED and exact. The build refuses unless the dataset's ``v0`` IS the grid-row
egomotion speed linearly re-sampled at fractional row ``i + s`` (the tripwire's 1e-3 m/s): an
identity that a wrong row, sign or shift breaks. The egomotion speed AT the shifted instant is
kept as ``speeds_egomotion``; its gap to ``v0`` is 10 Hz-vs-100 Hz interpolation error (MEASURED
on the 139 eval clips: median 0.0012, max 0.163 m/s) -- reported in coverage, never gated. ``--pose-sync-sign`` and ``--pose-sync-raw-offset``
have NO default: they are the run's ``config.json`` ``pose_sync.eval.sign`` and the eval
cache's ``n_stack − 1`` (the dump manifest's ``provider_to_raw_frame_offset``). A clip the
sidecar does not cover is built UNSHIFTED, as the trainer reads it, and counted.

FOUR STATES, NEVER TWO (``lead_source``): LEAD / NO_LEAD / NOT_STRAIGHT / NO_LABEL.
NO_LABEL (no ``obstacle.offline`` for the clip, or the window's horizon leaves the
labelled span) is NEVER free flow. 2 of the 147 eval clips carry no
``obstacle.offline`` member in their chunk (MEASURED by the range pull; the dataset
card says 97.44 % coverage) — their rows are NO_LABEL with the reason in ``coverage``.

CONVENTIONS — inherited from ``lead_source`` / ``lead_metrics`` unchanged: window-origin
ego frame at t0, x forward / y left, metres, clip-local; gap = along − size_x/2 (rig
origin to the lead's REAR face); selection strictly causal (cuboids ≤ t0, staleness
≤ 0.5 s, |lat| < 2 m, gap ≤ 80 m, vehicle classes); the straight-driving gate is the
corridor's own half-width. The lead's FUTURE track is GT ENVIRONMENT — admissible for
SCORING a plan (like GT waypoints), never a model input.

THE PULL (``--pull``): ``obstacle.offline.chunk_{c:04d}.zip`` are 3–111 MB each and
the 147 clips span 122 chunks (158 GB in total on the hub). The corpus release's
``index/clip_to_chunk.parquet`` names each clip's chunk; the zip's central directory
is range-read from its final 256 KB and ONLY the ``{clip}.obstacle.offline.parquet``
member is fetched (the ``tools/pull_egomotion_range.py`` recipe of the release).
MEASURED: 145 members, 77 MB, 59 s. Every member is parse-verified (schema columns)
before it is banked; sha256 recorded.

CONTAINER (``np.savez_compressed``; loads through ``eval_four_families.load_lead_block``):
    clip_id  <U36 [R]   frame int64 [R] (RAW v2ep index)   t0_s float64 [R] (clip s)
    eid      <U36 [R]   = clip_id — the episode-cluster unit
    leads    float64 [R, K, 2]  lead CENTRE at t0 + ts_rel in the t0 frame; NaN = none
    lead_lens float64 [R]  size_x   speeds float64 [R]  ego speed at t0 (= poses[i, 3])
    state    <U12 [R]   gap0_m float64 [R]   has_lead bool [R]
    ts_rel_s float64 [K] = dt·(1..K)   dt_s float64 [1]
    lead_track_id <U [R]  rel_speed_mps [R] (lead along-speed − ego speed at t0, + = pulling
      away; least-squares over the lead's own samples in the last 0.6 s)  headway_time_s [R]
      (gap0 / v_ego, NaN below MIN_SPEED_MPS)
    gt_headway_min_m / gt_time_gap_min_s / gt_min_ttc_s / gt_n_steps_in_corridor [R]
      — ``lead_metrics.distance_keeping`` of the GT ego path (the D-LEAD-1 GT arm's
      value on this window; the known value the kinematic-contract control must match)
    straight_dep_m / straight_checked_m [R]   coverage_json uint8   meta_json uint8
    ONLY with ``--pose-sync-sidecar``: t0_pose_s float64 [R] (the pose-clock instant; ``t0_s``
      is then the SHIFTED instant)  pose_sync_shift_s float64 [R]  speeds_egomotion float64 [R]
      (``speeds`` is then the dataset's pose-synced ``v0``)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import tarfile
import time
import zipfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))       # <repo>/taniteval/tools
_TE_PARENT = os.path.dirname(_HERE)                       # <repo>/taniteval
_REPO = os.path.dirname(_TE_PARENT)                       # <repo>
_TE_PKG = os.path.join(_TE_PARENT, "taniteval")


def _bootstrap_paths() -> None:
    """The namespace-package shadow guard (see refav1_arm.py): evict a wrongly
    bound outer ``taniteval`` and preflight the real package."""
    for p in (os.path.join(_REPO, "stack"), _TE_PARENT):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    m = sys.modules.get("taniteval")
    if m is not None:
        paths = [os.path.normcase(os.path.abspath(p))
                 for p in (getattr(m, "__path__", None) or [])]
        if os.path.normcase(os.path.abspath(_TE_PKG)) not in paths:
            for k in [k for k in sys.modules
                      if k == "taniteval" or k.startswith("taniteval.")]:
                del sys.modules[k]
    import taniteval.lead_source  # noqa: F401
    import taniteval.lead_metrics  # noqa: F401


_bootstrap_paths()
from taniteval import lead_metrics as lm  # noqa: E402
from taniteval import lead_source as ls  # noqa: E402

VERSION = "b1-eval-lead-block v1 (2026-09-02)"
K_DEFAULT = 10
DT_DEFAULT = 0.2
STATE_DTYPE = "<U12"                       # NOT_STRAIGHT is 12 chars; <U8 truncates it
#: the lead's own samples inside this window before t0 fit its along-speed
REL_SPEED_LOOKBACK_S = 0.6
#: the v2ep cross-check tolerances (the brief: centimetres / < 1 degree)
XCHECK_TOL_M, XCHECK_TOL_DEG, XCHECK_TOL_MPS = 0.01, 1.0, 0.01
HF_DATASET = "nvidia/PhysicalAI-Autonomous-Vehicles"
HF_OBS_BASE = (f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/"
               "labels/obstacle.offline/")
OBS_NEED_COLS = ("timestamp_us", "track_id", "center_x", "center_y", "size_x",
                 "label_class")
REQUIRED_KEYS = ("leads", "lead_lens", "speeds", "state", "eid",
                 "clip_id", "frame", "t0_s", "ts_rel_s")
#: --pose-sync-sidecar: the dataset's pose-synced v0 against the grid-row egomotion speeds re-sampled
#: at the same fractional row -- the speed tripwire's own tolerance (build_clip_rows)
PSYNC_V0_TOL_MPS = 1e-3


def _p(*a):
    print(*a, flush=True)


def _sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# the sibling ingest helpers, IMPORTED (never copied): ego_series / obs_series  #
# --------------------------------------------------------------------------- #
def _sibling():
    spec = importlib.util.spec_from_file_location(
        "build_lead_block_for_b1", os.path.join(_HERE, "build_lead_block.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SIB = None


def ego_track(ego_df) -> dict:
    """``{t, x, y, yaw (unwrapped), v}`` from an egomotion parquet DataFrame —
    ``build_lead_block.ego_series``, the pod builder's own reader."""
    global _SIB
    if _SIB is None:
        _SIB = _sibling()
    return _SIB.ego_series(ego_df)


def obs_track(obs_df) -> dict | None:
    """``build_lead_block.obs_series`` (+ its reference-frame skew check)."""
    global _SIB
    if _SIB is None:
        _SIB = _sibling()
    return _SIB.obs_series(obs_df, ls.VEHICLE_CLASSES)


# --------------------------------------------------------------------------- #
# the episode grid + poses, from the corpus release's own inputs                #
# --------------------------------------------------------------------------- #
def episode_grid(t_frames) -> tuple[np.ndarray, float, int]:
    """``(t_query_native, unit, n_target)`` — ``v2_compressed._resampled`` /
    ``physicalai.build_episode`` lines 711-718, with ``TARGET_HZ`` imported from
    the stack so the two cannot drift apart silently. Pinned against REAL v2ep
    poses by ``stack/tests/test_refav1_lead_block.py`` (float32-identical)."""
    from tanitad.data.physicalai import TARGET_HZ
    t_frames = np.asarray(t_frames, dtype=np.float64)
    if t_frames.size < 2:
        raise ValueError("timestamps parquet has < 2 frames")
    span = t_frames[-1] - t_frames[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):                 # ns/us/ms clocks -> seconds
        if span / cand > 1.0:
            unit = cand
            break
    n_target = max(int(span / unit * TARGET_HZ), 4)
    return np.linspace(t_frames[0], t_frames[-1], n_target), unit, n_target


def episode_poses(ego_df, t_query) -> np.ndarray:
    """``physicalai.signals_at`` — THE builder's pose path — ``[n, 4]`` float32
    (x, y, yaw wrapped, v = hypot(vx, vy))."""
    from tanitad.data.physicalai import signals_at
    return signals_at(ego_df, np.asarray(t_query, dtype=np.float64))[1]


def crosscheck_v2ep(v2ep_path: str, poses_rec: np.ndarray) -> dict:
    """Compare reconstructed poses with the banked v2ep record's RAW poses."""
    import torch
    d = torch.load(v2ep_path, map_location="cpu", weights_only=False, mmap=True)
    pv = d["poses"].float().numpy().astype(np.float64)
    pr = np.asarray(poses_rec, dtype=np.float64)
    n = min(len(pv), len(pr))
    dxy = np.hypot(pr[:n, 0] - pv[:n, 0], pr[:n, 1] - pv[:n, 1])
    dyaw = np.abs((pr[:n, 2] - pv[:n, 2] + np.pi) % (2 * np.pi) - np.pi) * 180.0 / np.pi
    dv = np.abs(pr[:n, 3] - pv[:n, 3])
    out = {"file": os.path.basename(v2ep_path), "T_v2ep": int(len(pv)),
           "T_reconstructed": int(len(pr)), "n_compared": int(n),
           "max_dxy_m": float(dxy.max()), "p95_dxy_m": float(np.percentile(dxy, 95)),
           "max_dyaw_deg": float(dyaw.max()), "max_dv_mps": float(dv.max()),
           "clip_id_in_record": d.get("clip_id"), "n_stack": int(d.get("n_stack", 0))}
    out["pass"] = bool(len(pv) == len(pr) and out["max_dxy_m"] <= XCHECK_TOL_M
                       and out["max_dyaw_deg"] <= XCHECK_TOL_DEG
                       and out["max_dv_mps"] <= XCHECK_TOL_MPS)
    return out


# --------------------------------------------------------------------------- #
# per-frame extras: GT path in the t0 frame, the lead's relative speed          #
# --------------------------------------------------------------------------- #
def gt_paths_t0frame(ego: dict, t0s, ts_rel) -> np.ndarray:
    """``[W, K, 2]`` GT ego positions at ``t0 + ts_rel`` in each window's t0 frame
    (x forward, y left) — egomotion interpolated, the same composition as
    ``lead_source.lead_track_in_window`` uses for the lead."""
    et = np.asarray(ego["t"], float)
    ex, ey, eyaw = (np.asarray(ego["x"], float), np.asarray(ego["y"], float),
                    np.asarray(ego["yaw"], float))
    t0s = np.asarray(t0s, float).reshape(-1)
    ts_rel = np.asarray(ts_rel, float).reshape(-1)
    tq = t0s[:, None] + ts_rel[None, :]                                   # [W,K]
    x0, y0 = np.interp(t0s, et, ex), np.interp(t0s, et, ey)
    yaw0 = np.interp(t0s, et, eyaw)
    xq, yq = np.interp(tq, et, ex), np.interp(tq, et, ey)
    dx, dy = xq - x0[:, None], yq - y0[:, None]
    c0, s0 = np.cos(yaw0)[:, None], np.sin(yaw0)[:, None]
    return np.stack([dx * c0 + dy * s0, -dx * s0 + dy * c0], axis=-1)


def lead_rel_speed(obs: dict, track, t0: float, ego: dict,
                   lookback_s: float = REL_SPEED_LOOKBACK_S) -> float:
    """Lead along-speed (t0 frame, world-referenced) minus the ego speed at t0.

    Least squares over the lead's OWN samples in ``[t0 − lookback, t0]`` (causal),
    each composed into the t0 frame by ``lead_source.lead_track_in_window`` at its
    exact sample time (staleness 0). NaN with fewer than 2 samples or a span
    under 0.15 s. ``+`` = the lead is pulling away."""
    ot = np.asarray(obs["t"], float)
    m = (np.asarray(obs["track"]) == track) & (ot <= t0) & (ot >= t0 - lookback_s)
    ts = np.unique(ot[m])
    if ts.size < 2 or (ts[-1] - ts[0]) < 0.15:
        return float("nan")
    xy = ls.lead_track_in_window(ot, obs["track"], obs["center_x"], obs["center_y"],
                                 track, float(t0), ts, ego["t"], ego["x"], ego["y"],
                                 ego["yaw"], max_stale_s=1e-6)
    ok = np.isfinite(xy[:, 0])
    if ok.sum() < 2:
        return float("nan")
    A = np.column_stack([np.ones(int(ok.sum())), ts[ok] - ts[ok][0]])
    coef, *_ = np.linalg.lstsq(A, xy[ok, 0], rcond=None)
    v_ego = float(np.interp(t0, np.asarray(ego["t"], float), np.asarray(ego["v"], float)))
    return float(coef[1]) - v_ego


# --------------------------------------------------------------------------- #
# the pose-sync clock: the TRAINER's own re-sampling, called per row            #
# --------------------------------------------------------------------------- #
def _pose_sync_mod():
    """``tanitad.data.pose_sync`` -- the module ``V3Dataset._pose_sync_apply`` uses (one
    implementation, never a copy)."""
    from tanitad.data import pose_sync as _ps
    return _ps


def pose_sync_track(ps, sid: int, poses_raw, raw_offset: int):
    """The dataset's pose-synced ego speed for every RAW row of one clip.

    For a window whose NOW is PROVIDER row ``now`` (RAW row ``i = now + raw_offset``),
    ``V3Dataset._pose_sync_apply`` sets ``pose_last = window_shifted_tracks(ps, sid, now,
    raw_offset, ep.poses)[0][now]`` on the PROVIDER-indexed track ``ep.poses =
    poses_raw[raw_offset:]``, and ``v0 = pose_last[3]``. This calls that very function once
    per row. Returns ``(v0 [n], shift_rows [n])`` as float64; a row with no provider row
    (``i < raw_offset``: no window can sit there) or outside the sidecar's rows is NaN.
    ``None`` when the sidecar does not hold the clip (the trainer reads it UNSHIFTED).
    """
    _ps = _pose_sync_mod()
    if ps.lookup(int(sid)) is None:
        return None
    off = int(raw_offset)
    pr = np.asarray(poses_raw, dtype=np.float32)
    n = int(pr.shape[0])
    prov = pr[off:]
    v0 = np.full(n, np.nan)
    sh = np.full(n, np.nan)
    for i in range(off, n):
        got = _ps.window_shifted_tracks(ps, int(sid), i - off, off, prov)
        if got is None:
            continue
        P, _a, s = got
        v0[i] = float(P[i - off, 3])
        sh[i] = float(s)
    return v0, sh


def pose_sync_clip(ps, sid: int, poses_raw, t_grid_s, raw_offset: int):
    """One clip's ``(t0_shift_s [n], v0_sync [n], info)`` for :func:`build_clip_rows` -- what ``main``
    does per clip. Row ``i`` moves by ``s_i * dt_grid`` (``s_i`` the trainer's row shift, ``dt_grid``
    the clip's own linspace spacing), i.e. to the instant the linear re-sample at fractional row
    ``i + s_i`` represents. Rows without a shift stay put (their speed stays the egomotion one). An
    uncovered clip: zero shift, all-NaN ``v0_sync`` (built UNSHIFTED, as the trainer reads it)."""
    t = np.asarray(t_grid_s, dtype=np.float64).reshape(-1)
    n = t.size
    dt_grid = float((t[-1] - t[0]) / (n - 1))
    got = pose_sync_track(ps, sid, poses_raw, raw_offset)
    if got is None:
        return np.zeros(n), np.full(n, np.nan), {"sid": int(sid), "covered": False}
    v0s, sh_rows = got
    shift_s = np.where(np.isfinite(sh_rows), sh_rows, 0.0) * dt_grid
    return shift_s, v0s, {"sid": int(sid), "covered": True, "dt_grid_s": dt_grid,
                          "dt_sidecar_s": float(ps.lookup(sid).dt_s)}


# --------------------------------------------------------------------------- #
# ONE clip -> per-frame rows + coverage (pure given arrays; unit-tested)        #
# --------------------------------------------------------------------------- #
def build_clip_rows(clip_id: str, ego: dict, t_grid_s, obs: dict | None, *,
                    k: int = K_DEFAULT, dt: float = DT_DEFAULT,
                    poses: np.ndarray | None = None,
                    obs_reason: str | None = None,
                    t0_shift_s=None, v0_sync=None) -> tuple[dict, dict]:
    """Rows for every RAW frame of one clip.

    ``t_grid_s`` [n] clip time of each RAW frame; ``obs`` ``None`` when the clip has
    no ``obstacle.offline`` (``obs_reason`` says why — it lands in coverage).
    ``poses`` (optional, [n, >=4]) is only used for the speed tripwire; it must be
    sampled at the instants the rows are built at (the SHIFTED ones under pose sync).
    ``t0_shift_s`` + ``v0_sync`` (both or neither, [n]): the pose-sync clock (module doc) --
    every row is built at ``t_grid_s + t0_shift_s`` and ``speeds`` is ``v0_sync`` where it
    is finite (the egomotion speed elsewhere, so a window there fails the join's proof).
    """
    if (t0_shift_s is None) != (v0_sync is None):
        raise ValueError("t0_shift_s and v0_sync come together (the pose-sync clock) or not at all")
    t0_pose = np.asarray(t_grid_s, dtype=np.float64).reshape(-1)
    n = t0_pose.size
    shift = None
    if t0_shift_s is not None:
        shift = np.asarray(t0_shift_s, dtype=np.float64).reshape(-1)
        if shift.shape != (n,) or not np.isfinite(shift).all():
            raise ValueError(f"{clip_id}: t0_shift_s must be {n} finite values")
    t0s = t0_pose if shift is None else t0_pose + shift
    ts_rel = np.arange(1, int(k) + 1, dtype=np.float64) * float(dt)
    obs_in = None
    if obs is not None and np.asarray(obs["t"]).size:
        obs_in = {kk: obs[kk] for kk in ("t", "track", "center_x", "center_y",
                                         "size_x", "is_vehicle")}
    blk = ls.lead_block(t0s, ts_rel, obs_in, ego)
    state = np.asarray(blk["state"]).astype(STATE_DTYPE)
    lead = state == ls.LEAD
    speeds = np.asarray(blk["speeds"], float)
    if poses is not None:
        pv = np.asarray(poses, float)[:n, 3]
        d = float(np.max(np.abs(pv - speeds))) if n else 0.0
        if d > 1e-3:
            raise RuntimeError(f"{clip_id}: lead_block speeds (egomotion interp at t0) "
                               f"disagree with poses[:, 3] by {d:.4f} m/s — the time "
                               f"base is not the poses' time base")
    psync = None
    if shift is not None:
        speeds_ego = speeds
        v = np.asarray(v0_sync, dtype=np.float64).reshape(-1)
        if v.shape != (n,):
            raise ValueError(f"{clip_id}: v0_sync must have {n} rows")
        have = np.isfinite(v)
        # THE GATE (an identity, not a closeness): the dataset's v0 IS the 10 Hz speed track
        # linearly re-sampled at fractional row u = i + s (pose_sync.sample_rows, extrapolating
        # past the ends exactly as it does), so the grid-row egomotion speeds re-sampled at the
        # SAME u must reproduce it to the speed tripwire's own 1e-3 m/s. A wrong row, sign or
        # shift fails it. ⚠️ NOT "v0 == egomotion at t0 + s*dt": MEASURED 2026-11-12 on the 139
        # covered eval clips (27,525 in-grid rows) that differs by 10 Hz-vs-100 Hz INTERPOLATION
        # error alone (median 0.0012, p95 0.0078, max 0.163 m/s, identical to the pure
        # interpolation error of the egomotion log), while the shifted instant still matches
        # better than the pose instant (median 0.0037) and the opposite sign (0.0077). That error
        # is a property of the 10 Hz track the model reads; it is reported, never gated.
        dtg = float((t0_pose[-1] - t0_pose[0]) / (n - 1)) if n > 1 else 1.0
        u = np.arange(n, dtype=np.float64) + shift / dtg
        v_grid = np.interp(t0_pose, np.asarray(ego["t"], float), np.asarray(ego["v"], float))
        i0 = np.floor(u).astype(np.int64).clip(0, max(n - 2, 0))
        f = u - i0
        lin = v_grid[i0] + f * (v_grid[np.minimum(i0 + 1, n - 1)] - v_grid[i0])
        dv = np.abs(v[have] - lin[have])
        dmax = float(dv.max()) if dv.size else 0.0
        if dmax > PSYNC_V0_TOL_MPS:
            raise RuntimeError(f"{clip_id}: the dataset's pose-synced v0 is not the 10 Hz speed track "
                               f"re-sampled at row + shift (off by {dmax:.4f} m/s > {PSYNC_V0_TOL_MPS}) — "
                               f"the shifted instant is not the one the dataset reads")
        speeds = np.where(have, v, speeds_ego)
        in_grid = have & (u >= 0.0) & (u <= n - 1)          # the egomotion lookup clamps beyond the grid
        e_int = np.abs(v[in_grid] - speeds_ego[in_grid])
        sh_ms = 1e3 * shift[have] if have.any() else np.zeros(0)
        psync = {"n_rows_synced": int(have.sum()), "n_rows_unsynced": int(n - have.sum()),
                 "n_rows_beyond_grid": int((have & ~in_grid).sum()),
                 "shift_ms": (None if not sh_ms.size else
                              {"min": round(float(sh_ms.min()), 4), "median": round(float(np.median(sh_ms)), 4),
                               "max": round(float(sh_ms.max()), 4)}),
                 "max_abs_v0_sync_minus_10hz_resample_mps": round(dmax, 6),
                 "interp_10hz_vs_egomotion_mps": (None if not e_int.size else
                                                  {"median": round(float(np.median(e_int)), 6),
                                                   "p95": round(float(np.percentile(e_int, 95)), 6),
                                                   "max": round(float(e_int.max()), 6)})}
    # -- the selected track + relative speed + headway time ------------------
    track_id = np.full(n, "", dtype=object)
    rel_v = np.full(n, np.nan)
    hw_t = np.full(n, np.nan)
    if obs_in is not None:
        for i in np.flatnonzero(lead):
            trk, g0, _sx = ls.select_lead_causal(
                obs_in["t"], obs_in["track"], obs_in["center_x"], obs_in["center_y"],
                obs_in["size_x"], obs_in["is_vehicle"], float(t0s[i]))
            if trk is None or abs(float(g0) - float(blk["gap0_m"][i])) > 1e-9:
                raise RuntimeError(f"{clip_id} frame {i}: select_lead_causal "
                                   f"disagrees with lead_block (drift)")
            track_id[i] = str(trk)
            rel_v[i] = lead_rel_speed(obs_in, trk, float(t0s[i]), ego)
    v_ok = lead & (speeds >= lm.MIN_SPEED_MPS)
    hw_t[v_ok] = np.asarray(blk["gap0_m"], float)[v_ok] / speeds[v_ok]
    # -- the GT arm's distance keeping on these rows (the known value) --------
    gt_paths = gt_paths_t0frame(ego, t0s, ts_rel)
    dk = lm.distance_keeping(gt_paths, blk["leads"], blk["lead_lens"], speeds, float(dt))
    rows = {
        "clip_id": np.full(n, clip_id, dtype="<U36"),
        "frame": np.arange(n, dtype=np.int64),
        "t0_s": t0s,
        "eid": np.full(n, clip_id, dtype="<U36"),
        "leads": np.asarray(blk["leads"], float),
        "lead_lens": np.asarray(blk["lead_lens"], float),
        "speeds": speeds,
        "state": state,
        "gap0_m": np.asarray(blk["gap0_m"], float),
        "has_lead": lead,
        "lead_track_id": track_id.astype("<U64"),
        "rel_speed_mps": rel_v,
        "headway_time_s": hw_t,
        "gt_headway_min_m": np.asarray(dk["headway_min_m"], float),
        "gt_time_gap_min_s": np.asarray(dk["time_gap_min_s"], float),
        "gt_min_ttc_s": np.asarray(dk["min_ttc_s"], float),
        "gt_n_steps_in_corridor": np.asarray(dk["n_steps_in_corridor"], np.int64),
        "straight_dep_m": np.asarray(blk["straight_dep_m"], float),
        "straight_checked_m": np.asarray(blk["straight_checked_m"], float),
    }
    if shift is not None:
        rows["t0_pose_s"] = t0_pose
        rows["pose_sync_shift_s"] = shift
        rows["speeds_egomotion"] = speeds_ego
    even = np.arange(n) % 2 == 0                      # the 0.2 s grid (frame 2t)
    counts = {s: int((state == s).sum())
              for s in (ls.LEAD, ls.NO_LEAD, ls.NOT_STRAIGHT, ls.NO_LABEL)}
    counts_even = {s: int((state[even] == s).sum())
                   for s in (ls.LEAD, ls.NO_LEAD, ls.NOT_STRAIGHT, ls.NO_LABEL)}
    n_lab = counts[ls.LEAD] + counts[ls.NO_LEAD]
    cov = {
        "clip_id": clip_id, "n_frames": int(n),
        "t0_first_s": float(t0s[0]) if n else None,
        "t0_last_s": float(t0s[-1]) if n else None,
        "obstacle_offline": obs_in is not None,
        "obstacle_reason": obs_reason,
        "label_span_s": blk["label_span_s"],
        "counts": counts, "counts_even_frames": counts_even,
        "n_labelled_straight": n_lab,
        # free flow = NO_LEAD among windows where a lead COULD have been seen
        # (labelled AND straight). NO_LABEL / NOT_STRAIGHT are outside the
        # denominator on purpose — counting them would manufacture free flow.
        "free_flow_share": (round(counts[ls.NO_LEAD] / n_lab, 4) if n_lab else None),
        "n_tracks_selected": int(len(set(track_id[lead].tolist()))),
        "gt_dk": {kk: dk.get(kk) for kk in ("n", "mean_headway_min_m",
                                              "mean_time_gap_min_s", "n_time_gap",
                                              "mean_min_ttc_s", "n_closing", "status")},
    }
    if psync is not None:
        cov["pose_sync"] = psync
    return rows, cov


def write_block(path: str, rows_list: list[dict], coverage: dict, meta: dict,
                *, k: int, dt: float) -> dict:
    """Concatenate per-clip rows and ``np.savez_compressed`` the container."""
    if not rows_list:
        raise ValueError("no rows to write")
    keys = list(rows_list[0])
    cat = {kk: np.concatenate([r[kk] for r in rows_list]) for kk in keys}
    cat["state"] = cat["state"].astype(STATE_DTYPE)
    cat["ts_rel_s"] = np.arange(1, int(k) + 1, dtype=np.float64) * float(dt)
    cat["dt_s"] = np.array([float(dt)], dtype=np.float64)
    tot = {s: int((cat["state"] == s).sum())
           for s in (ls.LEAD, ls.NO_LEAD, ls.NOT_STRAIGHT, ls.NO_LABEL)}
    meta = dict(meta)
    meta["counts"] = tot
    meta["n_rows"] = int(cat["state"].size)
    meta["n_clips"] = int(len(rows_list))
    meta["k"], meta["dt_s"] = int(k), float(dt)
    cat["coverage_json"] = np.frombuffer(
        json.dumps(coverage, default=str).encode("utf-8"), dtype=np.uint8).copy()
    cat["meta_json"] = np.frombuffer(
        json.dumps(meta, default=str).encode("utf-8"), dtype=np.uint8).copy()
    # numpy appends ".npz" to a name without it, so the temp name keeps the suffix
    tmp = (path[:-4] if path.endswith(".npz") else path) + ".part.npz"
    np.savez_compressed(tmp, **cat)
    os.replace(tmp, path)
    return {"counts": tot, "n_rows": meta["n_rows"], "n_clips": meta["n_clips"]}


def read_block_json(blk: dict, key: str) -> dict:
    """Decode the ``coverage_json`` / ``meta_json`` uint8 fields of a loaded block."""
    v = blk.get(key)
    if v is None:
        return {}
    return json.loads(np.asarray(v, dtype=np.uint8).tobytes().decode("utf-8"))


# --------------------------------------------------------------------------- #
# the HF range pull (opt-in; token read in place, never printed)               #
# --------------------------------------------------------------------------- #
class _HTTPRangeFile(io.RawIOBase):
    """Seekable file over an HTTP resource honouring Range — the corpus release's
    ``tools/pull_egomotion_range.py`` recipe (central directory from the final
    256 KB; the redirect resolved ONCE)."""
    TAIL = 262144

    def __init__(self, url: str, auth: dict, retries: int = 4):
        import urllib.request as U
        self._U, self.auth, self.retries = U, auth, retries
        self.url = self._resolve(url)
        self.size = self._length()
        self.pos = 0
        self._tail_off = max(0, self.size - self.TAIL)
        self._tail = self._fetch(self._tail_off, self.size - 1)

    def _resolve(self, url):
        U = self._U
        for i in range(self.retries):
            try:
                with U.urlopen(U.Request(url, headers={**self.auth, "Range": "bytes=0-0"}),
                               timeout=120) as r:
                    return r.url
            except Exception:                                   # noqa: BLE001
                time.sleep(1.5 * (i + 1))
        return url

    def _length(self):
        U = self._U
        with U.urlopen(U.Request(self.url, headers={**self.auth, "Range": "bytes=0-0"}),
                       timeout=120) as r:
            cr = r.headers.get("Content-Range")
            return (int(cr.rsplit("/", 1)[1]) if cr and "/" in cr
                    else int(r.headers.get("Content-Length", 0)))

    def _fetch(self, a, b):
        U = self._U
        last = None
        for i in range(self.retries):
            try:
                with U.urlopen(U.Request(self.url, headers={**self.auth,
                                                            "Range": f"bytes={a}-{b}"}),
                               timeout=180) as r:
                    return r.read()
            except Exception as e:                              # noqa: BLE001
                last = e
                time.sleep(1.5 * (i + 1))
        raise last

    def readable(self): return True          # noqa: E704
    def seekable(self): return True          # noqa: E704
    def tell(self): return self.pos          # noqa: E704

    def seek(self, off, whence=0):
        self.pos = (off if whence == 0 else self.pos + off if whence == 1
                    else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size)
        if self.pos >= self._tail_off:
            s = self.pos - self._tail_off
            out = self._tail[s:s + (end - self.pos)]
        else:
            out = self._fetch(self.pos, end - 1)
        self.pos += len(out)
        return out


def _hf_token(keys_path: str) -> str:
    for _ in range(6):
        try:
            with open(keys_path, encoding="utf-8", errors="ignore") as fh:
                m = re.search(r"hf_[A-Za-z0-9]+", fh.read())
            if not m:
                raise SystemExit(f"no hf_ token in {keys_path}")
            return m.group(0)
        except OSError:
            time.sleep(3)
    raise SystemExit(f"could not read {keys_path}")


def pull_obstacle_offline(clips, chunk_of: dict, out_dir: str, keys_path: str,
                          workers: int = 8) -> dict:
    """Fetch ``{clip}.obstacle.offline.parquet`` for every clip not yet in
    ``out_dir`` by HTTP range reads of its chunk zip. Returns the manifest
    (per-clip sha256 / bytes / rows / span; every miss named, never silent)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pandas as pd
    try:
        import truststore
        truststore.inject_into_ssl()             # the dev box's TLS proxy
    except ImportError:                          # pragma: no cover
        pass
    os.makedirs(out_dir, exist_ok=True)
    auth = {"Authorization": f"Bearer {_hf_token(keys_path)}"}
    by_chunk: dict[int, list[str]] = {}
    no_chunk = []
    for c in clips:
        if os.path.exists(os.path.join(out_dir, f"{c}.parquet")):
            continue
        if c not in chunk_of:
            no_chunk.append(c)
            continue
        by_chunk.setdefault(int(chunk_of[c]), []).append(c)
    lock = threading.Lock()
    results, errors = {}, []
    t0 = time.time()

    def one(chunk):
        name = f"obstacle.offline.chunk_{chunk:04d}.zip"
        got = []
        try:
            f = _HTTPRangeFile(HF_OBS_BASE + name, auth)
            with zipfile.ZipFile(f) as z:
                names = z.namelist()
                for cid in by_chunk[chunk]:
                    member = next((m for m in names
                                   if m.rsplit("/", 1)[-1].startswith(cid)), None)
                    if member is None:
                        with lock:
                            errors.append({"chunk": chunk, "clip": cid,
                                           "error": "member not in chunk"})
                        continue
                    data = z.read(member)
                    df = pd.read_parquet(io.BytesIO(data))
                    miss = [c for c in OBS_NEED_COLS if c not in df.columns]
                    if miss:
                        with lock:
                            errors.append({"chunk": chunk, "clip": cid,
                                           "error": f"schema missing {miss}"})
                        continue
                    p = os.path.join(out_dir, f"{cid}.parquet")
                    with open(p + ".part", "wb") as fh:
                        fh.write(data)
                    os.replace(p + ".part", p)
                    got.append({"clip_id": cid, "chunk": chunk, "zip": name,
                                "member": member, "bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "n_rows": int(len(df)),
                                "n_tracks": int(df["track_id"].nunique()),
                                "t_min_s": float(df["timestamp_us"].min() / 1e6),
                                "t_max_s": float(df["timestamp_us"].max() / 1e6)})
        except Exception as e:                                  # noqa: BLE001
            with lock:
                errors.append({"chunk": chunk, "clip": None,
                               "error": f"{type(e).__name__}: {str(e)[:160]}"})
        return got

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, c) for c in sorted(by_chunk)]
        for i, fu in enumerate(as_completed(futs), 1):
            for r in fu.result():
                results[r["clip_id"]] = r
            if i % 20 == 0 or i == len(futs):
                _p(f"  [pull {i}/{len(futs)} chunks] clips={len(results)} "
                   f"errs={len(errors)} {time.time() - t0:.0f}s")
    return {"n_requested": len(list(clips)), "n_pulled_this_run": len(results),
            "clips_without_chunk_mapping": no_chunk, "errors": errors,
            "seconds": round(time.time() - t0, 1), "source": HF_OBS_BASE,
            "clips": results}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def read_clip_list(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        txt = fh.read()
    if path.endswith(".json"):
        j = json.loads(txt)
        if isinstance(j, dict) and isinstance(j.get("clips"), dict):
            return sorted(j["clips"])
        if isinstance(j, list):
            return sorted(str(x) for x in j)
        raise SystemExit(f"{path}: expected a clip_index json ('clips' dict) or a list")
    return sorted(l.strip().split()[0] for l in txt.splitlines()
                  if l.strip() and not l.startswith("#"))


def _tar_member(tf: tarfile.TarFile, names: dict, key: str):
    m = names.get(key)
    return None if m is None else tf.extractfile(m).read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", required=True, help="clip_index_eval.json or a text list")
    ap.add_argument("--ego-tar", required=True, help="egomotion tar ({clip}.parquet)")
    ap.add_argument("--ts-tar", required=True,
                    help="camera timestamps tar ({clip}.timestamps.parquet)")
    ap.add_argument("--obs-dir", required=True,
                    help="dir of {clip}.parquet obstacle.offline files (see --pull)")
    ap.add_argument("--pull", action="store_true",
                    help="range-pull missing obstacle.offline parquets from HF")
    ap.add_argument("--chunk-map", default=None,
                    help="clip_to_chunk.parquet (index: clip_id; column: chunk)")
    ap.add_argument("--keys", default=os.path.join(_REPO, "Keys.txt"))
    ap.add_argument("--v2ep-dir", default=None,
                    help="dir of {clip}.v2ep.pt to CROSS-CHECK the reconstructed "
                         "poses (refuses the build on a failure)")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument("--pose-sync-sidecar", default=None,
                    help="build every row on the run's POSE-SYNC clock (module doc): the sidecar "
                         "the run's config.json pose_sync.eval names")
    ap.add_argument("--pose-sync-sign", type=float, default=None,
                    help="REQUIRED with --pose-sync-sidecar: the run's config.json pose_sync.eval.sign")
    ap.add_argument("--pose-sync-raw-offset", type=int, default=None,
                    help="REQUIRED with --pose-sync-sidecar: the eval cache's n_stack - 1 (the dump "
                         "manifest's corpus.frames.provider_to_raw_frame_offset)")
    ap.add_argument("--out", required=True, help="output .npz FILE")
    ap.add_argument("--report", default=None, help="json report (default <out>.report.json)")
    a = ap.parse_args(argv)
    if os.path.isdir(a.out):
        sys.exit(f"--out must be a FILE, got a directory: {a.out}")
    ps = None
    if a.pose_sync_sidecar:
        if a.pose_sync_sign is None or a.pose_sync_raw_offset is None:
            sys.exit("--pose-sync-sidecar needs --pose-sync-sign AND --pose-sync-raw-offset (no defaults: "
                     "they are the run's own values)")
        if a.pose_sync_raw_offset < 0:
            sys.exit("--pose-sync-raw-offset must be >= 0")
        ps = _pose_sync_mod().read_pose_sync_sidecar(a.pose_sync_sidecar, sign=a.pose_sync_sign)
    elif a.pose_sync_sign is not None or a.pose_sync_raw_offset is not None:
        sys.exit("--pose-sync-sign / --pose-sync-raw-offset without --pose-sync-sidecar")
    import pandas as pd

    t_start = time.time()
    clips = read_clip_list(a.clips)
    _p(f"[clips] {len(clips)} from {a.clips}")
    pull_man = None
    if a.pull:
        if not a.chunk_map:
            sys.exit("--pull needs --chunk-map")
        cm = pd.read_parquet(a.chunk_map)
        chunk_of = {str(i): int(r["chunk"]) for i, r in cm.iterrows()}
        pull_man = pull_obstacle_offline(clips, chunk_of, a.obs_dir, a.keys)
        _p(f"[pull] {pull_man['n_pulled_this_run']} pulled this run, "
           f"{len(pull_man['errors'])} errors, {pull_man['seconds']} s")
        with open(os.path.join(a.obs_dir, "pull_manifest.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(pull_man, fh, indent=1)

    ego_tf = tarfile.open(a.ego_tar)
    ego_names = {m.name.rsplit("/", 1)[-1]: m for m in ego_tf.getmembers()}
    ts_tf = tarfile.open(a.ts_tar)
    ts_names = {m.name.rsplit("/", 1)[-1]: m for m in ts_tf.getmembers()}
    _p(f"[tars] egomotion {len(ego_names)} members · timestamps {len(ts_names)} members")

    rows_list, coverage = [], {}
    n_no_obs, n_no_ego, n_no_ts, n_xfail = 0, 0, 0, 0
    obs_sha = {}
    ps_uncovered, ps_dt_gap = [], 0.0
    for j, cid in enumerate(clips):
        ego_b = _tar_member(ego_tf, ego_names, f"{cid}.parquet")
        ts_b = _tar_member(ts_tf, ts_names, f"{cid}.timestamps.parquet")
        if ego_b is None or ts_b is None:
            n_no_ego += int(ego_b is None)
            n_no_ts += int(ts_b is None)
            coverage[cid] = {"clip_id": cid, "status": "NO_EGO_OR_TIMESTAMPS",
                             "egomotion": ego_b is not None,
                             "timestamps": ts_b is not None, "n_frames": 0}
            _p(f"  [{j + 1}/{len(clips)}] {cid[:8]} REFUSED: egomotion={ego_b is not None} "
               f"timestamps={ts_b is not None}")
            continue
        ego_df = pd.read_parquet(io.BytesIO(ego_b))
        ts_df = pd.read_parquet(io.BytesIO(ts_b))
        tcol = next(c for c in ts_df.columns if "time" in c.lower())
        t_query, unit, n_target = episode_grid(ts_df[tcol].to_numpy(np.float64))
        poses = episode_poses(ego_df, t_query)
        t_grid_s = t_query / unit
        ego = ego_track(ego_df)
        xc = None
        if a.v2ep_dir:
            vp = os.path.join(a.v2ep_dir, f"{cid}.v2ep.pt")
            if os.path.exists(vp):
                xc = crosscheck_v2ep(vp, poses)
                if not xc["pass"]:
                    n_xfail += 1
                    _p(f"  [{j + 1}/{len(clips)}] {cid[:8]} ⛔ v2ep CROSS-CHECK FAILED: {xc}")
        op = os.path.join(a.obs_dir, f"{cid}.parquet")
        obs, why = None, None
        if os.path.exists(op):
            obs_df = pd.read_parquet(op)
            obs = obs_track(obs_df)
            obs_sha[cid] = _sha256(op)
            if obs is None:
                why = "obstacle.offline parquet present but EMPTY"
        else:
            why = ("no obstacle.offline parquet — the clip has no member in its "
                   "chunk (range pull) or was never pulled; NO_LABEL, never free flow")
        if obs is None:
            n_no_obs += 1
        skew = obs.pop("frame_skew_s", 0.0) if obs else None
        shift_s = v0s = None
        ps_clip = None
        poses_rows = poses
        if ps is not None:
            from tanitad.data.v2_dataset import stable_episode_id
            shift_s, v0s, ps_clip = pose_sync_clip(ps, int(stable_episode_id(cid)), poses, t_grid_s,
                                                   a.pose_sync_raw_offset)
            if not ps_clip["covered"]:
                ps_uncovered.append(cid)
            else:
                ps_dt_gap = max(ps_dt_gap, abs(ps_clip["dt_sidecar_s"] - ps_clip["dt_grid_s"]))
            poses_rows = episode_poses(ego_df, t_query + shift_s * unit)
        rows, cov = build_clip_rows(cid, ego, t_grid_s, obs, k=a.k, dt=a.dt,
                                    poses=poses_rows, obs_reason=why,
                                    t0_shift_s=shift_s, v0_sync=v0s)
        if ps_clip is not None:
            cov.setdefault("pose_sync", {}).update(ps_clip)
        et = np.asarray(ego["t"], float)
        cov.update({
            "status": "OK",
            "grid": {"a_s": float(t_grid_s[0]), "b_s": float(np.diff(t_grid_s).mean()),
                     "n_target": int(n_target), "n_cam_frames": int(len(ts_df)),
                     "unit": unit,
                     "formula": "linspace(t_cam[0], t_cam[-1], int(span_s*TARGET_HZ)) "
                                "— v2_compressed._resampled / physicalai.build_episode"},
            "egomotion": {"n": int(et.size), "t_min_s": float(et.min()),
                          "t_max_s": float(et.max()),
                          "hz_median": float(1.0 / np.median(np.diff(et)))},
            "obstacle": None if obs is None else {
                "n_rows": int(np.asarray(obs["t"]).size),
                "n_tracks": int(len(set(np.asarray(obs["track"]).tolist()))),
                "n_vehicle_rows": int(np.asarray(obs["is_vehicle"]).sum()),
                "t_min_s": float(np.min(obs["t"])), "t_max_s": float(np.max(obs["t"])),
                "reference_frame_skew_s": skew, "sha256": obs_sha.get(cid)},
            "v2ep_crosscheck": xc,
        })
        coverage[cid] = cov
        rows_list.append(rows)
        c = cov["counts"]
        _p(f"  [{j + 1}/{len(clips)}] {cid[:8]} n={cov['n_frames']} LEAD {c[ls.LEAD]:3d} "
           f"NO_LEAD {c[ls.NO_LEAD]:3d} NOT_STRAIGHT {c[ls.NOT_STRAIGHT]:3d} "
           f"NO_LABEL {c[ls.NO_LABEL]:3d}"
           + ("" if xc is None else f" | v2ep xcheck dxy {xc['max_dxy_m']:.5f} m "
                                    f"dyaw {xc['max_dyaw_deg']:.5f}deg")
           + ("" if obs is not None else " | NO obstacle.offline"))
    if n_xfail:
        sys.exit(f"⛔ {n_xfail} clip(s) FAILED the v2ep cross-check — the reconstructed "
                 f"time base does not reproduce the banked poses; refusing to write a "
                 f"block that would put every lead at the wrong time")
    if not rows_list:
        sys.exit("no clip produced rows")

    meta = {
        "version": VERSION, "tool": "taniteval/tools/build_lead_block_b1.py",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"clips": {"path": a.clips, "sha256": _sha256(a.clips), "n": len(clips)},
                   "ego_tar": {"path": a.ego_tar, "sha256": _sha256(a.ego_tar)},
                   "ts_tar": {"path": a.ts_tar, "sha256": _sha256(a.ts_tar)},
                   "obs_dir": a.obs_dir, "obs_sha256": obs_sha,
                   "v2ep_dir": a.v2ep_dir, "pull": pull_man},
        "frame_index_space": ("RAW v2ep 10 Hz index i in [0, n_target): the refav1 "
                              "loader's timeline (cache step t <-> frame 2t); NOT the "
                              "post-n_stack-trim provider index"),
        "time_base": ("clip seconds; t0_s = linspace(t_cam[0], t_cam[-1], n_target)/unit "
                      "(the builder's own formula), poses = physicalai.signals_at"),
        "horizon": {"k": int(a.k), "dt_s": float(a.dt),
                    "ts_rel_s": (np.arange(1, a.k + 1) * a.dt).round(6).tolist()},
        "conventions": ls.lead_block(np.zeros(1), np.array([a.dt]), None,
                                     {"t": np.array([0.0, 1.0]), "x": np.zeros(2),
                                      "y": np.zeros(2), "yaw": np.zeros(2),
                                      "v": np.zeros(2)})["conventions"],
        "states": {"LEAD": "causal in-corridor vehicle ahead at t0, future track present",
                   "NO_LEAD": "labels present, straight, road clear within 80 m",
                   "NOT_STRAIGHT": "ego's own future path leaves the corridor within "
                                   "the claimed range — its own state, never NO_LEAD",
                   "NO_LABEL": "no obstacle.offline for the clip / horizon leaves the "
                               "labelled span — NEVER free flow"},
        "refusals": {"clips_no_ego": n_no_ego, "clips_no_timestamps": n_no_ts,
                     "clips_without_obstacle_offline": n_no_obs},
        "pose_sync": None if ps is None else {
            "sidecar": a.pose_sync_sidecar, "sha256": _sha256(a.pose_sync_sidecar),
            "sign": float(a.pose_sync_sign), "raw_offset": int(a.pose_sync_raw_offset),
            "n_clips_covered": int(len(rows_list) - len(ps_uncovered)),
            "clips_uncovered_built_unshifted": ps_uncovered,
            "max_abs_dt_sidecar_minus_dt_grid_s": ps_dt_gap,
            "rule": ("row i is built at t_pose[i] + s_i * dt_grid, s_i = PoseSync.shift_for(sid, i) (the "
                     "trainer's row shift); speeds = the dataset's v0 via pose_sync.window_shifted_tracks "
                     "(V3Dataset._pose_sync_apply), which must equal the grid-row egomotion speeds re-sampled "
                     f"at row i + s_i within {PSYNC_V0_TOL_MPS} m/s or the build refuses; speeds_egomotion = "
                     "egomotion at the shifted instant (10 Hz-vs-100 Hz interpolation error, reported per clip)"),
        },
        "wallclock_s": round(time.time() - t_start, 1),
    }
    if ps is not None:
        meta["time_base"] += ("; ⭐ POSE-SYNC: t0_s is the SHIFTED instant (t0_pose_s + pose_sync_shift_s) "
                              "and speeds is the dataset's pose-synced v0 (meta.pose_sync)")
    summary = write_block(a.out, rows_list, coverage, meta, k=a.k, dt=a.dt)
    rep_path = a.report or (a.out + ".report.json")
    n_clips_lead = sum(1 for c in coverage.values() if c.get("counts", {}).get(ls.LEAD, 0))
    report = {"out": a.out, "sha256": _sha256(a.out), **summary,
              "n_clips_requested": len(clips), "n_clips_with_any_lead": n_clips_lead,
              "n_clips_without_obstacle_offline": n_no_obs,
              "refusals": meta["refusals"], "coverage": coverage, "meta": meta}
    with open(rep_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    tot = summary["counts"]
    n_rows = summary["n_rows"]
    _p(f"\n[out] {a.out}  sha256={report['sha256'][:16]}…  rows {n_rows} clips "
       f"{summary['n_clips']}/{len(clips)}")
    for s, v in tot.items():
        _p(f"  {s:12s} {v:6d}  ({100.0 * v / max(n_rows, 1):.2f} %)")
    _p(f"  clips with any LEAD frame: {n_clips_lead}  · without obstacle.offline: "
       f"{n_no_obs}  · v2ep cross-checked: "
       f"{sum(1 for c in coverage.values() if c.get('v2ep_crosscheck'))}")
    if ps is not None:
        _p(f"  pose-sync clock: {meta['pose_sync']['n_clips_covered']} clip(s) shifted · "
           f"{len(ps_uncovered)} uncovered (built unshifted) · sign {a.pose_sync_sign} · raw offset "
           f"{a.pose_sync_raw_offset}")
    _p(f"[report] {rep_path}")
    _p("LEAD_BLOCK_B1_DONE")


if __name__ == "__main__":
    main()

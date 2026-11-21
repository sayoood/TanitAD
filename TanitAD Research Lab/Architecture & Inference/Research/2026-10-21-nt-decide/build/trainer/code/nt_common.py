#!/usr/bin/env python3
"""nt_common -- shared facts for the H-NT-DECIDE-1 navtrain fine-tune builders (E2 labels, E3 joins, E4 guard).

Research Lab (Architecture & Inference), 2026-10-21. TANITAD VENV, CPU only. Nothing here touches a GPU.

THE CONVENTIONS (each one load-bearing; each restated from source, never invented):

* ONE navtrain token = ONE v2ep "token-episode" (E1, DataFlyWheel) = ONE trainer window (``--window-starts first``).
  raw row ``k`` of the episode sits at ``t0 + (k - NOW_RAW_ROW) * DT_S`` with ``NOW_RAW_ROW = 9`` (= ``t + w - 1 +
  n_stack - 1`` at ``t = 0``, ``w = 8``, ``n_stack = 3``; ``v9_labels.py`` module docstring).
* THE EPISODE CLOCK is CLIP-LOCAL: ``grid_start_s = 0.0``, ``dt_s = 0.1`` (``clip_clock.read_clip_clock_sidecar``
  refuses ``|grid_start_s| > 1 s``, so a log-relative origin is not admissible). The window NOW is therefore
  ``now_ep = 0.0 + 9 * 0.1`` in the trainer's own float arithmetic (``refc_v3_train._now_s``: ``float(g0) + (r +
  raw_offset) * float(dt)``) -- :func:`episode_now_s` reproduces that expression literally.
* LABEL GEOMETRY is computed on the LOG timeline (seconds since the log's first 2 Hz frame), where the turn table
  and the ego future live; only the release's ``now_s`` column is written on the episode clock.
* clip ids: E1 owns them (``INTERFACE.md``). :func:`clip_id_provisional` is used ONLY until a manifest exists, and
  every output records which source named the clips.
* the NavSim nav token: ``driving_command`` one-hot at t0 -> ``NAVSIM_LEGAL_NAV_CMD`` (literal mirror of
  ``refcv8_train.py:72`` and ``refcv8_bridge.py:163``): left -> TURN_L 1, forward -> FOLLOW 0, right -> TURN_R 2,
  unknown -> FOLLOW 0; a non-one-hot vector is REFUSED.
* the road plane: the SAME constant the NavSim bridge feeds at inference (``run_bridge8.py:137-138`` reads
  ``summary.median`` of ``road_plane_navhard_warmup_logs.json``, md5 ``e37064e1fa11432e8b5500fb4c24c9ff``,
  median ``-0.3557311968663991`` m in the NavSim ego frame). Training must see the lift the bridge builds (K1);
  a per-log plane is measured as a DIAGNOSTIC only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------------------- literals
NOW_RAW_ROW = 9
N_STACK = 3
WINDOW = 8
DT_S = 0.1
GRID_START_S = 0.0
LOG_DT_S = 0.5
T_RAW_FULL = 70                        # 2 (stack) + 8 (window) + 60 (MAX_H_EXT) raw rows
T_OUT_MIN = WINDOW + 20 + 1            # build_window_index: T_out >= window + max_horizon(20) + 1
#: refcv8_train.py:72 / refcv8_bridge.py:163, LITERAL (a check derived from the value it checks is green forever)
NAVSIM_LEGAL_NAV_CMD = {"left": 1, "forward": 0, "right": 2, "unknown": 0}
NAVSIM_ONEHOT_KEY = {0: "left", 1: "forward", 2: "right", 3: "unknown"}
NAV_NAMES8 = ("FOLLOW", "TURN_L", "TURN_R")
#: v7 nav token per v9 token id (vocab_v7.NAV_COMMAND_TOKENS order = refb.NAV_COMMANDS 0..2)
V7_NAV_OF_V9 = {0: "NAV_FOLLOW_ROAD", 1: "NAV_TURN_L", 2: "NAV_TURN_R"}
#: the bridge's road plane (run_bridge8.py:137-138), recorded by md5 and value; re-read and checked when available
ROAD_PLANE_FILE_MD5 = "e37064e1fa11432e8b5500fb4c24c9ff"
ROAD_PLANE_MEDIAN_M = -0.3557311968663991
#: E3 class map (PREREG / DESIGN sec. 3, declared): OpenScene gt_names -> refc's 10-class enum (bev_raster.py:93-94)
CLASS_MAP = {"vehicle": "automobile", "pedestrian": "person", "bicycle": "rider"}
IGNORED_CLASSES = ("generic_object", "traffic_cone", "barrier", "czone_sign")


class NTRefused(SystemExit):
    """A refusal (the repo's SystemExit idiom: a run cannot catch it by accident)."""


# ----------------------------------------------------------------------------------------------- ids / hashing
def stable_episode_id(clip_id: str) -> int:
    """``tanitad/data/v2_dataset.py:69`` (blake2b-8 >> 1) -- restated, pinned equal by the test suite."""
    return int.from_bytes(hashlib.blake2b(clip_id.encode("utf-8"), digest_size=8).digest(), "big") >> 1


def sha12(clip_id: str) -> str:
    return hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest()[:12]


def clip_id_provisional(token: str) -> str:
    """PROVISIONAL clip id used only until E1's INTERFACE.md manifest exists (every output records the source)."""
    return f"navtrain-{token}"


def md5_file(p, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def sha256_file(p, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def track_int(track_token: str) -> int:
    """OpenScene ``track_tokens`` are 16-hex strings; the VIS-1 sidecar keys tracks as int64 and refuses a
    non-digit ``track_id`` (``vis1_block_for_rows``). The first 15 hex digits (60 bits) fit int64; a collision
    inside one frame is checked by the caller and REFUSED."""
    s = str(track_token)
    if not s or any(c not in "0123456789abcdefABCDEF" for c in s):
        raise NTRefused(f"[nt] track token {s!r} is not hex")
    return int(s[:15], 16)


def episode_now_s(g0: float = GRID_START_S, dt: float = DT_S, t: int = 0, w: int = WINDOW,
                  n_stack: int = N_STACK) -> float:
    """``refc_v3_train.V3Dataset._now_s`` literally: ``float(g0) + (r + raw_offset) * float(dt)`` with
    ``r = t + w - 1`` and ``raw_offset = n_stack - 1``."""
    r = int(t) + int(w) - 1
    return float(g0) + (r + max(int(n_stack) - 1, 0)) * float(dt)


# ----------------------------------------------------------------------------------------------- logs
def load_log(path) -> list:
    with open(path, "rb") as fh:
        fr = pickle.load(fh)
    if not isinstance(fr, list) or not fr:
        raise NTRefused(f"[nt] {path}: not a non-empty OpenScene log list")
    return fr


def log_name_of(frames: list, path=None) -> str:
    ln = str(frames[0].get("log_name"))
    if path is not None and Path(path).stem != ln:
        raise NTRefused(f"[nt] {path}: file stem != log_name {ln!r}")
    if any(str(f.get("log_name")) != ln for f in frames):
        raise NTRefused(f"[nt] {path}: mixed log_name within one log file")
    return ln


def yaw_of_e2g(m) -> float:
    m = np.asarray(m, dtype=np.float64)
    return float(math.atan2(m[1, 0], m[0, 0]))


def devkit_yaw(q) -> float:
    """Heading of a ``(w, x, y, z)`` ``ego2global_rotation`` EXACTLY as the NavSim v1.1 devkit reads it:
    ``pyquaternion.Quaternion(*q).yaw_pitch_roll[0]`` = ``atan2(2(wz - xy), 1 - 2(y^2 + z^2))`` on the normalised
    quaternion (pyquaternion's z-y'-x'' form, re-implemented here from its source formula). ⛔ NOT the textbook ZYX
    ``atan2(R10, R00)`` (``+xy``): MEASURED 2026-10-21 on the smoke logs, that form is ~1e-4 rad off and moved the
    bridge history 0.1-2.4 mm off the devkit export (K1); E1 found the same on the future (its v1.2)."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n > 0 and abs(n - 1.0) > 1e-14:
        w, x, y, z = w / n, x / n, y / n, z / n
    return math.atan2(2.0 * (w * z - x * y), 1.0 - 2.0 * (y * y + z * z))


def ego_arrays(frames: list) -> dict:
    """Per 2 Hz frame: ``t_s`` (seconds since the log's FIRST frame), global ``x, y`` (minus the first frame's,
    for conditioning only; ``ego2global_translation``, as the devkit), unwrapped ``yaw`` (:func:`devkit_yaw` of
    ``ego2global_rotation`` -- the devkit's convention, so labels and NavSim targets share ONE frame), ``v = hypot(vx, vy)`` of
    ``ego_dynamic_state[:2]`` (the EgoStatus velocity -- ``refcv6_bridge.speed_of``), ego-frame ``vx``, and
    ``rev`` = moving backwards (> 0.3 m/s with ego-frame vx < 0; ``build_v9_labels.load_log``'s 0.3 m/s rule)."""
    ts = np.array([int(f["timestamp"]) for f in frames], dtype=np.int64)
    if np.any(np.diff(ts) <= 0):
        raise NTRefused("[nt] log timestamps are not strictly increasing")
    xy = np.stack([np.asarray(f["ego2global_translation"], dtype=np.float64)[:2] for f in frames])
    x, y = xy[:, 0], xy[:, 1]
    yaw = np.unwrap(np.array([devkit_yaw(f["ego2global_rotation"]) for f in frames], dtype=np.float64))
    eds = np.asarray([list(f["ego_dynamic_state"])[:2] for f in frames], dtype=np.float64)
    v = np.hypot(eds[:, 0], eds[:, 1])
    rev = ((v > 0.3) & (eds[:, 0] < 0.0)).astype(np.float64)
    dc = np.asarray([np.asarray(f["driving_command"], dtype=np.float64) for f in frames])
    return {"ts_us": ts, "t_s": (ts - ts[0]).astype(np.float64) / 1e6, "x_g": x, "y_g": y,
            "x": x - x[0], "y": y - y[0], "yaw": yaw, "v": v, "vx_ego": eds[:, 0], "rev": rev,
            "driving_command": dc, "tokens": [str(f["token"]) for f in frames],
            "log_t0_us": int(ts[0])}


def nav_token_of_driving_command(dc) -> dict:
    """The NavSim t0 command -> the v9 per-frame token, exactly ``refcv8_bridge.nav_input8`` (REFUSES a
    non-one-hot vector, never arg-maxes)."""
    c = np.asarray(dc, dtype=np.float64).reshape(-1)
    if c.shape != (4,) or not np.all(np.isin(c, (0.0, 1.0))) or c.sum() != 1.0:
        raise NTRefused(f"[nt] driving_command is not a 4-dim one-hot: {c.tolist()}")
    k = int(np.argmax(c))
    key = NAVSIM_ONEHOT_KEY[k]
    tok = int(NAVSIM_LEGAL_NAV_CMD[key])
    return {"nav_token": tok, "name": NAV_NAMES8[tok], "navsim_argmax": k, "navsim_key": key}


def future_ticks_available(ea: dict, idx: int) -> int:
    """10 Hz ticks of logged future after the token (a 2 Hz gap > 1.0 s ends the usable future)."""
    t = ea["t_s"]
    j = idx
    while j + 1 < len(t) and (t[j + 1] - t[j]) <= 1.0 + 1e-6:
        j += 1
    return int(math.floor((t[j] - t[idx]) / DT_S + 1e-6))


def t_raw_for(ea: dict, idx: int) -> int:
    """The token-episode length E1 is asked to write (INTERFACE_REQUESTS R1): 10 + min(60, future ticks)."""
    return int(NOW_RAW_ROW + 1 + min(60, future_ticks_available(ea, idx)))


# ----------------------------------------------------------------------------------------------- token table
def read_navtrain_filter(path) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    return {"tokens": set(d["tokens_sorted"]), "logs": set(d["log_names_sorted"]),
            "n_tokens": int(d["n_tokens"]), "n_logs": int(d["n_logs"]),
            "file_sha256": d.get("file_sha256"), "source": d.get("source")}


def read_e1_manifest(path) -> dict:
    """E1's token table (INTERFACE.md): token -> row dict with at least ``clip_id``, ``log_name``.

    Accepts JSON (a list of rows, or ``{"rows": [...]}`` / ``{"tokens": {token: row}}``) or JSON-lines.
    ⛔ REFUSES a duplicate token or a duplicate clip id."""
    p = str(path)
    rows = []
    if p.endswith(".jsonl") or p.endswith(".jsonl.gz"):
        import gzip
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    else:
        d = json.load(open(p, encoding="utf-8"))
        if isinstance(d, list):
            rows = d
        elif "rows" in d:
            rows = d["rows"]
        elif "tokens" in d and isinstance(d["tokens"], dict):
            rows = [dict(v, token=k) for k, v in d["tokens"].items()]
        else:
            raise NTRefused(f"[nt] {p}: unrecognised E1 manifest shape")
    out, cids = {}, set()
    for r in rows:
        tok = str(r["token"])
        cid = str(r["clip_id"])
        if tok in out:
            raise NTRefused(f"[nt] {p}: token {tok} twice")
        if cid in cids:
            raise NTRefused(f"[nt] {p}: clip_id {cid} twice")
        cids.add(cid)
        out[tok] = r
    return out


def build_token_table(log_paths, navtrain: dict, heldout_logs: set, clip_ids: dict | None = None) -> dict:
    """Every navtrain token of the given logs that is NOT on a held-out log.

    Returns {"rows": [...], "census": {...}}; each row: token, log_name, frame_idx, clip_id, clip_id_source, sid,
    t0_us, log_t0_us, t0_log_s, T_raw, future_ticks, driving_command."""
    rows, n_held, n_not_nav, n_missing_cid = [], 0, 0, 0
    for lp in log_paths:
        fr = load_log(lp)
        ln = log_name_of(fr, lp)
        if ln in heldout_logs:
            n_held += sum(1 for f in fr if str(f["token"]) in navtrain["tokens"])
            continue
        ea = ego_arrays(fr)
        for i, f in enumerate(fr):
            tok = str(f["token"])
            if tok not in navtrain["tokens"]:
                n_not_nav += 1
                continue
            if clip_ids is not None:
                if tok not in clip_ids:
                    n_missing_cid += 1
                    continue
                cid, src = str(clip_ids[tok]["clip_id"]), "e1_manifest"
            else:
                cid, src = clip_id_provisional(tok), "PROVISIONAL(nt_common.clip_id_provisional)"
            rows.append({"token": tok, "log_name": ln, "frame_idx": i, "clip_id": cid, "clip_id_source": src,
                         "sid": stable_episode_id(cid), "t0_us": int(ea["ts_us"][i]), "log_t0_us": ea["log_t0_us"],
                         "t0_log_s": float(ea["t_s"][i]), "T_raw": t_raw_for(ea, i),
                         "future_ticks": future_ticks_available(ea, i),
                         "driving_command": [int(x) for x in ea["driving_command"][i]]})
    sids = [r["sid"] for r in rows]
    if len(set(sids)) != len(sids):
        raise NTRefused("[nt] stable_episode_id collision inside the token table")
    return {"rows": rows, "census": {"n_rows": len(rows), "n_heldout_navtrain_tokens_skipped": n_held,
                                      "n_non_navtrain_frames": n_not_nav, "n_missing_clip_id": n_missing_cid,
                                      "n_short_future": sum(1 for r in rows if r["T_raw"] < T_RAW_FULL),
                                      "n_untrainable_T_out_lt_29": sum(1 for r in rows
                                                                       if r["T_raw"] - (N_STACK - 1) < T_OUT_MIN)}}


def json_dump(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


def tree_stack(tree: str | None = None) -> str:
    """The clean launch tree's ``stack`` (sys.path'd); refuses an import from G: (the editable install)."""
    t = Path(tree or os.environ.get("NT_TREE") or (HERE.parent / "tree_fb47a58"))
    st = str(t / "stack")
    for p in (st, str(t / "stack" / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return st


def assert_not_g(mod) -> None:
    f = str(getattr(mod, "__file__", "") or "")
    if f.upper().startswith("G:"):
        raise NTRefused(f"[nt] {mod.__name__} imported from G: ({f}); set PYTHONPATH to a clone's stack")

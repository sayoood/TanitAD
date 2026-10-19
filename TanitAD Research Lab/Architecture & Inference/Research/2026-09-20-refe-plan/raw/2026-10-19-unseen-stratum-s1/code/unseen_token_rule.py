"""UNSEEN-STRATUM S1: a DB-only rule that names the evaluation TOKENS of a nuPlan log the way NAVSIM's navtrain did.

FROZEN by the sha256 recorded in raw/rule_freeze.json (S1 RESULT.md).  CPU only; sqlite is opened READ-ONLY; no label,
no PDM score, no model output is read.  The rule reads exactly four things from a log DB: lidar_pc (token, timestamp,
scene_token), CAM_F0 and CAM_L0 image timestamps, scene.roadblock_ids (emptiness only), and -- for the fetch list only -- the
four cameras' image rows (timestamp, filename_jpg).

THE RULE (every clause quotes its source; devkit = C:/Users/Admin/navsim-crun/devkit, the NAVSIM checkout on this box)

  C1  CADENCE.  A navtrain "frame" is an OpenScene frame, 2 Hz, = every 10th nuPlan lidar_pc (20 Hz, dt 0.050 s MEASURED).
      PHASE: the grid starts at the first lidar_pc that has a synchronised image on BOTH CAM_F0 and CAM_L0 (index = the larger of the two
      first-paired indices), and the FIRST grid frame is not in the frame list (the list starts one grid step later).
        source: devkit/docs/cache.md:3 ("... sensor data at 2Hz"); the stride, the phase and the dropped first frame are MEASURED
                (raw/identity_*.json), not read from a source file -- the OpenScene metadata builder is not on this box.  The identity
                check is what licenses them, and two of the three were corrected on the 214 logs; the 248 top-level navtrain logs are the
                out-of-sample test (raw/identity_toplevel248.json).
  C2  WINDOW.   A scene is a window of num_history_frames + num_future_frames = 4 + 10 = 14 consecutive frames, stride
      frame_interval = 1; its TOKEN is the token of frame index num_history_frames - 1 = 3.  A window shorter than 14 is dropped.
        source: devkit/navsim/common/dataloader.py:24-26 (split_list: `input_list[i : i + num_frames] for i in range(0, len, frame_interval)`),
                :47 (the call), :49-50 (`if len(frame_list) < scene_filter.num_frames: continue`), :57 (`token = frame_list[scene_filter.num_history_frames - 1]["token"]`);
                config  TanitAD Research Lab/.../2026-09-20-refe-plan/splits/navtrain.yaml:3-6  (num_history_frames: 4, num_future_frames: 10,
                frame_interval: 1, has_route: true) -- byte-identical to devkit/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml:3-6.
        => with M grid frames in the log (before the dropped first one), the candidate grid positions are k = 4 .. M-11: 3 frames of history (1.5 s) and
           10 frames of future (5.0 s) INSIDE the post-drop list.
  C3  ROUTE.    A window is dropped when the CURRENT frame (index 3) carries no route.  The nuPlan DB carries the route as scene.roadblock_ids
      (a space-separated roadblock-id string); a frame has a route iff its scene's roadblock_ids is non-empty.
        source: devkit/navsim/common/dataloader.py:52-54 (comment, then `if scene_filter.has_route and len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0: continue`).
  C4  NOT IN THE RULE.  navtrain's token LIST (the 103,288 tokens of navtrain.yaml `tokens:`) is a PDM-SCORE filter applied on top of C1-C3:
      "We remove highly simplistic scenes by detecting if the ... constant velocity agent exceeds a PDMS of 0.8 ... we remove scenes in which the human
      trajectory results in a PDMS of less than 0.8" (Dauner et al. 2024, NAVSIM, sec. 3.1; products/P7-TanitEval/benchmarks/NAVSIM_PROTOCOL.md:386-391).
      That needs a simulator score, i.e. a label; the DB cannot name it.  So the rule names a SUPERSET of navtrain.

  KNOWN LIMIT (C1 phase): 7 of the 248 out-of-sample logs (2.8 %, all one camera-timing signature of vehicle veh-26) have the true grid 2 lidar_pc later than this
      rule's phase; their candidates are VALID frames (images within 25 ms on F0 and L0) but not on the OpenScene grid.  No rule tried recovered them (raw/identity_*.json).

    candidates(log) -> the tokens of C1-C3.   Import and call; there is no command line.
"""
from __future__ import annotations

import sqlite3

import numpy as np

# ---- the frozen parameters (each one's source is in the module docstring) ------------------------------------------
NUM_HISTORY_FRAMES = 4      # navtrain.yaml:3
NUM_FUTURE_FRAMES = 10      # navtrain.yaml:4
FRAME_INTERVAL = 1          # navtrain.yaml:5
HAS_ROUTE = True            # navtrain.yaml:6
LIDAR_STRIDE = 10           # 20 Hz lidar_pc -> 2 Hz OpenScene frames (MEASURED identity)
PAIR_CAMS = ("CAM_F0", "CAM_L0")   # the cameras whose first paired lidar_pc defines the phase (max of the two; MEASURED 214/214 logs of the fit set, 241/248 out of sample; logs with recall >= 0.5: CAM_F0 alone 208/214 and 209/248, CAM_L0 alone 213/214 and 240/248; raw/identity_*.json red arms)
DROP_FIRST_FRAME = 1        # the OpenScene frame list starts one grid step after the first paired lidar_pc (MEASURED: 0 of 18,122 navtrain tokens at the first grid frame)
PAIR_TOL_US = 25_000        # a lidar_pc is "paired" on a camera when its nearest image of that camera is <= 25 ms away: the two populations sit at 12-14 ms and 36-38 ms (MEASURED, 40 DBs, 191,903 lidar_pc)
CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")   # the four cameras of Step A's bundle (refe/planner.py:54 CAMERAS; make_stepA_bundle.py:82 FrameResolver)
MAX_DT_MS = 120.0           # refe/planner.py:71 FrameResolver max_dt_ms


def _ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def read_log(path: str) -> dict:
    """The four DB reads the rule needs.  Arrays are ordered by lidar_pc timestamp."""
    c = _ro(path)
    try:
        lp = c.execute("SELECT lower(hex(token)), timestamp, lower(hex(scene_token)) FROM lidar_pc ORDER BY timestamp").fetchall()
        im = {ch: np.array([r[0] for r in c.execute("SELECT i.timestamp FROM image i JOIN camera c ON c.token = i.camera_token WHERE c.channel = ? ORDER BY i.timestamp", (ch,))], dtype=np.int64) for ch in PAIR_CAMS}
        sc = {r[0]: bool((r[1] or "").strip()) for r in c.execute("SELECT lower(hex(token)), roadblock_ids FROM scene")}
    finally:
        c.close()
    return {"tokens": [r[0] for r in lp], "ts": np.array([r[1] for r in lp], dtype=np.int64),
            "scene": [r[2] for r in lp], "img_ts": im, "scene_has_route": sc}


def paired_mask(ts: np.ndarray, img_ts: np.ndarray, tol_us: int = PAIR_TOL_US) -> np.ndarray:
    """lidar_pc i is paired iff its nearest image of that camera is within tol_us."""
    if len(img_ts) == 0 or len(ts) == 0:
        return np.zeros(len(ts), dtype=bool)
    j = np.searchsorted(img_ts, ts)
    d0 = np.abs(ts - img_ts[np.clip(j - 1, 0, len(img_ts) - 1)])
    d1 = np.abs(ts - img_ts[np.clip(j, 0, len(img_ts) - 1)])
    return np.minimum(d0, d1) <= tol_us


def first_paired(rec: dict, tol_us: int = PAIR_TOL_US, cams: tuple[str, ...] = PAIR_CAMS) -> int | None:
    """Index of the first lidar_pc paired on EVERY camera of `cams` (the max of the per-camera first-paired indices), or None."""
    firsts = []
    for ch in cams:
        pm = paired_mask(rec["ts"], rec["img_ts"][ch], tol_us)
        if not pm.any():
            return None
        firsts.append(int(np.argmax(pm)))
    return max(firsts)


def frame_indices(rec: dict, stride: int = LIDAR_STRIDE, phase: int | None = None, tol_us: int = PAIR_TOL_US,
                  cams: tuple[str, ...] = PAIR_CAMS, drop_first: int = DROP_FIRST_FRAME) -> np.ndarray:
    """C1: the lidar_pc indices of the OpenScene frames: the grid phase, phase + stride, ..., minus the first `drop_first` grid frames.
    `phase` overrides the measured phase (RED arms only)."""
    n = len(rec["tokens"])
    if phase is None:
        phase = first_paired(rec, tol_us, cams)
        if phase is None:
            return np.zeros(0, dtype=np.int64)
    return np.arange(phase, n, stride, dtype=np.int64)[drop_first:]


def candidates(rec: dict, num_history: int = NUM_HISTORY_FRAMES, num_future: int = NUM_FUTURE_FRAMES, has_route: bool = HAS_ROUTE,
               stride: int = LIDAR_STRIDE, phase: int | None = None, tol_us: int = PAIR_TOL_US, cams: tuple[str, ...] = PAIR_CAMS,
               drop_first: int = DROP_FIRST_FRAME) -> dict:
    """C1-C3 on one log.  Returns {'frame_idx': lidar_pc indices of ALL frames, 'k': candidate positions in the frame list,
    'tokens': candidate lidar_pc tokens, 'lp_idx': their lidar_pc indices, 'n_frames': M (frames in the post-drop list), 'n_dropped_route': windows C3 removed}."""
    fi = frame_indices(rec, stride=stride, phase=phase, tol_us=tol_us, cams=cams, drop_first=drop_first)
    m = len(fi)
    lo, hi = num_history - 1, m - 1 - num_future          # k with a full window: k - 3 >= 0 and k + 10 <= m - 1
    ks = np.arange(lo, hi + 1, dtype=np.int64) if hi >= lo else np.zeros(0, dtype=np.int64)
    dropped = 0
    if has_route and len(ks):
        keep = np.array([rec["scene_has_route"].get(rec["scene"][fi[k]], False) for k in ks], dtype=bool)
        dropped = int((~keep).sum())
        ks = ks[keep]
    lp = fi[ks]
    return {"frame_idx": fi, "k": ks, "lp_idx": lp, "tokens": [rec["tokens"][i] for i in lp], "n_frames": m, "n_dropped_route": dropped}


def nearest_jpgs(path: str, ts_us: int, cams: tuple[str, ...] = CAMERAS, max_dt_ms: float = MAX_DT_MS) -> dict | None:
    """The jpg name the Step A bundle's FrameResolver would load for a token at timestamp ts_us: per camera the image whose
    timestamp is nearest (np.argmin |ts - t|), refused beyond max_dt_ms.  Returns {cam: (filename_jpg, dt_ms)} or None."""
    c = _ro(path)
    out = {}
    try:
        for ch in cams:
            row = c.execute("SELECT token FROM camera WHERE channel = ?", (ch,)).fetchone()
            if not row:
                return None
            rs = c.execute("SELECT timestamp, filename_jpg FROM image WHERE camera_token = ? ORDER BY timestamp", (row[0],)).fetchall()
            if not rs:
                return None
            a = np.array([r[0] for r in rs], dtype=np.int64)
            j = int(np.argmin(np.abs(a - ts_us)))
            dt = abs(int(a[j]) - int(ts_us)) / 1000.0
            if dt > max_dt_ms:
                return None
            out[ch] = (rs[j][1], dt)
    finally:
        c.close()
    return out

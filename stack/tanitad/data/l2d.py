"""yaak-ai/L2D -> TanitAD episode contract (the map + horizon + ego-indicator source).

L2D is **LeRobot v3.0**, Apache-2.0 (tier ``ship``): 100,000 episodes / 26,466,954
frames @ 10 fps / 735 h, KIA Niro EV 2023, 6 cameras (1080x1920) + a rendered BEV
``map`` channel (360x640). Schema + drive-dedup verified in
``Research/2026-07-22-l2d-adapter-schema-and-slice.md``.

Two facts drive this adapter (both MEASURED 2026-07-22, not the card):

1. **Episodes are sliding windows over continuous drives.** Every episode carries a
   native ``session_id`` (the drive) and ``canonical_name`` (``veh/date--time``).
   Consecutive 30 s episodes overlap 93.9 % of the time (median stride 15.0 s) and
   the shared frames are byte-identical (0.000000 m GPS disagreement). So the
   de-dup is a ``groupby(session_id)`` + non-overlapping unix-time tiling — EXACT,
   never a heuristic. Ingesting episodes naively double-counts ~half the frames.
   Splits MUST be drive-disjoint (``split_unit_id = session_id``).

2. **No camera intrinsics ship anywhere** (only ``extrinsic_RDF.yaml``). The state
   layers (poses / actions / vocab) need no camera geometry and are admitted in
   full. The camera-pixel path is admitted ONLY with an ESTIMATED, flagged focal —
   never our asserted ``f_eff=266`` as truth (see ``frame_source`` / ``intrinsics``).

Contract produced (identical to comma2k19 / the D-015 stack):

    frames  [T, 3*n_stack, S, S] uint8   (n_stack consecutive frames, latest last)
    actions [T, 2]        (steer_road_rad [kinematic], accel m/s^2)  -> pose_derived
    poses   [T, 4]        (x_east, y_north, yaw, v)  in a drive-local ENU frame
    episode_id            stable 63-bit hash of ``l2d:{canonical_name}:{episode_index}``

Units handled (MEASURED trap): ``observation.state.vehicle[0]`` "speed" is **km/h**
(reads ~70 on a road posted 70), not m/s -> /3.6; ``action.continuous[2]`` steering
is NORMALIZED (no wheel-angle scale ships) -> steer is derived kinematically, not
read from CAN. The one CAN signal we DO trust is ``action.discrete[1]`` turn_signal
-> the SIGNAL vocab slot.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tanitad.data.toy_driving import ToyEpisode

FPS = 10
KIA_NIRO_WHEELBASE_M = 2.72        # PUBLISHED KIA Niro EV 2023 spec (kinematic steer)
KMH_TO_MS = 1.0 / 3.6
EARTH_M_PER_DEG_LAT = 110540.0     # local equirectangular (drive-local, 30 s..min)
EARTH_M_PER_DEG_LON = 111320.0     # * cos(lat0)

# I7 task-identity fingerprint (matches comma2k19 CORPUS_META so windows are shared).
CORPUS_META = {
    "channels": 9, "image_size": 256, "f_eff_px": 266.0, "hz": 10.0,
    "actions": ("steer_road_rad", "accel_mps2"),
    "poses": ("x_east_m", "y_north_m", "yaw_rad", "v_mps"),
}

# vehicle[8] axis order (MEASURED, meta/info.json).
VEH = {"speed": 0, "heading": 1, "heading_error": 2, "lat": 3, "lon": 4,
       "alt": 5, "acc_x": 6, "acc_y": 7}
TURN_SIGNAL = {0: "none", 1: "indicator_left", 2: "indicator_right"}


# --------------------------------------------------------------------------- #
# Episode index (meta/episodes parquet) -> per-episode dicts                    #
# --------------------------------------------------------------------------- #
_META_COLS = [
    "episode_index", "length", "canonical_name", "session_id", "tasks",
    "data/chunk_index", "data/file_index", "dataset_from_index", "dataset_to_index",
    "stats/observation.state.timestamp/min", "stats/observation.state.timestamp/max",
    "videos/observation.images.front_left/file_index",
    "videos/observation.images.front_left/chunk_index",
    "videos/observation.images.front_left/from_timestamp",
    "videos/observation.images.front_left/to_timestamp",
    "videos/observation.images.map/file_index",
    "videos/observation.images.map/chunk_index",
    "videos/observation.images.map/from_timestamp",
    "videos/observation.images.map/to_timestamp",
]


def read_episode_index(meta_parquet: str | Path) -> list[dict[str, Any]]:
    """The 140-col episodes-meta parquet -> one lean dict per episode.

    Absolute unix-ns span comes from ``stats/observation.state.timestamp/{min,max}``
    (no need to touch the data parquet). Video mapping locates the episode inside a
    bundled mp4 (``file_index`` + ``[from,to]`` seconds)."""
    import pyarrow.parquet as pq
    d = pq.read_table(str(meta_parquet), columns=_META_COLS).to_pydict()
    out = []
    for i in range(len(d["episode_index"])):
        out.append({
            "episode_index": int(d["episode_index"][i]),
            "length": int(d["length"][i]),
            "canonical_name": d["canonical_name"][i],
            "session_id": d["session_id"][i],
            "tasks": list(d["tasks"][i] or []),
            "data_chunk": int(d["data/chunk_index"][i]),
            "data_file": int(d["data/file_index"][i]),
            "row_from": int(d["dataset_from_index"][i]),
            "row_to": int(d["dataset_to_index"][i]),
            "ts_min": int(d["stats/observation.state.timestamp/min"][i][0]),
            "ts_max": int(d["stats/observation.state.timestamp/max"][i][0]),
            "front_file": int(d["videos/observation.images.front_left/file_index"][i]),
            "front_chunk": int(d["videos/observation.images.front_left/chunk_index"][i]),
            "front_from": float(d["videos/observation.images.front_left/from_timestamp"][i]),
            "front_to": float(d["videos/observation.images.front_left/to_timestamp"][i]),
            "map_file": int(d["videos/observation.images.map/file_index"][i]),
            "map_chunk": int(d["videos/observation.images.map/chunk_index"][i]),
            "map_from": float(d["videos/observation.images.map/from_timestamp"][i]),
            "map_to": float(d["videos/observation.images.map/to_timestamp"][i]),
        })
    return out


def episode_id_of(ep: dict) -> int:
    """Stable 63-bit episode id (unique across 100 k L2D episodes; int64-safe)."""
    key = f"l2d:{ep['canonical_name']}:{ep['episode_index']}"
    return int(hashlib.sha1(key.encode()).hexdigest()[:16], 16) & ((1 << 63) - 1)


# LeRobot v3 relative paths (info.json data_path / video_path templates).
def data_rel_path(ep: dict) -> str:
    return f"data/chunk-{ep['data_chunk']:03d}/file-{ep['data_file']:03d}.parquet"


def video_rel_path(ep: dict, key: str = "front_left") -> str:
    ck = ep["front_chunk"] if key == "front_left" else ep["map_chunk"]
    fi = ep["front_file"] if key == "front_left" else ep["map_file"]
    return f"videos/observation.images.{key}/chunk-{ck:03d}/file-{fi:03d}.mp4"


# --------------------------------------------------------------------------- #
# Drive reconstruction + the drive-level de-dup (trap #2)                       #
# --------------------------------------------------------------------------- #
def reconstruct_drives(index: list[dict]) -> dict[str, list[dict]]:
    """``groupby(session_id)`` -> {drive_id: [episodes sorted by unix start]}.

    Native, exact — L2D ships ``session_id``, so no overlap-chaining heuristic is
    needed. Each list is one continuous recording session (up to ~63 min)."""
    drives: dict[str, list[dict]] = defaultdict(list)
    for ep in index:
        drives[ep["session_id"]].append(ep)
    for k in drives:
        drives[k].sort(key=lambda e: e["ts_min"])
    return dict(drives)


def select_nonoverlapping(drive_eps: list[dict]) -> list[dict]:
    """Tile one drive into NON-OVERLAPPING windows by absolute unix time (the
    de-dup). Greedy earliest-start: keep an episode iff it starts at/after the last
    kept episode's end. Absorbs recording gaps for free. Halves a typical drive
    (34 raw -> 17 kept, MEASURED)."""
    kept, last_end = [], -1
    for ep in sorted(drive_eps, key=lambda e: e["ts_min"]):
        if ep["ts_min"] >= last_end:
            kept.append(ep)
            last_end = ep["ts_max"]
    return kept


def dedup_index(index: list[dict]) -> list[dict]:
    """Full corpus -> the de-duplicated episode set (drive-level, all drives)."""
    out = []
    for eps in reconstruct_drives(index).values():
        out.extend(select_nonoverlapping(eps))
    return out


def split_by_drive(index: list[dict], val_frac: float = 0.2, seed: int = 0
                   ) -> dict[str, list[dict]]:
    """Drive-disjoint split (I3): partition on ``session_id`` so no drive's windows
    straddle train/val — the leak that made REF-A's I-JEPA val unusable. Operates on
    an already-de-duplicated index."""
    drives = sorted({ep["session_id"] for ep in index})
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(drives), generator=g).tolist()
    n_val = max(1, int(len(drives) * val_frac))
    val_ids = {drives[i] for i in perm[:n_val]}
    train = [ep for ep in index if ep["session_id"] not in val_ids]
    val = [ep for ep in index if ep["session_id"] in val_ids]
    return {"train": train, "val": val}


# --------------------------------------------------------------------------- #
# Instruction + speed-limit parsing (-> the strategic vocab slots)             #
# --------------------------------------------------------------------------- #
_DIST_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(km|m)\b", re.I)


def parse_distance_m(text: str) -> float | None:
    """First metric distance in an instruction -> metres (``0.7 km`` -> 700.0)."""
    m = _DIST_RE.search(text or "")
    if not m:
        return None
    v = float(m.group(1))
    return v * 1000.0 if m.group(2).lower() == "km" else v


def parse_route_token(text: str) -> str | None:
    """Instruction -> a frozen ROUTE token (map provenance), or None if unnamable.

    Keyword rules over the real instruction corpus (MEASURED examples). ``follow``
    is reserved for the kinematic 'road-following' default; a named 'go straight'
    maps to ``straight``."""
    t = (text or "").lower()
    if "roundabout" in t:
        return "roundabout"
    if "u-turn" in t or "u turn" in t:
        return "u_turn"
    if "merge" in t:
        return "merge"
    if "exit" in t:                                   # highway/roundabout exit
        if "left" in t:
            return "exit_left"
        if "right" in t:
            return "exit_right"
        return "exit_right"                           # roundabout 'first exit' etc.
    if "turn left" in t or "bear left" in t or "keep left" in t:
        return "turn_left"
    if "turn right" in t or "bear right" in t or "keep right" in t:
        return "turn_right"
    if "go straight" in t or "straight on" in t or "continue straight" in t:
        return "straight"
    return None


def parse_max_speed_kmh(s: str | None) -> float | None:
    """``observation.state.max_speed`` string -> km/h float, or None.

    ``'50.0'`` -> 50.0; ``'0.0'`` (no posted limit / unknown-zero) and ``'NA'`` ->
    None (honest gap, never a guessed cap)."""
    if s is None:
        return None
    s = str(s).strip()
    if s in ("", "NA", "nan"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v > 0.0 else None


def speedpolicy_token(limit_kmh: float | None) -> str | None:
    """Posted speed limit -> the strategic SPEEDPOLICY band (map provenance).

    Banded, not a leak: <=30 cap_low, <=60 cap_med, else cap_high; None -> None
    (slot stays unknown). ``nominal`` is reserved for 'no cap active' contexts."""
    if limit_kmh is None:
        return None
    if limit_kmh <= 30.0:
        return "cap_low"
    if limit_kmh <= 60.0:
        return "cap_med"
    return "cap_high"


# --------------------------------------------------------------------------- #
# Native vocab-slot minting (L2D map/CAN -> frozen v3 slots + candidate)        #
# --------------------------------------------------------------------------- #
def l2d_native_slots(*, instruction: str | None, max_speed_kmh: float | None,
                     turn_signal: int | None) -> dict[str, Any]:
    """The slots L2D fills NATIVELY, each a provenance-stamped ``vocab.slot``.

    Returns ``{goal_slots, routedist, slot_source}`` where ``goal_slots`` are
    FROZEN-vocab slots to overlay onto a goal tuple, ``routedist`` is the v1.1
    CANDIDATE (kept OUT of the frozen goal — enrolment is a vocab bump), and
    ``slot_source`` records l2d_native vs derived per slot (separate from the
    frozen ``prov`` axis, which only has kinematic/map/vlm/human/sim/engineered)."""
    from tanitad.lake import vocab as V
    goal: dict[str, dict[str, str]] = {}
    src: dict[str, str] = {}

    # SIGNAL <- CAN turn_signal (human, native) — the one commercially-clean
    # ego-indicator; follows goal_labels' convention of stamping CAN blinker human.
    if turn_signal is not None and int(turn_signal) in TURN_SIGNAL:
        goal["SIGNAL"] = V.slot(TURN_SIGNAL[int(turn_signal)], "human")
        src["SIGNAL"] = "l2d_native"

    # VTARGET / VSOURCE / SPEEDPOLICY <- posted speed limit (map)
    if max_speed_kmh is not None:
        goal["VTARGET"] = V.slot(V.vtarget_band(max_speed_kmh * KMH_TO_MS), "map")
        goal["VSOURCE"] = V.slot("sign_limit", "map")
        src["VTARGET"] = "derived"        # band derived from the native number
        src["VSOURCE"] = "l2d_native"
        sp = speedpolicy_token(max_speed_kmh)
        if sp is not None:
            goal["SPEEDPOLICY"] = V.slot(sp, "map")
            src["SPEEDPOLICY"] = "derived"

    # ROUTE <- NL instruction (map)
    rt = parse_route_token(instruction or "")
    if rt is not None:
        goal["ROUTE"] = V.slot(rt, "map")
        src["ROUTE"] = "derived"

    # ROUTEDIST (v1.1 CANDIDATE — NOT a frozen slot; overlay into the goal would
    # break the frozen 18-slot/114-token counts). Returned separately for a sidecar.
    routedist = None
    dm = parse_distance_m(instruction or "")
    if rt is not None:                                # only meaningful with a maneuver
        band = V.routedist_band(dm, observed_arc_m=V.ROUTEDIST_LOOKED_ENOUGH_M)
        routedist = {"token": band, "prov": "map", "dist_m": dm}
        src["ROUTEDIST"] = "derived"

    return {"goal_slots": goal, "routedist": routedist, "slot_source": src}


def episode_native_slots(rows: dict) -> dict[str, Any]:
    """Episode-representative native slots from decoded state rows (``state_rows``).

    Uses the episode's dominant instruction + the median posted limit + the modal
    non-zero turn signal (the tactical intent for the window)."""
    from collections import Counter
    instr = rows["instruction"]
    limits = [v for v in rows["max_speed_kmh"] if v is not None]
    limit = float(np.median(limits)) if limits else None
    sigs = [s for s in rows["turn_signal"] if s]
    signal = Counter(sigs).most_common(1)[0][0] if sigs else 0
    return l2d_native_slots(instruction=instr, max_speed_kmh=limit,
                            turn_signal=signal)


# --------------------------------------------------------------------------- #
# State rows -> poses / actions (pure numpy; no video)                          #
# --------------------------------------------------------------------------- #
def read_state_rows(data_parquet: str | Path, row_from: int, row_to: int) -> dict:
    """Slice one episode's rows out of a data parquet -> plain arrays/lists.

    Reads only the columns the contract + vocab need. Row range comes from the
    episode-meta ``dataset_from/to_index``."""
    import pyarrow.parquet as pq
    cols = ["observation.state.vehicle", "observation.state.timestamp",
            "observation.state.max_speed", "observation.state.road",
            "observation.state.lanes", "action.discrete", "task.instructions",
            "task.policy", "frame_index"]
    t = pq.read_table(str(data_parquet), columns=cols).slice(row_from, row_to - row_from)
    d = t.to_pydict()
    veh = np.asarray(d["observation.state.vehicle"], dtype=np.float64)      # [T,8]
    disc = np.asarray(d["action.discrete"], dtype=np.int64)                 # [T,2]
    instrs = d["task.instructions"]
    return {
        "vehicle": veh,
        "unix_ns": np.asarray(d["observation.state.timestamp"], dtype=np.int64),
        "max_speed_kmh": [parse_max_speed_kmh(s) for s in d["observation.state.max_speed"]],
        "road": d["observation.state.road"],
        "lanes": d["observation.state.lanes"],
        "turn_signal": disc[:, 1].tolist(),
        "gear": disc[:, 0].tolist(),
        "instruction": instrs[len(instrs) // 2] if instrs else "",
        "policy": (d["task.policy"] or ["EXPERT"])[0],
        "frame_index": np.asarray(d["frame_index"], dtype=np.int64),
    }


def poses_actions_from_state(veh: np.ndarray, unix_ns: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """[T,8] vehicle + [T] unix-ns -> poses [T,4] and actions [T,2].

    poses: GPS lat/lon -> drive-local ENU (ref = first frame), yaw from the GPS
    track direction (robust to L2D's heading convention), v = speed(km/h)/3.6.
    actions: accel = dv/dt (finite diff), steer_road = kinematic bicycle
    ``atan(L*yawrate/v)`` (L2D's CAN steering is normalized/unscaled -> not used).
    Times are real (unix ns), so dt is measured, not assumed."""
    T = veh.shape[0]
    lat, lon = veh[:, VEH["lat"]], veh[:, VEH["lon"]]
    lat0 = float(lat[0])
    x = (lon - lon[0]) * EARTH_M_PER_DEG_LON * math.cos(math.radians(lat0))
    y = (lat - lat[0]) * EARTH_M_PER_DEG_LAT
    v = veh[:, VEH["speed"]] * KMH_TO_MS
    t_s = (unix_ns - unix_ns[0]).astype(np.float64) / 1e9
    dt = np.diff(t_s)
    dt = np.where(dt <= 0, 1.0 / FPS, dt)

    # yaw from track direction; fall back to previous yaw when nearly stationary
    dx, dy = np.diff(x), np.diff(y)
    step = np.hypot(dx, dy)
    yaw = np.zeros(T)
    yaw[:-1] = np.arctan2(dy, dx)
    yaw[-1] = yaw[-2] if T >= 2 else 0.0
    moving = step > 0.05
    last = 0.0
    for i in range(T - 1):
        if moving[i]:
            last = yaw[i]
        else:
            yaw[i] = last
    yaw[-1] = last
    poses = np.column_stack([x, y, yaw, v]).astype(np.float32)

    # accel + kinematic steer
    accel = np.zeros(T)
    accel[:-1] = np.diff(v) / dt
    accel[-1] = accel[-2] if T >= 2 else 0.0
    yaw_un = np.unwrap(yaw)
    yaw_rate = np.zeros(T)
    yaw_rate[:-1] = np.diff(yaw_un) / dt
    yaw_rate[-1] = yaw_rate[-2] if T >= 2 else 0.0
    v_safe = np.maximum(v, 0.5)
    steer = np.arctan(KIA_NIRO_WHEELBASE_M * yaw_rate / v_safe)
    steer = np.clip(steer, -0.7, 0.7)
    actions = np.column_stack([steer, accel]).astype(np.float32)
    return poses, actions


# --------------------------------------------------------------------------- #
# Video decode (optional) — front camera OR the intrinsics-free BEV map         #
# --------------------------------------------------------------------------- #
# L2D ships NO intrinsics. front_camera uses an ESTIMATED focal (flagged); bev_map
# is geometry-defined and needs none. Default is state-only so nothing silently
# depends on an unproven focal.
L2D_FRONT_HFOV_DEG_ASSUMED = 60.0      # ESTIMATED; the follow-on pins it properly


def _decode_window(mp4: str | Path, from_s: float, to_s: float, size: int,
                   est_focal_px: float | None, native_wh: tuple[int, int]) -> torch.Tensor:
    """Decode [from_s, to_s) of a bundled mp4 -> uint8 [T,3,size,size].

    Seeks to ``from_s`` and decodes forward. ``est_focal_px`` (front camera) drives
    the D-016 canonical crop toward f_eff=266 but is ESTIMATED — the record flags
    it. ``None`` (BEV map) center-crop-resizes with no focal claim."""
    import av
    from tanitad.data.calib import focal_crop_resize
    frames = []
    with av.open(str(mp4)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        c.seek(int(from_s / st.time_base), stream=st, any_frame=False, backward=True)
        for fr in c.decode(st):
            ts = float(fr.pts * st.time_base)
            if ts < from_s - 1e-3:
                continue
            if ts >= to_s - 1e-9:
                break
            frames.append(torch.from_numpy(fr.to_ndarray(format="rgb24")).permute(2, 0, 1))
    if not frames:
        raise ValueError(f"no frames decoded from {mp4} in [{from_s},{to_s})")
    vid = torch.stack(frames)                                         # [T,3,H,W]
    if est_focal_px is not None:
        return focal_crop_resize(vid, est_focal_px, size)            # est f_eff~266
    # BEV map: plain center-square crop + resize (no focal semantics)
    _, _, h, w = vid.shape
    c_ = min(h, w)
    vid = vid[..., (h - c_) // 2:(h - c_) // 2 + c_, (w - c_) // 2:(w - c_) // 2 + c_]
    return torch.nn.functional.interpolate(vid.float(), size=(size, size),
                                           mode="bilinear", align_corners=False
                                           ).clamp(0, 255).to(torch.uint8)


def stack_frames(vid_u8: torch.Tensor, n_stack: int = 3) -> torch.Tensor:
    """[T,3,S,S] -> [T-(n-1), 3n, S, S] (frames t-(n-1)..t stacked, latest last)."""
    parts = [vid_u8[i:vid_u8.shape[0] - (n_stack - 1) + i] for i in range(n_stack)]
    return torch.cat(parts, dim=1)


def estimated_front_focal_px(width: int = 1920,
                             hfov_deg: float = L2D_FRONT_HFOV_DEG_ASSUMED) -> float:
    from tanitad.data.calib import nominal_focal_from_hfov
    return nominal_focal_from_hfov(width, hfov_deg)


# --------------------------------------------------------------------------- #
# build_episode — state (always) + optional frames                             #
# --------------------------------------------------------------------------- #
def build_episode(ep: dict, data_parquet: str | Path, *, size: int = 256,
                  n_stack: int = 3, frame_source: str = "none",
                  video_path: str | Path | None = None) -> ToyEpisode:
    """One L2D episode -> a contract :class:`ToyEpisode`.

    ``frame_source``: ``'none'`` (state-only proof — synthesises a zero frame stack
    ONLY to satisfy the tensor contract; do NOT ship), ``'front_camera'`` (real
    pixels, ESTIMATED focal — flagged), or ``'bev_map'`` (intrinsics-free BEV).
    Actions/poses are aligned to the LATEST frame of each stack when frames exist."""
    rows = read_state_rows(data_parquet, ep["row_from"], ep["row_to"])
    poses, actions = poses_actions_from_state(rows["vehicle"], rows["unix_ns"])
    T = poses.shape[0]
    k = n_stack - 1
    eid = episode_id_of(ep)

    if frame_source == "none":
        # Minimal contract filler for state-only smoke — a mid-gray stack (NOT for
        # a shipped shard; use front_camera/bev_map for real records).
        frames = torch.full((max(T - k, 1), 3 * n_stack, size, size), 128,
                            dtype=torch.uint8)
        n = frames.shape[0]
        return ToyEpisode(frames=frames,
                          actions=torch.from_numpy(actions[k:k + n]),
                          poses=torch.from_numpy(poses[k:k + n]), episode_id=eid)

    if frame_source == "front_camera":
        est = estimated_front_focal_px()
        vid = _decode_window(video_path, ep["front_from"], ep["front_to"], size,
                             est, (1920, 1080))
    elif frame_source == "bev_map":
        vid = _decode_window(video_path, ep["map_from"], ep["map_to"], size, None,
                             (640, 360))
    else:
        raise ValueError(f"frame_source must be none|front_camera|bev_map, "
                         f"got {frame_source!r}")
    n = min(vid.shape[0], T)
    stacked = stack_frames(vid[:n], n_stack)                         # [n-k,3n,S,S]
    m = stacked.shape[0]
    return ToyEpisode(frames=stacked,
                      actions=torch.from_numpy(actions[k:k + m]),
                      poses=torch.from_numpy(poses[k:k + m]), episode_id=eid)

"""Dense nvidia/Alpamayo2-Super reasoning: one generation per v9 row (default 10 Hz = EVERY row), pod-side, resumable, sharded.

Why: the corpus `alpamayo/records.parquet` (= HF `Sayood/tanitad-alpamayo2-augmentation`) holds Alpamayo2-Super outputs at ONE anchor
per clip (t0_us = 5,100,000; 4,729 clips; MEASURED). The PI asked for reasoning at every frame (2026-10-25). This script produces it.

INPUT CONTRACT -- reconstructed from NVIDIA's own code, not guessed (all pinned):
  * code   github.com/NVlabs/alpamayo2 @ 59776920bbd981f2c989213bba5ee8594e71ed1d (2026-10-06), package `alpamayo2_super`:
      - `load_physical_aiavdataset.py`: 16 history ego poses at t0-1.5 s .. t0 (0.1 s), 64 future poses t0+0.1 .. t0+6.4 s, both in the
        t0 ego frame; image timestamps t0-0.3, -0.2, -0.1, t0 s, each decoded as the closest frame AT OR BEFORE the request
        (`physical_ai_av.video.SeekVideoReader`, searchsorted(side="right")-1); `assert t0_us > 1_600_000`.
      - `input_profiles.py`: trajectory / meta_action / auto_labeling use camera ids (0,1,2,3,5,6) x frame indices (0,1,2,3)
        (= the records' raw_json `input_profile`, `camera_indices` [0,1,2,3,5,6], `frame_indices` [0..3]).
      - `text_tasks.py`: prompts (meta_action: cot -> meta_action -> traj_future; auto_labeling: cot_auto_labeling, conditioned on the
        FUTURE ego trajectory -- ground truth by default), sampling top_p 0.98, temperature 0.6, top_k None, max_new_tokens 512 / 1024,
        meta_action stops one token after <|traj_future_start|>, trajectory tokens masked from the CoT logits.
      - notebooks/meta_actions.ipynb, autolabeling.ipynb: `torch.cuda.manual_seed_all(42)` before each generate (records: seed 42).
  * weights HF nvidia/Alpamayo2-Super @ 00554695e729a6ff0b6281fd2c81b18d06e33dbe (HF API sha, 35,813,934,322 BF16 params, 71.6 GB
    safetensors; licence openmdw-1.1; VLM = Qwen3VLForConditionalGeneration, 64 layers, 8 KV heads x 128).
  * data   nvidia/PhysicalAI-Autonomous-Vehicles via `physical_ai_av` (0.2.3); only the clip's members are range-read
    (`get_clip_feature_paths(..., members=[...], maybe_stream=True)`), the chunk path per camera is in DataEng/v10/camera_ref.

The per-frame input assembly here (`build_frame_data`) re-implements NVIDIA's loader + `select_input_profile` so a clip's videos are opened
ONCE (not once per frame); `--self-test-assembly K` compares it tensor-by-tensor (and the fused prompt ids) with NVIDIA's own
`select_task_input(load_physical_aiavdataset(...))` on K frames before a run. `--check-anchors` regenerates the 4,729 existing
t0 = 5.1 s rows and `--report-anchors` scores agreement with records.parquet.

LEAK NOTE (CLAUDE.md, labels may use ego, inference is vision-only): auto_labeling is conditioned on the ego's FUTURE trajectory and both
tasks on the ego history. Every output here is a LABEL / teacher signal. Never feed any of it to a deployable model at inference.

Frame clock (V9_SCHEMA.md): v9 row j of a clip = episode row j, raw row k = k0 + j, now_s = grid_start_s + k * dt_s (~10 Hz);
t_us = round(now_s * 1e6) on the clip-local upstream clock (same clock as records t0_us; build_v10_reasoning.py). `--hz` 5 / 2 / 1 keeps
every 2nd / 5th / 10th row, phased per clip so the row nearest t0 = 5.1 s is always kept (cross-checkable against records).

Output: <out>/<split>/shard_XXXX.parquet (finalised) or <out>/<split>/shard_XXXX.parts/part_*.parquet (in progress); key
(split, sha12, v9_row, task); a key already present anywhere under <out>/<split>/ is skipped (idempotent, re-shardable). Rows that
cannot be generated under NVIDIA's input contract are WRITTEN with error "ineligible:<reason>" (never silently dropped).

    PYTHONPATH=/workspace/TanitAD/stack python stack/scripts/alpamayo_dense_reasoning.py run --split train --shard 0/8 --backend vllm ...
    (full commands: DataEng/v10/reasoning_dense/RUNBOOK.md)
"""
from __future__ import annotations

import argparse
import functools
import glob
import hashlib
import json
import os
import re
import socket
import sys
import threading
import time
import queue
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# ============================================================================ pinned provenance
MODEL_ID = "nvidia/Alpamayo2-Super"
MODEL_REVISION = "00554695e729a6ff0b6281fd2c81b18d06e33dbe"          # HF API `sha`, read 2026-10-25
A2S_CODE_REPO = "https://github.com/NVlabs/alpamayo2"
A2S_CODE_COMMIT = "59776920bbd981f2c989213bba5ee8594e71ed1d"         # main @ 2026-10-06
PAIAV_VERSION = "0.2.3"
SCHEMA = "tanitad.reasoning_dense/1"

CAMERA_NAMES = {0: "camera_cross_left_120fov", 1: "camera_front_wide_120fov", 2: "camera_cross_right_120fov",
                3: "camera_rear_left_70fov", 4: "camera_rear_tele_30fov", 5: "camera_rear_right_70fov", 6: "camera_front_tele_30fov"}
DRIVING_CAMERA_IDS = (0, 1, 2, 3, 5, 6)        # input_profiles.DRIVING_SIX_CAMERA_FOUR_FRAME
FRONT_ONLY_CAMERA_IDS = (1,)                    # DEGRADED: not an NVIDIA-validated profile
NUM_FRAMES, FRAME_DT_US = 4, 100_000
NUM_HIST, NUM_FUT, TRAJ_DT_US = 16, 64, 100_000
HIST_ASSERT_US = NUM_HIST * TRAJ_DT_US          # NVIDIA: assert t0_us > 1_600_000
ANCHOR_T0_US = 5_100_000
HZ_STRIDE = {10: 1, 5: 2, 2: 5, 1: 10}
TOP_P, TEMPERATURE, SEED = 0.98, 0.6, 42
MAX_NEW_TOKENS = {"meta_action": 512, "auto_labeling": 1024, "trajectory": 512}
TASKS = ("meta_action", "auto_labeling", "trajectory")   # trajectory = CoT of the trajectory task (hf backend only, needs the expert)
DEFAULT_TASKS = ("meta_action", "auto_labeling")

KEY = ["split", "sha12", "v9_row", "task"]
COLUMNS = ["split", "sha12", "clip_id", "v9_row", "t_us", "task", "model_id", "model_revision", "quantisation", "input_profile", "seed",
           "wall_s", "raw", "cot", "meta_action", "coc", "error",
           # additions (provenance / audit): explicit degraded flag, decoded first-frame time, write time, run id
           "front_only", "camera_tmin_us", "written_unix", "run_id"]
DTYPES = {"split": "string", "sha12": "string", "clip_id": "string", "v9_row": "int32", "t_us": "int64", "task": "string",
          "model_id": "string", "model_revision": "string", "quantisation": "string", "input_profile": "string", "seed": "int64",
          "wall_s": "float64", "raw": "string", "cot": "string", "meta_action": "string", "coc": "string", "error": "string",
          "front_only": "bool", "camera_tmin_us": "int64", "written_unix": "float64", "run_id": "string"}


def sha12_of(u: str) -> str:
    return hashlib.sha256(u.encode()).hexdigest()[:12]


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


# ============================================================================ v9 clock + frame enumeration
@dataclass
class V9Clock:
    split: str
    sha12: list
    row0: np.ndarray
    n_rows: np.ndarray
    k0: np.ndarray
    dt_s: np.ndarray
    now_s: np.ndarray                   # row__now_s, all rows of the split
    k: np.ndarray                       # row__k

    @functools.cached_property
    def index(self) -> dict:
        return {s: i for i, s in enumerate(self.sha12)}

    def clip_now(self, s: str) -> np.ndarray:
        i = self.index[s]; r0 = int(self.row0[i]); return self.now_s[r0:r0 + int(self.n_rows[i])]


def load_v9_clock(npz_path: str, split: str) -> V9Clock:
    z = np.load(npz_path, allow_pickle=True)
    return V9Clock(split=split, sha12=[str(s) for s in z["clip__sha12"]], row0=z["clip__row0"].astype(np.int64),
                   n_rows=z["clip__n_rows"].astype(np.int64), k0=z["clip__k0"].astype(np.int64), dt_s=z["clip__dt_s"].astype(np.float64),
                   now_s=z["row__now_s"].astype(np.float64), k=z["row__k"].astype(np.int64))


def load_uuid_map(paths: Iterable[str]) -> dict:
    """clip_index_v7.2_{train,eval}.json (['clips'] = UUIDs) -> {sha12: uuid}."""
    out = {}
    for p in paths:
        for u in json.load(open(p))["clips"]: out[sha12_of(u)] = u
    return out


def shard_clips(sha_list: Sequence[str], i: int, n: int) -> list:
    """Deterministic, exact partition: clips sorted by sha12, clip of rank r goes to shard r % n."""
    if not (0 <= i < n): raise ValueError(f"shard {i}/{n} out of range")
    return [s for r, s in enumerate(sorted(sha_list)) if r % n == i]


def anchor_phase(now_s: np.ndarray, stride: int, anchor_s: float = ANCHOR_T0_US / 1e6) -> int:
    return int(np.argmin(np.abs(now_s - anchor_s))) % stride


def row_t_us(now_s) -> np.ndarray:
    return np.rint(np.asarray(now_s, np.float64) * 1e6).astype(np.int64)


def parse_clip_filter(spec: str | None, uuid_map: dict) -> set | None:
    """'a,b' or '@file' of sha12 or UUIDs -> set of sha12 (None = all)."""
    if not spec: return None
    items = open(spec[1:]).read().split() if spec.startswith("@") else spec.split(",")
    out = set()
    for x in (t.strip() for t in items if t.strip()):
        out.add(x if re.fullmatch(r"[0-9a-f]{12}", x) else sha12_of(x))
    return out


def enumerate_frames(clock: V9Clock, uuid_map: dict, hz: int = 10, clips: set | None = None, shard=(0, 1)) -> pd.DataFrame:
    """Target frames of one split: (split, sha12, clip_id, v9_row, v9_row_global, t_us), ordered clip-by-clip, row-ascending.
    Sharding is computed on the FULL split clip list (stable membership), then `clips` filters."""
    stride = HZ_STRIDE[int(hz)]
    sel = shard_clips(clock.sha12, *shard)
    if clips is not None: sel = [s for s in sel if s in clips]
    missing = [s for s in sel if s not in uuid_map]
    if missing: raise KeyError(f"{len(missing)} clips have no UUID in the clip index (first: {missing[:3]})")
    parts = []
    for s in sel:
        i = clock.index[s]; r0, n = int(clock.row0[i]), int(clock.n_rows[i])
        now = clock.now_s[r0:r0 + n]
        j = np.arange(n, dtype=np.int64)
        if stride > 1: j = j[(j % stride) == anchor_phase(now, stride)]
        parts.append(pd.DataFrame(dict(split=clock.split, sha12=s, clip_id=uuid_map[s], v9_row=j.astype(np.int32),
                                       v9_row_global=(r0 + j).astype(np.int64), t_us=row_t_us(now[j]))))
    cols = ["split", "sha12", "clip_id", "v9_row", "v9_row_global", "t_us"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def anchor_frames(records: pd.DataFrame, clocks: dict, clips: set | None = None, shard=(0, 1)) -> pd.DataFrame:
    """--check-anchors: every records.parquet clip at t0 = 5.1 s exactly. split = v9 membership (train / eval139) or 'not_v9';
    v9_row = the v9 row within dt/2 of t0 (build_v10_reasoning alignment) or -1."""
    rec = records[records.task == "meta_action"][["clip_id", "t0_us"]].drop_duplicates("clip_id")
    sha = {sha12_of(u): (u, int(t)) for u, t in zip(rec.clip_id, rec.t0_us)}
    sel = shard_clips(list(sha), *shard)
    if clips is not None: sel = [s for s in sel if s in clips]
    rows = []
    for s in sel:
        u, t0 = sha[s]; split, j, g = "not_v9", -1, -1
        for c in clocks.values():
            if s in c.index:
                i = c.index[s]; now = c.clip_now(s); jj = int(np.argmin(np.abs(now - t0 / 1e6)))
                split = c.split
                if abs(now[jj] - t0 / 1e6) <= 0.5 * float(c.dt_s[i]) + 1e-9: j, g = jj, int(c.row0[i]) + jj
                break
        rows.append(dict(split=split, sha12=s, clip_id=u, v9_row=j, v9_row_global=g, t_us=t0))
    return pd.DataFrame(rows, columns=["split", "sha12", "clip_id", "v9_row", "v9_row_global", "t_us"]).astype(
        dict(v9_row="int32", v9_row_global="int64", t_us="int64"))


def frame_seed(seed: int, mode: str, sha12: str, v9_row: int, task: str) -> int:
    if mode == "fixed": return int(seed)
    h = hashlib.sha256(f"{seed}|{sha12}|{v9_row}|{task}".encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


# ============================================================================ eligibility under NVIDIA's input contract
def frame_eligibility(t_us: int, task: str, ego_range: tuple, cam_ts: dict, camera_ids: Sequence[int], relax_hist: bool = False,
                      future_source: str = "gt") -> str | None:
    """None if the frame can be generated exactly as NVIDIA's loader would, else 'ineligible:<reason>'.
    ego_range = (first, last) egomotion timestamp (us; physical_ai_av interpolators do not extrapolate); cam_ts[cam] = frame times."""
    t_us = int(t_us); lo, hi = int(ego_range[0]), int(ego_range[1])
    if not relax_hist and t_us <= HIST_ASSERT_US: return "ineligible:t0<=1.6s(nvidia_history_assert)"
    if t_us - (NUM_HIST - 1) * TRAJ_DT_US < lo or t_us > hi: return "ineligible:ego_history_out_of_range"
    for c in camera_ids:
        ts = cam_ts[c]
        if len(ts) == 0 or t_us - (NUM_FRAMES - 1) * FRAME_DT_US < int(ts[0]) or t_us > int(ts[-1]):
            return f"ineligible:camera{c}_out_of_range"
    if task == "auto_labeling" and future_source == "gt" and t_us + NUM_FUT * TRAJ_DT_US > hi:
        return "ineligible:ego_future<6.4s(auto_labeling_gt_future)"
    return None


# ============================================================================ clip sources (camera + egomotion)
def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64); q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)
    return m.reshape(q.shape[:-1] + (3, 3))


class ClipSource:
    """One clip's egomotion + cameras. Subclasses implement `_ego(ts)`, `ego_range`, `camera_timestamps`, `_decode(cam, idx)`.
    Decoded frames are cached per camera (bounded) so the 3 frames shared by consecutive 10 Hz rows are decoded once."""
    clip_id: str
    cache_frames: int = 96

    def __init__(self):
        self._cache: dict = {}

    def ego_pose(self, ts_us: np.ndarray) -> tuple:                     # -> xyz (N,3) f64, quat xyzw (N,4) f64
        return self._ego(np.asarray(ts_us, np.int64))

    def frames_for(self, cam: int, ts_req: np.ndarray) -> tuple:
        """Closest frame at or before each requested time (physical_ai_av.video semantics) -> (uint8 (N,H,W,3), frame times)."""
        ts = self.camera_timestamps(cam)
        if ts_req.min() < ts.min() or ts_req.max() > ts.max(): raise ValueError(f"camera {cam}: requested times outside the video")
        idx = np.searchsorted(ts, ts_req, side="right") - 1
        return self.frames_by_index(cam, idx), ts[idx]

    def frames_by_index(self, cam: int, idx: np.ndarray) -> np.ndarray:
        c = self._cache.setdefault(cam, OrderedDict())
        need = sorted({int(i) for i in idx if int(i) not in c})
        if need:
            for i, im in zip(need, self._decode(cam, np.asarray(need, np.int64))): c[i] = im
        out = np.stack([c[int(i)] for i in idx])
        for i in idx: c.move_to_end(int(i))
        while len(c) > max(self.cache_frames, len(idx)): c.popitem(last=False)
        return out

    def warm(self, cam: int, ts_req: np.ndarray) -> int:
        """Decode, in ONE reader call, every frame a block of rows will need (a single-frame request re-seeks to the previous
        keyframe each time, ~GOP x slower). The cache is allowed to grow to the block's size. Returns the number of new decodes."""
        ts = self.camera_timestamps(cam)
        ts_req = np.asarray(ts_req, np.int64); ts_req = ts_req[(ts_req >= ts.min()) & (ts_req <= ts.max())]
        if not len(ts_req): return 0
        idx = np.unique(np.searchsorted(ts, ts_req, side="right") - 1)
        c = self._cache.setdefault(cam, OrderedDict())
        need = [int(i) for i in idx if int(i) not in c]
        if need:
            for i, im in zip(need, self._decode(cam, np.asarray(need, np.int64))): c[i] = im
        for i in idx: c.move_to_end(int(i))
        while len(c) > max(self.cache_frames, len(idx)): c.popitem(last=False)
        return len(need)

    def close(self): self._cache.clear()


class PaiavClipSource(ClipSource):
    """Pod source: NVIDIA's physical_ai_av. Only this clip's members are range-read (subset cache under HF_HOME)."""

    def __init__(self, avdi, clip_id: str, camera_ids: Sequence[int]):
        super().__init__()
        import io
        from physical_ai_av import egomotion as pego, video as pvid
        self.clip_id = clip_id
        p = avdi.get_clip_feature_paths(clip_id, "egomotion", members=["egomotion"], maybe_stream=True)
        df = pd.read_parquet(p["egomotion"])
        t = df["timestamp"].to_numpy(copy=True)
        self._interp = pego.EgomotionState.from_egomotion_df(df).create_interpolator(t)
        self.ego_range = (int(t.min()), int(t.max()))
        self._ts, self._readers = {}, {}
        for c in camera_ids:
            pp = avdi.get_clip_feature_paths(clip_id, CAMERA_NAMES[c], members=["video", "frame_timestamps"], maybe_stream=True)
            ts = pd.read_parquet(pp["frame_timestamps"])["timestamp"].to_numpy(copy=True)
            self._ts[c] = ts
            self._readers[c] = pvid.SeekVideoReader(video_data=io.BytesIO(open(pp["video"], "rb").read()), timestamps=ts)

    def _ego(self, ts):
        st = self._interp(ts)
        return np.asarray(st.pose.translation, np.float64), np.asarray(st.pose.rotation.as_quat(), np.float64)

    def camera_timestamps(self, cam): return self._ts[cam]

    def _decode(self, cam, idx): return self._readers[cam].decode_images_from_frame_indices(idx)

    def close(self):
        super().close()
        for r in self._readers.values():
            try: r.close()
            except Exception: pass


def build_frame_data(src: ClipSource, t_us: int, camera_ids: Sequence[int]) -> dict:
    """Model input dict for one frame == select_input_profile(load_physical_aiavdataset(clip, t0_us), profile) of alpamayo2 @ pinned
    commit, restricted to the profile's cameras (verified on the pod by `--self-test-assembly`)."""
    import torch
    t0 = int(t_us)
    hist_ts = t0 + np.arange(-(NUM_HIST - 1), 1, dtype=np.int64) * TRAJ_DT_US
    fut_ts = t0 + np.arange(1, NUM_FUT + 1, dtype=np.int64) * TRAJ_DT_US
    hx, hq = src.ego_pose(hist_ts)
    x0, q0 = hx[-1].copy(), hq[-1].copy()
    R0 = quat_xyzw_to_mat(q0)
    lo, hi = src.ego_range
    has_future = int(fut_ts[-1]) <= hi
    data: dict[str, Any] = {}
    data["ego_history_xyz"] = torch.from_numpy((hx - x0) @ R0).float()[None, None]
    data["ego_history_rot"] = torch.from_numpy(np.einsum("ji,njk->nik", R0, quat_xyzw_to_mat(hq))).float()[None, None]
    if has_future:
        fx, fq = src.ego_pose(fut_ts)
        data["ego_future_xyz"] = torch.from_numpy((fx - x0) @ R0).float()[None, None]
        data["ego_future_rot"] = torch.from_numpy(np.einsum("ji,njk->nik", R0, quat_xyzw_to_mat(fq))).float()[None, None]
    img_req = np.array([t0 - (NUM_FRAMES - 1 - i) * FRAME_DT_US for i in range(NUM_FRAMES)], np.int64)
    cams = sorted(int(c) for c in camera_ids)
    frames, stamps = [], []
    for c in cams:
        im, ts = src.frames_for(c, img_req)
        frames.append(torch.from_numpy(np.ascontiguousarray(im)).permute(0, 3, 1, 2))     # t h w c -> t c h w
        stamps.append(torch.from_numpy(ts.astype(np.int64)))
    image_frames = torch.stack(frames, 0)
    absolute = torch.stack(stamps, 0)
    camera_tmin = int(absolute.min().item())
    data.update({
        "image_frames": image_frames,
        "camera_indices": torch.tensor(cams, dtype=torch.int64),
        "camera_names": [CAMERA_NAMES[c] for c in cams],
        "ego_available": torch.tensor(True),
        "ego_t0": torch.tensor([t0], dtype=torch.int64),
        "ego_t0_relative": (torch.tensor([t0], dtype=torch.int64).float() - float(camera_tmin)) * 1e-6,
        "ego_t0_frame_idx": torch.tensor([NUM_FRAMES - 1], dtype=torch.int64),
        "prediction_start_offset": torch.zeros(1, dtype=torch.float32),
        "ego_history_tvals": torch.arange(-NUM_HIST + 1, 1, dtype=torch.float32) * float(TRAJ_DT_US / 1e6),
        "ego_future_tvals": torch.arange(1, NUM_FUT + 1, dtype=torch.float32) * float(TRAJ_DT_US / 1e6),
        "relative_timestamps": (absolute - camera_tmin).float() * 1e-6,
        "absolute_timestamps": absolute,
        "camera_tmin": camera_tmin,
        "ego_t0_xyz": torch.from_numpy(x0).float().view(1, 1, 3),
        "ego_t0_inv_quat": torch.tensor([q0[3], -q0[0], -q0[1], -q0[2]], dtype=torch.float32).view(1, 1, 4),
        "t0_us": t0,
        "clip_id": src.clip_id,
        "input_profile": {"source_camera_ids": list(CAMERA_NAMES), "source_camera_names": list(CAMERA_NAMES.values()),
                          "camera_ids": cams, "camera_names": [CAMERA_NAMES[c] for c in cams],
                          "frame_indices": list(range(NUM_FRAMES)), "num_cameras": len(cams), "num_frames_per_camera": NUM_FRAMES},
    })
    return data


def input_profile_str(camera_ids: Sequence[int], future_source: str, relax_hist: bool) -> str:
    base = "a2s_front_only_4f_DEGRADED" if tuple(camera_ids) == FRONT_ONLY_CAMERA_IDS else (
        "a2s_driving_6cam_4f" if tuple(camera_ids) == DRIVING_CAMERA_IDS else "a2s_custom_DEGRADED")
    s = f"{base}|cams={','.join(map(str, camera_ids))}|fut={future_source}"
    return s + ("|hist=relaxed" if relax_hist else "")


# ============================================================================ output parsing (mirror of alpamayo2_super, pure python)
_META_SPLIT = re.compile(r"(?P<head>(?:Longitudinal|Lateral|Lane)\s*:)")      # token_utils._META_ACTION_SPLIT_RE
_AUTO_KEYS = ("critical_components_analysis", "ego_vehicle_motion_analysis", "trajectory_analysis", "chain_of_causation")


def _assistant_text(raw: str, task: str) -> str:
    """token_utils.extract_text_tokens.assistant_text."""
    marker = "<|im_start|>assistant\n"
    if marker in raw: raw = raw[raw.rfind(marker) + len(marker):]
    if task in ("trajectory", "meta_action"): raw = raw.split("<|traj_future_start|>", 1)[0]
    return raw.split("<|im_end|>", 1)[0].strip()


def parse_auto_labeling_json(text: str) -> dict:
    """text_tasks.parse_auto_labeling_json: first decodable JSON object, the 4 keys normalised (blank -> None)."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{": continue
        try: payload, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError: continue
        if isinstance(payload, dict):
            out = {}
            for k in _AUTO_KEYS:
                v = payload.get(k)
                if v is None: out[k] = None; continue
                v = v.strip() if isinstance(v, str) else str(v)
                out[k] = v or None
            return out
    raise ValueError("auto-labeling output did not contain a JSON object")


def parse_generation(task: str, raw: str) -> dict:
    """raw generation text (decoded with special tokens) -> dict(cot, meta_action, coc, coc_json). Same split as alpamayo2_super."""
    if task == "trajectory":
        d = json.loads(raw); c = d.get("cot") or d.get("cots") or [""]
        return dict(cot=(c[0] if isinstance(c, list) else str(c)) or None, meta_action=None, coc=None, coc_json=None)
    text = _assistant_text(raw, task)
    if task == "meta_action":
        m = _META_SPLIT.search(text)
        if m is None: return dict(cot=text.rstrip() or None, meta_action=None, coc=None, coc_json=None)
        return dict(cot=text[:m.start()].rstrip() or None, meta_action=text[m.start():].lstrip() or None, coc=None, coc_json=None)
    if task == "auto_labeling":
        js = parse_auto_labeling_json(text)                     # text_tasks: `text or cot` == the assistant text
        return dict(cot=None, meta_action=None, coc=js["chain_of_causation"], coc_json=js)
    raise ValueError(f"unknown task {task!r}")


def normalise_raw(raw: str | None) -> str:
    """Raw text without special tokens / whitespace edges (the HF path pads '<|endoftext|>' after the meta_action stop; vLLM does not)."""
    return re.sub(r"<\|[^|]*\|>", "", raw or "").strip()


# ============================================================================ backends
@dataclass
class Request:
    key: tuple
    task: str
    data: dict
    seed: int


@dataclass
class Generation:
    raw: str | None
    wall_s: float
    error: str | None = None


class Backend:
    name = "abstract"
    model_id = MODEL_ID
    model_revision = MODEL_REVISION
    quantisation = "bf16"

    def generate(self, reqs: list) -> list: raise NotImplementedError

    def close(self): pass


class MockBackend(Backend):
    """CPU smoke backend: deterministic fake text in the real raw formats, clearly marked MOCK. Exercises I/O, assembly, resume."""
    name, model_id, model_revision, quantisation = "mock", "MOCK", "MOCK", "none"

    def __init__(self, fail_every: int = 0):
        self.fail_every, self.n = fail_every, 0

    def generate(self, reqs):
        out = []
        for r in reqs:
            self.n += 1
            if self.fail_every and self.n % self.fail_every == 0: out.append(Generation(None, 0.0, "mock_failure")); continue
            tag = f"MOCK {r.key[1]} row {r.key[2]} t0 {r.data.get('t0_us')} cams {list(map(int, r.data['camera_indices']))}"
            if r.task == "meta_action":
                raw = f"{tag}.Longitudinal: Maintain Speed.\nLateral: Go Straight.\nLane: Lane Keep.<|traj_future_start|><|endoftext|>"
            elif r.task == "auto_labeling":
                raw = json.dumps({"critical_components_analysis": None, "ego_vehicle_motion_analysis": None,
                                  "trajectory_analysis": None, "chain_of_causation": tag}) + "<|im_end|>"
            else:
                raw = json.dumps({"cot": [tag]})
            out.append(Generation(raw, 0.0))
        return out


def _cached_processor_patch():
    """alpamayo2_super.text_tasks.prepare_text_generation_inputs reloads AutoProcessor on every call; cache it (same object/args)."""
    from alpamayo2_super import helper, text_tasks
    orig = helper.get_processor
    cache: dict = {}

    def get_processor(tokenizer, model_config):
        k = (id(tokenizer), id(model_config))
        if k not in cache: cache[k] = orig(tokenizer, model_config)
        return cache[k]
    text_tasks.get_processor = get_processor
    helper.get_processor = get_processor
    return get_processor


class HFBackend(Backend):
    """transformers path = NVIDIA's notebooks (one sample per call, `torch.cuda.manual_seed_all(seed)` before each generate).
    quant 'nf4' = bitsandbytes 4-bit NF4 on the LM backbone (vision tower, lm_head, expert kept bf16): a RECONSTRUCTION of the
    records' 'NF4-backbone-4bit-UNVALIDATED' -- the original quantisation config was not published with the records."""
    name = "hf"

    def __init__(self, model_path: str, revision: str | None = MODEL_REVISION, quant: str = "bf16", device: str = "cuda:0",
                 attn: str = "sdpa", autolabel_future: str = "gt", diffusion_steps: int = 10):
        import torch
        from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
        self.torch, self.device, self.autolabel_future, self.diffusion_steps = torch, device, autolabel_future, diffusion_steps
        kw: dict[str, Any] = dict(dtype=torch.bfloat16, device_map=device, attn_implementation=attn)
        if os.path.isdir(model_path): revision = None
        if revision: kw["revision"] = revision
        if quant == "nf4":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                                           llm_int8_skip_modules=["visual", "lm_head", "expert"])
            self.quantisation = "NF4-backbone-4bit-RECON(bnb nf4; skip visual,lm_head,expert)"
        elif quant != "bf16": raise ValueError(f"hf backend quant must be bf16|nf4, got {quant}")
        self.model = Alpamayo2Super.from_pretrained(model_path, **kw).eval()
        self.model_revision = revision or f"local:{model_path}"
        _cached_processor_patch()

    def _predict_future(self, data, seed):
        from alpamayo2_super import helper
        torch = self.torch
        inp = helper.to_device(helper.prepare_model_inputs(data, self.model.config, self.model.tokenizer), "cuda")
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, rot, _, extra = self.model.sample_trajectories_from_data(data=inp, top_p=TOP_P, temperature=TEMPERATURE, num_traj_samples=1,
                                                                          diffusion_kwargs={"inference_step": self.diffusion_steps},
                                                                          return_extra=True)
        return xyz, rot, extra

    def generate(self, reqs):
        from alpamayo2_super import helper
        from alpamayo2_super.text_tasks import generate_text, prepare_text_generation_inputs
        torch, out = self.torch, []
        for r in reqs:
            t = time.time()
            try:
                if r.task == "trajectory":
                    xyz, _, extra = self._predict_future(r.data, r.seed)
                    raw = json.dumps({"cot": [str(extra["cot"].reshape(-1)[0])], "pred_xyz": xyz[0, 0, 0].float().cpu().tolist()})
                else:
                    fx = fr = None
                    if r.task == "auto_labeling" and self.autolabel_future == "predicted":
                        xyz, rot, _ = self._predict_future(r.data, r.seed); fx, fr = xyz[:, 0, 0].cpu(), rot[:, 0, 0].cpu()
                    inp = prepare_text_generation_inputs(data=r.data, model_config=self.model.config, tokenizer=self.model.tokenizer,
                                                         task=r.task, future_xyz=fx, future_rot=fr)
                    inp = helper.to_device(inp, "cuda")
                    torch.cuda.manual_seed_all(r.seed)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        res = generate_text(self.model, inp, top_p=TOP_P, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS[r.task])
                    raw = res["raw_outputs"][0]
                    mine = parse_generation(r.task, raw)
                    ref = res["cot_auto_labeling_json"][0]["chain_of_causation"] if r.task == "auto_labeling" else (res["meta_action"][0] or None)
                    if (mine["coc"] if r.task == "auto_labeling" else mine["meta_action"]) != ref:
                        out.append(Generation(raw, time.time() - t, "parser_mismatch_vs_alpamayo2_super")); continue
                out.append(Generation(raw, time.time() - t))
            except Exception as e:                                              # noqa: BLE001 -- recorded per row, never silent
                out.append(Generation(None, time.time() - t, f"{type(e).__name__}: {str(e)[:300]}"))
        return out


class VllmBackend(Backend):
    """vLLM path (batched). Serves ONLY the VLM (text tasks never touch the 2.3 B expert) from a checkpoint exported by
    `export-vllm` (keys `vlm.*` -> `*`, Alpamayo tokenizer with the 4,000 trajectory tokens). Prompt = NVIDIA's chat-template string
    with the trajectory placeholders replaced by the encoded trajectory tokens (the strings `<iN>`), images = the same 24 float frames
    with `do_rescale=False`; vLLM's Qwen3-VL processor expands the image pads. The first `parity_checks` prompts are compared token-for-
    token with NVIDIA's own processor + `fuse_traj_tokens` and the run ABORTS on any mismatch. Sampling RNG differs from torch's, so
    text is not bit-identical to the HF path even at equal weights (that is what `--check-anchors` measures)."""
    name = "vllm"

    def __init__(self, a2s_model_path: str, vllm_model_dir: str, tp: int = 1, quant: str = "bf16", max_model_len: int = 8192,
                 gpu_mem: float = 0.92, max_num_seqs: int = 64, parity_checks: int = 8, prefix_caching: bool = True, seed_base: int = 0,
                 image_mode: str = "uint8"):
        from vllm import LLM
        from alpamayo2_super.config import Alpamayo2SuperConfig, build_alpamayo2_super_tokenizer
        import hydra.utils as hyu
        self.cfg = Alpamayo2SuperConfig.from_pretrained(a2s_model_path)
        self.tok = build_alpamayo2_super_tokenizer(a2s_model_path, self.cfg.history_vocab_size, self.cfg.future_vocab_size)
        self.hist_tok = hyu.instantiate(self.cfg.hist_traj_tokenizer_cfg, load_weights=False)
        self.fut_tok = hyu.instantiate(self.cfg.future_traj_tokenizer_cfg, load_weights=False)
        self.get_processor = _cached_processor_patch()
        ti = self.cfg.traj_ids
        lo, n = min(ti["history_id0"], ti["future_id0"]), self.cfg.traj_vocab_size
        # == MaskDiscreteTrajectoryLogitsProcessor; ids >= len(tokenizer) (untrained padding rows of the 155,776 vocab) also excluded
        self.allowed = [i for i in range(len(self.tok)) if not (lo <= i < lo + n)]
        self.parity_left, self.image_mode = parity_checks, image_mode
        self.quantisation = "bf16" if quant == "bf16" else f"vllm-{quant}"
        self.llm = LLM(model=vllm_model_dir, tokenizer=vllm_model_dir, dtype="bfloat16", tensor_parallel_size=tp,
                       quantization=None if quant == "bf16" else quant, max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                       max_num_seqs=max_num_seqs, enable_prefix_caching=prefix_caching, limit_mm_per_prompt={"image": 6 * NUM_FRAMES},
                       mm_processor_kwargs={"min_pixels": self.cfg.min_pixels, "max_pixels": self.cfg.max_pixels,
                                            **({"do_rescale": False} if image_mode == "float" else {})},
                       seed=seed_base, trust_remote_code=False)
        self.model_revision = MODEL_REVISION + "+vllm_export"
        if image_mode != "float": self.quantisation += "|img=uint8"
        try:
            import vllm; self.vllm_version = vllm.__version__
        except Exception: self.vllm_version = "?"

    def _traj_data(self, data, task):
        td = {"ego_history_xyz": data["ego_history_xyz"], "ego_history_rot": data["ego_history_rot"]}
        if task == "auto_labeling": td.update(ego_future_xyz=data["ego_future_xyz"], ego_future_rot=data["ego_future_rot"])
        return td

    def _prompt(self, r: Request):
        from alpamayo2_super.text_tasks import build_text_task_messages, summarize_auto_labeling_conditioning
        from alpamayo2_super.models.utils import SPECIAL_TOKENS, tokenize_history_trajectory, tokenize_future_trajectory
        if r.task == "auto_labeling": summarize_auto_labeling_conditioning(r.data)
        proc = self.get_processor(self.tok, self.cfg)
        text = proc.apply_chat_template(build_text_task_messages(r.data, self.cfg, r.task), tokenize=False, add_generation_prompt=True,
                                        add_vision_id=False)
        td, ti = self._traj_data(r.data, r.task), self.cfg.traj_ids
        spans = [("traj_history", tokenize_history_trajectory(self.hist_tok, td, ti["history_id0"]))]
        if r.task == "auto_labeling": spans.append(("traj_future", tokenize_future_trajectory(self.fut_tok, td, ti["future_id0"])))
        for name, ids in spans:
            ids = ids[0].tolist(); pad = SPECIAL_TOKENS[name]
            if text.count(pad) != len(ids): raise ValueError(f"{name}: {text.count(pad)} placeholders vs {len(ids)} encoded tokens")
            text = text.replace(pad * len(ids), "".join(self.tok.convert_ids_to_tokens(ids)), 1)
        fr = r.data["image_frames"].flatten(0, 1).permute(0, 2, 3, 1)                     # 24 x H x W x 3 (view)
        if self.image_mode == "float":                                                      # NVIDIA-exact: float [0,1], do_rescale False
            return text, [im.float().div(255.0).contiguous().numpy() for im in fr]          # ~600 MB per request -- small batches only
        return text, [np.ascontiguousarray(im.numpy()) for im in fr]                       # uint8: standard vLLM path (see RUNBOOK)

    def _expected_ids(self, r: Request) -> list:
        """NVIDIA's own tokenisation (prepare_text_generation_inputs + fuse_traj_tokens) for the parity check."""
        from alpamayo2_super.text_tasks import prepare_text_generation_inputs
        from alpamayo2_super.models.utils import fuse_traj_tokens
        inp = prepare_text_generation_inputs(data=r.data, model_config=self.cfg, tokenizer=self.tok, task=r.task)
        return fuse_traj_tokens(self.hist_tok, self.fut_tok, inp["tokenized_data"]["input_ids"], self._traj_data(r.data, r.task),
                                self.cfg.traj_ids)[0].tolist()

    def generate(self, reqs):
        from vllm import SamplingParams
        prompts, params, out, ok = [], [], [None] * len(reqs), []
        t = time.time()
        for i, r in enumerate(reqs):
            try:
                text, imgs = self._prompt(r)
            except Exception as e:                                               # noqa: BLE001
                out[i] = Generation(None, 0.0, f"{type(e).__name__}: {str(e)[:300]}"); continue
            prompts.append({"prompt": text, "multi_modal_data": {"image": imgs}})
            stop = [self.cfg.traj_ids["future_start"]] if r.task == "meta_action" else None
            params.append(SamplingParams(n=1, temperature=TEMPERATURE, top_p=TOP_P, top_k=-1, max_tokens=MAX_NEW_TOKENS[r.task],
                                         seed=r.seed, stop_token_ids=stop, allowed_token_ids=self.allowed, skip_special_tokens=False,
                                         detokenize=False))
            ok.append(i)
        if prompts:
            res = self.llm.generate(prompts, params, use_tqdm=False)
            per = (time.time() - t) / len(prompts)
            for i, ro in zip(ok, res):
                r = reqs[i]
                if self.parity_left > 0:
                    self.parity_left -= 1
                    exp = self._expected_ids(r)
                    if list(ro.prompt_token_ids) != exp:
                        raise RuntimeError(f"vLLM prompt != NVIDIA prompt for {r.key} (len {len(ro.prompt_token_ids)} vs {len(exp)}); "
                                           "refusing to run -- see RUNBOOK 'prompt parity'")
                ids = list(ro.outputs[0].token_ids)
                out[i] = Generation(self.tok.decode(ids, skip_special_tokens=False), per)
        return out


def make_backend(a) -> Backend:
    if a.backend == "mock": return MockBackend()
    if a.backend == "hf":
        return HFBackend(a.model, revision=a.model_revision, quant=a.quant, device=a.device, attn=a.attn,
                         autolabel_future=a.autolabel_future, diffusion_steps=a.diffusion_steps)
    if a.backend == "vllm":
        if a.autolabel_future != "gt" or "trajectory" in a.tasks:
            raise SystemExit("vllm backend serves the VLM only: tasks meta_action/auto_labeling with --autolabel-future gt")
        return VllmBackend(a.model, a.vllm_model_dir, tp=a.tp, quant=a.quant, max_model_len=a.max_model_len, gpu_mem=a.gpu_mem,
                           max_num_seqs=a.max_num_seqs, parity_checks=a.parity_checks, prefix_caching=not a.no_prefix_caching,
                           image_mode=a.vllm_image_mode)
    raise SystemExit(f"unknown backend {a.backend}")


# ============================================================================ output store (resumable, idempotent)
def shard_name(i: int) -> str: return f"shard_{i:04d}"


def _atomic_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def output_files(out_root: str, split: str) -> list:
    d = os.path.join(out_root, split)
    return sorted(glob.glob(os.path.join(d, "shard_*.parquet")) + glob.glob(os.path.join(d, "shard_*.parts", "part_*.parquet")))


def read_outputs(out_root: str, split: str, columns=None) -> pd.DataFrame:
    fs = output_files(out_root, split)
    if not fs: return pd.DataFrame(columns=columns or COLUMNS)
    return pd.concat([pd.read_parquet(f, columns=columns) for f in fs], ignore_index=True)


def load_done_keys(out_root: str, split: str, retry_errors: bool = False) -> set:
    """Keys already written under <out>/<split>/ (any shard -> re-sharding is safe). With retry_errors, rows whose error is not
    'ineligible:*' do not count as done (ineligibility is deterministic; it is never retried)."""
    df = read_outputs(out_root, split, columns=["split", "sha12", "v9_row", "task", "error"])
    if retry_errors and len(df):
        e = df["error"].astype("string")
        df = df[e.isna() | e.str.startswith("ineligible:").fillna(False)]
    return set(zip(df["split"].astype(str), df["sha12"].astype(str), df["v9_row"].astype(int), df["task"].astype(str)))


def to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.astype(DTYPES)


class ShardWriter:
    def __init__(self, out_root: str, split: str, shard_i: int, flush_every: int = 256):
        self.out_root, self.split, self.shard_i, self.flush_every = out_root, split, shard_i, flush_every
        self.parts_dir = os.path.join(out_root, split, shard_name(shard_i) + ".parts")
        self.final = os.path.join(out_root, split, shard_name(shard_i) + ".parquet")
        self.buf: list = []
        self.n_written = 0

    def add(self, row: dict):
        self.buf.append(row)
        if len(self.buf) >= self.flush_every: self.flush()

    def flush(self):
        if not self.buf: return
        os.makedirs(self.parts_dir, exist_ok=True)
        n = len(glob.glob(os.path.join(self.parts_dir, "part_*.parquet")))
        _atomic_parquet(to_frame(self.buf), os.path.join(self.parts_dir, f"part_{n:06d}_{os.getpid()}.parquet"))
        self.n_written += len(self.buf); self.buf = []

    def finalize(self) -> str:
        """Compact parts (+ any earlier final file of this shard) into shard_XXXX.parquet; dedupe on KEY (latest write wins)."""
        self.flush()
        fs = sorted(glob.glob(os.path.join(self.parts_dir, "part_*.parquet")))
        if os.path.exists(self.final): fs = [self.final] + fs
        if not fs: return self.final
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        df = dedupe(df)
        _atomic_parquet(df, self.final)
        chk = pd.read_parquet(self.final, columns=KEY)
        if len(chk) != len(df): raise RuntimeError(f"finalize verify failed: {len(chk)} != {len(df)}")
        for f in fs:
            if f != self.final: os.remove(f)
        try: os.rmdir(self.parts_dir)
        except OSError: pass
        return self.final


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per KEY: prefer error-free rows, then the latest write."""
    if not len(df): return df
    df = df.assign(_ok=df["error"].isna().astype(int)).sort_values(["_ok", "written_unix"], kind="stable")
    df = df.drop_duplicates(KEY, keep="last").drop(columns="_ok")
    return df.sort_values(["split", "sha12", "v9_row", "task"], kind="stable").reset_index(drop=True)


# ============================================================================ the run loop
@dataclass
class RunCfg:
    tasks: tuple = DEFAULT_TASKS
    camera_ids: tuple = DRIVING_CAMERA_IDS
    future_source: str = "gt"
    relax_hist: bool = False
    seed: int = SEED
    seed_mode: str = "fixed"
    batch: int = 1
    flush_every: int = 256
    max_frames: int = 0
    run_id: str = ""
    warm_block: int = 16


def _row(frame, task, gen: Generation | None, backend: Backend, cfg: RunCfg, seed: int, camera_tmin: int, error: str | None = None) -> dict:
    raw = gen.raw if gen else None
    err = error if error is not None else (gen.error if gen else None)
    cot = ma = coc = None
    if raw is not None and err is None:
        try:
            p = parse_generation(task, raw); cot, ma, coc = p["cot"], p["meta_action"], p["coc"]
        except Exception as e:                                                    # noqa: BLE001
            err = f"parse:{type(e).__name__}: {str(e)[:200]}"
    elif raw is not None and err is not None and err.startswith("parser_mismatch"):
        try:
            p = parse_generation(task, raw); cot, ma, coc = p["cot"], p["meta_action"], p["coc"]
        except Exception: pass
    return dict(split=frame.split, sha12=frame.sha12, clip_id=frame.clip_id, v9_row=int(frame.v9_row), t_us=int(frame.t_us), task=task,
                model_id=backend.model_id, model_revision=backend.model_revision, quantisation=backend.quantisation,
                input_profile=input_profile_str(cfg.camera_ids, cfg.future_source, cfg.relax_hist), seed=int(seed),
                wall_s=float(gen.wall_s) if gen else 0.0, raw=raw, cot=cot, meta_action=ma, coc=coc, error=err,
                front_only=tuple(cfg.camera_ids) == FRONT_ONLY_CAMERA_IDS, camera_tmin_us=int(camera_tmin),
                written_unix=time.time(), run_id=cfg.run_id)


def run_frames(frames: pd.DataFrame, source_factory, backend: Backend, writer: ShardWriter, cfg: RunCfg, done: set,
               log=print) -> dict:
    """Generate every (frame, task) not in `done`. `source_factory(clip_id) -> ClipSource`. Clip-by-clip; requests are batched up to
    cfg.batch (vLLM batches them; HF runs them one by one). Ineligible frames are written with their reason, no model call."""
    stats = dict(frames=0, requests=0, ineligible=0, errors=0, skipped_done=0, clips=0)
    pending: list = []                                                          # (frame, task, seed, camera_tmin, Request)

    def flush_pending():
        if not pending: return
        gens = backend.generate([p[4] for p in pending])
        for (fr, task, seed, ctm, _), g in zip(pending, gens):
            if g.error: stats["errors"] += 1
            writer.add(_row(fr, task, g, backend, cfg, seed, ctm))
        pending.clear()

    n_left = cfg.max_frames or 1 << 62
    for clip_id, g in frames.groupby("clip_id", sort=False):
        todo = [(fr, t) for fr in g.itertuples(index=False) for t in cfg.tasks if (fr.split, fr.sha12, int(fr.v9_row), t) not in done]
        stats["skipped_done"] += len(g) * len(cfg.tasks) - len(todo)
        if not todo: continue
        stats["clips"] += 1
        try:
            src = source_factory(clip_id)
        except Exception as e:                                                  # noqa: BLE001 -- clip-level failure, every row recorded
            for fr, t in todo:
                writer.add(_row(fr, t, None, backend, cfg, frame_seed(cfg.seed, cfg.seed_mode, fr.sha12, int(fr.v9_row), t), -1,
                                error=f"source:{type(e).__name__}: {str(e)[:200]}"))
                stats["errors"] += 1
            continue
        cam_ts = {c: src.camera_timestamps(c) for c in cfg.camera_ids}
        by_frame: OrderedDict = OrderedDict()
        for fr, t in todo: by_frame.setdefault((int(fr.v9_row), int(fr.t_us)), (fr, []))[1].append(t)
        items = list(by_frame.items())
        for bi, ((_, t_us), (fr, tasks)) in enumerate(items):
            if n_left <= 0: break
            if bi % cfg.warm_block == 0:                                        # batch-decode the next block's frames per camera
                blk = np.array([tu for (_, tu), _ in items[bi:bi + cfg.warm_block]], np.int64)
                req = (blk[:, None] - np.arange(NUM_FRAMES - 1, -1, -1)[None, :] * FRAME_DT_US).ravel()
                for c in cfg.camera_ids:
                    try: src.warm(c, req)
                    except Exception: pass                                      # per-frame assembly reports the error
            n_left -= 1; stats["frames"] += 1
            data, data_err, ctm = None, None, -1
            for t in tasks:
                seed = frame_seed(cfg.seed, cfg.seed_mode, fr.sha12, int(fr.v9_row), t)
                why = frame_eligibility(t_us, t, src.ego_range, cam_ts, cfg.camera_ids, cfg.relax_hist, cfg.future_source)
                if why:
                    stats["ineligible"] += 1; writer.add(_row(fr, t, None, backend, cfg, seed, ctm, error=why)); continue
                if data is None and data_err is None:
                    try:
                        data = build_frame_data(src, t_us, cfg.camera_ids); ctm = int(data["camera_tmin"])
                    except Exception as e:                                       # noqa: BLE001
                        data_err = f"assembly:{type(e).__name__}: {str(e)[:200]}"
                if data_err:
                    stats["errors"] += 1; writer.add(_row(fr, t, None, backend, cfg, seed, ctm, error=data_err)); continue
                pending.append((fr, t, seed, ctm, Request((fr.split, fr.sha12, int(fr.v9_row), t), t, data, seed)))
                stats["requests"] += 1
                if len(pending) >= cfg.batch: flush_pending()
        flush_pending()
        src.close()
        if n_left <= 0: break
    flush_pending(); writer.flush()
    log(json.dumps(stats))
    return stats


# ============================================================================ anchor agreement (records.parquet vs regenerated)
def _boot(x, rng, n=2000):
    x = np.asarray(x, float)
    if not len(x): return [None, None]
    b = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def _kappa(a, b):
    a, b = np.asarray(a, object), np.asarray(b, object)
    if not len(a): return None
    po = float(np.mean(a == b)); labs = set(a) | set(b)
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in labs))
    return None if pe >= 1 else (po - pe) / (1 - pe)


def anchor_agreement(gen: pd.DataFrame, records: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Agreement of regenerated t0 = 5.1 s rows with records.parquet. Unit = clip (one anchor per clip); 95 % CI = clip bootstrap.
    Metrics: exact text (normalised raw, cot, meta_action string, coc) and meta-action CLASS agreement per axis (Longitudinal,
    Lateral, Lane string) + mapped to v9 classes (tanitad.hicap.reasoning.meta_to_v9), with Cohen's kappa; camera_tmin parity vs the
    records' trajectory raw_json (input-assembly check, model-free)."""
    from tanitad.hicap.reasoning import parse_meta_action, meta_to_v9
    rng = np.random.default_rng(seed)
    rec = records.copy(); rec["sha12"] = [sha12_of(u) for u in rec.clip_id]
    g = gen[gen["error"].isna()]
    out: dict = dict(n_generated_rows=int(len(gen)), n_error_rows=int(gen["error"].notna().sum()),
                     error_kinds={k: int(v) for k, v in gen["error"].dropna().str.split(":").str[0].value_counts().items()},
                     model_id=sorted(gen.model_id.dropna().unique().tolist()), quantisation=sorted(gen.quantisation.dropna().unique().tolist()),
                     records_quantisation=sorted(records.quantisation.unique().tolist()))
    # --- meta_action
    rm = rec[rec.task == "meta_action"].set_index("sha12")
    gm = g[g.task == "meta_action"].set_index("sha12")
    common = sorted(set(rm.index) & set(gm.index))
    res = dict(n=len(common))
    if common:
        rj = [json.loads(rm.at[s, "raw_json"]) for s in common]
        r_raw = [normalise_raw(j["raw_outputs"][0]) for j in rj]; g_raw = [normalise_raw(gm.at[s, "raw"]) for s in common]
        r_cot = [(j["cot"][0] or "").strip() for j in rj]; g_cot = [(gm.at[s, "cot"] or "").strip() for s in common]
        r_ma = [(j["meta_action"][0] or "").strip() for j in rj]; g_ma = [(gm.at[s, "meta_action"] or "").strip() for s in common]
        for name, a, b in (("raw_exact", r_raw, g_raw), ("cot_exact", r_cot, g_cot), ("meta_action_exact", r_ma, g_ma)):
            x = np.array([u == v for u, v in zip(a, b)], float); res[name] = dict(rate=float(x.mean()), ci95=_boot(x, rng, n_boot))
        rf = [parse_meta_action(t) for t in r_ma]; gf = [parse_meta_action(t) for t in g_ma]
        for ax in ("Longitudinal", "Lateral", "Lane"):
            a = [f.get(ax, "<missing>") for f in rf]; b = [f.get(ax, "<missing>") for f in gf]
            x = np.array([u == v for u, v in zip(a, b)], float)
            res[f"class_{ax}"] = dict(rate=float(x.mean()), ci95=_boot(x, rng, n_boot), kappa=_kappa(a, b))
        rv = [meta_to_v9(f) for f in rf]; gv = [meta_to_v9(f) for f in gf]
        for ax in ("lon", "lat"):
            a = [str(v[ax][0]) for v in rv]; b = [str(v[ax][0]) for v in gv]
            x = np.array([u == v for u, v in zip(a, b)], float)
            res[f"v9class_{ax}"] = dict(rate=float(x.mean()), ci95=_boot(x, rng, n_boot), kappa=_kappa(a, b))
        allx = np.array([all(f1.get(k) == f2.get(k) for k in ("Longitudinal", "Lateral", "Lane")) for f1, f2 in zip(rf, gf)], float)
        res["class_all_three"] = dict(rate=float(allx.mean()), ci95=_boot(allx, rng, n_boot))
    out["meta_action"] = res
    # --- auto_labeling
    ra = rec[rec.task == "auto_labeling"].set_index("sha12"); ga = g[g.task == "auto_labeling"].set_index("sha12")
    common = sorted(set(ra.index) & set(ga.index)); res = dict(n=len(common))
    if common:
        def rcoc(s):
            j = json.loads(ra.at[s, "raw_json"]); js = (j.get("cot_auto_labeling_json") or [{}])[0] or {}
            return (js.get("chain_of_causation") or "").strip()
        a = [rcoc(s) for s in common]; b = [(ga.at[s, "coc"] or "").strip() for s in common]
        x = np.array([u == v for u, v in zip(a, b)], float); res["coc_exact"] = dict(rate=float(x.mean()), ci95=_boot(x, rng, n_boot))
        r_raw = [normalise_raw(json.loads(ra.at[s, "raw_json"])["raw_outputs"][0]) for s in common]
        g_raw = [normalise_raw(ga.at[s, "raw"]) for s in common]
        x = np.array([u == v for u, v in zip(r_raw, g_raw)], float); res["raw_exact"] = dict(rate=float(x.mean()), ci95=_boot(x, rng, n_boot))
    out["auto_labeling"] = res
    # --- input assembly parity (camera_tmin of the 6-camera window; records keep it on the trajectory row)
    rt = rec[rec.task == "trajectory"].set_index("sha12")
    gc = gen[gen.camera_tmin_us >= 0].drop_duplicates("sha12").set_index("sha12")
    common = sorted(set(rt.index) & set(gc.index)); res = dict(n=len(common))
    if common:
        x = np.array([int(json.loads(rt.at[s, "raw_json"]).get("camera_tmin", -2)) == int(gc.at[s, "camera_tmin_us"]) for s in common], float)
        res["camera_tmin_equal"] = dict(rate=float(x.mean()), ci95=_boot(x, rng, n_boot))
    out["input_assembly"] = res
    out["estimator"] = f"clip bootstrap, {n_boot} resamples, seed {seed}; one anchor per clip so the clip is the cluster"
    out["note"] = ("exact-text agreement is bounded by sampling: records were drawn with temperature 0.6 / top_p 0.98, seed 42 under "
                   "torch's CUDA RNG with an UNVALIDATED NF4 quantisation; a different RNG (vLLM), dtype or GPU changes the draw. "
                   "Class agreement is the decision-relevant number.")
    return out


# ============================================================================ self-test vs NVIDIA's loader (pod)
def self_test_assembly(avdi, frames: pd.DataFrame, camera_ids, k: int, a2s_model: str | None = None, log=print) -> dict:
    """Compare build_frame_data with alpamayo2_super's select_input_profile(load_physical_aiavdataset(...)) on k frames; optionally the
    fused prompt ids (needs only the CPU tokenizer/processor). Raises on any mismatch."""
    import torch
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.input_profiles import InputProfile, select_input_profile
    prof = InputProfile(camera_ids=tuple(camera_ids), frame_indices=tuple(range(NUM_FRAMES)))
    keys = ["image_frames", "camera_indices", "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot", "absolute_timestamps",
            "relative_timestamps", "ego_t0", "ego_t0_relative", "ego_t0_frame_idx", "ego_history_tvals", "ego_future_tvals"]
    frames = frames[(frames.t_us > HIST_ASSERT_US) & (frames.t_us < 13_000_000)]       # NVIDIA's loader needs t0-1.5 s .. t0+6.4 s
    pick = frames.sample(n=min(k, len(frames)), random_state=0) if len(frames) else frames
    rep = []
    for fr in pick.itertuples(index=False):
        try: ref = select_input_profile(load_physical_aiavdataset(fr.clip_id, t0_us=int(fr.t_us), avdi=avdi, include_calibration=False), prof)
        except Exception as e:                                                  # noqa: BLE001
            rep.append(dict(clip=fr.clip_id, t_us=int(fr.t_us), skipped=f"nvidia_loader:{type(e).__name__}:{str(e)[:120]}")); continue
        src = PaiavClipSource(avdi, fr.clip_id, camera_ids)
        mine = build_frame_data(src, int(fr.t_us), camera_ids); src.close()
        diffs = {}
        for kk in keys:
            a, b = mine.get(kk), ref.get(kk)
            if a is None or b is None: diffs[kk] = "missing"; continue
            if a.shape != b.shape: diffs[kk] = f"shape {tuple(a.shape)} vs {tuple(b.shape)}"; continue
            if a.dtype.is_floating_point: d = float((a.double() - b.double()).abs().max())
            else: d = float((a.long() - b.long()).abs().max())
            if d > (1e-5 if a.dtype.is_floating_point else 0): diffs[kk] = d
        if mine["camera_tmin"] != ref["camera_tmin"]: diffs["camera_tmin"] = (mine["camera_tmin"], ref["camera_tmin"])
        if a2s_model:
            from alpamayo2_super.config import Alpamayo2SuperConfig, build_alpamayo2_super_tokenizer
            from alpamayo2_super.text_tasks import prepare_text_generation_inputs
            from alpamayo2_super.models.utils import fuse_traj_tokens
            import hydra.utils as hyu
            cfg = Alpamayo2SuperConfig.from_pretrained(a2s_model)
            tok = build_alpamayo2_super_tokenizer(a2s_model, cfg.history_vocab_size, cfg.future_vocab_size)
            ht = hyu.instantiate(cfg.hist_traj_tokenizer_cfg, load_weights=False); ft = hyu.instantiate(cfg.future_traj_tokenizer_cfg, load_weights=False)
            for task in ("meta_action", "auto_labeling"):
                ids = []
                for d in (mine, ref):
                    inp = prepare_text_generation_inputs(data=d, model_config=cfg, tokenizer=tok, task=task)
                    td = {kk: d[kk] for kk in ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")}
                    if task == "meta_action": td = {kk: td[kk] for kk in ("ego_history_xyz", "ego_history_rot")}
                    ids.append(fuse_traj_tokens(ht, ft, inp["tokenized_data"]["input_ids"], td, cfg.traj_ids)[0].tolist())
                if ids[0] != ids[1]: diffs[f"prompt_ids_{task}"] = sum(x != y for x, y in zip(*ids)) + abs(len(ids[0]) - len(ids[1]))
        rep.append(dict(clip=fr.clip_id, t_us=int(fr.t_us), diffs=diffs))
        log(json.dumps(rep[-1]))
    bad = [r for r in rep if r.get("diffs")]
    if bad: raise SystemExit(f"self-test FAILED on {len(bad)}/{len(rep)} frames: {bad[:3]}")
    n_cmp = sum("diffs" in r for r in rep)
    if n_cmp == 0: raise SystemExit(f"self-test compared NO frame (all {len(rep)} skipped): {rep[:3]}")
    return dict(n=len(rep), n_compared=n_cmp, ok=True, frames=rep)


# ============================================================================ vLLM export of the VLM sub-model
def export_vllm(a2s_dir: str, out_dir: str, log=print) -> dict:
    """Write a standalone Qwen3VLForConditionalGeneration checkpoint: tensors `vlm.<k>` -> `<k>` (expert.* dropped, bytes unchanged),
    config = the A2S `vlm_config`, the Alpamayo tokenizer (base + <i0..i3999> + Alpamayo special tokens), processor configs with the
    A2S pixel budget (min 163,840 / max 196,608 px per image), generation_config and chat template copied."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    cfg = json.load(open(os.path.join(a2s_dir, "config.json")))
    vcfg = dict(cfg["vlm_config"]); vcfg["architectures"] = ["Qwen3VLForConditionalGeneration"]; vcfg["torch_dtype"] = "bfloat16"
    json.dump(vcfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    idx = json.load(open(os.path.join(a2s_dir, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    wmap, total = {}, 0
    for sh in shards:
        tensors = {}
        with safe_open(os.path.join(a2s_dir, sh), framework="pt") as f:
            for k in f.keys():
                if not k.startswith("vlm."): continue
                t = f.get_tensor(k); tensors[k[4:]] = t; total += t.numel() * t.element_size()
        if tensors:
            name = sh.replace("model-", "vlm-")
            save_file(tensors, os.path.join(out_dir, name), metadata={"format": "pt"})
            for k in tensors: wmap[k] = name
            log(f"export {sh}: {len(tensors)} tensors")
    json.dump({"metadata": {"total_size": total}, "weight_map": wmap}, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=1)
    from alpamayo2_super.config import build_alpamayo2_super_tokenizer
    tok = build_alpamayo2_super_tokenizer(a2s_dir, cfg.get("history_vocab_size", 1000), cfg.get("future_vocab_size", 3000))
    tok.save_pretrained(out_dir)
    for f in ("chat_template.jinja", "generation_config.json", "video_preprocessor_config.json"):
        p = os.path.join(a2s_dir, f)
        if os.path.exists(p): open(os.path.join(out_dir, f), "wb").write(open(p, "rb").read())
    pp = json.load(open(os.path.join(a2s_dir, "preprocessor_config.json")))
    pp.update(min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"], size={"shortest_edge": cfg["min_pixels"], "longest_edge": cfg["max_pixels"]})
    json.dump(pp, open(os.path.join(out_dir, "preprocessor_config.json"), "w"), indent=1)
    man = dict(source=a2s_dir, source_revision=MODEL_REVISION, n_tensors=len(wmap), total_bytes=total, a2s_code_commit=A2S_CODE_COMMIT,
               traj_vocab=dict(history_id0=cfg["traj_ids"]["history_id0"], future_id0=cfg["traj_ids"]["future_id0"], size=cfg["traj_vocab_size"]))
    json.dump(man, open(os.path.join(out_dir, "EXPORT_MANIFEST.json"), "w"), indent=1)
    if len(tok) > int(vcfg["text_config"]["vocab_size"]): raise SystemExit(f"tokenizer {len(tok)} > vocab {vcfg['text_config']['vocab_size']}")
    return man


# ============================================================================ CLI
def _paths(a):
    v9 = {"train": os.path.join(a.v9_dir, "v9_labels_train.npz"), "eval139": os.path.join(a.v9_dir, "v9_labels_eval139.npz")}
    ci = [os.path.join(a.clip_index_dir, f"clip_index_v7.2_{s}.json") for s in ("train", "eval")]
    return v9, ci


def _avdi(a):
    import physical_ai_av
    return physical_ai_av.PhysicalAIAVDatasetInterface(revision=a.dataset_revision, cache_dir=a.cache_dir,
                                                       confirm_download_threshold_gb=float("inf"))


def _parse_shard(s: str):
    i, n = (int(x) for x in s.split("/")); return i, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--v9-dir", required=True, help="dir with v9_labels_{train,eval139}.npz")
        p.add_argument("--clip-index-dir", required=True, help="dir with clip_index_v7.2_{train,eval}.json")
        p.add_argument("--split", default="all", choices=["train", "eval139", "all"])
        p.add_argument("--hz", type=int, default=10, choices=sorted(HZ_STRIDE))
        p.add_argument("--clips", default=None, help="comma list or @file of sha12/UUIDs")
        p.add_argument("--shard", default="0/1", help="i/n over the sorted clip list of each split")

    p = sub.add_parser("plan", help="frame counts only (no data, no model)"); common(p)
    p = sub.add_parser("run", help="generate"); common(p)
    p.add_argument("--out", required=True, help="output root (reasoning_dense)")
    p.add_argument("--check-anchors", default=None, metavar="RECORDS_PARQUET", help="regenerate the records' t0=5.1 s rows instead of v9 frames")
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    p.add_argument("--front-only", action="store_true", help="DEGRADED: front-wide camera only (flagged in every row)")
    p.add_argument("--autolabel-future", default="gt", choices=["gt", "predicted"])
    p.add_argument("--relax-history-assert", action="store_true", help="allow t0 <= 1.6 s when the ego log covers t0-1.5 s (flagged)")
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf", "mock"])
    p.add_argument("--model", default=MODEL_ID, help="HF id or local snapshot dir of nvidia/Alpamayo2-Super")
    p.add_argument("--model-revision", default=MODEL_REVISION)
    p.add_argument("--vllm-model-dir", default=None, help="output of `export-vllm`")
    p.add_argument("--quant", default="bf16", help="hf: bf16|nf4 ; vllm: bf16|fp8")
    p.add_argument("--device", default="cuda:0"); p.add_argument("--attn", default="sdpa")
    p.add_argument("--tp", type=int, default=1); p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem", type=float, default=0.92); p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--parity-checks", type=int, default=8); p.add_argument("--no-prefix-caching", action="store_true")
    p.add_argument("--vllm-image-mode", default="uint8", choices=["uint8", "float"],
                   help="float = NVIDIA-exact pixels (do_rescale False) but ~600 MB host RAM per request")
    p.add_argument("--diffusion-steps", type=int, default=10)
    p.add_argument("--batch", type=int, default=0, help="requests per backend call (default: 1 hf, 128 vllm ~ 10 GB host RAM)")
    p.add_argument("--seed", type=int, default=SEED); p.add_argument("--seed-mode", default="fixed", choices=["fixed", "per_frame"])
    p.add_argument("--retry-errors", action="store_true"); p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--flush-every", type=int, default=256)
    p.add_argument("--dataset-revision", default=None, help="PhysicalAI-AV git revision (default: main, resolved sha is logged)")
    p.add_argument("--cache-dir", default=None, help="physical_ai_av / HF cache (member subset cache lives here)")
    p.add_argument("--no-finalize", action="store_true")
    p = sub.add_parser("prefetch", help="range-read this shard's camera/egomotion members into the cache (no GPU)"); common(p)
    p.add_argument("--front-only", action="store_true"); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dataset-revision", default=None); p.add_argument("--cache-dir", default=None)
    p = sub.add_parser("self-test-assembly", help="build_frame_data vs NVIDIA's loader on K frames"); common(p)
    p.add_argument("--k", type=int, default=8); p.add_argument("--front-only", action="store_true")
    p.add_argument("--model", default=None, help="also compare fused prompt ids (local A2S dir or HF id; CPU only)")
    p.add_argument("--dataset-revision", default=None); p.add_argument("--cache-dir", default=None)
    p = sub.add_parser("report-anchors", help="agreement of check-anchors output with records.parquet")
    p.add_argument("--out", required=True); p.add_argument("--records", required=True)
    p = sub.add_parser("export-vllm", help="standalone VLM checkpoint for vLLM")
    p.add_argument("--a2s-dir", required=True); p.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "export-vllm":
        print(json.dumps(export_vllm(a.a2s_dir, a.out_dir), indent=1)); return 0
    if a.cmd == "report-anchors":
        gen = read_outputs(a.out, "check_anchors")
        rep = anchor_agreement(dedupe(gen), pd.read_parquet(a.records))
        rep["inputs"] = dict(records_sha256=sha256_file(a.records), n_output_files=len(output_files(a.out, "check_anchors")))
        p = os.path.join(a.out, "check_anchors", "AGREEMENT.json")
        json.dump(rep, open(p, "w"), indent=1); print(json.dumps(rep, indent=1)); return 0

    v9p, cip = _paths(a)
    splits = ["train", "eval139"] if a.split == "all" else [a.split]
    clocks = {s: load_v9_clock(v9p[s], s) for s in splits}
    umap = load_uuid_map(cip)
    clips = parse_clip_filter(a.clips, umap)
    shard = _parse_shard(a.shard)
    cams = FRONT_ONLY_CAMERA_IDS if getattr(a, "front_only", False) else DRIVING_CAMERA_IDS

    if a.cmd == "plan":
        for s in splits:
            f = enumerate_frames(clocks[s], umap, a.hz, clips, shard)
            print(json.dumps(dict(split=s, hz=a.hz, shard=a.shard, clips=int(f.sha12.nunique()), frames=int(len(f)),
                                  t_le_1p6s=int((f.t_us <= HIST_ASSERT_US).sum()))))
        return 0
    if a.cmd == "prefetch":
        from concurrent.futures import ThreadPoolExecutor
        avdi = _avdi(a)
        ids = sorted({u for s in splits for u in enumerate_frames(clocks[s], umap, a.hz, clips, shard).clip_id.unique()})
        feats = ["egomotion"] + [CAMERA_NAMES[c] for c in cams]

        def pull(u):
            t = time.time(); n = 0
            for ft in feats:
                mem = ["egomotion"] if ft == "egomotion" else ["video", "frame_timestamps"]
                for pth in avdi.get_clip_feature_paths(u, ft, members=mem, maybe_stream=True).values(): n += os.path.getsize(pth)
            return u, n, time.time() - t
        tot = 0
        with ThreadPoolExecutor(a.workers) as ex:
            for i, (u, n, dt) in enumerate(ex.map(pull, ids)):
                tot += n; print(json.dumps(dict(i=i, clip=u, bytes=n, s=round(dt, 1), total_gb=round(tot / 1e9, 2))), flush=True)
        print(json.dumps(dict(clips=len(ids), bytes=tot, dataset_revision=avdi.revision))); return 0
    if a.cmd == "self-test-assembly":
        avdi = _avdi(a)
        fr = pd.concat([enumerate_frames(clocks[s], umap, a.hz, clips, shard) for s in splits], ignore_index=True)
        fr = fr[fr.t_us > HIST_ASSERT_US]
        print(json.dumps(self_test_assembly(avdi, fr, cams, a.k, a.model), indent=1)); return 0

    # ---- run
    tasks = tuple(t.strip() for t in a.tasks.split(",") if t.strip())
    bad = [t for t in tasks if t not in TASKS]
    if bad: raise SystemExit(f"unknown tasks {bad}; supported {TASKS}")
    a.tasks = tasks
    run_id = f"{socket.gethostname()}-{os.getpid()}-{int(time.time())}"
    cfg = RunCfg(tasks=tasks, camera_ids=cams, future_source=a.autolabel_future, relax_hist=a.relax_history_assert, seed=a.seed,
                 seed_mode=a.seed_mode, batch=a.batch or (128 if a.backend == "vllm" else 1), flush_every=a.flush_every,
                 max_frames=a.max_frames, run_id=run_id)
    backend = make_backend(a)
    avdi = _avdi(a)                       # real camera/egomotion data for every backend (mock = data path without a model)
    factory = functools.partial(PaiavClipSource, avdi, camera_ids=cams)
    jobs = []
    if a.check_anchors:
        jobs.append(("check_anchors", anchor_frames(pd.read_parquet(a.check_anchors), clocks, clips, shard)))
    else:
        for s in splits: jobs.append((s, enumerate_frames(clocks[s], umap, a.hz, clips, shard)))
    for split_dir, frames in jobs:
        meta = dict(schema=SCHEMA, run_id=run_id, argv=sys.argv, split_dir=split_dir, shard=a.shard, hz=a.hz, tasks=list(tasks),
                    camera_ids=list(cams), backend=backend.name, model_id=backend.model_id, model_revision=backend.model_revision,
                    quantisation=backend.quantisation, a2s_code=dict(repo=A2S_CODE_REPO, commit=A2S_CODE_COMMIT),
                    dataset_revision=getattr(avdi, "revision", None), generation=dict(top_p=TOP_P, temperature=TEMPERATURE, top_k=None,
                    max_new_tokens=MAX_NEW_TOKENS, seed=a.seed, seed_mode=a.seed_mode), vllm_version=getattr(backend, "vllm_version", None),
                    v9_npz_sha256={s: sha256_file(v9p[s]) for s in splits}, n_frames=int(len(frames)), started_unix=time.time())
        os.makedirs(os.path.join(a.out, split_dir), exist_ok=True)
        json.dump(meta, open(os.path.join(a.out, split_dir, f"{shard_name(shard[0])}.{run_id}.meta.json"), "w"), indent=1)
        done = load_done_keys(a.out, split_dir, a.retry_errors)
        w = ShardWriter(a.out, split_dir, shard[0], a.flush_every)
        st = run_frames(frames, factory, backend, w, cfg, done)
        exp = {(r.split, r.sha12, int(r.v9_row), t) for r in frames.itertuples(index=False) for t in tasks}
        have = load_done_keys(a.out, split_dir)
        if not a.no_finalize and exp <= have and not a.max_frames:
            print(json.dumps(dict(finalized=w.finalize(), **st)))
        else:
            print(json.dumps(dict(finalized=None, missing=len(exp - have), **st)))
    backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

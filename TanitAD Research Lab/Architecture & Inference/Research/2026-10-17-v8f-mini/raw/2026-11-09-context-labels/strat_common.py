"""strat_common: geometry labels, schema, session key, splits for tanitad-context-labels (label_version strat-labels-v1).

Run with  C:/Users/Admin/venvs/tanitad/Scripts/python.exe  and the repo tip export on the path (TIP_STACK).
Labels are BUILD-TIME ONLY. None of these fields is an inference input.
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

TIP_STACK = os.environ.get("STRAT_TIP", "F:/strat_scratch/tip/stack")   # git export of mirror tip 26fb0bc1 (read-only copy)
for _p in (TIP_STACK, TIP_STACK + "/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EGO_DIR = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo"
CAM_DIR = "C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov"
META = "C:/Users/Admin/tanitad-data/physicalai/metadata/data_collection.parquet"
CLIP_INDEX = "C:/Users/Admin/tanitad-data/physicalai/clip_index.parquet"
LABEL_VERSION = "strat-labels-v1"

HZ = 10.0                       # label track rate (same as s2_geom_emit_v7 / egomotion_source)
N_FRAMES = 101                  # k = 0..100, t_s = 0.2 k, video frame = 6 k (30 fps source)
CKPT_T = (6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0)   # SPEC_HWM section 3
TAC_BAND = (2.0, 6.0)
STR_BAND = (6.0, 30.0)
V9_TAC_BAND = (2.0, 8.0)        # v9 `LITERALS["band_s"]`
V72_STR_BAND = (8.0, 30.0)      # v7.2 strategic band
GAP_BAND = (6.0, 8.0)

COUNTRIES = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
             'Hungary', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia',
             'Slovenia', 'Spain', 'Sweden', 'United States']


def video_frame(k: int) -> int:
    return int(round(30.0 * 0.2 * k))


# ------------------------------------------------------------------------------------------------ ego track
def load_track(clip_id: str, hz: float = HZ):
    """Same arithmetic as tanitad.data.egomotion_source.load (read there), without its zip-scanning index.

    Returns poses [T,4] = x, y, unwrapped yaw, |v| on a `hz` grid from the first egomotion sample, and span_s."""
    import pandas as pd
    df = pd.read_parquet(f"{EGO_DIR}/{clip_id}.parquet")
    return track_from_df(df, hz)


def track_from_df(df, hz: float = HZ):
    ts = df["timestamp"].to_numpy(dtype=np.float64) / 1e6
    ts = ts - ts[0]
    v = np.linalg.norm(df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64), axis=1)
    qx, qy, qz, qw = (df[c].to_numpy(dtype=np.float64) for c in ("qx", "qy", "qz", "qw"))
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    grid = np.arange(0.0, ts[-1] + 1e-9, 1.0 / hz)
    yaw_u = np.unwrap(yaw)
    poses = np.stack([np.interp(grid, ts, df["x"].to_numpy(float)), np.interp(grid, ts, df["y"].to_numpy(float)),
                      np.interp(grid, ts, yaw_u), np.interp(grid, ts, v)], axis=1)
    return poses, float(ts[-1])


def checkpoints(poses: np.ndarray, i0: int, hz: float = HZ, times=CKPT_T):
    """Ego-frame (x forward, y left) position and relative heading at +T s from index i0; mask = log long enough."""
    n = len(poses)
    x0, y0, h0 = poses[i0, 0], poses[i0, 1], poses[i0, 2]
    c, s = math.cos(-h0), math.sin(-h0)
    out = np.full((len(times), 3), np.nan, np.float32)
    valid = np.zeros(len(times), bool)
    for m, T in enumerate(times):
        j = i0 + int(round(T * hz))
        if j <= n - 1:
            dx, dy = poses[j, 0] - x0, poses[j, 1] - y0
            dh = poses[j, 2] - h0
            out[m] = (c * dx - s * dy, s * dx + c * dy, (dh + math.pi) % (2 * math.pi) - math.pi)
            valid[m] = True
    return out, valid


# ------------------------------------------------------------------------------------------------ tokens
@contextlib.contextmanager
def bands(tac, strat, gap):
    """Temporarily set the band constants of s2_geom_emit_v7 (module globals read at call time)."""
    import s2_geom_emit_v7 as S2
    old = (S2.TACTICAL_S, S2.STRATEGIC_S, S2.GAP_S)
    S2.TACTICAL_S, S2.STRATEGIC_S, S2.GAP_S = tuple(tac), tuple(strat), tuple(gap)
    try:
        yield S2
    finally:
        S2.TACTICAL_S, S2.STRATEGIC_S, S2.GAP_S = old


def _tactical_token(S2, poses, i0, inplan):
    """Tactical goal token for the band that is currently set: the first in-band turn, else a stop, else FOLLOW_LANE."""
    turns = [m for m in inplan if S2.is_turn(m)]
    if turns:
        m = turns[0]
        side = "L" if m[2] > 0 else "R"
        return f"TURN_{side}", float(m[0]), S2._arc_to(poses, i0, m[0], HZ)
    lo = i0 + int(round(S2.TACTICAL_S[0] * HZ))
    hi = min(len(poses) - 1, i0 + int(round(S2.TACTICAL_S[1] * HZ)))
    if hi > lo and float(poses[i0, 3]) > S2.V_STOP_MS:
        import tanitad.data.ego_manoeuvre as EM
        st = EM.stop_episodes(poses[lo:hi + 1, 3], HZ)
        if st:
            t = round(S2.TACTICAL_S[0] + st[0][0] / HZ, 1)
            return "STOP_POINT", t, S2._arc_to(poses, i0, t, HZ)
    return "FOLLOW_LANE", math.nan, math.nan


class SeqCache:
    """`manoeuvre_sequence(poses, key)` of s2_geom_emit_v7 for every key of one track, computed once.

    The smoothed yaw rate and the |rate| >= threshold runs do not depend on the key; only the truncation of the run that straddles
    the key does. Same arithmetic as the original (equality is a test: test_seq_cache_equals_original)."""

    def __init__(self, poses):
        import s2_geom_emit_v7 as S2
        self.S2, self.p = S2, np.asarray(poses, dtype=np.float64)
        p = self.p
        self.yaw = np.unwrap(p[:, 2])
        om = np.gradient(self.yaw) * HZ
        k = max(1, int(round(0.5 * HZ)))
        self.rate = np.degrees(np.convolve(om, np.ones(k) / k, mode="same"))
        act = np.abs(self.rate) >= S2.TURN_RATE_DEG_S
        self.runs, i = [], 0
        while i < len(act):
            if act[i]:
                j = i
                while j + 1 < len(act) and act[j + 1]:
                    j += 1
                self.runs.append((i, j))
                i = j + 1
            else:
                i += 1
        self.need = int(round(S2.MIN_SEG_S * HZ))

    def seq(self, key):
        S2, p = self.S2, self.p
        out = []
        for s, e in self.runs:
            if e < key:
                continue
            a, j_end = max(s, key), e
            if j_end - a + 1 < self.need:
                continue
            b = min(j_end, len(p) - 1)
            d = float(np.degrees(self.yaw[b] - self.yaw[a]))
            if abs(d) >= S2.MIN_TURN_DEG:
                seg = p[a:b + 1]
                arc = float(np.sum(np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1]))))
                R = float(arc / np.radians(abs(d))) if abs(d) > 1.0 else float("inf")
                vmin = float(p[a:b + 1, 3].min())
                out.append((round((a - key) / HZ, 1), round((j_end - key + 1) / HZ, 1), round(d, 1), round(R, 1), round(vmin, 2)))
        return out


def frame_tokens(poses: np.ndarray, i0: int, cache: "SeqCache | None" = None) -> dict:
    """Tokens for one key frame at index i0 of the 10 Hz track. Pure geometry (no CoT, no classifier)."""
    n = len(poses)
    horizon = (n - 1 - i0) / HZ
    out = {}
    with bands(TAC_BAND, STR_BAND, (6.0, 6.0)) as S2:               # new cut: tactical 2-6, strategic 6-30, no gap
        seq = cache.seq(i0) if cache is not None else S2.manoeuvre_sequence(poses, i0, HZ)
        inplan, post, _g = S2.split_by_band(poses, i0, seq, HZ)
        tok, by, within = _tactical_token(S2, poses, i0, inplan)
        out.update(g_tac6_token=tok, g_tac6_by_time_s=by, g_tac6_within_m=within, g_tac6_valid=bool(horizon >= TAC_BAND[1]))
        g, a = S2.strategic(poses, i0, seq, HZ)
        out.update(g_str_token=g["token"], g_str_by_time_s=float(g["args"].get("by_time_s", math.nan)),
                   g_str_within_m=float(g["args"].get("within_m", math.nan)), a_str_token=a["token"],
                   g_str_valid=bool(horizon >= STR_BAND[1] or (g["token"] != "FOLLOW_ROUTE" and g["args"].get("by_time_s", 1e9) <= horizon)))
    with bands(V9_TAC_BAND, V72_STR_BAND, GAP_BAND) as S2:          # old cuts, kept to reproduce and to tabulate changes
        seq = cache.seq(i0) if cache is not None else S2.manoeuvre_sequence(poses, i0, HZ)
        inplan, post, gap = S2.split_by_band(poses, i0, seq, HZ)
        tok, _, _ = _tactical_token(S2, poses, i0, inplan)
        out["tac_v9band_token"] = tok
        g, _a = S2.strategic(poses, i0, seq, HZ)
        out["g_str_v72band_token"] = g["token"]
        gt = [m for m in gap if S2.is_turn(m)]
        out["g_gap_6_8s_token"] = (f"TURN_{'L' if gt[0][2] > 0 else 'R'}" if gt else "NONE")
        out["g_gap_6_8s_valid"] = bool(horizon >= GAP_BAND[1])
    out["horizon_s"] = float(horizon)
    return out


def clip_rows(clip_id: str, meta: dict, session_key: str, chunk: int, split: str, n_video_frames: int | None = None):
    """101 rows of geometry labels for one clip. n_video_frames (from the timestamps file) sets `frame_present`."""
    poses, span = load_track(clip_id)
    cache = SeqCache(poses)
    rows = []
    for k in range(N_FRAMES):
        i0 = int(round(0.2 * k * HZ))
        r = {"clip_id": clip_id, "frame_idx": k, "t_s": round(0.2 * k, 1), "video_frame": video_frame(k),
             "frame_present": bool(n_video_frames is None or video_frame(k) < n_video_frames),
             "country": meta["country"], "month": int(meta["month"]), "hour_of_day": int(meta["hour_of_day"]), "meta_source": "data_collection"}
        ck, cv = checkpoints(poses, i0)
        r["ckpt_t_s"] = list(CKPT_T)
        r["ckpt_x"], r["ckpt_y"], r["ckpt_heading"] = ck[:, 0].tolist(), ck[:, 1].tolist(), ck[:, 2].tolist()
        r["ckpt_valid_mask"] = cv.tolist()
        r.update(frame_tokens(poses, i0, cache))
        r.update(session_key=session_key, chunk=int(chunk), split=split, label_version=LABEL_VERSION)
        rows.append(r)
    return rows


# the pinned column list of the geometry part (context columns are pinned in extract_context.CTX_COLUMNS)
GEOM_COLUMNS = ["clip_id", "frame_idx", "t_s", "video_frame", "frame_present", "country", "month", "hour_of_day", "meta_source",
                "ckpt_t_s", "ckpt_x", "ckpt_y", "ckpt_heading", "ckpt_valid_mask",
                "g_tac6_token", "g_tac6_by_time_s", "g_tac6_within_m", "g_tac6_valid",
                "g_str_token", "g_str_by_time_s", "g_str_within_m", "a_str_token", "g_str_valid",
                "tac_v9band_token", "g_str_v72band_token", "g_gap_6_8s_token", "g_gap_6_8s_valid", "horizon_s",
                "session_key", "chunk", "split", "label_version"]


def rows_to_table(rows):
    import pyarrow as pa
    cols = {}
    for c in GEOM_COLUMNS:
        vals = [r[c] for r in rows]
        if c in ("ckpt_t_s", "ckpt_x", "ckpt_y", "ckpt_heading"):
            flat = pa.array(np.asarray(vals, np.float32).ravel(), pa.float32())
            cols[c] = pa.FixedSizeListArray.from_arrays(flat, len(CKPT_T))
        elif c == "ckpt_valid_mask":
            flat = pa.array(np.asarray(vals, bool).ravel(), pa.bool_())
            cols[c] = pa.FixedSizeListArray.from_arrays(flat, len(CKPT_T))
        elif c in ("frame_idx", "video_frame", "month", "hour_of_day", "chunk"):
            cols[c] = pa.array(vals, pa.int32())
        elif c in ("t_s", "g_tac6_by_time_s", "g_tac6_within_m", "g_str_by_time_s", "g_str_within_m", "horizon_s"):
            cols[c] = pa.array(vals, pa.float32())
        elif c in ("frame_present", "g_tac6_valid", "g_str_valid", "g_gap_6_8s_valid"):
            cols[c] = pa.array(vals, pa.bool_())
        else:
            cols[c] = pa.array(vals, pa.string())
    return pa.table(cols)


# ------------------------------------------------------------------------------------------------ sessions and splits
def session_groups(clip_ids_by_chunk: dict, pair_edges=()):
    """Union-find: group key = chunk, merged with continuation pairs (a, b) of clip ids. Returns clip -> 'S<min chunk>'-style key
    (the smallest chunk id of the merged component, zero padded, so the key is stable across runs)."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    chunk_of = {}
    for ch, clips in clip_ids_by_chunk.items():
        find(ch)
        for c in clips:
            chunk_of[c] = ch
    for a, b in pair_edges:
        if a in chunk_of and b in chunk_of:
            union(chunk_of[a], chunk_of[b])
    return {c: f"S{find(ch):05d}" for c, ch in chunk_of.items()}, chunk_of


def stable_hash(s: str, seed: str = "ctx-ho-v1") -> int:
    return int.from_bytes(hashlib.sha256((seed + ":" + s).encode()).digest()[:8], "big")


def pick_ctx_ho(cands, train_groups: set, per_country: int = 40, max_per_chunk: int = 2):
    """cands: list of dicts(clip_id, country, chunk, session_key). Excludes any session touching train; seeded-hash order;
    `per_country` per country (fewer where fewer are eligible). Chunk cap: `max_per_chunk`, raised per country to
    ceil(per_country / free chunks) when the country has too few free chunks to reach `per_country` at the base cap (Hungary has 1)."""
    elig = [r for r in cands if r["session_key"] not in train_groups]
    free_chunks = {}
    for r in elig:
        free_chunks.setdefault(r["country"], set()).add(r["chunk"])
    cap = {c: max(max_per_chunk, math.ceil(per_country / len(s))) for c, s in free_chunks.items()}
    out, per_c, per_ch = [], {}, {}
    for r in sorted(elig, key=lambda r: stable_hash(r["clip_id"])):
        if per_c.get(r["country"], 0) >= per_country or per_ch.get(r["chunk"], 0) >= cap[r["country"]]:
            continue
        per_c[r["country"]] = per_c.get(r["country"], 0) + 1
        per_ch[r["chunk"]] = per_ch.get(r["chunk"], 0) + 1
        out.append(r["clip_id"])
    return out


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

"""bank_presteps I/O: compact per-record tables from the v8f DAgger banks + the logged native tracks.

Everything here READS banked artifacts; nothing is written outside this package.
Memory discipline: one npz MEMBER at a time (np.load on an npz is lazy per member), shard by shard.
Run with `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I` (isolated); this file's directory is put on sys.path by the caller.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ART = Path("E:/Projects/TanitAD-artifacts")
BANK_DIRS = {"s0": ART / "v8f-dagger" / "bank_s0", "s1": ART / "v8f-dagger" / "bank_s1"}
TRACKS = ART / "sfresh_inputs" / "inputs" / "native_tracks.pt"
TRACKS_SHA = "1db431f112b1bd7c426073c43c7df5fc75b2a032736c1e2020f02ffb118669e7"   # = bank header inputs_sha256.tracks
PKG = ART / "dagland" / "bank_presteps"
RAW = PKG / "raw"
REASONS = ("ok", "off_lat", "off_head", "end_edge", "nonfinite")
OK, OFF_LAT, OFF_HEAD, END_EDGE, NONFINITE = range(5)
N_WINDOWS = 306566


def free_ram_gb() -> float:
    import subprocess
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                         capture_output=True, text=True).stdout.strip().replace(",", ".")
    return float(out) / (1024 * 1024)


def require_ram(min_gb: float = 3.0):
    g = free_ram_gb()
    if g < min_gb:
        raise SystemExit(f"free RAM {g:.2f} GB < {min_gb} GB: not starting (write HANDOFF.md)")
    return g


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_tracks(verify_sha: bool = True) -> dict:
    """{clip_id: float64 [n_raw, 4]} (x, y, yaw, v) native 0.1 s track; rows 0-1 NaN for TRAIN clips."""
    import torch
    if verify_sha:
        s = sha256_file(TRACKS)
        if s != TRACKS_SHA:
            raise SystemExit(f"native_tracks.pt sha256 {s} != bank header {TRACKS_SHA}")
    o = torch.load(TRACKS, map_location="cpu", weights_only=True)
    return {c: t.double().numpy() for c, t in o["tracks"].items()}


def grid_of(track):
    """(grid [T,3] = (x,y,yaw) on the 0.2 s cache grid, v [T] logged speed) from a native track: even raw frames."""
    tr = np.asarray(track, np.float64)[0::2]
    return tr[:, :3], tr[:, 3]


def read_shard_members(path, keys):
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in keys:
            out[k] = z[k]
        hdr = json.loads(bytes(z["header_json"]).decode("utf-8"))
    return out, hdr


def build_table(seed: str, tracks: dict, log=print) -> dict:
    """One compact table for a whole bank (all 4 shards), in the bank's record order. Keys:
    clip_g (int32 global clip index), t0, tick, j, r, reason (int8), v_own (p10f[-1,0]), kin3_v (kin3[0]), e_lat, dpsi, ox, oy, oyaw (own pose at j)
    plus `clip_names` (global index -> clip id) and the bank manifest sha."""
    bdir = BANK_DIRS[seed]
    man = json.loads((bdir / "bank.json").read_bytes().decode("utf-8"))
    cols = {k: [] for k in ("clip_g", "t0", "tick", "j", "r", "reason", "v_own", "kin3_v", "e_lat", "dpsi", "ox", "oy", "oyaw", "px", "py")}
    for s in man["shards"]:
        f = bdir / s["file"]
        log(f"[{seed}] shard {s['file']}  free RAM {free_ram_gb():.2f} GB")
        arr, hdr = read_shard_members(f, ("ci", "t0", "tick", "j", "r", "reason", "e_lat", "dpsi", "kin3"))
        with np.load(f, allow_pickle=False) as z:
            p10f = z["p10f"]
            v_own = np.array(p10f[:, -1, 0], np.float64)
            del p10f
            pose = z["pose"]
            own = np.array(pose[:, -1, :3], np.float64)
            prev = np.array(pose[:, -2, :2], np.float64)          # own pose of row j-1 (AMENDMENT 1, A3)
            del pose
        cl = {int(k): v for k, v in hdr["clips"].items()}
        # global indices are assigned after all shards are read: store the clip NAME per record for now
        local = np.asarray(arr["ci"], np.int64)
        name_arr = np.empty(len(local), dtype=object)
        keys = np.array(sorted(cl))
        vals = np.array([cl[k] for k in keys], dtype=object)
        pos = np.searchsorted(keys, local)
        name_arr[:] = vals[pos]
        cols["clip_g"].append(name_arr)
        for k in ("t0", "tick", "j", "r", "reason", "e_lat", "dpsi"):
            cols[k].append(arr[k])
        cols["v_own"].append(v_own)
        cols["kin3_v"].append(np.asarray(arr["kin3"], np.float64)[:, 0])
        cols["ox"].append(own[:, 0]); cols["oy"].append(own[:, 1]); cols["oyaw"].append(own[:, 2])
        cols["px"].append(prev[:, 0]); cols["py"].append(prev[:, 1])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    uniq = np.array(sorted(set(out["clip_g"].tolist())))
    idx = {c: i for i, c in enumerate(uniq.tolist())}
    out["clip_g"] = np.array([idx[c] for c in out["clip_g"].tolist()], np.int32)
    out["clip_names"] = uniq
    out["reason"] = out["reason"].astype(np.int8)
    out["bank_sha256"] = np.array(sha256_file(bdir / "bank.json"))
    return out


def clip_offsets(clip_names, tracks):
    """Concatenated logged arrays over the bank's clips: offsets, T per clip, grid xyz [sumT,3], v [sumT]."""
    Ts, G, V = [], [], []
    for c in clip_names.tolist():
        g, v = grid_of(tracks[c])
        Ts.append(len(g)); G.append(g); V.append(v)
    Ts = np.array(Ts, np.int64)
    offs = np.concatenate([[0], np.cumsum(Ts)[:-1]])
    return offs, Ts, np.concatenate(G), np.concatenate(V)

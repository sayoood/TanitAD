#!/usr/bin/env python3
"""build_strategic_geom.py -- writes ``strategic_geom_{train,eval}.npz`` (+ manifest) on the 0.2 s cache grid.

Clip universe = the v7.2 release records (train 4,572; eval 147; ``release/v72``), clock = the v9 clock rule
(sidecar, else ``clip_clock.pose_dt`` of the v9 manifest poses, else the nominal 0.1 s with grid start 0). Steps per
clip: 101 (= the 20.1 s cache, ``t_ep`` 201, ``n = (t_ep - 1) // 2 + 1``) unless the v2 manifest poses say otherwise.
Cache step j <-> raw frame k = 2 j <-> log time now = grid_start_s + k * dt_s.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_v9_verbatim as B  # noqa: E402
import strategic_ck as CK  # noqa: E402

REL = "E:/Projects/TanitAD-artifacts/_s2build-copy-20260919/release/v72/s2_labels_v7.2_{}.jsonl.gz"
SIDECAR_BY_SPLIT = {"train": "C:/Users/Admin/tanitad-data/add203/out/refcv8_clip_clock_sidecar_ts33.jsonl", "eval": "E:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl"}   # the sidecar and manifest of the 4,572-clip v9 file training reads (md5 83e30bd3)
MAN = {"train": "C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt",
       "eval": "E:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt"}
CACHE_EVAL = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/manifest.jsonl"
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/{}.parquet"
SCHEMA = "tanitad.band_labels.strategic_geom/1"
N_DEFAULT = 101


def n_steps_from_T(T: int) -> int:
    """T provider rows -> t_ep = T + 2 raw frames -> cache steps (t_ep - 1) // 2 + 1 (T = 199 -> 101)."""
    return (T + 2 - 1) // 2 + 1


def release_ids(split: str) -> list:
    with gzip.open(REL.format(split), "rt", encoding="utf-8") as f:
        return [json.loads(l)["clip_id"] for l in f if l.strip()]


def clip_universe(split: str, limit: int = 0):
    """[(clip_id, g0, dt, clock_src, n_steps, in_v9_manifest)] in release order."""
    import torch
    ids = release_ids(split)
    side = B._read_sidecar(SIDECAR_BY_SPLIT[split])
    man_ids, man_poses = B._manifest(MAN[split])
    mp = dict(zip(man_ids, man_poses))
    cache = set()
    if split == "eval":
        cache = {json.loads(l)["clip"] for l in open(CACHE_EVAL, encoding="utf-8") if l.strip()}
    out = []
    for cid in (ids[:limit] if limit else ids):
        sid = B.stable_episode_id(cid)
        poses = mp.get(cid)
        if sid in side or poses is not None:
            g0, dt, src = B.clip_clock(sid, side, poses)
        else:
            g0, dt, src = 0.0, 0.1, "nominal_dt"
        n = n_steps_from_T(int(poses.shape[0])) if poses is not None else N_DEFAULT
        out.append((cid, g0, dt, src, n, poses is not None, cid in cache))
    return out


def build_clip(cid, g0, dt, n, ego=EGO):
    log = B.load_log(ego.format(cid))
    k = 2 * np.arange(n)
    nows = g0 + k * dt
    xyh, valid, why = CK.checkpoints(log, nows)
    _, valid_span, _ = CK.checkpoints(log, nows, g_max=float("inf"))
    gap = CK.max_gap_matrix(log.ts, nows).astype(np.float32)
    return k, nows, xyh, valid, why, log.t_end, valid_span, gap


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "eval"))
    ap.add_argument("--out-dir", default="F:/strat_out/band_labels")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress", default=None)
    a = ap.parse_args(argv)
    B.assert_no_g_imports()
    U = clip_universe(a.split, a.limit)
    cols = {k: [] for k in ("k", "now_s", "ck_xyh", "ck_valid", "ck_why", "ck_valid_span", "ck_gap_s")}
    tab = {k: [] for k in ("sid", "off", "n", "grid_start_s", "dt_s", "clock_src", "log_end_s", "in_v9_manifest", "in_cache141")}
    ids = []
    t0, off = time.time(), 0
    for i, (cid, g0, dt, src, n, inm, inc) in enumerate(U):
        k, nows, xyh, valid, why, t_end, valid_span, gap = build_clip(cid, g0, dt, n)
        for nm, v in (("k", k.astype(np.int16)), ("now_s", nows), ("ck_xyh", xyh.astype(np.float32)), ("ck_valid", valid), ("ck_why", why), ("ck_valid_span", valid_span), ("ck_gap_s", gap)):
            cols[nm].append(v)
        for nm, v in (("sid", B.stable_episode_id(cid)), ("off", off), ("n", n), ("grid_start_s", g0), ("dt_s", dt),
                      ("clock_src", ("sidecar", "pose_dt", "nominal_dt").index(src)), ("log_end_s", t_end),
                      ("in_v9_manifest", int(inm)), ("in_cache141", int(inc))):
            tab[nm].append(v)
        ids.append(cid)
        off += n
        if (i + 1) % 100 == 0 or i + 1 == len(U):
            msg = f"[strategic_geom {a.split}] {i + 1}/{len(U)} {time.time() - t0:.0f}s"
            print(msg, flush=True)
            if a.progress:
                with open(a.progress, "w", encoding="utf-8") as f:
                    f.write(msg + "\n")
    out = {"clip_ids": np.array(ids), "horizons_s": np.array(CK.HORIZONS_S, np.float32)}
    out.update({k_: np.concatenate(v) for k_, v in cols.items()})
    out.update({"clip__" + k_: np.asarray(v) for k_, v in tab.items()})
    os.makedirs(a.out_dir, exist_ok=True)
    npz = os.path.join(a.out_dir, f"strategic_geom_{a.split}.npz")
    np.savez_compressed(npz, **out)
    md5 = B.file_md5(npz)
    man = {"schema": SCHEMA, "split": a.split, "n_clips": len(ids), "n_rows": int(off),
           "keys": sorted(out), "npz_md5": md5, "wall_s": round(time.time() - t0, 1),
           "horizons_s": list(CK.HORIZONS_S), "why_codes": CK.WHY,
           "clock_rule": "now_s = grid_start_s + k * dt_s, k = 2 * cache_step (v9 rule; sidecar, else pose_dt, else nominal)",
           "literals": {k_: (list(v_) if isinstance(v_, tuple) else v_) for k_, v_ in B.LITERALS.items() if k_.startswith(("rc_", "g_max", "track"))},
           "sources": {"release_clip_list": [REL.format(a.split), B.file_md5(REL.format(a.split))],
                       "sidecar": [SIDECAR_BY_SPLIT[a.split], B.file_md5(SIDECAR_BY_SPLIT[a.split])], "manifest": [MAN[a.split], B.file_md5(MAN[a.split])],
                       "ego_store": EGO.split("{")[0]},
           "code": {"build_v9_verbatim.py": B.file_md5(B.__file__), "strategic_ck.py": B.file_md5(CK.__file__)}}
    with open(os.path.join(a.out_dir, f"strategic_geom_{a.split}.manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(f"[strategic_geom {a.split}] done -> {npz} md5 {md5}")


if __name__ == "__main__":
    main()

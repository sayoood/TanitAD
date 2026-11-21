"""extract_eval_agents_r8.py -- pull the eval139 clips' GT agent boxes out of the B1 3-D agent join into ONE small pickle.

CPU only, streaming (never holds the 391 MB join in memory).  Output keys are SHORT EPISODE INDICES (never clip UUIDs):

    {clip_index: {raw_frame: ndarray[n, 8] float32 = (cx, cy, yaw, l, w, occ, cz, h)  +  cls list}}

Frame key = the join's `frame` = the RAW v2ep frame index (build_b1_agent_join.py docstring: `frame` is RAW, `frame_idx` is the post-trim
index = frame - (n_stack - 1)).  Agents are in the per-frame EGO frame (+x forward, +y LEFT).  An ABSENT (clip, frame) line is NO_LABEL;
an EMPTY agents list is "labelled clear" -- the two are kept distinct (absent frames are simply not in the dict).

    python extract_eval_agents_r8.py --out <dir>/eval139_agents3d.pkl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import pickle
import sys
from pathlib import Path

import numpy as np

JOIN3D = "E:/refcv6_eval_kit/data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz"
EVAL_DIR = "E:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139"
BATTERY_SUMMARY = "C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc/battery_summary.json"


def clip_table():
    """{clip_id: clip_index} from the file names (UUIDs stay in memory only) + the battery's clip_sha12_by_index."""
    by = json.load(open(BATTERY_SUMMARY, encoding="utf-8"))["stages"]["roll_s0"]["extras"]["clip_sha12_by_index"]
    inv = {v: int(k) for k, v in by.items()}
    out = {}
    for f in os.listdir(EVAL_DIR):
        if not f.endswith(".v2ep.pt"):
            continue
        cid = f.split(".")[0]
        s12 = hashlib.sha256(cid.encode()).hexdigest()[:12]
        if s12 in inv:
            out[cid] = inv[s12]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--join", default=JOIN3D)
    a = ap.parse_args()
    ct = clip_table()
    print(f"[extract] {len(ct)} eval clips mapped to indices", flush=True)
    res = {i: {} for i in ct.values()}
    n_lines = n_ag = 0
    with lzma.open(a.join, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            ci = ct.get(rec["clip_id"])
            if ci is None:
                continue
            n_lines += 1
            ag = rec["agents"]
            arr = np.zeros((len(ag), 8), np.float32)
            cls = []
            for k, x in enumerate(ag):
                arr[k] = (x["cx"], x["cy"], x["yaw"], x["l"], x["w"], x.get("occ", 0), x.get("cz", 0.0), x.get("h", 1.5))
                cls.append(x.get("cls", "?"))
            n_ag += len(ag)
            res[ci][int(rec["frame"])] = (arr, cls)
    cov = {i: len(v) for i, v in res.items()}
    print(f"[extract] lines={n_lines} agents={n_ag} clips_with_lines={sum(1 for v in cov.values() if v)} of {len(cov)}", flush=True)
    with open(a.out, "wb") as f:
        pickle.dump({"agents": res, "coverage_frames_per_clip_index": cov,
                     "frame_key": "RAW v2ep frame index (== stacked row + n_stack - 1, n_stack 3)",
                     "frame_convention": "ego frame at that frame, +x forward, +y left",
                     "columns": ["cx", "cy", "yaw", "l", "w", "occ", "cz", "h"], "source": os.path.basename(a.join)}, f)
    print("[extract] wrote", a.out, os.path.getsize(a.out), "bytes", flush=True)


if __name__ == "__main__":
    sys.exit(main())

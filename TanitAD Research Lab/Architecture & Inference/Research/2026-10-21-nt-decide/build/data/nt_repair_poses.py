#!/usr/bin/env python3
"""nt_repair_poses.py -- v1.2 repair: recompute poses/actions of every token-episode already written (yaw convention fix in
``nt_e1.quat_yaw``), keep the PNG frames byte-for-byte, re-verify, and rewrite MANIFEST_A1.jsonl atomically.
Run ONLY while nt_e1_run.py is stopped.  Prints ZZREPAIR_DONE n=<n> max_shift_m=<x> / ZZREPAIR_FAIL_<why>."""
from __future__ import annotations

import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import nt_e1 as E  # noqa: E402

NT = "/home/nvidia/r8ns/nt"


def main() -> int:
    torch.set_num_threads(2)
    idx = json.load(gzip.open(f"{NT}/TOKEN_INDEX_A1.json.gz", "rt"))["index"]
    recs = json.load(gzip.open(f"{NT}/inputs/a1_inputs.json.gz", "rt"))["tokens"]
    rows = [json.loads(ln) for ln in open(f"{NT}/MANIFEST_A1.jsonl") if ln.strip()]
    rows.sort(key=lambda r: (idx[r["token"]]["log"], idx[r["token"]]["pos"]))
    cache = (None, None)
    out, shifts = [], []
    for i, r in enumerate(rows):
        t = r["token"]
        ln = idx[t]["log"]
        if cache[0] != ln:
            cache = (ln, E.N.load_log_pickle(f"/home/nvidia/r8ns/navtrain/navsim_logs/trainval/{ln}.pkl")[0])
        path = f"{NT}/v2ep_a1/{r['file']}"
        d = torch.load(path, map_location="cpu", weights_only=False)
        tr = E.token_poses(recs[t], cache[1], idx[t]["pos"])
        old = d["poses"].numpy()
        if old.shape != tr["poses"].shape or not np.array_equal(old[:10], tr["poses"][:10]):
            print(f"ZZREPAIR_FAIL_history_rows_changed {t}", flush=True)
            return 3
        shifts.append(float(np.abs(old[10:, :2] - tr["poses"][10:, :2]).max()))
        d["poses"] = torch.from_numpy(tr["poses"])
        d["actions"] = torch.from_numpy(tr["actions"])
        d["nt"]["schema"] = E.SCHEMA
        d["nt"]["writer"] = {"tool": "nt_e1.py", "sha256": E.code_sha256(), "repaired_by": "nt_repair_poses.py (v1.2 yaw)"}
        nb = E.save_atomic(d, path)
        law_sha = r["law_frame_sha16"]
        ver = E.verify_file(path, r["t0_frame_sha16"], law_sha, tr)
        if ver["problems"]:
            print(f"ZZREPAIR_FAIL_verify {t} {ver['problems']}", flush=True)
            return 4
        r2 = dict(r)
        r2.update({"bytes": nb, "file_sha256": E.file_sha256(path), "repaired": "v1.2-yaw"})
        out.append(r2)
        if (i + 1) % 1000 == 0:
            print(f"[repair] {i + 1}/{len(rows)} max_shift_m={max(shifts):.4f}", flush=True)
    tmp = f"{NT}/MANIFEST_A1.jsonl.repair.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r2 in out:
            fh.write(json.dumps(r2) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(f"{NT}/MANIFEST_A1.jsonl", f"{NT}/MANIFEST_A1.pre_v1.2_yaw.jsonl")
    os.replace(tmp, f"{NT}/MANIFEST_A1.jsonl")
    s = np.asarray(shifts)
    print(f"ZZREPAIR_DONE n={len(out)} max_shift_m={s.max():.5f} p50={np.median(s):.5f} p99={np.quantile(s, 0.99):.5f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

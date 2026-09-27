#!/usr/bin/env python3
"""ov_probe.py -- overlay audit, step 0: WHAT data exists for a LiDAR clip (by 8-hex prefix), read-only.

Prints numbers and sha12s only (never a raw clip id). Run on Thor with the tanitad-train venv.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

C8 = sys.argv[1]  # the 8-hex file prefix ON THOR, runtime only; outputs carry sha12
D = "/home/nvidia/data"
S = "/home/nvidia/sam3map/data"


def sha12(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def scrub(s):
    import re
    return re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<uuid>", str(s))


lid = sorted(glob.glob(f"{S}/{C8}-*.lidar_top_360fov.parquet"))
cid = os.path.basename(lid[0]).split(".")[0] if lid else None
print("clip sha12", sha12(cid) if cid else None)
caches = sorted(glob.glob(f"{D}/refcv6-b1-416x1024-*/{cid}*.v2ep.pt")) if cid else []
print("416 caches:", [scrub(c) for c in caches])
fw = sorted(glob.glob(f"{S}/frontwide/{cid}*")) + sorted(glob.glob(f"{S}/native7/{cid}*front_wide*"))
print("front-wide raw:", [scrub(os.path.basename(x)) for x in fw])
sw = sorted(glob.glob(f"/home/nvidia/sam3map/data/sweeps/{C8}/*.npy"))
print("sweeps:", len(sw), os.path.basename(sw[0]) if sw else None, os.path.basename(sw[-1]) if sw else None)

import pyarrow.parquet as pq
import pandas as pd
if lid:
    pf = pq.ParquetFile(lid[0])
    print("lidar schema:", [f.name for f in pf.schema_arrow])
    sp = pq.read_table(lid[0], columns=["spin_start_timestamp", "spin_end_timestamp"]).to_pydict()
    a = np.asarray(sp["spin_start_timestamp"], np.int64)
    b = np.asarray(sp["spin_end_timestamp"], np.int64)
    print("n spins", len(a), "spin dur ms median", float(np.median(b - a)) / 1e3, "span s", (b[-1] - a[0]) / 1e6,
          "t0 us", int(a[0]))
for kind in ("sensor_extrinsics", "camera_intrinsics"):
    for p in sorted(glob.glob(f"{S}/calib/{kind}.chunk_*.parquet")) + sorted(glob.glob(f"{S}/{kind}.chunk_*.parquet")):
        t = pd.read_parquet(p)
        t = t.reset_index()
        if "clip_id" in t.columns and cid in set(t["clip_id"]):
            rows = t[t["clip_id"] == cid]
            print(kind, os.path.basename(p), "columns", list(rows.columns))
            for _, r in rows.iterrows():
                d = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else scrub(v))
                     for k, v in r.items() if k != "clip_id"}
                print("  ", json.dumps(d))
            break
ext = json.load(open(f"{D}/refcv6_train_eval139_extrinsics.json", encoding="utf-8"))
e = ext.get(cid)
print("renderer extrinsics table entry:", None if e is None else json.dumps({k: v for k, v in e.items()
                                                                            if k != "source"}, default=str)[:600])
import torch
if caches:
    pay = torch.load(caches[0], map_location="cpu", weights_only=False, mmap=True)
    print("payload keys:", sorted(pay.keys()))
    for k in ("frame", "codec", "n_stack", "fps", "dt", "t_query", "t_cam", "meta"):
        if k in pay:
            v = pay[k]
            if torch.is_tensor(v):
                print(" ", k, tuple(v.shape), v.dtype, v.reshape(-1)[:4].tolist())
            else:
                print(" ", k, scrub(json.dumps(v, default=str))[:400])
    for k, v in pay.items():
        if torch.is_tensor(v):
            print("  tensor", k, tuple(v.shape), v.dtype)
        elif isinstance(v, (list, tuple)):
            print("  list", k, len(v))

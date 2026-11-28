#!/usr/bin/env python3
"""Merge a log's GPU dump and CPU labels into the final cache file W/out/<log>.npz, validate it, and append its token rows to W/index.csv.

    sh_merge.py --log <log> --split train|heldout
Validation (FAIL LOUD, nothing written on failure): every token of the final file has a dump record AND a label record (tokens missing a label are
dropped and listed in the report, never filled); shapes props [n,64,20,3], s [n,64,256], logits [n,64,6], tr_* [n,64]; all finite; tr_* in [0,1];
pick in 0..63. The file is written with fixed zip timestamps (identical bytes for identical content) and its sha256 goes to the index.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sh_lib as L  # noqa: E402


def merge(log, split, dump_dir=None, lab_dir=None, out_dir=None, index=None):
    dump_dir, lab_dir, out_dir = dump_dir or L.wp("dump"), lab_dir or L.wp("lab"), out_dir or L.wp("out")
    index = index or L.wp("index.csv")
    d = np.load(os.path.join(dump_dir, f"{log}.dump.npz"))
    lb = np.load(os.path.join(lab_dir, f"{log}.lab.npz"))
    dt = [str(t) for t in d["token"]]
    lt = {str(t): i for i, t in enumerate(lb["token"])}
    keep = [i for i, t in enumerate(dt) if t in lt]
    dropped = [dt[i] for i in range(len(dt)) if i not in set(keep)]
    li = [lt[dt[i]] for i in keep]
    n = len(keep)
    if n == 0:
        raise SystemExit(f"FAIL {log}: no token has both a dump record and a label record")
    out = {"token": np.array([dt[i] for i in keep]), "log": np.array(log), "split": np.array(split),
           "props": d["props"][keep], "props8": d["props8"][keep], "s": d["s"][keep], "logits": d["logits"][keep], "pick": d["pick"][keep], "meta": d["meta"]}
    for k in L.LAB_KEYS:
        out[k] = lb[k][li]
    shp = {"props": (n, L.N_PROP, L.N_STEP, 3), "s": (n, L.N_PROP, L.S_DIM), "logits": (n, L.N_PROP, L.N_HEAD), **{k: (n, L.N_PROP) for k in L.LAB_KEYS}}
    for k, s in shp.items():
        if out[k].shape != s:
            raise SystemExit(f"FAIL {log}: {k} shape {out[k].shape} != {s}")
    for k in ("props", "props8", "s", "logits", *L.LAB_KEYS):
        if not np.isfinite(out[k]).all():
            raise SystemExit(f"FAIL {log}: non-finite values in {k}")
    for k in L.LAB_KEYS:
        if out[k].min() < -1e-6 or out[k].max() > 1 + 1e-6:
            raise SystemExit(f"FAIL {log}: {k} outside [0,1] ({out[k].min()}, {out[k].max()})")
    if out["pick"].min() < 0 or out["pick"].max() >= L.N_PROP:
        raise SystemExit(f"FAIL {log}: pick outside 0..63")
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"{log}.npz").replace("\\", "/")
    L.save_npz_det(fp, **out)
    sha = L.sha256_file(fp)
    rows = [(t, log, split, os.path.basename(fp), sha) for t in out["token"]]
    existing = set()
    if os.path.exists(index):
        existing = {r[0] + "|" + r[1] for r in csv.reader(open(index, encoding="utf-8"))}
    new = not os.path.exists(index)
    with open(index, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if new:
            w.writerow(["token", "log", "split", "file", "sha256"])
        for r in rows:
            if r[0] + "|" + r[1] not in existing:
                w.writerow(r)
    rep = {"log": log, "split": split, "records": n, "dropped_no_label": dropped, "file": fp, "sha256": sha, "bytes": os.path.getsize(fp)}
    json.dump(rep, open(os.path.join(out_dir, f"{log}.merge.json"), "w"), indent=0)
    print("ZZMERGE " + json.dumps({k: v for k, v in rep.items() if k != "dropped_no_label"}) + f" dropped {len(dropped)}", flush=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    merge(a.log, a.split)

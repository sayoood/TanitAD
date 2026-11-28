#!/usr/bin/env python3
"""Measure 5 effectiveness run (eval/PREREG_MEASURE5.md §3a): the navtrain sample draw, fixed by the pre-registration.

  rows    rank-0 rows of D:/Projects/TanitAD/data/refe_navtrain10/r0/targets_rank0.jsonl whose log has a local navtrain DB
          (D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval)
  split   the DB-local logs, sorted, permuted with np.random.default_rng(20260927): the first 44 are VAL, the other 170 TRAIN
  draw    the same generator permutes each split's rows (sorted by log|token|step); TRAIN takes the first 1,600, VAL the first
          300 -- the permutation ORDER is kept, so a row that fails the pod gate is replaced by the next one in that order
Writes <out>/samples.json: {train: [...], val: [...], order_train: [...], order_val: [...], logs_train, logs_val, source sha}.
Each sample: {key: "log|token|step|rank", log_name, token, step, rank, row_sha (key + teacher traj), row_line_index}.

    python eval/m5_samples.py --out D:/Projects/TanitAD/data/refe_m5
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np

SRC = "D:/Projects/TanitAD/data/refe_navtrain10/r0/targets_rank0.jsonl"
DBS = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"
SEED = 20260927
N_VAL_LOGS, N_TRAIN, N_VAL = 44, 1600, 300


def row_sha(r: dict) -> str:
    """The pre-registered gate's content: the key and the teacher trajectory, as values (json repr of the floats)."""
    blob = json.dumps([r["log_name"], r["token"], int(r["step"]), int(r.get("rank", 0)), r["traj"]],
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    local = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(DBS, "*.db"))}
    rows = []
    with open(SRC, encoding="utf-8") as f:
        for li, line in enumerate(f):
            r = json.loads(line)
            if int(r.get("rank", 0)) != 0 or r["log_name"] not in local:
                continue
            rows.append({"key": f"{r['log_name']}|{r['token']}|{int(r['step'])}|0", "log_name": r["log_name"],
                         "token": r["token"], "step": int(r["step"]), "rank": 0, "row_sha": row_sha(r),
                         "row_line_index": li})
    rows.sort(key=lambda x: x["key"])
    if len({x["key"] for x in rows}) != len(rows):
        raise SystemExit("duplicate keys among the eligible rows")
    logs = sorted({x["log_name"] for x in rows})
    rng = np.random.default_rng(SEED)
    perm = [logs[i] for i in rng.permutation(len(logs))]
    val_logs, train_logs = sorted(perm[:N_VAL_LOGS]), sorted(perm[N_VAL_LOGS:])
    out = {"seed": SEED, "source": SRC, "source_sha256": hashlib.sha256(open(SRC, "rb").read()).hexdigest(),
           "db_dir": DBS, "eligible_rows": len(rows), "eligible_logs": len(logs),
           "logs_val": val_logs, "logs_train": train_logs}
    for name, lg, n in (("train", set(train_logs), N_TRAIN), ("val", set(val_logs), N_VAL)):
        pool = [x for x in rows if x["log_name"] in lg]
        order = [pool[i] for i in rng.permutation(len(pool))]
        out[f"order_{name}"] = order
        out[name] = order[:n]
        out[f"pool_{name}"] = len(pool)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "samples.json"), "w", encoding="utf-8"), indent=0)
    print(f"  eligible rows {len(rows):,} over {len(logs)} DB-local logs; train logs {len(train_logs)} "
          f"(pool {out['pool_train']:,}), val logs {len(val_logs)} (pool {out['pool_val']:,}); "
          f"drawn {len(out['train'])} + {len(out['val'])}")
    print("ZZM5_SAMPLES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

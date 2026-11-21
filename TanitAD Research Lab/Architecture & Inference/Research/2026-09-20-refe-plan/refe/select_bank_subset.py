#!/usr/bin/env python3
"""Pick a seeded random subset of the on-policy sets the TRAINER ACTUALLY SERVES and write their QUEUE rows (proposals +
teacher rollout) as subset queue files, so a relabeller labels exactly those sets.

"Served" = train.OnPolicyBank's rule: per (log, token, step, rank) the line with the largest (ckpt_step, label_version);
held-out logs excluded. Only the key fields are read from the set files (regex on the line head), so the 11 GB of sets
stream in minutes.
    python refe/select_bank_subset.py --sets <dir> --queue <dir> --heldout-logs <txt> --n 15000 --seed 20261004 --out <dir>
Writes <out>/props_r{rank}_subset.jsonl and <out>/subset_keys.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

RX = {k: re.compile(r'"%s":\s*("([^"]*)"|-?\d+)' % k) for k in ("log_name", "token", "step", "rank", "ckpt_step", "label_version")}


def head_fields(line: str) -> dict:
    out = {}
    for k, rx in RX.items():
        m = rx.search(line)
        if m:
            out[k] = m.group(2) if m.group(2) is not None else int(m.group(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--heldout-logs", required=True)
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ho = set(open(a.heldout_logs, encoding="utf-8").read().split())
    best: dict = {}
    for fp in sorted(glob.glob(os.path.join(a.sets, "onpolicy_*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                if '"onpolicy_set"' not in line[:400]:
                    pass
                f = head_fields(line[:2000] + line[-400:])
                if not {"log_name", "step", "ckpt_step"} <= set(f) or f["log_name"] in ho:
                    continue
                k = (f["log_name"], f.get("token", ""), int(f["step"]), int(f.get("rank", 0)))
                rk = (int(f["ckpt_step"]), int(f.get("label_version", 1)))
                if k not in best or best[k] < rk:
                    best[k] = rk
    keys = sorted(best)
    random.Random(a.seed).shuffle(keys)
    pick = {k: best[k][0] for k in keys[: a.n]}
    os.makedirs(a.out, exist_ok=True)
    outs = {}
    found = 0
    for fp in sorted(glob.glob(os.path.join(a.queue, "props_*.jsonl*"))):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                f = head_fields(line[:3000] + line[-600:])
                if not {"log_name", "step", "ckpt_step"} <= set(f):
                    continue
                k = (f["log_name"], f.get("token", ""), int(f["step"]), int(f.get("rank", 0)))
                if pick.get(k) == int(f["ckpt_step"]):
                    r = int(f.get("rank", 0))
                    if r not in outs:
                        outs[r] = open(os.path.join(a.out, f"props_r{r}_subset.jsonl"), "w", encoding="utf-8")
                    outs[r].write(line if line.endswith("\n") else line + "\n")
                    found += 1
                    pick[k] = -1                                  # take each set once
    for fh in outs.values():
        fh.close()
    json.dump({"served_keys": len(best), "picked": len(keys[: a.n]), "queue_rows_found": found, "seed": a.seed,
               "keys": [list(k) for k in keys[: a.n]]}, open(os.path.join(a.out, "subset_keys.json"), "w"))
    print(f"ZZSUBSET served {len(best)} picked {min(a.n, len(keys))} queue_rows_found {found}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

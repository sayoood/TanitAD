#!/usr/bin/env python3
"""A SMALL on-policy bank for a memory-light launch-gate smoke: every line (all checkpoint / label versions) of the sets
whose key appears in a label side-file directory, so train.OnPolicyBank's served rule (largest (ckpt_step,
label_version) per key) picks exactly what it would pick from the full bank.

Why (2026-10-04): a full-bank smoke beside a live scorer fine-tune pushed the pod's 50 GB cgroup over its limit and the
OOM killer took the fine-tune (SFT-1, update 2,550 / 2,733). A smoke only has to exercise the gates on real rows, so it
loads the few thousand sets that carry the labels being tested instead of all 177,836.
    python refe/smoke_bank_subset.py --sets <dir> --labels <dir> [--labels <dir> ...] --max-per-dir 1500 --out <dir>
Writes <out>/onpolicy_smoke.jsonl (the name matches train._onpolicy_files) and prints ZZSMOKEBANK <keys> <lines>.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from select_bank_subset import head_fields


def label_keys(path: str, cap: int) -> set:
    out = set()
    for fp in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                if len(out) >= cap:
                    return out
                try:
                    k = json.loads(line).get("key")
                except json.JSONDecodeError:
                    continue
                if isinstance(k, list) and len(k) == 4:
                    out.add((k[0], k[1], int(k[2]), int(k[3])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--labels", action="append", required=True)
    ap.add_argument("--max-per-dir", type=int, default=1500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keys = set()
    per_dir = {}
    for d in a.labels:
        ks = label_keys(d, a.max_per_dir)
        per_dir[d] = len(ks)
        keys |= ks
    os.makedirs(a.out, exist_ok=True)
    n_lines = 0
    hit = set()
    with open(os.path.join(a.out, "onpolicy_smoke.jsonl"), "w", encoding="utf-8") as out:
        for fp in sorted(glob.glob(os.path.join(a.sets, "onpolicy_*.jsonl"))):
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    f = head_fields(line[:2000] + line[-400:])
                    if not {"log_name", "step"} <= set(f):
                        continue
                    k = (f["log_name"], f.get("token", ""), int(f["step"]), int(f.get("rank", 0)))
                    if k in keys:
                        out.write(line if line.endswith("\n") else line + "\n")
                        n_lines += 1
                        hit.add(k)
    json.dump({"label_keys_per_dir": per_dir, "keys": len(keys), "keys_found": len(hit), "lines": n_lines},
              open(os.path.join(a.out, "smoke_bank.json"), "w"))
    print(f"ZZSMOKEBANK keys {len(keys)} found {len(hit)} lines {n_lines}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Merge shard bank files into ONE: every line parsed, torn lines dropped, duplicate keys dropped.

⛔ WHY NOT `cat`. The shard builders append as they go and a killed process can leave a TORN last
line -- `json.loads` in the trainer would then die on it at load time, days later. And a shard
count changed between launches (the pod's throughput tuning does exactly that) can put one row in
two shard files. `cat` passes both straight to the trainer; the trainer now refuses a duplicated
scene-rank (R24), which is correct but late. This merge makes the bank clean at the source and
SAYS what it dropped.

  python merge_bank.py --key log_name,token,step,rank --out r0/targets_rank0.jsonl "r0_s*/targets_rank0.jsonl"
Lines are written byte-for-byte as found (first occurrence wins); the output is replaced atomically.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="comma-separated row fields that identify a row")
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+", help="files or glob patterns")
    a = ap.parse_args(argv)
    keys = [k.strip() for k in a.key.split(",") if k.strip()]
    files = sorted({f for g in a.inputs for f in (glob.glob(g) or [])
                    if os.path.abspath(f) != os.path.abspath(a.out)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    seen, n_in, torn, dup = set(), 0, 0, 0
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as g:
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    n_in += 1
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        torn += 1
                        continue
                    k = tuple(r.get(x) for x in keys)
                    if k in seen:
                        dup += 1
                        continue
                    seen.add(k)
                    g.write(line if line.endswith("\n") else line + "\n")
    os.replace(tmp, a.out)
    print(f"  merged {len(seen):,} rows from {len(files)} file(s) -> {a.out}   "
          f"(read {n_in:,}; torn {torn}; duplicate keys {dup})", flush=True)
    return 0 if files else 1


if __name__ == "__main__":
    sys.exit(main())

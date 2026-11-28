#!/usr/bin/env python3
"""A SMALL bank for p2_gate.py in which G3's mutation is NOT inert (eval/PREREG_P2_AMENDMENT_1.md, section 2).

p2_gate.py's G3 needs sets WITHOUT PDM targets in the bank it is given: its mutation arm (the same bank without
--pdm-only) must show them, else it prints ok=false. After a COMPLETE relabel every served set carries a label, so the
real train_pdm directory can never exercise G3. This builds both inputs from the real ones:

  <out-sets>/onpolicy_gate.jsonl   every line (all ckpt / label versions) of --n-unlabelled sets whose label is WITHHELD
                                   (the first sets of the stream) and of --n-labelled sets that HAVE a PDM label;
  <out-pdm>/pdm_gate.jsonl         the PDM label lines of the labelled sets ONLY (copied byte for byte).

The withheld sets are chosen FIRST and the labelled ones from the remaining labels, so a withheld set exists whenever
the stream holds enough sets at all. The gate then runs on real rows, a few dozen sets (no 50 GB bank load beside the
pod's RAM), and the withheld sets are exactly the "set without a PDM target" case.
Prints ZZGATEPREP <labelled> <withheld> <lines>; exit 1 if the gate's minimum (2 labelled, 1 withheld) cannot be met.
    python p2r_gate_prep.py --sets <sets dir> --pdm <train_pdm dir> --out-sets <dir> --out-pdm <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from select_bank_subset import head_fields  # noqa: E402


def key_of(line):
    f = head_fields(line[:2000] + line[-400:])
    if not {"log_name", "step"} <= set(f):
        return None
    return (f["log_name"], f.get("token", ""), int(f["step"]), int(f.get("rank", 0)))


def first_keys(files, n):
    """The first n distinct set keys of the stream (file order, line order)."""
    out = []
    seen = set()
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                k = key_of(line)
                if k is None or k in seen:
                    continue
                seen.add(k)
                out.append(k)
                if len(out) >= n:
                    return out
    return out


def first_labels(pdm_dir, n, exclude=()):
    """{key: line} for the first n parseable PDM label lines whose key is not in `exclude`."""
    out = {}
    for fp in sorted(glob.glob(os.path.join(pdm_dir, "pdm_*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                if len(out) >= n:
                    return out
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = d.get("key")
                if not (isinstance(k, list) and len(k) == 4 and "ckpt_step" in d):
                    continue
                k = (k[0], k[1], int(k[2]), int(k[3]))
                if k in exclude:
                    continue
                out[k] = line if line.endswith("\n") else line + "\n"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--pdm", required=True)
    ap.add_argument("--out-sets", required=True)
    ap.add_argument("--out-pdm", required=True)
    ap.add_argument("--n-labelled", type=int, default=24)
    ap.add_argument("--n-unlabelled", type=int, default=8)
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.sets, "onpolicy_*.jsonl")))
    withheld = set(first_keys(files, a.n_unlabelled))
    lab = first_labels(a.pdm, a.n_labelled, exclude=withheld)
    if not lab or not files:
        print(f"ZZGATEPREP_FAIL labels={len(lab)} set_files={len(files)}", flush=True)
        return 1
    os.makedirs(a.out_sets, exist_ok=True)
    os.makedirs(a.out_pdm, exist_ok=True)
    found, n_lines = set(), 0
    with open(os.path.join(a.out_sets, "onpolicy_gate.jsonl"), "w", encoding="utf-8", newline="\n") as out:
        for fp in files:
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    k = key_of(line)
                    if k is None or not (k in lab or k in withheld):
                        continue
                    if k in lab:
                        found.add(k)
                    out.write(line if line.endswith("\n") else line + "\n")
                    n_lines += 1
    with open(os.path.join(a.out_pdm, "pdm_gate.jsonl"), "w", encoding="utf-8", newline="\n") as out:
        for k in sorted(found):                                       # labels of the labelled sets ONLY
            out.write(lab[k])
    print(f"ZZGATEPREP {len(found)} {len(withheld)} {n_lines}", flush=True)
    return 0 if len(found) >= 2 and len(withheld) >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())

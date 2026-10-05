#!/usr/bin/env python3
"""Thor side: merge K token-disjoint navtest shard runs of ONE arm into one arm directory (any python + pandas).

    python finalize_navtest_arm.py --label r7thor_s50400_R7_VMAXOFF --shard-root /dev/shm/navsim/thor_scores/navtest/_shards \
        --k 4 --out /dev/shm/navsim/thor_scores/navtest --order-ref /dev/shm/navsim/subsets/orderref_navtest.csv

Refuses unless EVERY shard's counts.json reads PASS (W3 guards: C1 PDMS identity, C2 counts, C3 token set,
seam calls), shard token sets are disjoint and their union is the full split (12,146), and all shards
used the same seam. Then ``merge_shards.py navtest`` writes ``<label>/<label>.csv`` (devkit's v1.1
``average`` row recomputed) and a merged ``<label>.counts.json`` in W3's shape (status, summary,
summary_x100_4dp, per-shard records).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TERMS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
         "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score")
SHORT = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "ego_progress": "EP",
         "time_to_collision_within_bound": "TTC", "comfort": "C", "driving_direction_compliance": "DDC", "score": "PDMS"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--shard-root", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order-ref", required=True)
    ap.add_argument("--expected", type=int, default=12146)
    a = ap.parse_args()
    fails, shards, csvs, toks, seams = [], [], [], [], set()
    for k in range(a.k):
        lab = f"{a.label}_k{k}of{a.k}"
        d = os.path.join(a.shard_root, lab)
        try:
            c = json.load(open(os.path.join(d, f"{lab}.counts.json"), encoding="utf-8"))
            t = json.load(open(os.path.join(d, f"{lab}.thor.json"), encoding="utf-8"))
        except (OSError, ValueError) as e:
            fails.append(f"shard {k}: {e!r}")
            continue
        sub = json.load(open(t["tokens"], encoding="utf-8"))["tokens"]
        shards.append({"k": k, "status": c["status"], "n": c.get("csv_valid_rows"), "agent_calls": c.get("agent_calls"),
                       "wall_s": c.get("wall_s"), "C1_max_abs_delta": c.get("C1_max_abs_delta"), "failures": c.get("failures")})
        if c["status"] != "PASS":
            fails.append(f"shard {k}: {c['status']} {c.get('failures')}")
        toks += sub
        seams.add(t.get("seam_sha256"))
        csvs.append(os.path.join(d, f"{lab}.csv"))
    if len(seams) != 1:
        fails.append(f"{len(seams)} distinct seams across shards")
    if len(toks) != a.expected or len(set(toks)) != len(toks):
        fails.append(f"shard token union {len(set(toks))} (rows {len(toks)}) != {a.expected}")
    odir = os.path.join(a.out, a.label)
    os.makedirs(odir, exist_ok=True)
    csv_p = os.path.join(odir, f"{a.label}.csv")
    merge, summ = None, None
    if not fails:
        r = subprocess.run([sys.executable, os.path.join(HERE, "merge_shards.py"), "navtest", "--csvs", *csvs,
                            "--order-ref", a.order_ref, "--out-csv", csv_p], capture_output=True, text=True)
        merge = {"rc": r.returncode, "stdout": r.stdout.strip()[-300:], "stderr": r.stderr.strip()[-300:]}
        if r.returncode != 0:
            fails.append("merge failed")
        else:
            import pandas as pd
            df = pd.read_csv(csv_p, index_col=0, float_precision="round_trip")
            tok = df[df["token"] != "average"]
            summ = {SHORT[t]: float(tok[t].mean()) for t in TERMS}
            summ["devkit_average_row_score"] = float(df[df["token"] == "average"]["score"].iloc[0])
            summ["n"] = int(len(tok))
    rep = {"label": a.label, "backend": "thor", "status": "FAIL" if fails else "PASS", "failures": fails,
           "n_expected": a.expected, "csv_valid_rows": (summ or {}).get("n"), "seam_sha256": (next(iter(seams)) if len(seams) == 1 else None),
           "agent_calls": {"seam": sum((s.get("agent_calls") or {}).get("seam", 0) for s in shards)},
           "summary": summ, "summary_x100_4dp": ({k: round(100 * v, 4) for k, v in summ.items() if k in SHORT.values()} if summ else None),
           "shards": shards, "merge": merge,
           "csv_sha256": (hashlib.sha256(open(csv_p, "rb").read()).hexdigest() if os.path.exists(csv_p) else None),
           "how": f"{a.k} token-disjoint shards (sorted navtest tokens, index % {a.k}) scored by score_navtest_thor.py "
                  "(W3 cmd_score unchanged), merged by merge_shards.py navtest (v1.1 main() average row verbatim)",
           "policy": "RULING_BACKEND_POLICY.md (2026-10-04-navsim-thor-backend)"}
    json.dump(rep, open(os.path.join(odir, f"{a.label}.counts.json"), "w", encoding="utf-8"), indent=1)
    json.dump({"backend": "thor", "seam_sha256": rep["seam_sha256"], "merged_from_shards": a.k},
              open(os.path.join(odir, f"{a.label}.thor.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in ("label", "status", "failures", "csv_valid_rows", "summary_x100_4dp")}))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())

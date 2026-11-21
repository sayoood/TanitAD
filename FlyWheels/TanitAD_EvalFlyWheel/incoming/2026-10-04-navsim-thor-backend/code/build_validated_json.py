#!/usr/bin/env python3
"""Assemble ``raw/THOR_BACKEND_VALIDATED.json`` FROM ARTIFACTS ONLY, and refuse unless every
pre-registered exact-reproduction diff in ``--diffs`` reads EXACT (any python).

    python build_validated_json.py --pkg <package dir> --diffs raw/V-NH_final_frame_diff.json ... \
        --throughput raw/throughput_*.json --extra raw/tree_hashes.json --out raw/THOR_BACKEND_VALIDATED.json

The verdict is computed here from the diff files' own ``verdict`` / ``max_abs_diff_over_all_columns``
/ ``n_cells_text_differ_total`` fields -- never typed by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--diffs", nargs="+", required=True)
    ap.add_argument("--throughput", nargs="*", default=[])
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    table, ok = [], True
    for d in a.diffs:
        j = json.load(open(os.path.join(a.pkg, d), encoding="utf-8"))
        row = {"diff_file": d, "ref": j["ref"].split("/raw/milestones/")[-1], "got": j["got"].split("/raw/")[-1],
               "n_tokens": j["n_tokens_compared"], "n_columns": len(j["columns_compared"]),
               "max_abs_diff": j["max_abs_diff_over_all_columns"],
               "n_cells_text_differ": j["n_cells_text_differ_total"], "verdict": j["verdict"],
               "per_column_max_abs": {c: v["max_abs_diff"] for c, v in j["per_column"].items()}}
        ok &= (j["verdict"] == "EXACT" and j["max_abs_diff_over_all_columns"] == 0.0
               and j["n_cells_text_differ_total"] == 0 and j["n_tokens_compared"] > 0)
        table.append(row)
    code = {f: sha(os.path.join(a.pkg, "code", f)) for f in sorted(os.listdir(os.path.join(a.pkg, "code")))
            if f.endswith(".py")}
    out = {"verdict": "VALIDATED_EXACT" if ok else "NOT_VALIDATED", "diff_table": table,
           "code_sha256": code,
           "throughput": {t: json.load(open(os.path.join(a.pkg, t), encoding="utf-8")) for t in a.throughput},
           "extra": {e: json.load(open(os.path.join(a.pkg, e), encoding="utf-8")) for e in a.extra}}
    json.dump(out, open(os.path.join(a.pkg, a.out), "w", encoding="utf-8"), indent=1)
    print(json.dumps({"verdict": out["verdict"], "n_diffs": len(table)}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

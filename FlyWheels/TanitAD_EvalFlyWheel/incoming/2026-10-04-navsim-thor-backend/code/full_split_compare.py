#!/usr/bin/env python3
"""Full-split Thor-vs-dev-box comparison, built for a DECISION (any python).

Merges K shard CSVs/frames from Thor (token-disjoint by construction), checks the union is exactly
the reference's token set, and reports per column: tokens differing, max |delta|; for the DISCRETE
sub-scores (values in {0, 0.5, 1}) the number of FLIPS; and the effect on the split-level means that
programme tables quote (mean of every numeric column over all tokens, Thor minus dev box).

    python full_split_compare.py --ref <dev-box csv/frame> --got shard0.csv shard1.csv ... --out x.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math

SUMMARY = ("extended_pdm_score", "average")
DISCRETE = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
            "traffic_light_compliance", "time_to_collision_within_bound", "lane_keeping", "history_comfort",
            "comfort", "two_frame_extended_comfort", "multiplicative_metrics_prod")


def load(p):
    with open(p, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {r["token"]: r for r in rows if not r["token"].startswith(SUMMARY)}, (list(rows[0].keys()) if rows else [])


def f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--got", nargs="+", required=True)
    ap.add_argument("--ignore", nargs="*", default=["", "index"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref, rcols = load(a.ref)
    got, gcols = {}, None
    dup = 0
    for p in a.got:
        g, gcols = load(p)
        dup += len(set(g) & set(got))
        got.update(g)
    fails = []
    if dup:
        fails.append(f"{dup} tokens appear in more than one shard")
    if set(got) != set(ref):
        fails.append(f"token sets differ: thor-only {len(set(got) - set(ref))}, ref-only {len(set(ref) - set(got))}")
    keys = sorted(set(got) & set(ref))
    cols = [c for c in rcols if c in (gcols or []) and c != "token" and c not in a.ignore]
    per = {}
    for c in cols:
        n_diff, mx, flips, sr, sg, nn = 0, 0.0, 0, 0.0, 0.0, 0
        for k in keys:
            x, y = ref[k][c], got[k][c]
            if x != y:
                n_diff += 1
            fx, fy = f(x), f(y)
            if fx is None or fy is None or math.isnan(fx) or math.isnan(fy):
                continue
            mx = max(mx, abs(fx - fy))
            if c in DISCRETE and fx != fy:
                flips += 1
            sr += fx
            sg += fy
            nn += 1
        per[c] = {"n_tokens_text_differ": n_diff, "max_abs_diff": mx,
                  "n_discrete_flips": (flips if c in DISCRETE else None),
                  "mean_delta_thor_minus_ref": ((sg - sr) / nn if nn else None), "n_numeric": nn}
    any_diff = sorted({k for k in keys for c in cols if ref[k][c] != got[k][c]})
    rep = {"ref": a.ref, "got": a.got, "n_tokens_ref": len(ref), "n_tokens_thor": len(got),
           "n_tokens_compared": len(keys), "n_tokens_any_column_differs": len(any_diff),
           "frac_tokens_any_column_differs": len(any_diff) / max(1, len(keys)),
           "total_discrete_flips": sum(v["n_discrete_flips"] or 0 for v in per.values()),
           "max_abs_diff_any_column": max((v["max_abs_diff"] for v in per.values()), default=None),
           "max_abs_mean_delta_any_column": max((abs(v["mean_delta_thor_minus_ref"] or 0) for v in per.values()), default=None),
           "per_column": per, "tokens_differing": any_diff[:500], "failures": fails,
           "bit_exact": (not fails and not any_diff)}
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in ("n_tokens_compared", "n_tokens_any_column_differs",
                                          "frac_tokens_any_column_differs", "total_discrete_flips",
                                          "max_abs_diff_any_column", "max_abs_mean_delta_any_column",
                                          "bit_exact", "failures")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

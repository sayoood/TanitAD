#!/usr/bin/env python3
"""Pick a navhard_two_stage validation subset made of COMPLETE two-stage groups (any python + PyYAML).

Why complete groups: the two-stage runner's per-token ``weight`` / ``two_frame_extended_comfort`` /
``score`` come from ``SceneAggregator`` over a group ``(orig, prev, [(s2_a, s2_b), ...])``. A subset
that cut a group would change those columns by construction and the exact-reproduction bar would
compare two different experiments. Rule (deterministic, written before any Thor number exists):
groups in ``reactive_all_mapping`` YAML order, taken whole, until the token count reaches ``--n``.

    python select_navhard_subset.py --yaml <devkit>/…/train_test_split/navhard_two_stage.yaml \
        --frame <banked final_scores_frame.csv> --n 200 --out raw/navhard_subset_200.json
"""
from __future__ import annotations

import argparse
import csv
import json

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--frame", required=True, help="banked dev-box frame: every chosen token must be in it")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    doc = yaml.safe_load(open(a.yaml, encoding="utf-8"))
    banked = {r["token"] for r in csv.DictReader(open(a.frame, encoding="utf-8"))}
    stage1, expected, groups = [], [], 0
    for orig, prev, pairs in doc["reactive_all_mapping"]:
        toks = [orig, prev] + [t for p in pairs for t in p]
        if not all(t in banked for t in toks):
            continue                                     # never pick a group the reference lacks
        stage1 += [orig, prev]
        expected += toks
        groups += 1
        if len(expected) >= a.n:
            break
    if len(set(expected)) != len(expected):
        raise SystemExit("a token appears in two groups -- refusing (groups would not be independent)")
    out = {"rule": f"whole reactive_all_mapping groups in YAML order until >= {a.n} tokens; "
                   f"every token present in the banked frame",
           "yaml": a.yaml, "frame": a.frame, "n_groups": groups, "n_stage1": len(stage1),
           "n_expected": len(expected), "stage1_tokens": stage1, "expected_tokens": expected}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: out[k] for k in ("n_groups", "n_stage1", "n_expected")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

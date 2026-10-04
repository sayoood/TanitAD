#!/usr/bin/env python3
"""Shard navhard_two_stage into K files of COMPLETE two-stage groups (any python + PyYAML).

Group g (YAML order of ``reactive_all_mapping``) goes to shard ``g % K``. Each shard file has the
``score_arm_thor.py --tokens`` format (``stage1_tokens`` + ``expected_tokens``). Because a group is
never split, every per-token column -- including the aggregator's ``weight`` /
``two_frame_extended_comfort`` / ``score`` -- is computed exactly as in a single full-split run, so
K shards = one full split, scored K-way in parallel. Tokens of the split that belong to NO group
(if any) are listed and go to shard 0 as plain stage-1 filters.

    python shard_navhard_groups.py --yaml <…/train_test_split/navhard_two_stage.yaml> --k 6 --out-prefix raw/navhard_shard
"""
from __future__ import annotations

import argparse
import json

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    doc = yaml.safe_load(open(a.yaml, encoding="utf-8"))
    shards = [{"stage1_tokens": [], "expected_tokens": [], "n_groups": 0} for _ in range(a.k)]
    seen = set()
    for g, (orig, prev, pairs) in enumerate(doc["reactive_all_mapping"]):
        s = shards[g % a.k]
        toks = [orig, prev] + [t for p in pairs for t in p]
        if seen & set(toks):
            raise SystemExit(f"group {g} shares tokens with an earlier group -- refusing")
        seen |= set(toks)
        s["stage1_tokens"] += [orig, prev]
        s["expected_tokens"] += toks
        s["n_groups"] += 1
    for i, s in enumerate(shards):
        s["rule"] = f"reactive_all_mapping group g -> shard g % {a.k}; this is shard {i}"
        json.dump(s, open(f"{a.out_prefix}_{i}of{a.k}.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({"n_groups": sum(s["n_groups"] for s in shards), "n_tokens": len(seen),
                      "per_shard": [len(s["expected_tokens"]) for s in shards]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

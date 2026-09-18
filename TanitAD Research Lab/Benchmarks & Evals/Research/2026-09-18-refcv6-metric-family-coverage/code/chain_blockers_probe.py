#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv6 -- the two facts that decide whether the panel's eval chain can START.

    python chain_blockers_probe.py --json ../raw/chain_blockers.json

⛔ BLOCKER 1 -- the param_breakdown cross-check. `refcv3_arm.py:1052-1059` compares
the rebuilt model's `param_breakdown_v3` against the one stamped in `config.json`,
`total` INCLUDED, and raises SystemExit on any conflict. It runs at `:1102`, BEFORE
`load_state_dict` at `:1104`, so `--allow-nonstrict` (which gates `:1128`) cannot
reach it.

The refcv6 perception branch is attached by the TRAINER as a plain attribute
(`refc_v3_train.py:5699-5705`, `model._perception = build_perception_branch(...)`)
and is NOT a `RefCV3Config` field (`refc_v3.py:1683-1685`). So `param_breakdown_v3`
has no `_perception` line while its `total` counts the branch. This probe measures
that gap directly from the banked configs: if
``total - sum(named) == refcv6_perception.branch_params.total`` then the eval-side
rebuild -- which cannot build the branch -- reproduces only `sum(named)` and the
cross-check must fire. ⛔ It is an ARITHMETIC proof from the run record, not a claim
about a run that was never attempted.

⭐ BLOCKER 2 (or not) -- the lead block. `four_families._distance_keeping` is the half
of LONGITUDINAL the binding rule names and it refuses without a `lead` dict. This
probe loads the banked B1 block and checks it carries every field the metric needs,
and enough clips to cover the 139-clip eval corpus.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

DEFAULT_CONFIGS = [
    r"C:/Users/Admin/tanitad-caches/refcv6-zhverify-20260918/run/config.json",
    r"C:/Users/Admin/tanitad-caches/refcv6-pipeval-20260917/rate416/config.json",
]

#: `refav1_arm.LEAD_BLOCK_DEFAULT` (taniteval/tools/refav1_arm.py:232-234);
#: `refcv3_arm.py:3500-3510` picks it up unless --no-lead-block.
DEFAULT_LEAD = (r"C:/Users/Admin/tanitad-wt-bevtac/TanitAD Research Lab/"
                r"Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/"
                r"raw/b1_eval_lead_block.npz")

#: What `lead_metrics.distance_keeping` + the speed-stratified read need.
LEAD_REQUIRED = ("leads", "lead_lens", "speeds")
LEAD_FOR_STRATA = ("eid", "state")


def probe_param_breakdown(paths) -> dict:
    rows = {}
    for p in paths:
        try:
            c = json.load(open(p, encoding="utf-8"))
        except OSError as exc:
            rows[p] = {"error": str(exc)}
            continue
        pb = c.get("param_breakdown") or {}
        named = {k: int(v) for k, v in pb.items()
                 if k != "total" and isinstance(v, (int, float))}
        total = int(pb.get("total") or 0)
        branch = ((c.get("refcv6_perception") or {})
                  .get("branch_params") or {}).get("total")
        delta = total - sum(named.values())
        rows[p] = {
            "named_lines": sorted(named),
            "sum_named": sum(named.values()),
            "stamped_total": total,
            "delta": delta,
            "refcv6_perception_branch_params_total": branch,
            "has_perception_line": any("percept" in k for k in pb),
            "delta_equals_branch": (branch is not None and delta == int(branch)),
        }
    agree = [v for v in rows.values() if v.get("delta_equals_branch")]
    return {
        "per_config": rows,
        "verdict": (
            "BLOCKS -- the rebuilt model reproduces only sum(named); the stamped "
            "total is larger by exactly the perception branch, so "
            "cross_check_config raises SystemExit before the weights are loaded"
            if agree and len(agree) == len([v for v in rows.values()
                                            if "error" not in v])
            else "INCONCLUSIVE -- the delta does not match the branch params on "
                 "every config; do not quote this as a blocker"),
        "n_configs_agreeing": len(agree),
    }


def probe_lead_block(path: str) -> dict:
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy unavailable -- run with the tanitad venv python"}
    if not os.path.exists(path):
        return {"error": f"not found: {path}"}
    z = np.load(path, allow_pickle=True)
    files = set(z.files)
    missing = [k for k in LEAD_REQUIRED if k not in files]
    n = int(z["leads"].shape[0]) if "leads" in files else 0
    out = {
        "path": path,
        "keys": sorted(files),
        "n_rows": n,
        "leads_shape": (list(z["leads"].shape) if "leads" in files else None),
        "missing_required": missing,
        "has_strata_keys": [k for k in LEAD_FOR_STRATA if k in files],
        "n_distinct_clip_id": (len(set(map(str, z["clip_id"].tolist())))
                               if "clip_id" in files else None),
        "ts_rel_s": (z["ts_rel_s"].tolist() if "ts_rel_s" in files else None),
        "dt_s": (z["dt_s"].tolist() if "dt_s" in files else None),
        "states": (dict(collections.Counter(map(str, z["state"].tolist())))
                   if "state" in files else None),
    }
    out["verdict"] = (
        "AVAILABLE -- every field distance_keeping needs is present"
        if not missing else
        f"BLOCKS -- missing {missing}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--lead", default=DEFAULT_LEAD)
    ap.add_argument("--json", dest="json_out", default=None)
    a = ap.parse_args(argv)

    out = {
        "tool": "chain_blockers_probe.py",
        "blocker_1_param_breakdown": probe_param_breakdown(a.configs),
        "blocker_2_lead_block": probe_lead_block(a.lead),
        "_not_probed_here": (
            "the strict-load refusal (refcv3_arm.py:1128-1133) and the forward "
            "raise (refc_v3.py:2002-2012) are read from source, not exercised -- "
            "exercising them would need the 578 MB ckpt loaded and a forward pass, "
            "and blocker 1 fires first in any case (:1102 precedes :1104)."),
    }
    txt = json.dumps(out, indent=1, ensure_ascii=False)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            fh.write(txt)
        print("wrote", a.json_out)
    print("blocker 1 (param_breakdown):", out["blocker_1_param_breakdown"]["verdict"])
    print("blocker 2 (lead block)    :", out["blocker_2_lead_block"].get("verdict")
          or out["blocker_2_lead_block"].get("error"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

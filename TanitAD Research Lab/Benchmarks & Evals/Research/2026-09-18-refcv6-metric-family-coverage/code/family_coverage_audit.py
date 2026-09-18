#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv6 -- what the eval path ACTUALLY EMITS, diffed against the four binding families.

    python family_coverage_audit.py --json ../raw/refcv6_family_coverage.json

# Why this reads the ARTIFACT and not only the source

A key the code CAN emit is not a key a run DID emit. `refc_v3_train.py`'s eval block
builds its row by iterating whatever `compute_losses_v3` happened to return for THAT
flag set -- so the emitted key set is a property of the ARM, not of the file. The only
sound census is over banked `metrics.jsonl` rows.

⛔ THE THREE CASES THIS TOOL KEEPS APART, because they need different actions:

    (a) WIRED      the instrument exists AND the refcv6 path calls it
    (b) UNWIRED    the instrument exists in `taniteval/` but nothing in the refcv6
                   path calls it -- the CHEAP win, and the one most often
                   mis-reported as (a) because the file is present in the repo
    (c) NO INSTRUMENT  nothing anywhere computes it

Presence of `taniteval/taniteval/four_families.py` on disk proves NOTHING about (a).
The test applied here is an IMPORT test over the emitting script, not a file-exists test.

# The estimator question

`taniteval.ci.paired_episode_cluster_bootstrap(a, b, eid, ...)` needs per-window arrays
`a`, `b` of equal shape and a per-window `eid` of the same length. This tool therefore
also censuses the artifact for (i) any non-scalar value and (ii) any episode/clip/window
id key. An aggregate-only dump cannot be bootstrapped, whatever its metric coverage.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

#: The banked refcv6 runs audited. Both are pipeline-validation runs of the
#: refcv6 flag set on the B1 139-clip eval corpus (the `--steps 1` shape).
DEFAULT_RUNS = [
    r"C:/Users/Admin/tanitad-caches/refcv6-zhverify-20260918/run",
    r"C:/Users/Admin/tanitad-caches/refcv6-pipeval-20260917/cov3_evalB",
    r"C:/Users/Admin/tanitad-caches/refcv6-pipeval-20260917/cov2_evalA",
    r"C:/Users/Admin/tanitad-caches/refcv6-pipeval-20260917/rate416",
]

#: The emitting script -- the ONE file whose eval block writes `metrics.jsonl`.
EMITTER = "stack/scripts/refc_v3_train.py"

#: Modules that compute the binding families. The (b)-vs-(a) test is whether
#: EMITTER imports any of them.
FAMILY_MODULES = (
    "taniteval.four_families", "taniteval.lead_metrics", "taniteval.lateral",
    "taniteval.nav_compliance", "taniteval.hierarchy", "taniteval.ci",
    "taniteval.strategic_optionset", "taniteval.refcv6_acceptance",
    "four_families", "lead_metrics", "nav_compliance", "refcv6_acceptance",
)

#: Every required criterion of the binding registry, and the emitted key that
#: would satisfy it. `None` = nothing the refcv6 row emits maps to it.
#: ⛔ This map is asserted against the emitted key set below; a key named here
#: that the runs do NOT emit is reported as a BROKEN MAPPING, never silently kept.
CRITERION_TO_EMITTED_KEY = {
    "long.target_speed":       None,
    "long.along_track_error":  None,
    "long.progress":           None,
    "long.distance_keeping":   None,
    "lat.cross_track":         None,
    "lat.heading":             None,
    "lat.curvature":           None,
    "lat.yaw_rate":            None,
    "tac.manoeuvre_decision":  None,
    "tac.confusion":           None,
    "tac.goal_selection":      None,
    "strat.decision":          None,
    "strat.route_goal":        None,
    "strat.nav_compliance":    None,
    "strat.nav_compliance_ctrl_shuffle": None,
    "strat.nav_compliance_ctrl_zero":    None,
}

#: Emitted keys that LOOK like a family metric and are not one. Listed because
#: each has already been read as coverage in a draft, and a near-miss is more
#: dangerous than an absence.
FALSE_FRIENDS = {
    "lat":        "the REFCV3 LATERAL TRAJECTORY LOSS term (a training loss), not lateral cross-track error",
    "lon":        "the longitudinal trajectory LOSS term, not speed error",
    "lat_tac":    "the tactical head's LATERAL-CLASS cross-entropy LOSS, not manoeuvre-decision quality",
    "lon_tac":    "the tactical head's LONGITUDINAL-CLASS cross-entropy LOSS, not target-speed accuracy",
    "tacv6_lat_ce":  "refcv6 tactical decoder lateral CE LOSS -- a training objective, not an agreement metric",
    "tacv6_lon_ce":  "refcv6 tactical decoder longitudinal CE LOSS -- same",
    "route":      "the ROUTE AUX LOSS weight term; with --no-strategic the strategic head is not built",
    "goal_tac":   "a goal-head LOSS, not tactical goal-SELECTION quality",
    "goal2s_err_m": "the geometric goal POINT's 2 s error -- a single point, not ADE and not a family metric",
    "anchor_acc": "top-1 anchor-vocabulary accuracy of the SELECTOR at train time, pooled over the batch, no n, no eid",
    "map_iou_drivable": "a PERCEPTION metric (SAM3 BEV map), not one of the four driving families",
    "box3d_yaw":  "3-D cuboid YAW loss for OTHER agents -- perception, not the ego's lateral yaw-rate error",
}


def read_rows(run_dir: str) -> list[dict]:
    p = os.path.join(run_dir, "metrics.jsonl")
    rows = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def census(runs) -> dict:
    keys, per_run, nonscalar, idkeys = {}, {}, [], []
    id_pat = re.compile(r"eid|episode|clip|window_id|uid", re.I)
    for r in runs:
        try:
            rows = read_rows(r)
        except OSError as exc:
            per_run[r] = {"error": str(exc)}
            continue
        ks = {}
        for row in rows:
            for k, v in row.items():
                ks[k] = ks.get(k, 0) + 1
                keys[k] = keys.get(k, 0) + 1
                if isinstance(v, (list, dict)):
                    nonscalar.append(k)
                if id_pat.search(k):
                    idkeys.append(k)
        per_run[r] = {
            "n_rows": len(rows),
            "n_keys": len(ks),
            "has_eval_row": any(k.startswith("eval_") for k in ks),
            "eval_windows": next((row.get("eval_windows") for row in rows
                                  if "eval_windows" in row), None),
        }
    return {
        "runs": per_run,
        "n_distinct_keys": len(keys),
        "keys": dict(sorted(keys.items())),
        "non_scalar_values": sorted(set(nonscalar)),
        "episode_or_window_id_keys": sorted(set(idkeys)),
    }


def emitter_imports(repo: str) -> dict:
    """Does the emitting script import ANY family instrument? (the (a)/(b) test)"""
    p = os.path.join(repo, EMITTER)
    with open(p, encoding="utf-8") as fh:
        src = fh.read()
    found = {}
    for mod in FAMILY_MODULES:
        pat = re.compile(r"^\s*(?:import\s+%s|from\s+%s\s+import)" % (re.escape(mod),
                                                                     re.escape(mod)),
                         re.M)
        m = pat.search(src)
        found[mod] = m.group(0).strip() if m else None
    return {
        "emitter": EMITTER,
        "imports_found": {k: v for k, v in found.items() if v},
        "imports_absent": sorted(k for k, v in found.items() if not v),
        "verdict": ("WIRED" if any(found.values()) else
                    "NOT WIRED -- the emitting script imports NO family instrument"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--repo", default=os.path.abspath(
        os.path.join(HERE, "..", "..", "..", "..", "..")))
    ap.add_argument("--json", dest="json_out", default=None)
    a = ap.parse_args(argv)

    cen = census(a.runs)
    imp = emitter_imports(a.repo)

    emitted = set(cen["keys"])
    broken = {c: k for c, k in CRITERION_TO_EMITTED_KEY.items()
              if k is not None and k not in emitted}
    satisfied = {c: k for c, k in CRITERION_TO_EMITTED_KEY.items()
                 if k is not None and k in emitted}

    out = {
        "tool": "family_coverage_audit.py",
        "question": ("when the refcv6 panel runs, will it report all four metric "
                     "families -- or will a family be silently missing?"),
        "artifact_census": cen,
        "emitter_import_test": imp,
        "criterion_mapping": {
            "satisfied_by_an_emitted_key": satisfied,
            "no_emitted_key_maps_to_it": sorted(
                c for c, k in CRITERION_TO_EMITTED_KEY.items() if k is None),
            "BROKEN_MAPPING_key_not_emitted": broken,
        },
        "false_friends": FALSE_FRIENDS,
        "estimator_reachability": {
            "required_by": "taniteval.ci.paired_episode_cluster_bootstrap(a, b, eid, ...)",
            "needs": "per-window arrays a and b of equal shape + a per-window eid of the same length",
            "artifact_has_non_scalar_values": bool(cen["non_scalar_values"]),
            "artifact_has_episode_ids": bool(cen["episode_or_window_id_keys"]),
            "verdict": ("REACHABLE" if cen["non_scalar_values"] and
                        cen["episode_or_window_id_keys"] else
                        "NOT REACHABLE -- aggregate-only dump, no per-window values "
                        "and no episode id. No rescore of this artifact can produce "
                        "an interval."),
        },
    }
    txt = json.dumps(out, indent=1, ensure_ascii=False)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            fh.write(txt)
        print("wrote", a.json_out)
    print("emitter import test :", imp["verdict"])
    print("distinct keys       :", cen["n_distinct_keys"])
    print("criteria with a key :", len(satisfied), "of", len(CRITERION_TO_EMITTED_KEY))
    print("estimator           :", out["estimator_reachability"]["verdict"])
    if broken:
        print("BROKEN MAPPINGS     :", broken)
    return 0


if __name__ == "__main__":
    sys.exit(main())

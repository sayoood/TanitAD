#!/usr/bin/env python3
"""Paired two-stage intervals R7_VMAXOFF minus each navhard floor (TANITAD VENV).

WHY THIS EXISTS: ``vmaxoff_legal_summary7.py``'s ``paired_intervals_vmaxoff_vs_each`` block dies with
FileNotFoundError, because it loads E2's ``parse_scores.py`` from ``<navsim>/../2026-09-19-navsim-refcv4b-bridge``.
That path resolves INSIDE ``2026-09-28-refcv7-standard-tests/``, not ``incoming/``, and the loaded module is never
used anyway. This re-runs exactly that block's statistics -- ``navsim_ci.rows_from_score_frame``,
``two_stage_key_contributions`` and ``paired_log_cluster_bootstrap`` with ``AGG_TWO_STAGE`` -- on the same
staged frames, minus the broken import.

    python vmaxoff_navhard_pairs.py --stage <stage root used by thor_vmaxoff_summary> --summary <summary .THOR.json> --out <json>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

SUITE = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim"
INPUTS = "C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("navsim_ci7", "D:/Projects/TanitAD/taniteval/adapters/navsim_ci.py")
    ci = importlib.util.module_from_spec(spec)
    sys.modules["navsim_ci7"] = ci
    spec.loader.exec_module(ci)
    doc = json.load(open(INPUTS, encoding="utf-8"))
    mapping = doc["reactive_all_mapping"]
    log_of = {t: r["log_name"] for t, r in doc["tokens"].items()}
    clusters = [log_of[str(m[0])] for m in mapping]
    summ = json.load(open(a.summary, encoding="utf-8"))
    off = {k: v.get("official_two_stage_EPDMS") for k, v in summ["arms"].items()}
    sfx = "__navhard_two_stage"
    floors = os.path.join(SUITE, "raw", "floors", "navhard_two_stage")
    src = {"R7_VMAXOFF": os.path.join(a.stage, "vmaxoff_legal", "scores_navhard"),
           "STOP_zero": floors, "CV_official": floors, "ECHO_ha0_ext": floors}
    contrib = {}
    for arm, d in src.items():
        tag = f"{arm}{sfx}"
        rows, _ = ci.rows_from_score_frame(os.path.join(d, f"score_{tag}_wrapper", f"{tag}_final_scores_frame.csv"))
        contrib[arm] = ci.two_stage_key_contributions(rows, mapping, "score")
    out = {"estimator": "navsim_ci.paired_log_cluster_bootstrap (AGG_TWO_STAGE), log-cluster resampling",
           "answers": "another draw of LOGS? (blind to training and inference variance)",
           "backend": "R7_VMAXOFF scored on Thor; floors scored on the dev box => MIXED-BACKEND under RULING_BACKEND_POLICY.md",
           "official_two_stage_EPDMS": off, "pairs": {}}
    for y in ("STOP_zero", "CV_official", "ECHO_ha0_ext"):
        out["pairs"][f"R7_VMAXOFF__minus__{y}"] = ci.paired_log_cluster_bootstrap(
            contrib["R7_VMAXOFF"], contrib[y], clusters, aggregation=ci.AGG_TWO_STAGE,
            official_a=off["R7_VMAXOFF"], official_b=off[y])
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    for k, v in out["pairs"].items():
        print(k, json.dumps({kk: v.get(kk) for kk in ("official_delta", "delta", "lo", "hi", "status", "separated")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Writes N5_PINS.json for a TF-N5 run tree (MM, at the freeze): sha256 of every .py/.json/.yaml under w/, repo/, n2/, n4/, n5/ (except N5_PINS.json itself), of the
REQUIRED inputs, of the harness files outside the tree, and the metric-cache listing digest. Refuses a tree whose families_run.py is not the TIP version.
python n5_make_pins.py --run-dir R --tip <sha> [--out R/n5/N5_PINS.json] [--skip-cache-listing]"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n5_nt_chain as CH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-cache-listing", action="store_true", help="tests only")
    a = ap.parse_args()
    run, files = a.run_dir, {}
    for top in CH.WATCH_TOP:
        for d, _, fs in os.walk(os.path.join(run, top)):
            if "__pycache__" in d:
                continue
            for f in fs:
                if f.endswith(CH.WATCH_EXT):
                    rel = os.path.relpath(os.path.join(d, f), run).replace(os.sep, "/")
                    if rel != CH.PINS_REL:
                        files[rel] = CH.sha256(os.path.join(d, f))
    for rel in CH.REQUIRED:
        p = rel if os.path.isabs(rel) or rel[1:2] == ":" else os.path.join(run, rel)
        files[rel] = CH.sha256(p)
    if files.get(CH.TIP_FAMILIES) != CH.TIP_FAMILIES_SHA256:
        sys.exit(f"REFUSING: {CH.TIP_FAMILIES} is {files.get(CH.TIP_FAMILIES)}, not the TIP version {CH.TIP_FAMILIES_SHA256}")
    for p in (CH.DEVKIT + "/navsim/evaluate/pdm_score.py", CH.DEVKIT + "/navsim/planning/metric_caching/metric_cache_processor.py",
              CH.DEVKIT + "/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
              CH.DEVKIT + "/navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py"):
        files[p] = CH.sha256(p)
    lsha, n = ("SKIPPED", 0) if a.skip_cache_listing else CH.cache_listing_sha(CH.CACHE_ROOT)
    out = {"card": "PREREG_TF_N5 C-PIN", "tip": a.tip, "metric_cache_root": CH.CACHE_ROOT, "metric_cache_listing_sha256": lsha, "metric_cache_files": n,
           "families_run_tip_sha256": CH.TIP_FAMILIES_SHA256, "files": dict(sorted(files.items()))}
    json.dump(out, open(a.out or os.path.join(run, CH.PINS_REL), "w", encoding="utf-8", newline="\n"), indent=1)
    print("wrote", len(files), "pins; cache files", n)


if __name__ == "__main__":
    main()

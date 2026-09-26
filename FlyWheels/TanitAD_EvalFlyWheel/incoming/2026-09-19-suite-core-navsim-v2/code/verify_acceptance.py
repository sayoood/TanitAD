#!/usr/bin/env python3
"""Check a run directory against the PRE-REGISTERED acceptance criteria (SPEC.md §3, ACC-1..ACC-6).

⛔ Assert on the ARTIFACTS, never on an exit code: every criterion below reads the files the run
left behind. Writes ``raw/acceptance_verdict.json`` and prints one line per criterion.

    python verify_acceptance.py <run_dir> [--rc <exit code of the run>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "taniteval"))

CV_BANKED = 0.1853562745165113
STOP_BANKED = 0.3009023137456225
CV_MD5 = "b4b33ac29c39c37832371b8b68c9f898"
STOP_MD5 = "4a15c3a1d59234e518291a1a9e1d5bff"
LB_X100 = 18.5356


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--rc", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    from taniteval.bench import contract as C
    run = a.run_dir
    # ⛔ APPLY THE SKIP RULE FIRST (W4, 2026-09-20). MEASURED the same day, on my own tooling: a
    # throwaway `ls -dt | head -1` "newest run" heuristic picked a TOMBSTONE directory — tombstones
    # are written AFTER the run they withdraw, so they are the newest thing in the tree — and this
    # verifier died on a traceback instead of saying so. A consumer never picks a run by mtime.
    ok, why = C.is_consumable_run(run)
    if not ok:
        print(f"⛔ REFUSED: {run} is not a consumable run — {why}. "
              f"Name the run directory explicitly; never pick one by modification time.")
        return 2
    rec = json.loads((run / "bench_run.json").read_text(encoding="utf-8"))
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    v: dict = {"run_dir": str(run).replace("\\", "/"), "run_id": rec.get("run_id"), "rc": a.rc,
               "status": rec.get("status"), "wall_s": rec.get("wall_s"), "criteria": {}}

    files = {f"scores/{x}.csv" for x in ("CV", "STOP")} | {f"artifacts/{x}.json" for x in ("CV", "STOP")} \
        | {f"criteria/{x}.txt" for x in ("CV", "STOP")}
    missing = [f for f in sorted(files) if not (run / f).exists() or (run / f).stat().st_size == 0]
    errs_run, errs_sum = C.validate_bench_run(rec), C.validate_summary(summ)
    v["criteria"]["ACC-1"] = {"what": "exit 0, both records validate, the six files exist non-empty",
                              "rc": a.rc, "status": rec.get("status"), "missing_or_empty": missing,
                              "bench_run_errors": errs_run, "summary_errors": errs_sum,
                              "pass": bool((a.rc in (0, None)) and rec.get("status") == "COMPLETE"
                                           and not missing and not errs_run and not errs_sum)}

    hcv = summ["arms"]["CV"]["headline"]
    v["criteria"]["ACC-2"] = {"what": f"CV headline == {CV_BANKED!r} from column `score`, x100 -> {LB_X100}",
                              "value": hcv.get("value"), "column": hcv.get("column"), "row": hcv.get("row"),
                              "x100_rounded": (None if hcv.get("value") is None else round(100 * hcv["value"], 4)),
                              "pass": bool(hcv.get("value") == CV_BANKED and hcv.get("column") == "score"
                                           and hcv.get("row") == "extended_pdm_score_combined"
                                           and round(100 * hcv["value"], 4) == LB_X100)}
    hst = summ["arms"]["STOP"]["headline"]
    v["criteria"]["ACC-3"] = {"what": f"STOP headline == {STOP_BANKED!r}", "value": hst.get("value"),
                              "pass": bool(hst.get("value") == STOP_BANKED)}

    cells = {}
    for arm in ("CV", "STOP"):
        rc_ = summ["arms"][arm]["controls"].get("reference_check") or {}
        cells[arm] = {k: rc_.get(k) for k in ("max_abs_diff", "nan_pattern_mismatch", "same_token_set",
                                              "same_row_order", "cells_identical", "n_numeric_cells",
                                              "first_or_worst_difference", "md5", "reference_md5",
                                              "file_identical", "headline_equals_banked")}
    v["criteria"]["ACC-4"] = {"what": "every cell equals E1/E2's banked CSV (max |Δ| = 0.0, NaN pattern identical)",
                              "detail": cells,
                              "pass": all(cells[a_]["cells_identical"] for a_ in ("CV", "STOP"))}
    md5s = {arm: hashlib.md5((run / f"scores/{arm}.csv").read_bytes()).hexdigest() for arm in ("CV", "STOP")}
    v["criteria"]["ACC-5"] = {"what": "file-level md5 identity with the banked CSVs", "md5": md5s,
                              "expected": {"CV": CV_MD5, "STOP": STOP_MD5},
                              "pass": bool(md5s["CV"] == CV_MD5 and md5s["STOP"] == STOP_MD5)}

    import copy
    mutated = copy.deepcopy(summ)
    mutated["arms"].pop("STOP")
    mutated["floors"] = ["CV"]
    mut_errs = C.validate_summary(mutated)
    v["criteria"]["ACC-6"] = {"what": "the floors are machinery: a NavSim summary without STOP is REFUSED",
                              "floors": summ.get("floors"),
                              "added_by_rule": {k: summ["arms"][k].get("added_by_rule") for k in summ["arms"]},
                              "mutation_errors": mut_errs[:4], "pass": bool(mut_errs)}

    v["ALL_PASS"] = all(c["pass"] for c in v["criteria"].values())
    v["external_reference"] = {"HF_warmup_LB_baseline_constant_velocity_x100": LB_X100,
                               "evidence_class": "INHERITED (NAVSIM_PROTOCOL.md §6.3)"}
    out = a.out or (Path(__file__).resolve().parents[1] / "raw" / "acceptance_verdict.json")
    out.write_text(json.dumps(v, indent=1), encoding="utf-8")
    for k, c in v["criteria"].items():
        print(f"{k}: {'PASS' if c['pass'] else 'FAIL'} — {c['what']}")
    print(f"ALL_PASS={v['ALL_PASS']} -> {out}")
    return 0 if v["ALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

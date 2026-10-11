#!/usr/bin/env python3
"""Is a PDM relabel COMPLETE? The launch condition of eval/PREREG_SFT4R.md "Changed 1", as a check that can say no.

SFT-4 launched on 146,185 of 174,912 labels (PREREG_SFT4 Amendment 4) because "complete" was never a computed fact.
This computes it from three independent sources and requires them to agree:

  1. the ELIGIBLE count -- recomputed here from the set files with the relabeller's own `served_keys` (imported, so the
     definition cannot drift), or, without --sets, the relabeller's printed "served sets: N";
  2. the relabeller's final marker `ZZRELABEL_PDM_DONE <labelled> <skipped> <failed>`. The relabeller resumes by
     (key, ckpt_step) and never writes a row for a guard-skipped or failed set, so its LAST invocation re-attempts every
     such set: its <skipped> and <failed> are totals, while its <labelled> counts only that invocation's rows;
  3. the rows ON DISK: unique (key, ckpt_step) over <out>/pdm_*.jsonl, restricted to the eligible keys when --sets is
     given (an earlier non-served-only pass can leave rows for superseded checkpoints; they are counted, not credited).

COMPLETE iff the DONE marker is present, <failed> == 0, and credited rows + <skipped> == eligible (and == --expect when
given). Anything else is INCOMPLETE with a named reason. Exit 0 / 1. Prints ZZPDM_COMPLETE or ZZPDM_INCOMPLETE <reason>.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

SERVED_RE = re.compile(r"^served sets: ([\d,]+)\s*$")
DONE_RE = re.compile(r"^ZZRELABEL_PDM_DONE (\d+) (\d+) (\d+)\s*$")


def parse_log(path):
    served = done = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            m = SERVED_RE.match(ln)
            if m:
                served = int(m.group(1).replace(",", ""))
            m = DONE_RE.match(ln)
            if m:
                done = tuple(int(x) for x in m.groups())
    return served, done


def eligible_from_sets(sets_dir, pattern, served_only):
    files = sorted(glob.glob(os.path.join(sets_dir, pattern)))
    if not files:
        return None, set()
    if served_only:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from onpolicy_relabel_pdm import served_keys                       # the relabeller's own definition
        best = served_keys(files)
        return len(best), {(k, rk[0]) for k, rk in best.items()}
    keys = set()
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") != "onpolicy_set":
                    continue
                keys.add(((r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))), int(r["ckpt_step"])))
    return len(keys), keys


def count_rows(out_dir):
    uniq, total, bad = set(), 0, 0
    for fp in sorted(glob.glob(os.path.join(out_dir, "pdm_*.jsonl"))):
        with open(fp, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                total += 1
                try:
                    d = json.loads(ln)
                    uniq.add((tuple(d["key"]), int(d["ckpt_step"])))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    bad += 1
    return uniq, total, bad


def check(log, out, sets=None, files="onpolicy_*.jsonl", served_only=True, expect=None):
    served, done = parse_log(log)
    uniq, total, bad = count_rows(out)
    rep = {"log": log, "out": out, "served_line": served, "done_marker": list(done) if done else None,
           "rows_total": total, "rows_unique": len(uniq), "rows_bad_json": bad, "expect": expect}
    eligible, keys = (None, set())
    if sets:
        eligible, keys = eligible_from_sets(sets, files, served_only)
        rep["eligible_recomputed"] = eligible
        credited = len(uniq & keys)
        rep["rows_superseded_or_foreign"] = len(uniq - keys)
    else:
        eligible = served
        credited = len(uniq)
    rep.update({"eligible": eligible, "rows_credited": credited})
    reason = None
    if done is None:
        reason = "no_done_marker"
    elif eligible is None:
        reason = "eligible_unknown"
    elif sets and served is not None and served != eligible:
        reason = "served_line_vs_recomputed_mismatch"
    elif expect is not None and eligible != expect:
        reason = "expect_mismatch"
    elif done[2] != 0:
        reason = "failed_sets"
    elif credited + done[1] != eligible:
        reason = "row_count_mismatch"
    rep["missing"] = None if eligible is None or done is None else eligible - credited - done[1]
    rep["verdict"] = "COMPLETE" if reason is None else "INCOMPLETE"
    rep["reason"] = reason
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="the relabeller's stdout log")
    ap.add_argument("--out", required=True, help="the relabeller's --out dir (pdm_*.jsonl)")
    ap.add_argument("--sets", default=None, help="the relabeller's --sets dir: recompute the eligible count")
    ap.add_argument("--files", default="onpolicy_*.jsonl")
    ap.add_argument("--all-versions", action="store_true", help="the relabel ran WITHOUT --served-only")
    ap.add_argument("--expect", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    rep = check(a.log, a.out, a.sets, a.files, not a.all_versions, a.expect)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    d = rep["done_marker"] or [None, None, None]
    print(f"eligible {rep['eligible']} credited {rep['rows_credited']} skipped {d[1]} failed {d[2]} "
          f"unique {rep['rows_unique']} total {rep['rows_total']} bad {rep['rows_bad_json']}", flush=True)
    if rep["verdict"] == "COMPLETE":
        print("ZZPDM_COMPLETE", flush=True)
        return 0
    print(f"ZZPDM_INCOMPLETE {rep['reason']}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())

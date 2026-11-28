#!/usr/bin/env python3
"""Synthetic test of pdm_label_complete.py: every INCOMPLETE reason must be reachable, and COMPLETE only when it is.

Run: python refe/test_pdm_label_complete.py   -> ZZTEST_PASS <n>
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdm_label_complete as C  # noqa: E402


def _world(tmp, n_sets=6, write=5, skipped=1, failed=0, done=True, served_line=True, extra_rows=(), bad=0):
    sets, out = os.path.join(tmp, "sets"), os.path.join(tmp, "out")
    os.makedirs(sets); os.makedirs(out)
    keys = [("logA", f"t{i}", i, 0) for i in range(n_sets)]
    with open(os.path.join(sets, "onpolicy_000.jsonl"), "w", encoding="utf-8") as f:
        for k in keys:
            f.write(json.dumps({"kind": "onpolicy_set", "log_name": k[0], "token": k[1], "step": k[2], "rank": k[3],
                                "ckpt_step": 7}) + "\n")
    with open(os.path.join(out, "pdm_onpolicy_000.jsonl"), "w", encoding="utf-8") as f:
        for k in keys[:write]:
            f.write(json.dumps({"kind": "pdm_targets", "key": list(k), "ckpt_step": 7}) + "\n")
        for k, s in extra_rows:
            f.write(json.dumps({"kind": "pdm_targets", "key": list(k), "ckpt_step": s}) + "\n")
        for _ in range(bad):
            f.write("{not json\n")
    log = os.path.join(tmp, "relabel.log")
    with open(log, "w", encoding="utf-8") as f:
        if served_line:
            f.write(f"served sets: {n_sets:,}\n")
        if done:
            f.write(f"ZZRELABEL_PDM_DONE {write} {skipped} {failed}\n")
    return log, out, sets


def _run(**kw):
    expect = kw.pop("expect", None)
    use_sets = kw.pop("use_sets", True)
    with tempfile.TemporaryDirectory() as tmp:
        log, out, sets = _world(tmp, **kw)
        return C.check(log, out, sets if use_sets else None, served_only=False, expect=expect)


def main():
    n = 0
    r = _run(); assert r["verdict"] == "COMPLETE", r; n += 1                                     # 5 rows + 1 skip == 6
    r = _run(use_sets=False); assert r["verdict"] == "COMPLETE", r; n += 1                       # from the served line
    r = _run(write=4); assert r["reason"] == "row_count_mismatch" and r["missing"] == 1, r; n += 1
    r = _run(done=False); assert r["reason"] == "no_done_marker", r; n += 1
    r = _run(failed=1); assert r["reason"] == "failed_sets", r; n += 1
    r = _run(expect=3137); assert r["reason"] == "expect_mismatch", r; n += 1
    r = _run(expect=6); assert r["verdict"] == "COMPLETE", r; n += 1
    r = _run(served_line=False, use_sets=False); assert r["reason"] == "eligible_unknown", r; n += 1
    # a superseded-checkpoint row must NOT be credited: 4 current rows + 1 stale + 1 skip != 6
    r = _run(write=4, extra_rows=[(("logA", "t5", 5, 0), 3)])
    assert r["reason"] == "row_count_mismatch" and r["rows_superseded_or_foreign"] == 1, r; n += 1
    # a duplicated row is counted once, a corrupt line is reported and does not count
    r = _run(extra_rows=[(("logA", "t0", 0, 0), 7)], bad=2)
    assert r["verdict"] == "COMPLETE" and r["rows_bad_json"] == 2 and r["rows_total"] == 8, r; n += 1
    # the printed served line and the recomputed count disagree -> refuse
    with tempfile.TemporaryDirectory() as tmp:
        log, out, sets = _world(tmp)
        with open(log, "a", encoding="utf-8") as f:
            f.write("served sets: 7\n")
        r = C.check(log, out, sets, served_only=False)
        assert r["reason"] == "served_line_vs_recomputed_mismatch", r; n += 1
    # the CLI: exit codes and markers
    with tempfile.TemporaryDirectory() as tmp:
        log, out, sets = _world(tmp)
        js = os.path.join(tmp, "r.json")
        assert C.main(["--log", log, "--out", out, "--sets", sets, "--all-versions", "--json", js]) == 0
        assert json.load(open(js, encoding="utf-8"))["verdict"] == "COMPLETE"; n += 1
        assert C.main(["--log", log, "--out", out, "--expect", "9"]) == 1; n += 1
    print(f"ZZTEST_PASS {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

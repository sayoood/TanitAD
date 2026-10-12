#!/usr/bin/env python3
"""REPLICATION CHECK (written 2026-10-12 AFTER the round-2 PASS; decides nothing, changes no rule): does the NEW `rl_fan_dump.compare_sets(..., mean_atol=M)` reproduce, on round 2's REAL dumps,
the per-arm numbers that `run_calibration2.py --phase analyse` wrote into `verdict_round2.json`?  The CLI's m is a re-implementation (`rl_fan_dump.set_mean_abs_xy_strict`, mirroring
`run_calibration.xy_moments`); this runs it over the dumped sets of every round-2 arm against the A40's banked reference and demands EXACT equality (==, not approx) of: sets matched, sets
above M, worst m and p50 m.  Also records what the NEW code reports about the max bound (0.5) on the same dumps.

    python verify_mean_atol_cli.py [--work D:/Projects/TanitAD-artifacts/_rlfan_work]      ->  mean_atol_cli_replication.json  (+ a table on stdout), exit 0 iff every arm matches

CPU only, reads the dumps in the work dir (the dumps are NOT in the repo; their sha256 is recorded per arm).  Evidence class: MEASURED (ours).
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(PKG, "refe"))
import rl_fan_dump as RF  # noqa: E402

STEP = 10075
CATASTROPHE = 0.5


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="D:/Projects/TanitAD-artifacts/_rlfan_work")
    a = ap.parse_args()
    W = a.work.replace("\\", "/")
    v = json.load(open(os.path.join(HERE, "verdict_round2.json"), encoding="utf-8"))
    M = v["CHOSEN_MEAN_ATOL"]
    assert M == 0.025, M
    ref = f"{W}/sets_10075_proxy.jsonl"
    assert sha(ref) == v["reference_sha256"], "reference sets differ from round 2's"
    arms = {"N_full": f"{W}/identity/new_full/onpolicy_rlfan_full_s0of1.jsonl"}
    runs = f"{W}/atol_cal2/runs"
    for name in sorted(os.listdir(runs)):
        rel = f"{runs}/{name}/relabelled_step{STEP}.jsonl"                    # a checkpoint of another step was judged offline with the step relabelled
        f = f"{runs}/{name}/onpolicy_rlfan_cal_s0of1.jsonl"
        if os.path.exists(rel):
            f = rel
        if os.path.exists(f):
            arms[name] = f
    expect = {}
    for sect in ("null_arms", "decision_arms", "info_arms"):
        for n, e in v[sect].items():
            if not e.get("missing"):
                expect[n] = {"sets": e["sets"], "above_M": e["above_M"], "worst_m": e.get("max_m", e.get("worst_m")), "p50_m": e.get("p50_m")}
    out, ok_all = {}, True
    print(f"{'arm':26s} {'sets':>5s} {'above M':>8s} {'worst m':>10s} {'p50 m':>10s}   {'verdict_round2.json':>34s}   equal")
    for name, path in arms.items():
        if name not in expect:                                              # not an arm the verdict tabulates (none expected)
            continue
        r = RF.compare_sets(path, [ref], STEP, atol=CATASTROPHE, mean_atol=M)
        q = r["mean_abs_delta_quantiles"]
        got = {"sets": r["reference_keys_matched"], "above_M": r["beyond_mean_atol"], "worst_m": q["max"], "p50_m": q["p50"]}
        e = expect[name]
        eq = {k: (got[k] == e[k]) for k in ("sets", "above_M", "worst_m") if e[k] is not None}
        if e["p50_m"] is not None:
            eq["p50_m"] = got["p50_m"] == e["p50_m"]
        ok = all(eq.values())
        ok_all &= ok
        out[name] = {"new_code": got, "verdict_round2": e, "equal": eq, "all_equal": ok, "beyond_catastrophe_atol_0.5": r["beyond_atol"], "max_abs_delta": r["max_abs_delta"],
                     "within_mean_atol": r["within_mean_atol"], "dump_sha256": sha(path)}
        print(f"{name:26s} {got['sets']:5d} {got['above_M']:8d} {got['worst_m']:10.6f} {q['p50']:10.6f}   "
              f"{e['sets']:>5d} {e['above_M']:>5d} {e['worst_m']:>10.6f} {(e['p50_m'] if e['p50_m'] is not None else float('nan')):>10.6f}   {ok}")
    missing = sorted(set(expect) - set(out))
    print("not replicated (no dump present):", missing)
    res = {"M": M, "catastrophe_atol": CATASTROPHE, "reference_sha256": v["reference_sha256"], "all_arms_equal": ok_all and not missing, "arms_missing": missing, "arms": out}
    with open(os.path.join(HERE, "mean_atol_cli_replication.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("ALL EQUAL" if res["all_arms_equal"] else "MISMATCH")
    return 0 if res["all_arms_equal"] else 1


if __name__ == "__main__":
    sys.exit(main())

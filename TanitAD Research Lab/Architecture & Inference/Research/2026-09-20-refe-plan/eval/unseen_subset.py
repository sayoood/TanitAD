#!/usr/bin/env python3
"""The "reported, not gating" read of RESULT_SFT4.md "Next arm" §1: the paired PDMS difference of an arm against the
deployed route-fix run on a token SUBSET, from the two CSVs that already exist -- no new inference.

Default = the 11,023 navtest tokens over 122 logs that no read of the SFT-4 u400 checkpoint had touched before its full
navtest (`navtest_unseen_by_sft4u400_tokens.json`, md5 ab59e479..., committed in RESULT_SFT4 before the run started).
Estimator: the same paired log-cluster bootstrap as stage 2 (proxy_eval.boot: 10,000 resamples, seed 20260927).
It prints no verdict word: this read decides nothing.

Controls (run them; they are how a reader knows the script is right):
  --arm-csv <BASE_CSV>                      identity: the deployed run against itself must read exactly +0.00 [+0.00, +0.00]
  --name sft4_B_u400 --tokens a5_confirm_tokens.json --expect-md5 '' --expect-n 923 --expect-logs 43
                                            reproduction: must give stage 2's banked +3.04 [+1.21, +4.87]

    python eval/unseen_subset.py [--name sft4_B_u400_navtest_full]   -> ZZUNSEEN <name> <mean> [lo, hi] n=<n> logs=<k>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proxy_eval as PE  # noqa: E402

D = Path("E:/Projects/TanitAD/data/refe_navtest")
BASE_CSV = D / "score" / "refe_navtest_final_routefix" / "refe_navtest_final_routefix.csv"   # = stage2_sft.BASE_CSV
UNSEEN = D / "navtest_unseen_by_sft4u400_tokens.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sft4_B_u400_navtest_full", help="the stage2_sft.py --name whose CSV is read")
    ap.add_argument("--arm-csv", default=None, help="override the arm CSV (the identity control passes BASE_CSV)")
    ap.add_argument("--tokens", default=str(UNSEEN))
    ap.add_argument("--expect-md5", default="ab59e479", help="prefix the token file must hash to ('' = no check)")
    ap.add_argument("--expect-n", type=int, default=11023)
    ap.add_argument("--expect-logs", type=int, default=122)
    ap.add_argument("--out", default=None, help="default raw/2026-10-05-sft-stage2/<name>/unseen_subset.json")
    a = ap.parse_args(argv)
    tok_p = Path(a.tokens)
    md5 = hashlib.md5(tok_p.read_bytes()).hexdigest()
    T = json.load(open(tok_p, encoding="utf-8"))
    toks, tl = sorted(T["tokens"]), T["token_log"]
    label = f"refe_stage2_{a.name}"
    arm_p = Path(a.arm_csv) if a.arm_csv else D / "score" / label / f"{label}.csv"
    arm = {t: r["score"] for t, r in PE.read_csv(arm_p).items()}
    base = {t: r["score"] for t, r in PE.read_csv(BASE_CSV).items()}
    # a row with an empty score is a token the harness marked failed: it is MISSING, never a zero
    miss_arm = [t for t in toks if arm.get(t) in (None, "")]
    miss_base = [t for t in toks if base.get(t) in (None, "")]
    n_logs = len({tl[t] for t in toks})
    gates = {"md5": {"ok": (not a.expect_md5) or md5.startswith(a.expect_md5), "md5": md5, "expect": a.expect_md5},
             "n_tokens": {"ok": len(toks) == a.expect_n, "n": len(toks), "expect": a.expect_n},
             "n_logs": {"ok": n_logs == a.expect_logs, "n": n_logs, "expect": a.expect_logs},
             "every_token_logged": {"ok": all(t in tl for t in toks)},
             "scored_arm": {"ok": not miss_arm, "missing": len(miss_arm)},
             "scored_deployed": {"ok": not miss_base, "missing": len(miss_base)}}
    res = {"name": a.name, "arm_csv": str(arm_p), "deployed_csv": str(BASE_CSV), "tokens": str(tok_p),
           "status": "REPORTED, NOT GATING (RESULT_SFT4.md 'Next arm' 1)",
           "estimator": "paired log-cluster bootstrap over the subset's logs, 10,000 resamples, 95 %, seed 20260927",
           "tier": "NAVSIM v1.1 PDMS = ego pseudo-simulation of an open-loop plan against logged agents",
           "gates": gates}
    out = Path(a.out) if a.out else HERE.parent / "raw" / "2026-10-05-sft-stage2" / a.name / "unseen_subset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not all(g["ok"] for g in gates.values()):
        res["result"] = None
        json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
        bad = [k for k, g in gates.items() if not g["ok"]]
        print("ZZUNSEEN", a.name, "GATE_FAIL", ",".join(bad), flush=True)
        return 1
    A = {t: 100.0 * float(arm[t]) for t in toks}
    B = {t: 100.0 * float(base[t]) for t in toks}
    Dd = {t: A[t] - B[t] for t in toks}
    mu, lo, hi = PE.boot(Dd, {t: tl[t] for t in toks})
    res["result"] = {"mean": mu, "ci95": [lo, hi], "pdms_deployed": float(np.mean(list(B.values()))),
                     "pdms_arm": float(np.mean(list(A.values()))), "n": len(toks), "n_logs": n_logs,
                     "better": sum(v > 0 for v in Dd.values()), "worse": sum(v < 0 for v in Dd.values()),
                     "tied": sum(v == 0 for v in Dd.values()), "zero_deployed": sum(v == 0 for v in B.values()),
                     "zero_arm": sum(v == 0 for v in A.values())}
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print("ZZUNSEEN", a.name, f"{mu:+.2f} [{lo:+.2f}, {hi:+.2f}]", f"n={len(toks)} logs={n_logs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

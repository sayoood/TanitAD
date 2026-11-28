#!/usr/bin/env python3
"""P4 follow-on F1 (eval/PREREG_P4_TEACHER_NAVTEST.md §9, REPORT-ONLY): does DriveRL's test-time search (N = 16)
repair the registered teacher's failures without harming its passes? -- the cheapest test of P5's re-query arm.

  select   python eval/p4_f1_tts.py select  --teacher-csv <registered teacher csv> --out-dir <dir>
           -> <dir>/f1_tokens.json (W3 format: fail sample (a) + pass sample (b)), f1_selection.json
  analyze  python eval/p4_f1_tts.py analyze --teacher-csv <csv> --tts-csv <F1 csv> --out-dir <dir>
           -> <dir>/f1_result.json, the committed outcome for P5's design.
Selection (registered): primary set = teacher-scored tokens minus the 36 C1-known; fail = NC*DAC == 0; rng =
random.Random(20261005); (a) = rng.sample(sorted(fail), 120) if > 120 else sorted(fail); (b) = rng.sample(sorted(pass), 80).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import proxy_eval as PE  # noqa: E402
import teacher_navtest_analyze as A  # noqa: E402


def primary_tokens(T: dict) -> list:
    known = set(json.load(open(A.C1_KNOWN, encoding="utf-8")).get("over_bar", {}))
    return sorted(t for t in T if t not in known)


def select(a) -> int:
    T = A.per_token(a.teacher_csv)
    P = primary_tokens(T)
    fail = sorted(t for t in P if T[t]["NC"] * T[t]["DAC"] == 0)
    pas = sorted(t for t in P if T[t]["NC"] * T[t]["DAC"] > 0)
    rng = random.Random(20261005)
    fa = rng.sample(fail, 120) if len(fail) > 120 else list(fail)
    pa = rng.sample(pas, 80)
    exp = json.load(gzip.open(A.EXPORT, "rt", encoding="utf-8"))["tokens"]
    toks = sorted(set(fa) | set(pa))
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({"rule": "P4 F1 (PREREG_P4 §9): teacher-fail sample (a) + teacher-pass harm control (b), "
                       "random.Random(20261005)", "tokens": toks, "token_log": {t: exp[t]["log_name"] for t in toks}},
              open(os.path.join(a.out_dir, "f1_tokens.json"), "w"), indent=0)
    sel = {"n_primary": len(P), "n_fail": len(fail), "n_pass": len(pas), "fail_sample": sorted(fa),
           "pass_sample": sorted(pa), "run": len(fail) >= 30}
    json.dump(sel, open(os.path.join(a.out_dir, "f1_selection.json"), "w"), indent=1)
    print(f"ZZF1_SELECT fail={len(fail)} sampled={len(fa)} pass_sampled={len(pa)} run={sel['run']}")
    return 0


def analyze(a) -> int:
    exp = json.load(gzip.open(A.EXPORT, "rt", encoding="utf-8"))["tokens"]
    tl = {t: exp[t]["log_name"] for t in exp}
    T, X = A.per_token(a.teacher_csv), A.per_token(a.tts_csv)
    sel = json.load(open(os.path.join(a.out_dir, "f1_selection.json"), encoding="utf-8"))
    fa = [t for t in sel["fail_sample"] if t in X]
    pa = [t for t in sel["pass_sample"] if t in X]
    rep = {t: 100.0 * (X[t]["NC"] * X[t]["DAC"] > 0) for t in fa}
    harm = {t: 100.0 * (X[t]["NC"] * X[t]["DAC"] == 0) for t in pa}
    res = {"what": "F1: TTS N=16 vs the registered teacher (tts 0), paired on the same tokens",
           "n_fail_scored": len(fa), "n_fail_asked": len(sel["fail_sample"]), "n_pass_scored": len(pa),
           "n_pass_asked": len(sel["pass_sample"]),
           "repair_rate_pct": A.ci(rep, tl), "harm_rate_pct": A.ci(harm, tl),
           "fail_set": {"teacher": A.block(T, fa, tl), "tts16": A.block(X, fa, tl),
                        "tts16_minus_teacher": A.paired(X, T, fa, tl)},
           "pass_set": {"teacher": A.block(T, pa, tl), "tts16": A.block(X, pa, tl),
                        "tts16_minus_teacher": A.paired(X, T, pa, tl)}}
    r, h = res["repair_rate_pct"].get("mean"), res["harm_rate_pct"].get("mean")
    if r is None or h is None:
        res["outcome"] = "INCOMPLETE"
    elif r >= 50.0 and h <= 2.0:
        res["outcome"] = "RE-QUERY IS A VIABLE P5 ARM"
    elif r < 20.0:
        res["outcome"] = "RE-QUERY IS NOT THE LEVER: P5 uses the DROP arm (or a different repair)"
    else:
        res["outcome"] = "MIXED: carry both arms into P5"
    res["rule"] = "repair >= 50 % AND harm <= 2 % => viable; repair < 20 % => not the lever; else MIXED (PREREG_P4 §9)"
    json.dump(res, open(os.path.join(a.out_dir, "f1_result.json"), "w"), indent=1)
    print(f"ZZF1 {res['outcome']} | repair {r} {res['repair_rate_pct'].get('ci95')} | harm {h} "
          f"{res['harm_rate_pct'].get('ci95')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("select", "analyze"))
    ap.add_argument("--teacher-csv", required=True)
    ap.add_argument("--tts-csv", default=None)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    return select(a) if a.mode == "select" else analyze(a)


if __name__ == "__main__":
    sys.exit(main())

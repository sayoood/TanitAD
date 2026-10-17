#!/usr/bin/env python3
"""zt5_confirm_read.py -- the ONE-SHOT navtest read of PREREG_ZT5 section 7 (run by the Master Mind only).

    python zt5_confirm_read.py --fixed <harness csv of the F_c seam> --base <u400 navtest_full harness csv> \
        --tokens D:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json --out <confirm.json> \
        [--unseen D:/Projects/TanitAD/data/refe_navtest/navtest_unseen_by_sft4u400_tokens.json] [--zt3 <per_token_A.jsonl>]

Statistic: per-token PDMS(fixed) - PDMS(base), mean over the paired tokens, paired log-cluster bootstrap over the
navtest logs, 10,000 resamples, seed 20260927 (zero_tail.Boot).  ADOPT iff lower bound > 0 AND no NAVSIM sub-score
(NC, DAC, EP, TTC, C) separates adversely (upper < 0); REFUTED iff upper < 0; else NOT PROVEN.  Gates: both CSVs carry
every token valid; the token sets are equal; the token file's count matches.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zero_tail as zt  # noqa: E402

COLS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "EP": "ego_progress",
        "TTC": "time_to_collision_within_bound", "C": "comfort", "PDMS": "score"}


def read_csv(p):
    out = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["token"] == "average":
                continue
            out[r["token"]] = ({k: float(r[c]) for k, c in COLS.items()}, r["valid"] == "True")
    return out


def paired(fx, bs, toks, tok_log):
    logs = sorted({tok_log[t] for t in toks})
    li = np.array([logs.index(tok_log[t]) for t in toks])
    boot = zt.Boot(li, len(logs))
    res = {"n_tokens": len(toks), "n_logs": len(logs)}
    for k in COLS:
        d = np.array([fx[t][0][k] - bs[t][0][k] for t in toks])
        res[k] = boot.ratio(d, np.ones_like(d), scale=100.0)
    return res


def verdict(r):
    adverse = [k for k in ("NC", "DAC", "EP", "TTC", "C") if r[k]["hi"] < 0]
    if r["PDMS"]["lo"] > 0 and not adverse:
        return "ADOPT", adverse
    if r["PDMS"]["hi"] < 0:
        return "REFUTED", adverse
    return "NOT PROVEN", adverse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unseen", default=None)
    ap.add_argument("--zt3", default=None)
    a = ap.parse_args()
    fx, bs = read_csv(a.fixed), read_csv(a.base)
    tf = json.load(open(a.tokens, encoding="utf-8"))
    toks, tok_log = list(tf["tokens"]), tf["token_log"]
    gates = {"same_token_set": set(fx) == set(bs) == set(toks), "all_valid_fixed": all(v[1] for v in fx.values()),
             "all_valid_base": all(v[1] for v in bs.values()), "n": len(toks)}
    out = {"prereg": "PREREG_ZT5.md section 7", "gates": gates}
    if not all(v for k, v in gates.items() if k != "n"):
        out["verdict"] = "GATE FAIL -- no statistic read"
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print(json.dumps(out, indent=1))
        return 3
    r = paired(fx, bs, toks, tok_log)
    out["primary"] = r
    out["verdict"], out["adverse_subscores"] = verdict(r)
    if a.unseen:
        u = json.load(open(a.unseen, encoding="utf-8"))
        ut = [t for t in (u["tokens"] if isinstance(u, dict) else u) if t in fx]
        out["reported_unseen_subset"] = paired(fx, bs, ut, tok_log)
    if a.zt3:
        tm = [json.loads(ln) for ln in open(a.zt3, encoding="utf-8")]
        tmt = [x["token"] for x in tm if x["tracker_made"]]
        out["reported_zt3_tracker_made"] = {"n": len(tmt), "dac_rescued": int(sum(fx[t][0]["DAC"] == 1.0 for t in tmt)),
                                            "pdms_gain_sum_x100": float(sum(fx[t][0]["PDMS"] - bs[t][0]["PDMS"] for t in tmt) * 100)}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

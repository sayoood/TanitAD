#!/usr/bin/env python3
"""EXPLORATORY: two snapshots compared UNDER ONE SELECTION RULE, paired on the same 200 tokens.

From SPEC Amendment 5 (2026-09-26 21:23 Berlin) the planner picks with NAVSIM v1's formula; every earlier snapshot was
picked with NAVSIM v2's EPDMS shape. The learning curve's "vs previous" then mixes a model change with a rule change.
This re-selects BOTH snapshots' stored E-6 tables (their own logits, the harness's own per-proposal PDMS; no model run,
no harness run) with the SAME rule and bootstraps the paired difference of the two picks over log clusters.
Checks before any number: each table's own shipped rule must reproduce its stored pick on 100 % of tokens, and the two
tables must hold the same tokens in the same order.

    python eval/snapshot_pair_under_rule.py --a sub200_ep013 --b sub200_ep014 [--boot 10000] [--out-json F]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

DATA = "D:/Projects/TanitAD/data/refe_navtest"
TOK = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
       "A1_sub200_tokens.json")
RULE_NAMES = {"v2_shape": "NAVSIM v2 EPDMS shape", "navsim_v1": "NAVSIM v1 formula"}


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def aggregate(logits, order, rule):
    """the planner's two rules (refe/planner.py): v2_shape NC*DAC*DDC*(5EP+5TTC+4C)/14, navsim_v1 NC*DAC*(5EP+5TTC+2C)/12"""
    p = sig(logits.astype(np.float64))
    g = {k: p[..., order.index(k)] for k in ("NC", "DAC", "EP", "TTC", "C", "DDC")}
    if rule == "navsim_v1":
        return g["NC"] * g["DAC"] * (5 * g["EP"] + 5 * g["TTC"] + 2 * g["C"]) / 12
    if rule == "v2_shape":
        return g["NC"] * g["DAC"] * g["DDC"] * (5 * g["EP"] + 5 * g["TTC"] + 4 * g["C"]) / 14
    raise ValueError(rule)


def load(name):
    T = np.load(os.path.join(DATA, "proptable", name, "table.npz"))
    order = [str(x) for x in T["head_order"]]
    shipped = str(T["rule"]) if "rule" in T.files else "v2_shape"
    rep = float((aggregate(T["logits"], order, shipped).argmax(1) == T["pick"]).mean())
    assert rep == 1.0, (name, "its own shipped rule reproduces the stored pick on only", rep)
    return {"token": [str(t) for t in T["token"]], "pdms": T["pdms"].astype(np.float64), "logits": T["logits"],
            "order": order, "shipped": shipped, "pick": T["pick"]}


def compare(a_name: str, b_name: str, boot: int = 10000, exclude=None) -> dict:
    """the pair under each rule; `eval/record_snapshot.py` calls this for every new snapshot. `exclude`: tokens left
    out of the pair (SPEC Amendment 8: the tokens its goal sanitisation fires on, when only one side has it)"""
    A, B = load(a_name), load(b_name)
    assert A["token"] == B["token"], "the two tables must hold the same tokens in the same order"
    ex = set(exclude or ())
    keep = np.array([t not in ex for t in A["token"]])
    for X in (A, B):
        X["token"] = [t for t, k in zip(X["token"], keep) if k]
        X["pdms"], X["logits"], X["pick"] = X["pdms"][keep], X["logits"][keep], X["pick"][keep]
    tl = json.load(open(TOK, encoding="utf-8"))["token_log"]
    logs = np.array([tl[t] for t in A["token"]])
    ul = np.unique(logs)
    idx = {l: np.where(logs == l)[0] for l in ul}
    rng = np.random.default_rng(20260927)
    draws = [np.concatenate([idx[l] for l in rng.choice(ul, size=len(ul), replace=True)]) for _ in range(boot)]
    N = len(A["token"])
    ar = np.arange(N)
    out = {"_label": "EXPLORATORY: two snapshots under one selection rule, paired on the same tokens (SPEC E-6 tables); "
                     "never a claim without a disjoint-token confirmation",
           "a": a_name, "b": b_name, "a_shipped_rule": A["shipped"], "b_shipped_rule": B["shipped"], "N": N,
           "n_logs": int(len(ul)), "boot": boot, "excluded_tokens": sorted(ex & set(tl)), "rules": {}}
    for rule in ("navsim_v1", "v2_shape"):
        pa = A["pdms"][ar, aggregate(A["logits"], A["order"], rule).argmax(1)]
        pb = B["pdms"][ar, aggregate(B["logits"], B["order"], rule).argmax(1)]
        d = pb - pa
        bs = np.array([100 * d[ix].mean() for ix in draws])
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        out["rules"][rule] = {"name": RULE_NAMES[rule], "a_pdms": round(100 * pa.mean(), 2), "b_pdms": round(100 * pb.mean(), 2),
                              "b_minus_a": round(100 * d.mean(), 2), "ci95": [round(lo, 2), round(hi, 2)],
                              "separated": bool(lo > 0 or hi < 0),
                              "a_is_shipped": rule == A["shipped"], "b_is_shipped": rule == B["shipped"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="the earlier snapshot, e.g. sub200_ep013")
    ap.add_argument("--b", required=True, help="the later snapshot, e.g. sub200_ep014")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    out = compare(a.a, a.b, a.boot)
    for rule, r in out["rules"].items():
        print(f"ZZPAIR {rule:9s} {a.a} {r['a_pdms']:6.2f}{'*' if r['a_is_shipped'] else ' '} -> {a.b} {r['b_pdms']:6.2f}"
              f"{'*' if r['b_is_shipped'] else ' '}  diff {r['b_minus_a']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
              f"{'  separated' if r['separated'] else '  not separated'}   (* = the rule that snapshot shipped with)")
    if a.out_json:
        json.dump(out, open(a.out_json, "w", encoding="utf-8"), indent=1)
        print("wrote", a.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

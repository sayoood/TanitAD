#!/usr/bin/env python3
"""EXPLORATORY (post-hoc, 2026-09-26 ~20:20 Berlin, after the Amendment 4 FAILURE was read): does the planner's
SELECTION RULE match the metric it is scored on?

`refe/planner.py:aggregate` picks by NC x DAC x DDC x (5 EP + 5 TTC + 4 C) / 14 -- NAVSIM v2's EPDMS shape, comfort
standing for v2's two weight-2 terms. The harness scores NAVSIM **v1** PDMS (navsim @ 3e8291b,
`pdm_scorer.py:38-42`): NC x DAC x (5 EP + 5 TTC + 2 C) / 12, driving direction at weight 0. This re-selects on the
SAME stored logits of each E-6 table and reads the TRUE PDMS of each rule's pick from the same table -- no model run,
no harness run. Label: exploratory, same 200 tokens, never a claim without a disjoint-token confirmation (SPEC E-6).

    python eval/rule_mismatch_diag.py [--names sub200_ep011 sub200_ep012 sub200_ep013] [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

DATA = "D:/Projects/TanitAD/data/refe_navtest"
TOK = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
       "A1_sub200_tokens.json")


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def rules(logits, order):
    """logits [N, M, 6] in the head order `order`; returns {rule: aggregate [N, M]}."""
    p = sig(logits.astype(np.float64))
    g = {k: p[..., order.index(k)] for k in ("NC", "DAC", "EP", "TTC", "C", "DDC")}
    return {
        "planner (v2-shaped, as shipped)": g["NC"] * g["DAC"] * g["DDC"] * (5 * g["EP"] + 5 * g["TTC"] + 4 * g["C"]) / 14,
        "NAVSIM v1 formula": g["NC"] * g["DAC"] * (5 * g["EP"] + 5 * g["TTC"] + 2 * g["C"]) / 12,
        "v1 formula without C": g["NC"] * g["DAC"] * (5 * g["EP"] + 5 * g["TTC"]) / 10,
        "v2-shaped without DDC": g["NC"] * g["DAC"] * (5 * g["EP"] + 5 * g["TTC"] + 4 * g["C"]) / 14,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=["sub200_ep011", "sub200_ep012", "sub200_ep013"])
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    tl = json.load(open(TOK, encoding="utf-8"))["token_log"]
    out = {"_label": "EXPLORATORY post-hoc rules on the same 200 tokens (SPEC E-6); never claimed without a "
                     "disjoint-token confirmation", "boot": a.boot}
    for name in a.names:
        T = np.load(os.path.join(DATA, "proptable", name, "table.npz"))
        order = [str(x) for x in T["head_order"]]
        pdms, logits, pick = T["pdms"], T["logits"], T["pick"]
        N = pdms.shape[0]
        ar = np.arange(N)
        logs = np.array([tl[str(t)] for t in T["token"]])
        ul = np.unique(logs)
        idx = {l: np.where(logs == l)[0] for l in ul}
        rng = np.random.default_rng(20260926)
        draws = [np.concatenate([idx[l] for l in rng.choice(ul, size=len(ul), replace=True)]) for _ in range(a.boot)]
        orc, rnd = pdms.max(1), pdms.mean(1)
        res = {}
        R = rules(logits, order)
        shipped = R["planner (v2-shaped, as shipped)"].argmax(1)
        res["_pick_reproduced"] = float((shipped == pick).mean())
        for k, agg in R.items():
            pk = pdms[ar, agg.argmax(1)]

            def stat(ix):
                o, r_, p_ = orc[ix].mean(), rnd[ix].mean(), pk[ix].mean()
                return 100 * p_, (p_ - r_) / (o - r_) if o - r_ > 1e-12 else np.nan

            m = stat(ar)
            bs = np.array([stat(ix) for ix in draws])
            dd = np.array([100 * (pk[ix] - pdms[ix, pick[ix]]).mean() for ix in draws])
            res[k] = {"pdms": round(m[0], 2), "pdms_ci95": [round(float(np.percentile(bs[:, 0], 2.5)), 2),
                                                            round(float(np.percentile(bs[:, 0], 97.5)), 2)],
                      "skill": round(float(m[1]), 3), "skill_ci95": [round(float(np.nanpercentile(bs[:, 1], 2.5)), 3),
                                                                      round(float(np.nanpercentile(bs[:, 1], 97.5)), 3)],
                      "minus_shipped_pick": round(float(100 * (pk - pdms[ar, pick]).mean()), 2),
                      "minus_shipped_ci95": [round(float(np.percentile(dd, 2.5)), 2), round(float(np.percentile(dd, 97.5)), 2)]}
        res["_oracle"], res["_random"] = round(100 * orc.mean(), 2), round(100 * rnd.mean(), 2)
        out[name] = res
    p = os.path.join(DATA, "proptable", "rule_mismatch_diag.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    for name in a.names:
        r = out[name]
        print(f"== {name}: oracle {r['_oracle']}, random {r['_random']}, shipped pick reproduced on {100 * r['_pick_reproduced']:.1f} % of tokens")
        for k, v in r.items():
            if k.startswith("_"):
                continue
            print(f"   {k:34s} PDMS {v['pdms']:6.2f} {v['pdms_ci95']}  skill {v['skill']:6.3f} {v['skill_ci95']}  "
                  f"vs shipped {v['minus_shipped_pick']:+6.2f} {v['minus_shipped_ci95']}")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

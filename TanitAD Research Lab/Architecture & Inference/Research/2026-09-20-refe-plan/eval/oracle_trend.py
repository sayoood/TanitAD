#!/usr/bin/env python3
"""EXPLORATORY: is the CEILING of the 64 proposals moving? Paired, per token, across snapshots on the same 200 tokens.

The E-6 tables hold the harness's own PDMS for every proposal (no model run, no harness run). For each pair of
snapshots this compares the best of 64 (the ceiling a perfect scorer would reach) and the mean of 64 (a random pick),
with the same paired log-cluster bootstrap as snapshot_pair_under_rule.py. Written 2026-09-27 when the best-of-64 read
84.2 after epoch 15 against 91.3 after epoch 12, while the pick kept rising: the question is whether the proposals
lose ground as the scorer gains it.

    python eval/oracle_trend.py --pairs 012:015 014:015 [--boot 10000] [--out-json F]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import snapshot_pair_under_rule as SP


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="AAA:BBB snapshot numbers, e.g. 012:015")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    tl = json.load(open(SP.TOK, encoding="utf-8"))["token_log"]
    out = {"_label": "EXPLORATORY: best-of-64 and mean-of-64 PDMS across snapshots, paired on the same tokens (E-6 tables)",
           "boot": a.boot, "pairs": {}}
    for pr in a.pairs:
        x, y = pr.split(":")
        A, B = SP.load(f"sub200_ep{x.zfill(3)}"), SP.load(f"sub200_ep{y.zfill(3)}")
        assert A["token"] == B["token"]
        logs = np.array([tl[t] for t in A["token"]])
        ul = np.unique(logs)
        idx = {l: np.where(logs == l)[0] for l in ul}
        rng = np.random.default_rng(20260927)
        draws = [np.concatenate([idx[l] for l in rng.choice(ul, size=len(ul), replace=True)]) for _ in range(a.boot)]
        res = {}
        for k, f in (("best_of_64", lambda T: T["pdms"].max(1)), ("mean_of_64", lambda T: T["pdms"].mean(1)),
                     ("share_ge_80", lambda T: (T["pdms"] >= 0.8).mean(1))):
            va, vb = f(A), f(B)
            d = vb - va
            bs = np.array([100 * d[ix].mean() for ix in draws])
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            res[k] = {"a": round(100 * va.mean(), 2), "b": round(100 * vb.mean(), 2), "b_minus_a": round(100 * d.mean(), 2),
                      "ci95": [round(lo, 2), round(hi, 2)], "separated": bool(lo > 0 or hi < 0)}
            print(f"ZZTREND {x}->{y} {k:11s} {res[k]['a']:6.2f} -> {res[k]['b']:6.2f}  diff {res[k]['b_minus_a']:+.2f} "
                  f"[{lo:+.2f}, {hi:+.2f}]{'  separated' if res[k]['separated'] else '  not separated'}")
        out["pairs"][f"{x}:{y}"] = res
    if a.out_json:
        json.dump(out, open(a.out_json, "w", encoding="utf-8"), indent=1)
        print("wrote", a.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""EXPLORATORY: WHICH proposals disappeared where the best of 64 got worse between two snapshots (same 200 tokens).

oracle_trend.py showed the best of 64 falling (91.3 after epoch 12 -> 84.2 after epoch 15) while the mean of 64 held.
This reads both E-6 tables (the harness's own per-proposal PDMS and sub-scores, and the proposals' poses; no model run,
no harness run) and asks, on the tokens whose best fell by >= --drop PDMS points: is the old best proposal's mode still
in the new set (distance from its endpoint to the nearest new endpoint), where did it sit in its own set (its path-length
rank; 0 = the shortest of the 64), and what do the new best proposal's sub-scores lose? Plus the proposal fan's lengths
on all tokens. Path length = the polyline from the origin through the 8 poses.

    python eval/lost_modes.py --a 012 --b 015 [--drop 0.10] [--out-json F]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

D = "D:/Projects/TanitAD/data/refe_navtest/proptable"


def path_len(P):
    xy = np.concatenate([np.zeros_like(P[..., :1, :2]), P[..., :2]], axis=-2)
    return np.linalg.norm(np.diff(xy, axis=-2), axis=-1).sum(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--drop", type=float, default=0.10, help="PDMS fraction the best must lose (0.10 = 10 points)")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    A = np.load(f"{D}/sub200_ep{a.a.zfill(3)}/table.npz")
    B = np.load(f"{D}/sub200_ep{a.b.zfill(3)}/table.npz")
    assert (A["token"] == B["token"]).all()
    names = [str(x) for x in A["sub_names"]]
    N = len(A["token"])
    ar = np.arange(N)
    pa, pb = A["pdms"], B["pdms"]
    ia, ib = pa.argmax(1), pb.argmax(1)
    lost = (pa.max(1) - pb.max(1)) >= a.drop
    ea = A["proposals"][ar, ia, -1, :2]
    dmin = np.linalg.norm(B["proposals"][:, :, -1, :2] - ea[:, None, :], axis=-1).min(1)
    la, lb = path_len(A["proposals"]), path_len(B["proposals"])
    rank_a = (la < la[ar, ia][:, None]).mean(1)
    med = lambda x: round(float(np.median(x)), 2)
    out = {"_label": "EXPLORATORY (E-6 tables, same tokens)", "a": a.a, "b": a.b, "drop": a.drop, "N": N,
           "tokens_best_fell": int(lost.sum()), "tokens_best_rose": int(((pb.max(1) - pa.max(1)) >= a.drop).sum()),
           "old_best_to_nearest_new_endpoint_m": {"lost_tokens_median": med(dmin[lost]), "other_tokens_median": med(dmin[~lost])},
           "old_best_length_rank_lost_tokens_median": med(rank_a[lost]),
           "lost_tokens_path_m": {"old_best": med(la[ar, ia][lost]), "new_best": med(lb[ar, ib][lost]),
                                  "new_shortest": med(lb.min(1)[lost]), "old_shortest": med(la.min(1)[lost])},
           "all_tokens_path_m": {"pick_a": med(la[ar, A["pick"]]), "pick_b": med(lb[ar, B["pick"]]),
                                 "shortest_a": med(la.min(1)), "shortest_b": med(lb.min(1)),
                                 "longest_a": med(la.max(1)), "longest_b": med(lb.max(1)),
                                 "p90_minus_p10_a": med(np.percentile(la, 90, 1) - np.percentile(la, 10, 1)),
                                 "p90_minus_p10_b": med(np.percentile(lb, 90, 1) - np.percentile(lb, 10, 1))},
           "lost_tokens_best_subscores": {n: [round(float(A["sub"][ar, ia][lost, k].mean()), 3),
                                              round(float(B["sub"][ar, ib][lost, k].mean()), 3)] for k, n in enumerate(names)}}
    print(json.dumps(out, indent=1))
    if a.out_json:
        json.dump(out, open(a.out_json, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

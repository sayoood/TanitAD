#!/usr/bin/env python3
"""pair_arms.py — PAIRED corridor-departure delta between two closed-loop arms
on the IDENTICAL windows.

Two single-arm intervals combined in quadrature is not merely weaker here — it
is invalid, because the arms share windows and the estimates are not independent
(CLAUDE.md; ``taniteval.corridor.paired_stratum_delta``). This joins two
``*_perwindow_K<K>.pt`` dumps on ``(episode_index, t0)`` and emits
``paired_episode_cluster_bootstrap`` deltas per stratum.

Orientation follows ``corridor.paired_stratum_delta``: POSITIVE => arm ``a``
departs the corridor LESS OFTEN, i.e. ``a`` wins.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

for _p in ("/root/TanitAD/stack", "/root/TanitAD/stack/scripts", "/root/taniteval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from taniteval import ci as _ci                                # noqa: E402
from taniteval import corridor as _corr                        # noqa: E402


def _key(d):
    return [(int(x), int(y)) for x, y in zip(d["epi"], d["t0"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="per-window .pt, arm A")
    ap.add_argument("--b", required=True, help="per-window .pt, arm B")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--threshold", type=float, default=1.75)
    ap.add_argument("--junction-deg", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = torch.load(args.a, weights_only=False)
    B = torch.load(args.b, weights_only=False)
    ka, kb = _key(A), _key(B)
    common = sorted(set(ka) & set(kb))
    oa = {c: i for i, c in enumerate(ka)}
    ob = {c: i for i, c in enumerate(kb)}
    ia = np.array([oa[c] for c in common])
    ib = np.array([ob[c] for c in common])

    lat_a = A["lat"].numpy()[ia]
    lat_b = B["lat"].numpy()[ib]
    assert lat_a.shape == lat_b.shape, (lat_a.shape, lat_b.shape)
    eid = [A["eid"][i] for i in ia]
    hd = A["hd2s"].numpy()[ia]
    spd = A["speed"].numpy()[ia]
    junc = _corr.junction_mask(hd, args.junction_deg)
    long_ = (~junc) & (spd >= np.median(spd))
    strata = {"overall": np.ones(len(hd), bool), "junction": junc,
              "longitudinal": long_, "other": (~junc) & (~long_)}

    out = {
        "_estimator": ("paired_episode_cluster_bootstrap (taniteval/ci.py), "
                       "B=2000, resampling unit = val EPISODE. NEVER "
                       "overlapping_holdout_se, and NEVER a quadrature "
                       "combination of two single-arm intervals — the arms share "
                       "windows, so those estimates are not independent."),
        "_orientation": (f"POSITIVE => `{args.label_a}` departs the corridor "
                         f"LESS often = `{args.label_a}` wins"),
        "arm_a": args.label_a, "arm_a_file": args.a,
        "arm_b": args.label_b, "arm_b_file": args.b,
        "horizon_K": int(lat_a.shape[1]),
        "horizon_s": round(lat_a.shape[1] * 0.1, 2),
        "n_common_windows": len(common),
        "n_common_episodes": len(set(eid)),
        "threshold_m": args.threshold,
        "n_by_stratum": {k: int(m.sum()) for k, m in strata.items()},
    }
    for nm, m in strata.items():
        idx = np.flatnonzero(m)
        if len(idx) < 2 or len(set(eid[i] for i in idx)) < 2:
            out[nm] = {"skipped": f"n={len(idx)} too small to bootstrap"}
            continue
        e = [eid[i] for i in idx]
        la, lb = lat_a[idx], lat_b[idx]
        out[nm] = {
            "n_windows": int(len(idx)), "n_episodes": int(len(set(e))),
            f"cdr_{args.label_a}": float(
                _corr.corridor_departure(la, args.threshold).mean()),
            f"cdr_{args.label_b}": float(
                _corr.corridor_departure(lb, args.threshold).mean()),
            "d_corridor_departure_rate": _corr.paired_stratum_delta(
                la, lb, e, threshold=args.threshold),
            "d_peak_xte_m": _ci.paired_episode_cluster_bootstrap(
                lb.max(1), la.max(1), e, n_boot=2000),
            f"peak_xte_{args.label_a}": float(la.max(1).mean()),
            f"peak_xte_{args.label_b}": float(lb.max(1).mean()),
        }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("n_common_windows", "n_by_stratum")}, indent=2))
    for nm in strata:
        n = out[nm]
        if "d_corridor_departure_rate" in n:
            d = n["d_corridor_departure_rate"]
            print(f"{nm:14s} n={n['n_windows']:4d} "
                  f"{args.label_a}={n[f'cdr_{args.label_a}']:.4f} "
                  f"{args.label_b}={n[f'cdr_{args.label_b}']:.4f} "
                  f"delta={d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]",
                  flush=True)
    print(f"PAIR_DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

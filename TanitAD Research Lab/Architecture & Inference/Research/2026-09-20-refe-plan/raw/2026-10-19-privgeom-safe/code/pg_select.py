"""PRIVGEOM-SAFE subset selection (LABEL-FREE): the pool is the f10 logs whose nuPlan DB is held locally; the order is a seeded permutation of whole logs; the subset is the longest prefix whose
measured-cost projection (x SAFETY) fits the CPU-h cap.  Reads store keys (token / step / rank / ckpt_step) and the census meta's NON-label members only.

    python pg_select.py --cost run/cost/cost.json --out run/subset.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                       # noqa: E402

import pg_extract as X                                                                   # noqa: E402

SEED = 20261019
CAP_CPU_H = 16.0
SAFETY = 1.5
MIN_LOGS = 8
META = "C:/Users/Admin/par18/fastneg/run/inputs/meta.npz"
META_NONLABEL = ("log_idx", "fold", "code", "key_log", "key_token", "key_step", "key_rank", "key_ckpt_step", "log_rank", "log_names")


def cost_model(rep: dict) -> dict:
    """seconds per log / token / set from the measured 20-set record (means; a token costs parts + prep)."""
    return {"s_per_log": float(np.mean(rep["t_build_per_log_s"])),
            "s_per_token": float(np.mean(rep["t_parts_per_token_s"]) + np.mean(rep["t_prep_per_token_s"])),
            "s_per_set": float(np.mean(rep["t_score_per_set_s"]) + np.mean(rep["t_geom_per_set_s"]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = json.load(open(a.cost, encoding="utf-8"))
    cm = cost_model(rep)
    elig = X.eligible_logs()
    pool = sorted(elig)
    stats = {}
    for n in pool:
        k = X.store_keys(n)
        stats[n] = (len(k["token"]), len(set(k["token"].astype("U").tolist())))
    order = [pool[i] for i in np.random.default_rng(SEED).permutation(len(pool))]
    cum, take = 0.0, []
    for n in order:
        c = cm["s_per_log"] + cm["s_per_token"] * stats[n][1] + cm["s_per_set"] * stats[n][0]
        if (cum + c) * SAFETY > CAP_CPU_H * 3600.0:
            break
        cum += c
        take.append(n)
    if len(take) < MIN_LOGS:
        print(f"STOP: only {len(take)} logs fit the {CAP_CPU_H} CPU-h cap (< {MIN_LOGS}); cost model {cm}")
        return 3
    z = np.load(META, allow_pickle=False)
    M = {k: z[k] for k in META_NONLABEL}                                                  # never indexes 'labels6'
    where = {(str(l), str(t), int(s), int(r), int(c)): i for i, (l, t, s, r, c) in enumerate(zip(M["key_log"], M["key_token"], M["key_step"], M["key_rank"], M["key_ckpt_step"]))}
    census_idx = []
    for n in take:
        k = X.store_keys(n)
        for j in range(len(k["token"])):
            key = (n, str(k["token"][j]), int(k["step"][j]), int(k["rank"][j]), int(k["ckpt_step"][j]))
            if key not in where:
                print("STOP: store row not in the census bank:", key)
                return 4
            census_idx.append(where[key])
    census_idx = np.array(sorted(census_idx), np.int64)
    in_sub = np.zeros(len(M["fold"]), bool)
    in_sub[census_idx] = True
    code = M["code"][in_sub]
    fold_logs = [int(len({int(x) for x in M["log_idx"][in_sub][M["fold"][in_sub] == f]})) for f in range(5)]
    doc = {"schema": "privgeom-subset-v1", "seed": SEED, "cap_cpu_h": CAP_CPU_H, "safety": SAFETY, "min_logs": MIN_LOGS, "cost_model_s": cm, "cost_json_sha256": hashlib.sha256(open(a.cost, "rb").read()).hexdigest(),
           "pool_logs": len(pool), "pool_sets": int(sum(v[0] for v in stats.values())), "pool_tokens": int(sum(v[1] for v in stats.values())), "pool": pool, "order": order,
           "logs_in_order": take, "n_logs": len(take), "n_sets": int(len(census_idx)), "n_tokens": int(sum(stats[n][1] for n in take)),
           "projected_cpu_s": cum, "projected_cpu_h": cum / 3600.0, "projected_cpu_h_with_safety": cum * SAFETY / 3600.0, "census_idx": census_idx.tolist(), "logs_per_fold": fold_logs,
           "proposals_per_stratum_S_Q1_Q2_Q3_Q4": [int((code == c).sum()) for c in range(5)], "n_f10_logs": 117, "n_f10_sets": int(len(M["fold"]))}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k not in ("pool", "order", "logs_in_order", "census_idx")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

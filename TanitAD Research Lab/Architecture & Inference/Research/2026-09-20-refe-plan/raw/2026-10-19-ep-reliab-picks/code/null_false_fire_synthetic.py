"""EP-RELIAB-PICKS: false-fire rate of the registered decision rule under a TRUE NULL, on SYNTHETIC data only (R51: compute a control's false-RED rate before registering it).

Run BEFORE the prereg was sealed.  Reads only the label-free per-log set counts (`log_idx`) of the landed arrays; no label, logit or pick is opened.
Model: per-set paired difference d_i = log effect u_l + set noise e_i, true mean 0; two EP-row training seeds share a fraction rho of the set noise (they share almost all picks).
Rule (per seed): OVER iff lo > 0, UNDER iff hi < 0 of the 95% percentile log-cluster bootstrap CI (B = 10,000, seed 20260927) of mean(d); outcome OVER / UNDER iff BOTH seeds agree, else WITHIN.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

B, SEED, N_LOGS = 10000, 20260927, 117


def main(arrays_path: str, sims: int = 1000) -> int:
    li = np.load(arrays_path, allow_pickle=False)["log_idx"].astype(np.int64)
    n_sets = np.bincount(li, minlength=N_LOGS).astype(np.float64)
    draws = np.random.default_rng(SEED).integers(0, N_LOGS, size=(B, N_LOGS))
    den = n_sets[draws].sum(axis=1)
    rng = np.random.default_rng(12345)
    out = {"note": "SYNTHETIC; ESTIMATED false-fire rates of the two-seed rule under a true null", "sims": sims, "cells": []}
    for sd_e in (0.1, 0.4):
        for sd_u in (0.0, 0.02):
            for rho in (0.0, 0.9, 1.0):
                cnt = {"over": 0, "under": 0, "seed_over": 0, "seed_under": 0}
                for _ in range(sims):
                    u = rng.normal(0, sd_u, N_LOGS)
                    g = rng.normal(0, 1, len(li))
                    cls = []
                    for s in range(2):
                        h = rng.normal(0, 1, len(li))
                        d = sd_e * (math.sqrt(rho) * g + math.sqrt(1 - rho) * h) + u[li]
                        num = np.bincount(li, weights=d, minlength=N_LOGS)
                        v = num[draws].sum(axis=1) / den
                        lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
                        cls.append("OVER" if lo > 0 else ("UNDER" if hi < 0 else "WITHIN"))
                    cnt["seed_over"] += cls[0] == "OVER"
                    cnt["seed_under"] += cls[0] == "UNDER"
                    cnt["over"] += cls[0] == cls[1] == "OVER"
                    cnt["under"] += cls[0] == cls[1] == "UNDER"
                p_any = (cnt["over"] + cnt["under"]) / sims
                p2of5 = 1 - (1 - p_any) ** 5 - 5 * p_any * (1 - p_any) ** 4
                out["cells"].append({"sd_set": sd_e, "sd_log": sd_u, "rho": rho, "single_seed_over": cnt["seed_over"] / sims, "single_seed_under": cnt["seed_under"] / sims,
                                     "two_seed_over": cnt["over"] / sims, "two_seed_under": cnt["under"] / sims, "two_seed_any": p_any, "p_at_least_2_of_5_replicates_fire": p2of5})
                print(out["cells"][-1], flush=True)
    worst = max(c["two_seed_any"] for c in out["cells"])
    out["worst_two_seed_any"] = worst
    out["worst_p_at_least_2_of_5"] = max(c["p_at_least_2_of_5_replicates_fire"] for c in out["cells"])
    print("WORST two-seed any-fire", worst, " worst P(>=2 of 5)", out["worst_p_at_least_2_of_5"])
    with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

#!/usr/bin/env python3
"""POST-HOC ablation (not in PREREG_H_ADAM.md): closed-loop regime P with the KL anchor OFF (--epo-beta 0), lam in {0.25, 1.0}; D(0) is the BCE-only run (beta is irrelevant at lam 0).
Question: does the closed-loop super-proportionality (S 0.56 with beta 0.1, open-loop 0.32) survive when the EPO term has no anchor?
    python closed_beta0_stats.py --closed-dir <scratch/closed> --out closed_beta0_stats.json"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from adam_sim import stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed-dir", required=True)
    ap.add_argument("--out", required=True)
    q = ap.parse_args()
    ld = lambda n: torch.load(os.path.join(q.closed_dir, n), weights_only=False)  # noqa: E731
    r0, r25, r1 = ld("closed_P_0.0.pt"), ld("closed_P_0.25_beta0.pt"), ld("closed_P_1.0_beta0.pt")
    names = r0["names"]
    res = {"tool": "closed_beta0_stats/1", "class": "ESTIMATED (closed loop, regime P, K=50, stream-0 draws, substituted sctx); POST-HOC ablation", "at": {}, "log": {}}
    for k in (25, 50):
        st = stats(r0["D"][k], r25["D"][k], r1["D"][k], names)
        res["at"][str(k)] = {x: st[x] for x in ("S_C80", "S_all", "cos_C80", "U_full_norm_ratio", "elasticity_e")}
        print("beta=0 CL regime P", k, {x: round(v, 3) for x, v in res["at"][str(k)].items()})
    for tag, r in (("lam0", r0), ("lam0.25_beta0", r25), ("lam1_beta0", r1)):
        lg = r["log"]
        res["log"][tag] = {"gn_last5": sum(x["grad_norm_pre_clip"] for x in lg[-5:]) / 5, "bce_last5": sum(x["bce"] for x in lg[-5:]) / 5, "epo_last5": sum(x["epo"] for x in lg[-5:]) / 5}
    json.dump(res, open(q.out, "w"), indent=1)
    print(res["log"])
    print("ZZBETA0_OK")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""What the `rl_terms` side files look like on real on-policy proposals, and the contract test with rl_reward.py (RL-S1 part 2).

Joins rl_terms_*.jsonl to pdm_*.jsonl, feeds BOTH into rl_reward.reward(...) with the R0/R1 config (D) and the R2 config (D + K + L ON) -- so a missing or
mis-shaped field raises HERE, not on the pod -- and reports per-term rates and the size of each term's contribution to R.  numpy only.
    python eval/rl_terms_onpolicy_summary.py --dir eval/raw/rl_terms/onpolicy_local --out eval/raw/rl_terms/onpolicy_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "refe"))
import rl_reward as RW                                                          # noqa: E402


def load(d, prefix, kind):
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, prefix + "*.jsonl"))):
        for ln in open(fp, encoding="utf-8"):
            r = json.loads(ln)
            if r.get("kind") == kind:
                out[(tuple(r["key"]), int(r["ckpt_step"]))] = r
    return out


def q(v):
    v = np.asarray(v, float)
    return {k: float(np.quantile(v, p)) for k, p in (("min", 0), ("p10", .1), ("p50", .5), ("p90", .9), ("max", 1))} if v.size else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rlt, pdm = load(a.dir, "rl_terms_", "rl_terms"), load(a.dir, "pdm_", "pdm_targets")
    keys = sorted(set(rlt) & set(pdm))
    cfg_core, cfg_r2 = RW.RewardCfg(), RW.RewardCfg(use_k=True, use_l=True)
    acc = {n: [] for n in ("d", "k", "k_raw", "lc", "dither", "gmax", "lab", "human", "route", "start_lane", "pdms", "R_core", "R_r2", "dR_D", "dR_K", "dR_L", "guard")}
    argmax_changed_core = argmax_changed_r2 = ties_pdms = ties_core = ties_r2 = 0
    top_r2_pdms_gap: list = []
    for k in keys:
        r, p = rlt[k], pdm[k]
        tg = RW.tg_from_pdm_record(p)
        core = RW.reward(tg, cfg_core, d=r["d"], pdms=p["pdms"])
        lane = {"lc_executed": r["lc_executed"], "label_fires": r["label_fires"], "dither": r["dither"]}
        r2 = RW.reward(tg, cfg_r2, d=r["d"], k=r["k"], lane=lane, pdms=p["pdms"])
        acc["d"].append(r["d"]); acc["k"].append(r["k"]); acc["k_raw"].append(r["k_raw"]); acc["lc"].append(r["lc_executed"]); acc["dither"].append(r["dither"])
        acc["gmax"].append(r["gmax"]); acc["lab"].append(r["label_fires"]); acc["pdms"].append(core["pdms"])
        lb = r.get("labels", {})
        acc["human"].append(lb.get("human_lc", False)); acc["route"].append(lb.get("route_needs", False)); acc["start_lane"].append(lb.get("start_lane", False))
        acc["R_core"].append(core["R"]); acc["R_r2"].append(r2["R"])
        acc["dR_D"].append(-cfg_core.lam_d * core["D"]); acc["guard"].append(core["guard"])
        acc["dR_K"].append(-cfg_r2.lam_k * r2["K"]); acc["dR_L"].append(cfg_r2.lam_l * r2["L"])
        argmax_changed_core += int(np.argmax(core["R"]) != np.argmax(core["pdms"]))
        argmax_changed_r2 += int(np.argmax(r2["R"]) != np.argmax(core["R"]))
        ties_pdms += int((core["pdms"] >= core["pdms"].max() - 1e-9).sum() > 1)
        ties_core += int((core["R"] >= core["R"].max() - 1e-9).sum() > 1)
        ties_r2 += int((r2["R"] >= r2["R"].max() - 1e-9).sum() > 1)
        # does the top-R2 proposal keep the top PDMS (is it among the PDMS-maximal set)?  -- K / L may reorder ties, they must not override PDMS (P4)
        top_r2_pdms_gap.append(float(core["pdms"].max() - core["pdms"][int(np.argmax(r2["R"]))]))
    A = {n: np.asarray(v) for n, v in acc.items()}
    n = len(keys)
    out = {
        "scenes": n, "proposals": int(A["d"].size), "source": "dev-box ckpt-4933 on-policy dry-run sets (data/refe_m5/sets), A7-repaired executed plans",
        "contract": "rl_reward.reward() ran on every scene with the R0/R1 config (D) and the R2 config (D + K + L ON), pdms column asserted (check_pdms)",
        "D": {"share_proposals_D_gt_0": float((A["d"] > 0).mean()), "D_among_positive": q(A["d"][A["d"] > 0]), "gmax_among_positive_m": q(A["gmax"][A["gmax"] > 0])},
        "K": {"k": q(A["k"]), "k_raw": q(A["k_raw"]), "scenes_k_differs_from_k_raw": int((np.abs(A["k"] - A["k_raw"]).max(axis=1) > 0).sum())},
        "L": {"scenes_label_fires": int(A["lab"].sum()), "scene_rate_label_fires": float(A["lab"].mean()),
              "scenes_human_lane_change_i": int(A["human"].sum()), "scenes_route_needs_iii": int(A["route"].sum()), "scenes_start_on_lane": int(A["start_lane"].sum()),
              "proposal_rate_lc_executed": float(A["lc"].mean()), "proposal_rate_dither": float(A["dither"].mean())},
        "reward_term_size": {"mean_abs_lam_d_D": float(np.abs(A["dR_D"]).mean()), "mean_abs_guard": float(np.abs(A["guard"]).mean()),
                             "mean_abs_lam_k_K": float(np.abs(A["dR_K"]).mean()), "mean_abs_lam_l_L": float(np.abs(A["dR_L"]).mean()),
                             "mean_abs_pdms": float(np.abs(A["pdms"]).mean())},
        "scenes_where_argmax_R_core_differs_from_argmax_PDMS": argmax_changed_core,
        "scenes_where_argmax_R_r2_differs_from_argmax_R_core": argmax_changed_r2,
        "note_argmax": "np.argmax takes the FIRST of exact ties, so the two counts above are tie-sensitive; the tie counts below say how many scenes have a tied top",
        "scenes_with_tied_top": {"PDMS": ties_pdms, "R_core": ties_core, "R_r2": ties_r2},
        "PDMS_given_up_by_the_R2_pick_vs_best_PDMS": {"max": float(np.max(top_r2_pdms_gap)), "mean": float(np.mean(top_r2_pdms_gap)),
                                                      "scenes_with_gap_gt_0.01": int((np.asarray(top_r2_pdms_gap) > 0.01).sum())},
    }
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print("ZZRLTSUM", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())

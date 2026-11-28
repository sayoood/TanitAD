#!/usr/bin/env python3
"""S0 -- GATE. Reproduce every committed number from raw BEFORE quoting a new one.

Also runs the three FIDELITY controls the pre-registration requires (A, B, C) and
establishes window identity between the REF-C fan family and v4's `eh2_cache`
latent features, which the goal head reads.

Usage:  python gi_gate.py
"""
from __future__ import annotations

import json

import numpy as np

from gi_common import (BAR_CONFIRM, BAR_STRONG, OUT, ade, ci_paired, ci_single,
                       eid_str, endpoint_err, goal_reference, load_eh2,
                       load_refc_fan, pick_nearest_to, r4)

COMMITTED = {
    "refc_xl_as_trained": 0.4714,
    "refc_xl_oracle_in_fan": 0.1640,
    "refc_xl_R_goal2s_realised": 0.2009,
    "refc_xl_goal_vs_as_trained": -0.2705,
    "refc_xl_goal_vs_nogoal": -0.6149,
    "refc_base_oracle_in_fan": 0.1914,
    "refc_small_oracle_in_fan": 0.2213,
}


def main() -> None:
    res = {"_stream": "2026-07-27-goal-input",
           "_stage": "S0 gate + fidelity controls",
           "_estimator": "episode_cluster_bootstrap B=2000, unit=episode",
           "_bars": {"CONFIRM": BAR_CONFIRM, "STRONG": BAR_STRONG},
           "committed": COMMITTED, "reproduced": {}, "checks": {}}
    ok = []

    for arm in ("xl", "base", "small"):
        d = load_refc_fan(arm)
        fan = d["fan"].numpy().astype(np.float64)
        gt = d["gt"].numpy().astype(np.float64)
        cv = d["cv"].numpy().astype(np.float64)
        sel = d["sel"].numpy()
        eid = eid_str(d)
        W = len(gt)
        err = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)     # [W,N]
        a0 = err[np.arange(W), sel]
        oif = err.min(1)
        res["reproduced"][f"refc_{arm}_as_trained"] = r4(a0.mean())
        res["reproduced"][f"refc_{arm}_oracle_in_fan"] = r4(oif.mean())

        if arm == "xl":
            # --- the goal arm, rebuilt through gi_common's own pipeline -----
            g = gt[:, -1]
            i_goal = pick_nearest_to(goal_reference(g), fan)
            v_goal = err[np.arange(W), i_goal]
            i_cv = pick_nearest_to(cv, fan)
            v_cv = err[np.arange(W), i_cv]
            res["reproduced"]["refc_xl_R_goal2s_realised"] = r4(v_goal.mean())
            res["reproduced"]["refc_xl_goal_vs_as_trained"] = \
                ci_paired(v_goal, a0, eid)["delta"]
            res["reproduced"]["refc_xl_goal_vs_nogoal"] = \
                ci_paired(v_goal, v_cv, eid)["delta"]

            # --- FIDELITY B: nearest-to-full-GT == oracle_in_fan ------------
            i_full = pick_nearest_to(gt, fan)
            v_full = err[np.arange(W), i_full]
            fidB = float(np.abs(v_full - oif).max())
            res["checks"]["fidelity_B_pick_nearest_gt_vs_oracle_max_abs"] = fidB
            ok.append(("fidelity_B", fidB == 0.0))

            # --- FIDELITY C: proximity metric == scoring metric -------------
            # matching in FLAT 8-dim L2 must give a DIFFERENT (worse) answer;
            # if it does not, the two metrics have been silently unified and the
            # control has lost its power.
            fl_f = fan.reshape(W, fan.shape[1], 8)
            fl_g = gt.reshape(W, 8)
            i_flat = np.linalg.norm(fl_f - fl_g[:, None], axis=-1).argmin(1)
            v_flat = err[np.arange(W), i_flat]
            res["checks"]["fidelity_C_flatL2_pick_ade"] = r4(v_flat.mean())
            res["checks"]["fidelity_C_metric_mismatch_cost"] = \
                r4(v_flat.mean() - oif.mean())
            ok.append(("fidelity_C_has_power", v_flat.mean() > oif.mean()))

            # --- the endpoint axis, for S1 ----------------------------------
            e_sel = endpoint_err(fan[np.arange(W), sel][:, -1], gt[:, -1])
            e_cv = endpoint_err(cv[:, -1], gt[:, -1])
            e_oif = endpoint_err(fan[np.arange(W), i_full][:, -1], gt[:, -1])
            res["endpoint_axis_m"] = {
                "as_trained_selector": ci_single(e_sel, eid),
                "constant_velocity": ci_single(e_cv, eid),
                "best_in_fan_by_ade": ci_single(e_oif, eid),
                "gt_endpoint_norm_mean": r4(np.linalg.norm(gt[:, -1], 1).mean()
                                            if False else
                                            np.linalg.norm(gt[:, -1],
                                                           axis=-1).mean()),
                "gt_endpoint_sd_along_m": r4(gt[:, -1, 0].std()),
                "gt_endpoint_sd_cross_m": r4(gt[:, -1, 1].std()),
            }
            res["ade_axis_m"] = {
                "as_trained": ci_single(a0, eid),
                "oracle_in_fan": ci_single(oif, eid),
                "R_goal2s": ci_single(v_goal, eid),
                "R_cv": ci_single(v_cv, eid),
                "headroom_A0_minus_goal": r4(a0.mean() - v_goal.mean()),
            }

    # --- FIDELITY A: delta=0 reproduces the committed 0.2009 exactly --------
    fidA = abs(res["reproduced"]["refc_xl_R_goal2s_realised"]
               - COMMITTED["refc_xl_R_goal2s_realised"])
    res["checks"]["fidelity_A_goal2s_abs_diff"] = r4(fidA)
    ok.append(("fidelity_A", fidA == 0.0))

    # --- window identity: eh2 latent features <-> the REF-C fan -------------
    eh = load_eh2()
    dxl = load_refc_fan("xl")
    v0_eh = eh["v0"].numpy().astype(np.float64)
    v0_fan = dxl["v0"].numpy().astype(np.float64)
    max_v0 = float(np.abs(v0_eh - v0_fan).max())
    res["checks"]["window_identity_eh2_vs_refc_max_abs_v0"] = max_v0
    res["checks"]["window_identity_n"] = [int(len(v0_eh)), int(len(v0_fan))]
    ok.append(("window_identity", max_v0 == 0.0 and len(v0_eh) == len(v0_fan)))
    # lat/lon/dist must be goal-INDEPENDENT (pure latent projections): the
    # `produced` and `neutral` variants must agree bit-for-bit.
    dl = max(float((eh[k] - eh[f"neutral|{k}"]).abs().max())
             for k in ("lat", "lon", "dist"))
    res["checks"]["latent_feats_goal_independent_max_abs"] = dl
    ok.append(("latent_goal_independent", dl == 0.0))

    # --- committed-number reproduction -------------------------------------
    for k, v in COMMITTED.items():
        got = res["reproduced"].get(k)
        ok.append((f"repro:{k}", got is not None and abs(got - v) < 5e-5))

    res["_checks_detail"] = {k: bool(v) for k, v in ok}
    res["_all_ok"] = all(v for _, v in ok)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gi_gate.json").write_text(json.dumps(res, indent=2))

    print("=== S0 GATE ===")
    for k, v in COMMITTED.items():
        print(f"  {k:34s} committed {v:>9.4f}   got "
              f"{res['reproduced'].get(k, float('nan')):>9.4f}")
    print("\n=== checks ===")
    for k, v in ok:
        print(f"  {'OK ' if v else 'FAIL'}  {k}")
    print("\n=== endpoint axis (metres, 2 s endpoint L2) ===")
    for k, v in res["endpoint_axis_m"].items():
        print(f"  {k:26s} {v if not isinstance(v, dict) else v['mean']}")
    print(f"\n_all_ok = {res['_all_ok']}")
    print("wrote", OUT / "gi_gate.json")
    if not res["_all_ok"]:
        raise SystemExit("GATE FAILED -- stream stops here (pre-registration S0)")


if __name__ == "__main__":
    main()

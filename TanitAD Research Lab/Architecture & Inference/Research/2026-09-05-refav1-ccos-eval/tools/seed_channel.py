"""D-REFAV1-CCOS-EVAL — THE SEED CHANNEL. The iteration-0 exclusion bound covers the RANDOM iCEM
population (colored noise: jerk_raw >= 4.711). It does NOT cover the two injected candidates that
carry (almost) no penalty: the canonical goal SEED (`plan()` seeds the search with
`canonical_controls(lat, lon, v0)[:plan_steps]` of the decoded tactical goal — constant kappa for a
TURN token, a geometric accel profile for a lon token; zero or tiny jerk, kappa penalty
0.05 * 0.08^2 = 3.2e-04) and the proposal head's mode. Whether the SEED beats cv on the TOTAL cost
(goal + penalty) is therefore the question that decides whether a metric can make the planner
"turn" — and it is a GOAL-FOLLOWING channel (the decoder's control wins), not a searched plan.

Reads the 282-window box panel (rows `seed0`, `proposal`, `cv`; `cost_<m>` = the closure's total at
shipped weights; `goal_<m>`, `jerk_raw`, `kap_raw` per row) and reports, per metric and stratum, the
fraction of windows on which the seed / the proposal beats cv at shipped weights and at the
compensated triple (goal + F * penalty), plus the seed's own penalty and kappa content.
"""
from __future__ import annotations
import argparse, json
import numpy as np

W_JERK, W_KAPPA = 0.02, 0.05


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True); ap.add_argument("--factor", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    J = json.load(open(a.panel, encoding="utf-8")); BN, rows = J["box_names"], J["rows"]
    i_cv = BN.index("cv"); i_seed = BN.index("seed0"); i_prop = BN.index("proposal")
    hold = np.array([r["goalresp_rel"] < 1e-6 for r in rows]); eid = np.array([r["ep"] for r in rows])
    seed_k = np.array([abs(r["box_kmean"][i_seed]) > 1e-9 for r in rows])
    seed_a = np.array([abs(r["box_a0"][i_seed]) > 1e-9 for r in rows])
    out = {"tool": "seed_channel.py", "source": a.panel, "n_windows": len(rows), "n_episodes": int(len(np.unique(eid))),
           "factor_compensated": a.factor, "seed_nonzero_kappa_windows": int(seed_k.sum()),
           "seed_nonzero_accel_windows": int(seed_a.sum()), "hold_windows": int(hold.sum()), "metrics": {}}
    for m in ("cos", "chord", "ccos"):
        C = np.array([r[f"cost_{m}"] for r in rows]); G = np.array([r[f"goal_{m}"] for r in rows])
        JR = np.array([r["jerk_raw"] for r in rows]); KR = np.array([r["kap_raw"] for r in rows])
        pen = W_JERK * JR + W_KAPPA * KR
        ship_seed = C[:, i_seed] < C[:, i_cv]; ship_prop = C[:, i_prop] < C[:, i_cv]
        comp = G + a.factor * pen
        comp_seed = comp[:, i_seed] < comp[:, i_cv]; comp_prop = comp[:, i_prop] < comp[:, i_cv]
        # consistency: the closure's total must equal goal + shipped penalty (up to float32)
        resid = float(np.abs(C - (G + pen)).max())
        blk = {"closure_total_minus_goal_plus_penalty_max_abs": resid,
               "seed_penalty": {"median": float(np.median(pen[:, i_seed])), "max": float(pen[:, i_seed].max())},
               "shipped": {"seed_beats_cv": int(ship_seed.sum()), "seed_beats_cv_nonHOLD": int(ship_seed[~hold].sum()),
                           "seed_beats_cv_HOLD": int(ship_seed[hold].sum()),
                           "seed_beats_cv_on_kappa_seed_windows": int((ship_seed & seed_k).sum()),
                           "proposal_beats_cv": int(ship_prop.sum())},
               "compensated": {"seed_beats_cv": int(comp_seed.sum()), "seed_beats_cv_nonHOLD": int(comp_seed[~hold].sum()),
                               "seed_beats_cv_on_kappa_seed_windows": int((comp_seed & seed_k).sum()),
                               "proposal_beats_cv": int(comp_prop.sum())},
               "goal_seed_minus_goal_cv_median_nonHOLD": float(np.median((G[:, i_seed] - G[:, i_cv])[~hold]))}
        out["metrics"][m] = blk
        print(f"{m:6s} resid {resid:.2e} | shipped: seed beats cv {blk['shipped']['seed_beats_cv']}/282 (nonHOLD {blk['shipped']['seed_beats_cv_nonHOLD']}/149, on kappa-seed windows {blk['shipped']['seed_beats_cv_on_kappa_seed_windows']}/{int(seed_k.sum())}), proposal {blk['shipped']['proposal_beats_cv']} | compensated x{a.factor:.0f}: seed {blk['compensated']['seed_beats_cv']}/282 (kappa-seed {blk['compensated']['seed_beats_cv_on_kappa_seed_windows']}), proposal {blk['compensated']['proposal_beats_cv']} | seed pen median {blk['seed_penalty']['median']:.3g}")
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

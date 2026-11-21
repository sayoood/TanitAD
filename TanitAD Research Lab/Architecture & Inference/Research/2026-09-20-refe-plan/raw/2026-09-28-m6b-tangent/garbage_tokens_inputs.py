#!/usr/bin/env python3
"""The 20 'garbage' tokens (whole fan beyond pi at step 19): does REFe RECEIVE the ego's measured speed, does it IGNORE
it, and is there anything slow in its fan? MEASURED from the eval cache rows (the planner's OWN ego vector and goal, as
`planner._ego_vec` / `_goal_for` built them) and the banked dumps (all 64 proposals). No GPU, no training, no scoring.

Code path (file:line, read 2026-09-28): planner.py:280-297 `_ego_vec` -> 7-D [vx, vy, ax, ay, yaw_rate, steer, speed]
from the nuPlan ego state at t0; planner.py:306-310 -> `self.model(img, ego_vec, goal, calib=calib)`; model.py:185
`ego_dim = 7`; model.py:512-514 `ego_enc = Linear(ego_dim + goal_dim) -> GELU -> Linear`; model.py:738
`ego_tok = ego_enc(cat([ego, goal_feat]))` -- one token added to all 64 queries. The proxy's eval cache decodes the same
way (eval/proxy_eval_cache.py:89-92, :138 `planner._ego_vec(ego)`).

    python raw/2026-09-28-m6b-tangent/garbage_tokens_inputs.py   -> garbage_tokens_inputs.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ECD = Path("D:/Projects/TanitAD/data/refe_proxy/cache_eval_ep015")
DT = 0.2                                                   # native step, 5 Hz


def main() -> int:
    rows = {}
    for l in open(ECD / "rows.jsonl", encoding="utf-8"):
        r = json.loads(l)
        rows[r["key"][1]] = r
    out = {"code_path": __doc__.split("Code path")[1].split("    python")[0].strip()}
    for arm in ("base", "Wt0", "T0"):
        z = np.load(HERE / "dumps" / f"{arm}.npz")
        T = z["traj"].astype(np.float64)
        H = z["human"].astype(np.float64)
        W = z["winner"]
        tok = np.array([str(t) for t in z["token"]])
        bad = (np.abs(np.load(HERE / "dumps" / "T0.npz")["traj"][..., 19, 2]) > np.pi).all(1)
        v0 = np.array([rows[t]["ego"][6] for t in tok])
        vx = np.array([rows[t]["ego"][0] for t in tok])
        goal = np.array([rows[t]["goal"] for t in tok], float)
        g2, g4 = np.linalg.norm(goal[:, 0:2], axis=1), np.linalg.norm(goal[:, 2:4], axis=1)
        v_init = np.linalg.norm(T[:, :, 0, :2], axis=-1) / DT              # step 0 is t = 0.2 s from (0, 0)
        d4 = np.linalg.norm(T[:, :, 19, :2], axis=-1)
        gt4 = np.linalg.norm(H[:, 7, :2], axis=-1)
        ar = np.arange(len(W))

        def blk(m):
            return {"n": int(m.sum()), "ego_speed_v0_mps_mean": float(v0[m].mean()), "ego_speed_v0_mps_median": float(np.median(v0[m])),
                    "ego_vx_mean": float(vx[m].mean()),
                    "winner_implied_initial_speed_mps_mean": float(v_init[ar, W][m].mean()),
                    "fan_implied_initial_speed_mps_median": float(np.median(v_init[m])),
                    "fan_min_implied_initial_speed_mps_mean": float(v_init[m].min(1).mean()),
                    "gt_dist_4s_m_mean": float(gt4[m].mean()),
                    "goal_point_dist_m_mean_[p1,p2]": [float(g2[m].mean()), float(g4[m].mean())],
                    "fan_dist_4s_m_[min,median,max]_mean": [float(d4[m].min(1).mean()), float(np.median(d4[m], 1).mean()),
                                                           float(d4[m].max(1).mean())],
                    "slots_within_+-30pct_of_gt_dist_mean": float((np.abs(d4[m] - gt4[m][:, None])
                                                                  <= 0.3 * np.maximum(gt4[m][:, None], 1.0)).sum(1).mean()),
                    "tokens_with_ANY_slot_within_+-30pct_of_gt_dist": int((np.abs(d4[m] - gt4[m][:, None])
                                                                          <= 0.3 * np.maximum(gt4[m][:, None], 1.0)).any(1).sum()),
                    "tokens_with_ANY_slot_shorter_than_gt": int((d4[m] < gt4[m][:, None]).any(1).sum())}
        out[arm] = {"garbage20": blk(bad), "rest": blk(~bad)}
        if arm == "T0":
            out["T0_per_token"] = [{"token": str(tok[i]), "v0": round(float(v0[i]), 3), "goal_dist_m": [round(float(g2[i]), 1), round(float(g4[i]), 1)],
                                    "gt_dist_4s_m": round(float(gt4[i]), 1),
                                    "fan_dist_4s_m_[min,max]": [round(float(d4[i].min()), 1), round(float(d4[i].max()), 1)],
                                    "winner_init_speed_mps": round(float(v_init[i, W[i]]), 2)}
                                   for i in np.nonzero(bad)[0]]
    # ---- the goal input: does its range separate the 20? (planner.py:449-474 `_goal_for`: route_goal_positions on the
    #      route polyline, cached per planner instance) -- and what range did TRAINING see (the proxy train cache rows)?
    z = np.load(HERE / "dumps" / "T0.npz")
    tok = np.array([str(t) for t in z["token"]])
    bad = (np.abs(z["traj"][..., 19, 2]) > np.pi).all(1)
    d2 = np.array([np.linalg.norm(np.array(rows[t]["goal"][2:4], float)) for t in tok])
    sep = {f"goal_p2_gt_{thr}m": {"tokens": int((d2 > thr).sum()), "of_them_garbage": int(((d2 > thr) & bad).sum()),
                                  "garbage_outside": int(((d2 <= thr) & bad).sum())} for thr in (100, 150, 200, 300)}
    tg = []
    for l in open(Path("D:/Projects/TanitAD/data/refe_proxy/cache_train_ep015") / "rows.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if "goal" in r:
            tg.append(float(np.linalg.norm(np.array(r["goal"][2:4], float))))
    tg = np.array(tg)
    out["goal_range"] = {"navtest_separation": sep, "navtest_rest_goal_p2_max_m": float(d2[~bad].max()),
                         "navtest_garbage_goal_p2_min_m": float(d2[bad].min()),
                         "train_cache_goal_p2": {"n": int(tg.size), "max_m": float(tg.max()),
                                                 "p99.9_m": float(np.percentile(tg, 99.9)), "n_gt_200m": int((tg > 200).sum()),
                                                 "n_gt_300m": int((tg > 300).sum())}}
    json.dump(out, open(HERE / "garbage_tokens_inputs.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(out["goal_range"], indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("T0_per_token", "code_path")}, indent=1))
    print(json.dumps(out["T0_per_token"][:20]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

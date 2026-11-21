#!/usr/bin/env python3
"""What do the ~20 'garbage-heading' tokens (PRIMARY (b)'s residual: the WHOLE fan beyond pi at step 19, two logs) cost
in PDMS, which NAVSIM term fails there, is A7's heading[18] also wrong there, and what distinguishes them from the ~586
tokens the plain / tangent losses DID fix? MEASURED on banked dumps and already-scored harness CSVs; no GPU, no training.

    python raw/2026-09-28-m6b-tangent/garbage_tokens_pdms.py   -> garbage_tokens_pdms.json
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "eval"))
import proxy_eval as PE  # noqa: E402

EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
M6, E2 = PKG / "raw" / "2026-09-27-m6-proxy", PKG / "raw" / "2026-09-27-m6-proxy" / "epoch2"
SUB = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
       "comfort", "driving_direction_compliance"]


def seams():
    out = {}
    for root, tag in ((M6, "M6e1"), (E2, "M6e2"), (HERE, "M6b")):
        for d in sorted((root / "score").glob("refe_m6_*")) if (root / "score").exists() else []:
            c = d / f"{d.name}.csv"
            if c.exists():
                out[f"{tag}:{d.name[8:]}"] = c
    return out


def main() -> int:
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    zT = np.load(HERE / "dumps" / "T0.npz")
    zb = np.load(HERE / "dumps" / "base.npz")
    tok = np.array([str(t) for t in zT["token"]])
    bad = set(tok[(np.abs(zT["traj"][..., 19, 2]) > np.pi).all(1)].tolist())
    base_wf = set(tok[(np.abs(zb["traj"][..., 19, 2]) > np.pi).all(1)].tolist())
    fixed = base_wf - bad
    res = {"n_tokens": len(tok), "garbage_tokens_T0": len(bad), "base_whole_fan_tokens": len(base_wf),
           "fixed_by_the_losses": len(fixed), "garbage_subset_of_base": bad <= base_wf}
    # ---- PDMS contribution and sub-scores, per scored seam
    pd = {}
    for name, c in seams().items():
        rows = PE.read_csv(c)
        if len(rows) != len(tok):
            pd[name] = {"skipped": f"{len(rows)} rows"}
            continue
        sc = {t: float(r["score"]) for t, r in rows.items()}
        g = [sc[t] for t in bad]
        rest = [sc[t] for t in tok if t not in bad]
        pd[name] = {"pdms_all_x100": 100 * float(np.mean(list(sc.values()))), "pdms_garbage20_x100": 100 * float(np.mean(g)),
                    "pdms_rest_x100": 100 * float(np.mean(rest)),
                    "whole_set_if_garbage_scored_like_rest_x100": 100 * float(np.mean(rest)),
                    "contribution_to_whole_set_x100": 100 * (float(np.mean(list(sc.values()))) - float(np.mean(rest))),
                    "subscores_garbage20_x100": {s: 100 * float(np.mean([float(rows[t][s]) for t in bad])) for s in SUB},
                    "subscores_rest_x100": {s: 100 * float(np.mean([float(rows[t][s]) for t in tok if t not in bad]))
                                            for s in SUB}}
    res["pdms_by_seam"] = pd
    # ---- A7's heading[18] on the garbage tokens (the repair copies heading[18] into heading[19])
    a7 = {}
    for n in ("base", "Wt0", "T0", "T1", "T2"):
        z = np.load(HERE / "dumps" / f"{n}.npz")
        T, H, W = z["traj"].astype(np.float64), z["human"].astype(np.float64), z["winner"]
        ar = np.arange(len(W))
        e18 = np.abs(PE.wrap(T[ar, W, 18, 2] - H[:, 7, 2]))
        th, m = PE.own_tangent19(T)
        g = np.isin(tok, sorted(bad))
        a7[n] = {"winner_h18_vs_gt4s_median_rad_garbage": float(np.median(e18[g])),
                 "winner_h18_vs_gt4s_median_rad_rest": float(np.median(e18[~g])),
                 "winner_h18_vs_gt4s_median_rad_fixed586": float(np.median(e18[np.isin(tok, sorted(fixed))]))}
    res["a7_heading18"] = a7
    # ---- what distinguishes the garbage tokens from the fixed ones (inputs the planner saw; GT motion)
    def feats(ts):
        v = np.array([exp[t]["ego_statuses"][-1]["ego_velocity"][0] for t in ts], float)
        a = np.array([exp[t]["ego_statuses"][-1]["ego_acceleration"][0] for t in ts], float)
        cmd = np.array([np.argmax(exp[t]["ego_statuses"][-1]["driving_command"]) for t in ts])
        i = np.isin(tok, ts)
        gt4 = np.linalg.norm(zT["human"][i, 7, :2], axis=-1)
        fan = np.linalg.norm(zT["traj"][i, :, 19, :2], axis=-1).mean(1)
        return {"n": len(ts), "ego_speed_mps_mean": float(v.mean()), "ego_speed_mps_median": float(np.median(v)),
                "share_speed_below_1mps": float(np.mean(v < 1.0)), "ego_accel_mean": float(a.mean()),
                "cmd_counts_[left,straight,right,unknown]": np.bincount(cmd, minlength=4).tolist(),
                "gt_dist_4s_m_mean": float(gt4.mean()), "fan_mean_dist_4s_m_mean": float(fan.mean()),
                "logs": len({exp[t]["log_name"] for t in ts}),
                "maps": {m_: sum(exp[t]["map_name"] == m_ for t in ts) for m_ in sorted({exp[t]["map_name"] for t in ts})}}
    res["features"] = {"garbage": feats(sorted(bad)), "fixed_by_losses": feats(sorted(fixed)),
                       "all": feats(list(tok))}
    slow = [t for t in tok if exp[t]["ego_statuses"][-1]["ego_velocity"][0] < 1.0]
    res["whole_fan_rate_T0_among_speed_below_1mps"] = {"n": len(slow), "garbage": len(set(slow) & bad)}
    T0 = zT["traj"].astype(np.float64)
    g = np.isin(tok, sorted(bad))
    step = np.linalg.norm(T0[..., 19, :2] - T0[..., 18, :2], axis=-1)
    res["tangent_mask_on_garbage_slots"] = {"slots": int(g.sum() * 64),
                                            "step19_below_0.2m": int((step[g] < 0.2).sum()),
                                            "median_step19_m": float(np.median(step[g]))}
    json.dump(res, open(HERE / "garbage_tokens_pdms.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

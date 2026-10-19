"""pick_episodes_r8.py -- choose the three showcase episodes BY RULE from the banked EMA-40k metrics (CPU, no GPU).

Rules (fixed here, stated in the video manifest; none of them looks at how good the plan looks):
  R1 cruise false-braking : among episodes with >= 8 GT-cruising S6 windows (battery dvclass 'cruising'), the one with the MOST flagged windows
                            (flag = GT 'cruising' AND refcv8 plan class 'decel', == `false_decel_on_dvcruise` of cruise_brake_retime_r8.py);
                            ties -> higher flagged FRACTION, then lower episode index.
  R2 sharp turn           : the episode with the largest |heading change| over any 3 s (30 rows) of the logged ego path (yaw column, unwrapped), the
                            whole 3 s span lying INSIDE the plan-window coverage [ws[0], ws[-1]] (AMENDED 2026-10-19 after the first pick, ep 31, turned out to
                            have its turn in rows 2-32, before its first plan window at ws 10 -- a showcase with no plan on it; no plan-quality number entered
                            the amendment, only coverage).
  R3 stop / slow-down     : among episodes whose logged speed reaches < 1 m/s after having been >= 5 m/s, the one with the largest drop
                            (max_t [v_t - min_{t..t+6 s} v]); ties -> lower episode index.
All three must have the map GT, the extrinsics, and an agent-join line for most frames; the three picks are distinct (a later rule skips an index already taken).
    python pick_episodes_r8.py [--tree step40000ema_unc] --out <raw>/PICKS_step40000ema.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r8v_data as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="step40000ema_unc")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = D.EvalBank(a.tree)
    rows = {}
    for ci in B.indices():
        e = B.episode(ci)
        ws, poses = e["ws"], e["poses"]
        # --- R1: false brake on S6 windows
        n_cru = n_flag = n_valid = 0
        for w6, v06, v6, ok in zip(e["s6_ws"], e["s6_v0"], e["s6_gt_v6"], e["s6_gt_valid"]):
            i = int(np.where(ws == w6)[0][0])
            fl, gc, pc, vend = D.false_brake(e["plan8"][i], v06, v6, ok)
            if fl is None:
                continue
            n_valid += 1
            n_cru += int(gc == "cruising")
            n_flag += int(bool(fl))
        # --- R2: sharp turn
        yaw = np.unwrap(poses[:, 2])
        lo, hi = int(ws[0]), int(ws[-1])                                  # plan-window coverage in rows
        dy = np.abs(yaw[30:] - yaw[:-30])                                 # dy[k] spans rows k .. k+30
        okk = np.arange(len(dy))
        dyaw = dy[(okk >= lo) & (okk + 30 <= hi)].max()
        # --- R3: stop / slow-down
        v = poses[:, 3]
        best_drop, reach_stop = 0.0, False
        for t in range(int(ws[0]), len(v)):
            if v[t] >= 5.0:
                seg = v[t:t + 61]
                if seg.min() < 1.0:
                    reach_stop = True
                    best_drop = max(best_drop, float(v[t] - seg.min()))
        rows[ci] = {"n_s6_valid": n_valid, "n_gt_cruising": n_cru, "n_flagged": n_flag, "flag_frac": (n_flag / n_cru if n_cru else None),
                    "max_abs_dyaw_3s_rad": float(dyaw), "reach_stop": reach_stop, "stop_drop_mps": best_drop,
                    "has_map_gt": B.map_gt_path(ci) is not None,
                    "agent_frames": (len(B.agents.get(ci, {})) if B.agents is not None else None), "rows": int(len(poses))}
    ok = [ci for ci, r in rows.items() if r["has_map_gt"] and (r["agent_frames"] or 0) >= 100]
    taken = []
    c1 = [ci for ci in ok if rows[ci]["n_gt_cruising"] >= 8]
    c1.sort(key=lambda c: (-rows[c]["n_flagged"], -(rows[c]["flag_frac"] or 0), c))
    taken.append(c1[0])
    c2 = sorted([c for c in ok if c not in taken], key=lambda c: (-rows[c]["max_abs_dyaw_3s_rad"], c))
    taken.append(c2[0])
    c3 = sorted([c for c in ok if c not in taken and rows[c]["reach_stop"]], key=lambda c: (-rows[c]["stop_drop_mps"], c))
    taken.append(c3[0])
    picks = {"cruise_false_brake": taken[0], "sharp_turn": taken[1], "stop_slowdown": taken[2]}
    out = {"tree": a.tree, "ckpt_step": B.ckpt_step, "rules": __doc__, "picks": picks,
           "pick_stats": {k: rows[v] for k, v in picks.items()},
           "runner_ups": {"cruise_false_brake": c1[1:4], "sharp_turn": c2[1:4], "stop_slowdown": c3[1:4]},
           "all_episodes": rows}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"picks": picks, "stats": out["pick_stats"], "runner_ups": out["runner_ups"]}, indent=1))


if __name__ == "__main__":
    main()

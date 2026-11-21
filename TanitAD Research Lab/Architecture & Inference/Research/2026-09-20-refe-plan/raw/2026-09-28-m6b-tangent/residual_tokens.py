#!/usr/bin/env python3
"""Where does PRIMARY (b) -- raw step-19 |heading| > pi -- still live after the plain / plain_tangent losses?

MEASURED on the banked dumps (no GPU, no training): per arm, the tokens whose WHOLE fan (all 64 slots) is beyond pi at
step 19, their overlap across arms, and per token the log, the ego status the planner saw (velocity, acceleration,
driving command), the GT heading at 4 s, and the winner's heading profile. Writes residual_tokens.json.

    python raw/2026-09-28-m6b-tangent/residual_tokens.py
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
M6 = PKG / "raw" / "2026-09-27-m6-proxy"
DUMPS = {**{n: HERE / "dumps" / f"{n}.npz" for n in ("base", "Wt0", "Wt1", "Wt2", "T0", "T1", "T2")},
         "P0_e1": M6 / "dumps" / "P0.npz", "P1_e1": M6 / "dumps" / "P1.npz",
         "P0_e2": M6 / "epoch2" / "dumps" / "P0e2.npz", "P1_e2": M6 / "epoch2" / "dumps" / "P1e2.npz"}


def main() -> int:
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    per_arm, whole = {}, {}
    for n, p in DUMPS.items():
        if not p.exists():
            per_arm[n] = {"missing": str(p)}
            continue
        z = np.load(p)
        T = z["traj"].astype(np.float64)
        tok = np.array([str(t) for t in z["token"]])
        beyond = np.abs(T[..., 19, 2]) > np.pi
        wf = beyond.all(1)
        whole[n] = set(tok[wf].tolist())
        per_arm[n] = {"pct_raw_h19_beyond_pi": 100.0 * float(beyond.mean()), "whole_fan_tokens": int(wf.sum()),
                      "beyond_pi_slots_in_whole_fan_tokens": int(beyond[wf].sum()),
                      "beyond_pi_slots_elsewhere": int(beyond[~wf].sum()),
                      "beyond_pi_steps_other_than_19": int((np.abs(T[:, :, :19, 2]) > np.pi).sum())}
    ref = "T0"
    for n in whole:
        per_arm[n]["overlap_with_T0"] = len(whole[n] & whole[ref])
    z = np.load(DUMPS[ref])
    T, H, W = z["traj"].astype(np.float64), z["human"].astype(np.float64), z["winner"]
    tok = np.array([str(t) for t in z["token"]])
    rows = []
    for i in np.nonzero(np.isin(tok, sorted(whole[ref])))[0]:
        es = exp[tok[i]]["ego_statuses"][-1]
        hs = T[i, W[i], :, 2]
        rows.append({"token": tok[i], "log": exp[tok[i]]["log_name"], "map": exp[tok[i]]["map_name"],
                     "ego_velocity": es["ego_velocity"], "ego_acceleration": es["ego_acceleration"],
                     "driving_command": es["driving_command"], "gt_heading_4s": float(H[i, 7, 2]),
                     "winner_heading_steps_0_10_18_19": [round(float(hs[j]), 4) for j in (0, 10, 18, 19)],
                     "winner_h19_wrapped_err_rad": float(abs(PE.wrap(hs[19] - H[i, 7, 2]))),
                     "winner_xy19": [round(float(T[i, W[i], 19, 0]), 2), round(float(T[i, W[i], 19, 1]), 2)]})
    allv = np.array([exp[t]["ego_statuses"][-1]["ego_velocity"][0] for t in tok], float)
    badv = np.array([r["ego_velocity"][0] for r in rows], float)
    out = {"what": "the tokens whose WHOLE fan is beyond pi at step 19 (PRIMARY (b)'s residual), per arm; MEASURED on "
                   "the banked dumps", "per_arm": per_arm, "T0_whole_fan_tokens": rows,
           "logs": {lg: sum(r["log"] == lg for r in rows) for lg in sorted({r["log"] for r in rows})},
           "ego_speed_mps": {"whole_fan_tokens_mean": float(badv.mean()) if badv.size else None,
                             "all_tokens_mean": float(allv.mean()), "all_tokens_p10": float(np.percentile(allv, 10))}}
    json.dump(out, open(HERE / "residual_tokens.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "T0_whole_fan_tokens"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

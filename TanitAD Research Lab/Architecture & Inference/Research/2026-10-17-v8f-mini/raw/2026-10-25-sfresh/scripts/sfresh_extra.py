#!/usr/bin/env python3
"""PREREG_V8F_SFRESH extra self-consistency checks (descriptive; they decide nothing in the outcome table).

(1) IDENTITY: a window whose REALISED ages are (a_t, a_s) = (0, 0) under the cycle schedule feeds the model exactly what --ages fresh feeds it,
    so its output must be bit-identical in the two runs (same CPU, same batch boundaries). A miss would mean the 'fresh' arm changed something other than the ages.
(2) AGE PROFILE: the banked cycle S error at 50 m by REALISED S age (0..6 cache steps = 0..1.2 s), against the same windows scored fresh, and the
    metres the ego travelled between S's tick and NOW (measured poses).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(OUT, "scripts"))
import sfresh_analyze as A  # noqa: E402

E, C = A.E, A.C


def main():
    grid, J, X = A.load_ctx()
    res = {}
    for s in (0, 1):
        ds, wins = A.ds_for(s)
        N = len(wins)
        rat, ras, trav = np.zeros(N, int), np.zeros(N, int), np.zeros(N)
        for i, (ci, t, _, _) in enumerate(wins):
            a_t = i % 3
            a_s = a_t + (i // 3) % 5
            it = ds.item(ci, t, a_t, a_s)
            rat[i], ras[i] = it["a_t"], it["a_s"]
            trav[i] = float(np.linalg.norm(np.asarray(it["pose_now"])[:2] - np.asarray(it["pose_s"])[:2]))
        cyc = np.load(os.path.join(A.EVD, f"eval_M2b_s{s}-sfresh-cycle.npz"))      # the CPU re-run of the banked cycle eval
        bank = A.banked_cycle_npz(s)
        fr = np.load(os.path.join(A.EVD, f"eval_M2b_s{s}-sfresh-fresh.npz"))
        z = (rat == 0) & (ras == 0)
        ident = {"n_age0_windows": int(z.sum()),
                 "max_abs_diff_traj30_fresh_vs_cycleRerun": float(np.abs(fr["pred"][z] - cyc["pred"][z]).max()),
                 "max_abs_diff_ckpt_fresh_vs_cycleRerun": float(np.abs(fr["ckpt"][z] - cyc["ckpt"][z]).max()),
                 "max_abs_diff_goal_prob": float(np.abs(fr["goal_prob"][z] - cyc["goal_prob"][z]).max()),
                 "max_abs_diff_traj30_fresh_vs_BANKED_cycle": float(np.abs(fr["pred"][z] - bank["pred"][z]).max())}
        # non-age-0 windows must differ (the lever does something)
        nz = ~z
        ident["mean_abs_diff_traj30_on_aged_windows"] = float(np.abs(fr["pred"][nz] - cyc["pred"][nz]).mean())
        prof = {}
        ok50 = X["rc_ok"][:, 1]
        for a in range(0, 7):
            m = (ras == a) & ok50
            if m.sum() == 0:
                continue
            prof[str(a)] = {"age_s": round(a * 0.2, 1), "n_windows": int(m.sum()),
                            "cycle_banked_S_pos_err_50m": float(bank["ck_pos_err"][m, 1].mean()),
                            "fresh_S_pos_err_50m": float(fr["ck_pos_err"][m, 1].mean()),
                            "fresh_minus_cycle": float((fr["ck_pos_err"][m, 1] - bank["ck_pos_err"][m, 1]).mean()),
                            "ego_travel_S_tick_to_now_m": float(trav[m].mean())}
        ok = ok50 & (ras > 0)
        r_aged = E.paired(fr["ck_pos_err"][ok, 1], bank["ck_pos_err"][ok, 1], X["eid"][ok], A.NB)
        ok0 = ok50 & (ras == 0)
        res[f"s{s}"] = {"identity_on_age0_windows": ident, "age_profile_50m": prof,
                        "Q_STALE_50m_aged_windows_only": A.pr(r_aged), "n_age0_windows_with_rc50": int(ok0.sum()),
                        "Q_STALE_50m_age0_windows_only_mean": float((fr["ck_pos_err"][ok0, 1] - bank["ck_pos_err"][ok0, 1]).mean()) if ok0.any() else None,
                        "mean_realised_S_age_steps": float(ras.mean()), "mean_ego_travel_S_tick_to_now_m": float(trav.mean()),
                        "mean_realised_T_age_steps": float(rat.mean())}
    A.jd(res, os.path.join(OUT, "sfresh_extra.json"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""diag_gate.py -- POST-HOC DIAGNOSTIC, run AFTER the HGATE read (READ.json) was fixed. NOT a PREREG row, carries NO bar, changes NO outcome.

Why it exists. HGATE came back CONSISTENCY NOT RESTORED (PRIMARY-C fails at both seeds) while PRIMARY-K passes. The committed next step is the PREREG's
('no further post-processing arm; the lever moves to T'). This script only ATTRIBUTES the residual consistency cost, so that the T-side work is aimed at
the right term. Nothing here may be quoted as a result, none of it can move a threshold, a trigger, a weight or a bar.

   tau_G everywhere = the blend with w^G and no gate (NOT HBLEND's H, which uses the unrestricted w).
   leave-one-trigger-out / single-trigger arms : which of T1..T4 carries the trade-off (T's own predictions, vision-only at inference; T1-class).
   ORACLE arms (T0: they READ a GT label at inference; a ceiling on what a perfect detector could buy, NEVER a result):
      O1  close where the GT STOP / TURN / LEAD rule is active (replaces T1-T3), T4 as in the arm
      O2  O1, and T4 replaced by the GT SPEED band: close where tau_G's speed exceeds the GT v_hi_ms in the band
      O3  close wherever tau_G (= H, the blend) VIOLATES any GT rule -- the ceiling of ANY per-window gate that can only choose between tau_B and tau_G
Same estimator (paired episode-cluster bootstrap, B = 2000), same eval-141 windows and same eval `evaluate()` as the PREREG rows.
"""
from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

import v8f_hgate as G  # noqa: E402

H = G.H
CKs = ("viol_any", "viol_SPEED", "viol_STOP", "viol_TURN", "viol_LEAD")


def main(nb: int = 2000):
    from taniteval import v8f_beta_eval as BE
    res = {"label": "POST-HOC DIAGNOSTIC (not a PREREG row, no bar, no outcome change)", "estimator": f"paired episode-cluster bootstrap, B={nb}, 141 episodes, seed 0",
           "tier": "T1-class for the trigger arms; ORACLE arms are T0 (read GT at inference) and never a result", "seeds": {}}
    for seed in H.SEEDS:
        W = G.load_world(seed)
        ev, D, X, z, tau, gt, eid, ex = (W[k] for k in ("ev", "D", "X", "z", "tau", "gt", "eid", "ex"))
        pwB = H.arm_pw(tau, gt, BE)
        vB = {k: z[k].astype(float) for k in CKs}
        ga = {r: np.asarray(v["active"], bool) for r, v in ev.C.violations(tau, ev.gt_constraints(X)).items() if r in ev.C.RULE_NAMES}
        gt_any = ga["STOP"] | ga["TURN"] | ga["LEAD"]
        vhi_gt = np.where(np.asarray(X["has"], bool), np.asarray(X["cons"][:, D.CONS_IDX["v_hi_ms"]], np.float64), np.nan)
        blend_plan, info = G.gate_of(W, W["wG"], force="open")
        blend = info["blend"]
        v_blend = W["kin_fn"](blend)["v"]
        speed_over_gt_vhi = (v_blend[:, W["band_knot"]:] > vhi_gt[:, None]).any(1)           # NaN v_hi -> False
        _, ev_blend = ev.evaluate(blend, X, ex, 10, with_families=False)
        h_violates_any = np.asarray(ev_blend["viol_any"], bool)
        trig = info["triggers"]
        t123 = trig["T1_follow"] | trig["T2_stop"] | trig["T3_turn"]
        arm_closed = {
            "tau_G everywhere (w^G, NO gate: all triggers off)": np.zeros(len(tau), bool),
            "HG (the PREREG arm: T1+T2+T3+T4)": t123 | trig["T4_vhi"],
            "T1+T2+T3 only (T4 off)": t123, "T4 only": trig["T4_vhi"],
            "leave out T1 (FOLLOW)": trig["T2_stop"] | trig["T3_turn"] | trig["T4_vhi"],
            "leave out T2 (STOP)": trig["T1_follow"] | trig["T3_turn"] | trig["T4_vhi"],
            "leave out T3 (TURN)": trig["T1_follow"] | trig["T2_stop"] | trig["T4_vhi"],
            "leave out T4 (v_hi)": t123,
            "ORACLE O1: GT STOP/TURN/LEAD active + T4": gt_any | trig["T4_vhi"],
            "ORACLE O2: GT STOP/TURN/LEAD active + GT-v_hi speed band": gt_any | speed_over_gt_vhi,
            "ORACLE O3: close wherever tau_G violates any GT rule": h_violates_any,
        }
        block = {}
        for name, closed in arm_closed.items():
            plan = np.where(closed[:, None, None], tau, blend)
            _, pwev = ev.evaluate(plan, X, ex, 10, with_families=False)
            pw = H.arm_pw(plan, gt, BE)
            row = {"open_share": float((~closed).mean()), "n_open": int((~closed).sum())}
            for k in ("ade2", "ade6", "fw_lon_accel_mae"):
                row[k] = BE.paired_row(pw[k], pwB[k], eid, nb)
            for k in CKs:
                row[k] = BE.paired_row(np.asarray(pwev[k], float), vB[k], eid, nb)
            row["C_upper_bound_le_0.02"] = bool(row["viol_any"]["hi"] <= 0.02)
            row["K_both_negative_and_separated"] = bool(row["ade2"]["mean"] < 0 and row["ade2"]["separated"] and row["ade6"]["mean"] < 0 and row["ade6"]["separated"])
            block[name] = row
        res["seeds"][seed] = {"arms": block,
                              "gt_v_hi_finite_windows": int(np.isfinite(vhi_gt).sum()),
                              "tau_G_violates_any_rule_n": int(h_violates_any.sum()),
                              "n_windows": int(len(eid))}
        print(seed, {n: (round(r["open_share"], 3), round(r["viol_any"]["mean"], 4), round(r["viol_any"]["hi"], 4), round(r["ade6"]["mean"], 4), r["ade6"]["separated"]) for n, r in block.items()}, flush=True)
    H.jdump(res, f"{G.OUT}/diag_gate.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

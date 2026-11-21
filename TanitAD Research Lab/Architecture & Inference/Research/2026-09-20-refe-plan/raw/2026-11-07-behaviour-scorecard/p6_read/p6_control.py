#!/usr/bin/env python3
"""p6_control.py -- the u400 CONTROL for the P6 read: the SAME code path (selc6_analyze) run on u400's banked C3 table and ego probe. CPU, banked data only,
reads nothing of P6. It must reproduce u400_reference (pinned content) leaf by leaf, pass the data gates, and every deliberate-regression arm must FAIL.
    python p6_control.py --out CONTROL_RESULT.json   (writes CONTROL_RESULT.json + CONTROL_RESULT.md next to it)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selc6_analyze as A  # noqa: E402

BANKED_SEAM = "E:/Projects/TanitAD/data/refe_navtest/seams/stage2_sft4_B_u400_navtest_full.npz"
BANKED_CSV = ("E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_sft4_B_u400_navtest_full/refe_stage2_sft4_B_u400_navtest_full.csv")


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(HERE, "CONTROL_RESULT.json")
    res = {"what": "u400 as the control for the P6 read: same code path, banked data, no P6 value read", "gates": {}, "RED_arms": {}}
    g = res["gates"]
    g["G6-PIN"] = A.g6_pin(code_pin_path=os.path.join(HERE, "CODE_PIN.json") if os.path.exists(os.path.join(HERE, "CODE_PIN.json")) else None)
    g["G6-REPRO"], (Du, B, S, r3) = A.g6_repro(HERE + "/u400_reference.pinned.json")
    g["G6-0"] = A.g6_0(Du)
    g["G6-2"] = A.g6_2(Du)
    g["G6-4"] = A.g6_4(Du)
    g["G6-FP32"] = A.g6_fp32(Du)
    g["G6-1"] = A.g6_1(Du, BANKED_SEAM) if os.path.exists(BANKED_SEAM) else {"PASS": False, "why": "banked seam not found"}
    Dc = dict(Du)
    Dc["red_pick_pdms"] = Du["pdms"][np.arange(len(Du["pick"])), Du["pick"]] + 0.01      # STAND-IN for the harness-scored +2 m shift (no harness on the control)
    g["G6-3"] = A.g6_3(Dc, BANKED_CSV) if os.path.exists(BANKED_CSV) else {"PASS": False, "why": "banked csv not found"}
    g["G6-3"]["note"] = "RED arm here is a +0.01 perturbation stand-in; the real arm is a harness call (selc6_score.py `red` column)"
    # the on-disk u400_reference.json (sha differs from the card pin): tie/R1/R3 sections compared, legacy reported
    disk = json.load(open(A.BEH + "/u400_reference.json"))
    pinned = json.load(open(HERE + "/u400_reference.pinned.json"))
    res["on_disk_vs_pinned"] = {k: A.max_abs_diff(disk[k], pinned[k]) for k in ("R1_weighted_cruise_slow_share", "R2_spearman_tie", "R3_u400_without_lane_factor", "R2_spearman_legacy")}
    # headline numbers asked for in the brief
    t = S["R2_spearman_tie"]
    res["headline"] = {"TTC_pred_paired": t["TTC"]["pred"], "NC_pred_paired": t["NC"]["pred"], "weighted_cruise_slow_share": S["R1_weighted_cruise_slow_share"]["point"],
                       "u400_without_lane": r3["weighted_slow_share"]["point"], "TTC_pred_all154": t["TTC"]["pred_unpaired_mean"], "NC_pred_all154": t["NC"]["pred_unpaired_mean"]}
    # the verdict machinery on u400 treated as P6: must be REFUTED (pessimism persists, share 0.412 > 0.20)
    an = A.analyze_p6(Du, Du, B)
    res["u400_as_P6_verdict"] = an["decision"]
    # RED arms (each MUST fail)
    RA = res["RED_arms"]
    RA["argsort_of_argsort_ranks (G6-REPRO)"] = {"fails_as_required": not (g["G6-REPRO"]["RED_argsort_of_argsort_tie_block_max_abs_diff"] <= 1e-6), "diff": g["G6-REPRO"]["RED_argsort_of_argsort_tie_block_max_abs_diff"]}
    RA["swap pred/true (G6-REPRO)"] = {"fails_as_required": bool(g["G6-REPRO"]["RED_swap_flips_every_gap_sign"])}
    unw = A.stats_block(Du, B, w_zero=0.5, w_random=0.5)
    RA["wrong stratum weights"] = {"fails_as_required": bool(A.max_abs_diff(unw["R1_weighted_cruise_slow_share"], S["R1_weighted_cruise_slow_share"]) > 1e-6)}
    bf = dict(Du)
    bf["traj"] = (np.round(np.asarray(Du["traj"], np.float32) * 4) / 4).astype(np.float32)
    RA["bf16-grid trajectories (G6-FP32)"] = {"fails_as_required": not A.g6_fp32(bf)["PASS"], "share_on_grid": A.g6_fp32(bf)["share_on_0.25m_grid"]}
    q = dict(Du)
    q["v4_all"] = A.sc.step_speed(bf["traj"].astype(np.float64))
    qs = A.stats_block(q, B)
    RA["bf16 grid corrupts the slow share (why G6-FP32 exists)"] = {"slow_share_fp32": S["R1_weighted_cruise_slow_share"]["point"], "slow_share_on_grid": qs["R1_weighted_cruise_slow_share"]["point"],
                                                                    "fails_as_required": bool(abs(qs["R1_weighted_cruise_slow_share"]["point"] - S["R1_weighted_cruise_slow_share"]["point"]) > 1e-6)}
    pk = dict(Du)
    pk["pick"] = (Du["pick"] + 1) % 64
    RA["wrong pick (G6-0)"] = {"fails_as_required": not A.g6_0(pk)["PASS"]}
    eg = dict(Du)
    eg["ego"] = Du["ego"].copy()
    eg["ego"][:, 6] += 1e-3
    RA["perturbed ego (G6-2)"] = {"fails_as_required": not A.g6_2(eg)["PASS"]}
    sb = dict(Du)
    sb["sub"] = Du["sub"].copy()
    sb["sub"][..., 2] += 0.01
    RA["perturbed sub-score (G6-4)"] = {"fails_as_required": not A.g6_4(sb)["PASS"]}
    res["all_data_gates_pass"] = bool(all(v["PASS"] for v in g.values()))
    res["all_RED_arms_fail_as_required"] = bool(all(v["fails_as_required"] for v in RA.values()))
    res["u400_as_P6_is_REFUTED"] = res["u400_as_P6_verdict"]["VERDICT"] == "REFUTED"
    res["PASS"] = bool(res["all_data_gates_pass"] and res["all_RED_arms_fail_as_required"] and res["u400_as_P6_is_REFUTED"])
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("headline", "all_data_gates_pass", "all_RED_arms_fail_as_required", "u400_as_P6_verdict", "PASS", "on_disk_vs_pinned")}, indent=1, default=float))
    print({k: (v["PASS"], ) for k, v in g.items()})
    print({k: v["fails_as_required"] for k, v in RA.items()})
    return 0 if res["PASS"] else 2


if __name__ == "__main__":
    sys.exit(main())

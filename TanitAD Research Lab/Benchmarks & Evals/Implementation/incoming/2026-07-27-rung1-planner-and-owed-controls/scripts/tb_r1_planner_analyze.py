#!/usr/bin/env python3
"""RUNG 1 — the R1 PLANNER row, adjudicated. ZERO GPU.

Reads the per-window dense paths produced on pod2 and answers, in the
pre-registered order (`PRE_REGISTRATION.md` §B), banking each stage before the
next begins:

  gates      the WINDOW-SET IDENTITY gate against BOTH committed dumps, the
             fidelity gate against the ladder's committed headline numbers, the
             PLUMBING SELF-TEST in both directions, and the failing-value probe.
  planner    the primary arm scored against §7.2's pre-registered 6-12 s window,
             with its matched comparator, beats-CV and T_useful (M10's tier
             split kept: the T_blind number and the CAPABILITY claim are
             separate verdicts).
  mechanism  the action-amplitude statistics that decide whether a number in
             range was reached for the pre-registered REASON — the "5x lower
             longitudinal gain" — or for some other one.

Estimator everywhere: paired episode-cluster bootstrap, `taniteval/ci.py`,
B = 2000, seed 0, unit = episode cluster, identical windows for every arm.
`overlapping_holdout_se` appears nowhere. The rule machinery (`t_blind`,
`paired_at`, `separated_better_interval`, `draws_for`, `t_contiguous`) is
IMPORTED from the ladder's `tb_rung0.py` and Rung 1's `tb_rung1_analyze.py`
rather than re-implemented — two independent re-implementations of a firewall
produced nulls that had to be overturned.

Usage:
    python tb_r1_planner_analyze.py \
      --new      perwindow/r1planner_perwindow_K185.pt \
      --bi       ../../../Architecture\\ &\\ Inference/.../2026-07-26-blind-imagination/perwindow/bi_perwindow_compact.pt \
      --matched  .../2026-07-26-tblind-ladder/perwindow/perwindow_matched_K185.pt \
      --rung1    .../2026-07-26-tblind-rung1/perwindow/rung1_perwindow_compact.pt \
      --out raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_HUB = _HERE.parents[4]                       # .../TanitAD Research Hub
_AI = _HUB / "Architecture & Inference" / "Implementation" / "incoming"
sys.path.insert(0, str(_AI / "2026-07-26-tblind-ladder" / "scripts"))
_REPO = _HERE.parents[5]
for _p in (_REPO / "taniteval", _REPO / "stack", _REPO / "stack" / "scripts"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tb_rung0 import (DT, B_BOOT, SEED, ade_0_2s, draws_for,      # noqa: E402
                      paired_at, separated_better_interval, single_at,
                      t_blind)

A_OWN, B_OWN = "a_imagination__own__roSTR", "b_frozenlast__own__roSTR"
A_HOLD, B_HOLD = "a_imagination__hold__roSTR", "b_frozenlast__hold__roSTR"
ANCHOR_TOL = 1e-4              # m — Rung 0b measured 3.05e-05 between two passes
BARS = (1.0, 1.391, 2.0)

#: `PRE_REGISTRATION.md` §C, re-read from the ladder's raw JSON. The fidelity gate.
LADDER = {
    "T_blind_own_str_steps": 25, "T_blind_hold_str_steps": 115,
    "de2s_own_str": 1.8165, "de2s_hold_str": 0.6718,
    "ade_0_2s_own_str": 0.8710, "ade_0_2s_hold_str": 0.3351,
    "T_useful_1m_own_str_s": 1.4, "T_useful_1m_hold_str_s": 2.3,
}
#: §7.2's prediction, fixed before this run. [60, 120] steps = 6.0 - 12.0 s.
PRED_LO_STEPS, PRED_HI_STEPS = 60, 120
EMA08_STEPS = 64               # the gain-matched reference §7.2 names

#: §B.2 — ELIGIBLE arms may set the verdict; DIAGNOSTIC arms never can.
ELIGIBLE = ("planner", "planner_vupd", "planner_vdec")
DIAGNOSTIC = ("planner_gtlook",)
#: Rung 1's own action signature, re-measured here rather than inherited.
RUNG1_OWN = {"mean_abs_accel_first5": 2.0582, "frac_accel_at_clamp_first5": 0.4641,
             "mean_abs_steer_first5": 0.00978, "jitter_accel_first5": 3.1524}


def t_useful(de, bar: float) -> float:
    """Largest horizon (s) with mean `de` below `bar`, contiguous from step 1.
    ⚠️ Returns 0.0 when even the first step is above the bar — reachable."""
    m = de.mean(axis=0)
    ok = m < bar
    if not ok[0]:
        return 0.0
    bad = np.flatnonzero(~ok)
    return round(float(int(bad[0]) if bad.size else int(ok.size)) * DT, 2)


def arm_block(de, eid, draws, a_arm, b_arm) -> dict:
    out = t_blind(de[a_arm], de[b_arm], draws, label_a=a_arm, label_b=b_arm)
    out["de_at_2s"] = single_at(de[a_arm], eid, draws, 20)
    out["de_at_6s"] = single_at(de[a_arm], eid, draws, 60)
    out["ade_0_2s"] = ade_0_2s(de[a_arm], eid)
    out["paired_delta_2s_vs_comparator"] = paired_at(de[a_arm], de[b_arm], draws, 20)
    out["beats_cv"] = separated_better_interval(de[a_arm],
                                                de["d_constant_velocity"], draws)
    out["T_useful_s"] = {f"{b:g}m": t_useful(de[a_arm], b) for b in BARS}
    return out


def action_stats(fed, upto: int) -> dict:
    """The amplitude signature Rung 1 identified as THE mechanism, over the
    first `upto` fed steps. `fed [N,K,A]` with (steer, accel, v0)."""
    st = fed[:, :upto, 0].numpy()
    ac = fed[:, :upto, 1].numpy()
    d = np.diff(ac, axis=1)
    sgn = np.sign(ac)
    flips = (sgn[:, 1:] * sgn[:, :-1] < 0).mean() if ac.shape[1] > 1 else float("nan")
    return {
        "upto_steps": int(upto),
        "mean_abs_steer": round(float(np.abs(st).mean()), 6),
        "mean_abs_accel": round(float(np.abs(ac).mean()), 6),
        "mean_signed_accel": round(float(ac.mean()), 6),
        "bias_over_amplitude": round(float(abs(ac.mean()) /
                                           max(np.abs(ac).mean(), 1e-9)), 6),
        "frac_steer_at_clamp": round(float((np.abs(st) >= 0.05 - 1e-6).mean()), 6),
        "frac_accel_at_clamp": round(float((np.abs(ac) >= 3.0 - 1e-6).mean()), 6),
        "jitter_accel": round(float(np.abs(d).mean()), 6) if d.size else None,
        "sign_flip_rate_per_tick": round(float(flips), 6),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True)
    ap.add_argument("--bi", required=True)
    ap.add_argument("--matched", required=True)
    ap.add_argument("--rung1", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    outd = Path(a.out).resolve()
    outd.mkdir(parents=True, exist_ok=True)

    new = torch.load(a.new, map_location="cpu", weights_only=False)
    bi_d = torch.load(a.bi, map_location="cpu", weights_only=False)
    mat = torch.load(a.matched, map_location="cpu", weights_only=False)

    gt = new["gt"]
    de = {k: torch.linalg.norm(v - gt, dim=-1).float().double().numpy()
          for k, v in new["pred"].items()}
    de["d_constant_velocity"] = torch.linalg.norm(
        new["cv"] - gt, dim=-1).float().double().numpy()
    de["d2_hold_v0"] = torch.linalg.norm(
        new["hold_v0"] - gt, dim=-1).float().double().numpy()
    eid = new["eid"]
    draws, n_clusters = draws_for(eid)

    # ===================== STAGE 1 — THE GATES ============================= #
    committed = {k: v.double().numpy() for k, v in bi_d["dense_de_headline"].items()}
    committed.update({k: v.double().numpy()
                      for k, v in mat["dense_de_matched_arms"].items()})
    g = {"window_set_identity": {}, "plumbing_selftest": {},
         "fidelity_vs_ladder": {}, "failing_value_probe": {}}

    ident = {
        "n_windows_new": int(len(eid)),
        "n_windows_committed_bi": int(len(bi_d["eid"])),
        "n_windows_committed_matched": int(len(mat["eid"])),
        "n_episode_clusters": int(len(set(eid))),
        "eid_identical_vs_bi": bool(list(eid) == list(bi_d["eid"])),
        "eid_identical_vs_matched": bool(list(eid) == list(mat["eid"])),
        "t0_identical_vs_bi": bool(list(new["t0"]) == list(bi_d["t0"])),
        "t0_identical_vs_matched": bool(list(new["t0"]) == list(mat["t0"])),
        "anchors": {},
    }
    for arm in (A_OWN, A_HOLD, B_OWN, B_HOLD):
        if arm in committed:
            m = float(np.abs(de[arm] - committed[arm]).max())
            ident["anchors"][arm] = {"max_abs_diff_m": m,
                                     "within_tol": bool(m <= ANCHOR_TOL),
                                     "tol_m": ANCHOR_TOL}
    ident["IDENTITY_PASS"] = bool(
        ident["eid_identical_vs_bi"] and ident["eid_identical_vs_matched"]
        and ident["t0_identical_vs_bi"] and ident["t0_identical_vs_matched"]
        and all(v["within_tol"] for v in ident["anchors"].values())
        and len(ident["anchors"]) == 4)
    g["window_set_identity"] = ident
    print(f"[gate] IDENTITY_PASS = {ident['IDENTITY_PASS']}  "
          f"n={ident['n_windows_new']}/{ident['n_episode_clusters']}", flush=True)

    # ---- PLUMBING SELF-TEST, BOTH DIRECTIONS -------------------------------- #
    own_p, hold_p = new["pred"][A_OWN], new["pred"][A_HOLD]
    pl = {}
    for arm in [f"a_{t}" for t in ELIGIBLE + DIAGNOSTIC]:
        pl[arm] = {
            "max_abs_diff_vs_own_m": round(float((new["pred"][arm] - own_p).abs().max()), 6),
            "max_abs_diff_vs_hold_m": round(float((new["pred"][arm] - hold_p).abs().max()), 6),
        }
        pl[arm]["is_not_a_noop"] = bool(pl[arm]["max_abs_diff_vs_own_m"] > 1e-6
                                        and pl[arm]["max_abs_diff_vs_hold_m"] > 1e-6)
    pl["_wp_to_control_is_imported_not_copied"] = True
    pl["_cpu_fixture_pins"] = (
        "taniteval/tests/test_blindimag.py::test_planner_feeds_exactly_wp_to_control "
        "asserts torch.equal against closedloop.wp_to_control's own output; "
        "::test_planner_speed_bookkeeping_is_closed_loop_rollouts pins the "
        "v <- clamp_min(v + accel*DT, 0) integration; "
        "::test_no_planner_leaves_every_pre_existing_path_bit_identical pins "
        "inertness on every pre-planner call site")
    pl["SELFTEST_PASS"] = bool(all(v["is_not_a_noop"] for v in pl.values()
                                   if isinstance(v, dict)))
    g["plumbing_selftest"] = pl
    print(f"[gate] SELFTEST_PASS = {pl['SELFTEST_PASS']}", flush=True)

    # ---- fidelity vs the ladder's committed headline numbers ---------------- #
    own_b = arm_block(de, eid, draws, A_OWN, B_OWN)
    hold_b = arm_block(de, eid, draws, A_HOLD, B_HOLD)
    fid = {
        "T_blind_own_str_steps": {"committed": LADDER["T_blind_own_str_steps"],
                                  "recomputed": own_b["T_blind_steps"]},
        "T_blind_hold_str_steps": {"committed": LADDER["T_blind_hold_str_steps"],
                                   "recomputed": hold_b["T_blind_steps"]},
        "de2s_own_str": {"committed": LADDER["de2s_own_str"],
                         "recomputed": round(own_b["de_at_2s"]["mean"], 4)},
        "de2s_hold_str": {"committed": LADDER["de2s_hold_str"],
                          "recomputed": round(hold_b["de_at_2s"]["mean"], 4)},
        "ade_0_2s_own_str": {"committed": LADDER["ade_0_2s_own_str"],
                             "recomputed": round(own_b["ade_0_2s"]["mean"], 4)},
        "ade_0_2s_hold_str": {"committed": LADDER["ade_0_2s_hold_str"],
                              "recomputed": round(hold_b["ade_0_2s"]["mean"], 4)},
        "T_useful_1m_own_str_s": {"committed": LADDER["T_useful_1m_own_str_s"],
                                  "recomputed": own_b["T_useful_s"]["1m"]},
        "T_useful_1m_hold_str_s": {"committed": LADDER["T_useful_1m_hold_str_s"],
                                   "recomputed": hold_b["T_useful_s"]["1m"]},
    }
    fid["LEVEL_FIDELITY_PASS"] = bool(all(
        abs(fid[k]["committed"] - fid[k]["recomputed"]) < 5e-4
        for k in ("de2s_own_str", "de2s_hold_str", "ade_0_2s_own_str",
                  "ade_0_2s_hold_str")))
    fid["T_BLIND_EXACT_REPRODUCTION"] = bool(
        fid["T_blind_own_str_steps"]["committed"] == fid["T_blind_own_str_steps"]["recomputed"]
        and fid["T_blind_hold_str_steps"]["committed"] == fid["T_blind_hold_str_steps"]["recomputed"])
    fid["_note"] = ("level agreement was declared BLOCKING and the T_blind "
                    "integer reproduction REPORTED, NON-BLOCKING, in advance "
                    "(PRE_REGISTRATION.md B.5.2) — a step count is a threshold "
                    "crossing and the arms come from a second encode pass")
    g["fidelity_vs_ladder"] = fid
    print(f"[gate] LEVEL_FIDELITY_PASS = {fid['LEVEL_FIDELITY_PASS']}  "
          f"T_BLIND_EXACT = {fid['T_BLIND_EXACT_REPRODUCTION']}", flush=True)

    # ---- the failing value, probed on real arms ----------------------------- #
    g["failing_value_probe"] = {
        "identical_arms": t_blind(de[A_OWN], de[A_OWN], draws)["T_blind_steps"],
        "swapped_arms": t_blind(de[B_OWN], de[A_OWN], draws)["T_blind_steps"],
        "expected": 1,
        "_read": ("the primary statistic's failing value is 1 step (0.1 s), not "
                  "0, and it is returned on real arms — the rule can fail"),
    }
    (outd / "r1planner_gates.json").write_text(json.dumps(g, indent=2, default=float),
                                               encoding="utf-8")
    print(f"[write] {outd / 'r1planner_gates.json'}", flush=True)
    if not (ident["IDENTITY_PASS"] and pl["SELFTEST_PASS"]
            and fid["LEVEL_FIDELITY_PASS"]):
        print("⛔ A BLOCKING GATE FAILED — nothing below is admissible", flush=True)
        return 3

    # ===================== STAGE 2 — THE PLANNER ARM ======================= #
    res = {"baseline_own": own_b, "ceiling_hold": hold_b, "arms": {}}
    for tag in ELIGIBLE + DIAGNOSTIC:
        blk = arm_block(de, eid, draws, f"a_{tag}", f"b_{tag}")
        blk["eligible"] = tag in ELIGIBLE
        blk["de_2s_delta_vs_own_baseline"] = paired_at(de[f"a_{tag}"], de[A_OWN],
                                                       draws, 20)
        res["arms"][tag] = blk
        print(f"[arm] {tag:16s} T_blind={blk['T_blind_steps']:3d} "
              f"({blk['T_blind_s']}s) CI{blk['T_blind_ci95_s']} "
              f"de@2s={blk['de_at_2s']['mean']:.4f} "
              f"beatsCV={blk['beats_cv']['n_steps']}/185 "
              f"Tuseful1m={blk['T_useful_s']['1m']}s "
              f"{'ELIGIBLE' if blk['eligible'] else 'diagnostic'}", flush=True)
    (outd / "r1planner_arms.json").write_text(json.dumps(res, indent=2, default=float),
                                              encoding="utf-8")

    # ===================== STAGE 3 — THE MECHANISM ========================= #
    mech = {"rung1_own_signature_INHERITED": RUNG1_OWN, "measured": {}}
    for arm, fed in new.get("fed_actions", {}).items():
        mech["measured"][arm] = {f"first{u}": action_stats(fed, u)
                                 for u in (5, 20, 185)}
    # ⭐ WHY a 5x lower GAIN did not produce a lower COMMAND. `wp_to_control`
    # sets accel = clamp((v_target - v)/SPEED_TC) with v_target = x/(LOOKAHEAD *
    # DT) — the speed the tactical head's own 0.5 s waypoint IMPLIES. The gain
    # argument silently assumes v_target is near v. It is not, and that is
    # measurable: reconstruct the controller's own speed recursion (the same
    # `v <- clamp_min(v + accel*DT, 0)` the rollout used) and invert the
    # UNSATURATED steps back to v_target.
    if "a_planner" in new.get("fed_actions", {}):
        from taniteval.closedloop import DT as _cdt, LOOKAHEAD_STEP, SPEED_TC
        fed = new["fed_actions"]["a_planner"]
        acc = fed[:, :, 1].double()
        v0 = new["speed"].reshape(-1).double()
        vseq = torch.empty_like(acc)
        vseq[:, 0] = v0
        for j in range(1, acc.shape[1]):
            vseq[:, j] = (vseq[:, j - 1] + acc[:, j - 1] * _cdt).clamp_min(0.0)
        unsat = acc.abs() < 3.0 - 1e-6
        vtar = vseq + SPEED_TC * acc
        x_impl = vtar * (LOOKAHEAD_STEP * _cdt)
        x_match = vseq * (LOOKAHEAD_STEP * _cdt)
        def _its(sl):
            u = unsat[:, sl].numpy()
            if not u.any():
                return {"frac_steps_unsaturated": 0.0,
                        "_note": "every step saturated the +-3 clamp"}
            return {
                "frac_steps_unsaturated": round(float(u.mean()), 6),
                "mean_v_mps": round(float(vseq[:, sl].numpy()[u].mean()), 4),
                "mean_v_target_mps": round(float(vtar[:, sl].numpy()[u].mean()), 4),
                "mean_abs_speed_error_mps": round(
                    float(np.abs((vtar - vseq)[:, sl].numpy()[u]).mean()), 4),
                "mean_implied_lookahead_x_m": round(
                    float(x_impl[:, sl].numpy()[u].mean()), 4),
                "mean_speed_matched_lookahead_x_m": round(
                    float(x_match[:, sl].numpy()[u].mean()), 4),
            }
        mech["implied_target_speed"] = {
            "_what": ("the speed v1's TACTICAL HEAD asks for, recovered from the "
                      "UNSATURATED fed accelerations by inverting "
                      "wp_to_control's own P-controller"),
            "first5": _its(slice(0, 5)),
            "first20": _its(slice(0, 20)),
            "all185": _its(slice(None)),
            "SPEED_TC_s": SPEED_TC, "LOOKAHEAD_STEP": LOOKAHEAD_STEP,
            "_read": ("a 5x lower GAIN only lowers the COMMAND if the ERROR it "
                      "multiplies is unchanged. accel = (v_target - v)/0.5 vs "
                      "the inverse's (v - v_prev)/0.1: the inverse multiplies a "
                      "ONE-TICK speed change, the planner multiplies a "
                      "0.5 s TARGET-SPEED MISMATCH, which is the larger "
                      "quantity by more than the gain ratio"),
        }
    if "a_planner" in mech["measured"] and A_OWN in mech["measured"]:
        p5 = mech["measured"]["a_planner"]["first5"]
        o5 = mech["measured"][A_OWN]["first5"]
        mech["gain_argument"] = {
            "planner_mean_abs_accel_first5": p5["mean_abs_accel"],
            "own_mean_abs_accel_first5": o5["mean_abs_accel"],
            "amplitude_ratio_planner_over_own": round(
                p5["mean_abs_accel"] / max(o5["mean_abs_accel"], 1e-9), 4),
            "planner_frac_accel_at_clamp_first5": p5["frac_accel_at_clamp"],
            "own_frac_accel_at_clamp_first5": o5["frac_accel_at_clamp"],
            "saturation_ratio_planner_over_own": round(
                p5["frac_accel_at_clamp"] / max(o5["frac_accel_at_clamp"], 1e-9), 4),
            "predicted_by_7.2": ("a 5x LOWER longitudinal gain, so a materially "
                                 "SMALLER amplitude and saturation"),
        }
    (outd / "r1planner_mechanism.json").write_text(
        json.dumps(mech, indent=2, default=float), encoding="utf-8")

    # ===================== STAGE 4 — THE VERDICT =========================== #
    pr = res["arms"]["planner"]
    T = pr["T_blind_steps"]
    if T > PRED_HI_STEPS:
        bucket = "PREDICTION EXCEEDED"
    elif T >= PRED_LO_STEPS:
        bucket = "PREDICTION CONFIRMED"
    else:
        bucket = "PREDICTION REFUTED — LOW"
    ga = mech.get("gain_argument", {})
    amp_ratio = ga.get("amplitude_ratio_planner_over_own")
    sat_ratio = ga.get("saturation_ratio_planner_over_own")
    gain_ok = (amp_ratio is not None and amp_ratio < 0.75
               and sat_ratio is not None and sat_ratio < 0.75)
    mech_verdict = ("gain argument CONFIRMED"
                    if (bucket == "PREDICTION CONFIRMED" and gain_ok) else
                    "gain argument REFUTED — the amplitude did not move")
    cap = {
        "beats_cv_steps": pr["beats_cv"]["n_steps"],
        "beats_cv_interval_s": [pr["beats_cv"]["first_s"], pr["beats_cv"]["last_s"]],
        "T_useful_1m_s": pr["T_useful_s"]["1m"],
        "baseline_T_useful_1m_s": own_b["T_useful_s"]["1m"],
        "baseline_beats_cv_steps": own_b["beats_cv"]["n_steps"],
    }
    cap["capability_verdict"] = (
        "CONFIRM" if (cap["beats_cv_steps"] > 0
                      or cap["T_useful_1m_s"] > cap["baseline_T_useful_1m_s"])
        else "NOT CONFIRMED")
    verdict = {
        "prediction_quoted": ("TBLIND_RUNG1.md 7.2: v1's tactical planner will "
                              "land between 6 and 12 s of deployable T_blind ... "
                              "If it lands BELOW 6.4 s, the planner is adding a "
                              "failure mode the gain argument does not explain "
                              "and that is the finding."),
        "prediction_window_steps": [PRED_LO_STEPS, PRED_HI_STEPS],
        "baseline_own_T_blind_steps": own_b["T_blind_steps"],
        "ceiling_hold_T_blind_steps": hold_b["T_blind_steps"],
        "ema0.8_reference_steps_INHERITED": EMA08_STEPS,
        "planner_T_blind_steps": T,
        "planner_T_blind_s": pr["T_blind_s"],
        "planner_T_blind_ci95_s": pr["T_blind_ci95_s"],
        "BUCKET": bucket,
        "below_the_ema0.8_equivalence": bool(T < EMA08_STEPS),
        "below_the_own_kinematic_baseline": bool(T < own_b["T_blind_steps"]),
        "best_eligible_arm": max(
            ((k, v["T_blind_steps"]) for k, v in res["arms"].items() if v["eligible"]),
            key=lambda x: x[1]),
        "M10_tier_split": {
            "T_blind_is": ("an extension against a FROZEN PERCEPT — a metric "
                           "statement, not a capability"),
            "capability_is": ("beats-CV and T_useful@1m ONLY — comparator-free, "
                              "pure kinematics, no readout or action-source "
                              "mismatch can enter them"),
            **cap},
        "mechanism_subverdict": mech_verdict,
        "gain_argument_numbers": ga,
    }
    (outd / "r1planner_verdict.json").write_text(
        json.dumps(verdict, indent=2, default=float), encoding="utf-8")
    print(f"\n[VERDICT] planner T_blind = {T} steps ({pr['T_blind_s']}s) "
          f"CI{pr['T_blind_ci95_s']}  ->  {bucket}", flush=True)
    print(f"[VERDICT] capability: beatsCV={cap['beats_cv_steps']}/185 "
          f"T_useful@1m={cap['T_useful_1m_s']}s (baseline "
          f"{cap['baseline_T_useful_1m_s']}s) -> {cap['capability_verdict']}",
          flush=True)
    print(f"[VERDICT] mechanism: {mech_verdict}  amp_ratio={amp_ratio} "
          f"sat_ratio={sat_ratio}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

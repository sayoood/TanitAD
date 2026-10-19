#!/usr/bin/env python3
"""Step B read: the pre-registered endpoint and decision of `PREREG_L2_STEPB.md` (sealed, Revision 1), on labelled search outputs.  BUILT AND TESTED ON PLANTED DATA ONLY (test_stepB.py); it has read NO real label.

ENDPOINT (L1 prereg section 4): per token and search seed, d = true PDMS(search output) - true PDMS(P2's pick), the pick's label being the BANKED one; the token's value is the mean over the 3 seeds; the interval is
`tts_cem.delta_ci` (ratio of sums, WHOLE LOGS resampled, 10,000, seed 20260927, the same draws every call).  UNITS: PDMS points = 100 x the [0, 1] difference (L1's "+0.30" is on the published 0-100 scale; the
sub-score terms are the same).  It answers 'another draw of LOGS' and nothing else: not another head seed (H-ESTIM-SEED-1), not another search seed (the spread of the 3 per-seed points is reported beside it).

DECISION (draft section 3), applied in this order, each term a pre-committed constant below:
  O4 INERT      the search output differs from the pick by > 0.5 m ADE on < 5 % of (token, seed) outputs      (`tts_cem.inert_verdict`)  -> reported INERT, not refuted
  O1 PASS       dPDMS point >= +0.30  AND lower CI > 0  AND dPDMS point > the search-seed range  AND dNC, dDAC, dTTC, dC points each >= -0.25
  TRADE         the three dPDMS terms clear and at least one of dNC / dDAC / dTTC / dC point < -0.25
  NOT CLEARED   anything else (the dPDMS bar is not met)
Per-family reads (LONGITUDINAL / LATERAL / TACTICAL; STRATEGIC PAUSED, PI ruling 2026-09-19 ITEM 25) are never pooled into one score.  ADE alone is never the result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import stepB  # noqa: E402

A, T = stepB.A, stepB.T

# ------------------------------------------------------------------------------------------------------------------------------ pre-committed constants (L1 prereg section 4 / 5, units: PDMS points)
BAR_DPDMS = 0.30                 # point >= +0.30
BAR_SAFETY = -0.25               # dNC, dDAC, dTTC, dC point each >= -0.25
SAFETY_TERMS = ("nc", "dac", "ttc", "c")
SUBS = A.SUBS                    # ("nc", "dac", "ep", "ttc", "c", "ddc")
POINTS = 100.0                   # [0, 1] -> points
OUTCOMES = ("O1_PASS", "TRADE", "NOT_CLEARED", "O4_INERT")
STRATEGIC = "PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE ITEM 25): reported as PAUSED, not as a pass"
STAMPS = ("SEEN-SCENE (L1 amendment A1.3); the head is TOKEN-FRESH and LOG-SHARED on carve B (400 tokens disjoint from the 400 the head was fitted on, inside the SAME 24 logs: NOT a fresh-log test); "
          "SELECTION tier, label-only (not T1, not navtest); one training seed per head arm (H-ESTIM-SEED-1 open); the interval is 'another draw of LOGS' only; no capability claim")
# Revision 1 (MM ruling, pre-seal): every outcome ends in a NAMED next arm, written before any outcome exists.  Closing a search is never the end of the turn.
NEXT_ARM = {
    "O1_PASS": ("EP-ADMISSION on BOTH banks, built ONLY now (after this O1): (i) the frozen literal in-fan bank 'P2 + the L2 EP row' (head seeds 0 / 1 = the candidate's two seeds; a head-file input to "
                "build_candidate_bank.py via l2_scorer.install_ep_head) and (ii) a search-output bank in the same schema from this run's labelled outputs; ADMIT needs BOTH to admit (thresholds, estimator and "
                "rules frozen; the candidate block is written before either bank is read).  Then the shared navtest look, ONLY IF Step A also read O1 (READ 1 or READ 2)."),
    "TRADE": ("build and run the TRADE-restricted arm ON this TRADE (spec: PREREG section 3: the same search restricted to the DAC/NC-safe subset of candidates by the scorer's own DAC and NC heads, "
              "no new head; its own identity and reach checks first)."),
    "NOT_CLEARED": ("L2-BRANCH: the last scorer-branch block trainable on cached ctx (dev-box GPU, shares SH5's lockstep trainer); UNBLOCKED BY its own amendment sealed before any run, a launch_gate.py "
                    "PASS token (it TRAINS, so launch_gate.py applies) and the dev-box GPU lock."),
    "O4_INERT": ("the raised-M widened arm ONCE: M = 256, 32 elites, widen 2.0, clips at the admitted hull, config stepB_config_O4.json (sha256 pinned in the sealed text), its own G-REACH-STEPB stamp per head.  "
                 "If that is ALSO INERT, the search is CLOSED for this scorer and the report must NAME the next lever: L2-BRANCH (UNBLOCKED BY its own sealed amendment + a launch_gate.py PASS token + the "
                 "dev-box GPU lock) or the P6 trunk (UNBLOCKED BY the PI's authorisation of the trunk compute).  Closing is never the end of the turn."),
}
LICENSING = ("Step B is a pre-registered DIAGNOSTIC on carve B and licenses nothing alone (MM Decision 0): EP-ADMISSION and the shared navtest look need a Step A O1 (READ 1 or READ 2) AND a Step B O1 on "
             "BOTH head seeds.  Every Step B outcome is reported whatever Step A reads.  Step A is NOT read by this chain.")


def classify_stepB(dpdms: dict, seed_range: float, safety_points: dict, inert: dict) -> dict:
    """the decision, as a pure function of the four read-outs (so every term has its own regression arm in test_stepB.py)"""
    if inert["inert"]:
        return {"outcome": "O4_INERT", "why": "the search output differs from the pick on < 5 % of outputs: INERT, not refuted (precedent P4 F1, 1 of 120)"}
    terms = {"point_ge_bar": bool(dpdms["point"] >= BAR_DPDMS), "lower_ci_gt_0": bool(dpdms["lo"] > 0.0), "point_gt_seed_range": bool(dpdms["point"] > seed_range)}
    safety = {k: bool(safety_points[k] >= BAR_SAFETY) for k in SAFETY_TERMS}
    pdms_clears = all(terms.values())
    if pdms_clears and all(safety.values()):
        out = "O1_PASS"
    elif pdms_clears:
        out = "TRADE"
    else:
        out = "NOT_CLEARED"
    return {"outcome": out, "dpdms_terms": terms, "safety_terms": safety}


def log_index(bundle: dict, idx) -> tuple:
    names = sorted(set(str(x) for x in bundle["log"]))
    li = {l: i for i, l in enumerate(names)}
    return np.array([li[str(bundle["log"][i])] for i in idx], dtype=np.int64), len(names), names


def _ci(per_token, log_idx, n_logs, boot):
    num = np.bincount(log_idx, weights=np.asarray(per_token, np.float64), minlength=n_logs)
    den = np.bincount(log_idx, minlength=n_logs).astype(np.float64)
    return A.ratio_ci(num, den, boot)


def _yaw_wrap(x):
    return (np.asarray(x) + math.pi) % (2 * math.pi) - math.pi


def geometry(out_exec, pick_exec) -> dict:
    """LATERAL geometry of the search output against the pick (descriptive): heading deviation at the last non-repaired pose and the largest yaw-rate deviation (rad/s, 0.2 s steps)"""
    o, p = np.asarray(out_exec, np.float64), np.asarray(pick_exec, np.float64)
    hd = np.abs(_yaw_wrap(o[:, :, -2, 2] - p[:, :, -2, 2]))
    yr = np.abs(_yaw_wrap(np.diff(o[..., 2], axis=-1)) / 0.2 - _yaw_wrap(np.diff(p[..., 2], axis=-1)) / 0.2)
    return {"heading_dev_end_rad_mean": float(hd.mean()), "yaw_rate_dev_max_rad_s_mean": float(yr.max(-1).mean())}


def analyse(bundle: dict, so: dict, lab: dict, B: int = A.N_BOOT, seed: int = A.SEED) -> dict:
    """one arm (one head seed, or the P2HEAD control): endpoint, seed spread, safety terms, INERT, decision and the family reads"""
    idx = np.asarray(so["bundle_index"], np.int64)
    n, S = so["exec"].shape[:2]
    if [str(t) for t in lab["token"]] != [str(t) for t in so["token"]]:
        raise ValueError("labels and search outputs are not for the same tokens in the same order")
    if not (np.isfinite(lab["pdms"]).all() and np.isfinite(lab["sub"]).all()):
        raise ValueError("non-finite labels: a partial labelling is not analysed (no silent drop of tokens)")
    pick = np.asarray(bundle["pick"], np.int64)[idx]
    pick_sub = np.asarray(bundle["sub"], np.float64)[idx, pick]                     # [n,6]  the pick's BANKED label
    pick_pdms = np.asarray(bundle["pdms"], np.float64)[idx, pick]                   # [n]
    li, n_logs, _ = log_index(bundle, idx)
    boot = A.boot_indices(n_logs, B, seed)
    d_pdms = POINTS * (np.asarray(lab["pdms"], np.float64) - pick_pdms[:, None])    # [n,S] PDMS points
    d_sub = POINTS * (np.asarray(lab["sub"], np.float64) - pick_sub[:, None, :])    # [n,S,6]
    ep = _ci(d_pdms.mean(1), li, n_logs, boot)
    per_seed = [_ci(d_pdms[:, s], li, n_logs, boot)["point"] for s in range(S)]
    spread = T.seed_spread(per_seed)
    sub_ci = {k: _ci(d_sub[:, :, j].mean(1), li, n_logs, boot) for j, k in enumerate(SUBS)}
    pooled_out = np.asarray(so["exec"]).reshape(-1, 20, 3)
    pooled_pick = np.repeat(np.asarray(so["pick_exec"]), S, axis=0)
    inert = T.inert_verdict(pooled_out, pooled_pick)
    inert["per_seed_frac_moved"] = [float(np.mean([T.ade_xy(o, p) > T.INERT_ADE_M for o, p in zip(np.asarray(so["exec"])[:, s], np.asarray(so["pick_exec"]))])) for s in range(S)]
    dec = classify_stepB(ep, spread["range"], {k: sub_ci[k]["point"] for k in SAFETY_TERMS}, inert)
    th = np.asarray(so["theta"], np.float64)
    moved = np.asarray(so["moved"], bool)
    cls_out = np.array([[A.command_class(e) for e in row] for row in np.asarray(so["exec"])])
    cls_pick = np.array([A.command_class(e) for e in np.asarray(so["pick_exec"])])
    spd_up = moved & (th[..., 0] > 0)
    spd_dn = moved & (th[..., 0] < 0)
    fam = {"LONGITUDINAL": {"dEP_points": sub_ci["ep"], "dTTC_points": sub_ci["ttc"], "share_outputs_speed_up": float(spd_up.mean()), "share_outputs_slow_down": float(spd_dn.mean()),
                            "dPDMS_points_given_speed_up (descriptive)": float(d_pdms[spd_up].mean()) if spd_up.any() else None,
                            "dPDMS_points_given_slow_down (descriptive)": float(d_pdms[spd_dn].mean()) if spd_dn.any() else None},
           "LATERAL": {"dDAC_points": sub_ci["dac"], "dDDC_points": sub_ci["ddc"], "mean_abs_offset_m_given_moved": float(np.abs(th[..., 1][moved]).mean()) if moved.any() else None,
                       **geometry(so["exec"], np.asarray(so["pick_exec"])[:, None])},
           "TACTICAL": {"command_class_change_share (proxy, HYPOTHESIS: nav_command bearing rule on the end point)": float((cls_out != cls_pick[:, None]).mean())},
           "STRATEGIC": STRATEGIC}
    # descriptive: the in-fan re-ranking by the arm's scorer, from BANKED labels (no new labelling): L2's own argmax vs P2's pick, and the search's gain BEYOND that argmax
    am = np.asarray(so["arm_argmax"], np.int64)
    am_pdms = np.asarray(bundle["pdms"], np.float64)[idx, am]
    infan = {"arm_argmax_differs_from_p2_pick_share": float((am != pick).mean()), "dPDMS_points_arm_argmax_minus_p2_pick (banked labels, no search)": _ci(POINTS * (am_pdms - pick_pdms), li, n_logs, boot),
             "dPDMS_points_search_output_minus_arm_argmax": _ci(POINTS * (np.asarray(lab["pdms"], np.float64).mean(1) - am_pdms), li, n_logs, boot)}
    return {"arm": json.loads(str(so["meta"])).get("arm") if "meta" in so else None, "n_tokens": int(n), "n_seeds": int(S), "n_logs": int(n_logs), "units": "PDMS points = 100 x the [0,1] difference",
            "endpoint_dPDMS_points": ep, "per_seed_points": spread, "sub_score_deltas_points": sub_ci, "inert": inert, "decision": dec, "families": fam, "in_fan_decomposition_descriptive": infan,
            "moved_share": float(moved.mean()),
            "tier_and_stamp": f"{STAMPS}; interval = paired log-cluster bootstrap (B={B}, seed {seed}); the search-seed spread is reported beside it"}


def paired_vs_control(bundle: dict, so: dict, lab_arm: dict, lab_ctl: dict, B: int = A.N_BOOT, seed: int = A.SEED) -> dict:
    """DESCRIPTIVE attribution (never gating): per-token mean PDMS of the L2 arm's search output minus the P2HEAD control's, same tokens, same search seeds, paired log-cluster bootstrap, in points.
    It separates 'the search found something' from 'the L2 head is why'."""
    if [str(t) for t in lab_arm["token"]] != [str(t) for t in lab_ctl["token"]]:
        raise ValueError("the arm and its control are not labelled on the same tokens in the same order")
    idx = np.asarray(so["bundle_index"], np.int64)
    li, n_logs, _ = log_index(bundle, idx)
    return _ci(POINTS * (np.asarray(lab_arm["pdms"], np.float64).mean(1) - np.asarray(lab_ctl["pdms"], np.float64).mean(1)), li, n_logs, A.boot_indices(n_logs, B, seed))


def combine_head_seeds(per_head: dict) -> dict:
    """the draft's rule for two head seeds: the WORST outcome over head seeds (order O1_PASS < TRADE < NOT_CLEARED < O4_INERT) is the combined one, so O1 needs every head seed to pass (as L2-ADMIT does,
    'on BOTH seeds') and a single non-pass head is visible, never averaged away."""
    order = {"O1_PASS": 0, "TRADE": 1, "NOT_CLEARED": 2, "O4_INERT": 3}
    outs = {k: v["decision"]["outcome"] for k, v in per_head.items()}
    worst = max(outs.values(), key=lambda o: order[o])
    return {"per_head_seed": outs, "combined": worst, "rule": "worst outcome over head seeds; O1 needs every head seed to pass"}


def final_verdict(res: dict, combined: dict | None, attribution: dict | None) -> dict:
    """the one-screen verdict the chain prints: the combined outcome (worst of the two head seeds), the per-head outcomes, the NAMED next arm, the licensing rule, the stamps, the family status.  A pure
    function of the analysis result, so it is tested on planted results (an outcome with no named next arm is a defect)."""
    if not combined:
        raise ValueError("no gated arm in the analysis: there is no verdict (a P2HEAD-only analysis is descriptive)")
    out = combined["combined"]
    if out not in NEXT_ARM:
        raise ValueError(f"unknown outcome {out!r}")
    heads = {k: {"outcome": res[k]["decision"]["outcome"], "dPDMS_points": res[k]["endpoint_dPDMS_points"]["point"], "ci": [res[k]["endpoint_dPDMS_points"]["lo"], res[k]["endpoint_dPDMS_points"]["hi"]],
                 "seed_range_points": res[k]["per_seed_points"]["range"], "moved_share": res[k]["moved_share"]} for k in combined["per_head_seed"]}
    return {"combined": out, "rule": combined["rule"], "per_head": heads, "next_arm": NEXT_ARM[out], "licensing": LICENSING, "stamps": STAMPS, "STRATEGIC": STRATEGIC,
            "families": "LONGITUDINAL / LATERAL / TACTICAL are read per arm under 'arms' (never pooled); STRATEGIC PAUSED",
            "attribution_vs_P2HEAD": "descriptive, not gating" if attribution else "no P2HEAD arm in this analysis"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--search", action="append", required=True, help="repeat per head seed / arm: NAME=path/to/search_out.npz")
    ap.add_argument("--labels", action="append", required=True, help="repeat, same order: NAME=path/to/labels_stepB.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    b = A.load_npz(a.bundle)
    S = dict(x.split("=", 1) for x in a.search)
    L = dict(x.split("=", 1) for x in a.labels)
    if set(S) != set(L):
        raise SystemExit("--search and --labels must name the same arms")
    res, raw = {}, {}
    for name in S:
        so = {k: v for k, v in np.load(S[name], allow_pickle=False).items()}
        lab = {k: v for k, v in np.load(L[name], allow_pickle=False).items()}
        res[name] = analyse(b, so, lab)
        raw[name] = (so, lab)
    gated = {k: v for k, v in res.items() if not k.upper().startswith("P2HEAD")}
    ctl = [k for k in res if k.upper().startswith("P2HEAD")]
    attribution = {k: paired_vs_control(b, raw[k][0], raw[k][1], raw[ctl[0]][1]) for k in gated} if ctl else None
    out = {"arms": res, "combined": combine_head_seeds(gated) if gated else None, "attribution_vs_P2HEAD_points (descriptive, not gating)": attribution, "strategic": STRATEGIC}
    out["verdict"] = final_verdict(res, out["combined"], attribution) if gated else None
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("ZZANALYSE", json.dumps(out["combined"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

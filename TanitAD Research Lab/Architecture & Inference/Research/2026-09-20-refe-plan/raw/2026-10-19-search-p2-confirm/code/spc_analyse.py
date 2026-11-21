#!/usr/bin/env python3
"""The registered READ C1 analysis of SEARCH-P2-CONFIRM.  Built and tested on PLANTED data only; it has read no B-REST label.

The endpoint, interval, seed spread, safety points, INERT test and the four-family blocks are `analyse_stepB.analyse` UNCHANGED (sealed Step B code).  This file adds ONLY: the C1 outcome classifier (the
sealed Step B O1 terms plus the NEGATIVE / DIRECTIONAL / NULL split the prereg prices), the committed NEXT arm of each outcome, the stamps, and the REPORTED-ONLY A3 / A4 / A5 concordance of the search outputs
(H-SEARCH-ADMIT-A3's quantity; ungated: no outcome depends on it).

Outcome order (each term a pre-committed constant): O4_INERT -> O1_CONFIRMED (point >= +0.30 AND lower CI > 0 AND point > the 3-seed range AND dNC, dDAC, dTTC, dC points each >= -0.25) -> TRADE (the three dPDMS
terms clear, a safety point < -0.25) -> NEGATIVE (upper CI < 0) -> O2_DIRECTIONAL (lower CI > 0) -> O3_NULL (everything else; the CI contains 0).  Units: PDMS points.
    python spc_analyse.py --bundle <B-REST bundle.npz> --search search_P2HEAD.npz --labels labels_P2HEAD.npz --out RESULT_SPC.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

STEPB_CODE = os.environ.get("SPC_STEPB_CODE", "C:/Users/Admin/par18/l2stepb/code")
L1_CODE = os.environ.get("L1_CODE", "C:/Users/Admin/par18/l1run/code")
for _p in (STEPB_CODE, L1_CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import analyse_stepB as AB  # noqa: E402
import stepA as A  # noqa: E402

BAR_DPDMS, BAR_SAFETY = AB.BAR_DPDMS, AB.BAR_SAFETY           # +0.30, -0.25 (the sealed Step B constants, imported not copied)
SAFETY_TERMS = AB.SAFETY_TERMS
OUTCOMES = ("O1_CONFIRMED", "O2_DIRECTIONAL", "TRADE", "O3_NULL", "NEGATIVE", "O4_INERT")
STRATEGIC = AB.STRATEGIC
STAMPS = ("MEASURED; SEEN-SCENE (all of B-REST is inside M-SCENE; M-PDM overlap 0); TOKEN-FRESH to the search and LOG-SHARED with Step A and carve B (the same 24 logs): NOT a fresh-log test; SELECTION tier, label-only "
          "(not T1, not navtest); ONE training seed, no head substitution (H-ESTIM-SEED-1 open); the interval is the paired log-cluster bootstrap (B = 10,000, seed 20260927) = 'another draw of LOGS' only, the "
          "3-search-seed spread beside it; READ C1 alone, never pooled with carve B; no capability claim; STRATEGIC PAUSED")
NEXT_ARM = {
    "O1_CONFIRMED": ("(i) build the search-output EP-ADMISSION bank from this read (candidate block in the sealed text; DQ6 adapter and 24-log error-rate re-simulation first), (ii) the navtest look C2 only after admission and its own "
                     "amendment, and in the same turn the SEARCH-DOSE arm (stepB_config_O4.json, M = 256) on the SAME bundle, CPU only, under its own amendment."),
    "O2_DIRECTIONAL": ("the SEARCH-DOSE arm on the same bundle (own amendment); the admission bank only if the seal licenses it on O2 (DQ2); the size question goes to navtest C2, whose power is far higher."),
    "TRADE": ("the TRADE-restricted arm (Step B sealed text section 3: elites and pick restricted to candidates whose sigma(NC) and sigma(DAC) are >= the pick's inside the same set; no new head), built ON this TRADE."),
    "O3_NULL": ("not a refutation (power at half the observed effect is about 27 % / 8 %): the SEARCH-DOSE arm on the same bundle (CPU only); if it also reads NULL the search is CLOSED for P2's scorer on this footing "
                "and the report NAMES the next levers with what unblocks them: the P6 trunk (PI authorisation of trunk compute) and an L2-BRANCH with a TTC-protecting term (own amendment, launch_gate.py PASS token, the GPU lock)."),
    "NEGATIVE": ("the search HURTS on this footing: reported plainly; the levers named under O3_NULL (the SEARCH-DOSE arm only if the moved share is < 5 %)."),
    "O4_INERT": ("the raised-M arm ONCE (stepB_config_O4.json, own G-REACH stamp), then CLOSE with a named next lever (the P6 trunk or L2-BRANCH, with what unblocks each)."),
}


def classify_c1(dpdms: dict, seed_range: float, safety_points: dict, inert: dict) -> dict:
    """pure; every term has its own regression arm in test_spc.py"""
    if inert["inert"]:
        return {"outcome": "O4_INERT", "why": "the search output differs from the pick on < 5 % of outputs"}
    terms = {"point_ge_bar": bool(dpdms["point"] >= BAR_DPDMS), "lower_ci_gt_0": bool(dpdms["lo"] > 0.0), "point_gt_seed_range": bool(dpdms["point"] > seed_range)}
    safety = {k: bool(safety_points[k] >= BAR_SAFETY) for k in SAFETY_TERMS}
    clears = all(terms.values())
    if clears and all(safety.values()):
        out = "O1_CONFIRMED"
    elif clears:
        out = "TRADE"
    elif dpdms["hi"] < 0.0:
        out = "NEGATIVE"
    elif dpdms["lo"] > 0.0:
        out = "O2_DIRECTIONAL"
    else:
        out = "O3_NULL"
    return {"outcome": out, "dpdms_terms": terms, "safety_terms": safety, "safety_fail_beside_directional": bool(out == "O2_DIRECTIONAL" and not all(safety.values()))}


def concordance(bundle: dict, so: dict, lab: dict, B: int = A.N_BOOT, seed: int = A.SEED) -> dict:
    """REPORTED ONLY: A3 / A4 / A5 of (P2's pick, search output) pairs from the banked logits and the labels (L1's stepA.admission_terms via Step B Amendment 1's own pair builder); no outcome depends on it.
    The model identity that Amendment 1 ran (the scorer re-run on 16 tokens) is NOT repeated here: it is stated as NOT RUN."""
    try:
        import amend1_concordance as AM
        bi = np.asarray(so["bundle_index"], np.int64)
        li, n_logs, _ = AB.log_index(bundle, bi)
        arm = {"name": "P2HEAD", "so": so, "lab": lab}
        P, lidx, sidx, counts = AM.build_pairs(bundle, arm, li, np.asarray(so["pred_logit_out"], np.float64))
        S = int(np.asarray(so["moved"]).shape[1])
        sums = AM.arm_sums(P, lidx, sidx, n_logs, S)
        idx = A.boot_indices(n_logs, B, seed)
        out = {"pooled": {k: AM.ci_dict(sums["pooled"][k], idx) for k in ("A3", "A4", "A5")},
               "per_seed": [{k: AM.ci_dict(sums["per_seed"][s][k], idx) for k in ("A3", "A4", "A5")} for s in range(S)], "pair_counts": sums["pair_counts"], "counts": counts,
               "status": "REPORTED ONLY, ungated; model identity NOT RUN here (Step B Amendment 1 ran it on 16 tokens)", "hypothesis": "H-SEARCH-ADMIT-A3 (one scorer here: n = 1, so the relation is stated, not tested)"}
        return AM._clean(out)
    except Exception as e:  # reported-only: never a reason to void the read
        return {"status": f"NOT COMPUTABLE ({type(e).__name__}: {e})"}


def analyse_c1(bundle: dict, so: dict, lab: dict, B: int = A.N_BOOT, seed: int = A.SEED, with_concordance: bool = True) -> dict:
    res = AB.analyse(bundle, so, lab, B=B, seed=seed)
    dp = res["endpoint_dPDMS_points"]
    sr = res["per_seed_points"]["range"]
    safety = {k: res["sub_score_deltas_points"][k]["point"] for k in SAFETY_TERMS}
    verdict = classify_c1(dp, sr, safety, res["inert"])
    out = {"read": "C1", "arm": "P2HEAD (no head substitution)", "outcome": verdict["outcome"], "classification": verdict, "next_arm": NEXT_ARM[verdict["outcome"]],
           "endpoint_dPDMS_points": dp, "per_seed_points": res["per_seed_points"], "safety_points": safety, "sub_score_deltas_points": res["sub_score_deltas_points"], "inert": res["inert"],
           "families": res["families"], "sealed_step_b_rule_for_reference": res["decision"], "moved_share": res["moved_share"], "n_tokens": res["n_tokens"], "n_logs": res["n_logs"], "units": res["units"],
           "stamps": STAMPS, "STRATEGIC": STRATEGIC, "not_computable_families": {"LONGITUDINAL.headway": "n = 0: the label-only tier carries no lead-agent headway row beyond TTC",
                                                                                  "LATERAL.curvature_and_cross_track": "n = 0: no closed-loop rollout at the selection tier",
                                                                                  "TACTICAL.manoeuvre_decision_quality": "n = 0: no manoeuvre decision is scored at this tier"}}
    if with_concordance:
        out["concordance_reported_only"] = concordance(bundle, so, lab, B, seed)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    b = A.load_npz(a.bundle)
    so = {k: v for k, v in np.load(a.search, allow_pickle=False).items()}
    lab = {k: v for k, v in np.load(a.labels, allow_pickle=False).items()}
    res = analyse_c1(b, so, lab)
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: None if (isinstance(o, float) and not math.isfinite(o)) else (o.item() if hasattr(o, "item") else str(o)))
    print("ZZANALYSE_SPC", res["outcome"], json.dumps(res["endpoint_dPDMS_points"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Derives the tables REPORT.md quotes from published_rows.py (PUBLISHED numbers) and the REFe registry numbers (MEASURED, T1).

Writes next to itself: published_subscores.csv, rl_step_deltas.csv, testtime_and_scorer_only_deltas.csv,
gap_arithmetic.json, lever_ranking.csv.   CPU only, stdlib only.   python make_tables.py

The trade rule (fixed here, BEFORE reading the deltas, applied identically to every pair):
  TRADE-SAFETY  if min(dNC, dTTC) < -0.25            (the same 0.25 band the register uses for NC/TTC regressions)
  TRADE-COMFORT if dC < -1.0
  DAC-DOWN      flag only, if dDAC < -0.5
  NO-TRADE      if dEP >= +0.5 and none of the above
  EP-DOWN       if dEP < -0.5 (the step bought safety with progress)
  EP-FLAT       otherwise
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
from published_rows import R  # noqa: E402

BY = {r["id"]: r for r in R}
FAM = ["NC", "DAC", "TTC", "C", "EP", "DDC", "PDMS", "EPDMS"]


def w(name, header, rows):
    with open(os.path.join(HERE, name), "w", newline="", encoding="utf-8") as f:
        c = csv.writer(f)
        c.writerow(header)
        c.writerows(rows)


# 1 ---- every published row, in one canonical column order ------------------------------------------------------
rows = []
for r in R:
    v = r["values"]
    rows.append([r["id"], r["paper"], r["key"], "Table " + r["table"], r["pdf_page"], r["row"], r["bench"], r["check"]] +
                [v.get(k, "") for k in FAM])
w("published_subscores.csv", ["id", "paper", "arxiv", "table", "pdf_page", "row", "bench", "check"] + FAM, rows)


def delta(a, b):
    va, vb = BY[a]["values"], BY[b]["values"]
    scale = 100.0 if BY[a]["bench"].startswith("NAVSIM v2 navhard") else 1.0
    out = {}
    for k in FAM:
        if k in va and k in vb:
            out[k] = round((vb[k] - va[k]) * scale, 2)
    return out


def classify(d):
    flags = []
    dnc, dttc, dc, dep, ddac = d.get("NC"), d.get("TTC"), d.get("C"), d.get("EP"), d.get("DAC")
    safety = [x for x in (dnc, dttc) if x is not None]
    if safety and min(safety) < -0.25:
        flags.append("TRADE-SAFETY")
    if dc is not None and dc < -1.0:
        flags.append("TRADE-COMFORT")
    if ddac is not None and ddac < -0.5:
        flags.append("DAC-DOWN")
    if dep is None:
        flags.append("EP-not-printed")
        return " ".join(flags)
    if dep < -0.5:
        flags.append("EP-DOWN")
    elif dep >= 0.5 and not any(f.startswith("TRADE") for f in flags):
        flags.insert(0, "NO-TRADE")
    elif dep < 0.5:
        flags.append("EP-FLAT")
    return " ".join(flags)


# 2 ---- RL / preference / hard-negative steps (same checkpoint, before -> after) -----------------------------------
# side = what the step UPDATED.  'proposal' = the proposal-generating policy / refiner / encoder; 'ranking' = only the
# function that ranks fixed proposals.  This column is the structural finding of REPORT.md section 3.
STEPS = [
    ("recog_pre", "recog_post", "ReCogDrive Diff-GRPO", "proposal", "RL: GRPO on the diffusion planner"),
    ("ddv2_dd", "ddv2_sel", "DiffusionDriveV2: selector swap only", "ranking", "NOT RL: replace DiffusionDrive's classifier by DDv2's mode selector"),
    ("ddv2_sel", "ddv2_rl", "DiffusionDriveV2: RL step, selector held", "proposal", "RL on the generator, selector matched"),
    ("cophy_a2", "cophy_b2", "CoPhy: refiner GRPO, physical reward only", "proposal", "RL: GRPO, reward weights 1/1/0.4/0.4/0.2"),
    ("cophy_a2", "cophy_b3", "CoPhy: refiner GRPO, cognitive + physical", "proposal", "RL: GRPO, dual reward"),
    ("pair_dd0", "pair_dd1", "PaIR-Drive on DiffusionDrive", "proposal", "RL branch refines the IL plan (parallel IL + RL)"),
    ("pair_tf0", "pair_tf1", "PaIR-Drive on Transfuser", "proposal", "same"),
    ("tsa_pre", "tsa_post", "TSA-MPR multi-objective DPO", "proposal", "DPO on single-deficit loser pairs, one deficit per pair"),
    ("drivedpo_wo", "drivedpo_full", "DriveDPO safety DPO", "proposal", "DPO on the policy distribution, imitation-based rejected-trajectory choice"),
    ("simvta_init", "simvta_final", "SimVTA GRPO + matched-video loss", "proposal", "GRPO + shared-encoder video loss + BC regulariser (3-seed mean)"),
    ("simvta_init", "simvta_exp0", "SimVTA trajectory-GRPO only (Exp0)", "proposal", "GRPO without the video loss (NC and PDMS printed only)"),
    ("beyond_mf0", "beyond_mf1", "BeyondDrive on MeanFuser", "proposal", "NOT RL: hard-negative repulsion on the generator"),
    ("beyond_wote0", "beyond_wote1", "BeyondDrive on WoTE", "proposal", "NOT RL: same, EP column"),
    ("cgrpo_sft_s1", "cgrpo_final_s1", "Constrained GRPO (navhard stage 1, NOT navtest)", "proposal", "RL: GRPO with scalarised standardised advantages; values x100"),
]
rows2 = []
for a, b, name, side, what in STEPS:
    d = delta(a, b)
    rows2.append([name, a, b, side, what] + [d.get(k, "") for k in FAM] + [classify(d)])
w("rl_step_deltas.csv", ["step", "before_id", "after_id", "updated", "what"] + ["d" + k for k in FAM] + ["verdict"], rows2)

# 3 ---- test-time search and scorer-only retraining (no policy RL) ----------------------------------------------------
PAIRS = [
    ("toad_ipad0", "toad_ipad1", "TOAD search on iPad", "test-time"),
    ("toad_hydra0", "toad_hydra1", "TOAD search on Hydra-MDP", "test-time"),
    ("toad_gtrs0", "toad_gtrs1", "TOAD search on GTRS (base row quoted via TOAD)", "test-time"),
    ("toad_ztrs0", "toad_ztrs1", "TOAD search on ZTRS (base row quoted via TOAD)", "test-time"),
    ("toad_rap0", "toad_rap1", "TOAD search on RAP-DINO", "test-time"),
    ("toad_drivor0", "toad_drivor1", "TOAD search on DrivoR", "test-time"),
    ("versatile_dd0", "versatile_dd1", "Versatile-samples scorer on DiffusionDrive (3-seed mean)", "scorer-only retrain"),
    ("versatile_mf0", "versatile_mf1", "Versatile-samples scorer on MeanFuser (3-seed mean)", "scorer-only retrain"),
]
rows3 = []
for a, b, name, kind in PAIRS:
    d = delta(a, b)
    rows3.append([name, a, b, kind] + [d.get(k, "") for k in FAM] + [classify(d)])
w("testtime_and_scorer_only_deltas.csv", ["step", "before_id", "after_id", "kind"] + ["d" + k for k in FAM] + ["verdict"], rows3)

# 4 ---- first-order gap arithmetic: P2-s0 (MEASURED, registry row 5524 at the tip) vs DriveZero Table A13 (PUBLISHED) -----
p2 = dict(NC=98.81, DAC=96.87, TTC=95.32, C=99.98, EP=82.34, PDMS=88.82)
dz = BY["dz_a13_dinov3L"]["values"]


def prod(v):
    return (v["NC"] / 100) * (v["DAC"] / 100) * (5 * v["EP"] + 5 * v["TTC"] + 2 * v["C"]) / 1200


base = prod(p2)
scale = p2["PDMS"] / 100 / base          # mean-of-products over product-of-means, P2's own
out = dict(
    note="ESTIMATED. Means-level linearisation PDMS ~ NC*DAC*(5EP+5TTC+2C)/12 on column means; the true PDMS is a mean of per-token products, "
         "so the single-term effects below are NOT additive and are not a decomposition of the measured gap. Ranking of terms only.",
    p2=p2, driveZero_A13=dz,
    gap_pdms=round(dz["PDMS"] - p2["PDMS"], 2),
    column_deficits={k: round(dz[k] - p2[k], 2) for k in ("NC", "DAC", "TTC", "C", "EP")},
    product_of_means_p2=round(base * 100, 2), mean_of_products_over_product_of_means_p2=round(scale, 4))
single = {}
for k in ("NC", "DAC", "TTC", "C", "EP"):
    v = dict(p2)
    v[k] = dz[k]
    single[k] = round((prod(v) - base) * scale * 100, 2)
out["single_term_pdms_if_that_column_alone_reached_DriveZero"] = single
out["sum_of_single_terms"] = round(sum(single.values()), 2)
with open(os.path.join(HERE, "gap_arithmetic.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)

# 5 ---- lever ranking heuristic (DISCLOSED, ESTIMATED; not a measurement) ------------------------------------------------
# key = mid-reach PDMS * evidence_weight / (1 + cost_h / 10).  Weights: PUBLISHED-controlled 1.0, PUBLISHED-transfer 0.7, ESTIMATED 0.5, HYPOTHESIS 0.3.
# cost_h = A40-hour equivalents (dev-box GPU hours counted 1:1; CPU-only counted 0).
LEVERS = [
    # id, name, reach_lo, reach_hi, cost_h, weight, evidence class
    ("L1", "Test-time CEM search on the scorer, admission test first (TOAD form)", 0.3, 1.5, 2.5, 0.7, "PUBLISHED (TOAD Tab. 5), transfer ESTIMATED"),
    ("L2", "SR-TTC extension (SR-TTC itself is registered, not re-proposed): EP-output contender pairs + designed negatives + EMA soft targets on SR-TTC's cached features", 0.1, 0.8, 1.0, 0.5, "ESTIMATED (policy-level DPO PUBLISHED; scorer-only analogue +0.2)"),
    ("L3", "Two-scorer sub-score ensemble on one fan (P2-s0 + P2-s1 logits)", 0.1, 0.4, 3.0, 0.3, "HYPOTHESIS (bounded by the measured seed floor +0.33 [+0.04, +0.64])"),
    ("L4", "RL that also updates the encoder under a representation/aux loss (not head-only EPO)", 0.0, 0.8, 15.0, 0.7, "PUBLISHED on one base (SimVTA Tab. 5), transfer HYPOTHESIS"),
    ("L5", "Alternating generator top-k / vector-Pareto self-distillation to clean the fan (CLOVER form)", 0.2, 1.0, 30.0, 0.7, "PUBLISHED (CLOVER Tab. 5-8), transfer ESTIMATED"),
    ("L6", "Confidence-gated verifier on the pick (DriveVer form)", 0.1, 0.5, 30.0, 0.5, "PUBLISHED on weaker bases, transfer ESTIMATED"),
    ("L7", "Proposal-anchored scorer features (iPad ProFormer form); needs a PI architecture decision", 0.3, 1.5, 60.0, 0.3, "PUBLISHED on ResNet-34 (iPad Tab. 4), transfer HYPOTHESIS"),
    ("L8", "DrivoR layer-wise scoring / refinement decoder", 0.1, 0.5, 36.0, 0.5, "PUBLISHED (DrivoR Tab. 6), transfer ESTIMATED"),
    ("L9", "DrivoR accelerated second target (full retrain)", 0.2, 0.6, 160.0, 0.7, "PUBLISHED (DrivoR Tab. 7; -1.6 on v2), transfer ESTIMATED"),
    ("L10", "SimScale / RAP-style synthetic data scale-up", 0.5, 1.0, 150.0, 0.7, "PUBLISHED (DrivoR Tab. 1, RAP Tab. 1), transfer ESTIMATED"),
]
rank = []
for i, n, lo, hi, c, wt, ev in LEVERS:
    mid = (lo + hi) / 2
    rank.append([i, n, lo, hi, mid, c, wt, round(mid * wt / (1 + c / 10), 4), ev])
rank.sort(key=lambda r: -r[7])
w("lever_ranking.csv", ["rank", "id", "lever", "reach_pdms_lo", "reach_pdms_hi", "reach_mid", "cost_a40h_equiv", "evidence_weight", "rank_key", "evidence_class"],
  [[k + 1] + r for k, r in enumerate(rank)])
cheap = [r for r in rank if r[0] in ("L1", "L2", "L3", "L4")]
print("gap", out["gap_pdms"], "single-term", single, "sum", out["sum_of_single_terms"])
print("rank order:", [r[0] for r in rank])
print("cheap stack L1-L4 reach: lo %.1f hi %.1f mid %.1f (NOT additive, overlapping)" % (
    sum(r[2] for r in cheap), sum(r[3] for r in cheap), sum(r[4] for r in cheap)))
for r in rows2:
    print(r[0], "|", r[5:13], "|", r[-1])
for r in rows3:
    print(r[0], "|", r[4:12], "|", r[-1])

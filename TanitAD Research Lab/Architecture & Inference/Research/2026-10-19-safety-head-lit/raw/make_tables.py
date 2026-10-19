#!/usr/bin/env python3
"""Derive the deltas, relative failure-rate reductions and the lever ranking used in REPORT.md.

Inputs
  published_subscores.csv   every row PDF-verified by verify_published_rows.py (PUBLISHED, banked primaries)
  MEASURED constants below  copied from the register / source at tip d568e539 with their artifact named (not re-run here)
Outputs
  published_deltas.csv      before/after pairs: delta NC, DAC, TTC, EP, PDMS (points) and relative reduction of (100 - sub-score)
  lever_ranking.csv         reach (ESTIMATED), cost, reach per cost, applicability under the probe outcomes
  gap_arithmetic.json       the headroom terms the reach is a fraction of

Run:  python make_tables.py
Every ESTIMATED number is a product of a PUBLISHED relative reduction (a range) and a MEASURED headroom term; change the
constants here and the CSV regenerates.  Nothing here is a measurement of REFe's levers.
"""
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- MEASURED headroom (x100 PDMS), register at tip d568e539
HEADROOM = {
    # D-REFE-P2-FULLTABLE: T1 NAVSIM v1.1 navtest, 12,146 tokens, P2's own table; oracle - pick = 9.41
    "oracle_minus_pick": 9.41,
    "non_zero_ranking": 5.88,
    "dac_zero_picks": 2.50,
    "nc_zero_picks": 1.01,
    "both_zero_picks": 0.02,
    # D-REFE-SELRULE-FIT (u400, RL-val selection tier): perfect-head selection ceilings
    "u400_O_DAC": 2.13, "u400_O_NC": 0.75, "u400_O_both": 2.68,
}
HEADROOM["safety_zero_picks"] = round(HEADROOM["dac_zero_picks"] + HEADROOM["nc_zero_picks"] + HEADROOM["both_zero_picks"], 2)
# D-REFE-ZEROTAIL-3: share of the u400 DAC-only selection zeros each lever can in principle reach (Y-share, MEASURED coverage)
ZT3_Y = {"P6_trunk_DAC_gradient": 0.58, "proposal_side": 0.42, "P7_1_map_supervision": 0.257, "P6_lane_rank_twins": 0.029}
# label cost, build_scorer_targets.py header (MEASURED 2026-09-20, dev box, single core)
SEC_PER_CAND_STRIDE2 = 0.444
S0_TOKENS = 8922          # D-REFE-HEADFIT-F10
CAND_PER_TOKEN_BANK = 24  # the fast-negative bank proposed in REPORT.md section 4 (ESTIMATED design choice)

# ---------------------------------------------------------------- published pairs: (id, key, table, after-label, before-label)
PAIRS = [
    ("hydra: sub-score heads vs overall-PDM-score only", "2406.06978", "Tab.1", "Hydra-MDP-V8192 (sub-score", "V8192-PDM"),
    ("hydra: inference weighting W", "2406.06978", "Tab.1", "V8192-W (weighted", "Hydra-MDP-V8192 (sub-score"),
    ("hydra: + EP head", "2406.06978", "Tab.1", "W-EP", "V8192-W (weighted"),
    ("hydra++: weighted confidence", "2503.12820", "Tab.3", "+ weighted confidence", "no weighted confidence"),
    ("hydra++: + perception aux tasks", "2503.12820", "Tab.3", "perception aux tasks", "W + temporal SE"),
    ("hydra++: ResNet34 -> V2-99", "2503.12820", "Tab.1", "V2-99", "ResNet34"),
    ("drivesuprim R34: recipe vs Hydra-MDP", "2506.06659", "Tab.2", "DriveSuprim ResNet34", "Hydra-MDP ResNet34"),
    ("drivesuprim ViT-L: recipe vs Hydra-MDP", "2506.06659", "Tab.2", "DriveSuprim ViT-L", "Hydra-MDP ViT-L"),
    ("drivesuprim v2: coarse-to-fine", "2506.06659", "Tab.5", "coarse-to-fine", "single-stage"),
    ("drivesuprim v2: + rotation aug", "2506.06659", "Tab.5", "rotation augmentation", "coarse-to-fine"),
    ("drivesuprim v2: + self-distillation", "2506.06659", "Tab.5", "self-distillation", "rotation augmentation"),
    ("wote: trajectory evaluation (no future)", "2504.01941", "Tab.3", "no future BEV states", "TransFuser baseline"),
    ("wote: + BEV future states", "2504.01941", "Tab.3", "world-model future states", "no future BEV states"),
    ("wote: sim-reward only vs imitation-reward only", "2504.01941", "Tab.4", "simulation reward only", "imitation reward only"),
    ("ipad: ProFormer vs BEVFormer(+refine)", "2505.15111", "Tab.4", "ProFormer (proposal-aware", "BEVFormer + proposal refinement"),
    ("ipad: + proposal-centric mapping", "2505.15111", "Tab.4", "proposal-centric mapping", "ProFormer (proposal-aware"),
    ("ipad: + proposal-centric prediction", "2505.15111", "Tab.4", "proposal-centric prediction", "proposal-centric mapping"),
    ("safedrive: ProposalNet (traj-guided deformable attn) vs DiffusionDrive head", "2602.18887", "Tab.4", "ProposalNet (trajectory", "DiffusionDrive head only"),
    ("safedrive: fine-grained vs scene-level safety (sparse world)", "2602.18887", "Tab.5", "scene-level + fine-grained", "sparse world + scene-level safety"),
    ("safedrive: PwNC only vs neither", "2602.18887", "Tab.6", "PwNC only", "neither"),
    ("safedrive: TwDAC only vs neither", "2602.18887", "Tab.6", "TwDAC only", "neither"),
    ("safedrive: both vs neither", "2602.18887", "Tab.6", "both", "neither"),
    ("da-wam: action-conditioned future vs none", "2608.19085", "Tab.3", "action-conditioned future, no hard", "no future prediction"),
    ("da-wam: + hard negatives", "2608.19085", "Tab.3", "+ hard negatives", "no hard negatives"),
    ("da-wam: hard negatives vs no future", "2608.19085", "Tab.3", "+ hard negatives", "no future prediction"),
    ("da-wam: shared global future vs none", "2608.19085", "Tab.3", "shared global future", "no future prediction"),
    ("mm-future: paired future vs history only", "2609.20377", "Tab.3", "history + paired", "history only"),
    ("flowr2a: wTTC 3 vs default", "2606.24231", "Tab.7", "(1,3,1)", "default"),
    ("flowr2a: wTTC 2 vs default", "2606.24231", "Tab.7", "(1,2,1)", "default"),
    ("ztrs: + progress reward", "2510.24108", "Tab.8", "V2-99 reward = safety + progress", "V2-99 reward = safety only"),
    ("ztrs: + comfort reward", "2510.24108", "Tab.8", "V2-99 reward = safety + progress + comfort (full)", "V2-99 reward = safety + progress"),
    ("toad: re-score iPad with DrivoR scorer", "2606.07170", "Tab.7", "re-scoring with DrivoR", "iPad"),
    ("toad: search iPad with GTRS scorer", "2606.07170", "Tab.7", "search with GTRS", "iPad"),
    ("toad: search iPad with SparseDriveV2 scorer", "2606.07170", "Tab.7", "search with SparseDriveV2", "iPad"),
    ("toad: TOAD (DrivoR scorer) on iPad", "2606.07170", "Tab.7", "TOAD (search", "iPad"),
    ("gtrs: 2-model ensemble vs best member (Aug)", "2506.06664", "Tab.2", "GTRS-E-Lite", "GTRS-Aug"),
    ("gtrs: 6-model ensemble vs best member (Aug)", "2506.06664", "Tab.2", "GTRS-E (6-model", "GTRS-Aug"),
]
# tables that print one metric per line (plain | a | b | both): handled as (metric rows)
LINE_PAIRS = [
    ("versatile DiffusionDrive: both designs vs plain", "2609.01799", "Tab.5", "DiffusionDrive", 0, 3),
    ("versatile DiffusionDrive: longitudinal-only vs plain", "2609.01799", "Tab.5", "DiffusionDrive", 0, 1),
    ("versatile DiffusionDrive: lateral-only vs plain", "2609.01799", "Tab.5", "DiffusionDrive", 0, 2),
    ("versatile MeanFuser: both designs vs plain", "2609.01799", "Tab.5", "MeanFuser", 0, 3),
    ("versatile MeanFuser: longitudinal-only vs plain", "2609.01799", "Tab.5", "MeanFuser", 0, 1),
    ("versatile MeanFuser: lateral-only vs plain", "2609.01799", "Tab.5", "MeanFuser", 0, 2),
]


def load():
    rows = []
    with open(os.path.join(HERE, "published_subscores.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["map"] = dict(zip(r["columns"].split(), r["printed_values"].split()))
            rows.append(r)
    return rows


def find(rows, key, tab, frag):
    ex = [r for r in rows if r["arxiv"] == key and r["table"] == tab and frag.lower() == r["row"].lower()]
    if len(ex) == 1:
        return ex[0]
    c = [r for r in rows if r["arxiv"] == key and r["table"] == tab and frag.lower() in r["row"].lower()]
    if len(c) != 1:
        raise SystemExit(f"ambiguous or missing row {key} {tab} {frag!r}: {len(c)}")
    return c[0]


def num(m, k):
    try:
        return float(m[k])
    except Exception:
        return None


def delta_row(name, a, b, page_a, page_b):
    out = {"pair": name, "pdf_page": page_a}
    for k in ("NC", "DAC", "TTC", "EP", "PDMS", "EPDMS"):
        va, vb = num(a, k), num(b, k)
        if va is None or vb is None:
            out[k] = ""
            out[k + "_relfail"] = ""
            continue
        out[k] = round(va - vb, 2)
        if k in ("NC", "DAC", "TTC") and (100 - vb) > 0:
            out[k + "_relfail"] = round(100 * ((100 - va) - (100 - vb)) / (100 - vb), 1)  # negative = fewer failures
        else:
            out[k + "_relfail"] = ""
    return out


def main():
    rows = load()
    out = []
    for name, key, tab, after, before in PAIRS:
        ra, rb = find(rows, key, tab, after), find(rows, key, tab, before)
        out.append(delta_row(name, ra["map"], rb["map"], ra["pdf_page"], rb["pdf_page"]))
    # one-metric-per-line tables
    for name, key, tab, planner, i_before, i_after in LINE_PAIRS:
        d = {"pair": name, "pdf_page": ""}
        for k in ("EPDMS", "NC", "DAC"):
            r = find(rows, key, tab, f"{planner} {k}:")
            v = r["printed_values"].split()
            d["pdf_page"] = r["pdf_page"]
            d[k] = round(float(v[i_after]) - float(v[i_before]), 2)
            if k in ("NC", "DAC"):
                d[k + "_relfail"] = round(100 * ((100 - float(v[i_after])) - (100 - float(v[i_before]))) / (100 - float(v[i_before])), 1)
        out.append(d)
    cols = ["pair", "pdf_page", "NC", "DAC", "TTC", "EP", "PDMS", "EPDMS", "NC_relfail", "DAC_relfail", "TTC_relfail"]
    with open(os.path.join(HERE, "published_deltas.csv"), "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for d in out:
            w.writerow(d)

    # --------------------------------------------------------------- lever ranking (ESTIMATED)
    safety_zero = HEADROOM["nc_zero_picks"] + HEADROOM["dac_zero_picks"]  # 3.51 (both-zero 0.02 left out)
    bank_cpu_h = S0_TOKENS * CAND_PER_TOKEN_BANK * SEC_PER_CAND_STRIDE2 / 3600.0
    W = {"cpu_h": 0.1, "devgpu_h": 1.0, "a40_h": 2.0}   # cost-equivalent weights: a CONVENTION, stated, not measured
    LEVERS = [
        # id, name, rel_low, rel_high, base term, cpu_h, devgpu_h, a40_h, evidence, CAP, FEAT_scorer_branch, FEAT_trunk, code
        ("SH1", "fast / lead-ward speed-profile negative bank, PDM-labelled, then head or scorer-branch refit on the cached visual_ctx",
         0.03, 0.21, safety_zero, bank_cpu_h, 5.0, 0.0, "PUBLISHED (Versatile T5 p8, DA-WAM T3 p10, CLOVER p16, W4S p5) + MEASURED context; transfer ESTIMATED; cost = one more arm in the registered HN screen (8-12 dev-box GPU-h for arms A+B x 3 seeds, H-REFE-HNSCREEN-1)",
         "yes", "yes", "partial", "refe/build_scorer_targets.py:144 candidate_set; refe/hnscreen_layer.py (bank is lateral-only, PREREG_P6HN_SCREEN s2.2); refe/ctx_cache.py"),
        ("SH2", "nonlinear rows: per-row MLP replacing score_head Linear(256,7) (+ per-step aux outputs)",
         0.0, 0.10, safety_zero, 0.5, 0.3, 0.0, "HYPOTHESIS (every published scorer uses MLP heads, none ablates linear vs MLP)",
         "yes", "no", "no", "refe/model.py:563 score_head (1,799 params); G-DVB entry needed"),
        ("SH6", "sub-score ensemble of two scorers (P2-s0 + P2-s1 or a different recipe)",
         0.05, 0.15, HEADROOM["dac_zero_picks"], 0.0, 3.0, 0.0, "PUBLISHED (GTRS T2 p4: DAC 95.8/98.0 -> 98.9/99.3) ; seed-only transfer HYPOTHESIS",
         "partial", "partial", "no", "planner.py:537 aggregate (average logits/probs per row before the rule); model.py:699 score_trajectories x2"),
        ("SH3", "scorer-branch retrain on cached visual_ctx: per-step motion + tracked-state features in the trajectory token, per-step NC/DAC/TTC outputs, continuous TTC label",
         0.08, 0.20, safety_zero, 20.0, 10.0, 0.0, "PUBLISHED (SafeDrive T6 p7, ZTRS p16, FlowR2A p20) ; transfer HYPOTHESIS",
         "no", "yes", "no", "refe/model.py:560 score_q_mlp, :552 score_dec, :699; refe/scorer_finetune.py; refe/ctx_cache.py"),
        ("SH4", "proposal-anchored sampling of the camera grids along the projected proposal (deformable / ROI) inside score_dec",
         0.10, 0.25, safety_zero, 0.0, 16.0, 0.0, "PUBLISHED on weaker bases (iPad T4 p8, SparseDriveV2 T6 p15, SafeDrive T4 p7) ; transfer HYPOTHESIS",
         "no", "yes", "no", "refe/model.py:552-553 CrossBlock cross-attn over visual_ctx; _frustum/_pos3d :579-697 for the projection"),
        ("SH5", "map-geometry factor at the pick: aux drivable field sampled at box key points, multiplied into p_DAC (H-REFE-AUXDRV-1 H2 + a factor variant)",
         0.5 * ZT3_Y["P7_1_map_supervision"], ZT3_Y["P7_1_map_supervision"], HEADROOM["dac_zero_picks"], 2.0, 0.0, 14.0, "PUBLISHED (SafeDrive Eq.4 p5, T6 p7) + MEASURED reach (ZEROTAIL-3: P7.1 Y-share 25.7 %); cost = H-REFE-AUXDRV-1 S1 (12-16 A40-h, Amendment A3)",
         "no", "yes", "no", "refe/aux_drivable_head.py; scorer_finetune.py --aux-drv-feed; planner.py:537 aggregate"),
        ("SH7", "per-candidate predicted future state conditioning the score (WoTE / DA-WAM / MM-Future form)",
         0.08, 0.15, safety_zero, 0.0, 0.0, 60.0, "PUBLISHED (WoTE T3 p7, DA-WAM T3 p10, MM-Future T3 p6) ; EP -0.2 to -0.9 side effect",
         "no", "partial", "yes", "new predictor module; invalidates ctx_cache if the trunk moves"),
        ("SH8", "trunk / P6 route (DAC-gradient into the trunk)",
         None, None, None, 0.0, 0.0, 169.0, "MEASURED coverage (ZEROTAIL-3: P6 trunk reaches Y 58 %, Y-PDMS 1.48 [1.04, 1.95]) ; effect UNMEASURED",
         "no", "no", "yes", "held by P6 (A40, PI-gated); invalidates ctx_cache"),
    ]
    lev_out = []
    for (lid, name, lo, hi, base, cpu, dev, a40, ev, cap, fs, ft, code) in LEVERS:
        if lo is None:
            rl, rh, rm = 1.04 * 0.0 + 0.0, 0.0, 0.0
            reach = ("n/a", "n/a", "n/a")
            r_lo = r_hi = r_mid = None
        else:
            r_lo, r_hi = lo * base, hi * base
            r_mid = 0.5 * (r_lo + r_hi)
        cost = cpu * W["cpu_h"] + dev * W["devgpu_h"] + a40 * W["a40_h"]
        lev_out.append({
            "id": lid, "lever": name,
            "reach_pdms_x100_low": "" if r_lo is None else round(r_lo, 2),
            "reach_pdms_x100_high": "" if r_hi is None else round(r_hi, 2),
            "reach_mid": "" if r_mid is None else round(r_mid, 2),
            "cpu_h": round(cpu, 1), "devbox_gpu_h": dev, "a40_h": a40, "cost_equiv": round(cost, 2),
            "reach_per_cost": "" if (r_mid is None or cost == 0) else round(r_mid / cost, 3),
            "evidence_class": ev,
            "applies_if_CAPACITY_LIMITED": cap, "applies_if_FEATURE_LIMITED_scorer_branch": fs, "applies_if_FEATURE_LIMITED_trunk": ft,
            "code_touched": code,
        })
    def rank_for(col):
        app = [d for d in lev_out if d[col] in ("yes", "partial") and d["reach_per_cost"] != ""]
        app.sort(key=lambda d: -d["reach_per_cost"])
        for i, d in enumerate(app, 1):
            d.setdefault("rank_" + col, "")
            d["rank_" + col] = f"{i}" + ("*" if d[col] == "partial" else "")
    for col in ("applies_if_CAPACITY_LIMITED", "applies_if_FEATURE_LIMITED_scorer_branch", "applies_if_FEATURE_LIMITED_trunk"):
        for d in lev_out:
            d["rank_" + col] = ""
        rank_for(col)
    have = [d for d in lev_out if d["reach_pdms_x100_low"] != ""]
    for tag, fld in (("rank_all_at_low_reach", "reach_pdms_x100_low"), ("rank_all_at_high_reach", "reach_pdms_x100_high")):
        for d in lev_out:
            d[tag] = ""
        for i, d in enumerate(sorted(have, key=lambda d: -(d[fld] / d["cost_equiv"])), 1):
            d[tag] = i
    lev_out.sort(key=lambda d: (-(d["reach_per_cost"] if d["reach_per_cost"] != "" else -1)))
    with open(os.path.join(HERE, "lever_ranking.csv"), "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=list(lev_out[0].keys()), lineterminator="\n")
        w.writeheader()
        for d in lev_out:
            w.writerow(d)
    ga = {"HEADROOM_MEASURED": HEADROOM, "ZT3_Y_MEASURED": ZT3_Y, "safety_zero_term_used": round(safety_zero, 2),
          "bank_cpu_hours_estimated": round(bank_cpu_h, 1),
          "bank_formula": f"{S0_TOKENS} tokens x {CAND_PER_TOKEN_BANK} candidates x {SEC_PER_CAND_STRIDE2} s / 3600",
          "cost_weights_convention": W,
          "reach_formula": "reach = [rel_low, rel_high] x safety_zero (3.51 = NC-zero 1.01 + DAC-zero 2.50 picks, P2 full navtest), rel_* = a PUBLISHED relative reduction of the sub-score failure rate (published_deltas.csv *_relfail), discounted for base strength where stated in REPORT.md"}
    json.dump(ga, open(os.path.join(HERE, "gap_arithmetic.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print(open(os.path.join(HERE, "published_deltas.csv"), encoding="utf-8").read())
    print(open(os.path.join(HERE, "lever_ranking.csv"), encoding="utf-8").read())


if __name__ == "__main__":
    main()

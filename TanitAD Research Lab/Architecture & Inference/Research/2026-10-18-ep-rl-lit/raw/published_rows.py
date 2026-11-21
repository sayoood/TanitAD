# -*- coding: utf-8 -*-
"""Published NAVSIM numbers used by REPORT.md, one record per table row, with the banked PDF page.

Evidence class of every record: PUBLISHED (primary PDF held in TanitAD Research Lab/Library/papers/, sha256 in library.json).
Tier: NAVSIM v1.1 navtest pseudo-simulation (T1) unless 'bench' says otherwise. Every number is the paper's own, not ours.
`tokens` = the numbers exactly as printed in the row, in print order. verify_published_rows.py re-opens the PDF page and
requires the tokens to appear in that order. A row marked check='manual' could not be matched by text extraction (layout)
and is listed with the reason; it is still a reading of the PDF, but not machine re-checked.
pdf_page = 1-based PDF page index (not the journal page number).
"""
V1 = "NAVSIM v1.1 navtest PDMS (T1)"
V2 = "NAVSIM v2 navtest EPDMS (T1-like, not PDMS)"
NH = "NAVSIM v2 navhard stage 1 EPDMS (NOT navtest)"
R = []


def add(id, paper, key, table, page, row, bench, order, tokens, check="auto", note=""):
    vals = dict(zip(order, [float(t) for t in tokens[:len(order)]]))
    R.append(dict(id=id, paper=paper, key=key, table=table, pdf_page=page, row=row, bench=bench,
                  order=order, values=vals, tokens=tokens, check=check, note=note))


O6 = ["NC", "DAC", "TTC", "C", "EP", "PDMS"]       # printed order NC DAC TTC C EP PDMS
O6b = ["NC", "DAC", "EP", "TTC", "C", "PDMS"]      # printed order NC DAC EP TTC C PDMS
O7 = ["NC", "DAC", "TTC", "C", "EP", "DDC", "PDMS"]
O10 = ["NC", "DAC", "DDC", "TL", "EP", "TTC", "LK", "HC", "EC", "EPDMS"]

# ---- DriveZero (the comparison target) ----
add("dz_a13_dinov3L", "DriveZero", "2609.06055", "A13", 31, "DINOv3 ViT-L, 4 cams, navtrain, no SimScale (THE 94.55 TARGET)", V1, O7,
    ["98.80", "99.25", "95.74", "99.94", "92.72", "95.87", "94.55"])
add("dz_a13_vfmL", "DriveZero", "2609.06055", "A13", 31, "DriveVFM ViT-L", V1, O7, ["99.00", "99.24", "95.97", "99.99", "93.10", "95.89", "94.83"])
add("dz_a13_vfmS", "DriveZero", "2609.06055", "A13", 31, "DriveVFM ViT-S", V1, O7, ["99.17", "99.14", "96.29", "99.98", "91.79", "95.97", "94.41"])
add("dz_t7_human", "DriveZero", "2609.06055", "7", 13, "ViT-S, supervision = human trajectories", V1, O6b, ["98.76", "98.95", "91.99", "95.30", "99.98", "93.92"])
add("dz_t7_driverl", "DriveZero", "2609.06055", "7", 13, "ViT-S, supervision = DriveRL teacher trajectories", V1, O6b, ["98.60", "98.81", "91.64", "95.27", "99.94", "93.61"])
add("dz_t7_goalaug", "DriveZero", "2609.06055", "7", 13, "ViT-S, DriveRL trajectories + goal augmentation", V1, O6b, ["99.17", "99.14", "91.79", "96.29", "99.98", "94.41"])
add("dz_t4_human", "DriveZero", "2609.06055", "4", 11, "Human driver", V1, O6, ["100", "100", "100", "99.9", "87.5", "94.8"], "manual",
    "Table 4 rows are split across lines by the extraction; row read from the page text")
add("dz_t4_drivor_scale", "DriveZero", "2609.06055", "4", 11, "DrivoR-Scale (as quoted by DriveZero)", V1, O6, ["99.1", "99.2", "96.9", "100", "91.6", "94.6"], "manual",
    "split rows; the same numbers are machine-checked in DrivoR Table 1 (drivor_sim134)")
add("dz_t4_dz", "DriveZero", "2609.06055", "4", 11, "DriveZero (camera, ViT-L, no SimScale)", V1, O6, ["99.0", "99.2", "96.0", "100", "93.1", "94.8"], "manual",
    "split rows; EP 93.1 agrees with Table A13 DriveVFM ViT-L (93.10)")
add("dz_t4_dz_scale", "DriveZero", "2609.06055", "4", 11, "DriveZero-Scale", V1, O6, ["99.2", "99.4", "97.3", "99.9", "92.5", "95.3"], "manual", "split rows")
# ---- DrivoR ----
add("drivor_train", "DrivoR", "2601.05083", "1", 5, "DrivoR (navtrain)", V1, O6, ["98.9", "98.3", "96.2", "100", "89.1", "93.1"])
add("drivor_trainval", "DrivoR", "2601.05083", "1", 5, "DrivoR (navtrain+navval)", V1, O6, ["99.0", "98.9", "96.7", "100", "90.0", "93.7"])
add("drivor_sim65", "DrivoR", "2601.05083", "1", 5, "DrivoR (+65k SimScale)", V1, O6, ["99.1", "99.0", "96.9", "100", "90.3", "94.0"])
add("drivor_sim134", "DrivoR", "2601.05083", "1", 5, "DrivoR (+134k SimScale)", V1, O6, ["99.1", "99.2", "96.9", "100", "91.6", "94.6"])
# ---- Hydra-MDP / ++ / DriveSuprim ----
add("hydra_w", "Hydra-MDP", "2406.06978", "1", 4, "Hydra-MDP-V8192 (no weighted confidence)", V1, O6b, ["97.9", "91.7", "77.6", "92.9", "100", "83.0"])
add("hydra_W", "Hydra-MDP", "2406.06978", "1", 4, "Hydra-MDP-V8192 -W (weighted confidence at inference)", V1, O6b, ["98.1", "96.1", "77.8", "93.9", "100", "85.7"])
add("hydra_WEP", "Hydra-MDP", "2406.06978", "1", 4, "Hydra-MDP-V8192 -W-EP (+ continuous EP head)", V1, O6b, ["98.3", "96.0", "78.7", "94.6", "100", "86.5"])
add("hydra_PDM", "Hydra-MDP", "2406.06978", "1", 4, "Hydra-MDP-V8192 -PDM (distil the aggregate PDM score only)", V1, O6b, ["97.5", "88.9", "74.8", "92.5", "100", "80.2"])
add("hydra_A_vitl", "Hydra-MDP", "2406.06978", "2", 4, "Hydra-MDP-A (ViT-L, Depth Anything init)", V1, O6b, ["98.4", "97.7", "85.0", "94.5", "100", "89.9"])
add("hydrapp_r34", "Hydra-MDP++", "2503.12820", "1", 7, "Hydra-MDP++ ResNet34", V1, O6b, ["97.6", "96.0", "80.4", "93.1", "100", "86.6"])
add("hydrapp_v299", "Hydra-MDP++", "2503.12820", "1", 7, "Hydra-MDP++ V2-99", V1, O6b, ["98.6", "98.6", "85.7", "95.1", "100", "91.0"])
add("hydrapp_abl_noW", "Hydra-MDP++", "2503.12820", "3", 7, "ablation: no weighted confidence (ResNet34)", V1, O6b, ["97.5", "92.0", "80.4", "91.8", "100", "85.0"])
add("hydrapp_abl_W", "Hydra-MDP++", "2503.12820", "3", 7, "ablation: + weighted confidence", V1, O6b, ["97.4", "96.0", "81.0", "92.8", "100", "86.5"])
add("suprim_r34_base", "DriveSuprim", "2506.06659", "2", 6, "Hydra-MDP ResNet34 (their baseline row)", V1, O6b, ["98.3", "96.0", "78.7", "94.6", "100", "86.5"])
add("suprim_r34", "DriveSuprim", "2506.06659", "2", 6, "DriveSuprim ResNet34", V1, O6b, ["97.8", "97.3", "86.7", "93.6", "100", "89.9"])
add("suprim_vitl_base", "DriveSuprim", "2506.06659", "2", 6, "Hydra-MDP ViT-L (their baseline row)", V1, O6b, ["98.4", "97.7", "85.0", "94.5", "100", "89.9"])
add("suprim_vitl", "DriveSuprim", "2506.06659", "2", 6, "DriveSuprim ViT-L", V1, O6b, ["98.6", "98.6", "91.3", "95.5", "100", "93.5"])
add("suprim_inf_coef", "DriveSuprim", "2506.06659", "8", 8, "inference coefficients, NAVSIM v1: imitation, NC, DAC, EP, TTC, C (alpha_avg = 8.0, App. A eq. 13)", V1,
    ["imi", "NC", "DAC", "EP", "TTC", "C"], ["0.05", "0.5", "0.5", "5.0", "5.0", "2.0"], "auto", "the tuning set for these coefficients is not stated in the paper")
add("suprim_abl0", "DriveSuprim", "2506.06659", "5", 7, "v2 ablation: baseline single-stage", V2, O10, ["97.2", "97.5", "99.4", "99.6", "83.1", "96.5", "94.4", "98.2", "70.9", "81.4"])
add("suprim_abl1", "DriveSuprim", "2506.06659", "5", 7, "v2 ablation: + multi-stage (check-mark columns lost; row identified by Table 6's final 82.4)", V2, O10,
    ["97.5", "96.4", "99.1", "99.6", "87.0", "96.4", "95.3", "98.2", "75.0", "82.4"])
add("suprim_abl2", "DriveSuprim", "2506.06659", "5", 7, "v2 ablation: + rotation augmentation data (reconstructed, cumulative)", V2, O10,
    ["96.9", "96.9", "99.4", "99.6", "87.9", "96.0", "95.5", "98.3", "76.5", "82.7"])
add("suprim_abl3", "DriveSuprim", "2506.06659", "5", 7, "v2 ablation: + self-distillation = full model (reconstructed, cumulative)", V2, O10,
    ["97.5", "96.5", "99.4", "99.6", "88.4", "96.6", "95.5", "98.3", "77.0", "83.1"])
# ---- iPad / SparseDriveV2 / FlowR2A / RAP / WoTE / CLOVER ----
add("ipad", "iPad", "2505.15111", "1", 7, "iPad ResNet34 camera-only", V1, O6, ["98.6", "98.3", "94.9", "100", "88.0", "91.7"])
add("ipad_abl_prof", "iPad", "2505.15111", "4", 8, "ablation: + ProFormer proposal-anchored encoder (NC DAC TTC EP)", V1, ["NC", "DAC", "TTC", "EP"], ["98.1", "96.5", "94.3", "84.2"])
add("ipad_abl_noprof", "iPad", "2505.15111", "4", 8, "ablation: refinement + BEVFormer, no proposal anchoring (NC DAC TTC EP)", V1, ["NC", "DAC", "TTC", "EP"], ["96.9", "93.2", "90.8", "71.5"])
add("sdv2", "SparseDriveV2", "2603.29163", "2", 11, "SparseDriveV2 ResNet34 (1024 path x 256 velocity anchors)", V1, O6, ["98.5", "98.4", "95.0", "99.9", "88.6", "92.0"])
add("flowr2a", "FlowR2A", "2606.24231", "1", 6, "FlowR2A ResNet34, 60 proposals", V1, O6, ["98.8", "98.0", "96.0", "100", "90.1", "92.8"])
add("rap_dino", "RAP", "2510.04333", "1", 6, "RAP-DINO", V1, O6, ["99.1", "98.9", "96.7", "100", "90.3", "93.8"])
add("rap_ipad", "RAP", "2510.04333", "1", 6, "RAP-iPad", V1, O6, ["98.2", "98.6", "94.6", "100", "90.1", "92.5"])
add("wote", "WoTE", "2504.01941", "1", 6, "WoTE (contrast row: EP not high)", V1, O6b, ["98.5", "96.8", "81.9", "94.9", "99.9", "88.3"])
add("clover", "CLOVER", "2605.15120", "1", 9, "CLOVER", V1, O6, ["99.1", "99.0", "96.9", "100", "91.7", "94.5"])
add("clover_drivor", "CLOVER", "2605.15120", "1", 9, "DrivoR (as quoted by CLOVER)", V1, O6, ["99.0", "98.9", "96.7", "100", "90.0", "93.7"])
# ---- test-time search and scorer-only retraining ----
add("toad_ipad0", "TOAD", "2606.07170", "5", 14, "iPad", V1, O6, ["98.6", "98.3", "94.9", "100", "88.0", "91.7"])
add("toad_ipad1", "TOAD", "2606.07170", "5", 14, "iPad + TOAD", V1, O6, ["99.0", "99.3", "96.4", "100", "89.1", "93.4"])
add("toad_hydra0", "TOAD", "2606.07170", "5", 14, "Hydra-MDP", V1, O6, ["98.5", "98.7", "94.9", "100", "85.5", "90.9"])
add("toad_hydra1", "TOAD", "2606.07170", "5", 14, "Hydra-MDP + TOAD", V1, O6, ["98.7", "98.0", "95.4", "100", "87.2", "91.4"])
add("toad_gtrs0", "TOAD", "2606.07170", "5", 14, "GTRS (V2-99) [GTRS's navtest v1 row is quoted here, not read in GTRS's own PDF]", V1, O6, ["99.0", "98.9", "95.9", "100", "82.8", "90.4"])
add("toad_gtrs1", "TOAD", "2606.07170", "5", 14, "GTRS + TOAD", V1, O6, ["98.9", "98.1", "96.3", "100", "84.8", "90.9"])
add("toad_ztrs0", "TOAD", "2606.07170", "5", 14, "ZTRS (V2-99) [quoted here]", V1, O6, ["98.8", "99.5", "95.1", "100", "74.7", "86.9"])
add("toad_ztrs1", "TOAD", "2606.07170", "5", 14, "ZTRS + TOAD", V1, O6, ["98.4", "98.6", "94.3", "100", "82.1", "89.0"])
add("toad_rap0", "TOAD", "2606.07170", "5", 14, "RAP-DINO", V1, O6, ["99.1", "98.9", "96.7", "100", "90.3", "93.8"])
add("toad_rap1", "TOAD", "2606.07170", "5", 14, "RAP-DINO + TOAD", V1, O6, ["99.0", "99.2", "96.6", "100", "90.1", "93.9"])
add("toad_drivor0", "TOAD", "2606.07170", "5", 14, "DrivoR", V1, O6, ["99.1", "99.2", "96.9", "100", "91.6", "94.6"])
add("toad_drivor1", "TOAD", "2606.07170", "5", 14, "DrivoR + TOAD", V1, O6, ["99.0", "99.3", "96.8", "100", "91.8", "94.7"])
add("versatile_dd0", "Versatile samples", "2609.01799", "1", 7, "DiffusionDrive + plain scorer (3-seed mean)", V1, O6, ["98.4", "97.5", "94.2", "100", "84.1", "89.3"])
add("versatile_dd1", "Versatile samples", "2609.01799", "1", 7, "DiffusionDrive + augmented scorer (3-seed mean)", V1, O6, ["98.6", "97.7", "94.3", "100", "84.2", "89.5"])
add("versatile_mf0", "Versatile samples", "2609.01799", "1", 7, "MeanFuser + plain scorer (3-seed mean)", V1, O6, ["98.2", "98.0", "93.8", "100", "84.8", "89.7"])
add("versatile_mf1", "Versatile samples", "2609.01799", "1", 7, "MeanFuser + augmented scorer (3-seed mean)", V1, O6, ["98.3", "98.3", "93.7", "100", "85.1", "89.9"])
# ---- RL / preference steps ----
add("recog_pre", "ReCogDrive", "2506.08052", "3", 9, "IL planner (row 3, before Diff-GRPO)", V1, O6, ["98.1", "94.7", "94.2", "100", "80.9", "86.5"], "manual",
    "extraction prints the table columns in blocks (NC DAC TTC Conf EP PDMS per row); rows 3 and 4 read from the block")
add("recog_post", "ReCogDrive", "2506.08052", "3", 9, "+ Diff-GRPO (row 4)", V1, O6, ["97.9", "97.3", "94.9", "100.0", "87.3", "90.8"], "manual", "as above")
add("ddv2_dd", "DiffusionDriveV2", "2512.07745", "9", 13, "DiffusionDrive, its own classifier", V1, O6, ["98.2", "96.2", "94.7", "100", "82.2", "88.1"])
add("ddv2_sel", "DiffusionDriveV2", "2512.07745", "9", 13, "DiffusionDrive + DDv2's mode selector", V1, O6, ["97.2", "97.0", "92.2", "100", "86.8", "89.1"])
add("ddv2_rl", "DiffusionDriveV2", "2512.07745", "9", 13, "DiffusionDriveV2 (RL, same selector)", V1, O6, ["98.3", "97.9", "94.8", "99.9", "87.5", "91.2"])
add("cophy_a2", "CoPhy", "2605.21139", "3", 9, "A2: IL + physical score + VLM distillation + cognitive score (before RL)", V1, O6b, ["98.3", "97.2", "83.2", "94.5", "100", "88.9"], "manual",
    "row label A2 and columns read from Table 3; the extraction separates the labels from the numbers")
add("cophy_b2", "CoPhy", "2605.21139", "3", 9, "B2: + physical-reward RL only", V1, O6b, ["98.9", "97.8", "85.0", "95.8", "100", "90.7"], "manual", "as above")
add("cophy_b3", "CoPhy", "2605.21139", "3", 9, "B3: + cognitive and physical reward RL (full)", V1, O6b, ["99.0", "98.2", "85.3", "96.8", "100", "91.4"], "manual", "as above")
add("pair_dd0", "PaIR-Drive", "2603.13842", "3", 7, "DiffusionDrive (IL)", V1, O6b, ["98.2", "96.2", "82.2", "94.7", "100.0", "88.1"], "manual",
    "the +delta rows are printed with labels offset by one line in the extraction; assignment follows the printed order and the paper's text (+3.1 PDMS)")
add("pair_dd1", "PaIR-Drive", "2603.13842", "3", 7, "DiffusionDrive w/ PaIR-Drive", V1, O6b, ["99.1", "97.6", "88.3", "98.5", "94.1", "91.2"], "manual", "as above")
add("pair_tf0", "PaIR-Drive", "2603.13842", "3", 7, "Transfuser (IL)", V1, O6b, ["97.7", "92.8", "79.2", "92.8", "100.0", "84.0"], "manual", "as above")
add("pair_tf1", "PaIR-Drive", "2603.13842", "3", 7, "Transfuser w/ PaIR-Drive", V1, O6b, ["99.1", "96.1", "88.1", "98.2", "93.1", "89.7"], "manual", "as above")
add("tsa_pre", "TSA-MPR", "2509.20938", "III", 5, "TISA model before multi-objective DPO", V1, ["NC", "DAC", "TTC", "EP", "PDMS"], ["98.0", "95.5", "93.8", "81.1", "86.8"])
add("tsa_post", "TSA-MPR", "2509.20938", "III", 5, "+ multi-objective DPO (MOPT)", V1, ["NC", "DAC", "TTC", "EP", "PDMS"], ["98.4", "97.6", "94.7", "84.8", "89.8"])
add("simvta_init", "SimVTA", "2610.13781", "1", 5, "Epona init (NAVSIM-adapted), the RL start", V1, O6, ["97.8", "95.0", "93.9", "99.9", "80.3", "86.2"], "manual",
    "labels are offset in the extraction; EP 80.3, TTC 93.9, DAC 95.0, PDMS 86.2 are also stated in the text on p7")
add("simvta_final", "SimVTA", "2610.13781", "1", 5, "SimVTA (GRPO + matched-video loss, 3-seed mean)", V1, O6, ["98.0", "97.2", "97.0", "100.0", "86.7", "91.3"])
add("simvta_exp0", "SimVTA", "2610.13781", "5", 8, "Exp0: trajectory GRPO only (NC, PDMS printed)", V1, ["NC", "PDMS"], ["97.1", "89.1"])
add("simvta_exp2", "SimVTA", "2610.13781", "5", 8, "Exp2: full (NC, PDMS printed)", V1, ["NC", "PDMS"], ["98.0", "91.3"])
add("beyond_mf0", "BeyondDrive", "2605.19771", "4", 12, "MeanFuser baseline", V1, O6b, ["98.6", "97.0", "82.8", "95.0", "100", "89.0"])
add("beyond_mf1", "BeyondDrive", "2605.19771", "4", 12, "MeanFuser + BeyondDrive (hard-negative repulsion)", V1, O6b, ["98.5", "98.4", "84.2", "95.1", "100", "90.3"])
add("beyond_wote0", "BeyondDrive", "2605.19771", "4", 12, "WoTE baseline", V1, ["NC", "DAC", "EP", "TTC", "C"], ["98.5", "96.8", "81.9", "94.4", "99.9"])
add("beyond_wote1", "BeyondDrive", "2605.19771", "4", 12, "WoTE + BeyondDrive", V1, ["NC", "DAC", "EP", "TTC", "C"], ["98.4", "97.3", "83.1", "95.1", "100"])
add("curious_final", "Curious-VLA", "2603.06049", "1", 7, "Curious-VLA after RL (single sample)", V1, O6b, ["98.4", "96.9", "88.5", "97.9", "98.1", "90.3"])
add("drivedpo_wo", "DriveDPO", "2509.17940", "1", 7, "DriveDPO w/o DPO (unified policy distillation only; also Table 4 p9)", V1, O6b, ["97.9", "97.3", "84.0", "93.6", "100.0", "88.8"])
add("drivedpo_full", "DriveDPO", "2509.17940", "1", 7, "DriveDPO full (safety DPO, imitation-based rejected-trajectory selection)", V1, O6b, ["98.5", "98.1", "84.3", "94.8", "99.9", "90.0"])
add("cgrpo_sft_s1", "Constrained GRPO", "2602.05863", "1", 8, "navhard stage 1, SFT checkpoint (Qwen2.5-VL-3B); fractions in [0,1]", NH,
    ["NC", "DAC", "DDC", "TLC", "EP", "HC", "EPDMS"], ["0.953", "0.789", "0.978", "0.993", "0.836", "0.976", "0.651"])
add("cgrpo_final_s1", "Constrained GRPO", "2602.05863", "1", 8, "navhard stage 1, Constrained GRPO ScAdv (ours); fractions in [0,1]", NH,
    ["NC", "DAC", "DDC", "TLC", "EP", "HC", "EPDMS"], ["0.957", "0.928", "0.989", "0.998", "0.796", "0.977", "0.774"])
add("drivever_dd", "DriveVer", "2607.00399", "1", 6, "DiffusionDrive with DriveVer (NC DAC TTC C EP; PDMS in a separate block)", V1, ["NC", "DAC", "TTC", "C", "EP"],
    ["98.3", "96.8", "94.2", "100", "84.1"], "manual", "PDMS printed in a column the extraction separates; the text gives +0.9 for DiffusionDrive")

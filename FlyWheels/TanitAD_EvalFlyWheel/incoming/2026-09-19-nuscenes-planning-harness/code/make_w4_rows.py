"""Build the W6 -> W4 nuScenes rows in W4's published_results.json row schema.

Self-contained and re-runnable: it resolves every row's library key through
`TanitAD Research Lab/Library/library.json`, RE-HASHES the banked PDF (sha256 must equal the
library's), extracts the cited page with pypdf and asserts every `page_tokens` string occurs in
that page's text.

⛔ THIS IS NOT AN INDEPENDENT CHECK, AND AN EARLIER VERSION OF THIS FILE SAID IT WAS.
The tokens were CHOSEN by reading the page with pypdf and are re-found with pypdf: the producer's
own derivation, re-run. That measures transcription and determinism - a typo'd token, a wrong page
- never whether the number is the one the table prints. W4's `verify_published_against_pdfs.py`
(PyMuPDF, not installed in this venv) is the admissible check, and **0 of these 39 rows share a
(paper, page, system) cell with any row W4 read independently**, so no value here is double-read.
See `code/recount_row_overlap.py` for the counts and the retraction.

    python make_w4_rows.py [--out <path>]
"""
import hashlib
import json
import os
import sys

REPO = "D:/Projects/TanitAD"
LIB = os.path.join(REPO, "TanitAD Research Lab/Library/library.json")
OUT = os.path.join(REPO, "FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                         "2026-09-19-nuscenes-planning-harness/code/"
                         "published_results_nuscenes.patch.json")
if "--out" in sys.argv:
    OUT = sys.argv[sys.argv.index("--out") + 1]

UNIAD_IMPL = "UniAD planning_metrics (at-timestep)"
STP3_IMPL = "ST-P3 metric code as re-used by VAD / AD-MLP (0.5 m collision grid)"
BEVP_IMPL = "BEV-Planner union re-implementation (0.1 m collision grid, 2312.03031)"
SD_IMPL = "SparseDrive re-implemented collision (box-vs-box, estimated yaw; L2 per VAD)"
PD_IMPL = "PARA-Drive standardized (oriented ego box, 0.1 m grid, first frame removed, + map compliance)"
U, S = "nuScenes_OL_L2_uniad", "nuScenes_OL_L2_stp3"

CMD_NOTE = ("GT-derived: the high-level command is the GT future's final lateral offset thresholded "
            "at +-2 m (VAD vad_nuscenes_converter.py:452-457, UniAD trajectory_api.py:272-280, "
            "ST-P3 NuscenesData.py:525-530, SparseDrive models/motion/decoder.py:167-173 - "
            "MEASURED-from-source at the pinned commits)")


def row(rid, protocol, impl, system, role, vals, key, table, page, tokens, basis, modality,
        ego, command, admissible=True, notes=None, reason=None, evidence=None, extra=None):
    r = {"id": rid, "protocol": protocol, "implementation": impl, "system": system, "role": role,
         "values": vals, "source": {"library_key": key, "table": table, "page": page},
         "page_tokens": tokens,
         "harness": {"fix151": "n/a", "basis": basis},
         "modality": modality, "ego_status": ego, "command": command,
         "comparison_admissible": admissible}
    if notes:
        r["notes"] = notes
    if reason:
        r["inadmissible_reason"] = reason
    if evidence:
        r["evidence_class"] = evidence
    if extra:
        r.update(extra)
    return r


def v(l2, col=None, **kw):
    out = {}
    if l2:
        for k, x in zip(("l2_1s", "l2_2s", "l2_3s", "l2_avg"), l2):
            if x is not None:
                out[k] = x
    if col:
        for k, x in zip(("col_1s", "col_2s", "col_3s", "col_avg"), col):
            if x is not None:
                out[k] = x
    out.update(kw)
    return out


UNIAD_BASIS = ("UniAD's at-timestep reduction. MEASURED-from-source: at UniAD's FIRST public commit "
               "(4e91222, 2023-03-29) nuscenes_e2e_dataset.py:1024-1036 printed value[i] ONLY; the "
               "'stp3' option arrived 2024-03-19 (33cd8ec) - so a 2023 UniAD number is at-t")
STP3_BASIS = ("averaged-up-to-t reduction: ST-P3 evaluate.py:166 @69aabef (three PlanningMetric "
              "instances at n_future 2/4/6, reported as value.mean()); VAD metric_stp3.py @1688c4b "
              "computes the same per sample")
PD_BASIS = ("PARA-Drive's standardized methodology: oriented ego box, 0.1 m grid, first frame "
            "removed; code not released (its own Table 2 shows the GT trajectory going 0.384 -> "
            "0.32 -> 0.00 as those corrections land)")

ROWS = [
    # ---------------- nuScenes_OL_L2_uniad (UniAD's own implementation) ----------------
    row("nus.uniad.ff", U, UNIAD_IMPL, "FF (LiDAR)", "reference",
        v([0.55, 1.20, 2.54, 1.43], [0.06, 0.17, 1.07, 0.43]), "2212.10156", "Tab. 7", 7,
        ["0.55", "1.20", "2.54", "1.43", "0.43"], UNIAD_BASIS, "LiDAR", "unstated", "unstated"),
    row("nus.uniad.eo", U, UNIAD_IMPL, "EO (LiDAR)", "reference",
        v([0.67, 1.36, 2.78, 1.60], [0.04, 0.09, 0.88, 0.33]), "2212.10156", "Tab. 7", 7,
        ["0.67", "1.36", "2.78", "1.60", "0.33"], UNIAD_BASIS, "LiDAR", "unstated", "unstated"),
    row("nus.uniad.stp3_as_printed", U, UNIAD_IMPL, "ST-P3 (as printed in UniAD's table)", "competitor",
        v([1.33, 2.11, 2.90, 2.11], [0.23, 0.62, 1.27, 0.71]), "2212.10156", "Tab. 7", 7,
        ["1.33", "2.11", "2.90", "0.71"], UNIAD_BASIS, "6 cameras", "not claimed", "yes",
        admissible=False,
        reason=("these are ST-P3's OWN averaged-up-to-t numbers reprinted inside an at-timestep "
                "table - a published cross-convention mix; the same numbers already appear under "
                "nuScenes_OL_L2_stp3 as nus.stp3.stp3")),
    row("nus.uniad.vad_rescored", U, UNIAD_IMPL, "VAD (re-scored under UniAD's methodology)", "competitor",
        v([0.50, 1.02, 1.68, 1.07], [0.02, 0.28, 0.85, 0.38]), "paradrive-cvpr2024", "Tab. 8", 8,
        ["0.50", "1.02", "1.68", "1.07"], UNIAD_BASIS, "6 cameras", "no", "yes",
        notes=("re-scored by PARA-Drive's authors under UniAD's methodology (their Tab. 8) - not a "
               "number UniAD's own code printed. It is the re-scoring that flips the ranking: "
               "UniAD 1.03 < VAD 1.07 here, VAD 0.72 < UniAD 0.76 under VAD's")),
    row("nus.uniad.paradrive_rescored", U, UNIAD_IMPL, "PARA-Drive (re-scored under UniAD's methodology)",
        "competitor", v([0.40, 0.77, 1.31, 0.83], [0.07, 0.25, 0.60, 0.30]),
        "paradrive-cvpr2024", "Tab. 8", 8, ["0.40", "0.77", "1.31", "0.83"], UNIAD_BASIS,
        "6 cameras", "no", "unverified"),
    row("nus.uniad.gt_human", U, UNIAD_IMPL, "GT HUMAN TRAJECTORY (the instrument's false-positive floor)",
        "control", v(None, [0.35, 0.38, 0.35, 0.36]), "paradrive-cvpr2024", "Tab. 8", 8,
        ["0.35", "0.38", "0.36"], UNIAD_BASIS, "-", "-", "-",
        notes=("the recorded human driver 'collides' at 0.36 % under this protocol. No collision "
               "number within ~2x of it is readable. The AV-stack rows EXCLUDE steps where the GT "
               "box collides (PARA-Drive fn. 4), so an in-protocol GT arm scores 0 by construction")),
    # ---------------- nuScenes_OL_L2_stp3 (ST-P3 / VAD implementation) ----------------
    row("nus.stp3.stp3", S, STP3_IMPL, "ST-P3", "competitor",
        v([1.33, 2.11, 2.90, 2.11], [0.23, 0.62, 1.27, 0.71]), "2207.07601", "Tab. 3", 13,
        ["1.33", "2.11", "2.90", "0.23", "0.62", "1.27"], STP3_BASIS, "6 cameras", "not claimed", "yes",
        notes=("the Avg columns (2.11 / 0.71) are computed by later papers - ST-P3's own table has "
               "none. BEV-Planner flags ST-P3's tail-sample GT defect (its ID-0 row carries a dagger)")),
    row("nus.stp3.vad_tiny", S, STP3_IMPL, "VAD-Tiny (ego status switched off)", "competitor",
        v([0.46, 0.76, 1.12, 0.78], [0.21, 0.35, 0.58, 0.38]), "2303.12077", "Tab. 1", 6,
        ["0.46", "0.76", "1.12", "0.78", "0.38"], STP3_BASIS, "6 cameras",
        "no (official ckpt: ego in BEV - 2312.03031 Tab. 1)", "yes"),
    row("nus.stp3.vad_tiny_ego", S, STP3_IMPL, "VAD-Tiny\u2021 (ego status as input)", "competitor",
        v([0.20, 0.38, 0.65, 0.41], [0.10, 0.12, 0.27, 0.16]), "2303.12077", "Tab. 1", 6,
        ["0.20", "0.38", "0.65", "0.41", "0.16"], STP3_BASIS, "6 cameras + ego status", "yes", "yes"),
    row("nus.stp3.uniad_rescored", S, STP3_IMPL, "UniAD (re-scored under VAD's methodology)", "competitor",
        v([0.48, 0.74, 1.07, 0.76], [0.12, 0.13, 0.28, 0.17]), "paradrive-cvpr2024", "Tab. 8", 8,
        ["0.48", "0.74", "1.07", "0.76"], STP3_BASIS, "6 cameras", "no", "yes",
        notes=("re-scored by PARA-Drive's authors under VAD's methodology (their Tab. 8). Under it VAD "
               "(0.72) beats UniAD (0.76); under UniAD's the ranking is the other way round")),
    row("nus.stp3.paradrive_rescored", S, STP3_IMPL, "PARA-Drive (re-scored under VAD's methodology)",
        "competitor", v([0.25, 0.46, 0.74, 0.48], [0.14, 0.23, 0.39, 0.25]),
        "paradrive-cvpr2024", "Tab. 8", 8, ["0.25", "0.46", "0.74", "0.48"], STP3_BASIS,
        "6 cameras", "no", "unverified"),
    row("nus.stp3.admlp_traj", S, STP3_IMPL, "AD-MLP ablation: past trajectory only", "floor",
        v([0.53, 0.91, 1.48, 0.97], [0.17, 0.46, 0.83, 0.49]), "2305.10430", "Tab. 1 (Ours, row 1)", 3,
        ["0.53", "0.91", "1.48", "0.97", "0.49"],
        "the authors use ST-P3's metric implementation (p.3)", "none (no perception)",
        "yes - past trajectory only", "no",
        notes="the ego-status ladder starts here: 0.97 m with no perception and no velocity"),
    row("nus.stp3.admlp_vel", S, STP3_IMPL, "AD-MLP ablation: + velocity", "floor",
        v([0.33, 0.48, 0.66, 0.49], [0.21, 0.29, 0.40, 0.30]), "2305.10430", "Tab. 1 (Ours, row 2)", 3,
        ["0.33", "0.48", "0.66", "0.49", "0.30"],
        "the authors use ST-P3's metric implementation (p.3)", "none (no perception)",
        "yes - + velocity", "no",
        notes="adding ONE ego scalar (velocity) halves L2: 0.97 -> 0.49, with no perception at all"),
    row("nus.stp3.admlp_acc", S, STP3_IMPL, "AD-MLP ablation: + acceleration", "floor",
        v([0.24, 0.32, 0.49, 0.35], [0.18, 0.22, 0.28, 0.23]), "2305.10430", "Tab. 1 (Ours, row 3)", 3,
        ["0.24", "0.32", "0.49", "0.35", "0.23"],
        "the authors use ST-P3's metric implementation (p.3)", "none (no perception)",
        "yes - + acceleration", "no"),
    row("nus.stp3.gt_human", S, STP3_IMPL, "GT HUMAN TRAJECTORY (the instrument's false-positive floor)",
        "control", v(None, [1.02, 0.96, 0.91, 0.96]), "paradrive-cvpr2024", "Tab. 8", 8,
        ["1.02", "0.96", "0.91"], STP3_BASIS, "-", "-", "-",
        notes=("0.96 % under VAD's methodology - WORSE than every model row in this table. Any "
               "collision number here is measuring rasterisation artefacts, not safety")),
    row("nus.stp3.senna", S, STP3_IMPL, "Senna (VLM + VAD)", "competitor",
        v([0.37, 0.54, 0.86, 0.59], [0.09, 0.12, 0.33, 0.18]), "2410.22313", "Tab. II", 6,
        ["0.37", "0.54", "0.86", "0.59"],
        "INFERRED: 'we incorporate Senna with VAD for a fair comparison' (p.6); the paper does not "
        "state the averaging convention", "6 cameras + VLM", "no", "yes",
        admissible=False, evidence="PUBLISHED; protocol tag INFERRED",
        reason="the paper never states its reduction; the tag is inferred from its VAD backbone"),
    row("nus.stp3.senna_ego", S, STP3_IMPL, "Senna* (ego status)", "competitor",
        v([0.11, 0.21, 0.35, 0.22], [0.04, 0.08, 0.13, 0.08]), "2410.22313", "Tab. II", 6,
        ["0.11", "0.21", "0.35", "0.22"],
        "INFERRED as above", "6 cameras + VLM + ego status", "yes", "yes",
        admissible=False, evidence="PUBLISHED; protocol tag INFERRED",
        reason="the paper never states its reduction; the tag is inferred from its VAD backbone"),
    # ---------------- BEV-Planner implementation (same L2 convention, 0.1 m collision) ----
    row("nus.bevp.stp3", S, BEVP_IMPL, "ST-P3 (ID-0, official ckpt) \u2020 erroneous tail GT", "competitor",
        v([1.59, 2.64, 3.73, 2.65], [0.69, 3.62, 8.39, 4.23], ccr_avg=8.37),
        "2312.03031", "Tab. 1 (ID-0)", 5, ["1.59", "2.64", "3.73", "2.65", "8.37"],
        "union implementation, 0.1 m grid", "6 cameras", "no", "yes"),
    row("nus.bevp.uniad_noego", S, BEVP_IMPL, "UniAD (ID-1, no ego status, reproduced)", "competitor",
        v([0.59, 1.01, 1.48, 1.03], [0.16, 0.51, 1.64, 0.77], ccr_avg=1.93),
        "2312.03031", "Tab. 1 (ID-1)", 5, ["0.59", "1.01", "1.48", "1.03", "1.93"],
        "union implementation, 0.1 m grid", "6 cameras", "no", "yes",
        notes="the row our vision-only arm is comparable with (no ego status anywhere)"),
    row("nus.bevp.uniad_ego_bev", S, BEVP_IMPL, "UniAD (ID-2, ego status in BEV - the OFFICIAL ckpt)",
        "competitor", v([0.35, 0.63, 0.99, 0.66], [0.16, 0.43, 1.27, 0.62], ccr_avg=1.72),
        "2312.03031", "Tab. 1 (ID-2)", 5, ["0.35", "0.63", "0.99", "0.66", "1.72"],
        "union implementation, 0.1 m grid", "6 cameras + ego status in BEV", "yes", "yes",
        notes="what the RELEASED UniAD checkpoint actually does - 1.03 -> 0.66 from ego status alone"),
    row("nus.bevp.uniad_ego_planner", S, BEVP_IMPL, "UniAD (ID-3, ego status in BEV + planner)",
        "competitor", v([0.20, 0.42, 0.75, 0.46], [0.02, 0.25, 0.84, 0.37], ccr_avg=1.59),
        "2312.03031", "Tab. 1 (ID-3)", 5, ["0.20", "0.42", "0.75", "0.46", "1.59"],
        "union implementation, 0.1 m grid", "6 cameras + ego status", "yes", "yes"),
    row("nus.bevp.vad_noego", S, BEVP_IMPL, "VAD-Base (ID-4, no ego status, reproduced)", "competitor",
        v([0.69, 1.22, 1.83, 1.25], [0.06, 0.68, 2.52, 1.09], ccr_avg=3.82),
        "2312.03031", "Tab. 1 (ID-4)", 5, ["0.69", "1.22", "1.83", "1.25", "3.82"],
        "union implementation, 0.1 m grid", "6 cameras", "no", "yes",
        notes="the second row our vision-only arm is comparable with"),
    row("nus.bevp.vad_ego_bev", S, BEVP_IMPL, "VAD-Base (ID-5, ego in BEV - the OFFICIAL ckpt)",
        "competitor", v([0.41, 0.70, 1.06, 0.72], [0.04, 0.43, 1.15, 0.54], ccr_avg=2.72),
        "2312.03031", "Tab. 1 (ID-5)", 5, ["0.41", "0.70", "1.06", "0.72", "2.72"],
        "union implementation, 0.1 m grid", "6 cameras + ego status in BEV", "yes", "yes"),
    row("nus.bevp.vad_ego_planner", S, BEVP_IMPL, "VAD-Base (ID-6, ego in BEV + planner)", "competitor",
        v([0.17, 0.34, 0.60, 0.37], [0.04, 0.27, 0.67, 0.33], ccr_avg=2.47),
        "2312.03031", "Tab. 1 (ID-6)", 5, ["0.17", "0.34", "0.60", "0.37", "2.47"],
        "union implementation, 0.1 m grid", "6 cameras + ego status", "yes", "yes"),
    # ---------------- SparseDrive implementation ----------------
    row("nus.sd.uniad_repro", S, SD_IMPL, "UniAD (reproduced with the official ckpt)", "competitor",
        v([0.45, 0.70, 1.04, 0.73], [0.62, 0.58, 0.63, 0.61]), "2405.19620", "Tab. 2b", 8,
        ["0.45", "0.70", "1.04", "0.73", "0.61"],
        "SparseDrive's re-implemented collision (box-vs-box, estimated yaw); L2 aligned with VAD",
        "6 cameras", "not claimed", "yes",
        notes="collision 0.61 vs UniAD's own 0.31 - SparseDrive re-adds the pedestrians UniAD's protocol excludes"),
    row("nus.sd.vad_repro", S, SD_IMPL, "VAD (reproduced with the official ckpt)", "competitor",
        v([0.41, 0.70, 1.05, 0.72], [0.03, 0.19, 0.43, 0.21]), "2405.19620", "Tab. 2b", 8,
        ["0.41", "0.70", "1.05", "0.72", "0.21"],
        "SparseDrive's re-implemented collision; L2 aligned with VAD", "6 cameras + ego status",
        "yes", "yes"),
    row("nus.sd.sparsedrive_s", S, SD_IMPL, "SparseDrive-S", "competitor",
        v([0.29, 0.58, 0.96, 0.61], [0.01, 0.05, 0.18, 0.08]), "2405.19620", "Tab. 2b", 8,
        ["0.29", "0.58", "0.96", "0.61", "0.08"],
        "SparseDrive's re-implemented collision; L2 aligned with VAD", "6 cameras (ResNet50, 256x704)",
        "predicted, not GT (the paper adds an auxiliary ego-status head to avoid the leak)", "yes"),
    row("nus.sd.sparsedrive_b", S, SD_IMPL, "SparseDrive-B", "competitor",
        v([0.29, 0.55, 0.91, 0.58], [0.01, 0.02, 0.13, 0.06]), "2405.19620", "Tab. 2b", 8,
        ["0.29", "0.55", "0.91", "0.58", "0.06"],
        "SparseDrive's re-implemented collision; L2 aligned with VAD", "6 cameras (ResNet101)",
        "predicted, not GT", "yes"),
    row("nus.sd.diffusiondrive", S, SD_IMPL, "DiffusionDrive", "competitor",
        v([0.27, 0.54, 0.90, 0.57], [0.03, 0.05, 0.16, 0.08]), "2411.15139", "Tab. 7", 8,
        ["0.27", "0.54", "0.90", "0.57", "0.08"],
        "built on SparseDrive's recipe; 'metric calculation follows ST-P3' (p.8)",
        "6 cameras (ResNet50)", "predicted, not GT (follows SparseDrive)",
        "yes (INFERRED: follows SparseDrive's recipe)"),
    # ---------------- PARA-Drive standardized ----------------
    row("nus.pdstd.uniad", U, PD_IMPL, "UniAD", "competitor",
        v([0.4774, 0.8947, 1.4703, 0.9474], [0.32, 0.29, 0.73, 0.45],
          l2_ave_all=0.8317, col_ave_all=0.40, offroad_pct=0.91, offlane_pct=1.74),
        "paradrive-cvpr2024", "Tab. 6 (val)", 8, ["0.4774", "0.8947", "1.4703", "0.8317", "1.74"],
        PD_BASIS, "6 cameras", "No (the paper's own column)", "yes"),
    row("nus.pdstd.vad", U, PD_IMPL, "VAD", "competitor",
        v([0.4084, 0.8577, 1.4596, 0.9086], [0.02, 0.26, 0.83, 0.37],
          l2_ave_all=0.7830, col_ave_all=0.30, offroad_pct=1.03, offlane_pct=1.93),
        "paradrive-cvpr2024", "Tab. 6 (val)", 8, ["0.4084", "0.8577", "1.4596", "0.7830", "1.93"],
        PD_BASIS, "6 cameras", "No", "yes",
        notes="the gap to UniAD collapses from 0.31 m (1.03 vs 0.72, cross-protocol) to 0.039 m here"),
    row("nus.pdstd.paradrive", U, PD_IMPL, "PARA-Drive", "competitor",
        v([0.2581, 0.5927, 1.1196, 0.6568], [0.00, 0.12, 0.65, 0.26],
          l2_ave_all=0.5574, col_ave_all=0.17, offroad_pct=0.12, offlane_pct=0.83),
        "paradrive-cvpr2024", "Tab. 6 (val)", 8, ["0.2581", "0.5927", "1.1196", "0.5574", "0.83"],
        PD_BASIS, "6 cameras", "No", "unverified"),
    row("nus.pdstd.admlp", U, PD_IMPL, "AD-MLP (re-implemented by PARA-Drive)", "floor",
        v([0.2267, 0.5847, 1.1782, 0.6632], [0.00, 0.14, 0.70, 0.28],
          l2_ave_all=0.5568, col_ave_all=0.20, offroad_pct=1.21, offlane_pct=2.45),
        "paradrive-cvpr2024", "Tab. 6 (val)", 8, ["0.2267", "0.5847", "1.1782", "0.5568", "2.45"],
        PD_BASIS, "none (no perception)", "Yes", "yes",
        notes=("⭐ the row that decides what this benchmark measures: a perception-free MLP ties the "
               "full stack on L2 (0.5568 vs 0.5574) while its map compliance is 10x worse off-road")),
    row("nus.pdstd.paradrive_plus", U, PD_IMPL, "PARA-Drive+ (ego states)", "competitor",
        v([0.2035, 0.5195, 1.0425, 0.5885], [0.00, 0.09, 0.49, 0.19],
          l2_ave_all=0.4939, col_ave_all=0.13, offroad_pct=0.11, offlane_pct=0.78),
        "paradrive-cvpr2024", "Tab. 6 (val)", 8, ["0.2035", "0.5195", "1.0425", "0.4939", "0.78"],
        PD_BASIS, "6 cameras + ego states", "Yes", "unverified"),
    row("nus.pdstd.uniad_targeted", U, PD_IMPL, "UniAD - 686 non-straight frames", "competitor",
        v([0.4698, 1.0921, 1.9162, 1.1594], [0.00, 0.00, 0.73, 0.24],
          l2_ave_all=0.9935, col_ave_all=0.15), "paradrive-cvpr2024", "Tab. 6 (targeted)", 8,
        ["0.4698", "1.0921", "1.9162", "0.9935"], PD_BASIS, "6 cameras", "No", "yes",
        extra={"subset": "targeted-686 (command != keep forward)"}),
    row("nus.pdstd.vad_targeted", U, PD_IMPL, "VAD - 686 non-straight frames", "competitor",
        v([0.5305, 1.2015, 2.0696, 1.2672], [0.00, 0.29, 0.87, 0.39],
          l2_ave_all=1.0840, col_ave_all=0.34), "paradrive-cvpr2024", "Tab. 6 (targeted)", 8,
        ["0.5305", "1.2015", "2.0696", "1.0840"], PD_BASIS, "6 cameras", "No", "yes",
        extra={"subset": "targeted-686"}),
    row("nus.pdstd.paradrive_targeted", U, PD_IMPL, "PARA-Drive - 686 non-straight frames", "competitor",
        v([0.3844, 0.9729, 1.8805, 1.0793], [0.00, 0.00, 0.72, 0.24],
          l2_ave_all=0.9082, col_ave_all=0.14), "paradrive-cvpr2024", "Tab. 6 (targeted)", 8,
        ["0.3844", "0.9729", "1.8805", "0.9082"], PD_BASIS, "6 cameras", "No", "unverified",
        extra={"subset": "targeted-686"}),
    row("nus.pdstd.admlp_targeted", U, PD_IMPL, "AD-MLP - 686 non-straight frames", "floor",
        v([0.3309, 0.9938, 2.0559, 1.1269], [0.00, 0.58, 3.62, 1.40],
          l2_ave_all=0.9360, col_ave_all=0.94), "paradrive-cvpr2024", "Tab. 6 (targeted)", 8,
        ["0.3309", "0.9938", "2.0559", "0.9360", "0.94"], PD_BASIS, "none (no perception)",
        "Yes", "yes", extra={"subset": "targeted-686"},
        notes=("⭐ where the shortcut breaks: 0.94 % collision vs PARA-Drive's 0.14 % on the same "
               "686 frames - 6.7x - while the two are indistinguishable on full val")),
    row("nus.pdstd.gt_human", U, PD_IMPL, "GT HUMAN TRAJECTORY under the UniAD methodology, corrected step by step",
        "control", {"col_ave_all_uniad_method": 0.384, "col_ave_all_plus_orientation": 0.32,
                    "col_ave_all_plus_finer_grid": 0.00},
        "paradrive-cvpr2024", "Tab. 2", 5, ["0.384", "0.32"], PD_BASIS, "-", "-", "-",
        notes=("the measurement that condemns the legacy collision metric: the GT human trajectory "
               "'collides' 0.384 % with an axis-aligned box on a 0.5 m grid, 0.32 % once the ego's "
               "orientation is used, and EXACTLY 0.00 % at 0.1 m")),
]


_ENTRIES = json.load(open(LIB, encoding="utf-8"))["entries"]
_CACHE = {}


def page_text(key, page):
    """Text of `page` (1-based) of the BANKED pdf for `key`, after a sha256 content check."""
    if key not in _CACHE:
        ent = _ENTRIES[key]
        path = os.path.join(REPO, ent["path"])
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if h != ent["sha256"]:
            raise SystemExit("sha256 MISMATCH for %s: %s != %s" % (key, h, ent["sha256"]))
        from pypdf import PdfReader
        _CACHE[key] = [pg.extract_text() or "" for pg in PdfReader(path).pages]
    pages = _CACHE[key]
    return pages[page - 1] if 1 <= page <= len(pages) else ""


def verify(rows):
    miss = []
    for r in rows:
        t = page_text(r["source"]["library_key"], r["source"]["page"])
        for tok in r["page_tokens"]:
            if tok not in t:
                miss.append((r["id"], tok, r["source"]))
    return miss


bad = verify(ROWS)
keys = sorted({r["source"]["library_key"] for r in ROWS})
# positive assertion, not a silent pass: name the PDFs whose bytes were re-hashed and matched
print("sha256 re-verified against library.json: %d/%d banked PDFs -- %s"
      % (len(_CACHE), len(keys), ", ".join(keys)))
print("rows:", len(ROWS), "| tokens not found on their cited page:", len(bad))
for b in bad:
    print("   MISS", b)
if bad:
    raise SystemExit("refusing to write: a page_token is not on its cited page")

# ⛔ MUTATION ARM: a checker that cannot go RED is not a check. Re-verify a row whose token has
# been replaced by one that is NOT on that page, and require the miss to be seen.
import copy
mut = copy.deepcopy(ROWS[0])
mut["page_tokens"] = ["0.1234567"]
assert len(verify([mut])) == 1, "the token checker is inert -- it did not catch a planted miss"
print("mutation arm: a planted bogus token IS caught (the checker is not inert)")

doc = {
    "_for": ("W4 — products/P7-TanitEval/benchmarks/published_results.json (W4 OWNS that file; this "
             "is a PATCH FILE, never a direct edit). Append `rows` to `results`; they use W4's row "
             "schema and W4's protocol keys, and none collides with the 8 nuScenes rows already there."),
    "_built": {"by": "W6 (E5)", "utc": "2026-09-20",
               "evidence_class": "PUBLISHED (every number re-read from the banked PDF named in its row)",
               "token_verification": ("every page_token occurs in the pypdf text of its cited page - "
                                      "checked by the script that built this file "
                                      "(FlyWheels/.../2026-09-19-nuscenes-planning-harness/code/"
                                      "make_w4_rows.py), which also re-hashes each PDF against "
                                      "library.json and refuses to write on a miss. "
                                      "⛔ NOT AN INDEPENDENT CHECK: the tokens were chosen "
                                      "with pypdf and re-found with pypdf - the producer's own "
                                      "derivation re-run, which measures transcription, not "
                                      "correctness. W4's verify_published_against_pdfs.py "
                                      "(PyMuPDF) is the admissible check and is still OUTSTANDING "
                                      "on these rows."),
               "independent_reads": ("NONE. 0 of these 39 rows share a (paper, page, system) cell "
                                     "with a row W4 extracted, and 0 carry a value-set W4 also "
                                     "published - so no number here has been read twice by two "
                                     "readers. 15 rows sit on a paper W4 also cites and 5 on an "
                                     "exactly-shared (paper, table); those 5+15 agree on the PAGE, "
                                     "which is a consistency check on layout, not on any value. "
                                     "An earlier draft offered the 15 as an independent control; "
                                     "that was wrong and is retracted - see RESULT.md F12."),
               "human_readable_twin": ("FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                       "2026-09-19-nuscenes-planning-harness/raw/nuscenes_external_rows.md")},
    "_already_in_w4": ["nus.stp3.vad_base", "nus.stp3.vad_base_ego", "nus.stp3.ad_mlp",
                       "nus.uniad.uniad", "nus.bevp.bev_planner", "nus.bevp.bev_planner_pp",
                       "nus.bevp.ego_mlp", "nus.bevp.gostraight"],
    "_harness_version_note": {
        "why": ("W4's finding applies here too: a published number needs its HARNESS, not just its "
                "table. For nuScenes there is no devkit at all - the harness IS the research "
                "codebase, and the codebases disagree by more than the methods do."),
        "reduction_pin": {
            "nuScenes_OL_L2_uniad": ("OpenDriveLab/UniAD@609ee083ea51c3521c323f1279dfc4cee0e60467 "
                                     "projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py:"
                                     "1043-1044. MEASURED: UniAD's FIRST public commit "
                                     "(4e91222da855fa83a8caa55f512536e82f9f9818, 2023-03-29) had "
                                     "value[i] ONLY; the 'stp3' option arrived 2024-03-19 "
                                     "(33cd8ecf7ad5ea4b7d779a2176841d3c1c4d4174, 'Add option for "
                                     "legacy planning metric definition') - so every 2023 UniAD "
                                     "number is at-timestep."),
            "nuScenes_OL_L2_stp3": ("OpenDriveLab/ST-P3@69aabefd2610951d9e34238142776ed2228673be "
                                    "evaluate.py:71-73,135-137,166; VAD@1688c4b1c3a9e2e7873ca9700ff"
                                    "8058170c0e3c8 metric_stp3.py; AD-MLP@4b93ba085ee47474152f282177"
                                    "865796ea577fc0 re-uses ST-P3's code (deps/stp3).")},
        "collision_implementations": {
            "0.5 m grid, axis-aligned ego box": "ST-P3 / UniAD / VAD (the legacy rows)",
            "0.1 m grid, estimated yaw, trajectory-level": "BEV-Planner (2312.03031)",
            "box-vs-box polygons, estimated yaw": "SparseDrive (2405.19620) and DiffusionDrive",
            "oriented box, 0.1 m, first frame removed": "PARA-Drive (code not released)"},
        "why_it_bites": ("the same VAD checkpoint reads 0.72 m under VAD's harness and 1.22 m with "
                         "the averaging removed (paradrive-cvpr2024 Tab. 1), and the UniAD/VAD "
                         "ranking flips between the two legacy harnesses (Tab. 8). Our own runs "
                         "carry (reduction, pipeline) on every row for exactly this reason."),
        "our_harness": ("taniteval/adapters/nuscenes_planning.py @ this commit + "
                        "taniteval/taniteval/bench/plugins/nuscenes_ol.py; both conventions ported "
                        "verbatim from the pins above; cv2.fillPoly ported from opencv@4.5.4 "
                        "(parity vs a real cv2: NOT RUN - no OpenCV in the venv).")},
    "_command_note": CMD_NOTE,
    "_our_row": {
        "status": "NOT MEASURED - pending the PI's nuScenes download (W6 RESULT.md, PI ACTION LIST)",
        "produced_by": "python -m taniteval.bench nuscenes_ol --ckpt <path> --split val (ONE run per convention)",
        "comparable_only_with": ("the ego-status-free rows - nus.bevp.uniad_noego (1.03), "
                                 "nus.bevp.vad_noego (1.25), nus.bevp.bev_planner (0.55) - because "
                                 "our binding rule forbids ego status at inference and our arm is "
                                 "command-free"),
        "must_carry": ["protocol", "implementation", "gt_control (the GT human trajectory's raw "
                       "collision under the same protocol)", "command_source", "the non-straight "
                       "subset with its n", "input observed_frac (our 120 deg frame sees ~43 % on "
                       "nuScenes' ~64.6 deg CAM_FRONT)"]},
    "rows": ROWS,
}
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(doc, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print("wrote", OUT)

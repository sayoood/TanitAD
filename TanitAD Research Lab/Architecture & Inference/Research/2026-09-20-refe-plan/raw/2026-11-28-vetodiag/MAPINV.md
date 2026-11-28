# MAPINV: vision drivable-area predictors already in the programme (2026-11-28, read-only inventory)

Source: git mirror `refs/heads/agent/arch-inf-20260803` (read via `git show`). GOALS = `Project Steering/GOALS_AND_CLAIMS.md`, REG = `Project Steering/MODEL_REGISTRY.md`, PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. All numbers below are INHERITED from those rows (not re-verified here) unless stated; the rows themselves are MEASURED by their authors. Not re-probed: model weights, F:/safehead_cache, pods.

## Headline
The programme has ALREADY tried this lever on REFe, under the name **H-VDAC-P6-1** ("vision-built drivable-area check on P6's 64 proposals"). Hard-veto ceiling and head quality are measured; the head never cleared its bar. The new "perfect DAC = +1.53 on C1" figure should be read against that record.

## 1. Candidate predictors
| name | output | cameras | trained on | measured quality | runs on navtest? | work to emit ego-frame raster to ~40-60 m |
|---|---|---|---|---|---|---|
| **P7 `aux_drv` head** (`PK/refe/aux_drivable_head.py`, SPEC_AUX_DRIVABLE_HEAD) | 40x32 BEV grid, 2 m cells, x[-8,72) y[-32,32), mask logit + signed distance (2.44 M params); image tokens only (admissible) | REFe's 4 (F0,B0,L0,R0 @960x512; navtest has them) | navtrain labels (nuPlan map), in P7 only | IoU [0,40) m **0.8225 [0.790,0.853]** on 1,058 C3 tokens; **0.766 on the 416 DAC-zero tokens** vs 0.858 elsewhere; head DAC AUC 0.8105 vs scorer 0.7874 (GOALS D-REFE-P7-MECH-1). P7 did NOT cut DAC zeros (249 -> 260) | yes (same pipeline as P7, 91.07 navtest) | head weights live in P7 `model_final` (sha 2613fc17, REG 14.4), NOT in C1 (head OFF in P5-C, `PREREG_P5C.md:115`). Head sits on P7's trunk; C1's LoRA moved 2,000 steps, so reuse needs a head re-fit on C1's trunk (small) |
| **VDAC HA / HB / HC heads** (frozen P6 trunk; HB = LoRA trunk copy; HC = 1 m cells + boundary weight) | same raster idea, 2 m (HA, HB) / 1 m (HC) | REFe 4 cams | navtrain | S1-P1 IoU x<40 m: HA 0.780, **HB 0.811**, HC 0.776 (bar 0.80); S1-Q AUC 0.833 / 0.838 / 0.835 vs bar 0.90, true-field ceiling 0.941 (GOALS D-VDAC-S1A-1, A5-HB-READ-1, A5-HC-READ-1). Stage 0: in-sample IoU 0.786, i.e. under-fit, not a generalisation gap | trainer-gated, navtrain only so far | HB is built and reachable; HBC (HB + 1 m + band weight) is drafted, ~13.4 L40-h, PI decision pending |
| **refcv8 map head** (`_map_hires`, 10 cm, 100 m x +-30 m; Physical AI AV) | 10 cm BEV classes incl. drivable/sidewalk/lane edge | ONE front cyl. 120 deg cam, 416x1024 | PhysicalAI-AV (+ derived labels) | refcv6 drivable IoU 0.70/0.62/0.73 on 3 clips; refcv7 lane IoU 0.164 (REG 5707); MAPFT FAIL: planner hard-filter off-drivable **0.171 vs GT-map 0.022** (REG 6136, GOALS H-R9-MAPFT-1) | refcv8 did run navtest (72.78, REG 16.8) but with a different camera rig than OpenScene's | would need cylindrical-vs-pinhole recalibration and a navtest adapter for the map path: high |
| **SAM3 map (D-SAM3-MAP-1)** | road map from SAM3 + LiDAR ground height, 7 virtual views | PhysicalAI rig | zero-shot | whole-clip vote on 2 clips | LiDAR-dependent, offline label producer, not an inference model | not a candidate |
| refcv9 MAP-2 (nf_prec_* sidewalk precision) | sidewalk-precision bars | single front | PhysicalAI | E1-T1, E2 FAILED (`nf_swedge_prec_lv_5_20` 0.41 vs base 0.75); precision bar 0.99 needed (H-R9-E0H) | n/a | refuted path |

## 2. Does C1 itself have a BEV/semantic head?
**No.** C1 = P7 + 2,000 steps on RJ5 labels (REG 14.6); aux head OFF. It is DriveZero-style: 4 cameras -> DINOv3 ViT-L + LoRA -> `visual` [B,7680,1024] -> `visual_ctx` [B,7680,256] read by the scoring decoder (SPEC_AUX_DRIVABLE_HEAD 2.1). There is no TransFuser BEV grid; the nearest thing is `visual_ctx`, on which the P7 head is a 2-block cross-attention decoder. The cheapest route is therefore a head on C1's own `visual_ctx`, not a new backbone.

## 3. navtrain drivable-area label code: EXISTS
`PK/refe/aux_label_store.py` (+ `refe/navsim_dac.py:drivable_polygons`): per-token signed-distance raster in the rear-axle frame (0.5 m cells, x[-8,72) y[-32,32)), ROADBLOCK + INTERSECTION + CARPARK polygons from the nuPlan map, sharded per log, sha-pinned manifest, navtest-leak refusal; fidelity vs NAVSIM's exact polygon verdict 99.0 % corners / 97.7 % proposals (SPEC 0). Built for navtrain (VDAC fit packs "all in label_store"). Labels use the map (privileged), inference stays vision-only, matching the PI rule.

## 4. Ceiling evidence for the lever (read this before planning)
- True-field hard veto on P6's selection (navtrain, in-sample): V0 +0.204 PDMS, only 35 % of the oracle (+0.586) (GOALS D-VDAC-S0). October's navtest veto on u400 kept 14 %.
- Soft rule on the TRUE field was harmful (R -0.2 to -0.4).
- Zero-tail census: DAC zeros are mostly near-twin grazes the scorer cannot see, 42 % made by NAVSIM's tracker not the plan (D-REFE-ZEROTAIL-3); an in-sample 2 m field cannot see centimetre grazes.
- Oracle-agent check is 2.6x larger than the DAC check (+1.67 vs +0.64) but a true-raster pose veto only reached capture 0.43 (D-VDAC-6-S0B).

## 5. Ranked recommendation
1. **Head on C1's `visual_ctx` using the existing `DrivableAuxHead` + label store, HB-style LoRA copy feeding only the head (C1's planner stays bit-identical), then a hard veto only.** Cheapest (code built, labels built, ~13 L40-h per HBC draft). Main risk: S1-Q has been flat at 0.83-0.84 across three heads (bar 0.90, true-field 0.94), so the head may again fail to separate; and a perfect check is worth +1.53 only before the 35 %-ish capture measured so far (expect ~+0.2-0.5).
2. **Re-use P7's trained head as a no-training zero-GPU read first**: run it on P7's 64 proposals' footprints (GOALS names this next step, "head-veto read on P7's 64 proposals"). Risk: trunk mismatch to C1; head weakest exactly on DAC-zero tokens (0.766).
3. **refcv8 map head via an adapter**: highest work (camera model, scale), worst measured precision (off-drivable 0.17 vs 0.02), no navtest-native path. Not recommended.

Wrong-scope guard: "does not exist at one location" claims above were checked in GOALS, REG and `PK/` listings; no `*bev*`/`road_mask` module for a navtest-native TransFuser-style head was found in the ls-tree grep (names matched only refcv/perception-redesign files).

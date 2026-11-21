# VDAC Stage 0 dev-box read (CPU), 2026-11-12

**Verdict (mechanical): S0-WEAK.** In-sample IoU [0,40 m) = 0.7864 [0.7803, 0.7924] < 0.85. Selects **HB and HC as one registered pair** (A5 draft sections 2, 3; draft row "S0-WEAK ... HB and HC as one registered pair"). No control failed (`invalid_controls: []`).
**s1f:** step-0 reproduction PASS (sign agreement 1.0 on 1,705 frames); S1-F holds (P1 lower 0.5120, Q lower 0.0891, bar both > 0); reader v2 verdict S1-WEAK; stage0 row and reader verdict agree.

## Provenance
- Code: mirror commit `28db77fc`. vdac_stage0.py sha256 04a3d5f4dde23dd669167c18a6090cb16a37c3a680ee3351729b4fd8b51dc2b7 (matches `tool_sha256` in the outputs); s1_read_v2.py 796aea21506e27d1646f5b7aa71b08089ced8c6d92a5b02429d35431692b09f4; frozen reader s1_read.py 58aa3a77b976a6fb4b08b4bc5126d572e8be8ef1fc37926d9c9ce0983eed718e (asserted by the tool).
- Frame list sha256 1d353309a1d81a78cc04197ac39bd4e7ef467853ff59f5f7d3e97fde538eda45, repro256 26cd3a11...2770df: frozen in `FROZEN_SHAS.txt` at 20:11:01Z (`ZZSTAGE0_LIST_FROZEN`), before the forward (`ZZSTAGE0_FWD_DONE` 20:24:23Z). The read re-verified the list sha. `head_pred_step0` sha 2ebe4b54... is RUNTIME-recorded (no earlier freeze).
- S1a inputs re-downloaded from HF `Sayood/tanitad-refe-scorer1p6`; all 6 npz/json + h48 list equal the manifest sha256 (head_s0_a 6b9ac2bc..., head_s0_b 64d2f71b...). Manifest was a re-pointed copy. Label store `E:/Projects/TanitAD-artifacts/aux_drivable/label_store`. CUDA_VISIBLE_DEVICES=-1, no GPU lock.
- Preflight: no K-ledger entry or freeze is required before the read in the A5 draft (section 7 freezes: tool, list sha, card; the list sha is frozen above). The A5 card sha / `--aux-drv-lora-lr` confirmation (draft section 7 item 3) is a precondition for HB/HC arms, not for this read; UNVERIFIED whether the MM has done it.
- Operational note: `eval/s1_read.py` at the tip is a DIFFERENT file (sha 80a9f410...) from the frozen reader and is found first by `s1_read_v2`'s sibling lookup; the tool REFUSED on it (sha assertion worked). Run with `S1_READ_V1` pointing at the frozen `.../vdac/s1a_read/s1_read.py`. The POD_PLAN commands need that env var.

## Numbers (MEASURED, `stage0_read.json`; estimator: frame-level bootstrap per tool, n_boot 10,000, seed 20260927; city quota 512 each)
| | value | 95 % CI |
|---|---|---|
| in-sample IoU [0,40) | 0.7864 | [0.7803, 0.7924] |
| per city LV / BOS / PIT / SG | 0.8942 / 0.7441 / 0.7366 / 0.7034 | |
| bands 0-20 / 20-40 / 40-72 | 0.8113 / 0.7611 / 0.6923 | |
| held-out S1-P1 (S1a) | 0.7802 | [0.7587, 0.7974] |
| in-sample minus held-out | +0.0062 | reported, not deciding |

Controls: C0-FLIP ok (0.7864 vs flipped 0.3419, margin 0.1); C0-ID ok (256 frames, sign agreement 1.0, banked IoU 0.84770 = repro IoU 0.84770, diff 0.0; bars >= 0.999, <= 0.002); C0-FLOOR (all-drivable constant head) 0.3764, reported.
s1f: reproduction sign agreement 1.0 on 1,705 frames; reader v2 outputs equal S1a to the digit (S1-P1 0.7802, S1-Q 0.8332 [0.7951, 0.8691], C-SHUF 0.4468, C-FLIP 0.7056).

## Reading
The head is weak on the frames it trained on (0.786, just 0.006 above held-out): the S1a shortfall is under-fit/representation-limited, not a generalisation gap. By the draft this is the S0-WEAK branch: HB (copy + LoRA, features) and HC (finer cells, boundary-weighted loss) are registered as one pair. Weakest slices: PIT/SG (~0.70-0.74) and far band 40-72 m (0.692).

## Does NOT show
Not which of HB/HC addresses the deficit (the draft registers both for that reason); not a driving or planner result (perception IoU only, T0-type); the fit bar 0.85 is the draft's threshold. One seed; the interval is frame-resample over this list, not a training-seed floor (H-ESTIM-SEED-1). Pooled in-sample IoU uses equal city quotas, not the city reweighting of S1-P1 (a small definitional difference).

## Manifest
Hold files: STAGE0_READ.md, in_sample_iou.json, stage0_read.json, read.log, s1f.log, s1f_read_v2.json, s1f_read_v2_tables.md, manifest_s1f_v2.json (single copy: yes, until landed; all re-creatable from stage0/ + HF + git).

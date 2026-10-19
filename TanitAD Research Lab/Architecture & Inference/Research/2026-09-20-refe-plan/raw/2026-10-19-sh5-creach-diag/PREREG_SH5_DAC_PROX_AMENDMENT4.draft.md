# PREREG SH5-DAC-PROX: AMENDMENT 4 (DRAFT, NOT SEALED; Research Lab, 2026-10-19). C-REACH: why the branch could not read a planted channel, and the recipe that does

Amends `PREREG_SH5_DAC_PROX.md` (sealed 2026-10-19T13:39:48+02:00, sha256 `687e08c8...0a0839`) after Amendments 1-3. **DRAFT: the Master Mind rules and seals.** Status of the sections: s1-s5 MEASURED and final; **s6-s9 PENDING the GPU diagnostic grid** (`work/diag1`, `work/diag2`, both queued on the dev-box lock at the time of writing; see `HANDOFF.md`).

## 0. What exists and what was read (disclosure)
- No real arm (S, Snoprox, Sshuf) has been trained, by the chain or by this diagnosis; `analyze` has never run; no `Delta_F`, no real-arm OOF logit exists.
- The chain's C-ID (seeds 0, 1; E = 0) and C-REACH pre-flight (`fit --stage PRE --folds 0 --seed 0`) ran: `ZZPREFLIGHT {"plant_s0": {"n_sets": 322, "auc_all_strata": 0.6336, "r2_reg": 0.4751}}` (`par18/sh5/work/chain_local.log`). The chain's pre-stated rule stopped it.
- This diagnosis read: the inputs and labels (as `fit` does), the ctx file (6 rows, probe 3), and the planted CONTROL's held-out outputs on fold 0 (the same quantity the chain's PRE printed). The diagnostic GPU runs train ONLY planted-arm learners on fold 0 (`code/diag_pre.py`, which imports the sealed modules read-only and applies every variant in its own process).

## 1. What the plant is, from source (MEASURED by reading the pinned code)
- **Built** by `sh5_branch.plant_block` (sh5_branch.py:185-201): one entry per (set, proposal, key time); channels 0..31 = 8-octave Fourier code of (x, y), wavelengths 128 m .. 1 m (line 196); channel 32 = clip(signed key clearance, +-10) / 10 (line 200); channel 33 = 1 (flag). Source: the privileged SIGNED key clearance (`sh5_lib.signed_key_clearance`), noise-free for `plant`.
- **Injected into the ctx ROWS** (not an appended feature, not the label): `Trainer._batch` overwrites the LAST n slots of the token's 7,680 (sh5_train_eval.py:287-291), n = sets x 576 <= 1,152 on LOCAL, at every step and at prediction. Label and targets are the arm's normal ones (`arm_arrays`).
- **PRE** (`STAGES["PRE"]`, sh5_train_eval.py:35) trains ONE `plant` learner on folds 1-4 with the registered recipe (sh5_lib.py:37-42): Adam lr 3e-4, wd 1e-4, 50-step warm-up (sh5_train_eval.py:309), 12 token passes (3 cycles x 4 inner, 2 shards), 8 tokens per chunk, `p_sub` 32, loss BCE + 0.1 x Huber(1 m) (sh5_branch.py:116), bf16 autocast, both heads ZERO-initialised (sh5_branch.py:93-94); ~1,200 optimizer steps.
- **Query** `point_features` (sh5_branch.py:40-48): x/30, y/10, cos/sin yaw + 6 octaves, wavelengths 128 m .. 4 m (FOURIER_K 6, sh5_lib.py:42).
- **Metrics:** `_fl_pairs` (sh5_train_eval.py:435-440) = the census within-set pair AUC over all strata ("ALLQ"); `r2_reg` (442-446) = pooled held-out R^2 of the 9 |clearance|. **`n_sets` 322 = fold 0** (LOCAL sets per fold 322 / 237 / 421 / 66 / 398, prereg s2; `cov = n_cover == 1`, line 686): not a defect.

## 2. The bar
- **SEALED** (prereg s7 "C-REACH", PREREG_SH5_DAC_PROX.md:85; `REACH_MIN_AUC, REACH_MIN_R2 = 0.95, 0.50` sh5_lib.py:51, read at sh5_train_eval.py:511-512): the plant's DAC-logit-alone **F-AUC (Q4) LOWER CI >= 0.95 AND R^2 >= 0.50 on BOTH seeds**, all five folds, log-cluster bootstrap. Untouched by Amendments 1-3 and by this draft.
- READY.md Q3's "near 1 and r2_reg >= 0.5" is informal; the chain's rule (point ALLQ AUC >= 0.95 AND r2 >= 0.5, fold 0) is the Master Mind's operationalisation: looser than the sealed bar (a point, all strata, one fold) and the right order. A pre-flight at 0.63 fails both. **This draft keeps the sealed bar and proposes that the pre-flight rule be read on fold 0 as: point F-AUC >= 0.97 AND point ALLQ >= 0.95 AND r2 >= 0.5 on BOTH seeds** (a point margin, because the sealed bar is a LOWER CI). [PROPOSAL, MM rules]

## 3. Hypotheses and the probes that killed them (CPU, MEASURED)

| | hypothesis | probe | result | verdict |
|---|---|---|---|---|
| (a) | the plant is too weak / noisy | oracle within-set AUC of `min_k sc_key` (`raw/probe1_oracle.json`) | fold 0 **ALLQ 0.9972, F 0.9974** (all LOCAL 0.9971 / 0.9984); P(unsafe \| min_k sc < 0) = 1.0, P(min_k sc < 0 \| unsafe) = 0.968. P2's own row on fold 0: 0.8676 / 0.7327; the unsigned |clearance| block: 0.8729 | **killed**: "near 1" was right; the learner (0.634) is below P2's row |
| (d) | stratum / token-order misalignment, ambiguous plant | t = 0 clearance across proposals and sets; conflicting entries within 0.1 m; cover mask | t = 0 identical across the 64 proposals and the 2 sets of a token (one frame); 3,685 conflicts in 2.15 M close pairs (0.17 %); <= 1,152 entries of 7,680 slots; the plant is computed per token from `D` inside `_batch`, never read from the ctx file, so row order cannot misplace it; cover = fold 0 | **killed** |
| (e) | the metric | `_fl_pairs` on P2's row and on the oracle | 0.9000 (all LOCAL; Amendment 1's dry run 0.9007) and 0.997 | **killed** |
| (c) | reachability (mask / pooling / detach / wrong column) | `raw/probe3_reach_grad.json`: sealed `_batch` + sealed model, 6 REAL ctx rows, CPU bf16 autocast, d loss / d ctx | step 0 (zero heads): **exactly 0.0** on the planted clearance, planted code and real slots (the control reads zero); after 3 Adam steps the planted clearance gradient **2.68e-6 > 0** (per element the same order as the real slots); SDPA reads all 7,680 slots (sh5_branch.py:74); attention mass on the plant at init 0.144 = its slot share 0.1375 | **killed as plumbing**: the plant reaches the loss |
| (f) NEW | the read needs SHARP attention, which the recipe never learns | `raw/probe2_ideal_reader.json`: an IDEAL softmax(beta x Fourier-kernel) read of the token's plant, no distractors | ALLQ at beta 4 / 16 / 64: K=6 (query today) **0.771 / 0.953 / 0.978**; K=8 0.754 / 0.983 / 0.989; K=10 0.736 / 0.989 / 0.994; K=12 0.733 / 0.992 / 0.996 | the bar needs an attention logit range of ~30 (beta >= 16) on the position match; a flat read (beta 4) caps EVERY resolution at 0.73-0.77, and the learner sits below even that |
| (b) | budget / recipe | GPU grids 1-2 (s6) | no cell of lr {3e-4, 1e-3} x {12, 36} ep x octaves {6/8, 8/8, 10/10, 12/12} passes; best final F 0.601, ALLQ 0.673 | **killed for lr <= 1e-3, <= 36 ep**; grid 3 (sharpness levers) PENDING |

**Why the sealed synthetic said the plant is readable** (`raw/probe4_synth_plant_only.json`, MEASURED): the synthetic C-REACH test (`test_sh5_pipeline.py:145-153`, AUC > 0.9, R^2 > 0.5) ran at lr 2e-3 (6.7x the registered 3e-4; the test's own docstring, line 119), 40 epochs, d 32, `fourier_k` 3, P = 1,280 slots (so the plant filled 45-90 % of the slots, against 7.5-15 % on the real ctx), fp32, and proposals 0.3-1.15 m apart at 0.5-4 s (real: 0.002-0.43 m). I suspected the synthetic arm read the road widths that `make_synth(info=True)` writes into EVERY slot rather than the plant. **REFUTED:** with an information-free ctx (info=False) the same recipe still reads the plant, ALLQ 0.9852 / R^2 0.628, against 0.9869 / 0.584 with the info ctx. So the synthetic proves the MECHANISM can read a plant. What it never tested is the registered RECIPE on the real geometry, where the plant is a small share of the slots and the proposals are sub-metre neighbours.

## 4. Root cause
PENDING s6 (to be stated with its evidence class). Leading candidate after s3: **the registered recipe never learns a sharp, position-matched read of the plant** (an optimisation/budget defect of the INSTRUMENT, not a property of the plant or of the plumbing), and the synthetic test that set the bar could not have caught it.

## 5. Root-cause class (for RETRACTION_LOG)
*A control validated in a test world where the control's job was easier than in production.* The synthetic world gave the plant 45-90 % of the slots, spread the proposals 2-3x wider, and trained at 6.7x the learning rate for 3.3x the epochs. The real plant needs a sharp positional read among sub-metre neighbours, with 85-92 % of the slots as distractors. This is a sibling of Amendment 1's class: the test world differed from production in a dimension it never exercised. Rule proposed: **validate a planted control on the REAL geometry, with the REAL slot share, at the REGISTERED recipe. Compute its ideal-reader ceiling before the bar is set.**

## 6. The fix (grids 1-2 MEASURED; grid 3 / 4 PENDING, `PREREG_GRID3.md`, sha256 in `PREREG_GRID3.md.sha256`)
Grids 1-2 (`work/diag1/diag_summary.json`, `work/diag2/diag_summary.json`, fold 0, final cycle, point values, one training run per cell and seed):

| cell | seed | ep | ALLQ | F | R^2 | zero_clear ALLQ / R^2 | shuffle_clear R^2 |
|---|---|---|---|---|---|---|---|
| base K6/o8 lr 3e-4 | 0 / 1 | 12 | 0.6489 / 0.6732 | 0.5105 / 0.5863 | 0.398 / 0.476 | - / 0.6723, 0.376 | - |
| K8/o8, K10/o10, K12/o12 lr 3e-4 | 0 | 12 | 0.647 / 0.635 / 0.648 | 0.528 / 0.553 / 0.601 | 0.503 / 0.552 / 0.519 | - | - |
| base / K12/o12 lr 3e-4 | 0 | 36 | 0.6158 / 0.6659 | 0.5292 / 0.5608 | 0.544 / 0.661 | - | - |
| K6/o8 lr 1e-3 | 0 / 1 | 36 | 0.6149 / 0.6263 | 0.4853 / 0.5475 | 0.437 / 0.644 | 0.6148, 0.433 / 0.6264, **0.024** | 0.437 / **0.630** |
| K10/o10 lr 1e-3 | 0 / 1 | 36 | 0.6237 / 0.5877 | 0.5191 / 0.4972 | 0.412 / 0.166 | 0.6236, 0.412 / 0.5877, 0.166 | 0.412 / 0.166 |

Reading: zero_clear moves ALLQ by <= 0.0043 everywhere, so the classifier never reads the planted clearance. Where the regression reads it (K6/o8 lr 1e-3 s1: R^2 0.644 -> 0.024 under zero_clear), shuffling the planted values across positions keeps R^2 at 0.630: it reads a TOKEN-LEVEL average of the plant, not the position-matched entry. That is finding (f) seen from the trained side: the learned read is flat.
Grid 3 tests the attention-sharpness levers first (learnable temperature, cosine attention with tau0 16, q/k init x 6; at K6/o8 and K8/o8, both seeds) and measures the trained logits' beta_eff against the ideal reader's beta >= 16; grid 4 (lr 3e-3 / p_sub 64 / 60 ep) runs only if grid 3 has no pass.
## 7. C-REACH evidence, both seeds (PENDING)
## 8. Pinned files and new pins (PENDING)
## 9. Power (Amendments 2 / 3)
The A1 power arithmetic does not read the training recipe: `sh5_design_calc.py` (mde, plant) uses PRIVGEOM-SAFE's banked arrays and an IDEAL reader; `sh5_amend2_power.py` / `sh5_amend3_power.py` read `mde.json`, the plan and the banked CPU false-fire rows, whose own config is hard-coded in `sh5_sim_falsefire.py:28-38` (d 32, fourier_k 3, CPU lr). A change of lr, epochs or query octaves for the real rig therefore leaves every number of Amendments 2 and 3 unchanged by construction; what it can change is the real rig's TRAINING variance, which those amendments already list as UNMEASURED (`H-ESTIM-SEED-1`). PENDING: re-run of `sh5_design_calc.py mde` to show it reproduces `work/mde.json` byte-identically.

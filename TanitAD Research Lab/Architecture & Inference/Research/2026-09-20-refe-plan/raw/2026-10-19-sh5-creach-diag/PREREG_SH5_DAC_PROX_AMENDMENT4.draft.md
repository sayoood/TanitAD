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
**Status 2026-10-20: PARTLY MEASURED, the binding term is narrowed but not yet fixed.** MEASURED (grids 1-4 + probe 5, s6):
- Every learner fits TRAIN through the visual ctx (memorisation): train F stays 0.83-0.95 with the plant removed (`raw/probe5_ckpt_seedsplit.json`).
- The planted read is the only path that generalises to fold 0.
- Dot-product attention commits a head to the plant on both seeds but never sharpens it: beta_eff plateaus at 5.6-6.5 over 60 ep, and the read stays token-level.
- Cosine attention (learnable tau, tau0 16) CAN sharpen. On seed 0 a committed head reached beta_eff 18.4 and gave the first position-matched read (F 0.846, collapsing to 0.61 under zero_clear and shuffle_clear).
- On seed 1 the K6 cosine learner never committed a head and read nothing from the plant.

HYPOTHESIS H5 (tested by grid 5, `PREREG_GRID5.md`): plant commitment versus memorisation is a per-seed race, followed by a slow, grokking-type sharpening.

Original candidate after s3, kept for the record: **the registered recipe never learns a sharp, position-matched read of the plant** (an optimisation/budget defect of the INSTRUMENT, not a property of the plant or of the plumbing), and the synthetic test that set the bar could not have caught it.

## 5. Root-cause class (for RETRACTION_LOG)
*A control validated in a test world where the control's job was easier than in production.* The synthetic world gave the plant 45-90 % of the slots, spread the proposals 2-3x wider, and trained at 6.7x the learning rate for 3.3x the epochs. The real plant needs a sharp positional read among sub-metre neighbours, with 85-92 % of the slots as distractors. This is a sibling of Amendment 1's class: the test world differed from production in a dimension it never exercised. Rule proposed: **validate a planted control on the REAL geometry, with the REAL slot share, at the REGISTERED recipe. Compute its ideal-reader ceiling before the bar is set.**

## 6. The fix (grids 1-4 MEASURED, no pass; grid 5 RUNNING, `PREREG_GRID3.md`, sha256 in `PREREG_GRID3.md.sha256`)
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
### Grids 3-4 (MEASURED; `work/diag3|4/diag_summary.json`, per-cell table `raw/grid34_cells.json`; fold 0, final cycle, point values, one training run per cell and seed)
No cell passes in either grid (`ZZDIAG3_PASSING []` ×2). Per PREREG_GRID3 grid 3 is **O3**: no lever cell reaches beta_eff >= 16 on both seeds with top-1 >= 0.5.

| cell (lever) | ep | F s0 / s1 | ALLQ s0 / s1 | R^2 s0 / s1 | max beta_eff s0 / s1 | zero_clear F s0 / s1 | shuffle F s0 / s1 |
|---|---|---|---|---|---|---|---|
| base_k8o8_60ep (none) | 60 | 0.653 / 0.609 | 0.715 / 0.679 | 0.765 / 0.823 | 6.5 / 5.6 | 0.639 / 0.590 | 0.642 / 0.595 |
| BEST_60ep (cos16_k6o8) | 60 | **0.846** / 0.562 | **0.842** / 0.597 | 0.868 / 0.690 | **18.4** / 1.2 | **0.614** / 0.561 | **0.604** / 0.558 |
| cos16_k6o8 | 12 | 0.566 / 0.570 | 0.665 / 0.640 | 0.571 / 0.576 | -0.8 / 0.0 | 0.563 / 0.569 | 0.561 / 0.569 |
| cos16_k8o8 | 12 | 0.611 / 0.523 | 0.698 / 0.614 | 0.571 / 0.667 | 4.0 / 2.4 | 0.612 / 0.522 | 0.611 / 0.523 |
| temp_k6o8 / temp_k8o8 | 12 | 0.537 / 0.567 ; 0.534 / 0.587 | 0.631 / 0.647 ; 0.648 / 0.677 | 0.398 / 0.469 ; 0.461 / 0.511 | 1.6 / 0.8 ; 3.8 / 1.1 | = unablated ±0.003 | = unablated ±0.003 |
| qk6_k6o8 / qk6_k8o8 | 12 | 0.515 / 0.534 ; 0.515 / 0.582 | 0.594 / 0.601 ; 0.568 / 0.624 | 0.383 / 0.330 ; 0.529 / 0.624 | 6.2 / 4.0 ; 7.7 / 19.7 (top-1 0.05) | = unablated ±0.016 | = unablated ±0.016 |
| BEST_lr3e-3 / base_k8o8_lr3e-3 | 12 | 0.495 / 0.538 ; 0.497 / 0.553 | 0.574 / 0.630 ; 0.484 / 0.549 | 0.668 / 0.691 ; -0.325 / 0.611 | 2.2 / 1.0 ; 2.9 / 2.7 | = unablated ±0.011 | = unablated ±0.009 |
| BEST_psub64 / base_k8o8_psub64 | 12 | 0.537 / 0.489 ; 0.504 / 0.510 | 0.684 / 0.601 ; 0.625 / 0.657 | 0.547 / 0.537 ; 0.466 / 0.519 | 0.3 / 1.2 ; 3.2 / 0.5 | = unablated ±0.002 | = unablated ±0.002 |

**Replicate floor:** grid-3 base cells vs grids 1-2, |Δ ALLQ| 0.0296 / 0.0009 / 0.0006.

**The seed split, read from the checkpoints and the per-head diagnostic** (probe 5 reproduces the logged beta_eff from every checkpoint within 0.5):
- **Seed 0.** BEST's L1 head 1 held plant mass 0.985-0.9999 from cycle 3 on. Its beta_eff ran -0.88, 0.07, 2.70, 6.62, 18.36 at cycles 3/6/9/12/15, and F ran 0.659, 0.603, 0.599, 0.675, 0.846. The run was still accelerating at the end.
- **Seed 1.** No head was committed: the max plant mass was 0.64, with beta -2.27. R^2 0.690 survives no_plant (0.680), so it is read from the visual ctx.
- **Per-learner loss curves do not exist.** The sealed trainer logs only a pooled last-batch mean over learners. Probe 5 substitutes final per-learner TRAIN fits, taken from the checkpoints.

Next: grid 5 (`PREREG_GRID5.md`, sha256 in `PREREG_GRID5.md.sha256`):
- cos16_k8o8, cos16_k8o8 + AdamW wd 1.0, BEST and base, all at 120 ep on seeds 0 and 1 (plus seed 2 for the two candidates);
- the committed fp32-attention arm at lr 3e-3.

## 7. C-REACH evidence, both seeds (PENDING)
## 8. Pinned files and new pins (PENDING)
## 9. Power (Amendments 2 / 3)
The A1 power arithmetic does not read the training recipe: `sh5_design_calc.py` (mde, plant) uses PRIVGEOM-SAFE's banked arrays and an IDEAL reader; `sh5_amend2_power.py` / `sh5_amend3_power.py` read `mde.json`, the plan and the banked CPU false-fire rows, whose own config is hard-coded in `sh5_sim_falsefire.py:28-38` (d 32, fourier_k 3, CPU lr). A change of lr, epochs or query octaves for the real rig therefore leaves every number of Amendments 2 and 3 unchanged by construction; what it can change is the real rig's TRAINING variance, which those amendments already list as UNMEASURED (`H-ESTIM-SEED-1`). PENDING: re-run of `sh5_design_calc.py mde` to show it reproduces `work/mde.json` byte-identically.

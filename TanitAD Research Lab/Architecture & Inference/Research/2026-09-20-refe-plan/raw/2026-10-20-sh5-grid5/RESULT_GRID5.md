# RESULT GRID 5: SH5 C-REACH diagnosis, read against `PREREG_GRID5.md` (Research Lab, 2026-10-20 ~03:50 Berlin)

## Headline
- **Outcome (PREREG_GRID5 s6, tested in its order): G1 FAILED, G3 not met, G2 not met, G4 not met, so the outcome is "Otherwise" = INSTRUMENT DEFECT, as the prereg names it.** `D.passing_cells({G5_s0, G5_s1}) = []`, re-run on the pinned `diag_pre3.py`.
- Descriptive caveat, which does not change the verdict: **two cells reach pass-grade, each on ONE seed and never the same cell on both.**
  - `BEST_120ep` (cos16_k6o8) passes on seed 0 (F 0.9706, ALLQ 0.9793, R2 0.9755) and has no committed head on seed 1 (F 0.524).
  - `cos16_k8o8_wd1_120ep` passes on seed 1 (F 0.9931, ALLQ 0.9921, R2 0.9725) and sits at F 0.808 on seed 0.
  - So the defect reads as **per-seed reach of the plant, not an unreadable plant**: the instrument can read it, and no single recipe does so on both seeds.
- **The committed next arm (visual-ctx dropout on ALL arms) is PARKED** behind the MM's SH5 deprioritisation (section 7). Nothing was run for it.
- **The fp32 control (F12) did NOT run: UNAVAILABLE** (section 6). The wd attribution is INCOMPLETE because seed 2 crashed.

## 0. Stamps
- Evidence: **MEASURED** on the planted-control arm of the SH5 rig (`plant`, fold 0 = `fit --stage PRE --folds 0`, n_sets 322, the LOCAL sets). **SIMULATED: no.** Tier: diagnosis only (not T0/T1/T2, no driving metric).
- Estimator: point values, **one training run per (cell, seed), no CI claimed** (as the prereg says). The question each number answers is "does THIS run of THIS recipe read the plant?". It does not answer "would another training run?" (H-ESTIM-SEED-1). Run-to-run floor: 0.030 ALLQ (grids 1-2 vs 3).
- Four families: this is an instrument check. LATERAL (DAC boundary clearance, planted) is the only family touched. LONGITUDINAL and TACTICAL n = 0. STRATEGIC = PAUSED (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` ITEM 25).
- Sources: `work/diag5/diag_G5_s{0,1,2}.json`, `work/diag5/diag_summary.json`, `work/diag5.log`, `work/diag5f.log`, `raw/probe5b_grid5_ckpt*.json`. Derived: `raw/grid5_cells.json`, `raw/grid5_tables.txt`, `raw/grid5_controls.json|txt` (scripts `code/read_grid5.py`, `code/read_grid5_b.py`).
- Pins: `sha256sum -c PREREG_GRID5.md.sha256` OK on all 11 files, re-run at read time (`raw/grid5_passing_crosscheck.txt`). The in-job `ZZDIAG3_PASSING` marker never printed, because the job crashed. PASS_REG was computed offline with the pinned code.

## 1. The outcome, test by test (literal thresholds)
**G1 (PASS_REG): not met.** F >= 0.97 AND ALLQ >= 0.95 AND R2 >= 0.50 at c30, and zero_clear AND shuffle_clear each ALLQ <= 0.80 AND F <= 0.80, on seeds 0 AND 1.
| cell | s0 | s1 |
|---|---|---|
| cos16_k8o8_120ep | fails (F 0.618) | fails (F 0.618) |
| cos16_k8o8_wd1_120ep | fails (F 0.808, ALLQ 0.847) | **passes** |
| BEST_120ep | **passes** (F 0.9706, a margin of +0.0006 on the F bar) | fails (F 0.524) |
| base_k8o8_120ep | fails (F 0.585) | fails (F 0.649) |

**G3 (sharp committed head on BOTH seeds, F < 0.97): not met.** A committed head = plant mass >= 0.8, sharp = beta_eff >= 16 AND top-1 >= 0.5.
- `BEST_120ep`:
  - s0: L1h1 beta 138.6, top-1 1.00, sharp from c15 (beta 57.1, top-1 0.76).
  - s1: the max plant mass at ANY eval is 0.41, and beta <= 0.3 at c30.
- `wd1`:
  - s1: L0h0 beta 105.3, top-1 0.99, sharp from c15 (beta 82.7, top-1 0.91).
  - s0: L0h2 has beta 34.1 but top-1 **0.15**, i.e. a sharp-in-scale head that does not land on the key.
- `cos16_k8o8` and `base`: no head reaches beta 16 on either seed (max beta any head <= 4.9 for cos16 s0, 2.3 for cos16 s1, 8.1 for base s0, 9.7 for base s1).
- Robust to the reading: the result is the same (a) at c30 only, (b) at any eval, (c) with the plant-mass condition dropped (`raw/grid5_controls.txt`, "G3 SENSITIVITY").

**G2 (committed head on both seeds AND F(c30) - F(c24) >= +0.03 on both): not met.**
| cell | dF s0 | dF s1 |
|---|---|---|
| cos16_k8o8 | -0.006 | -0.001 |
| wd1 | -0.012 | +0.003 |
| BEST | +0.018 | -0.020 |
| base | -0.031 | +0.008 |

No cell is above +0.03 on either seed. (wd1 is the only cell committed on both seeds at c30.)

**G4 (seed 1 commits no head in ANY cell at ANY eval): not met.** Seed 1 commits a head in cos16_k8o8 (c9, c18, c24, c27, c30), in wd1 (every eval) and in base (every eval). It commits none only in BEST_120ep.

## 2. Per-cell, per-seed table at c30 (the final cycle; 120 ep, 12,359 steps)
"max-mass head" = the head with the largest plant mass at c30: (mass, beta_eff, top-1). Ablation columns are F / ALLQ.
| cell | seed | F | ALLQ | R2 | zero_clear | shuffle_clear | max-mass head | F c24 | dF c30-c24 | best F (cycle) | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cos16_k8o8_120ep | 0 | 0.618 | 0.638 | 0.621 | 0.615 / 0.636 | 0.615 / 0.636 | L0h2: 0.64, 4.9, 0.05 | 0.624 | -0.006 | 0.626 (c6) | no |
| cos16_k8o8_120ep | 1 | 0.618 | 0.653 | 0.716 | 0.614 / 0.651 | 0.616 / 0.653 | L1h2: 0.99, 1.4, 0.03 | 0.619 | -0.001 | 0.632 (c21) | no |
| cos16_k8o8_wd1_120ep | 0 | 0.808 | 0.847 | 0.907 | 0.507 / 0.533 | 0.476 / 0.527 | L0h2: 0.97, 34.1, 0.15 | 0.820 | -0.012 | 0.820 (c24) | no |
| cos16_k8o8_wd1_120ep | 1 | 0.993 | 0.992 | 0.972 | 0.550 / 0.499 | 0.518 / 0.510 | L0h0: 1.00, 105.3, 0.99 | 0.990 | +0.003 | 0.993 (c30) | **YES** |
| BEST_120ep | 0 | 0.971 | 0.979 | 0.976 | 0.559 / 0.549 | 0.539 / 0.541 | L1h1: 1.00, 138.6, 1.00 | 0.952 | +0.018 | 0.971 (c30) | **YES** |
| BEST_120ep | 1 | 0.524 | 0.597 | 0.611 | 0.523 / 0.596 | 0.523 / 0.596 | L1h1: 0.33, 0.3, 0.02 | 0.544 | -0.020 | 0.575 (c21) | no |
| base_k8o8_120ep | 0 | 0.585 | 0.690 | 0.775 | 0.576 / 0.687 | 0.578 / 0.689 | L0h0: 0.74, 8.1, 0.17 | 0.616 | -0.031 | 0.624 (c21) | no |
| base_k8o8_120ep | 1 | 0.649 | 0.721 | 0.857 | 0.621 / 0.698 | 0.618 / 0.701 | L0h0: 0.95, 9.6, 0.14 | 0.641 | +0.008 | 0.655 (c18) | no |

- **Seed 2 is PARTIAL (the first two cells only, to c12; no final, no checkpoint).** It is not used for any verdict.
  - cos16_k8o8: F 0.576, ALLQ 0.642, R2 0.822. Max-mass head L1h0 (mass 0.99, beta 4.0, top-1 0.04).
  - wd1: F 0.538, ALLQ 0.610, R2 0.762. Max-mass head L1h0 (mass 0.99, beta 1.5, top-1 0.02). L0h3 has beta 14.6 and top-1 0.17 and is rising (8.8, 12.1, 14.6 at c6, c9, c12).
- **Secondary reading (never a pass): the first eval cycle that clears F, ALLQ and R2, reported WITHOUT ablations.**
  - `BEST_120ep` s0: c30 (c27 reads F 0.962).
  - `wd1` s1: **c15** (F 0.977, ALLQ 0.979, R2 0.960). No other (cell, seed) ever clears.
- Per-cycle curves with every committed head: `raw/grid5_tables.txt`.

## 3. How solid are the two single-seed passes? (context; the prereg rule is final-cycle only and is applied as written)
- **BEST s0** clears the F bar only at c30. F by cycle: 0.908 / 0.924 / 0.936 / 0.952 / 0.962 / **0.971**. The margin on the F bar is +0.0006, against a rig floor of up to 0.030, so a replicate could fall below it.
- **wd1 s1 is not monotone.** The three level bars are cleared at c15, c18, c21, c24, NOT at c27 (F 0.786, ALLQ 0.840, R2 0.903, with the sharp head dropping to beta 78.3 / top-1 0.70 and four committed heads), and cleared again at c30. The final-cycle rule happens to land on a good eval. A one-cycle swing of 0.2 F is exactly the kind of instability that would hurt a real arm trained on this recipe.
- Longer training alone does nothing for the plain cells. Plain `cos16_k8o8` goes from F 0.611 / 0.523 (12 ep, grid 3, s0 / s1) to 0.618 / 0.618 (120 ep). `base_k8o8` stays at or below F 0.655 at every eval.

## 4. Replicate controls (PREREG_GRID5 s4; |d| against the 0.030 floor, which was measured on ALLQ, so applying it to F is an extension)
The reference values quoted in the prereg were re-read from the raw JSON (`diag4/diag_W60_s*.json`, `diag3/diag_S12_s*.json`) and all 6 match (`raw/grid5_controls.txt`).
| control | seed | grid 5 (F / ALLQ) | reference (F / ALLQ) | dF | dALLQ |
|---|---|---|---|---|---|
| BEST_120ep c15 vs W60 BEST | 0 | 0.908 / 0.924 | 0.846 / 0.842 | +0.062 (beyond) | +0.081 (beyond) |
| BEST_120ep c15 vs W60 BEST | 1 | 0.538 / 0.600 | 0.562 / 0.597 | -0.024 (within) | +0.003 (within) |
| base_k8o8_120ep c15 vs W60 base | 0 | 0.587 / 0.692 | 0.653 / 0.715 | -0.066 (beyond) | -0.023 (within) |
| base_k8o8_120ep c15 vs W60 base | 1 | 0.630 / 0.711 | 0.609 / 0.679 | +0.020 (within) | +0.031 (beyond by 0.001) |
| cos16_k8o8_120ep c3 vs grid 3 | 0 | 0.582 / 0.669 | 0.611 / 0.698 | -0.029 (within) | -0.029 (within) |
| cos16_k8o8_120ep c3 vs grid 3 | 1 | 0.522 / 0.646 | 0.523 / 0.614 | -0.001 (within) | +0.032 (beyond by 0.002) |
| BEST_lr3e-3_rep vs W12 (F12) | 0, 1 | **UNAVAILABLE** | 0.495 / 0.574, 0.538 / 0.630 | n/a | n/a |

- **C1, the one pre-registered rule: the W60 seed-0 transition is REPRODUCED.** The rule says s0 at c15 reading F < 0.75 means "NOT reproducible", and it reads 0.908. The prereg's rule is one-sided, so the +0.062 above W60 is not a failure. UNVERIFIED: whether c15 of a 30-cycle run is the same training state as the end of W60 (it depends on the lr schedule), so the size of that difference is not interpreted. The seed-1 non-commit also replicates (F within the floor, max plant mass 0.41 at any eval).
- C2 and C3 carry no pre-registered rule. Two of the ALLQ differences sit 0.001-0.002 above the floor (an edge case, not a clear departure). One F difference (base s0, -0.066) is clearly beyond it. They are reported, not interpreted.
- **The fp32 control did NOT run, so the fp32 lever is NEITHER "a lever" NOR "ELIMINATED".** The prereg's elimination clause needs the BEST_fp32 vs BEST_rep comparison on both seeds, which does not exist. The pre-launch CPU prediction ("no fp32 effect above the floor", `smoke_diag5_cpu.py`) is therefore UNTESTED. Do not substitute anything for it.

## 5. Mechanism: probe 5b (H5), CPU fp32, the 8 grid-5 checkpoints
`code/probe5b_grid5_ckpt.py` is a COPY of the pinned `probe5_ckpt_seedsplit.py` with only the checkpoint list and the docstring head changed. 48 train tokens and the first 24 fold-0 tokens. Cells with K6 (BEST) and K8 use their own K. **`attn_reproduces` is true for all 8** (tolerance |d beta| <= 0.5), so each checkpoint IS the evaluated learner.
| learner | train F | zero_clear | shuffle_clear | no_plant | reads the plant on train? |
|---|---|---|---|---|---|
| s0 cos16_k8o8 | 0.989 | 0.990 | 0.989 | 0.981 | no |
| s0 cos16_k8o8_wd1 | 0.927 | 0.802 | 0.744 | **0.644** | yes (partly) |
| s0 BEST | 0.996 | 0.762 | 0.694 | **0.580** | yes |
| s0 base | 0.970 | 0.969 | 0.972 | 0.952 | no |
| s1 cos16_k8o8 | 0.994 | 0.994 | 0.994 | 0.979 | no |
| s1 cos16_k8o8_wd1 | 0.997 | 0.409 | 0.539 | **0.412** | yes |
| s1 BEST | 0.990 | 0.990 | 0.991 | 0.980 | no |
| s1 base | 0.977 | 0.977 | 0.976 | 0.959 | no |

- **Every learner that reads the plant on TRAIN is one of the three whose committed head has beta_eff >= 16** (BEST s0, wd1 s0, wd1 s1; wd1 s0 is sharp in scale but not on the key, top-1 0.15). All five others fit train at F >= 0.95 with the plant removed, i.e. through the visual ctx. This is the grid-5 form of H5 (memorisation competes with the plant read) and it is **observational, n = 8 checkpoints, 48 train tokens: HYPOTHESIS-grade, not a test.**
- **wd attribution (PREREG s6) is INCOMPLETE, not confirmed.** The prereg needs an effect beyond the floor on seeds 0, 1 AND 2.
  - Seeds 0 and 1 meet it: wd1 minus plain at c30 is dF +0.190 / +0.375, dALLQ +0.209 / +0.339.
  - And the necessary H5 half is met on both: wd1's train F under no_plant is 0.644 / 0.412 against 0.981 / 0.979 for the plain cell, so wd lowers the visual-ctx fit.
  - Seed 2 stopped at c12, where wd1 trails the plain cell (dF -0.039, against +0.214 / +0.144 on seeds 0 / 1 at c12). The seed-2 half is unknown, and **the wd effect is not established across seeds.**

## 6. The crashes, their cause, and what they cost
| job | when | what the log says (MEASURED) | cause |
|---|---|---|---|
| `grid5L` seed 2 (`G5_s2`) | 2026-10-20 03:30:48, 712 s in, after the c12 eval of its first two cells | `numpy _ArrayMemoryError: Unable to allocate 2.25 MiB (1152, 256) float64`, in `sh5_branch.plant_block` <- `plant_rows` <- `_batch` <- `_train_step`. `ZZGPURELEASED yes`; `diag_G5_s2.json` is a partial (curve to c12, no `final`, no checkpoint) | A host-commit limit with C: full, so the pagefile cannot grow (the MM's diagnosis, **INHERITED**; consistent with grid-3 attempt 1 and the PREREG s7 risk note). The 2.25 MiB request is a victim, not a cause. |
| `grid5F` (`F12_s0`, `F12_s1`) | ~03:32, the chained `&` job | after `cycle 1/3 shard 1/2: 709 tokens loaded in 88 s`: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB ... 3.58 GiB is free ... 3.33 GiB allocated by PyTorch`, in `diag_attn5.py:32` (`ln_q`). No eval ran; `work/diag5f/` holds no JSON | **Not a plain VRAM shortfall** (a 20 MiB request failing with 3.58 GiB free). The MM attributes it to the same host-commit exhaustion; INHERITED, plausible, not independently reproduced. |

- Cost: G5_s0 and G5_s1 are complete and banked (all 4 cells, 30 cycles each, with ablations at c30). Lost: seed 2 (the ROBUST test, moot after the PASS_REG failure) and the whole fp32 control.
- At read time C: had 0.1-0.4 GB free and host free virtual memory was ~22 GB, so the same failure is reproducible whenever C: fills. Pagefile on a full system drive kills long GPU jobs at an arbitrary small allocation; the traceback names the allocation, not the cause.

## 7. Committed next arm for this outcome: **PARKED**
- **What the prereg commits (s6 "Otherwise", same as G4):** a **visual-ctx dropout regulariser applied to ALL arms** (the anti-memorisation lever H5 names). Then a supervised attention target on the plant, as a diagnostic only (not yet implemented).
- **Why it is parked (MM ruling):** SH5 LOCAL / FULL are deprioritised behind the search line, because PRIVGEOM-CLEAN measured the held-out proximity complement at about +0.010 AUC (+0.0107 [+0.0004, +0.0252] seed 0; +0.0101 [+0.0001, +0.0244] seed 1), below the +0.02 bar (`D-REFE-PRIVGEOM-CLEAN-1`). Nothing in grid 5 makes SH5 cheaper or more urgent. It adds a reason to be careful: the two recipes that read the plant do so on one seed each, so a real SH5 arm on either could be NO-CALL on the other seed.
- **Unblock condition (exact; for the MM to rule on):**
  1. The MM records that SH5 is un-parked, either because a proximity complement of >= +0.02 fast-band within-set AUC, with the lower CI above 0, is shown on held-out logs (the bar PRIVGEOM-CLEAN missed), or by an explicit MM / PI decision to fund SH5 regardless.
  2. `sh5_gpulock` owner `sh5_diag` is available. The dev-box GPU belongs to another session's battery, so this is not mine to take.
  3. C: has headroom before launch (proposed >= 2 GB free and a host-commit check). Two grid-5 jobs died on a full C:.
  4. `PREREG_GRID6.md` plus its sha256 is written before any number. Proposed content, which the prereg itself must fix:
     - dropout in a new pinned module;
     - live-lever tests, i.e. a non-zero gradient on the dropout path and an exact-0 control;
     - a RED test that refuses the no-dropout recipe;
     - all four grid-5 recipes as arms, seeds 0, 1, 2 (SEEDS, NOT WINDOWS), 120 ep, plus a replicate arm;
     - the F12 fp32 control re-run as a separate cheap arm (12 ep x 2 seeds, same lock).
- **Not done, not blocked on compute alone:** the amend4 draft (`PREREG_SH5_DAC_PROX_AMENDMENT4.draft.md`) s6-s8 are NOT filled, because G1 did not fire, so there is no recipe to seal. Proposed one-paragraph addition for its s6 is in section 8.
- RULE ZERO check: the cheapest discriminating item that could still run on CPU was done (probe 5b). Every remaining lever needs the GPU lock, C: space, or an MM un-parking decision, and each is named above.

## 8. Proposed register rows (TEXT ONLY; no register was edited)
**GOALS_AND_CLAIMS.md**, a new row (grids 1-4 have no GAC row yet; this one carries them):

`| **D-SH5-CREACH-GRID5-1** (2026-10-20) | **SH5 C-REACH grid 5 reads OTHERWISE (INSTRUMENT DEFECT, PREREG_GRID5 s6): G1 failed, G3 / G2 / G4 not met, so no recipe has read the planted channel on both seeds. Two cells reach pass-grade on ONE seed each, never the same cell on both: BEST_120ep (cos16_k6o8, 120 ep) passes on seed 0 (F 0.9706, ALLQ 0.9793, R2 0.9755, zero_clear / shuffle_clear collapse to F 0.559 / 0.539) with no committed head on seed 1 (F 0.524); cos16_k8o8_wd1_120ep passes on seed 1 (F 0.9931, ALLQ 0.9921, R2 0.9725) and reads F 0.808 on seed 0.** Neither pass is robust (BEST s0 clears the F bar by +0.0006; wd1 s1 drops to F 0.786 at c27). The W60 seed-0 transition REPRODUCES (c15 F 0.908, prereg rule threshold 0.75). Longer training alone moves nothing (plain cos16_k8o8 F 0.618 / 0.618, base <= 0.655). Probe 5b: every learner that reads the plant on TRAIN is a learner whose committed head has beta_eff >= 16 (train F under no_plant 0.580 / 0.644 / 0.412); the other five fit train at >= 0.95 without the plant. UNAVAILABLE: the fp32 control F12 (CUDA OOM at ~03:32, host-commit, C: full) and seed 2 (numpy MemoryError at 03:30:48, partial to c12), so the fp32 lever is neither a lever nor eliminated and the wd attribution is incomplete (needs seeds 0, 1, 2). MEASURED on the planted-control arm of the SH5 rig, fold 0, n_sets 322, diagnosis tier, one run per (cell, seed), no CI (H-ESTIM-SEED-1); LATERAL only, STRATEGIC PAUSED. Package `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-20-sh5-grid5/` (`RESULT_GRID5.md`). **Committed next arm (prereg s6): visual-ctx dropout on ALL arms, then a supervised plant-attention target as a diagnostic. PARKED** behind the search line (`D-REFE-PRIVGEOM-CLEAN-1`); unblock in `RESULT_GRID5.md` s7. | MEASURED (diagnosis tier) |`

A hypothesis row (so H5 has an ID and a status):

`| **H-SH5-CREACH-RACE-1** (2026-10-20) | HYPOTHESIS H5: whether an attention head commits to the plant before memorisation through the visual ctx saturates is a per-seed event, and weight decay (decoupled, 1.0) shifts the race toward the plant. STATUS: PARTIAL, INCOMPLETE. For: wd1 lowers train F under no_plant (0.644 / 0.412 vs 0.981 / 0.979, seeds 0 / 1), and the dF at c30 is +0.190 / +0.375 on seeds 0 / 1. Against / unknown: seed 2 stopped at c12 where wd1 trails the plain cell (dF -0.039), so the prereg's "all of seeds 0, 1, 2" condition is not met, and probe 5b is n = 8 checkpoints, 48 train tokens, observational. It becomes supported or refuted only on seeds 0-2 under a new prereg. | HYPOTHESIS |`

**BACKLOG.md**, an UPDATE for the `SH5-DAC-PROX` row:
`UPDATE 2026-10-20 ~04:00: C-REACH grid 5 DONE for seeds 0 and 1 (seed 2 and the fp32 control F12 crashed on a full C: / host commit): outcome "Otherwise" = INSTRUMENT DEFECT (PREREG_GRID5 s6). BEST_120ep passes on seed 0 only, cos16_k8o8_wd1_120ep on seed 1 only; no cell passes on both (D-SH5-CREACH-GRID5-1). The amend4 draft stays unsealed. Next arm (visual-ctx dropout for all arms) is PARKED behind the search line; unblock in RESULT_GRID5.md s7.`

and a new row:
`| SH5-CREACH-GRID6 | **PARKED** (MM: SH5 behind the search line, D-REFE-PRIVGEOM-CLEAN-1). Visual-ctx dropout on ALL arms (base, BEST, cos16_k8o8, wd1) x seeds 0-2 x 120 ep + replicate + the F12 fp32 control; own PREREG + sha256 before any number. Unblock: MM un-parks SH5 (a >= +0.02 proximity complement with lower CI > 0, or an explicit decision) AND the sh5_gpulock is available AND C: >= 2 GB free. CPU-only items already done: probe 5b. |`

A TRAPS candidate (one line): "A job on a full system drive dies at an arbitrary tiny allocation (numpy 2.25 MiB, CUDA 20 MiB with 3.58 GiB free): the traceback names the victim allocation, not the host-commit cause. Check `Get-PSDrive C` before blaming the code or the GPU."

**Proposed s6 paragraph for the amend4 draft** (the draft is not edited):
`Grid 5 (PREREG_GRID5, MEASURED, fold 0): no cell passes on both seeds. BEST_120ep (cos16_k6o8, 120 ep) passes on seed 0 and cos16_k8o8_wd1_120ep passes on seed 1; the other cells and seeds do not commit or do not sharpen. No recipe is picked for the amend4 implementation. s7 and s8 stay PENDING.`

## 9. What I did not do, and the deliverable
- Did not train, relaunch, take any lock, touch a pod, run S / Snoprox / Sshuf, run `analyze`, open navtest, or edit a register, a pinned file or `HANDOFF.md`.
- Ran on CPU only: `read_grid5.py`, `read_grid5_b.py`, `probe5b_grid5_ckpt.py` (8 checkpoints; ran as 7 + 1 because the first call hit its 270 s timeout after 7), and one import of the pinned `diag_pre3` for the cross-check.
- Checkpoints (`work/diag5/ckpt/*.pt`, 8 files, 6.9 MB each) are NOT staged and live on the dev box only (`C:/Users/Admin/par18/sh5diag/work/diag5/ckpt/`). That is deliberate, and they exist in that one place only.
- Grids 3-4 (and grid 5's prereg, pins, code, early logs) are ALREADY on the tip at `a8eeef6`: 89 of the 104 candidate files are byte-identical to the tip's `raw/2026-10-19-sh5-creach-diag/` (`raw/tip_compare_grid5.json`), so they are not re-staged. Only the 14 files not on the tip and the one that differs (`work/diag5.log`, 97,445 B against the tip's 350 B stub) are staged, with `RESULT_GRID5.md`, `HANDOFF_AFTER_GRID5.md` and four derived files (`raw/grid5_tables.txt`, `raw/grid5_controls.txt`, `raw/grid5_passing_crosscheck.txt`, `raw/tip_compare_grid5.json`): 21 items in all. No log exceeds 2 MB, so none is truncated.
- Manifest: `out/paths_grid5.json` (`[repo path, abs src, "raw"]`), repo dir `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-20-sh5-grid5/`.

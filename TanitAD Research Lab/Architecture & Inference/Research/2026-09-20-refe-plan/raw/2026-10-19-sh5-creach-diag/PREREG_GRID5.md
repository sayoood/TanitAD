# PREREG GRID 5: SH5 C-REACH, follow the seed-0 transition on both seeds (Research Lab, 2026-10-20, written BEFORE any grid-5 number)

Diagnostic grid, NOT a registered SH5 stage. It trains only the planted control (`plant`) on fold 0 (= `fit --stage PRE --folds 0`) via `code/diag_pre5.py`. That driver is `diag_pre3.main`, unchanged and sha256-pinned by PREREG_GRID3, with two module-global swaps. No real arm, no `analyze`, no navtest.
- Tier: none. This is an instrument check on fold 0 of the LOCAL SH5 sets.
- Estimator: point values on fold 0 (n_sets 322). Each number answers "does THIS training run of THIS recipe read the plant?"
- Run-to-run noise comes from the seeds and the replicate controls below (H-ESTIM-SEED-1). No CI is claimed.

## 1. What grids 3-4 measured (MEASURED; `work/diag3|4/diag_summary.json`, `work/diag4/diag_W60_s*.json`, `raw/grid34_cells.json`, `raw/probe5_ckpt_seedsplit.json`)
- **No grid-3 or grid-4 cell passes** (`ZZDIAG3_PASSING []` in both logs).
  - Grid 3 is **O3** of PREREG_GRID3: no lever cell reaches beta_eff >= 16 on BOTH seeds with top-1 >= 0.5. The only beta >= 16 is qk6_k8o8 s1 L0 19.7, with top-1 0.05.
  - Grid 4 also fails, so PREREG_GRID3's committed next arm is **fp32 attention on BEST at lr 3e-3**. It runs here as F12.
- **The one large effect, on ONE seed:** W60_s0 BEST_60ep (cos16_k6o8, lr 3e-4, 60 ep) reads F 0.846 / ALLQ 0.842 / R^2 0.868 at cycle 15.
  - The read is **position-matched**, the first in this series. zero_clear gives F 0.614 / ALLQ 0.673. shuffle_clear gives F 0.604 / ALLQ 0.670. Both are <= 0.80. no_plant gives R^2 -0.030.
  - Mechanism: ONE head (L1 h1) committed to the plant by cycle 3 (plant mass 0.985). Its beta_eff then climbed -0.88 / 0.07 / 2.70 / 6.62 / **18.36** at cycles 3 / 6 / 9 / 12 / 15, with top-1 0.31 at the end. It was still accelerating when the run stopped.
- **Seed 1, same recipe:** F 0.562 / ALLQ 0.597.
  - No head commits: the max plant mass is 0.64 (L1 h0), with a NEGATIVE beta of -2.27.
  - R^2 0.690 survives no_plant (0.680) and zero_clear (0.680). Seed 1 fits from the visual ctx and ignores the plant.
- **The checkpoints are the evaluated learners.** Probe 5 re-derives the logged final beta_eff from all 6 checkpoints within 0.5 (`attn_reproduces` true ×6).
- **Probe 5 (CPU, 32 train / 24 fold-0 tokens): every learner fits TRAIN without the plant, i.e. it memorises through the visual ctx.**

  | learner | train F | train F, no_plant | train F, zero_clear |
  |---|---|---|---|
  | W60 BEST s0 | 0.972 | 0.882 | 0.930 |
  | W60 BEST s1 | 0.967 | 0.949 | 0.967 |
  | W60 base s0 | 0.886 | 0.884 | 0.893 |
  | W60 base s1 | 0.933 | 0.886 | 0.930 |
  | S12 cos16_k8o8 s0 | 0.868 | 0.840 | 0.869 |
  | S12 cos16_k8o8 s1 | 0.833 | 0.834 | 0.832 |

  The plant read is the only path that generalises to fold 0, and it competes with memorisation, which is learned faster. Once train is fitted, little gradient is left for the plant read. Hence the **mechanism hypothesis H5**: *whether a head commits to the plant before memorisation saturates is a per-seed event, and the plant read then sharpens slowly (a grokking-type transition).*
- **Commitment is seed-robust with K8 queries.** Grid-3 cos16_k8o8 (12 ep) commits a head on BOTH seeds from cycle 1:
  - s0 L0 h2: plant mass 0.845 / 0.856 / 0.855.
  - s1 L1 h2: plant mass 0.876 / 0.969 / 0.932, with beta 0.90 / 1.83 / 2.40, rising.
  - cos16_k6o8 commits only on s0.
- Probe 2: the ideal K8 reader passes at beta 16 (F 0.988 / ALLQ 0.983). The ideal K6 reader does not (0.967 / 0.953) and needs beta ~64.
- Base (dot-product) attention commits on both seeds but does NOT sharpen: the L0 h0 beta plateaus at 5.6-6.5 over 60 ep, and its read is token-level (shuffle_clear keeps R^2 0.71 / 0.76).
- Replicate floor, grid 3 vs grids 1-2:
  - base_k6o8 s0: 0.6193 vs 0.6489, |Δ ALLQ| 0.0296.
  - base_k6o8 s1: 0.6742 vs 0.6732, 0.0009.
  - base_k8o8 s0: 0.6461 vs 0.6467, 0.0006.
  - **The rig's run-to-run floor is up to 0.030 ALLQ.**
- Pre-launch CPU check (`code/smoke_diag5_cpu.py`, SMOKE5_OK, random weights and ctx, not an outcome):
  - Attention-only fp32 changes the output by up to 5.9e-3.
  - It cuts the mean error vs an all-fp32 forward only from 1.677e-3 to 1.665e-3, because the projections stay bf16.
  - ⇒ **prediction: F12 shows no fp32 effect above the floor.**

## 2. Grid 5 cells (lr 3e-4 unless stated, p_sub 32, ablations zero_clear / shuffle_clear / no_plant on every cell at the final cycle)
**Long runs** (`work/grid5L.json` -> `work/diag5/`): 120 ep = 30 cycles x 4, evaluated at cycles 3, 6, ..., 30. They run in this order: **G5_s1, G5_s0, G5_s2**.

| cell | the ONE change vs a tested cell | why |
|---|---|---|
| cos16_k8o8_120ep | grid-3 cos16_k8o8 trained 10x longer | commits on both seeds (sec. 1); the K8 ideal reader passes at beta 16 |
| cos16_k8o8_wd1_120ep | + AdamW, decoupled wd 1.0 on the weight matrices (not t_emb, biases, LN or tau) | H5: weight decay penalises the memorisation path. Cosine attention is scale-invariant in q/k, so wd cannot flatten the logits. Fixed by rule: time constant 1/(lr·wd) = 3,333 steps ≈ 1/4 of the 12,360-step run |
| BEST_120ep | W60 BEST (cos16_k6o8), 2x longer | trend continuation; its first 60 ep replicate W60 |
| base_k8o8_120ep | W60 base_k8o8, 2x longer | the no-lever control at matched epochs: is the lever the effect, or the epochs? |

G5_s2 runs only the first two cells (a third seed for the candidate cells: SEEDS, NOT WINDOWS).

**Committed O3 arm** (`work/grid5F.json` -> `work/diag5f/`): F12_s0 and F12_s1, 12 ep, evaluated at cycles 1-3.
- BEST_fp32_lr3e-3: attention in fp32, autocast off inside SDPA.
- BEST_lr3e-3_rep: the bf16 replicate of W12 BEST_lr3e-3, i.e. the paired control.

## 3. PASS (PREREG_GRID3's rule, unchanged: `diag_pre3.learner_pass`)
- **PASS_REG**: the cell passes on seeds 0 AND 1 at the FINAL cycle (30).
  - F >= 0.97 AND ALLQ >= 0.95 AND R^2 >= 0.50.
  - zero_clear AND shuffle_clear each read ALLQ <= 0.80 AND F <= 0.80.
  - It is computed by `passing_cells` on the {G5_s0, G5_s1} sub-summary.
- **ROBUST**: PASS_REG holds AND seed 2 also passes (the two candidate cells only).
- The in-job `ZZDIAG3_PASSING` marker includes seed 2 for those cells, so for them it is the ROBUST test.
- Secondary, never a pass: the first eval cycle at which F, ALLQ and R^2 clear the bar. It is reported without ablations and labelled that way.

## 4. Replicate controls (known values; report each |Δ| against the 0.030 floor)
- BEST_120ep at c15 vs W60 BEST: s0 F 0.846 / ALLQ 0.842, s1 0.562 / 0.597.
  - If s0 at c15 reads F < 0.75, the W60 s0 transition is classed as **NOT reproducible under run-to-run noise**. That is a miss of > 3x the floor.
- base_k8o8_120ep at c15 vs W60 base: s0 0.653 / 0.715, s1 0.609 / 0.679.
- cos16_k8o8_120ep at c3 vs grid-3: s0 0.611 / 0.698, s1 0.523 / 0.614.
- BEST_lr3e-3_rep at c3 vs W12: s0 0.495 / 0.574, s1 0.538 / 0.630.

## 5. Readouts (from the logged attention diagnostic; no new instrument)
- **Committed head**: any head with plant mass >= 0.8 at an eval.
- **Sharpening**: that head's beta_eff and top-1.
- **Memorisation**: probe 5 re-run on the grid-5 checkpoints after the job (train F under no_plant).

## 6. Outcomes and committed next arms (evaluated in this order)
- **G1: >= 1 cell is PASS_REG.**
  - Pick by fewest changes: base_k8o8 < BEST (cos16_k6o8) < cos16_k8o8 < cos16_k8o8_wd1.
  - Next: the amend4 implementation of that recipe (lever, FOURIER_K, 120 ep, optimizer) per HANDOFF steps 3-5, then `confirm_pre.py` on seeds 0,1.
  - If ROBUST fails, add seed 2 to the confirm and state the seed risk in the headline.
- **G3: no pass.** Some cell has a committed head with beta_eff >= 16 AND top-1 >= 0.5 on BOTH seeds 0 and 1, but F < 0.97.
  - Sharpness is learned and the readout binds.
  - Next: a softmin key-time pool on that cell (PREREG_GRID3's O2 lever).
- **G2: no pass.** Some cell has a committed head on both seeds 0 and 1 AND F(c30) - F(c24) >= +0.03 on both.
  - Still climbing.
  - Next: that cell at 240 ep, same grid otherwise.
- **G4: no pass, and seed 1 commits no head in ANY cell at ANY eval.**
  - Commitment binds. Report **INSTRUMENT DEFECT**.
  - Next: a visual-ctx dropout regulariser applied to ALL arms (the anti-memorisation lever H5 names).
  - Then the supervised attention target on the plant, as a diagnostic only. It is not yet implemented.
- **Otherwise:** **INSTRUMENT DEFECT** with each lever's best F / ALLQ / R^2 / beta_eff. The next lever is the visual-ctx dropout, as in G4.
- **fp32 (F12):**
  - fp32 counts as a lever only if BEST_fp32_lr3e-3 beats BEST_lr3e-3_rep in final F by more than 0.030 on BOTH seeds.
  - Otherwise fp32 at lr 3e-3 is ELIMINATED and logged.
- **wd attribution:**
  - cos16_k8o8_wd1 vs cos16_k8o8 on the same seed. It is an effect only beyond the floor on all of seeds 0, 1 and 2.
  - H5 is supported only if the wd cell ALSO lowers train F under no_plant (probe 5 re-run).

## 7. Cost and launch
- Estimated GPU time is ~2.6 h after the lock (ESTIMATED from W60: ~7.4 s per learner-epoch, including evals).
- Launch: `work/launch_diag5.ps1`, WMI detached, through the sealed `sh5_gpulock`, owner `sh5_diag`, max wait 6 h per grid.
- `grid5L` and `grid5F` are chained with `&`, so F12 runs even if the long runs fail.
- Risk: C: has ~0.9 GB free. Grid-3 attempt 1 died of a host MemoryError while C: was full.

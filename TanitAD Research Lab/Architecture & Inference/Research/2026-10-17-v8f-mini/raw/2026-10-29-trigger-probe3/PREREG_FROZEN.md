# PREREG_FROZEN -- trigger probe 3: does a FINER image grid (8x20 vs 4x10) carry the go/stay trigger?  (H-CRAWL-06 precursor, continued)

Zero training of the planner, dev-box CPU, `CUDA_VISIBLE_DEVICES=-1`. Written and hashed BEFORE any 8x20 feature, label statistic or probe output of this package is computed
(`FROZEN_SHA.txt` carries the sha256 of this file and the wall clock).  Pre-freeze actions, all structure-only: read probes 1/2 code + RESULTS, read `finetoken_scope/SCOPE.md`,
listed file names / array shapes / dtypes of `windows_train.npz`, `sample_*.npz`, `eval141_features.npz` and one cached clip, and timed sklearn fits on SYNTHETIC random data (no label, no image feature).

## 0. Question and what is held fixed
Probe 1 read the frozen DINOv3-L/16 `pool4x10` of the CURRENT frame through a single PCA-64 of the flattened 40,960-d vector and found a small image increment over ego (V+E - E: P-STOP +0.054, P-MOVE +0.053 linear).
The scope (`finetoken_scope/SCOPE.md`) established that the 8x20 pooling (2x2 token cells) is already cached for all 4,572 train clips and the eval-141 clips.
**This probe holds the READOUT fixed and varies ONLY the grid resolution**, both grids taken from the SAME cached 8x20 tensor: **G8** = the 8x20 grid (160 tokens x 1024); **G4** = its exact 2x2 average (40 tokens x 1024).
Same windows, same E, same folds, same bootstrap resamples as probes 1 and 2.  Nothing about the encoder, the trunk or the planner is touched.

## 1. Windows and samples [REUSED, asserted]
* The 12,000 P-STOP + 12,000 P-MOVE TRAIN windows of probe 1: `dagland/trigger_probe/raw/windows_train.npz` (sha256 asserted equal to `a9a57b32ffac558929b8f57e49a979826b88d17be17e5e9d7f9347e5875783ec`, the value in its `windows_provenance.json`)
  and `raw/sample_pstop.npz`, `raw/sample_pmove.npz` (`idx` into the 306,566 canonical windows; their sha256 is recorded in `raw/INPUTS_SHA256.json`).  GO = the registered launch mask (`v6 >= v0 + 1`), P-STOP v0 < 0.5, P-MOVE 0.5 <= v0 < 6.0.
* Folds: `outer = int(sha12,16) % 5` over the clip, `inner = (int(sha12,16)//5) % 3`, exactly `tp_lib.outer_fold/inner_fold` (`tp_lib.py` copied byte-identically; sha256 asserted equal to the probe-1 copy).  The holdout `540:600` is never opened.
* **E** = the 8 numbers O reads (`windows_train.npz["E"]`).  The E arm is NOT refitted: its out-of-fold scores are taken from probe 1's banked `raw/oof_<pop>_V_linear.npz["E"]` and `oof_<pop>_V_mlp.npz["E"]` (same rows, same folds; row order asserted via banked `y`, `clip`).

## 2. Features [PIN]
* Source: `F:/TanitAD-scratch/speedread/cache/train/clips/<clip>.pt`, key `pool8x20` `[T,160,1024]` fp8-e4m3, row `t` = the CURRENT cache step of the window (exactly as probe 1 read `pool4x10[t]`).  fp8 -> fp16 is exact, so the stored memmap `G8_<pop>.npy` (fp16, `[12000,163840]`, row-major 8x20 tokens, token index = r*20 + c) is LOSSLESS w.r.t. the cache.
  Written to `F:/TanitAD-scratch/trigger_probe3_scratch/` (E: is nearly full).  `T` must equal the manifest `n_frames` (probe-1 assert); a clip that fails raises.
* **G4 := the exact 2x2 average of G8** in float32 with the fixed order `(((a + b) + c) + d) * 0.25` over `(2r,2c),(2r,2c+1),(2r+1,2c),(2r+1,2c+1)`, giving a 4x10 grid, token index = r*10 + c.  **ASSERT (pytest + extraction):** this function equals an independent pure-Python loop implementation BIT-FOR-BIT
  on every checked row (max abs diff must be exactly 0.0).
* **Alignment / continuity gate (extraction):** for ALL 24,000 rows, relative L2 between G4 and probe 1's cached `pool4x10` row (`F:/TanitAD-scratch/aside_from_E/probe_scratch/trigger_probe_scratch/V_<pop>.npy`, probe 1's own memmap) is computed.  SCOPE measured 3.0 % mean (two fp8 round trips).
  **Gate: median <= 0.05, 99.9th percentile <= 0.15; any row > 0.5 REFUSES the package** (a misaligned row or a wrong token order reads ~1.0).
* **Readout (identical for G8 and G4) = per-token PCA to 16 dims:** ONE shared 1024 -> 16 linear map per grid, applied to every token position independently; centred, NOT scaled; exact eigendecomposition of the token covariance (float64) of a **token subsample**: per window 16 token positions drawn
  without replacement (`np.random.default_rng(1000 + population index)`: 160 candidate positions for G8, 40 for G4), i.e. about **153,600 tokens per outer-train fold (192,000 for the all-rows fit) for EACH grid**; fitted on the outer-TRAIN-fold windows only (the held-out fold never enters; the inner C / alpha choice reuses that PCA, an unsupervised step, stated).
  Six fits per grid and population (5 outer folds + 1 on all rows for the eval-141 confirmation and the weight map).  Features: **G8 = 160 x 16 = 2,560, G4 = 40 x 16 = 640**, token-major (token t, component j at column 16 t + j).
  (The brief allowed IncrementalPCA; an exact covariance eigendecomposition of the same subsample is the same estimator without the incremental approximation.)  `n` tokens, explained-variance ratios and sv are saved.
* Arms: **E** (banked), **G4**, **G8**, **G4+E**, **G8+E** (E appended, 8 columns), linear and MLP.  Continuity arm (beside, never decisive): probe 1's **V+E** (PCA-64 of the whole flattened cached 4x10, banked OOF scores).

## 3. Probes, CV, metric [PIN]
* **Linear:** `LogisticRegression(penalty=l2, solver=lbfgs, max_iter=2000)` after a `StandardScaler` fitted on the training rows of the fold; **C chosen from {1e-4, 1e-3, 1e-2, 1e-1, 1} by inner 3-fold (grouped by clip) mean AUC** on the outer-train rows, refit on all outer-train rows (`tp_lib.fit_probe` logic, grid changed: the probe-1 grid {1e-3..10} was built for 72 features; at 2,568 features the top decade is a near-interpolating fit and the slowest to converge, the bottom decade is where a 2,560-d readout should sit.  The SAME grid is used for G4+E and G8+E and every control, so the paired contrast is not affected by it).
* **MLP:** `MLPClassifier((64,), relu, adam, lr 1e-3, early_stopping=True, validation_fraction 0.15, n_iter_no_change 10, max_iter 300, random_state 0)`, `alpha` from {1e-2, 1, 10} by the same inner CV, standardised inputs (`tp_lib` unchanged).
* **Primary metric: the MEAN OF PER-FOLD AUCs** (`tp_lib.auc_ci_fold`, the post-freeze correction of probe 1, adopted from the start here); the pooled out-of-fold AUC is reported beside.  **CI = episode(clip)-cluster bootstrap, B = 2,000, seed 0, resampling the sample's clips**
  (`tp_lib.boot_counts`; the resamples are identical to probe 1/2 for the same population); every difference is the PAIRED dAUC on the same windows with the SAME resample (`tp_lib.dauc_ci_fold`).
  The estimator answers "would another draw of EPISODES say this?"; it is blind to training variance (H-ESTIM-SEED-1) -- but NO model is trained here beyond the probes themselves and the frozen features are one fixed encoder, so the open variance is the probe fit (inner-CV C and MLP init), not a training run of the planner.  Say so wherever a CI is quoted.

## 4. DECISION RULE (committed as written; per population and readout; TRAIN; "dAUC" = fold-mean AUC(G8+E) - AUC(G4+E), paired, CI = 2.5/97.5 percentiles)
Four cells: {P-STOP, P-MOVE} x {linear, MLP}.
* **RESOLUTION HELPS** iff in at least ONE cell the point estimate dAUC >= 0.03 AND the CI excludes 0 (lower bound > 0).
* **RESOLUTION DOES NOT HELP AT 8x20** iff in ALL FOUR cells dAUC < 0.01.
* **Otherwise INCONCLUSIVE** (with the four numbers).
* A cell whose controls FAIL (section 6) is VOID: it cannot support HELPS, and a verdict of NOT-HELP needs all four cells non-void.  A missing cell (a job not run) makes the verdict INCOMPLETE, never NOT-HELP.  (C2 handling is pinned in section 6.)
* The MLP decision cells use only the arms G4+E and G8+E.  Nothing beside (strata, map, eval-141, V+E) can change a verdict.

## 5. BESIDE, never decisive
1. **Strata (TRAIN, as probe 2's `strata2.py`; same flags, same "too small" rule: >= 60 windows, >= 30 per class, >= 20 clips in the SAMPLE):** TL_RED / TL_GREEN / TL_YELLOW (clip-level, VLM-derived `vlm-cot`, disputed, untimed: coarse), NO_TL, lead_now, no_lead, lead_depart (reads the future: a stratum, never a feature).
   Within each stratum, linear OOF scores: fold-mean AUC of E, G4+E, G8+E, V+E; paired dAUC(G4+E - E), dAUC(G8+E - E), dAUC(G8+E - G4+E).  The two strata the brief names for attention: **green light** and **lead departure** (probe 1 found V+E - E = -0.058 and -0.019 there).
2. **Spatial readout map:** the G8 logistic weights on P-STOP.  Arm G8+E refitted ONCE on all 12,000 P-STOP rows (all-rows PCA, C by inner CV on all rows = the same fit used for the eval-141 confirmation); the weight of token (r,c) = the L2 norm of its 16 coefficients (standardised-feature scale), arranged as an 8x20 heatmap (PNG + `.npy`), with the row-marginal and column-marginal share of squared weight and the share in the lower half.  P-MOVE alongside.
   Descriptive only: an L2-regularised readout spreads weight over correlated tokens, so the map is not an attribution (said in RESULTS).
3. **Eval-141 no-refit confirmation (unseen by every choice here):** probes fitted ONCE on all sampled TRAIN windows (PCA on all rows, C / alpha by inner CV), applied unchanged to the eval-141 windows of `trigger_probe/raw/eval141_features.npz` (608 windows with v0 < 6; P-STOP n = 101, P-MOVE n = 507): AUC of E, G4, G4+E, G8, G8+E (linear and MLP), episode bootstrap by eval episode, paired G8+E - G4+E and G4+E - E, G8+E - E.  The 8x20 eval tokens come from `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/clips` (141 clips; the same cache probe 1's V came from), row `t0`; alignment checked against probe 1's `V_eval141.npy` by the same relative-L2 gate as TRAIN.
4. **Continuity:** probe 1's V+E (cached 4x10 through the whole-vector PCA-64) next to G4+E and G8+E, with paired deltas.

## 6. CONTROLS (the pipeline must see a signal, refuse to invent one, and prove the arm reads the 8x20 data) -- run on LINEAR G4+E and G8+E (and E where noted) per population
* **C1 label shuffle** (labels permuted, seed 1): fold-mean AUC CI of G8+E and G4+E must cover 0.5, AND the paired null dAUC(G8+E - G4+E) CI must cover 0.
* **C2 episode-shuffled labels** (donor-episode labels, `tp_lib.donor_labels`, seed 2): fold-mean CI must cover 0.5.  **Pinned in advance (the probe-1/2 lesson: a single donor draw can miss 0.5 by chance, 1 draw in 11 in probe 2):** if seed 2 misses for an arm, seeds 3-6 are run for that arm; the cell is NOT voided iff the 5-draw mean fold-mean AUC lies in [0.47, 0.53] (recorded as a donor-draw artifact); otherwise the cell is VOID.
* **C3 deliberate leak** (the GT label appended as a feature): AUC >= 0.99 for E, G4+E and G8+E.
* **C4 G4 == mean2x2(G8) exactly** (section 2) -- asserted at extraction and in pytest.
* **C5 reachability mutation ("built, tested, unreachable" class).**  (a) The G8 entry point asserts `D == 163,840`; calling it with the 4x10 file (`D == 40,960`) MUST raise (RED).  (b) Content mutation: the G8 source replaced by the nearest-neighbour upsample of G4 (same shape 163,840): the G8+E P-STOP linear arm run on it must give a PCA basis, a feature matrix and an AUC that DIFFER from the real G8+E (|AUC difference| > 1e-4, feature sha256 differs); its dAUC against G4+E is reported (expected near 0: it carries no extra information).
* **C6 planted-resolution known answer (pytest, synthetic tokens through the REAL code path):** a label signal planted as +x / -x on two horizontally adjacent 8x20 tokens cancels exactly in the 2x2 average: the G8+E pipeline must read AUC > 0.9 and the G4+E pipeline 0.5 +- CI; and a null (no planted signal) must give dAUC(G8 - G4) CI covering 0.
* **C7 the probe-1 trap (`run_probe.py:108`, `if src == "V"` skips C1-C3 for any other source name):** my copy runs the controls for EVERY source name; a pytest asserts the control-arm list is non-empty for "G4" and "G8" and for an arbitrary name.
* MLP controls: C1 label shuffle on MLP G4+E / G8+E, run LAST; reported, and if run and failed they void the MLP cell.  If they are not run before the hand-back the MLP cells are flagged "MLP controls not run: UNVERIFIED" (the cell is still reported).
* A failed control voids the arm it guards: reported FAILED-CONTROL, no verdict drawn from that cell.

## 7. Resource rules
RAM free >= 3 GB before every heavy load (PowerShell `Win32_OperatingSystem.FreePhysicalMemory`, polled in code; a step never starts below it); RSS < 2.5 GB; BLAS/OMP threads 4, `torch.set_num_threads(4)`; memmaps and chunks of <= 250 windows; `CUDA_VISIBLE_DEVICES=-1`;
`C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`.  Big arrays on `F:/TanitAD-scratch/trigger_probe3_scratch/`, everything else under `E:/Projects/TanitAD-artifacts/dagland/trigger_probe3/`.  Nothing is deleted (unwanted files move to `aside/`); no git add / commit / push; no box 2, no GPU, no `Keys.txt`, no other process touched.

## 8. What this probe can and cannot say (committed caveats)
1. It tests ONE fixed readout (per-token PCA-16 + a linear / 64-unit MLP head).  A NOT-HELP at 8x20 says this readout on these frozen DINOv3 features at 2x2-pooled resolution does not carry more; it does NOT say 16x40 would not, nor that a different (learned, attention-based) readout would not.
2. G8+E has 4x the features of G4+E; the paired contrast includes the cost of the extra parameters under L2 + inner-CV C.  A HELPS survives that cost; a small NOT-HELP is partly that cost.
3. TL strata are clip-level and VLM-derived; lead_depart reads the future.  Strata CIs are wide; none decides.
4. Eval-141 is small (P-STOP n = 101, 19 episodes) and cannot refute a TRAIN effect.
5. Evidence class: MEASURED; tier: T0-class probe of representations on logged inputs, NOT driving performance.

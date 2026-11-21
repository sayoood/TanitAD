# PREREG_FROZEN -- trigger probe 4: does a SPATIAL readout recover the go/stay trigger from frozen DINOv3 fine tokens where flat readouts did not?  (H-CRAWL-06 precursor, continued)

Zero training of the planner, dev-box CPU, `CUDA_VISIBLE_DEVICES=-1`, torch / BLAS threads <= 6.  This file and the sha256 of every code file are written and hashed BEFORE any real-label fit of this package (`FROZEN_SHA.txt` carries the hashes and the wall clock).
Pre-freeze actions: read probe 3's `PREREG_FROZEN.md` (sections 1-3, 6), `g_lib.py`, `tp_lib.py`, `run_g.py`, `strata_g.py`, `pca_g.py`, `eval_g.py`, `chain.sh`; listed file names / array shapes / dtypes of probe 3's `S8_<pop>.npy`, `S4_<pop>.npy`,
`pcaG_<pop>.json/npz`, probe 1's `windows_train.npz`, `eval141_features.npz`, banked OOF files; wrote the code; ran pytest on SYNTHETIC tokens only (the planted known answer, the uniform-attention mutation, a null, the pipeline end to end on a synthetic bundle).
No real label of this package was fitted, no label statistic was computed, and no real-data fit was started while probe 3 runs.

## 0. Question and what is held fixed
Probe 1 read the frozen DINOv3-L/16 pool4x10 through one PCA-64 of the flattened vector (V+E - E: P-STOP +0.054 linear).  Probe 3 held that kind of FLAT readout fixed and varied the grid (per-token PCA-16 features: G8 = 160 x 16 = 2,560, G4 = 40 x 16 = 640, flattened
into one logistic regression / MLP) and found no 8x20 gain (interim P-STOP G8+E - G4+E linear negative).  A flat readout over 2,560 columns spends its capacity on position-specific weights; the open question is whether a readout that POOLS OVER TOKENS
(softmax attention with a learned position embedding; max pooling) recovers a trigger that the flat readouts did not.  **This probe holds the TOKENS fixed (probe 3's fold-proper per-token PCA-16 features) and varies ONLY the readout** (and, for the second question, the grid under that readout).
Same windows, same E, same folds, same bootstrap resamples as probes 1-3.  Nothing about the encoder, the trunk or the planner is touched.

## 1. Inputs [REUSED, asserted, never modified]
* **Windows / labels / E / folds:** the 12,000 P-STOP + 12,000 P-MOVE TRAIN windows of probe 1 via `trigger_probe/raw/windows_train.npz` (sha256 asserted equal to `a9a57b32ffac558929b8f57e49a979826b88d17be17e5e9d7f9347e5875783ec`) and `raw/sample_<pop>.npz`; GO = the registered launch mask;
  P-STOP v0 < 0.5, P-MOVE 0.5 <= v0 < 6.0; E = the 8 columns O reads; `outer = int(sha12,16) % 5`, `inner = (int(sha12,16)//5) % 3` over the clip (`tp_lib`, copied byte-identically, sha256 `44b87977...ee8932` asserted in code and in pytest).  The holdout `540:600` is never opened.
* **Tokens:** probe 3's `F:/TanitAD-scratch/trigger_probe3_scratch/S8_<pop>.npy` and `S4_<pop>.npy`: float32 `[6, 12000, T*16]` (T = 160 for G8, 40 for G4), token-major (token t, PCA component j at column 16 t + j; **token t = r*20 + c on the 8x20 grid, r*10 + c on the 4x10 grid**).
  Axis 0 is the PCA fit: **f = 0..4 is the per-token PCA fitted on the outer-TRAIN windows of outer fold f (outer != f), f = 5 the all-rows fit** (read from `pca_g.py` / `g_lib.fit_readout`).  **For outer fold f this probe uses S[f] for ALL rows of that fold's train and test windows (fold-proper).  S[5] is used only for the eval-141 confirmation.**
  Shapes and dtype are asserted in code (`p4_lib.layer_shape_ok`, a G8 request with the S4 file raises) and the first 1,000 rows of S[5] are hashed against the fingerprint probe 3 recorded in `pcaG_<pop>.json`.
* **Comparators, reused as banked out-of-fold scores (row order asserted by banked `y` and `clip`):** E and V+E (probe 1, `oof_<pop>_V_linear.npz`; the MLP V+E / E from `oof_<pop>_V_mlp.npz` BESIDE), and probe 3's linear **G4+E / G8+E** (`oofg_<pop>_linear.npz`; their re-computed fold-mean AUC is checked against probe 3's `resg_<pop>_linear.json` to 1e-9 in the tables).
  Nothing in this package refits E, V+E or the probe-3 arms.
* **Statistics:** `tp_lib.boot_counts` (episode(clip)-cluster bootstrap, B = 2,000, seed 0, the same resamples as probes 1-3), **primary metric = fold-mean AUC** (`tp_lib.auc_ci_fold`), paired `tp_lib.dauc_ci_fold` on the same resample; the pooled AUC is reported beside.

## 2. The readouts [PIN]
Input per window: tokens `x in R^{T x 16}` of S[f]; **token standardisation** = per-dimension mean / std over all tokens of the outer-train rows, shared across token positions (16 + 16 numbers); **E standardised** with the mean / std of the outer-train rows (8 columns).
* **R-ATT (attention pooling):** `h_i = LayerNorm_16(x_i + pos_i)`, `pos_i = row_emb[r] + col_emb[c]` (learned separable 2-D embedding, 16-d each, init N(0,1)); `K_i = Linear(16 -> 32) h_i`, `V_i = Linear(16 -> 32) h_i`; 4 learned queries `q_m in R^32` (init N(0,1));
  `alpha_m = softmax_i( q_m . K_i / sqrt(32) )` over the T tokens; pooled = concat_m `sum_i alpha_mi V_i` (4 x 32 = 128); output logit = `Linear(136 -> 1)` of `[pooled, E_std]`.
* **R-MAX (max pooling):** `h_i = ReLU(Linear(16 -> 32) x_i)` per token (NO position information: permutation invariant over tokens, by design), `m = max_i h_i` (32), logit = `Linear(40 -> 1)` of `[m, E_std]`.
* **Training (both):** AdamW lr 1e-3, weight decay 1e-2, batch 256, <= 40 epochs, BCE-with-logits, no class weights, **early stopping (patience 5, best weights restored) on the validation BCE of an inner 20 % split of the outer-train rows, GROUPED BY CLIP**:
  val iff `(int(sha12,16)//15 + seed + fold) % 5 == 0` (independent of the outer digit `sha % 5` and the probe-1 inner digit `(sha//5) % 3`).  Init and batch order are seeded by `seed*100 + fold`.  No hyperparameter is tuned on any label; the init scales (pos 1.0, query 1.0) are the first and only values run on the synthetic known answer.
* **Arms (both populations):** G8att+E, G4att+E (decisive), G8max+E, G4max+E (BESIDE).  G8 = S8 (8x20), G4 = S4 (4x10).
* **Training variance:** each arm is trained with **3 readout seeds (0, 1, 2)** in every outer fold.  **The arm's out-of-fold score = the MEAN of the 3 seeds' held-out logits.**  Each seed's fold-mean AUC is reported and the **seed spread = max - min of the three fold-mean AUCs**.

## 3. DECISION RULE (committed as written; per population; TRAIN; dAUC = paired fold-mean AUC difference on the same windows and the same resamples; CI = 2.5 / 97.5 percentiles of the episode-cluster bootstrap)
Cells: **A_pop = dAUC(G8att+E - V+E)** and **B_pop = dAUC(G8att+E - G4att+E)**, pop in {P-STOP, P-MOVE}: four cells.  V+E is the banked single-fit probe-1 comparator, its seed spread is 0.
* **SPATIAL READOUT HELPS** iff in at least ONE population, A >= 0.03 AND the CI lower bound > 0 AND A > 2 x the seed spread of G8att+E.
* **RESOLUTION HELPS UNDER A SPATIAL READOUT** iff in at least ONE population, B >= 0.03 AND the CI lower bound > 0 AND B > 2 x the larger of the seed spreads of G8att+E and G4att+E.
* **DOES NOT HELP** iff all four dAUCs (A, B in both populations) are < 0.01 and all four cells are complete with passed controls.
* **Otherwise INCONCLUSIVE** (report the four numbers).  Both HELPS statements can hold at once; they are reported separately.
* **A cell whose controls failed is CONTROL-FAILED**: it cannot support HELPS and cannot support DOES NOT HELP.  Cell A needs the controls of G8att+E; cell B needs the controls of G8att+E, G4att+E and the paired null (section 5).  **A cell that is missing, or whose controls were not run, is INCOMPLETE** and the verdict is INCOMPLETE, never DOES NOT HELP.
* **BESIDE, never decisive:** R-MAX arms, strata, attention maps, eval-141, the MLP V+E contrast, the single-seed views, the probe-3 contrasts.
* The estimator answers "would another draw of EPISODES say this?"; the 3-seed spread is the training-variance companion and enters the rule through the `2 x spread` clause; the CI is blind to training variance (H-ESTIM-SEED-1) and this is said wherever a CI is quoted.

## 4. BESIDE (descriptive; none can change a verdict)
1. **Strata** exactly as probe 3 section 5.1 / `strata_g.py`: TL_ANY, TL_RED, TL_GREEN, TL_YELLOW, NO_TL, lead_now, no_lead, lead_depart; the same "too small" rule (>= 60 windows, >= 30 per class, >= 20 clips in the SAMPLE).  Within a stratum the OOF scores of E, V+E, G4att+E, G8att+E (and R-MAX, probe-3 G8+E):
   fold-mean AUC, paired dAUC(G4att+E - E), (G8att+E - E), (G8att+E - G4att+E), (G8att+E - V+E).  TL_* are clip-level and VLM-derived (disputed, untimed); lead_depart reads the future (a stratum, never a feature).
2. **Attention maps:** the HELD-OUT softmax weights of the decision arms (each window is scored by the model that did not train on its fold), averaged over the 4 queries, the 3 seeds and the windows of a stratum, separately for GO and STAY, as 8x20 (G8att) and 4x10 (G4att) PNG + npz, with the lower-half share, the normalised entropy and the largest token share.
   **Descriptive: a mean softmax weight is where the readout looks, not an attribution of the decision.**
3. **Eval-141 no-refit confirmation:** if probe 3's `G8_eval141.npy` exists and its alignment gate passed (`trigger_probe3/raw/eval_g_extract.json`), R-ATT is fitted ONCE on all 12,000 rows of the population (S[5], 3 seeds, inner clip-grouped 20 % early stopping) and applied unchanged to the eval-141 windows of the population
   (projection through the all-rows PCA, probe 3's `build_xy` recipe): AUC of E (logistic, `tp_lib.fit_probe`), G4att+E, G8att+E with the episode bootstrap by eval episode and the paired differences.  Eval-141 is small (P-STOP n = 101) and cannot refute a TRAIN effect.  If the eval features are absent the stage writes "NOT RUN".
4. **Seed views:** per-seed dAUC(G8att+E - V+E) and dAUC(G8att+E - G4att+E) (the single-fit view of the 3-seed logit mean).

## 5. CONTROLS (the pipeline must see a signal, refuse to invent one, and prove the attention path is live)
* **C1 label shuffle WITHIN clip-folds** (labels permuted inside each outer fold, seed 1, so each fold keeps its prevalence): the fold-mean AUC of G8att+E and of G4att+E must have a CI covering 0.5 AND a point estimate in **[0.47, 0.53]** (probe 3's band); the paired null dAUC(G8att+E - G4att+E) CI must cover 0.
* **C2 episode-shuffled labels** (donor-episode labels, `tp_lib.donor_labels`, seed 2): the fold-mean CI of each arm must cover 0.5.  **Conditional band, as probe 3 section 6:** if seed 2 misses for an arm, donor seeds 3-6 are run for that arm and the cell is NOT invalidated iff the 5-draw mean fold-mean AUC lies in [0.47, 0.53]; otherwise CONTROL-FAILED.
* Controls C1 and C2 are run on G8att+E and G4att+E, both populations, **one readout seed (0) per control fit** (stated; the decision arms use the 3-seed mean).  R-MAX arms run no controls (beside).
* **Known answer (pytest, synthetic S8-shaped tokens through the REAL training code):** `+3.0 * y` planted in dimension 0 of token (6,10) (index 130) of standard-normal data: R-ATT must read held-out AUC > 0.9, and **at least one of the 4 queries must place >= 50 % of its window-mean attention mass on token (6,10)** (the all-query mean is reported).
  MEASURED pre-freeze on the first run: AUC 0.973, query masses [0.95, 0.0002, 0.88, 0.00001].  A null (no planted signal) must read 0.5 +- 0.04.
* **Mutation (pytest):** the softmax replaced by uniform 1/T attention (`uniform=True`): the planted known answer must DROP (uniform AUC < 0.75 and live - uniform > 0.2; MEASURED pre-freeze 0.568 vs 0.973), proving the attention path is the live carrier.  R-MAX liveness: AUC > 0.7 on the planted amplitude (MEASURED 0.758).
* **Reachability ("built, tested, unreachable" class):** requesting G8 from the S4 file (or G4 from the S8 file) raises; pytest asserts the fold mapping (S[f] for outer fold f, never S[5]), the standardisers are fitted on the outer-train rows only, the controls exist for both decision arms, `tp_lib` is byte-identical, the grid copies equal probe 3's `g_lib` (g4_from_g8 bit-for-bit; projection to float32 rounding, run in a subprocess), and that the whole pipeline (arms -> controls -> strata / maps -> eval-141 -> tables) runs on a synthetic bundle with the same functions the chain calls.
* A failed control invalidates the cell it guards (section 3).

## 6. Resource rules and order of work
Dev-box CPU only; `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`; torch / BLAS threads <= 6 (two per-population workers with 3 threads each; 6 for the sequential stages); RAM free >= 4 GB before every heavy step (PowerShell `Win32_OperatingSystem.FreePhysicalMemory`, polled in code).
**Probe 3 is running: its files are read-only here and no real-data fit starts until ALL of** `trigger_probe3/raw/CHAIN_c2band.done, CHAIN_mlpctl.done, CHAIN_evalmlp.done, CHAIN_mutate.done, CHAIN_pf.done` **exist** (`chain4.sh` waits for them, launched detached through WMI).
Order (priority, so a killed chain still yields value): per population in parallel: G8att, G4att (3 seeds) -> controls C1, C2 (+ conditional band) -> G8max, G4max; then strata + maps, eval-141, tables (`VERDICT.json`), pytest, manifest.  Every arm-seed and control is banked as it finishes; the chain is resumable.
Big arrays on `F:/TanitAD-scratch/trigger_probe4_scratch/` (held-out attention weights), everything else under `E:/Projects/TanitAD-artifacts/dagland/trigger_probe4/`.  Nothing is deleted; no git add / commit / push; no GPU, no box 2, no `Keys.txt`, no process of probe 3 touched.

## 7. What this probe can and cannot say (committed caveats)
1. It tests ONE fixed attention readout and ONE max readout over probe 3's per-token **PCA-16** features.  Those tokens were compressed 1024 -> 16 by an UNSUPERVISED PCA fitted on a subsample (explained variance about 0.57 for G8, 0.64 for G4); a NOT-HELP says these readouts on these 16-d tokens do not carry the trigger, NOT that the 1024-d tokens, a larger / differently trained head, or 16x40 resolution would not.
2. The decision arm is a **3-seed logit mean (a small ensemble)** compared against the SINGLE-FIT banked V+E; the ensemble gain is part of the measured dAUC.  The per-seed dAUCs are reported beside, and the rule requires A > 2 x the seed spread.
3. V+E (whole-vector PCA-64 of the cached 4x10 pool, logistic) and the G-arms use different feature sets; A is a contrast between the BEST banked flat readout and the spatial readout, B holds the readout fixed and varies the grid.
4. G8att has more positions to attend over than G4att; the extra positions are the thing under test, and the learned position embedding grows with the grid (8 + 20 vs 4 + 10 rows of 16 numbers).
5. Early stopping on a clip-grouped inner split chooses the epoch; a fixed epoch budget (<= 40) may under-train an attention readout.  No hyperparameter search is run.
6. TL strata are clip-level and VLM-derived; lead_depart reads the future; strata CIs are wide; none decides.  Eval-141 is small and cannot refute a TRAIN effect.
7. Evidence class: MEASURED; tier: T0-class probe of representations on logged inputs, NOT driving performance.  Estimator: episode-cluster bootstrap, B = 2,000, seed 0 (blind to training variance; the 3-seed spread is the companion).

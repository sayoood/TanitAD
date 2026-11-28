# AMEND_HWMPLAN_7 (PRE-REGISTRATION, frozen 2026-11-17 before any MLP fit): RA7, a nonlinear 0-2 s acceleration readout on the 1,000-clip latents; the FIT gate first, the RA pass on SCORE only if it passes

Tier T0 everywhere. The FIT rows read true-future latents and R35 rollouts under the logged actions (teacher-forced); the SCORE rows plan toward HWM-ORACLE. A T0 row is a world-model diagnostic and never driving performance. Dev-box CPU only: no GPU, no box 2, no world-model training, no spend. Evidence classes are stated per number.

**Parents.** `amend6/AMEND_HWMPLAN_6.md` (sha256 `db89e128…`), whose section 6 row R6-SAME commits this read; its result `amend6/readout/RESULT_AMEND6.md` (R6-SAME) and the MM reading `raw/2026-11-08-v8f-bands-build/hwm_plan/lonplan/amend6/result/MM_READING_AMEND6.md` on `agent/arch-inf-20260803`; `amend5/AMEND_HWMPLAN_5.md` (sha256 `de6f7477…`) for the pass, the arms, the label rules, the families and the controls.

Paths are relative to `F:/TanitAD-scratch/actrsp/lonplan/` unless named. "amend1" means `raw/2026-11-16-v8f-actrsp-prereg/amend1/` on `agent/arch-inf-20260803`.

## 0 What changes (the whole amendment)

| thing | AMEND_6 | AMEND_7 |
|---|---|---|
| the readout the pass reads | RA6: ridge on D = z₁₀ − z₀ (128 inputs, linear), `amend6/readout_in/readout_K10_R35_s{k}.pt`, sha256 `34ec61ab…` (s0) / `c39743aa…` (s1) | **RA7: an MLP on [z₀, z₁₀] (256 inputs) in the RY head's shape**, fit on the same 1,000 clips; `amend7/readout_in/readout7_MLP_R35_s{k}.pt` (seed set A) and `…_B.pt` (seed set B), sha256 recorded in `amend7/RA7_FIT_s{k}.json` and checked by every run |
| order | fit, then the SCORE read | **the FIT stage first, with its own gate G7-FIT (section 4).** The SCORE read (sections 6-7) runs only on R7-FIT-PASS |
| everything else | | unchanged: the fit set (141 + 859 clips, `amend6/clips6.json`, `amend6/lat6/CHUNKS.json` sha256 `e144d1e3…`), the target, the pair rule, the FIT-CURVE fold map, the pass (`amend5/build/ra_l1.py`), the two R35 chassis, the 994 SCORE windows in 71 episodes, A0 as base (B3), HOLD_c, λ = 0, the estimators, the label rules, the families, AMEND_5's section 8 controls |

## 1 Why, and what is expected (stated before any fit)

* **The committed lever.** AMEND_6 read R6-SAME: Δ_R (RA6-L1 − RA-L1, 0-2 s `acc_mae`, T0) +0.0065 [−0.0040, +0.0163], floor 0.0042 (s0); −0.0051 [−0.0149, +0.0046], floor 0.0104 (s1). MEASURED, paired episode-cluster bootstrap B 2,000 seed 0, 994 windows / 71 episodes, `amend6/readout/RESULT_AMEND6.json`. Its R6-SAME row names this read: a nonlinear 2 s readout in the RY head's shape, CPU only, FIT-CURVE first, SCORE only on a separated R² gain.
* **Why a nonlinear readout could matter (HYPOTHESIS).** RA is linear in D, so its gradient in the pass points the same way at every state. A readout whose gradient depends on z₀ can weight R35's response differently per state. A linear probe's ceiling is a statement about the linear class only (`Project Steering/TRAPS.md`, "A NEGATIVE FROM A LINEAR PROBE IS NOT A NEGATIVE ABOUT LEARNABILITY").
* **What the measurements already say against it:**
  1. **Readout precision did not reach the score.** RA's held-out R² rose from 0.222 to 0.281 (s0) and from 0.208 to 0.279 (s1) between 141 and 1,000 clips, and RA's direction moved (cos(W6, W5) 0.784 / 0.512, `amend6/RA6_FIT_s{0,1}.json`), while Δ_R stayed inside the floor. MEASURED, `amend6/readout/FIT_CURVE_s{0,1}.json`, clip-cluster bootstrap B 1,000 seed 0.
  2. **The committed arm is bounded by MEAN-L1.** If a readout reads R35's response exactly and the per-knot response is uniform, the pass from HOLD_c ends at HOLD_c plus one uniform shift to the source's 2 s mean, which is MEAN-L1 (the analytic identity of AMEND_5's C-PLANT(i)). R35's per-knot response is not uniform, so on R35 this is a limit approached, not an identity (HYPOTHESIS). MEASURED on SCORE: RA6-L1 − MEAN-L1 = −0.0071 (s0) and +0.0034 (s1), not separated; RA6-L1 − HOLD = +0.4547 / +0.4886; hence MEAN-L1 − HOLD = +0.4618 / +0.4852 (a difference of paired point means on the same windows, no interval). `amend6/readout/RESULT_AMEND6_full.md`. RA at 141 clips read the same against MEAN-L1 (−0.0136 / +0.0085, AMEND_6 section 1).
  3. **The ceiling row failed for reasons a readout alone does not fix.** CEIL-ORACLE6 is False at both chassis (MEASURED, `amend6/readout/RESULT_AMEND6_full.json`). On TRAIN (in-sample, AMEND_5 section 3): RA on the TRUE displacement, held flat, reads +0.19 / +0.20 above HOLD (r 0.47-0.48 against a0's 0.78), and the inversion through R35 adds about +0.35 / +0.39 more (R35's gain along W, q10 0.01-0.04).
  4. **The RY recipe overfit at 141 clips.** The state-only MLP of that shape (12 epochs, wd 1e-2) read held-out R² −0.479 [−0.706, −0.326] (s0) and −0.238 [−0.428, −0.107] (s1) at 8,306 pairs. MEASURED, amend1 `out/kstep_R35_s{0,1}.json` (blobs `bb8d2454` / `ce17c4bb`), clip-cluster bootstrap B 1,000 seed 0. Hence the selection grid of section 2.3, which reaches far stronger regularisation than the RY recipe.
* **Expected (ESTIMATED from 1-4):** R7-FIT-FAIL, or R7-FIT-PASS followed by R7-SAME with CEIL-ORACLE7 failing. Under item 2, R7-CEILING is out of reach for the RA-L1 arm (MEAN-L1 sits +0.46 / +0.49 above HOLD), and R7-PRECISION needs a Δ_7 beyond floors of the size of AMEND_6's (0.0042 / 0.0104), which is the size of the whole RA6-L1 − MEAN-L1 gap. The read still runs: it costs CPU only, it settles the readout's function class with a measurement instead of an expectation, and its FIT rows are inputs ACTRSP Option B needs (section 5.2).
* **Why G7-FIT has two parts.** The pass compares two reads from the SAME z₀ (AMEND_5 section 2.1: "any state part of the read cancel[s]"). An MLP on [z₀, z₁₀] can raise its R² from z₀ alone, and the pass never sees that part. So beside the MM-ordered R² gain, G7-FIT requires a gain on the same-state contrast the pass uses (r_wm, section 3.3).

## 2 The readout RA7

### 2.1 Shape: the RY head
* **RY's definition:** amend1 `amend1_kstep.py` (git blob `6a805800`), header line 5 ("MLP 256-256 GELU on [z_t, z_t+10] … (the IDM head's shape), 12 epochs, AdamW 1e-3, wd 1e-2"); `mlp_fit` lines 112-126, the network at line 116; `mlp_pred` lines 128-132; the cross-fit call at line 143 (seed = fold); the frozen fit at line 191.
* **RA7 keeps:** Linear(256, 256) → GELU → Linear(256, 256) → GELU → Linear(256, 1); input [z₀, z₁₀], standardised by the fit rows' mean and sd + 1e-6; AdamW, lr 1e-3, default betas; batch 256; MSE on the unscaled target y; float32 training, float64 metrics; no early stopping inside a fit.
* **The target y** is `fit_ra6.pairs`' Yf: the mean scaled acceleration over the 10 transitions of a stretch of 11 consecutive frames with finite actions (`amend6/build/fit_ra6.py:22-49`).
* **RA7 changes two things, both fixed here:**
  * the epoch count and the weight decay are selected per fit from the grid of 2.3 (RY's 12 and 1e-2 is one grid point);
  * the seed sets the batch order as well as the initialisation. RY's batch order was `RandomState(epoch)` whatever the seed, so a seed replicate would have shared it.
* **Differentiable:** the pass back-propagates through RA7 to the acceleration knots, as through RA (C-GRAD7).

### 2.2 Fit set, pairs and folds: AMEND_6's FIT-CURVE at N = 1,000
* **Latents:** `F:/TanitAD-scratch/hwm_plan/latents_train_R35_s{k}.pt` (the 141 clips) plus `amend6/lat6/chunk_000.pt` to `chunk_017.pt` (859 clips; each chunk's sha256 checked against `CHUNKS.json`), merged by `fit_ra6.merge_latents`, paired by `fit_ra6.pairs`. **58,934 pairs in 1,000 clips at both chassis** (MEASURED, `FIT_CURVE_s{0,1}.json`). n = 58,934; d = 256 (RA7) and 128 (RA).
* **The "same folds":** order = `F:/TanitAD-scratch/hwm_plan/ts/clips141.json` then `clips6.json[:859]`; a pair's fold = `(RandomState(0).permutation(1000) % 5)[order index of its clip]` (`fit_ra6.py:155`, `:216-221`).
* **The comparator RA:** AMEND_6's FIT-CURVE cross-fit at t = 1,000 (λ per outer fold by `choose_lam` on the four training folds, `fit_ra6.py:66-72` and `:159-160`), rerun with its per-fold models kept. It must reproduce the 1,000 row of `FIT_CURVE_s{k}.json` (C-CURVE7): R² **0.2808** [0.2550, 0.3046] (s0) and **0.2788** [0.2538, 0.3034] (s1).

### 2.3 Selection protocol: inner folds only
The outer test fold and every SCORE window are never read by selection or by fitting. (`TRAPS.md`: a probe that tunes on the data it scores manufactures a result.)
* **Grid, 12 configs, in this order** (a tie goes to the earlier, more regularised config): epochs {2, 6, 12, 24} as the outer loop, ascending; weight decay {1.0, 0.1, 0.01} as the inner loop, descending. Width, depth, activation, lr and batch stay fixed (2.1).
* **Outer fold f:**
  * training rows = the folds ≠ f;
  * for each config, the inner error = Σ over the four folds g ≠ f of the squared error on fold g of the model fit on the folds ∉ {f, g};
  * the config with the least inner error is refit on all folds ≠ f and predicts fold f.
* **The SCORE-pass readout** (used only on R7-FIT-PASS): the same selection with the five folds as inner folds over all 1,000 clips, then a refit on all 58,934 pairs.
* **Seeds.** Every fit takes seed = S₀ + 10·f + j: f is the outer fold 0-4, or 8 for the final readout's selection fits and 9 for its refit; j is the inner fold 0-4, or 9 for a refit.
  * `torch.manual_seed(seed)` sets the initialisation; `np.random.RandomState(1000·seed + epoch)` sets that epoch's batch order. All 12 configs of one slot share the seed.
  * **Seed set A: S₀ = 0, the committed readout. Seed set B: S₀ = 500, the replicate.** Set B repeats the selection and the fits in full.
* **F_T,k, the readout-training floor:** |Q(B) − Q(A)| on the full-set point of each FIT quantity Q, per chassis. It answers another READOUT-TRAINING run.

## 3 FIT stage: arms, quantities, estimator

### 3.1 Arms
Each is a full cross-fit on the 1,000 clips with the selection of 2.3, seed set A unless named.

| arm | input / target | role |
|---|---|---|
| RA | D / y, ridge, AMEND_6's procedure | comparator (C-CURVE7) |
| MLP-A | [z₀, z₁₀] / y | the committed readout |
| MLP-B | as MLP-A, seed set B | replicate, gives F_T |
| KCTL | LinearD (3.5) / y | linear-equivalence K-control |
| PLANT | [z₀, z₁₀] / y_plant (3.4); RA and KCTL are also fit on y_plant | planted-signal control |
| SHUF-T | [z₀, z₁₀] / y permuted over the pairs by `RandomState(20261120)` | shuffled-target control |
| ZERO | [z₀, z₀] / y (z₁₀ replaced by z₀) | state-only arm: the zero control of r_wm, and STATE7 (beside) |
| CONST | the training rows' mean of y | constant-only control (`TRAPS.md` probe rule) |

### 3.2 R², the quantity the MM ordered
R² = 1 − mean((p − y)²) / mean((y − ȳ)²) over the pooled held-out predictions (`fit_ra6.py:166`). **ΔR² = R²(MLP-A) − R²(RA)** on the same pairs.

### 3.3 r_wm, the same-state contrast the pass reads
* **Two rollouts per pair i**, through `ra_l1.roll` (`amend5/build/ra_l1.py:161-164`, the function the pass calls; context 1; read after 10 actions):
  * F_true,i: from z₀,i under the stretch's 10 logged scaled actions, both channels;
  * F_alt,i: the same, with the acceleration channel set to ā = 0.0026808911934494972 (scaled; amend1's `a_mean_wm[0]` in `kstep_R35_s{0,1}.json`) and the logged yaw kept.
* **ΔR_i** = R(z₀,i, F_true,i) − R(z₀,i, F_alt,i). Each pair is read by the fold model that did not see its clip: RA reads (F − z₀) through its ridge; RA7 reads [z₀, F].
* **r_wm = Pearson r(ΔR, y).** Both sides share z₀ and yaw, so the state part cancels as it does in the pass, and the readout's scale cancels as it does at the pass's minimum. **Δr_wm = r_wm(MLP-A) − r_wm(RA).**
* Beside, no bar: r_swap and s_swap, amend1's grafted swap (partner rule `amend1_kstep.py:71-80`, applied to this table's frames; input definition at line 83).

### 3.4 The planted target (fixed here)
* z₀ˢ and Dˢ are z₀ and D, column-standardised over all pairs.
* With rng = `RandomState(20261118)`, draw u, v₁, v₂ ∈ R¹²⁸ in that order (standard normal, each scaled to unit norm).
* t₁ = std(Dˢ·v₁); t₂ = std(the residual of Dˢ·v₂ on t₁); s = std(the residual of z₀ˢ·u on t₁ and t₂). Here std means centred to mean 0 and scaled to sd 1 (population sd, ddof 0). Each of t₁, t₂ and s is then clipped to [−3, 3], so that a few heavy-tailed pairs cannot dominate t₂².
* f = t₁·(1 + tanh s) + 0.5·(t₂² − 1); g = std(f); y_plant = g + ε, with ε = `RandomState(20261119).standard_normal(n)`.
* R²_oracle = 1 − mean(ε²) / var(y_plant), computed and recorded (about 0.5 by construction).
* The z₀ × D interaction is the kind of structure RA7 is meant to add over RA.

### 3.5 The K-control model
* **LinearD** takes the RA7 input [z₀, z₁₀], forms D = z₁₀ − z₀, standardises D by the training rows' mean and population sd + 1e-6 (as `fit_ra6.ridge_fit`, lines 52-55), and applies Linear(128, 1).
* Loss: mean((p − y)²) + λ_f‖w‖², bias unpenalised, with λ_f = RA's λ in outer fold f. Solved by full-batch L-BFGS in float64 (strong Wolfe line search, at most 2,000 iterations, gradient tolerance 1e-12).
* It runs through RA7's own input builder, fold assignment, pooling, R² and bootstrap; only the module and the optimiser differ. It is RA7's MLP restricted to a linear map of D.

### 3.6 Estimator
* **Paired clip-cluster percentile bootstrap, B 2,000, seed 0, over the 1,000 clips.** Each replicate draws `RandomState(0).randint(0, 1000, 1000)`; every statistic of the replicate is computed on that one resample, with both models' held-out predictions fixed.
* It answers another draw of CLIPS only. F_T answers another readout-training run. Neither answers another training run of R35: as in AMEND_5 section 5, the two chassis agreeing answers that.
* C-CURVE7 reruns FIT-CURVE's own estimator (B 1,000, seed 0).

### 3.7 Beside rows (no bar)
* **STATE7:** ZERO's held-out R², the part of the MLP's R² that [z₀] alone gives.
* **GAIN7,** the quantity Option B's term reads at its step 0:
  * anchors every 10 frames of each clip (amend1's rule, `amend1_kstep.py:195`);
  * contrast ±δ on the acceleration channel of the logged sequence, δ = 0.42 WM units (`RSP_DELTA[0]`, `actrsp/optb/tree/stack/tanitad/models/v8f_mini.py:168`);
  * G = (R(+) − R(−)) / (2δ·s). For RA5, s is Option B's own s_a (0.20076 / 0.20323, `AMEND_OPTB.md` section 1), so its row is Option B's step-0 G; for RA6 and RA7-A, s is that readout's own s_swap at N = 1,000;
  * per chassis: mean G_aa, q10 / q50 / q90, and the share of anchors with G < 0.25;
  * readouts: RA5 (AMEND_5's frozen `readout_K10_R35_s{k}.pt`, Option B's readout), RA6, and RA7-A (final readouts; in-sample on these clips, and RA5 also on the 859 new clips alone, where it is held out);
  * clip-cluster bootstrap B 1,000 seed 0 on the mean.
* **r_wm of RA5** on the 859 new clips, where RA5 is held out.

## 4 Gate G7-FIT (decides whether the SCORE read runs; both chassis required)

M_k is the upper half-width of RA's own 1,000-clip R² interval in `FIT_CURVE_s{k}.json`: **M_s0 = 0.0238, M_s1 = 0.0246**.

* **(a) The R² part, at chassis k:** the ΔR² paired CI lies wholly above 0; the ΔR² point exceeds max(F_T,k(R²), M_k); and seed set B's ΔR² CI also lies wholly above 0.
* **(b) The pass-contrast part, at chassis k:** the Δr_wm paired CI lies wholly above 0; the Δr_wm point exceeds F_T,k(r_wm); and seed set B's Δr_wm CI also lies wholly above 0.
* **Why M_k.** Below it, RA7's point stays inside the interval another draw of clips gives RA alone. It is about a third of the R² gain AMEND_6 measured with no score change (0.059 / 0.071). It is a floor for "worth a SCORE read", not a claim that it suffices.
* Every FIT control of 8.1 reads green, and each of its regression arms reads RED.

| outcome | condition (both chassis; one chassis only is SEEDS-DISAGREE and the less favourable label applies) | next lever (Rule Zero; same work package) |
|---|---|---|
| **R7-FIT-PASS** | (a) and (b) | the SCORE read (sections 6-7) runs at once, in the same chain |
| **R7-STATE-ONLY** | (a), not (b) | no SCORE read: the gain lies in the part of the read that cancels in the pass. Next lever as R7-FIT-FAIL |
| **R7-FIT-FAIL** | not (a). Qualifier **TRANS-ONLY** if (b) holds at both chassis | the RY-shape readout is ELIMINATED as a lever at N = 1,000 at the FIT stage (a result, logged). Next: **ACTRSP Option B** (PI-approved 2026-11-17; build in `actrsp/optb/`; card `AMEND_OPTB.md` sha256 `7cb0245c…`), handed what section 5.2 lists. Under TRANS-ONLY, a SCORE read of RA7 is an MM decision (about 1.3 h of CPU), never automatic |
| **FIT-INVALID** | a FIT control RED, or a regression arm not RED | nothing is read; the control is fixed and the FIT stage rerun. If PLANT fails P1 alone, the grid is re-carded by an amendment before any rerun, never edited in place |

## 5 What a FIT result hands on

### 5.1 To the SCORE read (R7-FIT-PASS only)
The final readouts A and B (2.3), with their sha256 in `RA7_FIT_s{k}.json`.

### 5.2 To ACTRSP Option B (every FIT outcome)
Option B trains R35's response along the frozen RA5 direction: `w = RA.W / RA.sd` and `G = read / (2 δ s_a)` (`actrsp/optb/AMEND_OPTB.md` section 1). This read hands it, per chassis:
1. **Whether a nonlinear readout beats the linear one on the contrast the pass uses** (G7-FIT (b)). Under FIT-FAIL or STATE-ONLY: the linear `fixed_k` term does not leave a better readout of this class unused at N = 1,000.
2. **GAIN7 for RA5 and RA6:** the step-0 value of its term on 1,000 clips, where Amendment 1 section 1.5 measured G_aa 1.244 / 1.202 on 141 clips (amend1 `kstep_R35_s{0,1}.json`), plus the low-gain tail that AMEND_5 found the inversion amplifies by about 1/g.
3. **RA6 against RA5:** cos 0.784 / 0.512 (`amend6/RA6_FIT_s{k}.json`), with r_wm and GAIN7 for both. Whether Option B's frozen readout should move to RA6 is an MM ruling. Its card names RA5's sha256, so a change needs an Option B amendment before its pilot.

## 6 SCORE read (only on R7-FIT-PASS): AMEND_5 sections 4-5 with RA7 in place of RA

### 6.1 The readout in the pass
* ra(D) becomes RA7([z₀, z₀ + D]). The cost is (RA7([z₀, F¹⁰(z₀, [a_acc, yaw_b])]) − â\*)², with â\* = RA7([z₀, F¹⁰(z₀, [src_acc, yaw_b])]).
* Everything else of AMEND_5 section 2.3 is unchanged: one candidate from HOLD_c; `st._optimize` at level-1 settings; knots 1-10 written; yaw and knots 11-30 are the base's, bit for bit. RA7's weights are frozen and in eval mode.
* **RA7-SHUF:** D permuted by `RandomState(0).permutation(128)` (the permutation of `ra_l1.py:75`), on both sides.
* **RA7-NEG:** â\*_NEG = 2·RA7([z₀, F¹⁰(z₀, [HOLD_c, yaw_b])]) − â\*.
* **RA7-ORACLE:** â\* = RA7([z₀, z₁₀]) with the TRUE frame-10 latent; yaw at HOLD_c.

### 6.2 Arms (A0 base, B3)

| arm | readout | inference seeds | role |
|---|---|---|---|
| RA7-L1 | A | 0, 1 | committed lever |
| RA7-L1-B | B | 0, 1 | readout-training replicate (F_T,score) |
| RA7-SHUF | A | 0 | deliberate regression (C-SHUF7) |
| RA7-NEG | A | 0 | beside |
| RA7-ORACLE | A | — | ceiling, beside; sets CEIL-ORACLE7 |
| RA-OFF | — | 0 | identity (C-OFF5) |

References are read from existing files, each sha256 checked against its folder's `SOLVER_RA.json` or `SOLVER.json`: HOLD (windows), A0 (`score/A0_s{k}`), MEAN-L1 (`amend6/score/MEANL1_A0_s{k}`), RA6-L1 (`amend6/score/RAL1_A0_s{k}`), RA6-ORACLE (`amend6/score/ORACLE_A0_s{k}`), and RA-L1 of AMEND_5 (`amend5/score/RAL1_A0_s{k}`, beside).

### 6.3 Contrasts, estimator, floors
* **Contrasts,** per band (0-2 s, 2-6 s, full 6 s) and per family, never pooled: RA7-L1 − HOLD, − A0, − MEAN-L1, **− RA6-L1 (Δ_7)**, and − RA-L1 of AMEND_5 (beside); RA7-L1-B − RA7-L1; RA7-SHUF − RA7-L1; RA7-NEG − RA7-L1; RA7-ORACLE − HOLD; RA7-ORACLE − RA6-ORACLE (beside). Mechanism rows DIR-ACC, DEP and HIT as in AMEND_5 section 5.
* **Estimator:** paired episode-cluster bootstrap, B 2,000, seed 0, over 994 windows in 71 episodes (`taniteval/ci.py` through `hwm_eval.contrast`). It answers another draw of EPISODES only.
* **Floors,** per chassis, on the row's point means:
  * F_I,k = max(|RA7-L1 seed 1 − seed 0|, |RA6-L1 seed 1 − seed 0|): another INFERENCE run, as in AMEND_6;
  * F_T,k = |RA7-L1-B − RA7-L1| at inference seed 0: another READOUT-TRAINING run, which `H-ESTIM-SEED-1` requires for a fitted readout;
  * **Floor_k = max(F_I,k, F_T,k).**
* **Definitions** as in AMEND_5 section 6, with Floor_k in place of F_I,k: Δ_H, Δ_B, Δ_M, G_x and D(X).
  * "Separated-better beyond floor": the CI lies wholly below 0, |Δ| > Floor_k, and both the seed-1 point and the readout-B point lie below 0.
  * "WORSE": the CI lies wholly above 0 and Δ > Floor_k.
  * **Δ_7 = RA7-L1 − RA6-L1** on 0-2 s `acc_mae`.

## 7 SCORE outcomes (both chassis required; one chassis only is SEEDS-DISAGREE, and the less favourable label's next step applies)

| outcome | condition | next lever (Rule Zero; started in the same work package unless blocked) |
|---|---|---|
| **R7-CEILING** | Δ_H separated-better beyond floor, G_x not WORSE, D ≥ 0.10 | AMEND_5's CEILING row with RA7: RA7-L1 is the candidate level-1 longitudinal cost; the 2-6 s band needs a K = 30 readout and its own pull; the T1-kin read is blocked on a goal head (Q2a) |
| **R7-PRECISION** | not CEILING; Δ_7 separated-better beyond floor, Δ_B separated-better, G_x not WORSE | the readout's function class was a lever. Next: (1) Option B by amendment, reading through RA7's local gradient (an MM ruling before Option B's pilot); (2) the pull of the remaining 3,512 eligible clips (box 2 read only; an MM / PI decision under P3). CEIL-ORACLE7 sets the order |
| **R7-SAME** | Δ_7 not separated beyond floor, not WORSE, not CEILING | the nonlinear readout is ELIMINATED as a lever at N = 1,000 (logged). With R6-SAME, readout size and readout class are both measured not to move RA-L1. Next: Option B, handed section 5.2 and CEIL-ORACLE7 |
| **R7-WORSE** | Δ_7 WORSE at either chassis | C-GRAD7, C-RA7-SHA, the K-control and C-SEL7 rerun first (wiring), then the R7-SAME row |
| **HARM / INVALID** | as AMEND_5 section 6 (Δ_B or G_x WORSE; any control RED) | as AMEND_5 |

* **CEIL-ORACLE7** = RA7-ORACLE − HOLD separated-better at both chassis. If it holds, a better read lifts the ceiling and the source's 2 s intent is the limit for RA-L1. If it fails, R35's response along the read, which is Option B's term, is the limit. ESTIMATED: it fails.
* **A failed bar is reported as FAILED, then its next lever runs.** No arm, grid, seed, fold map, margin, constant or criterion changes after any FIT or SCORE number is seen.
* **Done** means a row of section 4 or section 7 is reached, with its next lever started, or named with its blocker and what unblocks it.

## 8 Controls (the read is INVALID if one fails; each has a regression arm that must read RED; `amend7/build/test_amend7.py` pins every mutation on synthetic data)

### 8.1 FIT stage

| control | green | regression arm (must read RED) |
|---|---|---|
| **C-PAIRS7** | the pair table equals `fit_ra6.pairs` on the merged latents bit for bit; 58,934 pairs in 1,000 clips at both chassis; every chunk's sha256 equals `CHUNKS.json` | a flipped byte in a chunk refuses |
| **C-ISO7** | the fit clips are exactly `clips141` + `clips6[:859]`; 0 of them are SCORE episodes | a SCORE eid planted in the table refuses |
| **C-CURVE7** | RA's cross-fit reproduces the 1,000 row of `FIT_CURVE_s{k}.json` (r2, r and both CIs) within 1e-9 | the fold map built from the clips sorted by id (`fit_ra6.fold_map`) |
| **K-control** | KCTL's pooled R² lies within 1e-4 of RA's, and max \|p_KCTL − p_RA\| ≤ 1e-4 (scaled units), at both chassis | the targets taken from each pair's swap partner (misaligned) |
| **C-FOLD7** | for every arm and outer fold f, the training clip set recorded at the fit is disjoint from fold f, and the two together are all 1,000 clips | fold f's rows added to its own fit |
| **C-SEL7** | every selection fit's validation clips lie inside its outer fold's training clips; the chosen config is the argmin of the recorded inner errors | the selection scored on the outer test fold |
| **C-CONST7** | CONST's pooled R² equals its closed form within 1e-12 | CONST predicting with the pooled mean (reads 0.0, not the closed form) |
| **C-ZERO7** | ZERO's ΔR is exactly 0.0 on every pair | ZERO read with the real z₁₀ |
| **C-WM7** | r_wm(RA) CI lies wholly above 0 at both chassis (a sign check; amend1's slope on 141 clips was 0.287 [0.208, 0.372] / 0.258 [0.198, 0.321], MEASURED, `kstep_R35_s{0,1}.json`); and the state after a 10-action `ra_l1.roll` equals the state after 10 steps of a 12-knot rollout (arbitrary knots 11-12) within 1e-6 on 64 pairs | F_true and F_alt exchanged |
| **PLANT** | P1: MLP-A's R² on y_plant ≥ 0.70·R²_oracle; P2: ΔR²(MLP-A − RA) on y_plant has its CI wholly above 0 and its point above M_k | P3: KCTL on y_plant must miss P1 |
| **SHUF-T** | MLP-A's R² ≤ 0 (point) | the shuffle replaced by the identity |

* **Gate test:** on synthetic data, G7-FIT reads PASS for a planted nonlinear truth and FIT-FAIL for a linear truth.

### 8.2 SCORE stage
AMEND_5's controls run on the amend7 folders: C-RA-SHA (as C-RA7-SHA, on the sha256 values in `RA7_FIT_s{k}.json` for readouts A and B), C-OFF5, C-YAW-ID / C-BAND-ID, C-MEAN, C-L4-RA (the built readout's kind `mlp` and its sha256), C-IN, C-READBACK, C-STRAT and C-TRAIN-ISO. New:

| control | green | regression arm (must read RED) |
|---|---|---|
| **C-GRAD7** | on the toy stack and a 16-window TRAIN smoke: the gradient is non-zero on knots 1-10 and exactly 0.0 on knots 11-12 and on yaw | RA7 detached inside the cost; RA7 reading after 12 actions |
| **C-PLANT7** | (i) a toy linear world, a fixed random readout of RA7's shape, a target formed from a known action sequence: the pass reaches the target read within 1e-6 and moves off HOLD_c; (ii) the TRAIN stage: median final cost ≤ 1e-4 | (i) a target formed from HOLD_c's own actions: the move check reads RED |
| **C-SHUF7** | RA7-SHUF − RA7-L1 separated-worse on SCORE 0-2 s `acc_mae` at both chassis, after a positive sign on the TRAIN stage | it is the regression arm |
| **C-RA7-SHA** | each readout file's sha256 equals `RA7_FIT_s{k}.json` | one changed byte refuses |

* The TRAIN stage (564 windows in 141 clips) is in-sample for RA7, as it is for R35. It is beside and never a result.

## 9 Families (all four; T0)

* **SCORE read:** all four families as in AMEND_5 section 7, through AMEND_5's generator, per band and never pooled.
  * STRATEGIC is reported in full (the item 25 pause ended by PI ruling 2026-11-16). It is computed at the full 6 s band, with n per cell (AMEND_6: `s_adh` 12 windows / 4 episodes; `s_false` 856 / 69). `S_CKPT`, 6-30 s intent, derived goals and detected info are NOT COMPUTABLE on this planner, each with its reason.
  * No outcome is decided on a STRATEGIC or TACTICAL cell.
* **FIT stage:** no plan exists, so the plan cells of LONGITUDINAL and all of LATERAL, TACTICAL and STRATEGIC are NOT COMPUTABLE there (n = 0; reason: a readout fit produces no trajectory). Its rows are readout diagnostics of the longitudinal channel, each stamped T0, MEASURED and with its estimator.
* **Under R7-FIT-FAIL or R7-STATE-ONLY,** the plans of record for this line stay AMEND_6's, and so does its four-family table (`amend6/readout/RESULT_AMEND6_full.md`).

## 10 Cost and memory (ESTIMATED, ±50 %)

| step | where | wall |
|---|---|---|
| pair tables and rollouts (F_true and F_alt for 58,934 pairs; GAIN7 contrasts), both chassis | dev-box CPU, 2 threads | about 15 min |
| tests (`test_amend7.py`, synthetic) | dev-box CPU | about 5 min |
| FIT: five selected cross-fits per chassis (MLP-A, MLP-B, PLANT, SHUF-T, ZERO), the cheap arms (RA, KCTL, CONST), two final readouts, the bootstraps | dev-box CPU | about 1.9 h with the chassis in parallel, about 3.7 h serially |
| SCORE path (R7-FIT-PASS only): controls, TRAIN stage, six SCORE arms per chassis, readout | dev-box CPU | about 1.3 h |

* **Basis.** One epoch of RA7's shape on 47,000 × 256 rows takes 0.55 s at 2 threads (MEASURED 2026-11-17 on the dev box, synthetic data, `amend7/card_inputs/bench_mlp_epoch.py` and `.txt`).
  * One inner fit runs on 3/5 of the pairs for 11 epochs on average: about 4.6 s. 240 inner fits plus 5 refits make one selected cross-fit: about 19.5 min.
  * Five selected cross-fits and two final selections (60 fits each on 4/5 of the pairs) make about 1.8 h per chassis.
  * AMEND_6's after-pull stages, read from `amend6/logs` file times: fit plus FIT-CURVE about 1 min per chassis, controls about 6 min per chassis, TRAIN arms 5-8 min, SCORE arms 14-22 min, readout about 10 min; about 57 min in all. AMEND_7 adds the readout-B arm.
* **Memory peak** (ESTIMATED from file sizes):
  * the pair-table stage: at most 1.5 GB (the 141-clip file is 20 MB per chassis; the chunks are 236 MB for both chassis; merged windows about 140 MB per chassis; R35 45 MB);
  * one FIT process: at most 1 GB (the pair table with both rollouts is about 125 MB in float32; RA7 is 0.13 M parameters);
  * SCORE processes as in AMEND_6.
  * Every stage waits for at least 5.5 GB of commit headroom, and the two chassis run side by side only with at least 11 GB (AMEND_6's `wait_commit.py` gate); otherwise they run serially. Commit headroom read on 2026-11-17 was 6.1 GB, so a serial run is the expectation.
* **Disk:** the pair tables, about 0.3 GB on F: (re-creatable from `lat6/` and R35; deleted at the end of the job that made them); readout files under 1 MB. F: had 18.4 GB free on 2026-11-17.
* **Approvals:** none beyond AMEND_6's R6-SAME row. CPU only; no box 2, no GPU, no world-model training, no spend.

## 11 Unchanged
Everything not named in section 0.

## 12 Register rows for the MM (to land with the freeze)
* **H-LONPLAN-RA7 (OPEN):** a nonlinear 2 s readout in the RY head's shape, on [z₀, z₁₀], beats linear RA on the same folds (G7-FIT) and moves RA-L1 beyond RA6-L1 (Δ_7). ESTIMATED expectation: R7-FIT-FAIL, or R7-SAME.
* **Bound on RA-L1 (HYPOTHESIS, with MEASURED support):** under an exact readout and a uniform per-knot response, the RA-L1 pass equals MEAN-L1 (C-PLANT(i)'s identity). Support: RA-L1 − MEAN-L1 not separated at N = 141 (−0.0136 / +0.0085) and at N = 1,000 (−0.0071 / +0.0034), while MEAN-L1 − HOLD = +0.4618 / +0.4852 (difference of point means). A readout lever on the RA-L1 arm therefore cannot reach CEILING; the source's 2 s intent and R35's response remain the terms that can.

## Log
* 2026-11-17: written by the pre-registration agent for the MM, then hashed into `AMEND_HWMPLAN_7.sha256`. Before the hash, no MLP had been fit on any latent of this study and no SCORE window had been read for it. The only new measurement is the synthetic timing in `card_inputs/`.

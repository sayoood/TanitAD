# PREREG_LHA0_D2 -- an a0-first decel head ("Design 2"): stage 1 = K-HEAD-D2 on eval139 (T0 label level). Bars set by the Master Mind; fit and instrument written by the LHA0-D2 agent, 2026-10-26.

Seal: the sha256 of this file is in `PREREG_LHA0_D2.sha256` together with the wall-clock time of sealing. **No eval139 window, label, feature row, bank or score file was opened, hashed, joined or scored by any code of this arm
before that file existed** (existing REPORTS of the sibling packages -- LHA0_DIAG.md, LHA0_EVAL_SIDE_CHECK.json, K_HEAD.json, the R9PF prereg -- were read for the design; s6 is the exact ledger of what WAS computed before the seal, all on the FRESH split). `code/d2_khead.py` refuses to run unless this file
matches the seal and every pin in s6 is recorded in this text.

## 0. Why this exists (disclosed)
`refcv9_lha0/result/LHA0_DIAG.md` (sealed rule: GO, dAUC(B - A) = +0.0241 [+0.0172, +0.0320]) and its **POST-HOC, NOT PRE-REGISTERED** `LHA0_POSTHOC.json` found that, on the FRESH split,
the measured t0 longitudinal acceleration a0 (a causal BACKWARD difference of the dataset's egomotion speed, PI-admissible since 2026-10-18) beats LH's latent head by a wide margin:
raw `-a0` decel AUC 0.7699; A = [z_tac, v0/10] (LH's feature set) 0.6699; fusing the two adds nothing over a0. That is a fresh-split, post-hoc observation. This registration
asks whether a small fitted head on [a0, v0/10, psi_dot0] reproduces an advantage over LH on eval139, at the label level, before any GPU is spent.
Function class (state it in the verdict): a 3-feature ridge **logistic** head. A NO-GO is "a linear 3-feature head does not clear LH by the bar", not "a0 carries no decel information".

## 1. Bars (set by the Master Mind; copied VERBATIM from the brief)
> * **Head D2.** Binary logistic P(decel set) on standardised [a0, v0/10, psi_dot0].
>   * The decel set is the SAME definition K-HEAD uses (`C:/Users/Admin/refcv8_pf/refcv9_lonhead/result/K_HEAD.json` and the code that wrote it).
>   * Fit on the FRESH split only, with the same 292-clip FIT rows (`--lead-clips-only`) and LH's 5 folds (numpy default_rng(0) over fresh episodes, as in `refcv9_lonhead/code/lh_fit.py`).
>   * Choose lambda from {1e-3, 1e-2, 1e-1, 1, 10, 100} by out-of-fold CE on the fresh folds only. Refit on all FIT rows and freeze (.pt or .json of plain numbers, sha256).
>   * Features come from `refcv9_lha0/result/lha0_features.npz` (fresh) and `lha0_features_eval139.npz` (eval). Reuse that package's feature code; do not re-derive a0 differently.
> * **K-HEAD-D2** (T0 label level, eval139, the EXACT window population on which LH's K-HEAD reads 0.7305; reproduce LH = 0.7305 first, and STOP if it does not reproduce).
>   * **GO iff AUC(D2) - AUC(LH) >= +0.030** and its paired episode-cluster bootstrap 95 % CI (B 2000, seed 0, over eval episodes) excludes 0.
>   * Otherwise NO-GO. Then report the gap, and stage 2 is not drafted.
> * **Controls, all on the same eval windows:**
>   * constant (must read 0.5000);
>   * raw `-a0` (no fit), beside;
>   * D2 with a0 and psi_dot0 PERMUTED WITHIN EPISODE (seed 0): must NOT clear the bar, else VOID;
>   * equal-weight fusion of z-scored D2 and LH logits (no fitted weight), beside.
> * **Clock coverage.** The eval clock sidecar covers 136 of 139 clips; the other 3 use the nominal-dt fallback that package documents. Report K-HEAD-D2 both with and without those clips' windows; the bar reads the full population.
> * Tier: T0 label-level diagnostic. One refcv8 training seed, one head fit. The interval answers "another draw of EPISODES" only. Print n (windows, episodes) and d.

## 2. The head, as fitted (FRESH split only; computed BEFORE the seal; `code/d2_fit.py`, `result/D2_FIT_INFO.json`)
* **Inputs.** `universe.npz` = the LHA0 diagnostic's own fresh table (z_tac / v0 / a0 / psi_dot0 per row, the fit-row index, the masks), sha256 `beadb7512d2ea323544ec5853d1114580768908d4aed0a69f90d1e15b8ec4d8f`;
  its a0 / psi_dot0 are `refcv9_lha0/result/lha0_features.npz` (sha256 `8aa08a5d04718671d2dc905e853df82645d4b443adad0241782b543bb2b44db7`) -- **not re-derived**.
* **Rows / folds.** The 8,178 FIT rows over 290 fresh episodes (the 292 lead clips; `LH_FIT_INFO.json`), LON6 class counts FOLLOW 1,796 / CRUISE 2,395 / BRAKE_TO 1,876 / CREEP 103 / HOLD 130 / ACCELERATE 1,878 (asserted).
  Folds = `lh_fit.folds_of(d_eid, 5, 0)`; fold sizes 1,611 / 1,632 / 1,662 / 1,624 / 1,649 rows, 58 episodes each. Standardisation (mean, sd + 1e-9) is fitted on the TRAINING rows of each fold only, and on all fit rows for the frozen head.
* **Decel set** = LON6 indices (2, 3, 4) = BRAKE_TO, CREEP, HOLD -- `l2_common.G_DECEL`, the numerator set of K-HEAD's renormalised score (asserted equal at fit time). K-HEAD's complement is `G_CRUISE` = (0, 1) = FOLLOW, CRUISE.
* **PRIMARY head `D2`: all 8,178 FIT rows, y = 1[class in decel set], every other class (incl. ACCELERATE) = 0.** This is the literal reading of the brief ("binary logistic P(decel set)", "the same ... FIT rows").
  **Beside head `D2NOACC`:** the K-HEAD-renormalisation reading -- ACCELERATE rows dropped from the fit (6,300 rows), complement = FOLLOW + CRUISE. Both are scored on eval; **only `D2` reads the bar.**
* **Objective** sum CE + lambda/2 ||w||^2, intercept unpenalised (sklearn C = 1/lambda, lbfgs, tol 1e-8) -- LH's own convention. Lambda by mean OOF binary CE over the head's fit rows; ties -> smaller lambda. Rank score = the head's logit.
* **Result.** `D2`: 2,109 / 8,178 positive (25.79 %); OOF CE by lambda 1e-3 .. 100 = 0.531222 / 0.531222 / 0.531222 / 0.531221 / **0.531215 (lambda 10)** / 0.531446 (flat to 5 decimals up to lambda 10; in-sample prior entropy 0.57084).
  Frozen (standardised units) w = [a0 -0.6701, v0/10 -0.1838, psi_dot0 -0.0310], b = -1.1519; mu = [0.0016, 1.1887, -0.0002], sd = [0.833, 0.8662, 0.0981]. n = 8,178 rows, d = 3.
  `D2NOACC`: 2,109 / 6,300 positive; lambda 1 (OOF CE 0.563409); w = [-0.7527, -0.5533, -0.1009], b = -0.7872.
* **Frozen files (plain-number JSON; re-running the fit reproduced both sha256 byte for byte):** `D2_FROZEN.json` `69135b9145e38e7b8b13fb1dac85ebda20b7db9b836c85032e7c4af2d25d2fc5`; `D2NOACC_FROZEN.json` `f9119ff8f5f199cec10daae6751a25d37e95cabdca66816186f1f89f8b5f858f`.
* **Fresh OOF decel AUC on the diagnostic's population P (5,038 rows, 1,468 decel, 280 episodes) -- DESCRIPTIVE, never gating, lambda was chosen on the same folds' CE (3 features, 6 lambdas: optimism immaterial but disclosed):**
  `D2` 0.7614; `D2NOACC` 0.7255; LH's own OOF (feature set A) 0.6699; raw `-a0` 0.7699. D2 - LH_OOF = +0.0915 [+0.0398, +0.1467]; D2 - (-a0) = -0.0085 [-0.0269, +0.0096]; D2NOACC - D2 = -0.0360 [-0.0541, -0.0186].
  **Do not read the fresh gap as the eval expectation:** LH reads 0.6699 OOF on fresh but 0.7305 on eval139, so the two populations are not equally hard and the transfer of a gap is not assumed.

## 3. K-HEAD-D2: exactly what is computed on eval139 (`code/d2_khead.py`; run ONCE, after the seal)
1. **Population + LH reproduction FIRST.** The K-HEAD population is rebuilt with `lha0_diag.prep`'s K-POP code (`l2_common.load_train_dump(EVAL_DUMP)`, S6-valid & compliant & GT in {decel, cruise}; z_tac from the L1 eval bank, `v0_in`; `LH_FROZEN.pt`
   sha `6fd3ea1e...d8f6`; score = P(BRAKE_TO+CREEP+HOLD)/(that + P(FOLLOW+CRUISE)) of the 6-way softmax). **STOP (no D2 score is computed) unless** s6_valid 3,668 / compliant 3,251 / GT-decel 639 / GT-cruise 1,856 / union 2,495 / 136 episodes
   AND the LH point AUC rounds to 0.7305 (|diff| < 5e-5). LH's bootstrap CI is also compared with K_HEAD.json's [0.6749, 0.7855], beside, non-gating.
2. **Features.** a0, psi_dot0 for every one of the 4,754 eval bank windows are read from `lha0_features_eval139.npz` (per-clip dt from the clock sidecar; sha256 recorded in `refcv9_lha0/MANIFEST.sha256` as `7fad83f1...04ac`, re-verified at run time).
   v0 = the bank's `v0_in` (what LH reads). **Nominal-dt fallback:** the 102 windows of the 3 clips with no sidecar row (71 of the 2,495 union windows -- asserted, else STOP) carry NaN in that table; for them
   a0 = (v[k] - v[k-1]) / 0.1 and psi_dot0 = wrap(yaw[k] - yaw[k-1]) / 0.1 from the eval dump's `decisions/ep*.npz['ep_poses']` at k = ws (the anchor rule is asserted exact: |pose speed - v0_in| <= 1e-9) -- the same formulas
   as the package's `Bnom` arm. No other value is touched.
3. **Arms scored on the SAME windows:** `LH` (K-HEAD's score); `D2` (primary); `constant` (zeros); `neg_a0_raw` (-a0, no fit); `D2_permuted` (D2 with a0 and psi_dot0 PERMUTED WITHIN EACH EVAL EPISODE over ALL of its bank windows,
   ONE shared permutation, episodes in sorted order, `np.random.default_rng(0)`; v0 stays with its window; the function is the one that reproduces LHA0_DIAG's C4 permutation bit for bit on the fresh table);
   `fuse_D2_LH` = z(D2 logit) + z(LH logit), each z-scored over the evaluated windows with no label (no fitted weight; the LH logit is log P(decel set) - log P(follow+cruise) in the log domain); beside: `D2NOACC`, `D2NOACC_permuted`.
4. **Estimator.** Mann-Whitney AUC with average-rank ties; paired episode-cluster bootstrap over the eval episodes (K-HEAD's loop, B 2000, seed 0, weight = times the episode was drawn, every arm on the same draws); CI = 2.5 / 97.5 percentiles of the paired difference.
   Re-run on the population without the no-sidecar clips (descriptive; the bar reads the full population). Printed: n windows, n decel, n episodes, d (D2: 3; LH: 513).
5. **Verdict** (`d2_common.clears_bar`, UNROUNDED): `D2` clears iff AUC(D2) - AUC(LH) >= +0.030 AND the paired CI lower bound > 0. `D2_permuted` is tested against the SAME bar. Word = **INVALID** if a known-value control fails (constant != 0.5000, unit-weight WAuc != AUC,
   population/LH/fallback gates); **VOID** if `D2_permuted` clears the bar; else **GO** if `D2` clears, else **NO-GO** with the gap `0.030 - delta` reported.

## 4. Committed outcomes (both sides, before any eval number exists)
* **GO** (D2 clears, permuted control does not): a0/psi_dot0, fed to a small fitted head, beat LH's decel decision at the label level on eval139. Consequence: stage 2 (`PREREG_LHA0_D2_ROLL_DRAFT.md`) is DRAFTED, not sealed, no roll run. It is T0 evidence only: the label-level gain of LH (+0.045 over native) did not translate downstream
  (LONHEAD FAIL_F), so a GO licenses a roll-side pre-registration and nothing else.
* **NO-GO:** report AUC(D2), the delta and CI, the gap to +0.030, and the beside rows (raw -a0, fusion, D2NOACC); the fresh-split observation stays an unconfirmed post-hoc finding; stage 2 is NOT drafted.
* **VOID:** the bar is passable without the per-window value (the episode-level a0 / psi distribution is enough); report it, no stage 2.
* **INVALID / STOP:** the instrument did not reproduce its known values; no claim is made.

## 5. Questions the interval answers, and does not (named)
Answers: "another draw of eval EPISODES with the scores held fixed". Does NOT answer: another TRAINING run (one refcv8 EMA-48,852 trunk; one head fit per variant -- the fold-seed spread of the fresh diagnostic, +0.0227 .. +0.0249 for B - A, is the only training-variance read and it
is for a different head), or another INFERENCE run (both heads deterministic; z_tac and v0_in bit-equal across inference seeds 0/1, K-HEAD.json). A separated interval is necessary, not sufficient (H-ESTIM-SEED-1). The a0 here is a backward difference of the DATASET's smooth offline egomotion speed,
not an IMU; the forward-difference contrast was uninformative on fresh (corr 0.974) and certifies nothing about leakage. Tier **T0**: never "driving performance".

## 6. Ledger (what existed before the seal) and pins
* **Computed before the seal, all FRESH-split or synthetic:** the fit (`d2_fit.py`), the fresh OOF descriptives above, `d2_selftest.py` (23 known-value / mutation tests incl. the registration gate on a temporary directory; `result/D2_SELFTEST.json`, ALL_PASS).
  **Forking path, disclosed:** the first fit run labelled the ACCELERATE-excluded head `D2` (primary) and the all-rows head the beside arm; its fresh OOF AUC read 0.7255 vs 0.7614. I then re-read the brief ("the same 292-clip FIT rows", "binary logistic P(decel set)") as the literal
  all-rows target, made THAT the primary, and kept the other as `D2NOACC`. The relabel was made after seeing the fresh AUCs (fresh split only), before the seal and before any eval read; the first-run log and info are kept in `logs/` (`d2_fit_run1_before_relabel.log`,
  `D2_FIT_INFO_run1_before_relabel.json`). Both heads are frozen and BOTH are scored on eval139; the bar reads the primary only.
* **eval139 looks in this decision-AUC family so far (INHERITED, from the sibling packages):** K-HEAD (LH 0.7305 vs native 0.6851), the LONHEAD rolls, R9-PF's provisional reads, and the LHA0 eval-side METADATA check (a0 / psi summary statistics, no labels, no scores). This registration adds one T0 look with ONE gating comparison; beside rows do not gate.
* **Pins:** `code/d2_common.py` `73b0a6107c32ca6f8b22a6b16919a2fe5829bb5977f2512fcf4b42cfa171aec7`; `code/d2_fit.py` `0af2816bf5fb7cfaa57017799f9080886d9467cc14c3491ef1e6f77c7f60f32f`;
  `code/d2_khead.py` `f5ca118e44dd1dcd4e6a16bda6430d187e5114ea841dac4bf28d77feed30cba7`; `code/d2_selftest.py` `39ffaf6a43237a9bda865f3c5303e4f2eee2aadb01798c6b31a607f3ba9f9e85`;
  `D2_FROZEN.json` `69135b9145e38e7b8b13fb1dac85ebda20b7db9b836c85032e7c4af2d25d2fc5`; `D2NOACC_FROZEN.json` `f9119ff8f5f199cec10daae6751a25d37e95cabdca66816186f1f89f8b5f858f`;
  `LH_FROZEN.pt` `6fd3ea1e3700a15b6506176bc855355d0cc1eefb3d87d390584e791d98e1d8f6` (registered, REGISTERED_LH.sha256).
* A bug in `d2_khead.py` found after the seal is fixed in a NEW file with a disclosed deviation; this text and its pins are not edited.

## 7. Stage 2 (only on GO; draft only -- do NOT seal, do NOT roll)
> Draft `PREREG_LHA0_D2_ROLL_DRAFT.md`: the D2 decision feeding the downstream composition used by `C:/Users/Admin/refcv8_pf/refcv9_r9pf/PREREG_R9PF_LH.md` (sha256 6de4e4bb...; read it, do not edit that dir -- another agent is working in it).
> Mirror that prereg's bars, controls and tier. Add two things: an attribution arm with a0 permuted within episode; the head-to-head versus the LH decision on the same windows.
> D2 gives P(decel set), while the R9 machinery wants a regime / class. State exactly how D2's output is mapped into what R9PF_LH consumes, and where an operating point would be chosen (FIT side only, never eval).

# PREREG_LHA0_DIAG -- do measured a0 and psi_dot0 AT t0 add decel-decision information to the LH head?

Sealed (sha256 in `PREREG_LHA0_DIAG.sha256`) BEFORE any a0 / psi_dot0 / AUC / OOF number was computed.
Author: LHA0 diagnostic agent for the Master Mind, 2026-10-26.  Package: `C:/Users/Admin/refcv8_pf/refcv9_lha0/`.
Tier: **T0 label-level DIAGNOSTIC** (a head read on a frozen trunk's z_tac at t0; no planner, no roll, no plan metric).

## 0. What is asked (Master Mind brief, copied)

refcv9's binding bottleneck is the longitudinal DECISION: a clean-split decel head "LH" reaches decel AUC 0.7305 using only two inputs,
refcv8's frozen tactical latent z_tac (512-d) and v0/10. The PI has ruled (2026-10-18) that measured longitudinal acceleration a0 and yaw
rate psi_dot0 AT t0 are admissible inference inputs (IMU-measurable). They must be derived causally: a BACKWARD difference only, never a
forward or centred one. Curvature is NOT admissible. Nobody has tested whether a0 / psi_dot0 add decel-decision information. Measure it.

### 0.1 Bars (VERBATIM from the Master Mind; set before any result)

* **Primary.** Out-of-fold decel AUC under LH's own recipe and folds, comparing feature set A = [z_tac, v0/10] with B = [z_tac, v0/10, a0, psi_dot0].
  * GO iff dAUC(B - A) >= +0.020 and its paired episode-cluster 95% CI excludes 0.
  * Otherwise NO-GO.
* **Controls** (all under the same folds):
  * C1 constant: must read exactly 0.5.
  * C2 v0-only and C3 a0-only: floors, reported.
  * C4 = B with a0 and psi_dot0 PERMUTED WITHIN EPISODE (seed 0): must NOT clear the bar. If C4 clears it, the diagnostic is VOID (structure surviving a shuffle is leakage).
  * C5 = B with a FORWARD-difference a0: reported BESIDE only, as the leak contrast. It must never be used as a feature.
* Report n (rows, episodes) and d for every arm.
* Estimator: paired episode-cluster bootstrap over fresh EPISODES, B 2000, seed 0.

## 1. Operationalisation (registered here; none of it was chosen after seeing a result)

### 1.1 Inputs (all read-only; sha256 of each is written into the result JSON at run time)

| input | path |
|---|---|
| z_tac bank, v0_in, (e_i, t) of the 10,456 fresh windows | `F:/l1c/bank/ztac_fresh.npz` (meta window_W = 8) |
| window list | `F:/l1c/kit_fresh/windows_s0.json` |
| fresh episodes (METADATA ONLY, never frames) | `F:/l1c/kit_fresh/data/refcv6-b1-416x1024-eval139/*.v2ep.pt`, read by `tanitad.data.v2_dataset._scan_meta` of tree `C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58/stack` |
| per-clip clock (dt) | `F:/l1c/kit_fresh/data/clock_sidecar.jsonl` (the cache's own clock; row key `sid` = `stable_episode_id(clip_id)`) |
| LH recipe, fit rows, folds, labels | `C:/Users/Admin/refcv8_pf/refcv9_lonhead/code/lh_fit.py` (`load_fresh(..., lead_clips_only=True)`, `folds_of`, `feats`, `std_of`, `fit_logistic`, `logits_logistic`, `log_softmax`, `ce_rows`), IMPORTED unedited, bytecode writing disabled |
| decel-AUC definition, kernels | `.../refcv9_lonhead/code/k_head.py` (`renorm_decel`, `auc_np`, `WAuc`, `ci`) and `refcv9_l2/code/l2_common.py` (`G_DECEL`, `G_CRUISE`, `load_train_dump`, `lon_indicators`' plan_class rule) |
| GT future (for the decel/cruise class) and compliance fields | `F:/l1c/tree/step48852ema_fresh/dump_s0` (`decisions/ep*.npz` `gt_future_ext`, `gt_future_valid_ext`; `refcv6_extras.npz` `v0`, `v_max_ms`, `v_max_valid`, `gt_max_speed_6s`) |

Pose columns (from SOURCE, `tanitad/data/physicalai.py` `signals_at`, line 641): `poses = [x_east_m, y_north_m, yaw_rad, v_mps]`, i.e. column 3 = speed
`v = hypot(vx, vy)`, column 2 = yaw (quaternion heading, wrapped to (-pi, pi]). `actions[:,0]` is NOT used anywhere (forward speed difference).
`_scan_meta` returns `poses[n_stack-1:]`; every index below is in that coordinate.

### 1.2 Mandatory calibration (positive assertions; STOP and report on failure)

1. **Anchor-frame rule.** For each of the 10,456 bank windows (e_i, t): window NOW `ws = t + W - 1` (W = 8, the dump's `ws`). Candidate rules
   `k = ws + delta`, `delta in {-3..+3}`. A rule MATCHES iff `max |v0_in - poses[k, 3]|` over ALL 10,456 windows is <= 4e-6 m/s (one float32 ulp at
   32 m/s; `v0_in` is float64, the cache stores float32) AND every other delta has max|diff| > 1e-3 m/s. Report the max|diff| per delta. No matching
   rule => STOP. The episode of each e_i is resolved by sha12 (`sha256(clip_id)[:12]`) against the dump's `clip_sha12_by_index`, and cross-checked by
   `decisions/ep<e_i>.npz['ep_poses'] == _scan_meta poses` (exact) for all 300 episodes.
2. **dt.** `dt` is the per-clip `dt_s` of the clock sidecar (cache's own clock, ~0.1007 s, not the nominal 0.1). Nominal-0.1 a0 is reported beside as a sensitivity only.
3. **a0 (primary)** `= (v[k] - v[k-1]) / dt`, **psi_dot0** `= wrap(yaw[k] - yaw[k-1]) / dt` with `wrap(x) = atan2(sin x, cos x)`.
4. **a0 (0.5 s backward variant, beside)** `= (v[k] - v[k-5]) / (5 dt)`.
   Where `k-1` (or `k-5`) does not exist the value is NaN, the row is DROPPED from every arm, and the drops are counted.
5. **Mutation check.** The forward difference `a0_fwd = (v[k+1] - v[k]) / dt` must differ from the backward one (`|diff| > 1e-6`) on the MAJORITY
   (> 50 %) of rows; the fraction is reported. If not, the backward/forward contrast is not a contrast and the diagnostic is INVALID.

### 1.3 Rows, label, population, score

* **Universe U** = the bank windows of the episodes that have at least one LH fit row (the 290 of the 292 lead clips; `lh_fit.load_fresh(lead_clips_only=True)`
  asserts the 8 no-lead clips are excluded whole) and finite a0 / psi_dot0 / a0_fwd.
* **Fit rows F** = LH's own fit rows (`D` of `load_fresh(lead_clips_only=True)`, 8,178 rows, 290 episodes, LON6 labels {0,1,3,4,5,7}) that are in U.
  Multinomial 6-class target = the trainer's LON6 label, EXACTLY LH's.
* **Folds** = LH's own: `lh_fit.folds_of(eid)` (5 folds over fresh episodes, `numpy default_rng(0)` permutation).
* **Score population P (PRIMARY)** = every window of U that is (S6-valid) AND (compliant) AND (K-HEAD plan class in {decel, cruise}), with the K-HEAD plan class
  `dv = |g(6 s) - g(5 s)| / 1 s - v0 ; decel iff dv <= -1.5, accel iff dv >= +1.5, else cruise` (`l2_common.load_train_dump` on the fresh dump; S6-valid = full 60-step valid
  future; compliant = `gt_max_speed_6s <= v_max_ms or v_max_valid < 0.5`). Positive = GT decel, negative = GT cruise, GT accel windows are not in P (K-HEAD's union).
* **Score** `s = P(BRAKE_TO+CREEP+HOLD) / (P(BRAKE_TO+CREEP+HOLD) + P(FOLLOW+CRUISE))` of the 6-way LON6 softmax (`k_head.renorm_decel`, LON6 index groups (2,3,4) / (0,1)).
* **OOF**: for fold k the head is fit on the F rows of the other folds only; the P rows of fold k are scored by it. AUC = Mann-Whitney with average-rank ties on the pooled OOF scores.
* **Sensitivity populations (beside, never gating):** (P-fit) P restricted to F rows (labelled rows only); (P-nocomp) P without the compliance filter.

### 1.4 Arms (identical rows P, F, identical folds; n and d printed for each)

Head recipe for every model arm: multinomial logistic, **lambda = 100 FIXED** (= `LH_FROZEN.pt` hp, the configuration LH froze on its own OOF cross-entropy before this
diagnostic existed; NO lambda is selected inside this diagnostic, so nothing is tuned on scored rows), sklearn `LogisticRegression(C=1/lambda, lbfgs, max_iter 5000, tol 1e-8)` via `lh_fit.fit_logistic`.
Standardisation is fitted on the outer-TRAIN fit rows only: z_tac by `lh_fit.std_of` (as LH); **a0 and psi_dot0 are standardised the same way (outer-train mean, sd + 1e-9)** so the ridge
penalty does not handicap their natural scales; `v0/10` is left as in `lh_fit.feats`.

| arm | features | d |
|---|---|---|
| **A** | z_tac_std(512), v0/10 | 513 |
| **B** | A + a0_std + psi_dot0_std | 515 |
| C1 | one global constant score (exactly 0.5 by construction) | 0 |
| C2 | v0/10 only (lambda 1, as lh_fit's v0_only control) | 1 |
| C3 | a0_std only (lambda 1) | 1 |
| C4 | B with (a0, psi_dot0) PERMUTED WITHIN EPISODE: ONE shared permutation of the episode's rows of U, `default_rng(0)`, episodes in sorted order, applied to fit rows and score rows alike | 515 |
| C5 (beside) | B with `a0_fwd` replacing a0 (psi_dot0 stays backward) -- the leak contrast, never a feature | 515 |
| B05 (beside) | B with the 0.5 s backward a0 | 515 |
| X1 (beside) | v0/10, a0_std, psi_dot0_std only (lambda 1) | 3 |
| B-nominal-dt (beside) | B with a0, psi_dot0 computed at dt = 0.1 | 515 |

### 1.5 Instrument controls (GATES: a failure makes the run INVALID; the claim is not evaluated)

* **K-REPRO-A**: arm A refit with `lh_fit` code reproduces LH's own OOF mean cross-entropy at lambda 100, `1.0917272119` (`LH_FIT_INFO.json LH.table.logistic:100`), to <= 1e-6, on the same F rows and folds.
* **K-POP**: the population/score code applied to the EVAL139 dump (`refcv8_eval_archive/battery/step48852ema_pf/dump_s0`) with `LH_FROZEN.pt` and the eval z_tac bank reproduces K-HEAD's
  union (`union_n 2495`, `gt_decel 639`, `gt_cruise 1856`) and `LH auc 0.7305` (+-5e-5).
* **K-CONST**: C1 AUC == 0.5 exactly.  **K-WAUC**: the weighted-AUC kernel at unit weights equals `auc_np` for every arm.
* **K-ID**: rows/windows join: every U window has a pose row and `v0_in == dump extras v0` (max|diff| reported).

### 1.6 Statistic and decision

* Estimator: **paired episode-cluster bootstrap**, B 2000, seed 0, over the fresh EPISODES of P (draw episodes with replacement; window weight = times its episode was drawn;
  `k_head.WAuc`); the OOF scores are held fixed. `dAUC(B - A)` is differenced INSIDE each draw; the 95 % CI is the 2.5 / 97.5 percentiles.
  **It answers "would another draw of EPISODES say this?" and nothing else; it is blind to training (fold-fit) variance** (H-ESTIM-SEED-1) and to inference variance (the head is deterministic).
* **GO** iff `dAUC(B - A) >= +0.020` AND the CI lower bound `> 0`. Otherwise **NO-GO**. Reported as written; no goalpost moves.
* **VOID** iff C4 clears the same bar (`dAUC(C4 - A) >= +0.020` and CI lower bound `> 0`).  **INVALID** iff any 1.2 / 1.5 gate fails.
* Order of the verdict word: INVALID > VOID > GO / NO-GO.
* **Function class stated in the verdict:** the primary head is a LINEAR (ridge multinomial logistic) map over standardised features. A NO-GO is "no >= +0.020 gain for a linear head at lambda 100",
  NOT "a0 / psi_dot0 carry no decel information". A GO is a lower bound on learnability. X1 and the MLP beside are descriptive.

### 1.7 Beside (never gating, always reported)

* **Fold-seed replicates** (seeds 1..4 of `folds_of`): the whole A / B / C4 comparison re-run on a fresh 5-fold partition; per-seed dAUC(B - A). This is the only part of this diagnostic that varies the
  TRAINING draw; a GO whose seed replicates do not all clear +0.020 is reported as `GO-NOT-REPLICATED`.
* **lambda sensitivity**: A and B at lambda 10 and 1000 (the frozen config sits at the grid edge).
* **MLP beside**: LH's own MLP recipe (256 ReLU, AdamW lr 1e-3, wd 1e-2 = LH's best MLP wd, batch 256, inner episode-grouped 20 % early stop) for A and B, descriptive.
* The sensitivity populations P-fit, P-nocomp; the nominal-dt arm; B05; X1; C5.
* Per-arm individual AUC with bootstrap CI.

### 1.8 Known limits stated in advance

* a0 / psi_dot0 come from the DATASET's egomotion speed / yaw channels (resampled to the 10 Hz grid by `np.interp`), not from a real IMU. The dataset egomotion is an offline estimate and may be smoother
  or less noisy than a deployed IMU, and `np.interp` between logged samples uses one logged sample that is at most one log-interval AFTER t0. A GO therefore bounds what an IDEAL causal measurement adds.
  The C4 / C5 controls guard against gross leakage; they do not certify the provenance of the channel.
* Fit-variance is not covered by the primary CI (one fit per fold). One lambda (the frozen one) is used.
* Eval-side (eval139) provenance of a0 / psi_dot0 is NOT part of this diagnostic (only inspected, if GO).

### 1.9 Hygiene

CPU only (`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=4`), `TMP=TEMP=F:/c_overflow_2026-10-26/tmp`. No git command; nothing written to G:. Clip ids appear only as sha12 in artifacts; logs that carry ids are excluded;
`python C:/Users/Admin/qland/work/mm_clipscan.py <package>` must print `ZZSCAN CLEAN control=True`. Nothing under `refcv9_lonhead/` is edited.

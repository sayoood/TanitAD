# SPEC_P1P: the reading of P1′ (THEORY_CRAWL_AMEND_1 §4), fixed before any eval-141 posterior or ρ is computed

Author: P1′ agent, 2026-11-06 (UTC) / 2026-11-07 (local, +0100). Evidence class of what this spec leads to: MEASURED (ours, dev-box CPU, zero training of the planner, no weight edit).
Tier: a tick-0 plan-level read of the T1-kin dumps (the plan is made from the logged ego history; no executed-plan depth is used). It is not a driving result.
Estimator: episode-cluster bootstrap over the eval-141 episodes. It answers "would another draw of EPISODES say this?" and is blind to training-seed and inference variance (H-ESTIM-SEED-1); the two planner seeds are read separately, not pooled.

## 0. Frozen inputs and what was looked at before this file

* `THEORY_CRAWL_AMEND_1.md` sha256 `9d39ac53fbb55773d40549fe97b7d0c857f8ea26d7c53dedd7fa8cdf0deb4fc0` (checked), beside `THEORY_CRAWL.md` `6e83451cdd2ca241ea1a502a3800e47e145b66796f751141cb7249f77c2d549d`. Neither is edited.
* The criterion, copied verbatim from AMEND_1 §4:
  * "The window set. On the T1-kin eval-141 windows with v0 in 3–8 m/s, take the shrinkage ratio ρ = â_plan / a_human over a 2 s horizon, restricted to windows with a_human > 0.5 m/s²."
  * "The confidence split. Split the windows in two by the out-of-fold posterior of the banked go/stay probe on V+E (§7.31's probe, its own folds, never refit on these windows): CONFIDENT: posterior ≥ 0.8; UNSURE: posterior in [0.35, 0.65]."
  * "P1′ holds iff, at both seeds: median ρ(CONFIDENT) − median ρ(UNSURE) > 0; separated, by an episode-cluster bootstrap with B 2,000, seed 0."
  * "P1′ fails if the difference is not separated at either seed, or is negative. Then the under-acceleration is not a mean over unresolved intent, and T1′ is wrong too."
  * "NOT EVALUABLE if either group holds fewer than 30 windows at a seed. That is named, never padded."
* Inspected BEFORE this file (structure and file identity only; no posterior, no ρ, no a_human, no count of windows in the v0 band, no group size was computed):
  * the file listings of `theory/`, `wsp_read/`, `trigger_probe/`, `trigger_probe/raw/`;
  * the keys, shapes and dtypes of `C:/Users/Admin/rr_scratch/out/C0C_s{0,1}/ticks/ticks_C0C_s{0,1}.npz` (7,197 rows = 3 ticks × 2,399 windows; tick 0 has `gt_ok` true on all 2,399 rows and no NaN in `gt`; the tick-0 rows of s0 and s1 carry identical `ci`, `t0`, `win`, `v0` and `gt`); the first ten `v0` values of s0 were printed by accident (all 7.8–10.3 m/s, window 0);
  * the largest absolute coordinate difference between `plan`, `plan_raw` and `plan_s` at tick 0 (a frame and projection question: `plan` is `plan_s` after the curvature clamp; no speed was formed);
  * the source of `v8f_t1_eval_ri.py` (what `plan`, `plan_raw`, `plan_s` mean) and the source of `trigger_probe/{PREREG_FROZEN.md,tp_lib.py,run_probe.py,eval_confirm.py,extract_v.py,build_windows.py,RESULTS.md}`;
  * the keys and shapes of one eval cache clip (`pool4x10` [T, 40, 1024] fp8, `kin3`, `p10`) and of `eval141_features.npz` (608 windows with v0 < 6; keys sel, E, Z, v0, eid, clip, t0, launch, stay, go).
* Input hashes (sha256, checked 2026-11-06): `tp_lib.py` `44b87977...8ee932`; `run_probe.py` `ad9ee3b2...91432`; `PREREG_FROZEN.md` `0eb1c25c...e6bff97`; `raw/windows_train.npz` `a9a57b32...5783ec`; `raw/sample_pmove.npz` `c549c093...165bb84`; `raw/oof_pmove_V_linear.npz` `5c8ef49d...51a990`; `raw/pca_V_pmove.npz` `846e38bb...a87064`; `raw/eval141_features.npz` `9237d1fd...3b1a1c`; `pcacomp_V_pmove_all.npz` `f9759d9a...4d9629da`. `V_pmove.npy` (983,040,128 bytes) and `V_eval141.npy` (49,807,488 bytes) are read from `F:/TanitAD-scratch/aside_from_E/probe_scratch/trigger_probe_scratch/` (moved there 2026-10-29, `SCRATCH_MOVED.txt`).

## 1. The probe

* **Population and label: P-MOVE, label GO.**
  * P-STOP is defined on v0 < 0.5 m/s. The P1′ window set is v0 in [3, 8). P-STOP and the window set do not overlap, so P-STOP cannot score a single P1′ window. P-MOVE is defined on 0.5 ≤ v0 < 6.0 m/s and overlaps the window set on [3, 6).
  * GO is the probe's decision label and the programme's registered launch definition (`PREREG_FROZEN.md` §2): GO = (v6 ≥ v0 + 1.0), v6 = the GT chord speed over the last 0.2 s of the 6 s horizon (knots 29 → 30). It is the label the probe's controls C1–C3 and its eval-141 confirmation were written for. It is the "go" of "go/stay": the car ends the horizon at least 1 m/s faster than it started.
  * GO24 (the chord at the end of the executed 2.4 s horizon) is the probe's own SECONDARY label ("reported, never decisive"). It is run here only as the declared sensitivity S4 in §4.
* **Features, V+E (as `PREREG_FROZEN.md` §4):** V = frozen DINOv3 `pool4x10` of the CURRENT frame, [40, 1024] fp8 → fp32 → flattened (40,960), centred, PCA-64 fitted on the training rows only (the probe's `StreamPCA`: oversample 32, 2 power iterations, seed 0, chunks of 500); E = `fk.kin_features(kin3[t0])`, the 8 numbers O's ego token reads; V+E = PCA-64(V) ++ E. Standardised with a scaler fitted on the training rows of the model.
* **Training windows and folds (the probe's own, nothing re-selected):** the 12,000 windows of `raw/sample_pmove.npz` (indices into `raw/windows_train.npz`; 6,000 GO and 6,000 not-GO; 2,140 clips), V read from `V_pmove.npy`, outer folds `fold(clip) = int(sha12, 16) % 5`, inner folds `(int(sha12, 16) // 5) % 3`.
* **Refit, by the probe's committed code and hyper-parameters, in a byte-identical copy of `tp_lib.py` (`p1p_tp_lib.py`, sha256 checked equal to `44b87977...8ee932`):**
  * five fold models: for f in 0..4, PCA-64 fitted on the rows with outer ≠ f, `LogisticRegression(penalty=l2, solver=lbfgs, max_iter=2000)`, C chosen from {1e-3, 1e-2, 1e-1, 1, 10} by the inner 3-fold grouped CV AUC on those rows (`fit_probe`), refit on all of those rows. Nothing is tuned on eval-141.
  * one all-train model (the probe's own eval-141 confirmation model, `eval_confirm.py`): PCA-64 on all 12,000 rows, `fit_probe` on all of them. Used only in S3 and in control R2.
* **Posterior (primary): the mean over the five fold models of `predict_proba[:, 1]`** = sigmoid of each model's decision function (the probe's own `decision_function` score passed through the logistic link; for `LogisticRegression` this is exactly `predict_proba`). Every fold model has never seen any eval-141 window, so each is out-of-fold for these windows; the five are averaged because the eval windows belong to no fold. The classes were balanced 50/50 in the training sample (the full-population GO rate is 53.8 %); the posterior is used as the model emits it, with no prior shift.
* **Domain note (named, not hidden).** The P-MOVE probe was trained on 0.5 ≤ v0 < 6. A window with 6 ≤ v0 < 8 lies outside that range, and the GO label is not defined there (`go_label` requires v0 < 6). The window set of P1′ is 3–8, so the PRIMARY reading scores those windows with the same probe, unchanged, as an extrapolation of a linear model. The in-domain restriction [3, 6) is the declared sensitivity S1.

## 2. Windows, a_human, â_plan, ρ

* **Source:** the T1-kin dumps of C0C, seed 0 `C:/Users/Admin/rr_scratch/out/C0C_s0/ticks/ticks_C0C_s0.npz` (ckpt `v8f-recover/M2b_s0_recft_C0`) and seed 1 `.../C0C_s1/ticks/ticks_C0C_s1.npz` (ckpt `M2b_s1_recft_C0`); `--smooth sg72 --feasible kbox`. The eval-141 window set (2,399 windows, 141 episodes) is the `tick == 0` rows. Ticks 1 and 2 are not used: their history is the arm's own, and no probe feature exists for them.
* **Fields:** `plan` [N, 30, 2] = the plan the executor was given (ego frame at NOW, knot j at (j+1)·0.2 s); `gt` [N, 30, 2] = the logged future in the same frame; `v0` = the measured ego speed at the tick (`kin3[t0, 0]`, checked against the probe's E); `gt_ok`; `ci` (clip index into `clip_names`), `clip` (name), `t0` (cache step).
* **Speed at +2 s, the P1 definition (`SPEC_P1.md`):** `sec(x) = hypot(x[11, 0] − x[7, 0], x[11, 1] − x[7, 1]) / 0.8`, a 0.8 s secant centred on 2.0 s (knots 7 and 11 are at 1.6 s and 2.4 s).
  * `a_human = (sec(gt) − v0) / 2` (m/s²);
  * `â_plan = (sec(plan) − v0) / 2` (m/s²), where `plan` is the dump field `plan` (primary; the same field `p1c_t1kin_fit.py` and AMEND_1 §2 use). The raw network plan `plan_raw` is the declared sensitivity S2;
  * `ρ = â_plan / a_human`.
* **Window filter (the primary window set):** `tick == 0` AND `gt_ok` AND `3.0 ≤ v0 < 8.0` (the band of P1) AND `a_human > 0.5` AND `â_plan`, `a_human`, `ρ` all finite. Every window removed by a rule is counted per rule in the result. No ρ is clipped or winsorised.
* The window set, `a_human`, `v0` and the posterior do not depend on the planner seed (the s0 and s1 tick-0 rows carry identical `gt`, `v0`, `ci`, `t0`), so the CONFIDENT and UNSURE groups hold the same windows at both seeds; only ρ differs. Group sizes are still checked per seed, as the rule is worded.
* **Window ↔ probe features:** V is read from the eval cache (`sfresh_inputs/cache/eval/clips/<clip>.pt`, key `pool4x10`, row `t0`, fp8 → fp16 → fp32), E from `D.yaw_kin_from_track(tracks[clip])` and `fk.kin_features(kin3[t0])`, for every window with `3 ≤ v0 < 8` (the tool tree `dagt1_tree_b43c7a9a`, as `extract_v.py`). Control R3 checks that this construction equals `V_eval141.npy` and `eval141_features.npz` bit-for-bit on the windows they cover.

## 3. Groups, statistic, bootstrap, verdict

* **Groups (on the primary window set):** CONFIDENT = posterior ≥ 0.8; UNSURE = 0.35 ≤ posterior ≤ 0.65 (both ends included). Windows with posterior below 0.35 or in (0.65, 0.8) belong to neither group; their count is reported.
* **Statistic per seed:** `Δ = median ρ(CONFIDENT) − median ρ(UNSURE)`.
* **Bootstrap:** episode-cluster, resampling the distinct clips present in the primary window set with replacement, B = 2,000, `numpy.random.default_rng(0)`, the multiplicity construction of `tp_lib.boot_counts` (copied byte for byte). One resample serves both groups and both seeds. Group membership is fixed; the group medians are recomputed in each resample (a median of the windows with their multiplicities, equal to `np.median` of the replicated array: unit test C4). A replicate in which a group is empty is dropped and counted. The interval is the 95 % percentile interval [2.5, 97.5].
* **Separated** means the 2.5th percentile of Δ is above 0.
* **Verdict, per the criterion as written:**
  * NOT EVALUABLE if either group holds fewer than 30 windows at either seed (named, with the counts, never padded; no threshold is moved and no band is widened to reach 30);
  * otherwise HOLDS iff, at both seeds, Δ > 0 and separated;
  * otherwise FAILS (Δ not separated at either seed, or Δ ≤ 0 at either seed).
  * If the primary verdict is NOT EVALUABLE, the sensitivities are still computed and reported, each labelled as not the verdict.
* **Per seed, also reported (descriptive, outside the verdict):** the median ρ of each group with its own bootstrap interval, the mean ρ, the 25th/75th percentiles, the number of episodes in each group, the posterior counts in the four bins, and the median `a_human` and `â_plan` of each group.

## 4. Sensitivities, declared now; none can change the verdict

Each is run through the identical verdict logic and shown beside the primary.
* **S1 in-domain band:** `3.0 ≤ v0 < 6.0` (the P-MOVE training range), same probe, same everything else.
* **S2 raw plan:** `plan_raw` in place of `plan`.
* **S3 all-train probe:** the posterior of the single all-train model (the probe's own eval-141 confirmation model) in place of the five-fold mean.
* **S4 GO24 label:** the same five-fold construction on the secondary label GO24 (`windows_train.npz` key `go24`).

## 5. Controls (a control that fails is reported as FAILED-CONTROL and the arm it guards is not read)

* **R1 refit reproduces the banked artifacts (train only):**
  * the five fold PCAs reproduce `raw/pca_V_pmove.npz` `S[f]` (max abs difference reported; pass if ≤ 1e-3 relative to the scale of S);
  * the five fold models' scores on their held-out rows reproduce `raw/oof_pmove_V_linear.npz` key `V+E` (pass if max abs difference ≤ 1e-6); the same for `GO24__V+E` with the GO24 models; the chosen C per fold is printed.
* **R2 the all-train model reproduces the banked eval-141 confirmation:** its linear V+E AUC on the eval-141 windows with 0.5 ≤ v0 < 6 reproduces `raw/results_eval141.json` `pops.pmove.arms["linear:V+E"].auc` = 0.748 (pass if within 0.005); the chosen C equals the banked `hp`.
* **R3 eval feature construction:** on the windows covered by `eval141_features.npz` (v0 < 6), E equals the banked E (max abs ≤ 1e-9), V equals `V_eval141.npy` (bit-equal after fp16 cast), the GO label computed from the dump's `gt` equals the banked `go`, and the dump's `v0` equals `kin3[t0, 0]` (max abs ≤ 1e-4).
* **K1 the probe's AUC on eval-141 itself:** AUC of the primary posterior against GO on the eval windows with 0.5 ≤ v0 < 6, with the episode-cluster bootstrap CI (B 2,000, seed 0, `tp_lib.auc_ci`).
* **K2 the same, in the P1′ band:** GO AUC on the eval windows with 3 ≤ v0 < 6 (the only part of the band where GO is defined).
* **K3 the two-second go indicator:** AUC of the primary posterior against `a_human > 0.5` on all eval windows with 3 ≤ v0 < 8 (not conditional on the filter; descriptive).
* **Reading of K2:** if the 95 % CI of K2 covers 0.5, the probe is at chance on the band where its label exists, and the report states in its first line that P1′ cannot test T1′ whatever the verdict says. If K1 is above chance and K2 is at chance, that is stated in the same words.
* **N1 posterior-shuffle control:** the primary window set with its posteriors permuted across windows (permutation seeds 1..20, group counts unchanged by construction), real ρ, the full verdict pipeline, both seeds. Pass if at most 3 of 20 permutations are separated at a seed (the probability of 4 or more at a 5 % rate is 0.016).
* **N2 identity control:** ρ computed with `plan := gt`. Every ρ is exactly 1.0, both medians are 1.0, Δ = 0.0 exactly with a [0, 0] interval, the verdict is FAILS (not separated). Pass iff those values are exact.
* **N3 planted-mechanism control:** ρ replaced by the window's own posterior (as if `â_plan = posterior · a_human`). The pipeline must return HOLDS with a separated Δ > 0. Pass iff it does.
* **C4 unit tests** (in `p1p_read.py`, run before the eval-141 windows are read): the weighted median equals `np.median` of the replicated array on random weights; the bootstrap counts equal `tp_lib.boot_counts`; the group masks put a posterior of exactly 0.8, 0.35 and 0.65 where §3 says (0.8 in CONFIDENT, 0.35 and 0.65 in UNSURE).

## 6. Reporting rules

* Files, in `wsp_read/`: `SPEC_P1P.md`, `SPEC_P1P.sha256` (this file's sha256 with UTC and local time, written before any eval-141 posterior), `p1p_tp_lib.py`, `p1p_probe_refit.py` (train only), `p1p_read.py`, `P1P_PROBE_MODELS.npz`, `P1P_PERWINDOW.npz`, `P1P_RESULT.json`, `P1P_RESULT.md`, `HANDOFF_P1P.md`. The code hashes are recorded in `P1P_RESULT.json`.
* The result states: the verdict exactly as written (HOLDS, FAILS or NOT EVALUABLE), the group counts per seed, every control, the sensitivities, the evidence class and tier. A defect in this spec found after the data is read is reported as a defect with a new dated spec beside it; this file is not edited.
* The result does not rewrite `WSP_READ.md`, `WSP_READ.json` or any frozen file; integration into them is the Master Mind's.

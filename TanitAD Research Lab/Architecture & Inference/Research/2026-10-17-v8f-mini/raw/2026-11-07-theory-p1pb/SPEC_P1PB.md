# SPEC_P1PB: the reading of P1′b (THEORY_CRAWL_AMEND_2), fixed before any holdout window is scored

Author: P1′b agent, 2026-11-07. Evidence class of what this spec leads to: MEASURED (ours, dev-box CPU, zero training, no weight edit).
Tier: a tick-0 plan-level read (the plan is made from the logged ego history of the window); not a driving result.
Estimator: episode-cluster bootstrap over the holdout clips, B 2,000, seed 0 (the `tp_lib` construction). It answers "would another draw of EPISODES say this?" and is blind to training-seed and inference variance (H-ESTIM-SEED-1). The two planner seeds are read separately.

## 0. Frozen inputs, and what was looked at before this file

* `THEORY_CRAWL_AMEND_2.md` sha256 `0427c32592434e3e07159124511a2afa073f73a816c7df9f86d1a16b0994f793` (verified equal to `FROZEN_SHA_THEORY_AMEND_2.txt`); `THEORY_CRAWL_AMEND_1.md` §4 sha256 `9d39ac53fbb55773d40549fe97b7d0c857f8ea26d7c53dedd7fa8cdf0deb4fc0` (verified); `SPEC_P1P.md` sha256 `73640c6730d66c84afb071a446e74688840bc2a9d9cd4253e8ae3ac5b4e40a97`.
* The rule is exactly P1′'s (AMEND_1 §4, AMEND_2 §2): the same window filter, the same ρ, the same group thresholds (CONFIDENT posterior ≥ 0.8; UNSURE posterior in [0.35, 0.65]), the same floor (30 windows per group per seed), the same statistic (median ρ(CONFIDENT) − median ρ(UNSURE) > 0 and separated), the same estimator and the same verdict words. The code is the P1′ code, loaded unchanged (`p1p_read.py` functions `read_one`, `verdict`, `groups`, `wmedian_rows`, `eval_features`, `score`, `sigmoid`, `sec`; `p1p_tp_lib.py`; the probe `P1P_PROBE_MODELS.npz`). Only the window source differs.
* The verdict is read on the holdout set alone. The pooled read (eval-141 primary windows ∪ holdout primary windows) is reported beside it and decides nothing.
* NOT EVALUABLE (either group below 30 windows at either seed) is named with the counts and never padded. In AMEND_2's words, it closes P1′ as untestable at this data scale, and the next test of T1′ is then P4 on the LoRA arm.
* Inspected before this file (structure and identity only; no feature of a holdout window was read, no model was run, no v0, a_human, posterior or ρ of a holdout window exists):
  * the holdout list and the admissibility checks (`P1PB_ADMISSIBILITY.json`, sha256 recorded in `SPEC_P1PB.sha256` below the spec hash): `admissible: true`; the holdout is clips 540:600 of the sha12 order of the 4,572 cache_full TRAIN clips (sha256 `f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9`, equal to the digest recorded by every run of the chain M2b → dagoacc → C0); C0's training list is the other 4,512 clips (sha256 `37f4f9ee…`); the probe's training windows (`windows_train.npz`, window-list digest `e144adcd…`, equal to C0's `train_window_list_sha256`) hold 0 holdout clips; the 60 files exist on the dev box with md5 equal to the box 2 manifest and the same encoder route as the eval-141 cache;
  * the structure of the RI tool and of `p2b_check.py` (what the tick-0 batch builder and the planner are), the T1-kin run header of C0C (`replan_steps 10`, `schedule periodic`, `scene logged`, `nav deadreckon`, `limit causal`, `smooth sg72`, `feasible kbox`);
  * the checkpoint config (`hist 4`, `window_start −1`, so the first window of a clip is t = 3; `HORIZON 30`); the v9 label releases (the dev box holds the f63ece41 release, 4,369 clips; the C0 chain trained on the 83e30bd3 release, 4,572 clips, whose rows for the old 4,369 clips are bitwise identical on the non-offset fields: INHERITED from `SPEC_REFCV8_DRAFT.md` line 1542, not re-verified here).

## 1. The window set

* **Clips:** the 60 holdout clips (`P1PB_HOLDOUT_60.json`).
* **Enumeration (PRIMARY): every window.** For each holdout clip with n cache steps, every t in [3, n − 30) (the training enumeration of the C0 chain and of the probe: first window t = 3, last window with a complete 6 s future). AMEND_2 says "every window passing the filter is read", and this is that. Windows are heavily overlapping within a clip; the group sizes are therefore reported together with the number of distinct clips in each group, and the bootstrap resamples clips.
* **Declared sensitivity G4 (the eval-141 construction):** the subset t ≡ 3 (mod 4) of the primary windows (eval-141's grid is stride 4, t = 3, 7, …).
* **Label coverage.** The nav input of the model is the v9 route label. The dev box holds the f63ece41 release. A holdout clip without rows in it has an all-zero nav input, a state C0 never saw in training (the 83e30bd3 release labels 100 % of the train windows). Rule: clips with no f63ece41 rows are excluded from the primary window set, named, and their windows counted. Declared sensitivity S5 includes them with the zero nav (as the eval-141 run did for any eval clip without rows).
* **Filter (P1′'s, verbatim):** tick 0, `gt_ok`, 3.0 ≤ v0 < 8.0 m/s, a_human > 0.5 m/s², and â_plan, a_human, ρ, posterior all finite. Every window removed by a rule is counted per rule. No ρ is clipped or winsorised.

## 2. The tick-0 plan (the only new computation)

* **Models:** `v8f-recover/M2b_s{0,1}_recft_C0/ckpt_last.pt` (the checkpoints the T1-kin dumps recorded). CPU, float32, `model.eval()`, no gradient. The checkpoint's `launch_intent_w` (a loss-side lever, 0.0) is dropped after asserting it is 0.0, as in `p2b_check.py`.
* **Path:** the RI tool's own functions, loaded by file path (`ri/code/v8f_t1_eval_ri.py`): `make_states(store, tracks, J, windows, "deadreckon")`, `build_item(ds, proxy, state, tick 0, a_t 0, a_s 0, "logged", "causal")`, `D.collate`, `model_planner(model, cpu, smooth="sg72", feasible="kbox")`. The plan is the tool's issued plan `plan` (after the Savitzky-Golay smoothing and the curvature projection), the field P1′ read from the dump; `plan_raw` (the raw network plan) is S2.
* **Store, tracks, join:** `ClipStore("F:/TanitAD-scratch/speedread/cache", "train", yaw kinematics)` over the 60 holdout clips (the cache the probe's V was extracted from); the native tracks `sfresh_inputs/inputs/native_tracks.pt` (the store checks the track against the cache pose grid per clip); a join built by `D.build_label_join` from the f63ece41 train release over the 60 clips, `speed=None`, `lead_npz=None` (neither feeds the model at tick 0: the speed bucket is causal, the lead gap is a label).
* **Fields per window:** `v0` = `p10f[-1, 0]` (the measured ego speed the prior reads); `gt` = the logged future in the tick frame (`gt_in_frame`); `plan`, `plan_raw`.
* **Speed at +2 s (the P1 definition):** `sec(x) = hypot(x[11,0] − x[7,0], x[11,1] − x[7,1]) / 0.8`; `a_human = (sec(gt) − v0)/2`; `â_plan = (sec(plan) − v0)/2`; `ρ = â_plan / a_human`.

## 3. The probe posterior (P1′'s, unchanged)

P-MOVE, label GO, V+E linear; the five fold models of `P1P_PROBE_MODELS.npz`; posterior = the mean of the five `predict_proba`. Features: V = the cache row `pool4x10[t0]` (fp8 → fp16 → fp32, flattened 40,960) read from the holdout clip's cache file; E = `fk.kin_features(kin3[t0])` from the native track. No refit, no tuning on any holdout window. Windows with 6 ≤ v0 < 8 lie outside the probe's training range and are scored by extrapolation, as in P1′.

## 4. Groups, statistic, bootstrap, verdict (P1′'s, verbatim in effect)

* CONFIDENT = posterior ≥ 0.8; UNSURE = 0.35 ≤ posterior ≤ 0.65. Δ = median ρ(CONFIDENT) − median ρ(UNSURE) per seed.
* Bootstrap: episode-cluster over the distinct holdout clips present in the primary window set, B 2,000, `numpy.random.default_rng(0)`, the `tp_lib.boot_counts` construction; 95 % percentile interval; separated = the 2.5th percentile above 0.
* Verdict on the holdout alone: NOT EVALUABLE if either group has fewer than 30 windows at either seed; otherwise HOLDS iff Δ > 0 and separated at both seeds; otherwise FAILS.
* Pooled read, beside: the eval-141 primary windows of `P1P_PERWINDOW.npz` (168 windows, 45 episodes, seen) plus the holdout primary windows; clips are distinct sets (checked: 0 overlap); the same logic. Not a verdict.
* Reported per seed besides the verdict: medians with their intervals, means, quartiles, episodes per group, the posterior bins, the median a_human and â_plan per group.

## 5. Sensitivities, declared now, each through the identical logic, none can change the verdict

G4 (grid stride, t ≡ 3 mod 4); S1 (v0 in [3, 6)); S2 (`plan_raw`); S3 (the all-train probe posterior); S4 (the GO24 posterior); S5 (include clips without v9 rows, zero nav).

## 6. Controls (a control that fails is reported as FAILED-CONTROL and the arm it guards is not read)

* **C4 unit tests** (the P1′ ones, plus the weighted median on the pooled path), run before any holdout window is scored.
* **FW forward reproduces the dump:** the same driver run on 240 evenly spaced eval-141 tick-0 windows (the eval store, `join_eval141.npz`, the eval grid), both seeds, must reproduce the dump's tick-0 rows: `v0` max abs ≤ 1e-6, `gt` max abs ≤ 1e-6, `plan` and `plan_raw` max abs ≤ 5e-3 m, `â_plan` max abs ≤ 2e-3 m/s². (The dumps were produced where a GPU may have been visible; the tolerance covers float32 device differences and is stated before the run.)
* **F1 holdout feature path:** per holdout clip, the cache pose grid equals the native track's even frames (the store's own check), n_frames equals the manifest, `gt_ok` true on every enumerated t; E rebuilt twice agrees; the v0 of the forward equals `kin3[t0, 0]` (max abs ≤ 1e-4).
* **Probe artifacts unchanged:** sha256 of `P1P_PROBE_MODELS.npz`, `p1p_tp_lib.py` (= `trigger_probe/tp_lib.py`), `p1p_read.py` equal the values in `P1P_RESULT.json`.
* **K1/K2/K3 (descriptive):** the probe's AUC on the holdout windows against GO (0.5 ≤ v0 < 6, and 3 ≤ v0 < 6) and against `a_human > 0.5` (3 ≤ v0 < 8), episode-cluster interval. If the K2 interval covers 0.5, the first line of the report says P1′b cannot test T1′ whatever the verdict.
* **N1 posterior shuffle** (20 permutations, real ρ, separated in at most 3 of 20 at each seed); **N2 identity** (`plan := gt`, so ρ ≡ 1: medians 1.0, Δ 0.0, interval [0, 0], not separated); **N3 planted** (ρ := posterior: separated Δ > 0). N2 and N3 are read on the statistic (the wording defect of `SPEC_P1P_AMEND_1.md`). They run on the holdout primary window set, using the NOT-EVALUABLE gate only for the verdict string.

## 7. Reporting

Files in `wsp_read/`: `SPEC_P1PB.md`, `SPEC_P1PB.sha256` (this file's hash with UTC and local time, written before any holdout forward), `P1PB_ADMISSIBILITY.json`, `P1PB_HOLDOUT_60.json`, `p1pb_admissibility.py`, `p1pb_forward.py`, `p1pb_read.py`, `P1PB_TICK0_s0.npz`, `P1PB_TICK0_s1.npz`, `P1PB_FW_CONTROL.json`, `P1PB_PERWINDOW.npz`, `P1PB_RESULT.json`, `P1PB_RESULT.md`, `HANDOFF_P1PB.md`, `P1PB_FILES.sha256`. A defect in this spec found after data is read is reported as a defect with a dated note beside it; this file is not edited. Integration into `WSP_READ.md` and `WSP_READ.json` adds text only.

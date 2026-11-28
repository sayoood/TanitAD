# HBLEND_RESULT: PREREG_V8F_HBLEND executed exactly as written (2026-10-25)

Dev box, CPU only, ZERO training, no GPU. Every number below is **MEASURED (ours)** from the artifact paths named; tier **T1-class open loop (own held outputs; not closed loop)**; estimator **paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0** unless a row says otherwise. The bootstrap answers 'would another draw of EPISODES say this?' only; training-seed variance enters only through the two seeds shown side by side and F = |ADE6(s0) - ADE6(s1)| = 0.12074 m (`H-ESTIM-SEED-1`). Inference is deterministic, so no inference-seed floor applies. `OUT` = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/hblend`.

## 0. Outcome (the PREREG outcome table, applied verbatim)

**PASS.** PRIMARY-S passes at both seeds, PRIMARY-N passes at both seeds, SECONDARY-A passes at both seeds. Committed next step from the table: *H becomes the deployed v8F output. The shrinkage weights become a model buffer in the next trained arm, and the FLAT and MIXO reads are re-scored with H applied to each arm's own fitted weights (descriptive).*

**But read section 7 before deploying anything.** The PREREG's bars do not include the CONSISTENCY read the programme registered (H-MINI-CONS-1) and H is **separated-worse than B on it at both seeds**; distance keeping is mixed (seed 0 worse on headway / time-gap / LEAD-VIOL, seed 1 not separated); H does **not** dominate P10 (acceleration MAE stays above P10's). None of these changes the table's outcome (the bars are fixed); all of them bear on the decision 'H becomes the deployed v8F output'. Escalated in the hand-back.

## 1. What was run

* **Route.** B's banked per-window plan tau = `pred` of the KIND run (own code), read from the repo mirror at `refs/heads/agent/arch-inf-20260803` (`.../raw/2026-10-25-kind/run/eval/eval_M2b_s{0,1}-kind-own.npz`, copied to `OUT/inputs/banked/`, sha256 in each score json). P = the checkpoint's own frozen P10 evaluated on the eval kinematic features (it needs no image), cross-checked to the P10-yaw-4512 file. GT = the eval's own context. The banked npz carries tau but not P, so P came from the checkpoint; the CPU forward was run on the 85 eval windows whose features are on this box (section 2).
* **Fit set (deterministic, declared).** The 4,512 trained clips were rebuilt from box 2's `manifest.jsonl` + `--exclude 540:600` and reproduce the trainer's recorded digests (`train_sha256` 37f4f9ee..., `excluded_sha256` f6a1dee0..., both read from box 2's `M2b_s0/selfcheck.json`). Sorted by sha12, **every k-th with k = 32**: k = 32 is the smallest k whose copy fits 3.0e9 bytes (k = 31 gives 146 clips = 3.05e9 B). **141 clips, 2,946,307,281 B** copied by sftp (`OUT/cache/train/clips/`, size-verified). clip list sha256 `458ce2975914b20c7dd91ed4b47032de55d86d9b8fda4df8a0cc0e3ca09921b1` (`OUT/fit_clips.json`).
* **Windows.** The trainer's own builder (`train_windows`, hist 4, start 3): **9,582 windows** (about 68 per clip), ALL windows of the 141 clips, in eval mode (`forward_infer`: p_gt 0, T's own code, no label read). Ages: the eval's own deterministic schedule (a_t = i mod 3, a_s = a_t + (i div 3) mod 5, i = index in the fit list). Eval-141 never enters the fit.
* **Fit.** Closed form per step and axis in the NOW ego frame (axis 0 = along/x forward, axis 1 = cross/y left), clip to [0, 1], then unweighted pool-adjacent-violators to non-decreasing in t, separately per seed from its own checkpoint. `v8f_hblend.py fit`; weights in `weights_M2b_s{0,1}.json`.
* **Declared interpretations (each was a choice the PREREG text left open; none was made after seeing eval).**
  1. *'120 constants'.* The model's plan on the scored grid is the 0.2 s knot plan (`traj30 = traj60[1::2]`), the GT exists only on that grid and the trainer's loss is on it. So w is **fitted on 30 steps x 2 axes = 60 numbers per seed**; the 60 midpoint weights of the 0.1 s `traj60` are not identifiable from the data and are never scored. `weights_*.json` carries them as `w60_derived_unscored` (mean of neighbouring knots). No scored number depends on them.
  2. *Isotonic* = unweighted PAV (the PREREG names no weights).
  3. *3 GB* = 3.0e9 bytes (the stricter reading; under 3 GiB k would be 29-30 and the footprint ~3.2 GB).
  4. *Bar of PRIMARY-N* uses the exact F = 0.12074 m (the PREREG prints 0.1207), so the bar is +0.06037 m (the PREREG prints +0.060). No upper bound lies between the two.
  5. *Fit ages* = the eval's own cycle (the PREREG is silent).


## 2. Identity and pipeline controls (run BEFORE any read; the script refuses to read if any fails)

| control | bar | seed 0 | seed 1 |
|---|---|---|---|
| w = 1: max abs diff of H vs B's banked `pred` (m) | <= 1e-4 | 0 | 0 |
| w = 1: ADE6 mean vs B's banked ADE6 (abs diff, m) | <= 1e-4 | 4.44089e-16 | 4.44089e-16 |
| w = 1: per-window ADE6 max abs diff vs banked | - | 0 | 0 |
| w = 0: ADE6 (m) | 2.4870 +- 1e-3 | 2.48699 | 2.48699 |
| w = 0: per-window ADE6 max abs diff vs `floors_perwindow.npz` P10_refit | - | 1.96183e-06 | 1.96183e-06 |
| w = 0: per-window ADE2 max abs diff vs `floors_perwindow.npz` | - | 6.0371e-07 | 6.0371e-07 |
| checkpoint's embedded prior vs P10-yaw-4512 file, max abs (m) | <= 1e-4 | 1.52588e-05 | 1.52588e-05 |
| banked ADE6 recomputed from `pred` + GT, max abs (ADE2 and FDE6: also 0.0) | - | 0 | 0 |
| fitted weights in [0, 1] and non-decreasing | required | True | True |
| **all identity controls pass** | | **True** | **True** |

* **M2b_s0, w = 1 through the eval's OWN `evaluate()`** (traj, consistency_a, strata, tactical, self_consistency_b, families lon/lat/tac) vs the banked JSON `raw` block: **0 mismatches** over keys ['consistency_a', 'estimator', 'families', 'self_consistency_b', 'strata_ade6', 'tactical', 'tier', 'traj']. Accel MAE: recomputed from per-window components 1.2058 vs banked family block 1.2058; P10 0.6376 vs the PREREG's 0.638.
* **M2b_s1, w = 1 through the eval's OWN `evaluate()`** (traj, consistency_a, strata, tactical, self_consistency_b, families lon/lat/tac) vs the banked JSON `raw` block: **0 mismatches** over keys ['consistency_a', 'estimator', 'families', 'self_consistency_b', 'strata_ade6', 'tactical', 'tier', 'traj']. Accel MAE: recomputed from per-window components 0.9313 vs banked family block 0.9313; P10 0.6376 vs the PREREG's 0.638.

* **CPU forward vs B's banked GPU output** (the PREREG's 1e-4 m bar): run on the eval windows whose features are on this box (1 + 4 clips = 85 windows; the other 136 clips' features, 2.8 GB, do not fit the 3.5 GB budget alongside the train copy). M2b_s0: max |traj30_cpu - banked pred| = 2.81e-05 m, max |embedded p10 - standalone prior| = 1.14e-05 m (pass: True); M2b_s1: max |traj30_cpu - banked pred| = 2.29e-05 m, max |embedded p10 - standalone prior| = 1.14e-05 m (pass: True). The full-population figure (4.6e-5 / 9.9e-5 m over all 2,399 windows) is the video agent's `controls.json`, INHERITED, not re-run. Artifact: `OUT/xcheck.json`.
* **Checkpoint identity.** sha256 of `M2b_s0.pt` / `M2b_s1.pt` re-computed by the fit script and asserted equal to box 2's recorded 601cb435...71dcc6f8 / 6104fb15...e50c2e6.
* **Self-tests** (`test_v8f_hblend.py`, 7 passed, `--basetemp C:/Users/Admin/pt_hblend`): PAV optimality vs brute force; closed-form identities (y = tau gives w = 1, y = P gives w = 0, y = P + 0.4 r gives 0.4 flat); clip-then-isotonic order; a deliberate-regression arm (GT silently replaced by the prior collapses w to 0 and H to P10, which the w = 0 control would flag as 'H = P10', never as a gain).

* **Independent re-derivation of the weights** (`verify_fit.py`, `fit_verify.json`): from the saved fit-window arrays, np.linalg.lstsq per step and axis reproduces the raw weights to M2b_s0: 3.6e-14, M2b_s1: 4.7e-14 and sklearn `IsotonicRegression` reproduces the final weights to M2b_s0: 3.2e-14, M2b_s1: 4.4e-14.
## 3. The fitted weights (the entire deployable artifact: 60 numbers per seed)

Fit set: 9,582 windows, 141 clips, k = 32. Final = clip then isotonic. Counts per seed (of 60 step-axis cells): M2b_s0: 4 clipped at 0, 2 clipped at 1, isotonic changed 31; M2b_s1: 4 clipped at 0, 2 clipped at 1, isotonic changed 20. Artifact: `OUT/weights_M2b_s{0,1}.json` (raw, clipped, final, sum r^2, derived 0.1 s grid).

| t (s) | s0 along | s0 cross | s1 along | s1 cross | s0 raw along | s0 raw cross |
|---|---|---|---|---|---|---|
| 0.2 | 0.000 | 0.656 | 0.000 | 0.609 | -0.048 | +2.206 |
| 0.4 | 0.035 | 0.656 | 0.016 | 0.609 | +0.035 | +0.313 |
| 0.6 | 0.115 | 0.670 | 0.138 | 0.641 | +0.132 | +1.377 |
| 0.8 | 0.115 | 0.670 | 0.138 | 0.641 | +0.098 | +0.343 |
| 1.0 | 0.120 | 0.670 | 0.138 | 0.641 | +0.617 | +0.881 |
| 1.2 | 0.120 | 0.670 | 0.138 | 0.641 | -0.056 | +0.657 |
| 1.4 | 0.120 | 0.670 | 0.138 | 0.641 | -0.058 | +0.556 |
| 1.6 | 0.120 | 0.670 | 0.138 | 0.641 | -0.027 | +0.586 |
| 1.8 | 0.120 | 0.670 | 0.138 | 0.641 | +0.019 | +0.668 |
| 2.0 | 0.120 | 0.749 | 0.138 | 0.700 | +0.084 | +0.749 |
| 2.2 | 0.178 | 0.802 | 0.141 | 0.729 | +0.178 | +0.802 |
| 2.4 | 0.311 | 0.841 | 0.270 | 0.748 | +0.311 | +0.841 |
| 2.6 | 0.429 | 0.861 | 0.439 | 0.763 | +0.429 | +0.861 |
| 2.8 | 0.498 | 0.869 | 0.588 | 0.772 | +0.498 | +0.869 |
| 3.0 | 0.534 | 0.873 | 0.690 | 0.779 | +0.534 | +0.873 |
| 3.2 | 0.555 | 0.875 | 0.754 | 0.787 | +0.555 | +0.876 |
| 3.4 | 0.568 | 0.875 | 0.790 | 0.795 | +0.568 | +0.878 |
| 3.6 | 0.576 | 0.875 | 0.808 | 0.801 | +0.576 | +0.877 |
| 3.8 | 0.583 | 0.875 | 0.817 | 0.803 | +0.583 | +0.874 |
| 4.0 | 0.591 | 0.875 | 0.822 | 0.806 | +0.591 | +0.872 |
| 4.2 | 0.601 | 0.875 | 0.828 | 0.812 | +0.601 | +0.872 |
| 4.4 | 0.609 | 0.875 | 0.834 | 0.819 | +0.609 | +0.874 |
| 4.6 | 0.613 | 0.875 | 0.838 | 0.827 | +0.613 | +0.874 |
| 4.8 | 0.617 | 0.876 | 0.843 | 0.835 | +0.617 | +0.876 |
| 5.0 | 0.627 | 0.881 | 0.849 | 0.846 | +0.627 | +0.881 |
| 5.2 | 0.636 | 0.886 | 0.857 | 0.856 | +0.637 | +0.886 |
| 5.4 | 0.636 | 0.886 | 0.858 | 0.862 | +0.642 | +0.888 |
| 5.6 | 0.636 | 0.886 | 0.858 | 0.864 | +0.636 | +0.885 |
| 5.8 | 0.636 | 0.887 | 0.858 | 0.868 | +0.631 | +0.887 |
| 6.0 | 0.641 | 0.906 | 0.876 | 0.887 | +0.641 | +0.906 |

Reading: on the along axis the residual is shrunk hard up to 2 s (final w 0.03-0.12 at 0.4-2.0 s for s0, 0.02-0.14 for s1; 0.000 / 0.000 at 0.2 s) and kept progressively later (0.64 / 0.88 at 6 s); the cross axis keeps 0.66-0.67 / 0.61-0.64 early (0.2-1.8 s) and 0.91 / 0.89 at 6 s. The two seeds agree in shape; s1 keeps more of its residual late. In-sample on the fit windows (descriptive, not a result): ADE2 tau / P / H = s0 0.3329 / 0.2186 / 0.2128 ; s1 0.3163 / 0.2186 / 0.2141; ADE6 tau / P / H = s0 2.1679 / 2.5502 / 2.0350 ; s1 2.0372 / 2.5502 / 1.9800.

## 4. The PREREG rows

| read | seed | value | 95 % CI | verdict | estimator | tier | artifact |
|---|---|---|---|---|---|---|---|
| **PRIMARY-S** ADE2 H - B (m) | s0 | -0.1033 | [-0.1407, -0.0754] | bar: < 0 and separated -> **PASS** | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s0.json::paired_H_minus_B`.ade2 |
| **PRIMARY-N** ADE6 H - B (m) | s0 | -0.2068 | [-0.2771, -0.1423] | bar: upper bound <= +0.06037 -> **PASS** | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s0.json::paired_H_minus_B`.ade6 |
| **SECONDARY-A** accel MAE H - B (m/s^2) | s0 | -0.4644 | [-0.5542, -0.3796] | bar: < 0 and separated -> **PASS** | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s0.json::paired_H_minus_B`.fw_lon_accel_mae |
| **PRIMARY-S** ADE2 H - B (m) | s1 | -0.0777 | [-0.1148, -0.0525] | bar: < 0 and separated -> **PASS** | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s1.json::paired_H_minus_B`.ade2 |
| **PRIMARY-N** ADE6 H - B (m) | s1 | -0.0697 | [-0.1011, -0.0419] | bar: upper bound <= +0.06037 -> **PASS** | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s1.json::paired_H_minus_B`.ade6 |
| **SECONDARY-A** accel MAE H - B (m/s^2) | s1 | -0.2277 | [-0.2894, -0.1740] | bar: < 0 and separated -> **PASS** | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s1.json::paired_H_minus_B`.fw_lon_accel_mae |

**DOMINANCE vs P10-yaw-4512** (descriptive: 'H dominates P10' iff ADE2, ADE6 and accel MAE point estimates are all <= P10's and ADE6 is separated):

| seed | ADE2 H / P10 (m) | ADE6 H / P10 (m) | ADE6 H - P10, paired | accel MAE H / P10 (m/s^2) | accel H - P10, paired | H dominates P10 |
|---|---|---|---|---|---|---|
| s0 | 0.2123 / 0.2221 | 1.9685 / 2.4870 | -0.5185 [-0.6349, -0.4066] separated | 0.7414 / 0.6376 | +0.1037 [+0.0795, +0.1344] separated | **False** (ADE2 True, ADE6 True, accel False) |
| s1 | 0.2137 / 0.2221 | 1.9848 / 2.4870 | -0.5022 [-0.6214, -0.3845] separated | 0.7036 / 0.6376 | +0.0660 [+0.0401, +0.1016] separated | **False** (ADE2 True, ADE6 True, accel False) |

H beats P10 on ADE2 (point estimates 0.2123 and 0.2137 against 0.2221; paired H - P10 separated at both seeds, section 5) and on ADE6 (separated at both seeds), but **its acceleration MAE stays above P10's** (paired H - P10 separated, positive, at both seeds). So the answer to 'does the planner now dominate its kinematic prior' is: on position yes at every horizon 1-6 s, on acceleration no. Artifact: `OUT/score_M2b_s*.json::paired_H_minus_P10`.

## 5. ADE and FDE at 1-6 s (beside, per PREREG)

**M2b_s0** (m; mean [95 % CI]; paired deltas paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0)

| metric | B | H | P10 | H - B | H - P10 |
|---|---|---|---|---|---|
| ADE@1s | 0.0819 [0.0714, 0.0953] | 0.0598 [0.0511, 0.0728] | 0.0633 [0.0545, 0.0764] | -0.0221 [-0.0270, -0.0174] * | -0.0036 [-0.0041, -0.0030] * |
| ADE@2s | 0.3155 [0.2765, 0.3621] | 0.2123 [0.1916, 0.2392] | 0.2221 [0.2008, 0.2491] | -0.1033 [-0.1407, -0.0754] * | -0.0098 [-0.0144, -0.0046] * |
| ADE@3s | 0.5772 [0.5254, 0.6401] | 0.4734 [0.4324, 0.5198] | 0.5337 [0.4865, 0.5871] | -0.1039 [-0.1433, -0.0747] * | -0.0603 [-0.0745, -0.0451] * |
| ADE@4s | 0.9838 [0.9078, 1.0702] | 0.8541 [0.7859, 0.9292] | 1.0129 [0.9246, 1.1071] | -0.1297 [-0.1696, -0.0959] * | -0.1588 [-0.1940, -0.1237] * |
| ADE@5s | 1.5179 [1.4027, 1.6470] | 1.3530 [1.2458, 1.4681] | 1.6628 [1.5163, 1.8107] | -0.1649 [-0.2181, -0.1176] * | -0.3098 [-0.3791, -0.2442] * |
| ADE@6s | 2.1753 [2.0071, 2.3568] | 1.9685 [1.8106, 2.1306] | 2.4870 [2.2606, 2.7135] | -0.2068 [-0.2771, -0.1423] * | -0.5185 [-0.6349, -0.4066] * |
| FDE@1s | 0.1220 [0.1066, 0.1443] | 0.1209 [0.1054, 0.1429] | 0.1288 [0.1129, 0.1509] | -0.0011 [-0.0047, +0.0022]  | -0.0079 [-0.0093, -0.0065] * |
| FDE@2s | 0.7022 [0.6156, 0.8066] | 0.5603 [0.5105, 0.6173] | 0.6089 [0.5556, 0.6695] | -0.1419 [-0.2317, -0.0792] * | -0.0485 [-0.0622, -0.0353] * |
| FDE@3s | 1.4754 [1.3587, 1.6069] | 1.3354 [1.2289, 1.4543] | 1.5911 [1.4524, 1.7375] | -0.1399 [-0.1918, -0.0949] * | -0.2556 [-0.3160, -0.1947] * |
| FDE@4s | 2.7303 [2.5108, 2.9629] | 2.4830 [2.2813, 2.6914] | 3.0894 [2.8142, 3.3663] | -0.2473 [-0.3432, -0.1588] * | -0.6064 [-0.7473, -0.4737] * |
| FDE@5s | 4.3146 [3.9548, 4.6900] | 3.9699 [3.6405, 4.3013] | 5.1174 [4.6429, 5.5918] | -0.3447 [-0.4902, -0.2130] * | -1.1475 [-1.4050, -0.8968] * |
| FDE@6s | 6.2810 [5.7498, 6.8269] | 5.8185 [5.3261, 6.3198] | 7.6673 [6.9298, 8.4116] | -0.4624 [-0.6569, -0.2849] * | -1.8487 [-2.2842, -1.4375] * |

`*` = CI excludes 0. Artifact: `OUT/score_M2b_s0.json::single / paired_H_minus_B / paired_H_minus_P10`.

**M2b_s1** (m; mean [95 % CI]; paired deltas paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0)

| metric | B | H | P10 | H - B | H - P10 |
|---|---|---|---|---|---|
| ADE@1s | 0.0706 [0.0613, 0.0832] | 0.0601 [0.0516, 0.0731] | 0.0633 [0.0545, 0.0764] | -0.0105 [-0.0143, -0.0074] * | -0.0032 [-0.0036, -0.0027] * |
| ADE@2s | 0.2914 [0.2541, 0.3397] | 0.2137 [0.1931, 0.2400] | 0.2221 [0.2008, 0.2491] | -0.0777 [-0.1148, -0.0525] * | -0.0084 [-0.0134, -0.0025] * |
| ADE@3s | 0.5525 [0.4983, 0.6177] | 0.4814 [0.4414, 0.5274] | 0.5337 [0.4865, 0.5871] | -0.0711 [-0.1125, -0.0443] * | -0.0522 [-0.0687, -0.0339] * |
| ADE@4s | 0.9361 [0.8585, 1.0205] | 0.8663 [0.7980, 0.9396] | 1.0129 [0.9246, 1.1071] | -0.0699 [-0.1034, -0.0449] * | -0.1466 [-0.1851, -0.1072] * |
| ADE@5s | 1.4403 [1.3246, 1.5633] | 1.3687 [1.2598, 1.4832] | 1.6628 [1.5163, 1.8107] | -0.0716 [-0.1027, -0.0459] * | -0.2941 [-0.3656, -0.2222] * |
| ADE@6s | 2.0546 [1.8859, 2.2323] | 1.9848 [1.8199, 2.1587] | 2.4870 [2.2606, 2.7135] | -0.0697 [-0.1011, -0.0419] * | -0.5022 [-0.6214, -0.3845] * |
| FDE@1s | 0.1214 [0.1062, 0.1432] | 0.1221 [0.1066, 0.1438] | 0.1288 [0.1129, 0.1509] | +0.0007 [-0.0028, +0.0040]  | -0.0067 [-0.0080, -0.0054] * |
| FDE@2s | 0.7080 [0.6210, 0.8234] | 0.5682 [0.5200, 0.6221] | 0.6089 [0.5556, 0.6695] | -0.1398 [-0.2347, -0.0795] * | -0.0407 [-0.0556, -0.0246] * |
| FDE@3s | 1.4102 [1.2925, 1.5347] | 1.3573 [1.2506, 1.4738] | 1.5911 [1.4524, 1.7375] | -0.0529 [-0.0805, -0.0269] * | -0.2337 [-0.3002, -0.1610] * |
| FDE@4s | 2.5848 [2.3687, 2.8044] | 2.5087 [2.3004, 2.7194] | 3.0894 [2.8142, 3.3663] | -0.0761 [-0.1111, -0.0424] * | -0.5807 [-0.7251, -0.4369] * |
| FDE@5s | 4.0775 [3.7120, 4.4422] | 4.0022 [3.6567, 4.3598] | 5.1174 [4.6429, 5.5918] | -0.0753 [-0.1243, -0.0308] * | -1.1152 [-1.3790, -0.8542] * |
| FDE@6s | 5.8582 [5.3224, 6.3994] | 5.8168 [5.2957, 6.3540] | 7.6673 [6.9298, 8.4116] | -0.0414 [-0.1003, +0.0132]  | -1.8504 [-2.2868, -1.4291] * |

`*` = CI excludes 0. Artifact: `OUT/score_M2b_s1.json::single / paired_H_minus_B / paired_H_minus_P10`.

## 6. The four families, beside (H - B and H - P10 paired; lower is better except the two tactical agreement rows)

**M2b_s0**

| family | metric | B | H | P10 | H - B | H - P10 |
|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 1.0674 | 0.8709 | 0.9802 | -0.1965 [-0.2466, -0.1506] * | -0.1093 [-0.1594, -0.0532] * |
| LONGITUDINAL | accel MAE (m/s^2) | 1.2058 | 0.7414 | 0.6376 | -0.4644 [-0.5542, -0.3796] * | +0.1037 [+0.0795, +0.1344] * |
| LONGITUDINAL | along MAE (m) | 1.8097 | 1.5925 | 1.8217 | -0.2173 [-0.2899, -0.1495] * | -0.2293 [-0.3137, -0.1424] * |
| LATERAL | cross-track offset MAE (m) | 0.7662 | 0.7557 | 1.1858 | -0.0105 [-0.0235, +0.0010]  | -0.4301 [-0.5259, -0.3366] * |
| LATERAL | heading MAE (deg) | 4.7804 | 4.0259 | 5.7302 | -0.7302 [-1.2215, -0.3694] * | -1.7043 [-2.1458, -1.2910] * |
| LATERAL | yaw-rate MAE (deg/s) | 8.1167 | 6.4244 | 6.3299 | -1.7768 [-2.5520, -1.1835] * | +0.0944 [-0.1340, +0.3448]  |
| LATERAL | curvature MAE (1/m) | 0.0815 | 0.0536 | 0.0462 | -0.0322 [-0.0542, -0.0178] * | +0.0074 [+0.0036, +0.0119] * |
| TACTICAL (plan-derived) | lateral decision agreement (higher better) | 0.8458 | 0.8629 | 0.7728 | +0.0171 [+0.0050, +0.0304] * | +0.0900 [+0.0658, +0.1148] * |
| TACTICAL (plan-derived) | longitudinal decision agreement (higher better) | 0.5790 | 0.5969 | 0.5336 | +0.0179 [+0.0017, +0.0346] * | +0.0634 [+0.0241, +0.1001] * |

**M2b_s1**

| family | metric | B | H | P10 | H - B | H - P10 |
|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 0.9112 | 0.8389 | 0.9802 | -0.0723 [-0.1000, -0.0507] * | -0.1413 [-0.1904, -0.0895] * |
| LONGITUDINAL | accel MAE (m/s^2) | 0.9313 | 0.7036 | 0.6376 | -0.2277 [-0.2894, -0.1740] * | +0.0660 [+0.0401, +0.1016] * |
| LONGITUDINAL | along MAE (m) | 1.6081 | 1.5573 | 1.8217 | -0.0508 [-0.0785, -0.0256] * | -0.2645 [-0.3444, -0.1782] * |
| LATERAL | cross-track offset MAE (m) | 0.8644 | 0.8375 | 1.1858 | -0.0269 [-0.0466, -0.0083] * | -0.3482 [-0.4508, -0.2530] * |
| LATERAL | heading MAE (deg) | 4.4383 | 4.3276 | 5.7302 | -0.1084 [-0.2376, +0.0376]  | -1.4042 [-1.9274, -0.8862] * |
| LATERAL | yaw-rate MAE (deg/s) | 6.9864 | 6.4447 | 6.3299 | -0.6327 [-0.8884, -0.4225] * | +0.1110 [-0.2027, +0.4630]  |
| LATERAL | curvature MAE (1/m) | 0.0626 | 0.0572 | 0.0462 | -0.0097 [-0.0172, -0.0035] * | +0.0110 [+0.0045, +0.0190] * |
| TACTICAL (plan-derived) | lateral decision agreement (higher better) | 0.8287 | 0.8470 | 0.7728 | +0.0183 [+0.0050, +0.0350] * | +0.0742 [+0.0463, +0.1033] * |
| TACTICAL (plan-derived) | longitudinal decision agreement (higher better) | 0.5919 | 0.6132 | 0.5336 | +0.0213 [+0.0083, +0.0346] * | +0.0796 [+0.0346, +0.1259] * |

Nothing in LONGITUDINAL (speed, accel, along), LATERAL or the plan-derived TACTICAL rows is separated-worse than B at either seed; the plan-derived tactical agreement is separated-better at both. (Plan-derived tactical = `four_families.tactical_from_trajectory`: the decision read OFF THE TRAJECTORY. It moves with H, because H moves the trajectory. Do not confuse it with T's own decision heads below.)

### 6a. Distance keeping (the other half of LONGITUDINAL), K = 30 (6 s) lead block, through the programme's own `v8f_dk_read.read_dk` -> `taniteval.lead_metrics`

Lead block `b1_eval_lead_block_k30.npz` (sha256 `7a879baa2b6f4c14...`), window states {'LEAD': 743, 'NO_LEAD': 552, 'NOT_STRAIGHT': 931, 'NO_LABEL': 173}, kinematic-contract check **OK**, arm-vs-itself identity control exactly +0 [0, 0]. Open loop: the lead's logged future does not react to the arm. No distance-keeping bar is registered for v8F-mini, so nothing here is PASS/FAIL. Higher min-TTC is a CENSORED mean (capped at 30 s when never closing); direction 'more gap' is not 'better'; the better/worse reading is LEAD-VIOL and the distance to the human.

| arm | headway min (m) | time-gap min (s) | min-TTC (s, censored) | LEAD-VIOL share of at-risk (n viol / 595) |
|---|---|---|---|---|
| human (GT path) | 25.665 | 3.995 | 19.790 | by definition 0 |
| B_s0 | 25.622 | 4.022 | 19.396 | 0.0353 (21) |
| H_s0 | 25.553 | 3.920 | 20.497 | 0.0487 (29) |
| B_s1 | 25.544 | 3.950 | 19.524 | 0.0605 (36) |
| H_s1 | 25.606 | 3.950 | 20.438 | 0.0588 (35) |
| P10 | 25.116 | 3.832 | 19.947 | 0.0807 (48) |

| paired contrast | headway (m) | time-gap (s) | min-TTC (s) | LEAD-VIOL at-risk (share) |
|---|---|---|---|---|
| H_s0 - B_s0 | -0.207 [-0.400, -0.036] * | -0.095 [-0.203, -0.018] * | +1.039 [+0.258, +1.871] * | +0.0134 [+0.0032, +0.0246] * |
| H_s1 - B_s1 | -0.004 [-0.113, +0.096]  | +0.003 [-0.025, +0.031]  | +0.946 [+0.441, +1.473] * | -0.0017 [-0.0081, +0.0036]  |
| H_s0 - P10 | +0.282 [-0.084, +0.649]  | +0.090 [+0.005, +0.190] * | +0.566 [+0.032, +1.150] * | -0.0319 [-0.0588, -0.0085] * |
| H_s1 - P10 | +0.381 [-0.041, +0.833]  | +0.119 [+0.009, +0.242] * | +0.520 [-0.156, +1.248]  | -0.0218 [-0.0498, +0.0017]  |

`*` = CI excludes 0. Artifact: `OUT/dk_read.json` (`pairs`, `arms`), per-window `OUT/dk_perwindow.npz`.

**Reading.** Seed 0: H holds LESS headway (-0.21 m) and time-gap (-0.10 s) than B and has more LEAD-VIOL (29 vs 21 of 595 at-risk windows, separated); seed 1: none of the three is separated and LEAD-VIOL is 35 vs 36. Min-TTC is higher at both seeds (separated); it is a censored mean (30 s cap), so it is not read as 'better' on its own. Against P10, H keeps a LONGER time-gap (separated at both seeds) and has fewer LEAD-VIOL (separated at seed 0 only). Seed-dependent, so not a finding about H; it is a flag.

### 6b. T's own decision heads and S's checkpoints: the structural identity

* **TACTICAL (T's heads: lat / lon macro-F1 and goal-token macro-F1):** H's block equals B's banked block **exactly** (json equality) at s0: True and s1: True. Values (identical for B and H): lat macro-F1 0.8157, lon 0.4992, goal 0.2618 (s0); s1: lat 0.8292, lon 0.5095, goal 0.2621.
* **The PREREG's sentence 'tactical ... must read EXACTLY B's values' holds for T's heads and does NOT hold for the plan-derived tactical agreement of section 6** (it is read from the trajectory and moves with H). This is a wording gap in the PREREG, reported here rather than hidden.
* **STRATEGIC: PAUSED** per the PI ruling of 2026-09-19 (`PI_DECISION_QUEUE.md` ITEM 25): the strategic layer is switched off temporarily; the family is reported as PAUSED, not as a pass. S's checkpoint errors are B's by construction (H never touches S; banked `ck_pos_err` carried unchanged; B's block is in `OUT/score_*.json::structural_identity.strategic_block_S_checkpoints`).

## 7. CONSISTENCY: the read the PREREG table does not carry, and where H pays

CONSISTENCY(a) = share of windows whose PLAN violates a GT-derived constraint (speed limit, stop, turn, lead gap); (b) = violation of the model's OWN predicted constraints. Lower is better. Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0. Artifact: `OUT/score_M2b_s*.json::consistency_beside`; P10 rates from `floors_perwindow.npz`.

| seed | rule | B | H | P10 | H - B | H - P10 |
|---|---|---|---|---|---|---|
| s0 | any | 0.1596 | 0.2601 | 0.3110 | +0.1005 [+0.0708, +0.1325] **WORSE** | -0.0509 [-0.0813, -0.0213] * |
| s0 | SPEED | 0.0955 | 0.1955 | 0.2418 | +0.1000 [+0.0717, +0.1277] **WORSE** | -0.0463 [-0.0717, -0.0204] * |
| s0 | STOP | 0.0188 | 0.0292 | 0.0338 | +0.0104 [+0.0050, +0.0167] **WORSE** | -0.0046 [-0.0096, -0.0008] * |
| s0 | TURN | 0.0042 | 0.0071 | 0.0342 | +0.0029 [+0.0004, +0.0067] **WORSE** | -0.0271 [-0.0400, -0.0154] * |
| s0 | LEAD | 0.0725 | 0.1042 | 0.1384 | +0.0317 [+0.0183, +0.0471] **WORSE** | -0.0342 [-0.0509, -0.0188] * |
| s0 | self (b) any | 0.2297 | 0.3697 | n/a | +0.1401 [+0.1113, +0.1700] **WORSE** | n/a |
| s1 | any | 0.2097 | 0.2851 | 0.3110 | +0.0754 [+0.0571, +0.0945] **WORSE** | -0.0258 [-0.0538, +0.0021]  |
| s1 | SPEED | 0.1255 | 0.2238 | 0.2418 | +0.0984 [+0.0755, +0.1221] **WORSE** | -0.0179 [-0.0392, +0.0050]  |
| s1 | STOP | 0.0271 | 0.0296 | 0.0338 | +0.0025 [+0.0008, +0.0046] **WORSE** | -0.0042 [-0.0083, -0.0012] * |
| s1 | TURN | 0.0067 | 0.0071 | 0.0342 | +0.0004 [-0.0017, +0.0025]  | -0.0271 [-0.0409, -0.0146] * |
| s1 | LEAD | 0.1084 | 0.1263 | 0.1384 | +0.0179 [+0.0100, +0.0267] **WORSE** | -0.0121 [-0.0271, +0.0046]  |
| s1 | self (b) any | 0.1430 | 0.2172 | n/a | +0.0742 [+0.0538, +0.0971] **WORSE** | n/a |

**H is separated-worse than B on CONSISTENCY(a) 'any' at both seeds (+0.101 and +0.075), on SPEED (+0.100 and +0.098), and on the model's own self-consistency (b) (+0.140 and +0.074).** It is still better than P10 (separated at s0 only for 'any'). Mechanism (HYPOTHESIS, consistent with the numbers, not isolated): B's residual was trained with the consistency penalties in the loss, so part of the residual IS the constraint-satisfying correction; shrinking it back toward the unconstrained P10 gives that part back, and the fit minimises position error only. The weights have no term for the constraints. Whether the trade (ADE and acceleration for constraint violations) is acceptable is a decision for the Master Mind / PI, not for this read; the PREREG's PASS does not decide it.

## 8. SECONDARY (descriptive) and diagnostics

**2-fold episode cross-fit on eval-141** (fold = parity of the sorted-episode index; fit on one fold, score the other, swap; **CROSS-FITTED, not a deployable weight**; SECONDARY, descriptive). Paired vs B:

| seed | ADE2 CF - B | ADE6 CF - B | accel CF - B | ADE2 / ADE6 CF (m) |
|---|---|---|---|---|
| s0 | -0.1026 [-0.1386, -0.0753] | -0.2009 [-0.2846, -0.1254] | -0.4655 [-0.5514, -0.3816] | 0.2130 / 1.9744 |
| s1 | -0.0747 [-0.1093, -0.0512] | -0.0401 [-0.0782, -0.0044] | -0.2125 [-0.2664, -0.1644] | 0.2167 / 2.0145 |

The cross-fit reproduces the train-fit result at both seeds (same sign, all separated; ADE6 s1 -0.040 vs -0.070 with half the episodes in each fit).

**Diagnostics, declared as diagnostics (never a bar):**

| diagnostic | seed | ADE2 - B | ADE6 - B | accel - B | reading |
|---|---|---|---|---|---|
| eval-oracle w (fitted ON eval-141, reads GT; the ceiling of this lever) | s0 | -0.1038 [-0.1416, -0.0754] | -0.2108 [-0.2920, -0.1367] | -0.5013 [-0.5958, -0.4111] | train-fitted H captures nearly all of it |
| noise control: w fitted on train with the residual PERMUTED across windows | s0 | -0.0949 [-0.1361, -0.0636] | +0.2802 [+0.1367, +0.4273] | n/a | w collapses to ~0.00-0.05 (H = P10), ADE6 gets WORSE than B |
| eval-oracle w (fitted ON eval-141, reads GT; the ceiling of this lever) | s1 | -0.0764 [-0.1121, -0.0521] | -0.0708 [-0.1086, -0.0364] | -0.2429 [-0.3042, -0.1883] | train-fitted H captures nearly all of it |
| noise control: w fitted on train with the residual PERMUTED across windows | s1 | -0.0709 [-0.1131, -0.0417] | +0.4044 [+0.2714, +0.5349] | n/a | w collapses to ~0.00-0.05 (H = P10), ADE6 gets WORSE than B |

Two things these show. (i) The train-fitted weights sit at the ceiling of the lever: H's ADE6 gain (-0.207 / -0.070) is within 0.004 m of what weights fitted on eval itself would give (-0.211 / -0.071), so a better weight fit has nothing left to buy. (ii) **PRIMARY-S is a bar against B, and B is easy to beat at 2 s by returning to P10**: the permuted-residual control (H = P10 essentially) also passes ADE2 H - B, separated at both seeds. So PRIMARY-S alone does not show that shrinkage found signal; what shows it is that the fitted H also beats P10 at 2 s (section 4, separated at both seeds) and that the permuted control LOSES to B at 6 s (separated worse at both seeds) while the fitted H wins.

## 9. The outcome table, applied verbatim

| outcome | condition | met? |
|---|---|---|
| **PASS** | PRIMARY-S and PRIMARY-N at both seeds | **True** (S: True, N: True) |
| SHORT ONLY | PRIMARY-S passes, PRIMARY-N fails | False |
| NO GAIN | PRIMARY-S fails | False |

**Result: PASS.** Committed next (PREREG): H becomes the deployed v8F output. The shrinkage weights become a model buffer in the next trained arm, and the FLAT and MIXO reads are re-scored with H applied to each arm's own fitted weights (descriptive).

The functional form, the clip range, the isotonic step and the bars were not changed after any eval was seen. The weights were fitted once per seed before the first eval read; the script refuses to read unless the identity controls pass. A first run of `score` with `--no-families --n-boot 100` (a smoke to flush bugs) was made after the fit and before the full read; it printed the same point estimates; no choice depended on it. `score` was then re-run once after adding the CONSISTENCY block (section 7); the bootstrap is seeded and every PREREG row reproduced to the printed digit.

## 10. What this does and does not show

* **Evidence class.** All rows MEASURED (this run). B's per-window plans are the banked KIND-run outputs (INHERITED from box 2's GPU run; re-derived metrics reproduce the banked arrays with max abs diff 0.0 and the full `evaluate()` block with 0 mismatches; the CPU forward reproduces them to <= 2.8e-5 m on 85 windows).
* **Tier.** T1-class open loop on 2,399 windows / 141 episodes. Not closed loop: nothing here says H drives better, only that its open-loop plan is closer to the logged future and smoother in acceleration.
* **Variance.** The bootstrap resamples episodes with the models fixed. Training-run variance is covered only by the two seeds, which agree in sign on every PREREG row but differ in size (ADE6 H - B -0.207 vs -0.070; the s1 arm starts from a better B, ADE6 2.175 vs 2.055, and shrinkage buys less). Two seeds are a floor check, not an interval.
* **Fit-set caveat (declared, in the PREREG).** The fit windows are windows B was trained on; the weights therefore shrink LESS than eval-optimal ones. The eval-oracle diagnostic and the cross-fit show the effect is not an artifact of that: the train-fitted weights reach the eval-oracle ceiling.
* **Not done.** The eval CPU forward over all 2,399 windows (the eval cache is 2.8 GB; the budget could not hold it beside the 2.9 GB train copy and deletion needs the PI). It is not needed for the read (B's tau is banked, P is image-free) and the 85-window cross-check covers the route. No H + hard-speed-projection read (needs traj60 on every eval window, i.e. the full CPU forward).

## 11. Rule Zero: the levers this leaves, ranked by measured effect (nothing below is a new bar; each needs its own PREREG)

1. **The committed next step (PASS):** weights become a model buffer in the next trained arm; FLAT and MIXO re-scored with H on each arm's own fitted weights (descriptive). Unblocked on the fit side (the 141-clip fit set is on disk, about 1 minute of fit per checkpoint); needs those arms' checkpoints (their location was not checked here).
2. **Constraint-aware H.** The measured cost of H is CONSISTENCY(a) (+0.075 to +0.100 'any', SPEED +0.10). Cheapest experiment: fit the SAME closed form but only on the windows where no constraint is active, or add the hard speed projection after H (B's own projection variant exists, `traj30_proj`). Blocked on: H on traj60 for every eval window = the full CPU forward (eval cache 2.8 GB, needs a disk decision) or one box-2 run.
3. **Acceleration gap to P10** (+0.104 / +0.066 m/s^2, separated): the weights are at their ceiling, so this is not a weight problem. Candidate: shrink per DCT coefficient instead of per step (the residual's high-frequency part is the jitter). Zero training, same fit machinery.

## 12. Deliverable manifest

All under `OUT` (dev-box scratchpad; **nothing staged to git, nothing pushed, nothing written on box 2**). The Master Mind lands them.

| artifact | path (under OUT) | note |
|---|---|---|
| this report | `HBLEND_RESULT.md` | generated by build_result.py from the JSONs below |
| executor (fit / score / read / dk / xcheck) | `v8f_hblend.py` | imports the eval's own `evaluate`, `taniteval.ci`, `v8f_beta_eval`, the DK reader |
| report generator | `build_result.py` |  |
| self-tests | `test_v8f_hblend.py` | 7 passed |
| independent weight re-derivation | `verify_fit.py, fit_verify.json` | lstsq + sklearn isotonic |
| fitted weights | `weights_M2b_s0.json, weights_M2b_s1.json` | THE deployable artifact: 60 numbers per seed + raw/clipped/derived |
| fit-set predictions | `fit_windows_M2b_s{0,1}.npz` | 9,582 windows: traj30, p30, gt, ages |
| fit clip list | `fit_clips.json, train_clip_order.json` | k = 32, 141 clips, sha256 above |
| per-seed score (every row, CI, blocks, controls) | `score_M2b_s0.json, score_M2b_s1.json` |  |
| per-window H / B / P10 arrays | `score_M2b_s{0,1}_perwindow.npz` | H, P, tau, per-window ADE/FDE/family components |
| outcome | `READ.json` | the table, verbatim |
| distance keeping | `dk_read.json, dk_perwindow.npz` | via v8f_dk_read.read_dk |
| route cross-check | `xcheck.json` | 85 windows CPU vs banked |
| logs | `fit_M2b_s*.log, score_M2b_s*.log, read.log, dk.log, copy_log.txt` |  |
| inputs fetched (read-only) | `inputs/ (banked B npz+json from the mirror, P10-4512 file, floors, lead block), box2/ (manifest, join_train.npz, argv, summaries, sftp batches), cache/train/clips/ (141 clips, 2.946 GB), deps/ (taniteval copy + DK reader)` | copies; originals untouched |
| smoke leftovers | `weights_smoke.json, fit_windows_smoke.npz` | 3-clip smoke of the fit; not used in any read |

Footprint on C: about 2.99 GB (2.946 GB train clips, 35 MB join, the rest small); nothing on E:; nothing deleted. `df -h /c` before the copy: 11 GB free; after: 7.4 GB (other processes also write to C:).

## 13. Reproduce

```
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY v8f_hblend.py fit   --seed M2b_s0   # ~52 s
$PY v8f_hblend.py fit   --seed M2b_s1
$PY v8f_hblend.py score --seed M2b_s0   # ~31 s, identity controls first
$PY v8f_hblend.py score --seed M2b_s1
$PY v8f_hblend.py read
$PY v8f_hblend.py dk
$PY v8f_hblend.py xcheck
$PY build_result.py
```
Paths are hard-coded to the scratchpad (`SCR` in `v8f_hblend.py`); relocate before landing. Pinned inputs: tip a99ca5d6 export at `viz/tree`, `taniteval` modules copied from box 2 (`viz/work/taniteval_b2`) plus `lead_metrics` / `lead_source` / `progress` / `v0_antiecho` from the mirror tip 25037c4.

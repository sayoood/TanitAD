# HPROJ_RESULT: PREREG_V8F_HPROJ executed exactly as written (2026-10-25)

Dev box, CPU only, ZERO training, no GPU, offline from banked outputs. Every number is **MEASURED (ours)** from the artifact paths named; tier **T1-class open loop (own held outputs; not closed loop)**; estimator **paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0** unless a row says otherwise. The bootstrap answers 'would another draw of EPISODES say this?' only; training-seed variance enters only through the two seeds shown side by side (`H-ESTIM-SEED-1`); inference is deterministic, so no inference-seed floor applies. `OUT` = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/hproj`. Contract: `Project Steering/PREREG_V8F_HPROJ.md` at c5ac9a9f. B = B's banked UNPROJECTED plan (its headline numbers), H = HBLEND's H, HP = the arm, Bp = B's own banked projected plan (`traj30_proj`), P10 = the P10-yaw-4512 plan.

## 0. Outcome (the PREREG outcome table, applied verbatim)

**CONSISTENCY NOT RESTORED. This is a FAIL of the pre-registered bar, reported as written.** PRIMARY-C fails at both seeds; PRIMARY-K fails at both seeds (s0: HP - B on ADE2 and ADE6 is positive and not separated; s1: both are positive and separated WORSE than B). The table's row 'CONSISTENCY NOT RESTORED' (PRIMARY-C fails) is the first to match; 'GAIN LOST' requires PRIMARY-C to PASS, so it does not apply. HP is **not** the deployed v8F output and H's weights do not enter `forward_infer` as a buffer on this evidence.

| read | seed | value | 95 % CI | bar | verdict |
|---|---|---|---|---|---|
| **PRIMARY-C** CONSISTENCY(a) 'any', HP - B | s0 | +0.0538 | [+0.0367, +0.0717] | upper bound <= +0.02 | **FAIL** (upper +0.0717) |
| **PRIMARY-C** CONSISTENCY(a) 'any', HP - B | s1 | +0.0313 | [+0.0146, +0.0488] | upper bound <= +0.02 | **FAIL** (upper +0.0488) |
| **PRIMARY-K** ADE6, HP - B (m) | s0 | +0.0071 | [-0.1231, +0.1553] | < 0 and separated | **FAIL** (not separated, worse) |
| **PRIMARY-K** ADE2, HP - B (m) | s0 | +0.0227 | [-0.0213, +0.0744] | < 0 and separated | **FAIL** (not separated, worse) |
| **PRIMARY-K** ADE6, HP - B (m) | s1 | +0.1273 | [+0.0107, +0.2714] | < 0 and separated | **FAIL** (separated, worse) |
| **PRIMARY-K** ADE2, HP - B (m) | s1 | +0.0479 | [+0.0027, +0.1015] | < 0 and separated | **FAIL** (separated, worse) |

Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0. Tier: T1-class open loop (own held outputs; not closed loop). Artifacts: `OUT/score_M2b_s{0,1}.json` (`consistency.paired.HP_minus_B.viol_any`, `paired.HP_minus_B.ade2/ade6`), outcome `OUT/READ.json`.

**Committed next step (PREREG, verbatim):** *H's violations are not of the posted limit: they are stops, leads or turns, which the cap does not see. Next: a constraint-aware fit under its own PREREG. It is the same closed form, fitted only on training windows with no active GT constraint, plus O's predicted stop / lead constraints in the projection.*

**Rule Zero.** The table's next step is a constraint-aware fit under its OWN PREREG. It is not run here: the PREREG fixes this arm, its bars and its fit, and a new fit is a new arm that needs its own pre-registered bars. What this run does leave behind is the measured mix the next PREREG must be written against, from two post-hoc diagnostics (sections 8 and 9; descriptive, no bar, they change no number above). Three measured facts, in order of effect:

1. **The PREREG's explanation is only partly what the data show.** It predicted that H's violations are 'stops, leads or turns, which the cap does not see'. The per-rule breakdown (section 7) puts the largest excess on **SPEED**, with LEAD second (HP - B on LEAD +0.0267 (s0), +0.0121 (s1), which the projection does not touch: HP - H on LEAD -0.0050, -0.0058) and STOP / TURN small. HP - B on SPEED is +0.0529 (s0), +0.0492 (s1) against 'any' +0.0538 (s0), +0.0313 (s1). But the SPEED rule's cap is min(posted limit, **v_hi_ms**), and v_hi_ms is a GT-derived speed-band edge (T predicts it as a `cons_pred` field), not the posted-limit input the projection reads: it is the binding cap in **2084 of 2399 windows**, and 356 of 356, 419 of 419 of HP's SPEED violations (s0, s1) sit in such windows. Projection onto the posted limit alone cannot remove them (by construction: wherever the posted limit is the binding cap, HP respects it). B's own projection (Bp) barely moves SPEED either (section 7).
2. **The projection as specified also costs position accuracy, for a separable reason.** `project_speed` caps every 0.1 s interval from t = 0, but the SPEED rule reads only intervals starting at tau >= 2.0 s. Applied alone to B it already costs ADE6 +0.2453 (s0), +0.1880 (s1) (separated worse). With the cap applied from 2.0 s (the rule's own band; section 9, diagnostic) the ADE2 cost disappears (HP ADE2 - B -0.0967 (s0), -0.0711 (s1)) but CONSISTENCY is still not restored with the posted limit alone (HP 'any' - B upper bound +0.0759, +0.0517).
3. **A deployable-class v_hi term is not enough on its own.** Adding T's own predicted v_hi_ms to the cap (section 9, banded) moves 'any' - B to +0.0229 (s0), +0.0188 (s1) (upper bounds +0.0396, +0.0354, bar +0.02) and costs ADE6 +0.0427, +0.0237 (T's v_hi head is off from the GT label by 1.68, 1.68 m/s MAE). The T0 ORACLE v_hi (reads a GT label; a ceiling, never a result) restores consistency ('any' - B -0.0463, -0.0784) and keeps ADE6 below B (-0.2760, -0.1753), so the constraint is recoverable if v_hi can be predicted well enough; the open lever is the quality of T's v_hi prediction, not the projection or the shrinkage.

**What blocks a finished result here** (named, per Rule Zero): the next arm (a banded, v_hi-aware, constraint-aware projection with its fit) needs its own PREREG with committed bars before any number is read, and it needs a decision on whether the consistency SPEED rule's GT-derived v_hi term is the right target for a deployable projection at all (the human violates the posted limit alone in 6.3 % of these windows, so the posted limit is not a constraint the GT itself respects). Both are Master Mind / PI calls; no GPU, no box 2.

## 1. What was run

* **Route (as the PREREG: eval-only, offline).** B's banked per-window plan `pred` (KIND run own code, `OUT/../hblend/inputs/banked/eval_M2b_s{0,1}-kind-own.{npz,json}`, sha256 in each score json) is the only model output used. P = the checkpoint's own frozen P10 on the eval kinematic features (no image needed). GT, cap source (`J['limit']`, `J['limit_ok']`) and the scoring context are the eval's own (`v8f_mini_data.eval_context`, `v8f_mini_eval.evaluate`). Weights: HBLEND's `weights_M2b_s{0,1}.json`, **not re-fitted**.
* **tau_B^(60)** = up(P10) + Basis c. Basis built from the formula (`cos(k pi (i+0.5)/60)` x ramp `(i+1)/60`, nb = 12 from the checkpoint cfg); c by least squares on the 30 eval-step rows (`basis[1::2]`, rank 12, condition number 22.5).
* **tau_H^(60)** = up(P10) + w60 (.) (tau_B^(60) - up(P10)), ego frame at NOW (axis 0 along/x, axis 1 cross/y); w60 = the banked 30 knots linearly interpolated to 0.1 s with w(0.1) = w(0.2) (equal to HBLEND's own derived grid to 0.0 but computed independently with `np.interp`).
* **HP** = `project_speed(tau_H^(60), cap)[:, 1::2]`, torch float32 CPU (the eval's own precision), cap per window = `limit_ms_from_onehot(J['limit'][clip], J['limit_ok'][clip])`, 1e9 where there is no limit (here every one of the 2399 windows has a posted limit; values 30 km/h, 50 km/h, 100 km/h, 120 km/h).
* **Scoring.** Every metric comes from the eval's OWN `evaluate()` (traj, consistency (a), self-consistency (b), tactical, families) on HP, H and Bp; paired rows via `v8f_beta_eval.paired_row` (episode-cluster bootstrap, B = 2000). Distance keeping via `v8f_dk_read.read_dk` -> `taniteval.lead_metrics`. Code: `OUT/v8f_hproj.py` imports HBLEND's loaders read-only.

## 2. Reconstruction control (run BEFORE any HP number; the script refuses to read and exits rc 2 if a binding control fails)

| control | bar | seed 0 | seed 1 | binding? |
|---|---|---|---|---|
| **C0** P10 plan file vs the model's internal p30, max abs (m) | <= 1e-4 | 1.526e-05 | 1.526e-05 | yes (task step 1) |
| **C1** up(P10) + Basis c, subsampled, vs B's banked traj30, max abs (m) | <= 1e-4 | 2.022e-05 | 2.092e-05 | **yes (PREREG)** |
| **C2** `project_speed` on B's reconstructed traj60: ADE6 2.42058 / 2.24260 vs banked `projected_r1_kamm` ADE6 2.4206 / 2.2426, abs diff (m) | <= 1e-3 | 2.326e-05 | 9.606e-07 | **yes (PREREG)** |
| **all binding controls pass** | | **True** | **True** | |

Diagnostics added by the executor (reported, not bars, cannot change the table):

| diagnostic | seed 0 | seed 1 |
|---|---|---|
| E1 basis formula vs the model's own `basis` buffer, max abs | 2.053e-06 | 2.053e-06 |
| E1 least-squares fit residual on the 30 rows, max abs (m) | 2.022e-05 | 2.092e-05 |
| E2 cap vs `eval_context` limit_ms (float64 route), max abs (m/s) | 1.272e-06 | 1.272e-06 |
| E3 reconstructed projected plan vs the eval's saved `traj30_proj`, max abs (m) / mean abs (m) | 1.564e-04 | 1.183e-04 |
| E3 mean abs (m) | 1.012e-06 | 1.001e-06 |
| E3 ADE6 of the saved `traj30_proj` (exact mean) vs reconstruction, abs diff (m) | 1.046e-07 | 5.281e-08 |
| E3 whole `projected_r1_kamm` block (traj, consistency_a, strata) recon vs banked json, mismatches at 1e-6 | 0 | 1 |
| E4 w60 vs HBLEND `w60_derived_unscored`, max abs | 1.665e-16 | 1.665e-16 |
| E5 H60 knots vs HBLEND's saved H, max abs (m) | 1.608e-05 | 1.796e-05 |
| E5 w = 1 gives tau60 / w = 0 gives up(P10) (max abs) | 0.000e+00 | 0.000e+00 |
| E5 numpy `up_np` vs torch `upsample_prior`, max abs (float32 rounding) | 7.629e-06 | 7.629e-06 |
| projection float64 vs float32, max abs (m) | 6.515e-05 | 7.689e-05 |
| **R1** deliberate regression: basis WITHOUT the ramp, knots max abs (m); must exceed 1e-4 | 0.458 (control fails as it must: True) | 0.531 (True) |
| **R2** deliberate regression: NO cap, ADE6 abs diff vs banked json (m); must exceed 1e-3 | 0.2423 (True) | 0.1875 (True) |
| **R3** deliberate regression: NO projection, ADE6 abs diff vs banked json (m); must exceed 1e-3 | 0.2453 (True) | 0.1880 (True) |
| **X1** CPU `forward_infer` TRUE traj60 vs the reconstruction, all 60 steps, max abs (m), 85 windows whose features are on this box | 2.755e-05 | 1.755e-05 |
| X1 same, the 0.1 s MIDPOINTS only (rows no banked file carries) | 2.681e-05 | 1.755e-05 |
| X1 cap built by the eval loader's own batch vs cap from `J`, max abs (m/s) | 0.0e+00 | 0.0e+00 |
| X1 `project_speed` on the model's TRUE traj60 vs on the reconstruction, max abs (m) | 8.011e-05 | 5.722e-05 |
| H block re-evaluated through `evaluate()` vs HBLEND's banked block, mismatches | 0 | 0 |

Reading: the reconstruction is exact to float32 rounding, including the 0.1 s midpoints that no banked file carries (X1), and the cap the eval loader builds is bit-identical to the one rebuilt from `J` (X1). The reference arm H, re-scored here, equals HBLEND's banked H block with 0 mismatches. The eval's saved `traj30_proj` differs from the reconstruction by at most 1.6e-04 m on any coordinate of any window (mean abs ~1e-6 m); that bound sits just above 1e-4 m at s0, which is why E3 is a diagnostic and the PREREG's binding forms are C1 and C2, which pass with margin (worst C1 2.09e-05 against 1e-4; worst C2 2.33e-05 against 1e-3). The E3 whole-block comparison has 0 mismatches at s0 and 1 at s1: `/traj/acts/zero_accel_frac: 0.03508645843814233 vs 0.035072084632964885` -- ONE of the 69571 acceleration steps crosses the 0.001 m/s^2 zero-acceleration tolerance (float noise in a counter, not in a plan); reported, not hidden.

## 3. The PREREG rows and the dominance row (HP against P10), beside rows

| row | seed | value | 95 % CI | estimator | tier | artifact |
|---|---|---|---|---|---|---|
| CONSISTENCY(a) 'any' HP - B | s0 | +0.0538 | [+0.0367, +0.0717] | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s0.json::consistency.paired.HP_minus_B.viol_any` |
| ADE6 HP - B (m) | s0 | +0.0071 | [-0.1231, +0.1553] | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s0.json::paired.HP_minus_B.ade6` |
| ADE2 HP - B (m) | s0 | +0.0227 | [-0.0213, +0.0744] | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s0.json::paired.HP_minus_B.ade2` |
| CONSISTENCY(a) 'any' HP - B | s1 | +0.0313 | [+0.0146, +0.0488] | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s1.json::consistency.paired.HP_minus_B.viol_any` |
| ADE6 HP - B (m) | s1 | +0.1273 | [+0.0107, +0.2714] | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s1.json::paired.HP_minus_B.ade6` |
| ADE2 HP - B (m) | s1 | +0.0479 | [+0.0027, +0.1015] | paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0 | T1-class open loop (own held outputs; not closed loop) | `OUT/score_M2b_s1.json::paired.HP_minus_B.ade2` |

**Dominance row, HP against P10-yaw-4512** (ADE2, ADE6, accel MAE; paired):

| seed | ADE2 HP / P10 (m) | ADE6 HP / P10 (m) | ADE2 HP - P10 | ADE6 HP - P10 | accel MAE HP / P10 (m/s^2) | accel HP - P10 | point estimates all <= P10 |
|---|---|---|---|---|---|---|---|
| s0 | 0.3383 / 0.2221 | 2.1824 / 2.4870 | +0.1162 [+0.0477, +0.2002] *WORSE* | -0.3046 [-0.4846, -0.1009] *better* | 0.7526 / 0.6376 | +0.1150 [+0.0919, +0.1412] *WORSE* | ADE2 False, ADE6 True, accel False |
| s1 | 0.3394 / 0.2221 | 2.1818 / 2.4870 | +0.1173 [+0.0487, +0.2008] *WORSE* | -0.3051 [-0.4851, -0.0997] *better* | 0.7176 / 0.6376 | +0.0800 [+0.0590, +0.0999] *WORSE* | ADE2 False, ADE6 True, accel False |

HP does not dominate P10: at 2 s it is separated WORSE than P10 at both seeds (the cap from t = 0), at 6 s it is separated better, and its acceleration MAE stays above P10's.

**HP against B's own projected plan Bp** (the other projection variant; paired):

| seed | ADE2 HP - Bp | ADE6 HP - Bp | accel MAE HP - Bp | CONSISTENCY 'any' HP - Bp |
|---|---|---|---|---|
| s0 | -0.0483 [-0.0565, -0.0405] *better* | -0.2382 [-0.3176, -0.1672] *better* | -0.3214 [-0.3685, -0.2761] *better* | +0.0613 [+0.0438, +0.0788] |
| s1 | -0.0302 [-0.0362, -0.0244] *better* | -0.0608 [-0.0872, -0.0347] *better* | -0.1315 [-0.1521, -0.1117] *better* | +0.0525 [+0.0404, +0.0655] |

Bp (B with the same projection) is better than HP on consistency (HP - Bp positive, separated at both seeds) and worse on ADE and acceleration (HP - Bp negative, separated). Neither dominates the other. (HYPOTHESIS, not isolated: the shrinkage moves the plan away from where B's consistency-trained residual put it, and the posted-limit projection does not bring that back.) `HP - H`, the cost of adding the projection to H, is in sections 4 and 7.

## 4. ADE and FDE at 1-6 s (beside)

**M2b_s0** (m; mean [95 % CI]; `*better*` / `*WORSE*` = CI excludes 0)

| metric | B | H | HP | Bp | P10 | HP - B | HP - H |
|---|---|---|---|---|---|---|---|
| ADE@1s | 0.0819 [0.0714, 0.0953] | 0.0598 [0.0511, 0.0728] | 0.1458 [0.1032, 0.1952] | 0.1648 [0.1218, 0.2140] | 0.0633 [0.0545, 0.0764] | +0.0639 [+0.0261, +0.1093] *WORSE* | +0.0860 [+0.0449, +0.1350] *WORSE* |
| ADE@2s | 0.3155 [0.2765, 0.3621] | 0.2123 [0.1916, 0.2392] | 0.3383 [0.2662, 0.4224] | 0.3866 [0.3139, 0.4705] | 0.2221 [0.2008, 0.2491] | +0.0227 [-0.0213, +0.0744] | +0.1260 [+0.0608, +0.2052] *WORSE* |
| ADE@3s | 0.5772 [0.5254, 0.6401] | 0.4734 [0.4324, 0.5198] | 0.6330 [0.5296, 0.7521] | 0.7147 [0.6078, 0.8391] | 0.5337 [0.4865, 0.5871] | +0.0558 [-0.0111, +0.1343] | +0.1597 [+0.0727, +0.2660] *WORSE* |
| ADE@4s | 0.9838 [0.9078, 1.0702] | 0.8541 [0.7859, 0.9292] | 1.0417 [0.9088, 1.1955] | 1.1751 [1.0340, 1.3376] | 1.0129 [0.9246, 1.1071] | +0.0579 [-0.0320, +0.1646] | +0.1876 [+0.0814, +0.3203] *WORSE* |
| ADE@5s | 1.5179 [1.4027, 1.6470] | 1.3530 [1.2458, 1.4681] | 1.5578 [1.3950, 1.7489] | 1.7430 [1.5656, 1.9474] | 1.6628 [1.5163, 1.8107] | +0.0399 [-0.0713, +0.1696] | +0.2048 [+0.0843, +0.3574] *WORSE* |
| ADE@6s | 2.1753 [2.0071, 2.3568] | 1.9685 [1.8106, 2.1306] | 2.1824 [1.9778, 2.4152] | 2.4206 [2.1936, 2.6816] | 2.4870 [2.2606, 2.7135] | +0.0071 [-0.1231, +0.1553] | +0.2139 [+0.0852, +0.3776] *WORSE* |
| FDE@1s | 0.1220 [0.1066, 0.1443] | 0.1209 [0.1054, 0.1429] | 0.2581 [0.1881, 0.3385] | 0.2692 [0.1990, 0.3501] | 0.1288 [0.1129, 0.1509] | +0.1361 [+0.0695, +0.2147] *WORSE* | +0.1372 [+0.0706, +0.2159] *WORSE* |
| FDE@2s | 0.7022 [0.6156, 0.8066] | 0.5603 [0.5105, 0.6173] | 0.7605 [0.6302, 0.9088] | 0.8170 [0.6855, 0.9698] | 0.6089 [0.5556, 0.6695] | +0.0584 [-0.0111, +0.1406] | +0.2002 [+0.0907, +0.3347] *WORSE* |
| FDE@3s | 1.4754 [1.3587, 1.6069] | 1.3354 [1.2289, 1.4543] | 1.5864 [1.3968, 1.8016] | 1.8002 [1.5916, 2.0397] | 1.5911 [1.4524, 1.7375] | +0.1110 [-0.0340, +0.2806] | +0.2510 [+0.1041, +0.4336] *WORSE* |
| FDE@4s | 2.7303 [2.5108, 2.9629] | 2.4830 [2.2813, 2.6914] | 2.7608 [2.5002, 3.0616] | 3.0960 [2.7991, 3.4326] | 3.0894 [2.8142, 3.3663] | +0.0305 [-0.1601, +0.2631] | +0.2778 [+0.1065, +0.5005] *WORSE* |
| FDE@5s | 4.3146 [3.9548, 4.6900] | 3.9699 [3.6405, 4.3013] | 4.2403 [3.8829, 4.6342] | 4.6688 [4.2648, 5.1144] | 5.1174 [4.6429, 5.5918] | -0.0743 [-0.2877, +0.1737] | +0.2704 [+0.0948, +0.5006] *WORSE* |
| FDE@6s | 6.2810 [5.7498, 6.8269] | 5.8185 [5.3261, 6.3198] | 6.0680 [5.5593, 6.5953] | 6.6272 [6.0598, 7.2386] | 7.6673 [6.9298, 8.4116] | -0.2130 [-0.4641, +0.0520] | +0.2494 [+0.0757, +0.4773] *WORSE* |

**M2b_s1** (m; mean [95 % CI]; `*better*` / `*WORSE*` = CI excludes 0)

| metric | B | H | HP | Bp | P10 | HP - B | HP - H |
|---|---|---|---|---|---|---|---|
| ADE@1s | 0.0706 [0.0613, 0.0832] | 0.0601 [0.0516, 0.0731] | 0.1462 [0.1035, 0.1956] | 0.1545 [0.1119, 0.2038] | 0.0633 [0.0545, 0.0764] | +0.0756 [+0.0364, +0.1210] *WORSE* | +0.0861 [+0.0450, +0.1349] *WORSE* |
| ADE@2s | 0.2914 [0.2541, 0.3397] | 0.2137 [0.1931, 0.2400] | 0.3394 [0.2671, 0.4235] | 0.3696 [0.2968, 0.4534] | 0.2221 [0.2008, 0.2491] | +0.0479 [+0.0027, +0.1015] *WORSE* | +0.1257 [+0.0606, +0.2043] *WORSE* |
| ADE@3s | 0.5525 [0.4983, 0.6177] | 0.4814 [0.4414, 0.5274] | 0.6362 [0.5338, 0.7532] | 0.6713 [0.5693, 0.7898] | 0.5337 [0.4865, 0.5871] | +0.0837 [+0.0171, +0.1634] *WORSE* | +0.1548 [+0.0700, +0.2589] *WORSE* |
| ADE@4s | 0.9361 [0.8585, 1.0205] | 0.8663 [0.7980, 0.9396] | 1.0442 [0.9115, 1.1944] | 1.0918 [0.9590, 1.2452] | 1.0129 [0.9246, 1.1071] | +0.1081 [+0.0188, +0.2120] *WORSE* | +0.1779 [+0.0755, +0.3047] *WORSE* |
| ADE@5s | 1.4403 [1.3246, 1.5633] | 1.3687 [1.2598, 1.4832] | 1.5612 [1.3956, 1.7483] | 1.6185 [1.4488, 1.8126] | 1.6628 [1.5163, 1.8107] | +0.1208 [+0.0148, +0.2477] *WORSE* | +0.1924 [+0.0760, +0.3385] *WORSE* |
| ADE@6s | 2.0546 [1.8859, 2.2323] | 1.9848 [1.8199, 2.1587] | 2.1818 [1.9716, 2.4120] | 2.2426 [2.0261, 2.4829] | 2.4870 [2.2606, 2.7135] | +0.1273 [+0.0107, +0.2714] *WORSE* | +0.1970 [+0.0736, +0.3559] *WORSE* |
| FDE@1s | 0.1214 [0.1062, 0.1432] | 0.1221 [0.1066, 0.1438] | 0.2588 [0.1889, 0.3393] | 0.2634 [0.1942, 0.3444] | 0.1288 [0.1129, 0.1509] | +0.1375 [+0.0709, +0.2160] *WORSE* | +0.1367 [+0.0705, +0.2150] *WORSE* |
| FDE@2s | 0.7080 [0.6210, 0.8234] | 0.5682 [0.5200, 0.6221] | 0.7652 [0.6364, 0.9108] | 0.7995 [0.6703, 0.9473] | 0.6089 [0.5556, 0.6695] | +0.0572 [-0.0149, +0.1390] | +0.1970 [+0.0895, +0.3291] *WORSE* |
| FDE@3s | 1.4102 [1.2925, 1.5347] | 1.3573 [1.2506, 1.4738] | 1.5835 [1.3951, 1.7986] | 1.6524 [1.4620, 1.8743] | 1.5911 [1.4524, 1.7375] | +0.1733 [+0.0367, +0.3352] *WORSE* | +0.2261 [+0.0879, +0.3923] *WORSE* |
| FDE@4s | 2.5848 [2.3687, 2.8044] | 2.5087 [2.3004, 2.7194] | 2.7651 [2.4987, 3.0627] | 2.8601 [2.5864, 3.1693] | 3.0894 [2.8142, 3.3663] | +0.1803 [+0.0092, +0.3956] *WORSE* | +0.2564 [+0.0921, +0.4697] *WORSE* |
| FDE@5s | 4.0775 [3.7120, 4.4422] | 4.0022 [3.6567, 4.3598] | 4.2435 [3.8658, 4.6592] | 4.3352 [3.9463, 4.7622] | 5.1174 [4.6429, 5.5918] | +0.1660 [-0.0173, +0.4095] | +0.2412 [+0.0674, +0.4724] *WORSE* |
| FDE@6s | 5.8582 [5.3224, 6.3994] | 5.8168 [5.2957, 6.3540] | 6.0223 [5.4833, 6.6000] | 6.0823 [5.5283, 6.6768] | 7.6673 [6.9298, 8.4116] | +0.1641 [-0.0150, +0.4006] | +0.2055 [+0.0370, +0.4405] *WORSE* |

Artifacts: `OUT/score_M2b_s*.json::single / paired.HP_minus_B / paired.HP_minus_H`.

## 5. The four families, beside (paired; lower is better except the two plan-derived tactical agreement rows)

**M2b_s0**

| family | metric | B | H | HP | P10 | HP - B | HP - H | HP - P10 |
|---|---|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 1.0674 | 0.8709 | 0.9222 | 0.9802 | -0.1452 [-0.1844, -0.1087] *better* | +0.0513 [+0.0238, +0.0854] *WORSE* | -0.0580 [-0.1172, +0.0072] |
| LONGITUDINAL | accel MAE (m/s^2) | 1.2058 | 0.7414 | 0.7526 | 0.6376 | -0.4532 [-0.5721, -0.3473] *better* | +0.0112 [-0.0280, +0.0428] | +0.1150 [+0.0919, +0.1412] *WORSE* |
| LONGITUDINAL | along MAE (m) | 1.8097 | 1.5925 | 1.8128 | 1.8217 | +0.0030 [-0.1266, +0.1544] | +0.2203 [+0.0896, +0.3864] *WORSE* | -0.0090 [-0.1654, +0.1747] |
| LATERAL | cross-track offset MAE (m) | 0.7662 | 0.7557 | 0.7576 | 1.1858 | -0.0086 [-0.0252, +0.0066] | +0.0019 [-0.0055, +0.0095] | -0.4282 [-0.5258, -0.3344] *better* |
| LATERAL | heading MAE (deg) | 4.7804 | 4.0259 | 3.9691 | 5.7302 | -0.7877 [-1.2759, -0.4197] *better* | -0.0568 [-0.1014, -0.0236] *better* | -1.7610 [-2.2108, -1.3423] *better* |
| LATERAL | yaw-rate MAE (deg/s) | 8.1167 | 6.4244 | 6.0400 | 6.3299 | -2.1676 [-3.0068, -1.4913] *better* | -0.3844 [-0.5739, -0.2249] *better* | -0.2899 [-0.5691, -0.0457] *better* |
| LATERAL | curvature MAE (1/m) | 0.0815 | 0.0536 | 0.0518 | 0.0462 | -0.0340 [-0.0562, -0.0191] *better* | -0.0018 [-0.0030, -0.0008] *better* | +0.0056 [+0.0024, +0.0093] *WORSE* |
| TACTICAL (plan-derived) | lateral decision agreement (higher better) | 0.8458 | 0.8629 | 0.8624 | 0.7728 | +0.0167 [+0.0042, +0.0304] *better* | -0.0004 [-0.0046, +0.0037] | +0.0896 [+0.0651, +0.1159] *better* |
| TACTICAL (plan-derived) | longitudinal decision agreement (higher better) | 0.5790 | 0.5969 | 0.5915 | 0.5336 | +0.0125 [-0.0083, +0.0367] | -0.0054 [-0.0183, +0.0134] | +0.0579 [+0.0221, +0.0922] *better* |

**M2b_s1**

| family | metric | B | H | HP | P10 | HP - B | HP - H | HP - P10 |
|---|---|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 0.9112 | 0.8389 | 0.8879 | 0.9802 | -0.0233 [-0.0393, -0.0071] *better* | +0.0490 [+0.0228, +0.0806] *WORSE* | -0.0923 [-0.1486, -0.0305] *better* |
| LONGITUDINAL | accel MAE (m/s^2) | 0.9313 | 0.7036 | 0.7176 | 0.6376 | -0.2137 [-0.3134, -0.1375] *better* | +0.0140 [-0.0284, +0.0459] | +0.0800 [+0.0590, +0.0999] *WORSE* |
| LONGITUDINAL | along MAE (m) | 1.6081 | 1.5573 | 1.7616 | 1.8217 | +0.1536 [+0.0337, +0.2982] *WORSE* | +0.2044 [+0.0785, +0.3665] *WORSE* | -0.0601 [-0.2160, +0.1244] |
| LATERAL | cross-track offset MAE (m) | 0.8644 | 0.8375 | 0.8363 | 1.1858 | -0.0282 [-0.0495, -0.0076] *better* | -0.0013 [-0.0060, +0.0042] | -0.3495 [-0.4506, -0.2548] *better* |
| LATERAL | heading MAE (deg) | 4.4383 | 4.3276 | 4.2788 | 5.7302 | -0.1572 [-0.2938, -0.0073] *better* | -0.0488 [-0.0817, -0.0230] *better* | -1.4530 [-1.9875, -0.9312] *better* |
| LATERAL | yaw-rate MAE (deg/s) | 6.9864 | 6.4447 | 6.1250 | 6.3299 | -0.9557 [-1.3036, -0.6547] *better* | -0.3197 [-0.4687, -0.1963] *better* | -0.2087 [-0.5612, +0.1502] |
| LATERAL | curvature MAE (1/m) | 0.0626 | 0.0572 | 0.0557 | 0.0462 | -0.0112 [-0.0189, -0.0049] *better* | -0.0015 [-0.0030, -0.0006] *better* | +0.0095 [+0.0032, +0.0172] *WORSE* |
| TACTICAL (plan-derived) | lateral decision agreement (higher better) | 0.8287 | 0.8470 | 0.8504 | 0.7728 | +0.0217 [+0.0079, +0.0392] *better* | +0.0033 [+0.0000, +0.0075] | +0.0775 [+0.0487, +0.1076] *better* |
| TACTICAL (plan-derived) | longitudinal decision agreement (higher better) | 0.5919 | 0.6132 | 0.5994 | 0.5336 | +0.0075 [-0.0121, +0.0304] | -0.0138 [-0.0292, +0.0071] | +0.0659 [+0.0221, +0.1118] *better* |

(`better`/`WORSE` already account for each metric's direction.) STRATEGIC family: **PAUSED** per the PI ruling of 2026-09-19 (`PI_DECISION_QUEUE.md` ITEM 25): the strategic layer is switched off temporarily; S is untouched by HP and its banked checkpoint errors are carried unchanged. It is reported as PAUSED, not as a pass. TACTICAL, T's own decision heads (lat / lon / goal macro-F1): HP's block equals B's banked block exactly (json equality) at s0: True and s1: True; the plan-derived tactical agreement above is read off the trajectory and does move with HP.

## 6. Distance keeping (the other half of LONGITUDINAL), K = 30 (6 s) lead block, via the programme's own `v8f_dk_read.read_dk` -> `taniteval.lead_metrics`

Lead block sha256 `7a879baa2b6f4c14...`, window states {'LEAD': 743, 'NO_LEAD': 552, 'NOT_STRAIGHT': 931, 'NO_LABEL': 173}, kinematic-contract check **OK**, arm-vs-itself identity control exactly +0 [0, 0]. Open loop: the lead's logged future does not react to the arm. No distance-keeping bar is registered; nothing here is PASS/FAIL. Min-TTC is a CENSORED mean (30 s cap).

| arm | headway min (m) | time-gap min (s) | min-TTC (s, censored) | LEAD-VIOL share of at-risk (n viol / 595) |
|---|---|---|---|---|
| human (GT path) | 25.665 | 3.995 | 19.790 | by definition 0 |
| B_s0 | 25.622 | 4.022 | 19.396 | 0.0353 (21) |
| H_s0 | 25.553 | 3.920 | 20.497 | 0.0487 (29) |
| HP_s0 | 25.840 | 3.939 | 20.948 | 0.0471 (28) |
| Bp_s0 | 25.974 | 4.047 | 20.472 | 0.0336 (20) |
| B_s1 | 25.544 | 3.950 | 19.524 | 0.0605 (36) |
| H_s1 | 25.606 | 3.950 | 20.438 | 0.0588 (35) |
| HP_s1 | 25.844 | 3.964 | 20.978 | 0.0521 (31) |
| Bp_s1 | 25.871 | 3.973 | 20.493 | 0.0555 (33) |
| P10 | 25.116 | 3.832 | 19.947 | 0.0807 (48) |

| paired contrast | headway (m) | time-gap (s) | min-TTC (s) | LEAD-VIOL at-risk (share) |
|---|---|---|---|---|
| HP_s0 - B_s0 | +0.053 [-0.285, +0.478] | -0.079 [-0.190, +0.003] | +1.509 [+0.393, +2.821] * | +0.0118 [+0.0017, +0.0229] * |
| HP_s0 - H_s0 | +0.257 [+0.022, +0.599] * | +0.017 [+0.002, +0.040] * | +0.464 [-0.051, +1.211] | -0.0017 [-0.0053, +0.0000] |
| HP_s0 - Bp_s0 | -0.266 [-0.446, -0.086] * | -0.101 [-0.209, -0.023] * | +0.414 [-0.203, +1.086] | +0.0134 [+0.0032, +0.0246] * |
| HP_s0 - P10 | +0.541 [+0.024, +1.119] * | +0.107 [+0.018, +0.209] * | +1.014 [+0.255, +1.906] * | -0.0336 [-0.0625, -0.0091] * |
| H_s0 - B_s0 | -0.207 [-0.400, -0.036] * | -0.095 [-0.203, -0.018] * | +1.039 [+0.258, +1.871] * | +0.0134 [+0.0032, +0.0246] * |
| HP_s1 - B_s1 | +0.235 [-0.047, +0.611] | +0.017 [-0.015, +0.052] | +1.487 [+0.620, +2.490] * | -0.0084 [-0.0206, +0.0000] |
| HP_s1 - H_s1 | +0.238 [+0.011, +0.573] * | +0.014 [+0.001, +0.035] * | +0.540 [+0.044, +1.256] * | -0.0067 [-0.0183, +0.0000] |
| HP_s1 - Bp_s1 | -0.062 [-0.168, +0.034] | -0.003 [-0.030, +0.025] | +0.500 [+0.149, +0.839] * | -0.0034 [-0.0087, +0.0000] |
| HP_s1 - P10 | +0.620 [+0.052, +1.243] * | +0.133 [+0.020, +0.259] * | +1.061 [+0.191, +2.068] * | -0.0286 [-0.0576, -0.0036] * |
| H_s1 - B_s1 | -0.004 [-0.113, +0.096] | +0.003 [-0.025, +0.031] | +0.946 [+0.441, +1.473] * | -0.0017 [-0.0081, +0.0036] |

`*` = CI excludes 0. Artifact: `OUT/dk_read.json`, per-window `OUT/dk_perwindow.npz`. Reading: s0: HP - B headway +0.053 m, LEAD-VIOL +0.0118 (separated); s1: HP - B headway +0.235 m, LEAD-VIOL -0.0084 (not separated).

## 7. CONSISTENCY(a) per constraint, and (b), for B, H and HP (the PREREG's PRIMARY-C lives here)

CONSISTENCY(a) = share of windows whose PLAN violates a GT-derived constraint (speed cap, stop, turn, lead gap); (b) = violation of the model's OWN predicted constraints. Lower is better. Estimator: paired episode-cluster bootstrap, B=2000, 141 episodes (`taniteval/ci.py`), seed 0. Artifact: `OUT/score_M2b_s*.json::consistency`; P10 rates from `floors_perwindow.npz`; Bp = B with the same projection.

**M2b_s0** (active windows: SPEED 2399, STOP 81, TURN 158, LEAD 1027 of 2399)

| rule | B | H | HP | Bp | P10 | H - B | **HP - B** | HP - H | HP - Bp |
|---|---|---|---|---|---|---|---|---|---|
| any | 0.1596 | 0.2601 | 0.2134 | 0.1521 | 0.3110 | +0.1005 [+0.0708, +0.1325] *WORSE* | **+0.0538 [+0.0367, +0.0717] *WORSE*** | -0.0467 [-0.0709, -0.0263] *better* | +0.0613 [+0.0438, +0.0788] *WORSE* |
| SPEED | 0.0955 | 0.1955 | 0.1484 | 0.0900 | 0.2418 | +0.1000 [+0.0717, +0.1277] *WORSE* | **+0.0529 [+0.0383, +0.0687] *WORSE*** | -0.0471 [-0.0717, -0.0259] *better* | +0.0584 [+0.0438, +0.0738] *WORSE* |
| STOP | 0.0188 | 0.0292 | 0.0288 | 0.0183 | 0.0338 | +0.0104 [+0.0050, +0.0167] *WORSE* | **+0.0100 [+0.0046, +0.0158] *WORSE*** | -0.0004 [-0.0013, +0.0000] | +0.0104 [+0.0050, +0.0163] *WORSE* |
| TURN | 0.0042 | 0.0071 | 0.0067 | 0.0050 | 0.0342 | +0.0029 [+0.0004, +0.0067] *WORSE* | **+0.0025 [+0.0004, +0.0054] *WORSE*** | -0.0004 [-0.0013, +0.0000] | +0.0017 [-0.0004, +0.0042] |
| LEAD | 0.0725 | 0.1042 | 0.0992 | 0.0700 | 0.1384 | +0.0317 [+0.0183, +0.0471] *WORSE* | **+0.0267 [+0.0142, +0.0409] *WORSE*** | -0.0050 [-0.0088, -0.0021] *better* | +0.0292 [+0.0158, +0.0446] *WORSE* |
| self (b) any | 0.2297 | 0.3697 | 0.3406 | 0.2255 | n/a | +0.1401 [+0.1113, +0.1700] *WORSE* | **+0.1109 [+0.0872, +0.1351] *WORSE*** | -0.0292 [-0.0449, -0.0150] *better* | +0.1150 [+0.0917, +0.1400] *WORSE* |

**M2b_s1** (active windows: SPEED 2399, STOP 81, TURN 158, LEAD 1027 of 2399)

| rule | B | H | HP | Bp | P10 | H - B | **HP - B** | HP - H | HP - Bp |
|---|---|---|---|---|---|---|---|---|---|
| any | 0.2097 | 0.2851 | 0.2409 | 0.1884 | 0.3110 | +0.0754 [+0.0571, +0.0945] *WORSE* | **+0.0313 [+0.0146, +0.0488] *WORSE*** | -0.0442 [-0.0680, -0.0246] *better* | +0.0525 [+0.0404, +0.0655] *WORSE* |
| SPEED | 0.1255 | 0.2238 | 0.1747 | 0.1063 | 0.2418 | +0.0984 [+0.0755, +0.1221] *WORSE* | **+0.0492 [+0.0292, +0.0683] *WORSE*** | -0.0492 [-0.0738, -0.0283] *better* | +0.0684 [+0.0525, +0.0842] *WORSE* |
| STOP | 0.0271 | 0.0296 | 0.0296 | 0.0267 | 0.0338 | +0.0025 [+0.0008, +0.0046] *WORSE* | **+0.0025 [+0.0008, +0.0046] *WORSE*** | +0.0000 [+0.0000, +0.0000] | +0.0029 [+0.0008, +0.0054] *WORSE* |
| TURN | 0.0067 | 0.0071 | 0.0071 | 0.0067 | 0.0342 | +0.0004 [-0.0017, +0.0025] | **+0.0004 [-0.0017, +0.0025]** | +0.0000 [+0.0000, +0.0000] | +0.0004 [-0.0021, +0.0033] |
| LEAD | 0.1084 | 0.1263 | 0.1205 | 0.1034 | 0.1384 | +0.0179 [+0.0100, +0.0267] *WORSE* | **+0.0121 [+0.0058, +0.0188] *WORSE*** | -0.0058 [-0.0108, -0.0021] *better* | +0.0171 [+0.0096, +0.0254] *WORSE* |
| self (b) any | 0.1430 | 0.2172 | 0.1717 | 0.1251 | n/a | +0.0742 [+0.0538, +0.0971] *WORSE* | **+0.0288 [+0.0087, +0.0487] *WORSE*** | -0.0454 [-0.0688, -0.0258] *better* | +0.0467 [+0.0312, +0.0641] *WORSE* |

**Transitions (windows), HP vs B** (violated by HP only / violated by B only / both):

| seed | any | SPEED | STOP | TURN | LEAD | self (b) any |
|---|---|---|---|---|---|---|
| s0 | 174 / 45 / 338 | 159 / 32 / 197 | 24 / 0 / 45 | 8 / 2 / 8 | 79 / 15 / 159 | 289 / 23 / 528 |
| s1 | 135 / 60 / 443 | 172 / 54 / 247 | 6 / 0 / 65 | 4 / 3 / 13 | 41 / 12 / 248 | 122 / 53 / 290 |

**Anatomy of the SPEED rule** (DIAGNOSTIC, descriptive): the eval's SPEED cap is min(posted limit, v_hi_ms), v_hi_ms being the GT speed-band edge. Windows by which cap is present, and where each arm's SPEED violations sit:

Windows (2399): active_total 2399; active_posted_limit_only (v_hi absent) 48; active_v_hi_only (no posted limit) 0; active_both_present 2351; active_both_present_and_v_hi_binding (v_hi < limit) 2084.

| seed | arm | SPEED violations | in windows where v_hi < posted limit | in posted-limit-only windows |
|---|---|---|---|---|
| s0 | B | 229 | 218 | 0 |
| s0 | H | 469 | 366 | 0 |
| s0 | HP | 356 | 356 | 0 |
| s0 | Bp | 216 | 216 | 0 |
| s1 | B | 301 | 258 | 0 |
| s1 | H | 537 | 431 | 0 |
| s1 | HP | 419 | 419 | 0 |
| s1 | Bp | 255 | 255 | 0 |

Reading (MEASURED): the projection recovers part of H's cost (HP - H is separated better on 'any' and SPEED at both seeds) but leaves HP above B: PRIMARY-C's upper bound is +0.0717 (s0) and +0.0488 (s1) against +0.02. The largest excess is SPEED, and SPEED is capped by v_hi where it binds, which is not the posted-limit input. LEAD is second and STOP / TURN are small; HP - B (STOP, TURN, LEAD) is s0: +0.0100, +0.0025, +0.0267 (all three separated), s1: +0.0025, +0.0004, +0.0121 (STOP and LEAD separated, TURN not). The projection does not touch STOP and TURN (HP - H about 0) and only slightly LEAD. Rule violations overlap, so the per-rule excesses do not add up to 'any'. Self-consistency (b) is also worse than B's at both seeds.

Projection effect on the plan: HP differs from H by more than 1e-3 m in 592 of 2399 windows (max 25.6 m, mean abs 0.178 m). ADE6 means: B 2.1753, H 1.9685, HP 2.1824, Bp 2.4206 (s0); B 2.0546, H 1.9848, HP 2.1818, Bp 2.2426 (s1).

## 8. POST-HOC DIAGNOSTIC 1 (not a PREREG row, no bar, changes no number above): the v_hi term of the SPEED rule

Run AFTER the read was fixed (`OUT/diag_vhi.py`, `OUT/diag_vhi.json`). Same windows, same estimator. Arms: `HPv_or` = H projected with cap min(posted limit, GT v_hi) (**T0 ORACLE**: reads a GT label at inference; a ceiling, never a result); `HPv_pr` = cap min(posted limit, T's predicted v_hi) (T1-class: T's own `cons_pred` head, vision-only); `Bv_pr` = B under the same cap. Cap applied from t = 0, as `project_speed` does. Paired vs B.

| seed | arm | ADE2 - B | ADE6 - B | CONS 'any' - B | CONS SPEED - B | self (b) - B |
|---|---|---|---|---|---|---|
| s0 | HPv_or | +0.2161 [+0.1432, +0.2941] *WORSE* | -0.0264 [-0.1686, +0.1284] | -0.0800 [-0.1075, -0.0533] | -0.0955 [-0.1216, -0.0725] | +0.0784 [+0.0538, +0.1038] |
| s0 | HPv_pr | +0.5208 [+0.3989, +0.6420] *WORSE* | +0.9099 [+0.6393, +1.2125] *WORSE* | -0.0004 [-0.0171, +0.0167] | +0.0067 [-0.0071, +0.0200] | -0.1392 [-0.1764, -0.1037] |
| s0 | Bv_pr | +0.5379 [+0.4171, +0.6580] *WORSE* | +1.1360 [+0.8599, +1.4312] *WORSE* | -0.0296 [-0.0442, -0.0171] | -0.0175 [-0.0279, -0.0092] | -0.1492 [-0.1868, -0.1133] |
| s1 | HPv_or | +0.2406 [+0.1653, +0.3207] *WORSE* | +0.0229 [-0.1189, +0.1789] | -0.1071 [-0.1351, -0.0816] | -0.1255 [-0.1560, -0.0975] | +0.0013 [-0.0175, +0.0196] |
| s1 | HPv_pr | +0.2102 [+0.1401, +0.2908] *WORSE* | +0.3696 [+0.2120, +0.5591] *WORSE* | +0.0079 [-0.0096, +0.0250] | +0.0246 [+0.0067, +0.0413] | -0.0734 [-0.1014, -0.0496] |
| s1 | Bv_pr | +0.2289 [+0.1599, +0.3081] *WORSE* | +0.4324 [+0.2720, +0.6205] *WORSE* | -0.0313 [-0.0467, -0.0175] | -0.0229 [-0.0375, -0.0104] | -0.0742 [-0.1033, -0.0496] |

Facts: GT plan vs the posted limit alone violates the SPEED rule in 6.34 % of windows (152 of 2399); GT v_hi_ms is below the posted limit in 2084 windows; T's predicted v_hi_ms vs the GT label: MAE 1.68 / 1.68 m/s (s0 / s1), 20 / 3 non-positive predictions floored at 0. Reading: capping at the oracle v_hi fixes SPEED to 0.000 and restores 'any' below B, but costs ADE2 (section 9 shows this is the cap acting at t < 2 s, where the rule does not read); capping at the predicted v_hi is neutral on 'any' (-0.0004 to +0.0079) and costs a lot of ADE (HYPOTHESIS: errors of the v_hi head become hard brakes from t = 0).

## 9. POST-HOC DIAGNOSTIC 2 (not a PREREG row, no bar, changes no number above): apply the cap from tau >= 2.0 s, the SPEED rule's own band

`OUT/diag_band.py`, `OUT/diag_band.json`. `project_speed` verbatim except the speed cap binds only on 0.1 s intervals j >= 20 (tau >= 2.0 s, the eval rule's `_BAND_KNOT`); Kamm everywhere. Identity control (band start 0 == `project_speed`): max abs 0.0e+00, 0.0e+00 m. Nothing fitted, 2.0 s is not tuned. Arms `HPb_lim` (posted limit only, deployable), `HPb_pr` (min(limit, T's predicted v_hi), T1-class), `HPb_or` (T0 ORACLE ceiling), and the same three on B (`Bpb_*`). Paired vs B.

| seed | arm | ADE2 - B | ADE6 - B | accel MAE - B | CONS 'any' - B | CONS SPEED - B |
|---|---|---|---|---|---|---|
| s0 | HPb_lim | -0.0967 [-0.1326, -0.0685] *better* | -0.1623 [-0.2328, -0.0945] *better* | -0.4203 [-0.5139, -0.3305] *better* | +0.0575 [+0.0404, +0.0759] | +0.0529 [+0.0383, +0.0687] |
| s0 | HPb_pr | -0.0967 [-0.1326, -0.0685] *better* | +0.0427 [-0.0675, +0.1507] | -0.4042 [-0.4952, -0.3206] *better* | +0.0229 [+0.0063, +0.0396] | +0.0067 [-0.0071, +0.0200] |
| s0 | HPb_or | -0.0967 [-0.1326, -0.0685] *better* | -0.2760 [-0.3494, -0.2038] *better* | -0.4365 [-0.5303, -0.3465] *better* | -0.0463 [-0.0746, -0.0188] | -0.0955 [-0.1216, -0.0725] |
| s0 | Bpb_lim | +0.0007 [-0.0040, +0.0061] | +0.0058 [-0.0027, +0.0156] | +0.0500 [+0.0294, +0.0767] *WORSE* | -0.0067 [-0.0113, -0.0029] | -0.0054 [-0.0100, -0.0017] |
| s0 | Bpb_pr | +0.0007 [-0.0040, +0.0061] | +0.1512 [+0.0909, +0.2176] *WORSE* | +0.0508 [+0.0305, +0.0769] *WORSE* | -0.0200 [-0.0305, -0.0108] | -0.0175 [-0.0279, -0.0092] |
| s0 | Bpb_or | +0.0007 [-0.0040, +0.0061] | -0.0910 [-0.1226, -0.0626] *better* | +0.0383 [+0.0184, +0.0648] *WORSE* | -0.0800 [-0.1050, -0.0584] | -0.0955 [-0.1216, -0.0725] |
| s1 | HPb_lim | -0.0711 [-0.1086, -0.0453] *better* | -0.0298 [-0.0608, +0.0044] | -0.1823 [-0.2545, -0.1239] *better* | +0.0346 [+0.0175, +0.0517] | +0.0492 [+0.0292, +0.0683] |
| s1 | HPb_pr | -0.0711 [-0.1086, -0.0453] *better* | +0.0237 [-0.0210, +0.0771] | -0.1810 [-0.2532, -0.1220] *better* | +0.0188 [+0.0017, +0.0354] | +0.0246 [+0.0067, +0.0413] |
| s1 | HPb_or | -0.0711 [-0.1086, -0.0453] *better* | -0.1753 [-0.2284, -0.1236] *better* | -0.2009 [-0.2727, -0.1431] *better* | -0.0784 [-0.1034, -0.0534] | -0.1255 [-0.1560, -0.0975] |
| s1 | Bpb_lim | +0.0017 [-0.0033, +0.0076] | +0.0061 [-0.0027, +0.0158] | +0.0548 [+0.0349, +0.0790] *WORSE* | -0.0171 [-0.0309, -0.0058] | -0.0192 [-0.0330, -0.0075] |
| s1 | Bpb_pr | +0.0017 [-0.0033, +0.0076] | +0.0467 [+0.0190, +0.0819] *WORSE* | +0.0561 [+0.0361, +0.0802] *WORSE* | -0.0217 [-0.0363, -0.0092] | -0.0225 [-0.0371, -0.0100] |
| s1 | Bpb_or | +0.0017 [-0.0033, +0.0076] | -0.1202 [-0.1547, -0.0873] *better* | +0.0420 [+0.0212, +0.0671] *WORSE* | -0.0913 [-0.1160, -0.0683] | -0.1255 [-0.1560, -0.0975] |

Reading (MEASURED, descriptive): banding removes the ADE2 cost (HPb ADE2 - B equals H's own gain). The posted limit alone still leaves 'any' - B at +0.0575, +0.0346; T's predicted v_hi brings it to +0.0229, +0.0188 (upper bounds +0.0396, +0.0354, still above +0.02) at an ADE6 cost +0.0427, +0.0237; the oracle clears both. **No deployable-class variant tested here clears PRIMARY-C.** None of these arms is a result; each would need its own PREREG, and they are listed only so that the committed constraint-aware PREREG is written against the measured mix.

## 10. The outcome table, applied verbatim

| outcome | condition | met? |
|---|---|---|
| PASS | PRIMARY-C and PRIMARY-K at both seeds | **False** (C: False, K: False) |
| **CONSISTENCY NOT RESTORED** | PRIMARY-C fails | **True** |
| GAIN LOST | PRIMARY-C passes, PRIMARY-K fails | False (requires PRIMARY-C to pass) |

No weight, cap or bar changed after any HPROJ number was seen. The weights are HBLEND's banked file (sha256 in each score json); the cap is the eval's; the bars are the PREREG's. The control ran first in the same process and the script refuses to score if it fails. The two post-hoc diagnostics were written and run after `READ.json` existed and are labelled as such.

## 11. What this does and does not show

* **Evidence class.** All rows MEASURED (this run). B's plans are the banked KIND-run outputs (INHERITED from box 2's GPU run; re-derived metrics reproduce the banked arrays, and the reconstructed `projected_r1_kamm` block reproduces the banked one with 0 mismatches at s0 and 1 at s1 -- a zero-acceleration counter flipping one step, see section 2).
* **Tier.** T1-class open loop on 2,399 windows / 141 episodes. Not closed loop: nothing here says HP drives better or worse, only that its open-loop plan is farther from the logged future than B's (ADE) and breaks more GT-derived constraints than B's.
* **Variance.** The bootstrap resamples episodes with the models fixed. Training-run variance is covered only by the two seeds, which agree in sign on every PREREG row (PRIMARY-C fails at both, PRIMARY-K fails at both). Two seeds are a floor check, not an interval. For scale, F = |ADE6(s0) - ADE6(s1)| = 0.1207 m (HBLEND): the s1 ADE6 cost (+0.127) is of that order, the s0 one (+0.007) is not distinguishable from zero.
* **The PREREG's explanation was incomplete.** It expected the remaining violations to be stops, leads and turns. They are mostly SPEED, against a cap (v_hi) that the projection's input does not carry, then LEAD. The outcome row is unchanged by this; the committed next step (a constraint-aware fit under its own PREREG) stands, and the PREREG for it should name v_hi and the 2.0 s band explicitly.
* **Not done / not claimed.** No new fit, no new weights, no trained arm, no closed-loop run. MIXO and FLAT were not re-scored (that follows only from a PASS).

## 12. Deliverable manifest

All under `OUT` (dev-box scratchpad; **nothing staged to git, nothing pushed, nothing written on box 2, nothing deleted**). The Master Mind lands them. Footprint on C: under 20 MB.

| artifact | path (under OUT) | note |
|---|---|---|
| this report | `HPROJ_RESULT.md` | generated by build_result.py from the JSONs below |
| executor (control / score / read / dk) | `v8f_hproj.py` | imports HBLEND's loaders read-only (`../hblend/v8f_hblend.py`), the eval's own `evaluate`, `taniteval.ci`, `v8f_beta_eval`, the DK reader |
| report generator | `build_result.py` |  |
| self-tests | `test_v8f_hproj.py` | 17 tests: reconstruction identities, wrong-basis deliberate regression, w60, blend, projection, cap, banded identity, outcome-table logic (6 scenarios) |
| reconstruction control (+ diagnostics, regressions, X1) | `control_M2b_s0.json, control_M2b_s1.json` | binding C0-C2, E1-E5, R1-R3, X1 |
| per-seed score (every row, CI, blocks, consistency, transitions) | `score_M2b_s0.json, score_M2b_s1.json` |  |
| per-window arrays | `score_M2b_s{0,1}_perwindow.npz` | HP, HP60, H60, tau60, cap, w60, per-window ADE/FDE/families and violation flags for every arm |
| outcome | `READ.json` | the table, verbatim |
| distance keeping | `dk_read.json, dk_perwindow.npz` | via v8f_dk_read.read_dk |
| post-hoc diagnostic 1 (v_hi) | `diag_vhi.py, diag_vhi.json` | not a PREREG row |
| post-hoc diagnostic 2 (2.0 s band) | `diag_band.py, diag_band.json` | not a PREREG row |
| logs | `score_M2b_s0.log, score_M2b_s1.log, read.log, dk.log` |  |
| inputs (read-only, from HBLEND) | `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/hblend/inputs/banked, /weights_M2b_s{0,1}.json, /inputs/b2_p10_eval_pred6.npy, /inputs/full_eval_floors_perwindow.npz, /deps` | not copied; nothing from box 2 |

## 13. Reproduce

```
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY v8f_hproj.py control --seed M2b_s0   # ~5 s, rc 2 on failure
$PY v8f_hproj.py score --seed M2b_s0     # ~65 s, control first
$PY v8f_hproj.py score --seed M2b_s1
$PY v8f_hproj.py read
$PY v8f_hproj.py dk                      # ~26 s
$PY diag_vhi.py && $PY diag_band.py      # post-hoc, after READ.json
$PY build_result.py
$PY -m pytest -q test_v8f_hproj.py --basetemp C:/Users/Admin/pt_hproj -p no:cacheprovider
```
Paths are hard-coded to the scratchpad (`SCR` in `v8f_hproj.py`, `HB` = HBLEND); relocate before landing. Pinned: tip a99ca5d6 export at `viz/tree`, `taniteval` copy in `hblend/deps`.

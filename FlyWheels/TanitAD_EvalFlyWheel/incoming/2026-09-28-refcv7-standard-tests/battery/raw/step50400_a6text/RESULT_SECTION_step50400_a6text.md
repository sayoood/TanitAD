## RESULT — refcv7 step50400_a6text (2026-10-05T02:09:30+0200)

* **G0 as registered: FAIL** — eval_agent_det_ap0p5_heavy_truck_0_20 [DETECTION] in-run 0.0 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_0_20 [DETECTION] in-run 0.07143 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_all [DETECTION] in-run 0.03571 vs seeds mean 0.01515 (sd 0)
* **G0-A2 (seeds 0..7): PASS**
* **G0-A6 (THE GATE, SPEC A6; 24 inference seeds): FAIL**; mutation terms moved: M1 22 (detected), M2 62 (detected), M4 2 (detected); wrapper clause PASS; reasons: ['eval_agent_det_ap0p5_heavy_truck_0_20 [DETECTION] in-run 0.0 vs seeds mean 0.03333 (sd 0)', 'eval_agent_det_ap1_heavy_truck_0_20 [DETECTION] in-run 0.07143 vs seeds mean 0.03333 (sd 0)', 'eval_agent_det_ap1_heavy_truck_all [DETECTION] in-run 0.03571 vs seeds mean 0.01515 (sd 0)', 'eval_agent_det_ap2_stroller_0_20 [DETECTION] in-run 0.33333 vs seeds mean 0.25 (sd 0)', 'eval_agent_det_ap2_stroller_all [DETECTION] in-run 0.16667 vs seeds mean 0.125 (sd 0)']
* **G0-A5 (24 seeds; reported beside the A6 gate): PASS**
* **G0-A6 (REGISTERED; measured numerics floor): FAIL** — eval_agent_det_ap0p5_heavy_truck_0_20 [DETECTION] in-run 0.0 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_0_20 [DETECTION] in-run 0.07143 vs seeds mean 0.03333 (sd 0); eval_agent_det_ap1_heavy_truck_all [DETECTION] in-run 0.03571 vs seeds mean 0.01515 (sd 0); floor-rescued terms: []; threshold-target terms: eval_tacv6_goal_conf_bce in-run 0.14591 interval [0.14224843595836026, 0.18356920461913012] OK
* ⭐ **G0 RECORD (gate source: REGISTERED A6 TEXT, corrected judge (Master Mind ruling 2026-10-04)):** **G0-A6 as registered (text): PASS, computed by the corrected judge after the coded verdict was read; the criterion is unchanged** · **G0-A6 as coded: FAIL (judge defect: the A2 low-support tuple omitted A6)** (`g0_A6_text.json`, sha256 f3c7de52db67…; re-verified by the judge in use)
* **Tier:** T1 self-action OPEN LOOP (status UNRULED for an action-free model); never closed loop, never driving performance. **Estimator:** FULL-SET point estimate; paired episode-cluster bootstrap, n_boot 2000, seed 0.
* **Inference-seed floor:** 0.0006 m ADE (training-seed floor NOT measured).

| bar | verdict | seed 0 | seed 1 |
|---|---|---|---|
| BAR-R7-1 | FAILED | -0.0102 [-0.0260, +0.0067] (n 4754/139) | -0.0096 [-0.0253, +0.0069] (n 4754/139) |
| BAR-R7-2 | PENDING (the refcv6@38k S2 rolls run LAST in the milestone chain; SPEC A4) | ABSENT | ABSENT |
| BAR-R7-3 | PASS (single training seed; training-seed floor unmeasured) | -0.9804 [-1.2940, -0.6815] SEP (n 3668/139) | -0.9796 [-1.2855, -0.6842] SEP (n 3668/139) |
| BAR-M7-1 | FAILED | (per band in perc_score.json) | — |
| BAR-M7-2 | FAILED | (per band in perc_score.json) | — |
| BAR-M7-3 | FAILED | (per band in perc_score.json) | — |
| BAR-M7-4 | FAILED | (per band in perc_score.json) | — |
| BAR-B7-1 | PASS (single training seed; no replicate floor measured for boxes) | +0.1051 [+0.0796, +0.1289] SEP (n 4532/139) | — |
| BAR-B7-2 | FAILED | -0.2204 [-0.2448, -0.1885] SEP (n 4532/139) | — |

| BAR-R7-N1 (NavSim) | not this battery | — | — |

**Four families, paired `os − ha0_ext` (seed 0):** 
* ADE: ade_m -0.0102 [-0.0260, +0.0067] (n 4754/139); fde_m -0.0360 [-0.0759, +0.0054] (n 4754/139)
* longitudinal: LON_speed_mae_mps +0.0040 [-0.0102, +0.0198] (n 4754/139); LON_along_mae_m -0.0086 [-0.0225, +0.0057] (n 4754/139); LON_accel_mae_mps2 +0.0534 [+0.0329, +0.0753] SEP (n 4754/139)
* lateral: LAT_cross_mae_m -0.0049 [-0.0158, +0.0059] (n 4754/139); LAT_heading_mae_deg -0.0961 [-0.2042, +0.0146] (n 4509/139); LAT_yaw_rate_mae_radps_valid -0.0047 [-0.0075, -0.0021] SEP (n 4473/139)
* tactical: TAC_traj_lat_correct +0.0206 [+0.0103, +0.0306] SEP (n 4754/139); TAC_traj_lon_correct -0.0133 [-0.0288, +0.0006] (n 4754/139)

**Family levels, seed 0 (T1; FULL-SET mean; intervals in `analysis_s0.json`):**

| arm | speed MAE m/s | target-speed acc | along MAE m | min headway m / time gap s / min TTC s (n, n_closing) | heading ° | curvature 1/m | yaw-rate °/s | cross-track m | tac lat / lon (traj) |
|---|---|---|---|---|---|---|---|---|---|
| os | 0.259 | 0.853 (±0.5 m/s) | 0.226 | 27.95 / 4.13 / 25.1 (n 1303, closing 482) | 1.35 | 0.0037 | 1.08 | 0.103 | 0.962 / 0.831 |
| os_filteroff | 0.259 | 0.853 (±0.5 m/s) | 0.226 | 27.95 / 4.13 / 25.1 (n 1303, closing 482) | 1.35 | 0.0037 | 1.08 | 0.103 | 0.962 / 0.831 |
| ha0_ext | 0.255 | 0.866 (±0.5 m/s) | 0.235 | 28.02 / 4.20 / 25.5 (n 1285, closing 446) | 1.45 | 0.0038 | 1.36 | 0.108 | 0.941 / 0.844 |
| ha | 0.255 | 0.866 (±0.5 m/s) | 0.235 | 28.05 / 4.21 / 25.6 (n 1291, closing 447) | 1.57 | 0.0041 | 1.47 | 0.124 | 0.937 / 0.844 |
| ha0 | 0.489 | 0.705 (±0.5 m/s) | 0.472 | 27.45 / 4.12 / 24.1 (n 1315, closing 499) | 2.82 | 0.0069 | 2.41 | 0.317 | 0.864 / 0.757 |
* TACTICAL lat decision, v6_behaviour_decoder: acc 0.7401, kappa 0.45663136165485746, n in band 1081 (label clock: the refcv7 loader's TRUE clip clock, `enable_clip_clock` + G3)
* TACTICAL lat decision, z_tac_v7_heads: acc 0.7299, kappa 0.3461702577189327, n in band 1081 (label clock: the refcv7 loader's TRUE clip clock, `enable_clip_clock` + G3)
* TACTICAL lon decision, v6_behaviour_decoder: acc 0.481, kappa 0.3184823321971728, n in band 1081 (label clock: the refcv7 loader's TRUE clip clock, `enable_clip_clock` + G3)
* TACTICAL lon decision, z_tac_v7_heads: acc 0.519, kappa 0.3631659980581842, n in band 1081 (label clock: the refcv7 loader's TRUE clip clock, `enable_clip_clock` + G3)
* TACTICAL goal selection (22 tokens): 2 scoreable classes (n_pos ≥ 200); per-class AUROC / AP / P / R in `tactical_v6_s0.json`
* strategic: NOT APPLICABLE (n 0): the strategic layer is OFF (--no-strategic); its route loss is gated off in training, so the route head is untrained
* ⚠️ baselines missing: ['b_refcv6_38k_s0', 'b_refcv6_38k_s1']

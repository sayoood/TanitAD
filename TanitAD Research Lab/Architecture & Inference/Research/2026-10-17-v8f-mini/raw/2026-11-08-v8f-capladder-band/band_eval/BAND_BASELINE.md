# BAND_BASELINE: the reactive planner and the kinematic floor scored in the operative (0-2 s) and tactical (2-6 s) bands

Item 1 of the change list in `AUDIT_MODEL.md` section 6. Descriptive baseline: no bars, no verdict words. Zero training, CPU only, from banked per-window dumps. 2026-11-08.

**Evidence class: MEASURED (ours).** **Tier: T1-kin** (self-action open loop; m = 10, causal bucket, dead-reckoned nav, logged scene; the lead agent follows its logged future, so a plan that closes on the lead is scored against where the lead was). **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap` (arm minus P10, same windows) and `episode_cluster_bootstrap` (single arm), B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed. It answers "would another draw of EPISODES say this?". It does not answer "would another training run say this?" (`H-ESTIM-SEED-1`): the two seeds are shown side by side and no seed floor is applied. Instrument layer: mirror tip `26fb0bc1` (`four_families._seq_geometry`, `ci`, `lead_metrics`, `refc_tactical`), read with `git archive`.

## 1. Which arms, and where the plan arrays are

DAG-OACC (the adopted reactive base, report §7.23) **is banked per window**, so no fallback to M2b was needed.

| arm | source (per-episode `ep*.npz`, keys `g, cl, fl, v0, eid, clip_index, ws`) | n |
|---|---|---|
| DAG-OACC s0 (`cl`), P10 floor (`fl`), GT (`g`) | `E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/dump_DAGOACC_s0_m10_causal_heads/` (cat-sha256 `9c874630e62e121f...`) | 141 ep / 2,399 win |
| DAG-OACC s1 (`cl`), same P10 and GT | `.../dump_DAGOACC_s1_m10_causal_heads/` (cat-sha256 `37ecd69c7735b8e9...`) | 141 ep / 2,399 win |
| lead agents (distance keeping) | `E:/Projects/TanitAD-artifacts/dagoacc_t1/inputs/b1_eval_lead_block_k30.npz`, sha256 `7a879baa...3d9` (matches the registered pin); joined by (clip, frame = 2 x ws), speed proof max 0 within 1e-3 m/s | 743 lead windows |

Locations probed for the plan arrays: (1) `v8f-dagoacc-t1/` (the dumps above), (2) `dagoacc_t1/` (reader, manifests and inputs only, no dumps), (3) `dagland/smooth_t1/raw/rollouts/out/SMOOTH_DAGOACC_s*` (a different, smoothing rollout; not used), (4) `v8f-dagoacc-t1/read/DAGOACC_READ.json` (aggregates only). P10 is the dump's `fl` track, which reads no model and is identical across the two seeds' dumps.

## 2. Band definitions and how they are sliced

The plan has 30 knots at 0.2 s; knot i (0-indexed) is time (i+1) x 0.2 s.

| band | knots | time |
|---|---|---|
| operative | 0 to 9 | 0.2 to 2.0 s |
| tactical | 10 to 29 | 2.2 to 6.0 s |
| full (check) | 0 to 29 | 0.2 to 6.0 s |

`four_families._seq_geometry` is run on the **whole** track (so the tactical band does not restart its origin at the band edge), and its per-step arrays are then restricted. A step (speed, heading, along, cross) belongs to a band when its knot is inside it. A pair (acceleration, yaw rate, curvature: between steps j and j+1) belongs to a band when both steps are inside it, so the one pair that straddles 2.0 s is in neither band (the full band holds all 29). Per-window means use the same masks as `t1k_read.comps`. ADE is the mean over the band's knots; FDE is the error at the band's last knot (2.0 s for operative, 6.0 s for tactical and full). Distance keeping slices the knots of both the plan and the lead track (headway and time gap are minima over the band's steps; time gap divides by ego speed at NOW).

**Checks before any arm number** (`ANALYTIC_CHECKS.json`, 44 of 44 hold; script `analytic_checks.py`):
* constant speed (GT 10 m/s, plan 10.5 m/s): speed MAE 0.5, acceleration 0, ADE 0.55 / 2.05 / 1.55 m, FDE 1.0 / 3.0 / 3.0 m, lateral rows 0, in the three bands;
* constant acceleration (1 m/s2): acceleration MAE 1.0 in all three bands;
* constant curvature (circle R 50 m against R 40 m at 10 m/s, exact chord formulas): yaw-rate MAE, curvature MAE, heading MAE (mean of (i + 1/2) x dtheta) and speed MAE match the closed form in each band;
* a slice over the full band equals `t1k_read.comps` with max |diff| 0 and an identical NaN pattern on synthetic windows and on both real tracks (DAG-OACC s0, P10);
* additive identity 10 x ADE(op) + 20 x ADE(tac) = 30 x ADE(full) on the real dump;
* **deliberate regression:** moving the band edge by one knot (10 to 11) changes every metric mean in both bands except the tactical FDE, which is the value at knot 30 for both edges by construction; the analytic ADE check reads 0.600 against the expected 0.550 and goes red.

**Reproduction of the banked full-horizon numbers.** The full-band rows below equal the §7.23 DAG-OACC minus P10 table to the printed digits at both seeds: ADE6 -1.2012 / -1.1231, speed MAE -0.7761 / -0.7030, along-track -0.6809 / -0.5876, accel -0.8172 / -0.6014, heading -2.5920 / -2.0539, yaw-rate +3.4726 / +8.6844, curvature +0.0854 / +0.1352, cross-track -0.9620 / -0.9742, DK-HW -0.3318 / -0.4466, DK-TG -0.0568 / -0.0724, TAC-LAT +0.1905 / +0.1972, TAC-LON +0.1396 / +0.1371.

## 3. Results

Format: point [95 % episode-cluster CI]. "delta" is arm minus P10 on the same windows (paired); "CI excludes 0" is stated as a fact about the interval and says nothing about a bar. Lower is lower error for every row except the two TAC accuracies. Each seed is a separate training run of DAG-OACC; P10 is one deterministic track. Rows are never pooled across families.

### 3.1 Operative band, 0.2 to 2.0 s

| family | row | DAG-OACC s0 | DAG-OACC s1 | P10 | delta s0 | delta s1 |
|---|---|---|---|---|---|---|
| PRIMARY | ADE (m) | 0.219 [0.198, 0.245] | 0.225 [0.204, 0.251] | 0.222 [0.201, 0.249] | -0.003 [-0.010, +0.003] | +0.003 [-0.005, +0.011] |
| PRIMARY | FDE at 2 s (m) | 0.563 [0.513, 0.620] | 0.579 [0.528, 0.636] | 0.609 [0.556, 0.669] | -0.046 [-0.065, -0.026] | -0.030 [-0.054, -0.004] |
| LONGITUDINAL | speed MAE (m/s) | 0.262 [0.243, 0.280] | 0.271 [0.250, 0.292] | 0.266 [0.248, 0.286] | -0.005 [-0.013, +0.003] | +0.004 [-0.005, +0.015] |
| LONGITUDINAL | acceleration MAE (m/s2) | 0.522 [0.484, 0.560] | 0.550 [0.510, 0.591] | 0.591 [0.546, 0.639] | -0.069 [-0.092, -0.049] | -0.041 [-0.066, -0.019] |
| LONGITUDINAL | along-track MAE (m) | 0.182 [0.163, 0.208] | 0.187 [0.166, 0.212] | 0.173 [0.153, 0.199] | +0.009 [+0.004, +0.015] | +0.014 [+0.006, +0.022] |
| LONGITUDINAL | DK headway abs error (m), n = 703 / 701 windows, 72 ep | 0.554 [0.379, 0.773] | 0.501 [0.343, 0.687] | 0.659 [0.429, 0.960] | -0.050 [-0.268, +0.119] | -0.104 [-0.302, +0.028] |
| LONGITUDINAL | DK time-gap abs error (s), n = 646 / 647 | 0.119 [0.062, 0.203] | 0.126 [0.058, 0.231] | 0.138 [0.066, 0.242] | -0.011 [-0.041, +0.010] | -0.011 [-0.040, +0.009] |
| LONGITUDINAL | DK-VIOL, share of 605 at-risk lead windows | 0.0116 (7) | 0.0099 (6) | 0.0083 (5) | not paired | not paired |
| LATERAL | heading MAE (deg), n = 2,313 / 2,312 | 2.250 [1.192, 4.052] | 2.270 [1.216, 4.065] | 2.530 [1.449, 4.323] | -0.251 [-0.305, -0.199] | -0.225 [-0.291, -0.162] |
| LATERAL | curvature MAE (1/m), n = 2,303 | 0.072 [0.048, 0.102] | 0.069 [0.046, 0.096] | 0.071 [0.049, 0.099] | +0.001 [-0.003, +0.004] | +0.001 [-0.004, +0.006] |
| LATERAL | yaw-rate MAE (deg/s), n = 2,303 | 8.060 [6.471, 9.874] | 7.973 [6.434, 9.699] | 8.404 [6.844, 10.192] | -0.329 [-0.476, -0.189] | -0.309 [-0.486, -0.134] |
| LATERAL | cross-track MAE (m) | 0.077 [0.068, 0.088] | 0.081 [0.071, 0.092] | 0.092 [0.081, 0.104] | -0.015 [-0.019, -0.010] | -0.011 [-0.016, -0.006] |

### 3.2 Tactical band, 2.2 to 6.0 s

| family | row | DAG-OACC s0 | DAG-OACC s1 | P10 | delta s0 | delta s1 |
|---|---|---|---|---|---|---|
| PRIMARY | ADE (m) | 3.254 [2.980, 3.551] | 3.368 [3.090, 3.654] | 5.054 [4.675, 5.465] | -1.800 [-2.088, -1.527] | -1.686 [-1.996, -1.376] |
| PRIMARY | FDE at 6 s (m) | 6.957 [6.338, 7.623] | 7.291 [6.668, 7.947] | 12.058 [11.112, 13.078] | -5.100 [-5.869, -4.333] | -4.767 [-5.585, -3.966] |
| LONGITUDINAL | speed MAE (m/s) | 1.537 [1.382, 1.706] | 1.642 [1.472, 1.827] | 2.699 [2.367, 3.072] | -1.162 [-1.468, -0.898] | -1.056 [-1.365, -0.788] |
| LONGITUDINAL | acceleration MAE (m/s2) | 1.604 [1.323, 1.927] | 1.932 [1.557, 2.355] | 2.770 [2.142, 3.523] | -1.165 [-1.700, -0.694] | -0.838 [-1.289, -0.441] |
| LONGITUDINAL | along-track MAE (m) | 2.575 [2.358, 2.805] | 2.712 [2.471, 2.956] | 3.601 [3.312, 3.888] | -1.026 [-1.271, -0.789] | -0.888 [-1.162, -0.626] |
| LONGITUDINAL | DK headway abs error (m), n = 516 / 508 windows, 58 / 57 ep | 2.363 [1.870, 2.919] | 2.413 [1.895, 3.020] | 2.359 [1.862, 2.968] | -0.184 [-0.408, +0.049] | -0.158 [-0.411, +0.085] |
| LONGITUDINAL | DK time-gap abs error (s), n = 489 / 485 | 0.396 [0.271, 0.541] | 0.416 [0.283, 0.569] | 0.439 [0.299, 0.611] | -0.077 [-0.129, -0.032] | -0.042 [-0.109, +0.030] |
| LONGITUDINAL | DK-VIOL, share of 520 at-risk lead windows | 0.0442 (23) | 0.0423 (22) | 0.0423 (22) | not paired | not paired |
| LATERAL | heading MAE (deg), n = 2,348 | 8.897 [7.126, 10.935] | 9.571 [7.655, 11.661] | 12.345 [10.436, 14.509] | -3.449 [-4.353, -2.546] | -2.774 [-3.748, -1.761] |
| LATERAL | curvature MAE (1/m), n = 2,340 | 0.185 [0.132, 0.245] | 0.260 [0.180, 0.353] | 0.056 [0.041, 0.073] | +0.129 [+0.087, +0.176] | +0.204 [+0.136, +0.284] |
| LATERAL | yaw-rate MAE (deg/s), n = 2,340 | 30.355 [22.791, 38.909] | 37.185 [27.587, 48.236] | 21.509 [15.828, 28.128] | +8.847 [+4.468, +13.447] | +15.676 [+9.945, +22.132] |
| LATERAL | cross-track MAE (m) | 1.398 [1.214, 1.597] | 1.378 [1.202, 1.568] | 2.833 [2.547, 3.146] | -1.436 [-1.654, -1.241] | -1.456 [-1.667, -1.256] |

### 3.3 Full 0-6 s plan (the check band; equals the §7.23 table)

| family | row | DAG-OACC s0 | DAG-OACC s1 | P10 | delta s0 | delta s1 |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 (m) | 2.242 [2.057, 2.447] | 2.320 [2.131, 2.516] | 3.443 [3.189, 3.722] | -1.201 [-1.395, -1.019] | -1.123 [-1.330, -0.916] |
| LONGITUDINAL | speed MAE (m/s) | 1.112 | 1.185 | 1.888 | -0.776 [-0.981, -0.600] | -0.703 [-0.909, -0.523] |
| LONGITUDINAL | acceleration MAE (m/s2) | 1.273 | 1.488 | 2.090 | -0.817 [-1.177, -0.500] | -0.601 [-0.906, -0.335] |
| LATERAL | yaw-rate MAE (deg/s) | 23.409 | 28.595 | 19.944 | +3.473 [+0.660, +6.331] | +8.684 [+5.169, +12.574] |
| LATERAL | curvature MAE (1/m) | 0.160 | 0.209 | 0.075 | +0.085 [+0.058, +0.117] | +0.135 [+0.089, +0.192] |

All 11 full-band rows with CIs are in `BAND_BASELINE.json` (`seeds.s*.full_0_6s`).

### 3.4 TACTICAL family: the existing decision reads, and their label band

TAC-LAT / TAC-LON are the existing trajectory-derived reads (`four_families.maneuver_kinematics` into `refc_tactical.factor_from_kinematics`): the executed plan's manoeuvre class against the GT plan's class, per window, accuracy, T1-kin, n = 2,399 windows / 141 episodes.

| row | DAG-OACC s0 | DAG-OACC s1 | P10 | delta s0 | delta s1 |
|---|---|---|---|---|---|
| TAC-LAT accuracy | 0.784 [0.740, 0.823] | 0.790 [0.750, 0.827] | 0.593 [0.555, 0.634] | +0.191 [+0.151, +0.232] | +0.197 [+0.160, +0.236] |
| TAC-LON accuracy | 0.511 [0.470, 0.550] | 0.509 [0.466, 0.551] | 0.371 [0.326, 0.417] | +0.140 [+0.077, +0.200] | +0.137 [+0.075, +0.200] |

**Label bands, stated side by side (the mismatch is reported, not hidden):**
* these two reads label the **whole 0-6 s plan** (path tangent at the last moving knot; chord speed at knot 1 against knot 30). They are not sliced into 0-2 s and 2-6 s: a manoeuvre label of a 2 s slice is a different instrument that does not exist yet;
* the T head's own labels (`build_v9_labels.py`) span **[NOW+2 s, NOW+8 s]**, so against R-TAC (2-6 s) they reach 2 s past the band, and the plan-derived reads above overlap the operative band by 2 s (AUDIT_MODEL F8, INHERITED here);
* the T head's outputs (goal, lat, lon, constraint logits) are **not in these dumps**, so T's decision quality against the v9 labels is not scored here. Context only, INHERITED from `DAGOACC_READ.json` (box 2 standard eval, T0, DAG-OACC s0, trajectory-derived, not recomputed): lateral accuracy 0.8383 (kappa 0.6301), longitudinal accuracy 0.5961 (kappa 0.3786), n = 2,399.

### 3.5 STRATEGIC: PAUSED

Reported as **PAUSED**, PI item 25 (`PI_DECISION_QUEUE.md`, 2026-09-19): the strategic layer is switched off temporarily. It is not a pass and it is not omitted. The 6-30 s band has no baseline row because neither arm predicts beyond 6 s.

## 4. What the table says arithmetically (MEASURED; no bars)

1. **Where the full-horizon gap sits.** The 30-knot ADE gap of DAG-OACC against P10 is the sum 10 x (operative gap) + 20 x (tactical gap) over 30. At s0 that is 10 x (-0.003) + 20 x (-1.800) = -36.0 (= 30 x -1.201); at s1, 10 x (+0.003) + 20 x (-1.686) = -33.7. The operative band carries under 0.1 % of it. In the operative band the two arms' ADE intervals contain 0 at both seeds.
2. **Operative band, 0-2 s.** Acceleration MAE and FDE at 2 s have intervals below 0 at both seeds; heading, yaw-rate and cross-track MAE too (small absolute sizes: 0.011 to 0.329 units). Along-track MAE is higher for the arm at both seeds (+0.009, +0.014 m). Curvature, speed and DK rows contain 0.
3. **Tactical band, 2-6 s.** Position, speed, acceleration, along-track, heading and cross-track rows are lower for the arm at both seeds. **Curvature and yaw-rate MAE are higher for the arm** at both seeds, and the excess is larger in this band than in the full-horizon rows (curvature +0.129 / +0.204 against +0.085 / +0.135; yaw-rate +8.8 / +15.7 deg/s against +3.5 / +8.7). In the operative band the same two rows do not exclude 0 (curvature) or are lower (yaw-rate). The seed spread in this band (curvature 0.185 vs 0.260; yaw-rate 30.4 vs 37.2) is larger than the operative-band spread.
4. **Distance keeping** has thin denominators: 72 episodes with a lead in the operative band and 57 to 58 in the tactical band (lead windows 743; headway finite 508 to 703 depending on arm and band). Its intervals contain 0 in both bands at both seeds except the tactical time gap at s0.

## 5. Caveats

* T1-kin rolls the arm on its own executed motion with a logged scene and a lead that does not react; the tactical-band values are plans scored against the human's path, which is one of several acceptable futures (the usual open-loop reading).
* The seam pair (acceleration, yaw rate and curvature between the 2.0 s and 2.2 s steps) is in neither band; moving it into one band changes those three rows by one pair of 9 or 19.
* The bootstrap resamples episodes with the models held fixed: a separated interval here is not a statement about training variance. DAG-OACC's own seed spread is visible in the s0 and s1 columns.
* P10 has no training seed; its row is the same in both seed blocks.
* DK-VIOL is the existing `lead_viol` sliced to the band's knots; its progress-ratio guard (`plan_arc`, `human_arc`) is computed on the band's arc, which is a shorter path in the operative band. The share is shown with counts and is not paired.
* Analytic checks cover geometry and slicing; the lead-join and bootstrap code are the existing instruments, exercised here through the full-band reproduction of the banked numbers.

## 6. Manifest (all single copies on E:, nothing staged, committed or pushed)

| file | what |
|---|---|
| `.../horizon_review/band_eval/BAND_BASELINE.md` | this report |
| `.../band_eval/BAND_BASELINE.json` | every row, both seeds, three bands, with arm, P10 and paired CIs, DK counts, TAC reads, inputs, tier and estimator (B = 2,000, seed 0) |
| `.../band_eval/ANALYTIC_CHECKS.json` | the 44 checks, including the edge-shift regression deltas |
| `.../band_eval/band_lib.py`, `analytic_checks.py`, `run_band_eval.py`, `make_md.py` | code. `band_lib` needs a code tree at `band_eval/tree`, re-creatable with `git -C C:/Users/Admin/tanitad-push archive 26fb0bc1b86bd290c9913ecee0b3676ba7684f05 taniteval/taniteval stack/tanitad/refs stack/tanitad/models stack/tanitad/data stack/tanitad/configs` and extracted into `tree/` (plus the rest of `stack/tanitad` it imports) |

Run: `run_band_eval.py` 32 s wall at B = 2,000, run detached via WMI with a done-marker (`DONE_OK`).

Temporary files deleted at the end: see the hand-back list.

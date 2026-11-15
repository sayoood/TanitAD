# VERDICT_MOTION_TARGET_FINAL: cards ADD18 (MOTION) and ADD19 (TARGET), the finished reads collated (2026-11-15)

Written by a teammate of the v8F proof programme, CPU only, no card edited, no banked file edited. Evidence class: MEASURED (ours; every number carries its file in `RESULT_MOTION_TARGET_FINAL.json`). Tier: **T0** (a world-model diagnostic on logged frames, never driving performance) for the step, lead, change and position reads; **T1-kin** (kinematic rollout on the model's own actions, logged frames) for the family read. STRATEGIC: **PAUSED** (PI item 25), never a pass. Intervals: episode-cluster bootstrap B 2,000 (lead and step reads: 139 episodes; change form: 48 episodes with a lead pair; position read: clips as clusters, 1,200 clips; family read: two-level bootstrap seeds then episodes, 141 episodes, 2,399 windows). A separated interval from a one-seed arm is necessary and not sufficient (H-ESTIM-SEED-1); a name that needs both seeds is given only where both seeds exist.

## 0. What is on disk (and what is not)

| arm | seeds read | role |
|---|---|---|
| PR (rollout loss, K 10, W 1.0, on the cell predictor; note ADD18_N1) | 0 and 1 | comparison arm P of the same seed for I and V; arm minus G for R (card ADD18 section 4) |
| RES (residual form, flat predictor) | 0 | comparison arm G_s0 |
| RR (RES + ROLL, flat predictor) | 0 | comparison arm G_s0 |
| PE (EGOV, speed as third action channel, cell predictor) | 0 and 1 | comparison arm P of the same seed |
| PF (FUTTOK, K 10, W 1.0, cell predictor) | 0 | comparison arms P_s0 and PR_s0 |

**Not trained, so not read:** ROLL_s0, ROLL_s1 (ROLL alone on the flat predictor), RR_s1, RES_s1, PF_s1, PEF_s0, PEF_s1 (no checkpoint under `C:/Users/Admin/v8proof_c/ckpt/`). PR is the rollout loss on the CELL predictor (note ADD18_N1); it is not the ROLL arm of card ADD18 section 3, which is the same loss on the flat predictor.

## 1. Card ADD18 (MOTION)

### 1.1 Controls first

* Build (card section 5, CPU, files in `F:/TanitAD-scratch/v8proof/spatial/motion/`): m1 defaults bit-identical to the tree of ADD16 for the argv of G, P and the mean default over 200 steps, max abs difference 0.0 (`checks/m1_identity.json`): **PASS**. m2 RES equals the copy at init, 0.0, flat and cell, control without it 3.8 to 4.4 (`checks/m2345_checks.json`): **PASS**. m3 the rollout term reaches the predictor and the state projection with a non-zero gradient, a target-only frame exactly 0.0: **PASS**. m4 the rollout inside the loss equals the rollout of the stack at read time, 0.0 (`MOTION_DEFS.md` item 9): **PASS**. m5 the total moves by exactly weight times the term, mutation form: **PASS** (m3 to m5 in `m2345_checks.json`). m6 the selfcheck reads the levers back from the BUILT model and refuses a wrong expectation (`checks/m6_selfcheck.json`, four argv shapes): **PASS**. Diagnostics flag on against off, same weights, max abs difference 0.0, shapes PR, RR, RES, ROLL (`checks_lead/lead_diag_check.json`): **PASS**. Tests: 55 passed (`tests.log`).
* Pod gate per arm (selfcheck on real data, PASS token before training): **NOT RE-VERIFIED HERE** (no pod access; the token files are on the pod). Evidence at hand: the dry run of the PR argv on real data read SELFCHECK PASS (row H-V8F-PROOF-35, `TIMES_NOTE.md` 02:13Z), and each of the five arms has a checkpoint with a recorded sha256 (`RESULT_S1_<ARM>.json`, meta `ckpt_sha256_record`).
* Lead read, per arm: t1 to t6 **PASS** for PR_s0, PR_s1, RES_s0, RR_s0 (`out_m/RESULT_RB_<ARM>.json`, `controls_pass` true); change form k1 to k3 **PASS** (`out_m/CHANGE_RB_*`); ADD17 variants x2 to x4 **PASS** (`ctx/RESULT_CTXROLL_<ARM>.json`); x1 (reproduce the banked pixel rollout) is **NOT EVALUATED** for these arms: the file reads `available false` because the pixel job of these arms was not run, and the tool prints PASS by default. It is evaluated for P_s0 only (`ctx/AFTERCHECK_CTXROLL_P_s0.json`, 0.00195 against 0.004). This is a gap in a control, not a failure; the names below are given with that flag.
* Step read: s1 (copy over copy is exactly 1) and s2 (shuffle keeps the point) **PASS** for all arms (`final/STEP_CTX_FINAL.json`).
* Position read (R): t1 to t4 **PASS** for every arm quoted (`spatial/s1/RESULT_S1_<ARM>.json`, `controls_pass` true).
* Family read (F): q1 to q5 **PASS** (window counts 2,399 windows of 141 episodes, 1,305 selection-critical and 1,094 common-case; self is zero 0.0; swap is minus 0.0; planted regression 0.899 m/s read back and named COST-LONGITUDINAL): `out_q3_twoseed/RESULT_Q3.json` (14 dumps) and `out_q3_oneseed/RESULT_Q3.json` (12 dumps), both ALL_PASS.

### 1.2 PR, both seeds (the only MOTION arm with both seeds)

| read | PR_s0 | PR_s1 |
|---|---|---|
| comparison arm of I, V | P_s0 | P_s1 |
| S: one step ahead, context 1, error ratio predictor over copy (1,948 windows, 139 episodes; below 1 is better) | 0.844 [0.831, 0.857] | 0.873 [0.864, 0.883] |
| S: same ratio minus that of the comparison arm | +0.056 [+0.047, +0.064] | +0.077 [+0.067, +0.086] |
| I: lead gap R2 at +2.0 s, rolled-out state (n 441 windows with a lead at both times) | 0.442 | 0.538 |
| I: present state (copy) | 0.440 | 0.513 |
| I: true future state (ceiling) | 0.498 | 0.597 |
| I: rollout minus copy | +0.002 [-0.062, +0.065] | +0.025 [-0.038, +0.084] |
| I: rollout minus ego-only floor (iv-b) | +0.263 [+0.068, +0.554] | +0.359 [+0.134, +0.684] |
| I: rollout minus rollout of the comparison arm, paired | -0.059 [-0.134, +0.013] | +0.032 [-0.043, +0.105] |
| V: change of the gap, skill of the rollout over persistence (441 windows, 48 episodes) | +0.014 [-0.092, +0.130] | +0.091 [-0.015, +0.212] |
| V: ceiling from the true future state | +0.159 [+0.026, +0.300] | +0.190 [+0.071, +0.328] |
| V: rollout skill minus that of the comparison arm, paired | +0.007 [-0.062, +0.085] | +0.035 [-0.063, +0.132] |
| beside: history-seeded rollout R3h minus copy (ADD17, names nothing here) | +0.020 [-0.020, +0.060] | +0.027 [-0.013, +0.066] |
| beside: latent error at +2.0 s, skill over copy, R1 | 0.251 | 0.239 |
| name at this seed: S / I / V | STEP-BELOW-COPY / MOTION-FLAT / MOTION-STILL | STEP-BELOW-COPY / MOTION-FLAT / MOTION-STILL |

| position, lead gap R2 (state) on the train probe set, 1,200 clips, paired over clips | difference | name by the rule of ADD16 C2 (six-seed spread 0.0450) |
|---|---|---|
| PR_s0 minus G_s0 | -0.000 [-0.019, +0.018] | KEEPS-POSITION |
| PR_s1 minus G_s1 | -0.024 [-0.043, -0.005] | KEEPS-POSITION |
| PR_s0 minus P_s0 | -0.011 [-0.029, +0.007] | KEEPS-POSITION |
| PR_s1 minus P_s1 | +0.009 [-0.011, +0.028] | KEEPS-POSITION |

**Names (both seeds, the card's rules):** S **`STEP-BELOW-COPY`** (0.844 [0.831, 0.857] and 0.873 [0.864, 0.883]; both intervals below 1; note that PR is worse on this read than P at both seeds, +0.056 [+0.047, +0.064] and +0.077 [+0.067, +0.086]). I **`MOTION-FLAT`** (rollout minus copy +0.002 [-0.062, +0.065] and +0.025 [-0.038, +0.084]; rollout minus the rollout of P, paired, -0.059 [-0.134, +0.013] and +0.032 [-0.043, +0.105]; no interval of the two deciding lines excludes 0 above). V **`MOTION-STILL`** (+0.014 [-0.092, +0.130] and +0.091 [-0.015, +0.212]). R **`KEEPS-POSITION`** (arm minus G: -0.0005 [-0.019, +0.018] and -0.024 [-0.043, -0.005]; the seed-1 difference is below 0 and smaller than the six-seed spread 0.0450, seed 0 is not below 0). F **`NO-COST`** for PR against G, against P and against HIER (section 1.5).

**Committed next step (card ADD18 section 6, quoted):** "`MOTION-FLAT` and `MOTION-STILL` with `STEP-BELOW-COPY`: the predictor fits the step and the latent target does not carry the motion of objects; next lever, own card: an explicit target of object motion (the frozen token grid of the future frame, cell by cell)." That lever is FUTTOK of card ADD19, arm PF, which has one seed (section 2.3). The second committed line, "`LOSES-POSITION` or any `COST-<family>`: the arm is not promoted", does not apply (no such name). The promotion line (`MOTION-IMAGINES` or `MOTION-MOVES` with `KEEPS-POSITION` and `NO-COST`) is not triggered by PR.

### 1.3 RES and RR, one seed each: reported with numbers, one seed, names nothing

| read | RES_s0 | RR_s0 |
|---|---|---|
| comparison arm of I, V | G_s0 | G_s0 |
| S: one step ahead, context 1, error ratio predictor over copy (1,948 windows, 139 episodes; below 1 is better) | 0.829 [0.814, 0.845] | 0.870 [0.858, 0.882] |
| S: same ratio minus that of the comparison arm | -0.404 [-0.456, -0.362] | -0.363 [-0.416, -0.317] |
| I: lead gap R2 at +2.0 s, rolled-out state (n 441 windows with a lead at both times) | 0.383 | 0.485 |
| I: present state (copy) | 0.366 | 0.435 |
| I: true future state (ceiling) | 0.425 | 0.502 |
| I: rollout minus copy | +0.017 [-0.021, +0.052] | +0.050 [+0.003, +0.098] |
| I: rollout minus ego-only floor (iv-b) | +0.205 [-0.008, +0.496] | +0.306 [+0.087, +0.619] |
| I: rollout minus rollout of the comparison arm, paired | -0.059 [-0.166, +0.056] | +0.043 [-0.066, +0.167] |
| V: change of the gap, skill of the rollout over persistence (441 windows, 48 episodes) | +0.029 [-0.032, +0.103] | +0.091 [-0.008, +0.212] |
| V: ceiling from the true future state | +0.116 [-0.039, +0.248] | +0.188 [+0.037, +0.354] |
| V: rollout skill minus that of the comparison arm, paired | +0.092 [-0.013, +0.264] | +0.153 [+0.050, +0.343] |
| beside: history-seeded rollout R3h minus copy (ADD17, names nothing here) | +0.021 [-0.015, +0.054] | +0.050 [+0.001, +0.101] |
| beside: latent error at +2.0 s, skill over copy, R1 | 0.221 | 0.313 |
| name at this seed: S / I / V | STEP-BELOW-COPY / MOTION-FLAT / MOTION-STILL | STEP-BELOW-COPY / MOTION-IMAGINES / MOTION-STILL |

| position, lead gap R2 (state) on the train probe set, 1,200 clips, paired over clips | difference | name by the rule of ADD16 C2 (six-seed spread 0.0450) |
|---|---|---|
| RES_s0 minus G_s0 | -0.021 [-0.038, -0.004] | KEEPS-POSITION |
| RR_s0 minus G_s0 | +0.029 [+0.012, +0.047] | KEEPS-POSITION |

RES_s0: one seed, names nothing. At this seed the rules read S `STEP-BELOW-COPY` (0.829 [0.814, 0.845]), I `MOTION-FLAT`, V `MOTION-STILL`, R `KEEPS-POSITION` (-0.021 [-0.038, -0.004], smaller than the spread 0.0450), F `NO-COST`. RR_s0: one seed, names nothing. At this seed the rules read S `STEP-BELOW-COPY` (0.870 [0.858, 0.882]), **I `MOTION-IMAGINES`** (rollout minus copy +0.050 [+0.003, +0.098], rollout minus the ego floor +0.306 [+0.087, +0.619], both excluding 0; the lower end of the first interval is 0.003; the history-seeded R3h gives +0.050 [+0.001, +0.101]), V `MOTION-STILL` (+0.091 [-0.008, +0.212]), R `KEEPS-POSITION` (+0.029 [+0.012, +0.047], above 0), F `NO-COST`. Between-seed differences of the same read in arm types that have two seeds, for scale: rollout minus copy G 0.018 (-0.054, -0.036), P 0.010 (+0.013, +0.023), PR 0.023 (+0.002, +0.025), PE 0.004 (+0.021, +0.017); the seed floor of the flat predictor with the residual form and the rollout loss is not measured (RR_s1 and RES_s1 do not exist).

### 1.4 Attribution sentence of the card

The card assigns an outcome of RR to RES, to ROLL, or to both, by which single-lever arm carries the same name. **It cannot be given.** RR_s0 reads `MOTION-IMAGINES` at one seed, which names nothing; RES_s0 reads `MOTION-FLAT` at its one seed; the other single-lever arm (ROLL_s0, ROLL_s1) was not trained. What the files do say: the residual form alone did not carry the I name at seed 0, and the rollout loss on the CELL predictor (PR, both seeds) is `MOTION-FLAT`. To name RR with both seeds the missing arms are RR_s1 (and ROLL_s0 for the attribution).

### 1.5 Four-family line (T1-kin, ADD12 Q3, 2,399 windows, 141 episodes; STRATEGIC PAUSED)

LONGITUDINAL, LATERAL, TACTICAL: PR **`NO-COST`** at both seeds against G, against P and against HIER (no cell reads WORSE by the rule; `out_q3_twoseed/VERDICT_Q3.md`); RES_s0 and RR_s0 `NO-COST` against G and against HIER at one seed (`out_q3_oneseed/VERDICT_Q3.md`; one seed, names nothing); STRATEGIC PAUSED (PI item 25). Said plainly beside the name: against HIER, PR has higher lateral errors at both seeds on three cells, heading +1.273 deg [+0.401, +2.205], curvature +0.0473 1/m [+0.0279, +0.0659], yaw rate +3.68 deg/s [+1.51, +5.98]; each interval excludes 0 on the worse side and each lies inside the six-seed spread of the baseline (2.551 deg, 0.1049 1/m, 10.65 deg/s), so by the rule of Q3 the name stays `NO-COST`. Against P, curvature error is +0.0443 1/m [+0.0090, +0.0863] (INSIDE-SPREAD, worse side); cross-track error is lower than P and HIER (-0.1212 m and -0.1357 m, INSIDE-SPREAD). All other cells OPEN (tables below).

**PR minus P**, both seeds, Q3 **NO-COST**

| cell (all windows, error, below 0 better) | per seed | D | two-level interval | rule |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1863 / -0.3037 | -0.0587 | [-0.3496, +0.2489] | OPEN |
| LONGITUDINAL:along_mae_m | +0.2492 / -0.2265 | +0.0114 | [-0.2787, +0.3262] | OPEN |
| LATERAL:cross_mae_m | -0.1721 / -0.0704 | -0.1212 | [-0.2108, -0.0425] | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.4613 / +0.2847 | +0.8730 | [-0.0467, +1.7859] | OPEN |
| LATERAL:curvature_mae_1pm | +0.0701 / +0.0184 | +0.0443 | [+0.0090, +0.0863] | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +7.2653 / -0.3248 | +3.4702 | [-1.4447, +8.3001] | OPEN |
| TACTICAL:lon_err_rate | +0.0367 / -0.0913 | -0.0273 | [-0.1076, +0.0530] | OPEN |
| TACTICAL:lat_err_rate | -0.0225 / -0.0263 | -0.0244 | [-0.0462, -0.0036] | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0267 / -0.1030 | -0.0381 | [-0.1182, +0.0429] | OPEN |
| TACTICAL:ade_tac_m | +0.1601 / -0.3799 | -0.1099 | [-0.4627, +0.2713] | OPEN |

**PR minus G**, both seeds, Q3 **NO-COST**

| cell (all windows, error, below 0 better) | per seed | D | two-level interval | rule |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1879 / -0.0396 | +0.0741 | [-0.0836, +0.2573] | OPEN |
| LONGITUDINAL:along_mae_m | +0.1709 / +0.0633 | +0.1171 | [-0.0082, +0.2677] | OPEN |
| LATERAL:cross_mae_m | -0.0271 / -0.0300 | -0.0285 | [-0.0773, +0.0215] | OPEN |
| LATERAL:heading_mae_deg | -0.1800 / +2.0900 | +0.9550 | [-0.4718, +2.5210] | OPEN |
| LATERAL:curvature_mae_1pm | +0.0047 / +0.0613 | +0.0330 | [-0.0056, +0.0717] | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.7494 / +6.0644 | +2.6575 | [-1.9584, +7.3994] | OPEN |
| TACTICAL:lon_err_rate | +0.0454 / -0.0192 | +0.0131 | [-0.0367, +0.0606] | OPEN |
| TACTICAL:lat_err_rate | -0.0246 / +0.0642 | +0.0198 | [-0.0417, +0.0798] | OPEN |
| TACTICAL:joint_err_rate | +0.0325 / +0.0079 | +0.0202 | [-0.0088, +0.0481] | OPEN |
| TACTICAL:ade_tac_m | +0.2119 / +0.0220 | +0.1170 | [-0.0717, +0.3329] | OPEN |

**PR minus HIER**, both seeds, Q3 **NO-COST**

| cell (all windows, error, below 0 better) | per seed | D | two-level interval | rule |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0999 / -0.0829 | +0.0085 | [-0.1340, +0.1632] | OPEN |
| LONGITUDINAL:along_mae_m | +0.0838 / +0.0146 | +0.0492 | [-0.0574, +0.1757] | OPEN |
| LATERAL:cross_mae_m | -0.0753 / -0.1960 | -0.1357 | [-0.2253, -0.0500] | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.7399 / +0.8062 | +1.2730 | [+0.4013, +2.2047] | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0420 / +0.0527 | +0.0473 | [+0.0279, +0.0659] | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +4.0918 / +3.2733 | +3.6825 | [+1.5144, +5.9755] | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0621 / -0.0083 | +0.0269 | [-0.0248, +0.0796] | OPEN |
| TACTICAL:lat_err_rate | +0.0400 / -0.0313 | +0.0044 | [-0.0500, +0.0529] | OPEN |
| TACTICAL:joint_err_rate | +0.0754 / -0.0192 | +0.0281 | [-0.0342, +0.0926] | OPEN |
| TACTICAL:ade_tac_m | +0.0648 / -0.1810 | -0.0581 | [-0.2660, +0.1813] | OPEN |

**RES minus G**, seed 0 only (one seed, names nothing), reads NO-COST

| cell (all windows, error, below 0 better) | per seed | D | two-level interval | rule |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0246 | +0.0246 | [-0.0788, +0.1300] | OPEN |
| LONGITUDINAL:along_mae_m | -0.0220 | -0.0220 | [-0.1336, +0.0904] | OPEN |
| LATERAL:cross_mae_m | -0.0183 | -0.0183 | [-0.0809, +0.0459] | OPEN |
| LATERAL:heading_mae_deg | -1.4610 | -1.4610 | [-2.1774, -0.8248] | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0450 | -0.0450 | [-0.0719, -0.0201] | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -3.1503 | -3.1503 | [-6.1465, -0.3944] | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0038 | +0.0038 | [-0.0288, +0.0371] | OPEN |
| TACTICAL:lat_err_rate | -0.0275 | -0.0275 | [-0.0592, +0.0029] | OPEN |
| TACTICAL:joint_err_rate | -0.0121 | -0.0121 | [-0.0417, +0.0188] | OPEN |
| TACTICAL:ade_tac_m | -0.0715 | -0.0715 | [-0.2279, +0.0862] | OPEN |

**RES minus HIER**, seed 0 only (one seed, names nothing), reads NO-COST

| cell (all windows, error, below 0 better) | per seed | D | two-level interval | rule |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0635 | -0.0635 | [-0.1739, +0.0470] | OPEN |
| LONGITUDINAL:along_mae_m | -0.1091 | -0.1091 | [-0.2407, +0.0165] | OPEN |
| LATERAL:cross_mae_m | -0.0665 | -0.0665 | [-0.1356, +0.0019] | OPEN |
| LATERAL:heading_mae_deg | +0.4589 | +0.4589 | [-0.2502, +1.1513] | OPEN |
| LATERAL:curvature_mae_1pm | -0.0077 | -0.0077 | [-0.0284, +0.0123] | OPEN |
| LATERAL:yaw_rate_mae_degps | +1.6909 | +1.6909 | [-1.1934, +4.3717] | OPEN |
| TACTICAL:lon_err_rate | +0.0204 | +0.0204 | [-0.0100, +0.0509] | OPEN |
| TACTICAL:lat_err_rate | +0.0371 | +0.0371 | [-0.0017, +0.0742] | OPEN |
| TACTICAL:joint_err_rate | +0.0308 | +0.0308 | [-0.0004, +0.0608] | OPEN |
| TACTICAL:ade_tac_m | -0.2185 | -0.2185 | [-0.4145, -0.0304] | INSIDE-SPREAD |

**RR minus G**, seed 0 only (one seed, names nothing), reads NO-COST

| cell (all windows, error, below 0 better) | per seed | D | two-level interval | rule |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0479 | -0.0479 | [-0.1407, +0.0475] | OPEN |
| LONGITUDINAL:along_mae_m | -0.0979 | -0.0979 | [-0.2095, +0.0223] | OPEN |
| LATERAL:cross_mae_m | -0.0254 | -0.0254 | [-0.1022, +0.0567] | OPEN |
| LATERAL:heading_mae_deg | -1.9430 | -1.9430 | [-2.7195, -1.2011] | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | -0.0486 | -0.0486 | [-0.0732, -0.0263] | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | -6.0283 | -6.0283 | [-8.7629, -3.5109] | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0133 | -0.0133 | [-0.0483, +0.0213] | OPEN |
| TACTICAL:lat_err_rate | -0.0521 | -0.0521 | [-0.0864, -0.0171] | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0313 | -0.0313 | [-0.0621, -0.0004] | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1612 | -0.1612 | [-0.3151, +0.0264] | OPEN |

**RR minus HIER**, seed 0 only (one seed, names nothing), reads NO-COST

| cell (all windows, error, below 0 better) | per seed | D | two-level interval | rule |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1360 | -0.1360 | [-0.2605, -0.0197] | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.1849 | -0.1849 | [-0.2986, -0.0759] | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0736 | -0.0736 | [-0.1498, +0.0044] | OPEN |
| LATERAL:heading_mae_deg | -0.0231 | -0.0231 | [-0.5034, +0.4793] | OPEN |
| LATERAL:curvature_mae_1pm | -0.0112 | -0.0112 | [-0.0318, +0.0091] | OPEN |
| LATERAL:yaw_rate_mae_degps | -1.1843 | -1.1843 | [-3.3412, +1.1348] | OPEN |
| TACTICAL:lon_err_rate | +0.0033 | +0.0033 | [-0.0287, +0.0346] | OPEN |
| TACTICAL:lat_err_rate | +0.0125 | +0.0125 | [-0.0267, +0.0496] | OPEN |
| TACTICAL:joint_err_rate | +0.0117 | +0.0117 | [-0.0188, +0.0421] | OPEN |
| TACTICAL:ade_tac_m | -0.3082 | -0.3082 | [-0.4896, -0.1420] | INSIDE-SPREAD |


## 2. Card ADD19 (TARGET)

### 2.1 Controls first

* Build (card section 5 and note N1, CPU, files in `F:/TanitAD-scratch/v8proof/spatial/target/`): defaults bit-identical to the tree of ADD18 (`checks/t1_identity.json`): **PASS**. Each lever reaches the loss, a control group exactly 0.0 (`checks/t2345_checks.json`, t2 to t5 flat and cell): **PASS**. Selfcheck reads the levers back and refuses a wrong expectation (`checks/t6_selfcheck.json`, default, PE, PF, PEF): **PASS**. Diagnostics flag leaves every weight equal, 0.0 (`checks/t8_diag.json`): **PASS**. Lead independent check: FUTTOK analytic value difference 0.0, frame shifted by one visible at 0.0045, third action channel equals speed over 10 (`checks_lead/lead_target_check.json`): **PASS**. Tests: 87 passed (`tests_lead.log`). Stated limit of N1: the FUTTOK rollout inside the loss equals the read-time rollout within 5.1e-7, not 0.0 (bar 1e-6).
* Pod gate per arm: **NOT RE-VERIFIED HERE** (no pod access). Evidence at hand: dry gates of PE_s0 and PF_s0 (row H-V8F-PROOF-39, `TIMES_NOTE.md` 04:11Z) and the checkpoint sha256 of each arm in `RESULT_S1_<ARM>.json`.
* e1 (read tools on a non-EGOV arm give the same numbers, max abs difference 0.0 on P_s0): **PASS** (`ctx/E1_CTXROLL_P_s0.json`, all files 0.0).
* e2 (EGOV: the speed channel shuffled across windows must raise the one-step error): **PASS** for PE_s0 ratio shuffled over true **1.078** and PE_s1 **1.099** (`ctx/RESULT_CTXROLL_PE_s0.json`, `_PE_s1.json`, control x5_speed_file; recomputed from `DIAG_CTX_PE_s*.npz`: 1.078, 1.099). The speed file equals the logged speed over 10, correlation with logged acceleration 0.99999999 (x5): **PASS**.
* Lead read, per arm: t1 to t6 **PASS** for PE_s0, PE_s1, PF_s0 (`out_t/RESULT_RB_<ARM>.json`); k1 to k3 **PASS**; x2 to x4 **PASS**; x1 **NOT EVALUATED** for these arms (pixel job not run; same flag as for the MOTION card).
* Step, position and family controls: as in the MOTION block (s1, s2; t1 to t4; q1 to q5, all PASS).

### 2.2 PE, both seeds

| read | PE_s0 | PE_s1 |
|---|---|---|
| comparison arm of I, V | P_s0 | P_s1 |
| S: one step ahead, context 1, error ratio predictor over copy (1,948 windows, 139 episodes; below 1 is better) | 0.785 [0.770, 0.801] | 0.791 [0.776, 0.807] |
| S: same ratio minus that of the comparison arm | -0.003 [-0.011, +0.004] | -0.005 [-0.012, +0.001] |
| I: lead gap R2 at +2.0 s, rolled-out state (n 441 windows with a lead at both times) | 0.521 | 0.495 |
| I: present state (copy) | 0.500 | 0.477 |
| I: true future state (ceiling) | 0.572 | 0.547 |
| I: rollout minus copy | +0.021 [-0.028, +0.073] | +0.017 [-0.044, +0.079] |
| I: rollout minus ego-only floor (iv-b) | +0.342 [+0.144, +0.624] | +0.316 [+0.117, +0.596] |
| I: rollout minus rollout of the comparison arm, paired | +0.021 [-0.041, +0.080] | -0.011 [-0.100, +0.065] |
| V: change of the gap, skill of the rollout over persistence (441 windows, 48 episodes) | -0.064 [-0.179, +0.039] | +0.042 [-0.059, +0.142] |
| V: ceiling from the true future state | +0.206 [+0.054, +0.359] | +0.179 [-0.010, +0.342] |
| V: rollout skill minus that of the comparison arm, paired | -0.071 [-0.203, +0.034] | -0.013 [-0.106, +0.059] |
| beside: history-seeded rollout R3h minus copy (ADD17, names nothing here) | +0.037 [-0.002, +0.079] | +0.031 [-0.009, +0.075] |
| beside: latent error at +2.0 s, skill over copy, R1 | 0.233 | 0.226 |
| name at this seed: S / I / V | STEP-SAME / TARGET-FLAT / TARGET-STILL | STEP-SAME / TARGET-FLAT / TARGET-STILL |

| position, lead gap R2 (state) on the train probe set, 1,200 clips, paired over clips | difference | name by the rule of ADD16 C2 (six-seed spread 0.0450) |
|---|---|---|
| PE_s0 minus P_s0 | +0.007 [-0.013, +0.026] | KEEPS-POSITION |
| PE_s1 minus P_s1 | +0.009 [-0.012, +0.031] | KEEPS-POSITION |

**Names (both seeds):** S **`STEP-SAME`** (one-step ratio minus that of P: -0.0031 [-0.0106, +0.0038] and -0.0048 [-0.0117, +0.0013]; both intervals include 0, so `STEP-IMPROVES` is not met; e2 passes, so the predictor does use the speed channel: shuffled over true 1.078 and 1.099). I **`TARGET-FLAT`** (rollout minus copy +0.021 [-0.028, +0.073] and +0.017 [-0.044, +0.079]; rollout minus the rollout of P, paired, +0.021 [-0.041, +0.080] and -0.011 [-0.100, +0.065]). V **`TARGET-STILL`** (-0.064 [-0.179, +0.039] and +0.042 [-0.059, +0.142]). R **`KEEPS-POSITION`** (+0.007 [-0.013, +0.026] and +0.009 [-0.012, +0.031]). F **`NO-COST`** against P and against HIER (section 2.4). The reference beside V that the card asks for, the ego block alone on the same windows, is **not produced by the read tools** (`rb_change.py` reads rollout and ceiling only); the value on the train probe clips, cited from the card, is +0.291 [+0.243, +0.334] (`spatial/s1/VERDICT_S1_G_s0.md`, in-sample) and is not the same windows. Limit of note ADD19_N1: PE also asks the inverse-dynamics head for the speed, and its auxiliary has another random draw than P of the same seed; a gain would not have been assigned to the predictor input alone (none was found).

**Committed next step (card ADD19 section 6, quoted):** "EGOV `STEP-SAME`: the hypothesis of section 1 is not supported; reported as written; the object-level slot is the next lever." The committed control PEi (note ADD19_N1) runs only if PE reads `STEP-IMPROVES` or better at seed 0: it does not. PEF runs only if PE or PF reads at least `TARGET-BEATS-BASE` or `STEP-IMPROVES` at seed 0: neither does (PE_s0 `STEP-SAME`/`TARGET-FLAT`; PF_s0 below), so PEF is not triggered by the card.

### 2.3 PF, one seed: reported with numbers, one seed, names nothing

| read | PF_s0 |
|---|---|
| comparison arm of I, V | P_s0 |
| S: one step ahead, context 1, error ratio predictor over copy (1,948 windows, 139 episodes; below 1 is better) | 0.914 [0.896, 0.933] |
| S: same ratio minus that of the comparison arm | +0.126 [+0.119, +0.134] |
| I: lead gap R2 at +2.0 s, rolled-out state (n 441 windows with a lead at both times) | 0.375 |
| I: present state (copy) | 0.458 |
| I: true future state (ceiling) | 0.503 |
| I: rollout minus copy | -0.082 [-0.206, +0.018] |
| I: rollout minus ego-only floor (iv-b) | +0.197 [-0.023, +0.495] |
| I: rollout minus rollout of the comparison arm, paired | -0.125 [-0.247, -0.019] |
| V: change of the gap, skill of the rollout over persistence (441 windows, 48 episodes) | -0.009 [-0.123, +0.090] |
| V: ceiling from the true future state | +0.189 [+0.074, +0.324] |
| V: rollout skill minus that of the comparison arm, paired | -0.015 [-0.138, +0.094] |
| beside: history-seeded rollout R3h minus copy (ADD17, names nothing here) | +0.015 [-0.026, +0.056] |
| beside: latent error at +2.0 s, skill over copy, R1 | -0.618 |
| name at this seed: S / I / V | STEP-SAME / TARGET-FLAT / TARGET-STILL |

| position, lead gap R2 (state) on the train probe set, 1,200 clips, paired over clips | difference | name by the rule of ADD16 C2 (six-seed spread 0.0450) |
|---|---|---|
| PF_s0 minus P_s0 | +0.003 [-0.015, +0.021] | KEEPS-POSITION |

PF_s0: one seed, names nothing. At this seed the rules read S `STEP-SAME` (the one-step ratio is 0.914 [0.896, 0.933], above P by +0.126 [+0.119, +0.134]; this is worse than P, not a gain), I `TARGET-FLAT` (rollout minus copy -0.082 [-0.206, +0.018]; minus the rollout of P -0.125 [-0.247, -0.019], below 0), V `TARGET-STILL` (-0.009 [-0.123, +0.090]), R `KEEPS-POSITION`, F `NO-COST` against P and HIER. Said plainly: PF_s0 has higher lateral errors than P (heading +1.09 deg, curvature +0.0626 1/m, yaw rate +7.05 deg/s) and higher than HIER on cross-track, heading, curvature, yaw rate and the tactical lateral error rate (+0.1109), every one INSIDE-SPREAD. K, the token read (pixel read, learned rollout minus copy at 2.0 s): **missing**. The pixel job was not run for PF_s0; it would come from `spatial/latvis_gw` / `latvis_p` (the pixel read of addendum 10, `gw_read_arm.sh` family), not from `read_t_arm.sh`. Beside: the +2.0 s latent error of PF_s0 from one state is above the copy (ratio 1.618 [1.527, 1.718] from frame 4, context 1), and the rollout with context grown from its own predictions (R3g) reads -0.871 [-1.384, -0.455] on the lead read; the history-seeded R3h reads +0.015 [-0.026, +0.056]. The card's line for FUTTOK `TARGET-FLAT` and `TARGET-STILL` at both seeds needs PF_s1, which does not exist; at seed 0 alone it is not a name.

### 2.4 Four-family line (T1-kin)

PE **`NO-COST`** at both seeds against P and against HIER (STRATEGIC PAUSED). PF_s0 `NO-COST` against P and HIER, one seed. PF_s0 minus PR_s0 reads `NO-COST GAIN-LONGITUDINAL GAIN-TACTICAL` at one seed (speed error -0.3205 m/s [-0.5074, -0.1625], BETTER outside the spread 0.2997; tactical lon error rate -0.0825 [-0.1189, -0.0500], BETTER); one seed, names nothing, and PR is a different arm from P.

**PE minus P**, both seeds, Q3 **NO-COST**

| cell (all windows, error, below 0 better) | per seed | D | two-level interval | rule |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0061 / -0.1839 | -0.0950 | [-0.2281, +0.0397] | OPEN |
| LONGITUDINAL:along_mae_m | +0.0065 / -0.2013 | -0.0974 | [-0.2492, +0.0582] | OPEN |
| LATERAL:cross_mae_m | -0.1242 / -0.0819 | -0.1031 | [-0.1642, -0.0479] | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.1791 / -1.4900 | -0.8346 | [-1.7213, +0.1995] | OPEN |
| LATERAL:curvature_mae_1pm | +0.0167 / -0.0304 | -0.0068 | [-0.0385, +0.0250] | OPEN |
| LATERAL:yaw_rate_mae_degps | +4.0165 / -4.3115 | -0.1475 | [-5.3136, +5.0331] | OPEN |
| TACTICAL:lon_err_rate | -0.0438 / -0.0888 | -0.0663 | [-0.1036, -0.0275] | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | -0.0859 / -0.0404 | -0.0632 | [-0.1022, -0.0217] | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0692 / -0.1138 | -0.0915 | [-0.1282, -0.0540] | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1355 / -0.3516 | -0.2436 | [-0.4227, -0.0585] | INSIDE-SPREAD |

**PE minus HIER**, both seeds, Q3 **NO-COST**

| cell (all windows, error, below 0 better) | per seed | D | two-level interval | rule |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0925 / +0.0368 | -0.0278 | [-0.1595, +0.0814] | OPEN |
| LONGITUDINAL:along_mae_m | -0.1589 / +0.0399 | -0.0595 | [-0.2255, +0.0984] | OPEN |
| LATERAL:cross_mae_m | -0.0275 / -0.2075 | -0.1175 | [-0.2343, +0.0078] | OPEN |
| LATERAL:heading_mae_deg | +0.0995 / -0.9685 | -0.4345 | [-1.1726, +0.5042] | OPEN |
| LATERAL:curvature_mae_1pm | -0.0112 / +0.0039 | -0.0037 | [-0.0221, +0.0116] | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.8474 / -0.7121 | +0.0676 | [-1.4837, +1.9348] | OPEN |
| TACTICAL:lon_err_rate | -0.0183 / -0.0058 | -0.0121 | [-0.0377, +0.0117] | OPEN |
| TACTICAL:lat_err_rate | -0.0233 / -0.0454 | -0.0344 | [-0.0613, -0.0094] | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0204 / -0.0300 | -0.0252 | [-0.0490, -0.0002] | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.2308 / -0.1528 | -0.1918 | [-0.3369, -0.0460] | INSIDE-SPREAD |

**PF minus P**, seed 0 only (one seed, names nothing), reads NO-COST

| cell (all windows, error, below 0 better) | per seed | D | two-level interval | rule |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.1342 | -0.1342 | [-0.2432, -0.0335] | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.0023 | -0.0023 | [-0.1128, +0.1017] | OPEN |
| LATERAL:cross_mae_m | -0.0055 | -0.0055 | [-0.0705, +0.0653] | OPEN |
| LATERAL:heading_mae_deg | +1.0914 | +1.0914 | [+0.5344, +1.6864] | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0626 | +0.0626 | [+0.0430, +0.0839] | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +7.0505 | +7.0505 | [+4.0388, +10.1873] | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0459 | -0.0459 | [-0.0783, -0.0166] | INSIDE-SPREAD |
| TACTICAL:lat_err_rate | +0.0484 | +0.0484 | [+0.0154, +0.0812] | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0246 | -0.0246 | [-0.0555, +0.0033] | OPEN |
| TACTICAL:ade_tac_m | -0.0227 | -0.0227 | [-0.1766, +0.1295] | OPEN |

**PF minus HIER**, seed 0 only (one seed, names nothing), reads NO-COST

| cell (all windows, error, below 0 better) | per seed | D | two-level interval | rule |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.2207 | -0.2207 | [-0.3477, -0.1004] | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | -0.1677 | -0.1677 | [-0.2952, -0.0480] | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.0912 | +0.0912 | [+0.0241, +0.1653] | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +1.3700 | +1.3700 | [+0.6910, +2.0976] | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0346 | +0.0346 | [+0.0097, +0.0621] | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +3.8799 | +3.8799 | [+0.3338, +7.3359] | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0204 | -0.0204 | [-0.0525, +0.0129] | OPEN |
| TACTICAL:lat_err_rate | +0.1109 | +0.1109 | [+0.0742, +0.1468] | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0242 | +0.0242 | [-0.0096, +0.0555] | OPEN |
| TACTICAL:ade_tac_m | -0.1180 | -0.1180 | [-0.3109, +0.0676] | OPEN |

**PF minus PR**, seed 0 only (one seed, names nothing), reads NO-COST GAIN-LONGITUDINAL GAIN-TACTICAL

| cell (all windows, error, below 0 better) | per seed | D | two-level interval | rule |
|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.3205 | -0.3205 | [-0.5074, -0.1625] | BETTER |
| LONGITUDINAL:along_mae_m | -0.2515 | -0.2515 | [-0.4744, -0.0659] | INSIDE-SPREAD |
| LATERAL:cross_mae_m | +0.1665 | +0.1665 | [+0.1095, +0.2281] | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | -0.3699 | -0.3699 | [-1.2086, +0.4256] | OPEN |
| LATERAL:curvature_mae_1pm | -0.0074 | -0.0074 | [-0.0389, +0.0193] | OPEN |
| LATERAL:yaw_rate_mae_degps | -0.2119 | -0.2119 | [-3.4269, +2.5333] | OPEN |
| TACTICAL:lon_err_rate | -0.0825 | -0.0825 | [-0.1189, -0.0500] | BETTER |
| TACTICAL:lat_err_rate | +0.0709 | +0.0709 | [+0.0371, +0.1046] | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0513 | -0.0513 | [-0.0889, -0.0188] | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | -0.1829 | -0.1829 | [-0.4724, +0.0747] | OPEN |


## 3. Six-seed spread of the trained-state read

The rule R (ADD16 C2) uses one quantity: the lead gap R2 of the trained state on the train probe set (in-sample for the model), the arm minus the comparison arm, against the **largest difference between baseline seeds on that number** (the range). Banked R names (rows H-V8F-PROOF-37 for P) used the **three-seed** spread (HIER_s0 to s2, range 0.0450, `spatial/s1/SPREAD_S1_HIER.md`); an earlier two-seed spread (0.0103, `SPREAD_S1_HIER_2seeds.md`) existed before that. The six seeds are now all read:

| arm | gap R2 (state), train probe set | file |
|---|---|---|
| HIER_s0 | 0.4556 [0.4106, 0.4962] | spatial/s1/RESULT_S1_HIER_s0.json |
| HIER_s1 | 0.4453 [0.3984, 0.4869] | spatial/s1/RESULT_S1_HIER_s1.json |
| HIER_s2 | 0.4106 [0.3584, 0.4583] | spatial/s1/RESULT_S1_HIER_s2.json |
| HIER_s3 | 0.4364 [0.3891, 0.4785] | spatial/s1/RESULT_S1_HIER_s3.json |
| HIER_s4 | 0.4240 [0.3747, 0.4706] | spatial/s1/RESULT_S1_HIER_s4.json |
| HIER_s5 | 0.4253 [0.3777, 0.4690] | spatial/s1/RESULT_S1_HIER_s5.json |

| quantity | six seeds s0 to s5 (min, max, range) | three seeds s0 to s2 (banked in C2 of ADD16) | two seeds s0, s1 |
|---|---|---|---|
| gap R2 (state) | 0.4106, 0.4556, **0.0450** | 0.4106, 0.4556, 0.0450 | 0.4453, 0.4556, 0.0103 |
| presence AUROC (state) | 0.7874, 0.7942, **0.0068** | 0.7874, 0.7897, 0.0023 | 0.7893, 0.7897, 0.0004 |
| change skill H | 0.2629, 0.2894, **0.0265** | 0.2629, 0.2678, 0.0049 | 0.2629, 0.2662, 0.0033 |
| change skill N | 0.2519, 0.2776, **0.0257** | 0.2519, 0.2776, 0.0257 | 0.2519, 0.2559, 0.0040 |
| change skill E | 0.2911, 0.2911, **0.0000** | 0.2911, 0.2911, 0.0000 | 0.2911, 0.2911, 0.0000 |

The range of the gap R2 over six seeds is **0.0450**, equal to the three-seed range (the three added seeds, 0.4364, 0.4240, 0.4253, lie inside [0.4106, 0.4556]). **No R name changes** when the six-seed spread replaces the three-seed spread: the names of PR, PE, PF, RES, RR and of the banked P arms stay `KEEPS-POSITION` (no difference is below 0 with an interval excluding 0 and larger than 0.0450; the largest is PR_s1 minus G_s1 at -0.024 and RR_s0 minus G_s0 at +0.029, a gain). Had the two-seed spread (0.0103) been used, RES_s0 minus G_s0 (-0.021 [-0.038, -0.004]) and PR_s1 minus G_s1 (-0.024 [-0.043, -0.005]) would each read below 0 and larger than the spread at their seed; neither arm has both seeds below 0, so no both-seed `LOSES-POSITION` would result for PR either. The six-seed spreads of the other quantities are wider than the three-seed ones (presence AUROC 0.0068 against 0.0023; change skill H 0.0265 against 0.0049); R does not use them. Reproduction check of the paired tool: P_s0 minus G_s0 recomputed from the `OOF_S1_*` files equals the banked +0.0104 (gap R2 0.5622 against 0.5518).

## 4. Limits

One training run per arm per seed; two seeds at most; a separated interval from a one-seed arm is necessary and not sufficient, and the six-seed spread of the baseline is measured on the position read only (the lead read has two baseline seeds of G and no spread of its own). The probes show what is linearly readable. The lead and change reads use 441 windows of 139 episodes (48 with a lead pair); a small effect on the change read cannot be separated on them. The position read is in-sample for the model. T1-kin is a kinematic rollout on logged frames, not driving performance. x1 of the rollout reads is not evaluated for the new arms (section 1.1). The per-arm pod gate tokens were not re-checked. The one-step difference of ratios (S of ADD19) was computed here with the resampling of `ctx_step.py` (B 2,000, seed 0) because no earlier file holds it. STRATEGIC: PAUSED (PI item 25).

## 5. Consistency with the banked seed-0 reports (rows H-V8F-PROOF-35 to 40, `GAC_ROWS_29.md` to `GAC_ROWS_33.md`)

Every seed-0 number in row 40 (PR_s0 +0.002 [-0.062, +0.065], PE_s0 +0.021 [-0.028, +0.073], change skill +0.014 and -0.064, latent skill 0.251 and 0.233, e2 1.078) is reproduced by the files read here. The row reported `IMAGINE-FLAT, no name` at seed 0; with seed 1 read, the card names are `MOTION-FLAT` (PR) and `TARGET-FLAT` (PE), the same direction. The row 40 sentence "if both seeds stay flat, the committed step of card ADD19 is the object-level slot" is the committed step recorded in section 2.2 for PE. For the cell predictor of ADD16 (`CELL-FLAT`, row 37) nothing changes. The only seed-0 report that differs from what a one-seed read of the new arms shows is RR_s0, which was not reported before and reads `MOTION-IMAGINES` at seed 0 alone.

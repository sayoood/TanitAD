# SHIP-R8 AMENDMENT 4d -- the H2cn stage re-trained at fit seeds 1 and 2

**REPLICATED**

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap, B 2000, bootstrap seed 0 (pair_read_r8.paired_cell -> taniteval.ci) over EPISODES with the models FIXED. It answers 'another draw of episodes' only. ONE refcv8 training seed; inference seeds 0 and 1 are never pooled; fit seeds 0/1/2 are never pooled (each read separately). R3 is the fit-seed floor made visible, never a lever.

## stages (sha256 of the file each read used)

* fit seed 1: `21780cf8a94f035f4b3996e314dd7daa4da881cc297f8d4e63f28109cabb620d` registered==file==forward record: True
* fit seed 2: `b1fb318ce6bd8d4714600bf6525b69829fa3186e5e4ebb84ba959849cc84dd1c` registered==file==forward record: True
* fit seed 0: `f293b7bdc06aaeec19a10fd8cac9af993c4116401034a9d4ff9c28649e256f32` registered==file==forward record: True

## R1 -- SHIP-R8 (1)-(7)+(1b), both inference seeds inside clauses 2-5

| fit seed | arm | OVERALL | per clause |
|---|---|---|---|
| 0 | H2CN_DEPLOY8 | **PASS** | {'1': 'PASS', '1b': 'PASS', '2': 'PASS', '3': 'PASS', '4': 'PASS', '5': 'PASS', '6': 'N/A', '7': 'PASS'} |
| 1 | H2CN_FS1_DEPLOY8 | **PASS** | {'1': 'PASS', '1b': 'PASS', '2': 'PASS', '3': 'PASS', '4': 'PASS', '5': 'PASS', '6': 'N/A', '7': 'PASS'} |
| 2 | H2CN_FS2_DEPLOY8 | **PASS** | {'1': 'PASS', '1b': 'PASS', '2': 'PASS', '3': 'PASS', '4': 'PASS', '5': 'PASS', '6': 'N/A', '7': 'PASS'} |

## R2 -- vs RETIME_DEPLOY8 on the same draw (S6 compliant for FD / MD; ADE compliant)

| fit seed | inf seed | E1 vs ha0_ext_pose (RETIME) | E3 vs ha0_ext_pose (RETIME) | ADE S2 vs RETIME | ADE S6 vs RETIME | FD vs RETIME | MD vs RETIME |
|---|---|---|---|---|---|---|---|
| 0 (beside) | 0 | -0.0471 [-0.0585, -0.0363] (-0.0342) | -1.1975 [-1.4942, -0.9168] (-0.8682) | -0.0129 [-0.0196, -0.0062] SEP- n=4159 | -0.3292 [-0.4828, -0.1833] SEP- n=3251 | -0.0765 [-0.1164, -0.0377] SEP- n=1856 | +0.0642 [-0.0118, +0.1473] ns n=639 |
| 0 (beside) | 1 | -0.0464 [-0.0576, -0.0360] (-0.0329) | -1.1909 [-1.4800, -0.9163] (-0.8676) | -0.0135 [-0.0202, -0.0073] SEP- n=4159 | -0.3232 [-0.4673, -0.1826] SEP- n=3251 | -0.0770 [-0.1197, -0.0380] SEP- n=1856 | +0.0610 [-0.0157, +0.1425] ns n=639 |
| 1 | 0 | -0.0470 [-0.0590, -0.0356] (-0.0342) | -1.1190 [-1.4281, -0.8256] (-0.8682) | -0.0127 [-0.0192, -0.0066] SEP- n=4159 | -0.2507 [-0.4123, -0.0885] SEP- n=3251 | -0.0776 [-0.1172, -0.0397] SEP- n=1856 | +0.0751 [+0.0048, +0.1517] SEP+ n=639 |
| 1 | 1 | -0.0463 [-0.0581, -0.0351] (-0.0329) | -1.1129 [-1.4089, -0.8271] (-0.8676) | -0.0133 [-0.0195, -0.0074] SEP- n=4159 | -0.2453 [-0.4063, -0.0874] SEP- n=3251 | -0.0781 [-0.1206, -0.0392] SEP- n=1856 | +0.0751 [+0.0048, +0.1506] SEP+ n=639 |
| 2 | 0 | -0.0471 [-0.0593, -0.0353] (-0.0342) | -1.1303 [-1.4331, -0.8401] (-0.8682) | -0.0129 [-0.0192, -0.0067] SEP- n=4159 | -0.2621 [-0.4214, -0.0991] SEP- n=3251 | -0.0673 [-0.1081, -0.0278] SEP- n=1856 | +0.0548 [-0.0234, +0.1363] ns n=639 |
| 2 | 1 | -0.0463 [-0.0583, -0.0349] (-0.0329) | -1.1247 [-1.4238, -0.8398] (-0.8676) | -0.0133 [-0.0194, -0.0076] SEP- n=4159 | -0.2571 [-0.4137, -0.0959] SEP- n=3251 | -0.0679 [-0.1098, -0.0280] SEP- n=1856 | +0.0532 [-0.0272, +0.1343] ns n=639 |

## R3 -- each replicate PAIRED vs H2CN_DEPLOY8 (fit seed 0) on the same draw: the fit-seed floor (never a lever)

| fit seed | inf seed | ADE S2 compl | ADE S6 compl | ADE S2 all | ADE S6 all | FD S6 compl | MD S6 compl |
|---|---|---|---|---|---|---|---|
| 1 | 0 | +0.0001 [-0.0023, +0.0027] ns n=4159 | +0.0785 [+0.0205, +0.1565] SEP+ n=3251 | -0.0001 [-0.0025, +0.0021] ns n=4754 | +0.0723 [+0.0194, +0.1394] SEP+ n=3668 | -0.0011 [-0.0080, +0.0049] ns n=1856 | +0.0110 [-0.0111, +0.0320] ns n=639 |
| 1 | 1 | +0.0001 [-0.0022, +0.0026] ns n=4159 | +0.0779 [+0.0209, +0.1554] SEP+ n=3251 | -0.0002 [-0.0025, +0.0021] ns n=4754 | +0.0723 [+0.0202, +0.1387] SEP+ n=3668 | -0.0011 [-0.0080, +0.0049] ns n=1856 | +0.0141 [-0.0074, +0.0355] ns n=639 |
| 2 | 0 | +0.0000 [-0.0030, +0.0031] ns n=4159 | +0.0671 [+0.0073, +0.1451] SEP+ n=3251 | -0.0004 [-0.0032, +0.0022] ns n=4754 | +0.0591 [+0.0031, +0.1285] SEP+ n=3668 | +0.0092 [-0.0038, +0.0210] ns n=1856 | -0.0094 [-0.0314, +0.0113] ns n=639 |
| 2 | 1 | +0.0001 [-0.0029, +0.0031] ns n=4159 | +0.0662 [+0.0063, +0.1435] SEP+ n=3251 | -0.0003 [-0.0031, +0.0024] ns n=4754 | +0.0589 [+0.0034, +0.1279] SEP+ n=3668 | +0.0092 [-0.0038, +0.0210] ns n=1856 | -0.0078 [-0.0292, +0.0130] ns n=639 |

## controls

{
 "stages_registered_equal": true,
 "alignment": true,
 "retime_bit_equal": true,
 "e1e3_recompute_equal": true,
 "identity_pairing_zero": true,
 "mispaired_widens_ci": true,
 "repro_seed0": true,
 "forward_controls_ALL_PASS": true,
 "K_RETIME_identity_each_fit_seed": true,
 "RETIME_TAC_identity_each_fit_seed": true,
 "K_DISJ_each_fit_seed": true,
 "K_REG_each_fit_seed": true,
 "ALL_PASS": true
}

### seed-0 reproduction

{
 "what": "the REGISTERED seed-0 stage re-read from this code copy (out_4d_fs0/) vs the landed ship_r8_sub/out/ -- exact leaf / bit equality",
 "forward": {
  "n_leaves": 18240,
  "n_mismatches": 0,
  "first_mismatches": [],
  "h2cn_stage_sha256_equal": true,
  "h2cn_stage_sha256": "f293b7bdc06aaeec19a10fd8cac9af993c4116401034a9d4ff9c28649e256f32",
  "K_REG_both_true": true,
  "ALL_PASS_both_true": true
 },
 "verdict": {
  "overall": [
   "PASS",
   "PASS"
  ],
  "per_clause_equal": true,
  "n_leaves": 2392,
  "n_mismatches": 0,
  "first_mismatches": []
 },
 "tac": {
  "H2CN_DEPLOY8": {
   "n_leaves": 1986,
   "n_mismatches": 0,
   "first_mismatches": []
  },
  "H2CN_NEAR": {
   "n_leaves": 1986,
   "n_mismatches": 0,
   "first_mismatches": []
  },
  "RETIME_DEPLOY8": {
   "n_leaves": 1986,
   "n_mismatches": 0,
   "first_mismatches": []
  }
 },
 "positions_summary": {
  "n_arrays_compared": 156,
  "n_arrays_differ": 0
 },
 "PASS": true
}

## the word (Amendment 4d section 3)

**REPLICATED**


## deviations (disclosed)

* Internal arm key: the replicate stages are scored under the forward's internal arm key H2CN_DEPLOY8 (renaming it would touch every script); out_4d_fs{1,2}/arms/h2cn/H2CN_DEPLOY8 holds the FIT-SEED stage, named by sha256 in the forward (h2cn_stage), in every DEPLOY8_POS npz meta (h2cn_stage_sha256) and in SHIP_R8_H2CN_FS{s}_DEPLOY8.json (stage_sha256_read).
* The seed-0 arm is the REGISTERED stage ship_r8_sub/H2CN_STAGE_FROZEN.pt (read-only), re-read from this code copy into out_4d_fs0/; H2CN_FS0_STAGE_FROZEN.pt (the repro copy) is not scored.
* K-PERSURF (S2 H2cn - H0 ADE == -0.0130 / -0.0136 within 1e-4) is the SEED-0 fit's confirmed cell: required for the seed-0 reproduction, recorded but NOT required for a replicate (forward controls.K_PERSURF_required false; verdict clause 1 states it). Its replicate values are in each forward's K_PERSURF_detail.
* Clause 1 for a replicate adds the Amendment 4d seal (re-hashed now == the forward's record) and uses REGISTERED_H2CN_FS{s}_STAGE.sha256 / H2CN_FS{s}_FREEZE_INFO.json.
* S2 per-window ADE (R2 / R3) has no existing loader: load_s2 mirrors l2_common.load_eval_pf's panel part on dump_s{seed}; it is validated by the E1 recompute control (equals the verdict's registered-code cell).
* R3 ADE is reported on the compliant set (the clauses' set) and on all windows; FD / MD on the S6 compliant set as h2cn_fdmd_diag.

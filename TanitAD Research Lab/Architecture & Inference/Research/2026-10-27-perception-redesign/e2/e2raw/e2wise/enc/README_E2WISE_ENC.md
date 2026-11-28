# E2-WISE ENC: the E2 rule on TRUNK (core.encoder)-interpolated weights, K2DC vs K0 -- verdict NO-GO

Pre-registered before any E2-WISE number: `../../PREREG_E2WISE_2026-11-17.md` (landed ad9f1cbe) + `../../PREREG_E2WISE_2026-11-17_ADDENDUM1.md`
(landed 5f647742). Weights: `stack/scripts/weight_interp.py` (landed 20ab9f54): theta = theta_init + 0.5 (theta_tail - theta_init) on S_enc, every other tensor copied from `model_ema_tail5k6k.pt`; init `/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt`. Both arms get the same
S and alpha. Zero training. DEV = eval139 even half, 70 episodes. One training run per arm (`H-ESTIM-SEED-1`): a separated interval
answers another draw of EPISODES only. Interpolated file md5: K2DC bf0eb53aa07c520b205db4289b1c0c4c, K0 ad0d7e3b7912c627b700a2722d99b03f (every dump.json
records it, `stageA/dump_headers/`).

## Stage A -- clause (a), map primary `nf_swedge_prec_lv_5_20` at each arm's OWN FIT thresholds (T0, MEASURED)

| arm | E2-TAIL | this read |
|---|---|---|
| K2DC | 0.4850 | **0.4589** [0.3765, 0.5340] |
| K0 | 0.4665 | **0.4524** [0.3734, 0.5224] |

* WISE-ENC changes `core.encoder`, so stage A is a real re-read: (a) PASS, (b) FAIL (K2DC 0.311640, K0 0.315529; E2-TAIL 0.307623 / 0.307665).

## Stage B -- the read's verdict (quoted, unedited from `stageB/read/E2_VERDICT.md`)

# E2 verdict: NO-GO  (K2 vs K0)

| clause | status | tier | seeds not PASS |
|---|---|---|---|
| (a) MAP primary nf_swedge_prec_lv_5_20: K2 > K0 (point) | **PASS** | T0 (label level) | - |
| (b) box AP@2 m head output no NMS: K2 > K0 (point) | **FAIL** | T0 (label level) | 0,1 |
| (c) planner sidewalk rate S6c-DEV: K2 < K0 (point) | **PASS** | T1 self-action open loop (S6c-DEV) | - |
| (d) no four-family row separated worse (paired, episodes) | **FAIL** | T1 self-action open loop | 0,1 |
| (e) W4 preview: K2 separated BETTER than camera-blind on ade_m + each family primary | **PASS** | T1 self-action open loop | - |

GO iff K2 beats K0 in point estimate on the map primary, box AP@2 m and the planner's S6c sidewalk rate, no four-family row is separated worse, and a partial W4 preview (K2 vs camera-blind) is separated -- at inference seeds 0 and 1 (REFCV9_DESIGN 4.4).
NO-GO names clause (b) box AP@2 m head output no NMS: K2 > K0 (point) [FAIL, seeds ['0', '1']] -- next lever: box lever: B2 perspective aux / B3 quality-aware presence + DN / B4 queries (REFCV9_DESIGN 4.3), re-read AP@2 m
NO-GO names clause (d) no four-family row separated worse (paired, episodes) [FAIL, seeds ['0', '1']] -- next lever: the separated-worse row(s) named in the clause: diagnose that family before the 214 h are spent

* (d) seed 0: separated WORSE: LONGITUDINAL/LON_dk_headway_min_m, LONGITUDINAL/LON_dk_min_ttc_s, LATERAL/LAT_heading_mae_deg, LATERAL/LAT_yaw_rate_mae_radps_valid, LATERAL/LAT_cross_mae_m, LATERAL/LAT_cross_abs_final_m, TACTICAL/TAC_traj_lat_correct
* (d) seed 1: separated WORSE: LONGITUDINAL/LON_dk_min_ttc_s, LATERAL/LAT_heading_mae_deg, LATERAL/LAT_yaw_rate_mae_radps_valid, LATERAL/LAT_cross_mae_m, LATERAL/LAT_cross_abs_final_m

## (d), row by row: E2-TAIL vs this read (paired K2DC - K0, episode-cluster bootstrap, T1, 70 episodes)

Every row separated worse at either seed in either read. WORSE = separated in the bad direction.

| family / row | tail s0 | enc s0 | tail s1 | enc s1 |
|---|---|---|---|---|
| LONGITUDINAL / `LON_dk_headway_min_m` | -0.0369 [-0.0976, +0.0224] | -0.0428 [-0.0936, -0.0011] **WORSE** | -0.0089 [-0.0553, +0.0431] | -0.0114 [-0.0635, +0.0490] |
| LONGITUDINAL / `LON_dk_min_ttc_s` | -0.3953 [-0.9620, -0.0229] **WORSE** | -0.3910 [-0.9688, -0.0292] **WORSE** | -0.3532 [-0.8438, -0.0070] **WORSE** | -0.3540 [-0.8576, -0.0288] **WORSE** |
| LATERAL / `LAT_heading_mae_deg` | +0.1096 [+0.0442, +0.1813] **WORSE** | +0.1036 [+0.0372, +0.1769] **WORSE** | +0.0804 [+0.0175, +0.1515] **WORSE** | +0.1013 [+0.0239, +0.1917] **WORSE** |
| LATERAL / `LAT_yaw_rate_mae_radps_valid` | +0.0022 [+0.0009, +0.0036] **WORSE** | +0.0018 [+0.0002, +0.0034] **WORSE** | +0.0016 [+0.0004, +0.0030] **WORSE** | +0.0019 [+0.0000, +0.0037] **WORSE** |
| LATERAL / `LAT_cross_mae_m` | +0.0126 [+0.0063, +0.0194] **WORSE** | +0.0127 [+0.0062, +0.0194] **WORSE** | +0.0091 [+0.0024, +0.0166] **WORSE** | +0.0122 [+0.0052, +0.0204] **WORSE** |
| LATERAL / `LAT_cross_abs_final_m` | +0.0288 [+0.0151, +0.0435] **WORSE** | +0.0279 [+0.0136, +0.0432] **WORSE** | +0.0203 [+0.0053, +0.0370] **WORSE** | +0.0271 [+0.0118, +0.0453] **WORSE** |
| TACTICAL / `TAC_traj_lat_correct` | -0.0071 [-0.0146, -0.0004] **WORSE** | -0.0075 [-0.0146, -0.0008] **WORSE** | -0.0071 [-0.0147, +0.0000] | -0.0083 [-0.0175, +0.0000] |

## Secondary readout: K2DC seed-0 `LAT_cross_mae_m` (decides nothing alone)

* K2DC 0.1153 m (tail 0.1160), K0 0.1026 m (tail 0.1034); gap +0.0127 (tail +0.0126). Both arms above their tail values: NO.
* STRATEGIC: the read prints PAUSED (its PI item-25 text). The PI ended that pause 2026-11-16; in these arms the strategic modules are
  equal to the init (S_plan excludes them), so there is no trained strategic row to compare.

## Next (Rule Zero)

* prereg s4 outcome 2, WISE-ENC branch B (WISE-ENC fails (d) as well): neither half alone carries the lateral deficit. Reported; the 0.02 arm stays the pre-registered next arm, but the PI's conditional approval does NOT fire on this read -> PI. Addendum 2's trigger fires (WISE-BOTH, kept). Addendum 3 (landed 93688696 after WISE-ENC's family rows, before this verdict file) runs PLAN-RESET (planner at alpha 0) first, because WISE-PLAN showed a dose response in planner drift and WISE-ENC moved neither arm's lateral. Caveat: core.encoder's BatchNorm running statistics are interpolated with the weights (their num_batches_tracked counters are copied). Other clauses not PASS: (b) FAIL (reported as written).
* E2, E2-RAW and E2-TAIL stand as written.

## Provenance

* Stage A files pulled from Thor `/home/nvidia/r8ns/r9e2/runs/E2WISE_ENC/stageA/`, sha256-equal. Rasters, perc_det.pkl and
  the interpolated .pt files are not banked (re-creatable: weight_interp.py on Thor's runs/r9e2-*/model_ema_tail5k6k.pt, md5 above). Stage B read files are
  byte-exact copies of the dev-box S6c driver's pull (sha256-verified against Thor `runs/E2WISE_ENC/stageB/`).

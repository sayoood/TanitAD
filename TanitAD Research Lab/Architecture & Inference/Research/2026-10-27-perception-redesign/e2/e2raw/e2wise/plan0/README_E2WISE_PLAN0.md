# E2-WISE PLAN0 (PLAN-RESET): the E2 rule on PLANNER-RESET (alpha 0)-interpolated weights, K2DC vs K0 -- verdict NO-GO

Pre-registered before any E2-WISE number: `../../PREREG_E2WISE_2026-11-17.md` (landed ad9f1cbe) + `../../PREREG_E2WISE_2026-11-17_ADDENDUM1.md`
(landed 5f647742) + `../../PREREG_E2WISE_2026-11-17_ADDENDUM3.md` (landed 93688696). Weights: `stack/scripts/weight_interp.py` (landed 20ab9f54): theta = theta_init + 0 (theta_tail - theta_init) on S_plan (20 planner prefixes), every other tensor copied from `model_ema_tail5k6k.pt`; init `/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt`. Both arms get the same
S and alpha. Zero training. DEV = eval139 even half, 70 episodes. One training run per arm (`H-ESTIM-SEED-1`): a separated interval
answers another draw of EPISODES only. Interpolated file md5: K2DC 29820dd0ec277939145ba5bc026407ee, K0 beae7c8cfded14aae54486b66a440b54 (every dump.json
records it, `stageA/dump_headers/`).

## Stage A -- clause (a), map primary `nf_swedge_prec_lv_5_20` at each arm's OWN FIT thresholds (T0, MEASURED)

| arm | E2-TAIL | this read |
|---|---|---|
| K2DC | 0.4850 | **0.4850** [0.3996, 0.5632] |
| K0 | 0.4665 | **0.4665** [0.3852, 0.5375] |

* Perception check (ADDENDUM 1, required by Addendum 3): map points equal E2-TAIL to 1e-9: YES; box AP@2m points equal E2-TAIL to 1e-6: YES (K2DC 0.307623 vs 0.307623, K0 0.307665 vs 0.307665). PLAN-RESET copies `core.encoder`, `_perception` and `core.agent_head` bit-exactly; (d) decides the lever.

## Notes on this read

* Stage A's last STATUS line says "PLAN-RESET (alpha 0.5) weights": that label is text inherited from the WISE-PLAN script. The weights are
  alpha 0: both provenance files record alpha 0.0, and every dump's `ckpt_md5` equals the alpha-0 file's md5 (guarded above).
* Addendum 3's control says `n_copied_in_S` 6 for both arms. MEASURED: K2DC 6, K0 0 (K0 has no decoder box_gate). The control was
  applied per arm against that arm's own WISE-PLAN provenance, which is what the addendum's equality means.
* Clause (b) is judged under the PI's 2026-11-17 ruling (landed 50856a00 before this read's S6c): clause (b) paired box AP@2m K2 - K0 -0.0000 [-0.0077, +0.0080] (paired episode-cluster bootstrap, B 2000), not separated -> PASS.

## Dose readout, seed 0 (paired K2DC - K0, T1, 70 episodes; secondary, decides nothing alone)

| row | E2-TAIL (alpha 1) | WISE-PLAN (alpha 0.5) | PLAN-RESET (alpha 0) |
|---|---|---|---|
| LATERAL / `LAT_cross_mae_m` | +0.0126 [+0.0063, +0.0194] **WORSE** | +0.0061 [-0.0002, +0.0124] | +0.0113 [+0.0048, +0.0179] **WORSE** |
| LATERAL / `LAT_cross_abs_final_m` | +0.0288 [+0.0151, +0.0435] **WORSE** | +0.0137 [+0.0005, +0.0269] **WORSE** | +0.0242 [+0.0101, +0.0389] **WORSE** |
| LATERAL / `LAT_heading_mae_deg` | +0.1096 [+0.0442, +0.1813] **WORSE** | +0.0505 [-0.0217, +0.1236] | +0.1136 [+0.0410, +0.1928] **WORSE** |
| LATERAL / `LAT_yaw_rate_mae_radps_valid` | +0.0022 [+0.0009, +0.0036] **WORSE** | +0.0012 [+0.0001, +0.0024] **WORSE** | +0.0025 [+0.0011, +0.0042] **WORSE** |
| LONGITUDINAL / `LON_dk_min_ttc_s` | -0.3953 [-0.9620, -0.0229] **WORSE** | +0.0706 [-0.2523, +0.3498] | +0.3561 [-0.0017, +0.7794] |
| TACTICAL / `TAC_traj_lat_correct` | -0.0071 [-0.0146, -0.0004] **WORSE** | -0.0088 [-0.0162, -0.0021] **WORSE** | -0.0079 [-0.0154, -0.0008] **WORSE** |

## Stage B -- the read's verdict (quoted, unedited from `stageB/read/E2_VERDICT.md`)

# E2 verdict: NO-GO  (K2 vs K0)

| clause | status | tier | seeds not PASS |
|---|---|---|---|
| (a) MAP primary nf_swedge_prec_lv_5_20: K2 > K0 (point) | **PASS** | T0 (label level) | - |
| (b) box AP@2 m head output no NMS: K2 not separated worse than K0 (paired, episodes) | **PASS** | T0 (label level) | - |
| (c) planner sidewalk rate S6c-DEV: K2 < K0 (point) | **PASS** | T1 self-action open loop (S6c-DEV) | - |
| (d) no four-family row separated worse (paired, episodes) | **FAIL** | T1 self-action open loop | 0,1 |
| (e) W4 preview: K2 separated BETTER than camera-blind on ade_m + each family primary | **PASS** | T1 self-action open loop | - |

GO iff K2 beats K0 in point estimate on the map primary and the planner's S6c sidewalk rate, K2 is not separated worse than K0 on box AP@2 m, no four-family row is separated worse, and a partial W4 preview (K2 vs camera-blind) is separated -- at inference seeds 0 and 1 (REFCV9_DESIGN 4.4) (clause (b) amended by the PI 2026-11-17: not separated worse).
NO-GO names clause (d) no four-family row separated worse (paired, episodes) [FAIL, seeds ['0', '1']] -- next lever: the separated-worse row(s) named in the clause: diagnose that family before the 214 h are spent

* (d) seed 0: separated WORSE: LONGITUDINAL/LON_along_mae_m, LONGITUDINAL/LON_dk_lead_progress_shortfall, LATERAL/LAT_heading_mae_deg, LATERAL/LAT_curvature_mae_1pm_valid, LATERAL/LAT_yaw_rate_mae_radps_valid, LATERAL/LAT_cross_mae_m, LATERAL/LAT_cross_abs_final_m, TACTICAL/TAC_traj_lat_correct
* (d) seed 1: separated WORSE: LONGITUDINAL/LON_dk_lead_progress_shortfall, LATERAL/LAT_heading_mae_deg, LATERAL/LAT_curvature_mae_1pm_valid, LATERAL/LAT_yaw_rate_mae_radps_valid, LATERAL/LAT_cross_mae_m, LATERAL/LAT_cross_abs_final_m, TACTICAL/TAC_traj_lat_correct

## (d), row by row: E2-TAIL vs this read (paired K2DC - K0, episode-cluster bootstrap, T1, 70 episodes)

Every row separated worse at either seed in either read. WORSE = separated in the bad direction.

| family / row | tail s0 | plan0 s0 | tail s1 | plan0 s1 |
|---|---|---|---|---|
| LONGITUDINAL / `LON_along_mae_m` | +0.0035 [-0.0058, +0.0125] | +0.0141 [+0.0007, +0.0281] **WORSE** | +0.0009 [-0.0081, +0.0101] | +0.0104 [-0.0019, +0.0231] |
| LONGITUDINAL / `LON_dk_min_ttc_s` | -0.3953 [-0.9620, -0.0229] **WORSE** | +0.3561 [-0.0017, +0.7794] | -0.3532 [-0.8438, -0.0070] **WORSE** | +0.3317 [-0.0666, +0.7293] |
| LONGITUDINAL / `LON_dk_lead_progress_shortfall` | -0.0022 [-0.0069, +0.0019] | +0.0094 [+0.0027, +0.0185] **WORSE** | -0.0004 [-0.0043, +0.0038] | +0.0095 [+0.0031, +0.0172] **WORSE** |
| LATERAL / `LAT_heading_mae_deg` | +0.1096 [+0.0442, +0.1813] **WORSE** | +0.1136 [+0.0410, +0.1928] **WORSE** | +0.0804 [+0.0175, +0.1515] **WORSE** | +0.0880 [+0.0079, +0.1639] **WORSE** |
| LATERAL / `LAT_curvature_mae_1pm_valid` | +0.0005 [-0.0001, +0.0010] | +0.0008 [+0.0003, +0.0015] **WORSE** | +0.0004 [-0.0001, +0.0009] | +0.0007 [+0.0002, +0.0012] **WORSE** |
| LATERAL / `LAT_yaw_rate_mae_radps_valid` | +0.0022 [+0.0009, +0.0036] **WORSE** | +0.0025 [+0.0011, +0.0042] **WORSE** | +0.0016 [+0.0004, +0.0030] **WORSE** | +0.0021 [+0.0005, +0.0037] **WORSE** |
| LATERAL / `LAT_cross_mae_m` | +0.0126 [+0.0063, +0.0194] **WORSE** | +0.0113 [+0.0048, +0.0179] **WORSE** | +0.0091 [+0.0024, +0.0166] **WORSE** | +0.0089 [+0.0015, +0.0166] **WORSE** |
| LATERAL / `LAT_cross_abs_final_m` | +0.0288 [+0.0151, +0.0435] **WORSE** | +0.0242 [+0.0101, +0.0389] **WORSE** | +0.0203 [+0.0053, +0.0370] **WORSE** | +0.0194 [+0.0034, +0.0356] **WORSE** |
| TACTICAL / `TAC_traj_lat_correct` | -0.0071 [-0.0146, -0.0004] **WORSE** | -0.0079 [-0.0154, -0.0008] **WORSE** | -0.0071 [-0.0147, +0.0000] | -0.0108 [-0.0183, -0.0042] **WORSE** |

## Secondary readout: K2DC seed-0 `LAT_cross_mae_m` (decides nothing alone)

* K2DC 0.1191 m (tail 0.1160), K0 0.1078 m (tail 0.1034); gap +0.0113 (tail +0.0126). Both arms above their tail values: YES.
* STRATEGIC: the read prints PAUSED (its PI item-25 text). The PI ended that pause 2026-11-16; in these arms the strategic modules are
  equal to the init (S_plan excludes them), so there is no trained strategic row to compare.

## Next (Rule Zero)

* Addendum 3 committed outcome 3 (a lateral gap at alpha 0 is not below its alpha-0.5 value: LAT_cross_mae_m, LAT_cross_abs_final_m, LAT_heading_mae_deg, LAT_yaw_rate_mae_radps_valid): the response is not monotone, so planner drift is not the simple carrier. Next diagnostic: the coupling between the planner and the modules outside S_plan (_map_hires, _perception), under its own prereg -- pre-registered as Addendum 4, WIRE-OFF (landed 81d409d9 after this read's family rows and before this verdict file): K2DC's two planner wires (--r9-plan-map-channels' zero-init map_chan_proj, --r9-plan-box-tokens' 6 zero-init box_gates; K0 has neither, and PLAN-RESET reset neither) set to 0, queued on Thor after WISE-BOTH. Clause (b): Addendum 3 says PLAN-RESET 'cannot GO' because (b) stays the E2-TAIL tie; that was written under the point rule. The PI's 2026-11-17 ruling ("Not separated worse", landed 50856a00 before this read's S6c) now judges (b): clause (b) paired box AP@2m K2 - K0 -0.0000 [-0.0077, +0.0080] (paired episode-cluster bootstrap, B 2000), not separated -> PASS. Verdict as read: NO-GO. Outcome-4 condition met (both arms' s0 cross-track above their tail values); reported, not a reason to tune alpha on DEV.
* E2, E2-RAW and E2-TAIL stand as written.

## Provenance

* Stage A files pulled from Thor `/home/nvidia/r8ns/r9e2/runs/E2WISE_PLAN0/stageA/`, sha256-equal. Rasters, perc_det.pkl and
  the interpolated .pt files are not banked (re-creatable: weight_interp.py on Thor's runs/r9e2-*/model_ema_tail5k6k.pt, md5 above). Stage B read files are
  byte-exact copies of the dev-box S6c driver's pull (sha256-verified against Thor `runs/E2WISE_PLAN0/stageB/`).

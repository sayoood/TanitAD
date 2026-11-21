# E2-WISE PLAN: the E2 rule on PLANNER-interpolated weights, K2DC vs K0 -- verdict NO-GO

Pre-registered before any E2-WISE number: `../../PREREG_E2WISE_2026-11-17.md` (landed ad9f1cbe) + `../../PREREG_E2WISE_2026-11-17_ADDENDUM1.md`
(landed 5f647742). Weights: `stack/scripts/weight_interp.py` (landed 20ab9f54): theta = theta_init + 0.5 (theta_tail - theta_init) on S_plan, every other tensor copied from `model_ema_tail5k6k.pt`; init `/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt`. Both arms get the same
S and alpha. Zero training. DEV = eval139 even half, 70 episodes. One training run per arm (`H-ESTIM-SEED-1`): a separated interval
answers another draw of EPISODES only. Interpolated file md5: K2DC dc8377e032b0e1226cd1915b6b38024e, K0 7f2b42054c79476bfa47e8ae90e8aa65 (every dump.json
records it, `stageA/dump_headers/`).

## Stage A -- clause (a), map primary `nf_swedge_prec_lv_5_20` at each arm's OWN FIT thresholds (T0, MEASURED)

| arm | E2-TAIL | this read |
|---|---|---|
| K2DC | 0.4850 | **0.4850** [0.3996, 0.5632] |
| K0 | 0.4665 | **0.4665** [0.3852, 0.5375] |

* Perception check (ADDENDUM 1): map points equal E2-TAIL to 1e-9: YES; clause (b) equals E2-TAIL to 1e-6: YES (K2DC 0.307623 vs 0.307623, K0 0.307665 vs 0.307665). WISE-PLAN copies `core.encoder`, `_perception` and `core.agent_head` bit-exactly, so it cannot GO while (b) reproduces the tail tie; (d) decides the lever.

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

* (d) seed 0: separated WORSE: LONGITUDINAL/LON_dk_lead_progress_shortfall, LATERAL/LAT_yaw_rate_mae_radps_valid, LATERAL/LAT_cross_abs_final_m, TACTICAL/TAC_traj_lat_correct
* (d) seed 1: separated WORSE: LATERAL/LAT_curvature_mae_1pm_valid, LATERAL/LAT_yaw_rate_mae_radps_valid, LATERAL/LAT_cross_mae_m, LATERAL/LAT_cross_abs_final_m, TACTICAL/TAC_traj_lat_correct

## (d), row by row: E2-TAIL vs this read (paired K2DC - K0, episode-cluster bootstrap, T1, 70 episodes)

Every row separated worse at either seed in either read. WORSE = separated in the bad direction.

| family / row | tail s0 | plan s0 | tail s1 | plan s1 |
|---|---|---|---|---|
| LONGITUDINAL / `LON_dk_min_ttc_s` | -0.3953 [-0.9620, -0.0229] **WORSE** | +0.0706 [-0.2523, +0.3498] | -0.3532 [-0.8438, -0.0070] **WORSE** | +0.0026 [-0.3085, +0.2923] |
| LONGITUDINAL / `LON_dk_lead_progress_shortfall` | -0.0022 [-0.0069, +0.0019] | +0.0033 [+0.0000, +0.0070] **WORSE** | -0.0004 [-0.0043, +0.0038] | +0.0029 [-0.0006, +0.0068] |
| LATERAL / `LAT_heading_mae_deg` | +0.1096 [+0.0442, +0.1813] **WORSE** | +0.0505 [-0.0217, +0.1236] | +0.0804 [+0.0175, +0.1515] **WORSE** | +0.0606 [-0.0017, +0.1282] |
| LATERAL / `LAT_curvature_mae_1pm_valid` | +0.0005 [-0.0001, +0.0010] | +0.0004 [-0.0000, +0.0008] | +0.0004 [-0.0001, +0.0009] | +0.0004 [+0.0001, +0.0008] **WORSE** |
| LATERAL / `LAT_yaw_rate_mae_radps_valid` | +0.0022 [+0.0009, +0.0036] **WORSE** | +0.0012 [+0.0001, +0.0024] **WORSE** | +0.0016 [+0.0004, +0.0030] **WORSE** | +0.0014 [+0.0004, +0.0027] **WORSE** |
| LATERAL / `LAT_cross_mae_m` | +0.0126 [+0.0063, +0.0194] **WORSE** | +0.0061 [-0.0002, +0.0124] | +0.0091 [+0.0024, +0.0166] **WORSE** | +0.0065 [+0.0011, +0.0122] **WORSE** |
| LATERAL / `LAT_cross_abs_final_m` | +0.0288 [+0.0151, +0.0435] **WORSE** | +0.0137 [+0.0005, +0.0269] **WORSE** | +0.0203 [+0.0053, +0.0370] **WORSE** | +0.0152 [+0.0037, +0.0274] **WORSE** |
| TACTICAL / `TAC_traj_lat_correct` | -0.0071 [-0.0146, -0.0004] **WORSE** | -0.0088 [-0.0162, -0.0021] **WORSE** | -0.0071 [-0.0147, +0.0000] | -0.0088 [-0.0159, -0.0021] **WORSE** |

## Secondary readout: K2DC seed-0 `LAT_cross_mae_m` (decides nothing alone)

* K2DC 0.1106 m (tail 0.1160), K0 0.1045 m (tail 0.1034); gap +0.0061 (tail +0.0126). Both arms above their tail values: NO.
* STRATEGIC: the read prints PAUSED (its PI item-25 text). The PI ended that pause 2026-11-16; in these arms the strategic modules are
  equal to the init (S_plan excludes them), so there is no trained strategic row to compare.

## Next (Rule Zero)

* committed outcome 2 (WISE-PLAN fails (d)): planner drift alone does not carry the lateral deficit. Next: WISE-ENC (S_enc = core.encoder, alpha 0.5), started on Thor 2026-11-17 11:29:32Z from the families rows; this verdict file confirms the (d) FAIL it was started on. Other clauses not PASS: (b) FAIL (reported as written).
* E2, E2-RAW and E2-TAIL stand as written.

## Provenance

* Stage A files pulled from Thor `/home/nvidia/r8ns/r9e2/runs/E2WISE_PLAN/stageA/`, sha256-equal. Rasters, perc_det.pkl and
  the interpolated .pt files are not banked (re-creatable: weight_interp.py on Thor's runs/r9e2-*/model_ema_tail5k6k.pt, md5 above). Stage B read files are
  byte-exact copies of the dev-box S6c driver's pull (sha256-verified against Thor `runs/E2WISE_PLAN/stageB/`).

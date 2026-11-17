# E2 verdict: NO-GO  (K2 vs K0)

GO iff K2 beats K0 in point estimate on the map primary, box AP@2 m and the planner's S6c sidewalk rate, no four-family row is separated worse, and a partial W4 preview (K2 vs camera-blind) is separated -- at inference seeds 0 and 1 (REFCV9_DESIGN 4.4).

| clause | status | tier | seeds not PASS |
|---|---|---|---|
| (a) MAP primary nf_swedge_prec_lv_5_20: K2 > K0 (point) | **PASS** | T0 (label level) | - |
| (b) box AP@2 m head output no NMS: K2 > K0 (point) | **FAIL** | T0 (label level) | 0,1 |
| (c) planner sidewalk rate S6c-DEV: K2 < K0 (point) | **PASS** | T1 self-action open loop (S6c-DEV) | - |
| (d) no four-family row separated worse (paired, episodes) | **FAIL** | T1 self-action open loop | 0,1 |
| (e) W4 preview: K2 separated BETTER than camera-blind on ade_m + each family primary | **PASS** | T1 self-action open loop | - |

NO-GO names clause (b) box AP@2 m head output no NMS: K2 > K0 (point) [FAIL, seeds ['0', '1']] -- next lever: box lever: B2 perspective aux / B3 quality-aware presence + DN / B4 queries (REFCV9_DESIGN 4.3), re-read AP@2 m

NO-GO names clause (d) no four-family row separated worse (paired, episodes) [FAIL, seeds ['0', '1']] -- next lever: the separated-worse row(s) named in the clause: diagnose that family before the 214 h are spent

## inference seed 0
* (a) PASS: nearfield.ci.nf_swedge_prec_lv_5_20.point: K2=0.484951 K0=0.466485 delta=+0.018466 (higher is better)
* (b) FAIL: boxes.ap2m_all.point: K2=0.307623 K0=0.307665 delta=-4.15437e-05 (higher is better)
* (c) PASS: S6c-DEV sidewalk (off_drivable) rate: K2=0.176543 K0=0.191358 delta=-0.0148148 (lower is better)
* (d) FAIL: separated WORSE: LONGITUDINAL/LON_dk_min_ttc_s, LATERAL/LAT_heading_mae_deg, LATERAL/LAT_yaw_rate_mae_radps_valid, LATERAL/LAT_cross_mae_m, LATERAL/LAT_cross_abs_final_m, TACTICAL/TAC_traj_lat_correct
* (e) PASS: K2 separated better than camera-blind on ade_m + LONGITUDINAL + LATERAL + TACTICAL primary rows
  * family LONGITUDINAL: FAIL
  * family LATERAL: FAIL
  * family TACTICAL: FAIL
  * family STRATEGIC: PAUSED -- PAUSED (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25): the strategic layer is switched off until tactical and operative drive with high quality; reported PAUSED -- never a pass, never omitted, never extended to the other families.

## inference seed 1
* (a) PASS: nearfield.ci.nf_swedge_prec_lv_5_20.point: K2=0.484951 K0=0.466485 delta=+0.018466 (higher is better)
* (b) FAIL: boxes.ap2m_all.point: K2=0.307623 K0=0.307665 delta=-4.15437e-05 (higher is better)
* (c) PASS: S6c-DEV sidewalk (off_drivable) rate: K2=0.185185 K0=0.196296 delta=-0.0111111 (lower is better)
* (d) FAIL: separated WORSE: LONGITUDINAL/LON_dk_min_ttc_s, LATERAL/LAT_heading_mae_deg, LATERAL/LAT_yaw_rate_mae_radps_valid, LATERAL/LAT_cross_mae_m, LATERAL/LAT_cross_abs_final_m
* (e) PASS: K2 separated better than camera-blind on ade_m + LONGITUDINAL + LATERAL + TACTICAL primary rows
  * family LONGITUDINAL: FAIL
  * family LATERAL: FAIL
  * family TACTICAL: PASS
  * family STRATEGIC: PAUSED -- PAUSED (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25): the strategic layer is switched off until tactical and operative drive with high quality; reported PAUSED -- never a pass, never omitted, never extended to the other families.

STRATEGIC: PAUSED (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25): the strategic layer is switched off until tactical and operative drive with high quality; reported PAUSED -- never a pass, never omitted, never extended to the other families.
Windows: DEV (eval139 even half, DEV_HOLD_SPLIT.json); HOLD is never read here
Caveat: H-ESTIM-SEED-1: the paired interval answers another draw of EPISODES, not another TRAINING run; K2 and K0 are ONE training run each.

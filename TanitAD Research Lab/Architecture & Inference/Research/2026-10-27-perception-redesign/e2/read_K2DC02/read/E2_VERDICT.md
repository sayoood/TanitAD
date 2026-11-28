# E2 verdict: NO-GO  (K2 vs K0)

GO iff K2 beats K0 in point estimate on the map primary and the planner's S6c sidewalk rate, K2 is not separated worse than K0 on box AP@2 m, no four-family row is separated worse, and a partial W4 preview (K2 vs camera-blind) is separated -- at inference seeds 0 and 1 (REFCV9_DESIGN 4.4) (clause (b) amended by the PI 2026-11-17: not separated worse).

| clause | status | tier | seeds not PASS |
|---|---|---|---|
| (a) MAP primary nf_swedge_prec_lv_5_20: K2 > K0 (point) | **FAIL** | T0 (label level) | 0,1 |
| (b) box AP@2 m head output no NMS: K2 not separated worse than K0 (paired, episodes) | **PASS** | T0 (label level) | - |
| (c) planner sidewalk rate S6c-DEV: K2 < K0 (point) | **FAIL** | T1 self-action open loop (S6c-DEV) | 0,1 |
| (d) no four-family row separated worse (paired, episodes) | **FAIL** | T1 self-action open loop | 0,1 |
| (e) W4 preview: K2 separated BETTER than camera-blind on ade_m + each family primary | **PASS** | T1 self-action open loop | - |

NO-GO names clause (a) MAP primary nf_swedge_prec_lv_5_20: K2 > K0 (point) [FAIL, seeds ['0', '1']] -- next lever: map primary lever: M1/M3 representation + w_map share (REFCV9_DESIGN 4.3), re-read nf_swedge_prec_lv_5_20

NO-GO names clause (c) planner sidewalk rate S6c-DEV: K2 < K0 (point) [FAIL, seeds ['0', '1']] -- next lever: planner wiring: W-MAP-2 projection / W-BOX tokens + gates (C8), re-read S6c sidewalk rate

NO-GO names clause (d) no four-family row separated worse (paired, episodes) [FAIL, seeds ['0', '1']] -- next lever: the separated-worse row(s) named in the clause: diagnose that family before the 214 h are spent

## inference seed 0
* (a) FAIL: nearfield.ci.nf_swedge_prec_lv_5_20.point: K2=0.406632 K0=0.750485 delta=-0.343853 (higher is better)
* (b) PASS: box AP@2 m K2 - K0: delta=+0.007 paired CI [0.0015, 0.0127] -> not separated worse (interval reaches 0 or above); K2 higher by point=True (information only)
* (c) FAIL: S6c-DEV sidewalk (off_drivable) rate: K2=0.188889 K0=0.188272 delta=+0.000617284 (lower is better)
* (d) FAIL: separated WORSE: LONGITUDINAL/LON_accel_mae_mps2, LATERAL/LAT_curvature_mae_1pm_valid
* (e) PASS: K2 separated better than camera-blind on ade_m + LONGITUDINAL + LATERAL + TACTICAL primary rows
  * family LONGITUDINAL: FAIL
  * family LATERAL: FAIL
  * family TACTICAL: PASS
  * family STRATEGIC: PAUSED -- PAUSED (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25): the strategic layer is switched off until tactical and operative drive with high quality; reported PAUSED -- never a pass, never omitted, never extended to the other families.

## inference seed 1
* (a) FAIL: nearfield.ci.nf_swedge_prec_lv_5_20.point: K2=0.406632 K0=0.750485 delta=-0.343853 (higher is better)
* (b) PASS: box AP@2 m K2 - K0: delta=+0.007 paired CI [0.0015, 0.0127] -> not separated worse (interval reaches 0 or above); K2 higher by point=True (information only)
* (c) FAIL: S6c-DEV sidewalk (off_drivable) rate: K2=0.190741 K0=0.185185 delta=+0.00555556 (lower is better)
* (d) FAIL: separated WORSE: LONGITUDINAL/LON_accel_mae_mps2
* (e) PASS: K2 separated better than camera-blind on ade_m + LONGITUDINAL + LATERAL + TACTICAL primary rows
  * family LONGITUDINAL: FAIL
  * family LATERAL: PASS
  * family TACTICAL: PASS
  * family STRATEGIC: PAUSED -- PAUSED (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25): the strategic layer is switched off until tactical and operative drive with high quality; reported PAUSED -- never a pass, never omitted, never extended to the other families.

STRATEGIC: PAUSED (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25): the strategic layer is switched off until tactical and operative drive with high quality; reported PAUSED -- never a pass, never omitted, never extended to the other families.
Windows: DEV (eval139 even half, DEV_HOLD_SPLIT.json); HOLD is never read here
Caveat: H-ESTIM-SEED-1: the paired interval answers another draw of EPISODES, not another TRAINING run; K2 and K0 are ONE training run each.

# refcv9 E2 read: K2M1N20 vs K0 — verdict NO-GO (2026-11-12)

Banked by the Master Mind (session 4a3d05). Thor run dir: `/home/nvidia/r8ns/r9e2/runs/E2_READ_K2M1N20/`
(read tree `read_tree_cf653ca4bb47_lfix_e77c39d4_g55f14ef2`). Windows: **DEV** (eval139 even half, `DEV_HOLD_SPLIT.json`,
70 clips / 2,308 map windows); HOLD is never read here. Pre-registered rule: REFCV9_DESIGN 4.4.

## Verdict (`read/E2_VERDICT.{json,md}`, written by the read, unedited)

| clause | status | tier | evidence |
|---|---|---|---|
| (a) map primary `nf_swedge_prec_lv_5_20`, K2 > K0 (point) | **FAIL** | T0 | K2 0.4137 vs K0 0.7505 (default cut) |
| (b) box AP@2 m, no NMS, K2 > K0 (point) | PASS | T0 | 0.3281 vs 0.3195 |
| (c) planner S6c-DEV sidewalk rate, K2 < K0 (point) | **FAIL** | T1 | s0 0.1963 vs 0.1883; s1 0.1938 vs 0.1852 |
| (d) no four-family row separated worse | **FAIL** | T1 | below |
| (e) W4 preview, K2 separated better than camera-blind | PASS | T1 | LON, LAT and TAC primaries + ade_m |

(d), paired episode-cluster bootstrap over 70 DEV episodes (K2 − K0, lower is better, seed 0 inference):
LON_accel_mae +0.2445 m/s² [0.1019, 0.3944]; LON_dk_lead_progress_shortfall +0.0061 [0.0010, 0.0122] (37 episodes);
LAT_heading_mae +0.0571° [0.0183, 0.0974]; LAT_yaw_rate_mae (valid) +0.0042 rad/s [0.0006, 0.0078];
LAT_cross_mae +7.4 mm [3.7, 11.1]; LAT_cross_abs_final +16.6 mm [8.7, 24.9]. Seed 1 keeps accel, heading, cross and
cross_final separated worse. TACTICAL: no separated row. **STRATEGIC: PAUSED** (PI 2026-09-19, ITEM 25).

Caveats: K2 and K0 are **one training run each** (H-ESTIM-SEED-1). The intervals answer "another draw of episodes",
not "another training run". Both inference seeds were read.

## Diagnostic: clause (a) under the design's own decode protocol (`diag_fitthr/`), NOT a verdict change

The E2 read decoded both map heads at the fixed default cut. REFCV9_DESIGN 2.1's decision rule is per-arm thresholds
fitted on a FIT split of TRAIN clips, never on eval (E1's `pswpedgeF`). `diag_fitthr/e2_fitthr_diag.sh` re-reads the
**same E2 DEV dumps** with each arm's own FIT thresholds. It follows E1's recipe: a FIT dump of E1's 1,112 TRAIN-DIAG
windows, `r9_map_fit_thresholds.py fit` (psw, then psw+pedge f_tol, with the A1 check that the psw levels are EQUAL),
then `r9_e1_eval.py --map-thresholds`. Estimator: episode-cluster bootstrap B=2000 seed 0, unpaired.

| arm | protocol | lv_5_20 (primary) | 0–5 m | 5–10 m | 10–20 m | 20–40 m |
|---|---|---|---|---|---|---|
| K2 | default cut | 0.4137 [0.337, 0.486] | 0.3741 | 0.3669 | 0.4346 | 0.5108 |
| K2 | fitted (psw 128/126/128, pedge 98/134/139) | **0.4008 [0.326, 0.472]** | 0.3086 | 0.3237 | 0.4412 | 0.5188 |
| K0 | default cut | 0.7505 [0.643, 0.833] | 0.2464 | 0.7388 | 0.7541 | 0.7647 |
| K0 | fitted (psw 88/90/79, pedge 7/7/5) | **0.4776 [0.395, 0.549]** | 0.1664 | 0.4403 | 0.4926 | 0.5607 |

Reading: under the design's own protocol the gap shrinks from −0.337 to **−0.077**, and K2 is still below K0 on
the point estimate. Clause (a) would still fail. K0's default-cut 0.75 came mostly from its 8-way head at the
default cut (E1 measured the same direction: C0 0.662 psw → 0.491 pswpedgeF). K2 is ahead on 0–5 m (0.309 vs 0.166)
and on `nf_prec_0_10` (0.531 vs 0.244). (c) and (d) do not depend on the map decode, so **E2 stays NO-GO** whatever
(a) reads.

⚠️ Protocol defect to fix before any re-read: the E2 read rule must decode the map with the per-arm FIT thresholds
(REFCV9_DESIGN 2.1). The verdict above stands as written, because it was the pre-registered read.

## Next levers (Rule Zero)

1. (d) LON_accel_mae is the largest separated row: attribute K2 vs K0 by lever (REFCV9_DESIGN 5.4, the trace-fixed
   K-PLAN-ID method), CPU-only on the existing dumps.
2. (c) planner wiring: the W-MAP-2 projection and W-BOX token gates (C8).
3. (a) the map lever M3N20L64 (PI-approved ~13 Thor-h), or K0's head. Any new E2-scale training (6,000 steps,
   ~17 Thor-h per arm) is outside the approved envelope and goes to the PI with this diagnosis.

## Files

`read/` holds the verdict, inputs, the two arms' T0 evals, families, distance-keeping, camera-blind and S6c per seed.
`logs/` holds the roll status, the 15:10Z distkeep abort marker (fixed by the pose-sync lead block, landed 2ab7f995),
and the S6c resume log. `diag_fitthr/` holds the diagnostic script, its status, the fitted
thresholds, the remap and the two fitted evals. `SHA256SUMS` covers every file.

# STRATEGIC family on the banked ACTSENS windows: a demonstration of the instrument

Tier: T0 HWM-ORACLE diagnostic: the goal that conditioned each plan is the encoded true future (never a driving number); demonstration of the instrument, not a decision.
Evidence: MEASURED on the banked ACTSENS read (F:/TanitAD-scratch/actsens/read/out/<arm>/win), navtok_table.npz sha256 prefix 2836369e, tip hwm_eval build in strategic/build/src.

Windows 994, episodes 71; nav-text rows attached at table step now + 0 (MEASURED, STRATEGIC_TEXT_OFFSET_CHECK.json): text_ok 910, flags {'0': 910, '3': 84}.
COMPLETE_M 20 m, theta 25 deg (SLIGHT 10 deg), false-turn margin 30 m: all HYPOTHESIS, to be pre-registered before any decisive read.

Estimator: paired episode-cluster bootstrap (taniteval/ci.py), B = 2000, seed 0. It answers: would another draw of EPISODES say this? (models held fixed). It does NOT answer: would another TRAINING run say this (H-ESTIM-SEED-1: a lever claim needs a replicate arm), and would another INFERENCE seed say this (the oracle planner samples; only inference seed 0 is read here).

Control, straight plan (zero action): S_ADH 0.000 (n 11, ep 5), reads exactly 0.0 on every scorable window: True; max |signed degrees| 0.000.
HOLD baseline (the NOW action repeated, yaw rate included, so it turns where the ego is turning): S_ADH 0.583 (n 12, ep 6). It is a baseline, not a zero control.
Logged path through the same cells: S_ADH ceiling 1.000 (n 10, ep 4); S_FALSE logged-path rate 0.131 (n 854, ep 69). S_FALSE is read only as plan minus logged, never as an absolute.

## Per arm (inference seed 0, HWM-ORACLE and FLAT-ORACLE plans)

| arm | planner | S_ADH rate | S_ADH minus logged | S_ADH minus NULL | S_ADH_DEG mean deg | S_FALSE rate | S_FALSE minus logged | S_FALSE minus NULL |
|---|---|---|---|---|---|---|---|---|
| C7_R35_s0 | hwm_oracle | 0.000 (n 12, ep 4) | -1.000 [-1.000, -1.000] n 7, ep 3, worse | -0.125 [-0.375, +0.000] n 8, ep 4, none | -4.2 | 0.056 (n 857, ep 70) | -0.075 [-0.129, -0.026] n 853, ep 69, lower | -0.013 [-0.038, +0.012] n 854, ep 70, none |
| C7_R35_s0 | flat_oracle | 0.143 (n 14, ep 5) | -1.000 [-1.000, -1.000] n 9, ep 4, worse | +0.000 [+0.000, +0.000] n 5, ep 3, none | +4.7 | 0.306 (n 850, ep 69) | +0.173 [+0.101, +0.249] n 849, ep 68, higher | +0.058 [-0.019, +0.139] n 848, ep 68, none |
| IDM_s0 | hwm_oracle | 0.000 (n 13, ep 4) | -1.000 [-1.000, -1.000] n 7, ep 3, worse | -0.125 [-0.375, +0.000] n 8, ep 4, none | -5.1 | 0.031 (n 859, ep 70) | -0.101 [-0.158, -0.052] n 853, ep 69, lower | -0.037 [-0.061, -0.015] n 855, ep 70, lower |
| IDM_s0 | flat_oracle | 0.062 (n 16, ep 6) | -0.857 [-1.000, -0.250] n 7, ep 4, worse | +0.000 [+0.000, +0.000] n 5, ep 3, none | -13.3 | 0.379 (n 855, ep 69) | +0.248 [+0.162, +0.334] n 851, ep 68, higher | +0.134 [+0.049, +0.219] n 852, ep 69, higher |
| DIV_s0 | hwm_oracle | 0.111 (n 9, ep 3) | -0.800 [-1.000, +0.000] n 5, ep 2, none | +0.000 [+0.000, +0.000] n 7, ep 3, none | -1.2 | 0.054 (n 855, ep 70) | -0.076 [-0.133, -0.029] n 851, ep 69, lower | -0.015 [-0.039, +0.008] n 852, ep 70, none |
| DIV_s0 | flat_oracle | 0.200 (n 10, ep 3) | -0.800 [-1.000, +0.000] n 5, ep 2, none | +0.250 [+0.000, +1.000] n 4, ep 2, none | -2.8 | 0.285 (n 848, ep 70) | +0.154 [+0.088, +0.217] n 846, ep 68, higher | +0.038 [-0.033, +0.102] n 847, ep 69, none |
| NULL_s2 | hwm_oracle | 0.125 (n 8, ep 4) | -0.800 [-1.000, +0.000] n 5, ep 3, none | reference | +9.2 | 0.069 (n 858, ep 70) | -0.064 [-0.115, -0.021] n 851, ep 69, lower | reference |
| NULL_s2 | flat_oracle | 0.125 (n 8, ep 5) | -1.000 [-1.000, -1.000] n 4, ep 2, worse | reference | +7.1 | 0.250 (n 857, ep 69) | +0.115 [+0.051, +0.181] n 850, ep 68, higher | reference |
| IDM_s1 | hwm_oracle | 0.000 (n 11, ep 5) | -1.000 [-1.000, -1.000] n 6, ep 4, worse | -0.333 [-0.750, +0.000] n 9, ep 5, none | +3.4 | 0.048 (n 855, ep 69) | -0.083 [-0.140, -0.031] n 851, ep 68, lower | -0.005 [-0.029, +0.018] n 853, ep 68, none |
| IDM_s1 | flat_oracle | 0.077 (n 13, ep 5) | -0.833 [-1.000, -0.750] n 6, ep 3, worse | +0.000 [-1.000, +0.250] n 7, ep 3, none | -17.1 | 0.303 (n 844, ep 69) | +0.171 [+0.096, +0.246] n 843, ep 69, higher | +0.011 [-0.073, +0.092] n 841, ep 69, none |
| DIV_s1 | hwm_oracle | 0.154 (n 13, ep 4) | -0.714 [-1.000, +0.000] n 7, ep 3, none | -0.100 [-0.750, +0.667] n 10, ep 4, none | +5.7 | 0.065 (n 857, ep 69) | -0.066 [-0.125, -0.009] n 849, ep 68, lower | +0.013 [-0.014, +0.041] n 852, ep 69, none |
| DIV_s1 | flat_oracle | 0.727 (n 11, ep 4) | -1.000 [-1.000, -1.000] n 2, ep 2, worse | +0.250 [-1.000, +1.000] n 4, ep 3, none | +29.1 | 0.294 (n 856, ep 70) | +0.163 [+0.095, +0.234] n 852, ep 69, higher | +0.001 [-0.076, +0.077] n 852, ep 70, none |
| NULL_s3 | hwm_oracle | 0.250 (n 12, ep 5) | -0.857 [-1.000, -0.400] n 7, ep 4, worse | reference | +3.7 | 0.053 (n 856, ep 70) | -0.079 [-0.129, -0.032] n 851, ep 69, lower | reference |
| NULL_s3 | flat_oracle | 0.125 (n 8, ep 4) | -0.833 [-1.000, +0.000] n 6, ep 3, none | reference | +0.5 | 0.294 (n 855, ep 70) | +0.166 [+0.096, +0.241] n 851, ep 69, higher | reference |

NULL is the chassis-matched NULL arm (NULL_s2 for chassis 0, NULL_s3 for chassis 1). A "none" label means the interval contains 0; a separated interval here is necessary and not sufficient for any claim about a lever (H-ESTIM-SEED-1).

## Sensitivity of the S_ADH cell to COMPLETE_M and to the table step (HWM-ORACLE plan; each cell n windows / episodes)

| setting | C7_R35_s0 | IDM_s0 | DIV_s0 | NULL_s2 | IDM_s1 | DIV_s1 | NULL_s3 |
|---|---|---|---|---|---|---|---|
| COMPLETE_M=10, table_step now+0 (verified) | 0.000 (17/5; ceiling 1.00) | 0.000 (16/5; ceiling 1.00) | 0.188 (16/5; ceiling 1.00) | 0.067 (15/6; ceiling 1.00) | 0.053 (19/5; ceiling 1.00) | 0.000 (20/7; ceiling 1.00) | 0.067 (15/5; ceiling 1.00) |
| COMPLETE_M=20, table_step now+0 (verified) | 0.000 (12/4; ceiling 1.00) | 0.000 (13/4; ceiling 1.00) | 0.111 (9/3; ceiling 1.00) | 0.125 (8/4; ceiling 1.00) | 0.000 (11/5; ceiling 1.00) | 0.154 (13/4; ceiling 1.00) | 0.250 (12/5; ceiling 1.00) |
| COMPLETE_M=30, table_step now+0 (verified) | 0.000 (8/4; ceiling 1.00) | 0.000 (8/4; ceiling 1.00) | 0.000 (3/2; ceiling 1.00) | 0.200 (5/3; ceiling 1.00) | 0.167 (6/4; ceiling 1.00) | 0.250 (8/4; ceiling 1.00) | 0.000 (5/3; ceiling 1.00) |
| COMPLETE_M=10, table_step now+3 (audit guess, not verified) | 0.095 (21/6; ceiling 0.95) | 0.045 (22/6; ceiling 0.95) | 0.133 (15/5; ceiling 0.95) | 0.053 (19/5; ceiling 0.95) | 0.042 (24/6; ceiling 0.95) | 0.000 (23/6; ceiling 0.95) | 0.000 (17/5; ceiling 0.95) |
| COMPLETE_M=20, table_step now+3 (audit guess, not verified) | 0.000 (11/4; ceiling 1.00) | 0.000 (14/4; ceiling 1.00) | 0.100 (10/3; ceiling 1.00) | 0.111 (9/4; ceiling 1.00) | 0.000 (14/4; ceiling 1.00) | 0.143 (14/5; ceiling 1.00) | 0.231 (13/4; ceiling 1.00) |
| COMPLETE_M=30, table_step now+3 (audit guess, not verified) | 0.000 (9/4; ceiling 1.00) | 0.000 (8/3; ceiling 1.00) | 0.000 (5/3; ceiling 1.00) | 0.200 (5/3; ceiling 1.00) | 0.143 (7/3; ceiling 1.00) | 0.273 (11/4; ceiling 1.00) | 0.286 (7/4; ceiling 1.00) |

COMPLETE_M 10, 20 and 30 are HYPOTHESIS values; the rows show how far the cell moves with the allowance and with the table step (the now + 3 row is the audit's unverified guess, kept to show its effect).

## Not computed (n/a entries carried by the report)

* STRATEGIC.intent_6_30s: n = 0: no arm outputs an intent or goal at 6-30 s (the HWM plan is 6 s; v8f-mini's S emits route checkpoints out to 80 m only). Would-have n, MEASURED count on the 994 ACTSENS windows by time to the announced manoeuvre d / v0 (strategic-audit count_n.py): 6-12 s 32 windows, 12-30 s 69, over 30 s 20. Unblocks with a PI go on a strategic head and training
* STRATEGIC.derived_goals: n = 0: no derived-goal output exists on the HWM planner. Would-have n: n/a (no output or label is defined; STRATEGIC_PLAN section 2)
* STRATEGIC.detected_info: n = 0: no detected-information output exists on the HWM planner (the HWM ctx token is an input, not scored). Would-have n: n/a (STRATEGIC_PLAN section 2)
* STRATEGIC.S_CKPT: structural n = 0: the HWM planner has no S head, so no route checkpoint exists to score. S_CKPT is read on v8f-mini arms only; would-have n on the eval141 grid, MEASURED (proto_eval141_counts.json): 46 / 120 / 214 windows (18 / 22 / 24 episodes) at 30 / 50 / 80 m of arc

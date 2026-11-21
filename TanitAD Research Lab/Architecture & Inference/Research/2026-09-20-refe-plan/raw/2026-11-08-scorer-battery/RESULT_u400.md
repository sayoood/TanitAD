# RESULT_u400: scorer battery, selection tier, u400 on the C3 table (2026-11-08)

Tier: selection tier (lookup of true per-proposal scores; T0/T1 style), CPU. Table: C3 `table.npz`, 1,058 navtest tokens (529 zeroed, 529 random; 123 logs; 154 CRUISE). Source of every number: `battery_u400.json` (sha in CODE_PIN.json), produced by `scorer_battery.py`.
Estimator: paired log-cluster bootstrap, B = 10,000, seed 20260927, 95 % percentile, shared draws for every arm and pair. **Question every interval answers: "would another draw of LOGS say this?"** It does NOT answer "would another training run say this" (H-ESTIM-SEED-1), and the planner is deterministic (no inference variance). Weighted = zeroed 529/12146 + random 11617/12146 (selc prereg weights).

## Positive control and gates (all green, 19 tests)
- u400 reproduces `u400_reference.json`: R1 weighted CRUISE slow share 0.4124 [0.2747, 0.5690] exactly (point, lo, hi, per-stratum shares, n); R2 tie-estimator Spearman for NC, DAC, EP, TTC exactly (pred, true, gap, n); R3 no-lane-factor re-pick (423 picks change, 61 on CRUISE) to 1e-15 (the reference script groups `W*sum/max` where the battery groups `W*(sum/max)`: 1 ulp). Banked picks reproduced 1,058/1,058.
- ⛔ **Finding: the legacy argsort-of-argsort estimator is numpy-version dependent.** Same code, same data: NC "true" is -0.0393 on numpy 2.5.1 (= banked `p1_diag_all.json`) and +0.0336 on numpy 1.26.4 (= the value in `u400_reference.json`). Average ranks give -0.0945 on both. This is the mechanism behind the sign flip in RETRACTION_LOG 2026-11-07. The battery's default is average ranks; `legacy` exists only to reproduce.

## Regression arms read as known
| arm | regret W | PDMS@pick W | pairwise acc (W) | CRUISE slow share W | overspeed W | v4 bias m/s |
|---|---|---|---|---|---|---|
| **u400** | 0.099 [0.091, 0.108] | 0.883 [0.874, 0.891] | 0.870 [0.845, 0.893] | 0.412 [0.275, 0.569] | 0.181 [0.073, 0.294] | -0.731 [-1.441, -0.097] |
| u400 without lane factor | 0.096 [0.087, 0.106] | 0.885 [0.875, 0.895] | in json | 0.366 [0.229, 0.521] | 0.274 [0.131, 0.422] | -0.408 [-1.145, 0.252] |
| always-slowest | 0.248 [0.224, 0.271] | 0.734 | 0.512 (unweighted) | 0.889 [0.807, 0.963] | 0.000 | -3.806 |
| always-fastest | 0.400 [0.363, 0.437] | 0.582 | 0.488 (unweighted) | 0.063 [0.013, 0.136] | 0.778 | +2.448 |
| random pick (seed 0) | 0.264 [0.231, 0.297] | 0.718 | 0.508 [0.492, 0.524] | 0.187 | 0.493 | +0.634 |
| oracle | **0.000 exactly** | 0.982 | 1.000 | 0.095 | 0.666 | +1.708 |

Paired vs u400 (arm minus u400, weighted): regret random +0.165 [0.131, 0.198], slowest +0.149 [0.123, 0.173], fastest +0.301 [0.264, 0.338], oracle -0.099 [-0.108, -0.091]; no-lane-factor regret -0.003 [-0.009, +0.004] (not separated), CRUISE slow share -0.046 [-0.104, -0.000].

## Selection-tier detail for u400
- **Regret is concentrated in the zeroed stratum:** zeroed 0.831 [0.776, 0.886] vs random 0.065 [0.058, 0.074] (PDMS units, 0 to 1). Pairwise ranking accuracy (>= 0.95 vs <= 0.50; 714 tokens with both groups, 248,930 pairs): zeroed 0.385 [0.306, 0.459], random 0.892 [0.867, 0.915]. Below 0.5 on the zeroed stratum.
- **Top-K oracle regret (W):** K=1 0.099, 2 0.077, 4 0.059, 8 0.043, 16 0.029, 32 0.014, 64 0.000.
- **Within-token Spearman vs v4/v0 on CRUISE** (average ranks; a token enters a head only if its true labels vary). Predicted minus true:

| head | pred | true | pred minus true | n paired (of 154) |
|---|---|---|---|---|
| TTC | -0.546 [-0.640, -0.449] | -0.094 [-0.181, -0.004] | **-0.451 [-0.565, -0.341]** | 52 |
| NC | -0.491 [-0.606, -0.389] | -0.094 [-0.178, -0.014] | -0.396 [-0.517, -0.285] | 38 |
| DDC | -0.505 [-0.594, -0.384] | -0.099 [-0.128, -0.068] | -0.407 [-0.504, -0.273] | 52 |
| DAC | -0.293 [-0.406, -0.213] | -0.036 [-0.073, -0.003] | -0.257 [-0.372, -0.178] | 119 |
| EP | +0.088 [-0.028, 0.182] | +0.439 [0.321, 0.540] | -0.351 [-0.484, -0.201] | 151 |
| C | n/a: true comfort label constant on all 154 CRUISE tokens | | | 0 |
| lane | pred -0.329 [-0.455, -0.224] vs speed; NO TRUTH in the table (LANE_TRUTH.md) | | | 154 predicted |

The safety heads (TTC, NC, DDC) rank slow proposals higher while the truth is nearly flat in speed: the P1 defect, here with tie-correct ranks.

## Four families (u400 picks)
- LONGITUDINAL: CRUISE slow share 0.412 [0.275, 0.569]; overspeed share (> 1.15) 0.181 [0.073, 0.294]; v4 plan minus GT -0.731 [-1.441, -0.097] m/s (n = 154); distance-keeping UNAVAILABLE at this tier (no lead track).
- LATERAL (all 1,058 tokens, chord headings): cross-track 0.329 [0.299, 0.361] m; heading error at t = 4 s 0.068 [0.058, 0.081] rad (n = 955 moving); yaw-rate error 0.265 [0.231, 0.307] rad/s and curvature error 0.166 [0.120, 0.229] 1/m (n = 1,047 with a moving step).
- TACTICAL: n/a at this tier (the table has no tactical labels or head).
- STRATEGIC: PAUSED (PI 2026-09-19, ITEM 25).

⚠ Column 2 of the REFe proposals is NOT heading (median 1.14 rad off the chord direction; the GT column is 0.008 rad), so lateral heading uses chord directions on both sides. The lateral numbers compare the pick with the human future; an oracle-PDMS pick is not lateral-optimal (oracle cross-track 0.633 m), so read them as descriptive, not as scorer quality.

## Caveats
Single checkpoint (u400). E1 is the set the defect was diagnosed on and is never decisive alone (PREREG_SCORER1 section 3). The random-pick arm is one seed (0). Per-stratum intervals are over logs within the stratum. Regression-arm bias/overspeed values are plan v4 minus GT v4 over CRUISE.

# CONFIRM_RESULT -- H-SEL-1 fresh-token confirmation of the FROZEN gBG(0.20, 0.05) (stream YSEL2)

**Committed verdict: ADOPT-for-sim** (mechanical, `PREREG_SEL_CONFIRM.md` sha256 `ad12c2d9204eb7be74c9b500e885d2cbf0394dc027e2a5d0459b8634f2781bab`). Written by `selc_analyze.py --verdict`; every number MEASURED, artifact `confirm_result.json`. **Tier T0.** Population: 5877 fresh navtest tokens / 75 logs (disjoint from the 1,058 C3 tokens, outside YSEL's 61 TRAIN logs, all non-zeroed), 591 CRUISE.

Estimator: paired log-cluster bootstrap over the fresh logs, 10,000 resamples, seed 20260927, 95 % percentile; answers 'another draw of logs' only (one checkpoint; deterministic planner, ego probe C1) -- says nothing about training or inference seed.
Mix: no fresh token is zeroed (all 529 are C3 tokens): mix = W_zero*0 (structural floor) + W_random*delta_random.

## The decision
| condition | bar | measured | holds |
|---|---|---|---|
| (i) PDMS non-inferior (zero-floor mix, x100) | lower CI > -0.5 | -0.223 [-0.450, -0.030] | YES |
| (ii) CRUISE slow-down share drop | <= -10 pts and CI-hi < 0 | -31.8 [-39.5, -24.4] pts | YES |
| harm (REFUTED branch) | point < -0.5 and hi < 0 | no | -- |
| slow-down did not fall (REFUTED branch) | point >= 0 | no | -- |

Changed picks 864 (better 687, worse 57 by true PDMS).

## Four families (banked -> rule; paired delta with CI)
| family | metric | banked | rule | paired delta |
|---|---|---|---|---|
| LONGITUDINAL | plan v4 - GT v4, CRUISE (m/s) | -1.432 [-1.992, -0.791] | -0.360 [-0.803, +0.114] | +1.072 [+0.868, +1.288] |
| | plan v4 - GT v4, ego gate set | -1.236 [-1.654, -0.764] | -0.123 [-0.455, +0.235] | +1.113 [+0.962, +1.274] |
| | slow-down share (v4/v0 < 0.85), CRUISE | +0.563 [+0.443, +0.678] | +0.245 [+0.167, +0.335] | -0.318 [-0.395, -0.244] |
| | overspeed share (v4/v0 > 1.15), CRUISE | +0.110 [+0.026, +0.240] | +0.112 [+0.028, +0.243] | +0.002 [+0.000, +0.006] |
| | mean plan v4/v0, CRUISE | +0.813 [+0.737, +0.896] | +0.928 [+0.865, +0.994] | -- |
| LATERAL (what the proposals allow) | ADE to human future, CRUISE (m; Amendment 1: 7 cruise / 17 tokens masked) | +1.631 [+1.431, +1.830] | -- | -0.513 [-0.665, -0.360] |
| | abs(y) at 4 s on changed picks (m) | -- | -- | +0.315 [+0.251, +0.383] |
| | DAC failure rate | +0.000 [+0.000, +0.000] | +0.004 [+0.002, +0.006] | +0.004 [+0.002, +0.006] |
| | DDC mean | +0.980 [+0.973, +0.987] | +0.980 [+0.972, +0.986] | -- |
| safety | NC fail rate | +0.001 [+0.000, +0.003] | +0.003 [+0.001, +0.005] | +0.002 [+0.001, +0.003] |
| | TTC fail rate | +0.034 [+0.026, +0.041] | +0.037 [+0.030, +0.044] | +0.003 [+0.001, +0.006] |
| PDMS | zero-floor mix x100 | +88.45 [+87.87, +89.06] | +88.23 [+87.62, +88.85] | -0.22 [-0.45, -0.03] |
| | unweighted x100 (random stratum) | +92.48 [+91.87, +93.11] | +92.25 [+91.61, +92.90] | -0.23 [-0.47, -0.03] |
| TACTICAL | -- | n/a: no manoeuvre-decision label on these tokens and REFe emits no tactical output here (n = 0) | | |
| STRATEGIC | -- | **PAUSED** (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25) | | |

## DESCRIPTIVE by log class (never the verdict; the leakage / shared-log check of PREREG section 3)
* heldout: 5580 tokens / 62 logs / 567 CRUISE: dPDMS mix -0.244 [-0.488, -0.042], d slow -31.2 [-39.1, -23.4] pts
* new: 297 tokens / 13 logs / 24 CRUISE: dPDMS mix +0.179 [-0.119, +0.392], d slow -45.8 [-73.9, -26.7] pts

The 64-column per-proposal table (secondary H-SEL-LANE-1 + descriptive rules) follows in `CONFIRM_TABLE_APPENDIX.md` when `markers/selc_full.DONE` exists.

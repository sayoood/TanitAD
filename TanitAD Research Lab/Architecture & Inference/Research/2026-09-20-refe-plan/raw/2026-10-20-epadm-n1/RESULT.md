# RESULT EPADM-N1 (Research Lab agent, 2026-10-20): the A2 statistic restricted to the CHANGED picks, re-simulated at the S0 own-fan geometry (MM ruling `MM_RULING_S0BANK.md` DQ3, arm N1)

**Escalation (integration + decision, headline):** the Master Mind lands `out/` (repo dir `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-20-epadm-n1/`). Outcome by the ruling's own table (applied mechanically): N1 misses at both geometries. Per DQ3: the frozen EP-ADMISSION gate cannot certify a search candidate at any available geometry; the MM then rules, in a SEPARATE sealed decision and BEFORE any navtest token is touched, what licenses C2 instead (that ruling may not use any B-REST outcome to choose its bar).

**Stamps:** every rate is SIMULATED on the real estimator (R1 simulator code, `SIM_DESIGN_N1.md` hashed BEFORE the first run), generator A primary, B-bootstrap 10,000, seed 20260927, NEW replicate salts 20261401 / 20261402; Base C = carve-B clones with real search outputs, Base X = real S0 labels with a synthetic candidate; neither is S0's own-fan search bank. Tier: SELECTION (label lookup, T0-class); one training seed (`H-ESTIM-SEED-1`: the interval answers "another draw of tokens and logs" only); STRATEGIC PAUSED. Nothing here is a navtest or driving claim.

## Plain statement (generated from `sim_results.json`)

* Controls: **ALL PASS**; unit suite `test_n1.py`: 39 passed in 3.11s; copied S0BANK suite: 22 passed in 0.17s; mutation arms: **32 of 32 caught**, unmutated harness PASS.
* **GS41**: cells meeting EVERY bar: **none**; geometry ALL-MET (MET on C and on X): **False**; per-bar status {"A1-r1 false refusal": "MISSED", "A2-N1 false refusal": "MET", "N1 DAC power x1.25": "MISSED"}; missed per cell {"C": ["A1_r1_false_refusal_wilson_upper_le_5pct", "DAC_power_x1.25_margin1.05_point_ge_70pct"], "X": ["DAC_power_x1.25_margin1.05_point_ge_70pct"]}.
* **GS117**: cells meeting EVERY bar: **none**; geometry ALL-MET (MET on C and on X): **False**; per-bar status {"A1-r1 false refusal": "MISSED", "A2-N1 false refusal": "MET", "N1 DAC power x1.25": "MISSED"}; missed per cell {"C": ["A1_r1_false_refusal_wilson_upper_le_5pct", "DAC_power_x1.25_margin1.05_point_ge_70pct"], "X": ["DAC_power_x1.25_margin1.05_point_ge_70pct"]}.
* GS41 Base C: A1-r1 FR 8.6 % [7.5, 10.0]; A2-N1 FR 3.5 % [2.7, 4.3]; N1 DAC power x1.25 9.3 % [8.2, 10.7] (bars: <= 5.0 % upper, <= 5.0 % upper, >= 70 % point).
* GS41 Base X: A1-r1 FR 2.0 % [1.5, 2.7]; A2-N1 FR 2.5 % [1.9, 3.2]; N1 DAC power x1.25 27.5 % [25.6, 29.5] (bars: <= 5.0 % upper, <= 5.0 % upper, >= 70 % point).
* GS117 Base C: A1-r1 FR 8.6 % [7.5, 10.0]; A2-N1 FR 3.2 % [2.5, 4.1]; N1 DAC power x1.25 13.5 % [12.1, 15.1] (bars: <= 5.0 % upper, <= 5.0 % upper, >= 70 % point).
* GS117 Base X: A1-r1 FR 3.3 % [2.6, 4.2]; A2-N1 FR 2.8 % [2.1, 3.6]; N1 DAC power x1.25 39.6 % [37.5, 41.8] (bars: <= 5.0 % upper, <= 5.0 % upper, >= 70 % point).
* **N1 does not touch A1.** The A1-r1 false-refusal bar is the S0BANK simulator's unchanged statistic; on Base C it is missed at GS41, GS117, so no cell of Base C can meet every bar whatever A2 does. S0BANK's `R-GEN` showed generator T (fixed logs) reads 0.0 % there, i.e. the Base C A1 miss is a property of generator A on 24 cloned source logs (R-DUP below). Whether A1 should be re-read on a different footing is a separate pre-registration for the MM; nothing was tuned here.
* The ONLY change to the S0BANK simulation is the A2 statistic (changed picks only, class-calibrated null, new seeds) plus the non-degenerate cross-check generator P; the margin (1.05), the three bars, the quantifier, B and the seed are the committed ones.

## Tables (machine-generated, `work/tables.md`)

#### N1-T1. THE THREE COMMITTED BARS (SIMULATED; primary NaN convention = real sigma, fail-closed convention in brackets). Wilson 95 %. Generator A.

| geometry | base (seeds) | A1-r1 false refusal (bar: upper <= 5.0 %) | A2-N1 (1.05) false refusal (bar: upper <= 5.0 %) | N1 DAC power x1.25, margin 1.05 (bar: point >= 70 %) | every bar met in this cell |
|---|---|---|---|---|---|
| GS41 | C carve-B clones (3) | 8.6 % [7.5, 10.0] (173/2000) | 3.5 % [2.7, 4.3] (69/2000) [100.0 %] | 9.3 % [8.2, 10.7] [9.7 %] | **False** (missed: A1_r1, DAC_power) |
| GS41 | X S0 stored G25 (2) | 2.0 % [1.5, 2.7] (40/2000) | 2.5 % [1.9, 3.2] (49/2000) [2.5 %] | 27.5 % [25.6, 29.5] [27.5 %] | **False** (missed: DAC_power) |
| GS117 | C carve-B clones (3) | 8.6 % [7.5, 10.0] (173/2000) | 3.2 % [2.5, 4.1] (64/2000) [100.0 %] | 13.5 % [12.1, 15.1] [13.5 %] | **False** (missed: A1_r1, DAC_power) |
| GS117 | X S0 stored G25 (2) | 3.3 % [2.6, 4.2] (66/2000) | 2.8 % [2.1, 3.6] (55/2000) [2.8 %] | 39.6 % [37.5, 41.8] [39.6 %] | **False** (missed: DAC_power) |

| geometry | A1-r1 FR | A2-N1 FR | N1 DAC power x1.25 | geometry ALL-MET (MET = met on C AND on X) | cells meeting every bar | R-NAN | R-SEED (C vs X, pp: A1 FR / N1 FR / DAC power) | R-GEN (P vs A on C, pp: N1 FR / DAC power; status differs) |
|---|---|---|---|---|---|---|---|---|
| GS41 | MISSED | MET | MISSED | **False** | none | True | 6.6 / 1.0 / 18.2 | 1.6 / 2.8; False |
| GS117 | MISSED | MET | MISSED | **False** | none | True | 5.3 / 0.5 / 26.2 | 1.7 / 4.3; False |

#### N1-T2. DESCRIPTIVE (not bars): N1 power by head and margin, A1 power, the A2L reference, generator P, and the INHERITED all-picks (uniform plant) A2 of the S0BANK simulation

| geometry | base-gen | N1 DAC x1.25 @1.00 / @1.05 / @1.10 | DAC x1.5 @1.05 | TTC x1.25 @1.05 | NC x1.25 @1.05 | NC x1.5 @1.05 | TTC x1.5 @1.05 | A2L (all picks, localised plant) DAC x1.25 @1.05 | A1 power +0.135 / +0.05 | A2-N1 FR (fail only / undefined verdicts) | INHERITED all-picks A2 (S0BANK, uniform plant) DAC x1.25 @1.05 | INHERITED all-picks A2 FR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GS41 | C-A | 12.0 % [10.7, 13.5] / 9.3 % [8.2, 10.7] / 7.2 % [6.1, 8.4] | 17.4 % [15.2, 19.9] | 3.2 % [2.3, 4.5] | 0.0 % [0.0, 0.4] | 0.0 % [0.0, 0.4] | 10.4 % [8.7, 12.4] | 2.7 % [1.9, 3.9] | 100.0 % [99.6, 100.0] / 100.0 % [99.6, 100.0] | 3.5 % / 0 | 7.6 % [6.6, 8.9] | 2.5 % [1.9, 3.2] |
| GS41 | C-P | 8.2 % [6.7, 10.1] / 6.5 % [5.1, 8.2] / 4.8 % [3.6, 6.3] | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.9 % / 0 | n/a | n/a |
| GS41 | X-A | 40.6 % [38.5, 42.8] / 27.5 % [25.6, 29.5] / 17.3 % [15.8, 19.1] | 68.5 % [65.6, 71.3] | 11.3 % [9.5, 13.4] | 8.0 % [6.5, 9.8] | 18.7 % [16.4, 21.2] | 37.7 % [34.7, 40.7] | 2.3 % [1.5, 3.4] | 100.0 % [99.6, 100.0] / 100.0 % [99.6, 100.0] | 2.5 % / 0 | 31.6 % [29.6, 33.6] | 1.9 % [1.4, 2.7] |
| GS117 | C-A | 18.5 % [16.9, 20.3] / 13.5 % [12.1, 15.1] / 10.3 % [9.0, 11.7] | 32.4 % [29.6, 35.4] | 7.9 % [6.4, 9.7] | 0.0 % [0.0, 0.4] | 0.0 % [0.0, 0.4] | 21.5 % [19.1, 24.2] | 4.9 % [3.7, 6.4] | 100.0 % [99.6, 100.0] / 100.0 % [99.6, 100.0] | 3.2 % / 0 | 13.8 % [12.4, 15.4] | 2.9 % [2.3, 3.7] |
| GS117 | C-P | 13.7 % [11.7, 16.0] / 9.2 % [7.6, 11.2] / 7.3 % [5.8, 9.1] | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.5 % / 0 | n/a | n/a |
| GS117 | X-A | 57.9 % [55.7, 60.0] / 39.6 % [37.5, 41.8] / 25.1 % [23.2, 27.0] | 92.3 % [90.5, 93.8] | 27.1 % [24.4, 29.9] | 9.0 % [7.4, 10.9] | 31.8 % [29.0, 34.8] | 69.9 % [67.0, 72.7] | 4.4 % [3.3, 5.9] | 100.0 % [99.6, 100.0] / 100.0 % [99.6, 100.0] | 2.8 % / 0 | 60.4 % [58.2, 62.5] | 2.4 % [1.8, 3.1] |

#### N1-T3. CONTROLS and flags

* `C-GS117-A`: controls_ok **True**; PC1a True; PC1b True; PC2 True; PC4 True; PC5 True; PC6 True
* `C-GS117-P`: controls_ok **True**; PC4 True; PC5 True
* `C-GS41-A`: controls_ok **True**; PC1a True; PC1b True; PC2 True; PC4 True; PC5 True; PC6 True
* `C-GS41-P`: controls_ok **True**; PC4 True; PC5 True
* `X-GS117-A`: controls_ok **True**; PC1a True; PC1b True; PC2 True; PC4 True; PC5 True; PC6 True
* `X-GS41-A`: controls_ok **True**; PC1a True; PC1b True; PC2 True; PC4 True; PC5 True; PC6 True
* global: PC3_N1_shortcut_and_restriction True; PC8_s0_code_reproduces_banked_R2CONFIRM True; PC9_BaseX_split_equals_banked_stats True; overall controls_ok **True**
* PC1b deviations: {"C-GS117-A": [7.771561172376096e-16, 1.1102230246251565e-15], "C-GS41-A": [6.661338147750939e-16, 6.661338147750939e-16], "X-GS117-A": [2.220446049250313e-16, 2.220446049250313e-16], "X-GS41-A": [2.220446049250313e-16, 2.220446049250313e-16]}
* R-NAN counts `C-GS117-A` CAL (replicate-seed-heads with NO observed unsafe mass at the changed picks; n = replicates x seeds): {"DAC": 0, "NC": 6000, "TTC": 0}; undefined replicate-seed-heads {"DAC": 0, "NC": 0, "TTC": 0}
* R-NAN counts `C-GS117-P` CAL (replicate-seed-heads with NO observed unsafe mass at the changed picks; n = replicates x seeds): {"DAC": 0, "NC": 3000, "TTC": 0}; undefined replicate-seed-heads {"DAC": 0, "NC": 0, "TTC": 0}
* R-NAN counts `C-GS41-A` CAL (replicate-seed-heads with NO observed unsafe mass at the changed picks; n = replicates x seeds): {"DAC": 7, "NC": 6000, "TTC": 0}; undefined replicate-seed-heads {"DAC": 0, "NC": 0, "TTC": 0}
* R-NAN counts `C-GS41-P` CAL (replicate-seed-heads with NO observed unsafe mass at the changed picks; n = replicates x seeds): {"DAC": 0, "NC": 3000, "TTC": 0}; undefined replicate-seed-heads {"DAC": 0, "NC": 0, "TTC": 0}
* R-NAN counts `X-GS117-A` CAL (replicate-seed-heads with NO observed unsafe mass at the changed picks; n = replicates x seeds): {"DAC": 0, "NC": 0, "TTC": 0}; undefined replicate-seed-heads {"DAC": 0, "NC": 0, "TTC": 0}
* R-NAN counts `X-GS41-A` CAL (replicate-seed-heads with NO observed unsafe mass at the changed picks; n = replicates x seeds): {"DAC": 0, "NC": 0, "TTC": 0}; undefined replicate-seed-heads {"DAC": 0, "NC": 0, "TTC": 0}

#### N1-T4. R-EVENTS / R-DOSE (replicate bases): mean changed picks and mean observed unsafe mass AT THE CHANGED PICKS per replicate base (CAL, generator A), by seed

| job | changed picks per base (by seed) | observed unsafe mass at the changed picks per base: NC | DAC | TTC |
|---|---|---|---|---|
| C-GS117-A | 2413, 2327, 2370 | 0.0, 0.0, 0.0 | 111.2, 89.1, 111.2 | 111.0, 111.0, 111.0 |
| C-GS117-P | 2408, 2321, 2364 | 0.0, 0.0, 0.0 | 111.3, 89.4, 111.3 | 111.3, 111.3, 111.3 |
| C-GS41-A | 751, 723, 737 | 0.0, 0.0, 0.0 | 34.6, 27.7, 34.6 | 34.5, 34.5, 34.5 |
| C-GS41-P | 749, 721, 736 | 0.0, 0.0, 0.0 | 34.5, 27.4, 34.5 | 35.0, 35.0, 35.0 |
| X-GS117-A | 3920, 3904 | 48.0, 49.0 | 116.0, 114.0 | 171.0, 171.0 |
| X-GS41-A | 1205, 1202 | 16.0, 16.0 | 52.0, 51.0 | 41.0, 41.0 |

#### N1-T5. R-POOLRATIO / R-EVENTS of the banked bases (point estimates only; descriptive; no interval)

**Base C pool (carve-B, 400 tokens), by search seed:**

| seed | changed picks (share) | head | distinct changed picks with unsafe mass | observed / predicted unsafe mass at the CHANGED picks (pooled ratio) | at the UNCHANGED picks (pooled ratio) |
|---|---|---|---|---|---|
| 0 | 108 (27.0 %) | NC | 0 | 0.00 / 0.86 (0.000) | 1.00 / 2.59 (0.386) |
| 0 | 108 (27.0 %) | DAC | 5 | 5.00 / 2.04 (2.446) | 5.00 / 8.24 (0.607) |
| 0 | 108 (27.0 %) | TTC | 5 | 5.00 / 3.43 (1.460) | 10.00 / 9.89 (1.011) |
| 1 | 104 (26.0 %) | NC | 0 | 0.00 / 0.82 (0.000) | 1.00 / 2.63 (0.380) |
| 1 | 104 (26.0 %) | DAC | 4 | 4.00 / 1.97 (2.027) | 7.00 / 8.37 (0.837) |
| 1 | 104 (26.0 %) | TTC | 5 | 5.00 / 3.25 (1.539) | 10.00 / 10.08 (0.992) |
| 2 | 106 (26.5 %) | NC | 0 | 0.00 / 0.85 (0.000) | 1.00 / 2.60 (0.384) |
| 2 | 106 (26.5 %) | DAC | 5 | 5.00 / 2.03 (2.459) | 5.00 / 8.22 (0.608) |
| 2 | 106 (26.5 %) | TTC | 5 | 5.00 / 3.40 (1.470) | 10.00 / 9.93 (1.007) |

**Base X on the GS117 logs, by seed:**

| seed | changed picks of sets (share) | head | observed / predicted unsafe mass at the CHANGED picks (pooled ratio) | at the UNCHANGED picks (pooled ratio) |
|---|---|---|---|---|
| 0 | 3920 (24.8 %) | NC | 48.00 / 48.55 (0.989) | 96.50 / 120.80 (0.799) |
| 0 | 3920 (24.8 %) | DAC | 116.00 / 125.97 (0.921) | 274.00 / 294.94 (0.929) |
| 0 | 3920 (24.8 %) | TTC | 171.00 / 181.44 (0.942) | 424.00 / 489.46 (0.866) |
| 1 | 3904 (24.7 %) | NC | 49.00 / 48.17 (1.017) | 97.50 / 120.95 (0.806) |
| 1 | 3904 (24.7 %) | DAC | 114.00 / 125.11 (0.911) | 274.00 / 295.44 (0.927) |
| 1 | 3904 (24.7 %) | TTC | 171.00 / 179.87 (0.951) | 425.00 / 489.88 (0.868) |

**Base X on the GS41 logs, by seed:**

| seed | changed picks of sets (share) | head | observed / predicted unsafe mass at the CHANGED picks (pooled ratio) | at the UNCHANGED picks (pooled ratio) |
|---|---|---|---|---|
| 0 | 1205 (24.4 %) | NC | 16.00 / 14.43 (1.109) | 26.00 / 34.28 (0.759) |
| 0 | 1205 (24.4 %) | DAC | 52.00 / 69.81 (0.745) | 137.00 / 160.39 (0.854) |
| 0 | 1205 (24.4 %) | TTC | 41.00 / 53.53 (0.766) | 105.00 / 134.81 (0.779) |
| 1 | 1202 (24.3 %) | NC | 16.00 / 14.42 (1.109) | 26.00 / 34.28 (0.758) |
| 1 | 1202 (24.3 %) | DAC | 51.00 / 69.65 (0.732) | 137.00 / 160.47 (0.854) |
| 1 | 1202 (24.3 %) | TTC | 41.00 / 53.35 (0.769) | 105.00 / 134.86 (0.779) |

* R-DUP (mean distinct logs of N under generator A, 2,000 outer draws, N1 salt): GS41: 26.2, GS117: 74.0


## Proposed register rows (NOT edited; the Master Mind lands them)

* `D-EPADM-N1-1` (arm N1 of `D-EPADM-S0BANK-1`): the A2 statistic restricted to the CHANGED picks, GS41 ALL-MET=False (cells meeting every bar: none); GS117 ALL-MET=False (cells meeting every bar: none). SIMULATED, SELECTION tier, generator A, one training seed, 3 search seeds (2 on Base X), STRATEGIC PAUSED, controls_ok=True. Artifact: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-20-epadm-n1/raw/results/sim_results.json`.
* RETRACTION_LOG candidate (class "a bar the proposed statistic cannot move"): only if the Base C A1-r1 miss above is present; N1 restricts A2 and leaves A1-r1 unchanged.

## Deliverable manifest

See `out/paths.json` (repo path, absolute source, "raw" for every item). Everything exists under `C:/Users/Admin/par18/n1sim/` (the only copy until landed); inputs read in place and never modified: `epadm_search/work/carveB_bank2`, `epadm_search/tipgate`, `epadm_search/tipr1`, the C1 code dir, `pickrule/run/prepare/arrays.npz` (sha256 pinned), `s0amend/work/sim/sim_results.json` (INHERITED reference rows only).

A 20-replicate plumbing smoke of the run/repro/report path was executed after the hash and discarded (scratch deleted); its rates were not used.

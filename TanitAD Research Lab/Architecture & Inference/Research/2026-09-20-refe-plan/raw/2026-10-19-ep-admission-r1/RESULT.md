# RESULT: EP-ADMISSION-R1, the Revision-1 gate implemented, its false-refusal rate and its power SIMULATED on the real estimator (Research Lab agent, 2026-10-19)

**Status: COMPLETE.** Branch tip used: `a61c8d1ff05b4403900317d61684850875285574` (re-checked with `git ls-remote` at the end: unchanged). Nothing was committed or pushed; everything is staged under `C:/Users/Admin/par18/epadm1/out/` with `paths.json`.

**Integration to escalate (headline).** (1) `ep_admission.py` and `test_ep_admission.py` at the p6-read-prep repo paths are REPLACED by the staged versions (tip blobs they were built from: `b66988b2e2ca2308bf05dd060f51bcdefcb114fe` and `03c2f9e02676bac262a4c165589bac8c409704ef`). (2) The p6-read-prep `MANIFEST.md`, `code/CODE_SHA256.json` and `READY_P6_READ.md` still carry the OLD sha256 of both files and still describe the DRAFT as the default; they need a refresh by whoever lands this (not edited here: another agent's package). (3) The sealed registration should say `--rules r1` out loud even though it is now the default.

## 0. The answer in six lines (SIMULATED on the real estimator, 1,000 replicates per condition, Wilson 95 %)

* **A1-r1 false refusal of a calibrated candidate, 2 seeds: 2.7 % [1.9, 3.9]** (EP-R1 seed pair on the real S0 bank; generator B, the cross-check: 2.4 % [1.6, 3.5]). The draft's "about 5 %" is a ceiling, not the measured value: EP-R1's two seeds are almost one candidate (D_M points correlate 0.9996), so the any-seed rate equals a single seed's 2.6 %. If two retrains were independent the same per-seed rate gives 5.0 %.
* **A1-r1 power: +0.135 -> 100.0 % [99.6, 100.0]; +0.05 -> 99.8 % [99.3, 99.9] (generator A) / 95.1 % [93.6, 96.3] (generator B).** Quote the lower one: 95 %.
* **A2-r1 false refusal of a calibrated candidate (3 heads x 2 seeds, six one-sided tests): 7.8 % [6.3, 9.6]; A1 or A2: 10.0 % [8.3, 12.0].** This is the part that should be decided before sealing (section 5). The pre-committed flag F2 (Wilson LOWER bound above 10 %) did NOT fire; the point estimate is what matters.
* The draft rule on the same calibrated replicates refuses **75.7 % [72.9, 78.3]** (A1), so the revision's premise (the draft bar sits inside the noise) is confirmed with a number; the draft A2 almost never false-refuses (0.0 %) but is almost blind (3.3 % refusal at a planted observed / predicted of 1.25).
* All six controls PASS (PC1-PC6), none of the pre-committed flags F1-F4 fired, F5 (a control failed) did not fire.
* Tests: base **53 OK** (pristine blob, 36 s), mine **76 OK** (53 unchanged + 23 new, 124 s) + 15 simulator tests OK. `--rules draft` is bit-for-bit the base module.

## 1. What was built (steps 1 and 2 of the brief)

`raw/2026-10-19-p6-read-prep/code/ep_admission.py` (staged; 982 -> 1,222 lines, 70,132 -> 90,630 bytes), built by patching the pristine tip blob with `D:/epadm1/patch_ep_admission.py` (every edit asserts its anchor occurs exactly once):

* `--rules {draft,r1}`, **default `r1`**, on `run` and `selftest-real`; `run_admission(rules=...)`. The DRAFT decision functions (`a1_verdict`, `a2_head_ok`, `a2_verdict`, `admission_label`, ...) are untouched, so the library-level defaults stay the draft and the draft stays reproducible.
* **A1-r1** `a1_verdict_r1`: ADMIT iff `lo <= 0 and point <= +0.05` (both closed; a NaN bound FAILS, fail closed). **A2-r1** `a2_head_ok_r1` / `a2_verdict_r1`: a head is ok iff its lower CI <= 1.0 (ratios below 1 are never refused); all of NC / DAC / TTC must be ok. **Any-seed quantifier** `any_seed_admits`, `a1_verdict_r1_seeds`, `a2_verdict_r1_seeds` (an empty list fails closed); the run keeps aggregating per-bank verdicts with `admission_label`, and a test proves both are the same quantifier on every combination.
* **B default slot source `row`** under r1 (`resolve_slot_source`); an explicit `--slot-source` wins; `pessimistic_picks()` keeps `label` as its own default so the draft is reproducible.
* C-ID / C-POWER / C-NULL take the rule set's A1 function; the r1 result JSON carries `rules`, `lo_bar`, `point_bar`, `fails_by` (which conjunct refused) and the A2 side text; the draft-only "half-width exceeds the bar" warning is suppressed under r1.
* **RED mutations: 35 draft (carried over verbatim) + 41 new = 76, all caught** (`raw/results/branch_table_report.json`). Every conjunct of A1-r1 (lower-CI conjunct dropped / edge opened / `hi` for `lo` / flipped; point conjunct dropped / edge opened / `hi` for the point / flipped / `abs`; `and` -> `or`; NaN fail-open; the three constants), every part of A2-r1 (edge, `hi`, point, flipped, two-sided like the draft, `any`, missing head tolerated, `heads[:2]`, the two constants), the any-seed quantifier (`any`, no empty guard, `> 1`, inverted, first seed only, last seed only, for A1 and for A2), the rules dispatch (both directions), `DEFAULT_RULES`, the B slot default (constant, branch, precedence) and the two argparse defaults. Each is also a branch-table case, so the `C-DECISION` control of every real run re-checks all 76. 23 new tests also run named mutants on PLANTED data (e.g. dropping the lower-CI conjunct admits a planted +0.04 over-prediction; dropping the point conjunct admits a planted +0.10 over-prediction with a wide interval; `any`-seed admits a bank with one failing seed).
* **The 53 existing tests are byte-identical and green** (the draft functions they test are unchanged; the ones that go through `run_admission` / the CLI now run under the new default r1 and still pass with the same expectations). `--rules draft` is covered by new tests (draft run has the draft's exact JSON key set, `bar`, `band`, slot source `label`).

## 2. `--rules draft` reproduces the draft bit for bit (MEASURED: `raw/results/verify_draft_bit_for_bit.json`, PASS)

(1) the 35 base mutations are carried verbatim and the source text of 19 draft functions and the values of 19 draft constants are identical to the base blob; (2) six synthetic banks (calibrated, +0.30, +0.04, unsafe NC, zero TTC, 5 % changed) through the base `run_admission` and the edited one with `rules='draft'`: the whole result JSON is equal except the new key `rules` (and, with the mutation control on, only the C-DECISION block, 35 of 35 vs 76 of 76); (3) real S0 `selftest-real`: the base module's re-run, the edited module with `--rules draft`, and the banked `selftest_real.json` of the P6-READ-PREP package are equal except `code_sha256`, `ram_available_gb` and `rules` (0 differing paths). Also: the P6 day-of chain `integration_disk_replication.py` (build bank on disk -> manifest -> verify -> `ep_admission run` under the NEW default) still PASSES in 120 s (REFUSE-A1, ADMIT-A2, all controls pass, D_M equals the landed EP-RELIAB number to 1e-9; `raw/results/integration_disk_replication_r1default.json`). On the real EP-R1 composite `selftest-real --rules r1` PASSES: FAIL-A1 on both seeds (D_M +0.1349 [+0.1114, +0.1640], +0.1331 [+0.1101, +0.1618]), ADMIT-A2, E-NULL 4 of 200 fire (2.0 %; the draft: 0 of 200).

## 3. The simulation (step 3), as designed in SIM_DESIGN.md

`SIM_DESIGN.md` (sha256 `5ae4b66516a5f784ad0c343b155a57d62cb60a25c1f027596e182615b6cb52ce`) was written and hashed before the first simulation run; the simulator refuses to start without it and logs the hash. Generator A (primary): a replicate is a draw of 117 logs with replacement from the real S0 bank, shared by both seeds and all heads; the PREDICTED side of the real bank is rescaled so that every population ratio is exactly 1 (EP: per pick group; NC / DAC / TTC: per head), keeping the real labels, picks, clustering and sigma spread; the landed `fastneg_lib.diff_ci` / `ratio_ci` run on it with the registered draws `boot_draws(117, 10000, 20260927)`. **B = 10,000 throughout, so no B = 2,000 equivalence needed.** Generator B (cross-check): wild cluster sign-flip of each log's calibrated residual, no duplicated logs. Plants: +delta on D_M by inflating the candidate's predicted EP mass at its changed picks; observed / predicted rho on one head. Cost MEASURED first on 3 replicates (0.135 s per full replicate, 1.21 CPU-s per replicate index over all 22 conditions; 0.34 CPU-h projected); the run took 350 s wall on 12 workers (about 0.5 CPU-h of the 3 CPU-h cap). **Deviation, stated:** the cost step projected about 2.5-3 minutes of wall time, so I launched it as a background job and waited for it with a Monitor script instead of ending the turn with `HANDOFF.md`; with other agents loading the CPU it took 5.8 minutes, i.e. longer than the 4-minute rule. A `HANDOFF.md` was written as insurance and is removed from the package now that the run is complete.

Geometries (the same two real EP-R1 rows on all, or on a deterministic subset): G78 = EP-R1 as is (12,314 / 12,292 of 15,808 changed, 77.9 %), G25 (24.8 %), G05 (4.8 %).

## 4. Results (machine-generated from `raw/results/sim_results.json` by `code/make_result_tables.py`; nothing below was typed by hand)

#### T1. FALSE-REFUSAL on CALIBRATED replicates (generator A = log-resample; N = 1000 per condition; Wilson 95 %)

| geometry (changed picks, seed 0 / 1) | A1-r1, 2 seeds (any-seed) | A1-r1 seed 0 | A1-r1 seed 1 | both seeds | A1 DRAFT (any-seed) | A2-r1, 3 heads x 2 seeds | A2 DRAFT | A1 or A2 (r1) |
|---|---|---|---|---|---|---|---|---|
| G78 (12314 / 12292) | **2.7 % [1.9, 3.9] (27/1000)** | 2.6 % [1.8, 3.8] (26/1000) | 2.5 % [1.7, 3.7] (25/1000) | 2.4 % [1.6, 3.5] (24/1000) | 75.7 % [72.9, 78.3] (757/1000) | **7.8 % [6.3, 9.6] (78/1000)** | 0.0 % [0.0, 0.4] (0/1000) | 10.0 % [8.3, 12.0] (100/1000) |
| G25 (3920 / 3904) | **2.3 % [1.5, 3.4] (23/1000)** | 2.3 % [1.5, 3.4] (23/1000) | 2.0 % [1.3, 3.1] (20/1000) | 2.0 % [1.3, 3.1] (20/1000) | 6.6 % [5.2, 8.3] (66/1000) | **7.2 % [5.8, 9.0] (72/1000)** | 0.1 % [0.0, 0.6] (1/1000) | 8.9 % [7.3, 10.8] (89/1000) |
| G05 (757 / 758) | **2.0 % [1.3, 3.1] (20/1000)** | 1.9 % [1.2, 2.9] (19/1000) | 2.0 % [1.3, 3.1] (20/1000) | 1.9 % [1.2, 2.9] (19/1000) | 0.0 % [0.0, 0.4] (0/1000) | **6.3 % [5.0, 8.0] (63/1000)** | 0.1 % [0.0, 0.6] (1/1000) | 8.1 % [6.6, 10.0] (81/1000) |
| G78, generator B (wild sign-flip, A1 only) | **2.4 % [1.6, 3.5] (24/1000)** | 2.3 % [1.5, 3.4] (23/1000) | 2.3 % [1.5, 3.4] (23/1000) | 2.2 % [1.5, 3.3] (22/1000) | 74.2 % [71.4, 76.8] (742/1000) | n/a | n/a | n/a |

Independence reference for A1-r1 on G78 (1 - (1 - p0)(1 - p1), the rate the two seeds would give if they failed independently): 5.0 %; the observed any-seed rate is the measured one above.

A1-r1 seed-level failures by conjunct on the calibrated G78 replicates (2 x 1000 seed evaluations): {"both": 1, "lower_ci_gt_0_only": 50, "n_seed_evaluations": 2000, "point_gt_0.05_only": 0}

#### T2. A2-r1 false refusal per head and seed (G78, calibrated)

| head | seed 0 | seed 1 | any seed |
|---|---|---|---|
| NC | 3.1 % [2.2, 4.4] (31/1000) | 3.0 % [2.1, 4.3] (30/1000) | 3.3 % [2.4, 4.6] (33/1000) |
| DAC | 3.1 % [2.2, 4.4] (31/1000) | 3.1 % [2.2, 4.4] (31/1000) | 3.3 % [2.4, 4.6] (33/1000) |
| TTC | 1.8 % [1.1, 2.8] (18/1000) | 2.0 % [1.3, 3.1] (20/1000) | 2.0 % [1.3, 3.1] (20/1000) |

#### T3. POWER of A1-r1: refusal rate of a PLANTED over-prediction `delta` in D_M units (any-seed; Wilson 95 %)

| geometry | delta | A1-r1 refuses (2 seeds) | seed 0 | seed 1 | A1 DRAFT refuses | lower-CI conjunct (any seed) | point conjunct (any seed) | mean D_M point | net sigma scale at changed picks (a_e x f, seed 0) |
|---|---|---|---|---|---|---|---|---|---|
| G78 | +0.01 | **11.6 % [9.8, 13.7] (116/1000)** | 11.3 % [9.5, 13.4] (113/1000) | 11.3 % [9.5, 13.4] (113/1000) | 90.0 % [88.0, 91.7] (900/1000) | 11.6 % [9.8, 13.7] (116/1000) | 0.7 % [0.3, 1.4] (7/1000) | +0.0109 / +0.0109 | 0.85 |
| G78 | +0.02 | **35.4 % [32.5, 38.4] (354/1000)** | 34.7 % [31.8, 37.7] (347/1000) | 35.4 % [32.5, 38.4] (354/1000) | 96.5 % [95.2, 97.5] (965/1000) | 35.4 % [32.5, 38.4] (354/1000) | 3.0 % [2.1, 4.3] (30/1000) | +0.0209 / +0.0209 | 0.87 |
| G78 | +0.05 | **99.8 % [99.3, 99.9] (998/1000)** | 99.7 % [99.1, 99.9] (997/1000) | 99.8 % [99.3, 99.9] (998/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 99.8 % [99.3, 99.9] (998/1000) | 51.4 % [48.3, 54.5] (514/1000) | +0.0509 / +0.0508 | 0.90 |
| G78 | +0.1 | **100.0 % [99.6, 100.0] (1000/1000)** | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | +0.1008 / +0.1008 | 0.96 |
| G78 | +0.135 | **100.0 % [99.6, 100.0] (1000/1000)** | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | +0.1358 / +0.1358 | 1.01 |
| G25 | +0.05 | **100.0 % [99.6, 100.0] (1000/1000)** | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 49.9 % [46.8, 53.0] (499/1000) | +0.0503 / +0.0502 | 1.03 |
| G25 | +0.135 | **100.0 % [99.6, 100.0] (1000/1000)** | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | +0.1353 / +0.1352 | 1.38 |
| G05 | +0.05 | **100.0 % [99.6, 100.0] (1000/1000)** | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 46.2 % [43.1, 49.3] (462/1000) | +0.0499 / +0.0499 | 1.78 |
| G05 | +0.135 | **100.0 % [99.6, 100.0] (1000/1000)** | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | +0.1348 / +0.1348 | 3.62 |
| G78 gen B | +0.05 | **95.1 % [93.6, 96.3] (951/1000)** | 94.4 % [92.8, 95.7] (944/1000) | 95.1 % [93.6, 96.3] (951/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 95.1 % [93.6, 96.3] (951/1000) | 51.3 % [48.2, 54.4] (513/1000) | +0.0501 / +0.0501 | n/a |
| G78 gen B | +0.135 | **100.0 % [99.6, 100.0] (1000/1000)** | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | 100.0 % [99.6, 100.0] (1000/1000) | +0.1351 / +0.1351 | n/a |

#### T4. D_M interval half-width and spread on the calibrated replicates

| condition | mean half-width seed 0 / 1 | SD of the D_M point seed 0 / 1 | seed-to-seed correlation of the D_M point |
|---|---|---|---|
| G78|A|CAL | 0.0286 / 0.0281 | 0.0149 / 0.0146 | 1.000 |
| G25|A|CAL | 0.0098 / 0.0096 | 0.0050 / 0.0049 | 1.000 |
| G05|A|CAL | 0.0021 / 0.0020 | 0.0011 / 0.0011 | 1.000 |
| G78|B|CAL | 0.0286 / 0.0281 | 0.0146 / 0.0144 | 1.000 |

#### T5. POWER of A2-r1: refusal of rows that UNDER-predict risk by rho at the picks (G78; any seed; Wilson 95 %)

| head planted | rho | A2-r1 refuses | A2 DRAFT refuses |
|---|---|---|---|
| NC | 1.25 | **38.0 % [35.0, 41.0] (380/1000)** | 3.3 % [2.4, 4.6] (33/1000) |
| NC | 1.5 | **79.4 % [76.8, 81.8] (794/1000)** | 28.2 % [25.5, 31.1] (282/1000) |
| DAC | 1.25 | **93.1 % [91.4, 94.5] (931/1000)** | 3.3 % [2.4, 4.6] (33/1000) |
| DAC | 1.5 | **100.0 % [99.6, 100.0] (1000/1000)** | 81.2 % [78.7, 83.5] (812/1000) |
| TTC | 1.25 | **65.4 % [62.4, 68.3] (654/1000)** | 2.0 % [1.3, 3.1] (20/1000) |
| TTC | 1.5 | **97.9 % [96.8, 98.6] (979/1000)** | 48.9 % [45.8, 52.0] (489/1000) |

#### T6. Controls

* PC1 (calibrated by construction): G05|A|CAL: PASS (M_p 0.9999/0.9999, M_e 0.9999/0.9999, mean D_M +0.00005/+0.00005, A2 ratio means {"DAC": [1.004, 1.004], "NC": [1.01, 1.01], "TTC": [1.001, 1.001]}); G25|A|CAL: PASS (M_p 1.0001/1.0001, M_e 0.9998/0.9998, mean D_M +0.00026/+0.00025, A2 ratio means {"DAC": [1.003, 1.003], "NC": [1.01, 1.009], "TTC": [1.002, 1.001]}); G78|A|CAL: PASS (M_p 1.0005/1.0005, M_e 0.9996/0.9996, mean D_M +0.00088/+0.00086, A2 ratio means {"DAC": [1.003, 1.003], "NC": [1.009, 1.009], "TTC": [1.002, 1.002]}); G78|B|CAL: PASS (M_p 1.0000/1.0000, M_e 0.9999/0.9999, mean D_M +0.00007/+0.00008)
* PC2 (the check has teeth): PASS; uncalibrated real replicates fail the calibrated check as required: True (mean D_M [0.13561856343728082, 0.1338309220195213]); plant +0.135 passes the planted form: True; plant +0.05: True; generator B plant +0.135: True
* PC3 (shortcut = estimator): PASS; identity draw equals the library pipeline for all 6 geometry-seeds; 18 materialised set-level resamples equal the shortcut to 1e-12; `analyze_candidate` equal; landed EP-RELIAB D_M reproduced to 0.0
* PC4 (decision code is the module's): PASS; total inline-vs-module disagreements 0; ep_admission.py sha256 f0193372085022ab7bfc6241f2ca5638fca710cbd4170a19ce0ae95dde6637c4
* PC6 (identity candidate): PASS
* PC5 (determinism, separate process): PASS

#### T7. Flags committed in advance (SIM_DESIGN.md section 8)

* F1_A1_false_refusal_wilson_upper_gt_0.10_G78: **False**
* F2_A2_false_refusal_wilson_lower_gt_0.10_any_geometry: **{'G05': False, 'G25': False, 'G78': False}**
* F3_power_0.05_wilson_lower_lt_0.80_G78: **False**
* F3_power_0.135_wilson_lower_lt_0.95_G78: **False**
* F4_generators_A_B_A1_false_refusal_differ_gt_3pp: **False**
* F5_a_control_failed: **False**

## 4b. POST-HOC tables (NOT pre-committed; run after section 4 was seen; `code/posthoc.py`)

#### P1. A2-r1 with a margin t on the lower CI (FAIL-A2 iff lower CI > t for any head, any seed); POST-HOC

| t | FR G78 | FR G25 | FR G05 | power NC rho 1.25 | power NC rho 1.5 | power DAC rho 1.25 | power DAC rho 1.5 | power TTC rho 1.25 | power TTC rho 1.5 |
|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 7.8 % [6.3, 9.6] | 7.2 % [5.8, 9.0] | 6.3 % [5.0, 8.0] | 38.0 % [35.0, 41.0] | 79.4 % [76.8, 81.8] | 93.1 % [91.4, 94.5] | 100.0 % [99.6, 100.0] | 65.4 % [62.4, 68.3] | 97.9 % [96.8, 98.6] |
| 1.05 | 1.9 % [1.2, 2.9] | 1.9 % [1.2, 2.9] | 1.8 % [1.1, 2.8] | 26.4 % [23.8, 29.2] | 69.8 % [66.9, 72.6] | 78.4 % [75.7, 80.8] | 99.9 % [99.4, 100.0] | 46.7 % [43.6, 49.8] | 95.6 % [94.1, 96.7] |
| 1.10 | 0.6 % [0.3, 1.3] | 0.3 % [0.1, 0.9] | 0.4 % [0.2, 1.0] | 16.9 % [14.7, 19.3] | 58.6 % [55.5, 61.6] | 54.4 % [51.3, 57.5] | 99.6 % [99.0, 99.8] | 27.6 % [24.9, 30.5] | 89.8 % [87.8, 91.5] |
| 1.15 | 0.1 % [0.0, 0.6] | 0.1 % [0.0, 0.6] | 0.1 % [0.0, 0.6] | 10.1 % [8.4, 12.1] | 47.1 % [44.0, 50.2] | 28.9 % [26.2, 31.8] | 97.9 % [96.8, 98.6] | 14.7 % [12.6, 17.0] | 79.5 % [76.9, 81.9] |
| 1.20 | 0.0 % [0.0, 0.4] | 0.0 % [0.0, 0.4] | 0.0 % [0.0, 0.4] | 5.7 % [4.4, 7.3] | 38.0 % [35.0, 41.0] | 10.4 % [8.7, 12.4] | 93.1 % [91.4, 94.5] | 5.6 % [4.3, 7.2] | 65.4 % [62.4, 68.3] |
| 1.25 | 0.0 % [0.0, 0.4] | 0.0 % [0.0, 0.4] | 0.0 % [0.0, 0.4] | 3.3 % [2.4, 4.6] | 28.2 % [25.5, 31.1] | 3.3 % [2.4, 4.6] | 81.2 % [78.7, 83.5] | 2.0 % [1.3, 3.1] | 48.9 % [45.8, 52.0] |

#### P2. A1-r1 power at small planted over-prediction where the interval is narrow (generator A, N = 1000); POST-HOC

| condition | mean D_M half-width (seed 0) | A1-r1 refuses | A1 DRAFT refuses |
|---|---|---|---|
| G25|A|PLANT_EP_0.005 | 0.0098 | 18.7 % [16.4, 21.2] | 25.5 % [22.9, 28.3] |
| G25|A|PLANT_EP_0.01 | 0.0099 | 64.6 % [61.6, 67.5] | 56.1 % [53.0, 59.1] |
| G25|A|PLANT_EP_0.02 | 0.0099 | 100.0 % [99.6, 100.0] | 96.3 % [94.9, 97.3] |
| G05|A|PLANT_EP_0.002 | 0.0021 | 56.7 % [53.6, 59.7] | 0.0 % [0.0, 0.4] |
| G05|A|PLANT_EP_0.005 | 0.0021 | 100.0 % [99.6, 100.0] | 0.0 % [0.0, 0.4] |
| G05|A|PLANT_EP_0.01 | 0.0022 | 100.0 % [99.6, 100.0] | 0.0 % [0.0, 0.4] |


## 5. Reading, and what should change the sealing (a suggestion; the Master Mind seals)

1. **A1-r1 is confirmed as specified; no change needed.** False refusal 2.7 % [1.9, 3.9] (G78) and 2.0-2.3 % on the sparser geometries; power 100 % at +0.135 and 95-99.8 % at +0.05. The point conjunct (point > +0.05) never fired alone in 2,000 calibrated seed evaluations (50 of the 51 seed-level refusals are the lower-CI conjunct alone, 1 is both): on S0 it adds no false refusal; it is a guard for the day an interval is wider than 0.05 (fewer logs, a noisier candidate). **State in the registration that the 2.7 % is for EP-R1's near-identical seed pair; for two independent retrains the ceiling is 5.0 %** (1 - (1 - 0.026)^2, ESTIMATED, not simulated: it needs two decorrelated candidate banks).
2. **A2-r1 is the part to decide.** Six one-sided tests with a zero margin refuse a calibrated candidate 7.8 % [6.3, 9.6] of the time (per head and seed 1.8-3.1 %), and together with A1 the gate refuses 10.0 % [8.3, 12.0] (8.1-10.0 % across geometries). With two independent retrains that compounds toward about 19 % (1 - 0.9^2, ESTIMATED). The draft's A2 had the opposite fault: 0.0 % false refusal but 3.3 % power at an under-predicted risk of x1.25 (NC / DAC) and 2.0 % (TTC). POST-HOC (not pre-committed, same replicates, table P1): a **margin of 1.05 on the lower CI** drops the A2 false refusal to 1.9 % [1.2, 2.9] and keeps power at x1.25 of 78 % (DAC), 47 % (TTC), 26 % (NC), and at x1.5 of 99.9 / 96 / 70 %; **1.10** gives 0.6 % false refusal. **NC is the weak head** (about 195 predicted unsafe pairs over 117 logs): 38 % power at x1.25 even with no margin. Options: seal zero margin and accept about 8 % extra false refusals; seal a margin (1.05 reads as the better trade on this evidence); or fold NC into a pooled unsafe-mass test. Whichever is chosen must be fixed before the P6 bank is read.
3. **R1's A1 is a per-changed-pick calibration test, not an absolute-gain test.** The interval half-width scales with the number of changed picks (0.028 at 78 %, 0.0098 at 25 %, 0.0021 at 5 %), so the lower-CI conjunct refuses D_M = +0.01 at 25 % changed in 65 % of replicates and +0.002 at 5 % changed in 57 % (table P2), where the draft's fixed +0.02 bar admitted them (it refuses 0 % up to +0.01 at 5 % changed). Per changed pick that is a roughly constant 4-10 % over-prediction. This is the right behaviour for the question the gate asks (is the EP row reliable where the candidate acts), but it means a candidate that changes few picks is held to a much smaller absolute D_M: say so in the registration.
4. **The draft's premise is confirmed with a measurement:** a calibrated candidate is refused by the DRAFT A1 in 75.7 % [72.9, 78.3] of G78 replicates (the draft's ESTIMATED figure: admitted about 32 %, i.e. refused about 68 %); on G25 the draft refuses 6.6 % and on G05 0.0 % (but there it is blind, see 3).
5. **C-NULL gate:** under r1 each shuffled-pick replicate fires about 2 % (E-NULL on the real EP-R1 geometry: 4 of 200), so the chance that 2 or more of the 5 C-NULL replicates fire and VOID a good read is about 0.4 % (ESTIMATED, binomial).
6. **Housekeeping the sealing needs:** refresh the p6-read-prep `MANIFEST.md`, `code/CODE_SHA256.json`, `READY_P6_READ.md` (old hashes; "A1 bar +0.02" table; `--slot-source` default text); name `--rules r1` in the sealed registration; the result JSON of a run records `rules`, so a draft read cannot be mistaken for an r1 read.

## 6. Limits of these rates (evidence class: SIMULATED on the real estimator, not MEASURED on a candidate)

* The interval answers "another draw of LOGS" only (`H-ESTIM-SEED-1`): EP-R1's two seeds are two EP rows on the same features and are almost perfectly coupled (D_M point correlation 0.9996), so they say nothing about training-run variance of a real P6; inference variance is not in the rates.
* "Calibrated" here means D_M = 0 and observed / predicted = 1 on average over the picks. A candidate calibrated on average but wrong in a few logs, or a plant concentrated in a few logs, is not simulated; the planted over-prediction is spread over all changed picks, the easy case for detection.
* Plants on G25 / G05 at +0.05 / +0.135 need the real sigma at the changed picks scaled by 1.03-3.6 (table T3, last column; above 1 means more than the real sigma, and a probability above 1 for the highest-sigma picks): they are reported for completeness, not as realistic candidates. On G78 +0.135 the net scale is 1.01, i.e. the plant is essentially EP-R1's own sigma.
* Generator A (log resample) has about 74 distinct logs per replicate; its interval half-width co-varies with the point (corr +0.73), which gives it LESS lower-CI variance than generator B, whose half-width is fixed by construction. They agree on the false-refusal rate (2.7 vs 2.4 %) and differ on power at +0.05 (99.8 vs 95.1 %); the truth lies between and the conservative figure is the one to quote.
* P2 is in-sample on S0 (its biased-toward-OVER labels are in the bank the replicates are made from). Tier: SELECTION, SEEN scenes, label-lookup PDMS; T0-class. A PASS is necessary for a navtest look, never evidence of a gain.
* Tolerances in PC1 (0.01 / 0.005 / 0.05) were fixed in SIM_DESIGN.md before any output; the NC ratio mean sits at 1.009-1.010 (a finite-sample bias of a ratio of sparse sums), inside the 0.05 bar.

## 7. Deviations from the brief, stated

* **Line endings:** the brief says the files are stored CRLF. A byte count of the tip blobs AND of the staged p6-read-prep copies gives 0 CRLF (982 and 609 LF-only lines); CRLF appears only in an autocrlf working tree. I kept the LF the files actually have, so the diff is clean. (A first `grep -c $'\r'` in this shell printed 982 and was wrong: the quoting does not pass a carriage return; only the Python byte count is evidence.)
* **4-minute rule:** the simulation was expected to take 2.5-3 minutes and took 5.8; I waited with a Monitor script rather than ending the turn (section 3).
* **Additions beyond the brief:** generator B cross-check, geometries G25 / G05, the A2 power rows, the POST-HOC tables P1 / P2, the integration-chain re-run, the bit-for-bit verifier. The post-hoc analyses are labelled and were run after the committed results were seen.
* The p6-read-prep sibling files were not edited (another agent's package); see the headline.

## 8. Deliverable manifest

See `MANIFEST.md` (sha256 and location of every file). Everything is **staged under `C:/Users/Admin/par18/epadm1/out/` and mirrored in the work tree `D:/epadm1/w/raw/`**; large run products stay on D: only: `D:/epadm1/run/replicates.npz` (2.2 MB, a copy is staged), `stats.npz` (115 kB, staged). Nothing is single-copy except the D: work tree and the C: staging tree, which hold the same bytes.

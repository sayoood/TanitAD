# ANATOMY_25 - why REPAIR-D left 25 navtest tokens at DAC = 0 (TF-N4 A1)

Descriptive. Decides nothing. All 25 tokens were seen on navtest (the whole programme has read them), so everything here is SEEN-ON-NAVTEST and is an instrument reading, not a result. CPU, dev box, 2026-11-13.
Evidence class: MEASURED, ours; tier T1 builder-side TRACKED scoring on the harness metric cache (the builder's own scorer, A1: labels from the nuPlan-DB ego). Arm: "RJ5 teacher + P5-R + REPAIR-D" (N4-D: full certificate; N4-DM: DAC-only certificate), base = N2-R plans. Raw traces: `raw/primary_s*.jsonl`; per-token JSON `anatomy_25.json`; rule `anatomy_rule.py` (sha256 f22e8e52f7e019f45dea88eb4a52d5e519e29c50705a6852fc4b2ae1bc744d7c, hashed 09:40 BEFORE the classified table was produced).

## 1. Instrument and controls
- `n4_repair_trace.py` calls the REAL pinned `n4_repair.repair_target` through the builder's own `make_repair` (no re-implementation); a proxy records every `labels`/`score` call and a patch of `return_vector` records each worst-outside reading. Hops (file:line, run tree `F:/tanitad_scratch/n4nt/run_a1/n4`): `n4_nt_build.py:78` make_repair, `:214` Ctx.from_parts, `:259` base score, `:268-270` repair call / `{"applied": False}`; `n4_repair.py:113` repair_target, `:47` return_vector, `:25`/`:29` certificates; `n4_geom.py:84/92/101` labels / score / ext_dist.
- CONTROL 1 (MEASURED, `raw/control.jsonl`, `test_run.log`): 3 repaired N4-D tokens (iterations 1, 3, 5) reproduce the shipped `n4_plans_N4D.npz` plan with max |d| = 0.0. Needs BLAS threads = 1 set before numpy is imported: with default threads the same label solve drifted 2.8e-6 (found by the first test run; the builder ran at 1 thread).
- CONTROL 2: all 25 reproduce `applied: false` for N4-D (asserted in `make_anatomy.py`) and `applied: true` for the 5 N4-DM-only tokens; builder's 6 candidates per token (iterations 0..5) each.
- CONTROL 3 (deliberate regression): margin forced to -0.05 m changes the control token's plan or makes it fail (`test_n4_repair_trace.py`, 4 passed).
- CONTROL 4 (consistency, MEASURED): a DAC-only certificate at margin 0.05 / cap 5 certifies exactly the 5 tokens N4-DM repaired and N4-D did not (`posthoc_25.json`).

## 2. Cause counts by the PRE-WRITTEN rule (precedence b, d, c, a, e)
| class | n |
|---|---|
| (b) a DAC = 1 candidate existed, the certificate refused it | **5** |
| (d) no outside corner / undefined return vector | 0 |
| (c) "NO-FOLLOW" as defined by the rule: cap hit, median realised extent reduction < 0.25 | **17** |
| (a) cap hit, extent still shrinking | **2** |
| (e) other (cap hit, extent shrinking in sequence but not at the last step) | 1 |

(b) clauses: **C (comfort) x3, TTC x1, NC + TTC x1**; exactly these 5 are the N4-DM-only repairs (`dm_only_clauses` in `anatomy_25.json`). So the full certificate blocked 5 of 25; DAC-only certification does not mean the plan is otherwise as good (the N4-DM arm's other families were read in `RESULT_N4_A1.md`).

## 3. Post-hoc reading: what the rule's label (c) hides  (POST-HOC, not pre-registered; `posthoc_25.json`, `exec_probe.py`)
The pre-written rule's name for (c) is wrong for most of its members, and the probe says so. For the 20 non-(b) tokens, iteration 0 requested a shift d = (ext + 0.05 m) * u at t_v:
- The executed (tracked) path DOES follow the request at the worst step: executed shift / requested >= 0.85 in **15 of 20** (12 of the 17 rule-(c) tokens); only 5 have a true low ratio (0.02 to 0.60, short t_v 0.6 to 2.3 s).
- The loop **oscillates**: the lateral sign of the return vector flips >= 2 times in **16 of 20**; the extent drops sharply at iteration 1 (e.g. 11.25 -> 0.85, 0.47 -> 0.04) and then ping-pongs at 0.03 to 0.15 m, each correction pushing a corner out on the other side or at another time step (worst time alternates between two values in many tokens). 12 of 20 are both oscillating and executed-following.
- Read: the dominant failure is a one-direction shift applied to a **two-sided** constraint (no single offset ramp clears both boundaries at every step), not an iteration cap and not a tracker that fails to follow. Whether a clear lateral interval exists at all (corridor narrower than the vehicle plus 5 cm) is NOT measured here; it is the first thing the navtrain development should test.

## 4. Per-token table
| token | rule cause | clause | extent seq (m, candidates 0..5) | lat. sign flips | exec/req it0 | N4-DM | whatif first full-cert iter m.05/.10/.20 |
|---|---|---|---|---|---|---|---|
| 74cc0056b2c05248 | a CAP-SHRINKING | - | 1.276, 0.328, 0.175, 0.161, 0.096, 0.088 | 4 | 0.94 | - | 10/8/6 |
| f004b5e4838455f3 | a CAP-SHRINKING | - | 0.442, 0.436, 0.233, 0.114, 0.050, 0.007 | 4 | 0.93 | - | 6/None/6 |
| 2b04c510144e5545 | b CERT-REFUSED | c | 8.777, 0.670, 0.065, 0.048, 0.000 | 0 | 0.99 | repaired | None/None/None |
| 6bd234c45e885cb2 | b CERT-REFUSED | c | 7.729, 0.225, 0.079, 0.000 | 0 | 0.98 | repaired | None/None/None |
| 9e7ae0e0f04255fa | b CERT-REFUSED | c | 0.544, 0.216, 0.130, 0.032, 0.000 | 2 | 0.95 | repaired | None/None/None |
| d49b986924cd5bed | b CERT-REFUSED | ttc | 0.269, 0.286, 0.000 | 0 | 0.96 | repaired | None/None/None |
| e7c485c237b352cd | b CERT-REFUSED | nc+ttc | 0.062, 0.057, 0.000 | 0 | 0.57 | repaired | None/None/None |
| 0b3d3edbc52d5ddd | c NO-FOLLOW (rule) | - | 0.063, 0.083, 0.065, 0.080, 0.063, 0.078 | 4 | 0.38 | - | None/None/3 |
| 11568fab86df5183 | c NO-FOLLOW (rule) | - | 0.473, 0.042, 0.053, 0.056, 0.054, 0.052 | 2 | 0.99 | - | None/None/None |
| 12b32c24d2cf5309 | c NO-FOLLOW (rule) | - | 0.011, 0.012, 0.013, 0.014, 0.015, 0.017 | 0 | 0.02 | - | None/None/None |
| 3c542be991515ccd | c NO-FOLLOW (rule) | - | 0.026, 0.028, 0.029, 0.027, 0.028, 0.026 | 4 | 0.60 | - | None/None/None |
| 50eebed7d9655bb5 | c NO-FOLLOW (rule) | - | 0.115, 0.097, 0.128, 0.150, 0.151, 0.146 | 2 | 0.91 | - | None/None/3 |
| 72be63ed04f15f97 | c NO-FOLLOW (rule) | - | 0.103, 0.129, 0.134, 0.134, 0.132, 0.135 | 4 | 0.57 | - | None/None/None |
| 7a7ff4cc1f1a501e | c NO-FOLLOW (rule) | - | 0.036, 0.039, 0.039, 0.038, 0.038, 0.038 | 4 | 0.95 | - | None/None/None |
| 7cb9260c8be35c5c | c NO-FOLLOW (rule) | - | 0.164, 0.119, 0.128, 0.152, 0.131, 0.126 | 4 | 0.95 | - | None/None/None |
| 8209b6700b535e8e | c NO-FOLLOW (rule) | - | 1.661, 0.240, 0.201, 0.195, 0.158, 0.153 | 4 | 0.98 | - | 20/16/10 |
| a305b24487775cfc | c NO-FOLLOW (rule) | - | 0.576, 0.340, 0.329, 0.392, 0.276, 0.307 | 0 | 0.87 | - | None/None/None |
| b2872eb05a455f97 | c NO-FOLLOW (rule) | - | 0.115, 0.074, 0.142, 0.116, 0.117, 0.116 | 2 | 0.97 | - | None/None/3 |
| c296ded44f9a509a | c NO-FOLLOW (rule) | - | 0.040, 0.049, 0.045, 0.050, 0.046, 0.051 | 4 | 0.94 | - | None/None/None |
| d1bae9e7d9785598 | c NO-FOLLOW (rule) | - | 1.027, 0.175, 0.161, 0.157, 0.092, 0.089 | 4 | 0.96 | - | 10/8/6 |
| e1b3e5e086a05ff9 | c NO-FOLLOW (rule) | - | 0.035, 0.029, 0.034, 0.029, 0.034, 0.029 | 3 | 0.60 | - | None/None/2 |
| f140d31943145068 | c NO-FOLLOW (rule) | - | 0.051, 0.049, 0.045, 0.029, 0.040, 0.036 | 1 | 0.97 | - | None/None/None |
| f1bdf7e4e9f5534e | c NO-FOLLOW (rule) | - | 0.035, 0.052, 0.048, 0.053, 0.049, 0.054 | 4 | 0.93 | - | None/None/None |
| fc7b8441f81b5296 | c NO-FOLLOW (rule) | - | 11.250, 0.847, 1.566, 0.722, 0.787, 0.858 | 2 | 0.96 | - | None/None/None |
| 4d31d76919495557 | e OTHER | - | 10.090, 0.845, 0.321, 0.074, 0.029, 0.030 | 0 | 0.98 | - | None/None/None |

## 5. N4-DM-only five (repaired by DAC-only, refused by N4-D)
2b04c510144e5545 (C), 6bd234c45e885cb2 (C), 9e7ae0e0f04255fa (C), d49b986924cd5bed (TTC), e7c485c237b352cd (NC + TTC). For all five the DAC = 1 candidate appears at iteration 2 to 4 and the loop stops because no outside corner remains; a larger margin or cap cannot change this (what-if: none of them certifies at any setting, `whatif_25.json`).

## 6. What-if, same 25 (DESCRIPTIVE ONLY, SEEN-ON-NAVTEST, not a result; `whatif_25.json`, `raw/whatif_s*.jsonl`)
Full certificate, tokens certified of 25 (DAC-only in brackets), margin x cap:
| margin \ cap | 5 | 10 | 20 |
|---|---|---|---|
| 0.05 m (A1 setting) | 0 [5] | 3 [9] | 4 [12] |
| 0.10 m | 0 [6] | 2 [10] | 3 [14] |
| 0.20 m | 4 [10] | 8 [15] | 8 [17] |
Control: 0.05 m / cap 5 / full = 0 (the A1 result reproduces). By cause at 0.20 m / cap 10: 6 rule-(c) + 2 rule-(a), 0 of (b). The 5 (b) tokens never certify under the full certificate at any setting; the lever for them is the guard, not the shift. Which tokens a larger margin rescues is a property of these 25; selecting margin or cap from this table would be tuning on navtest.

## 7. Limits
Builder-side scoring only (harness re-score of any what-if plan was not run); one navtest set; n = 25; no estimator or CI is attached because nothing here is an effect. Executed-shift ratios are read at one step (the worst step of iteration 0).

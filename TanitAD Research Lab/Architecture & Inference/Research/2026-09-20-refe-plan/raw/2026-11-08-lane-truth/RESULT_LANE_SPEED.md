# RESULT_LANE_SPEED: does u400's lane head read speed as danger? (2026-11-08)

**READ LABEL: CONDITIONAL. G-L1 FAILED as written (71.8 % within 1 mm vs the 99 % bar, n = 34 sets); G-L1b PASSED (AMENDMENT_2 section 4). Outcome under PREREG section 6: O1 (G = -0.243, interval [-0.384, -0.133], n_entered 153 of 154), 0.093 from the nearest threshold, so the 0.03 O4 rule does not trigger. This does NOT unlock arm S1L by itself:** `PREREG_SCORER1_AMENDMENT_2.md` blocks S1L while G-L1 fails; S1L needs G-L1 as written, or the PI's explicit acceptance of this disclosure.

Tier **T0** (table read, scorer-calibration diagnostic, not driving performance) on every number. Family touched: **LONGITUDINAL** (speed). LATERAL n/a (the lane distance is used only as a label ranked against speed; no lateral quantity is measured). TACTICAL n/a (no manoeuvre or goal decision involved). STRATEGIC **PAUSED** (PI 2026-09-19, ITEM 25): not passed, not omitted.
Evidence class: MEASURED (ours), artifacts in this package, unless marked INHERITED. Estimator for all intervals: **paired log-cluster bootstrap over logs, B = 10,000, seed 20260927, 95 % percentile** (`sel_common.log_boot_idx` / `boot_mean`). It answers "another draw of logs" only: one checkpoint, deterministic planner, one rebuild of the labels. It does not answer training variance (`H-ESTIM-SEED-1`), inference variance, or label-pipeline variance.

## 1. Retraction of the 21.9 % finding
AMENDMENT_1's "G-L1 fails at 21.9 %, max 1.673 m" was a join bug: key `(log, token)` instead of the row identity `(key, ckpt_step)`; the banked file has rank 0 and rank 1 rows per token (3,137 rows, 1,705 distinct (log, token)). Retracted as a statement about the labeller (AMENDMENT_2, `code/gl1_key.py`). Class: join key narrower than the row's identity. The trainer's loader is not affected.

## 2. Gates (MEASURED, T0; `gl1_gates.json`, `lane_speed_result.json`, `gl1b.json`)
Control: 40 held-out queue rows (16 logs); **34 labelled in both arms**, 6 failed in both (the same 6): `scenario build KeyError` on the log name. Intersection n = 34 sets, 2,176 proposals. Failed keys (log / token): `2021.06.14.19.22.11_veh-38_02275_02455 / 571d0412fde95f52`, `2021.06.14.14.03.45_veh-38_01624_01811 / 55feefc7d86c53b0`, `2021.06.14.18.13.35_veh-26_01537_01717 / 83a73d8c0412574f`, `2021.06.23.14.54.32_veh-16_00862_01000 / cf250195d1fc5235`, `2021.09.15.14.00.15_veh-28_01953_02255 / 37419afa8bd057f5`, `2021.09.15.14.00.15_veh-28_01953_02255 / 301334eae15b5a1b`. (Cause beyond the KeyError not investigated.)

| gate | value | bar | result |
|---|---|---|---|
| **G-L1 as written** | 71.8 % of proposals within 1 mm (613 of 2,176 unequal); soft target within 0.01 on 94.2 %; max abs d 0.0841 m; unequal ones: median 0.007 m, p90 0.046 m. Identical for the real and placeholder arm. | >= 99 % and >= 99 % | **FAIL** |
| G-L2 placeholder vs real teacher | 100 % identical, max 0.0 (n = 2,176) | 100 % within 1e-6 | PASS |
| G-L3 coverage | 154 of 154 CRUISE tokens labelled, all 64 finite | >= 150 | PASS (joined by key; order vs `tokens_cruise.json` not separately asserted) |
| G-L4 instrument | lane Spearman (legacy) -0.32872 = banked, n 154; lane dominant on 31 of 55 = expected | within 0.001; exact | PASS |
| **G-L1b** (AMENDMENT_2 s.3) | 34 sets, **33 entered**; mean abs diff in rho 0.0042; abs mean diff 0.0008; max 0.0466; mean rho rebuilt -0.1959, banked -0.1967 | <= 0.05, <= 0.03, >= 20 sets | **PASS** |

The mismatch is real but small and set-specific: the 613 unequal proposals sit in 17 of the 34 sets (the other 17 are exact); 229 of them have our value larger. The PREREG premise "cross-machine noise ~1e-7" was INHERITED and wrong.
**CHAIN3 variants** (8 rows attempted, **6 labelled, 2 failed** each; 384 proposals, a weak test): `nocache` 62.8 % within 1 mm, `norepair` 62.8 %; the baseline on the same 6 sets is also 62.8 %. Both are bit-identical to baseline, so **neither map/query caches nor `--repair-last-heading` explain the residual, and neither passes G-L1 as written.** No relabel with a variant was run (nothing to relabel with).

## 3. Primary read (CONDITIONAL; 154 CRUISE tokens, 51 logs; tie-correct average ranks; valid proposals; truth = soft target `clip(1-(d-0.5)/1.5, 0, 1)`)
| within-token Spearman vs v4/v0 | point | 95 % CI (log bootstrap) |
|---|---|---|
| lane head, entered tokens (n_entered 153) | -0.330 | [-0.457, -0.225] |
| lane TRUTH, same tokens | -0.087 | [-0.134, -0.047] |
| **G = head - truth (paired)** | **-0.243** | **[-0.384, -0.133]** |

One CRUISE token has a constant true target and does not enter (not imputed). Rule: G < -0.15 and hi < 0 gives **O1**; distance to the nearest threshold 0.093 > 0.03. Meaning: the lane head ranks faster proposals as worse than the rebuilt truth does; the truth itself slopes negative (-0.087), so about a quarter of the head's slope is geometry and the rest is excess. The PREREG O1 action (lane as a fourth pairwise head) is gated by the S1L block above.
Sensitivity (labelled, never decides), truth variable `-d`, n 154: truth -0.079 [-0.129, -0.035], G = -0.250 [-0.395, -0.135].
**Legacy regression arm** (`argsort(argsort)`, `p1_diag.py::spearman_rows`, n 154, does not decide): head -0.3287, truth -0.0811 [-0.124, -0.044], G = -0.2476 [-0.387, -0.137]; difference to tie-correct -0.0042.

## 4. Margin block (descriptive; slow-pick CRUISE tokens: pick v4/v0 < 0.85 and a faster valid proposal exists, n = 55 of 63; mean of d log agg)
| term | f_sc (scorer's best fast) | f_or (oracle) |
|---|---|---|
| margin | 0.1618 [0.118, 0.221] | 0.5092 [0.379, 0.650] |
| lane | 0.0608 [0.048, 0.076] | 0.1666 [0.118, 0.220] |
| NC / DAC / EP / TTC / C | 0.0207 / 0.0435 / 0.0094 / 0.0273 / 0.00001 | 0.0531 / 0.1683 / 0.0567 / 0.0645 / 0.00006 |
| lane share of margin (sum ratio) | 37.6 % | 32.7 % |
| dominant term counts | lane 31, DAC 17, TTC 4, NC 3 | lane 26, DAC 23, NC 4, TTC 2 |

**Identity check:** max |sum of terms - margin| = 5.0e-16 <= 1e-9, PASS (both comparators).
Lane truth on the 55: true target pick 0.739, f_sc 0.756 (f_or 0.846); head output pick 0.794, f_sc 0.748. Truth ranks f_sc strictly above pick on 14 and equal on 24; the head prefers the pick over f_sc on 38 of the 55 where truth has f_sc at least as high (14 strict, 24 ties). Where truth is equal the mean lane term is still +0.052 against f_sc: the head penalises the faster proposal where the truth sees no difference.

## 5. Next lever (Rule Zero)
G-L1b shows the 28 % label disagreement moves the speed-rank of the truth by 0.004 on average, so it is not what drives O1. What it blocks is the clean as-written read and the S1L gate. Cheapest experiment that could remove it: tabulate the 17 differing vs 17 exact control sets by log and map, then relabel the differing logs under the other nuPlan DB root (3,055 vs 2,979 DBs; one CPU run, about 15 min). If the root explains it, G-L1 passes as written and the S1L block lifts without a PI exception. If not, the choice is the PI's: accept the CONDITIONAL read with this disclosure. Not run in this package. Also open: why 6 control logs are absent from the scenario builder (all 154 CRUISE tokens built).

## 6. Pins
table.npz `cf30519051bea03b...`, results.npz `a8823edfa92c21671...`, tokens_cruise `57db811e644218c0...`, PREREG `e99b2576...` (unchanged), `sel_common.py` `50727023525c86b2`, `p1_diag.py` `7bd7220dffa1ae11`, `analyze_lane.py` (patched: intersection, read label) `11b471887db2944d`, `gl1b.py` `fc0321bdd707bdcb`, `gl1_gates.py` `f302e2094c4dd118`. The banked label file and the labeller pins are as in PREREG s.9 (not re-hashed here).

## 7. Master Mind note on the next lever (2026-11-08 ~17:25Z, MEASURED, `code/ll_split.py`, control sets n = 34)
The DB-root lever of section 5 is withdrawn. The 2026-10-12 smoke run indexed 2,979 DBs and today's indexes 3,055, and their labels are bit-identical (AMENDMENT_1), so the DB root does not change these labels.
Exact vs differing sets are mixed in every city (Las Vegas 8 / 7, Singapore 5 / 3, Boston 1 / 1, Pittsburgh 3 / 6) and inside 6 of the 11 logs. The disagreement is therefore not a map file, a city or a log property. It is a per-set effect between the machine that built the banked labels and this one. Caches and the repair flag do not explain it (n = 6, weak).
Blocker for G-L1 as written: relabel the control sets on the machine class that built the banked labels. No such machine is free: a training pod must not take added load. Unblocks when a Linux pod is idle (about 15 CPU-min), or by the PI accepting this CONDITIONAL read.

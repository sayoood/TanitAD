# Verdict reading: PREREG_V8F_N1DC. The skip from T's code into O's plan head gains nothing on D1-HO at either seed, and the eval-141 guard carries a cost at each seed, on different cells: N1DC-COST as the card is written (2026-11-20)

Card `raw/2026-11-20-n1dc/PREREG_V8F_N1DC.md`, sha256 2eae7964, FROZEN before any number. Executor of record T1-kin/hb0.2 (`v8f_t1_eval_hb.py --kappa-max 0.2`, blob sha256 74fd8445); T1-kin/old beside. Estimator: paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES" only; training variance is the two seeds. Evidence class: MEASURED (ours; dev-box CPU dumps of the two trained arms, zero training in the read).

## 1. Labels

| part | s0 | s1 | reads |
|---|---|---|---|
| D1-HO deciding cell, TAC-LAT N1DC - N1D (bar 0.042364) | +0.0045 [-0.0041, +0.0132], not separated | +0.0068 [-0.0030, +0.0156], not separated | no gain, no cost, at either seed |
| eval-141 guard, 24 cells, N1DC - N1D, floor F2 | cost: TAC-LON | cost: operative 0-2 s ade, along, headway | the guard does not hold at either seed |
| ROUTE-COST (RC-A, 12 cells per seed) | none | none | none |
| **outcome, card section 4 as written** | | | **N1DC-COST**: "guard cost at both seeds". Not adopted |

**The one reading choice, stated before the label is used.**
* The card's guard (section 3, "no cost at either seed") and the frozen reader (`n1dc_read141.py`, written before data: "The guard holds at a seed iff NO cell is a cost") define the guard **per seed**. A cost at s0 and a cost at s1 is then a guard cost at both seeds, the literal condition of N1DC-COST. This is the label of record.
* The N1DW card worded its cost row per **cell** ("a guard cell a cost at both seeds"), and the N1DW verdict read two different cells at two seeds as not meeting it. Under that wording no cell here is a cost at both seeds, and the label would be N1DC-NO-GAIN.
* **Both readings give the same decision:** N1DC is not adopted and nothing enters the recipe. They differ only in the row's named next step (section 8 carries both).

## 2. D1-HO (5,736 windows / 497 held-out episodes, T1-kin/hb0.2; `d1ho/RESULT_N1DC_D1HO.json` sha256 6f458165)

| row | s0 | s1 |
|---|---|---|
| accuracy N1R / N1D / N1DC | 0.6102 / 0.6346 / 0.6391 | 0.6348 / 0.6515 / 0.6583 |
| **DECIDING N1DC - N1D** | +0.0045 [-0.0041, +0.0132] | +0.0068 [-0.0030, +0.0156] |
| same, T1-kin/old beside | +0.0030 [-0.0058, +0.0115] | +0.0002 [-0.0104, +0.0110] |
| N1DC - N1R (beside, separated, below the bar) | +0.0289 [+0.0160, +0.0411] | +0.0235 [+0.0112, +0.0353] |
| T0 decoded-plan TAC-LAT, N1DC (N1D) | 0.6224 (0.6266) | 0.6379 (0.6349) |
| T0 N1DC - N1D | -0.0042 [-0.0091, +0.0007] | +0.0030 [-0.0048, +0.0105] |
| **Z-USE**: real minus the skip zeroed (`o_code.weight` = 0, the exact "output x 0") | -0.0005 [-0.0051, +0.0042] | +0.0049 [+0.0003, +0.0099], separated |
| D-channel zero twin, N1DC (N1D) | +0.0457 (+0.0377) | +0.0248 (+0.0190) |

* Bar 0.042364 = max(0.0383, F_HO 0.0424, |N1D draw| 0.0169, |N1DC draw| 0.0192).
* **The skip is barely read.** Zeroing it moves TAC-LAT by -0.0005 at s0 and +0.0049 at s1. The decoded plan does not approach the card's 0.73 target: 0.6224 / 0.6379 against N1D's 0.6266 / 0.6349.
* The D channel is still read, by about as much as in N1D (zero twin +0.0457 / +0.0248).

## 3. eval-141 guard (2,399 windows / 141 episodes; `eval141/RESULT_N1DC_EVAL141.json` sha256 cc822257)

Each cell reads N1DC - N1D, delta [95 % CI]. **C** = cost (separated beyond F2 in the worse direction), s = separated inside the floor or in the better direction.

| cell | hb0.2 s0 | hb0.2 s1 | old s0 (beside) | old s1 (beside) | floor F2 |
|---|---|---|---|---|---|
| operative 0-2 s ade_m | +0.0003 | **+0.0015 [+0.0003, +0.0029] C** | +0.0003 | **+0.0015 C** | 0.0005 |
| operative 0-2 s along_mae_m | +0.0001 | **+0.0012 [+0.0001, +0.0023] C** | +0.0001 | **+0.0012 C** | 0.0005 |
| operative 0-2 s headway abs err (m) | +0.0086 | **+0.0176 [+0.0000, +0.0411] C** | +0.0086 | **+0.0176 C** | 0.0039 |
| operative 0-2 s accel_mae | -0.0004 | -0.0074 s | -0.0004 | -0.0074 s | 0.0225 |
| tactical 2-6 s accel_mae | +0.0248 | -0.0736 s | -0.1293 s | -0.1011 | 0.1783 |
| tactical 2-6 s yaw_rate_mae | -0.0081 | -0.8162 s | -1.9363 s | -0.8499 | 8.1322 |
| TAC-LAT | -0.0029 | -0.0004 | +0.0004 | +0.0008 | 0.0067 |
| **TAC-LON** | **-0.0154 [-0.0279, -0.0042] C** | +0.0050 | -0.0025 | +0.0071 | 0.0150 |

The other 16 cells (FDE, speed, cross-track, heading, yaw rate and curvature at 0-2 s; ADE, FDE, speed, along, cross-track, heading, curvature, headway and time gap at 2-6 s) are not separated at either seed under either executor. The full 24-cell table is in the JSON (`guard_eval141_hb02_N1DC-N1D`, `beside_eval141_old_executor`).

* **Size.** The s1 costs are 1.5 mm ADE, 1.2 mm along-track and 1.8 cm headway error over 0-2 s. The s0 TAC-LON cost is -0.0154 against a floor of 0.0150. No cell is a cost at both seeds.
* **Executor.** The operative 0-2 s cells are identical under both executors (MEASURED; that the bound first acts after the first replan is the HYPOTHESIS for why), so the s1 cost is not the heading bound's. The s0 TAC-LON cost appears under hb0.2 only; old reads -0.0025, not separated. Under T1-kin/old alone the guard carries a cost at s1 only.
* **Families.** LONGITUDINAL: cost at s1 (along, headway). LATERAL: none. TACTICAL: cost at s0 (TAC-LON). ADE / FDE: cost at s1 (operative ADE). Distance-keeping is computed (headway, time gap; operative and tactical).

## 4. STRATEGIC in full (T0 at the window's first tick; `eval141/N1DC_STRATEGIC_contrast.json` sha256 efdf1c3e, S1-slot `N1DC_STRATEGIC_S1slot.json` b8b65f14)

* **Route setting, RC-A** (position and heading error at 30 / 50 / 80 m): N1DC - N1D, no cell separated at either seed. ROUTE-COST: none (also none against N1R).
* **Route following, S_CKPT** under real, donor and zero nav: N1DC - N1D not separated at s0. At s1, at 30 m, -0.0533 [-0.1061, -0.0130] under donor nav and -0.0667 [-0.1169, -0.0210] under zero nav; real nav not separated. These rows sit beside the ROUTE-COST rule, which reads RC-A only.
* **S1-slot view:** N1DC reads S-PARTIAL (real - donor separated at both seeds, not above the best constant at either), the same label as N1D. Controls K7 and K8 read their expected values.
* Fog is unscored (PI 2026-11-17).

## 5. Controls (all read their committed values)

* D1-HO, both executors: K-RE PASS, K1 PASS, K-CONST 0.338215, K-T0 PASS. REPLAY of N1DW GREEN, its mutant RED.
* eval-141: K-RE PASS, K1 self-pair exactly 0, K-HB0-141 PASS (the wrapper with `--kappa-max none` reproduces the banked old N1D_s0 dump array for array), Z-EVAL exactly 0.0 at all four hb0.2 arms (2,160 zero-history windows / 130 episodes), K-FL-HB 8 of 8 (section 6).

## 6. The first-run fix in the eval-141 reader (before any guard number was read)

* The first run stopped in the loader. `fl`, the FLOOR planner's executed path, of the hb0.2 dumps differed from the old-executor reference dump C0_s0.
* MEASURED (`eval141/fix/check_fl.json`): the floor is rolled through the bounded executor too. eid, clip_index, ws, g and v0 equal C0_s0 in all 8 hb0.2 dumps. `fl` differs in 324 of 2,399 windows and is identical across all 8 hb0.2 dumps. The 2 old-executor dumps match on all six arrays. `fl` enters no cell.
* Fix (`eval141/fix/n1dc_read141.diff`): hb0.2 dumps check `fl` against the hb0.2 floor (N1D_s0 hb0.2) and every other array against C0_s0. New control K-FL-HB. Old-executor dumps are unchanged. No bar, cell, arm or rule changed.
* The chain's FAIL and DONE lines went through `tee -a`, which never reached the log under the WMI launch, so a waiter on the log line would have run to its cap. Now plain appends (`eval141/fix/run_n1dc141.sh.pre` against `eval141/run_n1dc141.sh`).
* Class: an instrument written for one executor carries an equality check whose reference belongs to that executor.

## 7. What this says

* **Access was not the limit.** A direct, trained path from T's code into O's plan head is almost unused (Z-USE about 0) and leaves the decoded plan at about 0.63. Whatever keeps O at 0.63 while T's code decodes 0.73 at T0, it is not the lack of a path from T's code.
* **The guard costs are small, cell-unstable across seeds and partly executor-dependent.** They are costs under the card's rule and are reported as such. They do not change the decision, which is "not adopted" under either reading.

## 8. Next (RULE ZERO)

* **The card's row, as written (N1DC-COST):** the cost family gets a card. The cheapest form is zero-training: eval-141 dumps of the zeroed-skip copies (`N1DC_s{0,1}_zuse`, already built on the pod). They would show whether the s1 operative cost runs through the skip's output or through the weights the skip changed in training. CPU, about 20 min on the dev box.
* **The deciding question stays open, and NO-GAIN's committed lever addresses it:** an auxiliary route-decision head on O's plan feature (kinematic label, CE), the T0T1 card's lever, under its own card. It needs a PI go for pod time.
* Nothing enters the recipe. N1D stays the reference arm.

# VERDICT, STOPPERC Q5 (416x1024 source, AMENDMENT 7): RES416-FLAT

Read 2026-11-16 12:13 local by `q5_read.py` (build of record in `../build/`), MM-checked. Card: `AMENDMENT_7_Q5_416.md` (sha256 `6be1adf9...`), frozen 2026-11-15 before any 416 token was encoded or scored.

* **Outcome: RES416-FLAT** (MEASURED, `run/RESULT_Q5.json`). No R(g, f, s) holds at any grid (40, 160), family (linear, MLP) or sample (S1, S2): no paired dAUC(Vg@416+E − Vg@256p+E) reaches the +0.020 bar with a CI above 0. The largest point is +0.0277 [−0.0157, +0.0737] (S2, g 160, MLP).
* **Tier:** frozen-feature probe. It is a diagnostic of what the inputs carry, never driving performance.
* **Estimator:** `taniteval.ci.stat_cluster_bootstrap`, B 2,000, seed 0, over 139 eval episodes, one call per sample. It answers "another draw of episodes". S2 answers "another draw of the training sample". The probes are deterministic, so there is no inference-seed term.

## Controls (all ok; a miss would have refused the read)
| id | value |
|---|---|
| C0 | 0 differing bytes (C0 refit bitwise, `c0/refit/`) |
| C1 | identity read 0.0 [0.0, 0.0] → RES416-FLAT |
| C2 | planted direction: AUROC 1.0 (minimum over samples and both families) → RES416-ADDS-40, so the read can see an ADDS when one exists (`c2/RESULT_C2.json`) |
| C3a-e | 1.0 / 0.0 / 526 / 0 / true |
| C4 | E bitwise equal to C0 on 4 of 4 dirs |
| C5 | label-shuffle null: real AUROC 0.8668 against a null max of 0.5950 |
| C6a / C6b / C6c | 1, within AMENDMENT 7a's bound of at most 1 / MAE 0.8206 on the 0-255 scale / kinematics bitwise |
| C7 / C8 | tests pass / 3 clips bitwise |
| C9 | eval 2,365 windows, 131 positives, 139 episodes; train 16,000 windows, 4,000 positives per sample |

## Paired differences (AUROC, 139 episodes; R = 416 − 256p at the same grid, A = 416 − E)
| sample | cell | linear | MLP |
|---|---|---|---|
| S1 | R, g 40 | −0.0020 [−0.0080, +0.0047] | +0.0077 [−0.0222, +0.0352] |
| S1 | A, g 40 | +0.0072 [−0.0320, +0.0450] | +0.0076 [−0.0436, +0.0578] |
| S1 | R, g 160 | −0.0034 [−0.0174, +0.0105] | −0.0177 [−0.0547, +0.0168] |
| S1 | A, g 160 | −0.0012 [−0.0484, +0.0434] | −0.0089 [−0.0688, +0.0457] |
| S2 | R, g 40 | −0.0009 [−0.0085, +0.0070] | +0.0056 [−0.0450, +0.0593] |
| S2 | A, g 40 | +0.0147 [−0.0260, +0.0530] | −0.0041 [−0.0660, +0.0561] |
| S2 | R, g 160 | −0.0009 [−0.0112, +0.0101] | +0.0277 [−0.0157, +0.0737] |
| S2 | A, g 160 | +0.0065 [−0.0458, +0.0535] | −0.0187 [−0.0791, +0.0366] |

The beside rows (mean token, ELIM, HU against BAL-NULL) are printed in `RESULT_Q5.json` under `beside`. They decide nothing, and none separates.

## Four families
* **TACTICAL:** stop-point AUROC, n 2,365 windows / 131 positives / 139 episodes (the table above).
* **LONGITUDINAL:** n = 0. The probe has no speed or distance-keeping output.
* **LATERAL:** n = 0. The probe has no path output.
* **STRATEGIC:** NOT COMPUTABLE, n = 0. A frozen-feature stop probe has no route, goal, intent or detected-information output.
  * The reader, frozen 2026-11-15, prints "PAUSED (PI item 25)". This read finished after the PI ended that pause on 2026-11-16, so its STRATEGIC entry is the reasoned n/a given here, not the pause.

## What it settles, and what comes next (committed in the card, section 2.4)
* **The stop-perception line on PhysicalAI-AV ends with this row.** It has four eliminations, each logged:
  1. the 256 grids (S0-P);
  2. a trained adapter with a stop auxiliary (A);
  3. a trained encoder (S0-L);
  4. 2.6x the pixels on the same field (Q5).
* **Next lever (committed):** an input that carries the stop location.
  * On PhysicalAI-AV that exists only as an ego-future label: admissible as a label, never as an input.
  * On nuPlan / NAVSIM, the map stop line exists. REFe `AMENDMENT_6_STOP_LINE.md` defines it as the start of the governing lane connector.
  * The next arm is a **map-carried stop-line input on that rig, under its own card**.
* **Blocker:** `QUESTIONS_416.md` Q2 is the PI's. Per the MM freeze of AMENDMENT 7, nothing that depends on it runs before the answer.
* Stage Q5-1, the dense 416 cache and its training arms, is not licensed and does not run.

# VERDICT_PLANTEST3 (Master Mind, 2026-11-09)
Card `PREREG_V8F_PLANTEST3.md` (`1f9609b4`), full tables `RESULT_PLANTEST3.md`, six controls green (`RESULT_PLANTEST3_controls.md`, 16 tests passed). **Tier T0, oracle (true-future) goal: a world-model diagnostic, never a driving result.** MEASURED; every interval is the paired episode-cluster bootstrap over 71 SCORE episodes (994 windows), B 2,000. It answers another draw of episodes; both training seeds are required, and the planner-seed replicate is printed beside.

## What the card's table says
| family | outcome, all 5 arms per family (FLAT1 DEC-END and DEC-PATH, plain and shrink; TD3 DEC-END) | HOLD | closest arm to HOLD (G-TRUE minus HOLD) |
|---|---|---|---|
| LONGITUDINAL (acceleration MAE, 12 logged actions) | **NO-GAIN** at both seeds | 0.3348 m/s^2 | FLAT1 DEC-END shrink: s0 -0.0012 [-0.0060, 0.0031], s1 +0.0017 [-0.0032, 0.0072]; both flagged HOLD-COLLAPSE, so neither counts as planning |
| LATERAL (yaw-rate MAE) | **NO-GAIN** at both seeds | 0.0256 rad/s | FLAT1 DEC-END shrink: s0 +0.0026 [0.0014, 0.0040], s1 +0.0021 [0.0010, 0.0034] (worse) |
| TACTICAL | not computed, n = 0: this read has no decision output | | |
| STRATEGIC | PAUSED (PI item 25) | | |

* **DECODER-WEAK at both seeds** (speed R2 0.271 / 0.314, yaw-rate R2 -0.122 / 0.268; bars 0.70 / 0.50).
* **KIN-CEILING does not beat HOLD:** actions computed from the decoded true-future quantities read 4.04 to 4.11 m/s^2 worse than HOLD and 0.035 to 0.037 rad/s worse. With R2 this low, differencing the decoded speeds amplifies the readout's error.
* **HIER-ADDS: no.** TD3 minus FLAT1 under G-TRUE is +0.049 to +0.056 m/s^2 and +0.0042 to +0.0048 rad/s (worse, separated, both seeds).

## What is positive and what limits it
* **The goal steers the plan.** G-TRUE minus G-RAND is separated-better on all 20 rows (for example -0.0595 [-0.0853, -0.0367] m/s^2, s0 FLAT1 DEC-END plain). The planners respond to the goal; they do not get closer to the logged actions than holding does.
* The planner-seed replicate differs by at most 0.0013 m/s^2 and 0.0002 rad/s, far below every effect in the table.
* Lambda came out at the top of the grid (1) in all 16 TUNE selections (8 arm and seed rows, 2 families), and TUNE MAE falls steadily as lambda rises. Lambda weights a penalty on deviation from HOLD, so the cost's pull away from HOLD lowers the score here, and a larger lambda would only move the plan toward HOLD.
* Decoded distance to the goal does not fall along the plan (Tab. 10 statistic -0.23 to +0.05 on the decoded cost), although planning lowered the cost below the best initial candidate in at least 99.9 % of windows.
* HOLD is a strong reference at this horizon: 12 knots at 5 Hz is 2.4 s, and the logged actions change slowly there. The MAE reads whether a plan recovers the logged actions, not whether it drives well.

## What the card triggers, and what it means here
NO-GAIN on every decoded arm, KIN-CEILING not beating HOLD, DECODER-WEAK: **"the decoded quantities are the limit; next is a nonlinear readout and multi-frame decoding on the ladder latents."**

A measurement sharpens this. **The ladder keeps the adapter fixed:** in all four finished ladder checkpoints, all 5 adapter tensors equal M2b's and all 4 `wm.proj` tensors differ (`theory/decinfo/COMPARE_LADDER_ADAPTER.json`). So a ladder z1 is a learned 128-D re-projection of the same 256-D mean-pooled token. The card's lever therefore runs, and a further question has to run beside it: at which point on the vision chain do speed and yaw rate get lost? That is `PREREG_V8F_DECINFO.md`, with its own committed outcomes.

Harness read on the ladder world models: still owed, after the ladder ends. PLANTEST and PLANTEST2 stay required gates.

## Where the data lives
Plan arrays and decoders (`runs/`, 62 files): PRIVATE `Sayood/tanitad-v8f-plantest3-runs`, verified by sha256 after re-download (see `HF_RUNS.md`). Per-file hashes: `MANIFEST_RESULT.txt`.

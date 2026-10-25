# PREREG_V8F_SFRESH: is the strategic layer worse than nav-only kinematics, or is it scored at a stale tick? (written 2026-10-25, before any SFRESH data)

**Status: pre-registered.** Eval-only on the BANKED M2b s0 / s1 checkpoints, with zero training. It runs under `PI-V8F-2026-10-25b`: on box 2 between training jobs, or on the dev box's CPU if the eval cache fits the disk budget.

## Why (MEASURED on banked numbers; `raw/full/eval/compare.json`, `floors_report.json`; cross-checked by the 2026-10-25 report draft)

`H-MINI-STRAT-1`'s first clause reads **S worse than the nav-only kinematic baseline at 50 m** at both seeds and in both arms, and it is separated:
* M1: learned S − baseline = **+3.37 [+2.14, +4.53] m**;
* M2b: **+3.28 [+2.06, +4.47] m**.

The baseline is 4.73 m. At 80 m the difference is ≈ +0.2 to +0.4 m and not separated. The nav-use control holds: shuffling nav costs +4.25 m at 50 m, separated.

**The confound.** In the default eval (`--ages cycle`), S's checkpoints come from its HELD tick, up to 1.2 s old. They are re-expressed to "now" and scored against checkpoints defined from "now" (`v8f_mini_eval._ckpt_to_now`). S therefore pays for the distance travelled since its tick, and the baseline does not. The register row still reads OPEN, and a decision on the strategic layer must not rest on a metric that may be scoring staleness.

## The read (ONE change: when S ticks; no training)

* **Arm F:** `v8f_mini_eval.py run … --ages fresh` on M2b s0 and s1. S and T tick at "now". Every other argument is the banked KIND own-code eval's, verbatim.
* **Reference:** the banked `--ages cycle` evals of the same checkpoints and the nav-only kinematic baseline, all on the same windows.
* Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes. Tier: T1-class open loop, DIAGNOSTIC: fresh ages are not the deployed schedule.

| read | definition | role |
|---|---|---|
| **Q-STALE** | S checkpoint position error at 50 m: fresh − cycle | is the stale tick a cost at all? |
| **Q-LAYER** | S checkpoint position error at 50 m: fresh S − nav-only baseline | **the decisive read** |
| beside | the same at 80 m; heading error; the four families under fresh ages (descriptive) | |

| outcome | condition | next (committed) |
|---|---|---|
| **STALENESS ARTEFACT** | Q-LAYER ≤ 0 (point) at both seeds, and Q-STALE < 0 separated at both seeds | STRAT-1's failure was the metric. Clause 1 is re-read on fresh ages: it PASSES iff Q-LAYER < 0 separated at both seeds. Then the deployed-schedule cost (Q-STALE) becomes the strategic lever: a faster S tick or a re-anchored checkpoint |
| **S GENUINELY WORSE** | Q-LAYER > 0 and separated, at both seeds | the S layer as built adds nothing over nav-only kinematics at 50 m. The PREREG clause *"if the scene adds nothing, the nav-only layer ships"* goes to the PI as a decision, with a pre-registered arm "S as a residual on the nav-only baseline" (an architecture change: needs the PI's go) |
| **NOT ESTABLISHED** | anything else | report it as such. Seeds come before windows (`H-ESTIM-SEED-1`) |

⛔ No bar moves after data. The 80 m reading stays beside: it was not separated on the banked numbers, and it decides nothing here.

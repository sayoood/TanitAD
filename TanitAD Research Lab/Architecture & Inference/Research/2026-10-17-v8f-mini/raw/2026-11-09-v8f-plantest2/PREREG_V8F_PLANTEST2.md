# PREREG_V8F_PLANTEST2: does a seeded and regularised solver make the level planners beat holding the action? (Master Mind, 2026-11-08, written before any PLANTEST2 code or run)

## Why
`PREREG_V8F_PLANTEST.md` read NO-GAIN on LONGITUDINAL and LATERAL for TD3 and FLAT1 at both M2b seeds (`theory/plantest/RESULT_PLANTEST_body.md`, tier T0, EC-BOOT over 141 episodes). G-TRUE beats G-RAND, so the plans depend on the goal. But G-TRUE is far worse than HOLD (M2b_s0 TD3 acceleration MAE 1.4228 against HOLD 0.3296 m/s^2). The planners start from wide prior candidates and carry no penalty for leaving a plausible action. That card commits the next lever on NO-GAIN: the planner's cost and solver, then the world model. Tier: **T0, an oracle-goal diagnostic, never a driving result.**

## Objects
As PLANTEST: M2b s0 and s1 world models, zero training, the 1,976 windows and 141 episodes, G-TRUE, G-RAND, HOLD, PRIOR, the same action prior fitted on TRAIN, the same metric code. Planners: TD3 and FLAT1 (FLAT1 is the flat arm).

## The changes (one family of solvers, nothing else moves)
* **SEED:** candidate 0 is HOLD at level 1. The other candidates are HOLD plus Gaussian noise with the prior's std (same 1.5x width and clipping as PLANTEST). Upper levels of TD3 seed their macro-actions from the macro encoding of the HOLD sequence.
* **REG(lambda):** SEED plus a cost term on level-1 primitive actions, lambda times the mean squared deviation from HOLD in units of the prior std. Lambda comes from the fixed grid {0, 0.01, 0.03, 0.1, 0.3, 1.0}. Lambda 0 is SEED alone.

## Tuning split (a read that tunes on what it scores is not admissible)
The 141 episodes are split once by a stable hash of the clip id (salt `plantest2-v1`) into TUNE and SCORE halves. Lambda is chosen per planner, training seed and family on TUNE, under G-TRUE, by the lowest action MAE. Every reported number is on SCORE only, and G-RAND uses the lambda chosen for G-TRUE.

## Read (paired episode-cluster bootstrap over the SCORE episodes, B 2,000, seed 0, both training seeds required)
Deciding, per family: action MAE as PLANTEST. Beside, no bar: speed, heading and ADE at 2.4 s, the distance of the plan from HOLD, the chosen lambda, cost monotonicity, a planner-seed replicate. TACTICAL: not computed (n = 0, no decision output). STRATEGIC: PAUSED (PI item 25).

## Committed outcomes (per planner and family, both seeds)
* **PLANNABLE:** G-TRUE minus HOLD is separated-better and G-TRUE minus G-RAND is separated-better.
* **GOAL-BLIND:** G-TRUE minus G-RAND is not separated.
* **NO-GAIN:** G-TRUE minus G-RAND is separated-better, and G-TRUE minus HOLD is not separated-better.
* **HOLD-COLLAPSE (flag on any of the above):** the plan's mean absolute distance from HOLD is below 10 % of the prior std. Then the planner is returning HOLD and the row does not count as planning.
* **HIER-ADDS (beside TD3):** TD3 minus FLAT1 under G-TRUE is separated-better at both seeds.

## What each outcome triggers (committed now, Rule Zero)
* **PLANNABLE on either family:** this solver becomes the HWM-PLAN solver, and it is rerun on the new ladder.
* **NO-GAIN at every lambda:** the next lever is the cost space (distance in decoded, action-relevant quantities instead of the raw latent), read on the ladder world model.
* **GOAL-BLIND:** action sensitivity first (inverse-dynamics weight, action-diverse sampler).
* **Every outcome:** PLANTEST and this solver are required gates of the ladder arms. The ladder is not dropped on a negative here, because these world models have a one-step loss and a 0.8 s top step.

## Controls (the read is INVALID if one fails)
1. **Reduction:** with lambda 0 and the seed replaced by the PLANTEST initialisation, the code reproduces a PLANTEST run's plans bit for bit on a 16-window subset.
2. **Limit:** with a very large lambda the plan equals HOLD to within a tolerance fixed in the implementation choices.
3. **Planted dynamics:** PLANTEST control 1 passes with SEED and REG, and a mutation disconnecting the actions fails it.
4. **Split:** TUNE and SCORE share no episode (asserted), and no SCORE window is read during tuning (mutation goes red).
5. **Goal plumbing and bit-repeat:** PLANTEST controls 2 and 3 on the new solver.

## Implementation choices
The implementer writes `theory/plantest2/IMPL_CHOICES_PLANTEST2.md` (the macro seeding, tolerances, batching, the split) with its sha256 before any eval window is read.

## Log
* 2026-11-08 16:05:23 frozen: prefix 4444 bytes, sha256 d8217dfa6c1e334390bd9cc263c8f61e1cbd3390f2c71e1db477e384143a1c89 (everything above this section). No PLANTEST2 code or run existed at this time. PLANTEST's deciding rows existed and motivate the card, stated in its Why.

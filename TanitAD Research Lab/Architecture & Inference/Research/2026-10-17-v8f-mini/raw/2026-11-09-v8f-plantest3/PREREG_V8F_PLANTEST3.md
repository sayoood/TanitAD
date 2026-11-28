# PREREG_V8F_PLANTEST3: does a cost in decoded, action-relevant quantities make the level planners beat holding the action? (Master Mind, 2026-11-09, written before any PLANTEST3 code or run)

## Why
`PREREG_V8F_PLANTEST2.md` (`d8217dfa`) read NO-GAIN on LONGITUDINAL and LATERAL for TD3 and FLAT1 at both M2b seeds (`raw/2026-11-09-v8f-plantest2/`, T0, EC-BOOT over 71 SCORE episodes), at every lambda of its grid. That card commits this lever: the cost space. Two measured reasons. (1) The planners lower their cost in every window and the goal steers them, but the level-1 latent distance hardly falls along the true future (monotonicity 0.1596 to 0.2557 for TD3, -0.1746 to -0.1475 for FLAT1, against 0.8948 to 0.9290 at levels 2 and 3). (2) `H-V8F-WMACT-1`: level 1 aims right and steps about twice too far (TRAIN-fitted shrink c 0.496 / 0.526). Tier: **T0, an oracle-goal diagnostic, never a driving result.** This card reads M2b first; the same harness is the read on the ladder world models when they finish, as PLANTEST2's trigger states.

## Objects
As PLANTEST2: M2b s0 and s1 world models, zero training, the same windows, the same TUNE and SCORE halves (salt `plantest2-v1`), G-TRUE, G-RAND, HOLD, the TRAIN action prior, the SEED + REG(lambda) solver, the same metric code. FLAT1 is the primary planner; TD3 is read after it (see Order).

## The decoder
`D_l`: a linear ridge readout from a level-l latent to two action-relevant quantities at that latent's time: speed (m/s) and yaw rate (rad/s), each divided by its TUNE standard deviation. Fitted per training seed on encoded (not rolled-out) latents of the TUNE windows only, ridge 1e-3 times the sample count. No gradient training takes place. Gate read before planning, on a second episode-disjoint split inside TUNE: held-out R2 of speed and yaw rate on encoded latents and on the rollout at k = 12 under logged actions.

## The arms (one change: the planner's cost; level 1 only for TD3)
* **DEC-END:** distance between `D_1` of the rolled-out level-1 state at the 12th knot and `D_1` of the goal latent at the 12th knot. Same goal information as PLANTEST2.
* **DEC-PATH:** the same distance summed over all 12 knots, against the decoded true-future latents at each knot. An oracle path goal: a ceiling read.
* **SHRINK(c):** factor on either arm: the level-1 rollout steps by c times the predicted change, with c from `RESULT_WMACT_SHRINK.json` per training seed (0.496 / 0.526). c = 1 is the plain rollout.
* **TD3:** levels 2 and 3 keep their latent cost. Level 1's cost becomes the squared distance under `D_2` between the level-2 lift of the level-1 rollout and the level-2 sub-goal. DEC-END form only.
* **KIN-CEILING (no world model, no planner, beside):** actions computed directly from the decoded true-future speed differences and yaw rates at the knots. It separates two causes: if this does not beat HOLD, the decoded quantities are too weak; if it beats HOLD and the world-model arms do not, the world model's action response is the limit.

## Tuning and order
Lambda from {0, 0.1, 1}, chosen on TUNE under G-TRUE by the lowest action MAE, per arm, training seed and family. Every reported number is on SCORE only; G-RAND uses the G-TRUE lambda. **Order (priority, a stopped run still yields a read):** (1) decoder gate; (2) KIN-CEILING; (3) FLAT1 on the four arms DEC-END and DEC-PATH, plain and SHRINK; (4) TD3 on SCORE only for DEC-END, in whichever of plain or SHRINK FLAT1 chose per seed and family, at FLAT1's lambda.

## Read (paired episode-cluster bootstrap over the SCORE episodes, B 2,000, seed 0, both training seeds required)
Deciding, per family: action MAE as PLANTEST2 (LONGITUDINAL acceleration, LATERAL yaw rate, 12 logged actions). Beside, no bar: speed, heading and ADE at 2.4 s, distance of the plan from HOLD, chosen lambda, decoded-cost monotonicity, a planner-seed replicate. TACTICAL: not computed (n = 0, no decision output). STRATEGIC: PAUSED (PI item 25). The interval answers another draw of episodes; the two training seeds and the planner-seed replicate are read beside it.

## Committed outcomes (per arm, planner and family, both seeds)
* **PLANNABLE:** G-TRUE minus HOLD is separated-better and G-TRUE minus G-RAND is separated-better.
* **GOAL-BLIND:** G-TRUE minus G-RAND is not separated.
* **NO-GAIN:** G-TRUE minus G-RAND is separated-better, and G-TRUE minus HOLD is not separated-better.
* **HOLD-COLLAPSE (flag):** the plan's mean absolute distance from HOLD is below 10 % of the prior std; the row does not count as planning.
* **HIER-ADDS (beside TD3):** TD3 minus FLAT1 under G-TRUE is separated-better at both seeds.
* **DECODER-WEAK (flag on the whole read):** held-out R2 on encoded latents below 0.70 for speed or below 0.50 for yaw rate at either seed.

## What each outcome triggers (committed now, Rule Zero)
* **PLANNABLE on DEC-END in either family:** this cost becomes the HWM-PLAN level-1 cost and is rerun on the ladder world models.
* **PLANNABLE on DEC-PATH only:** a single end goal is too little; the next lever is a goal path, supplied by the tactical level's goal knots up to 6 s.
* **NO-GAIN on every decoded arm, KIN-CEILING beats HOLD:** the world model's action response is the limit; next is the action-sensitivity lever (inverse-dynamics weight, action-diverse sampler) on the ladder arms.
* **NO-GAIN on every decoded arm, KIN-CEILING does not beat HOLD, or DECODER-WEAK:** the decoded quantities are the limit; next is a nonlinear readout and multi-frame decoding on the ladder latents.
* **GOAL-BLIND:** action sensitivity first.
* **Every outcome:** the harness is read on the ladder world models when they finish. PLANTEST and PLANTEST2 stay required gates.

## Controls (the read is INVALID if one fails)
1. **Reduction:** with the latent cost selected, the code reproduces a PLANTEST2 run's plans bit for bit on a 16-window subset.
2. **Planted dynamics:** a planted linear world model with a planted linear decoder recovers the planted actions under DEC-END and DEC-PATH, thresholds as PLANTEST2's control; a mutation disconnecting the actions fails it.
3. **Decoder:** a synthetic known linear map is recovered; shuffled targets read held-out R2 at or below 0.02.
4. **Split:** the decoder fit and lambda choice read no SCORE window (asserted; a mutation goes red).
5. **Shrink identity:** c = 1 reproduces the plain rollout bit for bit; c = 0 gives a constant rollout and a decoded-cost gradient of exactly 0.0.
6. **Goal plumbing and bit-repeat:** PLANTEST controls 2 and 3 on the new cost.

## Implementation choices
The implementer writes `theory/plantest3/IMPL_CHOICES_PLANTEST3.md` (where speed comes from on the eval windows, batching, tolerances, the inner split) with its sha256 before any eval window is read.

## Log
* 2026-11-09 00:53:38 frozen: prefix 6783 bytes, sha256 1f9609b40eaf7d3f5862dcd7fbd42c8dea852aa46965e5fea8da29fbd9b348f9 (everything above this section). No PLANTEST3 code or run existed at this time. PLANTEST2's deciding rows existed and motivate the card, stated in its Why.

# PREREG_V8F_PLANTEST: is today's hierarchical latent space plannable? (Master Mind, 2026-11-08, written before any PLANTEST code or run)

## Why
The PI ruled on 2026-11-08 that the multi-hierarchy world-model planner is the baseline of the redesign (`theory/horizon_review/REQ_HORIZONS.md`, R-PLAN). Rung gamma of the October plan (`Project Steering/PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` section 4.3, gradient planning through the levels) was never run on driving. The instrument exists and is unused: `HJEPAStack.plan_topdown` (`stack/tanitad/models/hjepa.py`, Eqs. 4 to 6 of H-JEPA, library key 2610.06805). This card asks the cheapest question first, with zero training: **given the true future as a goal, does planning through the learned levels recover what the human did, better than controls?** It is experiment E4 of `research_imagination/PLAN_IMAGINATION.md`, and it is the baseline the new ladder must beat.

Tier: **T0, a world-model diagnostic, never a driving result.** The true-future goal is an oracle and is never an inference input. A deployable arm needs a predicted goal, which is a later card.

## Objects (zero training of the model)
* Models: M2b s0 and s1 world models (`E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/M2b_s{0,1}/ckpt_last.pt`), levels at 0.2 / 0.4 / 0.8 s, with the inverse-dynamics term.
* Windows: IMAG1B's eval-141 deciding set (`theory/wmact/latents.pt`, 1,976 windows, 141 clips). If a window has no encoded frame 2.4 s after NOW, it is dropped and counted.
* Action priors: `fit_action_prior` on TRAIN windows only.

## Planners
* **TD3:** `plan_topdown` from level 3, `top_horizon` 3 (2.4 s), with subgoal counts chosen so level 1 plans the full 12 primitive actions (2.4 s). `num_candidates`, `n_iters` (at most 90), `lr` and the action bounds are fixed by the implementer from the paper's App. E before any eval window is read, and recorded in `theory/plantest/IMPL_CHOICES_PLANTEST.md` with its sha256.
* **FLAT:** the same solver at level 1 only, 12 steps, with the goal lifted into level 1 (the true-future level-1 latent). Same candidates and iterations as TD3's level 1.

## Goals
* **G-TRUE:** the encoded true future 2.4 s after NOW, in the planner's goal level.
* **G-RAND (control):** the same quantity from a window of a different episode.
* **G-STAY (control):** the current latent.

## References (no planning)
* **HOLD:** the current measured action (a0, v0*kappa0) held for 12 steps.
* **PRIOR:** the fitted prior mean.

## Read (paired episode-cluster bootstrap over the 141 episodes, B 2,000, seed 0; both training seeds required)
* **Deciding:** action recovery against the logged actions, over the 12 steps: acceleration MAE (LONGITUDINAL) and yaw-rate MAE (LATERAL).
* **Beside, no bar:** speed error at 2.4 s, heading error at 2.4 s, ADE at 2.4 s of the unicycle integration of the plan; cost monotonicity per level (H-JEPA Tab. 10); the fraction of windows where planning lowered the cost.
* Family statement: LONGITUDINAL and LATERAL as above. TACTICAL: not computed, this read has no decision output (n = 0, reason stated). STRATEGIC: PAUSED (PI item 25).

## Committed outcomes, per planner (both seeds required, per family)
* **PLANNABLE (family):** G-TRUE minus HOLD is separated-better **and** G-TRUE minus G-RAND is separated-better.
* **GOAL-BLIND (family):** G-TRUE minus G-RAND is not separated: the plan does not depend on where the goal is.
* **NO-GAIN (family):** G-TRUE minus G-RAND is separated-better, and G-TRUE minus HOLD is not separated-better: the goal matters, but planning does not beat holding the action.
* **HIER-ADDS (beside TD3):** TD3 minus FLAT under G-TRUE is separated-better on a family at both seeds.

## What each outcome triggers (committed now, Rule Zero)
* Every outcome: this test becomes a required gate of the new ladder (band-matched steps and multi-step loss), rerun with the same instrument. The ladder is not dropped on a negative here, because this card reads a world model with a 0.8 s top step and a one-step loss, which the PI has already rejected.
* **PLANNABLE** on either family: the top-down scorer and gradient-refinement arms are designed on this instrument at once.
* **GOAL-BLIND** on a family: the next lever is action sensitivity at that family (the inverse-dynamics weight and the action-diverse sampler), read by `action_sensitivity`, before the planner is touched.
* **NO-GAIN** on a family: the next lever is the planner's cost and solver (the cost space, the prior and the action bounds), then the world model.

## Controls (the read is INVALID if one fails)
1. **Planted dynamics:** on a synthetic linear system with a known action effect, in the same code path, the planner recovers the planted actions to within a tolerance fixed in IMPL_CHOICES before the run. A deliberate-regression arm (actions disconnected from the predictor) must fail it.
2. **Goal plumbing:** swapping G-TRUE with G-RAND at the input swaps the outputs exactly (same seed).
3. **Bit-repeat:** the same seed reproduces the same plans on a 16-window subset.
4. **No-leak statement:** the implementer states with file:line that logged future actions enter only the metric and the prior fit on TRAIN, never the planner on eval windows.

## Beside, no bar
Two untrained networks through the same pipeline on a subset of at least 256 windows, to show what the instrument reads with no learned dynamics.

## Log
* 2026-11-08 10:18:38 frozen: prefix 5420 bytes, sha256 d5dbcc637afedde21d87a790eac2c432be02f0fe6899bec87116a99a14957783 (everything above this line). No PLANTEST code or run existed at this time.

# PREREG_REWARD_2, Amendment 1 (Master Mind, 2026-11-09): the operational rule for the F class

**Written and frozen before any state of seed 20261111 was drawn.** No bar, threshold, state rule or outcome branch of `PREREG_REWARD_2.md` changes. The card stays as frozen (sha `9f8eeb12...`); this file only supplies the rule the card left out.

## Why
Card section 3 says each eligible state is classified "from pre-simulation metadata only" and defines F as "no lead or conflict agent in the window, and launch speed >= 1.0 m/s". It gives no operational rule. The first implementation read the agents from the rollouts of the six ladder rungs (`lead_gap` or `conf_ttc` in any cycle of any rung). That is not pre-simulation: the rungs' positions decide which agents are met, so a state whose fast rung reaches a lead leaves F, and that same contact changes the fast rung's return, which enters the KNOWN rule. On the used states that rule gave 5 of 5 KNOWN (`REWARD_2_COUNTERFACTUALS.md` Q2), which is why it may not decide a section 4 branch without this check.

## The rule (`scorer_gym/presim_v6.py`)
F is computed from the log and the state build only, before any rollout, by a chain stage that runs after the state list is pinned and before the first rollout. Its output `free_presim.json` is written once.

* Launch speed: `v0 = ego0.dynamic_car_state.rear_axle_velocity_2d.x`; requires `v0 >= 1.0 m/s`.
* Window: the 17 ladder cycles, iterations `5, 10, ..., 85` of the log, clipped to the log's length. A shorter log uses the iterations it has; the count is recorded.
* At each iteration the ego pose is the **logged** ego pose and the objects are the **logged** tracked objects. The lead and conflict geometry is the reward evaluator's, unchanged (`gym_env.py` cycle): corridor = PDM centreline from the projected ego position, length `max(2.5 v, 20 m)`, half-width 1.5 m; objects within 60 m; lead = centre ahead of the rear axle and box intersects the corridor; conflict = speed >= 0.3 m/s, box outside the corridor now, and box translated at its own velocity intersects the corridor at any of 0.5 to 4.0 s in 0.5 s steps.
* F iff `v0 >= 1.0` and no lead and no conflict at any window iteration.

## What is reported
* **`B3_F` uses this rule and is the one read in section 4.**
* `B3_F_rollout_any_rung` is also written, labelled as not pre-simulation. It never decides a branch.
* A state whose build fails has `free_presim = null`, is not in F, and is counted.
* Agreement of the two classes on the fresh states is reported. No threshold in this rule may be changed after the draw.
* Instrument check on used states only (seed 20261110, 39 eligible, no fresh state): the rule is run once as written and its class is compared with `REWARD_2_COUNTERFACTUALS.md` Q2. If a disagreement appears it is reported; the rule is not tuned on it.

Tier: simulator validity only. No T0, T1 or T2 number.

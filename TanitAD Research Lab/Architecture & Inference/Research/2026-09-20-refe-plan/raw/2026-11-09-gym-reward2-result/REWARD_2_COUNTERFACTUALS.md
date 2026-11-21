# REWARD-2 counterfactuals on stored REWARD-1 records (agent, 2026-11-08)
Scope: CPU re-scoring of stored cycle records only (`pilot_v5/raw/run_main`, `raw/ladder_20_160`, `raw/results_v5.json`). No simulation, no draw, no freeze, `PREREG_REWARD_2.DRAFT.md` untouched. Tier: ENVIRONMENT validity only. Script `reward2_counterfactuals.py` (plain python, no numpy), output `reward2_counterfactuals.json`. Evidence class MEASURED on stored records unless marked.
Instrument check: the script re-derives KNOWN with `gym_env.ladder_known` rules from stored per-cycle rewards: all 39 verdicts (29 KNOWN, 10 not) and the crawl returns (1e-9) reproduce before any counterfactual runs.

**Limit.** Everything below was looked at on the same 39 states that produced the failure. A counterfactual that "fixes" them says nothing about whether it works. Any choice must be judged on freshly drawn states (seed 20261111) under a frozen pre-registration; REWARD-1's B3 FAILED verdict stands unchanged.

## Q1. State 108: why is q_free never applicable on the crawl
Record facts (`json` `Q1`): on all 17 crawl cycles `lead_gap` and `conf_ttc` are None (no lead, no conflict on ANY of the 6 rungs, 100 cycles) and `active.q_free` is False. Applicability is `lead is None and conf_ttc is None and vstar >= 1.0 and not ctx.get("stalled_by_red")` (`gym_env.py:628`). Two conjuncts are shown true by the records, so it is `vstar < 1.0` or `stalled_by_red`.
* `stalled_by_red`: grep over `pilot_v5/scorer_gym` finds ONE occurrence, the read at `gym_env.py:628`. Nothing ever sets it (dead condition, harmless). Eliminated.
* So `vstar = min(limit, curve_speed(s_end)) < 1.0` (`gym_env.py:625`). Limit is 11.18, so `curve_speed` (`gym_env.py:433-441`) returned below 1 m/s, i.e. a curvature k > 2 rad/m (radius under 0.5 m) somewhere in the 30 m ahead. That is not a road: it is a heading step across near-coincident centreline points (`ds` is floored at 1e-3, k is the max of the point-wise differences with no smoothing).
* Position signature agrees: free-geometry cycles are inapplicable up to cumulative path 11.1 m (crawl), 18-19 m (0.3x-1.0x rungs) and applicable from 20.7 m on (the rungs that pass that point), the pattern of a single feature about 20 m ahead of launch dropping out of the 30 m look-ahead. (Cumulative path = sum of `progress_legacy` x limit x 0.5; start offset ignored.)
* **Verdict: a code defect (inference by elimination, mechanism UNVERIFIED), `gym_env.py:433-441` consumed at `:625/:628`.** One probe closes it: print `curve_speed(s)` along state 108's `cl_pts` (needs the nuPlan map, not available to this agent).
* Not specific to 108: over the 158 states with a ladder, 34 have crawl cycles with no lead and no conflict where q_free is inapplicable (JSON key `states_crawl_free_geometry_cycles...`); of the 39 eligible: 20 (17/17), 108 (17/17), 16 (9/17), 17 (1/9). State 20 (fails, crawl/best 0.411) has the same cause.
* Effect if repaired (Q1b, approximation: vstar := lane limit on those cycles, true vstar can only be lower so the penalty is an upper bound; floor 0.20): fixes **20 (crawl/best 0.091, marginal) and 108 (0.047)**, 0 regressions of 29, total KNOWN 31/39; touches states 16, 31, 94, 108, 121. It costs state 108's 0.9x/1.0x rungs 19 % and 18 % (they too were unpriced for 18 m).

## Q2. Free-road classification (39 eligible; 29 KNOWN, 10 not)
Features per state in JSON `Q2_state_table`. **Caveat (UNVERIFIED as "pre-simulation"):** launch speed `v0`, lane limit and `n_obj0` are pre-simulation. "Agent in window" is read from the logged `lead_gap`/`conf_ttc` of the log-replayed rollouts: `crawl` = any cycle of the crawl rung (a lower bound, ego stays within about 11 m); `any-rung` = any cycle of any rung (upper bound, depends on how far each rung drives); `cycle0` = first cycle of any rung (closest to a launch snapshot). A real sampler would project the logged agent tracks on the route centreline instead; this stands in for it.
Threshold reasons (grid, no best-cell pick): v0 1.0 = the reward's own moving gate (`q_gap`, `q_yield` need v > 1.0); 1.5 = the crawl's speed; 3.0 = twice the crawl; 5.0 = just above the smallest eligible limit (4.0).

| definition of free road | kept of 39 | kept KNOWN (of 29) | kept not-KNOWN (of 10) | failing states still kept |
|---|---|---|---|---|
| all eligible (REWARD-1) | 39 | 29 | 10 | all |
| v0 >= 1.0 / 1.5 / 3.0 (identical) | 27 | 24 | 3 | 20, 121, 125 |
| v0 >= 5.0 | 22 | 21 | 1 | 125 |
| no agent at cycle0 | 28 | 19 | 9 | all but 46 |
| no agent in window, crawl | 14 | 10 | 4 | 20, 108, 118, 119 |
| no agent in window, any rung | 10 | 7 | 3 | 108, 118, 119 |
| v0 >= 1.0 and no agent, crawl | 8 | 7 | 1 | 20 |
| v0 >= 1.0 and no agent, any rung | 5 | 5 | 0 | none |
| v0 >= 5.0 and no agent, any rung | 4 | 4 | 0 | none |
(Full grid, 4 thresholds x 3 agent definitions: JSON `Q2_defs`.)
Reading: (a) the Kind B states (31, 121, 125) all have an agent in the window: they are not free-road states and drop out of any agent-based definition. (b) The Kind A states 108, 118, 119 are free road by every definition except a launch-speed one: **no free-road definition removes them; a rest-launch window with an empty road is exactly where the crawl is priced too high.** A speed restriction removes 7 of the 10 by dropping rest launches (and 5 KNOWN ones with them) and keeps 20, 121, 125. (c) The only definitions that pass all kept states keep 4-5 of 39 states (about 12 %); on a fresh 160-state draw that is likely to sit at or under the 5-state minimum (UNDETERMINED risk, ESTIMATED), and it would be a restriction chosen after seeing the failures.

## Q3. Progress-side pace term, exact re-scoring of stored rungs
Per cycle the new reward is the stored reward x w. `w = min(1, ds / ds_ref)^gamma`, `ds` = the cycle distance (`progress_legacy` x limit x 0.5 s, `gym_env.py:596`), `ds_ref` = what a policy covers in the same cycle: P1-P3 accelerate at a_nom from the launch speed to the lane limit ("reach", the ladder's own accel bound is 2 m/s^2, `gym_env.py:758`), P4-P5 use a flat 0.9 x limit. **Exact** on stored records for the six scripted rungs: they do not react to the reward, only the scoring of the same cycles changes; hard-cycle rewards stay 0. **Approximate / not tested:** the reference uses the lane limit, not the curve speed; the PDM arms, B1 oracle and B4 arms were NOT re-scored (the cycle logs exist, it was out of scope); the term's effect on a learning agent is unknown.

| form | fixed of 10 | regressions of 29 | f0.9 retained (mean, min) | f1.0 retained (mean, min) | f1.0 on 12 rest launches (mean) | crawl/best on the 10 (max) |
|---|---|---|---|---|---|---|
| P1 reach a=2, g=1 | 8 (46,73,108,118,119,121,125,134) | 0 | 0.908, 0.879 | 0.979, 0.939 | 0.959 | 0.162 (state 20), else <= 0.061 |
| P2 reach a=2, g=2 | 10 (all) | 0 | 0.825, 0.775 | 0.959, 0.884 | 0.921 | 0.063 |
| P3 reach a=1, g=1 | 6 (46,73,108,119,125,134) | 0 | 0.960, 0.894 | 0.999, 0.985 | 1.000 | 0.162 |
| P4 flat 0.9L, g=1 | 7 (all but 20,31,125) | 0 | 0.932, 0.746 | 0.942, 0.761 | 0.856 | 0.169 |
| P5 flat 0.9L, g=2 | 10 (all) | 0 | 0.889, 0.631 | 0.907, 0.635 | 0.777 | 0.072 |
The 0.9x/1.0x rungs ARE moved (REWARD-1 tolerated 3.4 %): P1 costs f0.9 9 % on average (12 % worst). The flat forms (P4, P5) hurt rest launches most, because a rung accelerating from 0 cannot meet 0.9 L. Never fixed by P1: **state 20** (crawl/best 0.162, limit 4.47, launch 4.1 m/s: crawl is 0.34 of the limit, Q4) and **state 31** (f0.7/best 0.899). Zero regressions in every form is expected (w is monotone in speed, so slow rungs lose at least as much as the best) and was confirmed. Per-state numbers: JSON `Q3`, baseline `Q3_baseline_fail`.

## Q4. Is 0.10 reachable by kinematics on a launch from rest
Model: the ladder rule (`gym_env.py:749-765`: accel <= 2, decel <= 3 m/s^2, 0.1 s steps), 17 cycles of 0.5 s, brake-baseline progress (`:461`), q = 1 (no reward effects). It matches the stored progress ratios (state 108: 0.191 vs 0.187; 118: 0.170 vs 0.164; 73: 0.170 vs 0.167; 31: 0.176 vs 0.169; 46: 0.190 vs 0.183).
* **Launch from rest: crawl/best = 0.369 at limit 4.47, 0.234 at 8, 0.191 at 11.18, 0.176 at 13.41, 0.170 at 15, and 0.168 for every limit >= 17 (the floor, since 17 m/s is all 8.5 s can reach). 0.10 is unreachable by kinematics at any limit; the lowest possible value is 0.168.** (Across the 12 rest launches among the eligible: 0.161-0.369.)
* So on a rest launch B3's `crawl <= 0.10 x best` cannot be met by progress alone: the soft factors must remove at least 1 - 0.10/0.17 = 41 % (limit 15) up to 73 % (limit 4.47) of the crawl's relative return. KNOWN rest-launch states pass only because q_free (or q_gap) already did that work; the failing ones are those where it did not.
* Launch at speed does not have this problem except at low limit: state 121 (v0 3.2) 0.127, 125 (v0 9.0) 0.073, state 20 (v0 4.1, limit 4.47) 0.302. States 20 and 134 (limit 4.47) are low-limit lanes where even a moving start leaves the crawl at 0.30-0.37 of the best by kinematics: this is candidate 4 (crawl relative to the limit), not a free-road question.

## Ranking
1. **Recommended next arm: P1 (progress x min(1, ds/ds_ref), reach a=2.0, gamma=1)**, with the 0.9x/1.0x tolerance rewritten (they lose 9-12 %, not 3.4 %) and the PDM arms / B1 / B4 re-scored on stored records BEFORE freezing (exact, CPU, not done here). Reason for P1 over the stronger P2/P5: its reference is defined by the ladder's own accel bound rather than tuned, it moves the best rung least among forms that fix 8; P2/P5 reach 10/10 only by cutting f0.9 up to 22-37 %, and choosing them for that would be choosing by hindsight. Expected: still FAIL on state 20 (low limit).
2. **Second-best: repair `curve_speed` applicability (Q1b) as its own freeze** (a defect fix independent of the data: no road has a 0.5 m radius), after the one-line `curve_speed` probe on states 108 and 20. It fixes 2 of 10 and would be predicted to FAIL B3 on the other 8, so it is a prerequisite or companion (separate off-switch for attribution), not a pass. Candidate 4 (crawl relative to limit) is the lever for states 20 and 134; population restriction (candidate 3) is not recommended: it either keeps failures (speed rule) or leaves 4-5 states (agent rule).

Pass criteria and all numbers must be re-frozen on seed 20261111; nothing here is a result for the new draw.

## Deliverables
`E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v6/`: `REWARD_2_COUNTERFACTUALS.md`, `reward2_counterfactuals.py`, `reward2_counterfactuals.json`. Not staged, not committed, no repo path.

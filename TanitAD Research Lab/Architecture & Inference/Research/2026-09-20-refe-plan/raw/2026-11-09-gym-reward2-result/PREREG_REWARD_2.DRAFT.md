# PREREG REWARD-2 - DRAFT (agent, 2026-11-08). NOT FROZEN. No states drawn, nothing built, nothing launched.
Scope: one reward-term change to the scorer gym, validated on a NEW fresh set (seed 20261111). Tier: ENVIRONMENT validity only; no model capability claim, no T0/T1/T2 number.
Evidence classes: MEASURED (artifact path) unless marked. Source: `pilot_v5/M_GYM_5_RESULT.md`, `pilot_v5/B3_DECOMP.md`, `pilot_v5/raw/b3_decomp_v5.json`.

## READ FIRST - the decomposition says this pre-named lever cannot pass the bar
M-GYM-5 B3 FAILED on 10 of 39 eligible informative states (`results_v5.json`). With FLOOR 0.15, 0.10, 0.05 and even 0.0, **none of the 10 becomes KNOWN** (`b3_decomp_v5.json` `floor_cf.*.would_be_known` all False). Two causes outside the floor: (A) 7 states where the crawl's progress alone is 0.16-0.35 of the best rung's (a launch from rest limits the fast rungs; q_free inactive or not binding on 20, 46, 108, 118, 119); (B) 3 states (31, 121, 125) where f0.7 earns >= best because q_gap/q_yield price the 0.9x/1.0x rungs. As written below the FLOOR 0.15 experiment is expected to FAIL B3 again; it is drafted only because the brief and PREREG_REWARD_1 section 4 name it. **The Master Mind should choose among the alternatives in section 7 instead of, or before, freezing section 1.** What would actually move the failing states: a progress-side pace term (A), a q_gap/q_yield change (B), or a population restriction by sampler (both).

## 0. Derivation-only states (never validation)
All states seen so far: the 20 M-GYM-4 states (seed 20261109), the 40 M-GYM-2 states, stalls 7, 10, 11, 15, and the 160 REWARD-1 states reached (seed 20261110; its 240-state list in `pilot_v5/PREREG_REWARD_1_STATES.json`, including the 80 unreached). The new draw excludes the logs of all of them.

## 1. The single change (as pre-named): `q_free` floor 0.20 -> 0.15
`q_free = clip(1 - 0.7 (0.9 v* - vbar)/(0.4 v*), FLOOR, 1.0)`, **FLOOR = 0.15**; slope, knee, applicability and every other term unchanged. Off-switchable: FLOOR 0.20 must reproduce M-GYM-5 bit for bit and FLOOR 0.30 M-GYM-4. Constant derived ONLY from used states (the 160 states of seed 20261110 and the earlier ones), by the REWARD-1 rule: PRE-NAMED, not re-selected. Counterfactual crawl/best with 0.15 on the 10 failing states: `b3_decomp_v5.json` `floor_cf["0.15"]` (all still NOT KNOWN).
Side effect to measure (UNVERIFIED for the 0.9x/1.0x rungs on the new set): REWARD-1 estimated 3.4 % on a launch from rest.

## 2. Fresh set (seed 20261111, fixed before drawing; no seed search)
Same rule as PREREG_REWARD_1 section 2: `select_states_v4(seed=20261111, ...)` with the section 0 exclusion; one fixed ordered list of 240 states (idx 0-19 full-arm: 10 random + 10 stop-and-go/junction, 5 logs x 2; 20-159 ladder-only; blocks 160-199 and 200-239 added only while < 5 eligible, never a new seed). No state replaced; build failures counted. Rate observed in REWARD-1: informative 41/158 (0.259), eligible 39. Sizing is therefore not the risk.

## 3. Bars (every REWARD-1 threshold unchanged)
* **B3:** informative = 0.5x, 0.7x, 0.9x, 1.0x hard-free; eligible = informative AND limit >= 4.0 m/s; KNOWN iff every hard-free slow rung (crawl, 0.3x, 0.5x, 0.7x) < 0.85 x max(0.9x, 1.0x) AND crawl <= 0.10 x max(0.9x, 1.0x). PASS iff >= 5 eligible and all KNOWN; FAIL iff >= 5 eligible and any not KNOWN; UNDETERMINED iff < 5 at the cap. Informative states with limit < 4.0 are reported (limit, crawl/limit) and not judged.
* B1, B2, B4, B5, B6, B7: text unchanged (PREREG_M_GYM_4 section 4), retention checks on the fresh full-arm set. B2 also reports the number of distinct real-collision states (M-GYM-4: 1, M-GYM-5: 3). A regression arm no longer KNOWN is a finding.
* Deliberate-regression arms: R1 FLOOR 0.20 on used states 3, 11, 17 reproduces the stored M-GYM-5 ladder returns bit for bit (and FLOOR 0.30 the M-GYM-4 ones), determinism holds; R2 q_free held 1.0 reads crawl/best 0.3648 / 0.1691 / 0.5214 and is NOT KNOWN on state 3; R3 always-slowest stays KNOWN. Add R4: on the 10 M-GYM-5 failing states, FLOOR 0.15 rungs reproduce `floor_cf["0.15"]` for the crawl (and the verdict on those states is stated in advance as NOT KNOWN, so that a PASS cannot arrive from a changed state).
* G-REPRO is written to match what was accepted in `PREREG_REWARD_1_GREPRO_NOTE.md`: crawl/best within 1 % of the table, 0.9x and 1.0x within 1 %, every rung >= its counterfactual. Not "1 % on all rungs".
* No estimator or CI applies (deterministic scripted policies, one gym seed).

## 4. Outcomes and next steps (committed with the freeze)
* B3 PASS with R1-R4 and B1-B6 retained: adopt FLOOR 0.15 (the decomposition makes this unlikely).
* B3 FAIL on states other than the 10 already failing: the reward still pays a slow rung; apply section 7 on those.
* B3 FAIL on the same Kind A / Kind B states: the floor is exhausted; the next lever is not q_free (see section 7). The seed 20261111 states are spent.
* B3 UNDETERMINED at 240: state availability; free-road sampler under its own pre-registration.
* R1/R2 wrong: stop; defect; fresh set not spent if seen before the first fresh rollout.
* Metric families: LONGITUDINAL only as reward terms and hard events; LATERAL only off_drivable/wrong_way events; TACTICAL n/a; STRATEGIC PAUSED (PI 2026-09-19, ITEM 25), not a pass.

## 5. Rules
CPU only, `CUDA_VISIBLE_DEVICES=-1`, at most 4 workers, detached, no GPU lock, no pod; scratch two levels deep on F: deleted by literal path at the end; code sha256-pinned and tests green before the fresh rollouts.

## 6. Pre-launch checklist for the Master Mind (not done here)
Freeze by sha256 before drawing; choose between section 1 and section 7; copy `pilot_v5/scorer_gym` to `pilot_v6/scorer_gym` and add the change; do not reuse seed 20261110.

## 7. Alternatives that the decomposition supports (each would replace section 1 and be its own freeze)
1. **Progress-side pace term (Kind A, 7 states):** price a rung's progress by distance relative to what a nominal-speed policy covers in the same cycle, so the crawl cannot earn 0.16-0.35 of the best rung's progress. Risk: also moves the 0.9x/1.0x rungs (REWARD-1 rejected a progress change for exactly that; quantify first on stored records, CPU).
2. **q_gap / q_yield trigger review (Kind B, 31, 121, 125):** the fast rungs lose 0.9-2.3 units to these terms where traffic is present. Needs a look at whether the penalty is justified (a scripted 1.0x rung that ignores a lead is correctly punished; then the BAR, not the reward, is wrong for traffic states).
3. **Population restriction by sampler:** judge B3 only on windows with ego speed at launch above a threshold and no lead/conflict within the window (the "free-road sampler" already named for UNDETERMINED). Changes the bar's population, not the reward; must be pre-registered before drawing.
4. **Crawl rung relative to the limit:** min(1.5, 0.35 x limit), named in PREREG_REWARD_1 section 4; relevant to ineligible states 66, 69 (limit 2.235) only.
5. Trace state 108 first (CPU): why q_free is never applicable on its crawl (UNVERIFIED: `vstar < 1.0` or `stalled_by_red`, `gym_env.py:626`). If it is a defect, it is a code fix and not a reward change.

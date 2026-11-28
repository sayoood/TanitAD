# PREREG REWARD-2 (Master Mind, 2026-11-08): the curve-speed repair, judged on fresh states

**Frozen by sha256 before any state of seed 20261111 was drawn.** Tier: simulator validity only; no model capability claim and no T0, T1 or T2 number. Replaces section 1 of `PREREG_REWARD_2.DRAFT.md`, whose floor 0.15 arm was eliminated at zero compute (`H-GYM-REWARD-1` RESULT: no floor from 0.15 to 0.0 makes any of the 10 failing states KNOWN).

## 0. Derivation-only states (never validation)
All states seen so far: the 20 M-GYM-4 states (seed 20261109), the 40 M-GYM-2 states, stalls 7, 10, 11, 15, and the 240-state list of seed 20261110 (`pilot_v5/PREREG_REWARD_1_STATES.json`). The new draw excludes the logs of all of them.

The fix was found by looking at the seed 20261110 states. On those 39 eligible states an exact re-simulation with the fix reads 34 of 39 KNOWN (`CURVE_SPEED_DEFECT.md`). That number motivated this test and is not evidence for it. **`H-GYM-REWARD-1` stays FAILED as written.**

## 1. The single change
`curve_min_seg_m = 0.05` in `gym_env.curve_speed` (`pilot_v6/scorer_gym/`): centreline points closer than 0.05 m are dropped before heading steps are divided by arc length. The value comes from the spacing histogram. Floor stays 0.20; every other term, slope and knee is unchanged. With `curve_min_seg_m = None` the code reproduces M-GYM-5.

**Not changed, deliberately:** the pace term (P1), `q_gap` and `q_yield`, and the crawl rung. A second change would make the result non-attributable. They are the next levers, under section 4.

## 2. Fresh set (seed 20261111, fixed before drawing; no seed search)
Same rule as `PREREG_REWARD_1` section 2: `select_states_v4(seed=20261111, ...)` with the section 0 exclusion; one fixed ordered list of 240 states (idx 0-19 full-arm: 10 random + 10 stop-and-go/junction, 5 logs x 2; 20-159 ladder-only; blocks 160-199 and 200-239 added only while fewer than 5 are eligible, never a new seed). No state is replaced, and build failures are counted.

## 3. Bars
**Primary, every REWARD-1 threshold unchanged:**
* **B3:** informative = 0.5x, 0.7x, 0.9x, 1.0x hard-free; eligible = informative AND limit >= 4.0 m/s; KNOWN iff every hard-free slow rung (crawl, 0.3x, 0.5x, 0.7x) < 0.85 x max(0.9x, 1.0x) AND crawl <= 0.10 x max(0.9x, 1.0x). PASS iff >= 5 eligible and all KNOWN; FAIL iff >= 5 eligible and any not KNOWN; UNDETERMINED iff fewer than 5 at the cap. Informative states with limit < 4.0 are reported and not judged.
* B1, B2, B4, B5, B6, B7: text unchanged (`PREREG_M_GYM_4` section 4), retention checks on the fresh full-arm set. B2 also reports distinct real-collision states.

**Secondary, reported and never a substitute for B3.** Each eligible state is classified from pre-simulation metadata only, and all states are listed:
* **F (free road):** no lead or conflict agent in the window, and launch speed >= 1.0 m/s. **B3-F** is B3's KNOWN rule on F.
* **Kinematic crawl share:** the crawl/best ratio that the rung definitions and the window give from kinematics alone (`REWARD_2_COUNTERFACTUALS.md` Q4). MEASURED there: on a launch from rest 0.10 is unreachable from progress alone (0.168 at limits of 17 m/s and above, 0.369 at 4.47 m/s).

**Regression arms and instrument checks:**
* R1: `curve_min_seg_m = None` on used states 3, 11, 17 reproduces the stored M-GYM-5 ladder returns bit for bit, and determinism holds.
* R2: `q_free` held at 1.0 reads crawl/best 0.3648, 0.1691, 0.5214 on those states, and is NOT KNOWN on state 3 (unchanged from REWARD-1, with the switch off).
* R3: always-slowest stays KNOWN.
* R4: with the switch on, the verdicts of the 39 seed-20261110 eligible states reproduce `CURVE_SPEED_DEFECT.md` before any fresh rollout (34 KNOWN; not KNOWN are 31, 46, 121, 125, 134).
* G-REPRO: crawl/best within 1 % of that table, 0.9x and 1.0x within 1 %.
* No estimator or CI applies (deterministic scripted policies, one gym seed).

## 4. Outcomes, committed now
* **B3 PASS with R1-R4 and B1-B6 retained:** adopt the fix; the simulator proceeds to the validity statement of `PREREG_M_GYM_4` section 5.
* **B3 FAIL (expected from the used states: about 5 of 39 not KNOWN):** report it as FAILED. Then read B3-F:
  * **B3-F has at least 5 states, all KNOWN:** what remains is a population question. The Master Mind may propose a free-road-only validity scope for B1 under its own pre-registration; whether that scope is acceptable is the PI's decision. Nothing here declares the simulator valid.
  * **B3-F has a state not KNOWN:** the reward still pays a slow rung on free road. Next lever: the P1 pace term as its own freeze, after re-scoring the PDM, B1 and B4 arms on stored records and writing its 0.9x and 1.0x tolerance in advance (`REWARD_2_COUNTERFACTUALS.md` Q3).
  * **B3-F has fewer than 5 states:** undetermined for that scope; free-road sampler under its own pre-registration.
* **B3 UNDETERMINED at 240:** state availability; free-road sampler under its own pre-registration.
* **R1-R4 wrong:** stop; a defect; the fresh set is not spent if that is seen before the first fresh rollout.
* The seed 20261111 states are spent once read.
* Metric families: LONGITUDINAL only as reward terms and hard events; LATERAL only off-drivable and wrong-way events; TACTICAL n/a; STRATEGIC PAUSED (PI 2026-09-19, ITEM 25), not a pass.

## 5. Rules
CPU only, `CUDA_VISIBLE_DEVICES=-1`, at most 4 workers, detached with the own-process-group launch and a PID check in the waiter, no GPU lock, no pod. Scratch two levels deep on F:, deleted by literal path at the end. Code sha256-pinned and tests green before fresh rollouts. Do not start while the dev box is held for the label pass. Do not reuse seed 20261110.

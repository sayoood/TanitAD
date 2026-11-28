# PREREG REWARD-1 - DRAFT (2026-11-08). NOT hashed, NOT frozen. The Master Mind reviews, edits and freezes.
Scope: one reward-term change to the scorer gym, validated on a NEW fresh set (seed 20261110). Tier: ENVIRONMENT (simulator) VALIDITY only; no model capability claim, no T0/T1/T2 number.
Evidence classes: MEASURED (artifact path) unless marked. Source diagnosis: `B3_DECOMP.md`, `b3_decomp/attrib.json`.
**Nothing for seed 20261110 has been drawn, built or simulated, and none may be before this file is frozen and its sha256 written. Code is not changed by this draft.**

## 0. Derivation-only states (never validation)
All 20 M-GYM-4 states (seed 20261109; states 3, 11, 17 are the informative ones and carry the constants), the 40 M-GYM-2 states (main + stall sets), stalls 7, 10, 11, 15. The new draw must exclude all their logs (as `select_states_v4` did).

## 1. The single change: `q_free` floor 0.30 -> 0.20
Current (MEASURED, `scorer_gym/gym_env.py:620-621`): `q_free = clip(1 - 0.7 (0.9 v* - vbar)/(0.4 v*), 0.30, 1.0)`. Proposed: `q_free = clip(1 - 0.7 (0.9 v* - vbar)/(0.4 v*), FLOOR, 1.0)`, **FLOOR = 0.20**. Slope, knee (0.9 v*), applicability rule and every other term unchanged. Off-switchable: `FLOOR = 0.30` must reproduce M-GYM-4 bit for bit.

Why this term and not progress (from `B3_DECOMP.md`): the only used state where the crawl is priced too high for a speed reason is state 3, where the crawl sits at the floor on 17 of 17 cycles (loss 3.48 of 4.97) and prog/best = 0.354 is fixed by speed/limit. A progress change would also move the 0.9x/1.0x rungs.

Constants, derived ONLY from used states: FLOOR from state 3 (binding: crawl/best 0.116 at 0.30, needs a floor <= 0.259), checked against state 11 (0.065 -> 0.044) and state 17 (unmoved, 0.447). Counterfactual table (exact for scripted rungs, upper bound on the cut, `b3_decomp/attrib.json` `floor_cf`): 0.25 -> 0.097, 0.20 -> 0.077, 0.15 -> 0.058. Selection rule committed now: the largest value of {0.25, 0.20, 0.15} that gives crawl/best <= 0.08 on state 3 (a 20 % margin for the fresh states' spread) = 0.20.

Side effect, MEASURED as an upper bound: the 0.9x/1.0x rungs lose 0.2 % (state 3) and 2.1-2.2 % (state 11, launch from rest), mean 0.5 % over the 10 hard-free cells of the 20 states; they are not strictly untouched.

Pre-launch gate G-REPRO (used states only): the implementer reruns the ladder of used states 3, 11, 17 with FLOOR 0.20; the result must match the counterfactual table within 1 % relative, otherwise the constants are re-derived before any draw.

Not changed, deliberately: `q_free` is off while a conflict agent projects a hit within 4 s, and `q_yield` needs v > 1.0 m/s (both seen in `B3_DECOMP.md` Q1). Neither causes a B3 failure, and a second change would be non-attributable. `q_stall` firing on a launch from rest (state 17, 0.3x rung, 0.15 units) is logged, unchanged.

## 2. Fresh set (seed 20261110, fixed before drawing; no seed search)
* **Full-arm set (B1, B2, B4, B5, B7):** 20 states, same split as M-GYM-4: 10 random + 10 stop-and-go/junction (5 logs x 2), `select_states_v4(seed=20261110, ...)`, excluding the logs in section 0.
* **Ladder-only extension (B3 decidability), same seed, same ordered candidate list:** a further 140 states (70 random, 70 stop-and-go/junction), only the six ladder rungs run on each (median 22 s, max 93 s per state, MEASURED, `raw/run_main/states.jsonl.gz` `ladder_wall`). 160 states in total, about 160 x 30 s / 4 workers = 20-25 min (ESTIMATED). If B3 is still undecidable (section 3) the draw continues down the same fixed list in blocks of 40 to a cap of 240; never a new seed.
* **Sizing (ESTIMATED, binomial arithmetic):** the informative rate was 3/20 (Wilson 95 % 5.2-36 %); eligible informative (limit >= 4.0 m/s) 2/19 (Wilson 2.9-31 %). Target: >= 5 eligible informative states. P(>= 5 | N = 160) = 1.00 at p = 0.10, 0.91 at p = 0.05 and 0.53 at p = 0.03 (near the lower end of the interval); at the cap N = 240 it is 0.99 at p = 0.05 and 0.85 at p = 0.03. Informative requires 0.5x..1.0x all hard-free over 8.5 s; 14 of the 17 non-informative M-GYM-4 states had a hard event on the 0.9x or 1.0x rung (MEASURED, `raw/run_main/states.jsonl.gz`), mostly standing starts.
* No state is replaced; a build failure is reported and counted. Selection uses only pre-simulation metadata (log, row, sampler); the limit is reported, not filtered on.

## 3. Bars (committed before the draw; every M-GYM-4 threshold unchanged)
* **B3 as written (PREREG_M_GYM_4 section 4):** informative = 0.5x, 0.7x, 0.9x, 1.0x hard-free. KNOWN iff every hard-free slow rung (crawl, 0.3x, 0.5x, 0.7x) < 0.85 x max(0.9x, 1.0x) AND crawl <= 0.10 x max(0.9x, 1.0x). PASS iff >= 2 informative and all KNOWN; FAIL iff >= 2 informative and any not KNOWN; UNDETERMINED iff < 2.
* **B3-N decidability (added by this draft; the Master Mind accepts or drops it):** a B3 PASS is reported as a result only if >= 5 informative states have limit >= 4.0 m/s ("eligible"); otherwise B3 is UNDETERMINED and the draw is extended (section 2). Reason: the bar as written is unsatisfiable below a limit of about 3.4 m/s for any floor >= 0.2 (`B3_DECOMP.md` Q3), and 2 states decide little. The limit and crawl/limit of every informative state are reported next to its verdict.
* **B1, B2, B4, B5, B6, B7: text unchanged from PREREG_M_GYM_4 section 4**, on the fresh full-arm set. They are retention checks of what M-GYM-4 KNEW:
  * B1 PASS; B2 PASS (reward-independent: the ttc rule is a hard event; also report the number of distinct real-collision STATES, M-GYM-4 had 1);
  * B4 always-slowest KNOWN (M-GYM-4: mean q_free 0.425 < 0.6, return 0.668 < PDM 2.673; a lower floor moves both the same way);
  * always-fastest KNOWN (89 vs 67 and 95 vs 51; the floor does not act at speed); do-nothing KNOWN (0.0016 <= 0.02); determinism bit-identical.
  * Any arm that is no longer KNOWN is a finding, not tuned away.
* **Deliberate-regression arms (must read their known values, else the instrument is broken and nothing is judged):**
  * R1 control: `FLOOR = 0.30` on used states 3, 11, 17 reproduces the 18 stored ladder returns (`raw/results_v4.json` `ladder.per_state[*].rets`, e.g. state 3 crawl 1.4904977443399394) bit for bit, and determinism holds.
  * R2 term-removed arm: q_free held at 1.0 must read crawl/best 0.3648 (state 3), 0.1691 (state 11), 0.5214 (state 17) (derived from the stored records by dividing out `q.q_free`; ESTIMATED until rerun) and must be NOT KNOWN on state 3. If it reads KNOWN, the B3 test cannot fail and the bar is void.
  * R3: the B4 always-slowest arm stays KNOWN.
* No estimator or CI applies: scripted policies on a deterministic simulator (bit-identical), one gym seed, counts over states. Nothing here is a lever effect on a trained model.

## 4. Outcomes and the next step for each (committed now)
* **B3 PASS (>= 5 eligible informative, all KNOWN) with R1, R2 and B1-B6 retained:** adopt FLOOR 0.20. The simulator proceeds to the validity statement of PREREG_M_GYM_4 section 5 (still conditional on B2 resting on one scenario unless the count of collision states rose; B6 UNDETERMINED is allowed).
* **B3 FAIL, every failing state has limit < 4.0:** a LADDER-DEFINITION defect, not a reward defect, and not a PASS. Next: a separate pre-registration defining the crawl rung relative to the limit (min(1.5, 0.35 x limit)); no further reward change.
* **B3 FAIL on any state with limit >= 4.0:** the reward still pays a crawl. Decompose that state with `b3_decomp/attrib.py` (CPU, minutes) first. Pre-named cheapest next lever: FLOOR 0.15 (derivation: state 3 -> 0.058, state 11 -> 0.034; costs the 0.9x/1.0x rungs up to 3.4 % on state 11). If the crawl sits above the floor (vbar > 0.5 v*), the q_free knee/slope is next, not progress. Run it on a further fresh set (seed 20261111).
* **B3 UNDETERMINED at the cap (240):** report the eligible informative count and the observed rate. The blocker is state availability, not the reward; unblock = a free-road sampler (standing-start filter) under its own pre-registration.
* **R1 or R2 reads wrong:** stop; the code change is defective. The fresh set is NOT spent if it shows at G-REPRO or on used states, SPENT if after the first fresh rollout.
* **B1, B2 or B4 regress:** report as a finding, name the cause. If code changes in response to a defect seen on the fresh states, they are SPENT and seed 20261111 is proposed.
* **Metric families:** LONGITUDINAL measured only as simulator reward terms and hard events; LATERAL only off_drivable/wrong_way events; TACTICAL n/a (scripted arms); STRATEGIC PAUSED (PI 2026-09-19, ITEM 25), not a pass.

## 5. Rules
CPU only, `CUDA_VISIBLE_DEVICES=-1`, at most 4 workers, detached; the dev-box GPU lock is never touched; no pod is touched; scratch two levels deep on F: and deleted by literal path at the end; code files sha256-pinned before the fresh rollouts and tests green before RUN A (as in M-GYM-4).

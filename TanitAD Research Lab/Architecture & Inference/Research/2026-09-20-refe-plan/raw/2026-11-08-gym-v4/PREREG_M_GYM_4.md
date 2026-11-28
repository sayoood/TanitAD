# PREREG M-GYM-4 (written 2026-11-08 BEFORE any fresh state is simulated)
Scope: validate, on 20 FRESH states, five fixes to the scorer-only RL simulator (code: `pilot_v4/scorer_gym/`, a copy of `pilot_v2/scorer_gym/`;
`pilot_v2/` and `pilot_v3_diag/` are untouched). Every PREREG_M_GYM_1 threshold is unchanged. M-GYM-2's oracle arm stays FAILED as written on its own
state sets; it is not re-judged here. The oracle definition used in this document (fix 1) is new and is judged only on the fresh states.
Evidence classes: MEASURED (artifact path) unless marked. Tier: environment validity of the gym itself; no model capability claim.

## 0. Disclosure of what was looked at before this file
* USED states only (M-GYM-2 main and stall sets): the M_GYM_3_DIAG measurements; plus, run today before writing this file,
  `explore/explore_corridor.py` on stall 7, 10, 11, 15 (24 arms, log mode, ttc OFF, shadow flags) and `explore/prescreen_dist.py` on all 20 main + 20 stall
  states. Outputs: `explore/ex_stall{7,10,11,15}.json`, `explore/prescreen_dist.json`. They are DERIVATION evidence, never validation.
* No fresh state has been built, simulated or inspected. (Disclosure: `select_states_v4` was executed once inside the selection unit test, output = PASS/FAIL lines only; the list was not displayed or used.) The fresh list is a deterministic function of the seed and sampler split in section 2; it is drawn
  only after this file's sha256 is written (`PREREG_M_GYM_4.sha256`) and recorded in `PREREG_M_GYM_4_STATES.json` with its own sha256.

## 1. The fixes (each off-switchable; each with a deliberate-regression test that must go RED and a control that must read the old behaviour)
Switch: `GymState(st, fixes="on"|"off")`; analysis-side rules `oracle_pick(..., rule)` and `cell_status(..., tie_rule)`.
`fixes="off"` + rule `argmax` + tie rule `none` reproduces pilot_v2 bit for bit (control test: stored M-GYM-2 returns of main 1 and stall 7 arms are reproduced exactly).

| # | fix | derived from (USED states) | exact specification |
|---|---|---|---|
| 1 | oracle = fewest hard events first, then return | stall 7 (oracle arm 23 return 6.689 collides at cycle 16; best clean arm pdm0 4.356; 6 of 24 arms clean) | per (state, mode) cell of 24 arms, oracle = argmax of (-hard, return), first index on ties; an episode has 0 or 1 hard event. Old rule `argmax` kept for reporting |
| 2 | ttc predicate also requires the object in the ego corridor | stall 15 (13 of 24 arms legacy-flagged, no collision with the predicate off), stall 11 (5 flagged), stall 10 (24 flagged), stall 7 (16 flagged, 14 of them real collisions) | legacy test unchanged (ego footprint advanced v*dt straight, object advanced along its velocity, dt 0.2..1.0 s, closing >= 0.3 m/s, object ahead) AND the object's advanced box at the hitting dt intersects the corridor = route centreline from 2 m behind the ego projection to 8 m + v*(1 s) ahead, buffered by 1.15 m (ego half-width, no extra margin, flat caps). So corridor flags are a SUBSET of legacy flags. Braking-awareness is NOT included: measured not to help (stall 10 still 24/24 at 3 and 6 m/s^2; stall 7 trims only 18 to 16 hard arms), so it has no justification |
| 3 | map pre-screen | main 10 (on-centreline footprint overshoots the strict drivable union by 0.254 m at s = 18.5-23 m; all 24 arms off_drivable) | at state build, Pacifica footprints on the route centreline (heading = tangent) at every 1.0 m over the next 40 m from the ego projection; `centreline_overshoot_m` = largest corner distance outside the strict union; state flagged `map_unreliable_prescreen` iff > **0.20 m**. `MAX_MAP_SLACK_M` stays 1.0 m and no slack is granted by this screen: it labels only |
| 4 | all-tie cycle-0 cells are label-free | main 19 (24/24 arms return exactly 0.0, all hard in cycle 0, 1.1 cm over the slack) | cell is `label_free_tie` iff all 24 arms ended in a hard event with exactly 1 cycle and all 24 returns are exactly equal. Reported separately from `unavoidable` (= all 24 arms hard, any cycle) |
| 6 | constant-speed ladder as a regression arm | pilot 5, stall 9 (informative), pilot 4, 6 (not) from `q3_*.json` | policies: crawl 1.5 m/s and 0.3/0.5/0.7/0.9/1.0 x lane limit, held on the PDM centreline (accel <= 2, decel <= 3), log mode, 17 cycles, from the initial speed. Item 5 (q_stall excuse tightening) is NOT implemented (low priority per M_GYM_3_DIAG; unchanged) |

Derivation numbers (MEASURED, `explore/ex_stall*.json`, n = 24 arms per state, log mode, ttc OFF so that the arm's true fate is visible):
| state | legacy-flagged arms | flagged by corridor | real collisions in the arm set | collisions flagged by legacy / by corridor | verdict on the corridor spec |
|---|---|---|---|---|---|
| stall 15 | 13 | 0 | 0 | - | clears all 13 false alarms |
| stall 11 | 5 | 5 | 0 | - | clears none |
| stall 10 | 24 | 24 | 8 | 8 / 8 (flag 10-11 ticks before contact) | clears none: the oncoming car IS inside the corridor; the arms diverge only after the flag, so no static test can separate them |
| stall 7 | 16 | 16 | 16 (2 of them unflagged by either predicate) | 14 / 14 | clears none |
Honest expectation committed in advance: the corridor predicate removes SOME false alarms (stall-15 type, a crossing object near but outside the ego corridor)
and does NOT remove oncoming-pass false alarms (stall-10 type). It was NOT shown to fix the stall-10 oracle miss; that miss is handled by fix 1 only if some
arm is clean, and otherwise reported as unavoidable. Pre-screen derivation (MEASURED, `explore/prescreen_dist.json`, n = 40 used states): see section 6.

## 2. Fresh state selection
* **Seed 20261109** (fixed before drawing; no search over seeds). Function `gym_env.select_states_v4(seed=20261109, n_random=10, n_stall_logs=5, exclude_logs=used logs)`.
* **Sampler split: 10 states from the random sampler** (1 per log, 10 logs, hash order, same margins as `select_states`) **+ 10 states from the stop-and-go/junction sampler**
  (5 logs, per log the hash-first stop-and-go row and the hash-first junction row, same tag lists and margins as `select_states_stall`). = 20 states, 15 distinct logs.
* Excluded: all 147 navtest logs; every log used by the M-GYM-2 main set or stall set (so the fresh set is disjoint from both by log, hence by state); the two sub-samplers share no log.
* No hand picking, no replacement. If a state fails to build (exception) it is reported as a failed state and counted against the bars where stated, not replaced.

## 3. Runs (all CPU, CUDA off, at most 4 workers, detached)
* RUN A `main`: fixes ON; 24 arms x {log, idm}; regression policies slowest (log), fastest (log, idm), stop (log); ladder (6 targets, log); shadow flag for the legacy ttc predicate recorded, not acting.
* RUN B `ttcoff` (ground truth for fix 2): same states, 24 arms x {log, idm}, ttc OFF, shadow flags legacy + corridor recorded, never terminating on ttc.
  A real collision = the arm's first real hard event in RUN B is an at-fault collision. Run A's outcome per arm must equal the outcome derived from RUN B's shadow flags
  (min of corridor flag, first real hard event): any mismatch is reported and voids B2.
* Determinism: fixes-ON state index 0, arm 5, both modes, run twice in one process in each of two fresh processes: all four traces bit-identical (==).
* Tests (`tests/test_v4.py`) must be green BEFORE RUN A starts (the chain aborts otherwise, nothing fresh is simulated). Code files are sha256-pinned in `CODE_PIN.json` before RUN A.

## 4. Bars (verdict per bar as written; all PREREG_M_GYM_1 thresholds unchanged)
* **B1 oracle (fix 1).** Over the 40 cells (20 states x 2 modes): PASS iff no lexicographic-oracle arm has a hard event in any cell where at least one of the 24 arms is clean.
  A cell with no clean arm is reported as UNAVOIDABLE (with its hard types) and is not counted as a pass. NOTE: given the definition this is true by construction; its content is
  the unavoidable count and the return cost of the definition (regret vs the old argmax, reported). The old argmax result on the same runs is reported next to it.
* **B2 ttc guard retention (fix 2).** From RUN B: R_legacy = real-collision arm-episodes (both modes) that the legacy predicate flagged at least one tick before contact;
  R_corr = those also flagged by the corridor predicate before contact; lost = R_legacy minus R_corr. PASS iff R_legacy >= 10 and R_corr/R_legacy >= 0.80.
  FAIL iff R_legacy >= 10 and ratio < 0.80 (a corridor fix that hides real collisions). UNDETERMINED (bar not met) iff R_legacy < 10. Reported with n: R_legacy, R_corr, lost,
  total real collisions, and false-alarm episodes (non-collision) flagged by legacy versus by corridor. Derivation reference (used states, NOT evidence): 22/22 retained.
* **B3 ladder (fix 6).** A fresh state is informative iff the 0.5x, 0.7x, 0.9x, 1.0x rungs are all hard-event-free. On an informative state the ladder is KNOWN iff every hard-free slow rung
  (crawl, 0.3x, 0.5x, 0.7x) returns < 0.85 x max(0.9x, 1.0x) AND the crawl returns <= 0.10 x max(0.9x, 1.0x). (Constants fixed from the two informative used states: 0.7x/best = 0.65, 0.74; crawl/best = 0.027, 0.031.)
  PASS iff >= 2 informative states and all KNOWN; FAIL iff >= 2 informative and any not KNOWN; UNDETERMINED iff < 2 informative. q_stall fires at the crawl are counted and reported (expected 0 by the code rule).
* **B4 PREREG_M_GYM_1 regression arms, unchanged text, on the fresh states.** always-slowest KNOWN iff mean q_free over applicable cycles < 0.6 AND mean return < PDM-Closed's;
  always-fastest KNOWN iff (hard events + lead-present cycles with q_gap < 0.7) strictly greater than PDM-Closed's in log or idm; do-nothing KNOWN iff mean per-cycle reward <= 0.02 over states moving at t0;
  determinism bit-identical. A regression arm that is not KNOWN is reported as a finding, not tuned away.
* **B5 cost.** PREREG_M_GYM_1 verdict unchanged: mean effective s/state at 12 workers (= single-worker seconds of the timer buckets / 12, ESTIMATED linear) GO <= 26, REDUCE 26-260, BLOCKED > 260.
  Reported with the measured 4-worker wall per state and states per 3 days (259,200 s) at that measured cost.
* **B6 map pre-screen coverage (fix 3).** Over cells where all 24 arms end off_drivable: PASS iff every such state carries the pre-screen flag; FAIL iff any does not; UNDETERMINED iff there is no such cell.
  Flags and overshoots of every fresh state are reported; the screen labels only and never exempts a state from B1.
* **B7 activation.** Per-term fired rates as in PREREG_M_GYM_1 (a term with fired rate < 1 % is flagged sparse); reported, no pass/fail.

## 5. Verdict on validity for sizing the B1 teacher, and the next step for each outcome (committed now)
* VALID iff B1 PASS, B2 PASS, B3 PASS, B4 all KNOWN with determinism bit-identical, B6 not FAIL, unavoidable cells <= 4 of 40 (10 %), cost verdict not BLOCKED, and 0 build failures.
  Otherwise the statement is NOT YET VALID with each failing bar named; the states-per-3-days figure is still reported, as conditional.
* B1 FAIL: impossible by construction, so it means an implementation defect: diagnose, the 20 states are SPENT, propose a new fresh set (seed 20261110).
* B2 FAIL: revert fix 2 (`ttc_mode="legacy"`), characterise the lost collisions, propose a corrected predicate for a new fresh set. B2 UNDETERMINED: keep fix 2 labelled "retention unproven",
  draw a second fresh set from the stop-and-go/junction sampler only (seed 20261110) until R_legacy >= 10. B2 PASS: adopt fix 2 as specified.
* B3 FAIL: the reward pays slowness on a free road: write a separate pre-registration for a reward-term change (q_free floor or progress), never retune here. B3 UNDETERMINED: add free-road states (second fresh set).
* B4 any arm not KNOWN: finding about the reward; next step is a reward fix under its own pre-registration. Determinism not identical: stop, find the nondeterminism before anything else.
* B5 GO: size at the measured cost. REDUCE: size at the measured figure and propose, for a separate test, fewer arms or a held plan (hold 1.0 s changed about half the argmax labels in M-GYM-2, so not adopted untested). BLOCKED: stop and report.
* B6 FAIL: widen the screened stretch (length/speed-dependent) under a new pre-registration. B6 UNDETERMINED: no action.
* More than 4 of 40 cells unavoidable: report their causes (map seam, predicate, log-replay cut-in) and propose the specific fix for each on a new fresh set.

## 6. Rules
* If a defect is found on the fresh states and code changes in response, those 20 states are SPENT: say so, stop, propose a new fresh set. No threshold or constant above is changed after the draw.
* Scratch: `F:/tanitad_scratch/mgym4/` (two levels deep). `F:/tanitad_scratch_mgym2` is not touched. The shared GPU lock file is never touched; no pod is touched.

### 6a. Pre-screen derivation (MEASURED, `explore/prescreen_dist.json`, n = 40 used states; filled before the sha)
38 of 40 used states have an on-centreline overshoot of exactly 0.000 m over the next 40 m. Two are non-zero: **main 10 = 0.251 m** (hard: off_drivable in all 24 arms, M-GYM-2) and
**stall 19 = 0.223 m** (no off_drivable event in any arm in M-GYM-2; a flagged-but-clean state, i.e. the screen labels a seam that did not bite there). Main 8 (human-log start overshoot 0.143 m) and main 19
(0.156 m) read 0.000 on the centreline: the screen does NOT catch the start-pose slop cases; those are handled by the existing slack and by fix 4. Tolerance 0.20 m is fixed between the
cluster at 0 and the two non-zero values, which it flags (2 of 40 used states flagged; no other used state could be flagged under any tolerance > 0). It is a label, not a threshold on a result.

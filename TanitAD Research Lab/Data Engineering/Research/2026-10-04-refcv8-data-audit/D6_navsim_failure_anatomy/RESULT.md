# RESULT -- D6: anatomy of the refcv7 NavSim failures (navhard DAC/NC zeros), refcv8 data audit, 2026-10-04

Stream D6 (Data Engineering / refcv8 data audit). Model under test: refcv7-r101-s0 `R7_A1` at step 30,000 (CUDA bf16 bridge,
inference seed 0), compared with `R7_A1_s1` (inference seed 1), step 5,000, `PRIOR_ha0p` (model-free kinematic prior), `STOP_zero`,
`CV_official`. Tier: NavSim open-loop benchmark (T1-family, never closed loop), zero-shot PhysicalAI-AV B1 -> nuPlan cameras,
non-parity. Zero GPU used; CPU only, <= 3 processes of ~0.5 GB. Source package:
`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/` (`PKG`).

## 0. Bottom line (read this first)

**Question (gap G1):** why does refcv7 fail navhard (official two-stage EPDMS 0.2269 vs STOP 0.2985) through DAC-zero on 26.0 % and
NC-zero on 15.6 % of stage-2 scenes? **Answer, measured scene by scene (all 5,912 scorer tokens, both stages), with exact re-scoring
of every failing scene:**

1. **DAC-zero is a steering-geometry failure, not an initial-state or sampler-lottery failure.** 1,563 scenes (145 stage-1 = 32.2 %,
   1,418 stage-2 = 26.0 %). **0 of 1,563 leave the drivable area at t = 0** (no initial-state failures); the first non-drivable instant
   is spread over the whole 4 s (0-1 s 32.4 %, 1-2 s 22.5 %, 2-3 s 21.0 %, 3-4 s 24.0 %; median 1.8 s). **72.7 % [68.2, 77.4]
   (1,136) have a DAC-clean path** (the PDM-Closed reference passes), so the failure belongs to the plan; **27.3 % [22.6, 31.8] (427)
   have no proven clean path** (the reference fails too; 296 = 18.9 % are scenes where STOP, PRIOR and the reference all fail).
   The failures are **stable across the inference seed** (second seed's plan is DAC-clean on only 121 of 1,563 = 7.7 %; Jaccard 0.86
   stage 2), so this is a systematic planner/selector/generator error, not a lucky-draw error.
2. **Geometry classes of the 1,563 DAC-zero scenes** (route-referenced; literals in section 2; log-cluster 95 % CI):
   LATERAL-DRIFT 483 (30.9 % [26.8, 35.2]) | ROUTE-FOLLOWING (under-turn) 363 (23.2 % [18.5, 28.1]) | ON-ROUTE 287 (18.4 %
   [15.3, 21.5]; threshold-dependent, 4-38 %) | OVER-STEER 203 (13.0 % [9.6, 16.2]) | SPEED 147 (9.4 % [7.7, 11.1]) | NO-RECOVERY 42 (2.7 %) |
   WRONG-SIDE 34 (2.2 % [1.1, 3.3]) | STOP-LIKE 4 (0.3 %). Restricted to the 1,136 with a clean path:
   DRIFT 346 / ROUTE-FOLLOWING 256 / ON-ROUTE 243 / OVER-STEER 170 / SPEED 66 / NO-RECOVERY 31 / WRONG-SIDE 24.
3. **NC-zero is a LONGITUDINAL failure: 94.8 % (843 of 889) are front collisions** -- ACTIVE-FRONT into a slower vehicle 504 (56.7 %
   [48.5, 65.1]; object 1.7 m/s, refcv7 6.1 m/s, 6.2 m ahead), STOPPED-TRACK (parked / stopped vehicle) 307 (34.5 % [26.2, 43.0]),
   front-VRU 24 + 8. Lateral-after-lane-departure only 27 (3.0 %); initial overlap 19 (2.1 %, STOP fails all 19). The emitted plan is
   **+3.87 m/s faster at 4 s than the PDM-Closed reference in NC-zero scenes (median) vs -0.48 m/s over all scenes**; 64.8 % of
   NC-zero plans are >= 3 m/s faster vs 23.3 % overall. The kinematic PRIOR collides in the same scenes (67 % / 60 % of the two
   big classes): the learned part does not buy collision avoidance over its own prior (A1 NC0 15.6 % vs PRIOR 16.3 %).
4. **Lever sizing on the official number (ESTIMATED counterfactual, other sub-scores held):** replacing A1's DAC by STOP's on the
   scenes where A1 is zero and STOP is not lifts official EPDMS 0.2269 -> **0.3504** (A1 - STOP -0.0716 -> **+0.0519**); the 145
   stage-1 DAC-zero rows alone give +0.0975 (-> 0.3245, already above STOP). NC likewise +0.0535, DDC +0.0109, TLC +0.0091.
   **DAC alone is enough to clear the NH1 bar; stage 1 matters out of proportion** because the stage-1 score multiplies its
   stage-2 scores inside every key (99 of the 225 keys have a zero first-branch stage-1 factor, the orig-frame score).
5. **Class mix 5,000 -> 30,000:** DAC-zero 2,113 -> 1,563 scenes. LATERAL-DRIFT 804 -> 483, ROUTE-FOLLOWING 626 -> 363,
   WRONG-SIDE 73 -> 34 shrank; **OVER-STEER 220 -> 203, ON-ROUTE 283 -> 287 did not move; SPEED 67 -> 147 doubled**; NO-RECOVERY flat.
   Training removed route-following and drift errors and left the turn-execution (over-steer) and boundary errors untouched.
6. **navtest is the same mechanisms at a lower rate** (DAC-zero 12.0 %, 1,460 tokens; vs the human future: ROUTE-FOLLOWING 19.9 %,
   OVER-STEER 19.5 %, LATERAL-DRIFT 15.7 %, ON-HUMAN 41.0 %, SPEED 3.2 %; NC-zero 8.3 %, 1,008: 46.0 % are plans >= 1.25x and >= 4 m
   further than the human, and in that class NC is zero for 52.8 % vs STOP 1.8 %).
7. **Master Mind stratifiers (D4):** (a) **the all-zero "unknown" max-speed row is NOT a disproportionate failure source**:
   navhard A1 DAC0 25.3 % unknown vs 27.4 % known, difference-in-differences vs STOP -1.7 pp [-5.7, +1.7]; navtest unknown shows
   +3.8 pp [+1.1, +6.3] over STOP, but the input-free PRIOR shows the same excess (standardised +4.4 pp vs A1 +4.1 pp; A1-vs-PRIOR
   DiD -1.1 pp [-5.1, +3.3]) -- a scene attribute, not the input. On NC the known limit is the riskier row (A1-vs-PRIOR DiD on
   navtest -3.6 pp [-5.7, -1.6]; the row shifts the emitted speed by ~-0.6 m/s beyond the control). No cheap train/deploy fix is
   indicated by this split. (b) **Command x geometry:** where the command is LEFT/RIGHT but the route is still straight inside the
   4 s horizon (724 scenes) the plan turns anyway in ~59 % and **DAC fails 37.9 % / 43.2 % when it turned vs 8.9 % / 25.0 % when it did
   not** (premature-turn association); curve turns are not worse than junction turns once STOP is subtracted
   (A1 - STOP: junction L +29.1 / R +24.5 pp, curve L +23.8 / R +23.3 pp).

**Rule-Zero position:** these are attributions, and each class below has a named lever and a costed next probe. The classes whose
lever is not yet identified (LATERAL-DRIFT, ON-ROUTE, OVER-STEER and NO-RECOVERY among the clean-path scenes = 790 = 50.5 % of all DAC-zero, plus the 843 front-collision NC scenes) have a
zero-training probe designed, costed and measured on the CPU side; the one blocker is **the GPU lock** (held by the NavSim 50,400 job) for the candidate-fan export (section 9).

## 1. Provenance and controls (every check below READ ITS KNOWN VALUE)

Inputs (all banked, read-only): per-token score frames `scores_navhard/score_<arm>__navhard_two_stage_wrapper/*_final_scores_frame.csv`
(sub-scores + the stage-2 Gaussian `weight` + the official per-token `score`), `*_hooks.json` (the plan the scorer received, the human future for
stage-1 tokens, v0), `bridge_navhard/rows_R7_A1.jsonl` (declared inputs: max-speed row, nav), the navhard agent-inputs export
(`C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json`: stage, log, driving command, ego status),
the floors frames (`raw/floors/navhard_two_stage`), and for the exact re-score the LOCAL NavSim metric cache
(`C:/Users/Admin/navsim-crun/exp/metric_cache_navhard_two_stage`, 5,912 per-token pickles) with the devkit's own `pdm_score`.
Navtest uses `D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz` (human future, commands) + the navtest score CSVs/hooks/rows.

| control | expected (known) value | read | result |
|---|---|---|---|
| official two-stage EPDMS recomputed from the frames with the devkit aggregation (`taniteval/adapters/navsim_ci.py`) | A1 30k 0.226926, A1 seed-1 0.223296, A1 5k 0.162430, PRIOR 0.130040, STOP 0.298532, CV 0.114816 (summary JSONs) | all six equal to <= 1.1e-16 | PASS |
| stage-2 multiplier zero-rates recomputed | summary JSON `S2_multiplier_zero_rates` (NC/DAC/DDC/TLC, six arms) | equal to <= 1e-12 | PASS |
| `score` == prod(NC,DAC,DDC,TLC) x (5EP+5TTC+2LK+2HC+2EC)/16 | identity | max abs diff 2.2e-16, 0 rows > 1e-9 | PASS |
| **exact re-score of the emitted plan in the local metric cache** (`code/d6_rescore.py`; devkit `PDMSimulator` + `PDMScorer`, IDM-reactive policy, human penalty filter) | the banked 8 sub-scores | **2,678 tokens (all 1,563 DAC-zero, all 889 NC-zero, 275 DDC-only, 200 random all-pass), max abs diff 0.0 on every sub-score, 0 non-OK** | PASS |
| NavSim command order [left, straight, right, unknown] (analytic: human 4-s heading by command, 450 stage-1 tokens) | LEFT > STRAIGHT > RIGHT | +0.718 / +0.007 / -0.702 rad medians | PASS |
| PDM-Closed reference lateral offset | exactly one of {-1, 0, +1} m (PDM-Closed's own offset set) | all 5,912 within 1 mm of {0, 1} in absolute value | PASS |
| classifier self-test (9 analytic plans incl. a left 90-degree turn driven straight, driven right, cut inside; a 3 m drift; a start offset kept) | the named class | 9 of 9 | PASS |
| class rule applied to the HUMAN future (stage 1, 450) | WRONG-SIDE ~ 0 | 1 of 450 (0.2 %); ON-ROUTE 74.4 % | PASS |
| STOP-LIKE plans | DAC0 rate must equal STOP's (a plan that does not move is STOP) | 1.2 % vs 1.2 % (n 341) | PASS |
| navtest: PDMS recomputed | A1 30k 71.8849, STOP 61.8202, A1 5k 65.5976 | 71.8849 / 61.8202 / 65.5976 | PASS |
| navtest: classifier on the human against itself | only ON-HUMAN / STOP-LIKE | 11,920 / 226 | PASS |
| max-speed known-flag identical between 5k and 30k bridges | identity | True | PASS |
| unreadable files / unparseable lines | 0 | 0 / 0 (geom 5,912 lines, rescore 2,678, hooks 5,912 x 4 arms) | PASS |

**Independent re-derivation** (`code/d6_verify.py` -> `raw/d6_verify.json`): the headline counts (DAC-zero 145 + 1,418, NC-zero 35 + 854, 1,563 / 889 from the PRE_AGGREGATION frames, the seed-1 clean count 121, reference-also-fails 427, t = 0 violations 0, front-collision NC 843) re-read from the official score CSVs, the PRE_AGGREGATION frames and the raw rescore JSONL without `d6_common`: **17 of 17 checks, including 1 mutation control** (a wrong expectation of 1,417 is rejected).

## 2. Definitions -- every threshold is a literal fixed BEFORE the class counts were read

Scorer facts used (from the devkit source in `C:/Users/Admin/navsim-crun/devkit/navsim`): DAC = 0 if any of the four ego-footprint corners is outside
the union of ROADBLOCK / INTERSECTION / DRIVABLE_AREA / CARPARK polygons at ANY of the 41 simulated states (0.1 s) of the LQR-tracked plan;
the human-penalty filter applies to ORIGINAL (stage-1) frames only, and **stage-2 has no banked human future** (the hooks carry `human_poses = null`),
so stage-2 is referenced to the PDM-Closed reference path and the route centreline. NC = 0 for an at-fault collision with an agent (ACTIVE_FRONT,
STOPPED_TRACK, or ACTIVE_LATERAL while the ego is in multiple lanes / non-drivable area); STOPPED_EGO and ACTIVE_REAR are not at fault.

Route features (`code/d6_rescore.py --geom-only`, `code/d6_part2_anatomy.py`): `lat` = signed offset of the plan point from the route centreline
(left +); `lat0` = start offset; `d_route` = route heading change over D = clip(4 v0, 10, 40) m ahead; `head_err` = plan yaw at 4 s minus the route tangent at
the plan's own progress; `junction` = any centreline sample within D lies in an INTERSECTION polygon; otherwise a turn is a `curve`.

| literal | value | use |
|---|---|---|
| T_TURN | 0.35 rad (20 deg) | route heading change that makes a turn |
| T_HEAD | 0.175 rad (10 deg) | heading deviation from the route that counts |
| T_LAT | 1.5 m | lateral offset (lane half-width ~1.75 m) |
| T_AWAY | 0.75 m | excursion beyond the start offset = drifting away (else NO-RECOVERY = inherited offset) |
| T_STOP | 1.0 m | 4-s endpoint distance below which a plan is STOP-LIKE (the banked stop-fraction literal) |
| SPEED | plan route-progress >= 1.25 x reference and >= 4 m more | SPEED |

Class cascade (first match): STOP-LIKE; on a turning route WRONG-SIDE (plan yaw opposite, |yaw| >= T_HEAD) -> ROUTE-FOLLOWING (plan heading lags the route by >= T_HEAD or
ends >= T_LAT outside the curve) -> OVER-STEER (>= T_HEAD more than the route or >= T_LAT inside); on a straight route LATERAL-DRIFT (excursion >= T_AWAY or |head_err| >= T_HEAD) -> NO-RECOVERY
(|lat| >= T_LAT kept from the start); then SPEED; else ON-ROUTE. Scene-level flag for DAC-zero scenes: INITIAL-STATE (non-drivable at t = 0), REF-ALSO-FAILS (PDM-Closed reference
fails DAC), else PLAN-INDUCED (a clean path exists -- the name means "the failure is the plan's", not "the plan caused the scene").
Navtest classes use the same literals against the HUMAN future at 4 s (`code/d6_part3_navtest.py`), so ON-HUMAN there is "within thresholds of the human endpoint" and misses mid-path excursions.
**Sensitivity (T10):** every literal x0.5 and x1.5 (one at a time and all together) moves ON-ROUTE 4.4-37.9 %, NO-RECOVERY 0.8-12.4 %, LATERAL-DRIFT 22.1-38.6 %, ROUTE-FOLLOWING 15.8-32.3 %,
OVER-STEER 5.8-21.1 %, SPEED 3.4-11.8 %, WRONG-SIDE 1.8-3.5 %; LATERAL-DRIFT and ROUTE-FOLLOWING are in the top 3 in all 17 scenarios (the third is ON-ROUTE or OVER-STEER; at all-x1.5 ON-ROUTE is first); quote class shares with that range.

## 3. Deliverable 1 -- the per-scene failure table

`raw/d6_scene_table_FULL_step30000.csv` (5,912 rows = every navhard scorer token; 122 columns). Columns: token, stage, log_name, map, command
(LEFT/STRAIGHT/RIGHT/UNKNOWN), v0 and band; for each of A1-30k, A1-seed-1, A1-5k, PRIOR, STOP, CV: NC, DAC, DDC, TLC, EP, TTC, LK, HC, EC, official score, stage-2 weight, zero-term string;
geometry (route direction, junction/curve, class at 30k / 5k / PRIOR / seed 1, lat0, lat4, yaw4, head_err, plan and reference progress and speed, `complied` with the command);
the exact re-score (first non-drivable step `nd_first`, reference DAC `ref_dac`, first at-fault collision class / object / speeds / relative position); the max-speed known flag; and
`dac_scene_flag`, `pattern_DAC|NC|DDC|TLC` (ALL-ARMS-FAIL / STOP-ALSO-FAILS / A1-AND-KINEMATIC-ARMS-FAIL / A1-ONLY), `refcv7_failed_where_STOP_passed_any_mult`.
Aggregates (`raw/d6_failure_patterns.json`): stage 2 -- A1 has a zero multiplier where STOP passes on **1,719 of 5,462** scenes (31.5 %), stage 1 on 160 of 450 (35.6 %);
A1's official per-scene score is exactly 0 on 2,331 stage-2 (42.7 %) and 178 stage-1 (39.6 %) scenes. DAC-zero stage 2: ALL-ARMS-FAIL 594 (41.9 %), A1 + kinematic arms fail but STOP passes 608 (42.9 %),
**A1-ONLY 206 (14.5 %)**, STOP-also-fails-only 10; NC stage 2: ALL-ARMS 193, A1+kinematic 462, **A1-ONLY 193**. T2 below gives who-else-fails per term and stage.
Reading rule for "refcv7 failed where STOP/prior passed": use the `pattern_*` columns; "every arm failed" = ALL-ARMS-FAIL.

## 4. Deliverable 2 -- DAC-zero anatomy (tables T3-T6, T10 in the appendix)

* **What the emitted plan did** (T4): the failures are not one thing. LATERAL-DRIFT (30.9 %) is the plan moving off the centreline (>= 0.75 m beyond its start offset, or a >= 10 degree swerve) on a road the route keeps straight;
  in the 1,526 scenes where A1's plan is in this class, A1 fails DAC in 31.7 %, STOP in 16.6 % and the input-free PRIOR in 60.9 %; the PRIOR's own DAC-zero set is 60.4 % LATERAL-DRIFT (1,684 of 2,788 scenes, T5) against A1's 483 -- the learned residual removes ~71 % of the prior's drift-class failures but not the rest.
  ROUTE-FOLLOWING (under-turn; 23.2 %) has the highest enrichment among the large classes (P(DAC0 | class) 47.8 % vs STOP 18.0 %) and OVER-STEER (13.0 %) the highest overall (60.4 % vs 16.7 %).
  WRONG-SIDE (2.2 %, 57.6 % vs 30.5 %) is rare. SPEED (9.4 %) is plans that outrun the reference on an otherwise correct line.
* **Command compliance** (T6): the plan turned toward a LEFT/RIGHT command in 77-91 % of the scenes whose route turns inside the horizon, yet DAC fails 39-55 % there (STOP 14-32 %): the dominant error is turn EXECUTION geometry
  (under-turn 273 junction + 90 curve scenes, over-steer 144 + 59), not the decision to turn. ROUTE-FOLLOWING by command: LEFT 118 / RIGHT 192 / STRAIGHT 53; by route kind junction 273 / curve 90;
  P(DAC0 | ROUTE-FOLLOWING) junction 43.3 % (n 630) vs curve 69.2 % (n 130). WRONG-SIDE: LEFT 20 / RIGHT 9 / STRAIGHT 5; junction 24 / curve 10.
* **Premature turn (association, UNVERIFIED as cause):** command LEFT/RIGHT with a straight route in the horizon -- LEFT n 389: plan turned 59.6 %, DAC0 26.2 % (turned 37.9 % n 232; not turned 8.9 % n 157); RIGHT n 335: turned 59.4 %,
  DAC0 35.8 % (turned 43.2 % n 199; not turned 25.0 % n 136). The refcv7 nav input is a per-clip oracle token on every window (CONTEXT), the NavSim command also fires before the geometry.
* **Curve vs junction** (D4 finding 2): of 1,058 LEFT / 1,254 RIGHT scenes the route turns inside the horizon in 63 % / 73 % (junction 488 / 769, curve 181 / 150) and is straight in 37 % / 27 %.
  A1 - STOP DAC0: junction LEFT +29.1 pp, RIGHT +24.5 pp; curve LEFT +23.8 pp, RIGHT +23.3 pp. Curves have higher absolute DAC-zero (51-55 % vs 39-45 %) because STOP also fails there more (27.6 / 32.0 %); the excess over STOP is not larger on curves.
  A STRAIGHT command with a turn in the horizon (280 junction, 36 curve scenes) has A1 DAC0 25.7 % / 50.0 % vs STOP 6.4 % / 8.3 %.
* **Step 5,000 vs 30,000** (T5): see section 0 item 5. Of the 1,563 DAC-zero scenes at 30k, 280 (17.9 %) were DAC-clean at 5k (regressions), so 30k trades some old errors for new ones.
* **Stage 1 (450 original frames)**: 145 DAC-zero (32.2 %); the human future passes DAC in all 145 (the human penalty filter exempted none); 97.2 % have a clean path; ROUTE-FOLLOWING 49 (33.8 %), ON-ROUTE 38 (26.2 %), LATERAL-DRIFT 31 (21.4 %), OVER-STEER 17 (11.7 %);
  the first non-drivable instant is late (114 of 145 after 2 s). 130 of the 145 (90 %) fail again with the other inference seed.
* **Where PRIOR and STOP stand** (T2/T3b): of A1's 1,563 DAC-zero scenes, a clean path exists and BOTH STOP and PRIOR are clean in 323 (20.7 %: refcv7 is worse than doing nothing AND worse than its own prior),
  PRIOR fails but STOP is clean in 486 (31.1 %), both fail in 315 (20.2 %), the reference fails too in 427 (27.3 %).

## 5. Deliverable 3 -- NC-zero anatomy (T7)

889 NC-zero scenes (35 stage-1, 854 stage-2), all re-scored (reproduction exact). Classes by the FIRST at-fault collision event: ACTIVE-FRONT-VEHICLE 504 (56.7 %), STOPPED-TRACK vehicle/object 307 (34.5 %),
LATERAL-AFTER-LANE-DEPARTURE 27 (3.0 %), ACTIVE-FRONT-VRU 24 (2.7 %), INITIAL-OVERLAP 19 (2.1 %), STOPPED-TRACK-VRU 8 (0.9 %). Collision time median 2.1-2.5 s (VRU stopped-track 3.1 s, initial overlap 0.1 s); the ego is at 5-6 m/s against a
1.7 m/s (or stationary) object 6 m ahead. Speed: plan 4-s speed minus reference 4-s speed median **+3.87 m/s** in NC-zero scenes vs -0.48 m/s overall. NC-zero rate by start speed: 8.0 % (0-2 m/s), 17.8 % (2-5), 20.6 % (5-8), 15.0 % (8-12), 8.8 % (>= 12)
vs STOP 1.1-5.7 %. **By command: LEFT 15.5 %, RIGHT 9.7 %, STRAIGHT 16.8 %** -- NC is not a turn problem. 249 of 889 (28 %) also fail DAC (138 of the 307 stopped-track scenes -- the off-lane plan meets parked vehicles).
Stability: seed 1 also NC-zero in 483 of 504 / 278 of 307. Co-failure: PRIOR 337 / 185, STOP 94 / 75.

## 6. Deliverable 4 -- navtest (single-stage PDMS v1; T9)

Same mechanisms, lower rate; the evidence is plan-vs-human at 4 s only (no re-score: the navtest cache needs the navsim-1.1 devkit tree, not run). DAC-zero 1,460 (12.0 %): ON-HUMAN 599 (41.0 %), ROUTE-FOLLOWING 291 (19.9 %), OVER-STEER 285 (19.5 %),
LATERAL-DRIFT 229 (15.7 %), SPEED 46 (3.2 %), WRONG-SIDE 10 (0.7 %); at 5k 1,876 (15.4 %) with ROUTE-FOLLOWING 501 (26.7 %). STOP also fails 219, PRIOR 895, seed 1 1,247 (Jaccard 0.74); STOP and PRIOR both clean in 502 (34 %).
Per-class DAC0 rates are lower than navhard's (ROUTE-FOLLOWING 27.2 % vs 47.8 %), so navhard is harder per class as well as richer in failures. NC-zero 1,008 (8.3 %): SPEED class 464 (46.0 %; P(NC0 | class) 52.8 % vs STOP 1.8 %),
plan >= 4 m further than the human in 57.0 % of NC-zero vs 17.2 % overall; STOP also NC-zero 23, PRIOR 434, 77 also DAC-zero. Command: LEFT DAC0 21.6 % (STOP 7.2 %), RIGHT 20.8 % (5.9 %), STRAIGHT 7.4 % (1.8 %); NC0 STRAIGHT 9.5 % (STOP 2.4 %).
Not available for navtest without the v1.1 re-score: time of first violation, collision type (cost in section 9, P3).

## 7. Master Mind stratifiers (D4) -- n per cell in T6 and T8

Max-speed input (T8): navhard 2,697 unknown / 3,215 known scenes (45.6 % / 54.4 %), navtest 5,488 / 6,658 (45.2 % / 54.8 %), confirming D4's ~45 %. Zero rates per stratum are in T8 with n; the standardised
(stage x v0 band x command) unknown-minus-known differences and the log-cluster DiD vs both control arms are in T8. Reading: navhard shows no excess in the unknown stratum (A1 DAC0 25.3 % vs 27.4 %, NC0 14.1 % vs 15.9 %).
navtest DAC excess in unknown (+4.06 pp standardised) is reproduced by the input-free PRIOR (+4.44 pp), so it is a scene property; NC is LOWER when the row is unknown (A1 -3.65 pp standardised; vs PRIOR DiD -3.58 pp [-5.70, -1.63]).
Plan speed: A1 emits 1.44 m/s less (vs the reference) in unknown scenes after standardisation, PRIOR 0.86 m/s -> the row's own effect is ~-0.6 m/s (`raw/d6_part2b_vmax_speed.json`). **Conclusion: the unknown row is not where DAC/NC failures concentrate; the cheap-fix hypothesis is not supported by this split.**
Caveat: no navhard VMAXOFF arm exists (the 200-token navtest VMAXOFF_sub is D5's SPD-4 row), so the causal statement rests on the PRIOR control.
Command / curve-vs-junction (T6, section 4): n per cell are in the tables.

## 8. Attribution table -- failure class x count x share x refcv8 lever (A1 step 30,000, navhard, both stages)

Shares are of the 1,563 DAC-zero scenes (DAC) or the 889 NC-zero scenes (NC); n and evidence class per row. All rows MEASURED (artifact paths in section 10) unless marked; "association" means no ablation exists.

| failure class | n | share [95 % log-cluster CI] | evidence | refcv8 lever it maps to | why this lever / what is missing |
|---|---|---|---|---|---|
| DAC / no clean path proven (reference also fails) | 427 | 27.3 % [22.6, 31.8] | M `d6_scene_table_FULL` | **NONE** (scene-level floor; 296 fail STOP+PRIOR+reference) | no arm in the package passes; do not spend a lever on it, but report it as an irreducible share beside any navhard number |
| DAC / LATERAL-DRIFT (clean path exists) | 346 | 22.1 % | M | **NONE-YET** (selector direction vs generator vs map) -> fan probe P1 | PRIOR has 1,684 drift-class DAC-zeros vs A1 483: the learned residual helps but is not enough; D5 NS-4 candidates (MAP-2/4, SEL-1) not separated |
| DAC / ROUTE-FOLLOWING (under-turn) + WRONG-SIDE | 280 | 17.9 % | M, association | **L2 time-localised nav + L1 dense tactical (turn execution)** | 339 of the 397 ROUTE-FOLLOWING/WRONG-SIDE scenes carry a LEFT/RIGHT command; the command is present, the plan under-turns; no navhard NAVOFF arm -> causal test P2 |
| DAC / ON-ROUTE (plan within thresholds, still fails) | 243 | 15.5 % (4-38 % over thresholds) | M, threshold-sensitive | **NONE-YET** (lane-edge / footprint / map polygons) | 55 of the 243 fail within 1 s; resolve with P1 (does any fan member pass) before assigning |
| DAC / OVER-STEER | 170 | 10.9 % | M | **L1 dense tactical (turn execution) / selector direction**; NONE-YET for which | did not shrink 5k -> 30k (220 -> 203); command LEFT/RIGHT in 191 of 203 |
| DAC / SPEED | 66 | 4.2 % | M | **L3 speed input / selector speed** | SPEED class doubled 5k -> 30k (67 -> 147 over both groups) |
| DAC / NO-RECOVERY (keeps a start offset) | 31 | 2.0 % | M | **NONE-YET** (candidate: perturbed-start recentring augmentation) | P(DAC0 | class) 11.0 % over 382 scenes vs STOP 5.2 %; small |
| NC / ACTIVE-FRONT-VEHICLE | 504 | 56.7 % [48.5, 65.1] | M (exact re-score) | **selector speed training + L1 tactical longitudinal labels (FOLLOW / BRAKE_TO / HOLD) + lead-vehicle perception (box)** -- which of the three is NONE-YET | plan +3.87 m/s over reference; PRIOR collides in 67 %; seed 1 in 96 %: systematic |
| NC / STOPPED-TRACK (parked or stopped vehicle) | 307 (+8 VRU) | 34.5 % [26.2, 43.0] | M | same as above; **box / perception** weighs more (a stationary object ahead) | 138 of 307 also DAC-zero (off-lane plans meet parked cars) |
| NC / LATERAL-AFTER-LANE-DEPARTURE | 27 | 3.0 % | M | follows the DAC lateral classes | consequence of DAC-type failures |
| NC / INITIAL-OVERLAP | 19 | 2.1 % | M | **NONE** (STOP fails all 19) | scene-level |
| NC / ACTIVE-FRONT-VRU | 24 | 2.7 % | M | box / VRU perception | small n |

Counterfactual weight (ESTIMATED, T1): DAC (where STOP is clean) +0.1235 official EPDMS, stage-1 DAC alone +0.0975, NC +0.0535 (stage-2 rows +0.0423, stage-1 rows +0.0177). Largest measured effect first (Rule Zero item 5): **DAC, and within it stage 1 and the turn/lateral classes.**

## 9. Cheapest next probes for what remains unattributed

> UPDATE: P3 below was RUN (section 12, Appendix B); P1 and P2 are REGISTERED as `SPEC_P1P2.md` + `SPEC_P1P2_A1.md` (the fan is 117 anchors; no WTA heads in this checkpoint) and queued behind the GPU lock. The text of this section is the plan as first written; the registered SPEC governs.

* **P1 -- candidate-fan probe (D_core analogue for DAC and NC): for every DAC-zero / NC-zero token, does the 117-anchor fan (+ 64 WTA proposals) hold a clean candidate, and does the pick take it?** Decides SELECTION (SEL-1) vs GENERATION vs MAP for the 790 clean-path LATERAL / ON-ROUTE / OVER-STEER / NO-RECOVERY scenes (50.5 % of all DAC-zero) and the 843 front-collision NC scenes.
  Needs: a ~5-line change in `refcv7_bridge.py` to export `out["anchor_traj"]` ([B,117,S,2], `refc_v3.py:2247`) and the `r7_*` WTA proposals, headings from the bridge's own pose builder. GPU (BLOCKED by the lock held by the NavSim 50,400 job): CUDA native path 0.35-0.42 s/scene (MEASURED on warmup, package RESULT s3) x 1,563 DAC-zero tokens
  = ~11 min + 4.3 s model load (seam manifest) ; with the 889 NC tokens ~16 min. CPU: scoring 117 candidates per token measured at **1.0-1.3 s/token** (`raw/d6_fanbench.json`: build 0.17 + simulate 0.14 + score 0.54-0.90 s; synthetic candidates, timing only) = ~30-35 min for the DAC-zero set, one process.
  For NC the environment is reactive to the ego (one IDM rollout per candidate, ~1 s each): score NC on the 5 best DAC-clean candidates only (+~2 h) or use the picked plan's environment as an approximation (state it).
* **P2 -- nav causal test (zero training):** run the NavSim bridge on the 724 scenes with a LEFT/RIGHT command and a straight route (and the 1,157 junction/curve turn scenes) with nav withheld (`R7_NAVOFF`, exists in the bridge) at step 30,000: ~1,900 x 0.4 s = ~13 min GPU (same lock) + CPU scoring ~1 h. Reads: does DAC fail less without the premature turn, and does the under-turn grow.
* **P3 -- zero-GPU, runnable now (not run, budget):** (a) speed-scaled counterfactual for NC: re-score A1's 889 NC-zero plans with along-track progress x 0.6 (CPU ~2 s each = ~30 min) -> how much of NC is pure speed commit; (b) lateral-snap oracle for the 790 clean-path lateral/on-route/over-steer scenes (shift the plan toward the centreline by <= 0.75 m, CPU ~1.2 s each = ~16 min) -> boundary-level vs gross error;
  (c) navtest re-score of the 1,460 DAC-zero / 1,008 NC-zero tokens needs the navsim-1.1 devkit tree `D:/Archive/devbox-C/navsim/navsim-3e8291bf...` (an environment hop, ~1 h CPU) for time-of-violation and collision type.
* **Blockers named (Rule Zero 3):** the GPU lock (NavSim 50,400 runner) for P1/P2; the bridge export change for P1 (owner: EvalFlyWheel / Master Mind); nothing else.

## 10. What I could not do / caveats (plainly)

* Stage-2 has no human future: stage-2 classes are referenced to the PDM-Closed reference and the route centreline, stage-1 to both (the class rule on stage-1 humans reads WRONG-SIDE 0.2 %). ON-HUMAN on navtest is endpoint-based and hides mid-path excursions.
* Class shares depend on the literals (T10 range). ON-ROUTE (4-38 %) and NO-RECOVERY (0.8-12 %) are the unstable ones; DRIFT / ROUTE-FOLLOWING / OVER-STEER are stable in rank.
* The route centreline is the cached PDM-Closed centreline, and `junction` is INTERSECTION-polygon membership of centreline samples; a junction label is not an announced-turn label (D4/refcv8's definition) and I did not have one.
* Intervals are log-cluster bootstraps (76 navhard logs, 136 navtest logs, B = 2000, seed 0): they answer "another draw of logs?" only; they are blind to training variance (H-ESTIM-SEED-1) and the only inference-variance evidence is the seed-1 overlap. No training-seed replicate exists; no class shift here is promoted to a lever effect.
* The counterfactual EPDMS rows hold every other sub-score (incl. EP normalisation) fixed -- upper bounds.
* "Premature turn" and "known limit raises collisions" are MEASURED ASSOCIATIONS; both are UNVERIFIED as causes (P2, VMAXOFF at scale).
* STOP's own DAC zeros (13.7 %) were not re-scored; I do not state their mechanism. A1 has no t = 0 violation, so INITIAL-STATE is empty by measurement, not by definition.
* The D5 inventory row NS-4 lists the fan probe as PROPOSED; this stream did the nearest zero-GPU thing (exact re-score + three baselines + second seed) and measured the fan-scoring CPU cost, but did NOT score a real fan: the fan is not banked.
* Navtest numbers use the package CSVs/hooks only; no navtest re-score.

## 11. Re-run

```
PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2   # tanitad venv for parts 1-6; navsim venv for d6_rescore.py / d6_fanbench.py
C:/Users/Admin/venvs/tanitad/Scripts/python.exe code/d6_part1_table.py        # controls, per-scene table, cross-arm patterns, counterfactuals
# navsim venv, PYTHONPATH="C:/Users/Admin/navsim-crun/devkit;C:/Users/Admin/navsim-crun/nuplan-devkit":
#   d6_rescore.py --geom-only ... --out raw/geom_all.jsonl ; d6_rescore.py --human --hooks <A1 30k hooks> --tokens raw/tokens_rescore_{A,B}*.txt --out raw/rescore_{A,B}*.jsonl ; d6_fanbench.py
C:/Users/Admin/venvs/tanitad/Scripts/python.exe code/d6_part2_anatomy.py ; d6_part3_navtest.py ; d6_part4_intervals.py ; d6_part5_sensitivity.py ; d6_part2b_vmax_speed.py ; d6_part6_merge.py ; d6_make_tables.py
```

## 12. Continuation after acceptance (Rule Zero): P3 oracles run, P1/P2 registered and queued

**12.1 Status (2026-10-04, local Berlin time).**
* **P3 done** (CPU only; tables in Appendix B). All three oracles were validated on known values first: the identity plan reproduces the banked sub-scores exactly (40 scenes, max abs 0.0), the speed-scale-0 plan reproduces the OFFICIAL STOP frame (max abs 8.3e-17), the identity snap stays DAC-zero on 1,563 of 1,563, and on navtest the v1.1 re-score reproduces the banked row on 2,491 of 2,491.
* **SPEC_P1P2.md REGISTERED** by the Master Mind (sha256 `4eaf655f8e000f5ae06a4b0ba93ff05be04a037064d1c16d108e7c11c1f5f4bd`, 2026-10-04T12:00:42Z) and **amendment A1 REGISTERED** (`SPEC_P1P2_A1.md`, sha256 `def9b4c24222d47dc49de0174690279c4819a7da1dc896954d7c6b582f1392bb`,
  12:09:57Z), both before any P1/P2 number existed (`raw/SPEC_SHA256.txt`). **A1 corrects a defect in my own SPEC found before launch:** refcv7-r101-s0 as launched has no WTA decoder and no refcv7 scorer (launch argv carries no `--refcv7`,
  `w_r7_wta = w_r7_scorer = 0.0`, no r7 group in `param_breakdown`), so the candidate universe is the 117 fan, and the "free lever" is "emit the decoder's own pick (E9 graft off)", not `refcv7_select`.
* **Bridge export built** in `code/bridge_fix/` (insert-only copy of `run_bridge7.py` / `refcv7_bridge.py`; `boot7.py`, `gpu_lock.py` byte-identical; base blobs in `BASE_BLOBS.txt`; `test_default_identical.py` 23/23 incl. a mutation control). The live `navsim/code/` files were not touched.
* **Queue state (read from the launch artifacts, not exit codes):** `raw/gate.log` shows the gate found `raw/SPEC_A1_REGISTERED.txt` at 14:10:21 and started the battery's `with_gpu_lock.py` (pid 34564); `raw/gpu_wait.log` shows the repeated
  `[gpu_lock] WAIT job=refcv8-d6-p1p2 holder={'job': 'refcv7-milestone-step50400', ...}`; `raw/gpu_rec.json` does not exist yet; the lock file still names the battery. **No P1/P2 number exists.** The CPU stage (`run_p1p2_cpu.py`) waits for the GPU chain's done-marker.

**12.2 What the P3 ceilings say (upper bounds on what a PERFECT fix of each kind would clear; Appendix B has n, CIs and the method).**

| lever kind | navhard (A1 30k) | navtest (A1 30k) |
|---|---|---|
| **selector speed** (the same path at 0.6 / 0.4 / best-of-three x the planned speed), share of NC-zero cleared | 30.6 % [26.0, 35.1] / 45.9 % [40.3, 51.0] / **46.0 % [40.4, 51.2]** (409 of 889); **200 (22.5 %) are not fixable by ANY speed** (STOP also collides); of the 689 scenes where STOP passes, 58.6 %; official EPDMS ceiling 0.2269 -> 0.2880 (+0.0611; still below STOP 0.2985) | 81.2 % / 93.4 % / **93.7 % (944 of 1,008)**; PDMS ceiling 71.88 -> 78.82 (+6.94) |
| **lateral fix, boundary-level** (shift toward the route centreline by <= 0.75 m / <= 1.5 m), share of DAC-zero cleared | 13.1 % [10.6, 15.9] / 17.3 % [14.5, 20.3]; EPDMS +0.0307 / +0.0369 | 24.0 % / 27.5 % |
| **lateral fix, route-follow oracle** (every point on the centreline at the plan's own along-route progress) | **51.4 % [46.4, 57.4] (803 of 1,563; 795 of the 1,136 clean-path scenes = 70.0 %, 8 of the 427 reference-also-fails)**; EPDMS ceiling +0.1134 -> 0.3403 (above STOP 0.2985) | **85.6 % (1,250 of 1,460)**; PDMS ceiling 71.88 -> 76.95 (+5.07) |
| **speed only, DAC** (0.6 / 0.4 x) | 27.2 % / 38.0 %; EPDMS +0.0674 (0.6 x) | 54.5 % / 67.1 % |
| union (FULL snap or 0.6 x or 0.4 x) | 59.8 % (934 of 1,563) | n/a |

Readings:
* **The DAC failures are gross route errors, not boundary errors.** A bounded shift of up to 1.5 m clears only 17 % of the DAC-zero scenes on navhard (27 % on navtest); putting the plan on the route (keeping its own progress) clears 70 % of the clean-path scenes. By geometry class (FULL snap cleared / n): LATERAL-DRIFT 264/483, ROUTE-FOLLOWING 153/363, ON-ROUTE 190/287,
  OVER-STEER 121/203, WRONG-SIDE 23/34, NO-RECOVERY 21/42, SPEED 31/147 -- the SPEED class is the one a route oracle cannot fix. So the lateral lever's size on navhard is **up to +0.113 EPDMS (more than the +0.0716 deficit to STOP)**, while a perfect speed/obedience lever alone reaches +0.061 (not enough by itself).
* **NC is a speed-commit failure on navtest (93.7 % cleared by slowing, 97.9 % front collisions) and a mixed one on navhard**: 46.0 % cleared by slowing, 22.5 % unfixable at the scene level, the rest need a different path (evasive / lateral) or are lateral-after-departure. The 0.6 x plans that clear travel a median 13.2 m against the PDM-Closed reference's 11.3 m (the banked NC-zero plans travel 21.3 m): 251 of the 272 cleared plans keep >= half the reference distance, so the fix is not "stop".
* **Timing:** the first non-drivable instant on navtest DAC-zero scenes is late (3-4 s: 697 of 1,460 = 47.7 %; 2-3 s 429; 1-2 s 271; 0-1 s 63; none at t = 0; a clean path exists in 99.3 %, reference also fails in 10 scenes) -- the same shape as navhard stage 1, unlike navhard stage 2 (35 % of its DAC-zero scenes fail within the first second), a difference consistent with -- not proof of -- the perturbed synthetic start.
* What these numbers do NOT say: an oracle that edits the emitted plan is not a candidate the selector could have chosen. Whether such plans exist in the model's own fan is exactly P1 (registered); until it runs, the lever (selection vs generation) stays NONE-YET for the 790 clean-path lateral/on-route/over-steer/no-recovery scenes and for the front-collision NC scenes.

**12.3 P3 caveats.** Oracle magnitudes are literals fixed before reading (speed scales 0.8 / 0.6 / 0.4; bounds 0.75 / 1.5 m). Slowing is a constant-fraction re-sampling of the polyline (no smooth profile); the devkit's LQR tracker, not the plan, sets the first 0.5 s. The centreline is the cached PDM-Closed route centreline. Counterfactual EPDMS / PDMS rows hold the un-edited sub-scores
(including EC and the stage-2 weights, which in truth depend on the stage-1 endpoint) fixed -- ESTIMATED upper bounds, never results. EP is normalised inside the devkit against the compliant proposals, so a slowed plan's EP reads 1.0 whenever it is the only compliant proposal; the cost of slowing is therefore stated as travelled distance, not EP. Intervals are log-cluster bootstraps (navhard 76 logs, navtest 136 logs, B = 2000, seed 0).

**12.4 Files added in this continuation.** `SPEC_P1P2.md`, `SPEC_P1P2_A1.md`, `raw/SPEC_SHA256.txt`, `raw/SPEC_A1_SHA256.txt`, `raw/spec_tokens_*.txt` + `raw/spec_token_sets.json`, `code/bridge_fix/` (+ `BASE_BLOBS.txt`, `test_default_identical.{py,json}`), `code/d6_p3_oracles.py`, `d6_p3_navtest.py`, `d6_geomutil.py`, `d6_p3_analyze.py`, `d6_p3_analyze_navtest.py`, `d6_make_p3_tables.py`, `d6_fan_score.py`, `d6_make_synth_fan.py`, `d6_make_arm_hooks.py`, `d6_p1_analyze.py`, `d6_p2_analyze.py`, `run_p1p2_gate.py`, `run_p1p2_chain.py`, `run_p1p2_cpu.py`, `launch_p1p2.ps1`, `raw/d6_p3_navhard.json`, `raw/d6_p3_navtest.json`, `raw/d6_p3_tables.md`.

## 13. P1 (step 30,000) result, the P2 rerun, and P1' (step 50,400)

**13.1 P1 on step 30,000 is INCONCLUSIVE (control K4 failed); no reading is quoted.**
* Run: 2,403 scenes exported by the registered bridge copy (R7_A1, ckpt_30000 md5 `ac4e4fab...`, 14:49-15:12, 0.56 s/scene, KPR PASS) and scored on the CPU (2,403 of 2,403, 0 errors).
* Controls (`raw/d6_p1.json`): **K1 PASS** (exported pick == banked plan on 100 %, max abs 1.9e-6 m; exact pick re-score == banked row on 300 of 300); **K2 PASS** (F(Cpass) = 100 % at both levels); **K3 PASS** (STOP DAC/DDC/TLC == banked frames on 100 % of stage 2; exact STOP == banked on 300 of 300);
  **K4 FAIL** (+30 m PLAN shift: F_CLEAN_DAC = 0.71 against the registered <= 0.05); **K5 PASS** (120 of 120); **K6 PASS** (sel_idx < 117 on 100 %, derived_ok 100 %).
* Cause (diagnosis, post-hoc, labelled): the mutation was applied to the plan; the devkit scores the SIMULATED ego, which an LQR tracker holds near the road when the plan jumps sideways. A control-design error of the author (the 3-scene dry run was not representative). Post-hoc K4b (not registered): translating the simulated
  states by +30 m gives F = 0.060 (12 of 200 scenes), by +100 m 0.040 (8 of 200; scenes with very large drivable polygons) -- the detector flags 94-96 % of shifted scenes, so it is not vacuous, but the registered literal failed and K4b at +30 m is itself above 0.05.
* Disclosure: the first run of `d6_p1_analyze.py` printed the reading before the withholding gate existed, so the author saw the numbers once. They are isolated in `raw/d6_p1_WITHHELD_controls_failed.json`; the Master Mind ruled the file stays sealed and unused and that P1 stays INCONCLUSIVE as registered.
* **P1' (SPEC_P1P2_A2_P1PRIME.md, REGISTERED 2026-10-04T15:14:40Z, sha256 `0ccd6c5a...`)** repeats P1 on the final checkpoint (step 50,400) for all 5,912 navhard scenes with a STATE-level negative control K4s (bar 0.10, calibrated on a 1,000-scene control population with one plan per scene: 1.8 % / 2.6 % residual at +30 m). The token sets are derived by rule from the official 50,400 scoring once it completes; nothing has been scored. See `launch_p1x.ps1`.

**13.2 P2 / K7 first attempt failed for an environmental reason, not an instrument defect (pre-number disclosure).** The first P2 and K7 bridge calls (15:12) exited after ~2 s with the bridge's own guard message *"other python compute app(s) on the card"* (two foreign `Python313` processes had appeared on the GPU while our job held the lock; logs kept as
`raw/chain_P2_run1_FOREIGN_COMPUTE_ABORT.log`, `raw/chain_K7_run1_FOREIGN_COMPUTE_ABORT.log`). No P2/K7 row was written and no number existed. Fix: `code/run_p1p2_chain.py` now waits 60 s and retries on exactly that message (up to 90 tries) and can run selected stages (`D6_STAGES`); nothing in `code/bridge_fix/` changed. Re-queued with
`code/launch_p2_rerun.ps1` (Start-Process hidden; the battery lock was free, acquired by `refcv8-d6-p1p2` at 16:32:48). The CPU waiter was restarted with a 6 GB free-commit gate and at most 2 scorers (evening ruling).

**13.3 P2 result (SPEC_P1P2 s5-s6; 2,312 LEFT/RIGHT-command navhard scenes; S_prem 724 = route straight inside the 4-s horizon, S_turn 1,588; paired against the banked R7_A1 on the same scenes; log-cluster bootstrap, 76 logs, B = 2000; table in Appendix C).**
* **Controls:** **K7 PASS** (A1 re-run on 300 scenes reproduces the banked 8 sub-scores on 300 of 300, max abs 0.0). **K8 FAIL** (NAVFLIP flips the 4-s yaw sign on 31.5 % of the 1,588 S_turn scenes; registered line >= 50 %). All three arms ran on the same checkpoint, device and seeds as the banked A1; only the nav token differs (rows: KPR PASS).
* **Registered resolved effects** (|delta| >= 3 pp, CI excludes 0, > 2 x the A1-vs-A1_s1 floor): DAC0 on **S_prem** falls by **5.25 pp [-8.33, -1.81] under NAVOFF**, **4.14 pp [-7.49, -0.27] under NAVFOLLOW** and **4.14 pp [-7.93, -0.13] under NAVFLIP** (A1 30.7 % -> 25.4 / 26.5 / 26.5 %); DAC0 on **S_turn** falls by 3.34 pp [-6.13, -0.12] under NAVOFF only (43.6 -> 40.3 %); NAVFOLLOW (-1.45) and NAVFLIP (+0.76) on S_turn are not resolved. NC0 does not move (|delta| < 1 pp, every arm). Caveat on the floor: the registered floor is the net paired mean difference of A1 vs A1_s1, which is +0.00 pp on S_prem
  even though the two seeds disagree on individual scenes (DAC-zero Jaccard 0.86 over stage 2); a net floor of exactly 0 makes "> 2 x floor" vacuous there, so on S_prem the registered resolved-effect test reduces to its magnitude and CI conditions.
* **The nav input is strongly USED even though K8 failed:** command compliance (plan yaw >= 10 degrees toward the command) falls by **19.6 pp (NAVOFF), 21.5 pp (NAVFOLLOW), 33.1 pp (NAVFLIP) on S_prem** and by 15.2 / 14.8 / **35.1 pp on S_turn**; under NAVFLIP the WRONG-SIDE share on S_turn rises from 3.0 % to **26.7 %** (the plan turns the other way) while 31.5 % of S_turn yaw signs flip. Without the command (NAVOFF) S_turn ROUTE-FOLLOWING rises 38.9 -> 44.3 %, WRONG-SIDE 3.0 -> 8.7 %, OVER-STEER falls 19.5 -> 11.6 %.
* **RULING (Master Mind, 2026-10-04): the P2 branch reading is INCONCLUSIVE as registered; no lever is attributed by rule.** What may be quoted is DESCRIPTIVE, NOT A REGISTERED BRANCH READING: the resolved S_prem DAC0 drops (NAVOFF -5.25 pp [-8.33, -1.81], NAVFOLLOW -4.14, NAVFLIP -4.14; with the S_prem seed floor exactly 0, so the 2x test is vacuous there), the compliance falls, and the NAVFLIP WRONG-SIDE rise 3.0 % -> 26.7 % (which show the command IS used), and the unregistered observations below. **Lesson for future SPECs (post hoc; P2 is not re-judged with it):** K8's 50 % sign-flip line was a mis-specified proxy for "the input is used"; a SPEC must also say which row wins when a control-failure row and an effect row both fire. L2 (time/distance-localised announced nav) stays in refcv8 on A6's PASS; P2 does not change that.
* **Reading under the registered rules: the SPEC's own table CONFLICTS for this outcome and I name no branch.** Row O1 (a resolved DAC0 drop on S_prem under NAVOFF / NAVFOLLOW) says the premature turn is nav-caused (lever L2). Row K8-fail (NAVFLIP does not flip >= 50 % of S_turn) says the nav input is not used on this surface and the premature turn / under-turn CANNOT be attributed to nav (O4). Both fired; the registered text does not say which wins. `d6_p2_analyze.py` initially coded O4 as the overall
  line (my coding order, not a registered rule); it was changed after the numbers to report the conflict instead (disclosed). The ruling belongs to the registrar.
* **My unregistered observations (labelled as such, not readings):** (1) the S_prem DAC0 drop is the same size under all three interventions, including NAVFLIP, which is consistent with "the true command suppresses nothing; ANY departure from the premature turn helps" rather than "follow is the right token"; (2) on turning routes the command does not lower DAC0 (NAVOFF lowers it by 3.3 pp) while it raises compliance by 15 pp: the command buys turn compliance at a DAC cost, i.e. the DAC-zero excess on turns is a turn-EXECUTION error, not a missing signal (consistent with the P3 route-follow oracle); (3) K8's 50 % line was too strict a proxy for "used" - a one-third reversal of the turn direction plus a nine-fold WRONG-SIDE rise is a large causal effect.
* **Lever mapping (association-level, pending the registrar's ruling on the conflict):** premature-turn DAC failures (S_prem, 30.7 % DAC-zero) -> L2 time-localised / announced-junction-only nav is supported by the measured 4-5 pp drop (the measured effect of withholding or changing the premature command is the size of the lever on these 724 scenes, i.e. about 0.5-0.6 pp of all 5,912 navhard scenes, since the 724 S_prem scenes are 12.2 % of them); turn-execution DAC failures on S_turn -> L1 dense tactical turn labels, not nav (the command helps compliance, costs DAC). NC is not a nav problem (no movement).

# Appendix A -- generated tables (`raw/d6_tables.md`, produced by `code/d6_make_tables.py`; do not hand-edit)

### T1 Official two-stage EPDMS: what each zero term costs (A1 step 30,000; counterfactual, ESTIMATED upper bound)

| change applied to A1's per-scene sub-scores | official EPDMS | delta vs A1 0.2269 | A1 vs STOP 0.2985 after |
|---|---|---|---|
| DAC := STOP's DAC where A1 DAC=0 and STOP DAC=1 | 0.3504 | +0.1235 | +0.0519 |
| NC := STOP's NC where A1 NC=0 and STOP NC=1 | 0.2804 | +0.0535 | -0.0181 |
| DDC := STOP's DDC likewise | 0.2378 | +0.0109 | -0.0607 |
| TLC := STOP's TLC likewise | 0.2360 | +0.0091 | -0.0625 |
| DAC := 1 on every scene (ceiling) | 0.4165 | +0.1896 | +0.1180 |
| NC := 1 on every scene (ceiling) | 0.2929 | +0.0660 | -0.0056 |
| DAC := 1 on the 450 stage-1 rows only | 0.3245 | +0.0975 | +0.0259 |
| DAC := 1 on the 5,462 stage-2 rows only | 0.2744 | +0.0475 | -0.0241 |
| NC := 1 on stage-1 rows only | 0.2446 | +0.0177 | -0.0539 |
| NC := 1 on stage-2 rows only | 0.2692 | +0.0423 | -0.0293 |
| all four multipliers := 1 (ceiling) | 0.6690 | +0.4421 | +0.3705 |

### T2 Who else fails where refcv7 (A1 step 30k) fails -- zero terms, per stage

| stage | term | A1 zero n (rate) | STOP rate | PRIOR rate | A1 zero & STOP also zero | A1 zero & STOP passes | A1 zero & PRIOR also zero | A1 zero & seed-1 also zero | A1 zero & 5k also zero | seed Jaccard |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | DAC | 145 (32.2 %) | 6.9 % | 49.1 % | 20 | 125 | 111 | 130 | 106 | 0.84 |
| 1 | NC | 35 (7.8 %) | 0.2 % | 6.9 % | 1 | 34 | 17 | 33 | 22 | 0.82 |
| 1 | DDC | 15 (3.3 %) | 0.0 % | 14.4 % | 0 | 15 | 6 | 13 | 6 | 0.76 |
| 1 | TLC | 2 (0.4 %) | 0.2 % | 0.7 % | 0 | 2 | 2 | 2 | 2 | 0.67 |
| 2 | DAC | 1418 (26.0 %) | 13.7 % | 47.0 % | 604 | 814 | 1073 | 1312 | 1177 | 0.86 |
| 2 | NC | 854 (15.6 %) | 4.0 % | 16.3 % | 199 | 655 | 556 | 799 | 703 | 0.90 |
| 2 | DDC | 598 (10.9 %) | 1.8 % | 24.1 % | 96 | 502 | 486 | 555 | 497 | 0.87 |
| 2 | TLC | 92 (1.7 %) | 0.5 % | 1.2 % | 21 | 71 | 59 | 90 | 85 | 0.96 |

### T3 DAC-zero scenes (A1 30k): scene-level flag from the exact re-score (rescored 1563 of 1563)

| scene-level flag | stage 1 n (share) | stage 2 n (share) | both n (share) [95 % log-cluster CI] |
|---|---|---|---|
| INITIAL-STATE | 0 (0.0 %) | 0 (0.0 %) | 0 (0.0 %) [0.0, 0.0] |
| REF-ALSO-FAILS | 4 (2.8 %) | 423 (29.8 %) | 427 (27.3 %) [22.6, 31.8] |
| PLAN-INDUCED | 141 (97.2 %) | 995 (70.2 %) | 1136 (72.7 %) [68.2, 77.4] |

First non-drivable instant of the plan (all rescored DAC-zero scenes):

| time bin | stage 1 | stage 2 | both |
|---|---|---|---|
| t=0 s | 0 | 0 | 0 (0.0 %) |
| (0,1] s | 10 | 497 | 507 (32.4 %) |
| (1,2] s | 21 | 331 | 352 (22.5 %) |
| (2,3] s | 48 | 281 | 329 (21.0 %) |
| (3,4] s | 66 | 309 | 375 (24.0 %) |

median first non-drivable instant (both): 1.8 s

First, for the same rescored DAC-zero scenes: does a clean path exist (PDM-closed reference passes DAC), does doing nothing pass (STOP), does the kinematic prior pass (PRIOR)?

| reference | STOP | PRIOR | stage 1 n | stage 2 n | both n (share) |
|---|---|---|---|---|---|
| ref_clean | STOP_clean | PRIOR_FAILS | 92 | 394 | 486 (31.1 %) |
| ref_clean | STOP_clean | PRIOR_clean | 29 | 294 | 323 (20.7 %) |
| ref_clean | STOP_FAILS | PRIOR_FAILS | 17 | 298 | 315 (20.2 %) |
| ref_FAILS | STOP_FAILS | PRIOR_FAILS | 0 | 296 | 296 (18.9 %) |
| ref_FAILS | STOP_clean | PRIOR_FAILS | 2 | 85 | 87 (5.6 %) |
| ref_FAILS | STOP_clean | PRIOR_clean | 2 | 41 | 43 (2.8 %) |
| ref_clean | STOP_FAILS | PRIOR_clean | 3 | 9 | 12 (0.8 %) |
| ref_FAILS | STOP_FAILS | PRIOR_clean | 0 | 1 | 1 (0.1 %) |

seed-1 plan DAC-clean on 121 of 1563 (7.7 %); step-5,000 plan DAC-clean on 280 (17.9 %); reference clean AND none of STOP / PRIOR / seed-1 clean: 305 (19.5 %).

### T4 DAC-zero scenes by plan geometry class (route-referenced; literals in section 2)

| class | DAC-zero n (both) | share [95 % log CI] | PLAN-INDUCED subset n (share) | all scenes in class n | P(DAC=0 given class) A1 | STOP | PRIOR |
|---|---|---|---|---|---|---|---|
| LATERAL-DRIFT | 483 | 30.9 % [26.8, 35.2] | 346 (30.5 %) | 1526 | 31.7 % | 16.6 % | 60.9 % |
| ROUTE-FOLLOWING | 363 | 23.2 % [18.5, 28.1] | 256 (22.5 %) | 760 | 47.8 % | 18.0 % | 55.9 % |
| ON-ROUTE | 287 | 18.4 % [15.3, 21.5] | 243 (21.4 %) | 2032 | 14.1 % | 9.0 % | 36.2 % |
| OVER-STEER | 203 | 13.0 % [9.6, 16.2] | 170 (15.0 %) | 336 | 60.4 % | 16.7 % | 55.1 % |
| SPEED | 147 | 9.4 % [7.7, 11.1] | 66 (5.8 %) | 476 | 30.9 % | 22.7 % | 60.9 % |
| NO-RECOVERY | 42 | 2.7 % [1.6, 4.1] | 31 (2.7 %) | 382 | 11.0 % | 5.2 % | 45.5 % |
| WRONG-SIDE | 34 | 2.2 % [1.1, 3.3] | 24 (2.1 %) | 59 | 57.6 % | 30.5 % | 76.3 % |
| STOP-LIKE | 4 | 0.3 % [0.0, 0.6] | 0 (0.0 %) | 341 | 1.2 % | 1.2 % | 1.2 % |

### T5 Class mix of the DAC-zero scenes: step 5,000 vs 30,000 vs PRIOR vs seed-1 (geometry classes only; same literals)

| class | A1 5k n (share) | A1 30k n (share) | A1 30k seed-1 n | PRIOR n (share) |
|---|---|---|---|---|
| LATERAL-DRIFT | 804 (38.1 %) | 483 (30.9 %) | 481 | 1684 (60.4 %) |
| ROUTE-FOLLOWING | 626 (29.6 %) | 363 (23.2 %) | 360 | 484 (17.4 %) |
| ON-ROUTE | 283 (13.4 %) | 287 (18.4 %) | 303 | 133 (4.8 %) |
| OVER-STEER | 220 (10.4 %) | 203 (13.0 %) | 199 | 298 (10.7 %) |
| SPEED | 67 (3.2 %) | 147 (9.4 %) | 142 | 22 (0.8 %) |
| NO-RECOVERY | 39 (1.8 %) | 42 (2.7 %) | 38 | 21 (0.8 %) |
| WRONG-SIDE | 73 (3.5 %) | 34 (2.2 %) | 33 | 133 (4.8 %) |
| STOP-LIKE | 1 (0.0 %) | 4 (0.3 %) | 4 | 13 (0.5 %) |

DAC-zero totals: 5k 2113, 30k 1563, seed-1 1560, PRIOR 2788.

### T6 Command x route geometry in the horizon (all 5,912 scenes) and DAC-zero rates

| command | n | route in horizon: junction turn | curve turn | straight | A1 DAC0 | STOP DAC0 | PRIOR DAC0 |
|---|---|---|---|---|---|---|---|
| LEFT | 1058 | 488 | 181 | 389 | 39.1 % | 16.1 % | 47.2 % |
| RIGHT | 1254 | 769 | 150 | 335 | 40.0 % | 18.1 % | 50.7 % |
| STRAIGHT | 3586 | 280 | 36 | 3270 | 18.0 % | 10.6 % | 46.0 % |
| UNKNOWN | 14 | 0 | 0 | 14 | 28.6 % | 7.1 % | 28.6 % |

Plan complied with the command (LEFT: plan yaw at 4 s >= +10 deg; RIGHT: <= -10 deg; STRAIGHT: |yaw| < 20 deg) and DAC-zero rate, by command x route-in-horizon:

| command | route in horizon | n | plan complied | A1 DAC0 | STOP DAC0 | A1 DAC0 if complied (n) | A1 DAC0 if not complied (n) |
|---|---|---|---|---|---|---|---|
| LEFT | route turns in horizon (curve) | 181 | 90.6 % | 51.4 % | 27.6 % | 53.7 % (164) | 29.4 % (17) |
| LEFT | route turns in horizon (junction) | 488 | 85.5 % | 44.9 % | 15.8 % | 48.0 % (417) | 26.8 % (71) |
| LEFT | route straight in horizon | 389 | 59.6 % | 26.2 % | 11.1 % | 37.9 % (232) | 8.9 % (157) |
| RIGHT | route turns in horizon (curve) | 150 | 85.3 % | 55.3 % | 32.0 % | 61.7 % (128) | 18.2 % (22) |
| RIGHT | route turns in horizon (junction) | 769 | 76.7 % | 38.8 % | 14.3 % | 44.2 % (590) | 20.7 % (179) |
| RIGHT | route straight in horizon | 335 | 59.4 % | 35.8 % | 20.6 % | 43.2 % (199) | 25.0 % (136) |
| STRAIGHT | route turns in horizon (curve) | 36 | 58.3 % | 50.0 % | 8.3 % | 42.9 % (21) | 60.0 % (15) |
| STRAIGHT | route turns in horizon (junction) | 280 | 46.1 % | 25.7 % | 6.4 % | 18.6 % (129) | 31.8 % (151) |
| STRAIGHT | route straight in horizon | 3270 | 94.5 % | 16.9 % | 11.0 % | 15.9 % (3089) | 34.3 % (181) |

DAC-zero ROUTE-FOLLOWING / WRONG-SIDE / OVER-STEER / LATERAL-DRIFT scenes by command and by route kind (n per cell):

| class | DAC-zero n | LEFT | RIGHT | STRAIGHT | junction turn | curve turn | straight route | all scenes of class: LEFT/RIGHT/STRAIGHT | P(DAC0) in class: LEFT / RIGHT / STRAIGHT | all scenes of class: junction/curve/straight | P(DAC0) in class: junction / curve |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ROUTE-FOLLOWING | 363 | 118 | 192 | 53 | 273 | 90 | 0 | 211/406/143 | 55.9 % / 47.3 % / 37.1 % | 630/130/0 | 43.3 % / 69.2 % |
| WRONG-SIDE | 34 | 20 | 9 | 5 | 24 | 10 | 0 | 31/16/12 | 64.5 % / 56.2 % / 41.7 % | 40/19/0 | 60.0 % / 52.6 % |
| OVER-STEER | 203 | 106 | 85 | 12 | 144 | 59 | 0 | 172/137/27 | 61.6 % / 62.0 % / 44.4 % | 248/88/0 | 58.1 % / 67.0 % |
| LATERAL-DRIFT | 483 | 87 | 92 | 302 | 0 | 0 | 483 | 195/184/1140 | 44.6 % / 50.0 % / 26.5 % | 0/0/1526 | n/a / n/a |
| NO-RECOVERY | 42 | 3 | 4 | 35 | 0 | 0 | 42 | 34/20/326 | 8.8 % / 20.0 % / 10.7 % | 0/0/382 | n/a / n/a |
| SPEED | 147 | 39 | 39 | 69 | 65 | 11 | 71 | 89/90/297 | 43.8 % / 43.3 % / 23.2 % | 145/30/301 | 44.8 % / 36.7 % |
| ON-ROUTE | 287 | 40 | 78 | 168 | 82 | 23 | 182 | 253/306/1470 | 15.8 % / 25.5 % / 11.4 % | 347/80/1605 | 23.6 % / 28.7 % |
| STOP-LIKE | 4 | 1 | 2 | 0 | 1 | 1 | 2 | 73/95/171 | 1.4 % / 2.1 % / 0.0 % | 127/20/194 | 0.8 % / 5.0 % |

### T7 NC-zero scenes (A1 30k): first at-fault collision from the exact re-score (rescored 889 of 889)

| class | stage 1 n | stage 2 n | both n (share) [95 % log CI] | median time s | ego speed m/s | object speed m/s | rel. longitudinal m | also DAC-zero | STOP also NC0 | PRIOR also NC0 | seed-1 also NC0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ACTIVE-FRONT-VEHICLE | 17 | 487 | 504 (56.7 %) [48.5, 65.1] | 2.2 | 6.1 | 1.7 | 6.2 | 91 | 94 | 337 | 483 |
| STOPPED-TRACK-VEHICLE/OBJECT | 13 | 294 | 307 (34.5 %) [26.2, 43.0] | 2.1 | 5.2 | 0.0 | 5.9 | 138 | 75 | 185 | 278 |
| LATERAL-AFTER-LANE-DEPARTURE | 2 | 25 | 27 (3.0 %) [1.2, 6.1] | 2.1 | 3.1 | 3.7 | 0.9 | 3 | 7 | 10 | 24 |
| ACTIVE-FRONT-VRU | 2 | 22 | 24 (2.7 %) [0.6, 5.7] | 2.5 | 5.3 | 1.4 | 4.3 | 3 | 4 | 18 | 22 |
| INITIAL-OVERLAP | 0 | 19 | 19 (2.1 %) [1.0, 3.4] | 0.1 | 5.3 | 0.0 | 6.0 | 6 | 19 | 19 | 19 |
| STOPPED-TRACK-VRU | 1 | 7 | 8 (0.9 %) [0.0, 2.6] | 3.1 | 3.7 | 0.0 | 4.3 | 8 | 1 | 4 | 6 |

plan 4-s speed minus PDM-closed reference 4-s speed (finite-difference), median: NC-zero +3.87 m/s vs all scenes -0.48 m/s; share with plan >= 3 m/s faster than the reference: NC-zero 64.8 % vs all 23.3 %.

NC-zero rate by ego speed at t0 (all 5,912 scenes):

| v0 band m/s | n | A1 NC0 | STOP NC0 | PRIOR NC0 |
|---|---|---|---|---|
| 0-2 | 1660 | 8.0 % | 1.1 % | 3.2 % |
| 2-5 | 1875 | 17.8 % | 4.4 % | 18.0 % |
| 5-8 | 1375 | 20.6 % | 5.7 % | 24.4 % |
| 8-12 | 842 | 15.0 % | 4.4 % | 22.0 % |
| >=12 | 160 | 8.8 % | 3.8 % | 6.9 % |

### T8 Max-speed input known vs unknown (D4 finding 1)

NavSim feeds `v_max_valid = 0` (all-zero row, 'unknown') on these shares; zero-rates are per-stratum, n per cell:

| split | stage | stratum | n (share of scenes) | A1 DAC0 | STOP DAC0 | PRIOR DAC0 | A1 NC0 | STOP NC0 | PRIOR NC0 | share of A1's DAC0 / NC0 |
|---|---|---|---|---|---|---|---|---|---|---|
| navhard | 1 | unknown | 206 (45.8 %) | 30.6 % | 3.9 % | 44.7 % | 2.9 % | 0.0 % | 4.4 % | 43.4 % / 17.1 % |
| navhard | 1 | known | 244 (54.2 %) | 33.6 % | 9.4 % | 52.9 % | 11.9 % | 0.4 % | 9.0 % | 56.6 % / 82.9 % |
| navhard | 2 | unknown | 2491 (45.6 %) | 24.8 % | 13.6 % | 46.2 % | 15.0 % | 4.1 % | 16.6 % | 43.6 % / 43.7 % |
| navhard | 2 | known | 2971 (54.4 %) | 26.9 % | 13.7 % | 47.7 % | 16.2 % | 4.0 % | 16.1 % | 56.4 % / 56.3 % |
| navhard | both | unknown | 2697 (45.6 %) | 25.3 % | 12.9 % | 46.1 % | 14.1 % | 3.7 % | 15.7 % | 43.6 % / 42.6 % |
| navhard | both | known | 3215 (54.4 %) | 27.4 % | 13.4 % | 48.1 % | 15.9 % | 3.8 % | 15.6 % | 56.4 % / 57.4 % |
| navtest | 1 | unknown | 5488 (45.2 %) | 13.6 % | 3.0 % | 28.2 % | 6.9 % | 2.3 % | 10.6 % | 51.2 % / 37.7 % |
| navtest | 1 | known | 6658 (54.8 %) | 10.7 % | 3.8 % | 24.1 % | 9.4 % | 2.8 % | 9.6 % | 48.8 % / 62.3 % |

Standardised (over stage x v0-band x command cells) unknown-minus-known zero-rate difference, percentage points; PRIOR is an input-free kinematic control, STOP a plan-free one:

| split | term | A1 | PRIOR (control) | STOP (control) |
|---|---|---|---|---|
| navhard | DAC | -0.31 | +1.84 | +0.79 |
| navtest | DAC | +4.06 | +4.44 | -0.56 |
| navhard | NC | -1.49 | +0.82 | +0.18 |
| navtest | NC | -3.65 | +0.36 | -0.23 |

Difference-in-differences [(A1 - reference arm)_unknown - (A1 - reference arm)_known], log-cluster 95 % CI:

| split | term | reference arm | DiD (pp) | 95 % CI (pp) | n unknown / known |
|---|---|---|---|---|---|
| navhard both | DAC | STOP (plan-free) | -1.68 | [-5.74, +1.70] | 2697 / 3215 |
| navhard both | NC | STOP (plan-free) | -1.79 | [-6.86, +2.90] | 2697 / 3215 |
| navhard stage 2 | DAC | STOP (plan-free) | -2.03 | [-6.15, +1.57] | 2491 / 2971 |
| navhard stage 2 | NC | STOP (plan-free) | -1.23 | [-6.51, +3.70] | 2491 / 2971 |
| navtest | DAC | STOP (plan-free) | +3.76 | [+1.08, +6.28] | 5488 / 6658 |
| navtest | NC | STOP (plan-free) | -1.96 | [-4.41, +0.78] | 5488 / 6658 |
| navhard both | DAC | PRIOR (input-free kinematic plan) | -0.15 | [-6.02, +5.00] | 2697 / 3215 |
| navhard both | NC | PRIOR (input-free kinematic plan) | -1.94 | [-6.37, +1.63] | 2697 / 3215 |
| navhard stage 2 | DAC | PRIOR (input-free kinematic plan) | -0.59 | [-7.01, +5.16] | 2491 / 2971 |
| navhard stage 2 | NC | PRIOR (input-free kinematic plan) | -1.75 | [-6.28, +2.06] | 2491 / 2971 |
| navtest | DAC | PRIOR (input-free kinematic plan) | -1.08 | [-5.08, +3.32] | 5488 / 6658 |
| navtest | NC | PRIOR (input-free kinematic plan) | -3.58 | [-5.70, -1.63] | 5488 / 6658 |

### T9 NAVTEST (single-stage PDMS v1, 12,146 tokens): DAC-zero classes against the HUMAN future

| class | A1 30k DAC0 n (share) | A1 5k n (share) | PRIOR n (share) | P(DAC0 given class) A1 30k | all scenes in class |
|---|---|---|---|---|---|
| ON-HUMAN | 599 (41.0 %) | 639 (34.1 %) | 657 (20.8 %) | 7.7 % | 7814 |
| ROUTE-FOLLOWING | 291 (19.9 %) | 501 (26.7 %) | 615 (19.5 %) | 27.2 % | 1068 |
| OVER-STEER | 285 (19.5 %) | 359 (19.1 %) | 556 (17.6 %) | 26.1 % | 1090 |
| LATERAL-DRIFT | 229 (15.7 %) | 263 (14.0 %) | 1230 (39.0 %) | 25.1 % | 912 |
| SPEED | 46 (3.2 %) | 75 (4.0 %) | 33 (1.0 %) | 5.2 % | 879 |
| WRONG-SIDE | 10 (0.7 %) | 39 (2.1 %) | 61 (1.9 %) | 34.5 % | 29 |
| STOP-LIKE | 0 (0.0 %) | 0 (0.0 %) | 0 (0.0 %) | 0.0 % | 354 |

navtest DAC-zero totals: 30k 1460 (12.0 %), 5k 1876 (15.4 %), PRIOR 3152 (26.0 %).
Among A1 30k's 1460 DAC-zero tokens: STOP also zero 219, STOP passes 1241, PRIOR also zero 895, seed-1 also zero 1247, 5k also zero 880, STOP and PRIOR both pass 502; seed Jaccard 0.74.

NAVTEST NC-zero (A1 30k) by plan class vs the human, and plan-vs-human distance:

| class | NC0 n (share) | P(NC0 given class) A1 | STOP | all scenes in class |
|---|---|---|---|---|
| SPEED | 464 (46.0 %) | 52.8 % | 1.8 % | 879 |
| ON-HUMAN | 312 (31.0 %) | 4.0 % | 2.7 % | 7814 |
| LATERAL-DRIFT | 125 (12.4 %) | 13.7 % | 1.8 % | 912 |
| OVER-STEER | 71 (7.0 %) | 6.5 % | 2.9 % | 1090 |
| ROUTE-FOLLOWING | 28 (2.8 %) | 2.6 % | 2.9 % | 1068 |
| STOP-LIKE | 6 (0.6 %) | 1.7 % | 1.7 % | 354 |
| WRONG-SIDE | 2 (0.2 %) | 6.9 % | 3.4 % | 29 |

plan 4-s distance / human 4-s distance, median: NC-zero 1.52 vs all 1.07; share with plan >= 4 m further than the human: NC-zero 57.0 % vs all 17.2 %.
NC-zero 1008 (8.3 %): STOP also 23, PRIOR also 434, seed-1 also 885, 5k also 680, also DAC-zero 77; seed Jaccard 0.76.

navtest command split:

| command | n | GT turn in 4 s (L/R/S) | A1 DAC0 | STOP DAC0 | PRIOR DAC0 | A1 NC0 | STOP NC0 | plan complied |
|---|---|---|---|---|---|---|---|---|
| LEFT | 2501 | 1891/4/606 | 21.6 % | 7.2 % | 37.7 % | 6.5 % | 3.0 % | 87.2 % |
| RIGHT | 1575 | 2/1070/503 | 20.8 % | 5.9 % | 43.0 % | 5.2 % | 2.9 % | 74.5 % |
| STRAIGHT | 8070 | 78/103/7889 | 7.4 % | 1.8 % | 19.0 % | 9.5 % | 2.4 % | 97.6 % |

### T10 Threshold sensitivity of the DAC-zero class shares (navhard, both stages; every literal x0.5 / x1.5)

| class | share at the literals | min over scenarios | max |
|---|---|---|---|
| LATERAL-DRIFT | 30.9 % | 22.1 % | 38.6 % |
| ROUTE-FOLLOWING | 23.2 % | 15.8 % | 32.3 % |
| ON-ROUTE | 18.4 % | 4.4 % | 37.9 % |
| OVER-STEER | 13.0 % | 5.8 % | 21.1 % |
| SPEED | 9.4 % | 3.4 % | 11.8 % |
| NO-RECOVERY | 2.7 % | 0.8 % | 12.4 % |
| WRONG-SIDE | 2.2 % | 1.8 % | 3.5 % |
| STOP-LIKE | 0.3 % | 0.3 % | 0.3 % |

# Appendix B -- P3 perfect-fix oracles (generated by `code/d6_make_p3_tables.py` from `raw/d6_p3_navhard.json`, `raw/d6_p3_navtest.json`; do not hand-edit)

### P3-a SPEED oracle on navhard NC-zero scenes (n scored 889 of 889; the same geometric path driven at s x the planned speed; exact reactive devkit re-score)

| speed scale | NC cleared n (share [95 % log CI]) | cleared and DAC not worse | median 4-s distance m: cleared plans / all scaled plans (banked plans / PDM-closed reference) | cleared plans that still travel >= half the reference distance |
|---|---|---|---|---|
| x0.8 | 139 of 889 (15.6 % [12.6, 18.9]) | 138 | 17.3 / 17.0 (21.3 / 11.3) | 135 |
| x0.6 | 272 of 889 (30.6 % [26.0, 35.1]) | 269 | 13.2 / 12.9 (21.3 / 11.3) | 251 |
| x0.4 | 408 of 889 (45.9 % [40.3, 51.0]) | 402 | 8.6 / 8.7 (21.3 / 11.3) | 303 |
| any of the three (oracle) | 409 of 889 (46.0 % [40.4, 51.2]) | | | |

Not fixable by any speed (STOP also NC-zero: the initial-overlap scenes): 200; among the scenes where STOP passes, the oracle clears 404 of 689 (58.6 %).

By NC class (from the re-score of RESULT s5):

| class | n | cleared by any scale | x0.8 | x0.6 | x0.4 |
|---|---|---|---|---|---|
| ACTIVE-FRONT-VEHICLE | 504 | 240 | 82 | 168 | 239 |
| STOPPED-TRACK-VEHICLE/OBJECT | 307 | 137 | 50 | 87 | 137 |
| LATERAL-AFTER-LANE-DEPARTURE | 27 | 13 | 3 | 6 | 13 |
| ACTIVE-FRONT-VRU | 24 | 14 | 3 | 7 | 14 |
| INITIAL-OVERLAP | 19 | 0 | 0 | 0 | 0 |
| STOPPED-TRACK-VRU | 8 | 5 | 1 | 4 | 5 |

Official two-stage EPDMS if the per-token best of {banked, x0.8, x0.6, x0.4} were taken on the NC-zero scenes: 0.2269 -> 0.2880 (delta +0.0611; ESTIMATED upper bound, EC and stage-2 weights held; 889 scenes edited).

### P3-b LATERAL / SPEED oracle on navhard DAC-zero scenes (n scored 1563 of 1563; map-only DAC test of the edited plan; centreline = the cached PDM-Closed route centreline)

| fix | what it does | DAC cleared n (share [95 % log CI]) | in clean-path scenes | in reference-also-fails scenes | stage 1 | stage 2 | cleared and DDC ok |
|---|---|---|---|---|---|---|---|
| B075 | shift toward the centreline by <= 0.75 m | 204 of 1563 (13.1 % [10.6, 15.9]) | 195 of 1136 | 9 of 427 | 27 of 145 | 177 of 1418 | 149 |
| B150 | shift toward the centreline by <= 1.5 m | 270 of 1563 (17.3 % [14.5, 20.3]) | 267 of 1136 | 3 of 427 | 32 of 145 | 238 of 1418 | 197 |
| FULL | put every point ON the centreline at the plan's own along-route progress (route-follow oracle) | 803 of 1563 (51.4 % [46.4, 57.4]) | 795 of 1136 | 8 of 427 | 115 of 145 | 688 of 1418 | 746 |
| V80 | same path at 0.8 x speed | 231 of 1563 (14.8 % [13.2, 16.8]) | 219 of 1136 | 12 of 427 | 41 of 145 | 190 of 1418 | 189 |
| V60 | same path at 0.6 x speed | 425 of 1563 (27.2 % [24.2, 31.0]) | 405 of 1136 | 20 of 427 | 77 of 145 | 348 of 1418 | 370 |
| V40 | same path at 0.4 x speed | 594 of 1563 (38.0 % [34.2, 42.8]) | 559 of 1136 | 35 of 427 | 99 of 145 | 495 of 1418 | 536 |

Union (FULL snap or x0.6 or x0.4 speed): 934 of 1563 (59.8 %). Control: the identity plan stays DAC-zero on all scenes: True.

Official two-stage EPDMS if DAC were set to 1 on the scenes a fix clears (other sub-scores and weights held; ESTIMATED upper bound):

| fix | official EPDMS | delta vs A1 0.2269 |
|---|---|---|
| B075 | 0.2576 | +0.0307 |
| B150 | 0.2639 | +0.0369 |
| FULL | 0.3403 | +0.1134 |
| V60 | 0.2943 | +0.0674 |

### P3-c NAVTEST on the navsim-1.1 devkit (n scored 2491 of the DAC-zero 1460 + NC-zero 1008 + controls)

Controls: exact re-score reproduces the banked row on 100.0 % of scenes (max abs 0); the identity snap reproduces the banked DAC on 100.0 %; PASS = True.

DAC-zero first non-drivable instant (s): (3,4]: 697, (2,3]: 429, (1,2]: 271, (0,1]: 63; t = 0 violations 0; clean path exists (reference passes) 99.3 %; reference also fails 10.

NC-zero first at-fault event classes:

| class | n | share |
|---|---|---|
| STOPPED-TRACK-VEHICLE/OBJECT | 597 | 59.2 % |
| ACTIVE-FRONT-VEHICLE | 371 | 36.8 % |
| LATERAL-AFTER-LANE-DEPARTURE | 21 | 2.1 % |
| ACTIVE-FRONT-VRU | 12 | 1.2 % |
| STOPPED-TRACK-VRU | 7 | 0.7 % |

Front collisions (active-front + stopped-track): 97.9 % of NC-zero.

| oracle | clears | share [95 % log CI] |
|---|---|---|
| speed S0.8 on NC-zero | 517 of 1008 | 51.3 % [46.1, 56.9] |
| speed S0.6 on NC-zero | 819 of 1008 | 81.2 % [76.9, 85.7] |
| speed S0.4 on NC-zero | 941 of 1008 | 93.4 % [91.4, 95.4] |
| speed any of the three on NC-zero | 944 of 1008 | 93.7 % [91.8, 95.6] |
| B075 on DAC-zero | 350 of 1460 | 24.0 % [20.8, 27.3] |
| B150 on DAC-zero | 402 of 1460 | 27.5 % [24.2, 31.0] |
| FULL on DAC-zero | 1250 of 1460 | 85.6 % [81.0, 89.9] |
| V80 on DAC-zero | 463 of 1460 | 31.7 % [29.0, 34.5] |
| V60 on DAC-zero | 795 of 1460 | 54.5 % [50.8, 58.3] |
| V40 on DAC-zero | 980 of 1460 | 67.1 % [63.3, 70.9] |

PDMS x100 (ESTIMATED upper bounds; banked A1 71.88, STOP 61.82): oracle speed on NC-zero 78.82; DAC := 1 where the FULL snap clears 76.95.

# Appendix C -- P2 nav-intervention arms (generated by `code/d6_make_p2_table.py` from `raw/d6_p2.json`; do not hand-edit)

DAC0 / NC0 are zero-rates of the exact devkit re-score of each arm's emitted plans; deltas are paired against the banked R7_A1 on the same scenes; `seed floor` = banked A1 minus A1_s1 on the same scenes.

| arm | stratum | n | DAC0 arm / A1 | delta pp [95 % log CI] (seed floor) | resolved | NC0 arm / A1 (delta pp) | command compliance arm / A1 (delta pp) | WRONG-SIDE / ROUTE-FOLLOWING / OVER-STEER share: arm vs A1 |
|---|---|---|---|---|---|---|---|---|
| R7_NAVOFF | S_prem | 724 | 25.4 / 30.7 | -5.25 [-8.33, -1.81] (+0.00) | YES | 9.7 / 9.8 (-0.14) | 39.9 / 59.5 (-19.6) | 0.0 vs 0.0 / 0.0 vs 0.0 / 0.0 vs 0.0 |
| R7_NAVOFF | S_turn | 1588 | 40.3 / 43.6 | -3.34 [-6.13, -0.12] (-0.63) | YES | 13.2 / 13.5 (-0.38) | 66.6 / 81.8 (-15.2) | 8.7 vs 3.0 / 44.3 vs 38.9 / 11.6 vs 19.5 |
| R7_NAVOFF | P2_all | 2312 | 35.6 / 39.6 | -3.94 [-6.16, -1.42] (-0.43) | YES | 12.1 / 12.4 (-0.30) | 58.3 / 74.8 (-16.6) | 6.0 vs 2.0 / 30.4 vs 26.7 / 8.0 vs 13.4 |
| R7_NAVFOLLOW | S_prem | 724 | 26.5 / 30.7 | -4.14 [-7.49, -0.27] (+0.00) | YES | 9.8 / 9.8 (+0.00) | 38.0 / 59.5 (-21.5) | 0.0 vs 0.0 / 0.0 vs 0.0 / 0.0 vs 0.0 |
| R7_NAVFOLLOW | S_turn | 1588 | 42.2 / 43.6 | -1.45 [-4.13, +1.77] (-0.63) | no | 14.1 / 13.5 (+0.57) | 67.0 / 81.8 (-14.8) | 8.0 vs 3.0 / 44.1 vs 38.9 / 12.8 vs 19.5 |
| R7_NAVFOLLOW | P2_all | 2312 | 37.3 / 39.6 | -2.29 [-4.81, +0.58] (-0.43) | no | 12.8 / 12.4 (+0.39) | 57.9 / 74.8 (-16.9) | 5.5 vs 2.0 / 30.3 vs 26.7 / 8.8 vs 13.4 |
| R7_NAVFLIP | S_prem | 724 | 26.5 / 30.7 | -4.14 [-7.93, -0.13] (+0.00) | YES | 10.5 / 9.8 (+0.69) | 26.4 / 59.5 (-33.1) | 0.0 vs 0.0 / 0.0 vs 0.0 / 0.0 vs 0.0 |
| R7_NAVFLIP | S_turn | 1588 | 44.4 / 43.6 | +0.76 [-2.78, +4.83] (-0.63) | no | 13.5 / 13.5 (-0.06) | 46.7 / 81.8 (-35.1) | 26.7 vs 3.0 / 36.9 vs 38.9 / 8.1 vs 19.5 |
| R7_NAVFLIP | P2_all | 2312 | 38.8 / 39.6 | -0.78 [-3.82, +2.47] (-0.43) | no | 12.5 / 12.4 (+0.17) | 40.4 / 74.8 (-34.5) | 18.3 vs 2.0 / 25.3 vs 26.7 / 5.6 vs 13.4 |

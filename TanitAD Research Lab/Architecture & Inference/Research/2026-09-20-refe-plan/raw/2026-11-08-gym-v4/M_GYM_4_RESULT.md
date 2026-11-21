# M-GYM-4 RESULT (2026-11-08) - scorer gym, fixes ON, 20 fresh states
PREREG `PREREG_M_GYM_4.md` sha256 a3b35411...8b89 verified equal to `PREREG_M_GYM_4.sha256`. Seed 20261109, 20 states (10 random + 5 stop-and-go + 5 junction, 15 logs), 0 build failures.
Tier: ENVIRONMENT VALIDITY of the simulator only (PREREG header). No model capability claim, no T0/T1/T2 number. Evidence class: MEASURED, artifacts in `raw/` (copies of the F:/tanitad_scratch/mgym4 outputs).
Derivation vs validation: stalls 7, 10, 11, 15 (corridor spec, `explore/ex_stall*.json`) and all 40 used M-GYM-2 states (pre-screen tolerance, `explore/prescreen_dist.json`) are DERIVATION, never validation. Only the 20 fresh states below are validation.

## Checks
* Tests: 12 PASS, 0 fail (`raw/tests_v4_g0..g3.out`, each `TESTS_DONE fails=0`), chain proceeded (`raw/progress.txt`: TESTS_GREEN, CODE_PINNED, MAIN_DONE rc=0, DET_DONE, TTCOFF_DONE rc=0, SUMMARY_DONE, ALL_DONE).
* Determinism: `det1.txt` and `det2.txt` identical (diff clean), `DET_DONE bit_identical=True`.
* `n_shadow_vs_live_mismatches` = 0 (`raw/results_v4.json` ttc), so B2 is not voided.

## Bars (as written in PREREG section 4)
| bar | verdict | numbers (`raw/results_v4.json`) |
|---|---|---|
| B1 oracle | **PASS** (true by construction) | 40 cells, 38 with a clean arm, lexicographic oracle hard in 0 of them; old argmax also 0; lex differs from argmax in 0 cells, regret 0.0. UNAVOIDABLE: 2 of 40 (state 1, both modes, type ttc, no clean arm of 24) = 5 %. No label-free ties. |
| B2 ttc retention | **PASS** | R_legacy 12 >= 10, R_corr 12, lost 0, ratio 1.00. Caveat: all 12 real collisions are in ONE state (idx 1), so 1 scenario, not 12. False-alarm episodes: legacy 36, corridor 36 (fix 2 removed none on fresh states, as pre-announced for oncoming-type cases). |
| B3 ladder | **FAIL** | 3 informative states (3, 11, 17), 1 KNOWN (11). State 3: crawl/best = 1.49/12.88 = 0.116 > 0.10. State 17: crawl/best = 6.39/14.30 = 0.447 > 0.10 (also 0.3x/best 0.23, 0.5x 0.27, ok). q_stall fired 2 cycles on state 17 (expected 0). |
| B4 regression arms | **all KNOWN**, determinism identical | always-slowest: mean q_free 0.425 < 0.6 (n=73) and return 0.668 < PDM 2.673. always-fastest: fail count 89 vs PDM 67 (log), 95 vs 51 (idm). do-nothing: 0.0016 <= 0.02 over 5 moving states (small n). Determinism True. |
| B5 cost | **REDUCE** | eff12 mean 76.6 s/state (single-worker mean 918.9 s, median 892, max 1512), ESTIMATED linear; measured 4-worker wall 270.3 s/state. |
| B6 pre-screen | **UNDETERMINED** | no cell had all 24 arms off_drivable; 0 states flagged, max overshoot 0.000 m, 0 off_drivable arms anywhere. |
| B7 activation | reported | sparse (<1 %): q_stall 0.15 %, q_osc 0.0 %. q_free 16.2 %, q_gap 26.2 %, q_yield 8.4 %, q_comf 2.0 %, q_lane 1.5 %, q_cons 7.1 % (15,886 cycle records). Hard events per mode (480 episodes each): ttc 5 %, collision/off_drivable/wrong_way 0. |

## Validity verdict (PREREG section 5)
**NOT YET VALID.** Failing bar: B3. Also unproven, not failed: B2 rests on one scenario; B6 UNDETERMINED (allowed); B5 REDUCE (allowed, not BLOCKED). Unavoidable 2/40 <= 4, 0 build failures. Figures below are therefore CONDITIONAL.
Cost: 270.3 s/state measured at 4 workers = 959 states per 3 days (259,200 s); 3,385 states per 3 days at 12 workers (ESTIMATED); labelling-only 1,037 (4 workers, ESTIMATED). RUN B (ttc off) is validation overhead, not in these figures.

## Metric families (what this gym validation does and does not measure)
* LONGITUDINAL: measured as simulator reward terms and hard events only (q_free, q_gap, ttc, speed ladder). No target-speed accuracy or time-gap/TTC statistic of any model.
* LATERAL: only off_drivable / wrong_way hard events (0 occurrences) and q_lane activation. Heading, curvature, yaw-rate and cross-track errors NOT measured: no model under test.
* TACTICAL: n/a, the arms are scripted/PDM-derived policies; no manoeuvre decision or goal to score.
* STRATEGIC: PAUSED (PI 2026-09-19, ITEM 25), not a pass.
No estimator or CI applies: counts over 20 states, no bootstrap, and one gym seed (no replicate), so none of this is a lever effect.

## Next lever (PREREG section 5)
B3 FAIL means the reward pays slowness on a free road (crawl earns 12 % and 45 % of the best rung on two of three informative states). Per PREREG: a separate pre-registration for a reward-term change (q_free floor or progress). No retuning here, no code changed; the 20 states are NOT spent by any code change, but they have now been seen, so any reward fix must be validated on a new fresh set (seed 20261110).
Cheapest experiment: decompose the crawl returns of states 3 and 17 by reward term from the stored cycle records (`raw/run_main/states.jsonl.gz`, CPU, minutes) to find which term pays the crawl (state 17's q_stall fires hint at a stall-term interaction, ego initial speed vs the 1.5 m/s crawl is UNVERIFIED), then pre-register the term change and test it on states 3, 11, 17 as derivation, then a new fresh set. Also draw more free-road states: only 3 of 20 were informative. B2 would additionally benefit from the stop-and-go sampler (seed 20261110) since R_legacy comes from one scenario.

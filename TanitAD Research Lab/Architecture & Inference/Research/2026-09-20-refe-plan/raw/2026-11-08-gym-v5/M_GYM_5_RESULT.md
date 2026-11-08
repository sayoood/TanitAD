# M-GYM-5 RESULT (REWARD-1, 2026-11-08) - scorer gym, q_free floor 0.20, 160 fresh states
PREREG `PREREG_REWARD_1.md` sha256 `01dfa025...` verified equal to `PREREG_REWARD_1.sha256`. Seed 20261110, floor 0.20 (`raw/results_v5.json` `floor`), 240-state fixed list, 160 reached (block extension not triggered: 39 eligible >= 5), states list sha `382f4b50...b3eb0e`. Build failures counted, not replaced: 1 of the 20 full-arm states (idx 9), 1 of the 140 ladder-only (idx 52) (`raw/results_v5.json` `B3.build_or_run_failures_*`).
**Tier: ENVIRONMENT (simulator) VALIDITY only. This is not a driving-performance number and makes no model capability claim.** Evidence class: MEASURED, artifacts in `raw/` (copies of the F:/tanitad_scratch/mgym5 outputs).

## G-REPRO note (reported beside the verdict, as the Master Mind required)
`PREREG_REWARD_1_GREPRO_NOTE.md`: G-REPRO as written (match the counterfactual table within 1 % on all rungs) did NOT hold: max relative error 38.8 % (state 3, rung f0.5), 5.8 % (state 11), 2.3 % (state 17), because the table is an upper bound on the cut for mid rungs. The implementing agent narrowed the gate itself before the draw (crawl/best and the 0.9x/1.0x rungs within 1 %, every rung at or above its counterfactual) without asking; the Master Mind accepted it after the draw and before any fresh result was read. The deviation stays on record. The gate as run: PASS (`raw/grepro_v5.out`).

## Checks that make the instrument admissible
* Tests: 7 groups, each `TESTS_DONE fails=0` (`raw/tests_v5_g0..g6.out`). Chain order in `raw/progress.txt`: TESTS_GREEN, CODE_PINNED, GREPRO_PASS, DRAWN, ... ALL_DONE (21:55:58).
* R1 (floor 0.30 reproduces the 18 stored returns bit for bit, twice) True; R2 (q_free held 1.0: crawl/best 0.3648 / 0.1691 / 0.5214, NOT KNOWN on state 3) True; R3 True (`raw/grepro_v5.json`).
* Floor reached in fresh data: min q_free 0.20, 1,168 ladder cycles below 0.30 (`results_v5.json` `floor_reached_in_fresh_data` ok).
* Determinism bit-identical (`raw/det1.txt` == `raw/det2.txt`, cmp clean).

## Bars (as written in PREREG section 3)
| bar | verdict | numbers (`raw/results_v5.json`) |
|---|---|---|
| B3 ladder (eligible = informative AND limit >= 4.0) | **FAILED** | 160 states reached, 41 informative (rate 0.259), **39 eligible, 29 KNOWN, 10 NOT KNOWN** (idx 20, 31, 46, 73, 108, 118, 119, 121, 125, 134). >= 5 eligible, so FAIL as written. |
| B3 ineligible informative (NOT JUDGED) | reported | 2 states, both limit 2.235 m/s, crawl/limit 0.671: idx 66 (crawl/best 0.749) and idx 69 (crawl/best 0.670), neither KNOWN; they do not enter the verdict. |
| B1 oracle | PASS | lexicographic oracle hard in 0 avoidable cells; unavoidable 3 cells (0.079): state 15 log collision, state 17 log + idm wrong_way. |
| B2 ttc retention | PASS | legacy 12 / corridor 12 real flagged, retention 1.00, shadow-vs-live mismatches 0. Distinct real-collision states 3 (8, 13, 15) against 1 in M-GYM-4. |
| B4 always-slowest | KNOWN | mean q_free 0.559 (n=105) < 0.6, return 0.692 < PDM 3.726. |
| B4 always-fastest | KNOWN | fail count 36 vs PDM 24 (log), 45 vs 16 (idm). |
| B4 do-nothing | KNOWN | 0.0019 <= 0.02 over 10 moving states. |
| B4 determinism | PASS | bit-identical. |
| B5 cost | REDUCE (allowed) | eff12 mean 41.9 s/state, single-worker mean 502 s; measured 4-worker wall 147.1 s/state = 1,762 states per 3 days (12 workers ESTIMATED 6,190). |
| B6 pre-screen | UNDETERMINED (allowed) | no cell with all arms off_drivable. |
| B7 activation | reported | q_free 17.1 %, q_gap 8.4 %, q_yield 4.3 %, q_comf 4.1 %, q_lane 4.7 %, q_cons 9.3 %; sparse: q_stall 0 %, q_osc 0.4 % (14,712 cycle records). |

`any_arm_not_known` = [] (no regression arm lost its known value).

## Outcome under PREREG section 4
**B3 FAILED: the reward still pays slowness on 10 of 39 eligible informative states.** The floor change fixed the state it was derived on, but not the population. Section 4 says: decompose (done: `B3_DECOMP.md`, `b3_decomp_v5.py`, `raw/b3_decomp_v5.json`); the pre-named next lever is FLOOR 0.15 on seed 20261111. **The decomposition indicates FLOOR 0.15 cannot fix the failing states**; a floor of 0.0 fixes none of the 10 either (see `B3_DECOMP.md`). The draft is `../pilot_v6/PREREG_REWARD_2.DRAFT.md` (not frozen, no states drawn, nothing launched; the Master Mind decides).
B1, B2, B4 retained. Validity statement (PREREG_M_GYM_4 section 5): NOT YET VALID (B3).

## Metric families
LONGITUDINAL only as simulator reward terms and hard events; LATERAL only off_drivable/wrong_way events; TACTICAL n/a (scripted arms); STRATEGIC PAUSED (PI 2026-09-19, ITEM 25), not a pass. No estimator or CI applies: scripted policies on a deterministic simulator, one gym seed, counts over states.

## UNVERIFIED
Why `q_free` is inactive over the whole crawl of state 108 (no lead, no conflict in the stored logs) was not traced to code; candidates are `vstar < 1.0` or `stalled_by_red` (condition at `scorer_gym/gym_env.py:626`). `GOALS_AND_CLAIMS.md` is not edited here (the Master Mind lands the register entry).

# M-GYM-1 RESULT (pilot, CPU, 20 states, 6 workers; written 2026-11-08)

Evidence class: MEASURED unless marked ESTIMATE / POST_HOC. Artifacts: `results.json`, `cost_rows.json`, `raw_states.jsonl.gz`,
`summarize.out`, `det1.txt`/`det2.txt` (all in this folder). Code frozen, hashes in `CODE_PIN.json`. Thresholds and statistic exactly as `PREREG_M_GYM_1.md`.

## 1. Completeness and determinism
* 20 of 20 states finished (`run.log`: `state N done ... (20/20)`, then `RUN_DONE 1252.9`; 0 tracebacks). 10 distinct logs, navtest-excluded (`states_used.json`).
* Determinism: `det1.txt` vs `det2.txt` (two fresh processes, state 0, arm 5, both agent modes, 17 cycles x 4 values) are **bit-identical** (`diff` empty). The in-process repeat plus fresh process is also test `test_deterministic_same_process_and_fresh_process` (PASS, 6/6 tests pass). **Determinism: PASS.**

## 2. COMMITTED VERDICT: **REDUCE**
Statistic = mean over 20 states of (single-worker seconds for K=24 first actions x 17 cycles, log + idm) / 12.

| | s/state at 12 workers |
|---|---|
| **mean (the verdict)** | **28.08** (single-worker 336.96 s) |
| median / max | 26.11 / 58.54 |
| mean, warm-map states only (POST_HOC) | 26.94 |
| log mode only / idm mode only | 11.14 / 18.23 |
| measured 6-worker wall per state (contention visible) | 62.65 s (= 31.3 s/state if it scaled to 12) |

GO needs <= 26. The mean misses by 8 % (3.6 % warm); the median sits 0.4 % above. REDUCE band (26-260): confirmed. At 10,000 states: 3.25 days (mean), 3.12 days (warm), 3.6 days (6-worker wall x 0.5). ESTIMATE of the largest scale in 3 days at K=24, 17 cycles: about 9,200 states.

## 3. Cost table (mean seconds per state, single worker; 20 states)
| bucket | log | idm |
|---|---|---|
| scene + map build (shared by both modes; cold-map surcharge included) | 15.20 | (same) |
| proposals (24 actions) | 0.30 | (same) |
| agent stepping | 10.62 | **90.92** |
| controller (LQR + bicycle) | 6.42 | 7.06 |
| scoring | 41.05 | 39.65 |
| PDM-Closed planning (continuation) | **58.08** | **61.49** |
| mode subtotal | 116.2 | 199.1 |

Buckets sum to 330.8 s of the 337.0 s measured wall (1.8 % untimed). Shares: planning 36 %, idm agent stepping 27 %, scoring 24 %, scene+map 4.5 %.
Scene+map is bimodal: 6 of 20 states paid a cold map load (18-117 s, mean 15.2 s over all); warm states cost 0.9 s. With 10,000 states grouped by map and workers keeping maps resident, the warm figure (26.9) is the fairer estimate, and it still misses GO. Standing states (v0 < 0.5, n=4) cost 310-702 s; moving states 307 s on average.

## 4. Per-term activation (fraction of cycle records with q < 1; 14,451 records = 7,103 log + 7,348 idm; the 24-arm set only, regression policies excluded)
| term | fired (all) | log | idm | note |
|---|---|---|---|---|
| q_free | 8.2 % | 9.0 % | 7.5 % | applicable on 37.5 % of cycles |
| q_gap | 16.5 % | 15.2 % | 17.8 % | applicable on 34.7 % |
| q_stall | **0.0 %** | 0.0 % | 0.0 % | **sparse -> `D-RL-COLL-SPARSE-1`** in the arm set. It does fire on the brake-to-stop policy (67 cycles), so the term works; no proposal ever stands still. |
| q_yield | 5.4 % | 10.4 % | **0.57 %** | idm sparse (<1 %); log-mode only 10 %: log agents do not yield, so the yield term is mostly a log-mode signal |
| q_comf | 10.6 % | 10.7 % | 10.4 % | section 6 |
| q_osc | 1.5 % | 1.6 % | 1.5 % | |
| q_lane | 3.2 % | 3.4 % | 3.0 % | |
| q_cons | 13.9 % | 13.5 % | 14.3 % | |

Only q_stall (and q_yield in idm) are below 1 %. Mean q is 0.91-1.0 for every term, so no single factor dominates.
Hard events per episode (480 episodes per mode): log 20.8 % (off_drivable 15.0 %, collision 5.0 %, wrong_way 0.8 %, ttc 0); idm 10.8 % (off_drivable 10.0 %, collision 0, wrong_way 0.8 %, ttc 0). The TTC hard rule never fired.

## 5. Deliberate-regression arms against what was committed
| arm | committed reading | measured | result |
|---|---|---|---|
| always-slowest | KNOWN iff mean q_free (applicable cycles) < 0.6 AND return < PDM-Closed's | q_free 0.573 (n=93 cycles); return 2.31 vs 7.37 | **KNOWN** (the q_free margin is 0.027 and n is small) |
| always-fastest | KNOWN iff (hard + lead-present cycles with q_gap < 0.7) > PDM-Closed's, log or idm | log 53 vs 45; idm 67 vs 40 | **KNOWN** (log marginal, idm clear) |
| do-nothing | KNOWN iff mean per-cycle reward <= 0.02 over moving states | **0.0832** (16 moving states) | **MISS, as written** |
| oracle | KNOWN iff no hard event on the 20 states | **6 hard events** (log 4: states 2, 8, 10, 19; idm 2: states 8, 19) | **MISS, as written** |
| determinism | bit-identical | identical | **PASS** |

POST_HOC (not the verdict): do-nothing reward on cycles where v < 0.05 is 0.0006 (147 cycles). Every one of the 6 oracle hard events is one where all 24 arms hit the same hard event (2: collision at cycle 13, log; 10: off-drivable at cycle 9, log; 8 and 19: off-drivable at cycle 0 in every arm, both modes). So the oracle is not failing; the states are unwinnable for 24 proposals. States 8 and 19 end at the first cycle for every action, so they are reward-0 labels: either the evaluator's drivable test (150 m radius around the start state) is misfiring at t0, or the log starts off the polygon. Needs a look before the 10,000-state run; filtering them is a selection change that would need its own note. The oracle criterion was written for a world without unavoidable events and should be restated for B1 (oracle must be best among arms, hard events only when all arms hard).

## 6. q_comf diagnosis
Firing: 513 of 960 arms (53 %) fire at least once in 17 cycles (so "every arm" was true of the smoke state only). PDM-Closed's best arm: 45 of 605 cycles (7.4 %).
* Channels over all fired cycles: lon_jerk 1,477 (72 %), mag_jerk 518 (25 %, correlated with lon_jerk), yaw_rate 50 (2.4 %); lon_acc, lat_acc and yaw_acc never fire in the arm set. The "2 channels" in the smoke test were lon_jerk + mag_jerk.
* By cycle: **34.6 % and 35.0 % at cycles 0 and 1**, then 0.6-13 % (cycles 2-16, mean about 7 %). Fire cycles have mean |a| 1.33 vs 0.43 m/s^2 otherwise. Standing cycles (v < 2) fire 3 of 3,266.
* Cause: the jerk is a **single-step finite difference of the tracker's acceleration at 10 Hz** (`gym_env.py` ~l.331-339: limit 4.13 m/s^3 trips on a 0.41 m/s^2 step between 0.1 s steps). Cycle 0 is the jump from the history acceleration to the proposed plan, cycle 1 the hand-over to the PDM replan; both are plan-switch transients that exist because the pilot re-plans every 0.5 s. nuPlan's own comfort metric does not use a raw one-step difference: it differentiates acceleration with a Savitzky-Golay filter (`state_extractors.py`, `approximate_derivatives`, `extract_ego_acceleration` window 8). So the pilot's channel is **stricter than the official metric by construction**; the steady-state 7 % is partly that and partly LQR command steps at replans. This split between the two is not separated by this data (the log stores booleans, not jerk values).
* Smallest fix: compute jerk through nuPlan's `approximate_derivatives` over a rolling acceleration history (carry the last 8 steps in `ctx`), and score the human-log ego with the same estimator as the calibration reference. Not applied; the verdict does not depend on it (q_comf is a 0.7^k factor with mean q 0.961).

## 7. Do-nothing finding
The "stationary plan" policy does not stand still: it **brakes to a stop**, and the progress term pays for the distance covered while braking. Over the 16 moving states it collects 13.57 reward in total, of which **13.48 (99.3 %) is earned in cycles with v >= 0.05**, equal to 9.3 % of the PDM-Closed best-arm return on the same states (146.3). After it stops the reward is 0.0006 per cycle. It reaches rest after 8-10 cycles in the 10 states where it survives all 17 cycles; 6 of 16 end early in a hard event (off-drivable 4, collision 2). Per state the mean reward per cycle is 0.004-0.565 (state 2: 0.565, a short episode ended by a hard event). So the miss (0.083 vs 0.02) is the braking payout, not a standing-still payout.
Proposal for the B1 pre-registration (not applied): pay progress **relative to a comfortable-braking baseline** per cycle, `progress_c = max(0, dist_c - dist_brake(v_start_c))` with dist_brake the distance a -2 m/s^2 stop would cover from the cycle's starting speed. Brake-to-stop then earns 0 by construction, constant-speed driving is still paid, and nothing else in the reward changes. Cost: one line in the progress computation. Alternative with the same effect: keep the reward and restate the do-nothing criterion as "reward after the vehicle has stopped <= 0.02" (POST_HOC already 0.0006), accepting that early braking is paid. I recommend the first, because an RL policy would otherwise learn to brake early into a free return of about 9 % of the expert's.

## 8. Caveats that limit extrapolation
1. 6 workers on a box shared with GPU chains, scaled **linearly** to 12 (ESTIMATE). The 6-worker wall (62.6 s/state, 31.3 at 12) is 11 % above the 28.08 mean because of tail states (up to 702 s); there is no 1-worker baseline run, so contention is not separated.
2. Cold map loads inflate the mean by about 1.1 s/state; warm is 26.9. Real runs will be warm only if work is grouped by map.
3. n=20 states, 10 logs, and 4 of 20 are standing: the cost has a 2x spread (310-702 s for standing states), and the 8 % GO miss is inside that spread.
4. Reward terms are pilot operationalisations of design 2.5 (corridor lead/conflict, no red-light stall beyond the lane-connector check, TTC rule with a closing-speed guard that ignores agents behind). q_stall and idm q_yield never fire in the arm set.
5. At-fault collision is a simplified classifier (rear contact below -1 m ignored); log-mode collisions (5 %) are partly agents that do not react.
6. PDM-Closed re-plans every 0.5 s, centreline rebuilt per episode, drivable map radius 150 m around the START state, which may explain the cycle-0 off-drivable events (states 8, 19).
7. A GO-class state count at K=24 and 17 cycles is the pre-registered question; the knob estimates below are linear ESTIMATES, not measurements.

## 9. REDUCE: what fits in 3 days (12 workers, ESTIMATE, linear in K and cycles)
Budget 259,200 s => 25.9 s/state at 10,000 states. Options: (a) keep K=24 and 17 cycles: about 9,200 states; (b) K=22: -8 % => 10,000 states fit; (c) continuation 15 cycles (7 s): about -11 % of the variable cost; (d) **cheapest in label quality: hold each PDM plan for 1.0 s instead of 0.5 s**: planning is the largest bucket (119.6 s, 36 %), so it roughly halves to about 60 s and the state cost drops to about 277 s single-worker = **23 s/state, GO**, and it would also cut the cycle 0-1 comfort transients; its effect on the continuation quality is untested. Also reduce idm agent stepping (91 s, 27 %), which is 8.6x the log agent cost for the same 24 x 17 steps.

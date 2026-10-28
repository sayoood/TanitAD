# H-ST-RECOVERYLABEL zero-training step (deliverable 2)

Frozen: `FROZEN_CARDS.md` `3202a62fbf76...`, Amendment 1 `4676bdb8d27c...` (the labeller construction; written before any real data), Amendment 2 `17cce25fa237...` (POST-HOC V2, written after I saw V1's 3,000-record smoke).
Evidence MEASURED (ours; `recovery_result.json`, `raw/recovery_s{0,1}.json`, `raw/recoveryV2_s{0,1}.json`). Tier: T1-kin rollout states on TRAIN; labels are offline. A census of the full banks (306,566 windows per tick), not a sample, so no sampling interval is quoted; the training-seed variance of the bank itself is the two seeds.

Prototype: `recovery/labeller.py` (V1, registered), `recovery/labeller_v2.py` (V2, post-hoc). Not in `stack/`. Envelope constants from `feasible_decode.py` (A_MAX 4.0, KAPPA_MAX 0.2, MU 0.7, 1 % margin); no accel/curvature limit exists in `P10Prior` itself (UNVERIFIED-as-'P10's').

## (a) identity, (b) deliberate regression, (d) feasibility

* (a) identity: **1000 / 1000** real windows reproduce the human label with `np.array_equal` (zero tolerance), also equal to the stack's `gt_waypoints`.
* (b) gate G-REC on 400 real off-route records: the real labeller emitted 67, **0 checker violations** (PASS); the offset-ignoring arm FAILS the gate on 304 / 305 records (99.7 %); the no-clip / unchecked arm fails on 257 / 257 (100.0 %).
* (d) feasibility violations among EMITTED labels, full banks, independent re-check of every label: V1 s0: 0 of 18707; V1 s1: 0 of 24436; V2 s0: 0 of 14760; V2 s1: 0 of 18707. Zero by construction (refusal gate) AND measured; a refusal is not a violation. My checker equals the stack's `recover_controls` to 1.8e-15 on 66 real labels.
* Context: the HUMAN labels themselves violate this envelope at 0.2 s resolution on 2.9 % / 2.6 % of sampled ok records (s0 / s1), so 'exactly 0' is a property of the recovery phase only.
* pytest: == recovery/ (labeller unit + real-panel) ........................                                                 [100%] 24 passed in 6.39s == scripts/ (bootstrap equivalence to taniteval/ci.py) ...                                                                      [100%] 3 passed in 0.13s 

## (c) kept fraction (tick 2; card predicts 53.9 % -> ~78 %; frozen band 75-81 %)

| variant | seed | tick-2 kept before | tick-2 kept after | ceiling (all off-route recovered) | recovered (tick 2) | tick 1 before -> after |
|---|---|---|---|---|---|---|
| V1 registered | s0 | 53.85 % | 59.51 % | 78.11 % | 17334 | 79.06 -> 79.50 % |
| V1 registered | s1 | 54.69 % | 61.95 % | 77.01 % | 22260 | 80.60 -> 81.31 % |
| V2 post-hoc | s0 | 53.85 % | 58.20 % | 78.11 % | 13327 | 79.06 -> 79.52 % |
| V2 post-hoc | s1 | 54.69 % | 60.07 % | 77.01 % | 16498 | 80.60 -> 81.32 % |

**Prediction NOT CONFIRMED** (frozen band 75-81 %): V1 reaches 59.5 % / 62.0 %. V2 (arc lookahead) does not help (58.2 % / 60.1 %). Refusals, V1 over both ticks:

| seed | emitted | refused_infeasible | no_logged_future (end-edge hidden by the matcher's cut order) | degenerate | path_exhausted |
|---|---|---|---|---|---|
| s0 | 18707 | 50170 | 22199 | 4297 | 557 |
| s1 | 24436 | 33368 | 22200 | 3440 | 529 |

Where V1 refuses (s0, tick 2, 74368 off-route records): overall emitted 23.3 %, refused_infeasible 47.8 %, no_logged_future 25.8 %. By own speed: 0-1 m/s emitted 6 %; 1-3 m/s emitted 11 %; 3-6 m/s emitted 34 %; 6-10 m/s emitted 39 %; 10-100 m/s emitted 33 %. By heading error: 0-30 deg emitted 39.6 %; 30-60 deg emitted 27.3 %; 60-90 deg emitted 6.6 %; 90-181 deg emitted 0.8 %.

## (e) label-vs-human deviation (emitted, world frame, metres)

| variant | seed | n | mean-over-k p50 | p95 | max | 6 s endpoint p50 | p95 |
|---|---|---|---|---|---|---|---|
| V1 | s0 | 18707 | 2.63 | 6.26 | 45.59 | 1.61 | 5.27 |
| V1 | s1 | 24436 | 2.40 | 5.93 | 34.20 | 1.31 | 4.61 |
| V2 | s0 | 14760 | 2.47 | 6.28 | 26.29 | 1.58 | 5.47 |
| V2 | s1 | 18707 | 2.14 | 6.08 | 22.52 | 1.23 | 4.83 |

## What it means

* The human label at the matched route point stays the only label for ~54 % of tick-2 states; a feasible recovery label is obtainable for about a quarter of the off-route states with this blend, so the card's ~78 % is a ceiling that is out of reach for two measured reasons: **~26 % of tick-2 off-route states have no complete 6 s logged future at r'** (the end-edge condition is hidden behind off_lat / off_head by the matcher's cut order, so the card's 78 % double-counts them) and **~48 % are refused as infeasible**, concentrated at own speed < 3 m/s and heading error > 60 deg (emitted 0.8 % above 90 deg).
* The cheapest remaining levers, named not run: (i) a curvature-bounded planner (Dubins / clothoid with R_min = 5.05 m) in place of the Hermite blend for the low-speed / large-heading states (the Hermite's curvature is the failing check at short rungs); (ii) partial-horizon recovery labels with a loss mask for the no-logged-future states (needs a trainer flag and a G-DVB entry, so box-2 work). Neither can lift tick-2 coverage to 78 %: with no_logged_future removed the labelable ceiling is 71.9 % / 70.6 % (s0 / s1).

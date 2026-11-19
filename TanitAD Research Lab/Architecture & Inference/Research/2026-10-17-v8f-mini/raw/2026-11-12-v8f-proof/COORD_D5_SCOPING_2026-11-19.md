# COORD_D5 : scoping card for a recognised yardstick and a Thor latency read of v8F (0 GPU; decisions are the PI's)

Coordination decision D5 of `COORD_V8F_2026-11-19.md`. Written by the v8f coordinator (v8proof fork). This card starts no job and spends nothing. Every fact below is INHERITED from a read-only search of the steering documents and state files on 2026-11-19 (the E: checkout can lag the repo tip); none was re-measured for this card. Paths are given so each can be checked.

## 1. Where v8F stands today

* Every v8F number of both sessions is T0 (diagnostic on logged frames) or T1-kin (kinematic rollout on the model's own actions, the lead does not react). None is driving performance and none is comparable with a published table.
* No latency was measured for any v8, v8F or v8F-mini model, nor for their encoder alone (NOT FOUND in `Project Steering/`, the four state files, `stack/scripts/`).

## 2. What the programme already has

| item | state | source |
|---|---|---|
| NAVSIM navtest PDMS scorer | runs on the dev box for the REFe / refcv7 line; cached scorer 560 s for 12,146 tokens on 3 processes | `LOOP_STATE_P6_2026-11-06.md:1327`, `GOALS_AND_CLAIMS.md:12228` |
| reference on that yardstick | DriveRL teacher MEASURED by us 89.81 PDMS [88.56, 91.00] against PUBLISHED 95.8; published leaders 86.5 to 91.2 | `GOALS_AND_CLAIMS.md:12310`, `:6159` |
| NAVSIM data | not on the pod; navtrain not on the dev box; cached features are PhysicalAI front-wide only | `LOOP_STATE_2026-11-02.md:3222`, `PI_DECISION_QUEUE.md:1515` |
| re-perception tier (T2) | plugin contract proven on Thor (gsplat, AlpaSim, NuRec scenes), no simulated drive yet; a v8F crawl pre-registration, playback executor and scorer exist in the Master Mind's lane, last result PARTIAL | `CLOSED_LOOP_FEASIBILITY_2026-09-03.md:17,246-250,353`, `LOOP_STATE_2026-11-02.md:235,319-324` |
| Thor latency of an earlier line | 60.3 ms p50 / 63.1 ms p95 per tick, the encoder about 54 ms of it; a 9-candidate fan costs 244 % of the 100 ms budget through a batch-1 engine and 56 % through a batch-9 engine | `GOALS_AND_CLAIMS.md:6503`, `stack/scripts/build_predictor_trt.py:3-9` |
| PI ruling | 100 ms end to end on Thor with 4 cameras; results first, never gate research on latency | memory `pi-ruling-latency-100ms-results-first` |

## 3. What is missing for v8F on navtest

NAVSIM frames encoded with our encoder at the v8F token grid, a fine-tune on navtrain (the corpus and camera rig differ from PhysicalAI), and a trajectory interface to the scorer (4 s, 8 poses). The other session's estimate for a four-camera perception LoRA on navtrain is 100 to 150 GPU-h (`LOOP_STATE_2026-11-02.md:3222`; ESTIMATED there).

## 4. Proposal of the coordinator (three steps, each its own go)

| step | what | cost | who decides |
|---|---|---|---|
| S0 | latency proxy of the v8F-specific part on the pod: world-model step, K-candidate imagined rollout as one batch at the planner's own horizon (10 steps, 2.0 s; 6 s beside), heads; eager and compiled; labelled a proxy (the pod GPU is not Thor); the encoder is taken from the measured earlier line until S2 | minutes of one pod slot, a build of a timing tool | coordinator (inside approved compute) |
| S1 | yardstick scoping measurement: encode the navtest front camera (12,146 tokens) with our encoder and time it; a 2,000-step fine-tune of one v8F-mini seed on a navtrain subset; one navtest PDMS read labelled T1 by the programme's ruling, beside the teacher and the refcv7 numbers | needs navtest and a navtrain subset on a GPU machine; ESTIMATED 10 to 20 GPU-h after the data is in place; data placement is the open cost | PI (new data on the pod, spend) |
| S2 | Thor latency of the integrated candidate (encoder with 4 cameras + world model + planner fan as a batch engine) | Thor time after the refcv9 run; engine build | PI and the Thor owner session |

T2 stays in the Master Mind's lane (its crawl harness); the integrated candidate of plan step 3 is its first real input.

## 5. Recommendation

S0 now. S1 as a card once the integrated candidate exists (plan step 3), unless the PI wants the yardstick earlier on a single v8F-mini seed. S2 after refcv9 releases Thor.

## 6. Limits

All facts of sections 1 to 3 are INHERITED and one checkout can lag; the cost lines of S1 are ESTIMATED without a measurement; a single front camera on navtest is not the four-camera product; a navtest PDMS is non-reactive.

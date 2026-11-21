# SPEC_NAVTEST Amendment 9 — result: the privileged planner's route fixes REFe's nav goal (ADOPT; deployment awaits the PI)

**MEASURED 2026-10-04.**
- **Pre-registration:** `PREREG_AMENDMENT_9.md`, sha256 `ecf58574…`, registered 01:25:16 CEST and landed in `aad28d9` before any Amendment 9 GPU forward existed.
- **Snapshot:** `model_final.pt` (md5 `b54773f8…`), Amendment 7 repair ON, rule navsim_v1, in every arm.
- **Baseline (OFF):** the full-navtest seam `refe_navtest_final.npz`.
- **Tier:** NAVSIM v1.1 PDMS, open loop.
- **Estimator:** paired log-cluster bootstrap, 10,000 resamples, seed 20260927. It answers *another draw of logs* only.
- **Analysis:** `a9_analyze.py` → `a9/result_a9.json`.

| arm | status | confirmation tokens / logs | PDMS OFF → arm | **PRIMARY Δ [95 % CI]** | FRESH-LOG Δ (tokens / logs) | picks scoring 0 | Δ on the full-navtest mean |
|---|---|---|---|---|---|---|---|
| **pdm_route** | **ADOPT** | 3,045 / 82 | 80.20 → 84.37 | **+4.17 [+1.98, +7.21]** | +12.05 [+4.03, +21.67] (578 / 37) | 354 → 229 | **+1.05** |
| navgoal_straight (map-free) | NOT PROVEN | 383 / 26 | 57.18 → 70.57 | +13.39 [−1.36, +26.84] | +12.80 [−6.62, +29.58] (244 / 20) | 128 → 83 | +0.42 |
| navgoal_arc | **VOID** (RETRACTION_LOG R29) | | identical to navgoal_straight on 407/407 tokens: its yaw-rate input is 0 on every nuPlan ego state at t0 | | | | |
| a8_lane (lane graph; inadmissible reference) | reported | 422 / 32 | 56.26 → 83.01 | +26.75 [+15.56, +36.86] | +28.61 [+15.24, +40.93] (279 / 26) | 147 → 35 | +0.93 |

**pdm_route validity gates:**
- (a) the goal changed on 3,045/3,045 confirmation tokens, in the run's own recorded inputs;
- (b) the regression `goal_fix=None` vs the pre-patch goals holds on all 12,146 tokens, max |Δ| 0.0;
- (c) the 24 unchanged controls reproduce the OFF seam bit for bit (24/24);
- (d) 3,069/3,069 seam rows, 0 misses, frame control 7.3e-15 m; every harness chunk PASSes with every token valid;
- (e) the cross-check against NAVSIM's own metric-cache centerline goal: median 0.34 m on the 475 off-route tokens, 0.09 m on the controls.

**Four families:**
- 0 of 21 longitudinal and lateral components separate adversely.
- Tactical (no interval), OFF → arm on these tokens:

  | decision | κ |
  |---|---|
  | lateral | 0.78 → 0.83 |
  | longitudinal | 0.37 → 0.39 |
  | 5-way | 0.69 → 0.73 |

- Strategic: UNAVAILABLE in NAVSIM.

## Reading

- On the full navtest the fix moves the final model from 83.24 to about **84.3**.
- The arm also changes the route on ~2,600 on-route tokens. NAVSIM's correction relinks broken route roadblocks and cuts loops there, so its PRIMARY effect (+4.17) is diluted by many small goal moves. The fresh-log read (+12.05) carries the off-route repair itself.
- The lane-graph reference (+26.75 on the 422 off-route tokens) shows the ceiling there: those tokens rise to 83.0, about the on-route level of 84.4.
- The map-free straight fallback captures about half of it and is not separated at 26 logs.

## Decision (as registered)

**ADOPT** makes `goal_fix="pdm_route"` the candidate default.
- ⛔ **It is NOT deployed until the PI confirms admissibility.**
- **Why that needs a ruling:** the correction reads the map's roadblock-graph topology to repair the ROUTE. No new input reaches the model; only the nav goal moves. It is the route NAVSIM's own privileged planner uses, and that the scorer measures EP and DDC against.
- **If the PI rules it out:** nothing admissible is adopted. navgoal_straight is NOT PROVEN.

## How it ran (declared)

- **Chunked GPU run.** The single 3,069-token pdm_route seam died after 25/84 logs with a native access violation (0xC0000005, no Python traceback).
  - Re-run in 6 log chunks with the unchanged seam (`raw/2026-10-04-navtest-final/run_a9_chunked.py`): every chunk ran clean (0 crashed logs), so the crash was transient.
- **Chunked scoring.** Scored per chunk, because 3,069 tokens exceed Windows' command line.
  - Two chunks first hit the harness's RAM guard (another session's job held the memory) and were re-scored.
  - Merged by `merge_chunk_scores.py`, which refuses unless every chunk PASSes and the token set matches exactly.
- **Analysis memory bug, fixed.** The first analysis run hit a MemoryError from a bug in `a9_analyze.py`: an in-loop `NpzFile[...]` pinned 12,146 copies of the pose array. Fixed by loading each member once; nothing else changed.
- **Drive letter.** The external drive was re-lettered D: → E: on 2026-10-03; every path was re-pointed by environment variable or by repointed copies, and nothing was edited in place.

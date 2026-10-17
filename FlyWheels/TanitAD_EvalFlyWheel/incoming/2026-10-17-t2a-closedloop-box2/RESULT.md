# RESULT — first closed-loop (T2) read of refcv7-50,400, on box 2 (2026-10-17)

**Tier T2: the loop is closed on PERCEPTION, not on INTERACTION.** The model's own trajectory drives a kinematic vehicle, and the NuRec scene is re-rendered from the executed pose. Nothing in the scene reacts; there is no map, so COLLISION and OFF-ROAD are UNAVAILABLE. STRATEGIC is PAUSED (PI ruling 2026-09-19). TACTICAL is degenerate on these straight scenes.

**Setup:**
* 2 junction-free scenes (`00040136`, `7c72937c`), both unseen by refcv7. 9 rollout starts × 50 ticks at 10 Hz, about 437 windows per arm per scene.
* Kit and reading rules: `../2026-10-17-t2a-closedloop-prep/` (`RUN_ON_BOX2.md` §6, committed before any T2 number existed).
* Raw results: `raw/results/` (per-arm `T2A_*.json`, paired `T2A_PAIR_*.json`, `T2A_SUMMARY.md`), `raw/preflight.json`, and logs `raw/t2a_run2.log`, `raw/t2a_preflight3.log`.

## Gates (in order)
1. **Preflight:** 40/40, 0 HARD failures. The renderer plus refcv7 peak at 2.127 GiB of 8 GiB. Two renders of the same frame are identical (max |diff| 0).
   * Getting there took two box-2-only patches to third-party code in `D:\tanitad\venv`, both semantics-preserving, with backups kept:
     * torch `CUDACachingAllocator.h`: parameter `small` renamed, because Windows' `rpcndr.h` defines `small` as `char`;
     * gsplat `_backend.py`: the GCC-only `-Wno-attributes` is dropped on Windows (MSVC error D8021).
2. **Zero control:** `cl_ha0` read `[ok]`, so every commanded steer and accel was exactly 0.0.
3. **Cross-host control:** not run. The 1.25 GB refc-base checkpoint was not shipped, so there is no cross-host statement; within-box-2 pairs stand.
4. **Inference-seed floor:** `refcv7_vmax_off_s1 − off_s0` per metric (column 5 below). Every effect is read against it.

**Estimator:** paired episode-cluster bootstrap over ROLLOUT STARTS, B = 2,000, 9 clusters per scene, all from ONE clip.
* It answers only "would another draw of rollout starts say this?".
* Training-seed variance is UNMEASURED (one refcv7 run).

## Verdict against the committed bar
The bar, per scene and per family, never pooled: `refcv7_vmax_off_s0 − floor` has a paired CI wholly on refcv7's side AND |Δ| > the inference-seed floor.

Aggregation within a family (disclosed; §6 does not specify it):
* **PASS** = every listed metric passes;
* **FAIL (floor better)** = any metric is separated against refcv7;
* **partial** / **not met** otherwise.

⚠️ **`dist_to_gt_traj_m` IS `cross_track_abs_m`.** `cl_metrics.py:370` sets `"dist_to_gt": abs(ct)`, so the ADE family's second metric is LATERAL's cross-track under another name, and the two columns match to the last digit in every row below. ⇒ The independent ADE evidence is `ade_0_2s` alone. **Re-read on `ade_0_2s` alone, every ADE verdict is unchanged** (scene 1: PASS vs `cl_p10`, FAIL vs `cl_p10_t0` and `cl_ha0_ext_live`, not met vs `cl_ha0_ext`; scene 2: PASS vs both live floors, not met vs both held). Any 'distance to the logged path' statement below is a LATERAL claim. Found while writing this up; no number changed.

| scene | floor | family | per-metric delta [95 % CI] (refcv7_off_s0 minus floor; negative = refcv7 better) | inference-seed floor | verdict |
|---|---|---|---|---|---|
| 00040136 | `cl_ha0_ext` | ADE | `ade_0_2s` +0.644 [-0.806, +2.018]; `dist_to_gt_traj_m` -0.286 [-0.881, +0.310] | 0.012, 0.024 | not met |
| 00040136 | `cl_ha0_ext` | LONGITUDINAL | `abs_target_speed_err_ms` +0.621 [-0.205, +1.410]; `along_track_ade_m` +0.687 [-0.794, +2.069] | 0.011, 0.011 | not met |
| 00040136 | `cl_ha0_ext` | LATERAL | `heading_err_rad` -0.036 [-0.056, -0.018] SEP; `cross_track_abs_m` -0.286 [-0.881, +0.310] | 0.000, 0.024 | partial |
| 00040136 | `cl_p10_t0` | ADE | `ade_0_2s` +0.771 [+0.141, +1.414] SEP; `dist_to_gt_traj_m` -0.236 [-0.863, +0.397] | 0.012, 0.024 | **FAIL (floor better)** |
| 00040136 | `cl_p10_t0` | LONGITUDINAL | `abs_target_speed_err_ms` +0.566 [+0.204, +0.896] SEP; `along_track_ade_m` +0.778 [+0.141, +1.432] SEP | 0.011, 0.011 | **FAIL (floor better)** |
| 00040136 | `cl_p10_t0` | LATERAL | `heading_err_rad` -0.037 [-0.053, -0.021] SEP; `cross_track_abs_m` -0.236 [-0.863, +0.397] | 0.000, 0.024 | partial |
| 00040136 | `cl_p10` | ADE | `ade_0_2s` -0.373 [-0.657, -0.037] SEP; `dist_to_gt_traj_m` -1.381 [-2.197, -0.629] SEP | 0.012, 0.024 | **PASS** |
| 00040136 | `cl_p10` | LONGITUDINAL | `abs_target_speed_err_ms` -0.233 [-0.413, -0.010] SEP; `along_track_ade_m` -0.328 [-0.629, +0.026] | 0.011, 0.011 | partial |
| 00040136 | `cl_p10` | LATERAL | `heading_err_rad` -0.052 [-0.077, -0.026] SEP; `cross_track_abs_m` -1.381 [-2.197, -0.629] SEP | 0.000, 0.024 | **PASS** |
| 00040136 | `cl_ha0_ext_live` | ADE | `ade_0_2s` +0.615 [+0.005, +1.148] SEP; `dist_to_gt_traj_m` -1.588 [-2.448, -0.762] SEP | 0.012, 0.024 | **FAIL (floor better)** |
| 00040136 | `cl_ha0_ext_live` | LONGITUDINAL | `abs_target_speed_err_ms` +0.581 [+0.170, +0.942] SEP; `along_track_ade_m` +0.714 [+0.136, +1.218] SEP | 0.011, 0.011 | **FAIL (floor better)** |
| 00040136 | `cl_ha0_ext_live` | LATERAL | `heading_err_rad` -0.067 [-0.094, -0.039] SEP; `cross_track_abs_m` -1.588 [-2.448, -0.762] SEP | 0.000, 0.024 | **PASS** |
| 7c72937c | `cl_ha0_ext` | ADE | `ade_0_2s` +0.303 [-0.835, +1.382]; `dist_to_gt_traj_m` -0.305 [-0.904, +0.326] | 0.001, 0.053 | not met |
| 7c72937c | `cl_ha0_ext` | LONGITUDINAL | `abs_target_speed_err_ms` +0.226 [-0.353, +0.820]; `along_track_ade_m` +0.316 [-0.934, +1.519] | 0.003, 0.006 | not met |
| 7c72937c | `cl_ha0_ext` | LATERAL | `heading_err_rad` -0.111 [-0.240, +0.039]; `cross_track_abs_m` -0.305 [-0.904, +0.326] | 0.010, 0.053 | not met |
| 7c72937c | `cl_p10_t0` | ADE | `ade_0_2s` -0.014 [-0.562, +0.538]; `dist_to_gt_traj_m` -0.188 [-0.592, +0.251] | 0.001, 0.053 | not met |
| 7c72937c | `cl_p10_t0` | LONGITUDINAL | `abs_target_speed_err_ms` -0.060 [-0.148, +0.036]; `along_track_ade_m` -0.111 [-0.614, +0.387] | 0.003, 0.006 | not met |
| 7c72937c | `cl_p10_t0` | LATERAL | `heading_err_rad` -0.068 [-0.170, +0.060]; `cross_track_abs_m` -0.188 [-0.592, +0.251] | 0.010, 0.053 | not met |
| 7c72937c | `cl_p10` | ADE | `ade_0_2s` -0.478 [-0.873, -0.057] SEP; `dist_to_gt_traj_m` -0.648 [-1.087, -0.224] SEP | 0.001, 0.053 | **PASS** |
| 7c72937c | `cl_p10` | LONGITUDINAL | `abs_target_speed_err_ms` -0.072 [-0.252, +0.136]; `along_track_ade_m` -0.331 [-0.773, +0.154] | 0.003, 0.006 | not met |
| 7c72937c | `cl_p10` | LATERAL | `heading_err_rad` -0.121 [-0.182, -0.060] SEP; `cross_track_abs_m` -0.648 [-1.087, -0.224] SEP | 0.010, 0.053 | **PASS** |
| 7c72937c | `cl_ha0_ext_live` | ADE | `ade_0_2s` -0.436 [-0.595, -0.208] SEP; `dist_to_gt_traj_m` -0.370 [-0.740, -0.016] SEP | 0.001, 0.053 | **PASS** |
| 7c72937c | `cl_ha0_ext_live` | LONGITUDINAL | `abs_target_speed_err_ms` -0.104 [-0.243, +0.034]; `along_track_ade_m` -0.261 [-0.487, +0.042] | 0.003, 0.006 | not met |
| 7c72937c | `cl_ha0_ext_live` | LATERAL | `heading_err_rad` -0.072 [-0.137, -0.007] SEP; `cross_track_abs_m` -0.370 [-0.740, -0.016] SEP | 0.010, 0.053 | **PASS** |

### What it says, plainly

* **⛔ The committed bar is NOT met overall.** refcv7 does not beat all four of its kinematic floors on any family in both scenes. On scene `00040136`, the held P10 floor and the live echo beat it on ADE and LONGITUDINAL, separated. That is a FAIL on those families, reported as written.
* **✅ It is a lateral driver in closed loop.** LATERAL passes against both **per-tick** kinematic drivers (`cl_p10`, `cl_ha0_ext_live`) on both scenes, and ADE passes against them on scene 2.
  * Its closed-loop cross-track to the logged path (the same number as `dist_to_gt_traj_m`), and its heading error, are separated-better than every floor that re-plans each tick. For example, `dist_to_gt_traj_m` is −1.381 [−2.197, −0.629] vs `cl_p10` on scene 1 and −0.648 [−1.087, −0.224] on scene 2.
  * Against the floors HELD from t0 it ties laterally: cross-track is n.s. on both scenes, and heading is better on scene 1 only.
* **⚠️ Its weak family is LONGITUDINAL, the same as open loop (S1).** The signed target-speed error is **+1.84 m/s** on scene 1 (too fast while the driver slows) and **−0.87 m/s** on scene 2 (too slow while the driver speeds up).
  * The held echo tracks speed better on both scenes (+0.84 / −0.25 m/s).
  * refcv7 does not follow speed changes.
* **The speed ceiling is inert.** Both ceiling arms, including the R4 code where the ceiling reaches the emitted plan, move nothing beyond the inference-seed floor on either scene (`r4code_log − off_s0`: ade +0.038 [−0.068, +0.148] and −0.003 [−0.034, +0.024]). So the longitudinal deficit is not a missing cap.
  * This answers §6's lever 1, "wrong speed vs bad steering", without a new run: the steering is the strength, the speed is the problem, and a ceiling does not fix it.

## Next levers (Rule Zero), in order of measured effect
1. **The P10 residual prior (v8F-α arm α1).** P10 beats the echo at 2 s in open loop (−0.038, separated), and refcv7 loses to P10 at 2 s on speed and along-track (S1, `../2026-10-17-p10-floors-refcv/`). This T2 read locates the same deficit in closed loop. Thor, after refcv8 (≈ 10-20), 2 seeds, behind the v8f launch gate.
2. **A longitudinal head conditioned on the tactical speed goal** (the v8F three-layer design): the tactical layer emits the speed target and the operative layer tracks it. CONSISTENCY's SPEED rule measures exactly this. Track M on box 2.
3. **More scenes.** n = 2 junction-free scenes is the binding limit of this instrument. More need the gated NuRec licence (PI decision, `../2026-10-17-t2a-closedloop-prep/BLOCKERS.md`).

## Not measured / caveats
* Training-seed variance (one refcv7 run).
* The cross-host control (refc-base not shipped).
* Collision and off-road (no actors, no map).
* Only 2 scenes, each one clip. The render domain gap applies: REF-C open-loop ADE is 1.5157 on reconstructions vs 0.4728 on real footage. Read within-sim relatively only.
* The scorer's `table` step crashed on a re-run, because `T2A_SUMMARY.json` (a list) matched its own `T2A_*.json` glob. It is fixed (`isinstance` guard) in the kit, on box 2 and in `../2026-10-17-t2a-closedloop-prep/code/t2a_score.py`, and the table was regenerated from the unchanged per-arm and pair JSONs.

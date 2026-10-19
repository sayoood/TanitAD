# RESULT: EP-ADMISSION-R2-CONFIRM, the A2 margin (1.05) and A1-r1 re-measured on FRESH replicates (Research Lab agent, 2026-10-19)

## SEAL AT 1.05. All three committed bars hold on G78 (SIMULATED on the real estimator, 1,000 fresh replicates per condition, Wilson 95 %; `raw/results/r2_confirm.json`).

| bar (committed in `MM_RULING_EPADM_R1.md` section 2, not changed) | fresh measurement, G78 | holds |
|---|---|---|
| A2(1.05) false refusal (any head, any seed): Wilson upper <= 5.0 % | 1.8 % [1.1, 2.8] (18/1000), upper 2.83 % | yes |
| DAC power at x1.25 (margin 1.05), point >= 70 % | 77.4 % [74.7, 79.9] (774/1000) | yes |
| A1-r1 false refusal (any seed): Wilson upper <= 5.0 % | 3.3 % [2.4, 4.6] (33/1000), upper 4.60 % | yes |

The tightest bar is A1-r1 (upper 4.60 against 5.0): see section 5. The fallback numbers (margin 1.10) are in T1 / T2 below and were not needed.

**Status: COMPLETE.** Branch tip used: `d22d3082` (remote tip is now `48dd115`; `git diff d22d308 48dd115` over the ep-admission-r1, ep-admission-r2-confirm and p6-read-prep/code paths is empty, so nothing here is stale). Nothing was committed or pushed; everything is staged under `C:/Users/Admin/par18/epadm2/out/` with `paths.json`.

**Integration to escalate (headline).** (1) `ep_admission_sim.py` and `test_ep_admission_sim.py` at the `2026-10-19-ep-admission-r1/code/` repo paths are REPLACED by the staged versions (tip blobs they were built from: `da9e9547e018879cb9552264ea71587d0b87474e` and `08025a65156dff7b48f488ea5fa8020de7aa3d42`; staged blobs `35cba20e5a1e5d859d3e0094026f5abf3f1586fa` and `67a60cae0deeb867d26293fd727cf24932c74692`). (2) Two NEW files land beside them: `code/r2_confirm.py` (blob `eb70b21b809d12177a492b72108af72df7d260d2`) and `code/a2_power_g25_g05.py` (blob `395ea6fdd577b840ad8eaa8a22d9c71ed3642e85`). (3) The sealed registration takes the A2 margin **1.05** and may cite this file for it. (4) `RESULT.md` of the R1 package (`2026-10-19-ep-admission-r1/RESULT.md`) still says in section 4b that the 1.05 was chosen post hoc on the same replicates; it needs one dated line pointing here (not edited: it is the previous agent's package).

## 1. What was built

`--rep-offset INT` (default 0) on `ep_admission_sim.py run` and `repro` (the diff is 38 insertions, 22 deletions; built by patching the pristine tip blob with `code/patch_sim.py`, every edit asserts its anchor occurs exactly once):

* `outer_idx(rep, offset=0)` and `wild_signs(rep, offset=0)` seed `default_rng([OUTER_SALT, rep + offset])` and `default_rng([WILD_SALT, rep + offset])`; `replicate(..., offset=0)` passes it to whichever generator the condition uses; the pool initialiser `_init(conds, rep_offset=0)` hands it to the workers; `run_condition_set(..., rep_offset=0)`.
* The registered bootstrap draws (`boot_draws(117, 10000, 20260927)`) are NOT offset: they are part of the estimator and are shared by every replicate in both sets.
* `precompute` takes no offset (it does not use replicate draws except its own PC3 materialised-resample check at indices 0-2, unchanged).
* `sim_results.json` gains one additive key, `rep_offset`. `repro` gains `--rep-offset`, `--against <replicates.npz>` and `--out`. The argparse construction moved into `build_parser()` so a test can read the defaults.

## 2. Offset 0 is the banked run, bit for bit (MEASURED)

* **Fresh `precompute` in the new workdir equals the banked one:** `raw/results/stats_vs_banked.json`: 96 of 96 arrays in `stats.npz` have the same keys, dtypes, shapes and bytes as the banked `stats.npz` (sha256 `4c34b63960ee0b2f3b5662e47f1a4a978a263f29c33963593cb9001b99d514a8`), and PC3 passes again.
* **Offset-0 replicates equal the banked ones:** `raw/results/repro_offset0_vs_banked.json`: replicates 0-19 of `G78|A|CAL` (calibrated), `G78|A|PLANT_EP_0.135` (planted) and `G78|B|CAL` (the wild generator), recomputed with `--rep-offset 0` from the new workdir, are `np.array_equal` to the banked `replicates.npz` (sha256 `eb9cac71e51f835566d90a32b19be6fad30e6f4d539c23e9b549d7c252940f6c`, the tip blob `ae021f97...`) on all four arrays (`dm`, `a2`, `mp`, `me`). True.
* **The identity check has teeth (RED arm):** the same comparison with `--rep-offset 1000000` against the banked file returns False and exit 3 (`raw/results/repro_offset1e6_vs_banked_RED.json`).
* **Determinism of the fresh run (PC5):** `raw/results/repro.json`: the first 20 replicates of the same three conditions recomputed in a separate process at offset 1,000,000 equal the stored fresh ones. True.
* **Tests** (section 6): offset 0 reproduces golden draws taken from the pre-edit module; offset 1,000,000 differs and equals the draw of index `rep + 1,000,000`; no fresh replicate of 1,000 reuses a banked draw.

## 3. The run

`chain.cmd` (launched detached, `cmd.exe /c ... < NUL`): `precompute` -> `stats_vs_banked` -> `repro` at offset 0 against banked -> `run --workdir D:/epadm2/run --n-reps 1000 --workers 12 --rep-offset 1000000` -> `repro` at offset 1,000,000. 23 conditions x 1,000 replicates (IDENTITY x 50), B = 10,000 throughout, the same conditions, generators, estimator and registered draws as the banked run. The run took **441 s** on 12 workers with other agents loading the CPU (the banked run: 350 s), so the wait was under the 10-minute line and a script waited for it; no `HANDOFF.md` was needed. `ep_admission.py` sha256 `f0193372085022ab7bfc6241f2ca5638fca710cbd4170a19ce0ae95dde6637c4` (the same as the banked run); `SIM_DESIGN.md` sha256 `5ae4b66516a5f784ad0c343b155a57d62cb60a25c1f027596e182615b6cb52ce` (unchanged).

**Controls on the fresh run:** PC1 (calibrated by construction) PASS on G78, G25, G05 and the wild generator; PC2 (the check has teeth) PASS; PC3 (shortcut = estimator) PASS; PC4 (inline decision restatement = the module's, 0 disagreements over all conditions) PASS; PC5 PASS; PC6 (identity candidate reads exactly 0) PASS. Pre-committed flags F1-F5 of `SIM_DESIGN.md` section 8: none fired. The A1-r1 inline restatement equals the module's decision on all three geometries (T3).

## 4. Results (machine-generated from `raw/results/replicates.npz` / `sim_results.json` by `code/r2_confirm.py`; nothing below was typed by hand)

#### D. THE DECISION (G78, the primary geometry; fresh replicates, rep_offset 1000000, N = 1000 per condition)

| bar (committed in MM_RULING_EPADM_R1.md s2) | measured | holds |
|---|---|---|
| A2(1.05) false refusal, any head, any seed: Wilson upper <= 5.0 % | 1.8 % [1.1, 2.8] (18/1000) -> upper 2.83 % | **True** |
| DAC power at x1.25 (margin 1.05), point >= 70 % | 77.4 % [74.7, 79.9] (774/1000) | **True** |
| A1-r1 false refusal, any seed: Wilson upper <= 5.0 % | 3.3 % [2.4, 4.6] (33/1000) -> upper 4.60 % | **True** |

**Seal at 1.05.** All three bars hold.

#### T1. A2 false refusal on CALIBRATED replicates (any head, any seed), by margin and geometry; Wilson 95 %

| geometry | margin 1.00 (r1 as specified) | **margin 1.05** | margin 1.10 (fallback) |
|---|---|---|---|
| G78 | 7.2 % [5.8, 9.0] (72/1000) | 1.8 % [1.1, 2.8] (18/1000) | 0.4 % [0.2, 1.0] (4/1000) |
| G25 | 6.5 % [5.1, 8.2] (65/1000) | 1.6 % [1.0, 2.6] (16/1000) | 0.1 % [0.0, 0.6] (1/1000) |
| G05 | 7.0 % [5.6, 8.8] (70/1000) | 1.4 % [0.8, 2.3] (14/1000) | 0.2 % [0.1, 0.7] (2/1000) |

#### T1b. A2 false refusal per head (any seed), G78

| head | 1.00 | 1.05 | 1.10 |
|---|---|---|---|
| NC | 2.7 % [1.9, 3.9] (27/1000) | 1.2 % [0.7, 2.1] (12/1000) | 0.2 % [0.1, 0.7] (2/1000) |
| DAC | 3.7 % [2.7, 5.1] (37/1000) | 0.5 % [0.2, 1.2] (5/1000) | 0.1 % [0.0, 0.6] (1/1000) |
| TTC | 1.6 % [1.0, 2.6] (16/1000) | 0.2 % [0.1, 0.7] (2/1000) | 0.1 % [0.0, 0.6] (1/1000) |

#### T2. A2 POWER (G78): refusal of rows that under-predict risk by rho at the picks; any seed; Wilson 95 %

| head planted | rho | margin 1.00 | **margin 1.05** | margin 1.10 |
|---|---|---|---|---|
| NC | 1.25 | 35.8 % [32.9, 38.8] (358/1000) | 25.4 % [22.8, 28.2] (254/1000) | 16.6 % [14.4, 19.0] (166/1000) |
| NC | 1.5 | 77.4 % [74.7, 79.9] (774/1000) | 68.0 % [65.0, 70.8] (680/1000) | 56.6 % [53.5, 59.6] (566/1000) |
| DAC | 1.25 | 93.8 % [92.1, 95.1] (938/1000) | 77.4 % [74.7, 79.9] (774/1000) | 53.2 % [50.1, 56.3] (532/1000) |
| DAC | 1.5 | 100.0 % [99.6, 100.0] (1000/1000) | 99.9 % [99.4, 100.0] (999/1000) | 99.5 % [98.8, 99.8] (995/1000) |
| TTC | 1.25 | 64.3 % [61.3, 67.2] (643/1000) | 47.1 % [44.0, 50.2] (471/1000) | 28.5 % [25.8, 31.4] (285/1000) |
| TTC | 1.5 | 98.1 % [97.1, 98.8] (981/1000) | 96.0 % [94.6, 97.0] (960/1000) | 88.9 % [86.8, 90.7] (889/1000) |

#### T3. A1-r1 false refusal on CALIBRATED replicates (generator A), by geometry; Wilson 95 %

| geometry | any seed (module) | any seed (inline restatement) | seed 0 | seed 1 |
|---|---|---|---|---|
| G78 | **3.3 % [2.4, 4.6] (33/1000)** | 3.3 % [2.4, 4.6] (33/1000) | 3.0 % [2.1, 4.3] (30/1000) | 3.2 % [2.3, 4.5] (32/1000) |
| G25 | **2.8 % [1.9, 4.0] (28/1000)** | 2.8 % [1.9, 4.0] (28/1000) | 2.8 % [1.9, 4.0] (28/1000) | 2.8 % [1.9, 4.0] (28/1000) |
| G05 | **2.8 % [1.9, 4.0] (28/1000)** | 2.8 % [1.9, 4.0] (28/1000) | 2.7 % [1.9, 3.9] (27/1000) | 2.8 % [1.9, 4.0] (28/1000) |

Generator B (wild sign-flip, G78, the cross-check): 3.0 % [2.1, 4.3] (30/1000).  A1-r1 power (G78, generator A): +0.05 -> 99.4 % [98.7, 99.7] (994/1000); +0.135 -> 100.0 % [99.6, 100.0] (1000/1000).

#### T4. FRESH (offset 1,000,000) beside the BANKED run (offset 0; the replicates the 1.05 was chosen on); point estimates

| quantity | banked (offset 0) | fresh (offset 1,000,000) |
|---|---|---|
| A2(1.05) false refusal G78 | 1.9 % [1.2, 2.9] (19/1000) | 1.8 % [1.1, 2.8] (18/1000) |
| A2(1.10) false refusal G78 | 0.6 % [0.3, 1.3] (6/1000) | 0.4 % [0.2, 1.0] (4/1000) |
| A2(1.05) false refusal G25 | 1.9 % [1.2, 2.9] (19/1000) | 1.6 % [1.0, 2.6] (16/1000) |
| A2(1.10) false refusal G25 | 0.3 % [0.1, 0.9] (3/1000) | 0.1 % [0.0, 0.6] (1/1000) |
| A2(1.05) false refusal G05 | 1.8 % [1.1, 2.8] (18/1000) | 1.4 % [0.8, 2.3] (14/1000) |
| A2(1.10) false refusal G05 | 0.4 % [0.2, 1.0] (4/1000) | 0.2 % [0.1, 0.7] (2/1000) |
| DAC power x1.25 margin 1.05 (G78) | 78.4 % [75.7, 80.8] (784/1000) | 77.4 % [74.7, 79.9] (774/1000) |
| DAC power x1.25 margin 1.10 (G78) | 54.4 % [51.3, 57.5] (544/1000) | 53.2 % [50.1, 56.3] (532/1000) |
| DAC power x1.5 margin 1.05 (G78) | 99.9 % [99.4, 100.0] (999/1000) | 99.9 % [99.4, 100.0] (999/1000) |
| DAC power x1.5 margin 1.10 (G78) | 99.6 % [99.0, 99.8] (996/1000) | 99.5 % [98.8, 99.8] (995/1000) |
| A1-r1 false refusal G78 | 2.7 % [1.9, 3.9] (27/1000) | 3.3 % [2.4, 4.6] (33/1000) |
| A1-r1 false refusal G25 | 2.3 % [1.5, 3.4] (23/1000) | 2.8 % [1.9, 4.0] (28/1000) |
| A1-r1 false refusal G05 | 2.0 % [1.3, 3.1] (20/1000) | 2.8 % [1.9, 4.0] (28/1000) |

#### T5. SUPPLEMENTARY: A2 power on G25 and G05 (planted A2 conditions are NOT in the banked condition set; same generator, fresh offset)

| geometry | head | rho | margin 1.00 | **margin 1.05** | margin 1.10 |
|---|---|---|---|---|---|
| G25 | NC | 1.25 | 29.7 % [26.9, 32.6] (297/1000) | 20.5 % [18.1, 23.1] (205/1000) | 12.2 % [10.3, 14.4] (122/1000) |
| G25 | NC | 1.5 | 72.5 % [69.7, 75.2] (725/1000) | 63.3 % [60.3, 66.2] (633/1000) | 50.6 % [47.5, 53.7] (506/1000) |
| G25 | DAC | 1.25 | 78.0 % [75.3, 80.5] (780/1000) | 58.2 % [55.1, 61.2] (582/1000) | 36.0 % [33.1, 39.0] (360/1000) |
| G25 | DAC | 1.5 | 99.4 % [98.7, 99.7] (994/1000) | 98.6 % [97.7, 99.2] (986/1000) | 95.5 % [94.0, 96.6] (955/1000) |
| G25 | TTC | 1.25 | 56.6 % [53.5, 59.6] (566/1000) | 37.6 % [34.7, 40.6] (376/1000) | 20.2 % [17.8, 22.8] (202/1000) |
| G25 | TTC | 1.5 | 95.5 % [94.0, 96.6] (955/1000) | 90.8 % [88.8, 92.4] (908/1000) | 83.0 % [80.5, 85.2] (830/1000) |
| G05 | NC | 1.25 | 33.1 % [30.3, 36.1] (331/1000) | 23.0 % [20.5, 25.7] (230/1000) | 13.5 % [11.5, 15.8] (135/1000) |
| G05 | NC | 1.5 | 79.3 % [76.7, 81.7] (793/1000) | 71.5 % [68.6, 74.2] (715/1000) | 57.9 % [54.8, 60.9] (579/1000) |
| G05 | DAC | 1.25 | 72.3 % [69.4, 75.0] (723/1000) | 54.0 % [50.9, 57.1] (540/1000) | 33.3 % [30.4, 36.3] (333/1000) |
| G05 | DAC | 1.5 | 99.1 % [98.3, 99.5] (991/1000) | 97.7 % [96.6, 98.5] (977/1000) | 93.2 % [91.5, 94.6] (932/1000) |
| G05 | TTC | 1.25 | 53.7 % [50.6, 56.8] (537/1000) | 34.6 % [31.7, 37.6] (346/1000) | 18.5 % [16.2, 21.0] (185/1000) |
| G05 | TTC | 1.5 | 95.4 % [93.9, 96.5] (954/1000) | 90.5 % [88.5, 92.2] (905/1000) | 81.4 % [78.9, 83.7] (814/1000) |

## 5. What this does and does not confirm (read before quoting)

* **"Fresh" means fresh replicate DRAWS on the same real bank.** The margin 1.05 was chosen on replicates 0-999; this run applies the committed rule to replicates 1,000,000-1,000,999 of the same generator. That removes the selection-on-the-replicates problem the ruling named. It does NOT test a different bank: every rate here is conditional on the real S0 bank (117 logs, the EP-R1 rows) and on the one registered bootstrap-draw set (seed 20260927), which both replicate sets share. A candidate scored on another bank has its own noise. The banked and fresh numbers agree within sampling noise on every A2 quantity (T4), which is what a confirmation on a new replicate set can show.
* **A1-r1 has the least headroom.** Fresh 3.3 % [2.4, 4.6] against the 5.0 % bar on the upper end (0.4 percentage points of headroom); the banked run read 2.7 % [1.9, 3.9]. The 0.6-point difference is about 0.8 standard errors (ESTIMATED by hand, pooled p about 3 %, n = 1,000 each), i.e. noise. Pooled over both replicate sets (A1's bar was not tuned on the banked replicates, so pooling is descriptive) it is 60/2,000 = 3.0 % [2.34, 3.84] (computed with the simulator's own Wilson function). The decision above uses the fresh set alone, as the rule says. Seed-level failures on G78, fresh: 53 lower-CI-only, 0 point-only, 9 both, of 2,000 seed evaluations.
* **NC is weak at any margin** (as the ruling already states): power at x1.25 and margin 1.05 is 25.4 % [22.8, 28.2] (banked 26.4 %); at x1.5 it is 68.0 % [65.0, 70.8]. The registration should state NC's power and not claim NC calibration is tested at x1.25.
* **A2 power falls with fewer changed picks (T5, SUPPLEMENTARY).** The banked and the R2 condition sets plant the A2 shortfalls on G78 only; the brief asked for G25 / G05 beside it, so `code/a2_power_g25_g05.py` plants them on G25 and G05 with the same generator, builder and fresh offset (these conditions have NO banked counterpart). DAC power at x1.25 and margin 1.05: G78 77.4 %, G25 58.2 % [55.1, 61.2], G05 54.0 % [50.9, 57.1]; at x1.5 it stays at 97.7 % or more. The committed bar is on G78 only and is met; on the smaller geometries the same x1.25 DAC bar would NOT be met. The cause was not investigated. The sealed registration should say that A2 power at x1.25 depends on the candidate's changed-pick share, the way A1-r1 is a per-changed-pick test.
* **A2 false refusal is flat across geometries** (1.4-1.8 % at margin 1.05, upper end at most 2.83 %), so the bar is not a G78 artefact.
* This run does not touch rules A1-r1 / A2-r1, the slot source, the seeds or P6; those stay as `MM_RULING_EPADM_R1.md` sections 1, 3 and 4 say.

## 6. Tests (the full files naming `ep_admission_sim.py`, `python -I test_ep_admission_sim.py -v`, CPU only)

* BASE (pristine tip blob `08025a65...` against the pristine tip `ep_admission_sim.py` blob `da9e9547...`): **15 tests OK** (`raw/logs/test_sim_BASE.log`).
* MINE: **23 tests OK** = the 15 unchanged + 8 new in `TestRepOffset` (`raw/logs/test_sim_MINE.log`): golden offset-0 draws from the pre-edit module for both generators; offset 1,000,000 differs and equals the draw of index `rep + offset`; none of 1,000 fresh draws equals any of the 1,000 banked ones; `replicate` default = offset 0 and offset shifts both generators; the pool worker applies the offset it was initialised with; the CLI default is 0 and the flag is an int; and three RED arms (a generator that ignores the offset, wild signs that ignore it, a worker that drops it) that the same assertions catch.
* `r2_confirm.py --selftest` (the decision function on constructed rates, including each bar missing alone and a boundary case): PASS (`raw/logs/r2_confirm_selftest.log`).

## 7. Deliverable manifest

Staged at `C:/Users/Admin/par18/epadm2/out/` (paths in `paths.json`, 3.4 MB in all, nothing only on D:). Work trees: `D:/epadm2/w` (repo layout), `D:/epadm2/run` (precompute + run), `D:/epadm2/banked` (the tip's banked `replicates.npz` / `stats.npz`).

| artifact | repo path (under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/`) | lives in |
|---|---|---|
| edited simulator (`--rep-offset`) | `2026-10-19-ep-admission-r1/code/ep_admission_sim.py` | staged `par18/epadm2/out` + `D:/epadm2/w` |
| edited sim tests (23) | `2026-10-19-ep-admission-r1/code/test_ep_admission_sim.py` | same |
| NEW decision script / supplementary A2 power script | `2026-10-19-ep-admission-r1/code/r2_confirm.py`, `.../a2_power_g25_g05.py` | same |
| this file | `2026-10-19-ep-admission-r2-confirm/RESULT.md` | same |
| helper scripts (chain, stats comparator, the two anchored patches, stager) | `2026-10-19-ep-admission-r2-confirm/code/` | same |
| fresh results: `sim_results.json`, `r2_confirm.json`, `result_tables.md`, `replicates.npz`, `stats.npz`, `precompute.json`, `a2_power_g25_g05.json` (+ its replicates), the three identity / determinism JSONs | `2026-10-19-ep-admission-r2-confirm/raw/results/` | same |
| logs: chain, precompute, run, repro (x3), r2_confirm, selftest, base / mine tests, supplementary run | `2026-10-19-ep-admission-r2-confirm/raw/logs/` | same |


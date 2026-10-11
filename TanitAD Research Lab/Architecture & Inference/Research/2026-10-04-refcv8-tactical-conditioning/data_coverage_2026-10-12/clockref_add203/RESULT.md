# refcv8 clip-clock reference for the 203 ADD clips, and the launch gate's own G-CLOCK verdict on the 4,572-clip train view (2026-10-12, 0 GPU)

Layer key: VIEW = Thor train symlink view `refcv8-b1-416x1024-train-4572` (manifest md5 `77dafb50af0aedd1b940cf12c12518ef`); SIDECAR = `refcv8_clip_clock_sidecar_ts33.jsonl` (md5 `c81c2739b5bef58e3c7e389245bd506a`); REF = an independent clip-clock reference JSON read by `launch_gate.py::load_clock_reference`. Evidence MEASURED (ours) unless tagged; every number cites an artifact under `raw/` (this directory). Tier: not an eval - a data-instrument check, no T0/T1 stamp applies. Estimators: deterministic arithmetic over every window (781,635) - there is no interval and none is claimed. Code tree for every gate run: Thor `/home/nvidia/refcv6_run/fa43d9c5ce` (launch_gate.py md5 `b20f3d0adcf41ec3e534a98358796a25` = the fa43d9c blob).

## 0. Verdict (TRAIN split, `launch_gate.py check --checks G-CLOCK`, the gate's own child job; `raw/verdict_summary.json`)
| run | view / sidecar | references | unverified / n_clips | frac vs cap 1 % (max 45 clips) | fallback | measured-unreferenced | violations / windows | G-CLOCK train |
|---|---|---|---|---|---|---|---|---|
| C control | old 4,369 / refcv6 sidecar (profile refcv7, refcv7's recorded launch argv) | ALLTRAIN | 22 / 4,369 | 0.5035 % | 22 | 0 | 0 / 746,946 | **PASS** |
| B before | 4,572 / ts33 | ALLTRAIN | 225 / 4,572 | 4.921 % | 22 | 203 | 0 / 781,635 | **FAIL** |
| A after-1 | 4,572 / ts33 | ALLTRAIN + ADD203 | 55 / 4,572 | 1.203 % | 22 | 33 | 0 / 781,635 | **FAIL** (10 clips over) |
| D after-2 | 4,572 / ts33 | ALLTRAIN + ADD203 + ADD203_slowregime | 42 / 4,572 | **0.9186 %** | 22 | 20 | 0 / 781,635, worst 1e-6 s | **PASS** (3 clips of margin) |
| E regression | as D, slow-regime ref shifted +0.2 s | ... | 42 / 4,572 | 0.9186 % | 22 | 20 | **2,219** / 781,635 (13 clips, worst 0.200001 s) | FAIL, as it must |

* The standard q4c reference (ADD203 file, 170 clips) is NOT enough: A fails by 10 clips. The 33 stationary ADD clips cannot be fitted by q4c (fewer than 20 rows above 2 m/s).
* D needs a third reference, built from the SAME instrument with its speed floor lowered 2.0 -> 0.5 m/s, backed by a pre-registered control (section 3). It verifies 13 of the 33. **Whether that reference is acceptable is a Master Mind / PI decision; I changed no rule, cap or clip set** - it is a data file handed to the existing `--clock-reference` option.
* The trainer's own G3 (`assert_label_clock_true`) reads PASS with 22 unverified (0.481 %) in B, A, D and E alike: it counts only clips absent from the sidecar. The previous hold's "22/4,572 = 0.481 % PASS" (landed_hold_add203 RESULT sec. 3a) was G3's view; the gate's G-CLOCK additionally requires an independent reference for every MEASURED clip and was FAIL (1.203 %) until D.
* Positive control: the gate's historical-clock control reads 772,652-774,871 violating windows in A/D (non-zero, as required), so the instrument can see the defect it exists for. Run E shows the slow-regime path is seen too.

## 1. Controls on the instrument (all passed; `raw/ctl_compare.txt`)
1. The UNCHANGED `q4c_grid_vs_egolog.py` (md5 `4f3943399cfe494f6cbe360d593dc5f8`, the fa43d9c blob; `_common.py` md5 `a8588fa92956f04d1cc01951fbbb5ec8`) was run via `main()` on Thor with only module constants patched (`EGO_DIR` = Thor `/home/nvidia/data/obstacle/egomotion_alpamayo`, 4,800 files = the stored run's `ego_dir_n_files`; `MANIFESTS`; `N_TRAIN_SAMPLE`) - driver `code/run_q4c.py`; Thor has no psutil, so a 10-line `/proc/meminfo` shim (`code/thor_shim_psutil.py`) stands in for it, nothing installed into the venv.
2. Brief control: 38 OLD train clips (20 random, seed 20261012, + the 19 clips ALLTRAIN did not reference) reproduce their stored ALLTRAIN entries: max |d grid_start| **1.2e-14 s**, |d dt| 1.4e-16 s (bar 1e-9 s); the 12 non-fits are the same 12; the 7 fits with resid >= 5 ms reproduce too (same values).
3. Stronger: the whole 4,572-clip view in one run reproduces **all 4,357 stored old entries** to max |d grid_start| 3.3e-14 s, |d dt| 2.4e-16 s, n_moving_rows identical; its 170 new fits equal the ADD203 file bit-for-bit.
4. K2 (synthetic known linspace, built into q4c): dt relative error 1.9e-9, start error 1.9e-5 ms (ADD203 run); 7e-9 / 5.4e-5 ms (control run).

## 2. The reference: `q4c_grid_vs_egolog_ADD203.json` (same schema, loadable by `load_clock_reference` next to ALLTRAIN)
* 203 ADD clips requested, 0 without egolog, **170 fit, all 170 with fit_resid_max_ms < 5** (max 0.0049 ms), n_moving_rows 21-199 (131 at 180+).
* **33 do not fit - all stationary** ("fewer than 20 rows above 2 m/s"); they are exactly the 33 the ts33 sidecar clocked by camera timestamps (`raw/add_notfit_sha12.json`). No fit failed for no-egolog or residual reasons.
* Keyed by sha12 of the clip id; the file carries no raw UUID (scanned).

## 3. Step 3 cross-check (`raw/xcheck_q4c_vs_ts33.json`, dev box) - what is and is not independent
* The ts33 sidecar's 170 moving-ADD rows were built by `build_clip_clock_sidecar.py`, which is the SAME position-inversion algorithm as q4c: q4c vs sidecar there is max |d grid_start| 3.0e-14 s - **determinism, not independence**. Only the 33 stationary rows are camera-timestamp rows.
* The genuinely different measurement is the camera-timestamp route (`pose_sync.resample_grid` on `camera_front_wide_120fov/<id>.timestamps.parquet` vs the ego log's first timestamp) recomputed fresh: vs q4c on the 170 moving ADD clips max |d grid_start| **1.24e-7 s**, |d dt| 3.1e-9 s. The ts33 route is therefore validated on the ADD population itself, not only on the older clips.
* For the 33 stationary rows, sidecar == camera-timestamp route exactly (same code): no information.

## 4. Why a third reference, and how it was built (`code/run_q4c_slow.py`, `raw/slow.log`, `raw/q4c_slowregime_raw.json`)
The gate wants an independent reference per MEASURED clip. For the 33 stationary ADD clips the only other kinematic measurement is the same ego-log position inversion at a lower speed floor. `fit_clip()` is untouched; only its module constant `V_MIN` is set. **Pre-registered before the run** (docstring of `run_q4c_slow.py`): a threshold of {0.1, 0.2, 0.3, 0.5} m/s is admissible iff on control S2 - the OLD moving clips with every row at >= 2.0 m/s masked out, so only the slow rows enter the fit, truth = the stored ALLTRAIN entry - EVERY clip has |d grid_start| <= 1 ms and |d dt| x 200 <= 1 ms; use the smallest admissible.
| V_MIN m/s | S2 clips | S2 max d grid_start | S2 max d dt x200 | admissible |
|---|---|---|---|---|
| 0.1 | 892 | 13.2 ms | 46.3 ms | no |
| 0.2 | 837 | 9.2 ms | 46.2 ms | no |
| 0.3 | 771 | 11.2 ms | 57.5 ms | no |
| **0.5** | **680** | **0.0325 ms** | **0.0431 ms** | **yes** (max resid 0.0139 ms; S1 on the 4,350 unmodified moving clips: 5.1e-7 s) |
The failures at <= 0.3 are stop-and-go fits that mis-invert (resid up to ~1.9 s); at 0.5 none occurs, so the result rests on one threshold of the grid, and the grid's largest. Targets (45 = 33 ADD + 12 old stationary) at 0.5: **25 fit, all resid < 5 ms (max 0.0013 ms): 13 ADD + 12 old**; 20 ADD do not fit. `q4c_grid_vs_egolog_ADD203_slowregime.json` holds the 13 ADD (the 12 old are fallback-clock clips and stay in the raw file only).
**Independence check on the very population (`raw/cross_slow_vs_ts.json`, dev box):** slow-regime ego-log inversion vs the camera-timestamp route on all 25: max |d grid_start| **7.6e-8 s**, |d dt| 5.8e-9 s, **1.1 microseconds over the whole 200-row window**. At gate level (run D) the worst |t_trainer - t_true| on the 13 clips is 1e-6 s. So for 13 of the 33, the sidecar's camera-timestamp clock is confirmed by a measurement from an unrelated stream.

## 5. What stays unverified, and what would unblock it (`raw/unverified_42_breakdown.json`, sha12 keys)
Under D: 42 = **22 old fallback-clock clips** (12 stationary, 7 with q4c resid >= 5 ms, 3 refused by the sidecar builder's stricter K3 gate although ALLTRAIN has them at resid < 0.003 ms) **+ 20 ADD clips**: **15 with no kinematic content at all** (cache path <= 0.07 m, max speed <= 0.29 m/s in the window; 14 of them show motion elsewhere in the 100 Hz log - the car is parked during the window - and one is parked for the whole log: `raw/probe_stationary.json`) and **5 below the instrument's built-in 20-row floor** (11-18 rows above 0.5 m/s: 9dee20474b44, 7201bcbf5652, 163d534424f4, b2f655eeea5a, 1ffcfbc5548a).
* **No independent measurement exists for the 15 parked clips.** Cache rows carry only pose/action (the v2ep file has no timestamps), the camera-timestamp route is the sidecar's own source, and the ego log cannot locate a row that never moved. By construction camera mp4 PTS carry no offset to the ego-log clock; whether the obstacle.offline timestamps are stamped independently of the camera parquet was NOT probed (UNVERIFIED) - and for these clips it would only be a second view of the same frames. Not built, by the brief's own condition.
* The 5 floor clips could be reached only by lowering the 20-row constant inside `fit_clip` (an instrument change I did not make) or V_MIN to 0.1, which failed the control.
* **Margin is thin: 42 of a maximum 45.** Three more unreferenced clips would fail. Acceptable levers (all MM/PI): (a) accept D as is; (b) clock the 3 K3-refused old clips - but a sidecar row copied from q4c and referenced by q4c is a tautology, so they would need the camera-timestamp route (the previous hold left the old 22 unclocked on purpose for refcv7 comparability); (c) raise the cap - forbidden here ("do not raise the cap"), a recorded PI decision (E2).

## 6. Integration (for the Master Mind - nothing here edits the gate)
1. Land `q4c_grid_vs_egolog_ADD203.json` and `q4c_grid_vs_egolog_ADD203_slowregime.json` (this directory).
2. The refcv8 profile inherits `clock_reference = (ALLTRAIN,)` from `_REFC` (launch_gate.py:187). The A2 `run` must pass `--clock-reference` THREE times (ALLTRAIN, ADD203, ADD203_slowregime) - passing it REPLACES the profile default, so ALLTRAIN has to be repeated - or the profile tuple is extended by the Master Mind. The token binds each file as `gate:clock-reference[i]`.
3. What I ran is the gate's real child job with a hand-written `ctx.json` (`raw/ctx_*.json`; placeholder `tree_sha256`; `--out` redirected; v9 / VIS-1 flags left at the existing old files because the train4572 releases are not built - neither is read by G-CLOCK; `--join-defect-masks` = the tree's `refcv8_join_label_defects_train4572.json`). It is the verdict's arithmetic on the real caches, NOT a launch token: the A2 token needs the real `run` on the final argv after the two blockers of landed_hold_add203 RESULT sec. 5 (v9 train release, VIS-1 sidecar) are built.
4. First attempt of B and A died on the smoke argv's `--join-defect-masks` pointing at a tree that is not on Thor (`raw/gate_*_try1.log`); fixed in the ctx, not in any repo file.

## 7. Not done / UNVERIFIED
* The eval split is run by the same job but is out of scope (3/139 unverified, E2(b) mode, unchanged by the ADD clips).
* The slow-regime control (S2) masks fast rows of moving clips; the real targets are slow clips in the log. Agreement with the camera-timestamp route on the 25 targets (1.1 microseconds) is the evidence that closes this gap for those 25; for any other slow clip it is not.
* Thor: MemAvailable >= 71 GB at every sample (abort floor 25; the 12 GB per-process RSS bound was NOT sampled - UNVERIFIED), CPU only, `nice 19`, all writes under the new `/home/nvidia/clockref203/` (30 MB); a find over refcv6/7/8 run dirs and `data/` found no other file changed.

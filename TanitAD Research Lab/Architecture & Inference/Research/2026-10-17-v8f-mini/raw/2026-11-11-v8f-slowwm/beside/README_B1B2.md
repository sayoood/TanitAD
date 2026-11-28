# SLOWWM beside readings B1 (decodability) and B2 (WMACT audit)

Code and tests only, CPU, tier T0. Protocols: `../BESIDE_RULES_SLOWWM.md` (frozen, read, not reinterpreted). Nothing staged, committed or pushed; no training, no box-2 access.
Imported, never edited: `read/lp_slowwm.py`, `ladder_read/lr_lib.py`, `ladder_plantest/lp_lib.py`, `wmact/*`, `slowwm_hook.py`, `slow_basis.py`, `taniteval` (tree under `F:/TanitAD-scratch/slowwm/tree`).

## Files (all in `theory/slowwm/beside/`)
| file | role |
|---|---|
| `decod_lib.py` | B1: rolled z1 (`stack.rollout`, context 1, k = `lr_lib.k_of(1, 2.0, 1)` = 10), targets, yaw-rate tangent mask, ridge probe (CV grouped by clip), R2 / MAE with the episode-cluster bootstrap (B 2,000, seed 0) |
| `decod_run.py` | B1 runner, specs `"<arm>\|<run_dir>\|<basis>\|<ubasis>"`, per-arm `decod_<arm>.json`, resumable, `DECOD_B1.json`; refuses without `CONTROLS_B1B2.json` all_pass; repeats controls (i), (ii) and red arms on each arm's own arrays |
| `decod_build_train.py` | per-frame inputs of the 141 WMACT train clips (`frames_train_s{0,1}.npz`, `ubasis_train_{slow,pca}.npz`). **Not run: needs the box-2 stream.** Its per-clip function is tested on an eval clip |
| `decod_controls.py` | controls B1 (i)-(iii) with red arms |
| `wmact_arm.py` | B2: collection (adapter = `wmact_lib.collect_latents`; basis = `z1_from_basis` of the live u), `stats_for` (port of the `run_wmact.py` body, all statistics from `wmact_lib`), control 4, `read_arm_wmact` |
| `wmact_run.py` | B2 runner, same spec format, `wmact_<arm>.json`, `WMACT_B2.json` |
| `wmact_controls.py` | B2 control and its checks (merges key `b2`) |
| `CONTROLS_B1B2.json` | all controls, `all_pass: true` (MEASURED 2026-11-11) |
| `tests/test_decod_wmact.py` | 10 tests, all pass (MEASURED 4.9 s) |

## Env
`C:/Users/Admin/venvs/tanitad/Scripts/python.exe`; `PYTHONPATH` = tree `stack;stack/scripts;taniteval` joined with `;`; `LADDER_READ_CODE_ROOT=F:/TanitAD-scratch/slowwm/tree`, `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`; `tanitad.__file__` asserted under the tree. Runners write to `$BESIDE_OUT` (default `F:/TanitAD-scratch/slowwm_beside_build/b1b2/out`), `--out DIR` overrides. Free RAM >= 8 GB is checked before every heavy step (MemoryError, nothing killed).
```
python decod_controls.py ; python wmact_controls.py         # then copy $BESIDE_OUT/CONTROLS_B1B2.json here
python decod_run.py --train-dir <DIR from decod_build_train.py> "SLOW_s0|<run_dir>|<basis_slow.npz>|<ubasis_slow.npz>" "R35H_s0|<ladder run>|" ...
python wmact_run.py "SLOW_s0|<run_dir>|<basis_slow.npz>|<ubasis_slow.npz>" ...
```

## Implementation choices where the rules are silent (none changes a rule)
* B1 windows: the 1,976 eval windows (and the train windows of `latents_train.pt`) whose 16 frames are all real, as WMACT and the planner read. NOW = window frame g = t - 15; target frame = g + 10.
* Speed target: `native_tracks` column 3 at row 2 x frame. Yaw-rate target: cached measured yaw rate. Yaw-rate mask: speeds at the target frame and the frame before both above `MIN_DS_MPS` 0.5 (`four_families`' pair rule).
* Ridge: ||y - Xw - b||^2 + alpha ||w||^2 on train-standardised X; CV ties go to the smaller alpha; folds by clip, clips dealt round robin after a seed-0 shuffle.
* Interval: bootstrap resamples eval episodes with the probe fixed ("another draw of episodes" only).
* Control (i) shuffles train targets with `hjepa.donor_indices` (every row takes a target from another clip). Control (ii) planted value = var(z.w) / (var(z.w) + 0.01) on eval, w scaled to unit std on train.

## What could not be implemented as written, and what was done
1. **B2 on arms with strides (3, 5) (R35H, and by the same argv the SLOWWM arms): `wmact_lib.KS` = level 2 k 1, 3 and level 3 k 1, 3 cannot all be computed.** A 16-frame window holds too few level-3 states: L3/k3 does not exist (MEASURED, `ks_not_computable` in the smoke rows; M2b, strides 2 and 2, holds every cell). `stats_for` computes the cells that exist (L1 k1/5/10, L2 k1/3, L3 k1) and lists the others per arm with the reason. The statistics of the computed cells are `wmact_lib`'s, unchanged. Master Mind: say if L3/k3 should instead be read on longer windows (that would change the protocol).
2. **B1 train latents for real arms need the box-2 train clips.** Real arm rows need `decod_build_train.py` output. Until then only `--smoke` runs (half split of eval-141 windows, labelled in the row). Controls (i) and (ii) use that split too (they test the probe, not the arm).
3. The B2 control tolerance (banked rows vs `stats_for`) is 1e-6, not tighter: first read gave 8.1e-8 on rows and 3.0e-8 on sens (float32 latents, different thread state). Red arm differs by 17.0.
4. `clip_inputs` reproduces the banked eval `zmean` to 1.9e-6 (thread drift, 1, 2 and 4 threads all non-bit-equal); `act` and `ok` bit for bit.

## Controls (MEASURED 2026-11-11, `CONTROLS_B1B2.json`)
* B1 (i): permuted-target eval R2 -0.19 speed / -0.11 yaw (R35H s0), -0.02 / -0.03 (planted SLOW arm), all <= 0.02. Red arm (scored on train rows): 0.28 and 0.03, both > 0.02 (the planted-arm margin is thin, 0.032).
* B1 (ii): planted recovery 0.9910 vs 0.9928 (diff 0.0017) and 0.9889 vs 0.9902 (0.0013). Red arm (targets rolled by half the rows): misses by 1.65 and 1.04.
* B1 (iii): R35H s0 through `prepare_arm_read` + B1 code equals `lp_lib.prepare_arm` bit for bit: Zw, rolled z1, targets, masks, all 1,976 windows (max abs 0.0). Red: 1e-6 perturbation flagged, step k+1 differs, a basis model on lp_lib's clip_z1 raises.
* B2: M2b_s0 through the new collection = banked `latents.pt` Z1 bit for bit on 64 windows (red: M2b_s1 model differs, 1e-7 perturbation flagged). Port check: 141 banked row values reproduced within 8.1e-8, sens within 3.0e-8; controls 1 and 2 read 0.0. Planted basis arm: window z1 vs an independent float64 5.2e-6, other basis 8.4; control 4 (`forward_basis` vs `stack.loss`) 0.0.

## Smoke (MEASURED; scratch outputs, deleted; planted arm = R35H s0 checkpoint with wm_basis slow, planted basis and planted token means)
* B1, R35H s0 and planted SLOW s0, half split (996 train / 980 eval windows): wall 1.4 s and 1.5 s after load; R2 speed 0.518 (R35H, interval 0.02 to 0.76) and -0.021 (planted); yaw R2 0.530 and 0.731. Numbers are machinery output on random planted weights or a split, not a result.
* B2, full 2,399 windows, 1,976 used: R35H s0 32.8 s, planted SLOW s0 23.9 s (RAM-gated start). Control 4 max abs diff 0.0 for both.
* A real full B1 + B2 arm costs seconds to minutes on CPU; the train-clip stream (box 2) is the cost.

## Not verified
* No real SLOW / PCA-B checkpoint or basis, no real train frames: the train-side path (`train_ctx`, `decod_build_train.main`) is unexercised beyond the shared per-clip function.
* The ubasis-file vs live-u cross-check (B2, basis arm) ran only with `--no-ubasis-crosscheck` (the planted ubasis comes from planted token means).

## Cleanup log
Scratch `F:/TanitAD-scratch/slowwm_beside_build/b1b2/` (planted bases, a 46 MB planted checkpoint, smoke outputs, probes, pytest basetemp), 61 MB, deleted by literal path at the end.

# M-LAUNCH build notes (PREREG_V8F_LAUNCH_INTENT section 7 / 11)

Zero training, dev-box CPU (`CUDA_VISIBLE_DEVICES=-1`), read-only on every input, no git write, no box 2, no ssh.
Everything below is **MEASURED** (artifact `MLAUNCH.json`, byte-identical on a second run, sha256 in the manifest) at tier **T1-kin** (own executed state, logged camera; never "closed loop"), estimator **taniteval.ci paired episode-cluster bootstrap, B = 2000, seed 0, over the 49 episodes of the launch windows**. That estimator answers "would another draw of EPISODES say this?"; it is blind to training variance (`H-ESTIM-SEED-1`).

## Result (registered statistic, reading R1; negative = the car is slower than the human)

`M-LAUNCH = not moved` (all 9 control groups pass: a, a2, a3, b1, b2, b4, b5, c, d). 327 launch windows, 49 episodes, in every cell.

| cell (X vs Y) | X mean | Y mean | X - Y [95% CI] | separated | \|X\|-\|Y\| [95% CI] | cell moved |
|---|---|---|---|---|---|---|
| main s0 (A vs DAG-OACC) | -0.8400 | -0.8747 | +0.0348 [-0.0299, +0.0923] | no | -0.0348 [-0.0923, +0.0299] | no |
| main s1 | -1.0431 | -1.0713 | +0.0282 [-0.0247, +0.0836] | no | -0.0282 [-0.0836, +0.0247] | no |
| replicate s0 (A_rep vs REP) | -1.4064 | -1.4307 | +0.0243 [-0.0281, +0.0846] | no | -0.0243 [-0.0846, +0.0281] | no |
| replicate s1 | -1.0659 | -1.0514 | -0.0146 [-0.0607, +0.0286] | no | +0.0146 [-0.0286, +0.0607] | no |

Units m/s. §11: "moved" needs all four cells moved; none is. Committed next (§11, not mine to decide): **LAUNCH-INTENT-HI** (W_HI 4.17852, the pair design, its own PI go).

Robustness (all give "not moved"): R2 executed-chord speeds (`sensitivity`); the sign reading of "toward 0"; tick 1 alone; tick 2 alone (the replicate pair is not separated toward 0 at either tick, at either seed, so no tick reading can reach four cells).

Context, beside, decides nothing (`beside_training_seed_floor`): the only look at training variance this rig has. REP - DAG-OACC (same recipe, other training seed) at s0 is **-0.5560 [-0.6994, -0.3975]** and A_rep - A is **-0.5665 [-0.7346, -0.3839]**; at s1 +0.0199 [-0.1009, +0.1373] and -0.0229 [-0.1353, +0.0850]. The lever's effect (+0.01 to +0.03) is an order of magnitude below the s0 seed-to-seed spread, and a separated CI would have been necessary, not sufficient.
Post-hoc look, NOT registered, not a finding: at tick 1 alone the main pair is separated toward 0 at both seeds (+0.0409 [+0.0227, +0.0615], +0.0318 [+0.0112, +0.0530]); tick 2 and the replicate pair are not. Eight per-tick looks, no multiplicity control.

## Files
`mlaunch.py` (computation) | `MLAUNCH.json` | `mlaunch.out` (log of the final run) | `test_mlaunch.py` (12 tests) | `mutation.json` (12 of 12 mutants of the pure functions killed, unmutated control green) | `raw/mlaunch_windows.npz` (per-window v_own, v_GT, launch mask, eid, for audit) | `ref/` (verbatim pinned sources: `t1klc_read.py`, `statekin.py`, `launch_mask_excerpt.py`) | `mlaunch_first_run_b2_b4_failed.out` (see Design history).
Reproduce: `CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I mlaunch.py` (about 20 s). Every logic source is sha256-pinned in `PINS`; a changed source REFUSES.

## What was read, with file:line

Tool tree `E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a` (the read of record's `--tool-root`); registered commit `d7dc9f11` read with `git show` from `C:/Users/Admin/tanitad-push`.

* **PREREG** `Project Steering/PREREG_V8F_LAUNCH_INTENT.md` @ d7dc9f11: definition line 222; "along-track MAE, speed MAE and L2 ... M2" line 223; "moved" and routing lines 288-295; reading "moved only when it moves in both pairs at both seeds" line 28 (§0 item 8 (b)).
* **tick.** `stack/scripts/v8f_t1_eval.py` (sha 61e140ed...): `rollout` tick loop `for n in range(H // m)` :407, hook called :418 before the plan is committed; `TickRecorder` :324, `FIELDS` :337; row fields `tick`, `j`, `t0`, `ci`, `win` :365. m = 10 so tick n is the executed state at step j = t0 + 10 n, n = 0, 1, 2 (3 ticks per window, 7,197 rows = 2,399 x 3).
* **v_own.** Stored per tick as `v0`: `v0 = batch["p10f"][:, -1, 0]` :354, appended :363. The own state is written by `WindowState._finish` :201 from `D.yaw_kin_from_track` (`stack/tanitad/data/v8f_mini_data.py:285`, `v0 = v[2j]`) over the Hermite frames of `hermite_frames` :85 (speed = Catmull-Rom tangent norm / 0.2 s, so the state speed at the tick knot is |K[i+1] - K[i-1]| / 0.4 of the plan issued one tick earlier). Control d reproduces the stored value from the previous tick's plan to <= 1.9e-6 m/s (float32), with no window over 1e-4, in all 8 runs and both ticks (16 of 16).
* **v_GT.** No per-tick GT speed is stored; GT positions are (dump `g`, t0 frame; ticks `gt`, tick frame). v_GT is `statekin.state_from_knots` (`ref/statekin.py:36`, lcdiag @ d7dc9f11, which calls the tool's own `hermite_frames` and `yaw_kin_from_track`) on `[origin; g]` with `start = 10 (n - 1)`: the same state rule applied to the GT knots, as `probe6_state.py` (`kinG`, eval `kin["GT"]`) and `DIAGNOSIS.md:81-88` ("O's own v0 - GT") did. Control d: it equals the closed form |g[10n] - g[10n-2]| / 0.4 exactly (max |d| 0.0).
* **launch windows.** `hmc_read.py:386-388` @ d7dc9f11, exactly: `v0 < 6.0` and `|g[29] - g[28]| / 0.2 >= v0 + 1.0` (`hmc_lib.py:23` DT = 0.2), `v0` and `g` from the dump. The train-side twin `stack/tanitad/models/v8f_mini.py:396, 401-406` @ d7dc9f11 gives the same set.
* **loading and matching.** Ticks by the DAG-ORC rig's own `t1hd_read.load_ticks` (`.../raw/2026-10-26-t1hdread/t1hd_read.py:134-158`: indexes rows by `(win, tick)`, asserts no duplicate or missing row, `j == t0 + 10 tick`, tick-0 ages 0, no `fired`); dumps by `dag_read.load_dump` (`dagorc_rebase/read/ref/dag_read.py:1023`); both AST-exec'd from the pinned files. Window matching is the rig's DUMP-WINDOWS test (`dag_orc_read.py:1222`): dump `clip_index`/`ws` equal ticks `ci`/`t0` in order; `load_run` repeats that equality for all 8 runs (all True). The dump's per-window `v0` equals the ticks file's tick-0 `v0` exactly.
* **estimator.** `taniteval/taniteval/ci.py` (sha 38db571e...): `paired_episode_cluster_bootstrap` :275 (a callable reducer is allowed, :200), `episode_cluster_bootstrap` :225, `_draws` :158. Reducers: `mean`; and `abs_mean` for |X| - |Y| on the same resampled episodes.
* **beside rows.** `comps` (`t1k_read.py:37`, row definitions: `along_mae_m`, `speed_mae_mps`), `paired` :62, `lat_rows` (`ref/t1klc_read.py:30`, `L2_curvature_mae_fast_1pm`), exactly as `rr_lib.py` of the DAG-REREAD reader loads them.

## Readings I had to choose (please confirm or name another)

1. **"Ticks 1-2" = 0-based ticks 1 and 2**, the two ticks whose state is the model's own executed motion. Tick 0 is the logged state and is identical in every arm (control a2: X - Y at tick 0 is exactly 0.0 in all four cells). The per-window value is the mean over the two ticks (equal to the pooled mean of the 2 x 327 rows). Per-tick values are in `per_tick`.
2. **v_own** = the stored tick `v0` (the own state speed, a central difference at the tick knot); **v_GT** = the same rule on the GT knots (above). Sensitivity R2 uses the executed-chord speed over the last 0.2 s before the tick on `cl` and `g`. Both readings give "not moved".
3. **v_GT cancels in X - Y** (X and Y share the windows and the GT), so the registered X - Y and its CI do not depend on how v_GT is defined; only the "toward 0" test does. Control b1 shows the cancellation (X - Y unchanged to <= 8.9e-16 in all four cells under a shuffled v_GT, while each arm's own mean moves by about 7 m/s).
4. **"Toward 0"** = |mean X| < |mean Y| *with the paired CI of |X| - |Y| entirely below 0* (reducer `|mean|` on the same resampled episodes). **A cell is moved** iff X - Y is separated AND that |X| - |Y| CI is below 0. The sign-only alternative (X - Y separated and signed against Y's mean) is in the JSON as `moved_sign_reading` and agrees here.
5. **"Moved" overall** = all four cells (main/replicate x s0/s1) moved (§11, §0 item 8 (b)). A missing cell is not a pass.
6. **Unit and clustering:** the window, clustered by the dump's `eid`; the bootstrap runs over the launch windows' 49 episodes (not the 141 of the full grid).
7. **Launch set** from the dump `v0`/`g` (the `hmc_read` form), the same 327 windows in all 8 runs (verified, including `g`, `v0`, `eid`, `ws`). `gok` of the train-side mask is true for every eval window (the tool refuses a window without a complete 6 s future).
8. The PREREG notes "~327 windows / 49 episodes (INHERITED)". Measured: **327 / 49**, by four recounts.

## Controls (all PASS, `controls` in the JSON)

* **a** X = Y (every run against itself, both readings): exactly 0.0, CI [0.0, 0.0], not separated. **a2** tick 0, X - Y: exactly 0.0 in all four cells (proves the pair's windows are aligned). **a3** v_own := v_GT reads exactly 0.0 per window.
* **b1** v_GT shuffled across all 2,399 windows (seed 0): each arm's mean moves (main s0 X: -0.8400 -> -7.9289); X - Y does not.
  **b2** launch mask swapped with its complement (2,072 windows): each arm's mean leaves its launch CI (main s0 X: -0.8400 -> -0.0913).
  **b4** Y's windows permuted across all windows: X - Y leaves its CI (main s0: +0.0348 -> -7.6029).
  **b5** a known effect injected into X: +0.5 m/s toward 0 reads back as the actual X - Y + 0.5 to 1.1e-16 and the cell reads MOVED; -0.5 away from 0 reads back exactly, is separated, and does NOT read moved. This is the control that proves the §11 verdict can say "moved".
* **c** window count: dump vectorised (hmc_read form) 327 / 49; ticks-file tick-0 rows 327 / 49; `v8f_mini.launch_mask` (torch, verbatim excerpt) 327 / 49; plain Python loop 327 / 49; identical sets; identical in all 8 runs. Threshold margins: min |v0 - 6| = 8.9e-4 m/s, min |vG6 - (v0 + 1)| = 2.5e-3 m/s, against a 7.6e-6 tick-vs-dump GT difference, so no window sits on a float boundary.
* **d** closed forms reproduce the tool's own rebuild (v_GT exactly; v_own to float32).

## Design history (disclosure)

The first run of `mlaunch.py` (`mlaunch_first_run_b2_b4_failed.out`) printed the **same cell numbers and the same word** and then refused: my first b2 and b4 asked that the regressed **X - Y** leave the actual X - Y CI. That criterion cannot hold when X and Y are near-identical arms (X - Y is about 0 +/- 0.06 on every window set, and a one-window roll lands on an adjacent launch-like window). The instrument was not wrong; the control was badly posed. I changed b2 to judge each arm's own mean, b4 to a full permutation, and added b5 (a known effect, which is the right sensitivity control). The controls only gate the output; they do not enter any number, and this change was made after seeing the numbers, so I record it here.

## Beside, M2 (decides nothing): along-track MAE, speed MAE, L2 on the launch subset, X - Y

Executed track `cl` vs GT `g`, 30-knot horizon (not ticks 1-2), the reader's `comps`/`lat_rows`, same paired bootstrap. Lower is better. L2 is defined only where GT >= 5.2 m/s (253-255 windows, 45 episodes).

| cell | along MAE (m) | speed MAE (m/s) | L2 (1/m) |
|---|---|---|---|
| main s0 | -0.1200 [-0.1957, -0.0483] sep | -0.0141 [-0.0712, +0.0480] | -0.0505 [-0.1487, +0.0256] |
| main s1 | +0.0291 [-0.0805, +0.1437] | **+0.3824 [+0.1367, +0.6489] sep (worse)** | -0.0099 [-0.1049, +0.0854] |
| replicate s0 | -0.0003 [-0.1034, +0.0945] | -0.0319 [-0.1006, +0.0512] | -0.1880 [-0.3169, -0.0707] sep |
| replicate s1 | -0.0274 [-0.1112, +0.0566] | -0.0975 [-0.2103, -0.0040] sep | +0.0899 [-0.0911, +0.2936] |

Mixed in sign and cell: no M2 row is separated in the same direction in both pairs at the same seed. Not read as an effect (each is one seed, one pair, against the training-seed floor above).

## Limits

* T1-kin tier; the estimator resamples episodes only. "Not moved" means no cell separated at CI half-widths of about 0.06 m/s on X - Y; it does not prove the lever has no effect smaller than that, and the s0 training-seed spread (0.56 m/s) is far larger than any X - Y read here.
* The register (`GOALS_AND_CLAIMS.md`, `H-V8F-LAUNCH-INTENT-1`) is not updated here (outside my work dir, no git write); that is the Master Mind's integration step.

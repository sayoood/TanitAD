# B3: planner with the level-1 goal cost restricted to SLOW's top-64 mu directions

Protocol: `../BESIDE_RULES_SLOWWM.md` section B3 (frozen, sha256 `d222f9d4f360b3239fe3054a2d7c0da80032af873ade932455c0b03c8d4013d0`, re-hashed by a test). Code and tests only, CPU. Nothing was trained, nothing ran on box 2, nothing is staged, committed or pushed. `ladder_plantest/`, `ladder_read/`, `slowwm_hook.py`, `slow_basis.py` and `read/*` are imported and not edited. Tier: T1-kin planner form, estimator EC-BOOT (paired episode-cluster bootstrap, B 2,000, seed 0: it answers "another draw of episodes" only). No bar and no outcome label anywhere in this package.

No real SLOW checkpoint exists yet. Everything below was exercised on a PLANTED basis arm built as control b of `../read/controls_read_slowwm.py` builds one (R35H s0's checkpoint with `wm_basis = slow`, a planted basis, planted token means), with PLANTED train frames for J-bar. No number here says anything about SLOW.

## Files
| file | role |
|---|---|
| `top64_lib.py` | train-frame sampling, J-bar by autograd (and by central difference), S / P_S, beside numbers, the cost wrapper `restricted_cost`, `build_arm` (-> npz with input sha256), `verify_projector` |
| `top64_controls.py` -> `CONTROLS_B3.json` | controls (i)-(iii), a red arm each, the planted builder run; `all_pass` |
| `top64_run.py`, `top64_env.cmd` | the runner: the deciding read's unit structure (p1 / tune / score, c1 and c3, FLAT1 level 1), resumable per unit, SLOW arms only |
| `top64_report.py` -> `REPORT_B3.json` / `.md` | m beside the same arm's full-width m, rank(S), variance share, principal angles |
| `tests/test_top64.py` | 17 tests |
| `top64_run_SLOW_s0_smoke.status.json` | the smoke run's status (wall time) |

## What the rule leaves open, and what was chosen (all recorded in the npz)
1. **K = 3 windows.** z1 of the R35H-built arms is `enc1` over three consecutive projected frames, so dz1/du has one block per frame. J-bar differentiates z1 with respect to a shift of the chosen u columns applied to all three window frames together. The newest-frame-only variant is stored as `jbar_current`; the principal angles between the two subspaces are a beside number (`angles_joint_vs_current_deg`). The planner uses the joint-shift S.
2. **4,096 train frames**: ok = True frames of the train cache written by `slow_basis.py cache --split train` (`<dir>/<clip>.npz` with `zbasis`, `ok`, and `BASIS_MANIFEST.json`), drawn without replacement with `default_rng(0).choice` over the flattened ok-frame index in sorted clip-name order; window indices are clamped at the clip start. The manifest's basis sha256 must equal the arm's, and every clip file read is hashed against the manifest.
3. J-bar is computed in float64 on a deep copy of the arm's `wm`; the planner stays float32. `eval z1` for the beside numbers = `slowwm_hook.clip_z1_basis` over the 141 eval clips (all frames, centred).
4. Injection: `st.goal_cost` is set as an instance attribute on the arm's stack and deleted on exit; levels above 1 call the library cost. `verify_projector` refuses a projector whose basis sha or checkpoint sha256 differs from the arm being read.

## Controls (MEASURED 2026-11-11, `CONTROLS_B3.json`, `all_pass` true, 256 s of planner controls)
* **(i) Identity.** P = I through the wrapper: plan, cost1, init1 and iters1 bit-equal to the full-width planner on 64 windows (context 1, prior seed) and on 8 windows (context 3, hold seed, lambda 0.3); a third run after the wrapper exits is bit-equal as well. Red arm: one diagonal entry zeroed changes cost and plan.
* **(ii) Bottom-64.** S from columns 192-255 (rank 64) gives a plan different from the full-width plan and from the top-64 plan on the same 64 windows. Red arm: "bottom" columns mutated to the top columns gives the top-64 plan bit for bit, which fails the control. Note: on the planted arm the max-abs plan difference reads the same 3.4393 in every comparison (one saturated action channel at its box limit); the cost1 difference is what the red arm of (i) asserts. This control is shown on a planted arm with random weights.
* **(iii) Jacobian.** autograd vs central difference (step 1e-3, float64), relative Frobenius error 2.9e-9 (joint shift) and 2.0e-9 (newest frame), tolerance 1e-3; window form of z1 equals the whole-clip form to 8e-15. Red arms: autograd columns shifted by one (rel 1.42) and the newest-frame form against the joint difference (rel 1.60) both fail the tolerance.
* Builder on the planted arm: 4,096 planted frames, rank(S) 64, idempotence 7.8e-16, symmetry 0, input sha256 recomputed equal; 20.8 s.

## Run (real arms, once the chain has banked)
```
call top64_env.cmd
python top64_run.py 2 "SLOW_s0|<run>|<basis_slow.npz>|<ubasis_slow.npz>|<runs>/proj_SLOW_s0.npz" "SLOW_s1|..." "SLOW_rep_s2|..."
```
1. Build each projector first: `top64_lib.build_arm(name, run_dir, basis_npz, ubasis_npz, train_cache_dir, <runs>/proj_<name>.npz)` (about 20 s on the planted arm). 2. Run `top64_controls.py` (all_pass must read true; it needs only the R35H s0 pull). 3. Detach the runner through WMI (`cmd /c ... < NUL`); runs folder env `SLOWWM_B3_RUNS` (default `F:/TanitAD-scratch/slowwm/beside_top64_runs`); workers log to `<runs>/worker.log`. 4. `python top64_report.py` (needs the deciding read's units in `F:/TanitAD-scratch/slowwm/plan_runs` and its `meta_*` files). A PCA-B or R35H arm name is refused.

## Smoke and cost
* Smoke: planted SLOW_s0, first 64 windows of every unit (p1 x 2 goals, tune x 6 lambdas, score at the chosen lambda, contexts c1 and c3), 2 workers, `top64_run_SLOW_s0_smoke.status.json`: **1,406 s wall, 0 errors** (MEASURED). About 8 minutes of that overlapped four orphan worker processes of my own earlier mis-launched runs, killed at 18:31, so it is an upper bound.
* Per 64-window unit with two workers on this shared box (MEASURED, smoke log): c1 p1 about 50 s, tune about 43 s, score about 32 s; c3 p1 about 137 s, tune about 138 s, score about 118 s.
* Full cost per arm (ESTIMATED = unit counts of the planted arm's real windows: 1,976 / 982 tune / 994 score windows at c1, 1,835 / 912 / 923 at c3, times the unit times above, divided by 2 workers): about 1.1-1.3 h for c1 plus 3.3-3.8 h for c3, **about 4.5-5 h per arm, 13-15 h for the three arms**, CPU, 2 workers, while other jobs share the box.

## Mistakes found and fixed during the build
* `RL.run_stage` maps the deciding runner's own `work` (full width, all windows), so the first runner version would have written full-width units under a B3 label. Caught on the smoke (no unit finished in 13 minutes), fixed with our own `run_stage`, pinned by `test_the_pool_maps_this_modules_work_not_the_full_width_one`.

## Not verified until the real basis and runs exist
No real SLOW checkpoint, basis, train cache or ubasis was read. rank(S), the variance share and the angles on a trained arm are unknown. The report was tested on synthetic unit files only, never on planner output of a full window set.

## Cleanup log
Scratch `F:/TanitAD-scratch/slowwm_beside_build/b3/` (planted basis, train cache, a 46 MB planted checkpoint, projector, smoke units, logs, pytest basetemp) deleted at the end by literal path: `F:/TanitAD-scratch/slowwm_beside_build/b3` (70 MB, re-creatable by `top64_controls.py` and the smoke command). The parent folder `F:/TanitAD-scratch/slowwm_beside_build/` is left (empty, shared with other build folders). Final test run before the delete: 17 passed (28 s).

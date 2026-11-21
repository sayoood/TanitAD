# SLOWWM read package (card: PREREG_V8F_SLOWWM.md, frozen sha256 7c2775e3...94ca, re-hashed before building)

Code and tests only. Nothing was trained or launched on box 2; nothing is staged, committed or pushed. `ladder_plantest/`, `ladder_read/` and `slowwm/slowwm_hook.py` are imported and not edited (the hook needed no fix). Tier T0.

## Files
| file | role |
|---|---|
| `lp_slowwm.py` | `prepare_arm_read` (the one path every arm takes, R35H included), `load_checked` (all refusals), `basis_clip_z1` (dispatches `lp_lib.clip_z1` to the hook for a basis model), `meta_of` |
| `run_lp_slowwm.py`, `job_lp_slowwm.cmd`, `env_slowwm.cmd` | the ladder runner unit for unit (p1 / tune / score, c1 and c3, FLAT1, level 1) over the arms; resumes per unit; `meta` unit per arm |
| `make_ubasis.py` | builds the eval `ubasis` file of a basis (hook's `build_eval_basis_frames`) |
| `report_slowwm.py` | PLANNABLE per arm and seed, margin m, floor, the four outcomes -> `RESULT_SLOWWM.json` / `.md` |
| `ia0_slowwm.py` | IA-0 skill per level at 2.0 / 6.0 / 12 s, BESIDE -> `results/ia0_<arm>.json`, `IA0_SLOWWM.json` |
| `controls_read_slowwm.py` -> `CONTROLS_READ_SLOWWM.json` | controls a (adapter identity), b (basis route), c (mismatch), d (outcome logic), each with red arms |
| `tests/test_read_slowwm.py`, `tests/RESULT.log` | 17 tests, all passed (82 s) |

## Inputs (paths are the assumed layout; the chain's bank step decides them)
* Run folders pulled from box 2 (argv.txt, ckpt_last.pt, selfcheck.json): `C:/Users/Admin/t2v8f_pulls/slowwm_{SLOW_s0,PCA-B_s0,SLOW_s1,PCA-B_s1,SLOW_rep}`; R35H: `C:/Users/Admin/t2v8f_pulls/ladder_R35H_s{0,1}` (present for s0; check s1).
* Bases: `C:/Users/Admin/t2v8f_pulls/slowwm_bases/basis_{slow,pca}.npz` (from `slow_basis.py fit`).
* Eval `ubasis` files: `F:/TanitAD-scratch/slowwm/read_cache/ubasis_{slow,pca}.npz`.
* Eval frames `ladder_read/cache/frames_s{0,1}.npz` (exist); the seed-2 replicate reads the frames of its s0 init.
* R35H planner units: the banked ladder runs `F:/TanitAD-scratch/ladder_plantest/runs/*R35H*` (the report reads them as they are).

## Run order (CPU, 2 workers, `CUDA_VISIBLE_DEVICES=-1`, needs >= 8 GB free RAM and >= 6 GB free on C:; the runner waits on its 6 GB RAM guard and never kills)
```
source theory/ladder_plantest/pyrun.sh-style env, or use env_slowwm.cmd (PYTHONPATH joined with ';', tanitad asserted under F:/TanitAD-scratch/slowwm/tree)
1. python make_ubasis.py <basis_slow.npz> <frames_s0.npz> <cache_root> <native_tracks.pt> F:/TanitAD-scratch/slowwm/read_cache/ubasis_slow.npz     (and the same for pca)
2. python controls_read_slowwm.py              # a b c d; about 2 min; all_pass must read True
3. launch detached through WMI:  cmd /c job_lp_slowwm.cmd < NUL      # SLOWWM planner units + R35H window metadata; log run_lp_slowwm.log, marker PLAN_SLOWWM.done
4. python ia0_slowwm.py "SLOW_s0|<run_dir>|<basis>|<ubasis>" ... "R35H_s0|<ladder run>|" ...     # beside, resumable
5. python report_slowwm.py [--space action]    # RESULT_SLOWWM.json / .md; the outcome reads UNUSABLE unless CONTROLS_READ_SLOWWM.json says all_pass
```
The full command for step 3 is `cmd /c E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\slowwm\read\job_lp_slowwm.cmd < NUL`.

## Rules the code applies (verbatim from the card unless stated)
* Rows: G-TRUE minus HOLD and minus G-RAND, LONGITUDINAL and LATERAL, PLANTEST (all windows) and PLANTEST2 (SCORE half at the TUNE-chosen lambda) forms, contexts 1 and 3, EC-BOOT (B 2,000). PLANNABLE = `plantest_lib.family_label` reads PLANNABLE. TACTICAL n = 0 with its reason; STRATEGIC PAUSED (PI item 25).
* m = mean(HOLD) - mean(G-TRUE) (positive = better); floor = |m(SLOW_rep) - m(SLOW_s0)| per read, context and family; "beats beyond the floor" means a margin gap above the floor at both seeds.
* Outcomes evaluated per family, then per cell (read x context). The report states `CELLS-AGREE` (with the shared outcome set) or `CELLS-DISAGREE` and shows each cell. SLOW-COSTS can hold together with NO-EFFECT (the card gives it NO-EFFECT's next step).
* **Both interpretations are fixed by `../AMEND_SLOWWM_1.md`** (frozen 2026-11-11 14:20:08, sha256 `5d2270d4…1549`, before any basis or run):
  * A1: the deciding metrics are LONGITUDINAL = `speed_err` and LATERAL = `heading_err`; `acc_mae` / `yaw_mae` are beside. The deciding run uses the default `--space traj`; `--space action` is a beside read only.
  * A2: the four cells are {PLANTEST, PLANTEST2} × {c1, c3}.
  * A3 / A4: `report_slowwm.committed`. One single outcome shared by all four cells is the read's outcome. Otherwise the read is CELLS-DISAGREE, with the least favourable cell's next step: MIXED counts as its worst member, and NONE-OF-FOUR counts as NO-EFFECT.
  * Tested by five tests in `tests/test_read_slowwm.py`, including a red arm against a majority rule.

## Controls (MEASURED 2026-11-11, `CONTROLS_READ_SLOWWM.json`, all four pass)
* **a, adapter identity.** R35H s0 through `prepare_arm_read` (the basis-capable path): every ctx array bit-equal to `lp_lib.prepare_arm`; `unit3` on 8 windows bit-equal to lp_lib's; the first 64 windows (batch 0) bit-equal to the banked `p1_R35H_s0_c1_flat1_true.npz` (max abs diff 0.0, 1 thread). Red arms: a 1e-6 perturbation and a different lambda are flagged; an adapter model sent down the basis branch raises.
* **b, basis route.** A planted basis arm (R35H s0's checkpoint with config `wm_basis = slow`, a planted basis and planted token means): z1 equals the proj tail of an independent float64 `(nm - mean) @ W` within 1e-4 (measured 5.2e-6); the `nm_of` route on planted features equals float64 within 2.2e-7. Red arm: the other basis' expected z1 differs by 12.1.
* **c, mismatch.** Wrong sha, wrong kind, no readback, readback cache problems, a ubasis built with another basis or kind, an argv that declares a basis the BUILT model lacks (and the reverse), and a name/kind mismatch are each refused before any scoring; the matching dir is accepted.
* **d, outcome logic.** Planted tables give SLOW-PLANS, BASIS-ONLY, NO-EFFECT, SLOW-COSTS, NO-EFFECT for a one-seed-only gap, and no outcome for a not-plannable SLOW; INCOMPLETE when a margin is missing; UNUSABLE when a control failed or did not run. Three mutants of `decide` (seed 0 only, floor ignored, sign flipped) each change a planted outcome.
* End-to-end smoke: `ia0_slowwm.py --smoke` on the planted basis arm read all 141 clips in 34 s (L1 and L2 above Copy-Last on random weights, copy-vs-itself 0.0).

## Cost (ESTIMATED from the ladder's `smoke_time.json`: about 27 s per 64-window batch at c1, 99 s at c3)
Per arm: c1 has 1,976 windows (31 batches), c3 slightly fewer; p1 x 2 goals, 6 lambdas of tune on half the windows, score x 2 goals at the chosen lambda(s). Order of 1.5 to 3 h CPU per arm at 2 workers, so about 8 to 15 h for five arms. IA-0 about 5 min per arm (EGOWM: 313 s at 2 threads for IA-0 plus beside).

## Not verified until the real basis and runs exist
* No real SLOW / PCA-B checkpoint, basis or `ubasis` exists: `prepare_arm_read` ran on a planted checkpoint (random-init weights from the R35H file), never on a trained basis arm. The run-folder naming and basis / ubasis paths above are assumptions.
* The planner harness has not run a full 1,976-window unit on a basis arm; no full R35H re-run through this path (identity is shown on the first 64 windows and 8-window comparisons).
* R35H s1's pulled folder and its banked units are read, not re-derived; the report asserts the window set equals across arms.
* IA-0 on R35H through `ia0_slowwm.py` was not compared with the banked `arm_R35H_s0.json` (the dispatch is a pass-through, unit-tested on a stub); decodability, WMACT audit and the top-64 mu planner (beside rows) are not built here.

## Cleanup log
Scratch `F:/TanitAD-scratch/slowwm_read_build/` (planted bases, planted run folders with 46 MB checkpoints, pytest basetemp; 320 MB, re-creatable by the controls) deleted at the end of the build by literal path. Nothing else was created outside `theory/slowwm/read/`.

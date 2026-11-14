# HANDOFF: MOTION levers (level-1 predictor), 2026-11-14

State: DONE on the CPU. Nothing trained, no git, no pod. Folder `F:/TanitAD-scratch/v8proof/spatial/motion/`. Base = `spatial/cellpred/tree_patch` (CELLPRED result).
Apply: `cd <tree root>; patch -p1 --binary < MOTION.patch` on a tree holding the CELLPRED files (re-applied to the base copies here: byte-identical to `patched/`, cmp, 4 files). Bank `patched/` under `stack/`. Pins: `PINS.txt`.
Details and every reading choice: `MOTION_DEFS.md`. Raw results: `checks/` (json + logs), `tests.log`.

Flags (all default to the base tree, bit for bit): `--wm-l1-residual {off,on}`, `--wm-l1-roll-k K` with `--w-wm-roll W` ((0, 0.0) or (K >= 2, W > 0), K <= hmax - 1 = 15), `--wm-l1-diag {off,on}`.
Residual on = wrapper `ResidualPredictor` (state-dict keys `predictors.0.inner.*`), last layer zero-initialised. Roll loss: one call of the shared rollout core per start (starts 0, 2, ...; a batched version moved the result by 1.7e-6).
With K > 0 the trainer also logs `wm_roll` (= roll/L1); G-LIVE requires it.

Results (all read from the output files, CPU):
* m1 identity, 200 steps, 3 argv shapes (grid G, cell, mean default), fresh run dirs: keys, init tensors, all 28 logged fields, final weights, ckpt cfg keys: max abs difference 0.0, PASS.
* m2 residual at init, real sizes (flat 256 wide, cell 128 wide): output minus input 0.0 for L = 1, 2, 3; control (plain predictor) > 1e-3.
* m3 roll/L1 alone: every group of the predictor and the state projection non-zero; last frame 0.0 exactly, non-start frames 0.0; control (one-step loss) reads non-zero on the last frame.
* m4 rollout in the loss vs `HJEPAStack.rollout(1, ..., context=3)`, K = 10, flat/cell, residual off/on: max abs difference 0.0 (bar 1e-6).
* m5 float64: total(W) - total(0) = W x roll/L1 to 2e-14 (flat and cell); deleting the line that adds the term gives 0.0 (RED). The test guards the line `level_loss = level_loss + c.roll_weight * roll`.
* m6 selfcheck, 4 arms (default, A, B, A+B+cell+diag): declared-vs-built PASS, G-LIVE pass, right gate_extra expectations rc 0, a wrong one per lever rc 1.
* m7 cost (tiny synthetic model, CPU, noisy, 3 reps, s/step): flat base 0.21-0.27, K=10 0.40-0.52 (x1.9 mean, x1.9 best); cell base 0.58-0.82, K=10 3.73-3.76 (x5.2 mean, x6.5 best vs the quickest base rep). Residual alone x0.8-1.3 (noise). A pod run will see the same order for cell.
* Tests: `test_v8f_motion_levers.py` 32 tests; with `test_v8f_cell_predictor.py` (11) and `test_v8f_spatial_state.py` (12): 55 passed, 0 failed (`tests.log`). An earlier run had 1 failure (the old cell test replaces `probe_pred` with a stub): fixed by reading the residual lever from the built module in `built_levers`.

Not run: real data, pod-size model through the trainer, `v8f_beta.py` and `v8f_t1_eval.py` (unchanged; a residual or rollout checkpoint builds through `V8FMiniConfig(**ck["cfg"])`, tested by strict load).
Scratch (nothing deleted): `C:/Users/Admin/v8proof_c/motion/` about 1.8 GB, above the 1.5 GB ask because deletes are refused here (`tmp/` run dirs, `tree_base`, `tree_patch`). Safe to remove once the patch is banked.

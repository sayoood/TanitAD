# HANDOFF: KAPPA-T1 (H-KAPPA-T1-1), 2026-11-02 ~16:00. Rollouts running detached; the read is wired into the chain.

Package `E:/Projects/TanitAD-artifacts/dagland/kappa_t1/`. Scratch `F:/TanitAD-scratch/kappa_t1/` (trees `tree_dag`, `tree_li` = SMOOTH-T1's trees + the patched tool; `out/` = rollouts).

## Done
* Card `PREREG_FROZEN.md` frozen (sha256 `9564d568...`, `FROZEN_SHA.txt`, 15:44:58 +0100) before any number.
* Gate G-KAPPA run: all known answers PASS. **Lever (a)** selected (s_HIGHK 0.838 / 0.805 at DAG-OACC s0 / s1). See `GATE_RESULT.md` and `raw/GATE_KAPPA.json`.
* Tool `code/v8f_t1_eval.py` (sha256 `260bd175...`) = the SMOOTH-T1 tool (`caa71055...`) + `--feasible {off,box}` (`code/apply_feasible_patch.py`, diff `code/v8f_t1_eval.feasible.diff`). It is installed in both scratch trees; the chain checks the sha.
* Tests on the patched tool (in `F:/TanitAD-scratch/kappa_t1/tree_dag/stack/tests/`):
  * `tests/test_v8f_t1_feasible.py`: 7 passed. It covers off returns the same object, the round-trip, the planted plan inside the box, refusals, off bit-identical to the flag absent AND to the SMOOTH-T1 tool (with and without `--smooth`), reach, and the mutation (deleting the call turns RED).
  * `test_v8f_t1_smooth.py`: 10 passed.
  * `test_v8f_t1`, `_causal`, `_heads`: 48 passed, 1 skipped.
  * Logs are in `logs/pytest_*.log`.
* Chain `kappa_t1_chain.sh`, launched via WMI (`kappa_t1_chain.cmd`, pid 27724) at 15:55:54. It makes 8 runs, one at a time at 16 threads: `SREP_*` = Y (S rerun), `FEAS_*` = X (S + box), DAG-OACC and SOUP at s0 / s1, each ~150 s. Then `code/kappa_t1_read.py` runs automatically and writes `raw/KAPPA_T1.json` and `logs/read.out`.

## Next step (fresh agent, Sonnet)
1. `tail logs/chain.out`. You want `ZZKAP-CHAIN-DONE` and `ZZKAP-READ rc=0`. On a FAIL, read `logs/run_<tag>.out`. To resume, relaunch `kappa_t1_chain.cmd` through WMI `Win32_Process.Create`; finished runs are skipped.
2. If the read returned 3, KA1 or KA2 failed. `raw/KAPPA_T1.json["controls"]` says which. KA1 (Y == `out_m16/SMOOTH_*` at 16 threads, <= 1e-4 m) failing means thread or tree drift. State it and do not read rows.
3. Write `RESULTS.md` from `raw/KAPPA_T1.json`:
   * the outcome per arm (DOMINANT / HARMS / MOVED / NO EFFECT, the card's precedence);
   * per seed, the X - P10 / X - Y / Y - P10 rows for all four families plus ADE6 and M1, with the estimator (paired episode-cluster bootstrap, B 2000, seed 0, EPISODES only) and the tier (T1-kin);
   * the number of separated-worse rows beside HARMS.
4. Rule Zero: launch the card's next arm for that outcome IN THE SAME TURN. All of them are zero-training flags of the same tool and need their own frozen card first:
   * HARMS => the kappa-only projection (`a_max` = inf).
   * MOVED => the gate's next-ranked lever composed on top: DIP => (b), the accel-bounded longitudinal decode with `clamp_entry` and v0. (b) is NOT implemented: the planner has no v0, but the TickRecorder reads `batch["p10f"][:, -1, 0]`, and the same works inside `model_planner`.
   * NO EFFECT => (c), the lateral shrink, alpha on TRAIN only. **Blocker:** the tool rolls out eval-141 only (`--eval-grid`/`--eval-join`). A TRAIN grid + join + cache is needed. UNVERIFIED whether DAgger's train-window rollout inputs exist on disk.
   * DOMINANT => re-read on a fresh seed or window set; the training lever is the PI's go.
5. Update `MANIFEST.txt` (add `raw/KAPPA_T1.json`, `raw/arrays/*`, `RESULTS.md`, logs) and hand back in 300 words or fewer.

## For the Master Mind (integration, PI)
* Landing: `--feasible` stacks on the unlanded SMOOTH-T1 tool. Land it after SMOOTH-T1's decision 2 (overwrite vs separate tool file), and move the sha pin in `tests/test_v8f_t1_feasible.py` (`caa71055`) to that landed blob.
* Register the claim row H-KAPPA-T1-1. The gate result alone is MEASURED, decides the lever and claims nothing.
* This card runs everything at 16 threads and compares only runs made here or in `out_m16`. SMOOTH-T1's thread question is untouched.

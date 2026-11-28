# HANDOFF: SMOOTH-T1 (final state, 2026-11-02 ~15:35; replaces the 14:48 handoff text, which described an untested reader)

Package `E:/Projects/TanitAD-artifacts/dagland/smooth_t1/`, scratch `F:/TanitAD-scratch/smooth_t1/` (regenerable). Deliverable DONE except two decisions that are not mine.

## State
* Card `PREREG_FROZEN.md` (sha256 `4030d22f...`) untouched. Five rollouts finished; reader fixed (six changes, `READER_FIXES.md`, none touches the rule / rows / bars / estimator) and run: `raw/SMOOTH_T1.json` (final run `logs/read5.out`), tables `raw/RESULTS_TABLES.md`, write-up `RESULTS.md`.
* **Outcome: REFUSED by the card's own control rule** (K1, K2 and the K5 floor clause fail under 1-thread vs banked 16-thread float32 noise; `--smooth off` is bit-identical to the unpatched tip tool at equal threads, `raw/K1_DIAG.json`). Descriptive reading: DAG-OACC NO EFFECT, SOUP MOVED. Thread replicate (`raw/THREADREP.json`): no row flips, same outcome words.
* Tests: `test_v8f_t1_smooth` 10, `test_v8f_t1` 23, `_causal` 12, `_heads` 14 passed on the patched tool; `test_v8f_t1_speedsub` 5 passed / 28 refused by its sha pin `61e140ed...` (33 passed on the unpatched tool).

## Open decisions (Master Mind / PI)
1. Thread count: lift the 4-thread cap for 16-thread reruns (`k1_matched16_chain.sh` ready for K1, NOT run; then the four sg72 runs via `smooth_t1_chain.sh` with `SMT_THREADS=16` into a fresh out dir), or amend K1 / K2 / K5-floor (dated, post-data).
2. Landing: overwrite `stack/scripts/v8f_t1_eval.py` and move the pin at `stack/scripts/v8f_t1_speedsub.py:44`, or land the patch as a separate tool file. Also land `tests/test_v8f_t1_smooth.py` into `stack/tests/`, the package into `.../2026-10-17-v8f-mini/raw/2026-11-02-smooth-t1/`, and add the claim row H-SMOOTH-T1-1.
3. Next arm per the card (both outcomes): curvature-bounded decode (`stack/tanitad/refs/feasible_decode.py`, KAPPA_MAX 0.2) as a flag of the same tool, composed with S for SOUP; needs its own pre-registration frozen before data. Post-hoc seam diagnostic (`raw/SEAM_DIAG.json`) supports a per-knot decode lever: the model's seam bump falls to about P10's under S, a near-uniform excess over P10 remains.

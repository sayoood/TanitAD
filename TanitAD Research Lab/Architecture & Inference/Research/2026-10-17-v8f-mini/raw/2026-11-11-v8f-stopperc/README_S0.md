# STOPPERC stage 0: implementation notes (S0-PRE, S0-C, S0-M)

Card `PREREG_V8F_STOPPERC.md` (frozen, sha256 `70c75176…8877`), amendment `AMENDMENT_1_S0.md` (sha256 `6c4ad83b…d123`). Both are pinned in `test_stopperc_s0.py`. Tier T1-kin first tick, eval-141, zero training. CPU, 2 threads. Read tree `F:/TanitAD-scratch/stopbal_read/tree`.

## Files
* `stopperc_s0.py`: the S0-C and S0-M reads. Stage `run` banks one npz per arm and read to `F:/TanitAD-scratch/stopperc_s0/`; stage `analyse` writes `RESULT_S0.json`.
* `test_stopperc_s0.py`: 8 tests. They cover the card and amendment hashes, k(c), the ranking (ties, set mask, k above the set refused), exact mixing, the clears rule (floor, separation, side) and the outcome table (FAR, NEAR, the 0.7 edge, none).
* `RESULT_S0.json`: every cell, control and the S0-C outcome label.
* `s0p/`: the S0-P probe pipeline (separate hand-back).

## S0-PRE (MEASURED 2026-11-11)
* Box 2 train cache `D:\tanitad\cache_full\train\clips`: 4,572 files; the first 3 in sorted order are 20,904,597 bytes each with `meta.grids` = ['4x10', '8x20'].
* Local eval-141 cache `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/clips`: 141 files; the first 3 in sorted order hold `pool8x20` fp8 [101, 160, 1024] beside `pool4x10` and `mean`.
* ⇒ Arm R needs no re-encode, so Q2's re-encode question does not arise.

## Implementation choices (where the card is silent)
* The model pass is `stoporc_read.run_arm`, imported and not edited. Its `MODES` hook is narrowed to (fgt, mutant) for S0-C and (mutant,) for S0-M. For S0-M, `speedsrc_read.tag_paths` is wrapped so that the arm's `limit` entry reads `off`. The wrapper asserts the dump's own mode is `causal` before it replaces it (AMENDMENT_1_S0.md).
* S0-C's mixed plan takes each window's plan from the pass that computed it. A window's plan depends only on its own code and batch row: IDENT = OWN bit for bit is the check.
* The descriptive k-scan (k = 0…117, point only, no CI) sits beside the committed grid and decides nothing.
* S0-M reads STOPCUE's statistic on the STOPDEC npz windows (`has`, positive = `stop_hit`). It pairs the `off` p with the causal p of the same npz, with the difference taken on the same episode resample.

## Scratch
`F:/TanitAD-scratch/stopperc_s0/` holds four per-arm npz files, re-creatable by `--stage run`. They are deleted once the verdict lands; their sha256 values are in `RESULT_S0.json` `banked`. The 64-window smoke folder (580 KB) and its two logs were deleted on 2026-11-11 at 23:02.

# Stage 0 ruling, addendum 2 (Master Mind, 2026-11-09): run stopped, dump made resumable

**No Stage 0 result exists.** `scores_step0_base.npz` and `stage0.json` were never written, so nothing was read. No bar, input or estimator of `PREREG_SCORER1` changes. `H-SCORER1-1` is unchanged.

## What happened
The dump-only run took the dev-box GPU lock at 20:42:46Z on 2026-11-08 (RTX 4060, 8,188 MiB). The Master Mind's estimate of about 22:10Z was missed. The v8 flagship session asked at about 23:05Z, under the PI's "labels go first" ruling, for the run to be stopped if it needed more than about 30 minutes or if that could not be told.

MEASURED before the stop:
* Both DataLoader workers read 0 bytes between 23:07:06Z and 23:13:52Z (counters 884,934,710 and 896,898,428), and their CPU time was flat. The trainer ran at 100 % of one core and the GPU at 100 %, with 7.8 to 7.9 GB allocated.
* `dump.log` has no line after `start` at 20:45:44Z; `evaluate()` logs no progress and writes scores only at the end.

ESTIMATED: the workers had read 1.78 GB including imports, against about 2.57 GB needed (3,137 rows, 4 frames each, 205 kB mean frame). So at most about 69 % of the data had been loaded after about 2.5 hours, and at least about an hour remained.

The trainer (PID 18728) was killed by PID at about 23:15Z. The chain's own `finally` released the lock and `labels_first` took it. `markers/stage0.FAILED` records the stop.

Cause: UNVERIFIED. HYPOTHESIS: GPU memory close to full at batch 16, with Windows paging.

## What changes for the next run (scratch copy only, all behind `--dump-step0-only`)
`dump_resumable.diff`, applied by `patch_resumable.py`:
1. A `dump_progress` line after batch 1 and every 5 batches: rows done, elapsed time, GPU memory.
2. A crash-safe partial save every 25 batches, and resume from it after checking that the restored rows match the held-out order.
3. `--dump-batch`, set to 8 for this run (16 in `evaluate()` otherwise). The trainer's training batch here is also 8.
4. Scores only: the per-batch metrics and the bootstrap tables are skipped. Stage 0 reads only `scores_step0_base.npz`.

**Batch 8 instead of 16 is a difference from the pod's evaluate path.** Scores can differ at the size of a bf16 rounding step between batch compositions; the result must say that batch 8 was used.

`test_resumable.py` (CPU): a dump interrupted and restored writes the same arrays as one that never stopped; the deliberate-regression arm that drops one row differs. **Not tested:** the changed `evaluate()` end to end on the GPU, because the GPU is held for the label pass. Its first `dump_progress` lines must be read before the run is left alone.

## Where it runs next
After the label pass and behind the P6 read on the dev box, or on the L40 once P7 has ended and before any SCORER1 arm. Stage 0 still has to be read before the pod arms launch (Z2 means no pod arms).

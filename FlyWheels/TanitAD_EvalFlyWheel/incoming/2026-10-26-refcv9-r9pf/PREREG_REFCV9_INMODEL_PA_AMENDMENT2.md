# PREREG_REFCV9_INMODEL_PA_AMENDMENT2 -- K-EQ made DEVICE-MATCHED, written AFTER K-EQ FAILED (Master Mind, sealed 2026-10-26 16:20:56 Berlin)

Amends PREREG_REFCV9_INMODEL_PA (67d17860...) + AMENDMENT1 (1c25a064...). **Disclosed as post-failure.** The Phase A verdict as run is
**VOID (K-EQ FAIL)** (`result/R9PF_PA_VERDICT.json` 8c6f3c46...) and that verdict is NOT overwritten; this amendment produces a SECOND,
labelled reading.

## What failed, MEASURED
K-EQ (AMENDMENT1 s3, same-input form): decisions agree on 515 / 515 windows at both inference seeds, but max |d| = 3.05e-5 m (s0) /
4.58e-5 m (s1), exceeding 2 x ulp_float32 on 28 / 24 of 515 windows. The in-forward head runs on CUDA; the offline composition runs on
CPU. K-OFF PASSED (515 windows, both seeds).

## Why this is a control-DESIGN fix, not a tolerance move
K-EQ exists to prove that the offline composition the bars use is the SAME FUNCTION as the in-forward path. A CUDA-vs-CPU comparison
mixes that question with device arithmetic (different reduction order / fused kernels). The tolerance is NOT changed.

## Change
1. **K-EQ (amended):** the in-forward roll is re-run with the model on **CPU** (same tree, same flags, same windows: >= 500, both inference
   seeds) and compared with the CPU offline composition on the forward's own selected path: max |d| <= 2 x ulp_float32(max |coordinate|)
   per window AND 100 % decision agreement. **If it FAILS, the VOID verdict stands** and the composition code is defective.
2. **Beside, always:** the CUDA-vs-CPU residual (max |d|, windows over 2 ulp, decision agreement), so a deployment on CUDA is described by
   the device it would run on.
3. **Bars, arms, population, estimator: unchanged.** If K-EQ passes under (1), the Phase A word is read from the already-computed bars
   (`result/R9PF_PA_BARS.json` 497a551d...) and labelled "K-EQ under AMENDMENT2 (post-failure, device-matched)". PREREG_R9PF_LH's s2
   precondition is then satisfied in the same labelled sense, and R9PF_LH runs.

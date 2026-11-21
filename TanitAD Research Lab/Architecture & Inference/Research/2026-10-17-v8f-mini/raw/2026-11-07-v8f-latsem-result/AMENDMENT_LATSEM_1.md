# AMENDMENT_LATSEM_1: control 1's known value for z1 was mis-specified (Master Mind, 2026-11-07, written after run 1 read INVALID and before any LATSEM statistic was looked at)

## What happened
* `PREREG_V8F_LATSEM.md` (frozen prefix 4,882 bytes, `0b0c2f73`) control 1 commits: every copy of one (episode, g) has identical z1 and raw actions, max abs difference 0.
* Run 1 (`latsem/run_latsem.run1.log`) read INVALID on control 1 alone: max abs difference 1.19e-6. Only the control values and the word INVALID were printed. No R1, R2 or R3 statistic was looked at.
* Located (a separate probe, stdout only): raw actions differ by exactly 0.0. z1 differs on 32 of 50,808 copies, by at most 9.5e-7 (s0) and 1.2e-6 (s1). |z1| reaches 4.92, and float32 eps times 4.92 is 5.9e-7, so the differences are one to two float32 rounding steps. `collect_latents` encoded the windows in batches of 64, and a CPU matmul's accumulation order depends on the batch, so one frame encoded in two batches can differ in its last bits.

## The amendment
* Control 1 reads: raw actions max abs difference exactly 0; z1 max abs difference at most 1e-5, the tolerance control 2 already uses for the same per-frame quantity.
* Nothing else changes: no statistic, bar, outcome, estimator or picture rule. Run 2 recomputes everything from the same inputs.

## Log (dated, append-only after the freeze)
* 2026-11-07 10:03:08 (dev-box clock, from `date`), Master Mind. **Frozen:** the first 1375 bytes above this log, sha256 `ddb07ff809b5060e83b043b07e645ce0965a5f464dcde474c886155696d9580a`, before any LATSEM statistic was looked at.

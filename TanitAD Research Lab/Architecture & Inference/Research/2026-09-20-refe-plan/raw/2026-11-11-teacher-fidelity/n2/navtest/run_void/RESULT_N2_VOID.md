# TF-N2: VOID at control C-R0. No N2 navtest score exists

2026-11-12, Master Mind. Card `../PREREG_TF_N2.md` (sha256 `15913173…`, frozen 14:22:36Z), K ledger n = 43. The entry stays consumed: a void read is still a read.

## What happened
- **Attempt 1** (14:23Z) stopped at `roll_R0` with 0 rows.
  - The cause: `teacher_navtest_seam.py:48` hardcodes the navtest export at `E:/Archive/...`, but the archive now lives on F:.
  - The fix leaves the pinned code untouched: an 8,188,529-byte copy of the export at the old path, sha256 `b0dc05f3…`, equal to the source. The copy is deleted after TF-N2.
- **Attempt 2** (14:34Z, the same command): `roll_R0` completed, 240 / 240 rows ok.
  - **C-R0 FAILED**: the re-rolled R0 (default knobs) against the banked navtest teacher rollouts (2026-10-05) exceeds 1e-6 on **111 / 240 tokens**, max **0.0183 m**, max heading **0.0077 rad** (`ctl_R0.json`).
  - Per the card the chain STOPPED. `roll_RJ5` never started, and no N2-R, N2-C or N2-W plan or PDMS exists.

## Contrast
On navtrain the same R0 re-roll reproduced the banked TF-2b rollouts at **0.0 m** on 469 / 469 tokens. Those were rolled recently, on this machine, with the same code. The navtest bank is older (2026-10-05).

The banked rows carry route and goal fields (`goal`, `corrected_route_n`, `raw_route_n`) that the fresh rows do not. So the drift may sit in route or goal construction, or in the code or environment version, not only in floating-point noise. This is UNDIAGNOSED.

## Lesson (the MM's own)
A cross-time bit-identity bar (1e-6) on an iterative rollout was frozen hours after the MM recorded, in memory, that such bars fail across environments. The card should have measured the navtest drift (a few tokens, no scoring) before freezing the bar.

## Next lever (Rule Zero)
Diagnose the R0 drift against the bank: fields, code version at bank time, environment.
- **If identity can be restored**, rerun as an amendment (new K entry) with the 1e-6 bar kept.
- **If it cannot**, the amendment rolls R0 AND RJ5 fresh in one environment, so the comparison is paired. N1-C is re-derived from the fresh R0. The drift against the bank is reported as descriptive.

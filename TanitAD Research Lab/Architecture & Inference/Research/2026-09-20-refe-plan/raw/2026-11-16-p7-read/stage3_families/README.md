# P7 stage-3 families, recomputed after the families6 MemoryError (MM, 2026-11-17)

## What failed
- The P7 read's stage 3 (`stage3_full.py`, 2026-11-16 18:07Z) exited rc 0, but its families6 call raised `numpy._ArrayMemoryError` (`stage3/p7_full_families.log`).
- As a result, `r2_result.json` reads `families_vs_base` / `families_vs_also` = "missing: NOT COMPUTED (never a pass)".

## Root cause (from source, `eval/harness/ev6e/families6.py:72`)
- The line `[z["poses"][idx[t]] for t in toks]` runs on a numpy `NpzFile`.
- Each `z["poses"]` re-reads the whole array, and the row view keeps that copy alive.
- On navtest that is 12,146 x 1.11 MiB, about 13.2 GB of live copies. The dev box had about 13.8 GB of commit free, shared with other jobs.

## The fix: no pinned file edited
- `families6.py` is sha256-pinned (`eval/harness_pins.py`) and cited by the SFT-4 / C2 / P6 / P7 results.
- `families6_lowmem.py` loads the pinned file unchanged and swaps only its `np.load`, for one that reads each array once.
- `check_families6_lowmem.py` is the identity check. On 300 P7 tokens the wrapper output equals the pinned output key for key; a +1 m regression arm differs. Result: `identity/identity_result.json` = PASS.
  - families6 blob: `59345cc3` (git), equal to the tip's.

## Results (MEASURED; T1-family; families6's own per-model intervals, n_boot 2000; unpaired non-overlap, REPORTED NOT GATING)
- `run/p7_full_families.json`: P7 on 12,146 tokens over 136 logs. Longitudinal, lateral and tactical all OK; strategic UNAVAILABLE in families6 (see the new `strategic_family.py`).
- `run/p7_families_vs.json`: `PE.fam_adverse`, exactly stage 3's call.
  - **vs P6:** 0 adverse of 21 components. The base is `p6_full_families.json`, md5 `b9dcbf62`, which equals stage3_full's pin.
  - **vs P2:** 1 adverse of 21. Longitudinal `ego_progress.progress_ratio_mean`, direction closer to 1: P2 CI [1.0055, 1.0418] vs P7 [1.0658, 1.1007]. P7's plans run further than the logged route.
- Tactical, which has no interval (accuracy / kappa, P6 -> P7):
  - lateral decision 0.9131 -> 0.9133;
  - longitudinal decision 0.6481 -> 0.6543;
  - 5-way collapsed 0.7537 -> 0.7583.

## For future stage-3 reads
Run families through `families6_lowmem.py` on a memory-tight box, or give the box more than 14 GB of free commit.

<title>PREREG TF-N5 A1</title>

# PREREG TF-N5 Amendment A1: rerun TF-N5 after its build-stage crash (merge out of memory), with the merge loading each npz once

**Status: FROZEN by the MM 2026-11-13 (sha256 in `PREREG_TF_N5_A1.sha256`), before the rerun.** It amends `PREREG_TF_N5.md` (sha256 `7411444c…`, frozen 2026-11-13, K n = 47). Everything not named here is unchanged: question, population, arms (N5-A, N5-X, N5-R builder-only, N4-D comparator), L1 and its frozen parameters, the residual split `inputs/n5/residual25.json`, scorer of record and full-seam scoring, every bar of section 6, every control of section 7, the estimator, the four-family guard, the naming rule.

## 1. What happened (MEASURED, `n5run_void/chain.log`, `n5run_void/build/merge.log`)
- The TF-N5 chain launched 2026-11-13 10:20Z.
  - Guards PASS (card `7411444c`, K 47, 956 pins, reach PASS, tip `families_run.py`).
  - `reach_real` DONE.
  - `derive` DONE 10:37:54Z (rc 0, 0).
- `build` FAILED at 10:38:50Z (`merge_rc` 1, `ZZN5_CHAIN STOP`). The merge raised `numpy.core._exceptions._ArrayMemoryError: Unable to allocate 1.11 MiB for an array with shape (291504,)` at `n5_nt_build.py:438`.
- **No control was computed and no N5 count or number exists.** The merge raised before its build report was written.
  - The derive stage's per-token rows (`n5run_void/build/derive_s*of2.jsonl`) exist. **Nobody has read them.** They are banked unread as void evidence.
  - No harness score of any N5 arm exists.
- K entry n = 47 stays counted.

## 2. Diagnosis (MEASURED, by reading the code; reproduced by the new test)
- `cmd_merge` and `check_unch` index NumPy `NpzFile` objects per token inside loops, e.g. `sb["poses"][i]` for each of the 12,146 tokens.
- An `NpzFile` re-reads and decompresses the whole member on every `[key]` access. Where the per-token slice is KEPT (the seam assembly at `:438`), each kept row is a view that keeps its full 1.1 MiB copy alive. That is 12,146 × 1.1 MiB ≈ 13 GiB, which exhausted virtual memory.
- **Why the 53 chain tests did not see it:** their fixtures have 8 tokens. The defect scales with N.
- **Class:** an instrument whose cost scales with N, exercised only at toy N. It is the same family as "AN ARTIFACT'S COST IS THE COST OF THE FILE ITS…" in `TRAPS.md`.

## 3. The amendment
1. **`n5_nt_build.py`:** a helper `_npz(path)` loads every member of an npz ONCE into a dict of arrays. The merge's and C-UNCH's six `np.load` sites use it.
   - The derive stage's `np.load` (`:243`, which ran fine) is unchanged.
   - No computation changes: the same arrays are read, only once each.
2. **`test_n5_chain.py`:** a new test counts `NpzFile` member reads during a merge and requires at most one read per member.
   - Its mutation arm (the pre-fix per-token access) must exceed that count, and it does.
   - Tests: **62 passed** (53 + 1 chain, 8 L1), MM 2026-11-13.
3. **Pins re-made:** `n5/N5_PINS.json` sha256 `73cf6cc89f6314586928c9c788a22378ce42651efd2b0cd62ff137f7c75acebb`, 956 files, tip `af48b101`.
   - Against the void's pins, exactly `n5/n5_nt_build.py` and `n5/test_n5_chain.py` changed.
   - The metric-cache listing is unchanged (`2ce6c713…`, 12,146 files).
4. **Fresh run:** the void outputs move to `n5run_void/`; the rerun starts from guards with no reused stage.

## 4. Disclosure: what was seen before this amendment
- Nothing about N5's outcome. The chain log shows only stage names, return codes and timings. The traceback names a line and an array shape, not a value.

## 5. K
- This read is **K n = 48**.
- Ledger key `TF-N5:<sha256 of this file>:E3`, nominee `TF-N5-A1`. It amends TF-N5 (n = 47, void at build, still counted).
- Bonferroni intervals use K = 48. The chain runs with `--card <this file> --k 48`.

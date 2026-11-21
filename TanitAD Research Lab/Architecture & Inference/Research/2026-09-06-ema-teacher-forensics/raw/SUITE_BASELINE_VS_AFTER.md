# The suite, as a CONTROLLED comparison

**2026-09-06 - ArchInf FlyWheel.** The baseline is NOT green, so a bare "suite green" would be inadmissible. The verdict is the **failure-ID SET DIFF in both directions**, not the counts.

## The two trees

| | tree | state |
|---|---|---|
| **BASELINE** | `C:\Users\Admin\tanitad-v7fbudget` | HEAD state, untouched by me. **Blob-verified against git HEAD before use**: `v6.py` `e9d733d0...`, `train_v6_staged.py` `46c4db1d...`, `predictor.py` `73dac418...` |
| **AFTER** | `C:\Users\Admin\tanitad-emaforensics` | a copy of that tree plus this turn's edits |

Two SEPARATE trees, so the control could not be contaminated by editing the tree it ran in. Identical invocation in both:

```
cd stack && PYTHONPATH=<tree>/stack OMP_NUM_THREADS=4 PYTHONUTF8=1 \
  python -m pytest -q -p no:cacheprovider --continue-on-collection-errors
```

## The result

| | failed | passed | skipped | xfailed | errors | distinct FAILED/ERROR ids | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BASELINE** | 65 | 6374 | 121 | 2 | 51 | 116 | 868 s |
| **AFTER** | 65 | 6395 | 121 | 2 | 51 | 116 | 860 s |

### Failure-ID set diff, BOTH directions

* **NEW failures in AFTER (`comm -13`): 0**
* **Newly FIXED (in BASELINE, not in AFTER) (`comm -23`): 0**

=> **EMPTY IN BOTH DIRECTIONS. ZERO REGRESSIONS.**

### Accounting for the passed delta

`passed` moves **6374 -> 6395 = +21**. This turn ADDS two test files: `test_declared_freeze_preflight.py` (**12** tests) and `test_horizons_refusal_is_wired.py` (**9**) = **21**. The delta matches exactly, so no existing test was quietly deleted or skipped to make the numbers work.

### Raw tails

```
BASELINE: 65 failed, 6374 passed, 121 skipped, 2 xfailed, 10 warnings, 51 errors in 868.22s (0:14:28)
AFTER:    65 failed, 6395 passed, 121 skipped, 2 xfailed, 10 warnings, 51 errors in 860.44s (0:14:20)
```

Raw logs are not banked (they are ~235 and ~236 KB of dots); the FAILED/ERROR id sets above are the decision-grade content.

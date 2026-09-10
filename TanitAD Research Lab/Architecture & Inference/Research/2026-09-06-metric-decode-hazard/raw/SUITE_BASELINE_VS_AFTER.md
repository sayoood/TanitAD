# The suite as a CONTROLLED comparison — metric-decode refusal

Two separate trees, so the control could not be contaminated by editing the tree it ran in.

| | tree | state |
|---|---|---|
| **BASELINE** | `C:\Users\Admin\tanitad-mdhaz-base` | blob-verified against git `HEAD` for all 8 touched paths with the 40-character shape check, and **re-verified after HEAD advanced twice mid-turn** (`63e2609` -> `1964d51`): still identical |
| **AFTER** | `C:\Users\Admin\tanitad-mdhazard` | the same tree plus this turn's edits |

A full recursive `diff -rq` between them lists **exactly** the 9 edited files plus 1 new test file
and nothing else, so nothing outside the change can be moving the numbers.

Identical invocation in both:
`cd stack && PYTHONPATH=<tree>/stack OMP_NUM_THREADS=<n> PYTHONUTF8=1 python -m pytest -q
-p no:cacheprovider --continue-on-collection-errors`

| | failed | passed | skipped | xfailed | errors | wall |
|---|---:|---:|---:|---:|---:|---:|
| **BASELINE** | **65** | 6,395 | 121 | 2 | **51** | 1,077.5 s |
| **AFTER (first run)** | 68 | 6,417 | 121 | 2 | 51 | 1,035.7 s |
| **AFTER (final)** | **65** | **6,420** | 121 | 2 | **51** | 818.8 s |

* ⭐⭐ **FAILURE-ID SET DIFF, FINAL: EMPTY IN BOTH DIRECTIONS.** `comm -13` (new) = **0** and
  `comm -23` (newly fixed) = **0**, over **116** ids on each side. The diff is over the IDs, not the
  counts, so an equal-count swap could not hide in it.
* ⭐ **The `passed` delta is +25 and is accounted for EXACTLY**: `test_metric_decode_refusal.py`
  collects **25**. 6,395 + 25 = **6,420**. No existing test was quietly deleted or skipped.
* ⛔⛔ **THE FIRST AFTER RUN HAD 3 REGRESSIONS, AND THEY WERE THE GUARD DOING ITS JOB.**
  `test_p9_saliency.py::{test_end_to_end_synthetic_smoke,
  test_speed_saliency_is_exactly_zero_outside_the_declared_support,
  test_support_guard_FIRES_on_a_misdeclared_support}` all build a FRESH tiny stack — at
  initialisation by construction — and ask `builtin_targets` for the speed/yaw targets, which decode
  METRES. ⭐ **The controlled comparison is the only thing that found them**: the targeted runs on
  the files I had edited were all green. Fixed by making the opt-in explicit where the test
  interrogates the SUPPORT GRAPH rather than a metric value, and by auto-allowing
  `--synthetic` — which is not a loophole: that arm is random weights AND random frames by
  construction and the script already stamps it `quotable: NONE`.
* ⚠️ **The baseline is NOT green, and that is why the verdict is the set diff.** Both trees are
  `stack/`-only copies without most repo-root data, so a large block of environment-dependent tests
  fails or errors in **both** columns and cancels out.
* ⚠️ The wall-clock differs (1,077 s vs 819 s) only because the two runs used different
  `OMP_NUM_THREADS` (2 vs 4) and did not overlap; it carries no signal.

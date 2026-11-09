# P6 read: frame control, Master Mind note (2026-11-09, before any P6 score existed)

**What happened.** The P6 navtest seam finished all 12,146 tokens at 07:56 CET with 0 misses, wrote `refe_p6final.npz`, and exited 1 because its frame control is 6.18 m against a 1 mm bar. `chain_p6.py` treated exit 1 as a failure and stopped. No score had been computed.

**Why this is not a new failure.** The same bar has failed the same way on every full-navtest seam: the declared departure in `eval/RESULT_NAVTEST_FINAL.md` (36 tokens in 4 logs, a nuPlan-DB against OpenScene pose disagreement, `raw/2026-10-04-navtest-final/frame_control_navtest_full.json`). The control compares the log future with the W3 export, so it depends on the data and not on the model. The P2 gate is "frame control equal to u400's". The `chain_p6.py` written for this read did not carry that rule.

**Check (MEASURED, reports in `E:/Projects/TanitAD/data/refe_navtest/seams/`).**

| seam | n | max m | median m |
|---|---|---|---|
| `refe_p6final` | 12,146 | 6.18063877792441 | 9.155133597044475e-16 |
| `stage2_sft4_B_u400_navtest_full` | 12,146 | 6.18063877792441 | 9.155133597044475e-16 |
| `stage2_p2s1_full` | 12,146 | 6.18063877792441 | 9.155133597044475e-16 |

Max and median are bit-identical to u400's and P2-s1's. This compares two summary statistics, not a per-token list.

**Action.** The seam is not re-run. `chain.FAILED` is moved aside, and the chain resumes at scoring because the seam file exists. The P6 result must state this departure and report the score with and without the 36 tokens, as `RESULT_NAVTEST_FINAL.md` did. No bar, gate or token set changes.

# MM ruling on the P6-READ-GAPS escalations (2026-10-22, Master Mind)

**Made BEFORE any P6 checkpoint has touched navtest.** No P6 navtest number exists. The rulings below define how the registered read is instrumented. They do not change the endpoint, the comparators, the estimator, the decision rule or the goal bar.

## Verification by the MM

`eval/test_p6_read_gaps.py` was re-run by the MM on an overlay of the mirror tip `323a13d`. That overlay is `eval/` + `refe/` with the package's files on top; nothing under those paths changed between `1664143` and `323a13d`. Result: **`ZZP6GAPS PASS 31/31`**. The package's `MANIFEST.sha256` verifies on all 93 entries.

## E1 / F1: the scoring caller. **RATIFIED: the tip's post-NT-FIX1 caller** (`eval/score_navtest_refe.py`), with the banked harness and `RUN_V1_FIX1_ALLOW_DRIFT=1`

- **Why:** NT-FIX1's three changes are safety fixes and do not change the scoring:
  - A: the cross-drive relpath;
  - B: stale csv files are quarantined;
  - C: no summary is written on a count mismatch.
- The declared drift is one line: `navsim_v1_win.py:49` no longer points into `%TEMP%`.
  - It is recorded as `run_v1_matches_pinned: false` in the compare JSON.
  - It is declared, not hidden.
- **Evidence (MEASURED, `raw/harness_ctl/`):** both callers re-score the same 24 banked u400 navtest tokens with the same csv sha256 `a71dd17c…`, so 192 of 192 per-token cells are bit-equal to u400's banked csv. Both RED arms fire.
- ⛔ **Read-day preflight, added by this ruling:** before the seam starts, re-run arm B (24 u400 tokens, NT-FIX1 caller, banked harness). It must reproduce csv sha256 `a71dd17c…`. A mismatch stops the read before any P6 output exists.

## E2 / F2: the families path. **ACCEPTED.**

The banked `eval/harness/ev6e/families6.py` is now resolved first by both stage 2 and stage 3. The selfcheck reproduces u400's banked families block with 0 differences.

## E3 / G2c: command adherence, DEFINED HERE

PREREG_P6 line 125 names the row and does not define it. This ruling defines it before the read. The instrument is BACKLOG `P6-ADHERENCE`.

| Q | ruling |
|---|---|
| Q1 class of the selected plan | The canonical labeller `refc_tactical.factor_from_kinematics`, as used by `taniteval/taniteval/four_families.py` `tactical_from_trajectory`. **No new thresholds.** The plan is classed as left / straight / right. |
| Q2 statistic and subset | Both **accuracy** and **Cohen's κ**, reported on **all tokens** (12,146: LEFT 2,501 / STRAIGHT 8,070 / RIGHT 1,575 / UNKNOWN 0) **and** on the **turn-active subset** (LEFT ∪ RIGHT commands, 4,076 tokens). Turn-active accuracy is the headline cell, because straight dominates the full set. The 3×3 confusion matrix is always reported. |
| Q3 which plan | The **executed plan after the A7 transform**, which is the plan PDMS scores (primary). The raw proposal is reported as descriptive. |
| Q4 interval | The paired log-cluster bootstrap (B = 10,000) on identical tokens, **against C1 and against C2 separately**. The level is reported with its CI. |
| Q5 floor | **Yes.** The random-proposal floor is the mean adherence of the 64 proposals per token, from the `--dump-proposals` dump, as for selection quality. |

**Authority:** TACTICAL is a reported family. It carries **no decision authority** over PREREG_P6's rule, which is PDMS against C1/C2 and the goal bar. It may be quoted only with the estimator and n above.

# RESULT -- SPEC_NAVTEST AMENDMENT 5 -- `a5confirm_ep013` (match the selection rule to the benchmark, PI option 3)

*Every number is read from the unchanged harness's per-token scores; none is typed. Written by `eval/rule_confirm.py`; the rule was fixed in Amendment 5 (blob `41e25c57`) before any v1-rule pick on these tokens was computed.*

**Confirmation tokens:** 923 from the 43 navtest logs that W3's 200-token subset does not touch; snapshot after epoch 13. **Gates:** G-A5a the shipped rule reproduces the planner's own pick on 923/923 tokens; G-A5b identical picks score identically (max |diff| 0.0).

## Verdict: **NO MEASURABLE DIFFERENCE**

| selection rule | PDMS | vs shipped, PDMS points [95 %] | tokens whose pick changed |
|---|---|---|---|
| shipped (NAVSIM v2 EPDMS shape) | 49.98 | -- | -- |
| **NAVSIM v1 formula (the rule under test)** | **50.76** | **+0.78 [-0.30, +1.82]** | 222 |
| v1 without comfort (reported, not gating) | 51.23 | +1.25 [-0.13, +2.65] | 360 |

Rule committed in Amendment 5: REFUTED iff the upper bound is below 0 (keep the shipped rule, back to the PI); otherwise the v1 formula becomes REFe's selection rule for every evaluation from this amendment on -- CONFIRMED GAIN iff the lower bound is above 0, NO MEASURABLE DIFFERENCE iff the interval straddles 0. Paired log-cluster bootstrap over the 43 logs, 10,000 resamples.

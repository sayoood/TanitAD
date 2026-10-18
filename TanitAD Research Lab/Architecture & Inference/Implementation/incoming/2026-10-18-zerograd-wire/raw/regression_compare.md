# base (tip export) vs edited tree -- per-test outcomes

| | base | mine |
|---|---|---|
| FAILED | 12 | 12 |
| passed | 2451 | 2473 |
| skipped | 9 | 9 |

regressions (pass/skip at base -> fail/error in mine): **0**

failing in BOTH (pre-existing on this box): 12
fixed in mine: 0
new tests only in mine: 22 (failing: 0)
tests present at base but missing in mine: 0

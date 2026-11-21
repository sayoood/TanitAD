# RESULT -- SPEC_NAVTEST AMENDMENT 4 -- `sub200_ep020` (the on-policy scorer, PI decision B)

*Every number below is read from `proptable/sub200_ep020/readout.json`, `points/sub200_ep020.json` and the trainer's `metrics.jsonl`; none is typed. Written by `eval/amendment4_readout.py`.*

**Eligibility:** epoch 19 read 68,467 on-policy sets at its start; rank 8 among on-policy snapshots -- the test was already decided (FAILURE at `sub200_ep013`); this read is REPORTED, NOT GATING.

## Reported, not gating: SUCCESS

Selection skill (pick - random) / (oracle - random), paired log-cluster bootstrap over 93 logs: **0.479 [0.366, 0.588]** against the bar 0.25: SUCCESS iff the lower bound exceeds it, FAILURE iff the upper bound is below it. E-6's own verdict (Amendment 3a): NOT_SELECTION_BOUND.

| snapshot | scorer learned from | best of 64 | pick | random | skill [95 %] |
|---|---|---|---|---|---|
| sub200_ep011 | fixed candidates | 91.2 | 43.8 | 43.2 | 0.013 [-0.089, 0.121] |
| sub200_ep012 | fixed candidates | 91.3 | 47.9 | 44.0 | 0.082 [-0.033, 0.198] |
| **sub200_ep020** | its own proposals | 99.0 | **84.8** | 71.9 | **0.479 [0.366, 0.588]** |

## Secondary (reported, not gating)

- **Drivable-area output, within-scene AUC:** 0.773 [0.723, 0.827] (n 97) -- ABOVE 0.6 (point estimate).
- **Pick vs standing still (same tokens), PDMS points:** +22.26 [18.65, 26.02] · separated.
- **Pick vs sub200_ep011:** +41.01 [34.35, 47.20] · separated.
- **Pick vs sub200_ep012:** +36.96 [30.45, 42.67] · separated.
- **Training-side on-policy skill, epoch 19 (steps 6543-6945):** 0.564 over 36,982 sets in 403 steps (pick 0.689, random 0.450, best 0.873; no interval -- the trainer logs window means; every set met for the first time).

A FAILURE goes to the PI with this table; nothing in the recipe changes again without the PI (SPEC_NAVTEST, AMENDMENT 4).

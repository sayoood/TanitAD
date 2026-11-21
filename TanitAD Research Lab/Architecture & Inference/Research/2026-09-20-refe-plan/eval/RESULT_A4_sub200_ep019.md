# RESULT -- SPEC_NAVTEST AMENDMENT 4 -- `sub200_ep019` (the on-policy scorer, PI decision B)

*Every number below is read from `proptable/sub200_ep019/readout.json`, `points/sub200_ep019.json` and the trainer's `metrics.jsonl`; none is typed. Written by `eval/amendment4_readout.py`.*

**Eligibility:** epoch 18 read 59,320 on-policy sets at its start; rank 7 among on-policy snapshots -- the test was already decided (FAILURE at `sub200_ep013`); this read is REPORTED, NOT GATING.

## Reported, not gating: SUCCESS

Selection skill (pick - random) / (oracle - random), paired log-cluster bootstrap over 93 logs: **0.396 [0.252, 0.530]** against the bar 0.25: SUCCESS iff the lower bound exceeds it, FAILURE iff the upper bound is below it. E-6's own verdict (Amendment 3a): NOT_SELECTION_BOUND.

| snapshot | scorer learned from | best of 64 | pick | random | skill [95 %] |
|---|---|---|---|---|---|
| sub200_ep011 | fixed candidates | 91.2 | 43.8 | 43.2 | 0.013 [-0.089, 0.121] |
| sub200_ep012 | fixed candidates | 91.3 | 47.9 | 44.0 | 0.082 [-0.033, 0.198] |
| **sub200_ep019** | its own proposals | 98.5 | **81.6** | 70.4 | **0.396 [0.252, 0.530]** |

## Secondary (reported, not gating)

- **Drivable-area output, within-scene AUC:** 0.787 [0.741, 0.838] (n 97) -- ABOVE 0.6 (point estimate).
- **Pick vs standing still (same tokens), PDMS points:** +18.98 [15.29, 22.96] · separated.
- **Pick vs sub200_ep011:** +37.74 [30.97, 44.51] · separated.
- **Pick vs sub200_ep012:** +33.68 [27.71, 39.35] · separated.
- **Training-side on-policy skill, epoch 18 (steps 6141-6542):** 0.517 over 31,940 sets in 402 steps (pick 0.667, random 0.448, best 0.872; no interval -- the trainer logs window means; every set met for the first time).

A FAILURE goes to the PI with this table; nothing in the recipe changes again without the PI (SPEC_NAVTEST, AMENDMENT 4).

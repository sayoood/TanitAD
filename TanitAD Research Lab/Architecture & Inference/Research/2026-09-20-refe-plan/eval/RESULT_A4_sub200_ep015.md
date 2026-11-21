# RESULT -- SPEC_NAVTEST AMENDMENT 4 -- `sub200_ep015` (the on-policy scorer, PI decision B)

*Every number below is read from `proptable/sub200_ep015/readout.json`, `points/sub200_ep015.json` and the trainer's `metrics.jsonl`; none is typed. Written by `eval/amendment4_readout.py`.*

**Eligibility:** epoch 14 read 20,659 on-policy sets at its start; rank 3 among on-policy snapshots -- the test was already decided (FAILURE at `sub200_ep013`); this read is REPORTED, NOT GATING.

## Reported, not gating: SUCCESS

Selection skill (pick - random) / (oracle - random), paired log-cluster bootstrap over 93 logs: **0.426 [0.328, 0.522]** against the bar 0.25: SUCCESS iff the lower bound exceeds it, FAILURE iff the upper bound is below it. E-6's own verdict (Amendment 3a): NOT_SELECTION_BOUND.

| snapshot | scorer learned from | best of 64 | pick | random | skill [95 %] |
|---|---|---|---|---|---|
| sub200_ep011 | fixed candidates | 91.2 | 43.8 | 43.2 | 0.013 [-0.089, 0.121] |
| sub200_ep012 | fixed candidates | 91.3 | 47.9 | 44.0 | 0.082 [-0.033, 0.198] |
| **sub200_ep015** | its own proposals | 84.2 | **61.2** | 44.1 | **0.426 [0.328, 0.522]** |

## Secondary (reported, not gating)

- **Drivable-area output, within-scene AUC:** 0.751 [0.703, 0.796] (n 124) -- ABOVE 0.6 (point estimate).
- **Pick vs standing still (same tokens), PDMS points:** -1.40 [-6.57, 3.77] · not separated.
- **Pick vs sub200_ep011:** +17.36 [10.08, 24.37] · separated.
- **Pick vs sub200_ep012:** +13.30 [7.00, 19.39] · separated.
- **Training-side on-policy skill, epoch 14 (steps 4531-4932):** 0.347 over 11,142 sets in 402 steps (pick 0.580, random 0.429, best 0.864; no interval -- the trainer logs window means; every set met for the first time).

A FAILURE goes to the PI with this table; nothing in the recipe changes again without the PI (SPEC_NAVTEST, AMENDMENT 4).

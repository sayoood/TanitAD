# RESULT -- SPEC_NAVTEST AMENDMENT 4 -- `sub200_ep016` (the on-policy scorer, PI decision B)

*Every number below is read from `proptable/sub200_ep016/readout.json`, `points/sub200_ep016.json` and the trainer's `metrics.jsonl`; none is typed. Written by `eval/amendment4_readout.py`.*

**Eligibility:** epoch 15 read 30,248 on-policy sets at its start; rank 4 among on-policy snapshots -- the test was already decided (FAILURE at `sub200_ep013`); this read is REPORTED, NOT GATING.

## Reported, not gating: SUCCESS

Selection skill (pick - random) / (oracle - random), paired log-cluster bootstrap over 93 logs: **0.452 [0.301, 0.598]** against the bar 0.25: SUCCESS iff the lower bound exceeds it, FAILURE iff the upper bound is below it. E-6's own verdict (Amendment 3a): NOT_SELECTION_BOUND.

| snapshot | scorer learned from | best of 64 | pick | random | skill [95 %] |
|---|---|---|---|---|---|
| sub200_ep011 | fixed candidates | 91.2 | 43.8 | 43.2 | 0.013 [-0.089, 0.121] |
| sub200_ep012 | fixed candidates | 91.3 | 47.9 | 44.0 | 0.082 [-0.033, 0.198] |
| **sub200_ep016** | its own proposals | 96.2 | **76.5** | 60.2 | **0.452 [0.301, 0.598]** |

## Secondary (reported, not gating)

- **Drivable-area output, within-scene AUC:** 0.828 [0.771, 0.891] (n 105) -- ABOVE 0.6 (point estimate).
- **Pick vs standing still (same tokens), PDMS points:** +13.94 [8.92, 19.34] · separated.
- **Pick vs sub200_ep011:** +32.70 [25.88, 39.57] · separated.
- **Pick vs sub200_ep012:** +28.64 [22.49, 34.34] · separated.
- **Training-side on-policy skill, epoch 15 (steps 4933-5335):** 0.367 over 16,370 sets in 403 steps (pick 0.595, random 0.436, best 0.869; no interval -- the trainer logs window means; every set met for the first time).

A FAILURE goes to the PI with this table; nothing in the recipe changes again without the PI (SPEC_NAVTEST, AMENDMENT 4).

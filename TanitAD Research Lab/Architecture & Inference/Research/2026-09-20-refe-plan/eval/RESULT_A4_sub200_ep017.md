# RESULT -- SPEC_NAVTEST AMENDMENT 4 -- `sub200_ep017` (the on-policy scorer, PI decision B)

*Every number below is read from `proptable/sub200_ep017/readout.json`, `points/sub200_ep017.json` and the trainer's `metrics.jsonl`; none is typed. Written by `eval/amendment4_readout.py`.*

**Eligibility:** epoch 16 read 40,155 on-policy sets at its start; rank 5 among on-policy snapshots -- the test was already decided (FAILURE at `sub200_ep013`); this read is REPORTED, NOT GATING.

## Reported, not gating: SUCCESS

Selection skill (pick - random) / (oracle - random), paired log-cluster bootstrap over 93 logs: **0.435 [0.305, 0.550]** against the bar 0.25: SUCCESS iff the lower bound exceeds it, FAILURE iff the upper bound is below it. E-6's own verdict (Amendment 3a): NOT_SELECTION_BOUND.

| snapshot | scorer learned from | best of 64 | pick | random | skill [95 %] |
|---|---|---|---|---|---|
| sub200_ep011 | fixed candidates | 91.2 | 43.8 | 43.2 | 0.013 [-0.089, 0.121] |
| sub200_ep012 | fixed candidates | 91.3 | 47.9 | 44.0 | 0.082 [-0.033, 0.198] |
| **sub200_ep017** | its own proposals | 98.8 | **82.6** | 70.2 | **0.435 [0.305, 0.550]** |

## Secondary (reported, not gating)

- **Drivable-area output, within-scene AUC:** 0.792 [0.735, 0.856] (n 101) -- ABOVE 0.6 (point estimate).
- **Pick vs standing still (same tokens), PDMS points:** +20.05 [15.56, 24.12] · separated.
- **Pick vs sub200_ep011:** +38.80 [32.17, 44.92] · separated.
- **Pick vs sub200_ep012:** +34.74 [28.21, 40.59] · separated.
- **Training-side on-policy skill, epoch 16 (steps 5336-5737):** 0.399 over 21,628 sets in 402 steps (pick 0.612, random 0.441, best 0.868; no interval -- the trainer logs window means; every set met for the first time).

A FAILURE goes to the PI with this table; nothing in the recipe changes again without the PI (SPEC_NAVTEST, AMENDMENT 4).

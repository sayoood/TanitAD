# RESULT -- SPEC_NAVTEST AMENDMENT 4 -- `sub200_final` (the on-policy scorer, PI decision B)

*Every number below is read from `proptable/sub200_final/readout.json`, `points/sub200_final.json` and the trainer's `metrics.jsonl`; none is typed. Written by `eval/amendment4_readout.py`.*

**Eligibility:** epoch 27 read 146,788 on-policy sets at its start; rank 16 among on-policy snapshots -- the test was already decided (FAILURE at `sub200_ep013`); this read is REPORTED, NOT GATING.

## Reported, not gating: UNDETERMINED

Selection skill (pick - random) / (oracle - random), paired log-cluster bootstrap over 93 logs: **0.375 [0.232, 0.517]** against the bar 0.25: SUCCESS iff the lower bound exceeds it, FAILURE iff the upper bound is below it. E-6's own verdict (Amendment 3a): UNDETERMINED.

| snapshot | scorer learned from | best of 64 | pick | random | skill [95 %] |
|---|---|---|---|---|---|
| sub200_ep011 | fixed candidates | 91.2 | 43.8 | 43.2 | 0.013 [-0.089, 0.121] |
| sub200_ep012 | fixed candidates | 91.3 | 47.9 | 44.0 | 0.082 [-0.033, 0.198] |
| **sub200_final** | its own proposals | 98.3 | **84.2** | 75.7 | **0.375 [0.232, 0.517]** |

## Secondary (reported, not gating)

- **Drivable-area output, within-scene AUC:** 0.737 [0.696, 0.783] (n 85) -- ABOVE 0.6 (point estimate).
- **Pick vs standing still (same tokens), PDMS points:** +21.60 [17.42, 26.18] · separated.
- **Pick vs sub200_ep011:** +40.36 [33.60, 47.05] · separated.
- **Pick vs sub200_ep012:** +36.30 [29.50, 42.41] · separated.
- **Training-side on-policy skill, epoch 27 (steps 9763-10074):** 0.637 over 61,372 sets in 312 steps (pick 0.734, random 0.480, best 0.879; no interval -- the trainer logs window means; every set met for the first time).

A FAILURE goes to the PI with this table; nothing in the recipe changes again without the PI (SPEC_NAVTEST, AMENDMENT 4).

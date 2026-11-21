# RESULT -- SPEC_NAVTEST AMENDMENT 4 -- `sub200_ep014` (the on-policy scorer, PI decision B)

*Every number below is read from `proptable/sub200_ep014/readout.json`, `points/sub200_ep014.json` and the trainer's `metrics.jsonl`; none is typed. Written by `eval/amendment4_readout.py`.*

**Eligibility:** epoch 13 read 10,885 on-policy sets at its start; rank 2 among on-policy snapshots -- the test was already decided (FAILURE at `sub200_ep013`); this read is REPORTED, NOT GATING.

## Reported, not gating: UNDETERMINED

Selection skill (pick - random) / (oracle - random), paired log-cluster bootstrap over 93 logs: **0.329 [0.233, 0.431]** against the bar 0.25: SUCCESS iff the lower bound exceeds it, FAILURE iff the upper bound is below it. E-6's own verdict (Amendment 3a): UNDETERMINED.

| snapshot | scorer learned from | best of 64 | pick | random | skill [95 %] |
|---|---|---|---|---|---|
| sub200_ep011 | fixed candidates | 91.2 | 43.8 | 43.2 | 0.013 [-0.089, 0.121] |
| sub200_ep012 | fixed candidates | 91.3 | 47.9 | 44.0 | 0.082 [-0.033, 0.198] |
| **sub200_ep014** | its own proposals | 88.0 | **58.9** | 44.7 | **0.329 [0.233, 0.431]** |

## Secondary (reported, not gating)

- **Drivable-area output, within-scene AUC:** 0.718 [0.677, 0.763] (n 127) -- ABOVE 0.6 (point estimate).
- **Pick vs standing still (same tokens), PDMS points:** -3.67 [-9.42, 2.51] · not separated.
- **Pick vs sub200_ep011:** +15.09 [8.96, 20.94] · separated.
- **Pick vs sub200_ep012:** +11.03 [4.80, 16.91] · separated.
- **Training-side on-policy skill, epoch 13 (steps 4128-4530):** 0.270 over 5,864 sets in 403 steps (pick 0.471, random 0.371, best 0.741; no interval -- the trainer logs window means; every set met for the first time).

A FAILURE goes to the PI with this table; nothing in the recipe changes again without the PI (SPEC_NAVTEST, AMENDMENT 4).

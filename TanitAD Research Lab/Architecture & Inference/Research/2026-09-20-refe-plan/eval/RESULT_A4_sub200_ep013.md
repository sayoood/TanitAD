# RESULT -- SPEC_NAVTEST AMENDMENT 4 -- `sub200_ep013` (the on-policy scorer, PI decision B)

*Every number below is read from `proptable/sub200_ep013/readout.json`, `points/sub200_ep013.json` and the trainer's `metrics.jsonl`; none is typed. Written by `eval/amendment4_readout.py`.*

**Eligibility:** epoch 12 read 1,618 on-policy sets at its start; rank 1 among on-policy snapshots -- this is the PRIMARY test.

## Primary: **FAILURE**

Selection skill (pick - random) / (oracle - random), paired log-cluster bootstrap over 93 logs: **0.073 [-0.047, 0.199]** against the bar 0.25: SUCCESS iff the lower bound exceeds it, FAILURE iff the upper bound is below it. E-6's own verdict (Amendment 3a): SELECTION_BOUND.

| snapshot | scorer learned from | best of 64 | pick | random | skill [95 %] |
|---|---|---|---|---|---|
| sub200_ep011 | fixed candidates | 91.2 | 43.8 | 43.2 | 0.013 [-0.089, 0.121] |
| sub200_ep012 | fixed candidates | 91.3 | 47.9 | 44.0 | 0.082 [-0.033, 0.198] |
| **sub200_ep013** | its own proposals | 86.3 | **49.4** | 46.5 | **0.073 [-0.047, 0.199]** |

## Secondary (reported, not gating)

- **Drivable-area output, within-scene AUC:** 0.650 [0.594, 0.713] (n 126) -- ABOVE 0.6 (point estimate).
- **Pick vs standing still (same tokens), PDMS points:** -13.18 [-18.87, -7.14] · separated.
- **Pick vs sub200_ep011:** +5.57 [-0.44, 11.51] · not separated.
- **Pick vs sub200_ep012:** +1.52 [-4.77, 7.47] · not separated.
- **Training-side on-policy skill, epoch 12 (steps 3726-4127):** 0.145 over 907 sets in 350 steps (pick 0.413, random 0.361, best 0.722; no interval -- the trainer logs window means; every set met for the first time).

A FAILURE goes to the PI with this table; nothing in the recipe changes again without the PI (SPEC_NAVTEST, AMENDMENT 4).

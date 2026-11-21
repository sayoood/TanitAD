# RESULT -- SPEC_NAVTEST AMENDMENT 4 -- `sub200_ep018` (the on-policy scorer, PI decision B)

*Every number below is read from `proptable/sub200_ep018/readout.json`, `points/sub200_ep018.json` and the trainer's `metrics.jsonl`; none is typed. Written by `eval/amendment4_readout.py`.*

**Eligibility:** epoch 17 read 49,819 on-policy sets at its start; rank 6 among on-policy snapshots -- the test was already decided (FAILURE at `sub200_ep013`); this read is REPORTED, NOT GATING.

## Reported, not gating: SUCCESS

Selection skill (pick - random) / (oracle - random), paired log-cluster bootstrap over 93 logs: **0.468 [0.369, 0.560]** against the bar 0.25: SUCCESS iff the lower bound exceeds it, FAILURE iff the upper bound is below it. E-6's own verdict (Amendment 3a): NOT_SELECTION_BOUND.

| snapshot | scorer learned from | best of 64 | pick | random | skill [95 %] |
|---|---|---|---|---|---|
| sub200_ep011 | fixed candidates | 91.2 | 43.8 | 43.2 | 0.013 [-0.089, 0.121] |
| sub200_ep012 | fixed candidates | 91.3 | 47.9 | 44.0 | 0.082 [-0.033, 0.198] |
| **sub200_ep018** | its own proposals | 99.3 | **83.7** | 70.0 | **0.468 [0.369, 0.560]** |

## Secondary (reported, not gating)

- **Drivable-area output, within-scene AUC:** 0.811 [0.763, 0.861] (n 101) -- ABOVE 0.6 (point estimate).
- **Pick vs standing still (same tokens), PDMS points:** +21.13 [17.02, 25.06] · separated.
- **Pick vs sub200_ep011:** +39.89 [32.68, 46.22] · separated.
- **Pick vs sub200_ep012:** +35.83 [29.01, 41.83] · separated.
- **Training-side on-policy skill, epoch 17 (steps 5738-6140):** 0.462 over 26,922 sets in 403 steps (pick 0.639, random 0.442, best 0.869; no interval -- the trainer logs window means; every set met for the first time).

A FAILURE goes to the PI with this table; nothing in the recipe changes again without the PI (SPEC_NAVTEST, AMENDMENT 4).

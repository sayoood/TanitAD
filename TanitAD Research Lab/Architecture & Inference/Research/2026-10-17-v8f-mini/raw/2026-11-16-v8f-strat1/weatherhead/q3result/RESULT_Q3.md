# RESULT: weather label audit Q3, read 2026-11-17

**Evidence class.** MEASURED (the PI's blind reading of 40 frames, scored by `score_q3.py` against `key.json`). Pre-registration `PREREG_Q3.md`, sha256 4d17a3dd, written before the draw. Not a model metric and has no tier. It measures the label's **precision** per class: when the label says X, how often the PI sees X.

| label class | frames | answered | agree | agreement [Wilson 95 %] | pre-registered reading |
|---|---|---|---|---|---|
| snow | 11 | 11 | 11 | 1.00 [0.74, 1.00] | **sound** (>= 0.80) |
| rain | 11 | 11 | 8 | 0.727 [0.434, 0.903] | **borderline** (0.70-0.80): a second blind draw of rain, about 20 frames, before any relabel |
| fog | 11 | 10 (1 can't tell) | 2 | 0.20 [0.057, 0.510] | **relabel candidate** (< 0.70) |
| clear | 7 | 7 | 6 | 0.857 [0.487, 0.974] | sanity check only, no bar |

**Confusion** (rows are the label, columns are the PI):

| label \ PI | clear | fog | rain | snow | can't tell |
|---|---|---|---|---|---|
| clear | 6 | 0 | 0 | 1 | 0 |
| fog | 3 | 2 | 1 | 4 | 1 |
| rain | 1 | 0 | 8 | 2 | 0 |
| snow | 0 | 0 | 0 | 11 | 0 |

Pooled: 27/39 = 0.692 [0.536, 0.814]. This is a stratified draw, not comparable to the 90.0 % of the earlier random draw. Prevalence-weighted over the train windows: 0.846 (ESTIMATED reweighting, no interval).

**What it means (committed reading).** The label is the weather lever for fog.

- **Fog:** of the 10 answered "fog" frames, the PI reads 4 as snow, 3 as clear and 1 as rain. The fog class is mostly other weather.
- **Rain:** borderline at n = 11. The pre-registered next step is a second blind draw.
- **Snow:** sound. Head accuracy against the snow label is a real signal.
- **Clear:** one "clear" frame reads snow, a recall gap of the label that the confusion records.

**Next (pre-registered).**

1. A second blind draw of rain, about 20 frames, with the same instrument.
2. A fog relabel card, as a new card with its own PI go: relabel from the PI's answers plus a stronger labeller.

The weather head's fog recall (0.75 against this label) is not interpretable until then.

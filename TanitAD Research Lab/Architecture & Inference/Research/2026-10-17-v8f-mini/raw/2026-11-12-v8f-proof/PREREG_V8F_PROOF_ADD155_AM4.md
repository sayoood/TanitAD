# ADD155 Amendment 4 — A1 temporal probe, linear form with the permutation reference (frozen before its data)

Written 2026-11-28 after the A1 temporal MLP (P-MLP) read was rendered. Under the card's rule P-MLP reads NOT-ENCODED on every cell, but its dR^2 is -0.15 to -2.0, including on ego `speed` (-0.005 to -0.028), whose target is inside EGO: the instrument overfits (673 inputs, early stopping on a random frame split that leaks within clips) and the card gave it no control. That read is reported as INSTRUMENT-LIMITED, not as an absence. This amendment adds the cheapest valid temporal instrument; its values have not been computed.

**Probe P-TLIN:** the card's P-LIN (ridge, alpha by inner 3-fold clip CV on the training folds, 5 outer clip folds) on [state at t-4, .., t (5 frames, 1.0 s), EGO] against [EGO]. Representations L1, L2, L3, TOK-MEAN. Targets `lead_gap`, `ag1_pos`, `lead_closing`, `lead_rel_acc`, `gap_change_2s`. Rows = A1's rows (t >= 10). Interval: the card's episode-cluster bootstrap (B 2,000, rng 0). Reference: Amendment 3's 19 clip derangements (rng seeds 100..118), refit per draw.

**Label (two-part, Amendment 3's rule):** ENCODED-BEYOND-EGO (temporal) if the card's rule holds (seed mean >= bar, bar 0.10 for `lead_gap` and `ag1_pos`, 0.05 otherwise; CI lower bound > 0 at each seed) AND the observed value exceeds all 19 permutation values at each seed; PARTIAL likewise with the mean below the bar; otherwise NOT-ENCODED. A cell that is ENCODED under P-TLIN and not under single-frame P-LIN (Amendment 3, where computed) is ENCODED-TEMPORAL-ONLY, which ranks B3 (temporal context) per card section 6.

Tier T0. Four families do not apply (card section 0).

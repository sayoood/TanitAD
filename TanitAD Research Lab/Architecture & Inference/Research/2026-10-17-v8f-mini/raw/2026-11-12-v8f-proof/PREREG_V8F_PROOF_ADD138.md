# PREREG_V8F_PROOF_ADD138 — RULELOG10KB: the split of card ADD137 with the execution part written with the engine's gap at the tick

Frozen before any tool of this card is written and before any value of this card or of card ADD137 exists: the tool of card ADD137 has read synthetic test data alone. Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m). CPU only, on banked runs and rule logs.

## 1. Why this card, and when it runs

Card ADD137 writes the execution part e_X with the gap g0 of the rule log and asks in control w2 that S = e_L + e_X hold within 1e-6 m, while its control w1 lets g0 differ from the engine's gap g_m(k_t) by up to 1e-3 m. The identity holds exactly only where g0 equals g_m(k_t); the rule works in single precision and the engine in double, so on real windows the two can differ by more than 1e-6 m while w1 passes. That is a property of the two files and not of the split the control guards.

Order, committed here: card ADD137 is run first, as frozen. If all its controls pass, its names stand and this card is not run. If its control w2 fails with w0, w1, w3, w4 and w5 passed, card ADD137 is reported as failed with no name and no value opened, and this card is run; its row is read by the other session before it is landed (fork rule of row 136). If another control of card ADD137 fails, the same control fails here and both cards are reported as failed.

## 2. Everything but one definition is card ADD137

Windows, measures R1 to R4, estimator, families, names with the prefix `RULELOG10KB-K0-` in place of `RULELOG10K-K0-`, committed readings, cost and limits: sections 2 to 4, 6 and 7 of card ADD137, by the tool of that card, imported unchanged. One definition differs: e_X = [a_m(k*) - a_m(k_t)] - [g_m(k_t) + v_L x 0.2 s x i - max(3.0 m, 1.2 s x v_m(k* - 1))], with the engine's gap g_m(k_t) in place of the logged g0. With this, S = e_L + e_X holds by construction.

## 3. Controls (all have to pass before any name or value is opened)

* **w0, w1, w3, w4, w5** — as in card ADD137 (w4 also holds the test output of this card's tool, dev box and pod).
* **w2 identity** — S equals e_L plus e_X within 1e-6 m in every window, with e_X as in section 2; and beside it the largest difference between the logged g0 and g_m(k_t) over the windows is at most 1e-3 m (the clause of w1, read again).

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

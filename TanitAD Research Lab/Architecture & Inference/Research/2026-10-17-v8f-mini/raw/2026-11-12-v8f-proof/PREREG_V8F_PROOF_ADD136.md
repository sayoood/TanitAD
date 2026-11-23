# PREREG_V8F_PROOF_ADD136 — LABELUNSAFE10KB: the read of card ADD135 with a share in its planted control

Frozen before any tool of this card is written and before any value of this card or of card ADD135 exists: the tool of card ADD135 has read synthetic test data alone. Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m). CPU only, on banked runs.

## 1. Why this card, and when it runs

Control u2 of card ADD135 asks that EVERY window the planted track makes unsafe counts for M4 (lowest model time gap more than 0.2 s below the logged driver's). When the tool was built it became clear, with no real file read, that this cannot hold for every real window: the engine floors the speed at 0.5 m/s, so at a near standstill an advance of 0.35 s of travel lowers the time gap by less than 0.2 s; and where the logged driver brakes, the step speed of the planted track falls with the advance. Both are properties of the plant and not of the measure the control guards.

Order, committed here: card ADD135 is run first, as frozen. If all its controls pass, its names stand and this card is not run. If its control u2 fails with u0, u1, u3 and u4 passed, card ADD135 is reported as failed with no name and no value opened, and this card is run; its row is read by the other session before it is landed (fork rule of row 136). If u0, u1, u3 or u4 of card ADD135 fails, the same control fails here and both cards are reported as failed.

## 2. Everything but one clause is card ADD135

Runs, windows, measures M1 to M5, estimator, families, the two names with the prefix `LABELUNSAFE10KB-K0-` in place of `LABELUNSAFE10K-K0-`, the committed readings, cost and limits: sections 2 to 4, 6 and 7 of card ADD135, by the tool of that card, imported unchanged.

## 3. Controls (all have to pass before any name or value is opened)

* **u0, u1, u3, u4** — as in card ADD135. u3 also holds the test output of this card's tool, dev box and pod.
* **u2 planted** — on the real windows with the model track replaced by the logged track, no window is unsafe and M4 has no window (as card ADD135). With the logged track advanced along the path by 0.35 s of travel at each step: one window or more becomes unsafe; of the windows that become unsafe, 0.90 or more count for M4; every window that becomes unsafe and does not count for M4 has a logged step speed below 1.0 m/s at one scored step or more, or a logged speed that falls by more than 0.2 m/s from one scored step to the next; none is unsafe by the gap alone unless its gap is below 2.0 m by the second routine.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

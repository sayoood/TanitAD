# X3 CEIL-OBEY + X10 NIGHT-LUMA / LEFT-N instruments (EvalFlyWheel, 2026-10-11)

The Master Mind wrote this from the implementing agent's hand-back, because the harness refused the agent's own report
file. Code and tests are in this commit. Nothing was run over the corpus.

## Built (MEASURED: tests run)

1. **CEIL-OBEY** — `taniteval/taniteval/ceil_obey.py`. It reports three things:
   * the SPEC share: windows whose emitted plan's max speed exceeds the fed **finite** ceiling by **> 0.5 m/s**;
   * the ladder's strict form beside it (> 1e-6);
   * the **exempt-row rate**: windows where no candidate in the fan complies, so the argmax filter keeps the whole fan.

   Violations are split into **exempt** and **non-exempt** (a compliant candidate existed but the plan still exceeded
   the ceiling).
   * The numpy statistic is pinned against the real `SpeedCeilingFilter` and its `speed_rows_empty` count on random fans.
   * `test_ceil_obey.py`: **18 passed**.
   * The ladder side landed separately: `ladder_eval.py` now stores `fan_min_vmax` (SPEC_WPB_LADDER_A2, clause 4).
2. **LEFT-N** — `taniteval/taniteval/nav_compliance.py` now carries **clip n** beside window n (additive; window n is
   unchanged): `n_*_clips`, `informative.*.n_left_clips`, a `left_n` block, and a `clip_id` argument.
   * `test_nav_compliance.py`: **35 passed** (31 existing + 4 new).
3. **NIGHT-LUMA** — `taniteval/taniteval/night_luma.py`. It provides:
   * a frame-level luma function;
   * a clip census against the clock label;
   * a window mask carrying window n and clip n.

   It follows D3's `c5_frames.py` exactly: 8-bit, plain channel mean, bottom strip excluded, mean over frames, night
   if strictly under 30.
   * `test_night_luma.py`: **15 passed**.

Expected values are hand-derived literals. Each item has deliberate-regression arms that go RED: 5 for X3, 5 for
NIGHT-LUMA, 2 for LEFT-N.

**Comparison against the tip:** the Master Mind compared `taniteval/tests` with and without this change (plus X7). The
failure sets were identical, and the 24 stack test files that import the changed modules also showed no new failure.

## Flags against `SPEC_REFCV8_DRAFT.md` — the PI / Master Mind must settle these before registration

* **CEIL-OBEY is named "obey" but defined as a violation share**, with the bar `<= 0.01`.
* **Its denominator is ambiguous:** finite-ceiling windows (used here) or all windows.
* **NIGHT-LUMA:** the draft gives no luma weights, strip handling or aggregation; D3's were used. Rec.601 weights would
  move a (90, 0, 0) frame from 30.0 to 26.91, i.e. across the threshold.
* **"Left-turn" is ambiguous:**
  * D3's 13 clips are nav-token `TURN_L`;
  * S-ROUTE's 849 left windows are ground-truth turns.

  The helpers take any mask, so both readings are computable.
* **UNVERIFIED:** `from_refcv3_dump` now passes `clip_id`, but it was not exercised on a real dump.

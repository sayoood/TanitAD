# ADDENDUM — what a minimal TWO-SEGMENT candidate family would buy

**RULE ZERO continuation.** §1.5 of `RESULT.md` proved the 117 candidates are constant-curvature
arcs (yaw monotone on 100.0000 % of pairs) and §1.9 measured the demand (1,297 / 18,615 windows,
79 / 141 clips). The decision the PI actually faces is not *"is the vocabulary short"* but
*"what does the cheapest extension buy, and how many tokens does it cost"*. Zero GPU, model-free.

**The extension, defined before any number.** A two-segment candidate holds `+a_lat` for `t_split`
seconds and `−a_lat` for the rest of the 6 s horizon, with `a_lon` held constant throughout — the
existing control alphabet, applied twice. Same integrator, same
`kappa = clamp(a_lat / max(v0, 4)², ±0.12)`, same model slots `[5,10,15,20,30,40,50,60]`.

**Scoring.** Best-in-fan ADE over the eight model slots against the **recorded 6 s ego path** — a
SUPPLY ceiling, never comparable to an achievement. 5,000 windows scored: **all 1,297** strict
lane-change windows kept, the remaining 3,703 subsampled from 17,318 with seed 0.

| extension | n new | LC ADE | LC base | **gain** | ALL ADE | ALL base | TURN ADE | TURN base |
|---|---|---|---|---|---|---|---|---|
| 8 `a_lat` × 3 splits, `a_lon = 0` | 24 | 1.3662 | 1.5315 | **0.1654** | 1.2646 | 1.3551 | 1.9065 | 1.9435 |
| 4 `a_lat` × 3 splits, `a_lon = 0` | 12 | 1.3664 | 1.5315 | **0.1651** | 1.2666 | 1.3551 | 1.9215 | 1.9435 |
| **2 `a_lat` (±0.75) × 3 splits (2/3/4 s), `a_lon = 0`** | **6** | **1.3670** | 1.5315 | **0.1645** | 1.2765 | 1.3551 | 1.9388 | 1.9435 |
| 8 `a_lat` × 1 split (3 s), `a_lon = 0` | 8 | 1.5011 | 1.5315 | 0.0304 | 1.3158 | 1.3551 | 1.9285 | 1.9435 |
| 2 `a_lat` × 1 split, `a_lon = 0` | 2 | 1.5011 | 1.5315 | 0.0304 | 1.3214 | 1.3551 | 1.9406 | 1.9435 |

⚠️ The `a_lon every 4th` rows in `raw/out_p1h_two_segment.json` are **NOT supersets** of the
`a_lon = 0` rows — `a_lon_grid[::4]` skips 0.0 — and their smaller gains are that artefact, not a
finding. Read only the `a_lon = 0` rows above.

**CONTROL — the turn column.** A constant-curvature arc is exactly what a junction turn needs, so a
two-segment family that "won" there would mean the scorer, not the geometry, was doing the work.
It does not: the 6-candidate extension moves the turn ceiling by **−0.0047 m (0.24 %)**, i.e.
nothing.

## What this licenses, and what it does not

* **The split point is what matters, not the magnitude.** Three split points at 2 / 3 / 4 s recover
  **0.1645 m** with only **two** `a_lat` magnitudes; a single split at 3 s recovers **0.0304 m**
  even with all eight magnitudes. ⇒ the cheap extension is **6 candidates (+5.1 % of the bank)**;
  going to 24 buys **0.0009 m** more and is not worth the tokens.
* **10.7 % of the lane-change supply gap, and it is not the whole fix.** 1.5315 → 1.3670 m is a
  real but modest ceiling improvement, and these windows stay harder than average (all-window
  ceiling 1.2765 m). ⛔ **The label half is load-bearing**: a head whose three lane-change logits
  have never received a gradient (D-SELQ-LC-HEAD-3, −31 logits, best rank 6 of 8) will not select
  the new candidates deliberately no matter how good they are. Extending the vocabulary WITHOUT
  fixing the emitter would add six candidates nothing is trained to want.
* ⚠️ **This is a SUPPLY number.** It says what the best possible selector could reach with the
  extended fan; it says nothing about what refcv4b's ranker would actually pick, and it may never
  be quoted beside an achievement (`GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md`).

**Recommended order, given the above:** (1) fix `s2_geom_emit_v7.py::tactical_actions()` so a lane
change can be emitted at all and rebuild the v7.2 labels; (2) add the 6-candidate two-segment
family; (3) retrain. Steps 1 and 2 are cheap; step 3 is the PI/compute decision, and the A40 is
committed to refcv5 until 2026-09-08 07:33 UTC.

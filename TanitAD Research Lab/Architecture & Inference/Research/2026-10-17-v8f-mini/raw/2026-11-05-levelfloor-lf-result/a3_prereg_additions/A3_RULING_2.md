# A3_RULING_2: the reader's lift control, ruled after the LF rollouts and before any rule P, G, F, D or LEAK-BORNE was read (MM, 2026-11-05 23:3x)

**What happened.** All 16 LF rollouts finished (`lf_read/raw/LFRO.done`, 23:29). The reader refused at its per-rollout tick controls
(`raw/LF_READ.json` key `refused`), on ONE control only: `lift_zero_where_unbound_and_never_negative`. Every other tool-against-card
control passed on every rollout. The reader stops at the controls, so no rule P, G, F, D or LEAK-BORNE was computed and no contrast
was seen.

**Cause (MEASURED on `RILF050_s0`, ticks dump).**
* Where F does not bind, the lift is exactly 0 on every rollout, so that half of the control holds.
* Where F binds, 52 of the 543 bound replans have a negative `lf_lift`. Across the four rollouts inspected, the minimum is −0.45 m/s
  and the first percentile is −0.025 to −0.0007 m/s.
* The operator is correct on those 52 replans. Every output knot over 0 to 3 s lies on F's input path (S's output; largest distance
  1e-6 m), at an arc position at or beyond its original progress s_k (minimum difference −0.0 m). The dumped lift equals the chord
  recompute.
* The instrument measures something else. `lf_lift` is defined (`lf_build/code/level_floor.py`) as the CHORD length of the output
  polyline over knots 1 to 15 minus the input's, over 3 s.
  * Where the input path bends or reverses between knots, points moved forward along it are joined by shorter chords, so the chord
    length can fall while the arc progress rises.
  * 9.6 % of the 52 have a direction reversal in the input 0 to 3 s path. The rest bend. Their median input speed is 2.914 m/s.
* The card (section 2, steps 4 and 5) guarantees arc progress along O's own path. It guarantees nothing about chord progress, so the
  control's "never negative where bound" was a wrong expectation for this instrument, not a tool defect.

**Ruling.**
1. The control becomes `lift_zero_where_unbound`: |lift| <= 1e-6 wherever F does not bind. This half of the old control holds on
   every rollout.
2. The negative chord lift where F binds is a RESULT about the frozen operator, reported beside each arm, never deciding: the share
   of bound replans with lift < 0, its minimum and its first percentile. On those replans the floor shortens the planned 0 to 3 s
   displacement, which the executor follows. The read measures the operator as frozen, including this effect.
3. Nothing else changes. The rules, outcomes, doses and rollouts stay as they are; the 16 rollouts are reused unchanged, and the
   reader is re-run on them.

The sha256 of this file is recorded in `a3_prereg/FROZEN_SHA.txt` before the reader is re-run.

# PREREG_MAPPATH AMENDMENT 1 -- K-PROG was mis-specified (chord vs along-path); replaced BEFORE any pick or bar (Master Mind, 2026-10-26 ~04:33 CET)

Sealed after `mappath_transform.py` ran and BEFORE `mapsoft_path.py` computed any pick, rate or bar. Base prereg `PREREG_MAPPATH.md` 34a030ee.

## What happened (disclosed in full)
The transform's control record (`result/MAPPATH_TRANSFORM.json`) read: K-ID **PASS** (native plan bit-equal on all 4,754 windows);
K-PROG as written **FAIL** -- 12.0 % of candidates within 1e-4 m, p99 deviation 0.70 m; 4,736 of 708,346 candidates are
zero-length (degenerate) paths. K-PROG compared `arc()` of the re-sampled 8 points (CHORD lengths between consecutive new points)
with the native per-instant arc. `at_progress` places each point at the requested ALONG-PATH distance on the candidate's polyline;
on a curving candidate the chords between those points cut the corners, so the control measured chord shortening, not the
transform. The quantity the design needs -- each output point lies at the native plan's along-path distance on its own candidate
path -- was never checked.

## The change (definition only; arms, objective, bars and next steps unchanged)
* **K-PROG (replaced):** an INDEPENDENT loop-based walker (no shared code with `RT.eval_polyline`: walk the candidate's vertices
  from the origin, accumulate segment lengths, interpolate; extrapolate along the last positive-length segment) recomputes the
  point at distance `s` for a seeded random sample of 5,000 non-degenerate (window, candidate, instant) triples; PASS iff the max
  abs difference to the transform's output is <= 1e-6 m. Plus a MUTATION arm: the same check against outputs built with `s + 0.05`
  must FAIL.
* **Descriptive, no pass/fail:** the chord-arc deviation (p50 / p99 / max) and the degenerate-candidate count, printed in the verdict.
  Degenerate (zero-length) candidates extrapolate along +x by the shipped machinery's documented fallback; they stay in the fan
  (0.67 % of candidates), and the count of windows whose PICK is a degenerate candidate is reported for every arm.

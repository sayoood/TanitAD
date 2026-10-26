# PREREG_LONHEAD_C_AMENDMENT1 -- FIT rows restricted to the 292 fresh clips that HAVE obstacle labels (Master Mind, sealed 2026-10-26 13:22:42 Berlin)

Amends PREREG_LONHEAD_C (6c81d4b0...). Written BEFORE any LH / DER fit output exists. The only fit output on disk is the mechanics-only
fit on the decel-only no-lead labels (`F:/refcv9_lonhead/mech/_mechanics_only`), which is not a deliverable and is never read for a bar.

## Why
* The PI approved the fresh obstacle.offline download (2026-10-26 ~13:05). It delivered 292 / 300 clips. The other 8 have NO member in their
  chunk zip: a second probe listed the full chunk zip for each (93-99 members, 0 match, 0 containing the id) -- a dataset coverage gap
  (PhysicalAI-AV obstacle.offline covers ~97.4 % of clips), not a transfer failure.
* The lead pass therefore runs on 292 / 300 clips. The 8 clips' windows carry the NO-LEAD label function, which is NOT the trainer's target
  function (K-LABEL-B: that recipe matches the battery on only 1,361 / 4,754 eval139 windows). Fitting on them would train on the wrong target.

## Change
* **FIT rows = fresh windows of the 292 clips whose release row has `clip__has_lead` True.** The 8 clips are excluded whole (all their
  windows), by sha12, recorded in the fit info. Implemented as `lh_fit.py --lead-clips-only` (asserts exactly 8 excluded clips and that
  every kept clip has_lead).
* Everything else in PREREG_LONHEAD_C is unchanged (bars, controls, eval read, committed outcomes). K-LABEL is re-read on the with-lead
  release for the kept clips and must PASS (trainer function == release targets on every kept window).
* Not a selection on outcome: the exclusion is decided by label-source availability alone, before any fit output.

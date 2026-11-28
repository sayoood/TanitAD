# PREREG_RECOVER_NEXT addendum 2: committed next arm for the RAPR outcomes the card's section 4 leaves open (frozen 2026-11-04, before any RAPR checkpoint or rollout exists)

Why this exists. The card's section 4 (sha256 01046fe6...) commits a next arm for four outcomes, and its combiner
(`read/code/recover_next_combine.py`, `next_arm`) reports every other outcome as `UNCOVERED` with the line "the Master Mind names the
next arm; this is a gap in the card's table, reported, not filled". Row 3 also names no setting when Q1 = `neither`, which is the
measured Q1 (RTL and RP01 reads, landed `55cd7f42` and `584954d8`). This addendum fills those gaps BEFORE the data. State at freeze
(MEASURED): the RAPR chain started on box 2 at 2026-11-04 14:32:47 (PI go 14:30) and RAPR_0 is in its launch gate or training; no RAPR
checkpoint, rollout or read exists. Every read rule, bar, control and row of the card and of addendum 1 (sha256 8c62b4da...) is
unchanged; this text only adds rows.

## Committed next arm (applied after card section 4: a card row that matches decides; these rows decide only what it leaves open)
| # | outcome (card section 3 definitions; combiner fields in brackets) | next arm |
|---|---|---|
| E | card row 3 selected and Q1 = `neither` [`row3_setting` = "NO setting is known not to harm"] | **RAPR-P01**: RAPR's frozen argv with `--recover-p 0.1`; 2 runs, seeds 0 / 1. RP01's dose is the one measured setting that kept ADE6 inside its guard at both seeds. |
| F | UNCOVERED, RAPR's closure exceeds R's at exactly one seed, and RAPRC fails D2 on M1 alone | **card row 2's arm** (RAPR at `--recover-p 0.5`, 2 runs). The direction is positive at one seed and the dose is the cheapest lever on it; a larger dose answers more than a replicate of the same dose. |
| G | UNCOVERED, RAPRC fails D1 or D3, or a composite was not computed | no new arm of this family: a composite not computed is a refused read (addendum 1 row B: fix the instrument, re-read); a D1 or D3 failure means the a-priori law costs a family, and the next arm is **card row 4's** level-from-image auxiliary, a PI-decision item (architecture change). |
| H | RAPRC DOMINANT at both seeds but NOT counted (frame-freeze reading not ROBUST or PARTIAL) | no evidence videos and no scale signal (only a counted DOMINANT carries them); report the frame-freeze reading as the blocker; next arm **card row 4** (PI-decision item). |

## What this does NOT do
* It does not authorise a box-2 job: every next arm needs the PI's explicit go (box 2 runs one job at a time, SAFE MODE).
* It does not change which outcome is a success: only a counted DOMINANT (card row 1, addendum 1 row A) is one.
* RAPR-P01 and the p 0.5 arm need their argv written from RAPR's frozen argv by changing one token, checked by token count, and a
  reader `ARM_SPEC` entry with a known answer before a read (instrument work, no box-2 time).

ESTIMATED (not results): each 2-run arm ~3.5 to 4 h on box 2 (the a-priori law builds no deviation table); read ~45 min on the dev box.

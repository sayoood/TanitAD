# PREREG_RECOVER_NEXT addendum: committed next arm for the RTL + RP01 outcomes (frozen 2026-11-04 01:02, before any RTL or RP01 rollout exists)

Why this exists. The PI scoped the box-2 chain to RTL_0 RTL_1 RP01_0 RP01_1 (2026-11-03, "RTL + RP01"); RAPR_0 / RAPR_1 were
not run. The card's section 4 commits a next arm for "any XC DOMINANT" and for three RAPR outcomes, and for nothing else. An
RTL or RP01 outcome that is not DOMINANT would therefore have no committed next arm. This addendum commits one per outcome
BEFORE the data. State at freeze (MEASURED): RTL_0 and RTL_1 finished on box 2 (RUN_RTL_{0,1}.done, 2026-11-04 ~00:50); no
rollout of either exists; RP01_0 is at its gate on box 2. Every read rule, bar, control and row of the card (sha256 01046fe6...)
is unchanged; this text only adds rows to section 4.

## Committed next arm (rows match in this order; the first match decides)
| # | outcome (card section 3 definitions) | next arm |
|---|---|---|
| A | RTLC or RP01C DOMINANT, counted (frame-freeze ROBUST or PARTIAL) | the card's section 4 row 1: evidence videos + the PI scale signal; a third seed pair of that arm on fresh inits. |
| B | a read REFUSES (any control fails) | no new box-2 arm: report the refused control, fix the instrument, re-read. A refusal is a result, not a label. |
| C | no counted DOMINANT, and Q1 = `blend` or `both` (RTL's guards pass at both seeds) | **RAPR-TL**: RAPR_s's frozen argv with `--recover-target logged` and without `--recover-rejoin-steps 10` (the exact RTL difference applied to RAPR); out dir `M2b_s{s}_recft_RAPRTL`; 2 runs, seeds 0 / 1. |
| D | no counted DOMINANT, and Q1 = `dose` or `neither` | **RAPR as frozen** (RAPR_0 / RAPR_1, argv already in `argv/`, unchanged). |

Read of the row C / D arm: the card's section 3 rules with X := RAPR-TL or RAPR; its DOMINANT bar, rows and controls are the
card's. The RAPR secondary (M1 share closed exceeds R's 0.220 / 0.132 at each seed) applies to both. If that arm is not
DOMINANT, the card's section 4 RAPR rows decide (dose up to p 0.5; the harmless target / dose; or the level-from-image
auxiliary as a PI-decision item).

## Why C and D are split this way (rationale, fixed now)
* `blend` / `both`: removing the rejoin blend removed the harm, so the arm that keeps it would test the law with a known harm in it.
* `dose`: p 0.1 removed the harm but also lowers the dose that moves M1; RAPR keeps R's p 0.3 and changes the law instead.
* `neither`: neither change removed the harm, so the law is the remaining pre-specified lever (card section 2, RAPR).

## What this does NOT do
* It does not authorise a box-2 job. Each next arm needs the PI's explicit go (box 2 runs one job at a time, SAFE MODE).
* It does not change which outcome is a success: only row A is DOMINANT, and only row A carries videos and the scale signal.
* The reader needs an `ARM_SPEC` entry for RAPR-TL before a row C read (instrument work, with a known answer on the shipped argv).

ESTIMATED: 2 runs ~3.5 to 4 h on box 2 (RAPR builds no deviation table, so no ~45 min table builds); read ~45 min per arm on the dev box.

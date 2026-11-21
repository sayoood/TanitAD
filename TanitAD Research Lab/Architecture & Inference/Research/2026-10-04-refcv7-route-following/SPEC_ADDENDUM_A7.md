# SPEC addendum A7 — A6's gating control G1 was mis-specified; it is replaced, nothing else changes

**Written 2026-10-04 by the Master Mind after A6 STOPPED at its gate (RESULT_A6.md) and before any A6 capture or arm
number exists.** A6 as registered is recorded as **STOPPED — gate G1 FAILED** (`announced(entries[0])` reproduced the
shipped `nav_command` side on 4,578 / 4,719 = 97.01 % < 99 %; the suppression-named subset passed 68 / 68).

## Why G1 failed (the instrument, not the function) — from RESULT_A6.md, read before writing this
`nav_command` is the builder's rule applied to the FIRST sustained manoeuvre `manoeuvre_sequence[0]` within 35 s;
`nav_30s.entries[0]` is the first TURN within 30 s. The two objects differ by construction on 141 records, all in two
structural classes, 0 unexplained: **curve-first** (70: `seq[0]` is a road curve → FOLLOW, `entries[0]` is the later
real turn) and **beyond-cap** (71: `nav_command` names a turn at 30.1–35 s that `nav_30s` cannot hold). The builder's
rule applied to `seq[0]` reproduces `nav_command` on 4,719 / 4,719. My G1 compared the right function to the wrong
object.

## G1′ (replaces G1; instrument checks — their values were read in RESULT_A6.md, which is admissible for a control)
1. The builder's rule on `manoeuvre_sequence[0]` reproduces `nav_command` on ≥ 99 % of the 4,719 records.
2. On records where `entries[0]` IS `seq[0]` (same start time within 0.05 s) and `nav_command.args.time_s ≤ 30`,
   `announced(entries[0])` reproduces the `nav_command` side on ≥ 99 %, and on 100 % of suppression-named records.

## Semantics fixed now (before any arm number)
* A real turn that follows a road curve **is announced** (a navigation system announces the turn, not the curve).
* The obstacle-pass suppression (PI 2026-08-29) is applied **clip-level**, exactly as the builder applies it.
* Premise correction to A6's motivation: A5's four "curve, not a turn" clips are curve-first — their later turn is
  real and stays announced. Of the 7 GT-turn windows that carried 54 % of A5's T2 turn gain, 5 lie on the 2
  obstacle-pass clips (removed by `announced`) and 2 on curve-first clips (kept).

## Everything else in A6 stands unchanged
Windows (dense capture of the same 139 held-out episodes, sampler seeds 0 and 1), arms (T3a REPORTED, T3 foil, T2a
secondary, T3a-c derangement control), SPEC §5's bar, the B1t fraction, the four-family block and A6's reading rule.

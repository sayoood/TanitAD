# MM ruling on FASTNEG-CENSUS's outcome and its committed next arm (Master Mind, 2026-10-19)

**The outcome stands as registered: INCONCLUSIVE** (A1.6 precedence: FEATURE on DAC + NO DEFICIT on NC). The powered null fired on 0 of 5 replicates per row, so the census is not VOID. It is not re-labelled.

**The committed next arm (one re-run at f = 25 %) is BLOCKED, and it is DEFERRED, not cancelled.**
- S0's bank exists only at f = 10 %. f = 25 % needs 14,984 new tokens (292 logs), about 6.7 dev-box GPU-h (ESTIMATED from S0's measured ~1.6 s/token), plus pod staging under the P6 overlap rule.
- Why defer:
  - The INCONCLUSIVE is the precedence mapping of two per-row classes, each decisive on both seeds. It is not a noise reading.
  - A larger f changes the outcome only if a row's class flips. DAC's coverage slope has an upper bound of 0.0047 against the 0.01 bar, with the fast band already holding 49.7 % of all unsafe rows. NC's fast-band difference is positive on both seeds.
  - The expected information is low (ESTIMATED) against 6.7 GPU-h on a contended GPU (another session's chain, then L1 Step A, then the HN screen).
- It is un-deferred when the dev-box GPU has no higher-value queued item, or if either new arm below returns a result that a row-class flip at f = 25 % would change.

**Two NEW arms, each with its OWN pre-registration written before any of its numbers.** ⛔ They are not promotions of the census's per-row reading; the census verdict remains INCONCLUSIVE.
1. **PICKRULE-AUDIT-NC (0 GPU, CPU minutes).** NC's rows resolve the fast band (no deficit), yet RL-1's head-wise composite lost NC on navtest. So test whether the NC loss arises in how the pick rule COMPOSES the rows, not in the rows themselves.
2. **PRIVGEOM-DAC (CPU).** DAC has a fast-band deficit that more fast labels do not close and a 2-layer head does not close. So test whether privileged drivable-area geometry closes it: a diagnostic only, never a deployed input.

**`H-REFE-FASTNEG-1` (COVERAGE-limited):** NOT SUPPORTED at f = 10 % on either row. It is not formally falsified, because its registered gate read INCONCLUSIVE. ⇒ FASTNEG-BANK stays BLOCKED.

# WORK PACKAGE — pin the navhard-two-stage EPDMS column + ingest the 9 sub-metrics

**From** Master Mind, 2026-08-28 · **Source** LAB-RUN-002 (committed `27021d506`),
package `TanitAD Research Lab/Opponent Analysis/Research/2026-08-28-navsim-v2-camera-lane/`
· **Priority** before ANY NavSim number exists · **Schema** TANITAD_PROGRAMME §3.

## The finding that makes this urgent

⛔ NavSim v2 has TWO "EPDMS" protocols **~30 points apart** — papers' navtest-EPDMS
(~76–88) vs the challenge's navhard-two-stage (~35–56). A single "EPDMS" column lets
the two be compared as one metric: the `overlapping_holdout_se` class (precise about
the wrong thing). And our reference **Drive-JEPA has NO navhard number at all**
(absence, 2 probes) — the 89.0-front-only positioning claim needs re-anchoring to the
camera-only navhard bar (**DrivoR 56.3 EPDMS @ ~40M params**; privileged ceiling 56.6).

## The work

1. `CRITERIA_REGISTRY.json`: pin **navhard-two-stage EPDMS + harness version** as the
   official NavSim column; navtest-EPDMS, if kept, is a SEPARATE non-comparable column
   with a blocking gate against cross-protocol comparison (join it to the existing
   `navsim.estimator_unit` family).
2. Ingest the 9 navhard sub-metrics so any NavSim number arrives with its family
   breakdown mapped onto the binding four-family rule.
3. Run the TanitAD_BenchmarkCriteria skill over the result before staging.

Constraints: stage never push; deliverable manifest; escalate integration to the
Master Mind. The ego-status two-arm entry design is a PI decision (in the 08-28
morning report) — do not resolve it here.

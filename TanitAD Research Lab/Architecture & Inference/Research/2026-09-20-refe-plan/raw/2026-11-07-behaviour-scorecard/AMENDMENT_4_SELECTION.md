# AMENDMENT 4 to BEHAVIOUR_SCORECARD.md (sha256 fac6c7f1...): Master Mind rulings on the token-selection readings

**Date:** 2026-11-08 ~10:00Z.
**Ruled before** any checkpoint was scored on any of these scenes. No threshold, window length or adoption bar changes. Item numbers follow `tsf_bundle2_20261107/AMBIGUITIES.md`. Both candidate lists were computed in the same pass, and both are reproducible byte for byte (`VERIFY_BUNDLE_RESULT.json`).

## The suite list is `SUITE_CANDIDATES_ALT.json`

**Principle for AMB-7 to AMB-10: a slot tests a behaviour only if the event happens while the model is driving.** The rollout starts at t0 − 2 s and the first model call is about 0.55 s later. Under the literal reading, 3 of the 5 lead-braking candidates have finished braking by t0 + 0.5 s, and 2 of the 5 "merge" candidates are crossing traffic at about 90°.

| id | ruling |
|---|---|
| AMB-1 | The PRIMARY reading stands (no lead within `max(4 s*v, 15 m)` AND `TTC_t > 8 s`), as in AMENDMENT_1 A-1. |
| AMB-7 | ALT. The steady-lead run lies inside `[t0, t0 + 15 s]`. |
| AMB-8 | ALT. Lead-braking onset >= t0. |
| AMB-9 | ALT. Stop-go fall >= t0. |
| AMB-10 | ALT. Heading difference to the path <= 30° at entry; crossing traffic is not a merge. |
| AMB-12 | Separate lists per predicate. The table's one "lead braking / stop-go" slot is filled alternately, starting with lead braking: tier 1 (n = 2) takes one of each, tier 2 (n = 4) takes two of each. |
| AMB-13 | Ranks above 5 are the continuation order when QC skips exhaust a slot. Cruise has only 11 logs, so its margin is thin; report any shortfall. |
| AMB-14 | Duplicates are resolved at fill time in this order: cruise, steady lead, lead braking, stop-go, merge/cut-in, then the AMENDMENT_3 junction and light slots. A candidate whose log or token is already used by an earlier-filled slot is skipped, and the skip is logged. |
| AMB-15 | The `tokens` key (154). |
| AMB-2 to 6, 11, 16 to 18 | Accepted as implemented. Selection (AMB-4: along-path relative speed) and scoring (A-3: scalar speeds) use different estimators; selection only picks scenes and scoring alone decides. |

The sim session's QC remains the only reason to skip a candidate apart from AMB-14.

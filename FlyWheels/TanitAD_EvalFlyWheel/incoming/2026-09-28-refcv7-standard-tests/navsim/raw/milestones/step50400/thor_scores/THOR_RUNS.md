# Thor-backend NavSim scores for refcv7 step 50,400 (early reads + cross-backend checks)

Scored on Thor (`backend: thor`, linux-aarch64), under the Master Mind ruling
`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-04-navsim-thor-backend/RULING_BACKEND_POLICY.md`.
The admission record is `…/2026-10-04-navsim-thor-backend/raw/THOR_BACKEND_VALIDATED.json` (verdict
ADMITTED_UNDER_POLICY; the registered exact bar FAILED).

* **These are NOT the milestone's standard artifacts.** The dev-box milestone owns
  `scores_<split>/` and `CLAIMED_ARMS.txt`. Thor names are `r7thor_s50400_<ARM>` (navtest) and
  `<ARM>_THOR` (navhard), so they can never collide with a dev-box name.
* Each arm directory carries:
  * `<label>.thor`: provenance;
  * `<label>.cross_backend.json`: written when the dev box has a PASS for the same arm. It holds the
    per-column diff and the policy checks: 0 discrete flips, per-token |Δ| ≤ 1e-9, split-mean |Δ| ≤ 1e-12.
* navhard arms are scored as 6 whole-group shards and merged (`code/merge_shards.py`). The
  fingerprint fallback was ACCEPTED in the ruling, with atol 1e-9.
* Mixed-backend comparisons, e.g. a Thor arm against the dev-box STOP floor, are admissible only
  under the policy tolerance, and are stated as mixed.

## Banked (2026-10-05)

| split | arm | dir | Thor | cross-backend vs dev box |
|---|---|---|---|---|
| navtest | R7_VMAXOFF (LEGAL) | `navtest/r7thor_s50400_R7_VMAXOFF/` (4 merged shards; shard runs in `navtest/_shards/`) | PASS, PDMS 70.2048 | dev box has no counts yet |
| navtest | R7_A1 / R7_A1_s1 / R7_CEILDECL_d | `navtest/r7thor_s50400_<ARM>/` | PASS | ADMISSIBLE |
| navtest | PRIOR_ha0p | `navtest/r7thor_s50400_PRIOR_ha0p/` | PASS, 58.6613 | dev box FAIL |
| navhard | R7_VMAXOFF (LEGAL) | `navhard/R7_VMAXOFF_THOR/` (6 merged whole-group shards) | PASS, EPDMS 19.8729 | dev box pending |
| navhard | PRIOR_ha0p | `navhard/PRIOR_ha0p_THOR/` | PASS, 13.0040 | ADMISSIBLE |
| navhard | R7_CEILDECL_d | `navhard/R7_CEILDECL_d_THOR/` | PASS, 20.4844 | dev box FAIL |

LEGAL-row summaries are the suite's own parser run on a staged copy:
`navtest/summary_navtest_vmaxoff_legal.THOR.json`, `navhard/summary_navhard_vmaxoff_legal.THOR.json`,
`navhard/pairs_navhard_vmaxoff_legal.THOR.json`.

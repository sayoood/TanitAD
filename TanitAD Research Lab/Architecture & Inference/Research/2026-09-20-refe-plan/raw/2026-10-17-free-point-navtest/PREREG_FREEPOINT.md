# PRE-REGISTRATION — navtest read of the zero-training scorer points (PI-approved 2026-10-17). Written by the Master Mind BEFORE any navtest number.

**Decision.** PI, 2026-10-17: *"Yes, on the idle A40"* — read the free logit-mixing point on navtest while the safety-objective arm is built.
**Why it was widened before data (disclosed).** Re-reading `raw/2026-10-14-rls1b-attribution/dose.json` while writing this: its mixture numbers are
against **C0-A**, and C0-A itself reads **PDMS −0.57 [−1.26, +0.09], TTC +0.83 [+0.13, +1.58] against the DEPLOYED base** (`pick_changes_vs_base`,
base = `r0 step0`, `dose_rls1b.py:40`). **R0-B against the deployed base reads PDMS +0.59 [+0.14, +1.02], TTC −0.17 [−0.71, +0.38]** (not separated).
So the mixture at alpha 0.25 is roughly FLAT against what is deployed, and the deployment-relevant candidate is R0-B itself. Both are read; the cost
is the same harness. (RL-val, ESTIMATED, selection tier; these motivate the read and decide nothing.)

## Points (fixed; no sweep, no picking)
| id | scorer | construction |
|---|---|---|
| BASE | the deployed scorer | `r0` `step0` (dose_rls1b.py:40) |
| C0A | RL-S1 control, arm A, update 1000 | `c0` `update1000` `A` |
| R0B | RL-S1 R0, arm B, update 1000 | `r0` `update1000` `B` |
| M25 | logit mixing at alpha = 0.25 | `s_C0A + 0.25 (s_R0B − s_C0A)` on every output (dose_rls1b.py:4), the FULL variant (not `prog`) |
Each picks one of the deployed model's 64 proposals per token by the trainer's own `sel_score` on its own (mixed) logits.

## Tier, estimator
**T1** = NAVSIM v1.1 PDMS on the FULL navtest (12,146 tokens, 136 logs), the picked proposal scored exactly as the banked P2/RL-S1 navtest reads score
a pick. Paired log-cluster bootstrap, 10,000 resamples, seed 20260927; it answers *another draw of LOGS* only (one training seed per state;
deterministic selection, so no inference variance). PDMS and every sub-score (NC, DAC, EP, TTC, C, DDC) x100.

## Committed outcome rows (vs BASE = the deployment comparison)
A point is **DEPLOYABLE-SAFE** iff, against BASE: PDMS lower bound > 0 AND NC, DAC and TTC are each NOT separated below 0 (CI upper bound >= 0).
- **FP-R0B**: R0B is DEPLOYABLE-SAFE ⇒ the banked RL arm R0 is already a safe gain over the deployed scorer on navtest. Reported to the PI
  (deploying is the PI's call); the safety-objective arm's navtest bar ADDS "beats R0B against BASE".
- **FP-M25** (R0B not, M25 yes): M25 is the safe point; the same consequence with M25.
- **FP-NONE**: neither is DEPLOYABLE-SAFE ⇒ the safety-objective arm proceeds exactly as pre-registered.
- Also reported (lever view, no row): R0B − C0A and M25 − C0A, the comparison dose.json made on RL-val.
**Families:** LONGITUDINAL (EP, TTC, C, plus any longitudinal component the harness emits), LATERAL (DAC, DDC, plus lateral components),
TACTICAL (rows without an interval if the harness has them; else "no instrument at this tier", with the reason), STRATEGIC = PAUSED (PI, 2026-09-19).

## Identity control (before any point is read)
BASE read through the harness must reproduce the banked deployed-model navtest PDMS for the same proposals and tokens (exact, or the banked
read's own tolerance), and C0A scored twice must be bit-identical.

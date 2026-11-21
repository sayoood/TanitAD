# PREREG H-R9-NAV-BRAKE-1 — AMENDMENT 1 (pre-data)

Registered 2026-10-19 ~23:45 Berlin, BEFORE any count was computed (the first run died at load with
`KeyError: plan_full_nav_zero` and wrote nothing). Sealed by `PREREG_NAV_BRAKE_AMENDMENT1.sha256`.

**Fact (MEASURED by archive listing):** the battery banked the nav-condition plans (`plan_full_nav_zero`,
`_nav_flipped`, `_nav_shuffled`) for inference seed 0 ONLY — 139/139 episodes at seed 0, 0/139 at seed 1,
on both the 40k and the 30k EMA trees. Seed 1 carries `plan_full_nav_true` and `nav_cmd` only.

**Amendment.** The prereg's "at BOTH seeds" cannot be evaluated. It is replaced, for the intervention
clauses only, by "at seed 0 of the 40k tree AND at seed 0 of the 30k tree" (two checkpoints of the same run,
same inference seed). This is a replicate across CHECKPOINTS, not across inference seeds; the measured
inference-seed floor on this rig is small (E1 0.00046 m, `raw/compliant_reread/trees/step40000ema_unc.json`),
which is why the substitution is admissible, and it is stated as a caveat. The label is reported as
SINGLE-INFERENCE-SEED. Seed 1 still enters the DESCRIPTIVE r8-vs-r7 strata (nav_true only).

* NAV-DRIVES-BRAKE iff dFD(nav_zero) on TURN ∧ GT-cruise is separated > 0 at 40k-s0 AND 30k-s0, AND its point
  exceeds the FOLLOW stratum's at both.
* NAV-NOT-THE-LEVER iff it is separated > 0 at neither.
* MIXED otherwise.

# QUALITY SYSTEM — maximize achievements, minimize false positives

`Created 2026-08-22 (TANITAD_PROGRAMME.md §6, expanded per the PI's request for
CONCRETE proposals). The goal is not finding mistakes; it is maximizing durable
assets. The main agent self-corrects too often — the remedy is moving
correctness UPSTREAM of execution, so errors are prevented or caught by
machinery, not by after-the-fact vigilance.`

## The diagnosis, from this week's measured record

One day (2026-08-22) produced: 4 estimator bugs in one probe family, a λ
confound that invalidated two sweeps, a gate that could never rule, a σ/σ²
statistic inversion, and a backward-compat break. **Every one was caught — but
each by after-the-fact checking, costing a re-run.** The common causes:

| root cause | count | upstream remedy |
|---|---|---|
| tuning/normalising on the scored data | 3 | spec template forces split declaration BEFORE run |
| statistic/scope ambiguity (σ vs σ², per-step vs accumulated) | 2 | named-statistic rule + pinning test |
| success asserted from exit codes / markers | 2 | verify-by-content helpers, not ad-hoc checks |
| new knob silently changing a second variable (λ with n) | 1 | one-variable check in the spec template |
| code path divergence (flag reaches log but not gate) | 1 | end-to-end test at introduction, not after |

## Concrete proposal 1 — the SPEC template is executable, not prose

`stack/scripts/new_workpackage.py <area> <slug>` scaffolds the §3 schema with a
SPEC.md whose front-matter is MACHINE-CHECKED by a pre-launch preflight:

```yaml
hypothesis: H-RANK-5           # must exist in GOALS_AND_CLAIMS.md
one_variable: sigreg_accum     # the ONLY thing that differs between arms
held_constant: [lambda, seed, corpus, steps]   # preflight greps the launch cmd
success:  "participation CI lower bound > baseline's upper bound"
failure:  "CI overlap after 2000 steps"        # BOTH outcomes, committed
controls: [constant, pixel_floor, regression_arm]
splits:   {fit: "clips 0..11", val: "carved from fit", test: "12..23, never tuned on"}
```

The preflight REFUSES launch if: an arm's command differs from baseline in more
than `one_variable`; a control is missing; the hypothesis ID is unknown. This
mechanises exactly the checks that failed by hand (the λ confound would have
been refused at launch: `sigreg_accum` changed the effective λ).

## Concrete proposal 2 — instruments are code, probes are not scripts

Promotion rule: **any probe run twice moves from scratchpad into
`stack/tanitad/eval/` with tests, in the same week.** Scratchpad scripts may
explore; only stack instruments may produce QUOTABLE numbers. An instrument
ships with:
- a **fixture where its verdict must be TRUE** (physics: the ego probe must read
  kinematics ≈1),
- a **fixture where it must be FALSE** (the deliberate-regression arm),
- a **fixture where its failure mode ACTIVATES** (the σ-inversion fixture:
  55 % top-1 energy + thin tail → the wrong statistic inverts, the right one
  doesn't).
This is TDD adapted to measurement: the test isn't "code runs", it's "the
instrument discriminates".

## Concrete proposal 3 — the two-key rule for quotable numbers

A number enters a report only with BOTH keys: (1) artifact path (raw JSON in
`raw/`), (2) evidence class + tier stamp. `/TanitAD_Review` greps reports for
naked numbers. MODEL_REGISTRY.md and raw eval JSON remain the only quotable
sources — unchanged, now enforced by review instead of memory.

## Concrete proposal 4 — false-positive taxonomy in the review skill

/TanitAD_Review checks each finding against the measured failure classes
(RETRACTION_LOG root-cause classes) BEFORE it is reported upward: scope errors
(df/free/step_s family), estimator selection on scored data, statistic
ambiguity, echo tests, content-vs-existence. A claim matching a known class
needs the class's specific control before it may be asserted. This converts the
RETRACTION_LOG from a diary into a checklist.

## Concrete proposal 5 — definition of DONE (per work package)

DONE = staged in the repo + raw artifacts banked + RESULT.md with evidence
classes + GOALS_AND_CLAIMS.md updated + tests green + (if procedure-shaped) a
skill drafted. Anything less is IN PROGRESS. "It ran and looked good" is not a
state.

## Concrete proposal 6 — self-correction budget

When a session's corrections exceed its shipped assets, STOP and convert: write
the failing pattern into (a) a test, (b) a trap entry, or (c) a skill — then
continue. Correction without conversion is how the same mistake recurs at the
next full context window.

# PRE-REGISTRATION — MM-E5: is drift POSITIVELY COUPLED to prediction quality?

**Written 2026-08-29 ~22:35, BEFORE the MM-E4 L2/L3/L4 arms report** (L2 is mid-training,
L3/L4 unstarted — precedence is in the git history) · **Author** Master Mind · **PI go**
2026-08-29 ("do 1, 2 and 3") · **Hypothesis** MM-E5 · **Tier** T0-DIAGNOSTIC.

## The claim under test

The programme has treated drift as a PATHOLOGY to minimise. Two measurements now suggest
it is instead COUPLED to the thing we want:

| lever | drift | prediction (cos) | scale |
|---|---|---|---|
| EMA teacher (E-DEC-69) | **+3.6 %** | **+24.5 %** | 30k |
| innovation-SIGReg (MM-E4 L1) | **−25.0 %** | **−95.6 %** | 2k |

Both move the two quantities in the SAME direction. If that is mechanism-independent, the
programme's objective is mis-stated: the target is not minimum drift but **maximum
prediction AT MATCHED DRIFT** — a frontier, and the dissociation blocker must be restated
as "we sit at a bad point ON the frontier", not "we have a defect to remove".

⚠️ n = 2, different mechanisms, different scales, no shared axis. This is a HYPOTHESIS.

```yaml
hypothesis: MM-E5
test_1 (free, already running): the MM-E4 arms L2 (frozen-teacher) and L4 (azimuthal
  crop) constrain the TARGET, not the innovations — a mechanism family disjoint from L1.
  Their (drift, cos) pairs are read against the same base o14fut10.
test_2 (the curve, PI-approved 2026-08-29): a DOSE sweep of the innovation constraint —
  `--o6-innovation` with `--w-o6` in {0.01, 0.03} added to the measured {0.1 = L1, and
  0 = base}, all else the o14fut10 line, 2k, ~35 min/arm. Four points on one axis with
  ONE mechanism varying — the clean coupling test L1 alone cannot give.
reads: (drift r, cos_ctr, nrmse) per arm, all via the shared rig (eval-mode safe, MM-C2)
outcomes (committed):
  COUPLED:      across test_2's four points, drift and cos move MONOTONICALLY together
                => the objective is restated as the frontier; the programme reports
                "prediction at matched drift" from then on, and MM-E4's remaining
                levers are re-read as frontier POSITIONS, not pass/fail
  DECOUPLED:    a point exists with lower drift AND cos within the seed band of base
                => coupling refuted; that point's weight is the operating point and the
                minimise-drift framing survives
  L1-SPECIFIC:  test_1's arms (different mechanism) do NOT trade along the same line
                while test_2's do => the coupling is a property of innovation
                constraints, not of drift — report as such, do not generalise
  NON-MONOTONE: the sweep is not monotone => no verdict; report the points and stop
                (a non-monotone dose curve means an uncontrolled variable, C136 family)
cost: 2 new arms x ~35 min Thor (~1.2 h) + probe reads on the 4060; test_1 is FREE
      (those arms are already queued in the E4 chain)
⛔ scheduling: yields to the v7f line — the sweep runs in the gap before the B1
      epcache build; the tau-ramp arm (D-EMA-ADOPT's gate) outranks it if they collide.
```

**Why this is worth 1.2 h**: if COUPLED, every anti-drift result in the programme —
including tonight's — is re-interpreted, and the paper's §14 open problem is restated.
If DECOUPLED, we get the innovation constraint's usable operating point, which is what
L1 failed to find by testing only one weight. Both outcomes are decision-grade.

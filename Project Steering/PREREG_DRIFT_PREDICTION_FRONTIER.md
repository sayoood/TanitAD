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

---

## OUTCOME — 2026-08-30 ~01:30, from MM-E4's five arms (MEASURED, T0-DIAGNOSTIC)

**Test 2 is VOID and CANCELLED** (its mechanism failed its own regression control —
MM-E4's verdict). **Test 1 answered**, and with the shuffled control it answers more than
it was designed to:

| arm | mechanism | drift Δ | cos | coupled? |
|---|---|---|---|---|
| L1 innovation-SIGReg | dynamics constraint | −25.0 % | 0.0108 | yes |
| L2 frozen-teacher | **target** constraint | −31.6 % | 0.0377 | yes |
| ctrl shuffled-g | **arithmetically meaningless** | −20.1 % | 0.0066 | yes |
| L3 L1+L2 | both | −15.5 % | 0.0816 | yes |
| L4 azimuthal crop | target **view** | **+4.5 %** | **0.2105** | n/a — no drift change, prediction kept |
| EMA (30k, E-DEC-69) | target source | **+3.6 %** | **+24.5 %** | yes, other direction |

**Verdict: COUPLED — but the coupling is TRIVIAL, not a frontier.** Every arm that moved
drift down moved prediction down, across four mutually unrelated mechanisms *including a
meaningless one*; the two arms that moved drift UP (EMA, and L4 slightly) kept or improved
prediction. A relationship that survives replacing the mechanism with noise is not a
trade-off between two capabilities — **it is one quantity seen twice.**

⭐ **THE REFRAMING, and it is the night's most consequential claim.** Drift measures the
predictability of Δz from z_t. Prediction quality measures whether the predictor can
produce Δz. **These are not opposed goals in tension; they are largely the same property
of the latent's temporal structure.** Destroying that structure lowers drift *because* it
destroys predictability. ⇒ **High drift is not, by itself, a pathology — it is partly what
a temporally-structured latent looks like.** The programme's real question was never "how
do we lower drift" but **"how much of the predictable structure is SELF-REFERENCE versus
ENVIRONMENT"**, and no instrument here separates those. That separation is the actual open
problem, and it is not the one MM-E4 was built to attack.

⛔ **What this does NOT say:** that the drift attractor is harmless. The T1 evidence stands
— arms sitting at 0.67 drift do not drive (ADE ~14.5 m vs the 0.5352 CV floor). It says
the *statistic* cannot be optimised directly, and that lowering it without a mechanism
that distinguishes self-reference from environment buys nothing.

**Consequences adopted:** (1) ⛔ no further arm is funded on "lower drift" as its primary
read; (2) the pre-committed frozen-teacher lever (§14.10) is **already spent** — L2 was it,
and it failed; (3) the successor question is an INSTRUMENT question: build a read that
separates self-referential predictability from environment-driven predictability, and only
then re-open the attack. Candidate: condition the drift ridge on scene content and compare
the residual against a scene-shuffled control — the E-DEC-63 rig already has the parts.

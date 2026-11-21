# E1e-B RESULT + the `lam_replay` axis is CLOSED — with a trade-off curve, not a null

**MEASURED 2026-07-28**, pod3. Artifact `pod3:/workspace/e1e/e1e_B_frontier.json`.
**Estimator: paired episode-cluster bootstrap** (`taniteval/ci.py`, B=2000), 44 held-out episodes /
43 clusters at K=185. `overlapping_holdout_se` used nowhere. Pre-registration `PRE_REGISTRATION_E1E.md`
(`a7a2781`), committed before either arm existed.

## 1. E1e-B (λ=8): VERDICT **BOUND**, 0/4 success — P1∧P2 **4/4**, guardrails **0/4**

| step | `dep_overall` Δ | `dep_junction` Δ | open-loop ADE@2s Δ [lo, hi] | P1 | P2 | Ga |
|---|---|---|---|---|---|---|
| 1000 | −0.2378 | −0.2748 | +0.0872 [+0.047, +0.133] | ✅ | ✅ | ❌ |
| 2000 | **−0.2891** | −0.2685 | +0.0696 [+0.038, +0.106] | ✅ | ✅ | ❌ |
| 3000 | −0.2674 | **−0.3468** | +0.0531 [+0.024, +0.086] | ✅ | ✅ | ❌ |
| 4000 | −0.2752 | −0.3270 | **+0.0500 [+0.023, +0.083]** | ✅ | ✅ | ❌ |

⭐ **The risk this arm was run to test did NOT materialise: P1 survives at λ=8.** Weighting replay 8×
against the closed-loop term does **not** destroy the corridor-departure gain — it fires at every
checkpoint, as it did at λ=3.

**Ga is not met and the CI says it is not reachable on this axis.** The lower bound walked
0.047 → 0.038 → 0.024 → **0.023 and flattened**. Reaching Ga needs it to touch 0; the remaining
distance is not closing, and buying it with more λ costs P1.

## 2. ⭐ THE AXIS PRODUCED A TRADE-OFF CURVE — that is the deliverable

| arm | best closed-loop | at open-loop cost | lowest open-loop | at closed-loop |
|---|---|---|---|---|
| **λ=1** (E1c) | **−0.4407** | +0.2158 | +0.1893 | −0.4205 |
| **λ=3** (E1e-A) | −0.3911 | +0.0958 | +0.0891* | — |
| **λ=8** (E1e-B) | −0.2891 | +0.0696 | **+0.0500** | −0.2752 |

*(E1e-A's +0.0891 is an in-training gate reading; its frontier minimum is +0.0958.)*

**Monotone in λ on both axes, with no crossing:** each arm is Pareto-optimal in its own region and
**none dominates another.** `lam_replay` is therefore a **well-behaved, calibrated control on the
closed-loop / open-loop trade**, not a failed knob — the program can now *choose* an operating point
rather than accept whatever CL-SFT produces.

⚠️ **What this does NOT license.** No point satisfies the pre-registered success criterion, so **there
is still no D-A deliverable**. A curve one can choose along is not the same as a checkpoint that
passes the gate, and the distinction is the whole reason the criterion was fixed in advance.

## 3. The axis is CLOSED — three one-dimensional levers, three BOUNDs

| lever | experiment | verdict | why it cannot be pushed further |
|---|---|---|---|
| **training time** | E1c | BOUND | open-loop cost **plateaus** from step 2250 (8 points inside ±0.02) |
| **weight space** | E1d | BOUND | interpolation path is **separated-WORSE at 5 consecutive interior α** — the endpoints are not linearly mode-connected (**C52**) |
| **loss weighting** | E1e-A/B | BOUND | λ sets the **asymptote**, preserves P1, but Ga's lower bound flattens at **+0.023** |

⛔ **A finer λ grid is NOT run** — pre-committed as inadmissible before any of this existed, exactly so
a near-miss (+0.023) could not tempt an unbounded sweep. Recording that the temptation was real and
was declined.

## 4. What follows, per the pre-registration

> *"The remaining hypothesis is that the **TARGET is wrong, not its weight**."*

E1d already measured the asymmetry that names the target: **junction recovery is cheap and monotone**
(better at every α, never separated-worse) while **overall-corridor recovery is expensive and
barrier-crossing**. The buffer supervises *all* recoverable pre-failure states.

⇒ **E1f — junction-restricted buffer.** Feasibility already MEASURED
(`…/2026-07-28-e1f-junction-buffer/`): the buffer carries `dpsi`, filterable at the evaluator's own
10° threshold, **733 of 3,537 records = 20.7 %** across 362 episodes.
🔴 **With its risk priced in advance:** a 4.8× smaller buffer means **~87× reuse per record** vs E1c's
~18× — a memorisation regime, the same bound that stopped GATE-1. E1f's pre-registration must fix a
held-out-**by-episode** check, a reuse-matched run length or batch, and the honest alternative that
**the junction gain may not survive restriction at all** — α-interpolation moved a model trained on
everything, which is not the same object as one trained only on junctions.

## 5. Honest bounds

- 43 episode clusters at K=185, **6 junction windows** — `dep_junction` CIs are wide; not a precise
  effect size.
- `lam_replay` and `lam_cl` are **not independent**: only their ratio matters up to the LR schedule.
  Three arms are **three points on one axis**, not three levers.
- All arms are **fine-tunes of one base checkpoint**. Any operating point chosen from the curve owes a
  from-scratch confirmation before it enters a headline.
- Base reproduction control passed **exactly, on all three runs** (`ADE 0.4747 / dep 0.5877 /
  junc_dep 0.8414 / peakXTE 38.944 / OODpeak 1.266`), which is what makes the cross-arm curve a
  comparison rather than three separate readings.

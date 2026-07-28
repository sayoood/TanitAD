# DECISION BRIEF — is **Ga** the right guardrail for closed-loop fine-tuning?

**For: Sayed. Prepared 2026-07-28.** One page of decision, one page of evidence.
Every number **MEASURED**, estimator **paired episode-cluster bootstrap** (`taniteval/ci.py`, B=2000),
44 held-out episodes / 43 clusters at K=185. `overlapping_holdout_se` used nowhere.
The held-out split is **content-verified clean at sensor level** against REF-C base's *own* training
corpus (0/44 episodes, 0 shared frames), so none of this is a leak artifact.

---

## 1. The decision, in one sentence

**Ga** = *open-loop ADE@2s must not be CI-separated-worse than base*. It is a **strict
non-regression**. Four closed-loop fine-tuning experiments have now failed it — **while three of them
delivered large, CI-separated closed-loop gains**. The question is whether Ga, as written, is the
right condition for this class of arm.

**This brief does not recommend relaxing it.** It puts both magnitudes side by side, because until
today only one of them was ever reported.

---

## 2. What the arms buy (closed loop, K=185 ≈ 18.5 s)

**The base REF-C arm diverges catastrophically**: peak cross-track excursion **38.944 m** against a
**1.75 m** corridor; mean **14.306 m**. Over 18.5 s it does not leave the lane — it leaves the road.

| arm | corridor departure Δ | peak_xte base→ft | Δ peak_xte |
|---|---|---|---|
| **E1c** λ=1 | **−0.4407** | **38.944 → 3.042 m** | **−35.90** [−49.33, −24.12] ✅ |
| E1e-A λ=3 | −0.3911 | 38.944 → 4.502 m | −34.44 ✅ |
| E1e-B λ=8 | −0.2891 | 38.944 → 7.790 m | −31.15 ✅ |
| E1f junction | ~0 (P1 failed) | 38.944 → 15.012 m | −23.93 ✅ |

**E1c cuts peak excursion 92 %.** All separated, every arm, on both `peak_xte` and `mean_xte`.

## 3. What they cost (open loop)

| arm | ADE@2s Δ | lateral | longitudinal | lateral p90 |
|---|---|---|---|---|
| E1c | +0.1947 | +0.1611 | +0.1838 | **+0.3945** (2.45× the mean) |
| E1e-A | +0.0990 | +0.0971 | +0.0926 | +0.1849 |
| E1e-B | **+0.0500** | +0.0500 | +0.0520 | +0.0756 |
| E1f | +0.0555 | +0.0460 | +0.0702 | +0.0422 (not separated) |

The cost is **real**, **separated on both axes in every arm**, and **tail-heavy laterally**.
Best Ga lower bound ever observed across all arms and steps: **+0.020** — it never touched zero.

**Scale of the trade at E1c: ~0.2 m open-loop concession against ~36 m of peak excursion — order
167 : 1.** Those are not commensurable quantities, and the gate weighs only the first.

## 4. Why this is a judgement and not another experiment

Four independent one-dimensional levers, four BOUNDs, four *different* structural reasons:

| lever | experiment | why it stops |
|---|---|---|
| training time | E1c | open-loop cost **plateaus** from step 2250 (8 points within ±0.02) |
| weight space | E1d | α-path is separated-**worse** at 5 consecutive interior points; endpoints not linearly mode-connected (**C52**) |
| loss weighting | E1e-A/B | λ sets the **asymptote**; Ga's lower bound flattens at **+0.023** |
| the target | E1f | junction-only supervision **halves** junction recovery — restriction gives *less of everything* (**C55**) |

**A fifth lever is not indicated.** The space of "configure CL-SFT differently" has been searched
along its four natural axes and the guardrail was never approached.

## 5. What the gain actually is (characterised, not assumed)

Knot-ADE spans **0.5–2.0 s**; corridor departure spans **18.5 s**. The arms take a **slightly
different early line** (E1c: +0.2154 m extra lateral deviation at ≤2 s) and are **dramatically more
stable long-horizon**. The frame confound I initially feared here **does not exist** — prediction and
ground truth share origin *and* rotation, and every window re-initialises at the recorded pose.

⚠️ **Not established:** whether that early deviation is a *recovery-oriented* line or merely tolerated.
No per-window traces exist; it needs a capture re-run. **It is an input to this decision, not a
consequence of it** — say the word and I'll run it on pod3.

---

## 6. The options, plainly

**A — Keep Ga as written.** Closed-loop fine-tuning is then closed as a direction on this
architecture; the four arms stand as a characterised negative result. *Cost:* forgoes a measured
92 % reduction in peak excursion.

**B — Replace the strict non-regression with a bounded one** (e.g. open-loop ADE may degrade by up to
X m, or the lateral p90 may not exceed Y). *Requires:* choosing X/Y — a statement about what the
programme will trade, which only you can make. **Pre-register before re-adjudicating**, or the four
existing arms become a post-hoc search over thresholds.

**C — Change what Ga measures.** The lateral **p90** is arguably the safety-relevant statistic and the
mean is not (the artifact's own guidance says gate on p90/p95). *Requires:* the same pre-registration
discipline.

**D — Decide it is undecidable on this evidence** and commission the recovery-vs-tolerated capture
first. ~1 h on pod3, no new training.

⛔ **No arm should be run against Ga until this is settled** — otherwise the threshold is being
selected after seeing the results.

---

## 7. Provenance

`…/incoming/2026-07-28-e1e-replay-weight/` (E1e-A, E1e-B, axis closure) ·
`…/incoming/2026-07-28-e1f-junction-buffer/` (E1f result, lateral/longitudinal split, XTE magnitude,
resolved knot-ADE question) · `RETRACTION_LOG.md` C52, C55.
⚠️ **Reporting failure disclosed:** `peak_xte`/`mean_xte` were in every artifact since E1c and I
reported only departure rates and open-loop ADE for four experiments. This decision sat with you
without §2 in front of it.

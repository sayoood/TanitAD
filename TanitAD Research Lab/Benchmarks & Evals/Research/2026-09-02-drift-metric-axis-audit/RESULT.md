# RESULT — E-BE-DRIFT-1

`2026-09-02 · Research Lab · Benchmarks & Evals · 0 GPU · raw/drift_axis.json`

**Tier:** `T0-DIAGNOSTIC` — inherited from the source reads. ⛔ Not a driving
claim; not comparable to any T1 number.
**Controls:** the `constant (control)` column reads **exactly 0.0 on all 16 banked
reads** (0 failures); `z_t` is itself the positive control, per E-DEC-59.

**Headline.** ⛔ **Backlog row FS-1's premise is wrong about our instrument** —
our "drift" is not a start-vs-end trajectory measure, so porting WorldRoamBench's
segment-based metric would not correct it. ⭐ **But FS-1's underlying concern is
valid, relocated to the `k` axis** — and the banked data sits in exactly the
configuration where that blind spot hides: **drift is flat across a 15× horizon
change on all three arms and both PCA bands** (−2.39 % to +0.38 %).

---

## F1 — ⛔ WHAT OUR "DRIFT" ACTUALLY MEASURES, FROM ITS OWN SOURCE

FS-1 states:

> *"Our drift is read **start-vs-end**, which by their construction cannot see
> non-monotonic mid-sequence collapse."*

`mm-e19-probes/latentmotion.py` (E-DEC-59) says what it measures, in its own
docstring:

> *"TARGETS — Δz = z_{t+k} − z_t projected on its top PCA directions"*
> *"COLUMNS: `z_t` — the **DRIFT** baseline, and the POSITIVE CONTROL"*

⇒ **Drift is the `r` of a k-fold-fit linear probe predicting `Δz` from `z_t`** — a
**per-window predictability statistic at one fixed `k`**. It is the operational
form of the question `V7_LAUNCH_GATE.md` states in prose (*"drift asks how much of
Δz is predictable from `z_t`"*).

**There is no sequence being reduced to its endpoints**, so the specific defect
FS-1 names cannot apply, and a segment-based port would not change the number.
*Root-cause class: a metric characterised by its NAME rather than by its source —
the same family as the `step_s` and `df` scope errors.*

## F2 — ⭐ FS-1's CONCERN SURVIVES, ON A DIFFERENT AXIS

The concern behind FS-1 is general and correct: **a single-number summary cannot
see non-monotone structure inside whatever it summarises.** Our drift summarises
over `k` — it is reported at **one** horizon. Two arms with identical drift at
k = 4 could differ arbitrarily at intermediate k.

⇒ FS-1 is not refuted; it is **mis-targeted**. The blind spot is real and lives on
the `k` axis, not on a sequence axis.

## F3 — ⭐⭐ MEASURED: DRIFT IS FLAT ACROSS A 15× HORIZON CHANGE, ON EVERY ARM AND BOTH BANDS

`raw/drift_axis.json` → `k_axis`. Six independent (arm, PCA-band) pairs:

| arm | PCA band | k 4 → 60 (15×) | relative change | n_rows |
|---|---|---|---|---|
| `postrain30k` | 0–8 | 0.6674 → 0.6625 | **−0.73 %** | 7,680 → 3,200 |
| `postrain30k` | 8–16 | 0.6712 → 0.6585 | −1.89 % | 7,680 → 3,200 |
| `k8clip05p30k` | 0–8 | 0.6618 → 0.6460 | −2.39 % | 7,680 → 3,200 |
| `k8clip05p30k` | 8–16 | 0.6598 → 0.6529 | −1.05 % | 7,680 → 3,200 |
| `k60clip05p30k` | 0–8 | 0.6331 → 0.6277 | −0.85 % | 7,680 → 3,200 |
| `k60clip05p30k` | 8–16 | 0.6510 → 0.6535 | **+0.38 %** | 7,680 → 3,200 |

⇒ **`Δz` over 0.4 s and `Δz` over 6.0 s are equally linearly predictable from
`z_t`** — across three arms that differ in horizon, clip and initialisation.

⭐ **Why that is a substantive finding, not a curiosity.** If the latent were
**transporting** a scene through time, the predictability of `Δz` from `z_t` should
**degrade with horizon** — the further ahead, the less the current state should
determine the change. It does not degrade at all over a 15× span. That is
independent support for **P5/L3**'s reading (*"the predictor is not transporting
the scene, it is restating it"*), measured on an axis P5 does not currently use.

⚠️ **Two caveats that travel with it, both load-bearing.**
1. The target is `Δz` projected on the top PCA directions **of that k**, so k = 4
   and k = 60 are not literally the same target vector. The claim is about the
   predictability of *each horizon's own leading `Δz` structure*, not about one
   fixed quantity.
2. ⛔ **Two points cannot establish monotonicity** — which is exactly F2's point.
   Equal endpoints are the *worst* case for detecting a non-monotone middle. This
   result therefore **motivates** the k-sweep; it does not substitute for it.

## F4 — ⚠️ "DRIFT" NAMES AT LEAST THREE DISTINCT QUANTITIES IN THIS PROGRAMME

| name | file:line | what it is | null |
|---|---|---|---|
| latent **drift** (the v7 campaign's) | `mm-e19-probes/latentmotion.py` | `r` of `Δz` on `z_t`, k-fold fit | (probe `r`) |
| `lat_drift` | `taniteval/taniteval/control.py:959`, registered `:1036`, described `:1063` | lateral control error, **m per m (steering error)** | 0.0 |
| `max_drift` | `stack/tanitad/eval/goal_provenance.py:310` | a **determinism** check over goal provenance | 0.0 |

⇒ An unqualified "drift" in a report, the register or the paper is **ambiguous**,
and the three have different units and different nulls. Every drift number should
name which one it is. *(This is the vocabulary-drift class the review discipline
exists to catch.)*

---

## ⇒ What FS-1 should become

**Not a port of a foreign metric — a `k`-sweep of the instrument we already have
and trust.**

* **Cheap:** `latentmotion.py` exists, runs on the dev-box 4060, and carries a
  control that reads **exactly 0.0** on all 16 banked reads. Adding k values is an
  argument, not an implementation.
* **Pre-committed read**, both outcomes:
  * drift **monotone or flat** in k ⇒ the single-k summary stands, and every
    banked drift number keeps its meaning;
  * ⛔ a **dip or peak at intermediate k** ⇒ every single-k drift number we hold is
    **incomplete**, and the drift column must be reported as a curve.
* **Suggested grid:** k ∈ {1, 2, 4, 8, 15, 30, 45, 60} on the two arms that
  already have both endpoints, both PCA bands, same clips.
* ⚠️ Report `n_rows` per k — it falls with k (7,680 → 3,200 between the banked
  endpoints), and a curve that hides a shrinking sample is not a curve.

### And a genuine gap FS-1 half-identified, which should be proposed separately

We appear to have **no rollout-trajectory drift metric** of the WorldRoamBench kind
— a per-step divergence along an imagined rollout. That is a real absence and a
reasonable thing to build; it is simply **not a correction to the metric FS-1
names**, and conflating the two would have re-scored a number that does not have
the defect while leaving the actual gap unfilled.

⚠️ **Stated as "not found at the searched locations", not as absence.** I probed
`stack/tanitad/`, `taniteval/taniteval/` and `stack/scripts/` by Python (not
`grep`, which under-reports on this mount) and read the metric's own source. Per
the two-probe rule that is not enough to assert non-existence; a third probe and
the tool that owns rollout scoring should confirm before anyone builds.

## Limits, stated

* **No interval.** These are probe `r` values with `t` statistics from the source
  reads; there is **no episode-cluster bootstrap** on this instrument, so the
  −2.39 %…+0.38 % spread carries no CI. Per the estimator rule, none is quoted
  rather than a wrong one. ⚠️ The consequence is that "flat" is a statement about
  point estimates across six pairs, not a tested null.
* All reads are the **HELD-OUT** split, 80 clips, 100 frames/clip, at step 30,000.
* The k = 60 reads use **3,200** rows against k = 4's 7,680 — a real difference in
  precision (`t` falls ~147 → ~86 accordingly), and it is printed with every row
  above rather than smoothed over.
* ⛔ The `k8_attribution` source files store their arm under the internal key
  `"k60clip05p30k"` (the `--arm-name` default documented in that package, §6a).
  `code/drift_axis_audit.py` remaps it by an explicit named constant; reading by
  the name in the file would have swapped two arms.

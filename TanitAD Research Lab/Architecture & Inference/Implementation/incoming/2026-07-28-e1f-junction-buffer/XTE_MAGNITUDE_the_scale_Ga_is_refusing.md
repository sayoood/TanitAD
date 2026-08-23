# The scale Ga is refusing: base wanders **38.9 m** off track; CL-SFT brings it to **3.0 m**

**MEASURED 2026-07-28**, extracted from the four committed frontier artifacts. **No GPU, no new run.**
**Estimator: paired episode-cluster bootstrap** (`taniteval/ci.py`, B=2000), 43 clusters at K=185.
`overlapping_holdout_se` used nowhere.

⚠️ **This is a reporting failure of mine, not a new measurement.** `peak_xte` and `mean_xte` have been
in every frontier artifact since E1c. I reported departure *rates* and open-loop ADE for four
experiments and never reported the **magnitude** of the excursion being prevented. The Ga judgement
has been sitting with the PI without this number in front of it.

## The numbers (step 4000 of each arm; corridor half-width 1.75 m)

| arm | `peak_xte` base → ft | paired Δ [lo, hi] | sep |
|---|---|---|---|
| **E1c** λ=1 full | **38.944 → 3.042 m** | **−35.9030** [−49.330, −24.124] | ✅ |
| **E1e-A** λ=3 full | 38.944 → 4.502 m | −34.4430 [−47.920, −22.446] | ✅ |
| **E1e-B** λ=8 full | 38.944 → 7.790 m | −31.1540 [−45.174, −18.404] | ✅ |
| **E1f** λ=3 junction | 38.944 → 15.012 m | −23.9328 [−37.140, −11.453] | ✅ |

| arm | `mean_xte` base → ft | paired Δ | sep |
|---|---|---|---|
| E1c | **14.306 → 1.391 m** | −12.9153 [−17.689, −8.559] | ✅ |
| E1e-A | 14.306 → 1.916 m | −12.3905 | ✅ |
| E1e-B | 14.306 → 3.248 m | −11.0580 | ✅ |
| E1f | 14.306 → 6.196 m | −8.1099 | ✅ |

**Every arm, both metrics, CI-separated.**

## What this means

The **base REF-C arm diverges catastrophically in closed loop** — a peak cross-track excursion of
**38.9 m** against a **1.75 m** corridor, and a *mean* of 14.3 m. It is not marginally leaving the
lane; over 18.5 s it leaves the road.

**E1c reduces peak excursion by 92 % (38.9 → 3.0 m).** Even E1f, the weakest arm, more than halves it.

⇒ **This is the scale of what Ga is currently refusing.** The guardrail blocks on an open-loop ADE
regression of **+0.05 to +0.20 m** (and an early-path lateral deviation of **+0.2154 m at ≤2 s** for
E1c). Against a **−35.9 m** reduction in peak excursion, that is a ratio of order **167 : 1**.

## ⚠️ How to read this, and how not to

- **It does NOT mean the arms pass the gate.** They do not, and no verdict changes. Ga is a
  pre-registered condition and every arm failed it; that record stands exactly as written.
- **It does NOT make the open-loop cost fictional.** It is real, CI-separated on both axes, and
  tail-heavy laterally (E1c's `cross_p90` +0.3945 — see
  `OPENLOOP_LATERAL_LONGITUDINAL_SPLIT.md`).
- **It DOES mean the two sides of the trade were never presented at comparable scale.** A ~0.2 m
  open-loop concession and a ~36 m closed-loop excursion reduction are not commensurable quantities,
  and the gate treats the former as decisive while never weighing the latter.
- ⚠️ **It is not an argument to relax Ga.** It is the missing magnitude, supplied so the PI's judgement
  is made against the real numbers on both sides rather than against departure *rates* alone.

## Bounds

- 43 episode clusters at K=185; the `peak_xte` CIs are wide (E1c: [−49.3, −24.1]) because peak
  statistics are heavy-tailed. The **sign and order of magnitude** are robust; the point value is not
  precise.
- All arms are **fine-tunes of one base checkpoint**; none is a from-scratch result.
- Step 4000 chosen for comparability, not each arm's best point.
- **The base figure (38.944 m) is the same in all four rows** because the base arm was re-rolled
  identically and reproduced exactly on all four runs — that is what makes the cross-arm comparison
  a comparison.

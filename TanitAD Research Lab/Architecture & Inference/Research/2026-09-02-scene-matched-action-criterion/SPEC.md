# E-ARCH-SMAS-1 — a scene-spread-aware criterion for action-conditioning arms

`Research Lab · Architecture & Inference · daily pass 2026-09-02 · 0 GPU`

## Why this exists

Backlog row **L-1** proposes replacing the `k60clip05p30k` arm's mitigation with a
**horizon curriculum** `K(step) = min(60, 8 + floor(step/Δ))`. That arm ends at
**k = 60**, so whatever the horizon does to the *scene* response will happen to it
too — and the programme's action-conditioning bar is a statistic whose denominator
is that same scene response:

```
ratio = action_spread / scene_spread          (PREREG_MM_E19, bar: ">= 10x rise")
```

Tonight's MM-E19 attribution made the problem concrete: against the one-variable
k=8/clip-0.5 control, `o5_k` 8 → 60 moved the **action** spread 0.83× and the
**scene** spread **1.71×**. ⇒ the ratio fell mostly because the **denominator
grew**. Judging the curriculum arm on the raw ratio would repeat that reading.

⛔ **The general defect:** `scene_spread` is not a normaliser fixed by the design —
it is a **second measurement of the model**, and it rises when the model gets
*better* at the scene. A statistic that divides by it therefore **penalises
world-model improvement**, and a stated "10×" bar is not a fixed bar: its
action-side requirement floats with the denominator.

## Hypotheses, both outcomes committed in advance

| id | hypothesis | outcome A | outcome B |
|---|---|---|---|
| **H-SMAS-1** | The MM-E19 ratio fall is mostly denominator movement. | If the action share dominates, the raw ratio is reporting what everyone thinks it reports and **no criterion change is warranted** — say so. | ⛔ If the scene share dominates, the ratio conflates two effects and the decision statistic must change. |
| **H-SMAS-2** | A stated ratio bar imposes an unstated, floating action-side bar. | If the effective bar ≈ the stated bar, the prereg was fine. | ⛔ If it differs materially, arms have been judged against a bar nobody set. |
| **H-SMAS-3** | Fixing the denominator changes the MM-E19 **verdict**. | The verdict flips ⇒ a banked conclusion needs revisiting. | ⚠️ The verdict holds ⇒ the fix changes the *magnitude, attribution and future bar*, not the past call — and this package must say that plainly rather than overselling. |

## Success criteria, committed in advance

1. The decomposition is **log-additive** — the only additive split of a quotient's
   movement — and the reconstructed ratio factor must match the observed one, or
   the two reads are not on the same instrument and nothing is scored.
2. ⛔ **The panel carries controls that must read known values**, per the
   ridge-probe post-mortem:
   * a **no-information control** (action-blind arm) reading **exactly 0.0**;
   * a **known-value synthetic control** whose injected factors must be recovered
     exactly (catches an algebra error that would otherwise look publishable);
   * the **empirical action-dead floor** (the h2/h4 rows of the same reads);
   * the source reads' **C0 identity control**, which must PASS on every arm or
     the arm is not scored.
3. `n` is stated with every number (24 clips / 144 windows / 8 action variants).
4. ⛔ **A criterion proposal that cannot state what it does NOT fix is incomplete.**

## ⚠️ What this package is not

SMAS (`action_spread / scene_ref`, denominator pinned to a named arm) is
**`action_spread` rescaled by a constant**. The code says so in its own docstring.
The contribution is not a cleverer statistic — it is **refusing to put a moving
quantity in the denominator of a decision statistic**, and keeping `scene_spread`
as a **co-primary report** instead of folding it in. That is the four-families
rule (*"per-family, never pooled into one score"*) applied one level down.

**Tier:** `T0-DIAGNOSTIC`, inherited from the source reads. ⛔ Nothing here is a
driving claim, and no number in this package is comparable to any T1 number.

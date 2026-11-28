# PI rulings on the lateral panels — and what they cost to implement

**Date** 2026-08-30 · **Owner** DataFlyWheel · Source: the PI's judgments on three
`ONE-SIDED-GEOM` panels in the D-LAT-AGREE inspection report.

## The rulings, as given

| clip | PI's judgment | current label |
|---|---|---|
| `00c628f3` (-6.62 m / 123 m, road bending) | **"definitely keep lane"** | `NUDGE_R` |
| `01acc9de` (-3.47 m / 75 m, curve) | **"definitely keep lane and adapt speed for curve"** | `NUDGE_R` |
| `00f3bda1` (+9.67 m, tunnel, "lane change to the left") | goal legitimately emitted; goals are not always followed | `NUDGE_L` |

⭐ **NEW PRINCIPLE (PI, verbatim intent):** *"we should differentiate between
safety-critical tactical goals and actions, or mandatory to follow the route, and
other ones which are optional."* A tactical goal that is not executed is not
automatically a label error — **but only for the OPTIONAL class**. An unexecuted
MANDATORY goal (safety-critical, or required by the route) remains a real defect.
This is a schema addition: `g_tac` tokens need a criticality tier.

## ⚠️ One correction to the second reading, before it is built on

The PI read `00f3bda1` as *"the real data is not including this, that means the
real driver drove differently"*. **MEASURED: the driver DID execute it.** Peak
lateral **+9.67 m**, end lateral **+9.67 m** — the car moved left and stayed
left, exactly as the text describes. The goal was emitted AND followed.

The two real defects in that clip are different, and both are ours:
1. `alpamayo_side` extracted **"straight"** from text that literally reads
   *"Lane change to the **left**"* — a **third** variant of the extractor miss
   (the other two take an obstacle's side; this one finds no side at all).
2. The action is recorded `NUDGE_L` rather than `LANE_CHANGE_L` (D-NUDGE-ABSORB).

⇒ The general principle stands on its own and is worth adopting; it just is not
what this particular clip demonstrates.

## ⛔ The rulings on 1 and 3 are CORRECT and NOT IMPLEMENTABLE from ego motion

I tried to turn "road curvature is not a nudge" into a rule and **it fails the
PI's own labelled cases** — which is exactly what a control is for.

**Attempt 1 — net heading change** (a bend turns the car; a lane change leaves it
parallel):

| clip | truth | net heading |
|---|---|---|
| `00c628f3` | KEEP LANE (road) | **7.0°** |
| `01acc9de` | KEEP LANE (road) | **4.9°** |
| `00f3bda1` | lane change | **12.8°** |

**Backwards.** The lane change has a LARGER heading change than both road-follows,
because the tunnel itself curves. A `>=15°` threshold classifies all three as
"not road-following" and contradicts two of the three rulings. *(Applied blind to
the corpus it would have relabelled **278 of 1,115** NUDGE clips — a confident,
plausible, and unvalidated 25 % reclassification.)*

**Attempt 2 — constant-curvature arc residual** (`lateral = path x heading / 2`;
a lane change should add lateral the arc does not predict):

| clip | truth | arc predicts | measured | residual |
|---|---|---|---|---|
| `00c628f3` | KEEP LANE | -7.58 m | -6.62 m | **+0.95 m** |
| `01acc9de` | KEEP LANE | -3.21 m | -3.47 m | **-0.26 m** |
| `00f3bda1` | lane change | +11.68 m | +9.67 m | **-2.00 m** |

Separates in magnitude (0.95 / 0.26 vs 2.00) but **the sign is wrong**: a LEFT
lane change should push the residual positive, and it is negative. The road is
not a constant-curvature arc (tunnels use transition spirals), so the residual is
contaminated by the road's own curvature profile. **n=3, one confound, wrong
sign — not admissible as a rule.**

⇒ **THE FINDING: no ego-only signal separates "following a curving road" from
"changing lane on a curving road", because both are the same ego motion. The
difference is defined RELATIVE TO THE ROAD, and we do not measure the road.**

## What this costs, stated plainly

The PI's ruling is a statement about **lane-relative** position. Our labels are
**ego-relative**. ⇒ implementing it **requires a lane reference** — it is not a
threshold tweak.

* ⭐ **This promotes the lane detector from an enhancement to a PREREQUISITE.**
  `TanitAD Research Lab/Data Engineering/Research/2026-08-29-lane-detector-deployment/`
  is the package; the extractor already names it at `s2_geom_emit_v7.py:261,:863`
  as the instrument for exactly this question. The PI's judgment is what makes it
  load-bearing rather than optional.
* **Until it exists, `NUDGE` cannot be split by measurement**, and the honest
  reading of the class is: 1,115 clips, of which ~1 % are nudges in the ordinary
  sense and the rest are road-following and lane changes that the label conflates.
* ⛔ **What I am NOT doing: relabelling on an unvalidated discriminator.** Both
  attempts above produce a confident-looking corpus-wide reclassification that
  the PI's own three examples refute. Shipping either would have manufactured a
  25 % relabel with no evidence behind it.

## Recommended next steps (PI decision)

1. **Adopt the criticality tier** on `g_tac` (mandatory / optional). Cheap,
   schema-only, and it makes "goal not executed" interpretable instead of
   ambiguous. I can draft the taxonomy against the existing token vocabulary.
2. **Fix the three extractor misses** (obstacle-side x2, no-side-found x1) — no
   ruling needed, and it removes ~54+ false conflicts.
3. **Decide whether the lane detector is funded now.** It is the only way to
   implement rulings 1 and 3, and it also settles the 505 one-sided cases and the
   NUDGE/LANE_CHANGE split in one instrument.

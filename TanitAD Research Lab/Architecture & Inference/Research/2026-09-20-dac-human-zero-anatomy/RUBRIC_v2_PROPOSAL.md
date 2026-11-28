# Adjudication rubric v2 — PROPOSAL for the Master Mind to land, the PI to rule

⛔ **STATUS: PROPOSAL.** Pack 1's 60 labels and pack 2's 58 were made under **v1** and are **not**
re-read under this. A rubric changed after labels exist may govern the NEXT pack only.

## What v1 could not express

**S53** (pack 2): a stationary manoeuvre in a snow-covered **car park**. v1 names pavement, verge,
island and kerb as non-roadway; it says nothing about a surface built for vehicles that is not
carriageway. The adjudicator's tie-break — judge whenever the surface is visible in ANY panel —
failed at the same moment, because the footprint projected **20 px below the frame** and no overlay
appeared at all. Both halves failed together, which is why that row is a `cannot-tell` that means
two different things at once.

## v2 — four cells, and the new one carries the definitional question rather than hiding it

| label | one line |
|---|---|
| `on-surface` | every part of the car stayed on **carriageway** a car may drive on — including paint, markings and crosswalks painted ON it |
| `over-boundary` | some part of the car crossed onto a surface **not built for vehicles** — pavement, verge, traffic island, or past a raised kerb |
| ⭐ `off-carriageway-trafficable` **(NEW)** | the car was on a surface **built for vehicles but not carriageway**: car park, forecourt, service yard, private driveway |
| `cannot-tell` | the images do not show the surface under the relevant part of the car well enough to decide |

⛔ **The new cell is REPORTED, never folded.** Until the PI rules whether the drivable-surface map is
meant to include car parks, windows labelled `off-carriageway-trafficable` are **excluded from
agreement scoring and counted separately**. Folding them into `on-surface` would silently decide the
question; folding them into `over-boundary` would silently decide it the other way.

⚠️ This is a **rubric** fix, not a map fix. What the SAM3 map actually does with a car park is a
separate measurement nobody has made; the two must not be assumed to agree.

## Two sheet-level fixes that go with it

1. **Say when there is no overlay.** MEASURED across both packs: **2 of 118 sheets (1.7 %)** show no
   corner in any panel, and **17 of 118 (14.4 %)** never show a complete footprint. The renderer
   knows this from the geometry, so the sheet should state it on its face. ⭐ It leaks nothing — it
   depends on the camera and the path, never on a map or a rule.
2. **Keep the depth-test disclosure.** Five of pack 2's eight borderline rows are tight turns whose
   inside corner is a kerb or island, and in each the far arc draws OVER the island because it lies
   beyond it. Until the overlay is depth-tested, that sentence is load-bearing.

## What v2 still cannot do

It cannot make this corpus able to test a rule for **missing** a departure. 118 adjudicated windows
produced zero `over-boundary` labels; a rubric cannot supply positives the corpus does not contain.

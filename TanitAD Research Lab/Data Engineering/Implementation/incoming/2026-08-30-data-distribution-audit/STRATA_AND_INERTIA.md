# The corpus can now state its own composition — plus the inertia statistic

**Date** 2026-08-30 · **Owner** DataFlyWheel · Rows D-B1-CRITIQUE, D-DATA-EFFICIENCY
· Blob `ee44875916ae7c0ac002c6716b9658ea` · 4,719 clips

## ⛔ AMENDMENT to my Part 1 headline — B1 WAS a designed selection

**Part 1 concluded B1 was "an availability intersection, NOT a designed sample".
That conclusion is WRONG and I am withdrawing it.** Verified in source myself
rather than on relay: `2026-08-06-alpamayo-augmentation/DESIGN.md` records the
PI-ordered design —

* `:3` *"select **100 hours** of well-distributed PhysicalAI-AV data"*
* `:12` *"100 h = **18,000 clips** ⇒ a **5.9 % stratified sample**"*
* `:49` *"stratified 18,000-clip selection (**equal-weight cells over road-class ×
  day/night × country**, proportional fill, **seed 0**, manifest committed)"*
* `:42` deliverable: *"selection manifest (clip_id + **strata cell**)"*

⚠️ **THE MECHANISM OF MY ERROR, because it generalises.** I measured B1 == the
Alpamayo record set *exactly* — 4,719/4,719 inside, **0** outside — and read that
as "no selection". **A set that equals its parent is not unselected; its PARENT
was selected.** The selection happened one level up, in choosing which clips were
sent for labelling. My measurement was right and my inference skipped a level.

⭐ **And my own Part 1 numbers were the evidence against my own conclusion.** I
reported near-uniform country shares (~5–6 % each) against 50.75 % US in the
parent, χ² = 7,300 — and attributed it to "Alpamayo's own labelling structure".
**That IS the stratification, showing through.** A designed country-balanced
sample is exactly what produces it. I had the fingerprint and misread whose it
was.

**ROOT-CAUSE CLASS — C85: INFERRING ABSENCE OF DESIGN FROM AN IDENTITY BETWEEN A
SET AND ITS SOURCE.** Recognition signal: the conclusion "X was not selected"
rests on *"X equals the set it was drawn from"*. That identity is equally
consistent with the selection having happened **upstream of the boundary you
measured**. ⇒ **before concluding no selection, look one level up the supply
chain for a selector**, and check whether the distribution carries a
selection fingerprint (here: uniformity that chance cannot produce).

## What the design achieved, and what it did not

| stratum axis | design intent | measured on our 4,719 | verdict |
|---|---|---|---|
| **country** | equal-weight cells | ~5–6 % per country vs 50.75 % US in parent (χ² 7,300) | ✅ **ran** |
| **day/night** | equal-weight cells | 53.7 % day / 46.3 % night | ✅ ran (near-balanced) |
| **road class** | equal weight ⇒ ~33 % each | urban **64.8 %** · intersection **18.8 %** · highway **16.3 %** | ⛔ **did NOT run** — the design flagged it an unfilled WORK ITEM |
| **volume** | 18,000 clips / 100 h | 4,719 clips / 26.2 h | truncated at **26 %** of target |

## ⭐ IMPLEMENTED: the strata cell is now in the manifest

`selection_manifest_with_strata.json` — 4,719 rows carrying
`clip_id · t0_us · strata_cell · road_class · daynight_clock · country ·
stopped_frame_frac`, so **every future subset question is answered by reading
rather than re-deriving**. This is the artifact whose absence let a designed
corpus be mistaken for an accidental one.

* **148 of 150 possible cells are occupied** (3 road-class × 2 day/night × 25
  countries); **140 cells hold ≥ 5 clips**; median 24 clips/cell, max 103. The
  cell grid is genuinely populated — further evidence the stratification ran.
* Road class from the **design's own rules** (`DESIGN.md:17`): highway =
  ≥ 20 m/s sustained (≥ 30 % of the clip), intersection = a stop **and** a
  heading change (≥ 25° yaw span), urban = the remainder. Ego-derived labels are
  admissible (labels-may-use-ego); this is offline derivation, never an
  inference input.
* ⚠️ **`daynight_clock` is DECLARED AS AN APPROXIMATION.** The design names
  day/night but defines no rule, and we hold no lat/lon (egomotion is clip-local
  metres), so true solar elevation is **not computable**. It is a fixed
  07:00–19:00 local-clock split and is named `daynight_clock` so nobody reads it
  as measured. It will mislabel shoulder hours and high-latitude summers.

## ⭐ CODEVILLA INERTIA — measured for the first time, and it does NOT apply to us

Codevilla's *more-data-is-worse* result was **not** an argument about high-speed
data. It was the **inertia problem**: over-represented STOPPED frames create a
spurious *low-speed → no-acceleration* correlation. Nobody had measured our
stopped-frame fraction. Now measured, over all 4,719 clips:

| statistic | value |
|---|---|
| stopped-frame fraction, **mean** | **5.6 %** |
| stopped-frame fraction, median | **0.0 %** |
| p90 | 23.5 % |
| clips > 50 % stopped | 80 = **1.7 %** |
| clips > 90 % stopped | 7 = 0.1 % |
| **clips with NO stopped frames at all** | **2,893 = 61.3 %** |

⇒ **B1 does not have Codevilla's inertia problem — it has the opposite shape.**
A median of zero stopped frames and 61.3 % of clips never stopping means the
spurious low-speed correlation has little mass to form on.

⚠️ **But do not read that as good news without stating the flip side**: stop-and-go,
queueing and creeping are *real driving competences*, and a corpus where 61.3 %
of clips never stop is thin in them. The inertia risk is low; a **stopped-behaviour
coverage** risk is the one to check instead. That is a different question and it
is now cheap to ask, because the fraction is in the manifest per clip.

## Deliverable manifest

| artifact | where |
|---|---|
| `raw/selection_manifest_with_strata.json` — 4,719 rows, the missing strata cell | repo, this package |
| `raw/b1_strata_cells.parquet` — full per-clip frame incl. inertia stats | repo |
| `code/build_strata_cells.py` — derivation, re-runnable | repo |

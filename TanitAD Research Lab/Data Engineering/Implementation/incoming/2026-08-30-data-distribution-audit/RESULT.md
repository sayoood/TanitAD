# Part 1 — What actually selected our 26 h

**Date** 2026-08-30 · **Owner** DataFlyWheel · **Trigger** PI: *"investigate the
distribution of our 26 h — I thought we sampled with strategy, but let's check it
again."* · Blob `s2_labels_v7.jsonl.gz` md5 `ee44875916ae7c0ac002c6716b9658ea`.

## ⛔ SUPERSEDED IN PART — read `STRATA_AND_INERTIA.md` first

**The headline below ("B1 is an availability intersection, NOT a designed
sample") is WITHDRAWN.** B1 IS the product of a PI-ordered stratified selection
(`2026-08-06-alpamayo-augmentation/DESIGN.md:3,:12,:49`). My measurement was
right — B1 equals the Alpamayo record set exactly — but **a set that equals its
parent is not unselected; its PARENT was selected**, one level up, when clips
were chosen to send for labelling. Class C85. The country table in §3 below is
the stratification's own fingerprint, which I attributed to the wrong actor.

**What still stands unchanged:** the r0/parity speed-gate finding (§2), the
metadata-tags correction (§3), and every measured number here.

## The answer in one line

**There are TWO corpora and they were selected in OPPOSITE ways — and the one we
DESIGNED is the one with the hole in it.**

| | how it was selected | evidence |
|---|---|---|
| **parity `e438721ae894`** (2,376) | **DESIGNED** — scored and ranked | `scripts/physicalai_r0.py:150` `sort_values("urban_score", ascending=False)` |
| **B1** (4,719 = our 26 h) | ⛔ **AVAILABILITY** — no selection at all | 4,719/4,719 inside the Alpamayo record set; **0** outside it |

## 1. B1 is an availability intersection — MEASURED, not inferred

| set | n |
|---|---|
| B1 corpus | 4,719 |
| Alpamayo records available | 4,729 |
| **B1 ∩ Alpamayo** | **4,719 = 100.00 % of B1** |
| B1 clips NOT in Alpamayo | **0** |
| Alpamayo clips not in B1 | 10 |

We took **everything Alpamayo happened to label**, minus ten. And
`alpamayo/selection_manifest.json` records **only `clip_id` + `t0_us`** — no
criteria, no score, no strata. There is no selection rationale banked because
there was no selection. ⇒ **the PI's belief that the 26 h was sampled with a
strategy is REFUTED for B1.**

## 2. The parity corpus WAS designed — and the design deletes high speed

`physicalai_r0.py::stage_select` is a real strategy, and its rules are explicit:

* **HARD GATE (`:100`)**: `if not (2.0 <= mean_v <= 14.0) or stop_frac > 0.8 or
  distance < 40.0: score = 0.0`
* **score (`:106-108`)**: `speed_band` peaking at **8 m/s**, plus a stop term
  saturating at 30 % stopped, plus a turn term — i.e. *maximally urban*
* **rank (`:150`)**: take the **top** urban scores, round-robin over countries
* **chunks (`:61-75`)**: country-spread, `random_state=0`

⛔ **14.0 m/s is 50.4 km/h. Every motorway, dual carriageway and fast arterial
clip scores exactly 0 and can never be selected.** That is not a sampling
imbalance; it is a structural exclusion in source.

### The comparison that matters

| mean-speed band | **r0 DESIGNED** | **B1 AVAILABILITY** |
|---|---|---|
| crawl < 7 km/h | 0.0 % | 2.0 % |
| slow urban 7–29 | **67.6 %** | 37.2 % |
| urban/arterial 29–50 | 32.4 % | 34.5 % |
| fast arterial 50–72 | **0.0 %** | 14.3 % |
| **highway > 72 km/h** | **0.0 %** | **12.0 %** |

*(r0 figures from the local 500-row `r0_selection.parquet`: median 7.19 m/s, max
**11.50**, zero above 14. The empirical zero is guaranteed by the source gate, so
the partial copy cannot understate it.)*

⭐ **THE FINDING: our deliberate strategy produced a corpus with NO high-speed
driving whatsoever, while the accident preserved 26.3 % of it.** And **88.7 % of
our oracle gap is LONGITUDINAL** — the axis the designed corpus structurally
deleted. The selection strategy removed the regime our largest known defect
lives in.

## 3. Stratum table — against the tags that actually exist

⚠️ **CORRECTION TO THE BRIEF: the weather / road-type / traffic-density / surface
tags DO NOT EXIST in the shipped metadata.** Two probes, full column dumps:
`metadata/data_collection.parquet` carries **5** columns — `country`, `month`,
`hour_of_day`, `platform_class`, `radar_config` — and
`metadata/feature_presence.parquet` carries 36 **sensor-presence** flags, no
scene tags. ⇒ **the card's stratification axes are not available to us as
metadata**; any weather/road-type stratification must be DERIVED (from pixels or
from Alpamayo text), which is a build, not a join.

**B1 vs the 306,152-clip parent, on the axes that do exist:**

| country | parent % | B1 % | ratio |
|---|---|---|---|
| **United States** | **50.75** | **6.40** | **0.13 ×** |
| Germany | 14.34 | 6.27 | 0.44 × |
| France | 3.39 | 6.12 | 1.81 × |
| Italy | 2.83 | 6.06 | 2.14 × |
| Sweden | 2.39 | 6.12 | 2.56 × |
| Austria | 1.78 | 5.55 | **3.12 ×** |
| Netherlands | 1.61 | 4.83 | **3.00 ×** |

B1 is **near-uniform across countries (~5–6 % each)**, so the US — **half the
parent corpus** — is **8× under-represented**, and small European countries are
2–3× over-represented.

| hour_of_day | parent % | B1 % | ratio |
|---|---|---|---|
| 21:00 | 4.98 | 7.29 | 1.46 × |
| 19:00 | 6.28 | 8.98 | 1.43 × |
| 20:00 | 5.98 | 7.93 | 1.32 × |
| 08:00 | 5.51 | 3.07 | **0.56 ×** |
| 09:00 | 5.63 | 3.71 | 0.66 × |
| 10:00 | 6.15 | 4.39 | 0.71 × |

**Evening and night are over-sampled ~1.4×; morning is under-sampled ~0.6×** —
a ~2.5× swing across the day. Nobody chose this.

## 4. Effective coverage

* **1.53 %** of the parent by clips (4,719 / 306,152).
* **1 of 7 cameras** read — `feature_presence` confirms all 7 are present on the
  corpus, so 6 are unused.
* Of the axes we could stratify on, **none was used for B1**; the parity corpus
  used exactly one (urban score) and used it to exclude a regime.

## What this means for Part 2 (the score)

The audit already names three concrete axes a distribution score must cover, and
one of them is measurable today with no new instrument:

1. **speed regime** — derivable from egomotion for every candidate clip; the
   parity corpus scores 0 on two of five bands.
2. **country / time-of-day** — a free join on the catalogue.
3. **weather / road type** — ⛔ NOT available as metadata; must be derived.

⚠️ **PARITY: none of this re-selects within `e438721ae894` / skip-hash
`f09e44db`.** Any stratified re-sample is a NEW NAMED CORPUS ARM by construction.

## Deliverable manifest

| artifact | where |
|---|---|
| `RESULT.md` (this) | repo, this package |
| `raw/b1_speed_profile.parquet` — per-clip mean/p95 speed, 4,719 rows | repo |
| `code/audit_distribution.py` — the measurement, re-runnable | repo |

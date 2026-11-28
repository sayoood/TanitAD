# Part 2 — DDS, a data distribution score that ranks subsets before training

**Date** 2026-08-30 · **Owner** DataFlyWheel · **Status** DESIGNED · controls PASS
· validation against trained outcomes NOT YET RUN (plan below).

## What it is

`dds(subset, parent)` returns a scalar plus per-axis detail, computed from
metadata and egomotion only. ⭐ **No training required** — that is the whole
point of a selection tool.

    dds = (coverage x effective_diversity x defect_regime_ratio) ^ (1/3)

* **coverage** — fraction of the parent's strata holding ≥ 5 clips.
* **effective_diversity** — `exp(Shannon entropy) / n_strata`, the effective
  fraction of strata actually exercised. A raw stratum count is satisfied by one
  token clip each; this is not.
* **defect_regime_ratio** — mass in `fast_arterial` + `highway` relative to the
  parent. **Floored and inside a geometric mean, so a zero here sinks the whole
  score.** Deliberate: a subset missing the regime our largest defect lives in
  must not be rescued by being broad elsewhere — which is exactly how the r0
  corpus would otherwise pass.

## Why diversity and not representativeness — from the primaries

* **Codevilla ICCV 2019**: more data was WORSE (best at 10 h of 100 h, failure
  RISING with volume) because *"the diversity of the dataset does not grow fast
  enough compared to the main mode of demonstrations"*. **Matching the parent's
  shape would maximise exactly that main mode.**
* **WOD-E2E**: a 12 h benchmark curated for <0.03 %-frequency scenarios does the
  work of ~40,000 h accumulated by volume — **curation beat volume ~3,300×**.
* **DINO-WM**: over 92× data, prediction fidelity moved +4 % while control
  competence moved 11.5× ⇒ the score must predict CONTROL competence, so it is
  validated against a control metric, never against reconstruction loss.

## ⛔ The reference is the PARENT, not B1 — measured, not assumed

The Master Mind flagged that "unselected" is not "representative". Probed:

| hypothesis | result |
|---|---|
| B1 is a RANDOM draw from the parent | ⛔ **REFUTED** — χ² **7,300** on df=24; US expected **2,395**, observed **302** |
| B1 is UNIFORM over its 25 countries | ⛔ refuted too — shares 1.50–6.40 % vs 4.00 % uniform; only **6/25** within ±30 % |

B1 spans 1,411 of 3,146 chunks, median 2 clips per chunk. ⇒ **the Alpamayo
labelling set carries its own country-capping structure.** Scoring against B1
would bake that bias into the score, so the parent catalogue is the reference.

## Controls — ALL PASS, at EQUAL SIZE (n = 302 per arm)

| arm | n | **DDS** | coverage | diversity | defect mass |
|---|---|---|---|---|---|
| **designed (speed-balanced)** | 302 | **1.0876** | 0.9461 | 0.8993 | 39.7 % |
| random | 302 | 0.8617 | 0.9039 | 0.8262 | 22.5 % |
| bad: single country | 302 | 0.7546 | 0.6522 | 0.5683 | 30.5 % |
| **bad: r0 gate emulated** | 302 | **0.0080** | 0.7194 | 0.7038 | **0.0 %** |
| bad: all-slow | 302 | 0.0076 | 0.6922 | 0.6380 | 0.0 % |

* ✅ designed **above** random — the score is not inert.
* ✅ ⭐ **the r0 gate emulation ranks near-bottom** — a control with a **known
  answer from source** (`physicalai_r0.py:100`), not a synthetic strawman.
* ✅ all-slow and single-country both below random.
* ✅ every arm the **same size**, so n rewards nothing.

⚠️ **The equal-size assertion earned its place immediately**: the first run used
n=800 and the single-country arm could only supply 302, so it FAILED. N is now
computed as the largest size every arm can reach. A control that never fails
would not have caught that.

## ⚠️ Scope and honest limits

* **BLIND AXES: weather, road type, traffic density, surface.** These columns do
  **not exist** in the shipped metadata (two full column dumps). They must be
  derived by an instrument we do not have; the score reports them as blind
  rather than implying coverage. Per the Master Mind's call, the score ships on
  three real axes rather than stalling behind a derivation build.
* **The parent's speed distribution is IMPUTED from B1**, because egomotion is
  not downloaded corpus-wide. Stated, not hidden — and the ARM ORDERING above
  does not depend on it, since the arms are compared against each other.
* `dds` is **not bounded to [0, 1]**: `defect_regime_ratio` may exceed 1 when a
  subset over-samples the defect regime relative to the parent. Intended for a
  curation tool, but it means DDS is a RANKING statistic, not a percentage.

## Validation plan — what would make this admissible

**Not yet run. Until it is, DDS is a designed instrument with passing controls,
NOT a validated predictor, and no decision should rest on it.**

1. **≥ 3 subsets of EQUAL size and EQUAL GPU-time** — e.g. `designed`, `random`,
   `r0-gate-emulated`, identical trainer, seed and step budget.
2. **Rank-correlate DDS against a CONTROL metric** — T1 closed-loop longitudinal
   error, never reconstruction loss (the DINO-WM lesson). Spearman ρ over the
   arms, with its n stated.
3. **The negative control is free and already known**: the r0-gated arm must
   train WORSE on longitudinal, because the regime is absent from it. If it does
   not, DDS is measuring something other than what we need.
4. **Report n and d** (arms and axes) on every number, and fit nothing on the
   arms being scored.

⭐ **This doubles as the publishable experiment the Research Lab could not find
in the literature: NO paper ablates diversity at FIXED hours.** Equal-size,
equal-compute arms differing only in distribution is precisely that gap.

## Deliverable manifest

| artifact | where |
|---|---|
| `code/dds.py` — the score | repo, this package |
| `code/dds_controls.py` — the control harness | repo |
| `raw/dds_controls.json` — the arm results | repo |

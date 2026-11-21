# INTAKE — curve-rebalance analyzer + measured mix recipe (FLEET_REVIEW P0#3)

**Discipline:** Data Engineering · **Date:** 2026-07-18 · **Slug:** `2026-07-18-curve-rebalance`
**Status:** proposed (awaiting MVP-orchestrator triage)

## What
A standalone analyzer (`curve_rebalance.py`, 12 tests ✓) that measures the
straight/gentle/sharp **curvature-stratum distribution** of any corpus on real
episode bytes — using the *exact* D1 eval convention (`|net yaw change @2 s| <5° /
5–20° / >20°`, copied verbatim from `stack/scripts/driving_diagnostic.py` and
test-guarded against drift) — and derives a **turn-weighted sampling recipe**
(single-knob `beta` on non-straight windows) that moves the mix to a target
straight-fraction.

This is the **data-side half of the REF-B curve failure** (the loss-side half
shipped in `refbpatch`) and the FLEET_REVIEW 2026-07-17 **P0#3** deliverable.

## Why
The #1 program risk (single-camera driving-capability gap) is *enabled* by a
mostly-straight corpus: a mix dominated by straight driving is satisfiable by the
ego-status shortcut ("keep going straight at v0") instead of learning
action→consequence on turns. The review asserted "~74% straight" without a
per-source breakdown; this quantifies it on real bytes and turns it into two
tunable knobs with numbers.

## Evidence (measured — local RTX-4060 dev box, CPU, ~1.5 min, $0)
630 real episodes / 125,247 windows from the local epcache (comma2k19 + PhysicalAI):

| Source | episodes | windows | **straight** | gentle | sharp |
|---|---|---|---|---|---|
| comma2k19 (highway) | 130 | 36,270 | **83.1%** | 10.5% | 6.4% |
| PhysicalAI (urban) | 500 | 88,977 | **56.0%** | 23.4% | 20.6% |
| **combined (natural pool)** | 630 | 125,247 | **63.9%** | 19.6% | 16.5% |

- **The "74% straight" claim reconciles to a comma-dominant mix:** at comma
  0.65–0.70 / PhysicalAI weighting the straight-fraction is 73.6–75.0%. The
  straightness is a **comma2k19/highway property**, not a whole-corpus one —
  PhysicalAI urban is already **56.0%**, at the 55–60% target.
- **Recipe to hit 57.5% straight:** turn-upweight `beta` = 1.31 (natural pool) →
  2.22 (comma-70% mix). Verified by construction (round-trip test).
- **Two independent levers, both quantified:** (1) **source-mix** — every +10 pp
  of comma weight adds ~2.7 pp straight; shifting weight to urban (PhysicalAI, and
  the incoming ZOD/PandaSet) is the primary knob; (2) **window turn-weighted
  sampling** (`WeightedRandomSampler` keyed by each window's stratum, weights
  `{straight:1, gentle:beta, sharp:beta}`).

## Tests run
`pytest .../2026-07-18-curve-rebalance/tests -q` → **12 passed** (1.7 s). Covers
the stratum buckets, per-episode vectorized counts (analytic arcs),
fraction/aggregation math, the `beta` round-trip, source-mix combination, and a
**drift guard** asserting the constants still equal `driving_diagnostic.py`.

## Proposed target location in `stack/`
- `stack/tanitad/data/curve_rebalance.py` (pure analysis; no new deps — numpy+torch).
- `stack/tests/test_curve_rebalance.py`.
- Consumer wiring is a **follow-up** (not in this package): a
  `WeightedRandomSampler` over `MixedWindowDataset` window strata, or a per-source
  mix-ratio change — both are training-recipe changes (D-018 **tactics →
  ESCALATE** to Sayed before flipping the live trainer). This package delivers the
  measurement + recipe, not a live-trainer change.

## Risk / rollback
- Analysis-only; imports nothing from the trainer, touches no `stack/` file. Zero
  runtime risk until a sampler/mix change is separately proposed.
- The stratum constants are duplicated from `driving_diagnostic.py`; the drift
  guard test fails loudly if the eval owner changes them — update both together.
- Rollback = delete the two files.

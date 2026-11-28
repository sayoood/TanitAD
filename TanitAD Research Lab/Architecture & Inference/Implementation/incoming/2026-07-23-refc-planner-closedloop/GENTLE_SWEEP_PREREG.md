# PRE-REGISTRATION — gentle-FT sweep: can decoder-only recovery-FT keep the departure win AND recover ADE?

**Stream D2** (`a1f26c92`). **Committed before the run.** The naive FT (λ_dev 0.5 / lat_max 1.75 / 1500 steps
/ lr 1e-4) WON the primary metric (held-out corridor_departure 0.0174→0.0085, dCDR +0.0089 [+0.0008,+0.0197]
SEP) but paid a **separated ADE cost** (closed_ade 0.587→0.875, dADE −0.288 [−0.428,−0.149] SEP), broad
across strata (worst on longitudinal −0.347). The FT `dev` (‖FT−base‖ on clean windows) rose 0.24→0.75 over
training → the decoder drifted from base **globally, and late**. Hypothesis: **fewer steps + stronger
on-path anchoring** keeps the departure win while recovering ADE.

## Arms (decoder-only, frozen encoder — WM stays safe; same held-out 28:40 eval each)

| cfg | steps | lat_max | yaw_max | clean_frac | λ_dev | lr | rationale |
|---|---|---|---|---|---|---|---|
| **base** | — | — | — | — | — | — | REF-C base (reference, re-measured each eval for the paired CI) |
| **naive** (done) | 1500 | 1.75 | 5° | 0.30 | 0.5 | 1e-4 | the WIN+ADE-cost result already banked |
| **g1** | **500** | 1.75 | 5° | 0.50 | 1.0 | 1e-4 | *fewer steps + stronger anchor, SAME envelope* → tests "less over-training is the fix" |
| **g2** | 700 | **1.0** | 3° | 0.50 | 1.0 | **5e-5** | gentler envelope + lower lr |
| **g3** | **400** | **0.75** | 3° | 0.60 | **2.0** | 5e-5 | gentlest — smallest perturbations, strongest anchor, fewest steps |

## Committed verdicts (primary = OVERALL held-out; paired episode-cluster bootstrap)

- **✅ NET WIN (usable → promote, NAME the config).** A config with BOTH:
  1. **departure held**: overall dCDR(base−ft) **≥ +0.005 and CI excludes 0** (still ~≥ base/2 real reduction), **and**
  2. **ADE recovered**: overall dADE(base−ft) **CI INCLUDES 0** (the ADE regression is no longer separated — within noise of base 0.587), **and** the peak_xte guard holds (dPEAK not separated < 0).
  → decoder-only recovery-FT can be a net closed-loop improvement; promote the gentlest such config, then AlpaSim.

- **⚠️ TRADE IS FUNDAMENTAL (for decoder-only).** No config satisfies both — every config either **loses the
  departure reduction** (dCDR CI includes 0 → too gentle) **or keeps a separated ADE regression** (dADE CI
  excludes 0, FT worse). → report the **Pareto frontier** (dCDR, dADE) over {naive, g1, g2, g3} and conclude
  the departure↓/ADE↑ trade is intrinsic to decoder-only recovery-FT → the lever then needs the **encoder in
  the loop** (unfreeze/light-FT the encoder so the off-path *features* separate, not just the decoder's read
  of them) or a **different mechanism** (e.g. a return-to-GT-speed / progress term alongside the gentler warp,
  or closed-loop-consistency rather than single-step recovery). This is a decision-grade bound, not a failure.

## Cost / safety
Decoder-only, frozen encoder; g1 ~500 steps (~7 min) + g2 ~700 + g3 ~400, each + held-out eval (~6 min) →
~40 min total. `gpu_lock refc-cl-improve` tied to the sweep PID, **released on completion** (D1's 40-ep
hardening is queued behind). REF-C deployed ckpt read-only; each FT writes a NEW dir. Bank each config's
numbers to `RESULTS.md` as it lands (not held for the set). Low-OOD LANE-KEEPING, not a safety rate.

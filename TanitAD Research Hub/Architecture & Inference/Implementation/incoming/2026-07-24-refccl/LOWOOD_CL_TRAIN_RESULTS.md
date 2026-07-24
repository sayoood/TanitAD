# LOWOOD-CL-TRAIN — the decisive renderer-free test: does the on-policy objective escape the trade?

**MEASURED on `tanitad-eval`, `gpu_lock lowood-cl-train`, 2026-07-24.** Pre-registered in
`Research/2026-07-24-low-ood-closedloop-renderer.md §7` (both outcomes committed). Closed-loop-CONSISTENT
recovery: roll REF-C forward on the real-footage low-OOD harness, train (decoder-only, WM-safe) to recover
from the poses it **actually visits on-policy** (RoaD/CAT-K), CAT-K-filtered to the ≤1.16× envelope.
Held-out 28:40 (12 eps / 264 windows), paired episode-cluster bootstrap — **identical protocol to
D2/RefcCL**. Raw: `corridor_lowood_cl.json`, `cl_provenance.json`, `lowood_cl_train.log`.

## Envelope honesty (the §7 requirement) — the on-policy distribution is faithfully low-OOD
3 DAgger rounds, ~10,660 real on-policy states each (~32k total). **frac_inside the ≤1.16× envelope =
0.948 / 0.948 / 0.949**; mean on-policy |drift| **0.130 / 0.273 / 0.231 m**. Non-self-referential (real
warped frames, not the WM's imagination → avoids the MEASURED DAGGER_HURTS trap). The harness-as-training-
source WORKS; the finding below is about the OBJECTIVE, not the instrument.

## Result (base cdr 0.0174, ade 0.587)

| stratum | dCDR@1.75 [CI] | dADE@2s [CI] | dPEAK |
|---|---|---|---|
| overall | −0.0460 [−0.120,+0.005] n.s. (departs MORE) | **−0.329 [−0.732,−0.034] S (worse)** | −0.649 S |
| junction | +0.0159 [−0.008,+0.042] n.s. | **+0.092** [−0.110,+0.241] n.s. (better) | +0.302 n.s. |
| longitudinal | −0.130 [−0.269,−0.018] S (departs much more) | −0.825 [−1.482,−0.195] S | −1.677 S |

Absolute overall: corridor 0.0174→**0.0634**, ADE 0.587→**0.916**.

## PRE-REGISTERED VERDICT: **BOUND** — the trade is DEEPER than the objective

dADE is **CI-separated-worse (−0.329)** → the on-policy CAT-K/RoaD objective **does NOT escape** the
departure↔ADE Pareto trade. It is in fact **worse** than the synthetic single-step configs:

| objective | overall dCDR | overall dADE |
|---|---|---|
| base | — | 0.587 (ref) |
| D2 naive (synthetic single-step) | +0.0089 **S** (fewer departures) | −0.288 S |
| D2 g2 (synthetic) | +0.0057 n.s. | −0.125 S |
| RefcCL-s2 (encoder + synthetic) | +0.0002 n.s. | −0.084 S |
| **LOWOOD-CL (on-policy consistent)** | **−0.0460 n.s. (departs MORE)** | **−0.329 S** |

**Why the on-policy objective is WORSE, not better (MEASURED-grounded):** the on-policy drift on the FT
episodes is **tiny** (mean 0.13–0.27 m, 95 % inside envelope) — because base REF-C already keeps the ego
near the path on the low-OOD source. So the RoaD/CAT-K premise ("train on the informative failure states you
visit") is **starved of informative failures**: the on-policy signal is dominated by ~0.13 m micro-
corrections, and heavily training on those makes the planner **over-react to tiny offsets**, which
generalizes to MORE departures + worse ADE on the held-out episodes. The synthetic D2 objective was actually
a **stronger** recovery signal because it could deliberately probe larger (up to 1.75 m) offsets. **On a
low-OOD source where the base policy rarely fails, on-policy recovery training is impoverished, not
superior.**

## The closed-loop-improvement direction, closed (D2 → RefcCL → LOWOOD-CL)

The departure↔ADE trade is **intrinsic to recovery-augmentation on the low-OOD lane-keeping instrument** —
independent of **objective** (synthetic single-step ✗ / on-policy consistent ✗) AND **parameter subset**
(decoder ✗ / safe encoder-in-loop ✗). Every one of the ~10 configs hits the same Pareto wall; the recurring
signature is **junction ADE improves (+0.09..+0.13) while longitudinal/straight ADE degrades** — the
recovery reactivity helps where heading error dominates (turns) but over-generalizes on straights. **The
cheap low-OOD instrument is EXHAUSTED for improving REF-C road-keeping via recovery training** (the §7 BOUND
branch fired).

**What this justifies (previously premature, now necessary):** the renderer paths in
`Research/2026-07-24-low-ood-closedloop-renderer.md`. Specifically —
1. **The road-keeping trade is real, not an objective artifact** → the map-free source's inability to say
   "you're actually fine here" (no lane geometry — it penalizes any deviation from the exact recorded path)
   is the deeper limit. A **map/lane-aware instrument** (or the recorded corridor treated as a *tolerance
   band*, not a knife-edge GT path) is needed to score road-keeping without punishing benign recovery.
2. **Reactive-agent collision (B)** remains the only place a renderer is unambiguously binding → **Path 3**
   (reactive-agent overlay on real frames via `obstacle.offline` + IDM), inside the envelope.

## Bankable POSITIVES from the whole arc (do not lose these)
- The **open quadrant** (renderer-free ∧ non-self-referential ∧ data-efficient recovery) is real: D2
  decoder-only **halves** held-out departures (+0.0089 S), generalizing where Gate-1 memorized.
- **REF-C's encoder is safely fine-tunable** (RefcCL canary holds at a material move) — de-risks future
  encoder work.
- The **on-policy low-OOD harness is a valid, non-self-referential training source** (95 % in-envelope,
  32k states, DAGGER_HURTS trap avoided) — the machinery is sound and reusable; only the recovery objective
  is Pareto-bound.
- **New diagnostic:** RoaD/CAT-K on-policy training needs a policy that **actually fails** on the source; on
  a low-OOD source where the base rarely drifts, synthetic perturbation is the stronger signal. This scopes
  where on-policy training pays off (harder/higher-OOD sources, or a weaker start policy).

## Honest bounds
Low-OOD LANE-KEEPING, not a safety rate. n=12 held-out eps (wide bands). Decoder-only (WM untouched — no
canary needed). Ground-plane-lateral warp optimistic; 94.9 % of on-policy states inside the measured
envelope, the rest CAT-K-filtered out. Single seed / one config. Within-instrument RELATIVE.

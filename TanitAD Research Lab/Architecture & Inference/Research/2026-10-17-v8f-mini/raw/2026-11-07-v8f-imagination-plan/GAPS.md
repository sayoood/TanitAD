# What the v8F world model does and does not imagine: the gaps (Master Mind, 2026-11-07)

Every row is MEASURED, tier T0 (a world-model diagnostic, never a driving claim), M2b s0 / s1, zero training, episode-cluster bootstrap B 2,000 unless stated. Sources are landed result folders under `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/`.

## What works (the positive base)
| finding | numbers | source |
|---|---|---|
| L2 and L3 beat "nothing changes" | L3 SKILL +0.1316 / +0.1351 at 0.8 s, +0.3283 / +0.3175 at 2.4 s | `2026-11-07-v8f-wmact-result/` |
| The prediction follows the logged longitudinal action at every level | ACTION-USE separated above 0 at all levels, both seeds | same |
| L2 and L3 decode a future speed better than "speed stays the same" | +0.168 / +0.157 at 1.2 s (L2), +0.233 / +0.223 at 2.4 s (L3) | `2026-11-07-v8f-latsem-result/` |
| The imagined speed answers accelerate vs brake | +1.35 to +3.63 m/s, 27 % to 43 % of the physical answer | same |
| L1 aims right and overshoots by about 2× | TRAIN-fitted shrink c 0.496 / 0.526; shrunk SKILL +0.0313 / +0.0310 at 0.2 s (its ceiling) | `afb3920a`, `AMENDMENT_WMACT_1` |
| Scene facts are encoded | lead present, the 7-class manoeuvre and a turn ahead at every level; agents within 30 m at L1 (R² +0.357 / +0.329) | `2026-11-07-v8f-latsem2b-result/` |

## The gaps
| # | gap | numbers | source | what it suggests |
|---|---|---|---|---|
| G1 | No level holds the car's motion: acceleration, yaw rate and the 2 s change are not encoded | R1 not separated for a, w, dv2, dpsi2 at any level | LATSEM | Each state reads ONE frame (window 1). Motion exists only in the predictor. MRATE's W1 tests this. |
| G2 | L1 overshoots and does not beat "speed stays the same" | c ≈ 0.5; SPEED-SKILL L1 not separated at both seeds | WMACT, LATSEM | L1's step (0.2 s) is below the token's motion resolution; its target may be dominated by noise. |
| G3 | Steering is barely imagined | w+ vs w− : L3 +0.012 / +0.062 of 0.5 rad/s; L2 seeds disagree; L1 the wrong sign | LATSEM | The lateral action has almost no pathway into the state. |
| G4 | No slow/fast split across levels (H-JEPA's signature) | TD-INCONSISTENT; the top level holds less speed than the bottom | LATSEM | Nothing forces the levels to differ except their step. |
| G5 | The lead gap is encoded at no level | S1 lead_gap not separated at both seeds anywhere | LATSEM2B | A pooled 4 × 10 token may not resolve a lead vehicle's distance; the WM loss never asks for it. |
| G6 | No level imagines the scene's future | S2 lead gap at the longest horizon: at most one seed separated | LATSEM2B | With G5, the gap probe has no headroom, so imagination of the gap cannot be measured yet. |
| G7 | The imagined scene does not answer the action the right way | S3-PARTIAL (speed removed) accelerate − brake = +0.72 to +3.30 m, the wrong sign everywhere | LATSEM2B | "Accelerate" moves the state toward faster-road scene types (a scene prior), not toward a closer lead. |
| G8 | Our only decoder of the imagination (nearest real frame) has no headroom | even the true future latent is not separated on R2 | LATSEM run 2 | We need a decoder that can show what a level predicts. |
| G9 | The planner reads no world-model output | by design today | v8F architecture | Imagination quality has no driving consequence yet; planning use is the strongest validation. |

## Process lessons from today (kept with the plan)
* A one-draw shuffled-label check judged by a bootstrap interval passes junk about 5 % of the time per row (18 / 360). Any per-row control needs a permutation reference (LATSEM2 run 1 INVALID, LATSEM2B fixed it).
* Probes of continuous targets are bounded below by large negative R² under the null, so the interval alone is nearly enough there; classification and imagination rows are where the permutation reference binds (58 % of permuted S2 rows pass the interval alone).

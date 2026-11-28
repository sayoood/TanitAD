# RESULT — the fallback-teacher bake-off: DINOv3 vs V-JEPA2 (P2)

**Measured** 2026-08-27 · **Evidence class** MEASURED (ours; dev-box RTX 4060; the
E-TRUNK-2 instrument imported verbatim — identical 5,617 frames, episode-disjoint
folds, dual-ridge with inner-split λ) · **Tier** T0-DIAGNOSTIC (decodability, never
capability) · **Raw** `raw/teacher_bakeoff.json`.

**Provenance:** official `facebook/vjepa2-vitl-fpc64-256` (326 M). ⛔ **No official
V-JEPA *2.1* repo exists on HF** (two probes, 2026-08-27); third-party re-uploads
were rejected on provenance. The survey's "2.1, Mar 2026" claim stays
PUBLISHED-SECONDARY until official weights exist.

## The panel (R² / AUC, out-of-fold; paired Δ = DINOv3 − V-JEPA2)

| target | DINOv3 | V-JEPA2 | pixels | Δ paired [CI95] |
|---|---|---|---|---|
| `n_agents` | **+0.360** | +0.239 | +0.007 | **+0.121 [0.052, 0.188] SEP** |
| `n_agents_log` | **+0.559** | +0.467 | +0.079 | **+0.092 [0.028, 0.162] SEP** |
| `lead_gap_m` | **+0.455** | +0.419 | +0.002 | **+0.036 [0.005, 0.071] SEP** |
| `occluded_frac` | **+0.123** | −0.052 | −0.005 | **+0.175 [0.091, 0.256] SEP** |
| `lead_is_vru` | **+0.926** | +0.882 | +0.665 | **+0.044 [0.001, 0.081] SEP** |
| `right_occupied` | **+0.842** | +0.812 | +0.481 | **+0.030 [0.002, 0.061] SEP** |
| `left_occupied` | +0.846 | +0.841 | +0.481 | +0.006 ns |
| `nearest_any_m` | +0.357 | +0.337 | +0.013 | +0.020 ns |
| `vru_ahead` | +0.704 | +0.686 | +0.619 | +0.018 ns |

## Verdict

⭐ **DINOv3 wins every target (9/9 point estimates), 6 of 9 paired-separated —
including the load-bearing environment targets** (`n_agents`, `lead_gap_m`,
`occluded_frac`). Both teachers clear the pixel floor everywhere. ⇒ **DINOv3 stays
the distill/fallback teacher for v7r; the PI's open decision 1 resolves with a
measurement.** The A8 `dinofrozen30k` fallback arm proceeds with DINOv3.

## Scope, stated so the loser is not over-buried

V-JEPA2 is a **video** model probed here through a minimal 2-frame tubelet on a
**static** decodability rig — its temporal strengths are structurally under-used.
That is the correct rig for the TEACHER role (per-frame distillation targets in our
pipeline), but this result must not be quoted against V-JEPA2 as a video WM, and a
future **dynamics-target** bake-off (predicting Δ-quantities) could reverse it.

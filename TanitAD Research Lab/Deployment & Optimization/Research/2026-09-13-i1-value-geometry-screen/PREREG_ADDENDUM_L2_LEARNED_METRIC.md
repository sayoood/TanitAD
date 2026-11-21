# PRE-REGISTRATION ADDENDUM — `E-DEP-VGEO-2` (POST-HOC lever): does a LEARNED metric over the frozen tokens clear the bar?

**Date** 2026-09-13 · Research Lab · **POST-HOC** — written after `E-DEP-VGEO-1` returned **PARTIAL**
(`raw/vgeo_screen.json`: short band ρ_latent_704 **0.4223 [0.3943, 0.4498]**, pixel floor **0.4950
[0.4651, 0.5243]**, paired latent−pixel **−0.0727 [−0.1067, −0.0404]**; all four controls at their known
values). SPEC.md's PARTIAL branch pre-named this lever: *"a learned metric head over frozen tokens … is the
next lever, not an encoder unfreeze."* Written **before** any learned-metric number exists.

## Design

* Same rows as VGEO-1 (every 4th row, 139 clips, 6,954 rows), same Δ ∈ [4, 160], same bands.
* **Clip-disjoint split, fixed before fitting:** clips shuffled with seed 20260913; **97 fit / 42 scored**.
  Every hyper-parameter and every fitted weight uses FIT clips only; scored clips are scored, never tuned on.
* Two learned metrics, applied identically to the latent (704-d mean-pool **and** 7,040-d 2×5 pool) and to
  the pixel floor (1,440-d) — a learned latent metric must beat a **learned pixel metric**, not the raw one:
  1. **M-diag** — non-negative diagonal metric: `d² = Σ_k w_k (a_k − b_k)²`, `w ≥ 0` fitted by NNLS of
     `log(Δ)` on the squared per-dimension differences (plus intercept) over `min(200 k, 40 M / d)` fit pairs (host-RAM cap set BEFORE running because a live MM arm leaves ~2 GB free: 704-d → 56,818; 1,440-d → 27,777; 7,040-d → 5,681 — ⚠️ n ≈ d for the 7,040-d diagonal fit, so that arm is underpowered by construction and is reported with its n and d).
  2. **M-pca** — PCA-whitened Euclidean, basis and whitening fitted on FIT rows, `k = 64` components
     (pixel and latent alike).
* Statistic and estimator unchanged: mean of per-clip Spearman ρ, clip-cluster bootstrap over the **42
  scored clips**, 2,000 resamples; paired differences on the same draws.
* Controls re-run on the scored clips with the fitted M-diag latent metric: **C-shuf ≈ 0**, **C-mut ≈ 0**,
  **C-const = 0**.

## Committed outcomes (short band decides)

| outcome | rule | consequence |
|---|---|---|
| **HEAD-UNBLOCKS** | best learned LATENT metric ρ ≥ 0.50 **and** paired (learned latent − learned pixel, same metric family) CI lower > 0 | I-1 proceeds with a metric head on the frozen trunk; no unfreeze needed for this purpose |
| **HEAD-CLEARS-BAR-BUT-PIXELS-TIE** | ρ ≥ 0.50 but paired CI includes or is below 0 | the goal-distance signal is recoverable but is not trunk-specific ⇒ the trunk adds nothing for value geometry; escalate: value belongs on an input path the trunk does not own |
| **HEAD-FAILS** | best learned latent ρ < 0.50 | frozen-trunk geometry is the binding constraint for a distance-shaped value ⇒ no value-head arm this cycle; I-1 blocked on the trunk-freeze (named blocker, MM decision) |
| **WORSE** | learned latent ρ below the raw VGEO-1 latent (0.4223) with CI excluding it | fitting overfits clip-specific structure; report and stop the metric line |
| **INVALID** | any control misses its value | no verdict |

⚠️ A one-seed frozen extraction; the CI answers the clip question only (`H-ESTIM-SEED-1`).

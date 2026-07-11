# Architecture & Inference — 2026-07-11

**D1 probe-capacity ladder + isotropy linkage — is trajectory info LOST or just less LINEAR?**
Plus theory-watch: Sub-JEPA (subspace SIGReg, arXiv 2605.09241) and the layerwise-probing
protocol (arXiv 2606.09646) that names exactly the linear-vs-MLP discriminator the loop is using.

> Calendar note: wall-clock run date **2026-07-11** (Wednesday agent, fired Saturday clock —
> narrative clock runs ahead per the Data-Eng/repo precedent). Worktree branch
> `worktree-arch-inf-20260711` (D-026). Budget: 4 web searches + 1 fetch, ~1.4 h, 1 measured
> experiment on the RTX 4060, $0.

---

## 0. Consumed since last run (2026-07-09) — the loop moved into my lane

Three developments landed in the Architecture/Inference lane between my last run and now; this
run builds directly on them rather than duplicating:

1. **D-027 accepted (`ff6a409`): rollout_k=4 adopted for all post-30k training** (Sayed).
   Evidence was the matched-compute K=4 arm (`859caa8`): 1-step `imag_rel` 8.13→**1.03**,
   recursive 14.5→**1.11**. **This is my backlog P0 #2b decision-graded and DONE by the loop** —
   the K-step thread I opened (2026-07-09, −64 % @1-step) is now settled. P0 #2b/#2c retire.
2. **REF-A DINO precompute shipped (`cda93df`): `stack/scripts/dino_precompute.py`**
   (v3-with-v2-fallback, latest-frame per-timestep, fp16 token grids) — my backlog P1 #4
   (frozen-DINOv3 WM arm) is now in motion by the loop. I leave it to the loop/pod and keep
   #4 as a watch item.
3. **D1 probe-capacity investigation OPENED TODAY by the loop (`0284a5c`):**
   `stack/scripts/d1_probe_capacity.py` — "is D1 position info LOST or just less LINEAR?" It
   fits ridge (3 α) + one MLP on two checkpoints (14k frozen vs current) and compares ADE@1s.
   **This is the live question and it is squarely mine — this run advances it.**

Also consumed: **Monday** (Tools&DevEnv — CARLA graphics-pod recipe gated on one `vulkaninfo`
probe; test-suite cold-I/O is Drive hydration, pin `stack/` offline); **Tuesday** (Data-Eng —
PhysicalAI-AV R1 96 % reachable from cached egomotion; WorldModel-Synthetic-Scenarios OpenMDW-1.1
ungated, pose field UNVERIFIED → D-022 proposed); **D-028** (recency-first arXiv listing scan now
mandatory; world-model/E2E releases are my seam).

## 1. The question, sharpened by theory (why linear-vs-MLP is the right probe)

D1 (open-loop ADE) regressed step-14k→21k (5.18 m → 11.52 m, camera-frame; small-sample preview).
The loop's discriminator asks the standard **LeWM/probing question**: freeze the encoder, train a
**linear probe** (tests whether the target is *directly, linearly accessible*) and an **MLP probe**
(tests whether the info is *present but nonlinearly encoded*). If the MLP recovers what the linear
probe misses → info intact, readout mismatch; if both fail → true information loss.

Theory pins why this is the right axis and predicts a specific outcome:
- **arXiv 2606.09646** (layerwise probing of video foundation models, U. Amsterdam, Jun-2026)
  states the protocol verbatim: linear probe = "directly accessible"; MLP probe = "present but
  not linearly encoded" — and finds probes that model temporal dynamics do best for V-JEPA.
- **arXiv 2605.26379** (When Does LeJEPA Learn a World Model?) — SIGReg gives *linear + orthogonal*
  identifiability; the isotropic Gaussian is the **unique** bias/variance minimiser for OLS/ridge.
  **Prediction I pre-registered:** if the readout is anisotropic (I measured
  `iso_ratio_active=0.250` at step-6500, 2026-07-10 orthogonality note), a LINEAR trajectory probe
  should *underperform* a nonlinear one — a large positive gap.
- **arXiv 2605.09241 (Sub-JEPA — Subspace Gaussian Regularization)** — regularise the
  *principal-variance subspace* toward Gaussian instead of all 2048 dims uniformly. This is a
  candidate **remedy** for the `iso_active` shortfall: full-dim SIGReg wastes its budget on the
  dead 2027-dim tail (global iso 2e-8); a subspace variant targets the active-k directions the
  planner actually uses. (Abstract-level; exact isotropy/probe numbers not extractable from the
  PDF — logged as a lever to characterise, not an adopted result. G-AI2.)

## 2. Experiment (G-H) — probe-capacity ladder + in-run isotropy, step-6500, RTX 4060

`probe_capacity_ladder.py` (intake `2026-07-11-d1-probe-capacity-ladder/`) extends the loop's
2-rung/2-checkpoint script into a **capacity ladder** and measures the **isotropy of the same
latents in the same run** so the mechanistic bridge is apples-to-apples. 12 comma2k19 val eps →
**408 latents (dim 2048)**, route-split by episode parity (I3-style, n_train=n_val=204), ADE@1s and
@2s, ego-frame targets. **36 s, $0.** Ladder run twice: on the **raw 2048-dim** latent and on the
**top-active_k (19) PCA subspace** (basis fit on train only).

**Isotropy (same latents) — cross-checks the 2026-07-10 instrument:**

| metric | value (07-11) | 07-10 branch | reading |
|---|---|---|---|
| `active_k` (99 %-energy knee) | 19 | 21 | ~tens-dim active subspace (spectral-sizing agrees) |
| `iso_ratio_active` | **0.269** | 0.250 | **anisotropic** within the used subspace (target →1) |
| `cond_number_active` | 230.6 | 246.3 | steep active spectrum |
| `participation_ratio` | 4.68 | 4.92 | ~5 dominant directions |

**Probe-capacity ladder — held-out ADE (metres, pose frame):**

| probe | raw-2048 @1s | PCA-19 @1s | raw-2048 @2s | PCA-19 @2s |
|---|---|---|---|---|
| `linear_ols` (unreg.) | **24.35** | 10.77 | 43.81 | 21.58 |
| `ridge_a1` | 10.31 | 10.71 | 20.82 | 21.46 |
| `ridge_a10` | 11.42 | 10.21 | 22.83 | 20.45 |
| `ridge_a100` | 11.65 | **8.84** | 23.20 | **17.55** |
| `mlp_256` (1 h) | 12.89 | 14.49 | 24.98 | 32.51 |
| `mlp_256×2` (2 h) | 11.93 | 10.18 | 23.49 | 24.33 |
| **best linear / best MLP** | 10.31 / 11.93 | **8.84 / 10.18** | 20.82 / 23.49 | **17.55 / 24.33** |
| **gap = lin − mlp** | −1.62 (**−15.7 %**) | −1.33 (**−15.1 %**) | −2.68 (−12.9 %) | −6.78 (**−38.7 %**) |
| zero-motion reference | 18.36 | 18.36 | 36.71 | 36.71 |

## 3. Findings (measured, honest)

**F-1 — the raw-2048 probe is underdetermined (D≫N); this confounds the loop's live d1 script.**
With 12 eps → n_train=204 and a 2048-dim latent, `linear_ols` overfits catastrophically
(**24.35 @1s**) while regularised ridge lands at 10.31 — a **2.4× swing driven by regularisation
strength, not capacity.** The loop's `d1_probe_capacity.py` uses the identical 12-ep/stride-8
collection, so its ridge-vs-MLP comparison sits in the same D≫N regime. **Fix (actionable):
PCA-reduce to `active_k` before probing** (basis on train only) — makes N>D and removes the
confound. In the PCA-19 subspace `linear_ols` drops 24.35→10.77 and the best probe is a clean
ridge (**8.84 @1s**, well below the 18.36 zero-motion floor → the top-19 subspace carries ~52 %
of the 1-s motion signal, linearly).

**F-2 — no nonlinear advantage at step-6500 (gap is NEGATIVE in every setting): −15 %/−13 %
(raw), −15 %/−39 % (PCA).** The MLP never beats the best linear probe. Taken at face value this
**disfavours the "less-linear" (info-present-but-nonlinearly-stored) branch** — the trajectory
info that *is* in the latent is linearly accessible; adding nonlinear capacity extracts nothing
more. **Caveat (P8, decisive for grading):** at n_train=204 the MLP is **data-starved**, so its
underperformance is *partly expected regardless of the linearity question*. So F-2 is
**directional, not decision-grade**: it is consistent with "linearly accessible," but a clean
info-lost-vs-less-linear verdict needs many more samples (≥50 eps) so the MLP is not starved.

**F-3 — anisotropy does NOT tax the linear trajectory probe (my pre-registered prediction is
REFUTED).** I predicted `iso_active`=0.27 ⇒ linear probe worse than MLP. Observed: linear ≥ MLP
throughout. **Ridge already absorbs the covariance anisotropy** (ridge *is* the isotropic-prior
estimator), so a non-white covariance does not, by itself, hide trajectory info from a linear
readout. Honest correction: the 2605.26379 orthogonality precondition matters for **latent-space
planning regret** (rotation-invariant cost, D4–D6), **not** for external-target probe
recoverability (D1). The isotropy admissibility gate (07-10) keeps its meaning for the *optimal-
planning* claim; it does **not** explain a D1 ADE shortfall. Negative result, logged (P8).

**Net for the live D1 investigation (hand-off to the loop/pod):** the "less-linear" explanation
for the D1 regression is **not supported at step-6500** once the probe is properly powered — but
the test must be re-run (a) PCA-reduced, (b) at ≥50 eps, (c) on the actual 14k-vs-21k pair — before
it is decision-grade. If a well-powered pod run *also* shows no positive gap, the D1 regression is
**not** a nonlinear-readout artifact → escalate the search toward coordinate-frame / highway
normalisation (cf. the D3 "highway normalization artifact", `c0b22b7`) or genuine drift, **not** a
decode-capacity fix. D1 remains the known structural-FAIL gate (camera-frame ADE unmeetable;
D4–D6 arbitrate — D-004).

## 4. Recommendations (actionable, G-B / G-AI1)

1. **Patch `stack/scripts/d1_probe_capacity.py` to PCA-reduce to `active_k` before probing**
   (train-only basis) and raise `--episodes` ≥ 50. Without this the ridge-vs-MLP read is a
   regularisation artifact (F-1). Gate that would falsify: none — this is instrument hygiene, not
   an architecture change. Shipped as this intake; proposed target `stack/scripts/`.
2. **Do not motivate any decode-capacity / nonlinear-head change from D1** until a well-powered
   probe shows a positive gap (G-AI1: no gate, no change; D-004: D1 is BLOCKED/structural-FAIL).
3. **Characterise Sub-JEPA subspace-SIGReg (2605.09241) as the `iso_active` remedy** — the
   isotropy gate, not D1, is the gate it would move (raises `iso_ratio_active` toward the
   optimal-planning precondition). Design-note first, escalate before any trained-config change
   (D-018 Tactic). Backlog P1.
4. **Merge the 2026-07-10 orthogonality branch** (`worktree-arch-inf-20260710`, still unmerged in
   main per the W29 report) — this run cross-validates its numbers (iso_active 0.25↔0.27,
   active_k 21↔19) and depends on them conceptually. Orchestrator triage item.

## 5. Ledger / decisions
- **H3 / H5** — no status change (P8): F-2/F-3 are instrument findings on one early checkpoint, not
  a hypothesis test. Evidence rows added: anisotropy present but linear-probe-benign; K-step
  settled at K=4 (D-027).
- No new decision proposed. D-018 classification of everything here: **EXECUTE** (instrument
  hygiene, intake, measurement) — nothing touches the trained config.

## Sources
- [arXiv 2605.26379 — When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379)
- [arXiv 2605.09241 — Sub-JEPA: Subspace Gaussian Regularization](https://arxiv.org/pdf/2605.09241)
- [arXiv 2606.09646 — Do Video Foundation Models Understand Intuitive Physics? A Layerwise Probing Analysis](https://arxiv.org/abs/2606.09646)
- repo: `stack/scripts/d1_probe_capacity.py` (`0284a5c`), `stack/scripts/dino_precompute.py` (`cda93df`), K=4 arm (`859caa8`), D-027 (`ff6a409`)
- repo: `Architecture & Inference/Research/2026-07-10-orthogonality-instrument-and-isotropy-theory.md` (branch `worktree-arch-inf-20260710`)
- artifact: `Implementation/incoming/2026-07-11-d1-probe-capacity-ladder/` + `Research/2026-07-11-probe_ladder_step6500.json`

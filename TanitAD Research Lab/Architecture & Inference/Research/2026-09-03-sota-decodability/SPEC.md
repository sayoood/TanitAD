<title>SPEC - SOTA pass: representation and decodability (theme 4 of 4)</title>

# SPEC - E-ARCH-SOTA-REP-1: transition-level probes, what strong systems decode, and whether anyone clears our L3

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-sota-decodability/` · Research Lab overnight pass 2026-09-03 · literature only · 0 GPU
**Written BEFORE any primary of this pass was opened.** Staged in the same turn it was written.
**Trigger.** `V7_LAUNCH_GATE` P5 (L3: the predictor adds nothing over `z_t`); `LAB_BACKLOG` rows P-4, P-5, GS-1, GS-9; readiness report §E "P5/L3 (+P3 dual read)".

## 1. Our MEASURED state

| fact | value | source (class · tier) |
|---|---|---|
| L1 (rank) PASSED | participation 3.80/3.62 (2k) → **25.58/26.96** (`rdw8p30k`, 30k parity) | `MODEL_REGISTRY.md` §13.0 · MEASURED · T0 |
| L2 (environment content) PASSED target-specifically | `splitp30k` `n_agents` **+0.3881** > frozen DINOv3 **+0.2754**; vs raw-pixel floor +0.6037 (t 30.86, 24/24) | §13.0d · MEASURED · T0 |
| … but NOT on the other targets | `lead_range_m` **−0.1611** (worst arm, below the raw floor); `d_ego` −0.0399 below the constant; P-5: behind DINOv3 on 4 of 5 targets; two panels disagree on `n_agents` (+0.1220/+0.0998 lead-matched vs +0.3881/+0.2754) — reconcile before quoting | §13.0d; `LAB_BACKLOG` P-5 (INHERITED) |
| `lead_gap_m` is a SUPERSEDED target (C150): 4.8 % no-lead frames at an 80 m default carried 59.9 % of the variance; the readout's 128→64 projection, not the pooling, is the cause (E-DEC-25) | §13 preamble · MEASURED · T0 |
| L3 FAILED: the predictor adds nothing over `z_t` | `splitp30k` predictor delta −0.0008 / −0.0320 / −0.0158, t −3.69 / −5.62 / −6.26 | INHERITED — readiness report §B citing `V7_RECIPE_AND_SCALEUP.md:104` (raw not re-opened) |
| ceiling beyond drift is small but non-zero and PIXEL-borne; the predictor adds −0.0023 (t −1.84, null) | E-DEC-63 · MEASURED · T0, `rdw8p30k`, n=7,280, 80 held-out clips |
| the low-variance-subspace hypothesis is CLOSED negative (30 matched null draws; more draws made p worse) | E-DEC-62 · MEASURED · T0 |
| our probe panel: cross-fitted ridge, PCA basis + λ fit on the FIT split only, constant control at exactly 0.0000, raw-pixel floor, paired LOEO over clips | §13.0 estimator row · MEASURED |
| our probes are STATE probes only (`z_t` → attribute); no transition-level rung (`Δz` → `Δx`) exists | GS-9 · INHERITED |
| `2607.27017` (banked, abstract-only): "only the full multimodal objective forecasts force beyond a persistence baseline"; certificate-gated protocol | INHERITED from KNOWLEDGE_BASE 2026-08-31 — read in full here |

## 2. Questions

- **Q-R1.** What is Delta-JEPA's transition-level probe protocol (Table 5): targets, probe families (linear / MLP), split rule, seeds, and the numbers — and what do they predict for a `Δz → Δx` probe on our banked latents?
- **Q-R2.** Which quantities do strong latent world models decode — ego kinematics (speed, yaw rate), agents (count, range), free space / occupancy — with which readout family (linear, MLP, attention pooling over tokens) and which CONTROLS (constant, raw-input floor, shuffled)?
- **Q-R3.** Does any primary compare the PREDICTOR's output against the encoder of the current frame (a persistence baseline) on a physical target — i.e. does anyone clear our L3 — and with what recipe?
- **Q-R4.** Is there MEASURED evidence that probes on `[z_t, z_{t+1}]` concatenations pass on endpoint cues (the GS-1 leak) — beyond Delta-JEPA's argument — and what probe shape avoids it?
- **Q-R5.** Do the field's readouts at token level (attention pooling vs. our 4×8 grid + 128→64 projection) explain our `lead_range_m` failure (E-DEC-25)?

## 3. Hypotheses — both outcomes committed

| id | hypothesis | outcome A | outcome B |
|---|---|---|---|
| **H-SOTA-R1** | The field's probe protocols carry the controls we require (constant-only, raw floor, n and d printed) | their numbers are comparable to ours and become priors | they do not ⇒ their numbers are NOT comparable to ours; RESULT.md says so per paper and no field number enters our tables as a bar |
| **H-SOTA-R2** | Transition-level probes (`Δz → Δx`) are reported with linear AND MLP heads, and reveal structure a state probe cannot see | GS-9 is pre-registered with their protocol + our controls on `postrain30k` / `postrain30k_freeze` / `splitp30k` (the dual P3/P5 read, readiness §E) | no primary reports them beyond Delta-JEPA ⇒ GS-9 runs on Delta-JEPA's protocol alone and is a TanitAD-novel rung |
| **H-SOTA-R3** | ≥ 1 primary reports the predictor beating a persistence / current-frame baseline on a physical target (our L3) | L3 is achievable; the recipe that achieved it is the prior for the v7f objective and P5's closing criterion is calibrated to it | nobody tests L3 ⇒ P5's bar is TanitAD-novel; the paper states it as a contribution, and the gate's criterion stays as written |
| **H-SOTA-R4** | Attention-pooled / token-level readouts are the field's standard for fine-geometry targets and grid-pooled projections are known to lose range | E-DEC-25's mechanism is field-corroborated; the readout re-architecture (GS-4's expensive branch) has a published prior | no such evidence ⇒ GS-4's cheap branch (explicit geometry loss) runs first, as already proposed |

## 4. What would change our plan

1. A published `Δz → Δx` protocol ⇒ GS-9 runs on banked checkpoints (0 GPU beyond probes) before the v7f freeze; the P3 dual read comes for free.
2. A published L3 clearance ⇒ P5's criterion is recalibrated to that recipe's effect size, and the recipe is a v7f candidate.
3. Evidence that our readout family is the outlier ⇒ GS-4's ordering flips (re-architect before the geometry loss).

## 5. Method and admissibility

As theme 1: banked PDFs only (tag `sota-2026-09-03-decodability`), ≥ 3 primaries read in full, every number stamped, agreement/disagreement with L1/L2/L3 stated per target, cheapest discriminating probe with both outcomes committed. Any proposed probe carries constant-only + raw-pixel floor + printed n/d, fits λ/PCA on the fit split only, splits by trajectory/clip.

## 6. Falsifiers

- H-SOTA-R3 B is refuted by any banked table with a "predictor vs. current-frame/persistence" row on a physical target.
- H-SOTA-R1 A is refuted paper-by-paper if the probe table lacks a control that must read the no-information value.

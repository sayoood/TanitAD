# VERDICT_LADDER_INTERIM (Master Mind, 2026-11-09; eight of nine runs, R35b_s0 still training)
Card `PREREG_V8F_LADDER.md`. Tier **T0**, a world-model diagnostic in latent space, never a driving result. MEASURED: `RESULT_LADDER_READ.md`, `results/arm_*.json`, `DIAG_SHRINK.md`. Interval: episode-cluster bootstrap over the 141 eval clips, B 2,000, with the 199-draw derangement reference. It answers "another draw of episodes"; both training seeds are required. LONGITUDINAL and LATERAL appear through the IA-2 targets and the IA-3 gains; TACTICAL: n = 0, no tactical target is read here; STRATEGIC: PAUSED (PI item 25).

## The card's label, as committed
**BAND-MET on C0, S35, R35 and R35H, at both seeds:** all three levels beat Copy-Last at their band end and are healthy. Skill at L1 2.0 s / L2 6.0 s / L3 12.0 s, seeds 0 / 1:

| arm | L1 2.0 s | L2 6.0 s | L3 12.0 s |
|---|---|---|---|
| C0 (control) | 0.064 / 0.080 | 0.278 / 0.267 | 0.359 / 0.401 |
| S35 | 0.069 / 0.096 | 0.294 / 0.339 | 0.472 / 0.480 |
| R35 | 0.178 / 0.190 | 0.419 / 0.430 | 0.535 / 0.541 |
| R35H | 0.259 / 0.267 | 0.546 / 0.555 | 0.672 / 0.687 |

**No difference between arms is a lever effect yet:** the card requires it to exceed the R35b floor, and R35b is pending. The final verdict and the choice of world model of record wait for it.

## Beside, written after reading the labels, decides nothing (`diag_shrink.py`, `DIAG_SHRINK.md`)
The control passing raised a question: a predictor that shrinks towards the average beats Copy-Last in squared error with no knowledge of the future. Two baselines with no world model and no action, cross-fitted over two clip folds: SHRINK (mean plus scalar times the anchor offset) and LINEAR (ridge from the anchor latent).

* **The trivial baselines are real but small.** SHRINK against Copy-Last reads 0.05 to 0.08 at L1, 0.14 to 0.20 at L2, 0.18 to 0.26 at L3.
* **Built context (three states, the form the k-step loss trained).** R35 and R35H beat SHRINK and LINEAR at every level at both seeds, on interval and reference. World model against SHRINK: R35 +0.129 / +0.144 (L1), +0.326 / +0.338 (L2), +0.429 / +0.441 (L3); R35H +0.201 / +0.208, +0.433 / +0.449, +0.583 / +0.579.
* **C0 and S35 at L1 do not clearly clear SHRINK** (C0 +0.007 and +0.026, not separated; S35 +0.012 not separated, +0.043 separated). So the Copy-Last bar is lenient at L1, and the card's label on the control there says less than it appears to.
* **Planning form (context 1): L1 does not hold.** No arm beats Copy-Last at L1 2.0 s (-0.059 to -0.012), and SHRINK beats the world model on all eight runs (world model against SHRINK -0.072 to -0.121, interval wholly below zero on all eight). L2 and L3 hold in this form too (R35H L2 0.461 / 0.449, L3 0.659 / 0.678 against Copy-Last).

## Action response (IA-3, beside, no bar; gain 1.0 = right size)
Longitudinal gain at L1 2.0 s: C0 0.300 / 0.353, R35 0.384 / 0.301, R35H 0.529 / 0.480. Lateral gain at L2 6.0 s: C0 -0.030 / +0.014, R35 0.142 / 0.131, R35H 0.439 / 0.378. Lateral gain at L1 2.0 s stays near zero on every arm (-0.074 to -0.010). All are below 1.0.

## What this says
Training the ladder on the PI's bands produces prediction inside the 6 s and 12 s bands that a no-action baseline does not explain. At 2 s the result depends on giving the rollout three past states, and the operative lateral response is still near zero. Every planner read so far (PLANTEST to PT3F) used context 1, the form in which L1 does not hold.

## Next (Rule Zero)
1. **PLANTEST and PLANTEST2 on the ladder world models**, the gates every outcome keeps, with the rollout at context 3 as a labelled beside arm, because past latents are available at inference.
2. **R35b_s0 read** when it is banked; the final verdict, the lever effects and the world model of record follow.
3. **Measured-speed world model:** card frozen on this read; runs on box 2 after the band retrain (PI, 2026-11-09). The near-zero operative lateral gain goes into its design as a measured defect.

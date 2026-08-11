# Morning report — overnight autonomous session 2026-08-10 22:30Z → 2026-08-11 ~04:30Z

**Mandate executed: all aligned measures, no exception. 22 pre-registered gates measured in
~30 h; ~40 commits; two datasets/instruments shipped; one programme-defining convergence.**

## The headline: one defect explains four failures — and its repair is training now

The night's separate investigations converged: **the v5f trunk's predictor responds to
actions with the right sign (lateral 99.5 %) inside a clean 3-dimensional latent subspace —
but at ~¼ of the physical magnitude** (W3 gain median 0.27 vs unicycle-analytic; longitudinal
sign only 74–79 %). This single measured defect explains:
1. §1.12's action echo (closed-loop near-straight driving),
2. W4b feat/kin scoring failures (0.560/0.564 — features don't separate candidates by consequence),
3. W4c spatial-attention failure (0.661 — same memorisation gap 0.139),
4. W7's ceiling (roll-cost is the programme's FIRST CALIBRATED selector, ρ 0.399, but
   collapses at dense K because muffled rolls make similar candidates indistinguishable).

Fast per-candidate scoring is **retired by pre-registration** (three surfaces, one failure
mode). **Stage-A predictor-only post-training launched at 03:30Z** with W3's numbers as its
gates — and the step-400 monitor already shows **left-channel gain 1.10, sign 1.0** (from
0.27). Held-out verdict ~06:30Z; PASS → W7 + T1 rows re-run on the repaired trunk.

## v5.8f state (registry §1.14)

| | best arm (top8+kincost) | control (frozen argmax) | v5f baseline |
|---|---|---|---|
| selected ADE (T0, 881) | **0.4815 [0.393, 0.577]** | 0.7933 [0.641, 0.976] | 0.4011 |
| fan oracle | **0.1077** | 0.1077 | 0.1975 |
| selected accel MAE | **0.515 m/s²** | 0.774 | 8.10 |

Trades +0.08 m ADE for 16× kinematics + 2× oracle; the deficit is pure selection, which the
stage-A→W7 path attacks. W4 (unicycle emission) PASSED both gates — the load-bearing wedge
holds. Full CIs banked (episode-cluster bootstrap, arms CI-separated).

## WM-interpretability battery (the PI's proof programme) — 7 of 9 probes measured

- **P3/P6 (W3)**: sign ✓ / gain ✗ 0.27 (THE defect) / subspace ✓ 3-dim.
- **P1**: predicted latents decode driving state BETTER than encoded (speed R² 0.99 vs 0.73,
  curvature 0.84 vs 0.51) — caveat stamped (action conditioning contributes).
- **P2**: nuisance probes ran (episode-ID fold deviation documented); lead-gap probe
  instrument-suspect (class-agnostic join) — fix queued, not a model verdict.
- **P5**: T1 instrument VALIDATED (E1.4 byte-close: reproduces §1.12 exactly).
- **P7**: v5f pairing calibrated (ρ 0.49 ✓); both v5.8f score-arms FAIL (0.26 / 0.05) —
  W7's roll-cost restores calibration (0.399 at K=8).
- **P8**: harness + occlusion join built (39/40 eps, 195,805 boxes); first run failed on
  train-coverage — join extended (episode-disjoint clips 40–140) and REBUILDING now;
  training queues today. P4 rides P8's occluded split. P9 rides P1/P8 saliency dumps.

## Shipped datasets & instruments

- **Alpamayo-2-Super augmentation COMPLETE: 4,800/4,800 clips, 23,999 rows, 5 tasks**, card
  live on HF (road-class coverage, provenance, honest holes note).
- REF-C vs v5f planner comparison (one shared algorithm; jitter = 10 Hz offsets, vocabulary
  measured innocent; REF-C's "richness" = FEWER cleaner modes, entropy 0.97 vs 2.22).
- W7 roll-rerank driver, stage-A trainer, P1/P2 harness, W3 probe pack, P8 harness + join
  tool (now with --skip-episodes), t1_eval promotion — all committed with green CPU suites.
- VLM strategic-labeling design (PI directive) + E6 efficiency prereg + scaling-ladder
  scaffold (S1–S4) + W4b/W4c preregs (both consumed with verdicts).

## Today's plan (in order)

1. **Stage-A gate (~06:30Z)** → if PASS: W7 re-run on repaired trunk (the selection verdict),
   then **E1.4 T1 rows for v5.8f** (the PRIMARY-tier numbers).
2. P8 occupancy training on the extended join → the decoded-BEV reel (I1c) + P9 saliency.
3. Four-families completion on the banked v5.8f windows → **registry §1.14 release row + HF**.
4. Lead-gap probe fix (vehicle-class filter) → P1 lead target re-run.
5. Then the sequencing gate for the VLM strategic-labeling pilot (PH0) clears.

*Every number above is MEASURED with its artifact banked in this directory or the registry;
estimator and tier stamps travel with each. Failed gates are recorded as verdicts, not
excused.*

# RESULT — Representational fixes for latent self-prediction drift BEYOND the EMA teacher

`Research Lab daily pass 2026-08-28, Architecture & Inference. Literature-only
(0 GPU). Seed: the v7 central blocker — content/prediction dissociation, trained
encoders drifting toward self-predictable latents. In-house today: the Drive-JEPA
EMA-teacher bake-off arms (EMA lane) and PREREG_O14_FUTURE_OBS (pixel-auxiliary
lane, R2). This survey covers the OTHER three fix families, each with the paper's
MEASURED effect and a 19M-tiny-rig transfer read. All keys banked; --verify clean.`

## Findings first — the three families, measured

**F1 — Variance–covariance regularization: works WITHOUT any teacher, and the
placement that matters is BOTH branches, patch-level.** [PUBLISHED, banked]
- VICReg (lib `2105.04906`): variance (prevents norm collapse) + covariance
  (prevents informational collapse) applied to **each branch separately** at the
  expander output; the ablation table shows removing the variance term collapses
  outright, and var+cov alone — no EMA, no stop-grad, no negatives — reaches 68.6
  IN-1K linear (R50), with EMA/stop-grad adding only +0.2/+0.9 on top.
- C-JEPA (lib `2410.19560`, NeurIPS 24): diagnoses that **EMA alone does not
  prevent entire collapse in I-JEPA** — directly our finding, at ImageNet scale —
  and adds VICReg var/cov inside I-JEPA (student branch, patch level): ViT-L/600ep
  linear 72.9 → 73.7, fine-tune 83.5 → 84.5 (+0.8/+1.0), with visibly faster
  convergence. The gain is modest but the STABILITY claim is the point.
- LeJEPA (lib `2511.08544`, already banked): SIGReg — the strongest beyond-EMA
  candidate; replaces the entire heuristic stack (EMA, stop-grad) with an isotropic
  Gaussian projection test. Var-JEPA (lib `2603.20111`, 2026) makes anti-collapse
  implicit in an ELBO — tabular-only instantiation so far; watch, don't adopt.
- **19M transfer read:** var/cov cost is O(d²) in latent width — trivial at our
  d ≤ 2k. Placement per the two primaries: on z of BOTH student and teacher paths
  (or post-projector), NOT only on the predictor output. [ESTIMATED transfer;
  mechanism PUBLISHED]

**F2 — Predictor-side capacity is a THRESHOLD lever on what the encoder keeps —
and it runs in the OPPOSITE direction of the bottleneck intuition.** [PUBLISHED, banked]
- IWM (lib `2403.00504`): with a 12-layer generic predictor the encoder goes
  invariant (transformation MRR 0.00 — content discarded); same rig with predictor
  conditioning + 18 layers reaches MRR 0.79–0.85, and jitter-equivariance flips
  from **1/5 seeds to 4/5 seeds** on depth alone. Mapping to our pathology: the
  encoder discards what the predictor cannot carry — dissociation can be relieved
  by making the predictor DO MORE (capacity + conditioning), not less.
- The compression counter-evidence: NextLat (lib `2511.05963`): ADDING latent
  transition prediction to next-token training yields effective latent rank
  **160.1 → 52.7** with valid trajectories UP (97.0 → 98.7 %) and detour
  robustness UP (85 → 95 %) — lower rank accompanied BETTER world-model behavior;
  smaller latent dims helped on two of their tasks.
- **19M transfer read:** the literature supports BOTH directions depending on
  whether lost content is nuisance or signal — exactly why our rank-AND-decodability
  doctrine (and PhyLatent, lib `2608.05720`) refuses bare rank reads. Predictor
  depth/width is a cheap tiny-ladder axis; neither direction may be assumed.
  [HYPOTHESIS either way at our scale]

**F3 — Auxiliary targets: pixels-as-MAIN-target measurably lose to features; the
literature therefore brackets O14's R2 as an auxiliary-only design, with a
feature-target fallback ready.** [PUBLISHED, banked]
- V-JEPA (lib `2404.08471`, Tab. 5–6): controlled ViT-L pair, same masking —
  feature-prediction beats pixel-reconstruction baselines on ALL frozen downstream
  reads except IN1K (74.8 vs 75.1), +6 % on SSv2, and pixel methods only catch up
  under full fine-tuning with longer schedules. Pixel targets pull capacity toward
  appearance; as the MAIN objective they are the weaker representation learner.
- This does NOT argue against O14 (pixels as a small-w AUXILIARY anchoring the
  encoder to observation content — Dreamer-style, not MAE-style). It DOES supply
  the pre-committed fallback: if O14 reads FLOODED at both weights, the next arm is
  the same auxiliary with a **frozen-teacher FEATURE target** (DINO-WM/FROST-Drive
  lineage, libs `2411.04983`/`2601.03460`) rather than more pixel-weight tuning.
- **19M transfer read:** at 32×80 grey the R2 head cost is negligible; the
  V-JEPA efficiency argument (decoder cost) does not bind at our scale. [MEASURED
  in-house: O14 tiny gates already ran clean 2026-08-27, prereg Amendment B]

**F4 — Instrument note.** RankMe (lib `2210.02885`): entropy effective rank
correlates with downstream linear performance in- and out-of-distribution and
selects hyperparameters label-free — **except degenerate solutions at full rank**,
its own stated caveat. Together with NextLat's rank-down-quality-up row, the
standing rule is re-confirmed from two independent sources: a rank statistic alone
can read BOTH failure and success as the same number; only rank + decodability
rules. [PUBLISHED]

## What this changes for TanitAD (≤3 recommendations)

1. **Register a VICReg-placement pair on the v7-tiny ladder** (via
   `/TanitAD_ValidateAIDesign`, both outcomes pre-committed): champion + var/cov on
   z at BOTH branches (VICReg defaults scaled), vs champion. Primary read = the
   E-DEC-63 pixel-marginal + G-RANK/G-DECODE, same as O14 — giving the programme a
   controlled three-way: EMA lane (bake-off, running) vs absorb lane (O14) vs
   constrain lane (VICReg) on one instrument. C-JEPA's result predicts stability,
   not headline gains — pre-commit that a null on gains with cleaner drift is a
   POSITIVE outcome for the constrain lane.
2. **Add ONE predictor-capacity arm before any predictor-shrinking arm.** IWM's
   1/5 → 4/5 seed flip says capacity acts as a threshold on what the encoder may
   keep; if dissociation is the encoder shedding hard content, the cheap test is a
   deeper/conditioned predictor at 2k steps — NOT the bottleneck first. NextLat
   licenses the bottleneck arm only as the counter-arm of the same pair.
3. **If O14 ends FLOODED or R1-EQUAL, the pre-committed successor is the
   frozen-teacher feature-target auxiliary** (same head, DINOv2-class target,
   teacher never updated — no new EMA machinery), per V-JEPA's measured
   pixels-vs-features gap. Log this in the O14 COMMS so the fallback needs no new
   decision meeting.

## Searches that came up empty
- A paper naming our exact "content/prediction dissociation" (encoder drifting
  toward self-predictable latents DURING world-model training, measured by a
  content probe) beyond PhyLatent's dynamics-collapse modes: none found newer than
  PhyLatent (2 queries; raw/search_log.md). The dissociation instrument we built
  (E-DEC-63 marginal) appears ahead of the published state.
- Measured VICReg-term results at ≤ 50M params on video world models: none —
  every quantitative result above is ImageNet/ViT-B+ scale or toy-world; the tiny
  ladder validation is genuinely necessary, not a formality.

## Evidence table
| # | claim | class | source |
|---|---|---|---|
| F1 | var removal ⇒ collapse; var+cov alone 68.6; EMA/SG add ≤ +0.9 | PUBLISHED | lib `2105.04906` §6 tables |
| F1 | EMA alone insufficient (entire collapse); +0.8/+1.0 ViT-L | PUBLISHED | lib `2410.19560` Tab. 2 |
| F2 | predictor 12→18 layers: jitter equivariance 1/5→4/5 seeds; MRR 0.00→0.85 | PUBLISHED | lib `2403.00504` Tab. 2 |
| F2 | NextLat: eff. rank 160.1→52.7 with validity 97.0→98.7 % | PUBLISHED | lib `2511.05963` Tab. 1 |
| F3 | frozen eval: features > pixels everywhere but IN1K 74.8 vs 75.1; SSv2 +6 % | PUBLISHED | lib `2404.08471` Tab. 5–6 |
| F4 | RankMe correlates w/ downstream, full-rank degenerate caveat | PUBLISHED | lib `2210.02885` |
| — | O14 tiny gates clean at 2k (drift ≤ +4.5 % rel, DR inert) | MEASURED | `Project Steering/PREREG_O14_FUTURE_OBS.md` gates table |

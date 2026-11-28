# Unified 120° FOV canvas + masked periphery + foveated patching (Sayed idea, 2026-07-12)

## The idea (Sayed)
Do NOT crop PhysicalAI's 120° down to comma's ~51° (which discards the hazard-rich urban periphery).
Instead **pad comma UP to a common 120° canvas**, with comma's real ~51° center surrounded by a
**masked "unobserved" region**. Both corpora then live in one 120° geometry:
- **PhysicalAI:** full 120° real pixels (periphery observed).
- **comma:** real ~51° center + masked periphery (periphery unobserved).

## Why this is strong — three wins in one
1. **Preserves urban peripheral vision.** The wide FOV where pedestrians/cross-traffic/cyclists appear
   is kept for PhysicalAI, instead of thrown away. Directly addresses the urban weakness.
2. **Unified action→pixel geometry at 120°** (the D-016 goal) WITHOUT sacrificing FOV — consistency
   achieved by padding up, not cropping down.
3. **A FREE, REAL training signal for H15 imagination (our moat).** Today H15 trains on *synthetic*
   sector masking. Here, comma's masked periphery is a **real, always-available unobserved region**,
   and PhysicalAI provides the **ground-truth periphery** the imagination must learn to predict. The
   model learns: *"in comma I must imagine the periphery; in PhysicalAI I observe what actually tends
   to be there."* Cross-corpus, self-supervised imagination training — stronger than synthetic masks.
   Plus **temporal-reveal consistency** (H15 advection prior): what is peripheral-and-imagined now
   becomes central-and-observed as the ego turns/advances — a self-supervised check on the imagination
   in BOTH corpora. This is the epistemic-humility + imagination capability that is core to the thesis.
4. **Partial-observability robustness:** training with a genuinely limited FOV + mask teaches the model
   to drive under incomplete information (real cameras occlude; this is realistic), and is a MAE-style
   masked-modelling regularizer for the encoder.

## The real considerations (honest — this is an architecture change, not a config flip)
1. **Projection model.** A *rectilinear* 120° canvas has extreme edge stretch (tan60°→∞) — unusable.
   The 120° canvas must be **f-theta or cylindrical/equirectangular** (angle-linear). comma (rectilinear
   ~51°) is reprojected into the canvas center; PhysicalAI (native f-theta) maps naturally. One shared
   angle-linear canvas → truly consistent per-pixel angular geometry (arguably BETTER than today's
   rectilinear crop). Needs comma's and PAI's real intrinsics (we now have both — the audit found PAI's
   f-theta poly; comma's 910px is known).
2. **Resolution dilution (ties to the resolution issue).** At a fixed 256px canvas, comma's ~51° center
   occupies only ~51/120 ≈ 43% of the width → comma's real content drops to ~110px effective → LOWER
   acuity for the corpus that was our best. Unacceptable with uniform patching → forces the patching
   redesign below. Options: larger canvas (compute), or foveated patching (preferred).
3. **Masking fraction.** For comma, the periphery is the MAJORITY of the 120° canvas (~57% by width,
   more by area) → comma frames become mostly-masked. Risk: over-reliance on imagination for comma,
   or wasted tokens on mask. Mitigation: don't tokenize pure-mask regions (variable token count) OR a
   learned mask-token; and cap the masked fraction (e.g. pad comma to ~90°, not full 120°, as a
   compromise) — a tunable.
4. **Compute.** A wider canvas at useful resolution = more tokens. This is exactly why patching must
   change.

## Resolution + encoder patching optimization (Sayed's paired request) — the enabler
Uniform 16px patching over a 120° canvas is the wrong tool: it either dilutes the center (fixed 256px)
or explodes tokens (large canvas). The answer is **foveated / non-uniform patching**, which also
optimizes the standalone resolution issue we flagged earlier:
- **Foveated layout:** FINE patches (e.g. 8px) in the central cone (road-ahead detail, far small
  objects, comma's real content), COARSE patches (e.g. 24–32px) in the wide periphery (context,
  often-masked, less acuity-critical). Keeps far-object acuity where it matters, covers 120° at a
  token budget close to today's — the same lever the resolution-sensitivity probe was scoped to test.
- **Shared across both cameras:** one foveated patcher serves comma (center-heavy real content) AND
  PhysicalAI (needs wide coverage) — the center cone is high-res for both; periphery is observed for
  PAI, imagined for comma. Position embeddings encode the foveation so the ViT reads geometry
  correctly.
- **Production-friendly:** a FIXED foveated layout (static shapes) is TensorRT/Orin-safe (dynamic
  token counts break engine plans — Prod G-P2). This beats dynamic pruning for deployment.
- **Lineage:** this is the efficient-encoding direction from the Alpamayo sweep (decouple tokens from
  resolution/FOV) and the foveation family (H16); `ENCODER_MULTICAM_OPTIMIZATION.md` addendum.

## Pre-registered experiment (before adopting — proofs, not vibes)
Sequence AFTER the focal-bug fix + clean-geometry baseline retrain (don't block that):
1. **Resolution-sensitivity probe (cheap, existing 27k ckpt):** encode val at 128/256/384-interp,
   measure D1-probe ADE per curvature/far-hazard stratum → how much acuity is resolution-bound
   (the number that says whether center-res even matters). Falsifier: flat → resolution isn't binding.
2. **Foveated-patcher pilot:** implement the fixed foveated layout on the encoder; smoke-train a small
   model comma+PAI, compare token count + far-object probe vs uniform 16px at matched budget.
3. **Unified-FOV pilot (the full idea):** build the 120° angle-linear canvas (comma padded+masked, PAI
   full) for a subset; fine-tune from a baseline; measure (a) urban ADE (does periphery help?),
   (b) H15 imagination quality on the masked periphery (does PAI ground-truth teach comma's
   imagination? temporal-reveal consistency), (c) comma ADE guardrail (no regress from center dilution).
   **Pre-registered prediction:** urban ADE improves + imagination calibration improves without comma
   regressing. **Falsifier:** comma regresses (center dilution dominates) or periphery adds no urban
   lift → keep the 51° crop + defer wide FOV to H2 multi-cam.

## Recommendation & sequencing
- **NOW:** fix the focal bug, retrain REF-A/REF-B/flagship on the corrected ~51° geometry (clean
  baseline — needed regardless, unblocks everything).
- **NEXT architecture experiment (Phase-0.5):** this unified-FOV + foveated-patching program, staged
  as the three pilots above. It is the natural convergence of the FOV question, the resolution issue,
  the encoder-patching optimization, AND the H15 imagination moat — high upside, but a real change that
  earns its place by experiment, not adoption-by-enthusiasm.
- Propose as a new hypothesis **H17: unified-FOV masked-periphery training strengthens urban capability
  and imagination** (add to HYPOTHESIS_LEDGER once Sayed confirms direction).

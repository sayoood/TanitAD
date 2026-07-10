# Vision-Encoder Optimization for Multi-Camera (Phase 1 prep)

**Directed by Sayed 2026-07-09 night. Status: investigation kickoff — candidate techniques with
production-readiness first-pass; deep validation via the Sonnet literature workflow + the
Wednesday agent's measured experiments.**

## The problem shape

Phase 1 moves from 1 front camera to N cameras (PhysicalAI rigs: 4 fwd + 7 fisheye; realistic
target 4–7). Naive scaling: 7 × (9.4 ms encode) = ~66 ms/tick on the 4060-class proxy — eats the
entire real-time budget before the predictor runs. The encoder is already 60%+ of our tick; it
becomes ~90% multi-cam. Constraint that shapes everything: **whatever we adopt must survive
TensorRT/INT8 export** (the production path) — research tricks that don't compile are decoration.

## Candidate techniques — first-pass assessment (to be validated with measurements)

| Technique | Idea | Expected win | Production readiness (first pass) |
|---|---|---|---|
| **Weight-shared encoder + camera-ID embedding** | one encoder, N passes, cam embedding like our domain tag | params flat; latency = N× (batchable! N cams = batch-N → amortizes like our K=9 select: measured 5.7 ms for 9) | ✅ trivial export; FIRST choice — batched multi-cam encode likely ~2× single-cam latency, not 7× |
| **Token merging (ToMe/ToMeSD-class)** | merge redundant tokens between blocks | 30–50% encoder FLOPs, quality −<1% reported | 🔶 dynamic shapes hurt TRT; static-merge-ratio variants export; needs a measured parity test |
| **Token pruning by driving relevance** | drop sky/hood tokens early (fixed masks from calibration!) | 20–35% tokens statically | ✅ STATIC mask = free and export-trivial; dashcam geometry gives it away; try FIRST |
| **Lower-res peripheral cams** | front @256, others @128–160 | ~2–3× on peripheral encodes | ✅ pure config; matches human foveation logic |
| **Frame-stack folding** (already ours) | 9-ch stacks = 3 frames in ONE pass | already banked (3× vs naive) | ✅ shipped (D-015) |
| **Cross-cam late fusion** | encode per-cam, fuse tokens in 1–2 shared blocks before readout | keeps per-cam cost, adds small fusion cost | ✅ standard; the readout grid extends naturally |
| **EfficientViT / linear-attn backbones** | replace ViT blocks | 2–4× claimed | 🔶 quality on SSL-from-scratch unproven for us; big architecture change — Phase 2 candidate, not Phase 1 |
| **Distillation to a smaller deploy encoder** | train big, distill small | 2–3× at deploy | ✅ orthogonal, production-standard; pairs with any of the above |
| **FP8/INT8 (ModelOpt path)** | quantize | 2–4× on Orin-class | ✅ already in the Production stream (TRT fp16 next; INT8 via ModelOpt — native-TRT ViT INT8 is the known trap) |

## Recommended attack order (cheap+safe first, each with a falsifier)

1. **Batched multi-cam encode measurement** (weight-shared, batch-N) — extend
   `latency_cnce_baseline.py` with N∈{1,4,7} batched encodes. Expected: 7-cam ≈ 2–2.5× single-cam
   latency, NOT 7×. Falsifier: if batching doesn't amortize (VRAM-bound), rethink. **Hours, local.**
2. **Static sky/hood token mask:** measure token-relevance from the trained 14k/30k encoder
   (attention-to-readout maps) → fixed per-rig masks → % tokens dropped at zero gate delta.
   Falsifier: D1/D2 delta > noise ⇒ reduce mask. **Days, local.**
3. **Peripheral-resolution study** once multi-cam data exists (Phase 1 data).
4. **ToMe static-ratio parity test** on our encoder + TRT export check (Production stream Saturday).
5. Distillation + quantization ride the existing Production roadmap.

## Deep-research questions (→ Sonnet workflow tomorrow)
- Published multi-camera encoder latency numbers on Orin (BEVFormer-lite class, NVIDIA DRIVE
  production patterns) — what do shipping stacks actually do?
- ToMe-class token reduction: TRT-compatible implementations in 2025/26? Measured accuracy at
  fixed merge ratios on driving data?
- Any 2026 work on shared-backbone multi-cam SSL (our exact setting: from-scratch SSL + N cams)?
- FP8 ViT inference on Thor: real numbers vs INT8?

## Owner & cadence
Architecture agent (Wed) owns the measured experiments (#1 first — it's a G-H-compliant experiment
with numbers in hours); Production agent (Sat) owns export-compatibility checks; findings roll into
the Phase-1 plan's encoder section. The architecture-design workflow panel takes these as inputs.

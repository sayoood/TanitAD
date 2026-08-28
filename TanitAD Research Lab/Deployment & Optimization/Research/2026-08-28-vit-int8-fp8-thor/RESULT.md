# RESULT — INT8/FP8 PTQ of ViT encoders for Jetson Thor (JetPack 7, sm_110): current best practice

`Research Lab daily pass 2026-08-28, Deployment & Optimization. Literature +
community-primary pass (0 GPU). Seed: the B1 production decision — quantisation
validation is its gate. Builds on LAB-RUN-001's component-wise finding (lib
2607.08029); today adds the ViT-specific PTQ recipe, calibration-size evidence,
the dense-head cliff, and one Thor-specific silent-failure trap that directly
threatens the B1 gate's validity.`

## Findings first

**F0 — ⛔ THE B1-GATE TRAP: TensorRT on Thor sm_110 has a DOCUMENTED silent
FP32 fallback under FP8/FP4 builder flags.** [PUBLISHED-WEB, verified fetch
2026-08-28; NOT a banked PDF — community bug report]
NVIDIA/TensorRT issue #4590 (opened 2025-10-04, no NVIDIA response in-thread at
fetch time): on Jetson Thor (compute capability 11.0), TensorRT 10.13.3.9
**accepts `BuilderFlag.FP8`/`FP4` and silently builds an FP32 engine** — no
warning; detected only via plan size (~FP32-sized) and `DataType.FLOAT` outputs.
A separate developer-forum report has FP4 failing on TRT 11.0.0.114 on Thor.
Consequence for us: a quantisation validation that trusts builder flags would
score an FP32 engine, PASS accuracy trivially, and certify a speedup that does
not exist — a false-positive generator for the B1 gate, same family as the
`df`/cgroup scope traps. **Precision must be verified by engine inspection +
measured latency, never by flags.** (In-house verification on Thor is required
before any B1 read; 0-GPU today, Thor access embargoed.)

**F1 — Calibration set size: tens of images suffice for ViT PTQ, and the result
is insensitive to more.** [PUBLISHED, banked]
- PTQ4ViT (lib `2111.12293`): **32 calibration images**, quantizes most ViTs in
  minutes; #ims=128 mainly increases time, top-1 "varies slightly" (their
  insensitivity ablation, Tab. 4). W8A8 near-lossless: **< 0.5 % top-1 drop**
  (ViT/DeiT/Swin), where naive base PTQ loses > 1 % — the gap comes from
  post-softmax and post-GELU distributions, not from weights.
- RepQ-ViT (lib `2212.08254`): 32 samples (classification), **1 COCO sample**
  (detection/instance-seg) — calibration data is not the bottleneck.
- ⇒ our 40-episode parity val cache is over-sufficient; no new data needed.

**F2 — Per-channel vs per-tensor: the published resolution is BOTH — calibrate
per-channel, deploy per-tensor via scale reparameterization.** [PUBLISHED, banked]
RepQ-ViT's core mechanism: LayerNorm activations carry severe inter-channel
variation → calibrate channel-wise, then fold scales into adjacent weights so
INFERENCE runs hardware-friendly per-tensor/layer-wise; softmax activations get a
log-√2 quantizer. Standard setting throughout the field: channel-wise WEIGHTS +
per-tensor ACTIVATIONS. At W4/A4 prior art degrades hard (FQ-ViT to 0.1 % top-1,
i.e. infeasible) — reparameterization is what makes low-bit survivable.

**F3 — The dense-prediction cliff is real, measured, and lives at ≤ 6-bit — not
at W8A8.** [PUBLISHED, banked]
- RepQ-ViT (COCO, Cascade Mask R-CNN Swin-T, prior methods at W4A4): **box AP
  −23.2, mask AP −19.3** — collapse on dense heads while classification W8A8 is
  near-lossless.
- PTQ4SAM (lib `2405.03144`, SAM = the dense-prediction stress test): root causes
  are a **bimodal post-Key-Linear activation distribution** and heterogeneous
  post-softmax distributions across self- vs two-way-cross-attention. Measured
  (Faster R-CNN prompt, SAM-L): FP 36.4 AP → W6A6 statistics-PTQ 32.0–33.1
  (−3 to −4), **W4A4 OMSE 5.4 AP** (catastrophic); their fixes reach lossless at
  6-bit on instance segmentation (~0.5 % drop, 3.9× theoretical acceleration).
- ⇒ W8A8 on dense heads: no published cliff; W6A6: small, method-dependent drop;
  W4/NVFP4 on DENSE heads: unsupported by every number we found — 4-bit belongs
  to weights-only/memory relief, not dense activation paths.

**F4 — INT8-first remains the right default on efficiency grounds; FP8 is a
targeted tool for outlier layers, not a blanket upgrade.** [PUBLISHED, banked]
Qualcomm study (lib `2303.17951`): dedicated FP8 compute is **at least 50 % less
efficient in area/energy than INT8**; INT8 exactly represents ~90 % of FP8-E4's
range once scaling is free; W4A8 matches or beats FP8-E4 on several nets at ~50 %
less compute. The transformer exception, stated by the same paper: ViT/BERT
carry large-outlier layers that favor more exponent bits (FP8-E4) — but
per-channel + reparameterized INT8 (F2) addresses the same outliers at lower
hardware cost. On Thor specifically, FP8's Transformer-Engine path must first be
PROVEN engaged (F0) before any FP8 claim is admissible.

## What this changes for TanitAD (≤3 recommendations)

1. **Harden the B1 gate protocol against F0 before anything else** (0-GPU
   deliverable: a checklist + script skeleton): every quantized engine read
   requires (a) `IEngineInspector` per-layer precision dump showing INT8/FP8
   layers actually present, (b) plan-size sanity vs FP16 baseline, (c) measured
   latency delta, (d) accuracy on the four metric families. A build that passes
   accuracy but shows no latency/size change is a FAILED build (silent FP32),
   not a robust quantisation.
2. **Recipe order for the trunk on Thor**: W8A8 INT8, channel-wise weights +
   per-tensor activations, 32–512 parity-val windows calibration,
   RepQ-ViT-style reparameterization if LN inter-channel variation appears
   (capture per-layer activation stats on one parity batch first — CPU-side,
   doable before 12:00). Expect ≤ 0.5 % scalar-head delta (PTQ4ViT) — treat a
   larger delta as an outlier-layer signal, not a recipe failure. FP8-E4M3 only
   for identified outlier layers AND only after F0 verification; NVFP4
   weights-only where memory matters; **never ≤ 6-bit activations on dense
   heads** (F3).
3. **Dense-head-first validation**: B1's accuracy read must score the dense
   readout/decoder paths (the four families), not the scalar heads — the
   measured cliff is invisible to top-1-style metrics until far past the point
   where dense outputs have degraded (RepQ-ViT: cls near-lossless while dense AP
   −23). This folds into H-DEPLOY-1's seam-dominated hypothesis: per-seam AND
   per-head deltas, never one pooled number.

## Searches that came up empty
- A published INT8/FP8 accuracy study ON Jetson Thor silicon (any model): none
  found — Thor-specific evidence is currently community bug reports + vendor
  blogs; every accuracy number above is from A100/desktop-class measurement.
  Stated consequence: transfer of the accuracy numbers to Thor is [ESTIMATED]
  until our own gate runs; the RECIPE (calibration size, per-channel policy,
  bit-width floors) is architecture-level and transfers.
- FP8 PTQ results for ViT DENSE heads specifically (segmentation/detection with
  FP8 activations): not found in this pass — the FP8 literature is
  classification/LLM-dominated.

## Evidence table
| # | claim | class | source |
|---|---|---|---|
| F0 | TRT 10.13.3.9 on sm_110: FP8/FP4 flags → silent FP32 engine | PUBLISHED-WEB (community, unresolved; needs in-house verification) | github.com/NVIDIA/TensorRT/issues/4590, fetched 2026-08-28 |
| F1 | 32 calib images; < 0.5 % W8A8 drop; insensitive to #ims | PUBLISHED | lib `2111.12293` |
| F2 | per-channel calib → per-tensor deploy (reparam); W4A4 prior art infeasible (0.1 %) | PUBLISHED | lib `2212.08254` |
| F3 | dense cliff: W4A4 box AP −23.2 / mask −19.3; SAM-L W4A4 OMSE 5.4 vs FP 36.4; 6-bit lossless with fixes | PUBLISHED | libs `2212.08254`, `2405.03144` |
| F4 | FP8 ≥ 50 % less area/energy-efficient than INT8; INT8 covers ~90 % of E4 range; ViT/BERT outliers favor E4 | PUBLISHED | lib `2303.17951` |
| — | Thor = Blackwell, CC 11.0 (sm_110), native NVFP4, JetPack 7 | PUBLISHED-WEB (vendor blog + dev docs, fetched 2026-08-28) | developer.nvidia.com Jetson Thor posts |

# Search log — 2026-08-28 Deployment & Optimization pass

Method: WebSearch to locate; PyMuPDF extraction from banked PDFs for every
quoted paper number; WebFetch of the TensorRT GitHub issue for the Thor trap
(community primary, quoted with fetch date, cannot be banked as a PDF —
flagged as PUBLISHED-WEB and marked for in-house verification on Thor).

## Queries run
1. `PTQ4ViT RepQ-ViT post-training quantization vision transformer calibration images per-channel per-tensor INT8 accuracy drop`
   → 2111.12293, 2212.08254 (+ RepQuant 2402.05628, DopQ-ViT 2408.03291 noted,
   not banked — no numbers quoted from them).
2. `"FP8 versus INT8" Qualcomm arxiv 2303.17951 transformer accuracy hardware efficiency E4M3 outliers`
   → 2303.17951 confirmed.
3. `Jetson Thor JetPack 7 TensorRT compute capability sm_110 Blackwell FP8 NVFP4 quantization support`
   → CC 11.0/sm_110 confirmed (vendor + RidgeRun); TRT issue #4590 (silent FP32
   fallback); forum report FP4 fails on TRT 11.0.0.114; JetPack 7/7.1 posts.
4. `post-training quantization semantic segmentation dense prediction INT8 accuracy drop mIoU vision transformer SAM measured`
   → 2405.03144 (PTQ4SAM), PQ-SAM (ECCV24, not banked — no numbers quoted).

## WebFetch
- github.com/NVIDIA/TensorRT/issues/4590 (fetched 2026-08-28): TRT 10.13.3.9 on
  Thor sm_110 accepts FP8/FP4 builder flags, silently builds FP32 (plan ~FP32
  size, DataType.FLOAT outputs); opened 2025-10-04; no NVIDIA response
  documented in-thread at fetch time.

## Numbers extracted from banked PDFs (verbatim contexts in session log)
| key | extracted |
|---|---|
| 2111.12293 | 32 calib images (random, train); #ims=128 → time up, top-1 "varies slightly"; W8A8 < 0.5 % drop; base PTQ > 1 % drop; post-softmax/post-GELU cause |
| 2212.08254 | 32 samples cls / 1 sample COCO calib; channel-wise weights + layer-wise activations; W4A4 prior: FQ-ViT 0.1 % top-1; Cascade Mask R-CNN Swin-T W4A4 prior methods: box AP −23.2, mask AP −19.3 |
| 2303.17951 | FP8 dedicated compute ≥ 50 % less area/energy-efficient than INT8; INT8 covers ~90 % of FP8-E4 range exactly with free scaling; W4A8 ≥ FP8-E4 on MobileNetV2/DeepLabV3 at ~50 % less compute; ViT/BERT outlier layers favor E4 |
| 2405.03144 | bimodal post-Key-Linear activations; heterogeneous post-softmax (self vs two-way cross attention); SAM-L: FP 36.4 AP, W6A6 MinMax 32.9 / OMSE 33.1, W4A4 OMSE 5.4 (SAM-H 7.4); PTQ4SAM 6-bit lossless (~0.5 %), 3.9× theoretical accel |

## Not quoted (headers/column mapping not confirmed from the PDF)
- Qualcomm per-format ViT delta columns (−1.33/−0.45/−0.04/−0.19/−76.69): column
  headers not resolved in extraction → deliberately NOT quoted per the
  quote-only-what-you-verified rule.
- Thor TFLOPS marketing figures conflict between sources (2070 vs 1035 dense) →
  not quoted; not decision-relevant.

## Papers banked this pass (Deployment & Optimization)
2111.12293 (PTQ4ViT) · 2212.08254 (RepQ-ViT) · 2303.17951 (FP8 vs INT8) ·
2405.03144 (PTQ4SAM). Already banked and relevant: 2607.08029 (component-wise
Jetson quantization, LAB-RUN-001).

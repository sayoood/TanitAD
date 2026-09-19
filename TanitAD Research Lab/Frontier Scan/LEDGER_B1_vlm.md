<title>LEDGER B1 — VLM / multimodal / omni models</title>

# LEDGER B1 — VLM / multimodal / omni models

`APPEND-ONLY. Created 2026-09-13 (LAB-RUN-012) — the track had no ledger file. Transfer question: does it give us a semantic channel the trunk lacks?`

## 2026-09-13-01 — Fine-grained sign recognition: the best published number is ~0.48, and it is about TYPES, not values

`FULL TEXT` · arXiv **2508.02047v1** (4 Aug 2025) · Garg, Aich (NEC Labs America) · lib `2508.02047`.
MVV: Mapillary-Vistas-derived validation set with expert fine labels. Table 3 accuracy, task T3 (Mapillary **signs**): **DINOv2 0.484** (training-free exemplar feature matching) vs **InternVL-3-9B 0.467** (best VLM), no interval, no seeds; T1 vehicles DINOv2 0.935; T4 BDD vehicles DINOv2 0.95 vs Gemma-3-4B 0.937. VLMs needed **3–4 days vs hours** for 2,000 crops.
⚠️ **The concession is in the body:** the abstract's *"consistently outperforms all VLM baselines"* is a **1.7-point** margin on signs, and the fine classes stop at sign **type** ("Speed Sign") — **reading the value is not measured**.
**Our position:** the admissible in-corpus route to a posted limit is a vision sign reader; this is the ceiling evidence for that reader and it is weak. Also TS-1M `2603.23034` (>1 M images, 454 classes; abstract-only).
→ `Data Engineering/Research/2026-09-13-posted-speed-limit-supplier/RESULT.md` F8

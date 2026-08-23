# VLM model survey for the strategic-labeling pipeline (2026-08-11)

**Scope:** open-source VLMs vs the PH0 baseline arms {`Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-27B-FP8`}
for the VLM_STRATEGIC_LABELING task: two low-fps driving clips (past+future, ~16–32 frames at
448 px), strict-JSON out, sign-OCR incl. small city names, fine-grained scene understanding,
one A40 48 GB, permissive license for a public gated dataset. All claims **PUBLISHED** (URL
cited) unless marked otherwise. Nothing here is MEASURED by us — PH0 is where that happens.

## 1. VERDICT — the Qwen3.5 vision question (pipeline-critical)

**Qwen3.5 is NATIVELY multimodal; there is NO separate Qwen3.5-VL line.** The family
(released 2026-02-16) uses early-fusion training on multimodal tokens and explicitly
"outperforms Qwen3-VL models across … visual understanding" — the standalone VL line ended at
Qwen3-VL (Oct 2025). PUBLISHED: [Simon Willison](https://simonwillison.net/2026/Feb/17/qwen35/),
[HF Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B),
[NVIDIA NIM card](https://docs.api.nvidia.com/nim/reference/qwen-qwen3-5-122b-a10b).
- `Qwen3.5-9B`: 9 B dense, **image AND video input** ("vision encoder supporting image and
  video inputs"), 262 K native ctx, **Apache 2.0**, open sizes 0.8/2/4/9/27B + 35B-A3B/122B-A10B/
  397B-A17B MoE. PUBLISHED: [Together AI model page](https://www.together.ai/models/qwen3-5-9b).
- Video is exercised in practice: official **Video-MME 78.4** for 9B (a user reproduction
  discrepancy thread confirms the official number exists).
  PUBLISHED: [QwenLM issue #86](https://github.com/QwenLM/Qwen3.6/issues/86).
- Bench standing at size: 9B **MMMU-Pro 70.1** (vs GPT-OSS-120B 59.7); 27B **MMMU 82.3**,
  MathVision 86.0, MMLU-Pro 86.1. PUBLISHED:
  [techie007 guide](https://techie007.substack.com/p/qwen-35-the-complete-guide-benchmarks),
  [kie.ai / millstone refs](https://www.millstoneai.com/inference-benchmark/qwen3-5-27b-fp8).
- A40 fit: 27B ≈ 54 GB bf16 / **~27 GB FP8** — "FP8 on a 48 GB card is the likely sweet spot";
  9B bf16 ≈ 18 GB. Both leave KV headroom for few-thousand-token + 32-image prompts.
  PUBLISHED: [willitrunai](https://willitrunai.com/blog/qwen-3-5-27b-vram-requirements),
  [apxml](https://apxml.com/models/qwen35-27b).

⇒ The design doc's "Qwen3.5 ships natively multimodal — VERIFY" checkbox: **verified, PUBLISHED.**
Still confirm the exact `video` chat-template path at PH0 on pod4 (that part is MEASURED-only).

## 2. Field survey (last ~6 months prioritized)

| model (HF id) | params / A40 fit | vision I/O | OCR / bench standing | license | verdict for our task |
|---|---|---|---|---|---|
| **Qwen3.5-9B / 27B-FP8** (`Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-27B-FP8`) | 9B bf16 ~18 GB ✓; 27B FP8 ~27 GB ✓ | native image+video, 262K ctx | 9B MMMU-Pro 70.1, Video-MME 78.4; family = "generational leap in OCR accuracy over Qwen3-VL" ([roboflow compare](https://playground.roboflow.com/models/compare/gemma-4-31b-vs-qwen3-5-27b)) | Apache 2.0 | **baseline stands** |
| **Gemma 4** 12B-unified / 26B-MoE / 31B-dense (Google, 2026-04-02) | 12B bf16 ✓; 26B/31B need FP8 (~31 GB) ✓ | **all sizes accept image+video**, 256K ctx | 31B **MMMU-Pro 76.9**, "leads on MMMU-Pro and MATH-Vision" vs Qwen3.5-27B; doc-parsing/handwriting OCR claimed, **no OCRBench number found** ([Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/), [tech report](https://arxiv.org/html/2607.02770v1), [roboflow](https://playground.roboflow.com/models/compare/gemma-4-31b-vs-qwen3-5-27b)) | **Apache 2.0** (new for Gemma) | **only credible challenger** |
| Qwen3.6-27B(-FP8) (2026-04-22) | FP8 ✓ | native image+text | "match or beat Qwen3-VL" — reoriented to **agentic coding**, no published vision gain over 3.5 ([nerova](https://nerova.ai/benchmarks-performance/qwen-3-6-explained-benchmarks-context-and-what-builders-should-know-2026)) | Apache 2.0 | no vision case to switch |
| GLM-4.6V (`zai-org/GLM-4.6V`, 2025-12-08) | **106B-A12B — does NOT fit** A40 | image, 128K, native tool-calling | "Qwen3-VL-235B level" ([Z.ai docs](https://docs.z.ai/guides/vlm/glm-4.6v), [VentureBeat](https://venturebeat.com/ai/z-ai-debuts-open-source-glm-4-6v-a-native-tool-calling-vision-model-for)) | MIT | excluded by VRAM |
| GLM-4.6V-Flash (`zai-org/GLM-4.6V-Flash`, 9B) | bf16 18–20 GB ✓ | image; video support secondary | MMBench 86.9, MathVista 82.7, **OCRBench 84.7**; "outperforms Qwen3-VL-8B" — comparison **pre-dates Qwen3.5** ([binaryverseai](https://binaryverseai.com/glm-4-6v-review-benchmarks-pricing-local-install/)) | **MIT** | fallback 9B arm only |
| Molmo 2 8B (`allenai/…`, 2025-12-17) | bf16 ✓ | video + multi-image; **SOTA video pointing/tracking/grounding**, beats Gemini 3 Pro on tracking ([Ai2 blog](https://allenai.org/blog/molmo2)) | OCR not a headline strength | Apache 2.0 | niche: grounded evidence-frames, not OCR |
| MiniCPM-V 4.6 / -o 4.5 (OpenBMB, 2026-05) | 4.6 is **1.3B** (Qwen3.5-0.8B base) — "Qwen3.5-2B-level" ([MindStudio](https://www.mindstudio.ai/blog/what-is-minicpm-v-4-6-vision-model)) | strong video compression (96×) | good OCRBench *for its size* | Apache-2.0 + registration for commercial | too small for PH1 quality bar |
| InternVL3.5 8B/38B (OpenGVLab, 2025-08) | 8B ✓ / 38B needs AWQ | image+video | strong, but all comparisons pre-date Qwen3.5 ([HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B)); no InternVL4 exists as of today | mixed (Qwen-derived LLMs) | superseded by Qwen3.5 |
| Llama 4 Scout/Maverick (2025-04) | 109B/400B MoE — int4 marginal / no fit | image (multi) | no OCR leadership published | **Llama Community License** (restrictions) | excluded: license + VRAM |
| Mistral Small 4 (2026-03-16) | ~24B-class ✓ | unified vision+reasoning (absorbs Pixtral, which is stale since 2024-09) ([serenitiesai](https://serenitiesai.com/articles/mistral-ai-models-2026-complete-guide), [ucstrategies](https://ucstrategies.com/news/pixtral-12b-specs-benchmarks-how-to-deploy-mistrals-vision-model-2026/)) | no published OCR/vision wins vs Qwen3.5 | Apache 2.0 | no evidence to prefer |
| DeepSeek-VL3 | **does not exist** (2026-08-11); DeepSeek-V4 (2026-04-24) is 284B-A13B/1.6T MoE ([MindStudio](https://www.mindstudio.ai/blog/deepseek-v4-open-source-frontier-model-review)) | — | — | MIT | excluded by VRAM |
| NVIDIA Alpamayo 1 (ex Alpamayo-R1, `nvidia/Alpamayo-R1-10B`, 2025-12-03) | 10B (8.2B backbone + 2.3B action expert) ✓ | driving multi-cam; Cosmos-Reason base, Chain-of-Causation ([NVIDIA](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development)) | trajectory-reasoning VLA, **not** a sign-OCR/strict-JSON labeler | NVIDIA open license | wrong tool for THIS task (already on pod4 for its own tasks) |
| Driving-tuned labelers (Senna, OpenDriveVLA, SteerVLA) | 7B-class | driving VQA | built on **Vicuna-7B/LLaVA-era** backbones ([Senna](https://github.com/hustvl/Senna)) — generations behind on OCR/instruction-following | research | no |

## 3. Ranked shortlist (≤3) for OUR task

1. **`Qwen/Qwen3.5-9B` — keep.** Only family with ALL five requirements simultaneously
   PUBLISHED: native video, Apache 2.0, best-at-size general benches (MMMU-Pro 70.1 ≈ models
   13× larger), family-level OCR leap, comfortable A40 fit with KV room for 32 frames.
2. **`Qwen/Qwen3.5-27B-FP8` — keep** as the quality arm (~27 GB weights, MMMU 82.3); FP8
   card states metrics "nearly identical" to bf16 ([HF](https://huggingface.co/Qwen/Qwen3.5-27B-FP8)).
3. **Gemma 4 31B (FP8) or 26B-MoE — the one evidence-backed challenger.** The only
   post-Qwen3.5 open release with a PUBLISHED head-to-head vision-reasoning win at our size
   class (MMMU-Pro 76.9; leads MATH-Vision vs Qwen3.5-27B), Apache 2.0, image+video input.
   Gap in the case: **no OCRBench/DocVQA-class number surfaced** in this survey, and sign-OCR
   is our gate metric — so it earns a pilot arm, not a swap.

## 4. Recommendation to the PI (decision, not a silent change)

**Do NOT replace the PH0 arms.** No surveyed model has PUBLISHED evidence of beating Qwen3.5
at A40-fittable size on the profile that gates PH0 (sign-OCR ≥0.9 + strict JSON + video-in).

**Option for the PI: ADD one challenger arm, Gemma 4 (26B-MoE or 31B-FP8), to the same 50
PH0 clips.**
- *For:* guards against Qwen-family monoculture; the MMMU-Pro delta (76.9 vs 27B-class Qwen)
  is real PUBLISHED evidence; same license class; PH0's 100-field human check scores it for free.
- *Against:* ~+2 h pod4 wall (+50 %), prompt/template porting, and its OCR standing is
  unproven — if the PI wants exactly one gate variable per phase, skip it and keep two arms.
- Non-option noted: `Qwen3.6-27B-FP8` is a drop-in same-family upgrade but its release is
  agentic-coding-oriented with no published vision gain over 3.5 — not worth the parity break.

*Survey method note: huggingface.co direct fetch is egress-blocked from this box; card facts
above come from search-indexed card text + secondary sources, so PH0 must re-verify the two
chosen cards' video-input template at run time (cheap, already planned in the design doc §3).*

# VLM Data-Augmentation Survey — Model Selection for TanitAD's Auto-Labeling Pipeline

**Author:** Data Engineering (research/planning session). **Date:** 2026-07-19. **Status:** research + design
only; no code, no pod, nothing committed. **Scope:** select the best recent (2025–2026) vision-language models to
**leverage + augment** our owned driving corpus — captioning, scene-tagging, Chain-of-Causation (CoC) reasoning-trace
generation, driving VQA, risk / critical-agent identification, corner-case mining, and label verification — for the
VLM auto-label → human-verify pipeline in `TANITDATASET_V1_STRATEGY.md` §3/§5. Deployment target: **RunPod A40 (48 GB)**.

> ⚠️ **Two flags up front, because they decide the whole recommendation.**
> 1. **OUTPUT licensing is the gate, not model quality.** We build two datasets (`TANITDATASET_V1_STRATEGY.md` §1):
>    a 🔴 **research set** (internal, NC-tainted, never shipped) and a 🟢 **commercial set** (shippable, permissive).
>    A label is only shippable if the VLM's **outputs may be used commercially AND to train a model that is then
>    distributed/sold**. That is a *stricter* test than "weights are open" and stricter than "commercial API use is
>    allowed." It cleanly splits the model roster: **Apache-2.0 / NVIDIA-Open-Model-License open-weight models →
>    commercial set; frontier APIs (GPT/Claude/Gemini) + non-commercial open weights → research set only.**
> 2. **Never run a frontier API on the commercial set.** GPT, Claude and Gemini all forbid using their outputs to
>    build *competing* models, and OpenAI additionally requires that any classifier/organizer model trained on
>    outputs **not be distributed**. A camera-first driving world model is almost certainly *non-competing*, but the
>    question is untested and the downside is the whole commercial dataset. The clean, defensible rule is
>    **frontier = research/internal teachers & verifiers only** (that set is already NC and never ships), which
>    sidesteps the ambiguity entirely.

---

## 0. TL;DR — the four picks

| Role | Pick | Why |
|---|---|---|
| **(a) Commercial-set workhorse** (at-scale auto-label) | **Qwen3-VL** — 8B native, 32B quantized on A40 | Apache-2.0 (cleanest possible outputs), SOTA open 2D/**3D** grounding + embodied-spatial + video-timestamp grounding, full vLLM/SGLang support |
| **(b) Physical-AI / driving specialist** | **NVIDIA Cosmos-Reason2-8B** (2B/8B/32B) | Post-trained on Qwen3-VL-8B for embodied/AV reasoning; NVIDIA Open Model License = commercial + outputs disclaimed; **Uber uses it for exactly our task** (AV training-data video captions); same ecosystem as our PhysicalAI/Alpamayo data |
| **(c) Research-set teacher / verifier** | **Gemini 2.5/3 Pro** (video), **GPT-5 / o-series** and **Claude Opus** (reasoning) | Best-quality gold traces & QA-of-labels — but outputs kept firewalled to the research set / eval only |
| **(d) Output-licensing landmines** | **Llama-3.2-Vision** (EU-excluded + "Llama" naming tax), **Qwen-Research-License small variants** (NC), **NVILA / Pixtral-Large / most driving-specialist weights** (NC) | See §8 |

**The elegant result:** the workhorse (Qwen3-VL-8B) and the specialist (Cosmos-Reason2-8B, *which is Qwen3-VL-8B
post-trained*) are the **same architecture family** — identical tokenizer/serving path, one vLLM deployment, one
prompt format. Run Cosmos-Reason2 for physics/causation/risk traces and Qwen3-VL for high-throughput captions &
tags on the *same* A40 image, hot-swapping weights.

---

## 1. Comparison table — OPEN-WEIGHT (commercial-set candidates)

A40 fit key: ✅ native bf16 · ⚙️ needs 4-bit/8-bit/FP8 quant to fit 48 GB · ❌ multi-GPU. (bf16 ≈ 2 GB/1 B params;
long-video KV-cache eats extra headroom, so treat "⚙️" sizes as frame-count-capped on a single card.)

| Model | Variants | A40 48 GB fit | AV-labeling strengths | Video / temporal | Weight license | **OUTPUT license (commercial + train-and-ship?)** | Recency |
|---|---|---|---|---|---|---|---|
| **Qwen3-VL** ⭐ | 2B/4B/8B/32B dense; 30B-A3B & 235B-A22B MoE; Instruct + Thinking | 2B–8B ✅ · 32B/30B-A3B ⚙️(AWQ~18–20 GB) · 235B ❌ | SOTA open **2D+3D grounding** (RefCOCO, RefSpatial, RoboSpatialHome, ERQA, VSIBench), OCR 32 langs, fine-grained perception | Native 256K ctx; interleaved-MRoPE + **text-timestamp alignment**; video ≈ Gemini 2.5 Pro / GPT-5 / Claude Opus 4.1 | **Apache-2.0** (whole family) | **✅ YES — unrestricted.** No field-of-use, no naming, train & sell freely | Sept–Oct 2025 |
| **Cosmos-Reason2** ⭐ | 2B / 8B / 32B | 2B/8B ✅ · 32B ⚙️ | Embodied/physical **reasoning**, risk & next-action, 2D/3D point+bbox, trajectory, OCR; #1 open on Physical-AI-Bench | Video-native, **256K** ctx (16K→256K vs Reason1); long chain-of-thought over clips | **NVIDIA Open Model License** | **✅ YES.** "Commercially usable," NVIDIA **disclaims output ownership**, derivative models OK. *Caveat:* keep a safety guardrail (auto-terminate clause) | ~2026 (successor to Cosmos-Reason1-7B, GTC-2025) |
| **Cosmos-Reason1-7B** | 7B (on Qwen2.5-VL-7B) | ✅ | Same family, AV-validated; superseded by Reason2 | Video, 16K ctx | NVIDIA Open Model License | ✅ YES (as Reason2) | 2025; **NVIDIA says migrate to Reason2** |
| **InternVL3 / InternVL3.5** | 1B/2B/8B/9B/14B/38B/78B (+3.5) | ≤14B ✅ · 38B ⚙️ · 78B ⚙️(4-bit, tight)/❌ | Strong reasoning (78B ≈ 72 MMMU), grounding, multi-image, video | Good multi-frame video | **MIT project + *upstream backbone*** | ⚠️ **Per-size.** 14B/38B (Qwen2.5-14B/32B = Apache) ✅ · 1B/2B (Qwen2.5-0.5B/1.5B = **Qwen Research = NC**) ❌ · 78B (Qwen2.5-72B = Tongyi, <100 M MAU) ✅* | 3.0 Apr 2025 · 3.5 Aug 2025 |
| **Ovis2.5** | 2B / 9B (Ovis2.6-30B-A3B newer) | 2B/9B ✅ · 30B ⚙️ | SOTA open <40B on OpenCompass (78.3), charts/STEM/OCR, native-res ViT, reflective CoT | Video understanding added in 2.5 | **Apache-2.0** (AIDC-AI) | ✅ YES* (verify small-size Qwen backbone) | Aug 2025 |
| **Molmo** | 7B-O, 7B-D, 72B | 7B ✅ · 72B ❌ | **2D pointing** (great for critical-agent localization), captions | Molmo-1 image-only; **Molmo 2** adds video/pointing/tracking | **Apache-2.0** weights | 7B-**O** (OLMo backbone) ✅ cleanest · 7B-D/72B on Qwen2 backbone ⚠️* · "research/edu intent" note | 0924; Molmo 2 2025–26 |
| **Pixtral-12B** | 12B (400M ViT) | ✅ | Native-res RoPE-2D, strong **OCR/ChartQA** (83.7), sign/text reading | Image-first; **weak multi-frame** | **Apache-2.0** | ✅ YES — unrestricted | Sept 2024 (still current) |
| **Pixtral-Large** | 124B | ❌ | Higher quality | image-first | **Mistral Research License = NC** | ❌ research only | 2024 |
| **NVILA** | 8B (incl. HD-Video, 4K/1K-frame) | ✅ | Efficient; **long-video** (up to 1K frames) | **Excellent** long-video | **CC-BY-NC-SA-4.0 = NC** | ❌ **research set only** | 2024–25 |
| **Llama-3.2-Vision** | 11B / 90B | 11B ✅ · 90B ❌ | Decent captions/VQA | image-first, weak video | **Llama 3.2 Community License** | ⚠️ outputs OK for synth-data/distill **BUT** trained model must be **named "Llama-…"**, 700 M-MAU clause, **multimodal rights NOT granted to EU-domiciled entities** → **effectively out for TanitAD** | 2024 |

\* "commercial OK" via Tongyi Qianwen License holds **below 100 M monthly active users** (a data-labeling backend
never approaches this); still verify per-checkpoint that the *specific* backbone isn't a Qwen **Research** (NC) variant.

---

## 2. Per-model notes — open weight

### Qwen3-VL — the workhorse (Apache-2.0)
Released Sept–Oct 2025 (dense 2B/4B/8B/32B; MoE 30B-A3B & 235B-A22B; Instruct + Thinking; tech report arXiv:2511.21631).
The **entire family is Apache-2.0** — the cleanest output license available: labels can be sold, redistributed, and used
to train the shippable world model with **zero** field-of-use, attribution, or naming constraints. Capability profile is
a near-perfect match for AV labeling: **SOTA-open 2D *and 3D* grounding** (RefCOCO, CountBench, ODinW-13, RefSpatial,
RoboSpatialHome, ERQA, VSIBench), fine-grained perception (V*, HRBench), 32-language OCR robust to low-light/blur/tilt
(sign & HMI text), and **native video with explicit text-timestamp alignment** (interleaved-MRoPE) — the multi-frame
temporal grounding that AV labels demand. Qwen's own report puts its video understanding **competitive with Gemini 2.5
Pro / GPT-5 / Claude Opus 4.1**. First-class **vLLM (≥0.11.0) and SGLang** support.
**A40:** 4B/8B run native bf16 with room for long context; **32B** fits at 4-bit AWQ (~18–20 GB weights) or 8-bit
(~34 GB) — best quality/throughput on a single card; 30B-A3B MoE quantized is fast (3B active); 235B needs 2–4× A40.

### Cosmos-Reason2 — the physical-AI / driving specialist (NVIDIA Open Model License)
The current NVIDIA embodied-reasoning VLM (2B/8B/32B), **post-trained on Qwen3-VL-8B-Instruct** — so it inherits the
workhorse's architecture and *adds* physical-common-sense + embodied-decision reasoning via SFT+RL. Ranked **#1 open
model** on Physical-AI-Bench and Physical-Reasoning leaderboards; outputs 2D/3D point localization, bounding boxes,
**trajectory data**, and OCR; 256K context (up from Reason1's 16K) → real multi-frame clip reasoning. **The license is
ideal for our gate:** NVIDIA Open Model License states models are *commercially usable*, you may *create and distribute
derivative models*, and **NVIDIA does not claim ownership of any outputs** — i.e. generated CoC traces/risk labels are
ours to ship and to train on. The one caveat is a guardrail-circumvention auto-termination clause (irrelevant to
labeling; don't strip safety filters). **Validation that this is the right tool:** NVIDIA's own launch cites **Uber
using Cosmos-Reason2 to produce searchable video captions for AV training data** — our exact use case. Deploy via NVIDIA
NIM (build.nvidia.com) or vLLM; community GGUF/quant builds exist. Use it for the **Chain-of-Causation / risk /
critical-agent / physics-plausibility** layer; use Qwen3-VL for high-volume captions & scene-tags.
*(Cosmos-Reason1-7B is the well-documented, AV-dataset-validated predecessor — NVIDIA explicitly recommends migrating
to Reason2. Cosmos 3, announced CES 2026, unifies reasoning + world-prediction/sim/action but is a heavier platform,
not a drop-in labeler — watch-item only.)*

### InternVL3 / InternVL3.5 (MIT project + upstream-backbone license)
Top-tier open reasoning (InternVL3-78B ≈ 72.2 MMMU, rivaling GPT-4o / Claude-3.5-class); ViT-MLP-LLM with InternViT +
Qwen2.5 or InternLM3 backbones; strong grounding, multi-image and video. **Output license is per-size**, because the
MIT wrapper inherits the LLM backbone: **14B (Qwen2.5-14B) and 38B (Qwen2.5-32B) sit on Apache backbones → clean**;
**1B/2B (Qwen2.5-0.5B/1.5B) inherit the Qwen *Research* (non-commercial) license → firewall**; 78B (Qwen2.5-72B) is
Tongyi (<100 M MAU OK). The InternLM3-backed 9B avoids the Qwen question. Solid #2 open option, but Qwen3-VL is a
cleaner blanket-Apache story. Verify the exact backbone per checkpoint before shipping any InternVL-derived label.

### Ovis2.5 (Apache-2.0)
Aug 2025, 2B/9B (+ newer Ovis2.6-30B-A3B). **Ovis2.5-9B = 78.3 OpenCompass, SOTA open <40B**; native-resolution ViT
(good for small distant signs/agents), reflective chain-of-thought with an optional "thinking" mode, charts/STEM/OCR
strength, and added video understanding. AIDC-AI ships it Apache-2.0. A strong lightweight alternative/ensemble partner
to Qwen3-VL-8B; verify the small-size backbone isn't a Qwen-Research variant.

### Molmo / Molmo 2 (Apache-2.0 weights)
Distinctive **2D pointing** capability (PixMo pointing data) — directly useful for *critical-agent identification* and
grounding "which object caused the decision." **Molmo-7B-O** (OLMo backbone) is the cleanest fully-Apache path;
**Molmo-7B-D / 72B** ride Qwen2 backbones (72B → Tongyi upstream). Molmo-1 is image-only; **Molmo 2** adds video
understanding, pointing and tracking. Note AllenAI's "research/educational" framing alongside the Apache grant —
weights are Apache but do your own diligence for a commercial pipeline. Good pointing-specialist ensemble member, not a
primary video labeler.

### Pixtral-12B (Apache-2.0)
Clean Apache-2.0, 128K context, native-resolution RoPE-2D vision, **strong OCR/ChartQA (83.7)** — excellent for
sign/plate/HMI text extraction on single frames. **Weak on multi-frame/temporal**, so it's a per-frame OCR/attribute
helper, not a clip-reasoner. (Pixtral-Large-124B is **Mistral Research License = NC** → research set only.)

### NVILA (CC-BY-NC-SA-4.0 = non-commercial)
Technically excellent **long-video** VLM (NVILA-8B-HD-Video: up to 4K res, ~1K frames) and A40-friendly, **but the
weights are CC-BY-NC-SA** → **research set only**, never a commercial-set labeler. Use as an internal long-video teacher
if useful; do not ship its outputs.

### Llama-3.2-Vision (Llama 3.2 Community License) — avoid for TanitAD
The community license *does* permit using outputs for synthetic-data/distillation, but three problems bite us:
(1) any model trained on Llama outputs **must carry "Llama" at the start of its name** (a permanent branding tax on our
world model); (2) the 700 M-MAU clause; and most decisively (3) **the multimodal-model rights are *not granted* to
individuals domiciled in, or companies with a principal place of business in, the EU.** TanitAD is EU-centric (GDPR
anonymization gate, EU-urban focus). **Treat Llama-3.2-Vision as out** for the commercial pipeline; if ever used, it's
research-set only and still carries the naming caveat.

---

## 3. Driving-specialized VLMs — reference architectures, **not** the labeler

DriveVLM (CoRL 2024), Senna / Senna-2, OmniDrive (CVPR 2025, counterfactual reasoning), DriveLM, OpenDriveVLA, and
**EMMA (Waymo)** are the state of the art in *driving* VLM design — but **none is a shippable auto-labeler**:
- **EMMA** fine-tunes **Gemini** and has **no open weights** (proprietary Waymo).
- **Senna / DriveVLM / OmniDrive / DriveLM** are academic and trained on **nuScenes / Waymo (non-commercial)** bases, so
  both the checkpoints and their outputs inherit NC/research terms → firewalled.

Their value to us is **schema and method**, exactly as `TANITDATASET_V1_STRATEGY.md` §3 already plans: copy the
*format* (Alpamayo Chain-of-Causation: `observation → critical agents → justification → decision → action`; DriveVLM's
"only label decision-relevant objects" efficiency trick; OmniDrive's counterfactual prompts) and generate the labels
ourselves with Cosmos-Reason2 + Qwen3-VL on owned pixels. **Do not ingest their weights or outputs into the commercial
set.**

---

## 4. Frontier / API — research-set teachers & verifiers only

Best raw quality for **gold CoC traces, hard-case adjudication, and label QA** — but their outputs are **firewalled to
the research set** per the §0 rule.

| Model | AV-relevant strength | Output-usage posture (for training data) |
|---|---|---|
| **Gemini 2.5 / 3 Pro** | Best **native video** + long-context (hours of footage, audio, PDFs), fine-grained localization | Paid/Vertex: Google **doesn't train on your data**; ToS: **no models that *compete* with the Services**; EEA/UK/CH get paid-terms on all tiers. → research/eval only |
| **GPT-5 / o-series** | Strong unified image+video reasoning, spatial/temporal grounding | ToS: **no outputs to build competing models**; classifier/organizer exception requires the trained model **not be distributed** → **do not use for shippable labels**; research/eval only |
| **Claude Opus (4.x)** | Strong reasoning/justification quality; customer **owns outputs** (Commercial Terms; API not trained on) | May train **non-competing** models on outputs; a driving world model is plausibly non-competing, but keep to research set to avoid the untested question |

**Usage pattern:** frontier models (1) mint a small **human-verified gold slice** of CoC traces to fine-tune/validate
the open-weight labelers, and (2) act as an **LLM-judge verifier** over open-weight labels for the internal eval set —
never as the bulk labeler for shippable data.

---

## 5. A40 (48 GB) deployment + throughput guidance

- **Serving:** vLLM (Qwen3-VL ≥0.11.0; Cosmos-Reason2 via NIM or vLLM) or SGLang. Both do continuous batching + paged
  KV — essential for millions of clips. FP8 (Hopper-style) isn't native on A40 (Ampere); use **AWQ/GPTQ 4-bit** or
  **bf16** and INT8 KV where supported.
- **Native on one A40:** 2B/4B/7–9B/14B run bf16 comfortably. **Best throughput/$ = 8B-class bf16** (Qwen3-VL-8B or
  Cosmos-Reason2-8B) — the recommended default for bulk labeling.
- **32B on one A40:** 4-bit AWQ (~18–20 GB weights) leaves ~25 GB for KV/vision tokens — viable, but **cap frames &
  resolution** for long clips. Reserve 32B for the *quality* tier (hard corner-cases, gold-slice pre-labeling).
- **Vision-token control is the throughput lever:** Qwen-family uses **dynamic resolution** → tokens scale with pixels.
  For labeling, downscale to ~0.5–1.0 MP and **subsample frames** (e.g. 8–16 keyframes/clip) — this dominates cost far
  more than model size. Keep output terse (structured JSON tags) to cut decode time.
- **Fleet pattern:** shard clips across N A40 pods, one 8B model per card (data-parallel); escalate only the mined
  hard-cases to a 32B card. Because Cosmos-Reason2 = post-trained Qwen3-VL, **one container image + one prompt schema**
  serves both by swapping weights.

---

## 6. OUTPUT-licensing landmines (the commercial-set gate — verify, don't assume)

1. **Frontier APIs → never on the commercial set.** OpenAI: no outputs to train competing models; classifier exception
   requires the model *not be distributed* — fatal for a shipped product. Google: no models that *compete with the
   Services*. Anthropic: no *competing* models (driving world model likely fine, but untested). **Mitigation:** frontier
   = research/internal only. **verify** with counsel if you ever want a frontier-labeled slice to ship.
2. **Llama-3.2-Vision: EU rights exclusion + "Llama" naming tax.** Multimodal rights not granted to EU entities; trained
   models must be named "Llama-…". **Avoid.**
3. **Qwen *Research* License hides inside small InternVL/Ovis/Molmo variants** (Qwen2.5-0.5B/1.5B/3B backbones = NC).
   The wrapper being MIT/Apache does **not** cure an NC backbone. **verify the exact backbone per checkpoint.**
4. **NVILA (CC-BY-NC-SA), Pixtral-Large (Mistral Research), all nuScenes/Waymo-based driving VLMs → non-commercial.**
   Research set only.
5. **Tongyi Qianwen License (Qwen 72B-class, Molmo-72B, InternVL-78B):** commercial OK **below 100 M MAU** (never a
   concern for a labeling backend) but you **can't relicense the weights** and there are redistribution conditions —
   fine for internal use, note it if weights are ever redistributed.
6. **NVIDIA Open Model License guardrail clause:** auto-terminates if you circumvent safety guardrails without a
   substitute — irrelevant to labeling, but don't strip filters.
7. **Human-in-the-loop is still the ground-truth guarantee.** Per the semantic survey, model-generated reasoning can
   launder hallucinations into "labels"; keep the auto-label → **human-verify a gold slice** discipline regardless of
   which VLM produces the bulk labels.

---

## 7. Open items to verify before production

- **Cosmos-Reason2 exact context/frame limits & A40 throughput** at our clip length (confirm on the HF model card /
  NIM docs; 256K ctx advertised).
- **Per-checkpoint backbone license** for any InternVL3/3.5, Ovis2.5, Molmo variant we actually deploy (Apache vs
  Qwen-Research vs Tongyi).
- **Ovis2.5-9B and Molmo-7B-O** as lightweight ensemble members vs Qwen3-VL-8B (pointing / native-res small-object
  recall) — quick A/B on a driving eval slice.
- **Frontier-output-to-ship question** — only if we ever want to commercialize a frontier-labeled slice; default is
  don't, so this stays theoretical.

---

## Sources

- Cosmos-Reason1-7B model card & license — https://huggingface.co/nvidia/Cosmos-Reason1-7B ; repo https://github.com/nvidia-cosmos/cosmos-reason1 ; research https://research.nvidia.com/labs/cosmos-lab/cosmos-reason1/
- Cosmos-Reason2 (2B/8B/32B) — https://huggingface.co/nvidia/Cosmos-Reason2-8B ; blog https://huggingface.co/blog/nvidia/nvidia-cosmos-reason-2-brings-advanced-reasoning ; VentureBeat https://venturebeat.com/orchestration/nvidias-cosmos-reason-2-aims-to-bring-reasoning-vlms-into-the-physical-world ; NIM docs https://docs.nvidia.com/nim/vision-language-models/1.6.0/examples/cosmos-reason2/api.html
- Qwen3-VL — GitHub https://github.com/QwenLM/Qwen3-VL ; HF https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct ; tech report https://arxiv.org/abs/2511.21631
- Qwen2.5-VL — HF https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct , https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct ; report https://arxiv.org/pdf/2502.13923
- Qwen license tiers (Apache / Tongyi Qianwen <100M MAU / Research=NC) — https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT ; https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20RESEARCH%20LICENSE%20AGREEMENT ; https://en.wikipedia.org/wiki/Qwen
- InternVL3 — paper https://arxiv.org/abs/2504.10479 ; blog https://internvl.github.io/blog/2025-04-11-InternVL-3.0/ ; InternVL3.5 https://arxiv.org/abs/2508.18265 ; backbone-license issue https://github.com/opengvlab/internvl/issues/900
- Ovis2.5 — https://huggingface.co/AIDC-AI/Ovis2.5-9B ; https://github.com/AIDC-AI/Ovis
- Molmo / Molmo 2 — https://huggingface.co/allenai/Molmo-72B-0924 , https://huggingface.co/allenai/Molmo-7B-D-0924 ; Molmo 2 https://allenai.org/blog/molmo2
- Pixtral-12B — https://mistral.ai/news/pixtral-12b/ ; paper https://arxiv.org/html/2410.07073v2
- NVILA — https://huggingface.co/Efficient-Large-Model/NVILA-8B ; https://huggingface.co/nvidia/NVILA-8B-HD-Video
- Llama-3.2-Vision — model https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct ; license https://www.llama.com/llama3_2/license/
- Driving VLMs — Senna https://arxiv.org/html/2410.22313v1 ; EMMA (Waymo) ; OmniDrive (CVPR 2025) ; OpenDriveVLA https://arxiv.org/html/2503.23463v1
- OpenAI output/competing-model terms — https://openai.com/policies/service-terms/ ; https://openai.com/policies/services-agreement/
- Anthropic outputs & training — https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model ; commercial terms
- Google Gemini API terms (compete / paid-vs-unpaid / EEA-UK-CH) — https://ai.google.dev/gemini-api/terms
- vLLM multimodal / Qwen serving — https://docs.vllm.ai/en/latest/features/multimodal_inputs/

# HiCAP stream R-E — NVIDIA/Cosmos backbones under a LATENCY criterion (2026-09-29)

**Status:** COMPLETE — staged in the index, **not committed**. Scope: update and re-orient `RA_vlm_backbones.md` after the three PI rulings of 2026-09-29
(a) a VLM is the preferred backbone, (b) the model source is `https://github.com/NVIDIA/Cosmos`, (c) the "sub-300 M" cap is removed — the constraint is now *system latency*.
Frozen VLM, prefill-only, hidden states at 3 depths pooled to ≤ 128 tokens, refreshed asynchronously at 2 Hz and cached; ~27 M trainable adapter/fusion/heads at 10 Hz; target Jetson Thor T5000.
Write-only-my-file stream (sibling streams RA/RB/RC/RD/RT and the latency-budget script untouched).

**Evidence classes** (CLAUDE.md rule 1): `P` = PUBLISHED-PRIMARY — I read the repo file / table / config myself this session (commit named in §0.2) ·
`M(ours)` = MEASURED-by-us — arithmetic or a command run here, script in Appendix A · `INH` = INHERITED (another doc/agent, not re-verified) ·
`R` = RECALLED from pre-2026 memory, not re-verified · `E` = ESTIMATED · `A` = ASSUMPTION · `H` = HYPOTHESIS · `U` = UNVERIFIED.
**Egress this session:** `huggingface.co`, `arxiv.org`, `research.nvidia.com`, `docs.nvidia.com` returned **403 CONNECT** (curl, this session); only GitHub (`git clone`/`ls-remote`, API search, `raw.githubusercontent.com`) was reachable.
⇒ **no Hugging Face model card or `config.json` was read.** Every "HF card says…" fact below is either read from a *GitHub README that quotes the card*, or is `U`.
**Nothing here is a TanitAD measurement of latency.** Every latency number is `E`, anchored on NVIDIA-published Thor rows; the on-Thor gaps are §6.

---

## Headline findings

1. ⭐ **`github.com/NVIDIA/Cosmos` resolves (GitHub is case-insensitive) to the repo `NVIDIA/cosmos`** — same repo id 910176789, same `HEAD` `3e3c6d6` for both spellings (`git ls-remote`, §0.1). It is the **Cosmos 3** home (three base models: **Edge 4 B / Nano 16 B / Super 64 B**). The earlier Cosmos-Reason1/-Reason2 repos live in the *other* org `nvidia-cosmos/*` and their READMEs now say *"no longer under active development … Visit the new Cosmos home"*. **So, read strictly, PI ruling (b) selects the Cosmos 3 reasoners — not Cosmos-Reason2.** Alpamayo is in a third org (`NVlabs/*`).
2. ⭐ **Cosmos 3 reasoners are two different architectures:** **Edge = Nemotron-dense 2 B LLM + SigLIP2-SO400M + 2×2 merger (2.43 B, `M(ours)` arithmetic on `P` config)**; **Nano = Qwen3-VL-8B architecture (8.76 B) and Super = Qwen3-VL-32B architecture (33.35 B)** — `P`: `cosmos-framework/.../nano_model_config.py` sets `model_name="Qwen/Qwen3-VL-8B-Instruct"`, `super_model_config.py` `Qwen/Qwen3-VL-32B-Instruct`. Cosmos-Reason2 (2B/8B/32B) is *also* Qwen3-VL-architecture (`P`: README "based on the Qwen3-VL architecture") and so is the Alpamayo-1.5 VLM (`P`: README "Cosmos-Reason2 backbone"; class `Qwen3VLForConditionalGeneration`, default `Qwen/Qwen3-VL-8B-Instruct`). Checkpoint `config.json` files are unread (egress) ⇒ the identity is `INH` until G5. ⇒ **the Qwen3-VL-2B/4B/8B Thor rows NVIDIA publishes are architecture-identical measured anchors for Cosmos-Reason2-2B/8B, the Cosmos 3 Nano reasoner and the Alpamayo VLM.**
3. ⭐ **Two RA "UNVERIFIED" items are now `P`:** Cosmos3-Edge ViT patch size = **16** (`modeling_cosmos3_reasoner_visual.py`: "Linear patch embed 768→1152", 768 = 3×16×16; 27 layers; merger 2×2 → 4608→11520→2048) ⇒ **160 tokens for a 256×640 frame** (32 px/token, same as Qwen3-VL); and the Alpamayo-1 (R1-10B) weights licence is stated as **OpenMDW-1.1** in the `NVlabs/alpamayo` README (HF card itself unread).
4. ⭐ **New since RA: TensorRT-Edge-LLM has a documented Cosmos3-Edge *reasoner* path** (export `--task reasoning` → `llm_build` + `visual_build` → `llm_inference`; native video in 0.11.0). **But NVIDIA's CI only tests it at FP16** (`Cosmos3-Edge-reasoning-fp16`), and **no TRT latency row for Edge exists** — the only NVIDIA Thor numbers for any Cosmos 3 model are **eager HF-Transformers BF16** (`inference_benchmarks.md`: 911-token image prompt **0.19 s**, text 1,705 tok **0.20 s = 8,717 tok/s**, decode 37–43 tok/s on T5000).
5. ⭐ **Latency verdict (all `E`, p50-style, ViT + prefill only; §2.3/§5):** at the corpus reality of **1 camera × 1 frame** the strict ≤ 50 ms reading is met by the ≤ 2.4 B tier in NVFP4 (19–22 ms, passes even with a 1.3× p95 guard + 15 ms overhead) and, on the raw p50 estimate only, by the ≤ 2.4 B tier in FP16 (39–41 ms) and the 8 B tier in NVFP4 (39 ms); 8 B at FP8/FP16 and eager BF16 fail it. At **3 cameras × 3 frames** the **async 2 Hz** reading (≤ 250 ms) is met with the guard by the ≤ 2.4 B tier (Cosmos-Reason2-2B ≡ Qwen3-VL-2B: **85 ms**; Cosmos3-Edge NVFP4 **109 ms** / FP16 **180 ms**) and by the 8 B tier at NVFP4 (**167 ms**); **nothing passes the ≤ 50 ms reading at 3 cameras** (ViT alone 21–29 ms); 7 cameras × 3 frames passes only the ≤ 2.4 B tier at NVFP4 and only on the raw estimate (185–239 ms, duty 0.40–0.51, no margin). 32 B-class (Cosmos-Reason2-32B, Cosmos 3 Super, Alpamayo-2-Super) is **teacher/auto-label only** (≥ 142 ms for a single camera even at NVFP4).
6. **Recommendation:** **primary = Cosmos3-Edge reasoner** (only model NVIDIA built and benchmarked for Thor; OpenMDW-1.1; ≥ 3 cameras × 3 frames within the async budget even at the CI-tested FP16), **with Cosmos-Reason2-2B / Qwen3-VL-2B as its measured-latency twin** (the only 2 B-class with real Thor TRT rows, by identity). **Paired driving-aware arm = Alpamayo-1.5's VLM vs its general parent Cosmos-Reason2-8B** (same Qwen3-VL-8B architecture, differ only by driving RL/SFT) — **passes the async reading but not the strict one**; no ≤ 4 B driving-tuned NVIDIA VLM exists. The RA R1/R2 (Qwen3.5-4B, Qwen-Drive) are **outside the PI's source ruling** and demoted to external references.
7. ⛔ **Biggest UNVERIFIED:** **no optimised (TRT/NVFP4/FP8) latency row exists for the recommended primary (Cosmos3-Edge) on Thor**, and no one has measured *any* of these models in the HiCAP mode — **prefill + 3-depth tap + concurrent 10 Hz fast path on the same 20-SM GPU** (refresh duty 0.1–0.5 of the GPU). All sub-50 ms Edge figures are extrapolated from Qwen3-VL-2B (identical body FLOPs: 50.3 M params/layer × 28) and Qwen3-VL-8B's ViT.
8. **Paper impact (not edited by me — this stream owns one file):** the sub-300 M framing, "HiCAP-S mandatory" and "Cosmos3-Edge has only eager-BF16 rows / patch size unverified" sentences in `Paper/HiCAP/sec_eff.tex` are stale (§7.4).

---

## 0. Provenance

### 0.1 What `https://github.com/NVIDIA/Cosmos` is — `M(ours)`, commands run 2026-09-29

```
$ git ls-remote https://github.com/NVIDIA/Cosmos HEAD   -> 3e3c6d61dc15d6517c4793beab5ec3894ffa07f1  HEAD
$ git ls-remote https://github.com/NVIDIA/cosmos HEAD   -> 3e3c6d61dc15d6517c4793beab5ec3894ffa07f1  HEAD        (identical)
GitHub search "repo:NVIDIA/Cosmos" -> 1 item: full_name "NVIDIA/cosmos", id 910176789, created 2024-12-30, pushed 2026-09-29T16:56:51Z,
                                       11,943 stars, license.spdx_id "NOASSERTION" (GitHub could not classify; the LICENSE file text is OpenMDW-1.1, below)
GitHub search "org:nvidia-cosmos"   -> 12 repos: cosmos-predict1/2/2.5, cosmos-transfer1/2.5, cosmos-reason1, cosmos-reason2, cosmos-rl, cosmos-cookbook,
                                       cosmos-xenna, cosmos-dependencies, .github      (no Cosmos-Reason3, no driving reasoner)
GitHub search "org:NVIDIA cosmos in:name" -> cosmos, cosmos-framework, cosmos-curator, cosmos-evaluator, Cosmos-Tokenizer (archived)
```
`P` (`NVIDIA/cosmos` `RELEASE.md`): the repo was the original Cosmos-1.0 home (release table still lists v1.0 2025-01-06, tokenizer v0.1 2024-11-06); `README.md` "What's new": **Jul 2026 Cosmos3-Edge released; May 2026 Cosmos 3 released**. `cosmos-reason2` README news: *"[June 1, 2026] … the Cosmos GitHub repositories have moved to the main NVIDIA GitHub organization"* and banner *"This repository is no longer under active development and will receive only limited maintenance updates."*
**Licence text (P):** `NVIDIA/cosmos/LICENSE` line 1 = **"OpenMDW License Agreement, version 1.1 (OpenMDW-1.1)"**; README: *"Source code and models are released under OpenMDW-1.1"*; every framework source file carries `SPDX-License-Identifier: OpenMDW-1.1`. The HF *model-card* licence field is unread (egress) — `U` until read.

### 0.2 Repos read (commit = what I read; each refreshed with `git fetch --depth 1 origin && git checkout -q FETCH_HEAD` — no change since RA except where noted)

| repo | commit | date | what I read |
|---|---|---|---|
| `NVIDIA/cosmos` | `3e3c6d6` (`…3e3c6d61dc15…`) | 2026-09-29 | `README.md`, `LICENSE`, `RELEASE.md`, `docs/reference/models.md`, `inference_benchmarks.md` (1,098 lines), `cookbooks/cosmos3/**` (README, reasoner README + prompt guide, NIM `support-matrix.md`/`configuration.md`) |
| `NVIDIA/cosmos-framework` | `cf5d68c` | 2026-09-23 | `configs/base/experiment/sft/models/{nano,super,edge}_model_config.py`, `inference/configs/model/Cosmos3-{Edge,Nano,Super}.yaml`, `reasoner/qwen3_vl/configs/Qwen3-VL-{2B,4B,8B,32B}-Instruct.json`, `reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json`, `cosmos3_edge/{configuration,modeling}_cosmos3_edge.py` |
| `nvidia-cosmos/cosmos-reason2` | `a3b4a1d` | 2026-06-07 | README, `scripts/inference_sample.py`, `docs/llmcompressor.md`, LICENSE |
| `nvidia-cosmos/cosmos-reason1` | `b84a3fa` | 2026-06-07 | README (new clone this session) |
| `NVIDIA/TensorRT-Edge-LLM` | `95515c2` (`…95515c2f87fb…`) | 2026-09-29 | `CHANGELOG.md` (0.11.0 dev), `supported-models.md`, `performance-benchmarks.md` (**latest tables = v0.10.0**, JetPack 7.2 / CUDA 13.2 / TRT 10.16), `examples/vla/{cosmos3,alpamayo}.md`, `developer_guide/models/cosmos3.md`, export/quantization sources, `tests/test_lists/*` |
| `NVlabs/alpamayo` (Alpamayo-1) | `11a0e01` | 2026-09-09 | README licence section, `models/base_model.py` |
| `NVlabs/alpamayo1.5` | `36aeb4c` | 2026-09-09 | README, `models/{alpamayo1_5,base_model}.py`, `helper.py`, loader |
| `NVlabs/alpamayo2` | `5e7975f` | 2026-09-15 | README, `models/expert.py`, `config.py` |
| `NVlabs/alpamayo-recipes` | `ae5bc10` | 2026-09-22 | README (model table), `recipes/alpamayo1_5_quant/README.md` |

---

## 1. What NVIDIA/Cosmos offers today, and which of it can be a per-tick reasoner backbone (priority 1)

### 1.1 The lineup (`P`: `NVIDIA/cosmos/README.md`, `docs/reference/models.md`)

| item | size (README) | what it is | usable as a frozen driving backbone? |
|---|---|---|---|
| **Cosmos3-Edge** | 4 B | omnimodal MoT; **Reasoner** (text+image+video → text) + Generator (video/action); no audio, no sound generation | ✅ via its **reasoner tower (2.43 B)** |
| **Cosmos3-Nano** | 16 B | as Edge + sound generation; reasoner "does not currently support the audio modality by design" | ✅ via **reasoner tower (8.76 B, Qwen3-VL-8B arch)** |
| **Cosmos3-Super** | 64 B | "teacher for distillation" (README) | ⛔ teacher only (reasoner tower 33.4 B) |
| Cosmos3-{Super-T2I, Super-I2V (+4Step), Nano-Policy-DROID, Edge-Policy-DROID} | — | post-trained *capability demos* "not part of the product line" (README) | ❌ generators / robot policies |
| Cosmos-Reason2-2B/8B/32B (`nvidia-cosmos/cosmos-reason2`) | 2/8/32 B | Qwen3-VL-architecture VLM, SFT+RL on physical common sense; **superseded by Cosmos 3, maintenance-only** | ✅ (2 B, 8 B); 32 B teacher |
| Cosmos-Reason1-7B (`nvidia-cosmos/cosmos-reason1`) | 7 B | Qwen2.5-VL-architecture (README line 119) | ⚠ older; superseded |
| Alpamayo 1 / 1.5 / 2-Super (`NVlabs/*`) | 10 / 10 / 34 B | driving VLA = VLM + diffusion expert | ✅ its **VLM** (8.2 B); 2-Super teacher |
| Cosmos-Predict/Transfer/Drive-Dreams | — | video *generators* | ❌ not encoders |
| "Cosmos Drive Nano" | — | **does not exist** in `NVIDIA/cosmos` (model tables), `nvidia-cosmos/*` (12 repos), `NVlabs/alpamayo*` (5 repos) — three more locations than RA probed | — |

### 1.2 Per-model specification (`P` unless tagged; parameter split = `M(ours)` arithmetic in Appendix A on the `P` configs; totals reproduce the published sizes: Qwen3-VL-8B 8.76 B vs 8.77 B official, Edge 2.43 B vs the "2.4 B reasoner")

| model | HF repo id (`P`: README/supported-models) | ViT + merger | LLM body | embed (+head) | architecture, hidden `d`, layers `L`, heads/KV, ffn |
|---|---|---:|---:|---:|---|
| **Cosmos3-Edge reasoner** | `nvidia/Cosmos3-Edge` | **0.49 B** (SigLIP2-naflex 27 L × 1152, patch 16, + 2×2 merger 4608→11520→2048) | **1.41 B** | **0.54 B** (131,072 × 2048, untied) | Nemotron-dense: `d=2048`, **28 attn+MLP pairs** (56 blocks), **16 q / 8 kv heads, head_dim 128**, ffn 9216 **relu² non-gated**, no q/k-norm on text, mRoPE [24,20,20], θ=1e8, ctx 131,072 (`Nemotron-2B-Dense-VL.json`, `configuration_cosmos3_edge.py`) — **rest of the 4 B = generator tower** (~1.6 B, `E`). The framework names the pre-MoT 2 B VLM `nvidia/Cosmos3-Edge-Reasoner` (`edge_model_config.py`, `_VLM_MODEL_SIZE == "2B"`); **whether that standalone repo is public is `U`** (the loader path is an internal S3 URI) |
| **Cosmos3-Nano reasoner** | `nvidia/Cosmos3-Nano` (tower initialised from `Qwen/Qwen3-VL-8B-Instruct` per `vlm_config.model_name`; weights then mid-trained, `cosmos3_ga_16bm8b_v2_midtrain`) | 0.57 B (27 L × 1152, patch 16, DeepStack taps [8,16,24]) | 6.95 B | 1.24 B (151,936 × 4096, untied) | Qwen3-VL-8B: `d=4096`, `L=36`, **32 q / 8 kv, hd 128**, ffn 12288 SwiGLU, ctx 262,144. Sum 8.76 B = **17.5 GB BF16 = 16.3 GiB ✔ vs README "~16–17 GB with vLLM/TRT/Transformers/NIM"**; framework backend loads the full omni weights ≈ 34 GB (`P` reasoner README) — rest of 16 B = generator (~7 B, `E`) |
| **Cosmos3-Super reasoner** | `nvidia/Cosmos3-Super` | 0.59 B | 31.21 B | 1.56 B | Qwen3-VL-32B: `d=5120`, `L=64`, 64 q / 8 kv, hd 128, ffn 25600; 33.35 B total — rest of 64 B = generator (~31 B, `E`) |
| **Cosmos-Reason2-2B** | `nvidia/Cosmos-Reason2-2B` | 0.40 B (24 L × 1024) | 1.41 B | 0.31 B (tied) | README: *"Cosmos-Reason2 is based on the Qwen3-VL architecture"*; size class = Qwen3-VL-2B: `d=2048`, `L=28`, 16/8, ffn 6144. **Own `config.json` unread ⇒ identity `INH`/`U`** |
| **Cosmos-Reason2-8B** | `nvidia/Cosmos-Reason2-8B` | 0.57 B | 6.95 B | 1.24 B | ≡ Qwen3-VL-8B (same caveat) |
| **Cosmos-Reason2-32B** | `nvidia/Cosmos-Reason2-32B` | 0.59 B | 31.21 B | 1.56 B | ≡ Qwen3-VL-32B (same caveat) |
| **Cosmos-Reason1-7B** | `nvidia/Cosmos-Reason1-7B` | 0.68 B (`R`) | 6.53 B | 1.09 B | Qwen2.5-VL-7B (`P` README): `d=3584`, `L=28`, 28/4 heads, ffn 18944 (**config `R`**, no clone carries it); total 8.29 B |
| **Alpamayo 1 Nano (R1-10B)** | `nvidia/Alpamayo-R1-10B` | — | **VLM 8.2 B + 2.3 B diffusion expert** (`P` recipes README) | — | recipes README: "Cosmos-Reason backbone (8.2 B)"; **the R1 repo's model class is `Qwen3VLForConditionalGeneration` with default `vlm_name_or_path="Qwen/Qwen3-VL-8B-Instruct"` (`alpamayo/.../base_model.py:207,368`)** ⇒ which lineage (Reason1/Qwen2.5-VL vs Qwen3-VL) the released R1 weights use is `U` (config unread) |
| **Alpamayo 1.5 Nano** | `nvidia/Alpamayo-1.5-10B` | — | VLM ≈ 8 B (≡ Qwen3-VL-8B arch) + expert | — | README: RL post-trained "on the Cosmos-Reason2 backbone"; `Qwen3VLForConditionalGeneration`; expert = `deepcopy(vlm.config.text_config)` + `expert_cfg` overrides, `embed_tokens` deleted (`alpamayo1_5.py:102-112`) ⇒ expert config values `U` (checkpoint unread). Whole-model size **~22 GB BF16 / ~11 GB FP8 / ~9 GB AutoQuant 6.5 bpe** (`P` quant recipe) |
| **Alpamayo 2 Super** | `nvidia/Alpamayo2-Super` | — | **32 B VLM + 2 B expert** (`P` alpamayo2 README) | — | "built on the NVIDIA Cosmos 3 backbone" (recipes README); 6 cameras × 4 frames; 2-GPU demo peaks **67 GiB (VLM GPU) + 71 GiB (expert GPU)** |

### 1.3 Interface facts: tokens at 256×640, fps, licence, driving tuning

**Token accounting (`M(ours)` arithmetic on `P` configs):** Qwen3-VL-lineage and Cosmos3-Edge: patch 16, 2×2 merge ⇒ **32 px per token** ⇒ 256×640 = 16×40 = 640 patches ⇒ **8×20 = 160 tokens/frame**. Edge: `P` `modeling_cosmos3_reasoner_visual.py` ("patch embed 768→1152", "27 × EncoderLayer", "2×2 merge reshape"). **256×640 = 163,840 px = exactly `MIN_PIXELS` of Alpamayo-1.5 (`helper.py:23`: 163,840 / max 196,608 ⇒ 160–192 tokens)** — the corpus resolution is Alpamayo's own lower pixel bound. ⚠ **Trap:** Cosmos-Reason2's own sample script sets `min_vision_tokens = 256` (`inference_sample.py:62`) which would resize a 256×640 frame up to ≈ **352×832 = 286 tokens** (`E`, Qwen smart-resize rule `R`) — override `min_pixels ≤ 163,840` or the token budget is 1.8× what the tables below assume. Qwen2.5-VL (Reason1): patch 14, merge 2 ⇒ 28 px/token ⇒ 9×23 = **207 tokens** (`E`). Qwen3-VL video mode compresses 2 frames per token block (`temporal_patch_size 2`, `P` configs) — not used below (frames counted as separate images, conservative).

| model | recommended fps / frames | licence (quoted) | driving-tuned? (data, hours) |
|---|---|---|---|
| Cosmos3-Edge/Nano/Super | video: `{"fps": 4, "do_sample_frames": True}` (reasoner prompt guide l.32) · NIM sample `fps 4.0` · TRT Edge server example `"fps": 2.0` · Edge eager benchmark = "Video 1 / 2 FPS" · Edge policy obs 736×544 | **OpenMDW-1.1** (LICENSE l.1; SPDX in sources; HF card `U`) | **not stated** in any file read. The reasoner prompt guide contains an *"autonomous vehicle planning system"* next-action prompt and a critical-object driving prompt (`reasoner_prompt_guide.md` l.337-352, 612-628) ⇒ AV content exists in post-training; **data source/hours `U`** (Cosmos 3 technical report unread — egress) |
| Cosmos-Reason2-2B/8B/32B | `--fps 4` (README l.228, 248) | code Apache-2.0; **models "NVIDIA Open Model License"** (README, terms unread) | *"post-trained with physical common sense and embodied reasoning data with SFT and RL"* — no AV-specific claim; AV hours `U` |
| Cosmos-Reason1-7B | — | same as Reason2 (README l.137-139) | as above; FP8 W8A8 PTQ script provided |
| Alpamayo 1 / 1.5 | **4 frames/camera at 0.1 s spacing** (t0−0.3…t0) × 4 cameras (`num_frames=4`, `time_step=0.1`); 1.5: flexible camera count | code Apache-2.0; **weights OpenMDW-1.1** ("see the HF Model Card", READMEs) | **yes:** R1 ">1 B images from 80,000 hours of driving" (recipes README); 1.5 RL post-trained; CoC labels. PhysicalAI-AV = "1,700+ hours … 306 K clips" (recipes README) — **80 k h ≫ 1.7 k h ⇒ training data is *not* only the public set; overlap with it unknown** |
| Alpamayo 2 Super | 6 cameras (ids 0,1,2,3,5,6) × 4 frames; pixel budget 163,840–196,608 (`config.py:68-69`) | code Apache-2.0; weights OpenMDW-1.1 (README) | yes: "trained on 110,000+ hours" (recipes README) |

---

## 2. Latency evidence (priority 2)

### 2.1 What NVIDIA has measured (all `P`; the *absent* rows are listed in 2.2)

**(a) Cosmos3-Edge reasoner — eager HF-Transformers BF16, "preliminary", `inference_benchmarks.md` §"Embedded-Platform Eager Transformers":**

| board (mode) | input | prompt tokens | prefill tok/s | prefill | decode tok/s |
|---|---|---:|---:|---:|---:|
| **Jetson AGX Thor T5000** (128 GB, MAXN) | text | 1,705 | 8,717 | **0.20 s** | 37.3 |
| | image | 911 | 4,845 | **0.19 s** | 42.6 |
| | video | 1,263 | 6,032 | **0.21 s** | 41.8 |
| Thor T4000 (64 GB) | text / image / video | 1,705 / 911 / 1,263 | 6,519 / 3,471 / 4,164 | 0.26 / 0.26 / 0.30 s | 34.1 / 40.3 / 38.1 |
| Thor T3000 (32 GB) | text / image / video | same | 5,230 / 2,710 / 3,388 | 0.33 / 0.34 / 0.37 s | 29.7 / 36.3 / 33.7 |
| Thor T2000 (16 GB) | text / image / video | same | 2,355 / 1,233 / 1,543 | 0.72 / 0.74 / 0.82 s | 15.7 / 19.6 / 18.0 |
| **AGX Orin** (64 GB) | text / image / video | same | 3,260 / 1,840 / 2,103 | **0.52 / 0.50 / 0.60 s** | 12.3 / 12.3 / 12.2 |

(No precision other than eager BF16; image resolution/patch count of the 911-token prompt not stated.) Derived (`M(ours)`): LLM-only rate 8,717 tok/s ⇒ **0.115 ms/token = 24.6 TFLOPS effective** on the 1.41 B body (same figure RA got); ViT share of the 911-token image row ≈ 190 − 104 = 86 ms ⇒ ≈ 23.5 µs/patch (`E`, assumes 911 tokens ≈ 3,644 patches).

**(b) Cosmos 3 reasoners, vLLM serving, concurrency 1, TTFT (ms), workload "input 50 tokens + video at 1 FPS / 2 FPS, output 1"** — discrete GPUs only; **the video clip length/resolution/token count is not stated ⇒ not convertible to a per-token rate**, kept only as an order-of-magnitude check that TTFT ≈ doubles with fps:

| model | RTX PRO 6000 BW | H100 SXM | B200 | B300 | H200 141 GB |
|---|---|---|---|---|---|
| Cosmos3-**Edge** (also RTX PRO 4500: 165.8 / 371.7) | 142.0 / 239.9 | — (not in Edge table) | — | — | — |
| Cosmos3-**Nano** | 187.6 / 316.9 | 145.5 / 228.8 | 115.6 / 169.0 | 80.7 / 126.3 | 142.5 / 229.7 |
| Cosmos3-**Super** | 534.7 / 978.8 | — | 212.1 / 350.3 | 176.2 / 301.5 | 327.2 / 592.2 |

**No Thor / Orin row exists for Cosmos3-Nano or -Super in `inference_benchmarks.md`** (GPU sections: RTX PRO 4500/6000, H20, H100 NVL/SXM, H200 NVL/SXM, B200, B300 only).

**(c) TensorRT-Edge-LLM v0.10.0, Jetson AGX Thor, batch 1, ViT FP16 + LLM NVFP4 (JetPack 7.2, CUDA 13.2, TRT 10.16; `performance-benchmarks.md` "v0.10.0 Runtime Performance Dashboard" + "`llm_bench`")** — these are **architecture-identity anchors** for Cosmos-Reason2 / Cosmos 3 Nano / Alpamayo (`INH`-grade identity, see §1.2):

| measured model (≡ which Cosmos model) | ViT ms (one COCO image, ~265 tok / 1,060 patches) | prefill @292 tok | prefill @2,048 tok (`llm_bench`) | INT4-AWQ @2,048 | decode tok/s |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-2B (≡ Cosmos-Reason2-2B) | 11.4 | 12.7 ms | 28.0 ms | 145.3 ms | 136 (152 in runtime table) |
| Qwen3-VL-4B (no Cosmos twin) | 11.6 | 22.3 | 62.2 | 364.4 | 73 |
| **Qwen3-VL-8B (≡ Cosmos-Reason2-8B ≡ Cosmos 3 Nano reasoner ≡ Alpamayo VLM)** | **15.7** | **32.1** | **104.5** | 662.2 | 44 |
| **Qwen2.5-VL-7B (≡ Cosmos-Reason1-7B)** | 23.4 (349 tok) | 25.0 (@376) | **83.1** | — | 59 |
| Qwen3.5-4B (RA's R1; not Cosmos) | 10.9 | 31.6 | 95.2 | 371.4 | 68 |

Other hardware for the same architectures (`P`, same file): **Jetson AGX Orin 64 GB, INT4-AWQ prefill @2,048: Qwen3-VL-2B 298 ms, 4B 758 ms, 8B 1,388 ms** (no NVFP4/FP8 on Orin — "Do not select FP8, MXFP8, FP4, or NVFP4 checkpoints for Orin", `supported-models.md`); **DGX Spark (GB10), NVFP4 @2,048: 2B 54.8, 4B 122.2, 8B 185.1 ms** (1.8× slower than Thor for the same 8B row). Batching 8 images through the ViT: 75 ms vs 8 × 11.4 = 91 ms (−18 %, `P`) ⇒ cameras do not batch for free.
⚠ **The "GPU Mem (MB)" column of that table (e.g. 1,510 MB for Qwen3-VL-8B NVFP4) is not the memory footprint** — the weights alone are ≥ 4 GB. It is an unspecified runtime counter; per CLAUDE.md's Thor-probe rule it is **not admissible for a memory budget** (use in-process `torch.cuda.max_memory_allocated()` or the engine's own device-memory query, §6).

**(d) Other NVIDIA statements:** Cosmos-Reason2 README: *"NVIDIA Jetson AGX Thor (Edge) | 13.0 | Transformers inference. vLLM inference is coming soon!"* — **no numbers**; min GPU memory **24 GB (2 B) / 32 GB (8 B)** (vLLM incl. KV cache). Alpamayo-1.5 on H100 80 GB: **~24 GB single-sample / ~40 GB (16 samples) / ~60 GB (16 + CFG)**. AR1 paper "99 ms on-vehicle" is `INH` from RA/`ALPAMAYO2_SUPER_ANALYSIS.md`, not Thor, not re-read.

### 2.2 Measured vs absent — the gap table

| model | NVIDIA-measured on Thor? | closest measured proxy | evidence class of the proxy |
|---|---|---|---|
| Cosmos3-Edge reasoner | **eager BF16 only** (2.1a); **no TRT row** | Qwen3-VL-2B TRT rows (same 1.41 B body FLOPs) + Qwen3-VL-8B ViT (same 27×1152 tower) | `E` (architecture-FLOP identity, not kernel identity: relu² vs SwiGLU, no q/k-norm) |
| Cosmos3-Nano reasoner | **no** (NIM supports Thor, no latency table) | Qwen3-VL-8B rows | `INH` identity (framework config `P`) ⇒ `E` |
| Cosmos3-Super reasoner | **no** | none (nearest: Gemma-4-31B NVFP4 787 ms @2,048; Qwen3.5-27B NVFP4 449 ms) | `E`, extrapolation 4.5× beyond the largest fitted body |
| Cosmos-Reason2-2B / 8B | **no** (Transformers inference "tested", no numbers) | Qwen3-VL-2B / 8B rows | `INH` identity ⇒ `E` |
| Cosmos-Reason2-32B | **no** | none | `E`, extrapolation |
| Cosmos-Reason1-7B | **no** | Qwen2.5-VL-7B rows | `R` identity ⇒ `E` |
| Alpamayo 1 / 1.5 / 2-Super VLM | **no** (FP16 TRT pipeline documented, no table) | Qwen3-VL-8B (FP16 not measured) | `E`, low confidence |

### 2.3 Estimates for the models with no row — method (same as RA §5, extended)

Method (`M(ours)`, Appendix A): NVFP4 LLM prefill is linear in tokens between NVIDIA's two anchors, `t(N) = t0 + s·N`; the three Qwen3-VL rows give **`s = 0.00612 ms/token per billion body params` (0.00593–0.00625 across 2/4/8 B — 5 % spread)** and a weight-read floor `t0 ≈ W_bytes / 195 GB/s` (**195 GB/s = 71 % of the 273 GB/s published peak**, calibrated on the 8 B row; NVFP4 = 0.5625 B/param incl. block scales `E`). ViT time scales with patches (640 per 256×640 frame vs 1,060 in NVIDIA's COCO image ⇒ **0.6×**; RA/`sec_eff.tex` Table `tab:thor` conservatively did not scale). Precision ladder (`E`, low confidence ±40 %): **FP8** slope ×1.3 (Gemma-4-31B FP8/NVFP4 @2,048 = 1036/787 = 1.32, `P`); **TRT FP16** ≈ 57 TFLOPS (mid of 43 = the AWQ FP16-math row and ~70 = Gemma-4-12B FP16 row with an assumed body); **eager BF16** = the Edge row (24.6 TFLOPS).
**Precision-ladder calibration rows (`P`, TRT-Edge-LLM v0.10.0, Thor, batch 1; Gemma-4 is the only family NVIDIA benchmarks at several precisions, so it is the only source for the ratios — it is *not* a Qwen/Cosmos architecture, hence `E` ±40 %):** Gemma-4-31B @2,048 tok: **NVFP4 786.9 ms · FP8 1,036.4 ms · INT4-AWQ 2,899.5 ms**; Gemma-4-12B FP16 @2,048: 622.8 ms; Gemma-4-E4B @292 tok: FP16 61.0 · FP8 45.1 · NVFP4 36.1 · AWQ 84.8 ms; E2B @292: 36.6 · 28.6 · 24.8 · 47.7 ms. Qwen3-VL-8B @2,048: NVFP4 104.5 vs INT4-AWQ 662.2 ms (6.3×). NVIDIA's FP16/NVFP4 ratio at ≈ 300 tokens is only 1.5–1.7× on the E-tier rows, but those models stream few weights (per-layer-embedding design) and are launch-bound; for a *dense* 8 B the FP16 weight read alone is 13.9 GB / 195 GB/s = 71 ms against 20 ms for NVFP4, so the model below predicts ≈ 4.5× at 292 tokens and ≈ 5.5× at 2,048 (the 6.3× AWQ gap is the measured bound). **No NVIDIA row tests a dense ≥ 7 B model in FP16/FP8 on Thor, so the FP16/FP8 columns are the least trustworthy numbers in this file.**

**Out-of-sample check (Qwen2.5-VL-7B, not used in the fit): the no-row model over-predicts by +35 % (@376 tok) / +21 % (@2,048)** ⇒ treat the tables as **upper-leaning, ±35 %**; the 32 B row is a **4.5× extrapolation of body size beyond the largest measured (6.95 B) — bracket only** (the sibling `hicap_latency_budget.py` refuses a point estimate beyond 2×; I give one point, flagged, because the criterion needs a yes/no — read it as "≈ 0.15–0.4 s").

**Per-refresh latency, ms — ViT + LLM prefill only** (excludes pre-processing, engine hand-off, 3-depth pooling ≈ 15 ms `A` per `hicap_latency_budget.py`); `N = cams × frames × 160 + 64` tokens; frames counted as separate images (no ViT-cache, no video-mode 2× compression); **`E` for every cell**:

*NVFP4 LLM + FP16 ViT (TRT) — Edge NVFP4 is untested by NVIDIA (CI is FP16 only):*

| model | 1c×1f (N=224) | 1c×3f = 3c×1f (N=544) | 3c×3f (N=1,504) | 7c×1f (N=1,184) | 7c×3f (N=3,424) |
|---|---:|---:|---:|---:|---:|
| Cosmos-Reason2-2B ≡ Qwen3-VL-2B (rows) | 19 | 36 | 85 | 69 | 185 |
| **Cosmos3-Edge** (2B-row LLM + SO400M ViT) | 22 | 43 | 109 | 87 | 239 |
| Cosmos-Reason2-8B / Cosmos 3 Nano / Alpamayo VLM (rows) | 39 | 71 | 167 | 135 | 360 |
| Cosmos-Reason1-7B (rows; 207 tok/frame) | 35 | 77 | 204 | 162 | 457 |
| Cosmos-Reason2-32B / Cosmos 3 Super (**extrapolated**) | 142 | 222 | 463 | 383 | 943 |

*FP8 (`E`; recipe exists: `cosmos-reason2` llmcompressor `--precision fp8`, Alpamayo-1.5 ModelOpt FP8, Cosmos-Reason1 W8A8 script, NIM Nano/Super FP8 profiles — none benchmarked on Thor):*

| model | 1c×1f | 1c×3f = 3c×1f | 3c×3f | 7c×1f | 7c×3f |
|---|---:|---:|---:|---:|---:|
| Cosmos-Reason2-2B | 23 | 40 | 92 | 75 | 196 |
| Cosmos3-Edge | 25 | 48 | 116 | 93 | 251 |
| 8 B tier | 58 | 94 | 204 | 168 | 424 |
| 32 B tier | 225 | 324 | 619 | 521 | 1,210 |

*TRT FP16/BF16 (`E`; **the only path NVIDIA documents for Alpamayo on Thor** (`examples/vla/alpamayo.md`: "Only FP16 is supported for Alpamayo export") and the only precision NVIDIA's CI tests for Cosmos3-Edge):*

| model | 1c×1f | 1c×3f = 3c×1f | 3c×3f | 7c×1f | 7c×3f |
|---|---:|---:|---:|---:|---:|
| Cosmos-Reason2-2B | 39 | 68 | 157 | 127 | 334 |
| **Cosmos3-Edge** | 41 | 76 | 180 | 145 | 389 |
| **8 B tier (Alpamayo VLM FP16)** | 135 | 232 | 523 | 426 | 1,105 |
| 32 B tier | 575 | 944 | 2,052 | 1,683 | 4,268 |

*Eager HF-Transformers BF16 (what the Cosmos-Reason2 README calls "tested on Thor"; Edge from NVIDIA's own row, 8 B by FLOP scaling):* **Edge 54 / 119 / 313 / 248 / 702 ms**; 8 B **206 / 408 / 1,013 / 811 / 2,223 ms**. ⇒ **eager BF16 alone does not meet even the async budget at 3 cameras × 3 frames** — an engine (TRT or compiled) is required.

### 2.4 Weight memory vs Thor's 128 GB (`M(ours)`; NVFP4/FP8 quantise the linear bodies only; embeddings/head + ViT kept FP16 = upper bound — a prefill-only build can drop the LM head, −0.6 to −1.6 GB)

| model | NVFP4 | FP8 | BF16 | **headroom left after weights** for perception/planning/OS (NVFP4 / FP8 / BF16) |
|---|---:|---:|---:|---|
| Cosmos3-Edge reasoner (2.43 B) | 2.8 GB | 3.5 | 4.9 | 64.0 / 63.3 / 61.9 GB |
| Cosmos-Reason2-2B (2.12 B) | 2.2 | 2.8 | 4.2 | 64.6 / 64.0 / 62.6 |
| Cosmos-Reason1-7B (8.29 B) | 7.2 | 10.1 | 16.6 | 59.6 / 56.7 / 50.2 |
| **8 B tier (8.76 B: Cosmos-Reason2-8B, Cosmos 3 Nano reasoner, Alpamayo VLM)** | **7.5** | **10.6** | **17.5** (= 16.3 GiB ✔ README) | 59.3 / 56.2 / 49.3 |
| 32 B tier (33.35 B) | 21.8 | 35.5 | 66.7 | 45.0 / 31.3 / **0.1** |

**Budget assumptions (`A`, PI to confirm):** headroom = 128 GB − weights − **17.2 GB** (NVIDIA's own default *host reserve* of 16 GiB on Thor, `P`: NIM `support-matrix.md` "Unified-memory thresholds"; Thor T5000 nominal shared memory **123 GiB**, so NVIDIA's effective pool is 123 − 16 = **107 GiB**) − **40 GB** other tenants (`A`) − **4 GB** engine workspace/activations (`A`). Read: every ≤ 8.8 B model leaves ≥ 49 GB in any precision; **32 B BF16 does not fit** (NVIDIA's own NIM floor for Cosmos 3 Super Reasoner BF16 TP1 is **135 GiB > Thor's total**; FP8 67 GiB / NVFP4 73 GiB fit but "Super is not recommended on these systems for practical turnaround", `P`). Cosmos3-Nano Reasoner NIM floor: **23.1 GiB/device in BF16, FP8 and NVFP4** (`P`), with `NIM_GPU_MEMORY_UTILIZATION` **0.80 (image-only) / 0.70 (video/mixed)** required on Thor (`P` configuration.md). Feature-cache disk, not RAM, is the memory that scales with `d`: §4.4.

---

## 3. Runtime support on Thor (priority 3)

`P` = `supported-models.md` @`95515c2` (JetPack 7.2 rows) / cookbook READMEs. "Supported" = the checkpoint ID is listed; NVIDIA states *"not every listed checkpoint has been fully verified on every supported platform and precision"*.

| model | TensorRT-Edge-LLM | vLLM / NIM / HF on Thor | status for HiCAP |
|---|---|---|---|
| **Cosmos3-Edge** (`cosmos3_edge`) | **listed** ("text and image/video to reasoning"); documented export `--task reasoning` → `llm_build` (`--maxInputLen 2048 --maxKVCacheCapacity 4096` in the example) + `visual_build` → `llm_inference`; **native video** in 0.11.0 ("Added native Cosmos3-Edge reasoner video input"); **CI-tested precision: fp16 only** (`test_lists/l1_pipeline_jedha_vlm.yml`, `l1_export_blackwell_large.yml`); reasoner LM = the shared `CausalLM` + one relu² MLP class ("quantized builds keep working" per docstring — untested) | vLLM ≥ 0.23.0 "native Cosmos3 Reasoner support" (no Thor statement); **not a NIM variant** (NIM = nano/super only); HF: `AutoModelForImageTextToText` on Transformers **`main` only**; **measured eager on T5000** | **works as-is at FP16 (needs `maxInputLen` ≥ 3,424 for 7c×3f — the Alpamayo example builds with 3,424)**; NVFP4/FP8 = *needs work/verification*; **3-depth tap = needs export change (§4)** |
| **Cosmos3-Nano** | **not listed** (only Edge). Architecture = Qwen3-VL-8B (framework `P`), a supported family, but the omni/MoT checkpoint layout is not stated as exportable | **NIM Reasoner Nano is a documented Thor target** (NVFP4 profile auto-selected on compute capability ≥ 10.0; DFlash speculative decode on by default — irrelevant to prefill); vLLM ≥ 0.23.0; TRT-LLM reasoner server (cookbook); HF `Cosmos3OmniForConditionalGeneration` loads the **reasoner tower only** | runnable **as a black-box server (no tap)**; in-process tap via HF; TRT-Edge-LLM = *needs work* |
| **Cosmos3-Super** | not listed | NIM Super FP8/NVFP4 TP1 meet the Thor memory floor; "not recommended … for practical turnaround" | teacher, off-vehicle |
| **Cosmos-Reason2-2B / 8B** | **listed by ID** under "Qwen3-VL and compatible checkpoints" (`qwen3_vl` family; export/NVFP4/AWQ paths as for Qwen3-VL) | vLLM ≥ 0.11.0 (README); *"Jetson AGX Thor: Transformers inference. vLLM inference is coming soon!"* (snapshot 2026-06-07) | **works as-is**; the only Cosmos-family entries with real TRT anchors (by identity) |
| **Cosmos-Reason2-32B** | not listed ("larger dense checkpoints … require case-by-case validation") | — | teacher |
| **Cosmos-Reason1-7B** | `nvidia/Cosmos-Reason1-7B` **not listed**; `Qwen/Qwen2.5-VL-7B-Instruct` (+AWQ, `nvidia/…-FP8`, `nvidia/…-NVFP4`) **is** | README PTQ recipe pins `vllm==0.9.2` + `llmcompressor` (FP8 W8A8 script) | same architecture ⇒ exportable in principle (`U`) |
| **Alpamayo-R1-10B** (`alpamayo_r1`) | **listed**; chained VLM + action engines, **FP16 only**, one `action_inference` call; visual engine `--minImageTokens 160 --maxImageTokens 18432 --maxImageTokensPerImage 192`, 16 images (4 cam × 4 frames) | — | VLM-only prefill extraction = custom (the shipped runner always runs the expert) |
| **Alpamayo-1.5-10B / 2-Super** | **not listed** | HF/PyTorch reference code only | as above; ModelOpt FP8/NVFP4 recipe exists for 1.5 (PyTorch/H100-class, not TRT-Edge-LLM) |

**Other 0.10.1/0.11.0 knobs that matter for a prefill-only frozen backbone (`P` CHANGELOG):** *DART visual-token pruning* (0.10.1) "to reduce VLM prefill work"; *media artifact cache that retains ViT and audio encoder embeddings across repeated media prefixes* (0.10.1) — a ready-made ViT-feature cache for frames repeated between refreshes; context/KV-cache reuse; *"Thor-optimized NVFP4-A16 dense … kernels"* and Blackwell SM110 NVFP4 GEMM/GEMV (0.10.1/0.11.0) — the 0.10.0 tables in §2.1c **predate** these, so NVFP4 numbers may be pessimistic (`H`). No 0.11.0 performance tables are published yet.

---

## 4. Feature-extraction fit for the frozen-tap design (priority 4)

Requirement: **(i)** hidden states at 3 depths (L/2, 3L/4, L), **(ii)** no decode. Verdict per family:

**4.1 Alpamayo (1 / 1.5 / 2-Super) — how the driving expert reads the VLM (`P`, quoted):**
* `alpamayo1_5.py:102-112`: `expert_config = copy.deepcopy(self.vlm.config.text_config)` … `self.expert = AutoModel.from_config(expert_config)` … `del self.expert.embed_tokens` — the expert is a **same-shaped transformer stack without embeddings** (config = VLM text config + `expert_cfg` overrides, values `U`).
* `:315-331`: `vlm_outputs = self.vlm.generate(…)` … `prompt_cache = vlm_outputs.past_key_values` — **the VLM first *decodes* the Chain-of-Causation text**, and the cache handed to the expert is the post-decode cache.
* `:372-381`: `self.expert(inputs_embeds=future_token_embeds, position_ids=position_ids, past_key_values=prompt_cache, attention_mask=attention_mask, use_cache=True, **forward_kwargs)` then `prompt_cache.crop(prefill_seq_len)`; `forward_kwargs["is_causal"] = False` when `expert_non_causal_attention`. Alpamayo-2-Super `expert.py`: `delta = vlm_outputs.rope_deltas + vlm_outputs.past_key_values.get_seq_length()`.
* **Implication:** the expert reads **the per-layer K/V of *all* VLM layers** (a per-layer read, not one pooled vector) **after decode**. A prefill-only tap at the end of the prompt is *not* the state the expert was trained on (no CoC tokens in the cache) ⇒ using the Alpamayo VLM as HiCAP's driving-aware arm measures the driving knowledge that survives in *prompt-time* states, not the deployed VLA path (`H`: magnitude unknown). Our heads train from scratch on the frozen features, so this is a *fairness/interpretation* caveat, not a blocker.
* KV size for reference: per token per layer `2 × 8 kv × 128 × 2 B = 4 KiB` ⇒ 8 B tier (36 L) **147 KiB/token**, 544 tokens ≈ 80 MB (`M(ours)`); taps of hidden states are more general than KV (KV_ℓ = W_kv · norm(h_{ℓ−1})).

**4.2 Qwen3-VL-architecture models (Cosmos-Reason2, Alpamayo VLM, Cosmos 3 Nano/Super via `Cosmos3OmniForConditionalGeneration`):** HF `Qwen3VLForConditionalGeneration` — hidden states via the standard `output_hidden_states=True` (L+1 tensors; `R`, not re-read here) or forward hooks on the decoder layers; **DeepStack** injects ViT mid-layer features into the *first three* LLM layers (`deepstack_visual_indexes` [8,16,24] for 8 B/32 B, [5,11,17] for 2 B/4 B — `P` configs), so mid/late taps are unaffected and truncating the LLM at 3L/4 (saving ≈ 25 % of prefill, `E`) is compatible. Cosmos 3's `Qwen3VLMoTConfig` wrapper carries the generator experts; the *Transformers* integration "loads only the Reasoner tower … and drops Generator, audio, and action parameters" (`P` reasoner README) — use it, **not** the framework backend (full omni weights ≈ 34 GB for Nano).

**4.3 Cosmos3-Edge:** framework-native port `modeling_cosmos3_edge.py` (`P`): `Cosmos3EdgeTextModel.forward(…, output_hidden_states=…)` appends `all_hidden_states += (hidden_states,)` **once per *block*** (lines 323-348) — **56 blocks alternate attention / MLP, so the tuple has 57 entries and "depth L/2, 3L/4, L" = tuple indices 28 / 42 / 56 (even = pair boundaries; 57 = post-final-norm)**, not 14/21/28. The port is **cache-free** ("attention takes no `past_key_values`"; generation = full-context recompute; `use_cache` accepted "for API parity") ⇒ no KV-cache interface in that port (irrelevant for prefill-only; relevant if a KV-read head is wanted — use vLLM/TRT). The Transformers-`main` class was not read; `P` (`cookbooks/cosmos3/README.md` l.479-482): *"Cosmos3-Edge uses a separate Transformers integration (`AutoModelForImageTextToText` / `Cosmos3EdgeForConditionalGeneration`) with `nvidia/Cosmos3-Edge`. Do not load Edge with `Cosmos3OmniForConditionalGeneration`. Edge support is on Transformers `main` and is not yet in a stable PyPI release."* The framework's `configuration_cosmos3_edge.py` parses the *renewed* root `config.json` in the **native schema, "no remote code"** — so a `--trust-remote-code` flag for Edge (the sibling `hicap_backbone_latency_bench.py` line 1434 adds one) is probably stale (`U`; the HF `config.json` is unread).

**4.4 TensorRT-Edge-LLM (what runs on Thor):** the LLM engine outputs `logits` + `present_key_values_i`; **but the shared `CausalLM` already has multi-layer hidden-state outputs for other purposes** (`P`, `modeling_default.py:916-987, 1075-1085`; `config.py:976`): `emit_hidden_states` (full-sequence last-layer normed hidden states as an extra ONNX output — used by Qwen3-Omni's thinker→talker hand-off); `--eagle-base` export "adds `hidden_states` output" of **3 target layers by default** (`return len(self.eagle3_target_layer_ids) or 3`); DFlash/JetSpec/DSpark base engines emit `[B, S, n_layers·H]` (DFlash default layer ids `[1, 8, 15, 22, 29]`, `model.py:424`). `Cosmos3ReasonerCausalLM` is a few-line subclass of that `CausalLM` (`modeling_cosmos3_reasoner_text.py`). **⇒ a 3-depth tap engine is a small export change, not new engine work (`H`; untested — whether `--eagle-base` accepts `cosmos3_edge` is `U`).** Also: the Python export API has a `num_decoder_layers` truncation parameter (`export.py:1012`, range-checked 1…L in `model.py:698-720`; **no CLI flag found**) ⇒ a 3L/4 truncated engine is feasible — **but the same code raises `NotImplementedError` when `num_decoder_layers` is combined with the eagle/dflash/jetspec/dspark base variants** (`model.py:704-716`), so *truncation + the eagle-base multi-layer tap cannot be combined as shipped* (a custom `emit_hidden_states` subclass would be needed; `U`). **vLLM/NIM servers expose no hidden-state tap** (`R`; CLM uses `--runner pooling`, last token only) ⇒ **the frozen-tap design forces an in-process engine** (HF/PyTorch-compiled or a custom TRT-Edge-LLM export), not the OpenAI-compatible servers.

**4.5 What it implies for HiCAP's design (each `H` until measured on Thor):**
1. Tap in *block* units for Edge and *layer* units for Qwen3-VL; store the tap indices in the cache manifest so features from different backbones are never mixed.
2. **Feature width per token `d`:** Edge/Cosmos-Reason2-2B **2,048**; Cosmos-Reason1-7B 3,584; 8 B tier **4,096**; 32 B tier 5,120. The paper's cache sentence ("128 tokens of width 2,048 at three depths, 1 byte ⇒ 148 GB") **is exactly right for the Edge/2 B tier and doubles for the 8 B tier** (`M(ours)`, 188,640 ticks × 128 × d × 3 × 1 B): **148 GB (d 2,048) · 260 GB (3,584) · 297 GB (4,096) · 371 GB (5,120)** at int8; ×2 at fp16. A 10 Hz refresh (paper: 742 GB at d 2,048 int8) is ×2 for the 8 B tier.
3. Prefill-only means the LM head is dead weight — drop it in the engine (saves 0.6–1.6 GB, and the head GEMM at the last token).
4. For Alpamayo, decide *before* the run whether the paired arm is "prompt-time states" (as here) or "post-CoC KV" (needs decode — violates the no-decode design).

---

## 5. Ranked recommendation under a LATENCY criterion (priority 5)

### 5.1 The criterion, as pre-registered by us (brief 2026-09-29) — two readings

* **Fast path:** trainable ~27 M network at 10 Hz, **p95 ≤ 25 ms** — *not* affected by backbone size **except through GPU contention** (below).
* **Reading A (strict, "in-line"):** VLM refresh **p95 ≤ 50 ms** per refresh at the chosen camera count.
* **Reading B (async 2 Hz):** refresh **≤ 250 ms** (= duty 0.5 at 2 Hz) **and** token age ≤ 500 ms + refresh (holds by construction whenever refresh < 500 ms; `hicap_latency_budget.py`'s worst case `A = T_r + overhead + 1/f_r`).
* **NVIDIA publishes means, not p95** ⇒ every verdict below is on a **p50-style `E` estimate**. Code: `GG` = passes even with a guard (1.3 × T_r + 15 ms `A` overhead ≤ threshold; 1.3 = `H`), `g.` = passes the raw p50 estimate only, `..` = fails. Duty = 2 Hz × (T_r + 15 ms).

### 5.2 Verdict matrix (from Appendix A output; A = ≤ 50 ms, B = ≤ 250 ms)

| model @ precision | 1c×1f (corpus reality today) | 3c×1f | 3c×3f | 7c×1f | 7c×3f |
|---|---|---|---|---|---|
| **Cosmos3-Edge @ NVFP4** (untested by NVIDIA) | A `GG` B `GG` (22 ms) | A `g.` B `GG` (43) | A `..` B `GG` (109; duty 0.25) | A `..` B `GG` (87) | A `..` B `g.` (239; duty 0.51) |
| **Cosmos3-Edge @ FP16** (CI-tested precision) | A `g.` B `GG` (41) | A `..` B `GG` (76) | A `..` B `GG` (180; duty 0.39) | A `..` B `GG` (145) | A `..` B `..` (389) |
| Cosmos3-Edge @ eager BF16 (NVIDIA-measured basis) | A `..` B `GG` (54) | A `..` B `GG` (119) | A `..` B `..` (313) | A `..` B `g.` (248) | A `..` B `..` (702) |
| **Cosmos-Reason2-2B @ NVFP4** (TRT rows by identity) | A `GG` B `GG` (19) | A `g.` B `GG` (36) | A `..` B `GG` (85; duty 0.20) | A `..` B `GG` (69) | A `..` B `g.` (185; duty 0.40) |
| **8 B tier @ NVFP4** (Cosmos-Reason2-8B / Cosmos 3 Nano / Alpamayo VLM) | A `g.` B `GG` (39) | A `..` B `GG` (71) | A `..` B `GG` (167; duty 0.36) | A `..` B `GG` (135) | A `..` B `..` (360) |
| 8 B tier @ FP8 | A `..` B `GG` (58) | A `..` B `GG` (94) | A `..` B `g.` (204) | A `..` B `GG` (168) | A `..` B `..` (424) |
| **8 B tier @ FP16 (Alpamayo's only TRT path)** | A `..` B `GG` (135) | A `..` B `g.` (232; duty 0.49) | A `..` B `..` (523) | A `..` B `..` (426) | A `..` B `..` (1,105) |
| Cosmos-Reason1-7B @ NVFP4 | A `g.` B `GG` (35) | A `..` B `GG` (77) | A `..` B `g.` (204) | A `..` B `GG` (162) | A `..` B `..` (457) |
| 32 B tier @ NVFP4 (**extrapolated**) | A `..` B `GG` (142) | A `..` B `g.` (222) | A `..` B `..` (463) | A `..` B `..` (383) | A `..` B `..` (943) |
| 32 B tier @ FP8 / FP16 | fails B from 3c×1f (324 / 944) | | | | |

(1c×3f equals 3c×1f in tokens and ViT patches under the no-cache convention, so it shares the column.)

**Two things the matrix cannot see (both `U`, both decisive):**
1. **GPU contention.** The refresh runs on the same 20-SM GPU as the 10 Hz fast path, duty 0.10–0.50 of wall-clock in the passing rows. CLAUDE.md records that Thor throughput saturates at batch 8 (flat 12.3–14.1 windows/s over a 6× batch range) ⇒ the ViT/LLM overlap buys little and a 100–200 ms prefill can delay fast-path kernels unless they run on a higher-priority stream. **Fast-path p95 under a concurrent refresh has never been measured for anything here.**
2. **Precision drift:** heads trained on BF16 features and run on NVFP4/FP8 features — magnitude `U`.

### 5.3 Ranking (replaces RA's implicitly size-capped R1..R3)

| rank | role | model | passes A? | passes B? | latency evidence | why / risks |
|---|---|---|---|---|---|---|
| **1** | **primary frozen backbone** | **Cosmos3-Edge reasoner** (2.43 B; OpenMDW-1.1) | 1c×1f (NVFP4 `GG`, FP16 `g.`); not at ≥ 3c×1f | **up to 3c×3f at FP16 (180 ms, duty 0.39) and NVFP4 (109 ms)**; 7c×1f; 7c×3f only NVFP4 raw | `E` from Qwen3-VL-2B + SO400M ViT; **NVIDIA-measured only eager BF16 on Thor (0.19 s / 911 tok)** | (+) the only model NVIDIA built *and benchmarked* for Thor/Orin (5 SKUs), in `NVIDIA/Cosmos`; TRT-Edge-LLM native export incl. video; permissive licence; newest physical-AI mid-training (yaml `…midtrain`); 160 tok/frame now `P`. (−) **no TRT row; NVFP4/FP8 untested**; framework port has no KV path; hidden states are block-granular; driving content of its training unknown; **not Qwen ⇒ unlike RA's R1/R2 there is no same-architecture driving-tuned twin** |
| **2** | **measured-latency twin / fallback / scale-ladder rung** | **Cosmos-Reason2-2B** (≡ Qwen3-VL-2B; NVIDIA Open Model License) | 1c×1f `GG`; ≤ 3c×1f `g.` | up to 7c×1f; 3c×3f `GG` (85 ms); 7c×3f `g.` | **TRT rows by identity (`INH`)** — the only 2 B-class with real Thor anchors | (+) explicit TRT-Edge-LLM checkpoint ID; same family as the 8 B (rung of a {2, 8, 32 B} Qwen3-VL ladder — **n = 3 ⇒ no exponent, matched-size ratios only**, CLAUDE.md); (−) superseded/maintenance-only repo; custom licence; **identity of Cosmos-Reason2's own `config.json` to Qwen3-VL-2B is `U`**; `min_pixels` trap (§1.3) |
| **3** | **paired driving-aware arm** | **Alpamayo-1.5's VLM (≈ 8 B)** vs its general parent **Cosmos-Reason2-8B** (and Cosmos 3 Nano reasoner as a Cosmos-3-generation general twin) — all Qwen3-VL-8B architecture, differ by driving RL/SFT and Cosmos-3 mid-training | only 1c×1f (39 ms, raw) at NVFP4 | **NVFP4: up to 3c×3f (167 ms, duty 0.36)**; **FP16 (the only Alpamayo TRT path): ≤ 3c×1f raw (232 ms, duty 0.49)** | `INH` identity ⇒ `E`; **FP16 8 B has no measured basis at all** | (+) the cleanest "does driving knowledge help a frozen interface" pair NVIDIA offers (same weights lineage — recipes README: Alpamayo-1.5 = "RL post-trained on the Cosmos-Reason2 backbone"); (−) **contaminated on our val** (§5.4); Alpamayo's driving state lives in post-CoC KV (§4.1); 8 B tier = 4.9× Edge's LLM body (3.6× total params); **no ≤ 4 B driving-tuned NVIDIA VLM exists** ⇒ the pair is 8 B vs 8 B, *not* size-matched to rank 1 (H-HC16 must state this) |
| 4 | reference / alt general 8 B | **Cosmos 3 Nano reasoner** | as rank 3 | as rank 3 | as rank 3 | the Cosmos-3 generation of the 8 B tier; **not TRT-Edge-LLM-listed**; NIM black box (no tap); needs HF in-process for taps |
| 5 | legacy | Cosmos-Reason1-7B | 1c×1f raw | ≤ 3c×1f `GG`; 3c×3f raw | rows by `R` identity | older Qwen2.5-VL arch, 207 tok/frame, not TRT-listed by ID; only worth it as the AR1-lineage ancestor |
| ✗ | teacher / auto-labeller only | Cosmos-Reason2-32B, **Cosmos 3 Super**, **Alpamayo-2-Super** (32 B VLM, 6 cams × 4 frames = 3,840–4,608 vision tokens ⇒ ≈ 1.0–1.3 s at NVFP4, `E`) | never | 1c×1f only | extrapolated | NIM itself says Super is "not recommended" on Thor; use offline for labels / score distillation |
| — | *(RA's R1/R2, outside ruling (b))* | Qwen3.5-4B, Qwen-Drive-1.0-4B VLM | see RA §5.3 (measured rows: 3c×1f 61 ms) | | | keep as **external references**, not as the Cosmos arm; **RA's Table remains valid for them** |

**If the PI wants the strict Reading A at 3 cameras:** nothing here passes — the ViT alone is 3 × (6.9–9.5) = 21–29 ms and the ≤ 2.4 B LLM 15 ms. Options (not evaluated): 1 camera per refresh with camera rotation (paper §elastic), 2×2-pooled ViT input, DART token pruning (TRT-Edge-LLM 0.10.1), or accept Reading B.

### 5.4 Contamination notes (driving-aware arms and the PhysicalAI-AV overlap)

* **What RA measured, restated precisely:** 13/400 of an *older* clean-val split and 76/2,400 parity-train clips are in `0417_5k_train_set_for_calibration_25.10.parquet` — **NVIDIA's 5,000-clip sample of the PhysicalAI-AV `train` split, published as the calibration set for quantising Alpamayo-1.5** (`alpamayo-recipes/recipes/alpamayo1_5_quant/`, default `--calib_parquet`). That proves **membership in the public train split NVIDIA used for calibration**; that Alpamayo *trained* on them is an **inference** (`H`), not a stated fact. RA wrote "NVIDIA's training list"; **the honest wording is "NVIDIA's train-split calibration sample"**. (`INH` from RA §6.2; not re-run — the v7 147-eval-clip identifiers live in a private HF dataset, unreachable here.)
* **Alpamayo family:** trained on ">1 B images from 80,000 hours" (R1) / "110,000+ hours" (2-Super) versus the **public** PhysicalAI-AV of "1,700+ hours, 306 K clips" (`P` recipes README) ⇒ the training pool is far larger than the public set; **overlap with any of our clips cannot be excluded or bounded** (`U`). Treat every Alpamayo-derived arm as **contaminated on any PhysicalAI-derived val** until a memorisation probe (log-likelihood gap, RA E-A2) says otherwise. The PI-visible consequence: **"driving-aware > general" on our v7 eval is inadmissible without decontamination (O3 of RA's pre-registration).**
* **Cosmos 3 (Edge/Nano/Super) and Cosmos-Reason2:** training data **undisclosed in every file read**; the prompt guide shows AV-planner prompts, and **PhysicalAI-AV is NVIDIA's own dataset** (licence "NVIDIA AV Dataset License", recipes README) ⇒ inclusion in Cosmos 3 post-training is *plausible* (`H`), not established. **Do not assume the Cosmos arms are the clean control** (RA's Qwen3.5 was the cleanest control; it is now out of source). Run the same memorisation probe on Edge before calling it "general".
* **The clean control within the ruling:** none is provably clean. The least-driving-specific is Cosmos-Reason2-2B/8B (physical common-sense SFT+RL, no AV-specific claim in the README) — `H`.

---

## 6. What we could not verify — and the exact measurement that closes each gap (feeds the on-Thor benchmark)

| # | gap (`U`) | consequence | closing command / measurement |
|---|---|---|---|
| G1 | **No TRT (or any optimised) latency row for Cosmos3-Edge on Thor** | every Edge sub-50 ms figure is extrapolated | Thor: `tensorrt-edgellm-export nvidia/Cosmos3-Edge $ONNX/reasoning --task reasoning` (x86) → on Thor `llm_build --onnxDir $ONNX/reasoning/llm --engineDir $ENG/reasoning --maxInputLen 3424 --maxKVCacheCapacity 4096` and `visual_build --onnxDir $ONNX/reasoning/visual --engineDir $ENG/reasoning --minImageTokens 8 --maxImageTokens 16384 --maxImageTokensPerImage 2048`; then `llm_bench --mode prefill --batchSize 1 --inputLen {224,544,1184,1504,3424} --warmup 3 --iterations 50 --profile` and `llm_bench --mode visual --imageSize 256x640 …` (×cameras). Record p50/**p95** (NVIDIA publishes means). |
| G2 | **NVFP4/FP8 of the Edge reasoner untested by NVIDIA** (CI = fp16) | NVFP4 rows are hypothetical | `tensorrt-edgellm-quantize llm --model_dir nvidia/Cosmos3-Edge --output_dir … --quantization nvfp4 --lm_head_quantization nvfp4 [--visual_quantization fp8]` on the x86 host (flag names from `features/quantization.md`, whose examples are Qwen3-VL-2B only — **Cosmos3-Edge is not documented as quantisable**; expect to hit a relu²/`Cosmos3ReasonerCausalLM` gap) with a driving calibration set → export → rebuild → repeat G1; **also compute feature drift**: cosine(BF16 tap, NVFP4 tap) per depth and head-accuracy retention on the frozen-feature panel |
| G3 | **Fast-path p95 under concurrent refresh** (contention) | criterion's 25 ms clause is untested for every backbone | run the 27 M network (CUDA-graphed, 10 Hz) while the refresh loop runs at 2 Hz on a second stream; report fast-path p95/p99 with and without refresh; sweep refresh priority; `OMP_NUM_THREADS=6` (CLAUDE.md trap) |
| G4 | 3-depth tap engine (`--eagle-base` reuse on `cosmos3_edge`; block-vs-layer indices) | tap design unproven on TRT | export with `--eagle-base` (or subclass `Cosmos3ReasonerCausalLM` with `emit_hidden_states=True` and layer ids 28/42/56 blocks = 14/21/28 pairs); assert engine hidden-state tensor equals HF `output_hidden_states` at those indices (cosine ≥ 0.999 FP16) |
| G5 | Identity Cosmos-Reason2 ≡ Qwen3-VL and Alpamayo-1.5 VLM ≡ Qwen3-VL-8B (HF `config.json` unread) | the TRT anchors are borrowed by identity | on any host with HF access: `hf download nvidia/Cosmos-Reason2-8B config.json preprocessor_config.json` and diff against `cosmos-framework/.../Qwen3-VL-8B-Instruct.json` (`hidden_size 4096, num_hidden_layers 36, intermediate_size 12288, vocab 151936`); same for `nvidia/Alpamayo-1.5-10B` (also read `expert_cfg`) and `nvidia/Alpamayo-R1-10B` (Qwen2.5-VL vs Qwen3-VL lineage) |
| G6 | Thor memory truly used | the NVIDIA "GPU Mem (MB)" column is not a footprint; `df`/`free`/`tegrastats` lie on Thor (CLAUDE.md) | in-process only: HF path `torch.cuda.max_memory_allocated()` after prefill; TRT path: serialized engine size + `ICudaEngine::getDeviceMemorySizeV2()` |
| G7 | Licences: HF model-card licence fields (Cosmos 3, Reason2, Alpamayo); NVIDIA Open Model License terms | cannot ship on an unread licence | read the cards' YAML `license:` and the NVIDIA Open Model License text (`nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license`) |
| G8 | Contamination: exact overlap of the v7 147-eval clips (and the 40 canonical val) with NVIDIA's train-split lists; Cosmos 3 training data | any driving-aware gain is uninterpretable | recover the UUIDs (private HF dataset `tanitad-v7-training-corpus`) and rerun RA Appendix B against the three parquet lists in `alpamayo-recipes`; read the Cosmos 3 technical report §data (`research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf`) from an unblocked host |
| G9 | Video-mode token accounting at 256×640 (2× temporal compression) and the `min_pixels` floor | ViT/LLM token counts ±1.8× | `processor(... images=[256×640 array])` with the shipped `preprocessor_config.json`; count `image_grid_thw` products / 4 |
| G10 | Qwen2.5-VL-7B config (`R`) for Cosmos-Reason1 | 207 tok/frame and 6.53 B body are recalled | same as G5 for `nvidia/Cosmos-Reason1-7B` |
| G11 | Alpamayo VLM FP16 8 B on Thor: no measured basis | the FP16 8 B rows are FLOP-scaled from a 57 TFLOPS guess | build the `alpamayo_r1` LLM+visual engines per `examples/vla/alpamayo.md` (`--maxInputLen 3424`), time the LLM engine prefill only |
| G12 | TRT-Edge-LLM 0.11.0 performance (Thor-optimised NVFP4-A16 kernels) | 0.10.0 anchors may be pessimistic | rerun the NVIDIA anchor rows (`llm_bench --mode prefill --inputLen 2048` on Qwen3-VL-2B/8B) on the deployed JetPack/TRT-Edge-LLM version — this calibrates *our* Thor against NVIDIA's before anything else is trusted |

---

## 7. Implications (for the HiCAP design owner; nothing here edits another file)

1. **Say "NVIDIA/Cosmos" precisely:** it is the Cosmos 3 repo; the Cosmos-Reason2 line is the *predecessor* (still TRT-supported and the only Cosmos-family source of real Thor TRT anchors); Alpamayo is `NVlabs`.
2. **The sub-300 M cap is gone, but the latency criterion is not automatic:** at the corpus reality (1 front camera, 1 frame) even the 8 B tier is 39 ms (NVFP4); the binding constraints are **(i) the strict ≤ 50 ms reading at ≥ 3 cameras (nothing passes), (ii) contention, (iii) NVFP4 drift, (iv) memory only for ≥ 32 B.**
3. **Test order (cheapest discriminating first):** G12 (calibrate our Thor against NVIDIA's rows) → G1/G4 on Cosmos-Reason2-2B (real anchor) and Edge FP16 (CI-tested) → G3 (contention) → NVFP4 (G2). Only then spend A40 days on the frozen-feature panel (RA E-A0) with Edge / Cosmos-Reason2-2B / the 8 B pair.
4. **Paper-impact list (owner of `Paper/HiCAP/*`; not edited here):** (a) `sec_eff.tex` "Cosmos3-Edge has only eager-BF16 rows" → add: TRT-Edge-LLM has a documented Cosmos3-Edge reasoner path, CI-tested FP16, native video in 0.11.0, still no latency row; (b) "Edge patch size unverified / 160 tokens assumed" → `P` (16; 160); (c) "everything ≥ 6.7× the sub-300 M target … HiCAP-S mandatory" → superseded by ruling (c); HiCAP-S is a fallback; (d) the backbone shortlist table (R1 Qwen3.5-4B, R2 Qwen-Drive) predates ruling (b) — the Cosmos-family ranking is §5.3 here; (e) "13/400 … in NVIDIA's training list" → "train-split calibration sample" (§5.4); (f) the cache-size sentence should state its width `d`: 148 GB is the d = 2,048 (Edge / 2 B) figure, 297 GB for the 8 B tier; (g) Alpamayo-R1 licence → OpenMDW-1.1 per its README.
5. **Do not compare** any latency here with a TanitAD number: none of them is ours; the only `M(ours)` items are arithmetic on published configs and the URL resolution.

---

## 8. Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-hicap/RE_cosmos_backbones.md` (this file, incl. Appendix A script + output) | **repo working tree, staged** (verify: `git ls-files --cached`) | in the repo; **not committed** (agent rule 1) |
| estimator script `re_cosmos_cost.py` + output `re_cosmos_cost_out.txt` | **embedded verbatim as Appendix A**; working copies `…/scratchpad/re/` (session scratch, disposable) | not stranded |
| shallow clones (cosmos `3e3c6d6`, cosmos-framework `cf5d68c`, cosmos-reason1 `b84a3fa`, cosmos-reason2 `a3b4a1d`, TensorRT-Edge-LLM `95515c2`, alpamayo `11a0e01`, alpamayo1.5 `36aeb4c`, alpamayo2 `5e7975f`, alpamayo-recipes `ae5bc10`) | session scratchpad only; reproducible at the commits in §0.2 | **exists in one place; disposable by design** |

**Escalations (headline, not a README plea):** (1) **PI decision — Cosmos 3 vs Cosmos-Reason2 as "the Cosmos arm":** the URL in ruling (b) is the Cosmos 3 repo; the recommended primary (Edge) has **no optimised latency row** (G1/G2). (2) **A Thor session is needed** for G12 → G1 → G3 before any latency claim in the paper is upgraded from `E`. (3) **Contamination screen before any driving-aware arm** (G8). (4) Integration: nothing to merge — this file is input to the HiCAP design synthesis and to the on-Thor benchmark author. **Notes for that author (`hicap_backbone_latency_bench.py`, untracked, not mine):** (i) its "TRT-Edge-LLM documents no API to return intermediate prefill hidden states" is answered in §4.4 with code pointers (`emit_hidden_states`, `--eagle-base` = 3 layers, `CausalLM` hidden-state ONNX outputs) — all untested; (ii) its `cosmos3-edge` loader is `UNVERIFIED` — §4.3 gives the documented class and the block-granular hidden-state tuple (57 entries, taps 28/42/56) which its Qwen3-VL `TAP_CONVENTION` (L+1 layers) does **not** match; (iii) its `--trust-remote-code` for Edge is probably stale (§4.3); (iv) it independently caught the 286-token `min_pixels` trap — agreed (§1.3).

---

## Appendix A — estimator (`M(ours)` arithmetic on `P` anchors; `python3 re_cosmos_cost.py`, numpy-free; output follows)

```python
# Appendix A -- HiCAP RE (Cosmos backbones) estimator. numpy-free, runs as written (python3 re_cosmos_cost.py).
# Method = RA_vlm_backbones.md section 5 (scale from NVIDIA-MEASURED Thor rows), extended with a precision ladder.
# Every input is tagged. P = PUBLISHED-PRIMARY (repo/table read this session), E = ESTIMATED, R = RECALLED, A = ASSUMPTION.

# ---------- 1. parameter arithmetic from the configs read in cosmos-framework@cf5d68c (P) ----------
def dense_body(h, L, nh, nkv, hd, inter, gated=True):
    attn = h * nh * hd + 2 * h * nkv * hd + nh * hd * h            # q, k, v, o
    mlp = (3 if gated else 2) * h * inter                          # SwiGLU (3 mats) or relu^2 non-gated (2 mats)
    return L * (attn + mlp)

def vit(depth, w, inter, merge_in, merge_mid, out, n_merger):
    layer = 4 * w * w + 2 * w * inter                              # qkv+o, fc1+fc2 (biases ignored)
    merger = merge_in * merge_mid + merge_mid * out
    return depth * layer + n_merger * merger

CFG = {  # name: (body_B, embed_B (untied x2 unless stated), vit_B, hidden, layers, kv_heads, head_dim)
 "Qwen3-VL-2B  (= Cosmos-Reason2-2B arch)": (dense_body(2048, 28, 16, 8, 128, 6144) / 1e9, 151936 * 2048 / 1e9, vit(24, 1024, 4096, 4096, 4096, 2048, 4) / 1e9, 2048, 28, 8, 128),
 "Qwen3-VL-8B  (= Cosmos-Reason2-8B / Cosmos3-Nano reasoner / Alpamayo VLM arch)": (dense_body(4096, 36, 32, 8, 128, 12288) / 1e9, 2 * 151936 * 4096 / 1e9, vit(27, 1152, 4304, 4608, 4608, 4096, 4) / 1e9, 4096, 36, 8, 128),
 "Qwen3-VL-32B (= Cosmos-Reason2-32B / Cosmos3-Super reasoner arch)": (dense_body(5120, 64, 64, 8, 128, 25600) / 1e9, 2 * 151936 * 5120 / 1e9, vit(27, 1152, 4304, 4608, 4608, 5120, 4) / 1e9, 5120, 64, 8, 128),
 "Cosmos3-Edge reasoner (Nemotron-dense 2B + SigLIP2 + merger)": (dense_body(2048, 28, 16, 8, 128, 9216, gated=False) / 1e9, 2 * 131072 * 2048 / 1e9, vit(27, 1152, 4304, 4608, 11520, 2048, 1) / 1e9, 2048, 28, 8, 128),
}
# Cosmos-Reason1-7B = Qwen2.5-VL-7B (R): h3584 L28 heads28/4 inter18944, ViT ~0.675B, vocab 152064 untied
CFG["Qwen2.5-VL-7B (= Cosmos-Reason1-7B arch; config RECALLED)"] = (dense_body(3584, 28, 28, 4, 128, 18944) / 1e9, 2 * 152064 * 3584 / 1e9, 0.675, 3584, 28, 4, 128)

print("== params (B): body / embed+head / ViT+merger / total ; hidden, layers, kv_heads")
for k, (b, e, v, h, L, kv, hd) in CFG.items():
    print(f"{k:78s} {b:6.2f} {e:5.2f} {v:5.2f} {b+e+v:6.2f} | d={h} L={L} kv={kv}")

# ---------- 2. weight memory (GB, decimal) per precision ----------
NVFP4_B, FP8_B, BF16_B = 0.5 + 1 / 16, 1.0, 2.0                     # bytes/param on quantised linears (NVFP4 = 4 bit + FP8 scale per 16) (E)
def weights_gb(b, e, v, prec):
    per = {"NVFP4": NVFP4_B, "FP8": FP8_B, "BF16": BF16_B}[prec]
    return b * per + e * 2.0 + v * 2.0                              # embeddings/lm_head + ViT kept FP16 (upper bound; prefill-only build can drop lm_head)
print("== weights GB (body quantised; embed+head and ViT FP16)")
for k, (b, e, v, h, L, kv, hd) in CFG.items():
    print(f"{k[:40]:40s} NVFP4 {weights_gb(b,e,v,'NVFP4'):5.1f}  FP8 {weights_gb(b,e,v,'FP8'):5.1f}  BF16 {weights_gb(b,e,v,'BF16'):5.1f}")

# ---------- 3. latency: anchors = NVIDIA TRT-Edge-LLM 0.10.0 Thor rows (P), batch 1, LLM NVFP4, ViT FP16 ----------
# name: (body_B, pf_short_ms, n_short, pf2048_ms, vit_ms_per_1060_patches)
ANC = {"Qwen3-VL-2B": (1.41, 12.7, 292, 28.0, 11.4), "Qwen3-VL-4B": (dense_body(2560, 36, 32, 8, 128, 9728) / 1e9, 22.3, 292, 62.2, 11.6),
       "Qwen3-VL-8B": (6.95, 32.1, 292, 104.5, 15.7), "Qwen2.5-VL-7B": (CFG["Qwen2.5-VL-7B (= Cosmos-Reason1-7B arch; config RECALLED)"][0], 25.0, 376, 83.1, 23.4)}
def fit(name):
    b, t1, n1, t2, _ = ANC[name]; s = (t2 - t1) / (2048 - n1); return t1 - s * n1, s  # t0 (ms), s (ms/token)
print("== linear prefill fits t = t0 + s*N (NVFP4, Thor, P rows): t0, s, s per B-body")
S_PER_B = []
for n in ANC:
    t0, s = fit(n); S_PER_B.append(s / ANC[n][0]); print(f"{n:16s} t0 {t0:5.1f} ms  s {s:.5f} ms/tok  s/B {s/ANC[n][0]:.5f}")
S_B = sum(S_PER_B[:3]) / 3          # Qwen3-family mean, in-sample
print(f"s per B-body (Qwen3-VL 2/4/8B mean) = {S_B:.5f} ms/tok/B ; Qwen2.5-VL-7B out-of-sample s/B = {S_PER_B[3]:.5f} (ratio {S_PER_B[3]/S_B:.2f})")
BW_EFF = 6.95 * NVFP4_B / fit("Qwen3-VL-8B")[0]                    # GB per ms => x1000 GB/s ; calibrated on the 8B row (t0 treated as weight-read floor)
print(f"BW_eff calibrated on Qwen3-VL-8B t0: {BW_EFF*1000:.0f} GB/s (73% of 273 GB/s published peak, brief)")
# out-of-sample check of the no-row model on Qwen2.5-VL-7B
b7 = ANC["Qwen2.5-VL-7B"][0]; t0p = b7 * NVFP4_B / BW_EFF; sp = S_B * b7
print(f"no-row model on Qwen2.5-VL-7B: pred pf@376 {t0p+sp*376:.1f} (meas 25.0) pf@2048 {t0p+sp*2048:.1f} (meas 83.1) -> over-predicts {((t0p+sp*376)/25.0-1)*100:.0f}% / {((t0p+sp*2048)/83.1-1)*100:.0f}%")

# precision ladder (E, low confidence): FP8 slope x1.3 (Gemma-4-31B FP8/NVFP4 pf@2048 = 1036.4/786.9 = 1.32, P), FP16 TRT ~57 TFLOPS (mid of 43..70; 43 = AWQ FP16-math row, 70 ~ Gemma-4-12B FP16 pf@2048 row w/ assumed body), eager BF16 = Edge row
S_FP8 = S_B * 1.3
S_FP16 = 2.0 / 57e3 * 1e3           # ms per token per B-body param at 57 TFLOPS  (2e9 FLOP / 57e12 FLOP/s = 3.51e-5 s)
S_EAGER = (200.0 - 1.41 * 2 / BW_EFF / 1000 * 1000) / 1705 / 1.41   # from Cosmos3-Edge eager text row: 1705 tok in 0.20 s (P)
def llm_ms(body, N, prec, row=None):
    if row and prec == "NVFP4":
        t0, s = fit(row); return t0 + s * N
    t0_nv = max(body * NVFP4_B / BW_EFF, 0.0) if not row else fit(row)[0]
    s_nv = S_B * body if not row else fit(row)[1]
    if prec == "NVFP4": return t0_nv + s_nv * N
    if prec == "FP8":  return t0_nv + (body * FP8_B - body * NVFP4_B) / BW_EFF + S_FP8 * body * N
    if prec == "FP16": return t0_nv + (body * 2 - body * NVFP4_B) / BW_EFF + S_FP16 * body * N
    if prec == "EAGER": return body * 2 / BW_EFF + S_EAGER * body * N
VIT_S = {"2B": 11.4 / 1060, "SO400M": 15.7 / 1060}                 # ms per patch, TRT FP16 (P rows)
VIT_EAGER = (190.0 - 911 / 8717 * 1000) / (911 * 4)                # ms per patch, eager BF16 (E: image row minus LLM share; assumes 911 tok ~ 3644 patches)
PATCH, TOK, TEXT = 640, 160, 64                                     # 256x640 frame: 16x40 patches -> 8x20 = 160 merged tokens (P: patch 16 + 2x2 merge; 32 px/token); +64 prompt/ego/query tokens (A)
MODELS = {  # key: (label, body_B, row-or-None, vit_key)
 "CR2-2B":  ("Cosmos-Reason2-2B  (= Qwen3-VL-2B arch)", 1.41, "Qwen3-VL-2B", "2B"),
 "Edge":    ("Cosmos3-Edge reasoner (relu2, SigLIP2-SO400M)", 1.41, "Qwen3-VL-2B", "SO400M"),   # body FLOPs identical to Qwen3-VL-2B (50.3M/layer x 28)
 "CR2-8B":  ("Cosmos-Reason2-8B / Cosmos3-Nano reasoner / Alpamayo-1.5 VLM (= Qwen3-VL-8B arch)", 6.95, "Qwen3-VL-8B", "SO400M"),
 "CR1-7B":  ("Cosmos-Reason1-7B (= Qwen2.5-VL-7B arch)", CFG["Qwen2.5-VL-7B (= Cosmos-Reason1-7B arch; config RECALLED)"][0], "Qwen2.5-VL-7B", None),
 "CR2-32B": ("Cosmos-Reason2-32B / Cosmos3-Super reasoner (= Qwen3-VL-32B arch)", CFG["Qwen3-VL-32B (= Cosmos-Reason2-32B / Cosmos3-Super reasoner arch)"][0], None, "SO400M"),
}
SC = [("1cam x1f", 1, 1), ("1cam x3f", 1, 3), ("3cam x1f", 3, 1), ("3cam x3f", 3, 3), ("7cam x1f", 7, 1), ("7cam x3f", 7, 3)]
def tick(mk, cams, frames, prec):
    lab, body, row, vk = MODELS[mk]; N = cams * frames * TOK + TEXT; nfr = cams * frames
    if mk == "CR1-7B":   # Qwen2.5-VL: patch 14 -> 9x23 = 207 tok/frame (E, processor rounding to 28 px); ViT row: 23.4 ms / (349 tok*4 = 1396 patches)
        N = cams * frames * 207 + TEXT; v = 23.4 / 1396 * (207 * 4) * nfr
    elif prec == "EAGER": v = VIT_EAGER * PATCH * nfr
    else: v = VIT_S[vk] * PATCH * nfr
    return N, v, llm_ms(body, N, prec, row if prec != "EAGER" else None)
print("== per-refresh latency, ms (p50-style; ViT + LLM prefill only; excludes preprocessing, engine hand-off, pooling). N = cams*frames*160 + 64")
for prec in ("NVFP4", "FP8", "FP16", "EAGER"):
    print(f"-- {prec}")
    for mk in MODELS:
        if prec == "EAGER" and mk not in ("Edge", "CR2-8B"): continue
        row = [f"{mk:8s}"]
        for nm, c, f in SC:
            N, v, l = tick(mk, c, f, prec); row.append(f"{nm}: {v+l:6.0f} (N={N})")
        print("  " + " | ".join(row))
# ---------- 4. feature cache width (paper sec_eff.tex: 188,640 ticks @2 Hz over 26.2 h; 128 tokens x d x 3 depths x 1 byte) ----------
TICKS = 188640
print("== feature-cache size, GB (128 tok x d x 3 depths x bytes x 188,640 ticks)")
for k, (b, e, v, h, L, kv, hd) in CFG.items():
    print(f"{k[:44]:44s} d={h:5d} int8 {128*h*3*TICKS/1e9:6.0f}  fp16 {128*h*3*2*TICKS/1e9:6.0f}  taps L/2,3L/4,L = {L//2},{(3*L)//4},{L}")
# ---------- 5. Thor memory budget (A): 128 GB unified; NIM default host reserve 16 GiB (P: cosmos NIM support-matrix); other tenants 40 GB (A); TRT workspace/act 4 GB (A)
print("== Thor 128 GB headroom after backbone weights, 16 GiB host reserve (P), 40 GB other tenants (A), 4 GB engine workspace (A)")
for k, (b, e, v, h, L, kv, hd) in CFG.items():
    print(f"{k[:40]:40s} " + "  ".join(f"{p}: {128-weights_gb(b,e,v,p)-17.2-40-4:5.1f}" for p in ("NVFP4", "FP8", "BF16")))

# ---------- 6. pass/fail against the pre-registered latency criterion (brief 2026-09-29) ----------
# Reading A: refresh p95 <= 50 ms.  Reading B (async 2 Hz): refresh <= 250 ms (= duty 0.5 at 2 Hz) and token age <= 500 ms + refresh (holds by construction if refresh < 500 ms).
# NVIDIA publishes means, not p95 -> "raw" = model-only estimate vs threshold; "guarded" = 1.3*T_r + 15 ms overhead vs threshold (1.3 = HYPOTHESIS p95/p50; 15 ms = ASSUMPTION from hicap_latency_budget.py).
def verdict(t, thr, over=15.0, k=1.3):
    raw = t <= thr; grd = k * t + over <= thr
    return "GG" if grd else ("g." if raw else "..")
print("== verdicts: A(<=50 ms) / B(<=250 ms) ; GG = passes guarded, g. = passes raw p50 only, .. = fails ; duty = 2 Hz x (T_r+15 ms)")
for prec in ("NVFP4", "FP8", "FP16", "EAGER"):
    print(f"-- {prec}")
    for mk in MODELS:
        if prec == "EAGER" and mk not in ("Edge", "CR2-8B"): continue
        cells = []
        for nm, c, f in SC:
            N, v, l = tick(mk, c, f, prec); t = v + l
            cells.append(f"{nm}: A {verdict(t,50)} B {verdict(t,250)} duty {2*(t+15)/1000:.2f}")
        print(f"  {mk:8s} " + " | ".join(cells))
```

**Output (run 2026-09-29; identical to the tables above):**

```
== params (B): body / embed+head / ViT+merger / total ; hidden, layers, kv_heads
Qwen3-VL-2B  (= Cosmos-Reason2-2B arch)                                          1.41  0.31  0.40   2.12 | d=2048 L=28 kv=8
Qwen3-VL-8B  (= Cosmos-Reason2-8B / Cosmos3-Nano reasoner / Alpamayo VLM arch)   6.95  1.24  0.57   8.76 | d=4096 L=36 kv=8
Qwen3-VL-32B (= Cosmos-Reason2-32B / Cosmos3-Super reasoner arch)               31.21  1.56  0.59  33.35 | d=5120 L=64 kv=8
Cosmos3-Edge reasoner (Nemotron-dense 2B + SigLIP2 + merger)                     1.41  0.54  0.49   2.43 | d=2048 L=28 kv=8
Qwen2.5-VL-7B (= Cosmos-Reason1-7B arch; config RECALLED)                        6.53  1.09  0.68   8.29 | d=3584 L=28 kv=4
== weights GB (body quantised; embed+head and ViT FP16)
Qwen3-VL-2B  (= Cosmos-Reason2-2B arch)  NVFP4   2.2  FP8   2.8  BF16   4.2
Qwen3-VL-8B  (= Cosmos-Reason2-8B / Cosm NVFP4   7.5  FP8  10.6  BF16  17.5
Qwen3-VL-32B (= Cosmos-Reason2-32B / Cos NVFP4  21.8  FP8  35.5  BF16  66.7
Cosmos3-Edge reasoner (Nemotron-dense 2B NVFP4   2.8  FP8   3.5  BF16   4.9
Qwen2.5-VL-7B (= Cosmos-Reason1-7B arch; NVFP4   7.2  FP8  10.1  BF16  16.6
== linear prefill fits t = t0 + s*N (NVFP4, Thor, P rows): t0, s, s per B-body
Qwen3-VL-2B      t0  10.2 ms  s 0.00871 ms/tok  s/B 0.00618
Qwen3-VL-4B      t0  15.7 ms  s 0.02272 ms/tok  s/B 0.00625
Qwen3-VL-8B      t0  20.1 ms  s 0.04123 ms/tok  s/B 0.00593
Qwen2.5-VL-7B    t0  11.9 ms  s 0.03475 ms/tok  s/B 0.00533
s per B-body (Qwen3-VL 2/4/8B mean) = 0.00612 ms/tok/B ; Qwen2.5-VL-7B out-of-sample s/B = 0.00533 (ratio 0.87)
BW_eff calibrated on Qwen3-VL-8B t0: 195 GB/s (73% of 273 GB/s published peak, brief)
no-row model on Qwen2.5-VL-7B: pred pf@376 33.9 (meas 25.0) pf@2048 100.6 (meas 83.1) -> over-predicts 35% / 21%
== per-refresh latency, ms (p50-style; ViT + LLM prefill only; excludes preprocessing, engine hand-off, pooling). N = cams*frames*160 + 64
-- NVFP4
  CR2-2B   | 1cam x1f:     19 (N=224) | 1cam x3f:     36 (N=544) | 3cam x1f:     36 (N=544) | 3cam x3f:     85 (N=1504) | 7cam x1f:     69 (N=1184) | 7cam x3f:    185 (N=3424)
  Edge     | 1cam x1f:     22 (N=224) | 1cam x3f:     43 (N=544) | 3cam x1f:     43 (N=544) | 3cam x3f:    109 (N=1504) | 7cam x1f:     87 (N=1184) | 7cam x3f:    239 (N=3424)
  CR2-8B   | 1cam x1f:     39 (N=224) | 1cam x3f:     71 (N=544) | 3cam x1f:     71 (N=544) | 3cam x3f:    167 (N=1504) | 7cam x1f:    135 (N=1184) | 7cam x3f:    360 (N=3424)
  CR1-7B   | 1cam x1f:     35 (N=271) | 1cam x3f:     77 (N=685) | 3cam x1f:     77 (N=685) | 3cam x3f:    204 (N=1927) | 7cam x1f:    162 (N=1513) | 7cam x3f:    457 (N=4411)
  CR2-32B  | 1cam x1f:    142 (N=224) | 1cam x3f:    222 (N=544) | 3cam x1f:    222 (N=544) | 3cam x3f:    463 (N=1504) | 7cam x1f:    383 (N=1184) | 7cam x3f:    943 (N=3424)
-- FP8
  CR2-2B   | 1cam x1f:     23 (N=224) | 1cam x3f:     40 (N=544) | 3cam x1f:     40 (N=544) | 3cam x3f:     92 (N=1504) | 7cam x1f:     75 (N=1184) | 7cam x3f:    196 (N=3424)
  Edge     | 1cam x1f:     25 (N=224) | 1cam x3f:     48 (N=544) | 3cam x1f:     48 (N=544) | 3cam x3f:    116 (N=1504) | 7cam x1f:     93 (N=1184) | 7cam x3f:    251 (N=3424)
  CR2-8B   | 1cam x1f:     58 (N=224) | 1cam x3f:     94 (N=544) | 3cam x1f:     94 (N=544) | 3cam x3f:    204 (N=1504) | 7cam x1f:    168 (N=1184) | 7cam x3f:    424 (N=3424)
  CR1-7B   | 1cam x1f:     55 (N=271) | 1cam x3f:    104 (N=685) | 3cam x1f:    104 (N=685) | 3cam x3f:    252 (N=1927) | 7cam x1f:    202 (N=1513) | 7cam x3f:    547 (N=4411)
  CR2-32B  | 1cam x1f:    225 (N=224) | 1cam x3f:    324 (N=544) | 3cam x1f:    324 (N=544) | 3cam x3f:    619 (N=1504) | 7cam x1f:    521 (N=1184) | 7cam x3f:   1210 (N=3424)
-- FP16
  CR2-2B   | 1cam x1f:     39 (N=224) | 1cam x3f:     68 (N=544) | 3cam x1f:     68 (N=544) | 3cam x3f:    157 (N=1504) | 7cam x1f:    127 (N=1184) | 7cam x3f:    334 (N=3424)
  Edge     | 1cam x1f:     41 (N=224) | 1cam x3f:     76 (N=544) | 3cam x1f:     76 (N=544) | 3cam x3f:    180 (N=1504) | 7cam x1f:    145 (N=1184) | 7cam x3f:    389 (N=3424)
  CR2-8B   | 1cam x1f:    135 (N=224) | 1cam x3f:    232 (N=544) | 3cam x1f:    232 (N=544) | 3cam x3f:    523 (N=1504) | 7cam x1f:    426 (N=1184) | 7cam x3f:   1105 (N=3424)
  CR1-7B   | 1cam x1f:    136 (N=271) | 1cam x3f:    259 (N=685) | 3cam x1f:    259 (N=685) | 3cam x3f:    626 (N=1927) | 7cam x1f:    504 (N=1513) | 7cam x3f:   1361 (N=4411)
  CR2-32B  | 1cam x1f:    575 (N=224) | 1cam x3f:    944 (N=544) | 3cam x1f:    944 (N=544) | 3cam x3f:   2052 (N=1504) | 7cam x1f:   1683 (N=1184) | 7cam x3f:   4268 (N=3424)
-- EAGER
  Edge     | 1cam x1f:     54 (N=224) | 1cam x3f:    119 (N=544) | 3cam x1f:    119 (N=544) | 3cam x3f:    313 (N=1504) | 7cam x1f:    248 (N=1184) | 7cam x3f:    702 (N=3424)
  CR2-8B   | 1cam x1f:    206 (N=224) | 1cam x3f:    408 (N=544) | 3cam x1f:    408 (N=544) | 3cam x3f:   1013 (N=1504) | 7cam x1f:    811 (N=1184) | 7cam x3f:   2223 (N=3424)
== feature-cache size, GB (128 tok x d x 3 depths x bytes x 188,640 ticks)
Qwen3-VL-2B  (= Cosmos-Reason2-2B arch)      d= 2048 int8    148  fp16    297  taps L/2,3L/4,L = 14,21,28
Qwen3-VL-8B  (= Cosmos-Reason2-8B / Cosmos3- d= 4096 int8    297  fp16    593  taps L/2,3L/4,L = 18,27,36
Qwen3-VL-32B (= Cosmos-Reason2-32B / Cosmos3 d= 5120 int8    371  fp16    742  taps L/2,3L/4,L = 32,48,64
Cosmos3-Edge reasoner (Nemotron-dense 2B + S d= 2048 int8    148  fp16    297  taps L/2,3L/4,L = 14,21,28
Qwen2.5-VL-7B (= Cosmos-Reason1-7B arch; con d= 3584 int8    260  fp16    519  taps L/2,3L/4,L = 14,21,28
== Thor 128 GB headroom after backbone weights, 16 GiB host reserve (P), 40 GB other tenants (A), 4 GB engine workspace (A)
Qwen3-VL-2B  (= Cosmos-Reason2-2B arch)  NVFP4:  64.6  FP8:  64.0  BF16:  62.6
Qwen3-VL-8B  (= Cosmos-Reason2-8B / Cosm NVFP4:  59.3  FP8:  56.2  BF16:  49.3
Qwen3-VL-32B (= Cosmos-Reason2-32B / Cos NVFP4:  45.0  FP8:  31.3  BF16:   0.1
Cosmos3-Edge reasoner (Nemotron-dense 2B NVFP4:  64.0  FP8:  63.3  BF16:  61.9
Qwen2.5-VL-7B (= Cosmos-Reason1-7B arch; NVFP4:  59.6  FP8:  56.7  BF16:  50.2
== verdicts: A(<=50 ms) / B(<=250 ms) ; GG = passes guarded, g. = passes raw p50 only, .. = fails ; duty = 2 Hz x (T_r+15 ms)
-- NVFP4
  CR2-2B   1cam x1f: A GG B GG duty 0.07 | 1cam x3f: A g. B GG duty 0.10 | 3cam x1f: A g. B GG duty 0.10 | 3cam x3f: A .. B GG duty 0.20 | 7cam x1f: A .. B GG duty 0.17 | 7cam x3f: A .. B g. duty 0.40
  Edge     1cam x1f: A GG B GG duty 0.07 | 1cam x3f: A g. B GG duty 0.12 | 3cam x1f: A g. B GG duty 0.12 | 3cam x3f: A .. B GG duty 0.25 | 7cam x1f: A .. B GG duty 0.20 | 7cam x3f: A .. B g. duty 0.51
  CR2-8B   1cam x1f: A g. B GG duty 0.11 | 1cam x3f: A .. B GG duty 0.17 | 3cam x1f: A .. B GG duty 0.17 | 3cam x3f: A .. B GG duty 0.36 | 7cam x1f: A .. B GG duty 0.30 | 7cam x3f: A .. B .. duty 0.75
  CR1-7B   1cam x1f: A g. B GG duty 0.10 | 1cam x3f: A .. B GG duty 0.18 | 3cam x1f: A .. B GG duty 0.18 | 3cam x3f: A .. B g. duty 0.44 | 7cam x1f: A .. B GG duty 0.35 | 7cam x3f: A .. B .. duty 0.94
  CR2-32B  1cam x1f: A .. B GG duty 0.31 | 1cam x3f: A .. B g. duty 0.47 | 3cam x1f: A .. B g. duty 0.47 | 3cam x3f: A .. B .. duty 0.96 | 7cam x1f: A .. B .. duty 0.80 | 7cam x3f: A .. B .. duty 1.92
-- FP8
  CR2-2B   1cam x1f: A GG B GG duty 0.08 | 1cam x3f: A g. B GG duty 0.11 | 3cam x1f: A g. B GG duty 0.11 | 3cam x3f: A .. B GG duty 0.21 | 7cam x1f: A .. B GG duty 0.18 | 7cam x3f: A .. B g. duty 0.42
  Edge     1cam x1f: A GG B GG duty 0.08 | 1cam x3f: A g. B GG duty 0.13 | 3cam x1f: A g. B GG duty 0.13 | 3cam x3f: A .. B GG duty 0.26 | 7cam x1f: A .. B GG duty 0.22 | 7cam x3f: A .. B .. duty 0.53
  CR2-8B   1cam x1f: A .. B GG duty 0.15 | 1cam x3f: A .. B GG duty 0.22 | 3cam x1f: A .. B GG duty 0.22 | 3cam x3f: A .. B g. duty 0.44 | 7cam x1f: A .. B GG duty 0.37 | 7cam x3f: A .. B .. duty 0.88
  CR1-7B   1cam x1f: A .. B GG duty 0.14 | 1cam x3f: A .. B GG duty 0.24 | 3cam x1f: A .. B GG duty 0.24 | 3cam x3f: A .. B .. duty 0.53 | 7cam x1f: A .. B g. duty 0.43 | 7cam x3f: A .. B .. duty 1.12
  CR2-32B  1cam x1f: A .. B g. duty 0.48 | 1cam x3f: A .. B .. duty 0.68 | 3cam x1f: A .. B .. duty 0.68 | 3cam x3f: A .. B .. duty 1.27 | 7cam x1f: A .. B .. duty 1.07 | 7cam x3f: A .. B .. duty 2.45
-- FP16
  CR2-2B   1cam x1f: A g. B GG duty 0.11 | 1cam x3f: A .. B GG duty 0.17 | 3cam x1f: A .. B GG duty 0.17 | 3cam x3f: A .. B GG duty 0.34 | 7cam x1f: A .. B GG duty 0.28 | 7cam x3f: A .. B .. duty 0.70
  Edge     1cam x1f: A g. B GG duty 0.11 | 1cam x3f: A .. B GG duty 0.18 | 3cam x1f: A .. B GG duty 0.18 | 3cam x3f: A .. B GG duty 0.39 | 7cam x1f: A .. B GG duty 0.32 | 7cam x3f: A .. B .. duty 0.81
  CR2-8B   1cam x1f: A .. B GG duty 0.30 | 1cam x3f: A .. B g. duty 0.49 | 3cam x1f: A .. B g. duty 0.49 | 3cam x3f: A .. B .. duty 1.08 | 7cam x1f: A .. B .. duty 0.88 | 7cam x3f: A .. B .. duty 2.24
  CR1-7B   1cam x1f: A .. B GG duty 0.30 | 1cam x3f: A .. B .. duty 0.55 | 3cam x1f: A .. B .. duty 0.55 | 3cam x3f: A .. B .. duty 1.28 | 7cam x1f: A .. B .. duty 1.04 | 7cam x3f: A .. B .. duty 2.75
  CR2-32B  1cam x1f: A .. B .. duty 1.18 | 1cam x3f: A .. B .. duty 1.92 | 3cam x1f: A .. B .. duty 1.92 | 3cam x3f: A .. B .. duty 4.13 | 7cam x1f: A .. B .. duty 3.40 | 7cam x3f: A .. B .. duty 8.57
-- EAGER
  Edge     1cam x1f: A .. B GG duty 0.14 | 1cam x3f: A .. B GG duty 0.27 | 3cam x1f: A .. B GG duty 0.27 | 3cam x3f: A .. B .. duty 0.66 | 7cam x1f: A .. B g. duty 0.53 | 7cam x3f: A .. B .. duty 1.43
  CR2-8B   1cam x1f: A .. B g. duty 0.44 | 1cam x3f: A .. B .. duty 0.85 | 3cam x1f: A .. B .. duty 0.85 | 3cam x3f: A .. B .. duty 2.06 | 7cam x1f: A .. B .. duty 1.65 | 7cam x3f: A .. B .. duty 4.48
```

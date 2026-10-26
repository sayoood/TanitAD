# HiCAP stream R-A — VLM backbones, VLM-as-embedder evidence, cost model (2026-09-29)

**Status:** COMPLETE — staged in the index, **not committed**. Scope: which frozen VLM should sit under REF-F / "HiCAP"
(CLM-8B transferred to driving; camera + ego state [+ audio] in, hierarchical tactical/trajectory decision out; Jetson Thor at 10 Hz).
Write-only-my-file stream; sibling streams (P1/P2/P3, toys) untouched.

**Evidence classes** (per `CLAUDE.md` rule 1): `PUBLISHED-PRIMARY` = I read the repo/README/config/table myself (commit named) ·
`PUBLISHED-SECONDARY` = one-hop (web-search summary of a paper/blog/HF card; arXiv, HF, developer.nvidia.com, research.nvidia.com,
docs.vllm.ai, marktechpost, andlukyane.com were **egress-blocked** for WebFetch — every arXiv-sourced number below is one hop unless PRIMARY) ·
`PUBLISHED-RECALLED` = from pre-2026 memory, not re-verified this session · `INHERITED` (repo doc, not re-verified) ·
`MEASURED (ours)` = computed here (scripts reproduced verbatim as Appendix A/B) · `ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.
**No TanitAD number here comes from a summary**; TanitAD facts used (27.8 ms bf16 encoder, frozen DINOv2/I-JEPA metric failure, Cosmos-Reason1 pilot flips, Alpamayo-2 paired run, PhysicalAI has no audio) are tagged `INHERITED` with their source file; clip-overlap and cost numbers are `MEASURED (ours)`.

## Headline findings

1. ⭐ **"Cosmos Drive nano" does not exist as a model.** Closest real objects: **Alpamayo 1 / 1.5 "Nano" (10 B, Cosmos-Reason(2) backbone — the *driving* one)**,
   **Cosmos 3 Nano (16 B) / Cosmos3-Edge (4 B, 2.4 B reasoner — the one built for Thor)**, and **Cosmos-Drive(-Dreams)** which is a *video generator*, not an encoder (§1.2).
2. ⭐ **Cosmos3-Edge's reasoner is NOT Qwen-based.** From NVIDIA's own framework code: **Nemotron-dense 2B LLM (28 attn+MLP layer pairs, hidden 2048, relu²) + SigLIP2 tower (hidden 1152) + 2×2 patch-merger**; no audio tower; text-out only (§1.4).
3. ⭐ **Thor numbers exist and are primary.** NVIDIA publishes measured Thor prefill/ViT/decode for Qwen3-VL-2B/4B/8B, Qwen3.5-0.8B…9B, Gemma-4 (§2 table, §5.1). E.g. **Qwen3-VL-4B NVFP4: ViT 11.6 ms + prefill 22.3 ms (292 tokens)**. Cosmos3-Edge reasoner has only **eager-BF16** rows (911-token image prompt: **0.19 s prefill**, 42.6 tok/s decode) — no TRT row.
4. ⭐ **A new, directly relevant public baseline appeared 3 weeks ago:** **Qwen-Drive-1.0-4B (2026-09-03, Apache-2.0)** = Qwen3.5-4B VLM + **1.0 B Planning Expert that reads the VLM's KV caches** (not a pooled vector) + BEV head; NAVSIM PDMS **88.2 SFT / 90.7 RL**, PhysicalAI open-loop ADE₃ₛ **0.37 m** (§4.1). Its own README table has *general* Qwen3.5-4B beating NVIDIA's Cosmos-Reason2-8B / Cosmos3-Nano / Alpamayo-1.5 on driving VQA (LingoQA 70.4 vs 59.6 / 65.0 / 64.0) — **vendor-run, one protocol, not reproduced**.
5. ⭐ **No published result found for a *raw, never-driving-tuned* frozen VLM + small heads doing driving decisions.** Alpamayo-2-Super's "frozen backbone" (`cotrain_expert_vlm:false`) is frozen *after* 80–115 k h of driving SFT; Qwen-Drive's VLM went through staged driving/perception training before the planner stage. HiCAP-on-a-raw-VLM is therefore a genuine bet, not a replication (§4).
6. **The penalty of freezing is interface- and task-dependent:** semantic retrieval ≈ 0 to slightly favourable when the *interface* is good (SLQ: frozen MLLM + latent queries beat VLM2Vec-7B LoRA on MMEB, 67.5 vs 62.9 — different backbones, read as direction); **geometry/metric tasks pay a lot** (Qwen-Drive: frozen-VLM BEV head −6.34 mAP vs a BEVFormerV2 control, unfreezing +10.46 mAP; classical visual odometry beats the best VLMs on EgoDyn-Bench) (§3.2, §4.2).
7. ⛔ **Contamination is the hidden cost of every driving-aware backbone.** Alpamayo (trained on PhysicalAI-AV + ~115 k h) and Qwen-Drive (evaluated on PhysicalAI, appears to have trained on its CoC labels — HYPOTHESIS) may have seen our 40 val episodes; the general backbones are far less likely to have (undisclosed data — probe it). Any driving-aware gain on our val is uninterpretable until decontaminated (§6).
8. **Ranked shortlist (test order):** **R1 general Qwen3.5-4B → R2 driving-aware Qwen-Drive-1.0-4B VLM (same architecture ⇒ a paired "driving-knowledge" ablation) → R3 small/edge Cosmos3-Edge reasoner (2.4 B; Qwen3-VL-2B as the measured-latency fallback; Gemma-4-E2B as the only small native-audio option)**. Alpamayo-1.5-10B / Cosmos3-Nano are **teachers**, not per-tick encoders (§6).
9. **Everything ≥ 2 B is ≥ 8× the "sub-300 M" programme target** — a distillation path is mandatory and NVIDIA's own edge story is exactly that (Alpamayo teacher → **0.5–2 B** students, `SECONDARY`). Whether "sub-300 M" survives is a **PI scope decision** (§6.4).

---

## 1. What NVIDIA actually ships (priority 1) — state at 2026-09-29

### 1.1 Access log (weights the evidence)

| source | how | status |
|---|---|---|
| `github.com/NVIDIA/cosmos` @`3e3c6d6` (2026-09-29): README, `docs/reference/models.md`, `inference_benchmarks.md`, `cookbooks/cosmos3/**` | shallow clone | PRIMARY |
| `github.com/NVIDIA/cosmos-framework` @`cf5d68c` (2026-09-23): `configuration_cosmos3_edge.py`, `Cosmos3-Edge.yaml`, `vision_siglip2.py` | blobless clone | PRIMARY |
| `github.com/nvidia-cosmos/cosmos-reason2` @`a3b4a1d`; `NVlabs/alpamayo-recipes` @`ae5bc10`; `NVlabs/alpamayo1.5` @`36aeb4c`; `NVlabs/alpamayo2` @`5e7975f` | shallow clone | PRIMARY |
| `github.com/NVIDIA/TensorRT-Edge-LLM` @`95515c2` (2026-09-29): `performance-benchmarks.md` (v0.10.0 Thor tables), `supported-models.md`, Cosmos3/Alpamayo examples | shallow clone | PRIMARY |
| `QwenLM/Qwen3-VL` @`9658872`, `Qwen3-VL-Embedding` @`393e297`, `Qwen3-Omni` @`e423585`, `QwenLM/Qwen-Drive-1.0` @`28091c1` (2026-09-03) | shallow clone | PRIMARY |
| `Contrastive-LM/CLM` @`bb42c6c`, `cover-vla` @`21a9960` | clones (also read by the F-R stream) | PRIMARY |
| HF model cards (Cosmos3-Edge/Nano, Alpamayo-1.5, Qwen3.5, Gemma-4), NVIDIA dev blogs, arXiv abstracts | 30 WebSearch summaries (budget exhausted) | SECONDARY |
| Cosmos 3 technical report, Alpamayo-R1 paper, Qwen-Drive report, Qwen3-VL/Omni reports | **not readable** (egress-blocked) | NOT READ — ablation tables in them are the main gap |

### 1.2 What did the PI mean by "Cosmos Drive nano"? (ranked)

Verified absence: no model of that name in the `NVIDIA/cosmos` model table, the `alpamayo-recipes` model table, the TRT-Edge-LLM supported list, or 4 targeted searches.

| rank | candidate | why plausible | what it actually is |
|---|---|---|---|
| **1 (most likely)** | **Alpamayo 1 / 1.5 "Nano"** | the only NVIDIA line that is *both* "Nano"-named *and* driving *and* Cosmos-lineage (Cosmos-Reason / Reason2 backbone) | 10 B VLA = ~8.2 B VLM + 2.3 B flow-matching action expert (PRIMARY, `alpamayo-recipes` README) |
| **2** | **Cosmos 3 Nano (16 B) / Cosmos3-Edge (4 B)** | "Cosmos … nano/edge"; NVIDIA markets Edge for road-scene understanding on Thor (SECONDARY) | general physical-AI omni-model, **not driving-tuned** in the base checkpoint (PRIMARY: model table lists only DROID robot policies as post-trained examples) |
| 3 | **Cosmos-Drive / Cosmos-Drive-Dreams** (arXiv 2506.09042) | literal name match | *video generation* pipeline (Cosmos-Transfer1-7B-Sample-AV, Single2Multiview…) for synthetic data — **cannot serve as a state encoder** (SECONDARY) |

**Practical reading:** what he wants = *"Cosmos-lineage reasoner with driving knowledge that fits Thor"* = **Alpamayo-1.5's VLM (driving knowledge, too big) ⊕ Cosmos3-Edge's reasoner (Thor-sized, no driving tuning)**. The two have not been combined by anyone; NVIDIA's own answer to the gap is *distillation* (§1.6).

### 1.3 The NVIDIA family, one row each

`P` = PRIMARY, `S` = SECONDARY, `R` = RECALLED. Tokens/image use **32 px per token** (patch 16, 2×2 merge) unless stated.

| model | size / architecture | input (res, tokens, video, multi-view, audio) | licence / weights | Thor/Orin latency | driving evidence |
|---|---|---|---|---|---|
| **Cosmos-Reason1** (2503.15558 S) | 7 B, Qwen2.5-VL-7B base (R) | video, dynamic res | NVIDIA Open Model License (R) | none | AR1 backbone (S); LingoQA 45.2 (vendor-run, Qwen README P) |
| **Cosmos-Reason2 2B / 8B / 32B** (Dec-2025, 32 B Apr-2026) | Qwen3-VL-2B/8B/32B post-trained — "follows the same architecture" (S, HF card) | video @ `--fps 4` recommended (P), Qwen3-VL processor | **NVIDIA Open Model License** (P: README); code Apache-2.0 | Thor: *Transformers inference tested, no numbers* (P); ⇒ **Qwen3-VL-2B/8B Thor rows apply by architecture identity** (ESTIMATED) | LingoQA 59.6 (8 B; vendor-run by Qwen, P) |
| **Cosmos 3 Edge** (Jul-2026) | **4 B MoT** (3.859 B BF16, S); **reasoner 2.4 B = Nemotron-dense-2B + SigLIP2 + merger** (§1.4); + diffusion/action generator tower | reasoner: text+image+video → text; **no audio tower**; 256p/480p generation; policy obs resized 736×544 (DROID, P) | **OpenMDW-1.1** (P: `NVIDIA/cosmos` README "code and models") | **eager HF-Transformers on Thor T5000: 911-token image prompt prefill 0.19 s (4,845 tok/s), decode 42.6 tok/s** (P: `inference_benchmarks.md`); policy 32-action chunk **median 1.528 s** (S); TRT-Edge-LLM 0.10.0 supports it but **no benchmark row** (P) | none published (robot policy = DROID) |
| **Cosmos 3 Nano** (May-2026) | **15.16 B MoT** (S); reasoner = Qwen3-VL-8B architecture (36 L, d 4096, 32/8 heads, ffn 12288 — S) ≈ 8 B + ≈ 8 B generator | as Edge + **sound generation**; Reasoner "**does not currently support the audio modality by design**" (P) | OpenMDW-1.1 | reasoner-only ≈ **16–17 GB** with vLLM/TRT (P); NIM on Thor needs explicit `NIM_GPU_MEMORY_UTILIZATION` 0.7–0.8 (P); no latency row on Thor | LingoQA 65.0; Ego3D RMSE 22.41 (vendor-run) |
| **Cosmos 3 Super** | 64 B MoT (README: "teacher for distillation", P) | — | OpenMDW-1.1 | H200/B200 class | = Alpamayo-2-Super's 32 B reasoner (P: recipes README "Cosmos 3 backbone") |
| **Alpamayo 1 Nano** (=R1-10B, 2511.00088) | Cosmos-Reason(1) **8.2 B** + **2.3 B** expert; 4 cams; >1 B images / 80 k h; 24 GB GPU (P) | 4 cameras × video + ego history + text | **UNVERIFIED** (search summaries conflict: "non-commercial" vs OpenMDW-1.1; the 1.5 and 2-Super READMEs say OpenMDW-1.1) | **TRT-Edge-LLM chained VLM+action pipeline on Thor, FP16 only** (P: `examples/vla/alpamayo.md`); AR1 paper: **99 ms** on-vehicle (INHERITED from `ALPAMAYO2_SUPER_ANALYSIS.md`, not Thor) | +12 % planning on hard cases, −35 % close-encounter (INHERITED) |
| **Alpamayo 1.5 Nano** (10 B) | Cosmos-Reason2 backbone, RL post-trained, flexible camera count, nav guidance, VQA (P) | **163,840–196,608 px/image ⇒ 160–192 tokens**, **4 frames/camera** (`helper.py`, `load_physical_aiavdataset.py`, P); 64 wp @10 Hz (S) | **OpenMDW-1.1** (P: README) | H100: 24 GB (1 sample) / 40 GB (16) / 60 GB (16+CFG) (P); **quant recipe: BF16 ~22 GB → FP8 ~11 GB → AutoQuant 6.5 bpe ~9 GB** (P); no Thor number | LingoQA 64.0; MMBench **7.5** / MMStar 26.1 in Qwen's protocol (format or ability loss after CoC tuning — vendor-run, cause not established) |
| **Alpamayo 2 Super** (34 B, 2026-08-04) | 32 B Qwen3-VL-arch (Cosmos 3 Super reasoner) + 2.3 B expert; 6-of-7 cams × 4 frames | 512×384-class dynamic (min/max 163,840/196,608 px) | OpenMDW-1.1 weights (P) | H100 72 GB; **not a per-tick model** (INHERITED analysis) | minADE₆ 0.911 m @6.4 s (best-of-6; INHERITED) |
| **Nemotron Nano V2 VL** (2511.03929) | 12 B, C-RADIOv2-H + hybrid Mamba-Transformer (S) | video w/ Efficient Video Sampling | NVIDIA licence (UNVERIFIED) | none | not driving |
| **Nemotron-3-Nano-Omni-30B-A3B** | 30 B MoE / **3 B active**; **C-RADIOv4-H vision + Parakeet-TDT-0.6B-v2 audio** (S) → text/image/video/**audio** in, text out | native audio + video | UNVERIFIED | **Thor NVFP4: prefill 163.6 ms @2,048 tok, decode 71.2 tok/s; TRT-reported GPU memory ≈ 19–20 GB in the v0.7/0.9 rows** (P) | not driving |
| **Cosmos-Embed1** | video–text dual encoder: EVA-ViT-G@224 → Q-Former → one clip vector (S, F-R stream) | sampled frames | NVIDIA licence (UNVERIFIED) | none | retrieval/dedup only — **one pooled vector = the regime CoVer avoids** |

### 1.4 Cosmos3-Edge reasoner, read from source (`cosmos-framework@cf5d68c`, "renewed `nvidia/Cosmos3-Edge` root `config.json`" defaults) — PRIMARY

| field | value |
|---|---|
| text tower | `hidden_size 2048`, **28 paired layers** (each = 1 attention block + 1 MLP block; "Nemotron-H 56-block layout"), `intermediate 9216`, **16 heads / 8 KV heads, head_dim 128**, `hidden_act relu2` (squared ReLU), `vocab 131,072`, `max_pos 131,072`, `tie_word_embeddings False`, mRoPE sections `[24,20,20]`, `rope_theta 1e8` |
| vision | SigLIP2 (`Siglip2VisionConfig`, packed variable-resolution "naflex" patches; `input_hidden_size 1152` ⇒ SO400M-class), **PatchMerger** 2×2 → `merger_intermediate 11520` → `out_hidden 2048` |
| my parameter check | per pair: attn 4.19+4.19+4.19 (q,kv,o) ≈ 12.6 M + MLP 2·2048·9216 = 37.7 M ⇒ 50.3 M × 28 = **1.41 B**; embeddings 2 × 131,072 × 2048 = **0.54 B**; ViT ≈ 0.41 B; merger ≈ 0.08 B ⇒ **≈ 2.43 B ≈ the "2.4 B reasoner"** ✔ (`MEASURED (ours)`, arithmetic on the config) |
| what is *not* verified | ViT patch size (16 assumed ⇒ 160 tokens for 256×640), the exact HF checkpoint config (blocked), driving content of the mid-training (`cosmos3_ga_4bm2b_v1_midtrain` in the yaml only names the run) |
| bandwidth sanity | 4.8 GB BF16 / 273 GB/s = **57 tok/s ceiling**; NVIDIA measured 37–43 ⇒ 65–75 % of ceiling ✔ (decode is bandwidth-bound, as the brief says) |

### 1.5 Thor latency facts that exist (all `PUBLISHED-PRIMARY` unless noted) and the gaps

* **NVIDIA TRT-Edge-LLM 0.10.0 (JetPack 7.2, TRT 10.16), Thor, batch 1** — rows in the §2 table, calibration in §5.1. ViT = FP16 engine; LLM = NVFP4 or INT4-AWQ.
* **Cosmos3-Edge eager BF16 on Thor T5000** (`inference_benchmarks.md`, "preliminary"): text 1,705 tok → 0.20 s (**8,717 tok/s**); image 911 tok → **0.19 s** (4,845 tok/s); video 1,263 tok → 0.21 s; decode 37.3–42.6 tok/s. Same table: Thor T4000 0.26 s, T3000 0.34 s, T2000 0.74 s, **AGX Orin 0.50 s** for the image prompt. ⇒ implied eager-BF16 LLM throughput 2·1.41 B·8,717 ≈ **24.6 TFLOPS** (`MEASURED (ours)` arithmetic). For scale: TanitAD's own bf16 encoder measured 27.8 ms (`INHERITED`, THOR_PROFILE.md; its FLOP count is not re-derived here).
* **Absent:** any Thor number for Cosmos-Reason2, Cosmos3-Nano, Alpamayo-1.5/2 (only Alpamayo-R1 has a TRT pipeline recipe, no table), Qwen-Drive. The "INT8 ~60 ms @10 Hz on DRIVE Thor" line from one search summary was **not** reproduced in a second search ⇒ `UNVERIFIED`, unused.

### 1.6 NVIDIA's own edge answer is distillation

`SECONDARY` (NVIDIA "Alpamayo 1.5: Distill Large AV Models into Edge Driving AI", via search summary): Alpamayo 1.5 is offered as a **teacher**; students are **0.5–2 B** parameters for DRIVE Orin/Thor; `alpamayo-recipes` (P) ships SFT, GRPO RL and **FP8 / NVFP4+FP8 quantisation** recipes, and the Cosmos 3 README calls Super the "teacher for distillation" (P). ⇒ **NVIDIA does not claim a 10 B+ VLA runs per-tick on Thor**; neither should HiCAP.

---

## 2. General VLM candidates as a frozen state encoder (priority 2)

**Token arithmetic.** Qwen3-VL-lineage processors (Qwen3-VL, Qwen3.5, Cosmos-Reason2, Alpamayo, Qwen-Drive): patch 16, 2×2 merge ⇒ **32 px per token**
(P: `Qwen3-VL/README.md` "compression ratio is 32"; `Qwen-Drive/docs/model.md`). ⇒ **448×448 → 196 tokens · 640×640 → 400 · 256×640 (8×20 merged) → 160**; video adds 2× temporal compression
(temporal patch 2, P README). The README recommends 256–1,280 tokens per image and 256–16,384 per video. `MEASURED (ours)` arithmetic.

**Thor columns** = NVIDIA-published TRT-Edge-LLM 0.10.0 rows (P; JetPack 7.2; batch 1; **one COCO image ≈ 265 merged tokens ≈ 1,060 patches**; ViT engine FP16; LLM NVFP4 unless stated):
`ViT ms` · `prefill ms @292 tok` · `prefill ms @2,048 tok (llm_bench)` · `INT4-AWQ prefill @2,048` · `decode tok/s`.

| model | params / vision tower | 256×640 tokens · video · multi-image | audio in | licence | **Thor (P)**: ViT · pf@292 · pf@2048 · AWQ@2048 · decode | note |
|---|---|---|---|---|---|---|
| **Qwen3-VL-2B** (2025-10-21) | ≈ 2 B; SigLIP2-**Large 300 M** (S: tech report 2511.21631); DeepStack (ViT mid-layer tokens added to first 3 LLM layers, S) | 160 · yes (2× temporal, timestamps) · yes | no | Apache-2.0 (P: LICENSE) | 11.4 · **12.7** · 28.0 · 145.3 · 136 | ≡ Cosmos-Reason2-2B architecture (ESTIMATED equivalence) |
| **Qwen3-VL-4B** | ≈ 4 B (LLM body ≈ 3.63 B by my arithmetic); SigLIP2-L 300 M | 160 · yes · yes | no | Apache-2.0 | 11.6 · **22.3** · 62.2 · 364.4 · 73 | fastest 4 B on Thor |
| **Qwen3-VL-8B** | ≈ 8 B (LLM body ≈ 6.95 B); SigLIP2-**SO400M** | 160 · yes · yes | no | Apache-2.0 | 15.7 · **32.1** · 104.5 · 662.2 · 44 | ≡ Cosmos-Reason2-8B / Alpamayo-1.5 VLM lineage |
| **Qwen3.5-0.8B / 2B / 4B / 9B** (2026-03-02, S) | natively multimodal, **Gated-DeltaNet hybrid**, 262 K ctx; ViT ≈ SigLIP2-class (ViT ms below) | 160 (same processor family, P via Qwen-Drive docs) · video yes · yes | no | Apache-2.0 (S) | 0.8 B: 3.8 · 13.2 · 32.7 · 72.4 · 230 ┃ 2 B: 10.9 · 18.2 · 43.3 · 151.7 · 122 ┃ **4 B: 10.9 · 31.6 · 95.2 · 371.4 · 68** ┃ 9 B: 14.4 · — · 135.3 · 690.5 · 40 | MMMU-Pro (S): 4 B **65.4**, 9 B **69.2** vs Qwen3-VL 4 B 52.0 / 8 B 56.6; prefill ≈ 1.5× slower than Qwen3-VL at equal size (hybrid kernels) |
| **Qwen3-Omni-30B-A3B** | 30 B MoE / **3 B active**, Thinker–Talker; audio = **AuT** encoder, 12.5 Hz tokens (S); min memory BF16 **68.7 GB (Thinking)** – 78.9 GB (P: README, 15-s video) | video yes | **yes, native** | **Apache-2.0** (P: LICENSE) | no Omni row; *text* Qwen3-30B-A3B NVFP4: pf@2048 **137.7** ms, decode 84 (P); `qwen3_omni_moe` is on the TRT-Edge-LLM supported list (P) | fits Thor memory (128 GB), too big for per-tick |
| Qwen3.5-Omni | Plus / Flash / Light; "not open-source", Light reported open (S — sources conflict) | — | yes | **UNVERIFIED** | none | ignore until weights verified |
| **Gemma-4 E2B / E4B** | 2.3 B / 4.5 B *effective* (5 B / 8 B with Per-Layer Embeddings); **150 M ViT**, **305 M USM-Conformer audio encoder** (40 ms chunks) (S) | image tokens configurable **70 / 140 / 280 / 560 / 1,120** (S) — 256×640 ⇒ 140 or 280 · video: UNVERIFIED | **yes** (P: text+image+audio) | **UNVERIFIED** | E2B: 16.7 · **24.8** (NVFP4) / 36.6 (FP16) · pf@2048 n/a · AWQ@292 47.7 · 101 (NVFP4) ┃ E4B: 16.7 · **36.1** · n/a · AWQ@292 84.8 · 59 | the only *small* natively-audio model with Thor rows |
| Gemma-4-12B | "unified" (encoder-free, S) | — | yes | UNVERIFIED | FP16: pf@292 146.8 ms, pf@2048 **622.8** ms, decode 10 | too heavy |
| **InternVL3.5 1 B–14 B** | InternViT-300 M/6 B + Qwen3 LLM (R) | 448² tile = 256 tok (R) ⇒ 256×640 ≈ **768** (2 tiles + thumbnail) or 256 squashed | no | (R) MIT/Apache | **no row** (supported list only, P) | token-hungry on panoramic crops (ESTIMATED 4.8× Qwen) |
| **Phi-4-multimodal** | 5.6 B, SigLIP-400 M + speech LoRA (R) | 448 crops (R) | **yes** (P supported list) | MIT (R) | no row | audio+vision in 5.6 B |
| SmolVLM2 256 M / 500 M / 2.2 B | SigLIP-B/16, pixel-shuffle (R) | 64–81 tok per tile (R) | no | Apache-2.0 (R) | no row | world knowledge too thin for the "general knowledge" premise |
| **Cosmos3-Edge reasoner** | 2.4 B (§1.4) | 160 (if patch 16 — UNVERIFIED) · video yes (P vLLM cmd) | **no** | OpenMDW-1.1 | eager BF16, 911-tok image: **190 ms** prefill, 42.6 tok/s (P) | no TRT row |

**Which give the best "world knowledge per Thor-millisecond"?** (`MMMU-Pro` is a *crude* proxy, S/vendor numbers; latency = NVFP4 `llm_bench` prefill @2,048, P.)
Pareto points, ms → MMMU-Pro: **Qwen3-VL-4B 62 → 52.0** · **Qwen3.5-4B 95 → 65.4** · Qwen3-VL-8B 104 → 56.6 (dominated) · **Qwen3.5-9B 135 → 69.2**. ⇒ **Qwen3.5-4B is the knee** (+13.4 MMMU-Pro for +33 ms over Qwen3-VL-4B); Qwen3-VL-8B is dominated by Qwen3.5-4B on this proxy. Caveat: MMMU-Pro is exam-style QA, not driving-state discrimination — `HYPOTHESIS` that it ranks frozen driving features.

**Audio (PI wish).** ⛔ **PhysicalAI-AV has no audio channel** (36 features = 7 camera + 6 calibration + 3 label + 1 lidar + 19 radar; `grep -ci audio` = 0 over 9 probe artifacts — `MEASURED` by the sibling P1 stream 2026-09-29, `INHERITED` here) ⇒ **no audio can be trained on the parity corpus**; an audio path is `HYPOTHESIS` until an audio-bearing driving corpus exists. Options if wanted, cheapest first: (a) **Gemma-4-E2B/E4B** (native, 305 M encoder, ~25 tok/s of audio) or **Nemotron-3-Nano-Omni** (Parakeet-0.6 B; Thor NVFP4 measured, P) or **Qwen3-Omni** (AuT, 12.5 Hz); (b) bolt a frozen ASR-class encoder (Qwen3-ASR-0.6B's AuT is **180 M, hidden 896**, S; Nemotron-3.5-ASR-streaming-0.6B, P supported list) to a *non-audio* VLM through a trained 2-layer projector (LLaVA-style recipe, R); (c) defer — measure first whether siren/horn cues exist in any available corpus.

**Three Thor-specific cautions** (all P, `performance-benchmarks.md` v0.10.0): (1) **hybrid-SSM prefill is not yet fast** — `Nemotron-3-Nano-4B` NVFP4 prefill **1,228 ms @2,048** vs Qwen3-4B **62 ms**; Qwen3.5's Gated-DeltaNet costs ≈ 1.5×; (2) **INT4-AWQ prefill is 5–6× slower than NVFP4** (W4A16 dequant + FP16 math) — quote which; (3) **batching cameras helps the ViT only ~18 %** (Qwen3-VL-2B: 8 images 75 ms vs 8 × 11.4 = 91 ms).

---

## 3. VLM-as-embedder evidence (priority 3)

### 3.1 The recipes

| system | backbone | pooling | layer | instruction | adaptation | headline | class |
|---|---|---|---|---|---|---|---|
| **CLM-8B** (`clm@bb42c6c`) | Qwen3-8B, **frozen** | **last token** (`vllm --runner pooling`) | final (no layer option in repo) | the *question* is appended after the state text; `--enable-prefix-caching`, `--max-model-len 2048` | 2 MLP heads 4096→1536×3→512 (**9,443,840 params/side**, F-R stream arithmetic) on cached embeddings | frozen-vs-adapted ablation **not in README/code** (blog only) | P |
| **CoVer-VLA** (2602.12281, `cover@21a9960`) | SigLIP 2 (open_clip, bf16), **frozen** | **trained** text-aware patch attention + 4-layer attention pooling | patch tokens | text | numeric action tower from scratch; symmetric InfoNCE | verifier over VLA candidates | P (F-R read) |
| **Qwen3-VL-Embedding-2B/8B** (`@393e297`) | Qwen3-VL-2B/8B | **EOS token, last layer** | final | **instruction-aware**, MRL (Matryoshka) | **LoRA r32/α32 on q,k,v,up,down,gate**; dual-tower | **MMEB-V2 all: 2 B 73.2, 8 B 77.8** vs VLM2Vec-2B 47.7, VLM2Vec-V2-2B 59.2, GME-7B 59.1, IFM-TTE-8B 74.1 (P, README table) | P |
| VLM2Vec (2410.05160) / V2 (2507.04590) | Phi-3.5-V, LLaVA-1.6, Qwen2-VL | last token (E5-V-style) | final | task instruction | **LoRA r8** | MMEB baseline | S |
| E5-V (2407.12580) | LLaVA-NeXT | last token + "in one word" prompt | final | prompt-as-embedder | text-only NLI training (R) | zero-shot-ish | R |
| **FreeRet** (2509.24621) | any MLLM | last token | **hidden state after the last *attention* block, before the last MLP** | controlled prompt | **none — training-free** | **+13.9 MMEB pts over E5-V on the same backbone; beats trained MM-Embed** | S |
| **SLQ** (2604.13710, ICML 2026) | Qwen/InternVL family, **frozen** | **shared latent queries appended** to text and image sequences (native causal attn aggregates) | final | — | trainable = queries only (**36 k params** on InternVL3-1B vs 4.4 M for LoRA) | **MMEB overall 67.5 (8 B), 64.5 (4 B) vs VLM2Vec-7B 62.9**; beats full-FT and LoRA on COCO/Flickr30K (S) | S |
| BToks (2604.11095) | single-encoder MLLM | learned bottleneck tokens | — | — | — | title only, not read | S |
| **jina-embeddings-v4** (2506.18902) | Qwen2.5-VL-3B (3.8 B) | **mean pool** 2048-d (→128) **and** multi-vector 128-d/token | final | task adapters | **3 LoRA adapters × 60 M**, backbone frozen | single- and multi-vector | S |
| **Omni-Embed-Nemotron** (2510.03458) | Qwen2.5-Omni-3B Thinker (4.7 B) | shared 2048-d | — | — | contrastive | text/image/**audio**/video in one space | S |
| NV-Embed-v2 (R) | Mistral-7B | **latent-attention pooling** + bidirectional attn | final | instruction | LoRA | MMTEB 56.3 (P table) vs Qwen3-Embedding-8B 70.6 | P/R |
| Skean et al. (2502.02013, ICML'25) | LLMs, SSMs, vision | — | **mid-depth layers up to +16 % over the final layer on 32 tasks** | — | probe-only | layer choice matters | S |

### 3.2 What they agree on — and what it means for HiCAP

1. **Pooling:** "last/EOS token conditioned by an instruction" is the industry default, **but** every recent frozen-backbone paper beats it with a *learned aggregator* (SLQ latent queries, CoVer patch-attention pooling, NV-Embed latent attention, FreeRet's layer surgery). ⇒ HiCAP should follow **CoVer/SLQ, not CLM's bare last-token** — the F-R stream reached the same conclusion from the CoVer side. (`HYPOTHESIS` that it transfers to driving.)
2. **Layer:** final-layer is not the optimum (Skean +16 %, FreeRet pre-last-MLP); both driving expert designs read **many layers** (Alpamayo: per-layer KV of a 64-layer expert; Qwen-Drive: **8 KV caches, one per 4 layers**). ⇒ expose **≥ 3 depths** (e.g. L/2, 3L/4, L) to the head with learned mixing — one forward, no extra cost (`HYPOTHESIS`); truncating the LLM at 3L/4 also saves ≈ 25 % of prefill (`ESTIMATED`).
3. **Instruction/question conditioning:** universal, and it is what makes CLM multi-headed: **question = suffix**. With prefix caching the state prefix (image + ego tokens) is computed once and each **family question** (longitudinal · lateral · tactical · strategic — the four binding eval families) is a ~10–20-token suffix ⇒ ≈ 4 × 20 tokens ≈ **≤ 1–2 ms** extra (`ESTIMATED`, from the 292→2,048 slope of 0.023 ms/token for a 4 B model).
4. **Frozen vs adapted — the measured penalty depends on task type:**

| task | frozen result | adapted/specialist result | penalty of freezing | class |
|---|---|---|---|---|
| semantic retrieval (MMEB) | SLQ 8 B **67.5** | VLM2Vec-7B (LoRA) 62.9; SLQ itself beats LoRA/full-FT on COCO/Flickr | **≈ 0 or negative** (different backbones across rows — read as direction) | S |
| semantic embedding (MMEB-V2) | — | Qwen3-VL-Embedding-8B (LoRA r32) **77.8** | no frozen row published | P |
| 3-D detection from VLM tokens (Qwen-Drive BEV) | frozen VLM + BEV head **−6.34 mAP** vs BEVFormerV2-with-same-encoder | unfreezing encoder+VLM **+10.46 mAP** | **large** (geometry) | S |
| ego-motion | frozen DINOv2 / I-JEPA fail metric ego-motion (`INHERITED (brief)`); classical VO beats best VLMs (EgoDyn-Bench, S) | — | **large** (metric) | S/INHERITED |
| **driving decisions from a frozen raw VLM** | **no published number** | — | **unknown — the measurement HiCAP exists to make** | — |

5. **CLM's own evidence is thinner than the brief suggests:** the *frozen Qwen3-8B + 9.44 M-param heads* result is real (P), but the README/code contain **no frozen-vs-fine-tuned-backbone ablation** and no scaling fit window/R² (blog unread) ⇒ "frozen suffices" is CLM's *design*, not a *measured* CLM finding (`UNVERIFIED`). The nearest measured support is SLQ/FreeRet on retrieval (S) — a different task class from state→action scoring.

---

## 4. Frozen or lightly-adapted VLM features for driving/planning (priority 4)

### 4.1 What has actually been published

| system | what is frozen / how the planner reads the VLM | result | class |
|---|---|---|---|
| **Alpamayo-R1 / 1.5 / 2-Super** | expert (2.3 B; 64 layers × 1,536 hidden in 2-Super) trained with `cotrain_expert_vlm:false` (2-Super config; R1/1.5 not checked) on a **backbone already SFT'd on 80–115 k h + 3.7 M CoC traces**; expert attends into the backbone's per-layer KV cache; unicycle (accel, curvature) action space, 10 Euler steps | R1: **+12 %** planning on hard cases, **−35 %** close encounters, 0.5 B→7 B monotone (INHERITED/S); 2-Super minADE₆ 0.911 m; our own paired run: ADE₂ₛ **0.2703 m** vs flagship 0.3303 m, speed bias **+0.057 vs +0.425 m/s** (INHERITED, `ALPAMAYO2_SUPER_ANALYSIS.md` §11, 39 clips) | INHERITED |
| **Qwen-Drive-1.0-4B** (`@28091c1`, Apache-2.0) | Qwen3.5-4B VLM; **1.0 B DiT expert (32 L, width 1,024, 16 q-heads d256, 4 KV)** cross-attends **8 cached KV layers** (one per 4); ego history (16 poses) + velocity + accel are MLP-encoded *at the expert*; nav command + ego status via AdaLN; 50 wp × (x,y,heading) @10 Hz, 10 Euler steps; planner trained with the VLM frozen (S — not in the repo) | NAVSIM PDMS **87.8 / 88.2 / 90.7** (no-reasoning / reasoning / RL); best-of-6 89.3 / 91.4; WOD-E2E RFS 7.76→7.91, ADE₃ₛ 1.19; **PhysicalAI (644 ex.) ADE₃ₛ 0.37, ADE₅ₛ 1.07, minADE₃ₛ 0.34**; RL *worsens* open-loop ADE (0.37→0.42) | P (README/docs); freezing = S |
| **VLA-R** (2511.12405) | frozen open-world VLM perception + Q-Former + vision–action contrastive retrieval | < 2 h of driving; no numbers read | S |
| **DriveVLM-Dual** (2402.12289) | VLM as slow System-2 next to a conventional planner | ~410 ms VLM latency, asynchronous | S (F-R stream) |
| EMMA (Waymo), SimLingo, OpenDriveVLA | VLM fine-tuned end-to-end (not frozen) | ids/numbers **not re-verified** | R |
| "Encoder Winners Do Not Reliably Transfer Across VLA Backbone Scale: A Frozen-Backbone Grafting Diagnostic" (2606.14153) | title only | warns that encoder rankings from a small backbone may not transfer to a larger one | S (title) |

⇒ **Two independent groups (NVIDIA, Qwen) converged on the same interface: a small flow-matching expert that reads the VLM's *KV caches at many layers*, not one pooled vector.** This is the strongest prior against CLM's single-vector bottleneck for a *trajectory* head — and it is compatible with frozen use because the caches are a by-product of the prefill you already pay for. For HiCAP's *scoring* (selector) role a pooled/learned-query vector may still suffice (the F-R stream's chooser argument) — that is the experiment (§6.3).

**Comparability warning.** Qwen-Drive's PhysicalAI ADE feeds **16 ego-history poses to the expert**; TanitAD's own finding is that open-loop L2 has an **ego-status shortcut** (`INHERITED`, `2026-07-17-openloop-l2-egostatus-shortcut.md`), and their sample count (6), horizon (3/5 s) and clip list ("clips listed in alpamayo-recipes") differ from our windows. **Do not compare 0.37 m to any TanitAD ADE.**

### 4.2 What breaks with frozen VLM features (with evidence class)

1. **Metric ego-motion / speed.** EgoDyn-Bench (2604.22851, ECCV 2026): *classical visual odometry beats the best VLMs*; dense kinematic time-series grounding beats prose summaries (S). Frozen DINOv2/I-JEPA fail metric ego-motion (`INHERITED (brief)`). Sibling measurement on **our** clips: Cosmos-Reason1-7B's longitudinal label agrees with the kinematic 85th-percentile band only **45 %**, and `LONMODE` **flips on 38 % of cosmetic weather repaints with identical ego motion** (`INHERITED`, `2026-07-20-cosmos3-vlm-pilot.md`, MEASURED there on generated labels — transfer to *embeddings* is `HYPOTHESIS`). ⇒ **ego kinematics must enter as numeric channels at the head/expert** (Qwen-Drive: MLP-encoded history; Alpamayo: 1,000-entry history-trajectory tokens), never be expected from pixels. *(Inference-time ego state is admissible for the decision model per the brief; the CLAUDE.md goal-path rule still forbids any situation-classifier output in the goal input.)*
2. **3-D geometry / range.** Frozen-VLM BEV head −6.34 mAP; the authors' own reading: the VLM "does not provide an explicit 3-D representation that can be read out directly" (S). Longitudinal distance-keeping (headway/TTC — 88.7 % of our oracle gap is longitudinal, CLAUDE.md) is exactly this.
3. **Small / far objects vs resolution.** Alpamayo: **160–192 tokens/image** (~512×384); Qwen-Drive: history frames ≈ 170 tokens but **current frame ≈ 900 tokens (921,600 px)**; ours: 160. No resolution ablation was readable ⇒ `HYPOTHESIS` that 256×640 starves far agents; discriminating test in §6.3 (256×640 vs 384×960 = 360 tokens, same backbone).
4. **Appearance sensitivity** (the 38 % repaint flip above) and **precision drift** (NVFP4/FP8 features vs the BF16 features a head was trained on: `UNVERIFIED` magnitude — must be measured, §6.3).
5. **Driving tuning can erode the general space:** Alpamayo-1.5 scores MMBench 7.5 in Qwen's protocol (vendor-run; format vs ability not established) whereas Qwen-Drive keeps 85.5 vs base 87.1 ⇒ the "general knowledge *and* driving knowledge" combination the PI wants is demonstrated only by Qwen-Drive (S/vendor).
6. **Temporal design differs wildly:** Alpamayo 4 frames at 0.1 s spacing (t0−0.3…t0; near-duplicate frames, motion from sub-frame differences); Qwen-Drive 4 frames at **2 Hz** (1.5 s history) + higher-res current frame. For HiCAP's ViT-feature cache the second is cheaper (fewer new frames per tick) — see §5.

---

## 5. Cost model for the top-3 backbones (priority 5)

**Method.** Linear-layer FLOPs ≈ **2 · P_body · N_tok**; attention adds **4 · L · N² · d** (≈ 3 % at 544 tokens, ≈ 8 % at 1.5 k for a 4 B model; less for Gated-DeltaNet layers); ViT ≈ 2 · P_vit · N_patch + 4 · L_v · N_patch² · d_v.
Latency is **not** derived from peak FLOPs; it is scaled from NVIDIA's *measured* Thor rows (§2), so the scale factor is empirical. Script reproduced in Appendix A (`MEASURED (ours)` arithmetic; every input tagged).

### 5.1 Calibration — what Thor actually delivers (all implied by P rows)

| path | implied effective throughput | source arithmetic |
|---|---|---|
| ViT, FP16, TRT | **≈ 64 TFLOPS** (same for the 300 M and the 400 M tower) | (2·0.41e9·1,064 + 4·27·1,064²·1,152) / 15.7 ms; (2·0.30e9·1,060 + 4·24·1,060²·1,024) / 11.6 ms |
| LLM prefill @2,048, **NVFP4**, TRT | **240–270 TFLOPS** (4 B: 239, 8 B: 272) but only ≈ 95 TFLOPS at 292 tokens (weight-read / launch floor) | 2·6.95e9·2,048 / 104.5 ms |
| LLM prefill @2,048, **INT4-AWQ** (FP16 math) | **≈ 41–43 TFLOPS** | 2·6.95e9·2,048 / 662 ms |
| eager BF16 (HF Transformers) | **≈ 25 TFLOPS** (Cosmos3-Edge 24.6) | 2·1.41e9·8,717 tok/s |
| decode, bandwidth-bound | 65–75 % of 273 GB/s ÷ weight bytes | Cosmos3-Edge 42.6 vs 57 tok/s ceiling |

⇒ **Precision is the biggest lever on Thor:** ≈ 6× (NVFP4 vs W4A16) to ≈ 10× (vs eager BF16) at 2,048 tokens, and ≈ 2.5–3× at per-tick token counts (R3: ≈ 115 ms eager for 3 cameras vs ≈ 43 ms estimated NVFP4) — larger than any architectural choice among the shortlist. The price is an *unmeasured* embedding drift (§4.2-4).

### 5.2 FLOPs per tick — 256×640 frame = 640 patches → 160 merged tokens; +64 ego/query tokens (HYPOTHESIS)

| backbone | body / ViT | 1 cam × 1 frame | 3 cam × 1 frame | 3 cam × 3 frames |
|---|---|---|---|---|
| **R1/R2: Qwen3.5-4B ≡ Qwen-Drive VLM** | LLM body ≈ 3.5 B (ESTIMATED: 9.1 GB dir − ViT − 248,320-vocab embeddings; hidden size **UNVERIFIED**); ViT ≈ 0.35 B (assumed from the 10.9 ms) | ViT 0.49 + LLM 1.57 = **2.06 TFLOP** | 1.46 + 3.81 = **5.27 TFLOP** | ViT 4.4 + LLM ≈ 11 ≈ **15 TFLOP** (LLM sees 1,504 tokens) |
| **R3: Cosmos3-Edge reasoner** | body **1.41 B** (§1.4), ViT ≈ 0.41 B | 0.58 + 0.63 = **1.21 TFLOP** | 1.73 + 1.53 = **3.26 TFLOP** | ViT 5.2 + LLM 4.2 ≈ **9.4 TFLOP** |
| ref: Qwen3-VL-8B (≡ Cosmos-Reason2-8B) | body 6.95 B, ViT 0.41 B | 0.58 + 3.11 = **3.69 TFLOP** | 1.73 + 7.56 = **9.29 TFLOP** | ≈ **26 TFLOP** |

### 5.3 Thor latency estimate (ms, p50-style, TRT NVFP4 LLM + FP16 ViT; **excludes** head, ego encoder, pre-processing, engine hand-off, CUDA-graph gains)

Scaled linearly in patches (ViT) and interpolated between the measured 292- and 2,048-token prefill rows (LLM). `ESTIMATED` from `MEASURED-by-NVIDIA` anchors; ±25 % is my honest band.

| scenario | LLM tokens | Qwen3-VL-2B (≈ R3 LLM) | Qwen3-VL-4B | **R1/R2 Qwen3.5-4B** | Qwen3-VL-8B |
|---|---:|---:|---:|---:|---:|
| 1 cam × 1 frame | 224 | 16.6 | 24.1 | **30.8** | 34.1 |
| 3 cam × 1 frame (≡ 1 cam × 3 frames, no cache) | 544 | 35.5 | 49.0 | **60.5** | 70.9 |
| 1 cam × 3 frames, **ViT-feature cache** (1 new frame) | 544 | 21.8 | 35.0 | **47.3** | 52.0 |
| 3 cam × 3 frames, ViT cache (1 new frame/cam) | 1,504 | 43.9 | 70.9 | **95.2** | 110.5 |

* **R3 (Cosmos3-Edge) TRT estimate** = Qwen3-VL-2B prefill (same 1.4 B body) + an SO400M ViT (9.5 ms/frame): **19 / 43 / 24 / 52 ms** for the four rows — *assumes a working NVFP4 TRT build for Edge (UNVERIFIED)*. **Measured** eager-BF16 alternative (P): 0.21 ms per prompt token all-in ⇒ ≈ **47 ms** (1 cam) / **≈ 115 ms** (3 cam) — i.e. eager BF16 alone does *not* make 3 cameras at 10 Hz.
* **Budget reading (10 Hz = 100 ms):** with NVFP4, **4 B-class fits 3 cameras × 1 frame with ~40 % margin (49–61 ms)**; 3 × 3 frames is at the edge (71–95 ms) *before* the ~10–20 ms of head/pre-processing/hand-off ⇒ needs the ViT cache **and** KV/token reuse or token pruning (§5.4). 8 B does not fit 3 × 3.
* **Generative-expert surcharge (Qwen-Drive style):** 1.0 B DiT × 50 waypoint tokens × 10 Euler steps ≈ 2·1.0e9·50·10 = **1.0 TFLOP ⇒ ≈ 17–25 ms at 40–60 TFLOPS** (FP16, ESTIMATED). CLM-style **scoring is prefill-only** (heads 2 × 9.4 M ⇒ < 1 ms; 4,096 × 512 cosine ≈ 2 MFLOP) — a real ~20 ms and a sampling-variance advantage, *if* scoring matches the expert's quality (§6.3).
* **Alpamayo-1.5 (8 B FP16 body) per tick:** 2·7e9·544 tokens ≈ 7.6 TFLOP at the measured FP16 rate (≈ 43 TFLOPS) ≈ **180 ms** for 3 cameras — plus a 2.3 B expert — ⇒ **teacher, not per-tick encoder** (consistent with §1.6).
* **Not measured anywhere:** end-to-end tick with two TRT engines (ViT → LLM) sharing Thor's 20 SMs — TanitAD's own finding is that Thor saturates by batch 8 (`INHERITED`, CLAUDE.md) so ViT/LLM overlap buys little.

### 5.4 Token-compression methods compatible with a *frozen* VLM

| method | where it acts | frozen-compatible? | Thor / TRT fit | evidence | verdict for HiCAP |
|---|---|---|---|---|---|
| **fixed ROI / resolution choice** (crop sky+hood, 256×640 vs 384×960) | pixels | exact | trivially static | — | first lever; the resolution test in §6.3 |
| **video path** (temporal patch 2 ⇒ 2 frames per token block) | ViT input | native | TRT Edge-LLM serves native video (P) | Qwen README | halves ViT tokens for paired frames — free |
| **post-ViT static-ratio prune/merge** (VisionZip, PruMerge, ToMe-style: R; HiPrune 2508.00553: **93.0 %** of Qwen2.5-VL score at **11.1 %** tokens; ZOO-Prune 2509.24837: **96.2 %** at 20 % on Qwen2.5-VL-7B — S) | between ViT and LLM | training-free | fixed K ⇒ CUDA-graph/TRT friendly (engine profile has `--minImageTokens/--maxImageTokens`, P) | VQA benchmarks **only** — not metric state | cuts LLM prefill (55–75 % of a 4 B tick) but **not** the ViT; must be re-validated on the four families |
| in-LLM attention-score pruning (FastV, PyramidDrop) | layers K… | training-free | needs attention maps ⇒ breaks fused/FlashAttention kernels | S titles | **avoid on Thor** |
| pre-ViT pruning ("PixelPrune", vLLM issue #39708, S title) | pixels | training-free | dynamic shapes | not read | the only method that cuts ViT cost; unproven |
| learned latent-query compression (SLQ/BToks-style; Q-Former) | after ViT or in-LLM | backbone frozen, **interface trained** (36 k params in SLQ) | static | S | consistent with §3.2; candidate for the head, not a backbone change |
| FP8 ViT (`--visual_quantization fp8`, P) / NVFP4 LLM | weights/activations | needs drift check | TRT Edge-LLM native | P (support) | required for the budget; **measure embedding drift** |

---

## 6. Recommendation (priority 6)

### 6.1 Ranked shortlist (rank = recommended test order; all three are *reference/teacher-class* — see 6.4)

| rank | slot | backbone | exact reasons | Thor (P / ESTIMATED) | risks | first experiment |
|---|---|---|---|---|---|---|
| **R1** | **general** | **Qwen3.5-4B** (Apache-2.0, natively multimodal, GDN hybrid) | (i) knee of MMMU-Pro vs measured Thor-ms (§2); (ii) **same architecture as R2 ⇒ the cleanest possible "does driving knowledge help a frozen interface" ablation**; (iii) far less likely than R2 to have trained on PhysicalAI (HYPOTHESIS — web-scale pretraining data is undisclosed, so run the same E-A2 memorisation probe) ⇒ **the best available contamination-clean control**; (iv) measured Thor rows; (v) the same family ships 0.8/2/4/9 B ⇒ a **size-scaling ladder** with one recipe | ViT 10.9 + prefill 31.6 (292 tok, P); **≈ 60 ms** 3-cam × 1 frame (E) | GDN prefill ≈ 1.5× Qwen3-VL; last-token has no metric ego-motion (§4.2); NVFP4 drift; 4 B ≈ 13× the 300 M budget; **exact HF config unread** | E-A0 |
| **R2** | **driving-aware** | **Qwen-Drive-1.0-4B VLM** (Apache-2.0; `Qwen-Drive-1.0-4B/` root = 9.1 GB VLM; expert 2.1 GB and BEV 0.5 GB are *separable*) | only public model that **kept general ability (MMBench 85.5 vs 87.1) and added driving** (LingoQA 77.8 vs 70.4; Ego3D RMSE 7.78 vs 13.17; PAI-CoC 41.3 vs 2.6) — the "general + driving knowledge" the PI asked for; ships a **reference expert-on-KV interface** to compare heads against; same Thor row as R1; permissive licence | as R1 | **3 weeks old, vendor-run, unreproduced; PhysicalAI training likely (HYPOTHESIS: CoC 41.3 vs 2.6 base implies PAI-AV CoC labels were trained on)** ⇒ contamination (§6.2); frozen-planner claim is SECONDARY; driving tuning may have moved the space away from pure-vision metric cues | E-A0 (paired vs R1) |
| **R3** | **small / edge** | **Cosmos3-Edge reasoner (2.4 B)** (OpenMDW-1.1) | NVIDIA-built for Thor/Orin (P); SigLIP2-SO400M + Nemotron-dense + mRoPE video; smallest with world-model/physical-AI mid-training (yaml `…midtrain`); body FLOPs = Qwen3-VL-2B's ⇒ **19–52 ms est.**; eager-BF16 numbers already published on 5 Jetson SKUs (P) | eager 190 ms / 911 tok (P); TRT est. 19–52 ms (E, needs build) | **no audio, no driving tuning shown, patch size & HF config unread, TRT NVFP4 path for Edge unverified, Transformers-`main`-only loader (P)** | E-A1 (Thor build) then E-A0 |
| alt-audio | small + audio | Gemma-4-E2B/E4B | only small native-audio model with Thor rows (ViT 16.7 + pf 24.8/36.1 ms) | measured (P) | licence UNVERIFIED; no PhysicalAI audio to train on | only if an audio corpus exists |
| teacher | driving-aware, big | Alpamayo-1.5-10B (OpenMDW-1.1), Alpamayo-2-Super, Cosmos3-Nano/Super | driving-SFT'd (80–115 k h), CoC labels, NVIDIA-supported quant recipes; **auto-labeller/teacher** per the 2026-08-05 analysis | ≈ 180 ms / 3-cam FP16 (E) | contamination; too big per tick | offline labelling / score distillation |

### 6.2 ⛔ Contamination check — MEASURED (ours), 0 GPU

**Question:** are TanitAD's PhysicalAI clips inside the pool a driving-aware backbone trained on? **Test:** exact clip-UUID intersection between (a) NVIDIA's two public clip lists shipped in `alpamayo-recipes@ae5bc10/recipes/alpamayo1_5_quant/` — `0417_5k_train_set_for_calibration_25.10.parquet` (**5,000 clips, all `split == train`**, used to calibrate the quantised Alpamayo-1.5) and `1005_7cam_gold_eval_metadb_public.parquet` (**644 clips** — same count as Qwen-Drive's "PhysicalAI 644-example split", which its docs define as "the clips listed in alpamayo-recipes"; identity inferred, not diffed) — and (b) TanitAD's `2026-07-22-idm-proof/rig_table.json` (**2,400 train clip ids**) and `2026-08-02-v2-clean-val-selector/v2_clean_val_manifest.json` (**400 clean-val clip ids**, `disjoint` from the 9,000-clip v2 train pool). Script: Appendix B.

| set | n | ∩ NVIDIA **train** (5,000-clip sample) | Wilson-95 % | ∩ NVIDIA **gold-eval** (644) |
|---|---:|---:|---|---:|
| TanitAD train (2,400) | 2,400 | **76 (3.17 %)** | [2.54, 3.95] % | 17 |
| TanitAD clean-val (400) | 400 | **13 (3.25 %)** | [1.91, 5.48] % | 2 |
| NVIDIA train ∩ NVIDIA gold-eval | — | 11 | — | — |

**Reading (each step labelled).** (1) `MEASURED`: **≥ 13 of our 400 clean-val clips were in NVIDIA's *training* list** — a **lower bound**, because the list is only a 5,000-clip sample of what Alpamayo trained on. (2) `MEASURED`: train and val hit-rates are **identical within noise (3.17 % vs 3.25 %)** ⇒ our val is drawn from the *same public pool* as our train and as NVIDIA's train sample — **it is not held out from Alpamayo's pool** (`ESTIMATED` inference; consistent with our own note that PhysicalAI-AV ships no drive/session id, so disjointness is provable only at clip level). (3) `MEASURED`: **2/400 val clips are in NVIDIA's gold-eval set** (fine for us, but it means NVIDIA/Qwen numbers and ours partly share windows). (4) **Not testable here:** the canonical **40-episode** eval set — `manifest_EVALPOD_val40.json` stores only the 4-ASCII-char clip-id prefix as `episode_id` (16 bits; a prefix screen gave 4/40 hits vs 2.95 expected by chance, p = 0.34 — uninformative). **Action (0-GPU, first thing): recover the 40 exact clip UUIDs from the pod epcache, rerun Appendix B.**
**Consequence:** any driving-aware arm (R2, Alpamayo, Cosmos-Reason2-tuned) is *presumed contaminated* on our val until decontaminated; R1 is the clean control; a driving-aware "win" is admissible only on windows whose clip ids are absent from NVIDIA's lists **and** (HYPOTHESIS) from the undisclosed remainder — which we cannot verify. This is the same class as the REF-A I-JEPA leak and the Alpamayo-2 note (§9 of that analysis).

### 6.3 Cheapest discriminating experiments (proposals for the PI; none run; outcomes committed before any data)

**E-A0 — frozen-feature panel (≈ 1 A40-day; 0 Thor).** Arms: **R1, R2, R3, Qwen3-VL-2B (R3 latency proxy), F-own (TanitAD frozen v1 encoder, control), SigLIP2-B mean-pool (CoVer-style, non-LLM control)**, plus the same-family **Qwen3.5-0.8B/2B/9B ladder**. Features per arm: (a) last token after a fixed driving-question suffix; (b) 3-depth learned mix (L/2, 3L/4, L); (c) 8 learned latent queries (SLQ-style). **Identical small head** (pre-registered: ridge + 2-layer MLP), **ego numerics injected at the head** (not via the VLM), targets minted at label time (target speed@2 s, curvature@2 s, factored lat/lon tactical class). Parity corpus only (`physicalai-train-e438721ae894`, skip-hash `f09e44db`; val = the 40 episodes, **no re-selection**). Score all **four families**, paired **episode-cluster bootstrap** (`taniteval/ci.py`), split steady / non-steady, versus **hold-speed and a VINN k-NN over the same frozen embedding** (the zero-training falsifier).
Committed outcomes:
* **O1** — Δ(R2 − R1) paired CI excludes 0 for R2 on non-steady longitudinal **and** the E-A2 decontamination check is clean ⇒ driving knowledge transfers through a frozen interface ⇒ adopt the R2 family as teacher; R1 stays the control in every report.
* **O2** — Δ(R2 − R1) CI spans 0 with half-width ≤ the pre-registered MDE (set from F-own's bootstrap width) ⇒ driving tuning adds nothing at the frozen-feature level ⇒ **choose R1** (clean, cheaper licence path) and spend the effort on the interface, not on driving-aware weights.
* **O3** — R2 > R1 but decontamination fails/undetermined ⇒ **inadmissible**; report only as "contaminated upper bound".
* **O4 — falsifier** — every VLM arm fails to beat hold-speed + F-own on non-steady windows (the frozen DINOv2/I-JEPA outcome) ⇒ pooled frozen-VLM state encoding is refuted for metric decisions ⇒ pivot to the **KV-read expert** (Qwen-Drive style) or a **LoRA-adapted** backbone; the freeze premise is dropped.
* **Scale ladder rule:** any size exponent from {0.8, 2, 4, 9 B} must carry **fit window, R² and n (n = 4 ⇒ no quotable exponent below R² 0.80; use matched-size ratios)** per CLAUDE.md; four points cannot decide a restart.
* **Side arms (same features):** resolution 256×640 vs 384×960 (160 vs 360 tokens); precision BF16 vs FP8/NVFP4 (**drift = cosine to BF16 + head-accuracy retention**); pooled-vector vs KV-read head at matched parameters.

**E-A1 — Thor latency (0 A40; 1 Thor session).** TRT-Edge-LLM 0.10.0 builds of R1, R2, R3 (+ Qwen3-VL-2B) at 256×640, 1 and 3 cameras, NVFP4 and FP8, p50/p99 over ≥ 50 iterations (the THOR_PROFILE harness), **`torch.cuda.max_memory_allocated()` only** for memory. Rule: **3-cam p99 ≤ 60 ms ⇒ per-tick viable; 60–100 ms ⇒ 5–10 Hz with ViT cache; > 100 ms ⇒ System-2/teacher only** (DriveVLM-Dual pattern).

**E-A2 — decontamination (0-GPU).** Recover the 40 exact clip UUIDs; rerun Appendix B; additionally probe memorisation (R2 CoC-style prompt on val vs never-seen clips: log-likelihood gap) — if the gap is significant, R2 is contaminated regardless of list membership.

### 6.4 Distillation / pruning towards a sub-300 M student

⚠️ **Scope decision for the PI (not mine):** every candidate is 2.4–4.5 B (8–15× the target), and the ViT alone (0.3–0.4 B) already exceeds 300 M. Two coherent positions: **(a) REF-F is a *reference/teacher* arm** (like REF-A…E) and the deployable sub-300 M student is distilled from it; **(b) the frozen VLM runs as an external Thor service and only trainable heads count** — this redefines the programme's claim and I do **not** recommend it silently.

| path | what | evidence | fit |
|---|---|---|---|
| **D1 score-level listwise distillation** | student (TanitAD encoder + pool + FiLM(question)) sees pixels + ego; target = teacher heads' **soft per-family softmax over the K candidates**, cached offline | Hydra-MDP: distilling per-candidate teachers, **a scalar score (80.2) < imitation-only (80.9) < five separate heads (83.0)** (INHERITED, F-R doc) ⇒ keep **per-family** targets | **recommended** — the deliverable is decisions, not embeddings; no embedding-space matching |
| **D2 embedding distillation** | regress the teacher's *question-conditioned* pooled embedding (cosine + InfoNCE) | multi-teacher vision students absorb 7 B teachers: C-RADIOv4 (SO400M/H; teachers SigLIP2-g-384, DINOv3-7B, SAM3; P via F-R doc); **no published example through an LLM in the loop** | HYPOTHESIS |
| **D3 depth/width pruning of the frozen VLM** | drop layers of R1/R3's LLM, re-align heads (Minitron/ShortGPT-style, R) | NVIDIA's Cosmos3-Edge reasoner is itself a compressed Nemotron-dense line (28 pairs, hidden 2048) (P) | gets 2.4 B → ~1.2 B, **not** 300 M; an intermediate rung |
| **D4 NVIDIA's own recipe** | teacher (Alpamayo-1.5/2-Super) → **0.5–2 B** student on DRIVE Orin/Thor | S (NVIDIA page via search summary) | the 2 B end is still ~7× the target ⇒ confirms the gap is structural |
| **D5 cached-target economics** | heads *and* students train on cached teacher outputs; one teacher pass over the parity corpus | teacher 3-cam ≈ 50–60 ms ⇒ ≈ 17–20 windows/s ⇒ **100 k windows ≈ 1.5–2 h** (ESTIMATED) | makes D1/D2 cheap |

---

## 7. Implications for HiCAP (≤ 12)

1. **Name the backbone precisely.** "Cosmos Drive nano" = Alpamayo-1.5-Nano (driving, 10 B, teacher-class) ⊕ Cosmos3-Edge (Thor-sized, 2.4 B reasoner, **Nemotron-dense + SigLIP2, not Qwen**, no audio, no driving tuning). No single model has both; do not write a plan that assumes one does.
2. **Order of test: R1 general → R2 driving-aware (same architecture) → R3 Cosmos3-Edge**; Alpamayo/Cosmos3-Nano/Super are teachers and labellers.
3. **Decontaminate first.** ≥ 3.25 % of our clean-val clips are in NVIDIA's *training* list (13/400, a lower bound); our val is not held out from Alpamayo's pool. Recover the 40 exact UUIDs (0-GPU) before any driving-aware number is quoted; R1 is the clean control in every table.
4. **Do not copy CLM's bare last-token.** Every recent frozen-backbone paper (SLQ, CoVer, FreeRet, NV-Embed) wins with a **learned aggregator**; both driving expert designs (Alpamayo, Qwen-Drive) read **KV caches of many layers**. Expose ≥ 3 depths; test pooled-vs-KV-read at matched parameters.
5. **Ego kinematics enter as numeric channels at the head/expert** (Qwen-Drive: MLP-encoded history + AdaLN; Alpamayo: history-trajectory tokens). Frozen VLMs and frozen DINOv2/I-JEPA do not read metric ego-motion (EgoDyn-Bench; our own pilot: 45 % band agreement, 38 % repaint flips). The CLAUDE.md goal-path rule stands: nothing from the situation classifier enters the goal input.
6. **Question-as-suffix + prefix caching is the free multi-head mechanism**: four family questions (longitudinal / lateral / tactical / strategic) cost ≈ ≤ 2 ms on top of one prefill — it matches the binding four-family reporting and never pools them into one score.
7. **Thor budget is feasible but precision-bound:** NVFP4 makes 3 cameras × 1 frame ≈ 49–61 ms for a 4 B model; 3 × 3 frames ≈ 71–95 ms needs the ViT cache + token pruning; eager BF16 is ≈ 2.5–3× slower at these sizes. **The unmeasured risk is embedding drift under NVFP4/FP8** — train heads on features of the deployed precision and report drift.
8. **Scoring beats sampling on cost**: prefill-only heads save ≈ 17–25 ms/tick versus a 1 B flow-matching expert and remove sampling variance — but the expert interface is the published-strong one (PDMS 90.7); the E-A0 side arm decides.
9. **Audio is currently untrainable here**: PhysicalAI-AV has no audio channel (36 features, `grep audio` = 0). Keep the door open with Gemma-4-E2B (native, measured on Thor) or a frozen ASR-class encoder + projector, but treat audio as HYPOTHESIS until an audio-bearing corpus exists.
10. **"Frozen general + driving" is a bet, not a replication:** no published raw-frozen-VLM driving result exists; Alpamayo and Qwen-Drive freeze *after* driving training. Pre-commit O4 (the falsifier) so a refutation redirects effort to KV-read/LoRA instead of being argued away.
11. **Sub-300 M cannot hold for R1–R3** (2.4–4.5 B; ViT alone > 300 M): treat REF-F as a reference/teacher arm and distil with **D1 (per-family score distillation)**; the scope call is the PI's.
12. **Never quote size/latency exponents bare**: the {0.8, 2, 4, 9 B} ladder has n = 4; report matched-size ratios, and any NVIDIA/Qwen number as vendor-run (`PUBLISHED-SECONDARY`/`INHERITED`), never as a TanitAD result.

---

## 8. Gaps and citation status (fail loud)

| gap | consequence | how to close |
|---|---|---|
| Cosmos 3 tech report, Alpamayo-R1 paper, Qwen-Drive report, Qwen3-VL/Omni reports **not readable** (egress) | no ablation tables (frozen-vs-unfrozen, token counts, resolution) | fetch from an unblocked box; the Qwen-Drive report holds the frozen-VLM BEV and planner-stage ablations |
| Qwen3.5-4B / Qwen-Drive **hidden size, layer types, tied embeddings** unverified | body-FLOPs (3.5 B) is an estimate | read `config.json` (HF blocked; Transformers config class needs the checkpoint) |
| Cosmos3-Edge ViT patch size, TRT NVFP4 build, driving content of mid-training | 160-token and 19–52 ms figures conditional | E-A1 |
| Qwen-Drive "planner trained with frozen VLM" | SECONDARY only (search summary) | tech report / training config |
| Licences UNVERIFIED: Gemma-4, Nemotron-Nano-VL/Omni, Cosmos-Embed1, Qwen3.5-Omni-Light, Alpamayo-1 (R1-10B) | do not ship on these | read the licence files |
| Qwen3.5 (Mar 2026), Gemma-4, Qwen3.8 facts are one-hop (search) | numbers (MMMU-Pro, 150 M/305 M encoders) not primary | model cards |
| 40-episode val exact clip ids missing from repo manifests | §6.2 test incomplete for the canonical val | recover from pod epcache |
| InternVL3.5, Phi-4-MM, SmolVLM2, E5-V, NV-Embed, Minitron/ShortGPT, EMMA/SimLingo/OpenDriveVLA facts are **RECALLED** | not load-bearing; re-verify before use | — |
| WebSearch budget (30) exhausted; all 6 WebFetch attempts hit egress blocks (only GitHub, via `git clone`/`curl`, was reachable) | breadth-first, not exhaustive | — |

**Citation status** — verified this session by PRIMARY read or search result list (id/title seen): Alpamayo-R1 **2511.00088** ✔ (search + repo READMEs) · Cosmos-Reason1 **2503.15558** ✔ (result list) · Cosmos 3 tech report (`research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf`; arXiv **2606.02800** "Cosmos 3: Omnimodal World Models for Physical AI" seen in a result list — id ✔, content ✗) · Qwen3-VL **2511.21631** ✔ · Qwen3-Omni **2509.17765** ✔ · Qwen3-VL-Embedding (repo, no arXiv id) ✔ · Qwen-Drive-1.0 **2609.00111** ✔ (repo citation block, P) · VLM2Vec **2410.05160**, V2 **2507.04590** ✔ · jina-v4 **2506.18902** ✔ · Omni-Embed-Nemotron **2510.03458** ✔ · FreeRet **2509.24621** ✔ · SLQ **2604.13710** ✔ · BToks **2604.11095** (title) · Skean **2502.02013** ✔ · EgoDyn-Bench **2604.22851** ✔ · Cosmos-Drive-Dreams **2506.09042** ✔ · Nemotron Nano V2 VL **2511.03929** ✔ · Qwen3.5-Omni **2604.15804** (title) · HiPrune **2508.00553**, ZOO-Prune **2509.24837** ✔ (result lists) · "Encoder Winners…" **2606.14153** (title) · CoVer-VLA **2602.12281**, VLA-R **2511.12405**, DriveVLM-Dual **2402.12289** (inherited from the F-R stream). **E5-V "2407.12580"** is from memory — **UNVERIFIED id.**

## 9. Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-hicap/RA_vlm_backbones.md` (this file) | **repo working tree, staged** (`git ls-files --stage` verified) | in the repo; **not committed** (rule 1) |
| cost-model script | **embedded verbatim as Appendix A** (also `…/scratchpad/hicap/appA.py`, session scratch — *not* a deliverable) | not stranded |
| contamination-overlap script + result | **embedded verbatim as Appendix B**; inputs are repo files + NVIDIA's public parquet (`alpamayo-recipes@ae5bc10`) | not stranded; re-run needs `pyarrow` |
| shallow clones used (cosmos, cosmos-framework, cosmos-reason2, alpamayo*, TensorRT-Edge-LLM, Qwen*, Qwen-Drive-1.0, CLM, cover-vla) | session scratchpad only (reproducible at the commits named in §1.1) | **exists in one place; disposable by design** |

**Escalations (headline, not a README plea):** (1) **PI decision — sub-300 M scope** for a frozen 2.4–4.5 B arm (§6.4); (2) **before any driving-aware arm runs — decontaminate** (recover the 40 exact clip UUIDs; §6.2); (3) **a Thor session** for E-A1 (TRT builds of R1/R2/R3; no Thor is available to this stream); (4) integration: nothing to merge — this file is input to the HiCAP design synthesis.

---

## Appendix A — cost model (`MEASURED (ours)` arithmetic on NVIDIA-published anchors; runs as written)

```python
# Appendix A -- HiCAP R-A cost model (numpy-free). Anchors = NVIDIA TRT-Edge-LLM v0.10.0 Thor rows (batch 1).
A = {  # vit_ms@265tok, prefill_ms@292 (NVFP4), prefill_ms@2048 (NVFP4)
 "Qwen3-VL-2B": (11.4, 12.7, 28.0), "Qwen3-VL-4B": (11.6, 22.3, 62.2),
 "Qwen3-VL-8B": (15.7, 32.1, 104.5), "Qwen3.5-4B": (10.9, 31.6, 95.2)}
def vit(m, patches): return A[m][0] * patches / (265 * 4)          # linear in patches
def pf(m, n):                                                       # interpolate 292 -> 2048; scale below 292
    a, b = A[m][1], A[m][2]
    return a * max(n, 100) / 292 if n <= 292 else a + (b - a) * (n - 292) / (2048 - 292)
PATCH, TOK, TEXT = 640, 160, 64          # 256x640 frame: 16x40 patches -> 160 merged tokens; +64 ego/query tokens (HYPOTHESIS)
S = [("1cam x1f", 1, 1, 1), ("3cam x1f | 1cam x3f (no cache)", 3, 1, 1), ("1cam x3f, ViT cache", 1, 3, 1),
     ("3cam x3f, ViT cache (1 new frame/cam)", 3, 3, 1)]     # (name, cams, frames, new_frames_per_cam)
for m in A:
    print(m)
    for name, c, f, nw in S:
        if name.startswith("3cam x1f"): c, f, nw = 3, 1, 1
        n = c * f * TOK + TEXT; v = vit(m, c * nw * PATCH); p = pf(m, n)
        print(f"  {name:40s} tok {n:5d}  ViT {v:5.1f}  prefill {p:6.1f}  total {v+p:6.1f} ms")
# FLOPs = 2*P_body*N + 4*L*N^2*d ; ViT = 2*P_vit*patches + 4*L_v*patches^2*d_v
for k, (pv, lv, dv, pl) in {"Qwen3.5-4B/Qwen-Drive (ViT~0.35B assumed, LLM body~3.5B est.)": (0.35, 24, 1024, 3.5),
                            "Cosmos3-Edge (ViT 0.41B, body 1.41B)": (0.41, 27, 1152, 1.41),
                            "Qwen3-VL-8B (ViT 0.41B, body 6.95B)": (0.41, 27, 1152, 6.95)}.items():
    for cams in (1, 3):
        vf = (2 * pv * 1e9 * PATCH + 4 * lv * PATCH**2 * dv) * cams / 1e12
        lf = 2 * pl * 1e9 * (cams * TOK + TEXT) / 1e12
        print(f"{k:62s} {cams}cam: ViT {vf:.2f} + LLM {lf:.2f} = {vf+lf:.2f} TFLOP")
# effective throughput implied by anchors (TFLOPS): prefill@2048 NVFP4 vs INT4-AWQ, Qwen3-VL-8B (body 6.95B)
print("8B pf@2048 NVFP4 %.0f TFLOPS ; AWQ %.0f TFLOPS" % (2*6.95e9*2048/0.1045/1e12, 2*6.95e9*2048/0.6622/1e12))
```

## Appendix B — clip-UUID overlap with NVIDIA's public clip lists (`MEASURED (ours)`; run from `TanitAD Research Hub/` after replacing `R` with the clone path; needs `pyarrow`)

```python
import json, math, pyarrow.parquet as pq
R = '<alpamayo-recipes@ae5bc10>/recipes/alpamayo1_5_quant/'
nv_tr = {r['key'] for r in pq.read_table(R+'0417_5k_train_set_for_calibration_25.10.parquet').to_pylist()}   # 5000, split=='train'
nv_ev = {r['key'] for r in pq.read_table(R+'1005_7cam_gold_eval_metadb_public.parquet').to_pylist()}          # 644
rt = json.load(open('Architecture & Inference/Implementation/incoming/2026-07-22-idm-proof/rig_table.json'))
ours_tr = {(v['clip_id'] if isinstance(v, dict) else eval(v)['clip_id']) for v in rt.values()}                  # 2400
ours_va = set(json.load(open('Data Engineering/Implementation/incoming/2026-08-02-v2-clean-val-selector/v2_clean_val_manifest.json'))['clip_ids'])  # 400
def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = p+z*z/(2*n); a = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-a)/d, (c+a)/d
for nm, S in (('train2400', ours_tr), ('cleanval400', ours_va)):
    k = len(S & nv_tr); print(nm, k, '/', len(S), 'Wilson95', wilson(k, len(S)), 'gold-eval', len(S & nv_ev))
# result 2026-09-29: train2400 76/2400 (3.17%) [2.54,3.95]%, gold-eval 17 | cleanval400 13/400 (3.25%) [1.91,5.48]%, gold-eval 2 | NVIDIA train∩gold-eval 11
```

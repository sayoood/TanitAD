# HiCAP stream RF — a native omni backbone vs. an audio adapter (2026-09-29)

**Status:** research note, staged in the working tree, **not committed**. Owner: stream RF. It **updates** `RD_cameras_tokens_audio.md` §4 and the audio rows of `RA_vlm_backbones.md`; it edits neither. No TanitAD number here comes from a summary; every number carries an evidence tag (legend below).

**Trigger (PI rulings, 2026-09-29).** (a) a VLM is the preferred backbone; (b) audio may optionally be used at inference; (c) the sub-300 M cap is removed *if whole-system inference time is very low*. The original request was *"audio as additional tokenized input, brought to the same embedding space as language and vision."* The current design (`Paper/HiCAP/sec_elastic.tex` L65–L88) reaches that with an **adapter**: a frozen ≈90 M tower → 8 tokens per 2 s → a zero-initialised gated cross-attention residual on the level-query outputs of the fusion transformer, with a frozen vision–language-only backbone. With the cap lifted, a **native omni** backbone (audio tokens inside the LLM's own input space) is a real alternative. This note compares them **for HiCAP's actual shape**: prefill-only backbone, refreshed at 2 Hz, hidden states at three depths pooled to ≤128 tokens, ≈27 M trainable fusion + heads at 10 Hz, Jetson Thor (273 GB/s, 128 GB unified).

**Evidence tags.** `[P]` primary source I read (path:line @commit, or a file I fetched from `raw.githubusercontent.com`) · `[S]` search-result summary only — the page itself is egress-blocked, so **not admissible for any GPU-day decision** · `[M]` measured/derived by me (grep, directory listing or the script in Appendix A) · `[E]` estimated · `[H]` hypothesis · `[R]` recalled from training data, not re-seen · `[I]` inherited from a sibling report, not re-verified · `[U]` unverified.

---

## 0. Headline (read this first)

1. **Recommendation: keep the gated-residual adapter as the primary audio path for HiCAP v1; run native omni as a pre-registered head-to-head arm on *one* small backbone (§2.6, E-N1) rather than choosing the backbone for its audio.** Four reasons, in order of weight: (i) the adapter keeps *every* audio-control arm on the cached VLM tokens, whereas a native path needs one backbone prefill per control arm (§2.4); (ii) a native tower is **mono**, so it cannot carry bearing — the bearing/DoA path stays a small side-channel adapter either way (§2.1); (iii) the parity claim "audio-null is bit-exact" is *free* for the adapter and only *by construction, not by test* for native (§2.3); (iv) a 30 B-A3B omni backbone raises the 2 Hz refresh duty from ≈0.18 to ≈0.29–0.42 (§1.3, `[E]`).
2. **What native buys (and the adapter does not):** the tower+projector were trained *with the LLM on real audio*, so the PI's "same embedding space as language" holds with **zero alignment training on our synthetic data**, and the model can be *asked* about a sound in language (a free zero-shot detector and a free labeler for real recordings). Qwen3-Omni additionally time-aligns audio with video (TMRoPE, `position_id_per_seconds = 25`). These are the things E-N0/E-N1 test.
3. **Cheapest route to native audio is not a 30 B MoE:** **Gemma-4-E4B** ships a native audio tower and has Thor rows at R1-class latency (NVFP4: ViT 16.7 ms + prefill 36.1 ms @292 tokens vs R1 Qwen3.5-4B 10.9 + 31.6 ms) `[P]` (TRT-Edge-LLM `performance-benchmarks.md:442`; RA table). Put it in the backbone bake-off; do not swap the backbone *for* audio.
4. **Largest UNVERIFIED:** **no NVIDIA-published Thor latency row includes audio.** Every Nemotron-Omni row is a COCO *image* run; the Gemma rows are image runs; there is no Qwen3-Omni row at all (§1.2). The audio cost in §1.3 is my arithmetic. Second: **all Hugging Face model cards, `config.json` and licence files were unreadable (egress 403)** — every licence cell except Qwen3-Omni (repo LICENSE) and MiniCPM-o (README + LICENSE) is `[U]` or `[S]`. Third: the audio stream of the shipped PhysicalAI mp4s is **still unprobed** (BACKLOG E8); §3.4 gives a tested zero-download probe.
5. **Corrections to earlier text (§1.4):** Gemma-4's audio token rate is **40 ms/token (25 tok/s), not the Gemma-3n 160 ms** RD quoted; **Phi-4-multimodal has no audio runner in TensorRT-Edge-LLM** (only a ViT runner) although its supported-models page says "also accepts audio" — RA's "audio: yes (P supported list)" for Phi-4 on Thor is wrong for the C++ runtime; the earlier "Nemotron-Omni GPU memory ≈19–20 GB on Thor" is confirmed (Thor 18.9–20.3 GB) but every such row is an *image* run.
6. **Integration items (for the orchestrator, not written into any other file):** (a) `sec_elastic.tex` says audio-null "is bit-exactly the original model" — true **at initialisation and under a hard bypass**, *not* after joint training with fusion weights unfrozen (§2.3); (b) the dose-response control must be defined as **SNR**, not overall gain, because Parakeet-style front-ends erase uniform gain (§2.3); (c) PI decision: admit Gemma-4-E4B to the backbone shortlist as "the native-audio candidate"; (d) RA row for Phi-4 (TRT audio) needs the correction above.

---

## 0.1 Source ledger (what I actually read)

| source | commit / fetch | how | read? |
|---|---|---|---|
| `NVIDIA/TensorRT-Edge-LLM` | `95515c2f87fba8982db5a519f9022277667b3cc9` (v0.11.0 merge, 2026-09-29) | shallow clone (scratchpad) | `docs/source/user_guide/getting_started/supported-models.md`, `…/performance/performance-benchmarks.md` (latest results section = **v0.10.0**; 0.11.0 has none yet), `…/examples/omni.md`, `…/examples/phi4.md`, `…/features/kv-cache-reuse.md`, `cpp/multimodal/**`, `cpp/runtime/melSpectrogram.h`, `tensorrt_edgellm/models/nemotron_omni/modeling_nemotron_omni_audio.py`, `CHANGELOG.md` |
| `QwenLM/Qwen3-Omni` | `e4235853125589c789f06a2dd83e9f4126df5e9d` (2026-04-23) | shallow clone | `README.md`, `LICENSE` (Apache-2.0, "Copyright 2025 Alibaba Cloud") |
| `huggingface/transformers` | `f339035b986aaf719bc6f5ea92342f73c498cb0e` (2026-09-29) | sparse clone: `models/{gemma4,gemma4_unified,gemma3n,qwen3_omni_moe,phi4_multimodal,nemotron_h_omni,parakeet,whisper,cosmos3_omni,audioflamingo3}` + `docs/source/en/model_doc` | config defaults, processors, feature extractors, model docs |
| `NVIDIA-NeMo/Nemotron` | `8749344f8ccefd50181c56b013ddc5c8a6154a8c` (2026-09-11) | shallow clone | `docs/nemotron/omni3/{architecture,inference,sft,README}.md`, `…/stage0_sft/config/audio_text.yaml` |
| `vllm-project/recipes` | `e7eeea74f59496da5a07953beab4e0a39b2ab7c7` (2026-09-29) | shallow clone | `models/nvidia/Nemotron-3-Nano-Omni-*.yaml`, `models/Google/gemma-4-*.yaml`, `Google/Gemma4.md`, `models/microsoft/Phi-4-multimodal-instruct.yaml`, `models/thinkingmachines/Inkling*.yaml`, `models/XiaomiMiMo/MiMo-V2.5.yaml` |
| `OpenBMB/MiniCPM-o` | `6ada8e8ef5e2979670fc94406f02b87c3c7e7ee0` (2026-09-08) | shallow clone | `README.md`, `LICENSE` |
| `carla-simulator/carla` | `1360bb9aff0f1aaa6216876ceee777528128306f` (2026-09-02) | shallow clone | repo-wide audio grep, `Docs/ref_sensors.md` |
| `NVlabs/alpasim` | `affc2eab209fa43bdfa2f26c0f8d437922d78a68` (2026-08-18) | earlier clone | repo-wide audio grep, all file types |
| `NVIDIA/cosmos` | `3e3c6d61dc15d6517c4793beab5ec3894ffa07f1` (2026-09-29) | earlier clone | `README.md`, `cookbooks/cosmos3/**` |
| `audioset/ontology`, `karolpiczak/ESC-50` | `master` via `raw.githubusercontent.com`, 2026-09-29 | curl | `ontology.json` (632 classes), `README.md` (CC BY-SA 4.0), ESC-50 `LICENSE` |
| **NOT READ — egress 403 / EGRESS_BLOCKED** | `huggingface.co` (all cards/configs/licences), `arxiv.org`, `docs.nvidia.com`, `developer.nvidia.com`, `ai.google.dev`, `qwen.ai`, `vllm.ai/blog`, `nature.com` | — | every `[S]` below comes from a WebSearch result summary of such a page |

---

## 1. Candidate table — native-audio backbones (priority 1)

Scope rule I used: **audio in, text out, open weights, and small enough to be resident on a 128 GB Thor.** Models that fail one of these are listed after the tables with the reason.

### 1.1 Identity and audio path

| model | total / active | vision | audio tower → tokens | how audio enters the LLM | licence |
|---|---|---|---|---|---|
| **Qwen3-Omni-30B-A3B** (Instruct / Thinking / Captioner) | "30B-A3B" `[P]` README title; HF class defaults: 128 experts, top-8, `moe_intermediate 768`, hidden 2048 `[P]` `qwen3_omni_moe/configuration_qwen3_omni_moe.py:112-190` (layer count in the class default, 28, is **not** the checkpoint's — `[U]`). Instruct = Thinker + Talker; dropping the Talker saves ≈10 GB `[P]` README:431 | ViT depth 27 / hidden 1152, patch 16, merge 2, DeepStack taps (8,16,24) `[P]` config L79-110 (class defaults; TRT calls it "Qwen3-VL ViT" `[P]` `omni.md:19`) | **AuT**, Whisper-style 128-mel front-end `[P]` `omni.md:18` and `melSpectrogram.h:85-91`; **3 stride-2 conv stages → 13 tokens per 100 mel frames ≈ 12.5–13 tok/s** `[P]` `modeling_qwen3_omni_moe.py:151-158` + `audioRunner.cpp:169-183`; HF defaults d_model 1280, 32 layers, 20 heads, ffn 5120 ⇒ **≈0.64 B `[E]`** (checkpoint config `[U]`) | `masked_scatter` of audio features onto `audio_token_id` positions `[P]` `modeling_qwen3_omni_moe.py:2136`; **TMRoPE** absolute-time position ids, 25/s `[P]` config L194,228 | **Apache-2.0** — repo `LICENSE` read `[P]`; the *weights'* licence file on HF is `[U]` (a search summary says Apache-2.0 `[S]`) |
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** (BF16 / FP8 / NVFP4 repos) | **31 B / 3 B** `[P]` vLLM recipe `models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16.yaml:parameter_count/active_parameters`; NeMo docs say "30B" `[P]` `architecture.md:13`. Mamba2+attention hybrid MoE decoder, context 262 K | **C-RADIOv4-H** (+3D-conv, EVS for video) `[P]` `architecture.md:23-29` | **Parakeet** FastConformer: 24 × 1024, ×8 conv subsampling, 128 mel, hop 160 @ 16 kHz ⇒ **12.5 tok/s** `[P]` `modeling_nemotron_omni_audio.py:15-33`, `test_preprocess_audio.py:130-133`, HF `parakeet` defaults ⇒ **≈0.61 B `[E]`** (Appendix A); *extended with Granary + Music Flamingo (paralinguistic, music, ambient sound)* — a **vendor claim** `[P]` `architecture.md:27-29`, capability not measured by us | projector `RMSNorm → Linear(1024→4096) → ReLU² → Linear(→LLM d)` `[P]` `modeling_nemotron_omni_audio.py` (`SoundProjection`); tokens `<so_start>` + N×`<so_embedding>` + `<so_end>`, then `masked_scatter` `[P]` HF `processing_nemotron_h_omni.py:72-74,210-213`, `modeling_nemotron_h_omni.py:423-427`. Audio is a **separate segment**, not time-interleaved with video frames; the vLLM recipe exposes `use_audio_in_video` `[P]` yaml L26,113 | "**NVIDIA Nemotron Open Model License**" — *stated* in NeMo docs `[P]` `architecture.md:89-91`; **licence text not read `[U]`** (commercial/derivative terms unknown to me) |
| **Gemma-4-E2B / E4B** | 2.3 B / 4.5 B effective, **5.1 B / 8 B with Per-Layer Embeddings** `[S]` press + `[P]` vLLM recipe `parameter_count "5B"/"8B"` (recipe also writes `active_parameters` = total, so ignore it) | ≈0.15 B: HF class default 16 layers × 768 `[P]` `configuration_gemma4.py` (Gemma4VisionConfig) ✓ RA's "150 M" `[S]`; 70–1,120 soft tokens per image, default 280 `[P]` `gemma4.md` | **conformer, HF class defaults: 12 layers, hidden 1024, 8 heads, *chunked attention: chunk 12, left context 13, right context 0*** `[P]` `configuration_gemma4.py:29-74` (checkpoint value `[U]`); **40 ms per token = 25 tok/s** `[P]` `processing_gemma4.py:63,72-75`; ≈0.305 B `[S]`, output projection dim 1536 | soft tokens between `boa`/`eoa` tokens, count `ceil(ms/40)` `[P]` `processing_gemma4.py:192-203,287-` | **Apache-2.0** per two press reports `[S]`; HF LICENSE **not read `[U]`** |
| **Gemma-4-12B "Unified"** | 12 B dense `[P]` recipe | **encoder-free**: 48×48 merged pixel patches → `LayerNorm→Dense→LayerNorm`, default 280 tokens `[P]` `gemma4_unified.md:24-58` | **no audio tower**: raw 16 kHz samples cut into **640-sample frames (40 ms)**, `RMSNorm → Linear`, token count `ceil(n/640)` ⇒ **25 tok/s** `[P]` `gemma4_unified.md:63-72` | the same `Gemma4UnifiedMultimodalEmbedder` for image and audio | as Gemma-4 (`[S]`/`[U]`) |
| **Phi-4-multimodal-instruct** | 5.6 B `[P]` recipe | SigLIP-400 M `[R]` | conformer, HF defaults 24 blocks, hidden 1024, ffn 1536, 16 heads, `time_reduction 8`, 80 mel `[P]` `phi4_multimodal/configuration_phi4_multimodal.py` AudioConfig ⇒ ≈0.4 B `[E]`, **≈80 ms/token only if the hop is 10 ms `[U]`** | **Mixture-of-LoRAs** routed per modality `[P]` HF `phi4_multimodal.md` | MIT `[R]` — not re-seen |
| **MiniCPM-o 4.5** | 9 B `[P]` README:374 (SigLIP2 + Whisper-medium + CosyVoice2 + Qwen3-8B) | SigLIP2 | Whisper-medium encoder; continuous audio+video streams, full-duplex `[P]` README:383 — token rate `[U]` | end-to-end streaming; details `[U]` | **Apache-2.0**, weights *and* code — README:2688 + repo `LICENSE` read `[P]` |

**Not usable, and why**

| model | reason |
|---|---|
| **Qwen3.5-Omni** Plus / Flash / Light | Plus/Flash are API-only; "Light" is reported open-weight but sources conflict `[S]`. **No `qwen3_5_omni` model doc or class exists in transformers@f339035** (only `qwen3_omni_moe`, `qwen2_5_omni`) and it is not on TRT-Edge-LLM's supported list `[P]` ⇒ cannot be run by our stack; ignore until a checkpoint id and a loader exist. |
| **Qwen3-Omni-Next** | TRT-Edge-LLM's *experimental builder* already has a `qwen3_omni_next` family (thinker/talker/MTP, **4-stage conv audio encoder** ⇒ ×16 subsampling, i.e. ≈6.25 tok/s if the hop is 10 ms) `[P]` `experimental/builder/models/registry.py:285-310`, `cpp/multimodal/common/modelTypes.h:34-35`. **No checkpoint id appears in `supported-models.md`** ⇒ whether this is Qwen3.5-Omni-Light is `[U]`. |
| MiMo-V2.5 | 311 B / 15 B active; vision 729 M + audio 261 M encoders; recipe minimum VRAM **374 GB** `[P]` `models/XiaomiMiMo/MiMo-V2.5.yaml` — exceeds Thor's 128 GB. |
| Inkling-Small / Inkling | 276 B/12 B and 975 B/41 B; text+image+audio in; NVFP4 needs ≥180 GB `[P]` `models/thinkingmachines/Inkling-Small.yaml` — exceeds Thor. |
| Cosmos 3 Reasoner (Nano/Edge) | "does not currently support the audio modality by design" `[P]` `NVIDIA/cosmos` `cookbooks/cosmos3/reasoner/README.md:43`. The **Generator** tower *does* synthesise audio `[P]` `README.md:74` — relevant only as a synthetic-data source (§3.3). |
| Voxtral-Mini-4B-Realtime, Qwen3-ASR, Nemotron-3.5-ASR | speech transcription only — towers/teachers, not scene backbones. |
| Audio Flamingo 3, PE-Audio, BEATs, Parakeet-0.6B | **adapter-side towers** (§2): AF3 = Whisper-style encoder + causal LM, replace-in-place fusion, "speech, environmental sounds, music" `[P]` HF `audioflamingo3.md` (licence `[U]`, `[R]` non-commercial); **PE-Audio** = contrastive audio–text encoder with a **frame-level head, one embedding per 40 ms**, plus a joint audio-video branch `[P]` HF `pe_audio.md`, `pe_audio_video.md` (licence `[U]`). |

### 1.2 Deployment on Thor — support, latency, memory

| model | TRT-Edge-LLM 0.11.0 **audio input** | vLLM audio | Thor rows (NVIDIA-published) — **any with audio?** | weights BF16 / FP8 / NVFP4 (GB) vs 128 GB |
|---|---|---|---|---|
| **Qwen3-Omni** | **yes**: `qwen3_omni_moe` `[P]` `supported-models.md:283`; a dedicated audio-encoder engine (`audio_build`) and mel pre-processing, six-engine layout `[P]` `omni.md:13-20,228-257`; calibration uses LibriSpeech audio `[P]` `omni.md:44-48` | yes, README example `[P]` README:500 | **none for Omni.** Only the *text* proxy `Qwen3-30B-A3B` NVFP4: prefill **137.7 ms @2048**, decode 84.1 tok/s `[P]` `performance-benchmarks.md:194` | thinker-only BF16 **68.74 GB minimum at 15 s video** `[P]` README:745; NVFP4 ≈19.5 `[E]` (17.2 LLM + 2.4 FP16 towers; `omni.md:44-48` keeps encoders FP16) |
| **Nemotron-Omni** | **yes**: `NemotronH_Nano_Omni_Reasoning_V3`, NVFP4 checkpoint `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` `[P]` `supported-models.md:196,209-211`; `NemotronOmniAudioRunner` `[P]` `cpp/multimodal/nemotron_omni/` | yes, recipe pins `vllm[audio]==0.28.0`; verified H100/H200/B200 only `[P]` yaml; a Thor container is documented for the *text* Nemotron-3-Nano `[P]` `NVIDIA/Nemotron-3-Nano-30B-A3B.md:52-135`; a Jetson-AI-Lab page for Omni exists `[S]` | **No audio in any row.** v0.9.0 Thor NVFP4, B=1, dataset **COCO images**: prefill **159.4 ms @1,699 tok**, ViT **127.3 ms @1,664 tok**, decode 72.6 tok/s, GPU mem **18,928 MB** `[P]` `performance-benchmarks.md:927`; `llm_bench` **163.6 ms @2048** `:828`; peak GPU memory **18.9–20.3 GB on Thor** (v0.9.0 `:927-928` 18,928/18,979; v0.8.0 `:1430-1431` 20,259/20,257; v0.7.x `:1601,1656` 20,327/20,267, platform not stated); the DGX-Spark rows (`:1225-1226`) read 18.4–19.8 GB and are *not* Thor | vLLM recipe minimum VRAM **75 / 38 / 28 GB** `[P]` (includes runtime overhead, H100/B200); estimate 62.0 / 31.0 / 20.0 `[E]` — the NVFP4 estimate lands inside the *measured* Thor 18.9–20.3 GB, a useful cross-check |
| **Gemma-4-E2B / E4B** | **yes**: E2B, E4B (and 12B) accept "text, image, and audio input" `[P]` `supported-models.md:222-225`; `gemma4AudioRunner.cpp` `[P]` | yes, `Gemma4.md:445-505` (`--limit-mm-per-prompt audio`) `[P]`; verified on H100/AMD/TPU/Xeon, **not Thor** | Thor v0.10.0, B=1, **image** runs: E2B NVFP4 ViT **16.7** + prefill **24.8 ms @292**; E4B NVFP4 **16.7 + 36.1 ms**; FP16 36.6 / 61.0 ms `[P]` `performance-benchmarks.md:422-443`. **No audio.** | recipe minimum VRAM BF16 **13 / 20 GB** (E2B/E4B), FP8 6 / 10 `[P]`. TRT's "GPU Mem" column reads 5.0 GB (E2B) and 5.9–6.1 GB (E4B) **flat across FP16/FP8/NVFP4** ⇒ it is not a weights footprint (PLE tables placed elsewhere?) `[U]` |
| **Gemma-4-12B Unified** | **yes**: `gemma4UnifiedAudioRunner.cpp`, `GEMMA4_UNIFIED_AUDIO` `[P]` `cpp/multimodal/gemma4/`, `modelTypes.h:42` | yes `[P]` `Gemma4.md:36,447` | FP16 only: prefill **622.8 ms @2048**, decode 10.1 tok/s `[P]` `performance-benchmarks.md:166`; runtime pf@292 146.8 ms, "ViT" 1.1 ms (a linear projection) `[P]` v0.10.0 Thor table. **No audio.** | 24 (BF16) / 12 / **6.8 NVFP4 `[E]`**; recipe min VRAM 29 GB BF16, RedHatAI NVFP4 8 GB `[P]` |
| **Phi-4-multimodal** | **No audio runner.** `cpp/multimodal/phi4mm/` contains only `phi4mmViTRunner.{cpp,h}`; the Phi-4 example builds an LLM and a *visual* engine only `[P]` `phi4.md:47-90`; yet `supported-models.md:175` says "Phi-4 also accepts audio" ⇒ **doc/implementation mismatch `[M]`** (directory listing @95515c2) | recipe exists, verified Xeon 6 only `[P]` | none | 11.2 / 5.6 / 3.1 `[E]`; recipe min VRAM 14 GB |
| **MiniCPM-o 4.5** | not on the supported list `[P]` | vLLM / SGLang / llama.cpp `[P]` README:2397 | none | 18 / 9 / 5.1 `[E]` |

**Notes that change how the Thor rows read.**
- **The Nemotron-Omni ViT is expensive**: 127.3 ms for 1,664 image tokens (≈13 k tok/s) against 11.6 ms for Qwen3-VL-4B's ≈265-token image `[P]`. Scaled linearly to HiCAP's 3 cameras × 160 tokens it is ≈37 ms `[E]` — fine, but it is 3× the R1 ViT column.
- TRT's **KV-cache reuse for audio is "implemented, not release-qualified"** `[P]` `kv-cache-reuse.md:25`; speculative requests with audio bypass reuse `:36`. Any design that leans on prefix-KV reuse *across audio variants* (§2.4) leans on an unqualified path.
- TRT already exports **one** accepted intermediate hidden layer per engine (`acceptHiddenLayer`, used to feed the Qwen3-Omni Talker) `[P]` `examples/llm/llm_inference.cpp:1841-1845,1915-1916`. HiCAP needs **three** depths; a generic multi-depth tap is **not documented `[U]`** (an RA-side question; noted because it interacts with the audio-readout design).
- 0.11.0 added "copy-free handoff of caller-owned PCM buffers" and a media-artifact cache for ViT/audio-encoder embeddings `[P]` `CHANGELOG.md:24,48` — the plumbing for a live microphone ring buffer exists.

### 1.3 What a MoE means on Thor for a prefill-only 2 Hz refresh (`[E]`, Appendix A)

A MoE's *compute* scales with **active** parameters, but its **weights must all be resident and — at HiCAP's prompt sizes — all be streamed once per refresh**:

- With 128 experts and top-8 routing, a prefill of *n* tokens touches, in expectation, at most `1 − (1 − 8/128)ⁿ` of each layer's experts (uniform-routing **upper bound**; skewed routing touches fewer): 0.80 at n=25, 0.96 at n=50, 0.998 at n=100, ≈1.0 at n=544. So **even the ≈25–50 extra audio tokens do not make the pass cheaper**; the pass costs its weight-stream floor.
- **Weight-stream floor** at the paper's convention (0.7 × 273 GB/s = 191 GB/s, `sec_eff.tex` Table `tab:bracket` caption): a 30 B-A3B NVFP4 LLM (≈17.4 GB) **≈90 ms**; FP8 ≈160 ms; BF16 ≈320 ms. A dense 4 B NVFP4 (≈2.3 GB) ≈12 ms.
- Refresh duty `D = f_r (T_r + o)`, `o = 15 ms`, `f_r = 2 Hz`, 3 cameras × 160 tokens + 64 prompt (n = 544), audio window included:

| backbone (NVFP4) | T_r (ViT + prefill) `[E]` | D at 2 Hz | age bound A = T_r + o + 500 ms |
|---|---:|---:|---:|
| R1 Qwen3.5-4B (paper Table `tab:thor`, interpolated from `[P]` rows) | 73.4 ms | **0.18** | 0.59 s |
| Gemma-4-E4B (ViT 3×16.7; prefill 45–55 by interpolation between the 292- and 2,089-token rows, batch confound `[U]`) | 95–105 ms | **0.22–0.24** | 0.62 s |
| Gemma-4-12B Unified, **FP16** (only Thor rows are FP16; slope 271 µs/token) | ≈218 ms | ≈0.47 | 0.73 s |
| Nemotron-Omni 30B-A3B (ViT ≈37 ms + prefill between the 91 ms floor and the measured 159 ms) | 128–196 ms | **0.29–0.42** | 0.71 s |
| Qwen3-Omni 30B-A3B (ViT ≈3×15.7 by Qwen3-VL-8B proxy + prefill 90–138 ms by the text proxy) | 137–185 ms | **0.30–0.40** | 0.70 s |

  All are under the paper's pre-registered `D ≤ 0.5`, but a 30 B omni consumes **0.11–0.24 more duty** than R1 — headroom the fast path and other tenants would otherwise have. Memory is **not** the constraint (≈20 GB NVFP4 of 128 GB); **bandwidth and duty are**.
- **Marginal cost of the audio tokens themselves is small**: from the published dense Thor slopes (Qwen3-VL-4B 22.7 µs/token) +25 tokens ≈ **0.6 ms**, +50 ≈ 1.1 ms `[E]`; for the 30 B-A3B the average 67 µs/token gives ≤1.7 ms per +25 tokens (upper bound). **Audio *tower* per 2-s window:** Parakeet-class ≈30 GFLOP but **≈6.4 ms weight-stream-bound** at FP16 (1.3 % duty at 2 Hz); Gemma-4 tower ≈3.2 ms; BEATs-base ≈0.9 ms `[E]`. So native audio costs roughly **+6 ms over the adapter per refresh** — negligible. **What is not negligible is the backbone that carries it.**

### 1.4 Corrections and updates to RA / RD (audio rows)

| item in RA/RD | now | class |
|---|---|---|
| RD §4.1: Gemma 3n "USM 160 ms/token (6.25 tok/s)" | that is Gemma **3n** (blog `[S]`). **Gemma 4 is 40 ms/token (25 tok/s)**; the 12 B Unified variant has no tower at all | `[P]` `processing_gemma4.py:72-75`, `gemma4_unified.md:63-72` |
| RA row Phi-4-mm: "audio in: yes (P supported list)" | supported-models text says so, **the C++ runtime has no Phi-4 audio runner** | `[M]` `cpp/multimodal/phi4mm/` listing |
| RA: Nemotron-Omni "TRT-reported GPU memory ≈19–20 GB in the v0.7/0.9 rows" | confirmed, Thor range 18.9–20.3 GB; **the rows are COCO image runs — no audio** | `[P]` `performance-benchmarks.md:927-928,1430-1431,1601,1656` |
| RA: TRT-Edge-LLM 0.10.0 | repository head is **0.11.0** (2026-09-29); its benchmark file still carries only v0.9/v0.10 tables | `[P]` `CHANGELOG.md:1`, `performance-benchmarks.md:154` |
| RA: Qwen3.5-Omni "Light reported open" | no loader in transformers@f339035, not in TRT — treat as unavailable | `[M]` |
| RD §4.4: AuT-class ≈0.65 B "UNVERIFIED" | HF class defaults give ≈0.64 B — still `[E]`, checkpoint config unread | `[E]` |
| new candidates not in RA/RD | Gemma-4-12B **Unified** (encoder-free audio), MiniCPM-o 4.5, PE-Audio (frame-level audio–text tower), Qwen3-Omni-Next (builder-only) | `[P]` |

---

## 2. Native vs. adapter, for *this* design (priority 2)

### 2.1 How each option puts audio "in the same space as language and vision"

| option | path | which space the audio tokens live in | who trained the alignment | what we would train |
|---|---|---|---|---|
| **A — adapter (v1 as written)** | frozen tower (BEATs-class, ≈90 M) → 2-layer perceiver, k = 8 per 2 s → `q′ = q + tanh(g)·CrossAttn(q, a)` on the **level-query outputs of the fusion transformer** (`sec_elastic.tex` L69-73) | the fusion transformer's 512-d space. The VLM **never sees audio**. "Same space as language" is obtained *indirectly*: `L_aud = 1 − cos(e′, e_teacher(image, caption))` distils the frozen text path (L77-79) | **us**, on synthetic events with metadata captions | perceiver + gates + cross-attn ≈ 15 M |
| **A′ — native tokens, adapter injection** (new; not in RD/RA) | frozen *native* tower **and its frozen projector** → soft tokens **in the LLM's input-embedding space** (Nemotron: `sound_projection` → LM width, `modeling_nemotron_omni_audio.py`; Gemma-4: `embed_audio`; Qwen3-Omni: audio features → thinker width) — **the LM is not run on them** → the same perceiver + gated cross-attn as A | the backbone family's own input-embedding space, i.e. literally the PI's wording, at adapter cost | the vendor (127 B tokens of mixed text+audio for Nemotron `[P]` `architecture.md:67-74`; "20 M h" AuT pre-training for Qwen3-Omni `[S]`) | perceiver + gates (as A) |
| **N — native, in the prefill** | same tower + projector, tokens **inserted at placeholder positions and processed by every LM layer**: `masked_scatter` at `<so_embedding>` (Nemotron) / `audio_token_id` (Qwen3-Omni, Gemma-4) `[P]` `modeling_nemotron_h_omni.py:423-427`, `modeling_qwen3_omni_moe.py:2136` | the LLM's input space *and* its residual stream at all depths | the vendor | nothing in the LM; **only the downstream reader** of audio-influenced hidden states (fusion branch) |
| **N-EF — native, encoder-free** (Gemma-4-12B Unified) | 640-sample waveform frames → `RMSNorm → Linear` → LM `[P]` `gemma4_unified.md:63-72` | LLM input space; **one linear map** between waveform and LM | the vendor | as N |

**Two facts that cut across all four**
1. **Every native tower is mono.** Parakeet, AuT/Whisper-style and Gemma-4 features are single-channel mel spectrograms; the HF Parakeet extractor **averages multi-channel input to mono** (`parakeet/feature_extraction_parakeet.py:219-230`) and the Whisper extractor **rejects** non-mono input (`whisper/feature_extraction_whisper.py:280`) `[P]`. **Direction of arrival is therefore not available from any native path.** A multi-microphone front end (GCC-PHAT / intensity-vector features → a small head) is needed in *every* option. Feeding L and R as two separate audio segments is untested — the LM was not trained to compare interaural level/phase — `[H]`, almost certainly weak.
2. **The place where audio meets the LM decides which cached quantities it contaminates** (§2.2). That, not the embedding-space elegance, is what separates the options for a *frozen-backbone, cached-feature* design.

### 2.2 The trainable interface for a hidden-state-tap design

HiCAP caches (i) the **visual-token** hidden states at three depths, pooled to ≤128 tokens, and (ii) the **last-token** state (`sec_method.tex` L21-23). Causal attention gives a hard rule: *the hidden state at position i depends only on tokens ≤ i*. Where the audio segment sits therefore decides what audio can and cannot change.

| layout | prompt order | what audio can change | consequence |
|---|---|---|---|
| **L0** (today) | `[sys][cams][query]` | nothing | the cache and the parity anchor M0 |
| **L2** audio-before-vision | `[sys][audio][cams][query]` | **every** visual-token state and the last token | cache invalid under any audio change; parity to M0 lost; but visual tokens become audio-conditioned |
| **L1** audio-after-vision | `[sys][cams][audio][query]` | the last-token/query states only | visual-token cache stays valid; the CLM readout still shifts on every audio-present window |
| **L3** *audio suffix* (my recommendation for N) | `[sys][cams][query]` **then** `[so_start][audio][so_end][audio-query][K readout]` | **only the appended positions** | every L0 quantity — the ≤128 pooled visual states **and** the CLM last-token state — is causally independent of audio; audio-absent = omit the suffix |

**L3 is the native layout that inherits the adapter's structure.** Audio is read only from the suffix positions, through a **zero-initialised gated branch in the fusion transformer** (the same `tanh(g)` construction as A), so M0's pathway is protected by the *causal mask* rather than by trained weights. Trainable interface for N under L3, cheapest first:
- **N-a (zero LM training).** A fixed audio question in the suffix ("Describe any emergency sounds and where the vehicle is heading.") and read the last-suffix-token states at the three depths. Trainable: the gated branch only (≈ `3·d·512` + gate). The LM's own language competence is the decoder; alignment is inherited.
- **N-b (K learned readout embeddings, K = 8).** K×d trainable input embeddings (≈16 k parameters at d = 2048) appended after the audio; needs a backward pass through the frozen LM for the suffix positions only (prefix KV fixed). Heavier; do only if N-a underperforms.
- **N-c (LoRA on the LM)** — rejected: breaks "frozen backbone" and parity.
- **Flamingo-style gated cross-attention *inside* the LM** — rejected: new weights inside the model, no benefit over the suffix.

**Cost of L3 in a single pass:** +25 (Nemotron/Qwen) or +50 (Gemma) audio tokens + ≈10 query + 8 readout tokens ≈ +0.8–2 ms dense, ≤ +3.5 ms MoE `[E]` (§1.3). **Numerical caveat `[H]`:** "independent of audio" is exact in real arithmetic; TensorRT may pick different GEMM tactics for different sequence lengths, so the L0 states of an L3 run need not be *bit*-equal to an L0 run. That is testable in one hour (E-N1 step 0, §2.6).

### 2.3 What breaks or gets easier in the seven controls

`Z` = audio-absent. Adapter = A/A′. Native = N under L3. Costs assume the §2.4 numbers.

| control (RD §4.3 / `sec_elastic.tex` L84) | adapter | native | verdict |
|---|---|---|---|
| **1 bit-exact audio-null (T1 parity)** | *Trivial* at initialisation (`tanh(0) = 0`) and *exact under a hard bypass* when the audio flag is 0. ⚠️ **Not exact after training** if the fusion weights were trained with the audio branch active (modality dropout 0.3 retrains them): "the original" is then the *retrained model at audio-absent*, not M0. Parity to M0 requires **freezing every non-audio parameter** while the audio branch trains. | Exact **by construction** for *absent* audio (no suffix ⇒ identical call) and, under L3, by the causal mask. **Audio-silent is not null**: silence is a non-empty signal and the suffix still runs. Bitwise equality of an L3 run's L0 states vs an L0 run is `[H]` (tactic dependence). | adapter easier; **spec precision needed for both**: state that T1 tests *absent*, not *silent* |
| **2 shuffle** (audio from another window, matched on speed bin, noise floor, event rate) | permute cached tower features (free) | one suffix pass per draw | adapter |
| **3 bearing flip** (mirror L/R) | meaningful — acts on the side-channel front end | **not expressible in the LM** (mono tower). The flip can only act on the side-channel. Any bearing-flip effect on the *native* branch is a **leak flag**, so N doubles as a *negative control* | adapter/side-channel; native gains a negative control |
| **4 level dose-response** (0/−10/−20/−30 dB) | meaningful **if the tower's front end keeps absolute level** (Gemma-4, Whisper-style; BEATs uses fixed global constants `[R]`) | ⚠️ **Parakeet subtracts a per-mel-bin mean and divides by a per-bin std over the clip** `[P]` `melSpectrogram.h:85-92`, HF `parakeet/feature_extraction_parakeet.py:268-273`, so a *uniform* gain change is (up to the 2⁻²⁴ log floor) **erased**: an "overall attenuation" dose-response is flat for Nemotron *by construction*, not because audio is unused; the loudness→distance cue is also lost. Whisper-style front ends (Qwen3-Omni) keep level (`(log10 mel + 4)/4` after a peak-relative clamp) `[P]` `whisper/feature_extraction_whisper.py:129-130`; Gemma-4's extractor has **no per-clip normalisation** — only *optional fixed* per-bin mean/std constants (`per_bin_mean/stddev = None` by default; the E4B values sit in a preprocessor config I could not read `[U]`) `[P]` `feature_extraction_gemma4.py:87-90,111-112,159-164` | **redefine the control as an SNR dose** (siren attenuated against a *fixed* ego-noise floor), valid for every tower; expect native-Parakeet to lose distance-by-level. Per-clip normalisation also *removes stationary ego/road noise* — a plus for sirens — but makes the normaliser depend on where the onset falls inside the 2-s window, entangling control 5 `[H]` |
| **5 onset shift** (±1–2 s) | direct | Qwen3-Omni can represent time-alignment with video (TMRoPE); Nemotron/Gemma treat audio as a separate segment; nothing to fix, but the window-normaliser entanglement above applies | tie |
| **6 no-vehicle / no-vehicle-with-siren** | independent branches: cheap and clean | the LM can *bind* audio to vision (audio-visual LLMs favour vision — video-SALMONN / arXiv 2604.14129 title, `[I]` from RD §4.1). This control is **more informative** for native: a siren with no visible EV should still register | native more informative, more costly |
| **7 audio-only label probe** | probe the perceiver output — clean | ⚠️ the suffix positions **attend to the vision tokens** (they come after them), so a probe on suffix states is **contaminated by vision** and a positive result means nothing. Run the probe on the **tower+projector output before the LM** (clean, no LM pass), or on an audio-only prompt with the cameras removed | adapter/A′ cleaner; native needs an explicit audio-only prompt |
| *(new)* **8 prompt invariance** | n/a | the fixed audio question can prime the readout; keep the query text identical between audio-present and audio-absent arms, and run one arm with a *neutral* question | native-only |

### 2.4 Cost of each option (`[E]` unless marked; Appendix A)

| dimension | A (BEATs-class) | A′ (native tokens, adapter injection) | N (native, L3 suffix) |
|---|---|---|---|
| audio tower per 2-s window (weight-stream-bound, FP16) | ≈0.9 ms | 3.2 ms (Gemma-4) – 6.8 ms (Parakeet/AuT) | same |
| extra LM prefill tokens | 0 | 0 | +25 / +50 audio + ≈18 |
| marginal LM prefill time | 0 | 0 | +0.8–2 ms dense; ≤ +3.5 ms MoE |
| **backbone duty at 2 Hz** | that of the chosen vision backbone | same | same, **but only omni-family backbones qualify: 0.22–0.24 (E4B) to 0.29–0.42 (30 B-A3B) vs 0.18 (R1)** |
| VLM-token cache | unchanged | unchanged | unchanged under L3 (+ readout tokens below) |
| audio-side cache per 10⁵ refreshes | tower features ≈99×768×2 B ≈ 15 GB (BEATs dims `[R]`) | 25–50 × d_LM ≈ 10–20 GB (fp16) | K=8 readouts × 3 depths × 2048 × 1 B ≈ **4.9 GB** |
| **one control arm over 10⁵ windows** | tower + fusion ≈ 2 ms ⇒ **≈23 min** | ≈ same | one LM suffix pass each: floor 13 ms (E4B) ⇒ **≈2.5 h**; 90 ms (30 B-A3B) ⇒ **≈17.5 h** on Thor; ≈12× faster on an H100 by bandwidth ratio |
| training with a *fresh audio draw per epoch* | free (VLM cache untouched) | free | needs a suffix pass per draw; caching the prefix KV is TB-scale (O(10–50 MB) per 544-token window `[E]`, geometry `[R]`) ⇒ **audio draws must be pre-materialised** (a finite set per window) |
| runtime plumbing on Thor | own tower engine (BEATs is not in TRT-Edge-LLM; a plain ONNX→TRT job) | reuse the audio-runner engine but take the pre-projector/projected tokens out `[U]` | audio-runner + LM exist for Qwen3-Omni, Nemotron, Gemma-4 `[P]`; KV-reuse with audio is "not release-qualified" `[P]`; a **3-depth** tap is undocumented `[U]` |

### 2.5 Recommendation

**Adopt A as the v1 default; make the tower choice empirical; carry native as a pre-registered arm.**

| criterion | A / A′ | N (L3) | edge |
|---|---|---|---|
| literal "audio tokenised in the LLM's embedding space" | A′ yes (omni-family backbone only); A indirect | yes | native / A′ |
| alignment training on our data | A needs it; A′ none | none | A′ / N |
| competence on *real* audio (not our renderer) | tower-level only (A); vendor-trained tower+projector (A′) | LM-level, promptable, zero-shot | native (unproven for sirens `[U]`) |
| bearing | side-channel | side-channel (mono) | tie |
| parity / null-exactness testable bit-for-bit | yes | by construction; bitwise `[H]` | A |
| control-arm cost | minutes | hours (E4B) – a day (30 B MoE) | A |
| fresh-audio training epochs | free | pre-materialise | A |
| refresh duty | +0 | +0.04–0.24 | A |
| runtime risk on Thor | own tower engine | audio KV-reuse unqualified; 3-depth tap undocumented | roughly even |

Reading: **the adapter wins on everything that costs GPU-days or threatens parity; native wins on everything that concerns the realism of the audio representation.** The realism advantage is exactly what a *synthetic-only* adapter cannot supply, so it is worth **measuring before deciding** rather than choosing on elegance — and A′ captures most of it at adapter cost. Do **not** pick the vision backbone for its audio: RA's R1 ranking is a vision-state ranking (MMMU-Pro proxy, `[I]`), and audio is the secondary, un-validatable modality.

### 2.6 Cheapest discriminating experiments — pre-registered, both outcomes committed

**E-A0 (gate, ≈10 min, 0 GPU):** probe the PhysicalAI mp4s (§3.4). *If a `soun` track exists*, E-N1 below is re-scoped to real cabin audio and the §4 ego-leak controls become the first gate; *if absent*, absence is confirmed at the container level (three locations + container).

**E-N0 — tower-level, no LM, no PhysicalAI (≈½ day on one GPU).** Frozen features → linear probe, 4 classes {siren, horn, tyre-squeal/crash, none}, siren/horn/squeal clips from ESC-50 / UrbanSound8K / AudioSet-EV / Sci-Data-EV (§3.2) mixed over road/wind/traffic backgrounds (AudioSet `Wind noise (microphone)`, `Traffic noise, roadway noise`; MIVIA road backgrounds) at SNR ∈ {+10, 0, −5, −10, −15} dB, **both level-preserving and per-clip-normalised**. Towers: Gemma-4-E4B, Nemotron Parakeet, Qwen3-Omni AuT (each *with* its frozen projector), BEATs-base, PE-Audio frame-level. Extras: (i) zero-shot yes/no AUROC from the omni LMs; (ii) the **"same-space" check** — cosine between mean-pooled projector tokens and the LM's input embeddings of the words *siren, horn, engine, wind*, top-1 word retrieval. Estimator: **clip-level paired bootstrap** (there are no episodes here; state that), δ from a 3-seed sd of the external-tower probe, fixed before launch.
- **Outcome 1 — native tower admitted:** some native tower ≥ best external tower − δ at every SNR ≥ −5 dB *and* zero-shot AUROC ≥ 0.90 at SNR ≥ +5 dB ⇒ carry A′ and N into E-N1.
- **Outcome 2 — external tower wins:** every native tower < best external − δ at SNR ≤ 0 dB ⇒ the external tower is the audio tower, native soft tokens are dropped, and N is run only if the *LM-level* zero-shot AUROC beats the probe.

**E-N1 — injection-site head-to-head, one frozen backbone (≈days).** Backbone **Gemma-4-E4B** (native audio; R1-class Thor latency; Apache-2.0 `[S]`): the question is *where audio enters*, not which backbone is better. Arms share vision path, fusion, heads, data, seeds: **Z** audio-absent (parity anchor M0) · **A** external tower → perceiver → gated cross-attn · **A′** native tower+projector tokens → the same perceiver/gate · **N** native L3 suffix (N-a, then N-b if needed). Data: synthetic event injection on the parity corpus, rendered from world state (RD §4.3), **plus real clips audio-only** for realism. Reported: T2 (presence AUROC, distance-bin accuracy; bearing from the shared side-channel), T3 (controls 1–8, with the SNR-defined dose), **T4 no-regression on all four metric families** — per-family, never pooled, paired episode-cluster bootstrap over the 40 val episodes (`taniteval/ci.py`), stratified, n < 30 windows ⇒ "not evaluable" — and Thor p50/p95 per refresh with audio, duty, cache GB. Step 0 (1 h): the L0 states of an L3 run vs an L0 run, max-abs-diff (0 ⇒ bit-exact null holds in practice).
- **Outcome N — adopt native for an omni-family backbone** iff: N ≥ max(A, A′) + δ on *real-clip* presence AUROC **and** ≥ max(A, A′) − δ on injected events; N passes ≥ 7 of the 8 controls; T4 paired CI inside the pre-set per-family harm margins δ_f; and Δduty ≤ +0.05. The adapter stays as the fallback and an ablation.
- **Outcome A — keep the adapter** iff: max(A, A′) ≥ N + δ on injected events, **or** N fails ≥ 2 controls, **or** the control-arm budget makes ≥ 3 of the controls infeasible. The better of A/A′ fixes the tower.
- **Outcome T — indistinguishable** (all |Δ| < δ): **adapter** (cheaper controls, cache intact) — tie-break committed *now*; and between A and A′ prefer **A′** when A′ ≥ A − δ, because its tokens are in the LLM's space at no measured loss.
δ (audio metrics) and δ_f (per family) are set **before launch** from ≥3-seed sd of arm A, as RD §3.8 does for camera elasticity. *No claim that audio improves driving follows from any outcome* — T2–T4 are plumbing, usage and no-regression tests (RD §4.3, `sec_elastic.tex` L83).

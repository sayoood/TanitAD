# HiCAP stream A / note RG — the audio bridge into the shared embedding space (2026-10-25)

**Status:** research note, staged in the working tree, not committed. Owner: stream A. It **adds to**
`RF_omni_audio_backbone.md` (native backbone vs adapter) and `HICAP_V10_SPEC.md` §3.2 (the bridge spec); it edits
neither. Code that implements it: `stack/tanitad/hicap/audio.py`, tests `stack/tests/test_hicap_audio.py`.

**PI request (2026-10-25):** *"include an audio encoder with bridge to the same embedding space as done in the omni
models of Qwen or in the Gemma embeddings."* **MEASURED today (spec §0):** there is no audio in PhysicalAI-AV or in our
corpus, so every audio sample we train on is external or synthetic.

**Evidence tags** (CLAUDE.md): **MEASURED** (ours, artifact path) · **PUBLISHED** (primary source read today, file:line
or URL) · **INHERITED** (another doc/agent or a mirror, not re-verified) · **ESTIMATED** · **HYPOTHESIS** ·
**UNVERIFIED** (source not reachable). arXiv, Zenodo, research.google.com and the UrbanSound site are egress-blocked
(403); everything PUBLISHED below comes from Hugging Face cards/configs, GitHub repos, or the vendors' own tech-report
PDFs shipped inside their GitHub repos.

---

## 0. Headline

1. **All four omni families use the same bridge, and it is tiny.** A pretrained audio encoder → a **one- or two-layer
   projector (2.4–4.6 M parameters, MEASURED from checkpoint headers)** → soft tokens written into the LM's input
   embedding sequence at placeholder positions (`masked_scatter`). **No Perceiver / Q-Former in any of them**: the
   token count is the encoder's frame rate (6.25–25 tok/s). Alignment is trained with the **LM's next-token loss on
   audio–text data**, never contrastively, and Qwen trains the **adapter first with the LLM frozen, then the encoder**.
2. **Our bridge (`audio.py`) follows that pattern with two deliberate differences**: (a) a 2-layer Perceiver fixes the
   output at **8 tokens per 2 s window** (the omni towers would give 25–50), because HiCAP reads audio through one gated
   cross-attention at 2 Hz and the VLM-prefill ablation is priced per token; (b) stage A1 aligns **contrastively to the
   frozen text path's embedding of a caption** (CLAP-style) instead of through the LM's next-token loss, because it needs
   no backward pass through the frozen backbone. The omni-faithful generative alignment (caption NLL through the frozen
   reasoner with the 8 tokens as a prefix, gradient into the bridge only — exactly Qwen's "adapter first, LLM fixed")
   is registered below as **A1-gen**, the arm that makes mode (b) honest.
3. **Recommended frozen tower: the Qwen3-Omni AuT encoder** (Apache-2.0; **647.9 M MEASURED**; 12.5 tok/s; trained on
   20 M h including audio-understanding data; block-window attention 1–8 s, so 2 s windows are native; ≈6.8 ms per
   window on Thor ESTIMATED = 1.4 % duty at 2 Hz). **Cheap contender: BEATs-base** (MIT, ≈90 M ESTIMATED, ≈0.9 ms).
   **The choice is made by the pre-registered tower probe E-N0 of note RF §2.6, not by this note.** This differs from both
   the paper's text ("BEATs-base-class") and spec §3.2 ("Whisper-class") — **an integration item for the orchestrator**.
4. **No siren/horn training set reached today is cleared for commercial use.** ESC-50 and UrbanSound8K are CC BY-NC,
   AudioCaps' own README says "academic purposes only" (while its LICENSE file says MIT), and AudioSet's audio is
   YouTube. The renderer's procedural sources (`audio.py`) are licence-free. **PI decision**: research-only data for A1,
   or fund a CC0-filtered / self-recorded set before any product claim.
5. **Corrections to earlier notes:** Qwen3-Omni's *checkpoint* sets `position_id_per_seconds = 13`, not the class
   default 25 that RF §0 quoted; Gemma-4-E4B's shipped `processor_config.json` has `per_bin_mean/stddev = null`, which
   settles RF's `[U]`: **no per-clip normalisation, absolute level is preserved** (the dose-response control is valid on
   it); Gemma-4 cards claim audio for **ASR and speech translation only** — not environmental sound.

---

## 1. How the omni models bridge audio into the LM space

### 1.1 Encoder, rate, projector (all PUBLISHED unless tagged)

| model | audio encoder | front end → rate | projector into the LM (shape) | where the tokens go | LM width |
|---|---|---|---|---|---|
| **Qwen2.5-Omni-7B / -3B** | Whisper-large-v3-initialised encoder (`Qwen2.5_Omni.pdf` §3, pdftotext L276; "audio encoder from Qwen2-Audio", L164-165); d_model 1280, 32 layers, 20 heads, ffn 5120, 128 mel (`config.json` L81-121 of Qwen2.5-Omni-7B); **639.6 M** (7B) / **637.7 M** (3B) MEASURED, `rg_audio_tower_params_result.json` | 16 kHz, 128 mel, 25 ms window, 10 ms hop (report L162-164); Conv1d(k3) → Conv1d(k3, s2) → transformer → **AvgPool1d(2,2)** → **40 ms per token = 25 tok/s** (report L165; `modeling_qwen2_5_omni.py:848-855`); attention in **2 s blocks** for chunked prefill (report L234-237; `n_window: 100` config L113) | `ln_post` → **one Linear(1280 → 3584)** (7B) / **(1280 → 2048)** (3B) (`modeling_qwen2_5_omni.py:853-855`; `output_dim` config L121); MEASURED 4,591,104 / 2,623,488 parameters = exactly 1280·out + out | `inputs_embeds.masked_scatter(audio_mask, audio_features)` at `audio_token_index` positions (`modeling_qwen2_5_omni.py:1908`); TMRoPE time ids, 25 per s (config L187) | 3584 / 2048 |
| **Qwen3-Omni-30B-A3B** | **AuT** (Audio Transformer), "trained from scratch on 20 million hours of supervised audio", "approximately 0.6B parameters" (`Qwen3_Omni.pdf` §2.2, L199-208); Table 1 says **650 M** (L290); **647.9 M MEASURED** (thinker.audio_tower.*). Training mix 80 % zh/en pseudo-labelled ASR, 10 % other-language ASR, **10 % audio understanding** (L203-205) | 128 mel, 25 ms / 10 ms (L215-216); **3 × Conv2d(3×3, stride 2, 480 ch)** = 8× downsampling → **12.5 Hz, ≈80 ms per token** (L200-202, L280; `modeling_qwen3_omni_moe.py:773-775`); flash attention with **dynamic windows 1–8 s** (L206-207) | `ln_post` → **Linear(1280,1280) → GELU → Linear(1280 → 2048)** (`modeling_qwen3_omni_moe.py:781-783`); MEASURED 4,263,168 parameters = (1280²+1280)+(1280·2048+2048) | `audio_token_id` 151675 (config L283), masked-scatter as 2.5; `position_id_per_seconds: 13` in the **checkpoint** (config L288) | 2048 |
| **Gemma 3n E4B** | Conformer described as the **Universal Speech Model's** "conf" section (`configuration_gemma3n.py:218-235`); hidden 1536, 12 layers, 8 heads, chunked local attention (chunk 12, left 13, right 0), 128 mel input (mirror `config.json` L12-16); **681.3 M** MEASURED from the `unsloth/gemma-3n-E4B-it` **mirror** (google repo gated, 401) — INHERITED-from-mirror | 16 kHz, 32 ms frame / 10 ms hop (`feature_extraction_gemma3n.py:160-161`); 2 stride-2 conv (SSCP, `configuration_gemma3n.py:301-304`) then `conf_reduction_factor = 4` decimation (`modeling_gemma3n.py:1517-1518`) → **160 ms per token**; card: *"Audio data encoded to 6.25 tokens per second from a single channel"* (mirror README L106); 188 tokens per 30 s clip (`audio_soft_tokens_per_image`, config L49) | `Gemma3nMultimodalEmbedder`: **RMSNorm(1536) → Linear(1536 → 2048, no bias) → RMSNorm (no scale)** (`modeling_gemma3n.py:1900-1904, 1925-1930`); MEASURED 3,345,408 = 128·1536 (hard-token table) + 1536·2048 + norms | soft tokens at audio-token positions | 2048 |
| **Gemma 4 E2B / E4B** | Conformer, hidden 1024, 12 layers, 8 heads, chunk 12 / left 13 / right 0, `output_proj_dims 1536` (`google/gemma-4-E4B-it` `config.json` audio_config); card: **"~300M"** audio encoder (README L62); **304.8 M MEASURED** (same tower in E2B and E4B) | 16 kHz, 128 mel, 20 ms frame / 10 ms hop, fft 512, **`per_bin_mean: null`, `per_bin_stddev: null`** (`processor_config.json` L10-19); **`audio_ms_per_token: 40` = 25 tok/s**, `audio_seq_length: 750` (L2-3); max audio **30 s** (README L426) | inside the tower: Linear(1024 → 1536, bias) (`modeling_gemma4.py:1915`); embedder: **RMSNorm (no scale) → Linear(1536 → 2560 E4B / 1536 E2B, no bias)** (`modeling_gemma4.py:2060-2064`); MEASURED 3,932,160 / 2,359,296 = 1536·2560 / 1536·1536 | `audio_token_id` 258881 (config); audio placed **after** the text in prompts (README L393) | 2560 / 1536 |
| *Gemma 4 12B "Unified"* | **no encoder**: "projecting raw image patches and audio waveforms directly into the LLM's embedding space through lightweight linear layers" (README L66) | — | linear | — | — |

### 1.2 How the alignment is trained

| model | stages (verbatim gist) | source |
|---|---|---|
| Qwen2.5-Omni | **S1**: "lock the LLM parameters and focus exclusively on training the vision encoder and audio encoder ... audio-text and image-text pairs"; the encoders "are trained separately on the fixed LLM, with both **initially focusing on training their respective adapters before training the encoders**". **S2**: unfreeze all, +300 B tokens of audio-related data, +100 B video-with-audio. **S3**: 32 k sequence length. | `Qwen2.5_Omni.pdf` §3, pdftotext L264-290 |
| Qwen3-Omni | **S1 "Encoder Alignment"**: LLM from Qwen3, audio encoder initialised with AuT, "trained separately on the fixed LLM ... adapters before training the encoders". **They explicitly abandon** "the stage ... where the encoder and adapter are trained jointly while keeping the LLM frozen, because this approach may cause the encoder to compensate for the limitations of the frozen LLM, which can lead to degraded perception capabilities." **S2** ≈2 T tokens mixed; **S3** 32,768 context. | `Qwen3_Omni.pdf` §3, L376-397 |
| Gemma 3n / Gemma 4 | **No alignment recipe published on the cards**; no Gemma 3n/4 tech report was reachable. | UNVERIFIED |

**What this means for us.** Both Qwen reports train the projector *first* against a frozen LM, with the LM's own loss,
and keep the encoder frozen during that phase. Our A1 has the same freeze pattern (encoder frozen, backbone frozen, only
the bridge trains) — and Qwen3's stated reason for not co-training the encoder against a frozen LM is a second argument
(besides cost) for keeping our tower frozen throughout. What we change is the loss: a contrastive target in the frozen
text path's space is cheap and is what HiCAP's mode (a) consumes; the next-token loss is what makes soft tokens
*readable by the LM* (mode (b)). Hence two A1 arms:

* **A1-con (default, implemented):** symmetric InfoNCE between the pooled bridge output and the frozen text-path
  embedding of the event caption, plus a cosine term (`BridgeAlignLoss`). Multi-positive targets for duplicate captions.
* **A1-gen (registered, for mode (b) only):** the 8 tokens as a soft prefix to the frozen reasoner, loss = caption NLL,
  gradient into the bridge only (no LoRA). Required before any "the VLM understands our audio tokens" claim; costs one
  backbone forward+backward over ≈8+30 tokens per sample.

---

## 2. Candidate frozen encoders for HiCAP

Cost: `rg_audio_tower_cost.py` → `rg_audio_tower_cost_result.json` (ESTIMATED: max of weight-stream floor at 191 GB/s FP16
and compute at 25.7 TFLOP/s effective — both conventions INHERITED from the paper; 2 s window, 2 Hz refresh).

| tower | params | licence | input | standalone? | tokens / 2 s | Thor per window (duty @2 Hz) | for us |
|---|---|---|---|---|---|---|---|
| **Qwen3-Omni AuT** | **647.9 M** MEASURED | Apache-2.0 (card L2-3 `license_name: apache-2.0`; repo `LICENSE`) | 16 kHz mono, 128 mel (Whisper-style, level kept — RF §2.3) | yes: `thinker.audio_tower.*` lives **entirely in shard 1 of 15** (525 tensors; 4.998 GB file) — MEASURED from `model.safetensors.index.json` and the HF tree API | 25 | **6.8 ms (0.014)** | **recommended default**: omni-trained, non-speech data in the mix, native 1–8 s windows, permissive |
| Gemma-4 E2B/E4B tower | **304.8 M** MEASURED | Apache-2.0 (card L3, L22) | 16 kHz mono, 128 mel, **no per-clip normalisation** (processor_config L18-19) | yes (`model.audio_tower`) | 50 | 3.2 ms (0.006) | strong second; card claims ASR/translation only → must earn its place on sirens in E-N0 |
| Whisper-large-v3 encoder | **637.0 M** MEASURED | Apache-2.0 (card L112); code MIT (`openai/whisper` LICENSE) | 16 kHz, 128 mel; **pads every input to 30 s** | yes | 100 (in a 2 s block) | 6.7 ms (0.013) in 2 s blocks; **89 ms (0.18) if run natively padded to 30 s** | the ancestor of the Qwen2.5-Omni tower; ASR-only training (1 M h weak + 4 M h pseudo-labelled, card L128) — dominated by AuT |
| Qwen2.5-Omni-7B tower | 639.6 M MEASURED | Apache-2.0 (card L2-4) | as Whisper | yes | 50 | 6.7 ms | fine; AuT supersedes it. **The 3B tower is `qwen-research` (non-commercial)** — MEASURED via HF API |
| Gemma-3n E4B tower (USM-based) | 681.3 M (mirror) | **Gemma Terms of Use** (`license:gemma`, HF API) | 16 kHz mono | yes | 12.5 | 7.1 ms (0.014) | the only released "USM"-lineage weights; custom licence; no advantage over Gemma 4 |
| **BEATs-iter3 base** | ≈90 M ESTIMATED (12 × 768) | MIT (repo `microsoft/unilm/LICENSE`; beats README L105-106 defers to it); checkpoints on OneDrive, **not on HF** | 16 kHz, 128 fbank | yes | 96 patches | **0.9 ms (0.002)** | cheapest, AudioSet-tagging pretraining (non-speech) — the E-N0 contender |
| CLAP-HTSAT (`laion/clap-htsat-unfused`) | ≈28 M audio branch ESTIMATED | Apache-2.0 card (L2); repo code CC0 | **48 kHz, fixed 10 s (repeat-pad)** (`preprocessor_config.json`) | yes | 1 pooled 512-d | 0.4 ms | already text-aligned (in RoBERTa space, not ours); use as a **retrieval baseline / sanity teacher**, not the tower; 10 s window blurs onset |
| USM (Google) | — | **not released** as standalone weights | — | no | — | — | only via Gemma 3n's tower |

All of them are **mono** (RF §2.1): bearing must come from our multi-channel front end (inter-channel level/phase
+ GCC-PHAT), which `audio.py` feeds to the bridge as a separate token stream.

---

## 3. External data for alignment (A1) and event heads (A2)

| set | size | driving-relevant classes | licence (commercial?) | source |
|---|---|---|---|---|
| **AudioSet** (+ strong labels) | 2,041,789 unbalanced + 22,160 balanced + 20,371 eval 10 s clips (≈85 % still downloadable in 2023) | ontology: Siren, Civil defense siren, **Emergency vehicle, Police car (siren), Ambulance (siren), Fire engine (siren)**, Vehicle horn, Air horn/truck horn, Reversing beeps, Skidding, Tire squeal, Air brake, Car alarm, Train horn, Train whistle, Bicycle bell, Motorcycle, Traffic/roadway noise, Engine (light/medium/heavy), Smash/crash | labels **CC BY 4.0**; ontology **CC BY-SA 4.0**; audio = YouTube, not distributed ⇒ **commercial use of the audio not cleared** | mirror card `agkphysics/AudioSet` L4, L149-163 (INHERITED-from-mirror); `audioset/ontology` README L49, `ontology.json` (class list MEASURED by grep); strong-label subset size UNVERIFIED (research.google.com 403) |
| **AudioCaps** v1 / 2.0 | 49,838 / 91,256 train captions | captions over AudioSet clips (traffic, sirens, horns appear) | **conflict**: `LICENSE` file = MIT, README = "free to use for academic purposes only"; audio is YouTube ⇒ treat as **non-commercial** | `cdjkim/audiocaps` README, LICENSE, `dataset/README.md` L14, `dataset2.0/README.md` L14 |
| **Clotho** | UNVERIFIED (Zenodo 403) | general captions (Freesound) | UNVERIFIED | `audio-captioning/clotho-dataset` README points to Zenodo DOI 10.5281/zenodo.3490684 |
| **UrbanSound8K** | 8,732 clips ≤ 4 s, 10 classes, 10 folds | **car_horn, siren, engine_idling**, street_music | **CC BY-NC** (mirror card says 4.0; original site unreachable) ⇒ **non-commercial** | mirror card `danavery/urbansound8K` L4, L44 (INHERITED-from-mirror) |
| **ESC-50** | 2,000 × 5 s, 50 classes × 40 | **Siren, Car horn, Engine, Train**, Helicopter, Airplane, Church bells | **CC BY-NC 3.0**; ESC-10 subset CC BY (ESC-10 has no siren/horn) | `karolpiczak/ESC-50` README L11-13, L159-165, LICENSE |
| Siren / emergency-vehicle sets (sireNNet, LSSiren, Kaggle EV sirens, "Sci-Data-EV" of RF) | UNVERIFIED | sirens | UNVERIFIED — none reachable; an HF dataset search for "siren"/"emergency vehicle" returned nothing authoritative (MEASURED) | — |
| **Procedural renders** (`audio.render_synthetic_event`) | unlimited | siren wail/yelp, horn, reversing beeper, tyre squeal, train horn | ours, licence-free | this repo |

**Recommendation for data.** A1 on AudioCaps + the driving-relevant AudioSet classes (captions rewritten from labels
for the class-only sets) **as research**; A2/A3 on procedural renders placed on real agent tracks. Mark every
A1-trained checkpoint *non-commercial* until a CC0/self-recorded set exists (HYPOTHESIS: a Freesound CC0 filter yields
enough siren/horn material — not checked).

---

## 4. Recommendation (with reasons)

1. **Tower:** Qwen3-Omni AuT, frozen, extracted from shard 1 (≈5 GB download, no 70 GB pull). Reasons: permissive
   licence; an omni encoder by definition (the PI's reference design); non-speech audio in its training mix and a
   published general-audio captioner on top of it (vendor claims, PUBLISHED, not measured by us); native 1–8 s windows so
   a 2 s window at 2 Hz is in-distribution; level-preserving front end; 1.4 % duty ESTIMATED. **Contender: BEATs**
   (7× cheaper, environmental-sound pretraining). **Decided by E-N0** (RF §2.6) on SNR-defined siren/horn probes.
2. **Bridge:** as implemented — frozen tower → LayerNorm+Linear → 2-layer Perceiver (8 latents, width 512) with a
   separate multi-channel spatial token stream → MLP → RMS-matched 8 tokens in R^{d_v}. Gemma-style output RMSNorm
   without scale, multiplied by the backbone's measured input-embedding RMS, so mode (b) tokens arrive at the scale the
   LM expects (Gemma's `embedding_post_projection_norm`, `modeling_gemma3n.py:1904`, is the same idea).
3. **Alignment:** A1-con now (CPU-testable); A1-gen before any mode-(b) claim.
4. **Do not** train the tower (Qwen3's own warning, §1.2); do not adopt a native omni backbone for audio (RF §2.5).

## 5. Integration items (escalated, not written elsewhere)
* `HiCAPConfig.d_audio` must be set to `d_v` (2048 / 4096) when `AudioBridge` feeds `GatedAudio` (model.py is the
  orchestrator's).
* Paper `sec_elastic.tex` L69 ("BEATs-base-class") and spec §3.2 ("Whisper-class") vs this note (AuT default, BEATs
  contender): one sentence needs to change in each once E-N0 reports.
* RF §0 item 2 quotes `position_id_per_seconds = 25` for Qwen3-Omni; the checkpoint says **13**.

## 6. What was built, and its size (MEASURED)

`stack/tanitad/hicap/audio.py`: `AudioFrontend` (level-preserving log-mel, 25 ms / 10 ms, 2 s window at 0.5 s hop via
`window_stream`; for arrays ILD + IPD on 16 coarse bands and a noise-gated, 40 ms-coherent GCC-PHAT over ±64 lags),
`FrozenAudioEncoder` protocol + `StandInEncoder` (tests only), `AudioBridge`, `BridgeAlignLoss`, `EventHeads`,
`render_synthetic_event` (retarded-time Doppler/ITD, 1/r, air absorption, directivity ILD, ego-noise floor),
`track_from_agent`, `caption_for_event`, the bearing-flip helpers, `splice_soft_tokens` (mode (b) interface).
Tests: `stack/tests/test_hicap_audio.py` (17 tests). Counts: `rg_audio_bridge_params.py` → `rg_audio_bridge_params_result.json`.

| d_v | tower width | mics | AudioBridge | EventHeads | bridge + heads | model.GatedAudio at d_audio = d_v (already in HiCAP) |
|---|---|---|---:|---:|---:|---:|
| 2048 | 1280 (AuT / Whisper-large) | 1 | 9,595,904 | 533,780 | **10,129,684** | 2,099,718 |
| 2048 | 1280 | 4 | 9,869,350 | 533,780 | **10,403,130** | 2,099,718 |
| 2048 | 768 (BEATs) | 4 | 9,606,182 | 533,780 | 10,139,962 | 2,099,718 |
| 4096 | 1280 | 1 | 11,695,104 | 1,062,164 | **12,757,268** | 3,148,294 |
| 4096 | 1280 | 4 | 11,968,550 | 1,062,164 | **13,030,714** | 3,148,294 |
| 4096 | 768 | 4 | 11,705,382 | 1,062,164 | 12,767,546 | 3,148,294 |

(+1 learnable logit scale in `BridgeAlignLoss`.) The Perceiver layers are 6.31 M at width 512 regardless of d_v; the
output MLP is 2.63 M (d_v 2048) / 4.72 M (4096). For comparison the omni projectors are 2.4–4.6 M (§1.1) — ours is
larger because it also pools (Perceiver) and reads the spatial stream.

**Mechanism checks (MEASURED, CPU, `test_hicap_audio.py`):** a toy alignment (7 procedural classes ↔ hashed caption
embeddings, frozen stand-in tower, 150 steps) drives held-out class retrieval to ≥ 0.5 top-1 against chance 1/7 —
this shows the bridge *can* align, nothing about real audio; Doppler peak of a 1 kHz tone at ±20 m/s radial speed is
within 1 Hz of `f c/(c ∓ v)`; mirrored channels flip the rule-based side estimate and exactly time-reverse the
front end's GCC; bridge tokens through `model.GatedAudio` at g = 0 leave the level queries **bit-identical**
(int32 view), as does `audio=None`.

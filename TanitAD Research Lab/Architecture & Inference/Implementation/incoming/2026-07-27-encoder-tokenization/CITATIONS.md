# CITATIONS — with FETCH DEPTH per source (retraction class C16)

**Why this file exists.** C16 was logged 2026-07-27 after a PDF fetch in a sibling stream
**FABRICATED a verbatim quotation, a section name and three numbers** — and the fabricated claim was
the most load-bearing one in that brief. The rules applied here:

- **PDF-summarisation output is a model-generated summary, NOT source text, and is INADMISSIBLE as a
  quotation.** No claim in `ENCODER_TOKENIZATION.md` rests on one.
- Quote only from **HTML full text** or **the abstract**, retrieved directly.
- **Record the depth per citation.** Depths used:
  - `FULLTEXT-HTML` — HTML full text (arxiv `/html/`, ar5iv, OpenReview HTML) read directly.
  - `ABSTRACT-HTML` — the abstract page HTML read directly.
  - `NOT-FETCHED` — not retrieved; background knowledge only ⇒ **UNVERIFIED**.
  - `PDF-ONLY` — only a PDF existed ⇒ **UNVERIFIED, no quotation permitted**.
- ⚠️ **Treat a conveniently perfect quote with MORE suspicion, not less.**
- Where a fetch returned **paraphrased** rather than verbatim text, that is recorded and **nothing is
  quoted** from it.

---

## Sources that carry weight in the report

| source | arXiv | depth | what we take from it | status |
|---|---|---|---|---|
| **STT — foveated ring tokenization**, Schmidt & Newcombe | [2506.11131](https://arxiv.org/abs/2506.11131) | **FULLTEXT-HTML** | 172 tokens vs SAM's 4096; STT-B 30.9 vs SAM-H 6533.7 GFLOPs; mIoU 0.412 vs 0.393 (Cityscapes). **Critically: the encoder was MAE-pretrained FROM SCRATCH on the foveated pattern** — our regime | ✅ admissible. ⚠️ the "24×" ratio is **our division**, not in the abstract |
| **Alpamayo AR1** | [2511.00088](https://arxiv.org/abs/2511.00088) | **FULLTEXT-HTML** | Default tokenizer = base VLM encoder 2× downsampled ⇒ **160 tokens/image at 448×280**, used for *all* reported experiments. Triplane is optional ("can additionally use"); the stated ratio is **3.9×**, not 3.6× | ✅ admissible, and it **corrects our repo** (§3.2) |
| **Vision Transformers Need Registers**, Darcet et al. | [2309.16588](https://arxiv.org/abs/2309.16588) | **ABSTRACT-HTML + FULLTEXT-HTML** | 4 registers, FLOP increase "below 2%"; DINOv2 sweep — only the three largest models show outliers, Base does not; artifacts emerge ~37 % depth and after ~⅓ of training | ✅ admissible |
| **ToMe — Token Merging**, Bolya et al. | [2210.09461](https://arxiv.org/abs/2210.09461) | **FULLTEXT-HTML (ar5iv)** | **Trained-from-scratch** setting: DeiT-S 300 ep, r=16 → **79.13 vs 79.96** (−0.83), 4.61→2.30 GFLOPs, ~1.5× training speedup | ✅ admissible — this is the *training* number the brief asked for |
| **Perceiver** | [2103.03206](https://arxiv.org/abs/2103.03206) | **FULLTEXT-HTML (ar5iv)** | **512** latents × 1024 ch, 8 cross-attends; ImageNet 78.0. Authors concede the bottleneck's severity | ✅ admissible |
| **Perceiver IO** | [2107.14795](https://arxiv.org/abs/2107.14795) | **FULLTEXT-HTML (ar5iv)** | architecture/latent sizing | ✅ admissible |
| **Flamingo** | [2204.14198](https://arxiv.org/abs/2204.14198) | **FULLTEXT-HTML (ar5iv)** | Perceiver Resampler at **64** latents; ablation 86.5 vs 83.2 vs 78.6 CIDEr — **frozen encoder AND frozen LM** | ✅ admissible; the frozen/captioning setting is the reason transfer is judged WEAK |
| **NaViT (patch-n-pack)** | [2307.06304](https://arxiv.org/abs/2307.06304) | **ABSTRACT-HTML + FULLTEXT-HTML** | matches top ViT at 4× less compute; token-drop curriculum ρ 0.2–0.8; <2 % packing padding | ✅ admissible. ⚠️ **the 4× must NOT be quoted as available to us** — the mechanism is corpus resolution diversity we do not have |
| **FlexiViT** | [2212.08013](https://arxiv.org/abs/2212.08013) | **ABSTRACT-HTML** | one weight set across patch sizes | ✅ admissible at abstract level only; **no per-task numbers verified** |
| **Look-Focus-Act / GIAVA** | [2507.15833](https://arxiv.org/abs/2507.15833) | **FULLTEXT-HTML** | foveated 20 tokens vs fine 324; 115.6 vs 1905.4 GFLOPs. ⚠️ **Coarse baseline is ALSO 20 tokens (126.9 GFLOPs)** ⇒ the compute win is token count, not foveation; and the fovea is **gaze-steered** | ⚠️ admissible **only with the confound stated** (§3.5) |
| **Sector Patch Embedding (fisheye)** | [2303.14645](https://arxiv.org/abs/2303.14645) | **ABSTRACT-HTML** | +0.75 % top-1 ViT, +2.8 % PVT on **synthetic** fisheye ImageNet | ✅ admissible; effect is tiny and classification-only |
| **PolarFormer** | [2206.15398](https://arxiv.org/abs/2206.15398) | **ABSTRACT-HTML** | polar coordinates applied to the **BEV/output** space, ordinary images as input | ⚠️ recorded as a **misattribution risk** — it is NOT log-polar image tokenization |

## Sources fetched but returning PARAPHRASED text — nothing quoted

| source | arXiv | depth | note |
|---|---|---|---|
| **Swin Transformer** | [2103.14030](https://arxiv.org/abs/2103.14030) | ABSTRACT-HTML, **paraphrased** | Used only for the general "linear complexity" property, which is standard. **No verbatim quotation taken.** The recommendation against Swin rests on **our own arithmetic** (§3.0), not on this paper |
| **DynamicViT** | [2106.02034](https://arxiv.org/abs/2106.02034) | ABSTRACT-HTML, **paraphrased** | 66 % token prune → 31–37 % FLOPs, <0.5 % accuracy drop. Treat the figures as low-confidence; the rejection rests on the **input-dependence** property, not the numbers |
| **EViT** | [2202.07800](https://arxiv.org/abs/2202.07800) | ABSTRACT-HTML, **paraphrased** | as above |

## ⛔ NOT VERIFIED — do not propagate any of these

| claim | why it is not admissible |
|---|---|
| **Ivanovic et al. 2025** — the triplane tokenizer itself | **NOT-FETCHED.** Every triplane efficiency number traces here |
| **Flex (Yang et al. 2025)** — the "20× token compression" | **NOT-FETCHED.** AR1 states it prospectively as a *cited* method |
| **AR1 §6.6 results** | **NOT read** — only the forward reference to it |
| **"Perceiver resamplers lose fine-grained/OCR detail"** | Seen only in **search snippets**. The specific figures circulating (e.g. 70.4 vs 72.2 OCR) are **UNVERIFIED — do not quote.** The only primary-source bottleneck evidence held is Perceiver's own one-line concession |
| **Look-Focus-Act's input resolution** | HTML fetch returned "NOT FOUND IN TEXT". Without it we **cannot** compute what angular resolution its 20 tokens cover — which is exactly the number that would decide transfer |
| **A-ViT** | **NOT-FETCHED** at all |
| **Perceiver's "707.2B FLOPs"** | Read out of an ablation table; units/context unclear. **Low confidence, do not propagate** |
| **All TensorRT / static-shape safety judgements** | **ESTIMATED**, inferred from static-vs-input-dependent token count. **No TensorRT documentation was fetched.** Not MEASURED |
| **"No published foveated tokenization in a video world-model setting"** | ⚠️ **SINGLE-PROBE ABSENCE.** Per the two-probe rule this is **NOT established** — only "not found on one search". Do not write it as a fact |

---

## Internal (non-published) sources, with their own classes

| source | class | note |
|---|---|---|
| `…/2026-07-26-situation-semantics/artifacts/t1_probe.json` + `t1_heldout_scores.npz` | **MEASURED (re-analysed by us)** | the rank-16 ladder's actual substrate; §0's paired contrasts are ours |
| `…/2026-07-26-situation-classifier/SITUATION_CLASSIFIER.md` §8 amendment A4 | INHERITED | records that three arms do not read the PCA rank at all |
| `Project Steering/Gates/flagship-v5-retrain.PREP.md` | INHERITED | bars, MDE inputs, T3's PDMS-lite CI, the ADE-vs-composite evidence |
| `stack/tanitad/models/{encoder,readout,dynamics_encoder}.py`, `stack/tanitad/config.py` | **MEASURED** | read at line level; params counted by instantiation |
| "the encoder is 60 %+ of our tick" (`Research/ENCODER_MULTICAM_OPTIMIZATION.md`) | **INHERITED, NOT re-verified** | flagged as a gap in §5.7 — it sizes the validation and must be measured |
| flagship total **286.34 M** | INHERITED (`MODEL_REGISTRY.md` via `CLAUDE.md`) | the **87.02 M** encoder figure is MEASURED here |

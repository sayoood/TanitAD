# THE ENCODER FOR A WIDER FIELD — what v5 should train with, and how to afford it

**Date:** 2026-07-27 (local, Europe/Berlin). **Stream:** `encoder-tokenization`.
**Owner of:** PREP-card item 7's *"encoder size and architecture, tokenization efficiency, and the
small validation itself"* (`Project Steering/Gates/flagship-v5-retrain.PREP.md` §3.7).
**Host:** dev box (`C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, RTX 4060 8 GiB).
⛔ **No pod was touched.** pod1 trains, pod2 runs an arm panel, pod3/eval run IDM v3.
⛔ **No file under `stack/tanitad/` was modified.** The non-square encoder path used here is the
geometry sibling's already-landed `EncoderConfig.image_width` / `SpatialGridReadout.token_grid`.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited, **with fetch depth**) ·
`INHERITED` (another agent/doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Tiers:** `PROVISIONAL` / `CONFIRMED` / `DECISION-GRADE`.
**Estimator:** episode/clip-cluster bootstrap (`taniteval/taniteval/ci.py`, B = 2000, seed 0), paired
where two arms share windows. ⛔ **`overlapping_holdout_se` appears nowhere in this document.**
🔒 PhysicalAI-AV is gated-confidential: no clip UUID or raw content appears in this folder.

---

# 0. HEADLINE — the finding that changes the architecture

**Tier: DECISION-GRADE. Class: MEASURED (re-analysis of banked held-out scores, ours).**

> ## ⛔ THE RANK-16 DOSE–RESPONSE DOES NOT SAY WHAT THE PROGRAM HAS BEEN QUOTING IT AS SAYING.
>
> The ladder — `ego 3.659× → +k16 3.685× → +k64 3.000× → +k256 2.116× → +k2048 1.59×` — is quoted in
> **at least four documents**, is listed under **"VALIDATED"** in the v5 PREP card, and was about to
> license a **Perceiver-style fixed-16-latent resampler**. Every rung is an **unpaired point
> estimate**. The source artifact contains a paired bootstrap of every arm *against chance* and
> **not one paired contrast between two rungs of the ladder**. So the ladder's *shape* — the thing
> the whole architectural argument rests on — **had never had an interval on it.**
>
> The raw held-out scores were banked, so the missing test cost seconds. Two results:
>
> **1. The "peak at k = 16" is NOT SEPARATED from ego alone.**
> `ego_win+img_pca16 − ego_win = +0.00085, 95 % CI [−0.02204, +0.02299]` (ridge; logistic
> `+0.00426 [−0.02142, +0.02971]`). **The CI is 27× wider than the effect.** The measurement cannot
> distinguish *"16 dimensions of vision help"* from *"vision contributes exactly zero"*.
>
> **2. The IMAGE-ONLY ladder is FLAT — and this is the discriminator.**
> With the image features as the *only* input, rank makes no difference whatsoever:
> `img_pca16 − img_t(raw 2048) = +0.00016 [−0.00106, +0.00214]`; `img_pca64 − img_t = +0.00000
> [−0.00001, +0.00003]`; `img_pca256 − img_t = +0.00000 [−0.00000, +0.00000]`.
> **If 16 dims were the visual state's true information content, the image-alone arm would peak at
> 16 and fall. It does not. It is flat to five decimal places.**
>
> The decline is real (`+k16 − +k256 = +0.05142 [+0.01701, +0.08792]`, separated) but it appears
> **only under concatenation**.
>
> ⇒ **MEASURED CONCLUSION: the ladder's shape is NOT a property of the visual content**, because the
> content's own rank curve is flat. It is a property of what happens when image dimensions are
> concatenated onto the ego block. ⇒ **it is not a property of the predictor, and not a property of
> the 51.4° crop either.**
>
> ⚠️ **The MECHANISM is a separate claim and it is a HYPOTHESIS, not a measurement.** The natural
> explanation — a single shared ridge penalty over `[ego | image]` at 198 train clip-clusters, i.e.
> the source stream's own "swamping" reading — **is not confirmed here. My attempt to confirm it by
> simulation FAILED (§1.3).** Nothing in §2–§6 depends on the mechanism; everything depends only on
> the two MEASURED facts above.

**Both validations passed, in both directions** (`artifacts/rank16_reanalysis.json`; 322 held-out
clip clusters, base rate 0.03276, B = 2000):
**C-FID** — all **24** arms reproduce the published `t1_probe.json` AP to < 1e-6, so the npz and the
JSON are provably the same experiment. **C-NEG** — column-shuffled features do **not** separate
(`+0.000176 [−0.002021, +0.002433]`). **Positive control** — `img_t − img_t_SHUFFLED = +0.02900
[+0.01497, +0.04662]`, **separated**, so the instrument *can* fire, and the visual state genuinely
does carry signal (1.89× base) — it simply carries it at every rank equally.

---

# 1. THE (a)-vs-(b) RESOLUTION — the brief's two readings, and why it is neither

The brief posed the fork as: **(a)** the predictor genuinely needs ~16 dimensions ⇒ build a
Perceiver resampler to a fixed small latent set; or **(b)** the 51.4° features are information-poor,
so rank-16 is a property of the crop and must be re-measured after widening.

| reading | verdict | why |
|---|---|---|
| **(a)** "the predictor needs ~16 dims" | ⛔ **REFUTED** | The evidence for "16" is a **+0.00085 non-separated** difference. Evidence that cannot distinguish 16 from **0** cannot license a 16-latent bottleneck. And the image-only ladder is flat, so no rank is preferred by the *content*. |
| **(b)** "rank-16 is a crop property; re-measure after widening" | ⚠️ **THE PREMISE IS FALSE, so the experiment is VOID** | Rank-16 is not a crop property — it is a reader property. **Re-running the ladder on wider-crop features would reproduce the same shape regardless of what the wider crop contains**, because the shape is driven by dimension count against n, not by content. It would answer nothing. |
| **(c)** *the actual finding* | ✅ **the ladder is a READER artifact** | Separated decline under concatenation + flat image-only ladder + non-separated peak. |

## ⭐ 1.1 The priority-1 experiment in my brief should NOT be run, and that is the finding

My brief named *"the rank-16 re-measurement on wider-crop features"* as priority 1, *"it decides the
architecture and nothing else should be designed before it."* **It does not decide the architecture,
and it should not be run.** The ladder's shape is not a measurement of vision, so re-measuring it on
different vision cannot discriminate between geometries. This **removes** a planned experiment rather
than adding one. *(Reported as an instruction I could not carry out as briefed, per rule 5 — not
quietly rescoped.)*

## 1.2 Three further defects in how the ladder has been propagated

1. ⚠️ **The ladder splices two different instruments.** Rungs 1–4 are a **linear ridge probe**
   (`t1_probe.json`). Rung 5 (`k2048 → 1.59×`) is **H2's 2.17 M-parameter attention head** — a
   different reader class, on a different ego baseline (**3.74×**, not 3.659×). The source document
   labels it honestly (`ego + k = 2048 (H2's head_img_ego)`); the *quotation* of it as one
   "monotone dose–response across five points" does not. There is **no `ego+raw2048` arm in
   `t1_probe.json` at all.** (MEASURED — I enumerated the arms.)
2. ⚠️ **It is not a measurement of "the downstream predictor".** The target is `NOT_T_seen`
   (`a_req_seen_res ≥ 0.5 m/s²`, frame-level binary anticipation), read by a **linear probe on
   frozen v1 features**. The world model's dynamics predictor was never in the loop. "Vision enters
   **the predictor** best at rank 16" over-generalises a linear probe on one binary target.
3. ⚠️ **"replicated by two independent streams, all ten arms selecting r=16"** is weaker than it
   reads. The replication's own amendment **A4** records that `head_img_ego_concat`, `head_ego` and
   `head_priv` **do not read the PCA rank at all**, so their `r16`/`r64` rows are *the same run
   repeated*. (INHERITED, `2026-07-26-situation-classifier/SITUATION_CLASSIFIER.md` §8 A4.)

## 1.3 ⛔ THE MECHANISM TEST FAILED — reported as a failure, not quietly dropped

`scripts/rank_ladder_mechanism.py` was built to confirm the shared-λ mechanism by reproducing the
ladder synthetically at the real shapes with a **pure-noise** image block (the discriminator: if
zero-information dimensions reproduce the collapse, the shape cannot be about content), plus a
per-block-λ arm. **It did not reproduce the phenomenon** (`artifacts/rank_ladder_mechanism.json`):

| arm | ladder `ego → +k16 → +k64 → +k256 → +k2048` | collapse | monotone? |
|---|---|---:|---|
| `noise_sharedlam` | 1.264 → 1.261 → 1.265 → 1.275 → 1.209 | **1.04×** | ❌ |
| `signal_sharedlam` | 1.252 → 1.251 → 1.254 → 1.264 → 1.211 | 1.03× | ❌ |
| `noise_perblocklam` (k ≤ 256) | 1.264 → 1.261 → 1.261 → 1.265 | 1.00× | ❌ |
| `signal_perblocklam` (k ≤ 256) | 1.252 → 1.249 → 1.250 → 1.254 | 1.00× | ❌ |

⛔ **The simulation is INADMISSIBLE as evidence in either direction, and the reason is visible in its
own baseline: synthetic `ego_alone` reaches only 1.26× base against the real experiment's 3.66×.**
My generative model is ~3× weaker than the substrate it was meant to stand in for, so the run never
entered the regime where the real collapse lives. **It does not support the shared-λ mechanism and it
does not refute it.** *(Kept and staged rather than deleted: a failed instrument that is silently
discarded is how a program learns nothing — and the next person to try this needs the calibration
failure, not a clean-looking gap.)*

⇒ **The mechanism stays a HYPOTHESIS.** §0's conclusion does not rest on it: the two facts that
matter — the non-separated k=16 peak and the flat image-only ladder — are **MEASURED on real data**
with a validated instrument, and they are sufficient for every architectural call in this document.

---

# 2. WHAT THIS MEANS FOR THE ARCHITECTURE — we already have the thing (a) would build

**Tier: CONFIRMED. Class: MEASURED (code + params).**

> ### ⭐ `SpatialGridReadout` IS ALREADY A FIXED-SIZE RESAMPLER. THE PREDICTOR'S COST IS ALREADY FLAT.
>
> `stack/tanitad/models/readout.py:20-66`: the token grid `[B, N, D]` is average-pooled to a fixed
> `grid × grid_w` cell layout and projected to `d_readout`, giving
> **`state_dim = 4 × 4 × 128 = 2048` for ANY input geometry.** A wide `256×640` input (16×40 tokens,
> pooled 4×10) yields the **same 2048** the predictor, the tactical/strategic policies and every
> grounding head already expect.
>
> ⇒ **The core argument for a Perceiver resampler — *"the encoder pays the wide-FOV cost once, emits
> a fixed small latent set, and the dynamics/predictor cost stays exactly flat"* — describes what the
> stack ALREADY DOES.** Adding a Perceiver would not make the predictor cost flat (it is flat); it
> would only replace a **parameter-free average-pool** with **learned cross-attention**. That is a
> legitimate but entirely separate question, it is *not* what the rank-16 finding licenses, and on
> §0's evidence there is no case for it.

This is independently corroborated by the literature (§3, Perceiver row): Perceiver's own authors use
**512** latents and **8** cross-attends and concede the bottleneck's severity; Flamingo's 64 latents
are a **frozen-encoder VLM captioning** adapter. Neither supports a 16-latent bottleneck in front of a
metric trajectory decoder.

## 2.1 ✅ The recommendation is verified END-TO-END, not asserted

**MEASURED** — the recommended geometry was instantiated and run through the real modules:

```
EncoderConfig(in_channels=9, image_size=256, image_width=640, patch_size=16,
              d_model=768, depth=12, n_heads=12)
  -> ViTEncoder token grid (16, 40) = 640 tokens
  -> SpatialGridReadout(token_grid=(16,40), grid=4, grid_w=4, d_readout=128)
  -> state [2, 2048]        # IDENTICAL to today's 256-token baseline
HFOV: 2*atan(320/266) = 100.53 deg   (today: 2*atan(128/266) = 51.39 deg)
```

⇒ **A wide 640-token encoder drops into the existing stack with `state_dim` unchanged at 2048.**
The predictor, tactical/strategic policies, imagination and every grounding head are untouched. This
is what makes the widening a **data + encoder** change rather than a whole-model redesign, and it is
now demonstrated rather than assumed.

---

# 3. THE TECHNIQUE SURVEY — ranked, with transfer judgement and costed on our shapes

**Class: PUBLISHED, with per-citation FETCH DEPTH (retraction class C16).** Depths are
`FULLTEXT-HTML` (HTML full text read directly) · `ABSTRACT-HTML` · `NOT-FETCHED` (background
knowledge ⇒ **UNVERIFIED**) · `PDF-ONLY` (**no quotation permitted**).
⚠️ **No claim below rests on a PDF summarisation.**

## 3.0 ⛔ FIRST — my own brief's O(N²) premise is WRONG, and it re-points the whole survey

My brief states *"at 640–1600 tokens attention becomes the dominant term."* **It does not.**

`attn_share = 2N²d / (12Nd² + 2N²d) = N / (6d + N)`, with `d = 768`:

| N | 256 (today) | 352 (foveated) | 640 (wide) | 1600 (square) | **4608** |
|---|---:|---:|---:|---:|---:|
| attention share of encoder FLOPs | **5.3 %** | 7.1 % | **12.2 %** | 25.8 % | **50.0 %** |

**Attention reaches 50 % only at N = 6d = 4608 tokens — 7× beyond our largest candidate.**
(MEASURED — deterministic arithmetic, `artifacts/encoder_shape_cost.json:attention_share_check`;
**independently re-derived by the literature sub-task**, two agreeing derivations.)

⇒ **Swin / windowed / linear attention attacks a 5–26 % term and is contra-indicated.** Everything
worth doing attacks **N itself**, which is 100 % of the cost.

## 3.1 The ranking

| # | technique | claimed win (numbers) | measured on what task | fetch depth | transfer to a from-scratch-SSL video WM | under **training**? | TRT-safe (static shapes)? |
|---|---|---|---|---|---|---|---|
| **1** | **Foveated ring tokenization (STT)** — Schmidt & Newcombe, [2506.11131](https://arxiv.org/abs/2506.11131) | 172 tokens vs SAM's 4096; STT-B 30.9 GFLOPs vs SAM-H 6533.7; mIoU 0.412 vs 0.393 Cityscapes | point-prompted segmentation, SA-1B | **FULLTEXT-HTML** | ⭐ **STRONG** — encoder was **MAE-pretrained from scratch on the foveated pattern**, i.e. our regime, not a fine-tuned classifier | ✅ **yes, from scratch** | ✅ fixed 172 tokens |
| **2** | **Register tokens** — Darcet et al., [2309.16588](https://arxiv.org/abs/2309.16588) | 4 registers, FLOP increase "below 2%" | DINOv2/DeiT-III/OpenCLIP artifacts | **ABSTRACT + FULLTEXT-HTML** | **PARTIAL** — cheap insurance; see §3.3, the honest answer is *probably not needed at our scale* | ✅ (added at pretraining) | ✅ |
| **3** | **FlexiViT** — [2212.08013](https://arxiv.org/abs/2212.08013) | one weight set across patch sizes; "usually matches" fixed-patch ViT | classification/detection/seg | **ABSTRACT-HTML** | **PARTIAL** — randomize patch size at train time, export one static size for Orin ⇒ a free deploy-time compute knob | ✅ it *is* a training method | ✅ export one size |
| **4** | **ToMe** — Bolya et al., [2210.09461](https://arxiv.org/abs/2210.09461) | trained from scratch DeiT-S 300 ep r=16: **79.13 vs 79.96** (−0.83), 4.61→2.30 GFLOPs, ~1.5× train speedup | ImageNet-1k; K400 video | **FULLTEXT-HTML (ar5iv)** | **PARTIAL** — merging is by feature similarity ⇒ merged tokens **lose spatial identity**, risky in front of a metric trajectory decoder | ✅ numbers are the trained setting | ✅ reduction is content-independent |
| **5** | **Triplane multi-cam tokenizer** (AR1's *optional* path) | 288 tokens/timestep irrespective of camera count/resolution; **3.9×** | AD planning — **as an option, not AR1's reported runs** | FULLTEXT-HTML for AR1's *description*; **source paper NOT-FETCHED** | **PARTIAL, numbers UNVERIFIED** — the property (tokens decoupled from FOV) is what we want, but it is a 3D-lifting module needing calibrated rig geometry | unknown | ✅ |
| **6** | **NaViT patch-n-pack** — [2307.06304](https://arxiv.org/abs/2307.06304) | matches top ViT at **4× less compute** | JFT-4B classification pretraining | **ABSTRACT + FULLTEXT-HTML** | **WEAK** — the win comes from multi-resolution packing over a hugely diverse corpus. **We have one fixed camera geometry; that mechanism does not exist for us** | ✅ training-only method | ✅ (<2 % padding) |
| **7** | **Perceiver / Perceiver-IO / Flamingo resampler** | Perceiver **512** latents, 8 cross-attends, IN 78.0; Flamingo **64** latents, CIDEr 86.5 vs 83.2 | ImageNet; VLM captioning with **frozen** encoder **and frozen** LM | **FULLTEXT-HTML (ar5iv)** | ⛔ **WEAK / CONTRA-INDICATED** — see §2 (we already have a fixed-size readout) and §3.4 | Perceiver yes; Flamingo's is a frozen-encoder adapter | ✅ |
| **8** | **Sector patch embedding (fisheye)** — [2303.14645](https://arxiv.org/abs/2303.14645) | +0.75 % top-1 ViT, +2.8 % PVT | **synthetic** fisheye ImageNet classification | **ABSTRACT-HTML** | **WEAK** — tiny effect, synthetic distortion, classification only, **no compute claim at all** | yes | ✅ |
| **9** | **Swin / windowed attention** — [2103.14030](https://arxiv.org/abs/2103.14030) | linear in image size; 87.3 % IN-1k | ImageNet/COCO/ADE20k | ABSTRACT-HTML (**returned paraphrased — not quoted**) | ⛔ **CONTRA-INDICATED by §3.0** — attacks a 5–26 % term; hierarchical stages also break the flat token grid the readout and predictor consume | yes | ✅ |
| **10** | **DynamicViT / EViT / A-ViT** | DynamicViT: prune 66 % of tokens → 31–37 % FLOPs, <0.5 % accuracy drop | ImageNet classification | ABSTRACT-HTML (paraphrased); **A-ViT NOT-FETCHED** | ⛔ **NOT-APPLICABLE** — pruning is explicitly **input-dependent**; token count varies per sample | masking yields no real saving without a gather | ⛔ **NO** — breaks TRT engine plans |
| **11** | **Log-polar tokenization** | — | — | — | ⚠️ **no admissible precedent found.** See §3.5 — **PolarFormer is a misattribution risk** | — | — |

## 3.2 ⛔ The Alpamayo AR1 claim in our own repo is WRONG on three counts

`Research/ENCODER_MULTICAM_OPTIMIZATION.md:56-63` states AR1 uses triplane + video tokenization with
*"3.6-20x token compression"*. Checked against `arxiv.org/html/2511.00088v1` (**FULLTEXT-HTML**):

1. **Wrong number** — the paper says **3.9×**, not 3.6×.
2. **Wrong attribution** — the 20× is a *cited* method (Flex, Yang et al. 2025), stated
   prospectively, **NOT-FETCHED** by us and therefore UNVERIFIED.
3. **Wrong status, and this is the important one** — triplane is **optional** ("can additionally
   use"). AR1's **default** tokenizer, *"the one used for all subsequent experiments"*, is the base
   VLM vision encoder 2× downsampled: **160 tokens per image** at 448×280.
   **Every headline AR1 result was produced WITHOUT triplane and WITHOUT the 20× compression.**

⭐ **The genuinely useful datapoint is the opposite of the one we wrote down:** a shipped on-vehicle
AD model runs at **160 tokens per camera** — *fewer than our current 256*. That is evidence that
**640 tokens is a generous budget by deployed-AD standards**, and that the wide arm is affordable
rather than exotic. → §7 flags the repo correction.

## 3.3 Registers — the direct answer

The published sweep **predicts we will NOT see the artifacts**: DINOv2 Tiny/Small/**Base** show none;
"only the three largest models exhibit outliers". **But** DeiT-III-B and OpenCLIP-B (Base scale) *do*,
so it is scale **×** objective, not scale alone — and our objective is neither. Artifacts also emerge
around **37 % depth** in a 40-layer ViT, which does not map cleanly onto our 12 layers.
⇒ **Cheap insurance (<2 % FLOPs, static shapes), but measure our own patch-norm distribution rather
than inheriting either conclusion.** Not a v5 blocker.

## 3.4 Where the literature actively CONTRADICTS adoption

- **Perceiver resampling at small M** — Perceiver's authors concede the bottleneck's severity and
  mitigate with **512 latents / 8 cross-attends**, *more* than our current 256 tokens. Flamingo's 64
  latents win on **caption CIDEr with a frozen encoder and frozen LM**. Cross-attention to an
  unordered latent set destroys the **spatial addressability** our readout, BEV head and predictor
  depend on. Perceiver pays off when N is ~50 k pixels, not when N is 640.
- **Swin / linear attention** — contradicted by §3.0's arithmetic, not by the papers.
- **DynamicViT / A-ViT / EViT** — input-dependent token counts violate the stated TRT constraint.

## 3.5 ⚠️ Two traps for whoever writes this up next

- **PolarFormer ([2206.15398](https://arxiv.org/abs/2206.15398)) is NOT log-polar image
  tokenization.** It applies polar coordinates to the **BEV/output** space and takes ordinary
  multi-camera images as input. It must not be cited as precedent for log-polar tokenization.
- **The strongest-looking foveation result is CONFOUNDED.** Look-Focus-Act
  ([2507.15833](https://arxiv.org/abs/2507.15833), FULLTEXT-HTML) reports foveated 20 tokens vs fine
  324 (16×). But its **Coarse baseline is also 20 tokens** (126.9 vs 115.6 GFLOPs) — **the compute
  win is token count, not foveation.** Foveation's own contribution is only readable at matched
  budget, and *that* comparison is confounded with **gaze**: their fovea is steered by a human/
  predicted gaze point. **We have no gaze signal**; our fovea would be *statically* placed near the
  vanishing point — a different and weaker proposition. **This is the same shape as the `nav_cmd=None`
  REF-C confound. Do not let this paper's success rates decide a GPU-day.**

## 3.6 ⚠️ What could NOT be verified — treat as UNVERIFIED

Ivanovic et al. 2025 (the triplane tokenizer itself) · Flex / Yang et al. 2025 (the 20×) · AR1 §6.6
results · "Perceiver resamplers lose fine-grained detail" (seen only in search snippets; the specific
figures circulating are **UNVERIFIED — do not quote**) · Look-Focus-Act's input resolution (the fetch
returned "NOT FOUND IN TEXT", so we cannot compute what angular resolution its 20 tokens cover) ·
A-ViT (not fetched) · EViT and Swin abstracts returned **paraphrased**, so nothing is quoted from
them · FlexiViT per-task numbers · **all TRT-safety judgements are ESTIMATED** (inferred from
static-vs-dynamic token count; no TensorRT documentation was fetched) · STT's "24×" is our division,
the abstract states no ratio.
⚠️ **"No published foveated tokenization in a video world-model setting" is a SINGLE-PROBE absence
and is therefore NOT established** — per the two-probe rule, only "not found on one search".

---

# 4. ENCODER SIZE — the params budget is NOT the binding constraint

**Tier: DECISION-GRADE for params/tokens/FLOPs (deterministic). Class: MEASURED,
`artifacts/encoder_shape_cost.json`.**

Backbone measured as instantiated from the flagship-v1 camera-encoder defaults
(`dynamics_encoder.py:221-229`: `d_model 768, depth 12, n_heads 12, patch 16, in_channels 9`).

| geometry | HFOV @ `F_REF` | px/deg | tokens | ×tok | **encoder params** | pos-embed | ×FLOPs | attn % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **today 256×256** | 51.39° | 4.64 | 256 | 1.00 | **87,022,848** | 196,608 | 1.00 | 5.3 % |
| **wide 640×256** | **100.53°** | **4.64** | 640 | 2.50 | **87,317,760** | 491,520 | **2.70** | 12.2 % |
| square 640×640 | 100.53° | 4.64 | 1600 | 6.25 | 88,055,040 | 1,228,800 | **7.98** | 25.8 % |
| **foveated 640×256, shared wings** | **100.53°** | 4.64 centre / 2.32 wings | **352** | 1.38 | **87,096,576** | 270,336 | **1.40** | 7.1 % |
| foveated 640×256, separate wing conv | 100.53° | same | 352 | 1.38 | 94,175,232 | 270,336 | 1.40 | 7.1 % |

## 4.0 ⚠️ FIRST — a headline program number does not reproduce, and I could not make it

**MEASURED, by instantiating `flagship4b_config()` → `WorldModel` and counting:**

| component | params | share of deployable |
|---|---:|---:|
| **encoder** | **87,022,848** | **33.0 %** |
| predictor | 91,360,512 | 34.7 % |
| tactical_pred | 26,534,912 | 10.1 % |
| tactical_policy | 22,736,141 | 8.6 % |
| imagination | 22,055,683 | 8.4 % |
| strategic_policy | 8,385,027 | 3.2 % |
| inv_dyn | 5,246,978 | 2.0 % |
| readout | 98,432 | 0.0 % |
| **WorldModel total (deployable)** | **263,440,533** | 100 % |
| `HierarchicalGrounding` (training-time, lives OUTSIDE the model) | 13,432,338 | — |
| **total trainable** | **276,872,871** | — |

⛔ **CLAUDE.md and the registry quote the flagship at 286.34 M. I cannot reproduce that number from
`flagship4b_config()`: the full trainable total is 276,872,871 — short by 9,467,129.** The config's
own docstring says *"Measured total ~260 M"*, consistent with the 263.44 M measured here, not with
286.34 M. **I am not silently picking one** — reported as a discrepancy (→ E6). It does not change any
conclusion below: the encoder is **33.0 %** of the deployable model / **31.4 %** of trainable, and the
widening cost is the same either way.

## 4.1 The three findings

**1. ⭐ The encoder is 87.02 M — 33.0 % of the deployable flagship (31.4 % of trainable).**
(MEASURED here, both the numerator and the denominator.)

**2. ⭐⭐ WIDENING THE FIELD IS ESSENTIALLY FREE IN PARAMETERS.** A ViT's parameters are independent
of token count **except the positional embedding**:
- 256 → 640 tokens: **+294,912 params = +0.34 % of the encoder, +0.11 % of the program.**
- 256 → 1600 tokens: +1,032,192 = +1.19 % of the encoder.
⇒ **The sub-300M invariant is NOT the constraint on widening.** The cost is **compute and activation
memory**, not parameters. Any argument against a wider field framed as "we cannot afford the
parameters" is answered: at 640 tokens the deployable model goes **263,440,533 → 263,735,445**, and
even the 1600-token square option lands at 264,472,725 — **all comfortably inside sub-300M.**

**3. ⚠️ But foveation's *naive* implementation is NOT free — and the fix is a one-line design change.**
A separate 32×32 wing convolution costs `9 × 32² × 768 = 7.08 M` extra params (**+8.1 % encoder**).
**The shared-weight variant avoids it entirely:** average-pool the wings 2× and apply the **same**
16×16 centre convolution. Identical token geometry, **+73,728 params (the positional embedding
alone)**, and the wing tokens land in the centre's embedding space rather than a separate one.
**If foveation is ever adopted, it must be the shared-weight form.** (MEASURED — both were built and
counted.)

## 4.15 ⭐ THE PROJECTION CHANGES THE TOKEN BILL BY ~30 % — and it is cheaper the right way

**MEASURED (deterministic arithmetic on `calib.py`'s own two formulas; `F_REF = 266`, patch 16,
16 token rows).** `calib.py` already ships **two** azimuth conventions —
`pinhole: HFOV = 2·atan((W/2)/f_ref)` and `cylindrical: HFOV = 2·(W/2)/f_ref` (equidistant).
**How many tokens 100–120° costs depends on which one v5 uses**, and nobody has costed that:

| width | token cols | **tokens** | ×FLOPs | **pinhole HFOV** | **cylindrical HFOV** | within the 120.5° sensor? |
|---:|---:|---:|---:|---:|---:|---|
| 256 (today) | 16 | 256 | 1.00 | 51.39° | 55.14° | ✅ |
| **448** | 28 | **448** | **1.82** | 80.20° | **96.50°** | ✅ |
| **512** | 32 | **512** | **2.11** | 87.81° | **110.28°** | ✅ |
| 576 | 36 | 576 | 2.40 | 94.55° | 124.07° | ⛔ **over** |
| **640** | 40 | **640** | **2.70** | **100.53°** | 137.85° | ⛔ **over** |

⇒ **Under a cylindrical projection the PI's 100–120° target is reached at 448–512 tokens, not 640 —
a 22–30 % smaller token bill (1.82–2.11× vs 2.70×) for the SAME field.** And cylindrical is
independently the better projection for this sensor: `calib.py` records that at a 50° half-angle
`tan 50° = 1.19`, so **a pinhole rectification spends 19 % of its new pixels stretching the
periphery**, while equidistant azimuth spends them uniformly.
⚠️ **Corollary the other way:** under cylindrical, **640 px would be 137.9° — beyond the camera's
120.5° physical field.** A width chosen under the pinhole convention and then run through a
cylindrical projection would silently ask for pixels the sensor never captured.

⛔ **A HARD CONSTRAINT ON THE WIDTH, and it is easy to miss:** `SpatialGridReadout` asserts
`token_cols % grid_w == 0`, and `grid_w` must stay **4** to hold `state_dim = 4×4×128 = 2048`.
⇒ **the input width must be a multiple of 64.** 464 px (the exact 100° cylindrical width) is **not**,
so it would break the readout — 448 or 512 are the nearest admissible widths.

**Coordination, not duplication:** *which* projection is `2026-07-27-geometry-configurable`'s and
`fov-crop-audit`'s call — `calib.py` already implements both. **This section supplies the token/compute
price of that choice, which is my lane, and it is a real input to their decision.** → E7.

## 4.2 Should the encoder be BIGGER for a wider field?

**No — not on current evidence, and the program's own history says the opposite.**
`raw-2048` is 17–25 % *worse* than rank-16 downstream (INHERITED); the frozen-DINO encoder was a
ceiling, not a gift (INHERITED, `MODEL_REGISTRY.md`); and §0 shows the visual state's usable signal is
**1.89× base and flat in rank** — a representation that is not capacity-limited at 2048 dimensions.
**There is no measured evidence that width or depth is the binding constraint.** A wider field at
constant `d_model`/`depth` gives the encoder **2.5× more tokens to look at with the same weights**,
which is the cheap direction. ⇒ **v5 keeps `d_model 768, depth 12`.** Revisit only if the §5
validation shows the wide arm *under-fitting* (train and held-out both improving at stop).

## 4.3 ⛔ THROUGHPUT IS REFUSED — the instrument declared itself invalid

The dev box was measured at **100 % foreign GPU utilisation with 5.8 GiB resident** throughout.
A **contention sentinel** (the same reference workload run before and after the sweep) drifted
**1.732× (73 %)**, and per-config latencies moved up to **5×** between two runs of the same script
(`square_640x640` train 1104.57 → 213.84 ms/sample). The spill filter fired once
(`foveated_shared`) at a peak allocation of **1.26 GiB** — nowhere near the 8 GiB card, i.e. a
**contention artifact, not a spill.**

⇒ **No latency or throughput number from this run is quotable, and none is quoted.** The admissible
compute claim is the **analytic FLOP ratio** (deterministic, and independently re-derived twice).
This is exactly the hazard the brief named, and the guard caught it rather than shipping fiction.
**To close: re-run `encoder_shape_cost.py` on an idle host** (it self-certifies via
`contention_sentinel.CONTENDED`).

---

# 5. THE SMALL VALIDATION — PRE-REGISTERED, written before any number exists

**This section is registered BEFORE any arm has been trained.** It is the PI's *"we will consider all
new trainings with the larger hfov after a small validation"*, and it gates all new training.

## 5.1 What it must decide, and what it must NOT be asked to decide

**Decides:** does v5 train at **today's 51.4° / 256 px / 256 tokens** or at **~100° / 640×256 /
640 tokens**?
**Does NOT decide** (owned elsewhere, do not duplicate): *which exact FOV and resolution* is best
(`2026-07-27-fov-crop-audit`, a frozen-encoder probe sweep over crops) · *the training-path plumbing*
(`2026-07-27-geometry-configurable`) · *comma2k19's role in the mix* (PI call, cost to be measured).
**These are complementary instruments:** the sibling asks *"is the information out there?"* on a
frozen encoder; this asks *"does a trained-from-scratch encoder use it?"*

⚠️ **The comparator problem, and why matched short runs are the only valid answer.** Every bar in the
PREP card §4 was registered against **old-geometry** numbers. **v1's 0.4271 is NOT a valid comparator
for a wide arm** — v1's encoder was *trained* at 51.4°, so feeding it 100° frames is out-of-distribution
and the number is meaningless in either direction. Re-evaluating v1 wide is invalid; training v5 at
both geometries costs two GPU-weeks. **Matched short runs are the only cheap valid design.**

## 5.2 Arms — identical in everything but geometry

Identical episodes (parity set `physicalai-train-e438721ae894`, skip-hash `f09e44db` — **a re-cache,
not a re-selection**), identical seed, identical step count, identical optimizer/schedule, identical
loss weights. **Only the input geometry and the encoder's `image_width` change.**

| arm | input | `f_eff` | HFOV | px/deg | tokens | isolates |
|---|---|---:|---:|---:|---:|---|
| **A_narrow** | 256×256 | 266 | 51.39° | 4.64 | 256 | **the baseline — today's geometry** |
| **B_wide** | 256×**W** | 266 | ~100–110° | 4.64 | **W/16 × 16** | **the candidate: more field at constant acuity** |
| **C_fov_only** | 256×256 | 107.4 | 100° | **1.88** | 256 | FOV **without** paying tokens — the "free lunch" arm |
| **D_blur** *(known-worse control)* | 256×256 | 266 | 51.39° | **1.88** | 256 | **2.48× acuity loss, ZERO information added** |

**`W` is set by the projection the geometry stream selects (§4.15), and must be a multiple of 64:**
`W = 512` under **cylindrical** (110.28°, **512 tokens**, 2.11× encoder FLOPs) or `W = 640` under
**pinhole** (100.53°, 640 tokens, 2.70×). ⚠️ **`B_wide` must declare its projection in the run
record** — the two conventions give different fields at the same width, and 640 px under cylindrical
would be 137.9°, beyond the sensor.

⚠️ **`D_blur` is the power guard, not a candidate.** It degrades acuity by exactly the factor
`C_fov_only` pays, while adding no field. **If the instrument cannot detect `D_blur` as worse with a
separated CI, it has no power to rank geometries at all.**

## 5.3 Metrics — the primary is the composite, NOT ADE

**Primary: the map-free composite (PDMS-lite).** ADE is a **diagnostic only**. Two independent lines
say ADE is the wrong target: the ADE-optimal pick collides **4.7×** more often (3.36 % vs 0.71 %,
separated), and published L2/ADE vs closed-loop Driving Score is **ρ = −0.36, p = 0.43** while Ego
Progress is **ρ = 0.83**. (INHERITED, PREP card §0.)
**Kill secondary:** `wm_canary_ade_2s ≤ 0.55` (carried verbatim, no new threshold).
**Estimator:** paired episode-cluster bootstrap, `taniteval/ci.py`, B = 2000, unit = **episode
cluster**. Arms share episodes, so every contrast is paired.

## 5.4 ⚠️ THE MDE — stated before the run, because a guard that cannot fail is class C13

**This is the part the program has shipped broken several times, so it is stated first and bluntly.**

The composite's *measured* resolution on this program is poor. At the T3 run's n, the paired CI on a
PDMS-lite delta had a half-width of **±0.0028** (`BCE_RULE − CE_CONTROL = +0.0002 [−0.0025,
+0.0031]`, INHERITED). Worse, **two of its terms carry no information on our fan** — DAC is missing
and comfort is a **literal constant (100.0000 % violation over 1,708,288 candidates)** — and the three
trained arms occupy **0.6096–0.6100, i.e. 0.2 % of the distance to random (0.3968)**.

⇒ **Registered MDE: the validation can only detect a composite delta of `|Δ| ≳ 0.006`** (2× the
measured paired half-width) **at T3's n.** Two consequences, both accepted in advance:

1. **If the geometry effect is smaller than 0.006 on the composite, this validation CANNOT detect it,
   and the honest output is `UNPOWERED`, not `NO DIFFERENCE`.** That outcome is registered as
   admissible and must be reported as such.
2. **Therefore the run is powered by n, and n is stated up front:** the validation runs on the
   **600-episode deployment set**, not the 40-episode val split. *(A 40-episode "not separated" is
   UNPOWERED, not refuted — PREP card §4.)* If ≥ 200 episode clusters are not available for a
   stratum, that stratum returns `UNPOWERED` and no verdict is emitted for it.

⚠️ **`D_blur` is what makes the MDE claim checkable rather than asserted:** it is a deliberately
large, known-signed effect. If a 2.48× acuity loss does not clear 0.006, the MDE estimate above is
**optimistic** and the whole validation returns `INSTRUMENT-BLIND`.

## 5.5 Outcome rules — fixed now, executed in code, and able to FAIL

Let `Δ(X) = composite(X) − composite(A_narrow)`, paired, B = 2000.

- **✅ WIDE WINS** — `Δ(B_wide)` CI excludes 0 **upward**, and `wm_canary_ade_2s ≤ 0.55` on `B_wide`.
  ⇒ **v5 trains at 100.6° / 640×256 / 640 tokens.**
- **⛔ WIDE LOSES** — `Δ(B_wide)` CI excludes 0 **downward**. ⇒ **v5 trains at today's geometry, and
  the PI's hypothesis is reported as REFUTED**, as cleanly as a win would have been.
- **➖ NO DIFFERENCE** — `Δ(B_wide)` CI contains 0 **and** `Δ(D_blur)` CI excludes 0 downward (the
  instrument demonstrably has power). ⇒ **v5 trains at today's geometry** (the wide arm costs 2.70×
  encoder FLOPs for no measured gain, and the burden of proof is on the change).
- **⚠️ FREE LUNCH** — `Δ(C_fov_only)` CI excludes 0 upward. ⇒ field matters more than acuity; take
  100° at **256 tokens** and pay nothing at all. *(Registered because it is cheap and would be the
  best outcome available; not expected.)*
- **⛔ INSTRUMENT-BLIND** — `Δ(D_blur)` CI does **not** exclude 0 downward. ⇒ **no verdict on any arm.**
  A sweep that cannot detect a 2.48× acuity loss cannot rank geometries. This returns instead of
  `NO DIFFERENCE`, which is the failure the program has shipped before.
- **⚠️ UNPOWERED** — fewer than 200 episode clusters, or `|Δ| < 0.006` for every arm with all CIs
  containing 0. ⇒ **reported as UNPOWERED, never as "no effect".**

## 5.6 Bi-directional validation (both required, both able to fail)

- **V-FID (fidelity).** `A_narrow` rebuilt under the new configurable-geometry path must reproduce
  today's cached crop **bitwise** on a sample of clips. If the "old geometry" arm is not byte-identical
  to today's pipeline, the comparison is not a geometry comparison and nothing is quotable.
  *(This also closes the silent-collision hole the PREP card records: build params carried `size` but
  never `f_ref`, so changing `F_REF` produced different pixels under the same cache key.)*
- **V-NEG (deliberately failing input).** A geometry arm fed **column-shuffled** frames must land at
  chance on the composite. If it separates, the harness is leaking.

## 5.7 Cost

**Class: ESTIMATED — and it is the one number here I could not measure**, because throughput was
refused (§4.3). Encoder FLOPs are **2.70×** for `B_wide` (MEASURED, deterministic). The encoder's
share of a training step is INHERITED at "60 %+ of our tick"
(`Research/ENCODER_MULTICAM_OPTIMIZATION.md`, **not re-verified**), which would put a wide step at
**~2.0× today's**. ⚠️ **This must be measured on the actual training host before the runs are sized** —
it is the difference between a 4-hour and an 8-hour validation. **Registered as a gap, not guessed.**

---

# 6. THE RECOMMENDATION — one plain paragraph

> **v5 should train on a WIDE, NON-SQUARE frame at `height 256`, `f_eff = 266`, uniform
> `patch_size = 16`, with the width set by the projection the geometry stream picks — `512 px`
> (110.3°, **512 tokens**, 2.11×) under **cylindrical**, or `640 px` (100.5°, 640 tokens, 2.70×) under
> **pinhole**. Cylindrical at 512 is the better buy: more field for fewer tokens.** The width **must
> be a multiple of 64** (§4.15). **Conditional on §5's validation returning WIDE WINS or FREE LUNCH.**
> Keep `d_model = 768`, `depth = 12`, `state_dim = 2048` and the existing `SpatialGridReadout`
> (`grid_w = 4`) — the readout is the geometry firewall and holds the predictor, the policies and
> every grounding head unchanged (**verified end-to-end, §2.1**). **Adopt no tokenization trick in
> v5.** Widening costs **+0.34 % parameters**, attention is still only **11–12 %** of encoder cost,
> and a shipped on-vehicle AD model runs at **160 tokens/camera** — 512–640 is affordable and carries
> zero research risk. Foveation (§4.1, **shared-weight form only**, 352 tokens, 1.40×) is the
> strongest *next* lever and the only survey entry with a from-scratch-SSL precedent, but its best
> published evidence is **confounded with gaze** (§3.5) and it should be earned by its own experiment
> **after** v5, not bundled into a GPU-week. **Reject** Perceiver resampling (we already have a
> fixed-size readout — §2), Swin/linear attention (attacks an 11–12 % term — §3.0), and all
> input-dependent pruning (breaks TRT). **Free options worth a line each if convenient:** 4 register
> tokens (<2 % FLOPs) and FlexiViT-style patch-size randomization (a deploy-time compute knob at no
> architectural cost).

**And the finding that outranks all of the above:** ⛔ **strike *"Vision enters at rank ≈ 16"* from the
PREP card's VALIDATED table (§1).** It is not validated. It is a non-separated point estimate whose
ladder shape is a reader artifact, and it must not license an architecture.

---

# 7. ESCALATIONS — decisions and corrections that need a human, not a README

*(Rule 3: escalate integration; do not write "please merge" into a doc nobody re-reads.)*

| # | escalation | who | why it cannot wait |
|---|---|---|---|
| **E1** | ⛔ **Strike "Vision enters at rank ≈ 16" from `flagship-v5-retrain.PREP.md` §1 VALIDATED**, and add the correction to `RETRACTION_LOG.md`. **Root-cause class: an unpaired point-estimate ladder quoted as a measured effect — a shape with no interval on it.** *(Nearest existing class: the `overlapping_holdout_se` family — a central value quoted without a valid estimator. This one is arguably worse: there was no estimator at all on the contrast.)* | PI / orchestrator | It is listed as VALIDATED and was about to license an encoder redesign in a GPU-week run. |
| **E2** | ⛔ **Correct `Research/ENCODER_MULTICAM_OPTIMIZATION.md:56-63`** — AR1's triplane/20× claim is wrong on number, attribution **and status** (§3.2). The correct, more useful fact is **160 tokens/camera by default**. | architecture stream | It is the repo's only "efficient AD tokenization" reference and it argues for a technique the source paper did not use. |
| **E3** | ⚠️ **Tell `2026-07-27-fov-crop-audit` and `2026-07-27-geometry-configurable` that the rank-16 premise is void.** The audit's Part 2 uses **"PCA rank 16 (the measured dose-response optimum)"** as a *pre-registered* choice. It remains a defensible dimensionality reduction (the image-only ladder is flat, so rank 16 costs nothing) — **but its stated justification is wrong and should be restated as "flat in rank, so 16 is chosen for cheapness", not "the measured optimum".** | both siblings | Their pre-registrations cite a now-void justification. The *choice* survives; the *reason* does not. |
| **E4** | ⚠️ **The v5 geometry decision needs the encoder-share-of-step measured on the training host** (§5.7). | v5 launch owner | It sizes the validation and is the only unmeasured number in the plan. |
| **E5** | ⚠️ **`MODEL_REGISTRY.md` needs the GEOMETRY column** the PREP card already called for; every pre-2026-07-27 row becomes a 51.4° / 256 px historical number. | registry owner | Without it, a wide v5 will be compared to narrow v1 by someone reading a table. |
| **E7** | ⭐ **The projection choice is also a COMPUTE choice, and it has not been costed anywhere.** Cylindrical reaches 96.5–110.3° at **448–512 tokens (1.82–2.11×)**; pinhole needs **640 tokens (2.70×)** for 100.5° (§4.15). **And the input width must be a multiple of 64** or `SpatialGridReadout` breaks `state_dim = 2048`. | `geometry-configurable` + `fov-crop-audit` | They own the projection call and are choosing it now; without this table they may pick a width that is 30 % more expensive than needed, or one the readout rejects. |
| **E6** | ⛔ **The flagship's quoted 286.34 M does not reproduce from `flagship4b_config()`.** MEASURED: `WorldModel` = **263,440,533**; + `HierarchicalGrounding` = **276,872,871** — **9,467,129 short**. The config docstring itself says *"Measured total ~260 M"*. **Either the quoted figure or the config is wrong, and both are load-bearing** (sub-300M is a program invariant). Needs the registry/model-registry agent, not a guess from me. | registry owner | It is quoted in `CLAUDE.md` — the file every agent reads — and it is the denominator of every "fraction of budget" claim in the program. |

---

# 8. DELIVERABLE MANIFEST

**Everything below is STAGED in the repo working tree (`git add`). Nothing is committed, nothing is
pushed, no branch was switched. Nothing lives in only one place.**

| artifact | path | class |
|---|---|---|
| this document | `repo: TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-27-encoder-tokenization/ENCODER_TOKENIZATION.md` | report |
| ⭐ the missing paired contrasts | `repo: …/2026-07-27-encoder-tokenization/scripts/rank16_reanalysis.py` | code |
| ⭐ their output (§0, §1) | `repo: …/2026-07-27-encoder-tokenization/artifacts/rank16_reanalysis.json` | MEASURED |
| params / tokens / FLOPs / throughput bench | `repo: …/2026-07-27-encoder-tokenization/scripts/encoder_shape_cost.py` | code |
| its output (§4) — **latency fields REFUSED, see `contention_sentinel`** | `repo: …/2026-07-27-encoder-tokenization/artifacts/encoder_shape_cost.json` | MEASURED (params/FLOPs) |
| ⛔ ladder-mechanism test — **FAILED, kept deliberately** (§1.3) | `repo: …/2026-07-27-encoder-tokenization/scripts/rank_ladder_mechanism.py` | code |
| its output — **INADMISSIBLE, calibration failure recorded** | `repo: …/2026-07-27-encoder-tokenization/artifacts/rank_ladder_mechanism.json` | negative result |
| citations + fetch depths (§3) | `repo: …/2026-07-27-encoder-tokenization/CITATIONS.md` | PUBLISHED |

**Source artifacts read, not modified:**
`…/2026-07-26-situation-semantics/artifacts/{t1_probe.json, t1_heldout_scores.npz}` ·
`stack/tanitad/models/{dynamics_encoder.py, encoder.py, readout.py}` · `stack/tanitad/config.py` ·
`taniteval/taniteval/ci.py` · `Project Steering/Gates/flagship-v5-retrain.PREP.md`.

**What this unblocks:** `Project Steering/Gates/flagship-v5-retrain.PREP.md` **item 7** — the encoder
size/architecture question is answered (§4, §6), the tokenization question is answered (§3, §6), and
the **small validation the PI asked for is pre-registered with its bar, its n, its MDE and a guard
that can fail** (§5). ⚠️ **Item 7's own "priority 1" — re-measuring the rank-16 dose-response on wider
crops — is struck as void (§1.1), and its VALIDATED entry must be struck too (E1).**

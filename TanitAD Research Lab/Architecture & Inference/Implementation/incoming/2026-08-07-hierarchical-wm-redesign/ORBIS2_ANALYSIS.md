# Orbis 2 — architecture, data, results, and what it changes for v6

**PI, 2026-08-14:** *"Let analyse Orbis 2, which architecture, which data and which results
did they achieve?"*

**Why this one matters more than any other external system we have looked at:** Orbis 2 is
**the first published *hierarchical* driving world model**, it uses a **compressed DINOv2
latent as its high-level target** (structurally the same move as our readout firewall), and
it **fine-tunes on NVIDIA PhysicalAI-AV** — *our corpus family*. It is our nearest published
analogue on three axes at once. Nobody else in the frontier tables is.

---

## ⚠️ Evidence class — read this before quoting anything below

Everything here is **PUBLISHED (cited)**, but **NOT from the paper PDF**. `arxiv.org`,
`openreview.net`, `lmb-freiburg.github.io`, `semanticscholar.org`, `huggingface.co`,
`alphaxiv.org` and `automotiveworld.com` are **all egress-blocked in this environment** —
`WebFetch` returns `EGRESS_BLOCKED` for each. Every figure below comes from **search-index
snippets of `arXiv:2607.15898`** plus the press release and the `lmb-freiburg/orbis` GitHub
README.

⇒ **Treat these as PUBLISHED-via-snippet, one confidence step below PUBLISHED-exact.**
Before any of them decides a GPU-day, someone with unblocked egress must confirm against the
PDF. Two figures already disagree between sources (§4) — which is exactly the reason this
caveat is at the top rather than in a footnote.

**Paper:** *Orbis 2: A Hierarchical World Model for Driving*, [arXiv:2607.15898](https://arxiv.org/abs/2607.15898)
(University of Freiburg LMB + NATIX). Lineage: [Orbis v1, arXiv:2507.13162](https://arxiv.org/abs/2507.13162).

---

## 1. Architecture

**The factorisation.** Orbis 2 splits future prediction across **two levels at distinct
temporal *and* abstraction scales**:

| level | job | horizon | operates on |
|---|---|---|---|
| **high-level predictor** | forecasts **coarse scene structure** | long | a **compressed DINOv2 latent** |
| **low-level generator** | produces **fine-grained detail**, conditioned on the high-level output | short | pixel-space video tokens |

Quoted: *"a high-level predictor that forecasts coarse scene structure over extended temporal
horizons, and a low-level generator that produces detailed predictions conditioned on the
high-level output. This decomposition yields high perceptual fidelity while also capturing
strong spatial and semantic representations."*

**The backbone.** The DiT block is extended into a **Spatio-Temporal Transformer
(ST-Transformer)**. The low-level detail predictor is attached by **spatially-aligned
conditioning via adaLN** — i.e. the high-level forecast modulates the detail generator's
normalisation, it is not concatenated as tokens.

**⭐ The high-level target is a compressed DINOv2.** A *lightweight projection* compresses
DINOv2 features into a **lower-dimensional latent**, and the paper credits that compression
with **stabilising training and improving long-horizon rollout quality**. The high-level
tokenizer is trained **primarily for DINOv2B embedding reconstruction, with a small pixel
reconstruction weight** — so the abstract level is explicitly *not* a pixel autoencoder.

### 1.1 Parameter budget

| component | params |
|---|---:|
| ST-Transformer backbone (per level) | **512 M** |
| low-level detail conditioning (adaLN, spatially aligned) | **+43 M** |
| → low-level generator subtotal as reported | **555 M** |
| **full hierarchical predictor** | **1 067 M** |

⚠️ **The reconciliation 512 + 43 = 555, and 512 + 555 = 1 067, is MINE (INFERRED), not
quoted.** It is the only arithmetic that closes, and it implies **both levels are 512 M
ST-Transformers** with the detail level carrying an extra 43 M of conditioning. Do not quote
the split as published until the PDF is read.

### 1.2 Training paradigm — the two-stage objective swap

**Pretrain with diffusion forcing → fine-tune with teacher forcing.** Quoted: *"Pretraining
with a diffusion forcing objective yields substantially richer internal representations than
the standard teacher forcing objective, while teacher forcing produces more stable
autoregressive rollouts."* Diffusion forcing specifically improved **linear-probe semantic
segmentation and depth**.

⇒ They found **the objective that builds the best representation is NOT the objective that
rolls out best**, and rather than choosing, they staged them. That is a real, transferable
finding and it is the single most actionable thing in the paper for us (§5.3).

---

## 2. Data

| | |
|---|---|
| **predictor training** | **a SINGLE epoch over 5 890 hours** of video, both levels |
| **tokenizer training** | **2.6 M frames**, curated mix: BDD100K · OpenDV · Honda HAD · Honda HDD · ONCE · nuScenes · nuPlan |
| **largest single source** | the **NATIX decentralised dashcam network — >40 %** of training data |
| **steering-conditional fine-tune** | a **500-hour subset of NVIDIA PhysicalAI-AV**, using **IMU annotations** to compute planar trajectory inputs |
| **held-out generalisation set** | **Waymo** (never trained on) |

**Three things worth noticing, because each is a decision for us:**

1. **One epoch over 5 890 hours** — they are firmly in the *fresh-data* regime, never the
   repeat regime. We are at **0.75 epochs over 13.3 hours** (`V6_DATA_REQUIREMENT.md`).
   Same epoch count, **443× less material**.
2. **The tokenizers and the predictors were trained on DIFFERENT data** — 2.6 M curated
   frames for the representation, 5 890 h of raw video for the dynamics. The representation
   problem and the dynamics problem were given separate, separately-sized corpora.
3. **PhysicalAI-AV appears only as a 500-hour ACTION-CONDITIONING fine-tune**, not as the
   pretraining corpus. They use our corpus for exactly the one thing it uniquely supplies
   (ego trajectory ↔ video pairing) and get the *world* from elsewhere.

---

## 3. Results

| axis | result |
|---|---|
| **headline** | outperforms **NVIDIA Cosmos-v2.5** with *"roughly half the parameters and one-third of the data"* |
| **generalisation** | beats Cosmos-v2.5 on **unseen Waymo** data |
| **long-horizon fidelity** | **FVD on 6 s rollouts** — SOTA |
| **representation quality** | **linear probing for semantic segmentation (Cityscapes) and depth** — SOTA |
| **steering responsiveness** | SOTA on **counterfactual** scenarios |
| **inference** | **3.64 FPS at 19 GB VRAM** — reported as the fastest in the comparison |
| **claimed overall** | SOTA across long-horizon generation fidelity, steering responsiveness, and internal representation quality |

### 3.1 The efficiency table — the part that should reframe our own cost thinking

| model | params | training compute |
|---|---:|---|
| GAIA-1 world model | 6.5 B | **23 k A100-h** |
| Epona | 2.5 B | **16 k A100-h** |
| Orbis **v1** | 469 M | **8 k A100-h** |
| **Orbis 2** | **1.07 B** | **< 6 k H100-h** (≈ 6 × 10²¹ FLOPs at 25 % MFU) |

⇒ **Orbis 2 is 2.3× the parameters of Orbis v1 at a fraction of the compute** — and roughly
**4× cheaper than GAIA-1 at 1/6 the size**. Their own framing is that **hierarchy is an
EFFICIENCY lever, not a capacity lever.**

⚠️ **DISCREPANCY, flagged rather than smoothed:** the paper snippet says **"under 6k
H100-GPU-hours"**; the [press release](https://www.automotiveworld.com/news/university-of-freiburg-and-natix-unveil-orbis-2-model/)
says **"under 3,000 H100 GPU-hours"**. These are not the same number. Most likely the 3 k is
one level or one stage and the 6 k is the total, but **that is a guess.** ⇒ **Quote "<6 k
H100-h (paper snippet); a press release states <3 k — unreconciled"** and never the bare 3 k.
*(This is the `MODEL_REGISTRY` rule in an external costume: the press summary and the primary
source disagree, so the primary source wins and the disagreement gets written down.)*

### 3.2 ⚠️ What is NOT in these results

- **No driving score.** No nuPlan, no NAVSIM, no PDMS, no closed-loop planning metric.
  Orbis 2 is evaluated as a **generative world model and a representation learner** — FVD,
  linear probes, counterfactual steering response. It **does not claim to drive.**
- **No T1-equivalent.** "Steering responsiveness on counterfactual scenarios" is the closest
  thing to our O1 counterfactual-action-response probe, and it is a *generation* metric
  (does the video change when you change the steering input), not an *action-closed-loop*
  metric. By `EVAL_DOCTRINE.md` this is a **T0-family diagnostic**, not a capability claim
  about driving.
- ⇒ **Orbis 2 and v6 are not comparable end-to-end**, and no number from it may enter
  `MODEL_REGISTRY.md` as a baseline for our ADE / four-families tiers. What transfers are
  **architectural and data-regime facts**, not scores.

---

## 4. Orbis 2 vs v6 — where we align and where we invert

| dimension | Orbis 2 | v6 (config E, running) |
|---|---|---|
| hierarchy | ✅ two levels, distinct temporal + abstraction scale | ✅ three bands (operative / tactical / strategic) |
| high-level target | **frozen-pretrained DINOv2B, compressed** | **self-trained ViT-5 encoder → 4×4×128 readout** |
| pixel generation | ✅ yes (low-level generator) | ❌ never renders — latent-only |
| **parameter mass** | **on the ABSTRACT level** (512 M high-level) | **on the OPERATIVE level (68.6 %)**; hierarchy = **12 %** |
| action conditioning | high-level, steering, fine-tuned separately | operative predictor, FiLM seam from tactical intent |
| training corpus | **5 890 h**, one epoch | **13.3 h**, 0.75 epochs |
| compute | **< 6 k H100-h** | ~150 h A40-class for S-W 30 k |
| evaluation | FVD + linear probes + counterfactual steering (T0-family) | T1 action-closed-loop + four metric families |

**⭐ The sharpest single finding: their parameter allocation is the INVERSE of ours, and it
is the third independent system to say so.** V-JEPA 2 spends 1 B of 1.2 B on the encoder and
22 M on the predictor. DINO-WM freezes the encoder entirely and spends everything on a 19 M
predictor. Orbis 2 spends its mass on the **abstract, long-horizon** level. **v6 spends 68.6 %
on the short-horizon operative predictor and leaves the hierarchy — the programme's entire
thesis — at 12 %.** No system in this class allocates the way we do.

⇒ This is **direct external support for config D over config C** in `V6_SIZE_VS_FRONTIER.md`:
D is the only in-band configuration that grows the *hierarchy* (56.7 M, 19.6 %) rather than
growing the operative predictor further. C reaches 260 M by scaling the imbalance.

⚠️ **The honest counter, stated because it is real:** Orbis 2's "low level" is a **pixel
generator**, and pixel generation is intrinsically parameter-hungry. Some of their 555 M
bottom level is paid for rendering, which we deliberately do not do. So the inversion is
*suggestive*, not dispositive — but the *high-level 512 M* is not a renderer, and that is the
number that indicts our 10.6 M hierarchy.

---

## 5. What this changes — three concrete items, priority-ordered

### 5.1 It substantially strengthens **lever P1: the frozen pretrained-encoder arm**

`V6_DATA_REQUIREMENT.md` recommended a frozen DINOv2 / V-JEPA 2 encoder arm on the argument
that it converts the encoder's data requirement from *ours* (13.3 h) to *theirs*. Orbis 2 is
now the **third** system to take that route, and the first to do so **for driving,
hierarchically, with a compression projection**, and to report that the compression itself
**stabilised training and improved long-horizon rollout**.

That last clause is the new information. Our readout (4×4 grid × 128 → state_dim 2048) is
already a compression projection and is already declared a geometry firewall — **Orbis 2
supplies external evidence that this shape helps long-horizon rollout**, which is exactly the
regime our 6 s / K=60 contract lives in. ⇒ **P1 moves from "worth an arm" to "the best-supported
unrun experiment in the programme."**

⚠️ **The cost we still own:** DINOv2B is 3-channel, narrow-FOV, non-driving-pretrained. Ours
is **9-channel wide-FOV cylindrical**. The adapter that bridges that is real work and is
precisely what the arm must measure — Orbis 2 does not de-risk it, because their input is
ordinary forward video.

### 5.2 The **objective-staging** idea transfers, and is nearly free

Their two-stage swap — *representation-optimal objective first, rollout-stable objective
second* — is not diffusion-specific. In our terms the analogue is: **pretrain S-W under the
aggressive masking / independent-per-step corruption schedule (best representation), then
fine-tune under the clean teacher-forced 6 s rollout (best rollout stability)**, rather than
running one fixed schedule for 30 k steps.

⇒ **Work item: a masking-schedule stage split for S-W.** It costs a scheduler change and no
new parameters. ⚠️ **HYPOTHESIS, not a prediction** — their evidence is for diffusion forcing
in a generative model, and our corruption is a different operator. It is a cheap
discriminating experiment, which is the bar; it is not a result.

### 5.3 It sets a **defensible data target**, which we did not previously have

`V6_DATA_REQUIREMENT.md` had to reach for Chinchilla (an LLM-on-text law, explicitly marked
non-transferable) and the V-JEPA 2 ratio (1 : 75 000, a *general video* pretrain). Orbis 2
gives us the number from **the same task, the same class of model, and the same corpus
family**: **5 890 hours of driving video, one epoch, for a 1.07 B hierarchical model.**

| | Orbis 2 | v6 today | ratio |
|---|---:|---:|---:|
| unique driving hours | **5 890 h** | **13.3 h** | **443×** |
| params | 1 067 M | 336.5 M | 3.2× |
| **hours per M-param** | **5.52** | **0.040** | **139×** |

⇒ **Normalised for model size we are 139× under the nearest comparable system.** That is a
far more defensible statement of the deficit than the Chinchilla ratio, because it is
task-matched. **It also directly ranks the levers:** lever 2 (enlarge the pretraining corpus
under a new declared parity key) is the only one that closes a 443× gap; levers 1, 4 and 5
are worth single-digit factors at best.

⚠️ **And it sharpens the tension with the 250–350 M directive.** At 5.52 h/M-param, the
corpus that would justify 336 M is **~1 860 hours**. The augmentation set takes us to ~26 h.
**Even the full PhysicalAI release is unlikely to reach 1 860 h**, which means the *only*
routes to a defensible 336 M are (a) a frozen pretrained encoder that imports someone else's
hours — §5.1 — or (b) a smaller model. Those are the two options; there is not a third.

---

## 6. Open, and not guessed

- **The PDF was never read.** Every number above is a search-index snippet. §4's
  compute figure is **internally contradicted between sources** and is flagged, not resolved.
- **The 512/43/1067 split is my arithmetic**, not their table.
- **No numeric FVD, mIoU or depth value is quoted here** — the snippets named the axes and
  the winner, never the values. Do not let "SOTA on FVD" become a number in a later doc; that
  is precisely the prose-propagation failure `CLAUDE.md` opens with.
- **No driving/planning score exists to compare against v6** (§3.2), and none should be
  invented by analogy.

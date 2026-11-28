# FROZEN ENCODERS IN ROBOTICS AND DRIVING — what the successful systems actually do, and where REF-A sat

**Date:** 2026-08-18 · **Question (PI):** *how have recent successful works in robotics and automated
driving used frozen encoders — for world models and for direct policy learning / planning — and what
is decisive in making a REF-A-like architecture work: the architecture, the training process, the
data, or something else?*

**Evidence class on every claim below.** `PUBLISHED-PRIMARY` = read from the paper's own arXiv HTML.
`PUBLISHED-SECONDARY` = read from an aggregator summary of the paper, **not yet confirmed against the
primary PDF** — admissible for orientation, ⛔ **not admissible for a paper citation until re-read**.
`MEASURED` = our own numbers, from `MODEL_REGISTRY.md` / banked raw JSON.

---

## 0. ⭐ THE ANSWER IN ONE PARAGRAPH

Frozen encoders work — in **two** distinct configurations, and REF-A was in **neither**. Configuration
**A** (driving, supervised head): freeze a *very large* VLM encoder, feed it *many cameras at high
resolution*, and keep a wide interface — `FROST-Drive` beats the *same encoder fine-tuned*, but its
frozen encoder is a 14–78 B VLM over 5 cameras at 448². Configuration **B** (robotics + driving world
models): freeze a *moderate* encoder (DINOv2 / V-JEPA / DINOv3) but change what sits on top — predict
**future features** self-supervised, and get behaviour from **test-time optimisation (CEM/MPC)** or
from a head that is *additionally* supervised by a feature-prediction loss. REF-A combined a **small
frozen encoder (DINOv2-B/14, 86 M) + a narrow low-FOV interface (256 tokens, 51.39°, monocular) + a
purely supervised action head + no test-time planning.** ⭐ **That specific combination is the one
configuration in which no published system succeeds.** The decisive factor is therefore **not**
"frozen vs not" and **not** the data volume — it is the pairing of *encoder strength × visual
interface width* with *what the objective asks the frozen features to do*.

---

## 1. THE EVIDENCE — configuration A: frozen encoder, supervised head, DRIVING

### `FROST-Drive` (arXiv 2601.03460, WACV-W 2026) — the decisive AD datapoint `PUBLISHED-PRIMARY`

Waymo Open E2E Dataset (4,021 scenes, 20 s at 4 Hz), **5 cameras at 448×448**, 5 s horizon
(H = 20 waypoints), 256 visual tokens per camera fused to 256, metric = Rater Feedback Score (RFS).
Architecture: **frozen** VLM vision encoder → transformer adapter → GRU waypoint decoder.

| arm | RFS ↑ | ADE@3s ↓ | ADE@5s ↓ |
|---|---|---|---|
| **frozen VLM 78 B** | **8.24** | **0.95 m** | **1.74 m** |
| frozen VLM 38 B | 8.17 | 0.98 m | 1.83 m |
| frozen VLM 14 B | 8.17 | 1.04 m | 1.88 m |
| ⚠️ **VLM 14 B, FINE-TUNED** | 8.13 | **1.47 m** | 2.19 m |
| frozen VLM 1 B | 8.09 | 1.84 m | 2.42 m |
| ViT, fully fine-tuned | 7.79 | 1.20 m | 2.15 m |
| ⛔ **ViT (ImageNet), FROZEN** | **7.39** | **2.28 m** | **3.08 m** |

Three readings, and the third is the one that matters to us:

1. **Freezing beat fine-tuning for the same encoder** (14 B: 8.17/1.04 frozen vs 8.13/1.47 tuned).
   Authors' mechanism: *"fine-tuning can degrade the rich, generalizable world knowledge learned
   during pre-training, forcing the model to over-specialize"* — the model wins its ADE back on the
   common cases and loses the tail.
2. **The frozen ImageNet ViT is the WORST arm in the entire table** — worse than fully fine-tuning the
   same ViT. ⇒ ⭐ **freezing is a multiplier on pre-training quality, not an independent good.**
   Frozen+strong > tuned+strong > tuned+weak > **frozen+weak**.
3. **Interface width is a real lever:** embedding-dim ablation on the frozen 38 B encoder gives
   RFS 8.17 at 5120-d vs **7.68 at 256-d** — same encoder, same data, ~0.5 RFS from the width of the
   channel between frozen features and the decoder.

Leaderboard: 3rd overall (RFS 7.856), **2nd in the "Spotlight" edge-case category** — i.e. the frozen
arm's advantage concentrates exactly where generalisation is tested.

### `DeepSight` (arXiv 2605.10564) — frozen encoder *plus* feature prediction, driving `PUBLISHED-PRIMARY`

**Vision encoder frozen; LLM fully fine-tuned.** Predicts **latent semantic features of 5 future
frames (2 s) in BEV**, targets extracted by **DINOv3-ViT-L/16**; loss = trajectory CE + CoT CE +
**MSE world-state loss against DINOv3 features**. Has a supervised trajectory head *as well*.
Bench2Drive **closed-loop**: DS 86.23 (+7.39 over prior SOTA), SR 71.36 % (+13.63). nuScenes
open-loop L2@3s 0.52 m, collision 0.27 %. Trained on 10,000 Bench2Drive segments.

### `LAW` (ICLR 2025) — self-supervised latent feature prediction as an *auxiliary* `PUBLISHED-SECONDARY`

Predicts future latent scene features from current features + ego trajectory; the self-supervised task
plugs into both perception-free and perception-based frameworks. Reported SOTA on nuScenes (open-loop),
NAVSIM, and CARLA (closed-loop). ⇒ the *feature-prediction objective* is independently load-bearing in
driving, even when the encoder is not frozen.

---

## 2. THE EVIDENCE — configuration B: frozen encoder, feature prediction, TEST-TIME PLANNING

### `DINO-WM` (ICML 2025, arXiv 2411.04983) — REF-A's actual ancestor `PUBLISHED-SECONDARY`

**Frozen:** DINOv2 **spatial patch** embeddings. **Trained:** only a causal ViT transition model + an
action→latent MLP. **Loss:** plain L2 between predicted and ground-truth patch embeddings — *"no
auxiliary reconstruction, reward, or terminal losses."* **No policy head at all.** Behaviour comes
from **CEM + MPC** at test time (CEM preferred because the latent dynamics are non-smooth and
gradient shooting fails); re-encode after each executed action.

| | PointMaze SR | Push-T SR | Wall SR | Rope CD ↓ | Granular CD ↓ |
|---|---|---|---|---|---|
| **DINO-WM** | 0.98 | **0.90** | 0.96 | **0.41** | **0.26** |
| DreamerV3 | 1.00 | 0.04 | 1.00 | 2.49 | 1.05 |
| IRIS | 0.74 | 0.32 | 0.04 | 1.11 | 0.37 |
| TD-MPC2 | 0.00 | 0.00 | 0.00 | 2.52 | 1.21 |

⭐ **The one ablation we must not ignore:** replacing spatial patch embeddings with *global* R3M,
ResNet-18, **or the DINOv2 CLS token** *"leads to significantly degraded performance, especially on
manipulation and spatial reasoning tasks."* Patch-level structure is the load-bearing property, not
"DINOv2-ness".

### `V-JEPA 2-AC` (Meta, 2025) — frozen encoder + action-conditioned predictor `PUBLISHED-SECONDARY`

Frozen web-scale video ViT; a **compact** transformer learns action-conditioned latent dynamics with
an L1 latent loss, post-trained on **< 62 hours** of Droid. No policy: planning by MPC minimising a
goal-conditioned energy to a goal image. Zero-shot in unseen labs, **65–80 %** on reach / grasp /
pick-and-place.

### `GPC` (arXiv 2502.00622) — the pattern stated explicitly `PUBLISHED-SECONDARY`

Wraps a **frozen** behaviour-cloning policy with an action-conditioned world model for online
planning, motivated by BC policies being *"brittle at deployment, lacking explicit mechanisms for
test-time correction"*. ⇒ in this literature the frozen-feature world model earns its keep **at
planning time**, not at fitting time.

---

## 3. THE EVIDENCE — what happens when you FINE-TUNE instead

* **`Preserving Pretrained Representations` (arXiv 2509.11417)** `PUBLISHED-PRIMARY`: direct
  fine-tuning of a VLM on robot data causes measurable representation degradation — OpenVLA PickCan
  drops **36.7 % → 12.1 %** under paraphrased instructions; background masking **55.5 % vs 76.4 %**.
  Their fix is the **middle path**: a *dual encoder* — one **frozen anchor** + one **trainable**
  branch, concatenated. Ablation: baseline 35.03 → **+dual encoder 55.55** → +string tokenizer
  +co-training **78.46**.
* **CortexBench / VC-1** `PUBLISHED-SECONDARY`: across 17 embodied tasks **no single frozen
  pre-trained representation dominates**; but **task/domain-specific adaptation of VC-1 yields
  substantial gains**, matching or beating the best known result on every benchmark.

⇒ ⭐ **The field's answer to "frozen or fine-tuned?" is "neither — adapt partially."** Dual encoders,
adapters/LoRA on late blocks, task-conditioned adaptation, co-training. Full fine-tuning and full
freezing are both corner cases that lose to partial adaptation in the majority of published ablations.

---

## 4. A KNOWN, DOCUMENTED DEFECT OF EXACTLY OUR CONFIGURATION `PUBLISHED-SECONDARY`

DINOv2 emits a single feature map at **H/14**, documented as insufficient for **small and distant
objects**, and it *"struggles to produce strong features for objects near image boundaries"* (it is
optimised for objects fully inside the frame). Recommended remedy in that literature: a multi-scale
encoder.

REF-A ran DINOv2-B/14 at **224×224 → 256 tokens → 51.39° HFOV, monocular** (`MEASURED`, the
pretrained-encoder-arm build). A lead vehicle at 60 m occupies a fraction of one patch. Our own v6
line already moved to **224×560 → 640 tokens → 120°**. ⇒ This is a *visual-interface* defect that is
independent of "is DINOv2 a good encoder", and it is the cheapest thing on this list to test.

---

## 5. ⭐ THE DECISIVE AXES, RANKED — and where REF-A sat on each

| # | axis | what the literature says | REF-A | verdict |
|---|---|---|---|---|
| 1 | **What the objective asks of the frozen features** | Successful frozen systems ask them to be *propagated forward in time* (feature prediction); REF-A asked a small adapter to *re-encode them into a control manifold* under a supervised head | supervised head, no feature-prediction loss | ⭐ **prime suspect** |
| 2 | **Where behaviour comes from** | CEM/MPC at test time (DINO-WM, V-JEPA2-AC, GPC) or head+world-model-aux (DeepSight, LAW) | feed-forward regression only | ⭐ **prime suspect** |
| 3 | **Strength of what is frozen** | frozen+weak is the **worst** arm in FROST-Drive; frozen+78B is the best | DINOv2-B, 86 M, image-only SSL | **major** |
| 4 | **Width of the visual interface** (tokens, FOV, cameras, embedding dim) | 5 cams @448², 5120-d; patch tokens mandatory, CLS/global fails | 1 cam, 256 tokens, 51.39°, narrow adapter | **major, cheapest to fix** |
| 5 | **Full fine-tuning** | degrades pretrained structure; partial adaptation wins | n/a (fully frozen) | not the fix |
| 6 | **Data volume** | 62 h (V-JEPA2-AC); 10 k segments (DeepSight); no expert demos (DINO-WM) | 2,376 episodes | ⛔ **least likely to be decisive** |
| 7 | **Trainable capacity on top** | successful systems train a **compact** predictor + MLP | 4-brain stack | ⚠️ cuts *against* "REF-A lost for want of 116.9 M params" |

⚠️ **Note axis 7 against our own open question.** The reconciliation stream lists REF-A's 116.9 M
trainable-parameter deficit as one of four unattributed differences. This literature does not support
capacity as the mechanism: DINO-WM and V-JEPA 2-AC beat heavier baselines with *deliberately small*
predictors. What they never do is ask that small module to solve a supervised control problem.

---

## 6. HOW THIS MEETS OUR OWN MEASUREMENTS

* Our five-stage ladder found the information **present and preserved** at every stage of REF-A's
  pipeline — raw features → trained adapter → predictor latent — with the trained adapter *not*
  collapsed (`MEASURED`, verdict `P2-PRESERVED`). The literature says the loss in a frozen-encoder
  system is on the **consumption** side. **Both point to the same place.**
* The only quantity REF-A's training separably improved was **ego speed** — exactly the target its
  aux losses named (`MEASURED`). That is precisely what axis 1 predicts: supervision moves what it
  points at and nothing else, because the adapter is not being asked to *preserve a world*, only to
  *hit a label*.
* ⛔ **But the recipe imports a dependency we have already measured as broken:** our CEM planner was
  **35.8 % worse than the constant-velocity baseline at T1** (`MEASURED`, C101 — "the loss is in the
  ACTION SEARCH, not the WM"). Configuration B concentrates its benefit exactly in that component.
  ⇒ **Adopting DINO-WM's recipe without repairing or replacing the action search would move our
  known-worst component onto the critical path.** This is the honest counterweight to §5.

---

## 7. WHAT I WOULD CHANGE IN THE PROGRAMME'S PLAN

1. ⭐ **Sharpen (do not cancel) `E-RECON-2`.** It varies *frozen vs trainable* holding architecture
   fixed. This review says the higher-information axis is **objective + consumer**, not trainability.
   A cheaper, sharper cell exists: **hold REF-A's already-cached frozen DINOv2 features constant
   (zero encode cost) and vary ONLY what consumes them** — the banked supervised head (REF-A, already
   run) vs a DINO-WM-style feature-prediction predictor. One variable, one training run, no new
   encoder pass. `E-RECON-2` remains the right *capacity/representation* discriminator; this is the
   *objective* discriminator, and §5 ranks it first.
2. **Promote backlog item #26 ("v3: DINO-WM proper")** from "someday" to the encoder question's main
   line — it is the exact recipe every successful frozen-encoder system uses, and we already wrote it
   down independently.
3. **Sequence the planner first, or borrow theirs.** Given C101, run the feature-prediction arm with
   DINO-WM's own CEM/MPC configuration (CEM, not gradient shooting; re-encode every step) rather than
   our current search, and treat the planner as part of the arm under test.
4. **Two cheap, high-value probes before any training spend:** (a) re-encode a REF-A slice at
   **640 tokens / 120°** and re-measure the readout ladder — axis 4, hours not days; (b) widen the
   adapter's output dim and re-run the banked admission — FROST-Drive measured ~0.5 RFS on this axis
   alone.
5. **If a pretrained encoder is ever fine-tuned here, do it as a DUAL ENCODER** (frozen anchor +
   trainable branch, concatenated), not as full fine-tuning — that is the configuration with the
   measured +20.5 pt gain, and full fine-tuning is measured to *degrade* pretrained structure.

---

## 8. LIMITATIONS OF THIS REVIEW — stated before use

* Five of the twelve numeric claims are `PUBLISHED-SECONDARY` (aggregator summaries): DINO-WM's
  ablation table, V-JEPA 2-AC's success rates, LAW's benchmark claims, CortexBench's adaptation
  result, and the DINOv2 small-object limitation. ⛔ **Re-read against the primary PDFs before any of
  these enters a paper or the registry.**
* FROST-Drive's Table 3/4 numbers are `PUBLISHED-PRIMARY` but come from one workshop paper on one
  dataset with one metric (RFS); its frozen-vs-tuned gap on ADE@3s (1.04 vs 1.47) is large, its RFS
  gap (8.17 vs 8.13) is small. **The two metrics disagree in magnitude** — do not quote the RFS gap
  as the headline.
* No claim here is a claim about *our* arms. Everything in §5–§7 is a **hypothesis ranking**, not a
  result. The only measured statements about REF-A are the ones tagged `MEASURED` in §6.
* Search coverage: 7 web searches + 6 primary fetches, English-language, arXiv-weighted. Absence of a
  counterexample here is **not** evidence of absence.

# How much data does v6 need — and how to need far less

**PI, 2026-08-13:** *"how can we derive the necessary amount of training data for our
approach and how can we massively reduce the amount of data by improving the data
distribution, improving the training process or leveraging any proven ideas in recent
research work."*

**The headline, MEASURED: we are data-limited by roughly two orders of magnitude, and
compute-limited by none.** At 30 k steps the running S-W sees **0.75 epochs** — it will not
even finish one pass over its own corpus. That single number reframes every other lever.

---

## 1. The accounting — MEASURED from the running job

| quantity | value |
|---|---|
| corpus | 2 400 episodes ≈ **13.3 hours** of driving |
| valid windows at k=60 | **319 002** (415 002 at k=20 — longer futures, fewer valid windows) |
| samples seen at 30 k × batch 8 | 240 000 → **0.75 epochs** |
| unique latent-steps | 319 002 × 66 = **21.1 M** |
| model | **336.5 M** parameters |
| Chinchilla-style target (20 tokens/param) | **6.73 B** |
| **unique data ÷ that target** | **0.31 %** |
| V-JEPA 2 pretraining | **> 1 000 000 hours** → ratio **1 : 75 000** |

⚠️ **Read the Chinchilla row carefully.** [20 tokens/parameter](https://lifearchitect.ai/chinchilla/)
is an LLM-on-text law; a latent world model on video is a different regime and the constant
is not transferable. What *is* transferable is the **order of magnitude**, and 0.31 % is not
a calibration quibble. The V-JEPA 2 ratio is the same message from our own family.

⭐ **The uncomfortable implication, stated plainly.** Config E/F at 336 M was chosen to sit
in the PI's 250–350 M band. On 13.3 hours of unique video that is **heavily
over-parameterised by frontier standards** — the model has ~16 parameters for every unique
latent-step it will ever see. This does not mean E is wrong; it means **the binding
constraint moved from parameters to data**, and spending the next effort on parameters
would be spending it in the wrong place.

---

## 2. Where the data actually is — the biggest single lever

**PhysicalAI-AV is far larger than 2 400 episodes. We restricted to 2 400 to hold PARITY.**
Parity exists so arms are comparable — it is a *measurement* discipline, not a *pretraining*
one. And v6 is a new architecture that no banked number compares against window-for-window.

> **Recommendation P0: pretrain S-W on a LARGER corpus under a NEW parity key, and keep
> evaluating on the SAME 40 val episodes.** Training-set size does not break comparability;
> re-selecting the *evaluation* set does. This is the one lever that moves the deficit by an
> order of magnitude rather than a few percent, and it costs no research risk.

⛔ **It must be a NEW, DECLARED parity key** (`physicalai-train-<hash>-vN`), never a silent
widening of the existing one — the invariant that "anything which re-selects episodes must
be refused" is what makes every v1→v5.8f comparison trustworthy, and it stays.

The augmentation set (4 729 clips ≈ 26 h) roughly doubles the corpus; the full PhysicalAI
release is much larger again.

---

## 3. Training-process levers — how to need less of it

### 3.1 We can afford **4× more epochs than we are taking** — free, today
[Scaling Data-Constrained Language Models](https://jmlr.org/papers/volume26/24-1000/24-1000.pdf)
finds repeated data keeps nearly the value of fresh data for the **first ~4 epochs**, decaying
after. We are at **0.75**. So the same corpus supports **~120 k steps** before the repeat
penalty meaningfully bites.
⇒ **The current 30 k step budget is not a data limit, it is an arbitrary stopping point.**
The right move is to run until the P-battery plateaus, not until a round number.

### 3.2 Saliency sampling — already implemented, and worth re-tuning
O4 draws windows by ego-kinematic saliency, **label-free**: `weight_max_over_min` **15.5×**
at k=60 (30.8× at k=20). Driving corpora are dominated by straight cruising; this is exactly
the QQT ("quality-quantity trade-off") lever the curation literature describes, and it is on.
⚠️ But `alpha=1.0` and `floor=0.25` were never swept. A cheap α ∈ {0.5, 1, 2} sweep at
step-500 is a real data-efficiency experiment with a measured gate.

### 3.3 Deduplication — untried, and driving data is the ideal case
"Repeated data is worth less" is the strongest consistent finding in the curation
literature, and *near-duplicate* windows are rampant here: consecutive windows of a car
waiting at a light are near-identical. Our windows are strided, not deduplicated.
⇒ **Latent-space near-duplicate pruning** (cluster the encoded window latents, cap per
cluster) is directly applicable, needs no labels, and is the standard
[concept-cluster pruning](https://arxiv.org/pdf/2401.04578) recipe.

### 3.4 The proven small-data results are about REASONING, not perception
LIMO and s1 show tiny curated sets beating large ones — but for *reasoning fine-tuning*, on
top of a model that already learned the world from a huge pretrain. ⚠️ **Do not read them as
"we can learn driving physics from 13 hours."** They apply to our **S-T/S-S** stages (goal
supervision on a frozen trunk), where a few thousand well-chosen labelled clips genuinely
may suffice — which is a real and encouraging result for the PH0 pipeline, and no help at
all for S-W.

### 3.5 The strongest structural lever: **don't learn the visual world from scratch**
[V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) spends 1 B of its
1.2 B parameters on an encoder pretrained on a million hours, then adapts with *a small
amount* of robot data. [DINO-WM](https://dino-wm.github.io/) goes further: **freeze a
pretrained encoder entirely** and train only a ~19 M predictor. [Orbis 2](https://www.automotiveworld.com/news/university-of-freiburg-and-natix-unveil-orbis-2-model/)
compresses **DINOv2** features as its high-level latent and beat Cosmos-v2.5 on **one third
of the data**.

> **Recommendation P1: run an arm with a FROZEN pretrained visual encoder** (DINOv2 or
> V-JEPA 2), our readout on top, and train only the predictor + hierarchy. It converts the
> encoder's data requirement from *ours* to *theirs* — exactly the 1 : 75 000 gap — and our
> architecture already supports it: the readout is a declared geometry firewall, and
> `shared_encoder`/E-ENC is an existing switch.
> ⚠️ Costs: their pretraining is not driving-specific and not 9-channel/wide-FOV; the
> adapter must bridge that. That is precisely what an arm measures.

---

## 4. Recommended order (cheapest discriminating experiment first)

| # | lever | cost | expected effect |
|---|---|---|---|
| 1 | **Train longer on the same data** (30 k → 120 k) | pure compute | 4× the effective data budget; free today |
| 2 | **Enlarge the pretraining corpus** under a new parity key | staging time | the order-of-magnitude lever |
| 3 | **Frozen pretrained encoder arm** (DINOv2 / V-JEPA 2) | ~1 day to wire | removes the encoder's data need entirely |
| 4 | O4 α sweep at step 500 | ~2 GPU-h | measured distribution tuning |
| 5 | Latent near-duplicate pruning | ~1 day | more unique information per step |

⚠️ **What I will not claim:** none of these is measured to fix our defects. The T1 failures
were conditioning and selection; E-ENC measured that more encoder capacity did not help at
step 500. This document says the *binding constraint* is data — it does not promise that
relieving it produces frontier driving.

**And the honest tension with the 250–350 M directive:** at 13.3 hours, a 336 M model is
over-parameterised. Levers 1–3 are what make that size *earn* its parameters. If none of
them is taken, the defensible size on this corpus is considerably smaller.

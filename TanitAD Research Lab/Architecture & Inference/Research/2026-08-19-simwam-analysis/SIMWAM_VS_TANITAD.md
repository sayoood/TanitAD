# SimWAM (2608.07468v3) vs the TanitAD programme — a hypothesis-by-hypothesis match

`PUBLISHED (cited)` for every SimWAM number · `MEASURED (ours)` for every TanitAD number,
each with its tier stamp · paper banked as the PRIMARY source, not read from a summary.

**Zhao et al., *SimWAM: A Simple World Action Model for End-to-End Autonomous Driving*,
Huazhong UST + Dongfeng R&D. Code + weights: `github.com/H-EmbodVis/SimWAM`.**

---

## 1. What the paper actually claims, and the one sentence that matters to us

World-Action Models transfer video-dynamics priors into action prediction. Existing driving
WAMs (DriveLaW, DriveWAM, Epona) use an **imagine-then-act** factorisation —

```
p(a | o,s,l) = ∫ p(z_future | o,s,l) · p(a | o,s,l,z_future) dz_future
```

— which puts video synthesis **inside the real-time planning loop**. SimWAM replaces it with

```
p(a | o,s,l) = p(a | z(o_t), s, l)
```

and moves future prediction to **training-time supervision only**.

> ⭐ **THE LOAD-BEARING CLAIM:** *"explicit future synthesis is unnecessary for effective
> world-action learning"* — video co-training helps **through training-time representation
> learning, not test-time future imagination.**

**Mechanism — the isolated attention mask.** One shared attention stream carries current-obs
latents `z(o_t)`, future-frame latents, and action tokens. Future-frame tokens and action
tokens **both attend to `z(o_t)` but are mutually invisible**. That single mask is the entire
structural modification. At inference the future branch (and its VAE decoder) is deleted.

**Their own ablation is the decisive evidence** (Tab. 3, PDMS):

| mask | PDMS |
|---|---|
| bidirectional | 90.2 |
| action → video | 90.1 |
| **isolated** | **90.3** |

⇒ *"exposing the action branch to the future-video tokens provides **no measurable benefit**"* —
while costing the whole future-generation latency.

---

## 2. Results, data, compute

| | |
|---|---|
| **Architecture** | video expert = Wan2.2-5B video DiT + its VAE + T5; action expert = 1.02B DiT, `d=1024`. **No shared parameters** — they interact only through attention. |
| **Total size** | ~**6B** |
| **Training** | joint flow matching, `L = L_act + λ·L_vid`, λ=1, AdamW, cosine, lr 1e-4, **100 epochs** |
| **Compute** | **32 GPUs** (`NNODES=4 × NPROC_PER_NODE=8`), DeepSpeed ZeRO-1. Duration not stated. |
| **Input** | **single front camera**, 384×672 |
| **Output** | 8 waypoints / 4 s / 2 Hz; video target 8 frames / 4 s / 2 Hz |
| **RL** | Flow-GRPO — ODE→SDE for exploration, LoRA rank 32 α=16 on action-expert attention only, G=8 samples, lr 5e-5, reward = NAVSIM PDM |
| **NAVSIM** | navtrain 103,288 scenes → navtest 12,146 |
| **PhysicalAI-AV** | **65K sampled frames** from the 150K training clips |

**Headline results (PUBLISHED):**

| benchmark | result |
|---|---|
| NAVSIM navtest | **91.5 PDMS** (SOTA), at substantially lower latency than WAM planners |
| NAVSIM-v2 navhard | **37.6 EPDMS**, +5.4 over Metis — **without RL** |
| nuScenes zero-shot | 0.96 m avg L2, **0.04 % collision** (best), no fine-tuning |

**Ablation ladder:** action-only **86.6** → + video co-training **90.3** → + RL **91.5**.
So the video prior is worth **+3.7 PDMS** and RL a further **+1.2**.

### 2b. ⭐ PhysicalAI-AV — they evaluate on OUR corpus

| method | params | ADE@3s ↓ | FDE@3s | ADE@4s | FDE@4s |
|---|---|---|---|---|---|
| VaVAM | 1.3B | 2.31 | 4.32 | — | — |
| Alpamayo-1.5 (NVIDIA) | 10B, ~80,000 h | 0.80 | 2.31 | 1.44 | 4.18 |
| DriveWAM | 5B + 8B | 0.47 | 1.35 | 0.83 | 2.47 |
| **SimWAM** | **6B**, 65K samples | **0.40** | **1.08** | **0.69** | **1.96** |

⛔ **DO NOT PUT OUR 0.4522 IN THIS TABLE.** Three incomparabilities, any one of which
invalidates the comparison:

1. **Different metric semantics.** Our `ade_0_2s = 0.4522` is stamped in the registry as
   ***"a WORLD-MODEL FIDELITY number, NOT a planning number"*** — T0. SimWAM's ADE is
   **open-loop planning**.
2. **Different horizon.** Ours is **2 s**; theirs 3 s and 4 s.
3. **Different split.** Ours is the 40-episode val (881 windows); theirs is *"the same
   1,000-clip test subset adopted by DriveWAM"* — neither is the other.

**What can honestly be said:** SimWAM at **6B** reaches 0.40 m ADE@3s on a PhysicalAI subset;
our deployed v1 is **263M** and our *closed-loop* (T1) numbers on our own benchmark are
catastrophic by comparison (`v5f` cl ADE **23.98 m**, stage-A-repaired **9.37 m**, with a
hold-action control at **0.42 m** beating the model by 22×). **The gap that matters is not
open-loop ADE — it is that we have no closed-loop competence and they report a strong
closed-loop-ish PDMS.**

---

## 3. Where SimWAM CONVERGES with what we already measured

### 3a. ⭐⭐ Imagination-in-the-loop hurts — we found it first, and then kept building on it

| source | evidence |
|---|---|
| **TanitAD, MEASURED** | flagship v1: open-loop ADE **0.45 m → closed-loop 1.69 m** (imagination-in-the-loop). "Open-loop does NOT predict closed-loop." |
| **TanitAD, MEASURED** | `v6.py:3879` refuses an imagination-led selection *verbatim*: *"an imagined-consistency-led refinement is the **REFUTED** roll-cost selection rule wearing MPC's name"*, and `mpc_w_consist` **defaults to 0.0** — imagination contributes **nothing** to selection at default. |
| **SimWAM, PUBLISHED** | isolated mask ≥ bidirectional (90.3 vs 90.2); future tokens give the action branch no measurable benefit. |

⇒ **Two independent programmes, opposite scales, same conclusion.** This is the strongest
external corroboration TanitAD has received on any architectural question. It also means our
`mpc_w_consist=0.0` default was right and should be **stated as a result**, not left as a
default nobody defends.

### 3b. Horizon coverage beats frame density

SimWAM Tab. 8: 4f/**2s**/2Hz **89.9** · 4f/**4s**/1Hz **90.2** · 8f/**4s**/2Hz **90.3**
⇒ *"broad temporal coverage is more important than dense frame sampling."*

This **directly supports REF-A v1's design choice** of three rates all reaching exactly 6.0 s
(op 0.2×30, tac 0.6×10, str 1.5×4) rather than one dense short window. Independent support for
a decision we made on hierarchy grounds alone.

### 3c. RL on the HARD subset beats RL on everything

SimWAM Fig. 3: training GRPO only on navtrain scenes with imitation PDMS < 90 **consistently
beats** training on all scenes; peak 91.5 at 15k steps, declining after. Reason given: easy
scenes *"contribute limited learning signals and dilute the benefit."*

⇒ Maps onto our **`sel_gap`** problem (`v5f`: *"the SELECTOR is the defect, not the curve"*).
If we ever post-train a selector, the training set should be the scenes where the selector
currently fails — not the corpus.

---

## 4. Where SimWAM CHALLENGES us

### 4a. ⛔ It is a direct challenge to REF-A v1's test-time planning

REF-A v1 as designed this session adopts DINO-WM's CEM/MPC at **test time**: 300 samples,
iCEM coloured noise β=2.5, coarse-to-fine, measured **21.5 s/tick**. SimWAM's entire thesis is
that this is the expensive half and it is **unnecessary** — and its Fig. 1 shows WAM planners
paying 1,600–3,200 ms for *worse* PDMS than its ~518 ms.

⚠️ **But the challenge is not fatal, and the distinction matters:**
* SimWAM's ablation removes **future-frame conditioning of an imitation policy**. It does
  **not** test **planning over a learned cost** — there is no search in SimWAM at all; the
  action expert is a flow-matching imitator refined by RL.
* DINO-WM's claim is about **zero-shot goal-reaching via planning**, a different capability
  from imitating an expert distribution.
⇒ The honest statement: **SimWAM refutes "condition the policy on generated futures". It does
not refute "plan against a learned world model".** REF-A v1 should be re-scoped to test the
second, and must stop citing the first as motivation.

### 4b. Their gain is mostly the PRETRAINED PRIOR, not the act of predicting

Tab. 4 — same architecture, same objective, different video backbone:

| backbone | PDMS |
|---|---|
| LTX-Video (light) | 88.7 |
| Wan2.1-1.3B | 90.2 |
| Wan2.2-5B | 90.3 |
| **Cosmos-Predict2.5** (driving-pretrained) | **90.4** |

**1.7 PDMS purely from prior quality**, and the driving-pretrained one wins. Our S-W stage
already performs future-latent prediction — **we have the objective and lack the prior.** That
reframes our frozen-encoder question: the issue may be less "frozen vs fine-tuned" than
"what was the encoder pretrained ON".

### 4c. Scale honesty

Action-expert scaling 0.21B → 1.02B buys 89.9 → 90.3 (**+0.4 for 5×**). Their total is 6B
against our sub-300M constraint. ⇒ **Their result does not transfer for free to our budget**,
and the small marginal return on the *action* expert suggests the video prior — the part we
cannot afford — carries the gain.

---

## 5. Implementation: theirs vs ours

| dimension | SimWAM | TanitAD v6 / REF-A v1 |
|---|---|---|
| future target | **pixel-space VAE latents** of future frames, flow matching | **feature-space latents** (JEPA/DINO-WM style), no pixel decoder |
| prior | pretrained **video generator** (Wan2.2-5B) | DINOv3 ViT-L/16 image encoder (REF-A v1); v6 trains its own trunk |
| coupling | **no shared parameters**; one attention stream + isolated mask | shared trunk feeding 4 brains |
| planner | flow-matching DiT, 10 sampling steps, **no search** | goal-conditioned selection over a fan + optional MPC top-K refinement |
| hierarchy | **none** — flat 8-waypoint output | **operative / tactical / strategic**, our thesis |
| inference cost | ~518 ms (A100) | REF-A v1 measured **21.5 s/tick** with CEM |
| RL | Flow-GRPO on LoRA, PDM reward | none |
| supervision | expert trajectories + future video | 4 metric families + tactical/strategic vocabularies |

⭐ **The architectural idea worth stealing outright is the *decoupling contract*:** two experts
with **no shared weights**, interacting only through a unified attention interface, so either
can be replaced or resized without touching the other's objective or the inference pipeline.
Our 4-brain shares a trunk, which is why our S-S/S-J stage boundaries need
`STAGE_INVALIDATES` bookkeeping at all. Their design makes that class of problem structurally
impossible.

---

## 6. How to extend this to OUR hierarchy — the part that is genuinely ours

SimWAM has **one** action-token group and **one** future horizon. Our hierarchy gives a natural
generalisation they cannot express:

> ⭐ **MULTI-HORIZON ISOLATED SUPERVISION.** Give each layer its own future-prediction target at
> its own horizon and rate — operative 2 s, tactical 6 s, strategic longer — all **isolated
> from inference by the same mask**. Their Tab. 8 already shows horizon coverage is what
> matters; a hierarchy is precisely a machine for covering several horizons at once.

Concretely, the attention stream becomes: `z(o_t)` ← attended by {future_op, future_tac,
future_str, a_op, a_tac, a_str}; every `future_*` group invisible to every `a_*` group, and
the layer-to-layer conditioning we already have (`g_str → P_T`, F-1) untouched. Inference
deletes all three future groups.

**Why this is a real contribution and not a re-skin:** SimWAM's isolated mask makes the
*single* future horizon a pure training signal. Nobody has tested whether **different layers
of a hierarchy benefit from different future horizons** — and our four metric families
(LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC) are already the instrument that could attribute
a gain to a specific layer.

### The cheapest discriminating experiment, pre-registerable now

**E-SIMWAM-1 (no new corpus, no new prior):** on the current v6 S-W checkpoint, compare
* **A** — planner reads only `z(o_t)` (isolated), vs
* **B** — planner reads `z(o_t)` + imagined future latents (our current path),

on identical windows, paired episode-cluster bootstrap, **all four metric families**, T1.
* If **A ≥ B**: SimWAM's finding reproduces in feature space at 300M, our
  `mpc_w_consist=0.0` default becomes a stated result, and imagination is confirmed as a
  training-time signal only.
* If **B > A**: their result is specific to pixel-space video priors and/or 6B scale, and our
  imagination-in-the-loop path is worth keeping.

Both outcomes are publishable and neither needs new data. ⚠️ It must be run at **T1** — at T0
the question is meaningless, since T0 *is* teacher-forced.

---

## 7. What to adopt, what to refuse

| | |
|---|---|
| ✅ **Adopt** | the **isolated-mask discipline** — future prediction as training-time supervision, never an inference dependency |
| ✅ **Adopt** | **RL on the hard subset only**, if we post-train a selector |
| ✅ **Adopt** | the **no-shared-parameters + attention-interface** contract for future arms |
| ✅ **Test** | **Cosmos-Predict2.5** as the prior — driving-pretrained, best in their table, and we already verified our access to Cosmos (ungated, commercial-OK) |
| ⚠️ **Re-scope** | REF-A v1's test-time planning: it must justify itself as *planning against a cost*, not as *imagine-then-act* |
| ⛔ **Refuse** | copying the 6B scale — the action-expert scaling curve (+0.4 PDMS for 5×) says the money is in the prior, which our sub-300M constraint forbids |
| ⛔ **Refuse** | quoting their PhysicalAI ADE against ours — different metric semantics, horizon and split |

---

## 8. Open questions this raises for us

1. **Is our S-W future-latent objective already delivering SimWAM's +3.7?** We have never run
   the action-only ablation. Their Tab. 2 is exactly the control we lack.
2. **Does the isolated mask matter in FEATURE space** (our setting) as it does in pixel space?
   Untested by anyone.
3. **Would a driving-pretrained prior close our closed-loop gap** — the 23.98 → 9.37 → ? axis —
   or is that gap the selector, as `v5f` concluded?

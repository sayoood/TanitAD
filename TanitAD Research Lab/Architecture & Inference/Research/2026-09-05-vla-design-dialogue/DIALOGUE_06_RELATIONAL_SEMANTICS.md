# TanitLang v6 — physical-world semantics in the latent, and the interaction problem

**2026-09-05 · TanitAD_TrainingFlyWheel.** Answers the PI's question: *how do we get semantics
about the physical world into the reasoning latent, and how do we intrinsically model the dynamic
interaction between agents, and between geometry and agents?* Status: DESIGN. Supersedes v5 §2.1
on the SHAPE of `z`.

---

## 0. ⛔ The gap the question exposes in my own design

v5 made `z` a **flat 128-dim vector**. The PI's example — *many pedestrians, a motorcyclist coming
from the rear, bad illumination* — cannot be represented well by a flat vector, and the reason is
precise: **"motorcyclist from the rear" is not a property of the motorcyclist. It is a RELATION
between the motorcyclist and the ego.** A flat latent has to re-derive every relation from scratch
at every step; nothing in its structure says relations exist.

**PUBLISHED, and it is exactly this point** — GraphPilot (`2511.11266`): vision-language models
struggle as driving planners *"due to lack of supervision that explicitly encodes relational
dependencies"*, and the fix is conditioning on **traffic scene graphs**. ⇒ the PI's instinct is
the literature's finding.

---

## 1. ⭐ The correction: `z` becomes RELATIONAL, not flat

```
  v5:   z ∈ R^128                          one vector, relations implicit
  v6:   z = ( {n_i}, {e_ij} )              per-entity latents + TYPED EDGES
        n_i   one latent per entity: ego, each agent token (WP-6), geometry slots
        e_ij  a typed edge latent per interacting pair
```

The recursion is unchanged in spirit — TRM's `z ← f(x,y,z)`, `y ← g(y,z)` — but the think-step is
now **message passing over the graph** rather than an MLP over a vector:

```
  e_ij ← e_ij + Edge(n_i, n_j, rel_ij)        # rel = relative pos, vel, TTC, bearing, closing
  n_i  ← n_i  + Node(n_i, Σ_j e_ij)           # each entity gathers what interacts with it
  y    ← y    + Answer(y, pool(n, e))         # REF-C's existing offset head, now graph-conditioned
```

⭐ **This makes the PI's two interaction types intrinsic rather than hoped-for:**

| interaction | how it is represented |
|---|---|
| **agent ↔ agent** | an edge between two agent nodes — a pedestrian *crossing in front of* the motorcyclist is one edge, and it exists whether or not either is near the ego |
| **geometry ↔ agent** | geometry enters as its own node type (WP-3's waypoint-indexed PV features are already **candidate-conditional**, so the "geometry" node can be *the candidate trajectory itself*) — an agent's edge to that node is literally *"this agent is in the way of this plan"* |
| **ego ↔ agent** | the ego node's edges. "From the rear" is `rel_ij` bearing, not a learned coincidence |

**PUBLISHED precedent:** GraphAD (`2403.19098`) — the **Interaction Scene Graph** models
interactions among *"the ego-vehicle, road agents, and map elements"* in one structure. That is the
three-way relation the PI named.

⚠️ **What refcv5 gives and what it does not.** WP-6 delivers **agent tokens** — the *nodes*. It
does not specify **edges**. ⇒ the graph is the increment TanitLang adds, and it is small: edges are
computed from geometry we already have (relative position, velocity, bearing, TTC).

---

## 2. Injecting physical semantics — the PI's own proposal, and how to do it safely

> *"maybe by using in some stages of the training a strong pretrained vision backbone which
> transforms the image into these semantics"*

⭐ **This is right, and the literature has the method.** CMCMD distils **DINOv3's open-world
semantic priors** into a student network via manifold-aware topological alignment; VLM-AD distils
VLM-generated *structured behaviour labels* into a planner's latent.

**How it fits our constraints:**

| stage | what runs | why it is admissible |
|---|---|---|
| **training only** | a strong frozen backbone (DINOv3-class) labels entities: pedestrian / cyclist / motorcycle / vehicle, plus illumination and occlusion state | ⛔ **teacher at train time, absent at inference** — the same discipline refcv5 WP-6 already enforces for `obstacle.offline`: *"an arm that reads the join at inference is REFUSED, not fixed"* |
| **inference** | the trunk predicts those semantics itself; nodes carry the predicted type | vision-only rule holds; latency unaffected — no second backbone on the tact |

⭐ **Bad illumination is a first-class node/global feature**, not an accident of the pixels: the
teacher can label it, so the student can *know it is uncertain* rather than merely being wrong.
That is the honest route to the PI's "the system says when it is uncertain".

⚠️ **What this does NOT solve.** A distilled label is a *category*. It does not tell you a
motorcyclist closes faster than a cyclist — that is **dynamics**, and it has to come from the edge
features (`closing rate`, `TTC`), not from the semantic class. **Semantics and dynamics are two
separate injections and conflating them is how a model learns "motorcycle ⇒ danger" as a prior
instead of computing it.**

---

## 3. ⭐ The complexity paradox the PI named, and what the graph gives us

> *"a complex situation ... which requires very fast reasoning"*

**Complex situations need MORE reasoning and allow LESS time.** A flat latent has no way to know it
is in one. **A graph does: edge count, and the number of edges with short TTC.**

| signal | source | use |
|---|---|---|
| n high-interaction edges | the graph itself | difficulty estimate, available *before* reasoning |
| top-2 candidate score margin | REF-C's scorer | is the decision contested? |
| illumination / occlusion state | the distilled semantics | is perception itself uncertain? |

⇒ These three give a **situation-derived compute budget**: R = 1 when the graph is sparse and the
margin wide; R = 2–3 when dense and contested. ⚠️ Still my proposal, not a published result —
⛔ TRM has **no** halting mechanism and HRM's ACT was among the ablated components. It needs a
fixed-R control or it merely measures spending more compute.

⚠️ **And the honest tension:** the hardest situations are exactly where we can least afford extra
steps. If the measurement says complex scenes need R=3 and R=3 does not fit the tact, the answer is
**a better graph, not more recursion** — and that is a result worth having either way.

---

## 4. What this changes in the plan

| | v5 | **v6** |
|---|---|---|
| `z` | flat 128-d | **per-entity nodes + typed edges** |
| think-step | MLP | **message passing** |
| agent↔agent | implicit | **an edge** |
| geometry↔agent | absent | **an edge to the candidate-trajectory node** |
| semantics | from the trunk alone | **+ distilled from a strong backbone, train-time only** |
| difficulty | top-2 margin only | **+ edge density + illumination state** |
| params | ~6 M | **~8 M** (edges are cheap; K=5 candidates × ~10 agents) |

**New work packages, both cheap and both gating:**

* **WP-F — do edges carry anything the nodes do not?** Fit best-candidate prediction from node
  features alone vs node+edge features. ⛔ Controls: shuffled edges (must collapse), constant-edge
  floor, n and d printed. *If edges add nothing on our corpus, the graph is decoration and v6
  reverts to v5.*
* **WP-G — is the distilled semantic label recoverable by the trunk?** Teacher labels vs student
  prediction, held out. ⚠️ If the trunk cannot predict "motorcycle vs bicycle" from our 256×640
  cylindrical frames, the distillation cannot help at inference and we should know before building
  the pipeline.

⛔ Unbanked, required before any number is quoted: `2511.11266` (GraphPilot), `2403.19098`
(GraphAD), the CMCMD distillation work, VLM-AD.

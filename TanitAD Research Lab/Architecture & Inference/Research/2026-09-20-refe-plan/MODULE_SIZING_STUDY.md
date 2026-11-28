# REFe module sizing — the five numbers the paper never gave

**Written 2026-09-20.** Scope: for each REFe module that DriveZero leaves unspecified, what is the
right size *for our context* — one front camera, DINOv3 ViT-L frozen with rank-32 Q/V LoRA, a
rehearsal bank of order 10³ tuples today and navtrain (103,288 frames) later, one A40.

**Evidence classes used throughout:** `MEASURED` (ours, with the artifact) · `PUBLISHED` (cited
**and banked**, with table/section) · `DERIVED` (arithmetic over published numbers, ours) ·
`ESTIMATED` · `HYPOTHESIS` · `UNSETTLED` (the literature does not decide it — an experiment is
pre-registered in §5).

⛔ **No model code was changed. This is a study. Every number below is a recommendation.**

---

## 0 · Headline — read this first

### 0.1 Three findings that change what should be built

**⭐ FINDING 1 — the paper DOES specify `pos3d`, by citation, and our learned table is not it.**
DriveZero p.8: *"the output visual tokens are enriched with 3D position embeddings **[50]**."*
Reference **[50] is PETR** (Liu et al., ECCV 2022). PETR's 3D PE is a **1×1-conv MLP over
analytically computed camera-frustum 3D coordinates** — not a table, and not indexed by token
position. `PUBLISHED`, and established twice independently: I read the bibliography entry out of
the banked PDF, and a separate stream resolved it through the Semantic Scholar citation-context
API, which returns that exact sentence.
⇒ **Our `pos3d` (a learned `1920 × 1024` table, 1,966,080 params) is a 2-D positional table
wearing a 3-D name.** §2.3.

**⛔ FINDING 2 — a GPU-free probe says that table is INFORMATION-WRONG, not merely
over-parameterised.** `MEASURED` over **all 1,349 nuPlan log DBs on D:** (`CAM_F0`, 1,349/1,349
read, 0 failures): **2 distinct intrinsics** (fx 1545.000/cx 960.000/cy 560.000 on 20 vehicles;
fx 1528.712/cx 970.460/cy 578.414 on 1) and **22 distinct extrinsic translations / 24 distinct
rotations**. A per-token table stores **one** vector per position for a corpus whose camera pose
differs per vehicle; it cannot condition on either. PETR's PE can, because it is a function of
K and the extrinsics. Control: the same probe on `CAM_L0` (a channel we do not use) returns the
same structure, so "variation found" is not a query artifact. §2.3.

**⛔ FINDING 3 — the paper's own published parameter count REFUTES a 12.6 M compression module.**
`DERIVED` from Table A12 (18.58 M trainable, FOUR cameras). Rebuilding *their* configuration with
a 1024-wide cross-attention compressor at MLP ratio 4 costs **24,667,714** trainable **with
`pos3d` set to zero** — **+6,087,714 over their published budget**. Under the most generous
possible assumption, LoRA = 0, it is still **21,521,986 = +2,941,986 over**. A full sweep puts the
current reading at **32,532,034 = +75.1 %** of a number it was not fitted to, while a
DrivoR-faithful reading lands at **17,531,458 = −5.6 %**. §3.2.

### 0.2 Recommendations

| module | current | **recommended** | evidence | changes with data scale? | changes with camera count? |
|---|---|---|---|---|---|
| `reg_compress` **MLP ratio** | 4× (12,598,272) | **1× (6,303,744)** | `PUBLISHED` Perceiver / Perceiver IO: latent blocks *"preserve the dimensionality of their inputs"* | **no** | **no** |
| `reg_compress` **blocks** | 1 | **1 — keep** | `PUBLISHED` Perceiver Table 6: front-stacked cross-attends degrade **76.7 → 76.7 → 75.9 → 73.7**; Perceiver IO deleted them | no | no |
| `reg_compress` **heads** | 8 (head_dim 128) | **16 (head_dim 64)** — costs **0 params** | `PUBLISHED` head_dim-64 convention (Scaling ViT Table 2, Three-Things Table 1); DINOv3 ViT-L is 16@1024 | no | no |
| `reg_compress` **mechanism** | bolt-on cross-attention | **`UNSETTLED` — E-SIZE-1**; ship the bolt-on at MLP 1 meanwhile | DrivoR's method is in-backbone (**90.0**); the bolt-on is its ablation row (e) (**89.3**) | no | no |
| `score_dec` **depth** | 4 | **4 — keep, now with a citation** | `PUBLISHED` DrivoR §3.4 + impl.: *"The decoders are 4-layer transformers with an inner dimension of 256"*, and the scorer *"uses an architecture mirroring that of trajectory generation"* | no | no |
| `pos3d` **form** | learned 1920×1024 table (1,966,080) | **PETR-style MLP over analytic 3D coords, ND=2, hidden C/2 → 528,896** | `PUBLISHED` PETR §3.2–3.3 + Table 3; `PUBLISHED` 3DPPE Table 9 licenses ND=2; `MEASURED` calibration variance | no | **YES if a table — NO as an MLP** |
| `ego_enc` **size** | 69,120 (12→256→256) | **keep 256/2-layer; add sinusoidal goal encoding → 85,504** | `PUBLISHED` DriveZero p.4 does exactly this in their own teacher; `PUBLISHED` Fourier Features | no | no |
| `dec_heads` (256-d) | 8 | **8 — keep** (`UNSETTLED`, E-SIZE-4, costs 0 params) | DETR uses 8@256; their own teacher uses 4@256; nobody has ablated it for a planner | no | no |
| `n_registers` | 16 | **16 — keep** (`UNSETTLED` hedge at 32, E-SIZE-5) | `PUBLISHED` DrivoR Table 4c: 5/8/16/32 per camera = 89.7/89.7/**90.0**/89.8 | no | **YES (per-camera budget)** |
| *(conformance, changes the arithmetic)* `score_q_mlp` | absent | **add (81,408)** | `PUBLISHED` DriveZero p.8 *"encodes each candidate trajectory as a score query"*; DrivoR §3.4 | no | no |
| *(conformance)* `score_head` | 1 Linear (1,542) | **six per-component MLPs (396,294)** | `PUBLISHED` DrivoR §3.4 *"a dedicated MLP for each score"* | no | no |
| `registers` init | `trunc_normal_(0.02)` | **`N(0, 1e-6)`** — costs 0 params | `PUBLISHED` DrivoR impl.: registers and initial trajectory tokens init `N(0, 1e-6)` | no | no |

**Net: 26,584,642 → 19,345,474 trainable (8.06 % → 6.00 %); 329,664,066 → 322,424,898 total.**
That is **+4.1 %** against DriveZero's published 18.58 M — for ONE camera against their four. §3.3.

### 0.3 ⛔ ESCALATION — three items for the PI

1. **`pos3d` is the one item I would treat as a defect rather than a preference.** Findings 1+2
   are independent of any taste argument: the paper names a different mechanism, and our corpus
   violates the assumption the table makes. It is also **the same shape (1,966,080) as the bogus
   trunk table this package already removed on 2026-09-20** for the same reason. Recommend it be
   replaced, not merely ablated.
2. **The PI's compress-at-1024 decision is NOT reopened here, and the DrivoR route sits inside
   it.** The PI fixed *where* (at backbone width, project afterwards). DrivoR's actual method —
   registers appended to the ViT input, harvested at the last layer, then projected to 256 — **is**
   compression at 1024 followed by a projection. So E-SIZE-1 chooses the *implementation* of the
   PI's own branch (i), not a return to the rejected project-then-compress branch (b).
3. **`POD_HANDOFF.md`'s arm table is internally inconsistent by exactly 9.0×.** `DERIVED`: on the
   basis that reproduces their published 608 GPU-hours to 0.1 %, arm A is **6.3** A40-h (claimed
   0.7) and arm B is **25.3** (claimed 2.8); arm D checks out at 608.5 vs 608. The same slip on
   both rows. This matters because "the first pod day costs under three GPU-hours" is the sentence
   the A40 request rests on. §5.1.

### 0.4 ⭐ The uncomfortable answer to "what is the optimal size at 10³"

The brief asks me to size five head modules for a 10³-tuple regime. **The literature says the head
modules are not the lever at 10³, and I would be inventing a recommendation if I pretended
otherwise.** `PUBLISHED`, arXiv 2410.18647 (banked), at ~800 demonstrations — the only measurement
I found *inside* our data regime that isolates these variables:

* **Table 2c**, scaling the *action decoder*: small **0.88** · base **0.90** · large **0.83**.
  Verbatim: *"scaling the action diffusion U-Net does not yield performance improvements."*
* **Table 2a**, scaling the *encoder adaptation* on the same bank: DINOv2 ViT-L fully fine-tuned
  **0.90** · LoRA rank-8 **0.72** · **frozen 0.00**.

⇒ At 10³, decoder capacity was flat-to-harmful and encoder adaptation was the whole result. Our
LoRA rank is **32** and our head is **23.4 M**. Per CLAUDE.md's *"prefer the lever with the largest
measured effect"*, **E-SIZE-7 (a LoRA rank ladder with a frozen floor) outranks every head-size
experiment in §5**, and I have ordered §5 accordingly rather than by the brief's module list.

⚠️ Scope: 2410.18647 is closed-loop robot manipulation with a Diffusion Policy, not NAVSIM
planning. It is the only 10³-regime isolation I found; it is directional, not decisive. The
counterweight is `PUBLISHED` VPT (banked): at **VTAB-1k = exactly 1,000 training examples**, a
**0.53 %-trainable** prompt beat *full* fine-tuning on **20 of 24 tasks**. Our setting sits between
those two, which is precisely why E-SIZE-7 is an experiment and not an assertion.

---

## 1 · What the source actually fixes, and what it leaves open

`PUBLISHED` — DriveZero, arXiv **2609.06055**, banked as
`Library/papers/2609.06055_DriveZero-End-to-End-Driving-Beyond-Human-Demonstrations.pdf`.

⚠️ **The banked copy is byte-different from the local `C:/Users/Admin/dz/DriveZero/drivezero_report.pdf`**
(sha256 `82a00048…` vs `e1ca1fd7…`; 9,392,899 B vs 9,396,758 B). I verified **by content, not by
name**: every sentence and every Table A12 cell quoted here appears in the banked copy at the same
page numbers. `MEASURED`.

**Table A12 (p.30), in full** — the complete model specification the paper gives:

> Sensors CAM_F0/B0/L0/R0 · Input resolution 960×512 · Prediction horizon 20 steps at 5 Hz ·
> Trajectory proposals 64 · Planning representation 256 dimensions · Proposal decoder 4 layers ·
> Register tokens 16 per camera · Visual backbone DriveVFM ViT-L (frozen) · Backbone adaptation
> Q/V LoRA rank 32 · **Trainable parameters 18.58 M** · **Full parameters 338.46 M**

**That is the whole table.** No scoring-decoder depth. No head count. No MLP ratio. No 3D-PE form.
No ego-encoder size. The main text (p.8, p.10) adds nothing further on any of them.

⚠️ **The paper's only head-count figure is Table 1, "Token width / attention heads — 256 / 4".
That is the DriveRL *teacher*, a vector-input nuPlan policy — a different model.** Quoting it as
the camera planner's head count would be a scope error. It is used below only as a weak
same-authors prior, and labelled as such.

### 1.1 The architecture sentence, and the ambiguity inside it

> *"DriveVFM, fine-tuned with LoRA [27], encodes each camera image, and the output visual tokens
> are enriched with 3D position embeddings [50]. Following DrivoR [34], learnable registers [16]
> then compress these tokens into a small set of scene tokens."* — p.8

Resolved references, read from the banked bibliography (`MEASURED`, my own extraction):

| ref | paper | arXiv | banked |
|---|---|---|---|
| [50] | PETR: Position Embedding Transformation for Multi-View 3D Object Detection | 2203.05625 | ✅ |
| [34] | Kirby et al., *Driving on Registers* (DrivoR), CVPR 2026 | 2601.05083 | ✅ |
| [16] | Darcet et al., *Vision Transformers Need Registers*, ICLR 2024 | 2309.16588 | ✅ |

⛔ **The sentence and its citation disagree, and this is the crux of §2.1.** The *word order* says
the compression happens **after** the trunk, on the enriched output tokens. The *citation* points
at DrivoR, whose compression happens **inside** the trunk. Three constraints, and no reading
satisfies all three:

| reading | matches the word order? | faithful to DrivoR? | fits 18.58 M? |
|---|---|---|---|
| in-backbone registers (DrivoR's method) | ✗ | ✅ | ✅ (≈0 params) |
| post-trunk cross-attention @1024, MLP 4 (**ours today**) | ✅ | ✗ (their ablation row e) | ⛔ **+75.1 %** |
| post-trunk cross-attention @1024, MLP 1 | ✅ | ✗ | ✅ (**+4.1 %**) |
| post-trunk cross-attention @256 | ✅ | ✗ | ✅ — but the PI ruled this branch out |

**I rank the constraints by hardness: the published parameter count (a number) > DrivoR's measured
ablation (89.3 vs 90.0) > the sentence order (prose).** That ordering is what produces the §0.2
recommendation and the E-SIZE-1 pre-registration.

---

## 2 · One section per module

### 2.1 `reg_compress` — 12,598,272 → 6,303,744, and a mechanism question worth one experiment

**What DrivoR actually does.** `PUBLISHED`, read by me from the banked PDF, §3.2 p.3:

> *"For each camera, we concatenate R camera registers of size D_ViT, along with pre-existing
> registers, classification token, and patch tokens. All registers and tokens are fed to the ViT.
> We then retrieve the R camera tokens at the final layer of the ViT."*

And, explicitly rejecting our design:

> *"This compression … is close in spirit to Perceiver approaches, with the noticeable difference
> that these approaches use cross-attentions for compression. Setting up such a mechanism in the
> encoder would require changes in the ViT architecture."*

**DrivoR Table 4b** (navval PDMS, ViT-S, all compressed rows at 16 cam tokens / 64 scene tokens):

| row | compression | backbone | optim M | PDMS |
|---|---|---|---|---|
| (b) | none (4k/16k tokens) | LoRA | 18.8 | **90.2** |
| (c) | pooling | LoRA | 18.2 | 89.7 |
| (e) | **decoder (cross-attention) — our design** | LoRA | 19.9 | **89.3** |
| (h) | **registers (their method)** | LoRA | 18.8 | **90.0** |
| (g) | registers | **frozen** | 18.2 | **84.4** |
| (d) | decoder (cross-attention) | **frozen** | 19.3 | 86.9 |

Three readings, in order of load:

1. **Our mechanism is the one that lost.** 89.3 vs 90.0, at matched token budget and *"roughly the
   same number of parameters"* in their ViT-S setting. ⚠️ Single point estimates on navval — **no
   CI, no seeds, anywhere in DrivoR's Tables 4–7**. A 0.7-PDMS gap with no replicate is suggestive,
   not decisive; that is exactly why E-SIZE-1 exists rather than a flat "switch".
2. **The frozen rows do not apply to us.** Registers+frozen is catastrophic (84.4) and
   cross-attention+frozen is better (86.9) — but **DriveZero and REFe both use rank-32 Q/V LoRA**,
   which is DrivoR's LoRA rows, where registers win. Reading (g) as an argument against the
   in-backbone route for REFe would be a scope error.
3. **Their cross-attention block scales to exactly our number.** `DERIVED`: DrivoR's row (e)
   overhead is ~1.1 M at D_ViT = 384; a cross-attention block with MLP 4 at 384 is 1,775,232, and
   ×(1024²/384²) = 12.6 M. **Our 12,598,272 is precisely "DrivoR's losing ablation arm, scaled to
   ViT-L width."** Nothing was mis-sized; the mechanism was inherited from the wrong row.

**The MLP ratio — the one knob the PI's decision leaves fully open, and it is a genuine fork.**

| anchor | ratio | applies to |
|---|---|---|
| `PUBLISHED` DrivoR impl. — *"The feed-forward network has a dilation factor of 4"* | **4** | its **decoders at 256** — not a compressor at ViT width |
| `PUBLISHED` Flamingo Table 4 — *"The hidden size of each feed-forward MLP is 4D"* | **4** | Perceiver Resampler (64 latents, 6 layers, D 1536, 16 heads) |
| `PUBLISHED` Scaling ViT Table 2 | **4** | every ViT backbone up to L/16 |
| `PUBLISHED` **Perceiver** §Appendix C — *"All linear layers … preserve the dimensionality of their inputs"*; *"The dense subblock … doesn't use a bottleneck"* | **1** | **a latent cross-attend bottleneck — our module's exact role** |
| `PUBLISHED` **Perceiver IO** language config: latent D 1280, *"FFW hidden dimension for latents 1280"* | **1** | same |

⇒ **I recommend ratio 1.** The ratio-4 anchors are all *decoder* or *backbone* blocks; the only
anchors for a **latent compression bottleneck** — which is precisely what `reg_compress` is — use
**1**. It saves **6,294,528** params (the single largest cut available) and is the change that
reconciles the PI's compress-at-1024 decision with the paper's own published budget.
⚠️ Flamingo is the counter-anchor: its resampler is a latent bottleneck *and* uses 4. But
Flamingo's Table 10(i) also measures that a **larger** resampler is **1.7 points worse** and
*"lead to unstable training"* — so even the ratio-4 anchor points down, not up, at the margin.

**Blocks: one.** `PUBLISHED` **Perceiver Table 6** — cross-attends stacked at the front:
**1 → 76.7 · 2 → 76.7 · 4 → 75.9 · 8 → 73.7.** Monotonically harmful. **Perceiver IO removed them
outright**: *"we found these to lead to relatively small performance improvements but to
significantly slow down training."* ⚠️ The one configuration that beat 1 (8 *interleaved*, 78.0)
requires re-reading the input from deep in the downstream stack — a different architecture.
⇒ **If depth is ever added here, add self-attention among the 16 registers, not a second
cross-attend** — Perceiver Table 5 is emphatic that cross-attention alone collapses (39.4 / 45.3).

**Heads: 16, not 8 — and it costs zero parameters.** `nn.MultiheadAttention` is `4d²+4d`
independent of `num_heads` (`MEASURED`: my analytic model carries no heads term and reproduced the
real instantiation exactly). At width 1024, 8 heads gives head_dim **128**; the universal ViT
convention is head_dim **64** (`PUBLISHED`: Scaling ViT Table 2 and Three-Things Table 1, every
model up to L/16), and **DINOv3 ViT-L itself is 16 heads at 1024** — the feature space these K/V
come from. Scaling ViT states the rule for exactly this module type (its MAP attention-pooling
head): *"We set the number of heads in MAP to be equal to the number of attention heads in the
rest of the model."*
⚠️ `PUBLISHED` *Are Sixteen Heads Really Better than One?* cuts the same way: encoder–decoder
(cross) attention is the **least** prunable layer type (−13.5 BLEU at one head), so erring low here
is the risky direction.

**Register count: keep 16.** `PUBLISHED` DrivoR Table 4c (per camera): 5 → 89.7 · 8 → 89.7 ·
**16 → 90.0** · 32 → 89.8. ⚠️ **The flatness is the finding**: 0.3 PDMS across a 6× range, single
point estimates. Two readings of "16 at one camera", and they agree:
* **per-camera reading** — 16 is their measured optimum. ✅
* **total-token reading** — our 16 total sits below their lowest tested total (20). ⚠️ But the
  compression ratio corrects for it: `DERIVED`, DrivoR compresses **3,936 → 16 = 246× per camera**;
  we compress **1,920 → 16 = 120×**. Our bottleneck is **2× gentler than theirs**.
⚠️ **The asymmetry argues for a cheap hedge.** `PUBLISHED` Perceiver IO Table 15 at matched FLOPs:
too **few** latents costs **−5.11** GLUE, too many costs **−0.03**. Going to 32 costs 16,384 params
and 0.23 GMAC. E-SIZE-5 tests it for ~5 A40-hours.
⚠️ **The DrivoR rationale for per-camera scaling is void at N=1.** Their reason is camera-awareness
(*"the model can differentiate if a given scene token is extracted from, e.g., the front, left or
right camera"*). With one camera there is nothing to differentiate. 16 survives on the
compression-ratio argument, not on theirs.

**Initialisation: `N(0, 1e-6)`, not `trunc_normal_(std=0.02)`.** `PUBLISHED` DrivoR impl.:
*"The camera registers as well as the initial trajectory tokens are randomly initialized with
normal distribution N(0, 10⁻⁶)."* Ours is ~4 orders of magnitude larger in scale. Free to change.
⚠️ **Keep DINOv3's own 4 `reg_token`s.** DrivoR concatenates its registers *"along with
pre-existing registers"* — it keeps them too. DrivoR Table 4c's DINOv2+random row reads 89.8 vs
90.0 for discarding, a 0.2 difference; keeping them also preserves this package's strict
tensor-consumption control.

### 2.2 `score_dec` — 4 layers is right, and now it has a citation instead of a guess

REFe's `score_dec_depth` field documents that the paper gives no depth and that mirroring
`dec_depth` is *"a CHOICE, not a quotation."* **That is correct about DriveZero and it is no longer
the whole story: the depth IS published, in the paper DriveZero follows.**

`PUBLISHED`, DrivoR, three statements that compose (read by me from the banked PDF):
* §3.3 p.3: *"**All decoders** use the architecture depicted in Fig. 2b … a stack of k transformer
  blocks, each made of a self-attention layer, followed by a cross-attention to the scene tokens,
  and a feed-forward network (FFN), all with residual connections."*
* §3.4 p.4: *"The scoring decoder … uses an architecture **mirroring that of trajectory
  generation**."*
* impl., p.4–5: *"**The decoders are 4-layer transformers with an inner dimension of 256.** The
  feed-forward network has a dilation factor of 4."*

⇒ **DrivoR's scoring decoder is 4 layers at 256 with FFN ratio 4** — identical to its trajectory
decoder. REFe's `score_dec = 4 × CrossBlock(256)` and `CrossBlock`'s self-attn → cross-attn → MLP-4
structure both match. **Recommend: keep 4, and record the citation so the next reader does not
re-open it.**

⚠️ **I am overruling a contrary conclusion from one of my own research streams.** That stream
surveyed DETR/VADv2/Hydra-MDP/GoalFlow/UniAD and concluded *"no published planner mirrors a deep
trajectory decoder with an equally deep scorer … mirroring 4 layers has no published precedent."*
**DrivoR is that precedent, and it is the paper DriveZero cites for this very branch.** Classic
*absence found at one location is not absence* — the survey did not include the lineage paper.

The counter-evidence is real but weaker, and I record it rather than bury it:

| source | number | why it does not overturn DrivoR |
|---|---|---|
| `PUBLISHED` DiffusionDrive Table 5, NAVSIM | cascade stages 1→**87.4**, 2→**88.1**, 4→**88.2** | depth saturates; 2→4 is +0.1 PDMS for +5 M. Different decoder family (diffusion cascade) |
| `PUBLISHED` UniAD | Planner **3** layers @256 vs 6 for TrackFormer/MapFormer | shallower, but a different scorer-free design |
| `PUBLISHED` DETR | classification head is **a single linear layer** | classification ≠ a six-component PDM regressor |
| `PUBLISHED` Transformer Table 3 | N=6 best (25.8); **N=8 is worse (25.5)** | depth is saturated by 6, consistent with 4 being safe |

⇒ 4 is inside every saturation point in the table and is the lineage's own value. **`score_dec`'s
depth is settled; do not spend GPU on it.**

⛔ **And it could not be tested today anyway — name the blocker.** The scorer's supervision on this
rig is defective in three independently MEASURED ways recorded in this package: coverage **22.7 %**
of frames; `comfort` varies across candidates in **0.0 %** of frames (`REFE_MODEL.md` §8.2 — it
cannot move the argmax, so a sixth of the head does no work); and components 3 and 6 are
**mis-mapped** (`PAPER_CONFORMANCE_REVIEW.md` D4a/D4b). **A depth ladder on a scorer that cannot
rank measures nothing.** Unblocking D3a/D4a/D4b comes before any `score_dec` experiment.

**Two structural pieces the scoring branch is missing** (conformance, but they change §3's
arithmetic and they are cheap):
* `PUBLISHED` DriveZero p.8: the scorer *"**encodes each candidate trajectory** as a score query"*;
  DrivoR §3.4: *"Each decoded trajectory is turned into a D_score-dimensional query using an MLP …
  Embedding the decoded trajectories into a new feature space **rather than reusing the trajectory
  decoder's output tokens is key in our architecture**."* REFe does `s = q.detach()` — it reuses
  the output tokens, the thing DrivoR calls out by name. **+81,408 params** to fix.
* `PUBLISHED` DrivoR §3.4: *"we predict the six score components using **a dedicated MLP for each
  score**."* REFe uses one `Linear(256, 6)`. **+394,752 params** to fix.

⚠️ **Context asymmetry, flagged with its cost.** DriveZero says the scorer *"attends to the visual
tokens"*; DrivoR says scene tokens. REFe follows DriveZero. `DERIVED`: that costs **1.25 GMAC** in
`score_dec` plus **0.50 GMAC** for `scene_proj(visual)` = **1.75 GMAC, 27 % of the whole head
stack** — for a wording difference. It is the right call on DriveZero's explicit text; it should be
recorded as a cost, and it is the cheapest thing to revisit if head compute ever binds (it does not
today — see §3.4).

### 2.3 `pos3d` — ⛔ the one module I would call wrong, not merely unjustified

**What the paper specifies.** By citation, and the citation is the whole specification: `[50]` =
**PETR**. `PUBLISHED`, PETR §3.2–3.3 — the mechanism in full:

1. discretise the camera frustum into a `(W_F, H_F, D)` meshgrid; a frustum point is
   `p = (u·d, v·d, d, 1)ᵀ`;
2. inverse-project with the camera matrix: `p³ᵈ = K⁻¹ p` (Eq. 1) — **this is where the intrinsics
   and extrinsics enter**;
3. normalise into the RoI (Eq. 2);
4. §3.3 verbatim: *"the P³ᵈ is first feed into a multi-layer perception (MLP) network and
   transformed to the 3D position embedding."*

The official implementation (`megvii-research/PETR`, `petr_head.py`) gives the shape the paper
prints only as a row label: `Conv2d(3·D_depth, 4·C, 1) → ReLU → Conv2d(4·C, C, 1)`, with
`depth_num = 64`, i.e. **192 → 4C → C**.

**PETRv2 states the property that matters here**, verbatim: *"3D PE in PETR is generated based on
the fixed meshgrid points in camera frustum space. All images from one camera view share the 3D PE,
making 3D PE data-independent."* ⇒ at inference it *is* a fixed tensor per camera — **functionally
what our table is — but it is generated by a function of 3D coordinates, so it re-derives itself at
any resolution and any calibration.** That is the entire difference, and it is the whole argument.

**PETR Table 3 — the load-bearing ablation** (nuScenes):

| 2D PE | MV | 3D PE | NDS | mAP |
|---|---|---|---|---|
| ✓ | | | 0.208 | **0.069** |
| ✓ | ✓ | | 0.224 | 0.089 |
| | | ✓ | 0.356 | **0.305** |
| ✓ | ✓ | ✓ | 0.359 | 0.309 |

Two readings: **a positional-only PE is catastrophic for 3D (0.069 vs 0.305 mAP, 4.4×)**, and
**3D PE alone is within 0.003 NDS of the full stack** — the multi-view term is not what makes it
work, which is why the result survives our single-camera setting.

**PETR Table 5(a)** — the PE network itself: `None` (raw normalised coords) 0.311/0.256 ·
**`1×1 ReLU 1×1` 0.359/0.309** · `3×3 ReLU 3×3` **0.017/0.000**.
⚠️ **The kernel must stay strictly pointwise; a 3×3 destroys the model entirely.**

**Does anyone use a learned per-token table at backbone width for 3D position?** `PUBLISHED`,
**no** — a clean negative across PETR, PETRv2, 3DPPE, CAPE, DETR3D, BEVFormer, LVSM and DrivoR.
The only learned tables in this literature are object-query embeddings and BEVFormer's BEV-grid
queries, both in **output** space. **Nobody indexes 3D position by visual-token index.**

**⛔ And for our corpus the table is not merely unconventional — it is information-wrong.**
`MEASURED` 2026-09-20, GPU-free, artifact `raw/refe_cam_calibration_variance.txt`:

```
sweeping 1349 log DBs for channel CAM_F0
  DBs read OK: 1349   unreadable/no CAM_F0: 0
  image size(s): {(1920, 1080): 1349}
DISTINCT CAM_F0 INTRINSICS : 2
   n=1304  vehicles=20  fx=1545.000 fy=1545.000 cx=960.000 cy=560.000
   n=45    vehicles=1   fx=1528.712 fy=1526.081 cx=970.460 cy=578.414
DISTINCT CAM_F0 EXTRINSIC TRANSLATIONS : 22
DISTINCT CAM_F0 EXTRINSIC ROTATIONS    : 24
SPREAD BETWEEN THE TWO INTRINSIC RIGS
   fx 1545.000 vs 1528.712   delta 16.288 px (1.05 %)
   cx 960.000 vs 970.460     delta 10.460 px
   cy 560.000 vs 578.414     delta 18.414 px
```

A learned per-token table stores **one** vector per token position. Our corpus has **two focal
geometries and 22–24 distinct camera mounting poses**. The table cannot condition on either;
PETR's PE is a function of exactly those quantities. **Control:** the identical probe on `CAM_L0`
— a channel REFe does not use — returns the same structure (2 intrinsics, 12 translations in the
24-DB sample), so "variation found" is not an artifact of the query.

**Resolution-lock, published and quantified.** `PUBLISHED` **CPVT** Table 2, DeiT-tiny tested
without fine-tuning at 224/384/448/512: **learnable PE 72.2 / 71.2 / 68.8 / 65.9 — a −6.3 pp
collapse**; CPVT's conditional encoding *improves* (72.4 / 73.2 / 71.8 / 70.3). `PUBLISHED`
**RoPE-ViT**, ViT-B: learned APE **83.4 → 80.5** at 512 while RoPE holds 82.9. `PUBLISHED` **DeiT**
§6: *"bilinear interpolation of a vector from its neighbors reduces its ℓ2-norm … we observe a
significant drop in accuracy if we employ use directly without any form of fine-tuning."*
`PUBLISHED` **NaViT** abandoned learned 2-D absolute PEs because *"every combination of (x,y)
coordinates must be seen during training."*

**⭐ And a third argument specific to REFe: the model already has a positional encoding.** DINOv3
computes **axial RoPE inside every attention block** (`model.py:build_axial_rope`, reconstructed
in this package against the released reference). `pos3d` is a **second** positional signal added at
the trunk output that carries **no 3-D information at all** — it is redundant with RoPE for 2-D
position and empty of the thing its name claims.

⚠️ **And it is the same shape as a table this package already removed.** `REFE_MODEL.md` and
`model.py:VitS16` record that a `trunc_normal_` table of exactly **1,966,080 = 1920 × 1024** was
deleted from the trunk on 2026-09-20 because it had no checkpoint counterpart. **`pos3d` is a
table of the identical shape, created nine lines later.** The two are not the same defect — `pos3d`
is added *after* the frozen trunk, so it does not corrupt the pretrained function — but the
resolution-lock objection applies verbatim to both.

**Sizing the replacement — and ⚠️ do NOT expect a parameter saving by default.** `DERIVED` at our
width C = 1024:

| design | params | vs the 1,966,080 table |
|---|---|---|
| learned table (current) | 1,966,080 | 1.00× |
| PETR's literal recipe, ND=64, hidden 4C | **4,985,856** | **2.54× — more expensive** |
| PETR recipe, ND=64, hidden C | 1,247,232 | 0.63× |
| PETR recipe, **ND=2, hidden C/2** | **528,896** | **0.27×** |
| 3DPPE recipe (sine front-end, MLP 1536→1024→1024) | 2,623,488 | 1.33× |

**Recommended: ND=2, hidden C/2 → 528,896.** The ND=2 licence is `PUBLISHED`, **3DPPE Table 9**:
ND=64 LID **0.338** NDS / 0.275 mAP · ND=32 UD 0.342 / 0.274 · **ND=2 0.345 / 0.276**, with the
paper's own reading *"the immune performances declare that N_D also does not largely affect the
results."*
⚠️ This is a **single-paper** licence on a nuScenes detection task. If the PI prefers the safer
route, PETR's literal 4C recipe (4,985,856) is the fully cited option and still lands the whole
model at 17,006,146 trainable (§3.2, row R1d) — it simply spends 4.5 M where 0.5 M would do.

**Implementation prerequisite, stated so it is not discovered later.** A PETR-style PE needs the
camera intrinsics and `sensor2lidar` extrinsics per frame. `MEASURED`: **they exist in the nuPlan
`camera` table** (columns `intrinsic, translation, rotation, distortion, width, height`, read
above) and `REFE_PLAN.md` records the same fields on NAVSIM's `Cameras.cam_f0`. **They are not
currently carried into the target bank** — `grep` over `refe/*.py` and `code/*.py` finds zero uses
of `intrinsic` / `sensor2lidar`. That is a bank-builder work item, not a blocker.

⚠️ **One-camera caveat, stated honestly:** I found **no published head-to-head of 3D PE vs 2D PE in
a single-camera setting**, across PETR/PETRv2/3DPPE/CAPE/DETR3D/BEVFormer/LVSM. PETR Table 3 row 3
(3D PE alone, no multi-view term, within 0.003 NDS of the full stack) is the strongest available
evidence that the result is not a cross-view effect — but it is an implication, not a measurement.
⭐ Working in our favour: CAPE's critique of PETR — *"the variation of camera extrinsics"* makes a
global 3D PE hard to learn — **does not bind on a single rig**, where CAPE's camera-local fix and
PETR's global coordinates coincide.

### 2.4 `ego_enc` — the size is fine; the input encoding is not

**69,120 params = 0.26 % of trainable. It is not a lever and no experiment should be spent on its
width or depth.** `UNSETTLED` in both sources: DriveZero p.8 says only *"The ego kinematics and the
command are embedded into one ego token"*; DrivoR §3.3 says only *"The ego status inputs …
are encoded and added to the trajectory queries."* Neither gives a size. **Recommend: keep
`Linear(·, 256) → GELU → Linear(256, 256)`.** It matches the planning width, which is the only
constraint either paper implies.

**⭐ But the FORM is specified, in the same paper, for the same input — and REFe does not follow
it.** `PUBLISHED` DriveZero p.4, describing their own teacher's goal handling:

> *"The two ego-frame goal points are embedded using **sinusoidal positional encoding** [79]
> followed by a shared MLP, then mean-pooled into a single goal feature."*

REFe concatenates 4 raw goal scalars (2 points × xy) with 8 ego scalars and feeds them straight
into a `Linear`. `PUBLISHED` **Fourier Features** measures exactly what that costs on coordinate
inputs: a parameter-free Fourier map lifts 2-D image regression **19.32 → 25.57 PSNR**, text
18.40 → 30.47, 3-D shape IoU 0.864 → 0.973 — and *"jointly optimizing these parameters does not
improve performance compared to leaving them fixed."*

⇒ **Recommend: sinusoidal-encode the two goal points before the MLP** (8 frequencies, sin+cos →
64 channels). `DERIVED` cost: input 12 → 76, `ego_enc` **69,120 → 85,504 (+16,384)**. This is the
cheapest published-sourced change in the whole study.
⚠️ `n_goal_points = 2` is already correct and already sourced (`MEASURED` from their released
config: `goal_count_probs [0.5, 0.5]`). Not reopened.

### 2.5 `dec_heads` — parameter-free, genuinely unsettled, and cheap to settle

**First, the fact that reframes it:** `MEASURED` — head count does not change the parameter count
at all. `nn.MultiheadAttention` is `4d² + 4d` regardless of `num_heads`; my analytic model carries
no heads term and reproduced the real instantiation to the parameter. **`dec_heads` is a pure
inductive-bias knob with zero budget consequence**, which is why it belongs in a near-free
experiment rather than in the arithmetic.

The anchors disagree, and none is about a planner at 256:

| anchor | value at width 256 | class |
|---|---|---|
| DETR | **8** heads @256 (head_dim 32) | `PUBLISHED` |
| DriveZero's own **teacher** (Table 1) | **4** heads @256 (head_dim 64) | `PUBLISHED` — ⚠️ different model |
| universal ViT convention (Scaling ViT Table 2, Three-Things Table 1) | head_dim **64** → **4** | `PUBLISHED` |
| Transformer Table 3 (d_model 512) | optimum at **8 heads, d_k = 64**; 1 head −0.9 BLEU, 32 heads −0.7 | `PUBLISHED` |
| DrivoR, VAD, VADv2, Hydra-MDP, GoalFlow, UniAD, BEVFormer | **NOT STATED** | — |

⇒ **`UNSETTLED`. Recommend keeping 8** — it is DETR's value at exactly 256, it is the incumbent,
and churning it without evidence buys nothing. **E-SIZE-4** settles it for ~5 A40-hours at ViT-S.
⚠️ **The 1024-wide compressor is a different question and there I do recommend a change** (§2.1):
16 heads, head_dim 64, matching the trunk whose feature space it reads.
⚠️ Do not read *Are Sixteen Heads* as licence to go low: it prunes a *trained* multi-head model at
test time, and its sharpest finding is that **cross-attention is the least prunable layer type**.

---

## 3 · The parameter arithmetic

### 3.1 The instrument, and why it is trustworthy

`MEASURED` — `scratchpad/refe_sizing_params.py` computes REFe's parameter count **two independent
ways**: (A) instantiating `refe/model.py` and counting real tensors, (B) an analytic formula
authored from `torch.nn` module definitions **without reading the model's reported totals**.

```
AGREEMENT CONTROL
  OK  reg_compress   analytic   12,598,272  measured   12,598,272
  OK  pos3d          analytic    1,966,080  measured    1,966,080
  OK  dec            analytic    4,213,760  measured    4,213,760
  OK  score_dec      analytic    4,213,760  measured    4,213,760
  OK  TOTAL          analytic  329,664,066  measured  329,664,066
  OK  TRAINABLE      analytic   26,584,642  measured   26,584,642
VERDICT: ROUTES AGREE
```

⭐ This matters because CLAUDE.md's rule is that re-running a producer's own derivation measures
**determinism, not correctness**. Route (B) is an independently authored reference, so the
agreement is evidence about the arithmetic, not about repeatability. Every count below uses it.

### 3.2 ⛔ The published budget refutes the current compression reading

**Their configuration, four cameras, target 18,580,000** (`DERIVED` — the module choices come from
DrivoR and PETR, **not** reverse-engineered from 18.58 M):

| reading | trainable | gap | rel |
|---|---|---|---|
| **A** DrivoR-faithful + PETR literal (4C) + per-component MLPs | **17,531,458** | −1,048,542 | **−5.6 %** |
| **C** DrivoR-faithful + PETR literal, single linear score head | 17,055,298 | −1,524,702 | −8.2 % |
| **B** DrivoR-faithful + PETR (hidden = C) + per-component MLPs | 13,792,834 | −4,787,166 | −25.8 % |
| **D** DrivoR-faithful + a LEARNED TABLE (4 cameras) | 20,409,922 | +1,829,922 | +9.8 % |
| **G** cross-attention @1024 MLP 1 + PETR literal | 23,359,042 | +4,779,042 | +25.7 % |
| **F** cross-attention @1024 MLP 4 + PETR literal | 29,653,570 | +11,073,570 | +59.6 % |
| **E** ⛔ **REFe's current reading**: cross-attention @1024 MLP 4 + learned table | **32,532,034** | **+13,952,034** | **+75.1 %** |

And the tightest form of the refutation, with `pos3d` set to **zero** so nothing is smuggled in:

| LoRA assumption | score_depth 4 | 2 | 1 | 0 |
|---|---|---|---|---|
| 3,145,728 (standard 24×1024 ViT-L) | 24,667,714 ⛔ | 22,560,834 ⛔ | 21,507,394 ⛔ | 20,453,954 ⛔ |
| **0 — the most generous possible** | 21,521,986 ⛔ | 19,415,106 ⛔ | 18,361,666 ✅ | 17,308,226 ✅ |

⇒ **A 1024-wide cross-attention compressor at MLP ratio 4, plus two 4-layer 256-d decoders, cannot
fit DriveZero's published trainable count — even if the LoRA adapters were free and the 3D PE cost
nothing.** `DERIVED`, and it is the hardest evidence in this study because it rests on a number the
authors printed rather than on prose.
⚠️ **The one assumption:** that DriveVFM ViT-L is a standard 24×1024 ViT for LoRA-counting
purposes. The LoRA = 0 row exists precisely so the conclusion does not depend on it.
⚠️ Reading D is also informative: **a learned per-camera table would put them 9.8 % over** — a
second, independent reason to doubt that DriveZero's `pos3d` is a table.

### 3.3 The recommended configuration

| module | current | recommended | delta | note |
|---|---|---|---|---|
| `registers` | 16,384 | 16,384 | +0 | 16 is DrivoR's per-camera optimum |
| `pos3d` | **1,966,080** | **528,896** | **−1,437,184** | learned table → PETR MLP over analytic 3D coords (ND=2, hidden C/2) |
| `reg_compress` | **12,598,272** | **6,303,744** | **−6,294,528** | MLP ratio 4 → 1; heads 8 → 16 (free) |
| `scene_proj` | 262,400 | 262,400 | +0 | |
| `ego_enc` | 69,120 | 85,504 | +16,384 | sinusoidal goal encoding |
| `queries` | 16,384 | 16,384 | +0 | |
| `dec` | 4,213,760 | 4,213,760 | +0 | 4 layers @256, FFN 4 — `PUBLISHED` |
| `traj_head` | 81,212 | 81,212 | +0 | |
| `score_q_mlp` | 0 | 81,408 | +81,408 | **NEW** — the score query is the trajectory |
| `score_dec` | 4,213,760 | 4,213,760 | +0 | 4 layers — `PUBLISHED` via DrivoR |
| `score_head` | 1,542 | 396,294 | +394,752 | six per-component MLPs |
| `LoRA(trunk)` | 3,145,728 | 3,145,728 | +0 | rank-32 Q/V — `PUBLISHED` |
| **TRAINABLE** | **26,584,642** | **19,345,474** | **−7,239,168** | **8.06 % → 6.00 %** |
| **TOTAL** | **329,664,066** | **322,424,898** | **−7,239,168** | |

**Against the reference points:**

| | total | trainable | fraction |
|---|---|---|---|
| DriveZero published (ViT-L, **four** cameras, Table A12) | 338,460,000 | 18,580,000 | 5.49 % |
| REFe current (ViT-L, **one** camera) | 329,664,066 | 26,584,642 | 8.06 % |
| **REFe recommended (ViT-L, one camera)** | **322,424,898** | **19,345,474** | **6.00 %** |
| if E-SIZE-1 selects the in-backbone route | 316,121,154 | 13,041,730 | 4.13 % |

⭐ The recommendation lands **+4.1 %** on their published trainable count and **−16.0 M** on their
total — a sane place for a one-camera reproduction to sit. The current build is **+43.1 %** on
trainable.
⚠️ **Neither figure gets under the programme's 300 M thesis**; the frozen ViT-L trunk alone is
303,079,424. The PI has already accepted that for REFe (`REFE_MODEL.md` §2). Not reopened.

### 3.4 Which sizes scale with camera count — I checked the code, and the answer is reassuring

The brief warns that *"several of our numbers were inherited from a 4-camera design without anyone
checking."* `MEASURED` — I checked. **Only two modules scale with camera count, and both are
already correctly scaled to one camera in `model.py`:**

| module | scales with N_cam? | in the code |
|---|---|---|
| `registers` | **YES** (`N × R`, DrivoR §3.2) | `cfg.n_cameras * cfg.n_registers` = 16 ✅ |
| `pos3d` **as a learned table** | **YES** (`N × tokens × width`) | `n_patch = cfg.n_cameras * …` = 1,920 ✅ |
| `pos3d` **as a PETR MLP** | **NO** — one shared MLP over per-camera coords | — |
| `reg_compress` | **NO** — same weights, more K/V tokens | ✅ |
| `scene_proj`, `dec`, `score_dec`, `traj_head`, `score_head` | **NO** — fixed 64 queries | ✅ |
| `ego_enc`, `queries` | **NO** | ✅ |
| **all head counts** | **NO** — and they cost 0 params anyway | ✅ |

⇒ **There is no 4-camera parameter-shape inheritance defect.** What *was* inherited unexamined is
a **justification**, not a shape: 16 registers is DrivoR's per-camera optimum measured at four
cameras, where the count bought *camera-awareness*. At N=1 that rationale is void (§2.1); 16
survives on the compression-ratio argument instead, which is a different and weaker support for
the same number. Recording that distinction is the honest form of "we checked."

### 3.5 Head sizing is not a compute question — the arithmetic, so nobody argues from vibes

`DERIVED`, `scratchpad/refe_flops.py`, one camera, ViT-L, batch 1:

```
frozen trunk (ViT-L, 1925 tokens)                     764.98 GMAC   99.15 %
HEAD TOTAL (current)                                    6.52 GMAC    0.85 %
   reg_compress cross-attn (16 q over 1920 kv @1024)    4.12
   score_dec 4 x CrossBlock over 1920 VISUAL tokens     1.50
   scene_proj(visual) 1920 x 1024 x 256                 0.50
   dec 4 x CrossBlock over 16 scene tokens              0.25
   reg_compress MLP 4x on 16 tokens                     0.13
trunk with +16 in-backbone registers (1941 tokens)    772.85 GMAC  (+1.03 %)
HEAD TOTAL (DrivoR route)                               0.51 GMAC   0.066 %
activation memory: trunk ~1.057 GiB/sample vs head ~0.010 GiB/sample  (109x)
```

Three consequences:
1. **The entire head is 0.85 % of the forward pass.** No head-sizing recommendation in this study
   can be rejected on compute grounds, and none should be defended on them either.
2. **The DrivoR route costs +1.03 % on the trunk and takes the head to 0.066 %** — which reproduces
   DrivoR's own published decomposition (*350 GFLOPs backbone ∥ 1 GFLOP rest*, Table 11). Our
   current head is 6.52 GMAC ≈ 13 GFLOP, **13× DrivoR's "rest"**.
3. **LoRA lives inside every trunk block, so backprop traverses the whole trunk and trunk
   activations are 109× the head's.** ⇒ **head sizing does not move the batch-size ceiling**, and
   the dev box's batch-2 limit will not be relieved by any change recommended here.

---

## 4 · What changes with data scale — and what does not

**The short answer: none of the module sizes. The schedule does.**

`PUBLISHED` **MOSAIC** (banked) is the only study I found that runs a NAVSIM planner at exactly our
budget — **100 / 200 / 400 / 800 / 1600 / 2400 navtrain clips, 3 seeds**, Hydra-MDP on a pretrained
VoVNetV2-99. **They did not shrink the model at small budgets. They changed the epoch count:
60 epochs at ≤800 clips, 50 at 1,600, 45 at 2,400.** Their EPDMS at those budgets: 84.66 (100
clips) → 86.69 (400) → 88.62 (1600), random selection, ±0.20–0.60.
⇒ The published response to small data, on our exact benchmark and task, is **schedule and data
selection**, not architecture. `PUBLISHED` **BiT** says the same thing from the other direction:
no weight decay, no dropout, no downstream regularisation — *"setting an appropriate schedule
length … provides sufficient regularization."*

**Per-module verdict:**

| module | scale-sensitive? | why |
|---|---|---|
| `reg_compress` MLP ratio | **no** | a bottleneck's job does not change with corpus size; Set Transformer's m=16 **beat** the unbottlenecked model *because* of the bottleneck's regularisation |
| `reg_compress` blocks | **no** | Perceiver's front-stacking penalty is architectural, not data-dependent |
| `score_dec` depth | **no** | 4 is the lineage value; every published depth ablation saturates at 2–6 regardless of scale |
| `pos3d` form | **no** | an analytic PE is *more* right at small scale, not less — fewer free parameters to fit |
| `ego_enc` | **no** | 0.26 % of trainable either way |
| head counts | **no** | zero parameters; pure inductive bias |
| `n_registers` | **no** | DrivoR's own scaling study (Fig. 3) varies training data at a **fixed 16** |
| **epochs** | ⭐ **YES** | MOSAIC: 60 / 50 / 45 across a 3× budget range |
| **LoRA rank** | ⭐ **possibly** | see below — and it is the open question worth GPU |

**⚠️ Do not import an LLM law here.** `PUBLISHED` Chinchilla gives a = 0.46–0.50, b = 0.50–0.54
(Table 2); the "20 tokens per parameter" figure is **`DERIVED` from Table 3, not stated by the
authors**, and they explicitly exclude the multi-epoch regime and frame cross-modality transfer as
an *expectation*. `DERIVED` — applied naively at 10³ examples × Waymo's own 849 tokens/example ÷ 20
it prescribes **~42,000 trainable parameters**, 460× below even the DrivoR-route recommendation.
The arithmetic is correct and the answer is meaningless. The admissible import is Waymo's
**direction** (`PUBLISHED` 2506.08228: N_opt ∝ C^0.63, D_opt ∝ C^0.44, so *"optimal model size
should grow ≈1.5 times as fast as the number of training examples"*) — a direction, never a
magnitude, and never across two decades of extrapolation.

**⚠️ And a published warning against fitting our own exponent later.** `PUBLISHED` 2504.04338
Table 4: the *same* estimator refit on only the low-data end gives an exponent **2.74× shallower**
(−0.1274 vs −0.3486) and an extrapolation loss **8.2× worse**, with ε∞ collapsing to exactly 0.
Their own appendix: *"the plateau becomes apparent only for datasets exceeding 10³ hours."*
This is CLAUDE.md's window-dependent-exponent rule, published, in our exact domain.

**⚠️ No published rule relates trainable parameters to sample count.** Probed four ways across
six banked scaling papers and the PEFT literature: **none exists.** Any number of that form in a
TanitAD document would be unsourced. What the literature publishes instead are measured
*crossovers between methods at a fixed data size* — VPT at N=1000 (0.53 % trainable beats full
fine-tuning on 20/24 tasks) and 2410.18647 at ~800 demos (frozen 0.00, LoRA-8 0.72, full 0.90).
**Our setting sits between them, and that gap is E-SIZE-7.**

**Scale reference, so "10³" is unambiguous.** `PUBLISHED` NAVSIM §3.1: navtrain = **103k samples**,
navtest 12k; MOSAIC: **1,192 logs, 103,288 frames at 2 Hz, 4,601 non-overlapping 10-s clips**.
`MEASURED` our rehearsal bank: **1,964 tuples trained** (2,182 built). ⇒ **~1.9 % of navtrain by
frame.** ⚠️ Frames and clips differ by 22×; state the unit.

---

## 5 · Pre-registered experiments — for what the literature does not settle

### 5.1 The cost basis, and a discrepancy it exposes

`MEASURED`: ViT-L, batch 2, 512×960 on the dev-box RTX 4060 = **1.56 s/step = 0.780 s/sample**
(`raw/refe_wta_batch_starvation.txt`). `ESTIMATED`: an A40 is ~3× → **0.260 s/sample**.
⭐ **Cross-check that makes the basis quotable:** at their 337 K samples × 25 epochs this yields
**608.5 A40-h**, against the paper's own **16 × H20 × 38 h = 608 GPU-hours** — a number the basis
was not fitted to.
⛔ **The ~3× multiplier is UNMEASURED** (`POD_HANDOFF.md` item 3 says so). **Replace it with a real
A40 timing in the first pod hour before committing to anything below.**

⛔ **The same basis contradicts `POD_HANDOFF.md`'s own arm table by exactly 9.0×:**

| arm | claimed | this basis | ratio |
|---|---|---|---|
| A rehearsal (1,746 × 50) | 0.7 A40-h | **6.3** | 9.0× |
| B mini-scale (14,000 × 25) | 2.8 A40-h | **25.3** | 9.0× |
| D full (337,000 × 25) | 608 A40-h | 608.5 | **1.0×** |

The identical slip on both small rows while the large row checks out. Escalated in §0.3 because
*"the first pod day costs under three GPU-hours"* is the sentence the A40 request rests on.

**Cost of one run** (`DERIVED`; ViT-S is 14.2× cheaper by depth × width²):

| bank | epochs | ViT-L A40-h | ViT-S A40-h |
|---|---|---|---|
| rehearsal (1,964) | 25 | 3.55 | 0.25 |
| arm B (~14,000) | 25 | 25.28 | 1.78 |
| arm C (~123,000) | 25 | 222.08 | 15.62 |

### 5.2 ⛔ The gate that applies to every experiment below

`MEASURED` (`REFE_MODEL.md` §4b, §9): winner-takes-all routes gradient to **one proposal per
sample**, so at batch B at most B of 64 proposals are touched per step. At the dev box's batch-2
ceiling this is **3.1 %** — *"1/2 on every logged step"*, both samples choosing the same proposal.
**Their batch was 256.**
⇒ **No architecture comparison is admissible on the dev box at any batch it can hold.** Every arm
below is a pod arm at batch ≥ 64, and each must print its WTA reach.

`MEASURED` and binding on interpretation (CLAUDE.md `H-ESTIM-SEED-1`): a separated episode-cluster
CI from a one-seed arm is **necessary, not sufficient** — two runs differing in *nothing* cleared
it on 14.3 % of cells. **Every experiment below therefore carries a replicate arm**, and no lever
effect is reported unless it exceeds the replicate spread.

### 5.3 The experiments, ranked by measured lever size — not by the brief's module order

#### ⭐ E-SIZE-7 — LoRA rank ladder with a frozen floor · **126.4 A40-h** (arm B) / **17.7** (rehearsal)
*Ranked first because it is the only knob with a published effect at 10³.*

| arm | config |
|---|---|
| a | rank 32 (current) — **control** |
| b | rank 8 |
| c | rank 64 |
| d | **frozen trunk, LoRA disabled** — the floor that must read a known-bad value |
| e | replicate of (a), same flags, different seed — **the noise floor** |

**Committed in advance.** ⓘ If (d) collapses toward the untrained baseline, the frozen-backbone
risk from 2410.18647 Table 2a is real on our task and encoder adaptation is where capacity belongs
— **rank becomes the primary sizing question and the head recommendations in §0.2 stand unchanged
as the cheap default.** ⓘ If (b) ≈ (a) ≈ (c) beyond the (a)/(e) spread, rank is saturated at 8,
**2.36 M trainable is recoverable**, and the head modules become the live lever after all. ⓘ If (c)
> (a) beyond the spread, the trainable budget should move *into* the trunk and *out* of the head —
the opposite of what a head-sizing study would otherwise conclude, and it must be reported that way.

#### E-SIZE-3 — `pos3d` form · **126.4 A40-h** (arm B) / **17.7** (rehearsal)

| arm | config |
|---|---|
| a | learned 1920×1024 table (current) — **control** |
| b | PETR-style MLP over analytic 3D coords, ND=2, hidden C/2 |
| c | **no `pos3d` at all** — the floor; the trunk's RoPE is the only positional signal |
| d | **a single learned vector broadcast to every token** — the no-position-structure control, which must read (c)'s value |
| e | replicate of (b) |

⭐ Arm (d) is the discriminator, and it is the cheap half: if (a) beats (c) by no more than (d)
does, the table's **per-position structure** is doing nothing and its 1,966,080 params are a bias
term. CLAUDE.md's rule that a control must read a known value, applied.
**Committed.** ⓘ (b) ≥ (a) beyond the (b)/(e) spread ⇒ **replace the table**, and the resolution
lock and calibration mismatch are removed for free. ⓘ (a) > (b) beyond the spread ⇒ the table is
earning its keep on a *fixed-geometry* bank; report it, and **re-run once the bank spans both
intrinsic rigs**, because that is the condition under which the table must fail. ⓘ inside the
spread ⇒ **adopt (b) anyway on the §0.1 arguments** — same score, 27 % of the parameters,
resolution-free.
⛔ **Not an experiment — an identity.** A trained arm (a) cannot be evaluated at a second input
geometry at all: the table's shape mismatches and it would have to be re-initialised. Report that
as a structural property, never as a measured degradation.
**Prerequisite:** the bank builder must carry `intrinsic` and `sensor2lidar_*` (§2.3). ~0 GPU.

#### E-SIZE-1 — `reg_compress` mechanism · **75.8 A40-h** (arm B) / **10.6** (rehearsal)

| arm | config |
|---|---|
| a | bolt-on cross-attention @1024, MLP 1, 16 heads (the §0.2 recommendation) — **control** |
| b | **in-backbone registers**: 16 trainable tokens appended to the ViT input, harvested at the last layer, projected to 256 |
| c | replicate of (b) |

⚠️ **Arm (b) has two implementation consequences that must be built, not assumed:** the RoPE offset
becomes `1 + 4 + 16` (the task registers carry no position and must not be rotated), and **`pos3d`
must move to the trunk *input*** or it cannot reach the compression at all.
**Committed.** ⓘ (b) ≥ (a) beyond the (b)/(c) spread ⇒ adopt the in-backbone route; **6,303,744
trainable is recovered**, REFe lands at 13.04 M / 316.12 M, and the reproduction becomes faithful
to the cited source. ⓘ (a) > (b) beyond the spread ⇒ **the bolt-on block earns its parameters
despite DrivoR's 89.3-vs-90.0**; record it as a measured, deliberate deviation and proceed to
E-SIZE-2. ⓘ inside the spread ⇒ adopt (b) on parameter grounds and say the comparison was flat.

#### E-SIZE-2 — `reg_compress` MLP ratio at 1024 · **101.1 A40-h** (arm B) / **14.2** (rehearsal)
*Runs only if E-SIZE-1 selects the bolt-on block.*
Arms: ratio **4** (12,598,272) · **1** (6,303,744) · **0**, no MLP (4,204,544) · replicate of the
winner.
**Committed.** ⓘ ratio 1 ≥ ratio 4 beyond the replicate spread ⇒ ship 1 and bank 6.29 M; this is
the outcome Perceiver/Perceiver IO predict. ⓘ ratio 4 > 1 beyond the spread ⇒ Flamingo's anchor
wins, ship 4, and **record that REFe's trainable count then exceeds the paper's by design.**
ⓘ ratio 0 ties the others ⇒ ship 0 and bank 8.39 M.

#### E-SIZE-4 — `dec_heads` · **5.3 A40-h** (ViT-S, arm B) / **0.75** (rehearsal)
Arms: **8** (current, DETR's value @256) · **4** (head_dim 64; their own teacher's value) · **16**.
Zero parameter change in every arm, which is what makes this the cheapest real question in the
study.
**Committed.** ⓘ any arm beats 8 beyond the replicate spread ⇒ adopt it; it is free. ⓘ all inside
the spread ⇒ **8 is confirmed as arbitrary-but-harmless and the question is closed**, which is a
result worth having rather than an open field.

#### E-SIZE-5 — register count · **5.3 A40-h** (ViT-S, arm B) / **0.75** (rehearsal)
Arms: **8** · **16** (current) · **32**. Parameter deltas: −8,192 / 0 / +16,384.
**Committed.** ⓘ 32 > 16 beyond the spread ⇒ adopt 32; Perceiver IO's asymmetry (too few costs
−5.11, too many −0.03) predicted it and it costs 16,384 params. ⓘ 8 ≈ 16 ≈ 32 ⇒ DrivoR's flatness
(0.3 PDMS over a 6× range) reproduces on one camera; **keep 16 and close the question.**

#### ⛔ E-SIZE-6 — `score_dec` depth · **BLOCKED, and the blocker is named**
**Not scheduled.** Depth 4 is `PUBLISHED` (§2.2), and the rig cannot test it regardless: scorer
coverage is **22.7 %**, `comfort` varies across candidates in **0.0 %** of frames, and components
3 and 6 are mis-mapped (D4a / D4b). **A depth ladder on a scorer that cannot rank measures
nothing.** Unblocked by: fixing D4a/D4b, replacing the inert `comfort` channel, and raising
coverage. Those are `PAPER_CONFORMANCE_REVIEW.md` work items, not sizing work items.

### 5.4 Totals, and what I would actually run

| plan | A40-h |
|---|---|
| all five live experiments at **rehearsal** scale (1,964 tuples) | **61.8** |
| all five at **arm B** scale (~14,000 tuples) | **440.4** |
| ⭐ **E-SIZE-7 + E-SIZE-3 at arm B, E-SIZE-4 + E-SIZE-5 at ViT-S** | **263.4** |
| ⭐⭐ **the cheapest useful first day: E-SIZE-4 + E-SIZE-5 at ViT-S rehearsal scale** | **1.5** |

⚠️ **The rehearsal-scale column is a smoke pass, not a result.** At 1,964 tuples, with the batch
constraint of §5.2 and no seed replication history on this rig, it tells you the arms **run** and
gives real timings. It does not discriminate architectures, and no arm from it should be quoted as
a sizing verdict.

⭐ **Recommended order:** E-SIZE-4 and E-SIZE-5 first (1.5 A40-h, zero-to-negligible parameter
change, and they close two `UNSETTLED` rows outright), then **E-SIZE-7** (the largest published
lever at our data scale), then E-SIZE-3, then E-SIZE-1. **E-SIZE-2 only if E-SIZE-1 says the block
should exist.**

---

## 6 · What I could not settle, stated plainly

1. **No published head-to-head of 3D PE vs 2D PE in a single-camera setting**, across
   PETR / PETRv2 / 3DPPE / CAPE / DETR3D / BEVFormer / LVSM. §2.3's recommendation rests on PETR
   Table 3 row 3 as an *implication*.
2. **The ND=2 licence rests on a single paper** (3DPPE Table 9) on a nuScenes detection task. The
   fully cited fallback is PETR's literal 4C recipe at 4,985,856.
3. **DrivoR's cross-attention ablation (row e) states no depth, width or head count** — *"roughly
   the same number of parameters"* is all the paper gives. Our 12.6 M matches it only under my own
   width-scaling `DERIVED` argument.
4. **No head count is stated anywhere in DrivoR, DriveZero's planner, VAD, VADv2, Hydra-MDP,
   GoalFlow, UniAD or BEVFormer.** E-SIZE-4 is unavoidable if the answer is wanted.
5. **Every DrivoR ablation number is a single point estimate on navval** — no CI, no seeds, in
   Tables 4–7. The 89.3-vs-90.0 mechanism gap and the 0.3-PDMS register sweep inherit that
   weakness, and E-SIZE-1 / E-SIZE-5 exist because of it.
6. **The A40/4060 multiplier is unmeasured.** Every GPU-hour in §5 is `ESTIMATED`.
7. **The paper is internally under-determined on the compression** (§1.1). I ranked the
   constraints; I did not resolve the contradiction, and no amount of re-reading will.
8. **DriveVFM's trunk is not a plain ViT-L.** `DERIVED`: their frozen part is 319.88 M against our
   303.08 M — **16.8 M unmodelled**. Reading A's +1.05 M residual (§3.2) is plausibly some of it.

---

## 7 · Deliverable manifest

### 7.1 Artifacts

| artifact | where it lives | class |
|---|---|---|
| **this study** | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/MODULE_SIZING_STUDY.md` — **repo, staged** | — |
| camera-calibration probe output | `…/2026-09-20-refe-plan/raw/refe_cam_calibration_variance.txt` — **repo, staged** | `MEASURED` |
| parameter accounting, two independent routes | `…/2026-09-20-refe-plan/code/refe_sizing_params.py` — **repo, staged** | `MEASURED` |
| four-camera reconstruction cross-check | `…/2026-09-20-refe-plan/code/refe_reconstruct_theirs.py` — **repo, staged** | `DERIVED` |
| final recommendation arithmetic | `…/2026-09-20-refe-plan/code/refe_final_reco.py` — **repo, staged** | `DERIVED` |
| FLOP / activation accounting | `…/2026-09-20-refe-plan/code/refe_flops.py` — **repo, staged** | `DERIVED` |
| experiment costing | `…/2026-09-20-refe-plan/code/refe_expt_cost.py` — **repo, staged** | `ESTIMATED` |
| calibration probe (source) | `…/2026-09-20-refe-plan/code/probe_cam_calib.py` — **repo, staged** | `MEASURED` |
| budget admissibility solver | scratchpad only (superseded by `refe_reconstruct_theirs.py`) | `DERIVED` |

⛔ **No model code was modified.** `refe/model.py` is untouched.

### 7.2 Library keys banked for this study

`python tools/kb_add.py --verify` → **552 entries, 0 orphans, 0 problems** (content hash, not
presence), run after the final ingest.
**Staged:** `Library/library.json`, `Library/LIBRARY.md`, and the **21 new PDFs** under
`Library/papers/` (tracked PDFs 524 → 545). ⚠️ Seven other untracked PDFs sit in that directory
from a different stream (`2510.24108`, `2511.09515`, `2602.11075`, `2606.08860`, `2606.19641`,
`2609.03572`, `2609.06370`) — **deliberately left unstaged**; they are not this study's work.

**Newly banked (18):** `2203.05625` PETR · `2206.01256` PETRv2 · `2211.14710` 3DPPE ·
`2103.03206` Perceiver · `2107.14795` Perceiver IO · `1810.00825` Set Transformer ·
`2204.14198` Flamingo · `2301.12597` BLIP-2 · `2508.10104` DINOv3 · `2106.04560` Scaling ViT ·
`2203.09795` Three things about ViT · `1905.10650` Are Sixteen Heads Really Better than One ·
`2012.12877` DeiT · `2307.06304` NaViT · `2006.10739` Fourier Features · `2005.12872` DETR ·
`1706.03762` Attention Is All You Need · `2106.09685` LoRA · `2203.15556` Chinchilla ·
`2203.12119` VPT · `2106.10270` How to train your ViT.

**Already banked, re-tagged `refe-sizing` with this study as `cited-by` (10):** `2609.06055`
DriveZero · `2601.05083` DrivoR · `2309.16588` Vision Transformers Need Registers ·
`2410.18647` Data Scaling Laws in Imitation Learning for Robotic Manipulation ·
`2604.08366` MOSAIC · `2504.04338` Data Scaling Laws for E2E AD · `2510.12796` DriveVLA-W0 ·
`2506.08228` Waymo Scaling Laws · `2412.02689` Data Scaling Laws for Imitation-Learning E2E AD ·
`2212.10156` UniAD.

**Also cited, already in the library:** `2411.15139` DiffusionDrive · `2203.17270` BEVFormer ·
`2102.10882` CPVT · `2403.13298` RoPE-ViT · `2212.08013` FlexiViT · `2010.11929` ViT ·
`2406.15349` NAVSIM.

⚠️ **Not banked, and therefore not load-bearing anywhere above:** BiT (`1912.11370`) and Prompt
Tuning (`2104.08691`) were read by a research stream from HTML and appear in this study only as
directional colour in §4; no recommendation rests on them.

### 7.3 Escalation

§0.3 items 1–3: the `pos3d` replacement (a defect, not a preference), the scope of the PI's
compress-at-1024 decision (E-SIZE-1 chooses an implementation *inside* it), and the 9.0×
inconsistency in `POD_HANDOFF.md`'s arm A/B costs.

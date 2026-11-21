# TanitLang — the Energy Bridge

**Architecture for semantically grounded reasoning on REF-C v5, with and without text.**
Author: TanitAD_TrainingFlyWheel · 2026-09-06 · status **DESIGN, pre-registration pending**
Supersedes: `DIALOGUE_02` (v2), `DIALOGUE_03` (v3), `DIALOGUE_05` (reasoner internals),
`DIALOGUE_06` (relational semantics). Those remain readable for the reasoning that
led here; **this file is the one to build from.**

---

---

# ⭐ STATE OF THE DESIGN — read this first (2026-09-06, after WP-I/L/M/N)

This document grew a body plus **six amendments**, several of which correct earlier ones.
The amendments are kept because the record of what was wrong is worth as much as the design.
**This section is the consolidated position; the body and amendments are the working.**

## What is MEASURED and stands

| # | established | number | where |
|---|---|---|---|
| 1 | **The re-ranking channel works.** A trained zero-init port moves the selected anchor, and is exactly inert at init. | flip **0.2324**, geo **0.2022 m**; zero-port identity **0/0/0 exact** | WP-M |
| 2 | **The channel is scene-dependent, and it REPLICATES.** A prior from a *different episode* degrades selection by a separated margin under **two independent window draws**, and **23/34 and 26/34 episodes** point the same way (medians −0.0251 / −0.0253). | **−0.0255** CI [−0.0425, −0.0093] and **−0.0375** CI [−0.0521, −0.0236] | WP-N, WP-O |
| 3 | **The port adds NEW ordering, not an echo** of the base scorer. | Spearman(Δ, S0) **0.096** | WP-M |
| 4 | **Selection headroom is real and large.** The fan already contains a far better anchor than the scorer picks. | selected **0.65 m** vs fan-best **0.2315 m** | WP-M, H-TL-1 |
| 5 | **A strong pretrained backbone carries agent semantics the trunk does not.** DINOv3 is the only arm of four clearing its own within-episode null on agent presence. | `vru<20m` **p 0.000**, `vru<60m` **p 0.020**; adj **+0.1935** vs trunk **+0.0792** | WP-I v3 |
| 6 | **That deficit is NOT a pooling artifact.** The full 8×8 map with zero pooling loss is the *weakest* arm. | adj **+0.0045**, p **0.495** | WP-L |
| 7 | **It is partly a linearity artifact — on the FAR band only.** An RBF read of the trunk clears where linear does not. | p **0.040** / **0.020**; near band: nothing clears (best p 0.155) | WP-L |
| 8 | **refcv5's grounding is AGENT TOKENS.** `agent_tok [B,100,384]`, gated cross-attention, two flags both defaulting off. **No BEV, LiDAR, map or lane token exists.** | `d_model` = **384**; budget 107.06 M → 114.13 M | AUDIT |

## What is WITHDRAWN or NARROWED — read before quoting the body

| claim | status |
|---|---|
| *"The Energy Bridge dissolves the influence/consistency tension"* | ⛔ **WITHDRAWN.** The frontier says *"neither design satisfies both desiderata"*. We have a continuous dial where the field has two discrete points; that is the contribution. |
| *"A large ‖S₁−S₀‖ establishes influence"* | ⛔ **INSUFFICIENT.** A shuffled CoT reproduces the effect and a `"depicts"`-token prior doubles it (`2606.12706`). Decide on `outcome_null_screen`, never on KL. |
| *"~89 % of the port's benefit is generic"* | ⛔ **RETRACTED.** The control mispaired by index on episode-grouped rows — **97.5 % same episode**. With a cross-episode control the effect is scene-specific and separated. |
| *"DINOv3 is the only arm tracking crowding"* | ⛔ **WITHDRAWN.** On the verified join all four arms clear `crowd` and DINOv3 is not best null-adjusted. |
| *"The trunk has no VRU signal / equals random weights"* | ⚠️ **TOO STRONG.** Neither reaches significance; that is not the same as being equal. |
| *"`lan_emb` is an existing language port"* | ⛔ **WRONG.** LAN is **Lane-Anchored Navigation**. No tokenizer, embedding table or text head exists anywhere in the model. |
| *"BEV ×161, agent ×73, map ×54"* | ⛔ **WRONG inventory.** Agent tokens only, 100 of them. |
| *"`maneuver_to_anchor` is the free live port in refcv5"* | ⛔ **ABSENT there** — `factored_maneuver` builds `lat/lon_to_anchor` instead. It exists only on the base checkpoint. |

## The binding constraints the measurements imposed

1. ⛔ **β·E must be comparable to `sd[S0]`, or consistency is unmeasurable.** REF-C's port runs
   at **~8 %** of the scorer's scale and `CON_rank` reads **64.40 of 128 against chance 64.50**.
   A timid port that stays timid produces an unfalsifiable consistency claim.
2. ⛔ **The bar is the CROSS-EPISODE control, not the no-port arm.** A 5-class manoeuvre prior
   already moves it by 0.0254 m.
3. ⛔ **Never supervise the rationale from the logged future trajectory** — trajectory
   anchoring bias (`2608.01755`); LCDrive cold-starts exactly that way on our corpus.
4. ⛔ **Consistency is PER FAMILY with a seed floor.** 88 % inconsistency from seed alone on
   our corpus; lateral 100 % vs longitudinal 62–69 %. A pooled score hides our own defect.
5. ⛔ **refcv5's seams are NOT in `HEAD`.** A fresh clone cannot build the model. Blocking;
   escalated to `BACKLOG.md`, not mine to commit.

## What is still HYPOTHESIS — no measurement behind it yet

The Energy Bridge itself (one energy read in two directions); the recursive core and its
halting head; the shared codebook tying `<TAC:…>`/`<STR:…>` to the planner's embedding rows;
language as a regulariser rather than a control-path link; the agent-perturbation
interaction test. **None of these is a result.**

## ⭐ STAGE A's beta IS NOW MEASURED: 1.5 (authority ~12 %)

| beta | authority | CON rank (chance 64.5) | ADE (m) |
|---:|---:|---|---:|
| 1.00 | 8.1 % | 64.40 [59.19, 69.20] overlaps chance | 0.6521 |
| **1.50** | **12.2 %** | **59.06 [53.79, 64.27] SEPARATED** | **0.6479 (best in sweep)** |
| 2.00 | 16.3 % | 53.66 [48.41, 58.92] SEPARATED | 0.6583 |
| 3.00 | 24.4 % | 38.70 SEPARATED | **0.8168, +0.1524 [+0.0608,+0.2547] SEPARATED WORSE** |

**beta = 1.5 is the SMALLEST authority at which consistency separates from chance, and it
carries the best ADE in the whole sweep.** Below it consistency is unmeasurable; above it
you pay ADE for consistency already available. The knee is beta = 3.0, where ADE degrades
and the scene-specificity gap stops being separated -- two independent signals on one grid
point.

⚠️ **Withdrawn point estimates, so they are not re-quoted:** across beta in [0.5, 2.5] the
ADE is **statistically indistinguishable from base** (every interval overlaps 0). "beta=2 is
better than base" and "beta=2 is worse than beta=1" are BOTH unestablished. What is
separated is the **scene-specificity gap**, monotone across seven consecutive betas
(+0.0115 -> +0.0845), and the CON_rank separation from beta 1.5.

⚠️ **Reporting rule:** every `CON_rank` is quoted with its **beta, its authority %, and its
ADE**. All three, or the number reports obedience rather than understanding.

## The next three decisions, in the order the evidence ranks them

1. **Distillation head targeting NEAR-agent semantics only** — WP-L showed the far band is
   already reachable nonlinearly, so a teacher there would be paying twice.
2. **Turn refcv5's agent tokens on and re-run WP-I against the agent-token bus** rather than
   `cond₀`. Blocked on the `HEAD` seams.
3. **Stage A at `tac_goal_head`**, with `β` sized from constraint (1) — otherwise the
   consistency half of the whole design is unmeasurable by construction.

---

## 0. The requirement, and the one genuinely hard part

The PI's requirement, restated so it can be falsified:

> The model must generate a chain-of-thought which **influences the behaviour** and
> is at the same time **consistent with the selected trajectory** and with the
> tactical and strategic planning.

Everything else in the brief — ≤1B parameters, shared embedding space, nav commands
and queries, 300–500 ms tact, optional text — is engineering. **This one sentence is
the research problem**, because its two halves pull in opposite directions:

| | mechanism | influence | consistency |
|---|---|---|---|
| **CoT as a caption** | reason *after* the plan, describe it | **zero** — it is an echo | perfect, and worthless |
| **CoT as an override** | reason *before* the plan, force it | high | unenforced — it may confabulate |

The programme has already been burned by exactly the degenerate case on the left.
MEASURED (`MODEL_REGISTRY.md`, flagship v1): the route head scored **1.0000** because
it was an **exact bijection of the nav signal we fed it** (369/369 and 81/81) — an echo
of its own input read as skill. A CoT trained to agree with the plan it did not cause
is the same defect with more words.

So the design must make influence and consistency **separately measurable**, and it
must make each one **falsifiable by a control that reads a known value**.

---

## 1. The resolution

> ⭐ **Influence and consistency are the same function, read in two directions.**

Introduce one new object: a scalar **compatibility energy**

```
        E_φ( z , τ )  ∈ ℝ          z = the reasoning state,  τ = a trajectory
```

low energy = "this reasoning and this trajectory agree".

Now use it twice:

```
  INFLUENCE     the reasoner's ONLY channel into behaviour is a re-ranking of the
                existing 128-anchor fan by E:
                        S₁(k) = S₀(k) − β · E_φ(z, τ_k)          k = 1…128
                so the plan changes iff E varies with z.

  CONSISTENCY   the reasoning that was emitted must score LOW against the
                trajectory that was finally selected:
                        L_cons = E_φ(z, τ_sel) − softmin_k E_φ(z, τ_k)
                so the CoT must not merely be compatible with the winner, it must
                prefer it over the other 127.
```

Both quantities are readings of **one** learned function. That is what dissolves the
tension:

* A CoT that is a **caption** makes `E` constant in `z` → the re-ranking `S₁ − S₀`
  vanishes → **influence is measurably zero**, and the arm is refuted by its own metric.
* A CoT that **confabulates** makes `E(z, τ_sel)` high → **consistency is measurably
  violated**, and the same metric refutes it.
* Neither failure can hide behind the other, because they are two evaluations of the
  same `φ`.

This is also why `E` must be a *scorer*, not a *generator*. A generator can always
satisfy a consistency loss by copying. A scorer cannot: to give the selected anchor a
lower energy than the other 127 it has to represent something about the scene that
discriminates them.

---

## 2. How the loop is broken

The PI identified the loop precisely: *"the reasoning must be consistent to the
trajectory fan, so there is a little of loop where we need to be careful."*

The loop is broken **in time**, by borrowing the two-pass structure of
classifier-free guidance:

```
  ┌── PASS 0 ── the BASE policy, reasoner ports OFF ───────────────────────────┐
  │  image → REF-C encoder → cond₀ → 128 anchors → 2-step diffusion → F₀       │
  │  scorer → S₀ → τ₀                                                          │
  │  ⇒ F₀ is produced WITHOUT the reasoner, so the reasoner cannot have        │
  │    caused it. It is a legitimate object to reason ABOUT.                   │
  └────────────────────────────────────────────────────────────────────────────┘
                        │  cond₀ , F₀ (as a PROPOSAL SET, never a target)
                        ▼
  ┌── REASONER R ── recursive, grounded, ≤1 B ────────────────────────────────┐
  │  z_T , goal tokens g , constraint tokens c                                 │
  └────────────────────────────────────────────────────────────────────────────┘
                        │  z_T  (and g, c through the zero-init prior ports)
                        ▼
  ┌── PASS 1 ── the GUIDED policy ────────────────────────────────────────────┐
  │  S₁(k) = S₀(k) − β·E_φ(z_T, τ_k) + Wg·g + Wc·c      → τ_sel               │
  └────────────────────────────────────────────────────────────────────────────┘
```

**The reasoner reasons against `F₀` and is scored for consistency against `τ_sel`.**
Two different objects, two different passes ⇒ no circularity, and the direction of
causation is explicit in the computation graph rather than asserted in prose.

> ⭐ **The guidance delta is the influence metric, and it is zero at initialisation
> by construction.** Every port the reasoner writes through (`Wg`, `Wc`, and `β`)
> is **zero-initialised**, exactly like REF-C's existing `maneuver_to_anchor` /
> `lat_to_anchor` / `lon_to_anchor` / `lan_gate` family. So at step 0 the TanitLang
> arm is **bit-identical to the REF-C v5 baseline**, and any later difference is
> attributable to the reasoner and to nothing else. This is the property that makes
> the arm comparable at all — a module that changes behaviour at init cannot be
> compared against its own base.

### Influence, defined

```
  INF_sel   =  1[ argmax S₁ ≠ argmax S₀ ]                 selection flip rate
  INF_geo   =  ‖ τ_sel(S₁) − τ_sel(S₀) ‖   (m, ADE-style)  geometric displacement
  INF_kl    =  KL( softmax S₁ ‖ softmax S₀ )               distributional influence
```

with the **no-influence control** available for free: the same checkpoint evaluated
with `β = 0, Wg = Wc = 0` **must** read `INF_sel = 0`, `INF_geo = 0.0000`,
`INF_kl = 0.0000` exactly. Not approximately — this is an identity, and if it does
not hold, the arm is mis-wired and no number from it is admissible.

### Consistency, defined

```
  CON_rank  =  rank of τ_sel under E_φ(z, ·) among the 128        (1 = perfect)
  CON_gap   =  E_φ(z, τ_sel) − min_k E_φ(z, τ_k)                  (0 = perfect)
  CON_text  =  agreement of the RENDERED CoT with τ_sel           (text arm only)
```

### The controls that make both readable

| control | what it does | what it **must** read |
|---|---|---|
| **shuffled-z** | pair this window's fan with **another window's** `z` | `E` must RISE and `CON_rank` degrade. If not, `z` carries nothing about *this* scene. |
| **shuffled-fan** | pair this `z` with another window's fan | same. Symmetric test of the other argument. |
| **constant-z floor** | replace `z` by its dataset mean | re-ranking must collapse: `INF_kl → 0`, `S₁ → S₀`. Any residual is `E`'s dependence on `τ` alone, which is *not* reasoning. |
| **β = 0 identity** | disable the bridge | `INF_* = 0` **exactly** (see above). |
| **raw-input floor** | `E` built on pooled pixels instead of `z` | a reasoning state that does not beat raw pixels has added nothing (the WP-J lesson, in the reasoner's costume). |
| **replicate arm** | same flags, re-run | ⛔ **mandatory.** MEASURED 2026-09-05: `A0b_replicate` produced "separated" differences on **3 of 18** metrics against an arm with *zero levers moved* — a ~17 % false-positive rate. A separated CI is **necessary and not sufficient** for a lever claim. |

> ⚠️ **The falsifiability requirement, stated so it can fail.** `E` must be able to
> *reject* a trajectory. The programme's standing anti-echo rule generalises here: ask
> whether the CoT could have been computed from the answer alone. Operationally —
> fit a probe `τ_sel → z`. Whatever `z` variance that probe explains is echo. The
> **residual** is the reasoning content, and a CoT whose residual is ~0 is a caption
> however fluent it reads.

---

## 3. The full architecture

```
                      ┌───────────────────────── REF-C v5, FROZEN in stage A ──┐
  image [9,256,256] ──► ResNet encoder                90.5 M  (86.8 % of REF-C)│
                      │      │                                                 │
                      │      ├──► cond₀            the conditioning bus        │
                      │      │                                                 │
   refcv5 grounding ──┼──► BEV tokens ×161 ─┐                                  │
   (already present)  │    agent tokens ×73 ─┼─► grounded scene set  X         │
                      │    map tokens   ×54 ─┘                                 │
                      │      │                                                 │
                      │      ▼                                                 │
                      │  128 anchors ──► 2-step truncated diffusion ──► F₀     │
                      │      │                                        [128,S,2]│
                      │      ▼                                                 │
                      │  scorer ──► S₀ ──► τ₀                          8.6 M   │
                      └────────────────────────────────────────────────────────┘
                             │                    │
              X, cond₀       │                    │  F₀ (proposal set)
                             ▼                    ▼
  ┌──────────────────── TANITLANG REASONER ───────────────────────────────────┐
  │                                                                            │
  │  ① GROUNDING FUSION      cross-attn(query=z, kv=[cond₀ ∥ BEV ∥ agent ∥ map])│
  │                          + optional DINOv3 semantic tokens (stage C)       │
  │                                                                            │
  │  ② RECURSIVE CORE  (TRM-style, weight-shared, T outer × n inner)           │
  │        for t in 1..T:                                                      │
  │            for i in 1..n:   z ← z + f_θ(z, X, y)      refine the REASONING │
  │            y ← y + g_θ(y, z)                          refine the BELIEF    │
  │        ⇒ the CoT IS the sequence z₁…z_T — natively a trace, renderable     │
  │                                                                            │
  │  ③ HALTING HEAD          q(z_t) → stop?     adaptive depth = "fast when    │
  │                                              easy, deep when crowded"      │
  │                                                                            │
  │  ④ HEADS                                                                   │
  │        goal tokens g   → strategic vocab  ─┐                               │
  │        constraint c    → tactical vocab   ─┼─ SHARED EMBEDDING TABLE       │
  │        (optional) text ← LM decoder over z ┘   (§5)                        │
  │                                                                            │
  │  ⑤ ENERGY BRIDGE   E_φ(z_T, τ_k)  for all 128 k, in ONE batched pass       │
  └────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
        S₁ = S₀ − β·E_φ(z_T, ·) + Wg·g + Wc·c        →   τ_sel
                    (β, Wg, Wc ALL zero-init)
```

### Parameter budget (ESTIMATED — the exact refcv5 split is being audited)

| block | params | note |
|---|---:|---|
| REF-C v5 encoder | **90.5 M** | MEASURED (`refc.py`, 86.8 % of 104.19 M) — frozen in stage A |
| REF-C v5 decoder + scorer + grounding | ~13.7 M | MEASURED total 104,191,577 minus encoder |
| ① grounding fusion (2 cross-attn blocks, d=512) | ~6 M | |
| ② recursive core (weight-shared, d=512, 2 blocks) | ~13 M | **shared across all T×n steps — depth is free in parameters** |
| ③ halting head | ~0.3 M | |
| ④ goal/constraint heads + shared codebook | ~2 M | |
| ⑤ energy bridge (traj encoder + bilinear) | ~4 M | |
| **TanitLang core, language-free** | **≈ 25 M** | the deployable arm |
| ⑥ optional LM decoder (≤1 B) | 0.3–1.0 B | **training-time + on-demand only, never in the control path** |

> ⚠️ **This directly answers the PI's objection** — *"it is not plausible for me why
> it's so small trainable part, in my opinion this won't work. In REF-C we had much
> more trainable parts."* The v3 draft trained ~2 M parameters and was wrong for a
> second, independent reason: PUBLISHED (`arXiv 2607.10172`, banked) reports that
> **freezing the VLM or LoRA-ing only the vision encoder significantly degrades
> performance**, with the recommendation LoRA r=32 **plus a full vision-encoder
> fine-tune**. The v3 configuration was the one measured to fail. Here the trainable
> surface is ≈25 M in stage A and **≈115 M in stage B (encoder unfrozen)** — an
> order of magnitude more than v3, and the staging is what keeps it comparable.

---

## 4. Why a recursive core and not a transformer stack

The PI asked to examine HRM and TRM. The transferable part is **not** the hierarchy.

* PUBLISHED (ARC Prize ablation of HRM): the hierarchical two-timescale structure
  contributes little; **the outer refinement loop is what carries the result.**
* PUBLISHED (TRM): a ~7 M-parameter recursive core matches or beats far larger
  models on ARC-style tasks by *iterating*, not by widening.

Mapped onto driving:

| TRM | TanitLang |
|---|---|
| `x` puzzle input | grounded scene set `X` = cond₀ ∥ BEV×161 ∥ agent×73 ∥ map×54 |
| `y` current answer | belief over the 128-anchor fan **+** goal/constraint tokens |
| `z` latent scratchpad | the CoT state — the thing that gets rendered to text |
| `n` inner / `T` outer | adaptive: the halting head sets depth per window |

⭐ **This is what makes "fast reasoning in a complex scene" a mechanism rather than
an aspiration.** The PI's example — *many pedestrians, a motorcyclist closing from
the rear, poor illumination* — is a scene where the halting head should spend more
outer steps, while an empty motorway spends one. Compute follows difficulty, and the
300–500 ms tact is met by capping `T` at the budget rather than by shrinking the model.

### ⚠️ The transfer risk, and what we already measured about it

TRM depends on a `puzzle_id` embedding; PUBLISHED ablation says replacing it with a
blank or random token yields **zero** accuracy — ARC Prize calls it *"a strong,
limiting dependency"*. In driving there is no puzzle identity: every window is new.

WP-B asked whether a **context-derived** situation code could play that role and read
no signal — but ⛔ **that result is void and is not evidence**: the features were
window-level and therefore *identical across the candidates being ranked*, so the
probe could only ever learn the marginal. Its "no signal" was a tautology. The repair
(WP-F, per-candidate features) found that hand-crafted **relational edges add nothing**
(shuffled-edge 0.4815 > real-edge 0.4704, MEASURED), which leaves the `puzzle_id`
analogue **open**: it must be a *learned* code trained end-to-end.

⛔ **And it may not be the situation classifier's output, in any form** — class
posterior, argmax, embedding, or any feature derived from them (binding PI ruling,
2026-08-03). A learned latent conditioned on the scene is admissible; the classifier's
output is not. Stated as the admissibility check: *could this code have been computed
from the situation classifier's output?* If yes, it is inadmissible until shown
otherwise.

---

## 5. The embedding space — how text, tactical and strategic tokens actually couple

The PI asked directly: *"how is the embedded space coupled, how are we combining text
tokens with our tactical vocab and strategic vocab?"* Concretely:

REF-C already carries the machinery. Its selection logits are the sum of a base score
and a family of **zero-initialised** `Linear(vocab → 128)` prior ports —
`maneuver_to_anchor`, `lat_to_anchor`, `lon_to_anchor`, `lan_gate` — and there is
already a language port, `lan_emb` / `lan_to_cond`, currently unused.

> ⭐ **The coupling is one embedding table with two consumers.**
>
> The tactical and strategic vocabularies are given entries in the LM's token
> vocabulary as **special tokens** — `<TAC:follow_lead>`, `<TAC:yield>`,
> `<STR:left_at_junction>` — whose embedding rows are **tied** to the planner's
> `maneuver_emb` / goal-embedding rows. One `nn.Embedding`, read by the LM as token
> embeddings and by the planner's prior ports as conditioning vectors.
>
> ⇒ **Emitting the token *is* conditioning the planner.** There is no translation
> layer to drift, no adapter to mis-align, and the "shared embedding space" the PI
> asked for is literal rather than metaphorical.

Text tokens then combine with tactical/strategic tokens the way any tokens combine —
they are the same sequence. A rendered CoT that ends in `<TAC:yield>` has, by
construction, pushed the `lat_to_anchor` port toward yielding anchors.

⚠️ **Directionality matters for the goal input.** Binding ruling (2026-08-03): a goal
signal is admissible, but it may not carry the situation classifier's output. The
shared table is safe *because the tokens are the planner's own vocabulary*, not the
classifier's posterior. State for any arm what the goal is computed from — and if a
shared trunk feeds both the goal path and the situation path, say so and justify why
it is not a back door.

---

## 6. Language: with, without, and why it is not in the control path

Three arms, and the PI asked for both directions to be developed:

| arm | `z` | text | tact |
|---|---|---|---|
| **TanitLang-0** *(language-free)* | latent only | none | the deployable arm; meets 300–500 ms |
| **TanitLang-T** *(language-driven)* | latent + LM | LM decodes `z₁…z_T` to an Alpamayo-style trace; text also accepted as **input** (nav command, query) | LM head is optional at runtime |
| **TanitLang-C** *(combined)* | latent, LM-supervised | text supervision during training; renderer on demand | deployable |

> ⭐ **The design decision worth stating crisply: language enters as a REGULARISER on
> `z` and an OPTIONAL renderer — never as a link in the control chain.**
>
> The control path is `image → cond₀ → z → E_φ → S₁`. The LM sits *beside* `z`, reading
> it and writing gradients into it. Skip the LM at deployment and the car still drives,
> with the same `z`. This is what buys the latency budget **and** what keeps the
> vision-only inference ruling intact.

**Yes, it generates Alpamayo-style CoT traces.** That was the requirement and the
recursive core is what makes it natural: `z₁…z_T` *is* a sequence of reasoning states,
so the renderer is a decoder over a trace, not a caption over a single vector.

⚠️ **What the trace can and cannot say — MEASURED, and it constrains the renderer.**
WP-A: the 11 structured CoT fields we can derive cover **27.2 %** of the Alpamayo text
variance (shuffled control 0.8 %). ⇒ **72.8 % of what the text says is not recoverable
from the fields**, so a renderer driven by fields alone would be reciting a template.
`z` must feed the renderer **directly**. WP-G: the corpus has **no `motorcycle` and no
`bicycle` class** (both collapse into `rider`), `occ` is constant-zero, and there is no
illumination field — so the PI's *"motorcyclist from the rear in bad illumination"*
cannot be *labelled* from our corpus at all. Those attributes can only arrive by
**distillation from a stronger backbone**, which is what WP-I is measuring tonight.

---

## 7. Injecting physical-world semantics — the PI's question, and the three routes

> *"how can we inject these semantics in the latent, maybe by using in some stages of
> the training a strong pretrained vision backbone which transforms the image into
> these semantics, but it is open how to intrinsically model the dynamic interaction
> between the agents themselves and the geometry and the agents."*

Two questions, and they have different answers.

**(a) Static semantics — what is in the scene.** Route: an auxiliary distillation head
on `cond₀` predicting a frozen teacher's pooled features (DINOv3 ViT-L/16, already on
local disk, zero HF quota). ⛔ **Gated on evidence, not enthusiasm**: WP-I is measuring
whether DINOv3 actually carries driving semantics our trunk does not, against a matched
pixel floor **and** a random-weight architecture floor. If the trained trunk already
matches the teacher, this route is closed and we say so.

**(b) Dynamic interaction — agents with each other and with the geometry.** This is the
harder half and the honest status is: **open**.
* Hand-crafted relational edges **do not work** — MEASURED (WP-F): shuffled-edge 0.4815
  beat real-edge 0.4704, i.e. the "relational" features were fit noise.
* What is untested is a **learned** interaction: attention among the agent tokens
  refcv5 already carries (×73), inside the recursive core, so interactions are refined
  over `T` steps rather than read off once.
* ⭐ The cheap discriminating experiment is **not another probe** — it is an
  **intervention**: perturb one agent token (remove it, displace it) and measure the
  change in `E_φ(z, τ_k)` across the fan. A model that has learned interaction must
  change its energy landscape *more* when the perturbed agent is on a collision course
  than when it is behind. A model that has not will change uniformly. That is a
  causal test with a built-in null, and it does not need new labels.

---

## 8. Staged build — and what each stage is allowed to claim

| stage | trains | frozen | claim admissible on completion |
|---|---|---|---|
| **A** bridge only | `E_φ`, `β`, heads (≈25 M) | REF-C entire | *"a reasoning state can re-rank the fan"* — needs `INF_* > 0` **and** shuffled-z degradation |
| **B** joint | + REF-C decoder/scorer (≈115 M) | encoder | *"the bridge improves selection"* — needs the H-TL-1 headroom to close, four-family, replicate arm |
| **C** semantics | + distillation head, encoder unfrozen | — | *"injected semantics help"* — needs WP-I to have said the teacher carries something first |
| **D** language | + LM decoder | — | *"the CoT is faithful"* — needs `CON_text` **and** the τ→z residual test |

⛔ **No stage may quote a number without its tier.** T0 is a world-model/diagnostic
tier and is never "driving performance"; **T1 (the model conditioned on its own
actions) is the primary tier for any capability claim.** And T1 as we run it is
**self-action open loop** — a planner feeding its own predictor is still open loop
(binding ruling); closed loop needs AlpaSim or a vehicle.

⛔ **Every eval reports four metric families, not ADE**: longitudinal (target speed,
headway/TTC), lateral (heading, curvature, yaw-rate, cross-track), tactical
(manoeuvre-decision quality, goal/anchor selection), strategic (route/goal setting) —
each per family with its own paired episode-cluster CI, never pooled into one score.

---

## 9. Why this is worth building — the headroom is MEASURED

H-TL-1 precondition, MEASURED (`RESULT_HTL1_HEADROOM.md`, T0, non-parity pilot,
n=120 windows, 128 anchors):

```
  SELECTED  0.654 m        what REF-C's scorer ships
  ORACLE    0.250 m        the best anchor the fan already contains
  headroom  0.404 m        CI95 [+0.267, +0.570]  SEPARATED
  best-anchor rank:  median 3/128 · top-5 75.0 % · top-1 32.5 %
```

⭐ **The fan already contains a much better trajectory than the scorer picks, and the
right one is in the top 5 three times out of four.** That is precisely a **re-ranking**
problem — which is exactly what the Energy Bridge is, and it is why the bridge writes
into *selection* rather than into trajectory regression.

⚠️ **Scope it honestly.** T0, non-parity pilot corpus, one seed, n=120 windows. A
separated CI on a one-seed arm is **necessary and not sufficient** (the `A0b_replicate`
result). This justifies *building the experiment*; it does not yet claim the bridge
will close the gap.

---

## 10. What would falsify this design

Pre-registered, both outcomes committed in advance:

1. **`INF_kl` stays ≈ 0 after stage A** ⇒ the energy is constant in `z`; the bridge is
   a caption generator. **Design refuted**, and the fallback is direct conditioning of
   the denoiser rather than the scorer.
2. **`INF` is large but shuffled-z does *not* degrade `CON_rank`** ⇒ `z` is scene-blind
   noise that merely perturbs selection. **Worse than refuted** — it would be a random
   re-ranker that could still improve ADE by luck, which is why the shuffled control is
   mandatory rather than optional.
3. **The τ→z probe explains nearly all of `z`** ⇒ the CoT is an echo of the answer.
   Design refuted on the consistency side.
4. **WP-I says the trunk already matches DINOv3** ⇒ §7(a) is closed; semantics must
   come from language supervision or from the interaction route, not from distillation.
5. **The replicate arm reproduces the bridge's effect with zero levers moved** ⇒ the
   effect is rig noise. This is the control that the programme learned the hard way to
   run *first*, not last.

---

## 11. Open items being measured or audited right now

| item | stream | what it decides |
|---|---|---|
| WP-I four-arm semantic floor (pixels / random-trunk / trunk / DINOv3) | running | §7(a): is distillation a live route at all |
| refcv5 surface audit — exact `cond` dims, grounding token counts, prior ports, param split, attachment points | running | turns §3's ESTIMATED budget into MEASURED, and names the exact zero-init port to reuse |
| agent-join extension beyond 15 episodes | running | the power ceiling on every probe above |
| verifier-guided decoding / CoT-faithfulness / energy-consistency literature | running | whether `E_φ`'s training objective should be contrastive, ranking, or a process reward |

---

### Evidence classes used above

`MEASURED (ours + artifact path)` — H-TL-1 headroom, WP-A/F/G, REF-C param split, the
`A0b_replicate` false-positive rate, the flagship route-head echo.
`PUBLISHED (cited, banked)` — arXiv 2607.10172 (VLM freezing degrades), HRM/TRM and the
ARC Prize ablation.
`ESTIMATED` — the TanitLang parameter budget (pending the refcv5 audit).
`HYPOTHESIS` — the Energy Bridge itself, the halting-head compute-follows-difficulty
claim, and the agent-perturbation interaction test. **None of these is a result.**

---

# ⛔ AMENDMENT 1 (2026-09-06) — four corrections from the refcv5 source audit

`AUDIT_REFCV5_SURFACE.md` read the actual tree with `file:line` citations and
same-breath controls. **It refutes four things asserted above.** Each was
`INHERITED` — carried from an earlier conversation turn rather than read from
source — and each is corrected here rather than quietly edited, because the
design that follows depends on them.

### C1 ⛔ The grounding token counts in §3 are WRONG. There is no BEV or map token.

§3's diagram shows `BEV tokens ×161 / agent tokens ×73 / map tokens ×54`.
**MEASURED:** refcv5's grounding is **agent tokens only** — `agent_tok [B, 100, 384]`
(`refc.py:1135`-region wiring, head at `refc_agents.py:158` `d_model=256`, projected to
384), fused by a **gated cross-attention** with `agent_gate` zero-init
(`refc.py:1217-1230`), behind **two flags that must both agree** (`AgentSeamConfig.enable`
**and** `DecoderConfig.cross_agent`), both defaulting `False`.

> **There is no BEV token, no LiDAR token, no map token and no lane-graph token in the
> model** (`AUDIT §2b`, two probes with controls; `bev_raster` is imported only for the
> label-side enum and FOV mask).

⚠️ **What survives and what does not.** The PI's correction — *"we addressed this in
refcv5"* — **stands**: refcv5 *does* add environment grounding. What was wrong was **my
inventory of it**. The design's §7(b) "learned interaction over the agent tokens
refcv5 already carries" is therefore **strengthened**, not weakened: agent tokens are
exactly what exists, and they are the only grounding the reasoner can read.
⚠️ And §7's map/BEV-flavoured framing must not be quoted: **a strategic/route topology
still has no source in this model**, consistent with the standing programme finding that
PhysicalAI-AV carries no map, lane graph, junction or traffic-light feature.

### C2 ⛔ `lan_*` is **Lane-Anchored Navigation**, not language. There is no language port.

§5 and every earlier draft called `lan_emb` / `lan_to_cond` *"an existing zero-init
language port"*. **It is not a language port.** `stack/tanitad/data/lan.py:1` defines LAN
as *Lane-Anchored Navigation*; `lan_to_cond` is `Linear(64, 384)`, zero-init
(`refc.py:1399-1401`), built only under `cfg.graft_lan`.

> **MEASURED (`AUDIT §4`, two probes with controls): no tokenizer, no embedding table and
> no text head exist anywhere in the model.** The only `language` symbol in the tree is a
> data-lake schema field.

⇒ §5's shared-codebook mechanism is **unchanged as a design** — it never depended on
`lan_*` — but it must be built **from nothing**, not grafted onto an existing port. That
is a real cost the design under-priced, and it moves the text arm strictly later than the
language-free arm.

### C3 ⚠️ `maneuver_to_anchor` — my measurement and the audit's are BOTH right, in different scopes

I measured `decoder.maneuver_to_anchor [128,5]` at **L2 6.375, max|w| 0.724** in
`refc-base-30k`. The audit reports it is **`None` on every registered arm**, because
`factored_maneuver=True` constructs `lat_to_anchor` / `lon_to_anchor` instead.

Both are true and they are **not about the same object**: mine is the **base** checkpoint,
the audit's is the **registered refcv5 arms**. ⛔ Stated so the number does not drift out
of its scope — that drift is the failure class this campaign has hit nine times.

⇒ **The claim that survives** is the weaker, sufficient one: *the zero-init prior-port
pattern is not inert on this architecture — a trained arm moves such a port off zero.*
⇒ **The claim that does NOT survive** is *"`maneuver_to_anchor` is the free live port to
reuse in refcv5"*. It is not; it does not exist there.

### C4 ⚠️ `d_model` is **384**, not the 512 the §3 budget assumed

`DecoderConfig.d = 384` (`refc.py:397`), pinned for every registered arm. The audit's
**MEASURED** budget at 256×640 replaces §3's ESTIMATED table's baseline:

| arm | params |
|---|---:|
| refcv4b base | **107,058,488** |
| + WP-4 | **108,246,216** |
| + WP-4 + WP-6 | **114,134,113** |
| the whole hierarchy cascade | **2,129,752** |

⭐ **A ≤1B reasoner is ~470× the entire hierarchy cascade.** That is the quantitative form
of the PI's objection to the v3 draft, and it argues for the staged budget in §8 —
≈25 M trainable in stage A, ≈115 M in stage B — rather than for hanging a billion
parameters off a two-million-parameter hierarchy.

### ⭐ C5 — a gift the audit found, and it changes the FIRST attachment point

§8 stage A did not name a specific port. The audit ranks six and flags **`tac_goal_head`**
(`refc_v3.py:761`, zeroed `:769-772`, `Linear(d_tac=512, k·GOAL_DIMS=12)`) as the best
*first measured* attachment for one reason that outranks convenience:

> **it is the only write port whose ANTI-ECHO INSTRUMENT ALREADY EXISTS** — `echo_ratio`
> is emitted every step.

That is precisely the quantity §2 and `H-TLANG-ECHO-1` need, already wired and already
producing a number. ⇒ **Stage A attaches at `tac_goal_head` first**, and
`echo_residual()` in `stack/tanitad/instruments/cot_influence.py` is validated against
the existing `echo_ratio` before it is trusted on anything new.
(Rank 1 by *least disturbance* is the `hierarchy_hook` block reusing `tgt_film`'s
zero-init — kept as the fallback if `tac_goal_head` proves too narrow at `GOAL_DIMS=12`.)

### ⛔⛔ C6 — THE BLOCKER, escalated rather than written down and left

**refcv5's model-side seams live in the WORKING TREE and are NOT IN `HEAD`.** MEASURED on
HEAD's blob of `refc.py` (`0e6103e5`), with same-breath controls that must read non-zero:

| symbol | HEAD | worktree |
|---|---:|---:|
| `agent_tok`, `control_head`, `cross_agent`, `time_mlp`, `sampler` | **0, 0, 0, 0, 0** | 19, 12, 11, 6, 52 |
| `lan_to_cond` / `maneuver_to_anchor` — **CONTROLS** | 7 / 4 | 7 / 5 |

HEAD's trainer already calls `_pin_refcv5_seams` (3×) and `AgentSeamConfig` (2×), while
`assert_seams_are_built` appears **0×** in HEAD. That is the false-provenance state the
guard was written to prevent: **a run that stamps `sampler="ddim"` on a model with no
denoiser.** A fresh clone of HEAD cannot build a refcv5 model at all.

⇒ **No TanitLang arm may attach until this is committed.** ⛔ I am not committing it: it
is the Architecture & Inference stream's in-progress model code, and sweeping another
stream's uncommitted work into my commit is the exact failure `CLAUDE.md`'s git-hygiene
rule exists to stop. **Escalated to the Master Mind as a blocking integration item.**
Separately: `refc_selector.py` (WP-7, 4,530,444 params) *is* committed and is wired to
nothing but an import probe.

---

# ⛔ AMENDMENT 2 (2026-09-06, same turn) — C5 above was WRONG, and the real find is bigger

I wrote in C5 that `echo_ratio` is *"precisely the quantity §2 and `H-TLANG-ECHO-1`
need"*. **It is not.** Read at source rather than from the audit's summary:

```
  refc_v3.py:1086     out["echo_ratio"] = cache["g_tac_delta_absmean"] / eb
  refc_v3_train.py:1453  "how much goal VISION contributes beyond the ego
                          self-extrapolation"
```

and the code's own warning, verbatim: *"It is a RATIO OF MAGNITUDES, not a verdict: a
large delta can still be wrong, and a small one can still be right if `ha0_ext` is
already near-optimal."*

⇒ `echo_ratio` measures **the magnitude of a path's contribution over a baseline**. That
is an **INFLUENCE**-family quantity — the same shape as `‖S₁ − S₀‖` — **not** the
CoT-reconstructibility quantity `echo_residual(z, τ)` computes. The two are complementary
and I conflated them.

**What C5's conclusion becomes.** `tac_goal_head` is still the right first attachment,
but for the corrected reason: it is the port whose **INFLUENCE** readout already exists
and is logged every step, so stage A's *"did the reasoner move anything"* question has a
live tell from step 1. The **consistency/echo** side still needs `echo_residual`, which
has no existing counterpart.

### ⭐ C7 — and the audit's pointer led somewhere better than the port

`echo_ratio`'s docstring names its verdict instrument: `tanitad.eval.echo_gate`. Reading
it changes the build plan, because **most of §2's control table already exists and is
tested**:

| §2 control | already implemented |
|---|---|
| constant-z floor | `echo_gate.constant_only_reference` |
| raw-input floor | `echo_gate.raw_pixel_floor` |
| both-directions intervention | **`echo_gate.ego_intervention_test`** — *"perturbing ego must move the output (else the channels are dead) AND varying the scene must move the output (else the arm is ECHOING). An arm that passes only the first is the failure this module exists to name."* |
| a gate that actually raises | `echo_gate.assert_not_echoing` |
| the bar to clear | `REQUIRED_REFERENCES = ("ha", "ha0_ext")` — ⛔ **not `ha0`**: MEASURED on refcv3, `ha` **0.2996** BEATS the model's **0.4799** @30k while `ha0` **0.6723** loses to it, so an `ha0`-only panel certifies an arm the trivial control already beats by 2× |

⇒ **`cot_influence.py` must INTEGRATE with `echo_gate`, not run beside it.** I was one
step from building a parallel instrument stack with its own conventions — which is the
duplication the operating standard's *"escalate integration"* rule exists to prevent, and
it would have produced two anti-echo verdicts that could disagree.

**Concretely, the revised instrument plan:**
1. `cot_influence` keeps only what is genuinely new — `rerank_logits`, `influence`,
   `consistency`, `echo_residual`, and the roll-based `z` control.
2. Its floors are **imported** from `echo_gate` (`constant_only_reference`,
   `raw_pixel_floor`), not re-implemented.
3. The TanitLang arm is certified by `echo_gate.assert_not_echoing` against
   `("ha", "ha0_ext")` like every other arm, with `INF_*`/`CON_*` as **additional**
   reasoner-specific readouts, never as a substitute bar.
4. ⭐ **§7(b)'s agent-perturbation interaction test is `ego_intervention_test` with the
   perturbed channel swapped from ego to an agent token.** It already has the
   both-directions logic and the dead-channel failure mode named. That is a much cheaper
   path to the PI's *"how do we model the dynamic interaction between the agents"*
   question than the new instrument I proposed.
5. ⛔ And it inherits the mandatory **deliberate-regression arm**: an arm *designed* to
   echo must be FAILED by the gate, or a PASS on the real arm means nothing.

⚠️ **A note on how both errors happened, because it is the same one twice.** C5 quoted the
audit's *summary* of `echo_ratio`; §3's token counts quoted an earlier turn's *summary* of
refcv5. Both were `INHERITED` presented as settled. The audit was right to exist and right
in what it said — I was wrong to build on a summary of it instead of on the source it
cited. **Evidence class is not a label you attach after the fact; it is a constraint on
what you may write next.**

---

# ⛔⛔ AMENDMENT 3 (2026-09-06) — the frontier already refuted §2's influence metric as SUFFICIENT

Source: `LIT_COT_INFLUENCE_CONSISTENCY.md` (40 primaries banked, library 447→487).
Evidence class **PUBLISHED (cited, banked)** where the PDF body was read; the lit pass marks
26 of 40 `PUBLISHED-ABSTRACT` and **no design change below rests on one of those**.

### A1 ⛔ A large `‖S₁ − S₀‖` proves the channel is WIRED, not that the CoT is MEANINGFUL

§2 defines influence as the guidance delta and treats it as the causal contribution.
**MEASURED by others, on this exact question** (`2606.12706`, VLADriveBench, PDF body read):
ORION's CoT shortened its 3 s prediction on **81.9 %** of steps, **−0.437 m**. Then:

| arm | effect |
|---|---|
| the real CoT | 81.9 % of steps, −0.437 m |
| **the same CoT with its words SHUFFLED** | **81.4 %, −0.429 m** — indistinguishable |
| **the CoT replaced by repeated copies of the token `"depicts"`** | **90.4 %, −0.846 m ≈ 2× STRONGER** |

⇒ **a semantically empty prior produced twice the influence of the real one.** Every
`INF_sel` / `INF_geo` / `INF_kl` in §2 would have reported that as a success.

**What changes.** §2's controls (shuffled-z, constant-z floor) were the right *idea* but the
wrong *specification*: `constant_energy_floor` uses the **mean** energy, which is
norm-**reduced**. The published failure mode is a **norm-preserved** null. ⇒ the control
table gains a mandatory arm:

> **SEMANTIC-NULL: a norm-matched constant prior, plus a permuted prior.** The real `z`
> must beat BOTH. An arm that does not exceed its own norm-matched null has demonstrated
> wiring and nothing else.

This is the cheapest experiment in the whole plan and it is now a **precondition** for
reading any influence number, not a follow-up. Landed in
`stack/tanitad/instruments/cot_influence.py::norm_matched_null`.

### A2 ⚠️ The influence/consistency tension is a PUBLISHED trade-off, not a defect in our design

Same paper, verbatim: *"Neither design satisfies both desiderata."*

⇒ §1's framing — *"the Energy Bridge dissolves the tension"* — is **too strong** and I am
withdrawing it as stated. The honest and still-valuable framing: the field has two
*discrete* designs, each satisfying one desideratum; we have **one architecture with a
continuous dial (`β`)** and can therefore **measure the frontier between them** — which is
the experiment their two-model conclusion asks for and their setup cannot run. That is a
better contribution than a dissolution claim we cannot yet support.

### A3 ⛔ Do NOT supervise the rationale from the logged future trajectory

`2608.01755` names this **trajectory anchoring bias** and measures lower causal faithfulness
and more hallucination, concentrated in exactly the hard causal scenarios we care about.
⚠️ **LCDrive (`2512.10226`) cold-starts precisely this way, on PhysicalAI-AV** — our corpus.
It is the recipe we would most naturally have copied.

⇒ §8 stage D is amended: the text arm is supervised from Alpamayo reasoning traces and from
the **energy's own ranking**, never from the logged future ego path. Same family as the
programme's binding rule that **labels may use ego but inference is vision-only** — here it
is the *rationale* that must not be built from the answer.

### A4 ⛔ Every consistency number needs a SEED-REPLICATE floor, and must be per-family

`2605.17268` ran **300 Alpamayo-R1 inferences on our corpus** and measured an **88 %
inconsistency rate from changing the random seed alone**. Split by family:

| family | consistency |
|---|---|
| lateral | **100 %** |
| longitudinal | **62–69 %** |
| — of all mismatches, **64 % are "I said stop" and then kept going** | |

⇒ two consequences, both binding. **(i)** `CON_rank` / `CON_gap` / `CON_text` are
uninterpretable without a same-seed and different-seed replicate — the third variance the
programme already documented (`H-ESTIM-SEED-1`, and the inference-seed floor on stochastic
planners). **(ii)** ⛔ **A pooled consistency number would have hidden this entirely**: the
lateral channel is perfect and the longitudinal one is broken, which is our own
88.7 %-longitudinal defect reappearing in the rationale channel. Consistency joins the
four-family rule — never one score.

### A5 ⚠️ Our novelty is NARROWER than §1 implies, and it is the measurement

* **AnchorVLA (`2607.03182`)** already publishes **anchors as the reasoning↔action interface**.
* **LCDrive (`2512.10226`)** already publishes **CoT emitted in the action vocabulary** — §5's
  shared-codebook idea is not new either.

⛔ Neither measures **any** influence or consistency quantity. ⇒ **our genuine first is the
MEASUREMENT, not the interface**: influence and consistency as two readings of one
distribution over the same anchor fan, with no judge, no parser and no LLM anywhere in the
measurement loop. §10's falsification list is the contribution; §3's diagram is not.

### ⭐ A6 — the guidance delta already exists in the code. VERIFIED AT SOURCE, not from the summary.

`refc.py` materialises `conf0` — the classifier surface **before** priors — at the
prefilter branch (`conf0, offset = self._decode(kv, cond, x0, 0, agent_tok, agent_pad)`,
with the very next block commented *"priors on the CLASSIFIER surface (unchanged
semantics)"*). ⇒ **`Δ = conf − conf0` is the per-anchor guidance delta, available as one
subtraction — no dual pass is needed for the anchor side.**

⚠️ I checked this in `refc.py` myself rather than taking the lit agent's report, because
twice this turn I built on a summary of a source instead of the source (§Amendment 1 C1,
§Amendment 2). The claim is confirmed.

### ⭐ A7 — a zero-cost non-degeneracy screen to run BEFORE any eval GPU-hour

`2410.08146` (PAV): a process signal is informative iff its **variance across the candidates
it ranks** is large, under a scorer *complementary* to the one being scored.

⇒ before training anything: compute `Var_a[prior]` over the anchors and its rank-correlation
with `conf0`. **Near-flat variance, or rank-identical to `conf0`, means the port is inert by
construction** and no amount of training will make it matter. Costs one forward pass.

---

## Consolidated: what §2's control table looks like after Amendments 1–3

| control | must read | status |
|---|---|---|
| `β = 0` identity | `INF_* = 0` exactly | ✅ implemented + tested |
| constant-z floor (norm-reduced) | re-ranking collapses | ✅ implemented + tested |
| **SEMANTIC-NULL: norm-matched constant prior** | **real `z` must BEAT it** | ⭐ **NEW (A1)** — the published failure mode |
| **permuted prior** | real `z` must beat it | ⭐ **NEW (A1)** |
| shuffled-z (roll, B ≥ 2) | `E` rises, `CON_rank` degrades | ✅ implemented + tested |
| raw-input floor | arm must beat pooled pixels | reuse `echo_gate.raw_pixel_floor` (Amendment 2) |
| **seed replicate — same seed AND different seed** | bounds the 88 % inconsistency floor | ⭐ **NEW (A4)** |
| `echo_gate.assert_not_echoing` vs `("ha", "ha0_ext")` | the arm clears its own kinematics | reuse (Amendment 2) |
| deliberate-regression arm (designed to echo) | **must FAIL** the gate | reuse (Amendment 2) |
| **non-degeneracy screen `Var_a[prior]`** | non-flat, not rank-identical to `conf0` | ⭐ **NEW (A7)**, costs one forward pass |

---

# ⭐ AMENDMENT 4 (2026-09-06) — §7(a) is NARROWED by measurement: distillation targets the NEAR band only

`RESULT_WPL_POOLING_KERNEL.md` asked whether the trunk's VRU deficit is a **pooling**
artifact, a **linearity** artifact, or real. Four reads of the same [704, 8, 8] map, same
windows, same folds, same nulls; every trunk arm passed its own positive control
(PC-TRUNK 0.86–0.96).

| arm | `vru<20m` adj (p) | `vru<60m` adj (p) |
|---|---|---|
| trunk 2×4 linear | +0.0792 (0.155) | +0.0327 (0.185) |
| **trunk 8×8 linear** (full map) | **+0.0045 (0.495)** | +0.0078 (0.420) |
| trunk 2×4 rbf | +0.0374 (0.305) | **+0.0893 (0.040)** |
| trunk 8×8 rbf | +0.0558 (0.235) | **+0.0585 (0.020)** |
| dinov3 2×4 linear | **+0.1935 (0.000)** | +0.0501 (0.025) |

1. ⛔ **Spatial pooling is NOT the bottleneck.** The **full 8×8 map with zero pooling loss
   is the WEAKEST arm on both targets** (p 0.495 / 0.420). Eight times the dimensions on
   ~1,900 rows buys variance, not signal.
2. ⭐ **On the FAR band a nonlinear read rescues the trunk** — RBF p **0.040** / **0.020**
   where both linear arms fail, and null-adjusted the RBF trunk (+0.0893) sits *above* a
   linear DINOv3 (+0.0501). The earlier linear negative was not a statement about the
   representation, exactly as the one-way linear-probe rule warns.
3. ⛔ **On the NEAR band nothing rescues it** — best trunk p **0.155**, DINOv3 p **0.000**
   with double the margin.

⇒ **§7(a) is narrowed, not withdrawn.** The distillation head targets **near-agent
semantics** — the safety-critical band. **Far-agent presence is already in the trunk
nonlinearly**, so the recursive core's own MLP reaches it for free and a teacher would be
paying twice for it. That is a smaller, better-aimed head than the WP-I v2 result implied.

⚠️ **And the run's own printed verdict was wrong**, which is worth carrying here because
the design would otherwise inherit it: the script computed `rescued` on the near band alone
and announced *"NOT RESCUED"* globally while the far band was clearing in the same table.
Fixed in code (per-target verdict + multiplicity note). ⚠️ Multiplicity is real: 10 tests at
α 0.05, ≈0.5 expected false positives, p resolution 0.005 — **the far-band rescue is a lead
awaiting a replicate**, not a settled result.

---

# ⛔ AMENDMENT 5 (2026-09-06) — the influence metric must be read on the OUTCOME axis, and §2's scale is a hard constraint

WP-M ran the Energy Bridge's exact mechanism on a real trained model, by ablating REF-C's
own `decoder.maneuver_to_anchor` and differencing the two passes: `S0` = port zeroed (which
is literally what the zero-init arm produces at step 0), `S1` = port as trained, so
`Δ = S1 − S0` is the per-anchor guidance delta by construction. Feeding `energy = −Δ` at
`β = 1` reproduces `S1` **identically** (asserted in the run), so the banked instrument
measures the real model with no special-casing. MEASURED, n = 1,360 windows / 34 episodes.

### A5.1 ⭐ The port mechanism is NOT inert — §2's first premise holds

```
  ZERO-PORT IDENTITY   flip 0.0000   kl 0.000000   geo 0.0000 m     <- exact
  the TRAINED port     flip 0.2324   kl 0.187789   geo 0.2022 m
```

A trained zero-init port moves the selection in **23.2 %** of windows and displaces the
chosen trajectory by **0.20 m**. The identity is exact, so that effect is attributable to
the port and to nothing else. **The channel TanitLang writes through works.**

### A5.2 ⛔ But `INF_kl` cannot decide whether a prior is MEANINGFUL — my own screen was on the wrong axis

```
  real            kl 0.187789
  norm-matched    kl 0.406544    <- a RANDOM prior, same per-window spread
  permuted prior  kl 0.193963    <- a REAL prior, attached to the wrong window
```

REF-C's genuinely trained prior **loses the KL contest to random noise, by more than 2×.**

⚠️ That is not evidence the port is empty — it is evidence that **KL is the wrong
statistic**. At matched spread a random direction generically diverges further than a
structured one, so magnitude cannot separate *moved meaningfully* from *moved hard*.
Amendment 3 §A1 imported the VLADriveBench failure mode correctly and then
**operationalised it incorrectly**: their finding is about the **behavioural effect**, and I
built a **magnitude** screen.

⇒ **`outcome_null_screen` added** to `stack/tanitad/instruments/cot_influence.py`: every arm
re-ranks, selects its argmax, and is scored by the **metric it actually achieved**.
`semantic_null_screen` is kept and re-scoped in its own docstring — it can show a prior is
**inert**; it can never show a prior is **empty**. Suite 29/29 green, including a test that
pins exactly the distinction the KL screen cannot make.

### A5.3 ⭐⭐ THE HARD DESIGN CONSTRAINT — β·E must be comparable to the scorer's own spread

```
  sd_a[Δ]           1.457     the port's spread across the 128 anchors
  sd[S0]           17.907     the base scorer's spread
  ratio              ~8 %
  Spearman(Δ, S0)   0.096     <- nearly orthogonal: NEW ordering, not an echo
```

⚠️ **At 8 % of the scorer's scale, consistency is not merely weak — it is UNMEASURABLE.**
`consistency(energy, sel)` reads mean rank **64.40 of 128 against a chance of 64.50**, and
top-1 **0.0000**.

⛔ That is **not** a finding that REF-C is inconsistent. The metric asks whether the energy
*alone* ranks the chosen anchor first, and a small additive prior cannot, by arithmetic, no
matter what it knows. Reporting it as a REF-C defect would have been a scope error of
exactly the kind this campaign keeps producing — so it is reported as a **calibration**.

⇒ **What it actually calibrates, and it is the most useful number WP-M produced:**
`CON_rank` reads chance whenever `β·E` is small relative to `sd[S0]`. **§2's consistency half
is therefore only measurable if the reasoner is given real authority over the decision** —
either `β` large enough that `β·sd_a[E] ≈ sd[S0]`, or an energy that *replaces* rather than
*nudges* the scorer. **A timid zero-init port that stays timid produces an unfalsifiable
consistency claim**, which is the failure mode §2 was written to prevent.

⭐ The Spearman of **0.096** is the encouraging half: the trained port is nearly orthogonal
to the base scorer — it contributes *new* ordering rather than echoing `S0`. That is exactly
the property the Energy Bridge needs, and it is already present in the mechanism.

### A5.4 The port's effect on the metric, and the headroom confirmed

```
  base selection    0.6636 m        guided selection  0.6521 m
  fan's best anchor 0.2315 m        port effect      -0.0115 m
  paired episode-cluster CI95  [-0.0417, +0.0184]  overlaps 0
```

⇒ **No measurable ADE benefit either way.** And the H-TL-1 headroom is independently
confirmed on this corpus and this join: the fan already contains an anchor at **0.2315 m**
while the model ships **0.65 m**. The selection problem is real, and REF-C's existing prior
is not solving it — which is the gap TanitLang exists to close.


---

# !!! AMENDMENT 6 (2026-09-06) -- A5.3's control was broken; the channel is scene-dependent

Amendment 5 imported WP-M's finding that ~89 % of REF-C's port benefit is generic. **That
rested on a control that was 97.5 % a no-op** -- `roll_control` mispairs by index, and the
rows are grouped by episode, so the "different window" prior was almost always the same clip
one timestep away. Retracted; full record in `RESULT_WPN_PORT_CONTRASTS.md`.

With a cross-episode control and a paired episode-cluster bootstrap:

    real - cross-episode   -0.0254 m   CI [-0.0425, -0.0093]   SEPARATED
    real - base            -0.0113 m   CI [-0.0416, +0.0175]   overlaps 0
    norm-matched - base    +0.0838 m   CI [+0.0600, +0.1087]   SEPARATED

**What this changes for the design:**

* **Section 1-2's premise is STRENGTHENED.** The re-ranking channel is demonstrably
  scene-dependent: swapping in another episode's prior degrades the selection by a separated
  margin. A5.3's pessimistic reading is withdrawn.
* **The bar for TanitLang is the CROSS-EPISODE control**, and it is now a real bar rather
  than a no-op. A 5-class manoeuvre prior already moves it by 0.0254 m; a reasoner must beat
  that, not merely beat "no port".
* !! **Unchanged:** "the port improves on no prior at all" is still unestablished, and
  A5.3's scale constraint (`beta*E` must be comparable to `sd[S0]`, or CON_rank reads chance)
  stands untouched -- it was measured, not inferred from the broken control.
* **Instrument fixed:** `roll_control`'s docstring now states that a roll is the wrong
  control whenever the batch has a grouping, and that the partner must be drawn ACROSS
  GROUPS with an assertion that no same-group pair survives.


---

# * AMENDMENT 7 (2026-09-06) -- A5.3's constraint is now a NUMBER, and the frontier is measured

`RESULT_WPP_FRONTIER.md` swept `beta` in `S1 = S0 - beta*E` on the banked tensors (zero GPU;
`beta = 1` reproduces the shipped model exactly, asserted).

| beta | authority | CON rank | ADE (m) |
|---:|---:|---:|---:|
| 0.0 | 0 % | 75.80 | 0.6636 |
| **1.0** | **8.1 %** | 64.40 *(chance 64.5)* | **0.6521** |
| **2.0** | **16.3 %** | **53.66** | **0.6583** |
| 5.0 | 40.7 % | 14.33 | **1.4864** |
| 160.0 | 1302 % | **1.15** | **7.6221** |

**A5.3 said `beta*E` must be comparable to `sd[S0]` or consistency is unmeasurable. The
number is now known:**

* ⭐ **The usable window is beta in [1, 2] -- authority 8-16 %.** At 16.3 % consistency
  becomes readable (64.40 -> 53.66) at **no ADE cost** (0.6583 vs base 0.6636). **Stage A
  starts there.**
* ⛔ **Beyond it, performance collapses long before consistency becomes good.** 40 %
  authority already doubles ADE; near-perfect consistency (rank 1.15) costs **11.5x** the
  base ADE. Near-perfect consistency is purchasable, and it costs the car.
* ⭐ **The control that matters behaves:** the real-vs-cross-episode gap grows ~3x across the
  usable regime (+0.0252 -> +0.0742), so the extra authority buys understanding there. Above
  beta = 5 the gap oscillates in SIGN and no trend is readable.

=> **Raising beta ALONE cannot deliver the PI's requirement. The energy must get BETTER, not
LOUDER** -- which is exactly why the target is re-ranking against a fan that already holds a
0.2315 m anchor while the model ships 0.65 m.

!! **New reporting rule, and this table is why:** any `CON_rank` is quoted **with its beta,
its authority %, and its ADE**. All three, or the number is not interpretable -- a high
consistency score without its ADE is a report of obedience.

!! **And the run's own first verdict was wrong**: it took `max(gap)` over the whole sweep
(+0.4290 at beta 40) and called it a trend, when the series oscillates in sign there. A
maximum over noise is not an estimate. Fixed in code; the trend is now read only where
`ADE <= base`.

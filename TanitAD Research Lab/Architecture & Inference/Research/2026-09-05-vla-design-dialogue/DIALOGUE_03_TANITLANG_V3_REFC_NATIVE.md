# TanitLang v3 — language as GUIDANCE on REF-C's diffusion decoder

**2026-09-05, TanitAD_TrainingFlyWheel. Supersedes v2 (`TANITLANG_DESIGN_V2.md`), which contained
an error the PI caught.** Status: DESIGN, nothing built.

---

## 0. ⛔ The correction I owe first

**v2 mixed REF-C with the world-model line (v6/v7).** Two concrete errors:

1. I wrote *"we have a world model natively"* as REF-C's advantage over LaST-VLA. **REF-C is not a
   world model.** It is an anchored diffusion planner: encoder → `cond` bus → anchor fan → truncated
   denoise → scorer. The world model is the v6/v7 line, a **different architecture**. That advantage
   may hold for v7; it is not REF-C's and I should not have claimed it here.
2. I imported `H-ARCH-ACTINS` (action-insensitivity, ~1 % closed-loop vs hold-action) as a
   constraint on this design. **That is a v7 world-model measurement.** REF-C has no action-rollout
   channel of that kind; the finding does not transfer and I withdraw it from this plan.

⇒ Everything below is derived from `stack/tanitad/refs/refc.py` **read at source**, not from the
v7 line.

---

## 1. ⭐ What I found in REF-C, and why it changes the whole design

v2 invented a `FanReasoner` to talk to the planner. **That was unnecessary: REF-C already defines
exactly how an upper layer conditions the fan, and it has four such ports, all zero-init.**

`AnchoredDiffusionDecoder.forward` (`refc.py:1611`), two distinct conditioning surfaces:

**(a) the `cond` bus — continuous, additive** (`:1644-1650`)
```python
cond = self.cond_proj(m)                       # measurement
cond = cond + self.ctx_to_cond(ctx)            # strategic context
cond = cond + self.lan_to_cond(lan_emb)        # ⭐ A LANGUAGE PORT. ALREADY EXISTS. zero-init.
```

**(b) the ANCHOR-PRIOR bus — discrete, log-space, summed into the selection logits**
(`:1715-1728`)
```python
terms.append(self.maneuver_to_anchor(log_softmax(maneuver_logits)))   # tactical manoeuvre
terms.append(self.lat_to_anchor(lat_prior))                           # lateral action
terms.append(self.lon_to_anchor(lon_prior))                           # longitudinal action
terms.append(self.lan_gate * self._lan_anchor_prior(lan_dir, bank))   # route
```

⭐⭐ **THAT is "goals and constraints conditioning the planner", and it is already built.** Each is
`Linear(vocab → 117)`, zero-init, gated. A higher layer does not "influence" the planner vaguely —
**it re-weights the 117 anchors directly, in log-space.**

⇒ **TanitLang does not need a new interface. It needs to produce better priors for the ports REF-C
already has.** That is what "extending REF-C" means concretely, and it is why v2 was unrecognisable:
it routed around this machinery instead of using it.

---

## 2. Your four questions, answered directly

### 2.1 Which parts are taken from REF-C?

| taken | from | used as |
|---|---|---|
| the **117-anchor fan** | `AnchorConfig`, `anchors [117,8,2]` + `controls` | the hypothesis set the reasoning ranges over — **the reasoning's vocabulary of possible futures** |
| **truncated denoising**, `steps=2` | `:1394-1399` | the refinement budget for the *latent CoT*, same idiom and same trained-operating-point limit |
| the **`cond` bus** | `:1644-1650` | where the continuous rationale enters |
| the **`lan_emb` / `lan_to_cond` port** | `LanConfig`, `:1648` | ⭐ the mount point — **it already exists and is zero-init** |
| the **`*_to_anchor` prior ports** | `:1715-1728` | ⭐ how goals become planner conditioning |
| the **scorer** (`SelectionConfig`, `sel_score`) | `:462` | the committer — **unchanged, still the authority** |
| `cons_head` / `cons_ctx` | `:1619` | the existing consistency seam in selection |

**Nothing new is invented at the interface.** The new module is a *producer* for ports REF-C ships.

### 2.2 How is the embedding space coupled?

⭐ **Not by projecting into a shared hidden dimension — by both sides speaking in the ANCHOR
SIMPLEX.** The planner's output is a distribution over 117 anchors. The language part's output is
*also* a distribution over 117 anchors (via `lat_to_anchor` etc.). **The shared space is Δ¹¹⁶, and
it needs no alignment loss because it is the same object by construction.**

That is a much stronger coupling than "project `pooled` into `d_lm`", and it is testable: a
disagreement between reasoning and plan is a **KL between two distributions over the same 117
anchors** — one number, no correspondence problem.

⚠️ The continuous side (`lan_emb → cond`) still exists and carries what the discrete vocab cannot
express. But the *decision-relevant* coupling is discrete and lives in anchor space.

### 2.3 How do text tokens combine with the tactical and strategic vocab?

**They do not need combining, because the vocab tokens ARE the interface.** The LM gets a
**constrained head** over exactly our released vocabularies:

| head | vocab | feeds |
|---|---|---|
| lat | `TACTICAL_LAT_ACTIONS_V7` (8) | `lat_prior` → `lat_to_anchor` |
| lon | `TACTICAL_LON_ACTIONS_V7` (8) | `lon_prior` → `lon_to_anchor` |
| strategic action | `STRATEGIC_ACTION_TOKENS_V7` (7) | `ctx_to_cond` / strategic seam |
| strategic goal | `STRATEGIC_GOAL_TOKENS_V7` (8) | ditto |
| manoeuvre | the manoeuvre set | `maneuver_logits` → `maneuver_to_anchor` |
| **free text** | the LM's own vocabulary | ⛔ **rendering ONLY — never reaches the planner** |

⭐ **The separation is the safety property.** Free text cannot influence the fan because it is not
wired to any anchor port. Only a distribution over a *released, masked, 8-wide vocabulary* can.
A hallucinated sentence is inert by construction.

⚠️ And the loss mask carries over: `NOT_YET_EXTRACTABLE` classes are masked in these heads exactly
as in the label consumer, so the language part cannot emit a goal the corpus cannot supervise.

### 2.4 How does the planner benefit?

Three mechanisms, in increasing ambition — and ⛔ **only the first is in this plan**:

1. **Better priors.** Today `lat_prior`/`lon_prior` come from small heads on the trunk. TanitLang
   produces them from a *reasoned* read of the scene + fan + nav + query. Same port, better
   distribution ⇒ a better-weighted fan ⇒ a better selection.
2. **Constraints as negative priors.** A constraint ("do not change lane, cyclist right") is a
   **mask/penalty over anchors** — the same log-space bus, negative. This is how "constraints"
   become planner-legible without a new mechanism.
3. ⛔ **Guidance during denoising** — not in this plan; §5 explains why it is the natural next step
   and why it must wait.

---

## 3. ⭐⭐ The loop you identified, and how CFG dissolves it

**Your concern, restated:** the CoT emits goals → goals re-weight the fan → the reasoning must be
consistent with the fan → but the fan was shaped by the reasoning. Consistency then becomes
tautological, and a wrong plan can never be flagged.

⭐ **REF-C is a diffusion model, and diffusion already has the answer: classifier-free guidance.**
CFG runs **two passes**, unconditional and conditional, and combines them. We use that structure:

```
  pass 0  (UNCONDITIONED)   fan F0, scores S0     ← all language ports gated to 0
  pass 1  (CONDITIONED)     fan F1, scores S1     ← language priors active
  guided                    S = S0 + w·(S1 − S0)
```

**Three consequences, all measurable:**

| quantity | what it is | why it breaks the loop |
|---|---|---|
| **consistency vs F0** | is the rationale true of the scene *as the planner saw it before being told anything*? | ⭐ **the reasoning is judged against the evidence it was given, NOT against the outcome it caused.** The circularity is cut here |
| **‖S1 − S0‖** | the guidance delta | ⭐ **the CoT's causal contribution is a number CFG computes anyway** — not an added probe. Zero delta ⇒ the language channel is inert |
| **rationale consistent with F1 but NOT F0** | post-hoc rationalisation | a *detectable* failure mode, which is what an explanation channel is for |

⛔ **This is the design's central claim and it is falsifiable.** If a rationale is only ever
consistent with the fan it produced, the module is a rationaliser and we report that.

⚠️ **And `w` is a knob with a trap.** Large `w` makes the language dominant — the 368 M module
silently overriding a 2.15 M cascade, dissolving the hierarchy. **`w` starts at 0** (the ports are
zero-init anyway) and any increase is a pre-registered arm with the shuffled-prior control beside it.

---

## 4. The reasoning module — small, and on the tact

**`AnchorReasoner`, ~6 M**, the only new thing on the 300–500 ms tact:

* reads: `pooled` tokens, `ctx`, the **fan F0 and its scores S0**, nav, `v0`
* **K = 16 latent thought tokens, 2 truncated refinement steps** — REF-C's own idiom
* writes: `lat_prior`, `lon_prior`, `maneuver_logits`, strategic tokens, and a `lan_emb`
* ⭐ **iterative token-space self-editing** (ReflectDrive-2 idiom, `2605.04647`): the prior is
  **edited twice**, not emitted once — propose, check against F0, revise. Bounded at 2 edits to fit
  the tact.

**Why latent and not text on the tact** — PUBLISHED, two independent results:

* `2602.01148` (*Fundamental Limits of Latent CoT*): latent reasoning is **strong at exploration**
  (97.0 % ProsQA), **weak at execution** (34.1 % GSM8K), the trade-off proven to be governed by
  decisional certainty. ⇒ **the latent CoT explores the fan; the existing scorer commits.** Each
  mechanism where it is measured strong. The same paper proves **curriculum training is necessary** —
  and REF-C's staged recipe already is one.
* Alpamayo-R1 (`2511.00088`, Tbl 14): 99 ms on an **RTX 6000 Blackwell workstation**, **71 % of it
  CoT decode**. ⇒ text on the tact is not a budget, it is a hope. **Text renders off-tact, ≤1 Hz.**

**The LM: SmolLM2-360M frozen + LoRA (~5 M trainable)**, off the tact, rendering the *committed*
choice and answering queries. Resident ≈ **370 M**, trainable ≈ **11 M**.

---

## 5. Frontier ideas taken, and the one deliberately deferred

| source | idea | taken? |
|---|---|---|
| [ReflectDrive-2](https://arxiv.org/pdf/2605.04647) | goal-point hypotheses condition a discrete-diffusion decoder; **token-space self-editing** | ✅ the 2-step prior edit (§4) |
| [CoVT](https://arxiv.org/pdf/2511.19418) | ~20 continuous thought tokens suffice | ✅ K=16 budget |
| [LaST-VLA](https://arxiv.org/pdf/2603.01928) | latent spatio-temporal CoT, physical-prior alignment | ✅ latent-on-tact; ⚠️ its world-model distillation is **not** ours to claim (§0) |
| [SOLVE](https://arxiv.org/pdf/2505.16805) | 36-trajectory bank, k-means per nav command, select by similarity | ✅ precedent that bank-and-select competes; our fan is 117 and predates it |
| [Neuro-Symbolic Drive](https://arxiv.org/html/2606.23938v1) | reasoning traces supervised by a **rule-based planner** | ✅ our unicycle-feasible anchors are that rule source |
| [Do Latent Tokens Think?](https://arxiv.org/pdf/2512.21711) | causal + adversarial analysis of continuous thought | ✅ the control literature — latent CoT is not automatically faithful |
| **CFG** (Ho & Salimans) | dual-pass conditional/unconditional guidance | ⭐ **the loop resolution (§3)** |

⛔ **DEFERRED, and this is the genuinely ambitious one: guidance INSIDE the denoising loop.**
REF-C denoises for 2 steps. A language prior could steer *each* step rather than only re-weighting
the final selection — true language-guided trajectory diffusion. ⚠️ I am not proposing it yet
because MEASURED this session: pushing REF-C past its trained step budget **diverges** (sel-ADE
0.654 → 1.529 m at steps 8, SEPARATED). Adding a per-step force to a 2-step schedule that is
already at its operating limit is a bigger intervention than it looks. **It becomes the natural v6
of this line once §3's delta is shown non-zero and safe.**

---

## 6. Hypotheses, both outcomes committed

| id | if TRUE | if FALSE |
|---|---|---|
| **H-TL-1** reasoned priors beat the current small heads on selection quality | the language part earns its place | it is an explainer only — say so |
| **H-TL-2** ‖S1−S0‖ is non-zero and *helpful* (selection improves) | the channel is causally live | ⛔ inert; report and stop |
| **H-TL-3** rationales are consistent with **F0**, not only F1 | grounding is real | ⛔ it is a rationaliser; the USP fails |
| **H-TL-4** shuffled-fan degrades the rationale | it reads the fan | grounding is decorative |
| **H-TL-5** constrained-vocab heads suffice; free text adds nothing to planning | the safety separation is free | reconsider — carefully |

**Controls, all of which must be SHOWN TO FIRE on a deliberate-regression arm before any number is
admissible:** shuffled-fan · shuffled-nav (built) · constant-prior floor · rationale-swap
(render from the runner-up's rationale) · **`w = 0` identity** (with guidance off the output must be
*bit-identical* to today's REF-C — the strict-subset property).

---

## 7. Work packages

| # | package | gate |
|---|---|---|
| **WP-0** | Thor latency + `max_memory_allocated` for K=16 × 2 steps | every estimate becomes MEASURED |
| **WP-1** | `AnchorReasoner` → `lat/lon/maneuver` priors, `w=0`, identity proven | ⛔ bit-identical output with guidance off |
| **WP-2** | The dual-pass F0/F1 readout + ‖S1−S0‖ + consistency-vs-F0 | H-TL-2, H-TL-3 answered |
| **WP-3** | The five controls, each shown to fire | ⛔ nothing quotable until this passes |
| **WP-4** | LM attach, constrained vocab heads, text off-tact | H-TL-5 |
| **WP-5** | Alpamayo CoT distillation | only if WP-1–4 hold |

⚠️ **Open, and honestly unresolved:** we have **no question corpus** — free-form QA is a data gap,
not a modelling gap. And nav is **1.313 bits/clip** and an oracle (`ego-future`, 4,719/4,719), so
nav cannot be what justifies the module; any nav-conditioned claim needs shuffled-nav beside it.

⛔ Seven papers cited here are **not yet banked**; `tools/kb_add.py` before any number enters the
registry.

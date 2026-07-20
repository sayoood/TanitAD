# Loss Rebalancing (v2.x) and Pure LeWM/JEPA Train+Decode (v3) — Deep Research + Recommendation

**Date:** 2026-07-18
**Author:** research agent (deep-research pass, no training code changed)
**Scope:** Two decisions. **PART A** = an immediate, safe loss-rebalance for v2.x. **PART B** = whether a v3 that *separates* SSL representation learning from action-conditioned prediction/imagination is warranted.
**Protected quantity:** the operative grounding forward-consistency **fwd-ADE = 0.033 m** (`g_op_fwd_ade_m`) must not regress. **Operational constraint:** any v2.x change must be provably incapable of destabilizing a 4–5 day unattended run.

---

## 0. TL;DR — the two decisions

**PART A (do this for v2.x — safest effective option):**
**Gradient-decouple the metric-inverse-dynamics *real-pair* grounding term from the encoder.** Concretely, apply a straight-through **gradient scale α** to the latents (`z_t`, `fut_states`) *as they enter* `grounding.invdyn[*]` in `metric_dynamics.grounding_losses`, leaving the forward-consistency rollout term (b), JEPA, and SIGReg fully attached. Start **α = 0.25**, ablate `{1.0 (control), 0.5, 0.25, 0.0}`. This removes ~6.0 of the ~9.0 units of encoder-shaping supervised mass **without touching the term that actually produces the 0.033 m metric** (the forward-consistency rollout, term b), keeps the invdyn heads training at full rate as *readout probes*, is forward-pass-identical, and cannot destabilize the run (it only shrinks one gradient path — SIGReg still prevents collapse, JEPA still trains). Gate at a 5 k mid-checkpoint before committing 30 k. **Fallback (zero-new-code-path):** drop the `invdyn` weight `2.0 → 1.0` (then `0.5`). **Do NOT** reach for PCGrad / GradNorm / naive uncertainty-weighting on this run — reasons in §2.3.

**PART B (verdict):** **A v3 that separates encoder training from action-conditioned prediction/imagination IS warranted by the literature — but only as a measurement-gated experiment with an *own-data* SSL phase-1, not as a wholesale switch, and not as "freeze DINOv2" (that is REF-A, already run and plateaued).** The entire JEPA/LeWM lineage (I-JEPA, V-JEPA/V-JEPA 2, **V-JEPA 2-AC**, **DINO-WM**, LeJEPA, LeCun's position paper) two-stages representation vs. action-conditioned dynamics, and there is consistent evidence this is *why* it generalizes (Kumar et al.'s fine-tuning-distortion result is the theorem behind it). **But** our own data contains an explicit warning: the pre-grounding JEPA latent had a **1.65 m oracle in-distribution decode ceiling** — a frozen trunk may cap the metric far above 0.033 m unless phase-1 is on our driving corpus and the predictor/adapter carries the metric. The clean tie-breaker: **Part A is the cheap (~1%-cost) ablation that de-risks Part B** — if relieving encoder grounding raises `vision_use` while holding the metric, the expensive two-phase v3 is worth building; if the metric collapses the moment you unshackle the encoder, v3's freeze will hit the same ceiling.

---

## 1. The verified problem (read from our code)

Files read: `tanitad/train/flagship_losses.py`, `tanitad/models/metric_dynamics.py`, `tanitad/models/sigreg.py`, `HYPOTHESIS_LEDGER.md` (H25/H26).

### 1.1 The joint objective and the weight arithmetic

`flagship_loss()` sums one big objective (`flagship_losses.py:380-393`). Grouping by character:

| Group | Terms (nominal weight) | Mass |
|---|---|---|
| **SSL world-model core** | `pred` 1.0 (JEPA op) + `tacpred` 0.5 (JEPA tac) + `roll` 0.5 (K-step) + `goal` 0.5 (goal-latent JEPA) + `sigreg` 0.1 | **≈ 2.6** |
| **Supervised metric-motion / dynamics-decode** | hierarchical grounding `invdyn` 2.0×3 + `fwd` 1.0×3 = **9.0**; `wp` 1.0; `goalwp` 1.0; `man` 0.5; `route` 0.5; `inv` 0.5 (A5) | **≈ 12.5** |

So supervised metric-motion objectives outweigh the SSL core **≈ 5 : 1** — the user's framing checks out against the code.

### 1.2 The no-stop-gradient coupling is confirmed

`grounding_losses` (`metric_dynamics.py:286-346`) reads `z_t = states[:, -1]` and `fut_states[...]` **with no `.detach()`**, and rolls `model.predictor` on `states`. So the grounding gradient reaches **the encoder** (via `states`/`fut_states`) and the **predictor**. There are two sub-terms, and the distinction is the whole key to Part A:

- **(a) metric-inverse-dynamics on REAL latent pairs** — `grounding.invdyn[lvl](z_t, fut_states[...])` regressing odometry `(Δx,Δy,Δyaw)` (`metric_dynamics.py:320-328`). Weight **2.0 per level = 6.0**. Per its own docstring this term exists *"to force the ENCODER latent to encode metric ego-motion"* — i.e. it is deliberately reshaping the trunk into an odometer. It is a **static supervised probe** that happens to be wired into the trunk.
- **(b) forward-metric-consistency on the predictor rollout** — decode `grounding.step[lvl]` on the true-action rollout, accumulate SE(2), match odometry (`metric_dynamics.py:330-338`). Weight **1.0 per level = 3.0**. **This is the term whose `g_op_fwd_ade_m` = 0.033 m is the protected metric.** It is action-conditioned and philosophically part of the *predictor/world-model* objective, not a static probe.

### 1.3 The measured symptoms (from HYPOTHESIS_LEDGER H25/H26, 2026-07-18)

- `vision_use` ≈ **12.9 %** flat (imagination panel) — the model leans on fed dynamics (`v0`/`yr0`) + the speed-input channel.
- The trained encoder **redundantly re-encodes** the fed ego: **in-latent yaw R² = 0.89** (the exact quantity LEVER B's `decorr` now penalizes post-hoc).
- `route` head is a pure command-echo: **`route_skill_vs_chance` = 0.0** (follow-acc == base rate), at 30 k still.
- H18 grounding **dominance grew** with training (Δ 2.70 m grounded-vs-ungrounded at 30 k) — the dynamics pathway keeps winning capacity.

### 1.4 One honest caveat that shapes the whole recommendation

**Loss *weights* are not gradient *magnitudes*.** The grounding losses are divided by `pose_scale` to O(1) and are MSE-type: as the model fits (0.033 m), `loss_fwd`/`loss_mid` **shrink toward zero**, so their late-training *gradient* contribution is far below their 9.0 nominal weight, while SIGReg and JEPA-pred stay O(1). The 5:1 ratio therefore describes the **early training trajectory that set the encoder's character**, not the steady-state gradient budget. Consequence: **a pure re-weight is a weak lever** — once the encoder is ego-motion-shaped (yaw R² 0.89), shrinking a small-valued term does little to un-shape it, and most of the low `vision_use` is driven by the *fed* ego + speed channels, not the grounding loss value. This is *why* the right move is to cut the grounding→encoder **coupling** (detach/scale), not merely its scalar — and why the fed-dynamics levers (future-action dropout, LEVER B decorr) are the complementary half.

---

## 2. PART A — Loss rebalancing

### 2.1 The literature menu (what comparable systems do)

**Multi-task loss balancing**
- **Uncertainty weighting** (Kendall, Gal & Cipolla, CVPR 2018): weight each task by learned homoscedastic uncertainty, `L = Σ (1/2σ_i²)L_i + log σ_i`. Balances heterogeneous units automatically. *Known failure for our case:* as a regression head fits (`L_i → 0`), its learned `1/σ_i²` can **grow without bound**, over-amplifying exactly the well-fit grounding heads — the opposite of freeing the encoder. Also adds learned parameters that can drift over a long unattended run.
- **GradNorm** (Chen et al., ICML 2018): dynamically tunes per-task weights so gradient magnitudes at a shared layer stay balanced (single `α` hyperparameter). Effective but **introduces a learned-weight dynamic** that needs tuning and can interact badly with a 4–5 day run.
- **PCGrad / gradient surgery** (Yu et al., NeurIPS 2020): project each task gradient off any conflicting (negative-cosine) task gradient. Reduces destructive interference. **Cost:** needs per-task gradients — with our ~13 loss terms that is many extra backward passes and a loop rewrite (high blast-radius for an unattended run).
- **Gradient-cosine gating** (Du et al., 2018, *Adapting Auxiliary Losses Using Gradient Similarity*): treat grounding as an **auxiliary** loss and gate it by the cosine similarity of its gradient with the main (SSL) gradient — apply it only when it does not conflict; **provably converges to critical points of the main task.** This is the principled version of "protect the representation."
- Loss-scale normalization / cosine annealing schedules: cheap, but blunt.

**SSL-representation protection**
- **Stop-gradient / detach the task head from the encoder** (decode on a detached latent). The cleanest lever. SimSiam (Chen & He 2021) shows stop-gradient is *the* ingredient that stops the trunk collapsing under a decoder; the same asymmetry logic says "let the head read the representation, don't let it rewrite it."
- **Decoupling representation from task in RL** (Stooke et al., 2021, *Decoupling Representation Learning from RL*; and follow-ups): explicitly test **stopping the critic/auxiliary gradient before the CNN encoder** — "for many environments, detached CNN representations were sufficient to learn an optimal policy." Direct precedent that a **detached readout keeps task accuracy while freeing the trunk.**
- **Projector-head isolation** (SimSiam/BYOL/VICReg): the loss is applied *through* an expendable projector/predictor MLP; the transferable representation is the trunk *before* it. Our grounding heads are already separate `nn.Module`s (`HierarchicalGrounding`) — they are the projector — the only issue is that gradient still flows *past* them into the trunk.
- Separate optimizer group / low-LR for the encoder: **note this does NOT solve our problem** — a lower LR on the *head* params does nothing to the gradient the head injects *into the encoder*; a lower LR on the *encoder* slows everything including the SSL core. The correct boundary to act on is the head→encoder gradient itself.

### 2.2 The specific tension, resolved

Rebalancing toward SSL must not destroy the 0.033 m metric. The naive move — **fully detaching all grounding from the encoder** — is **unsafe**, because our own evidence (`metric_dynamics.py` docstring) is that the *pre-grounding JEPA latent did not encode metric ego-motion* (1.65 m oracle ceiling); the encoder's metric content was *put there by grounding*. Remove it entirely and the metric likely regresses.

The resolution is the **(a) vs (b) split** from §1.2:
- Term **(a)** (invdyn, real pairs, weight 6.0) is the *"make the encoder an odometer"* term — a static probe. **This is the thing we want to dial back.**
- Term **(b)** (forward-consistency rollout, weight 3.0) is the term whose ADE **is** 0.033 m and is action-conditioned world-model dynamics. **Keep it.** Critically, (b) *also* reaches the encoder (via `states` into the predictor rollout), so keeping (b) attached **retains encoder metric-shaping through the "good" dynamics path** while we relieve the static-probe path.

So we can free encoder capacity (dial back a) *and* preserve the metric (keep b) — the two are separable in our code.

**Why gradient-scale (a)'s encoder path beats a plain re-weight of `invdyn`:** a plain re-weight (`2.0→x`) scales the gradient into the invdyn *heads* too, so the readout probe trains slower/worse. Gradient-scaling only the *encoder-bound* path leaves the heads at full learning rate (better probes) while relieving the trunk — strictly better for preserving readout accuracy at the same encoder relief.

### 2.3 RANKED recommendation for v2.x

**#1 — PRIMARY (safest effective): soft gradient-decouple of grounding term (a) from the encoder.**
- **Mechanism:** a straight-through gradient-scale `α` (forward identity; backward ×α) applied to `z_t` and `fut_states` *only where they feed* `grounding.invdyn[lvl]` — i.e. in the (a) branch of `grounding_losses`. The (b) forward-consistency branch, JEPA, SIGReg, and all other heads are untouched. `α = 1.0` reproduces today exactly; `α = 0.0` is a full probe-detach of (a).
- **Starting setting:** `α = 0.25`. Ablate `{1.0, 0.5, 0.25, 0.0}`.
- **Why it can't wreck 0.033 m:** the metric is produced by term (b), which is untouched; (a)'s heads keep training as full-rate readouts; the encoder still receives (b)'s dynamics-path gradient. Worst case is a *mild* metric softening if the encoder was leaning on (a) — caught by the mid-checkpoint gate (below).
- **Why it can't destabilize a 4–5 day run:** it only *removes* a gradient component. The remaining objective is a strict subset of forces; SIGReg (0.1) still prevents collapse; JEPA still trains. There is no new learned parameter, no per-batch gradient algebra, no schedule to diverge.
- **Why it should raise `vision_use`:** it removes the dominant pressure that writes ego-motion linearly into the trunk (the very yaw-R²-0.89 source LEVER B fights post-hoc), reallocating the encoder's relative gradient budget to JEPA scene-prediction. Synergises with LEVER B (`decorr`) and future-action dropout, which attack the *fed*-dynamics half.

**#2 — PRINCIPLED FALLBACK (if a fixed knob feels too manual): gradient-cosine gating of the grounding→encoder auxiliary (Du et al. 2018).** Gate the grounding-into-encoder gradient by its cosine with the SSL (JEPA+SIGReg) gradient; zero it only when it conflicts. Convergence guarantee to the SSL objective's critical points. **Cost:** needs the SSL and grounding gradients separately (one extra partial backward or a grouped `autograd.grad`), so it is more loop-invasive than #1 — acceptable but not the first thing to run unattended.

**Ultra-safe minimal fallback (no new code path): re-weight `invdyn` `2.0 → 1.0` (then `0.5`).** One number in `LossWeights`. Weaker (see §1.4 caveat and §2.2) and slightly degrades the readout, but *provably* harmless and a useful cheap probe.

**NOT recommended for this run (with reasons):**
- **Naive uncertainty weighting** — the `1/σ²→∞` failure on well-fit grounding heads (§2.1) would *amplify* the exact terms we want to relieve.
- **GradNorm / PCGrad** — learned-weight dynamics / per-task gradient surgery over ~13 terms; both raise blast-radius and instability risk on a long unattended run for no advantage over #1 here. Fine as *offline* research probes, not as the v2.x change.
- **Encoder low-LR optimizer group** — wrong boundary (§2.1); would also slow the SSL core.

### 2.4 Concrete settings + ablation gate + safety valves

- **Change:** add `α` (grounding-invdyn encoder-gradient scale); default `1.0` (no-op) so the switch is opt-in. Recommend first run at `α = 0.25`. Optionally, as a *second, separate* change for clean attribution, `invdyn 2.0 → 1.0`.
- **Keep untouched:** `fwd` 1.0, `pred` 1.0, `sigreg` 0.1, `wp`, `man`, `route`, `inv`. Do **one** change at a time.
- **Gate (canonical val):** `g_op_fwd_ade_m` non-regress (hold ≤ ~0.05 m vs the 0.033 baseline) **AND** `vision_use` ↑ (target >15–20 %, H25 gate) **AND** in-latent `ego_r2` ↓, at flat `ade_0_2s`.
- **Safety valve:** evaluate at a **5 k mid-checkpoint** before letting it run to 30 k; auto-rollback to `α=1.0` if `g_op_fwd_ade_m` regresses beyond the gate. Because α is a pure gradient scale, rollback is bit-clean.

---

## 3. PART B — Pure LeWM/JEPA train+decode for generalization

### 3.1 What the lineage actually does — and why it generalizes

The evidence that the JEPA/world-model lineage **separates representation learning from action-conditioned prediction/decoding**, and that the separation is *load-bearing for generalization*, is consistent across the canon:

- **V-JEPA 2 / V-JEPA 2-AC** (Meta, 2025, arXiv:2506.09985): stage 1 self-supervised video encoder on >1 M hours; stage 2 **freezes the encoder** and post-trains a *compact action-conditioned predictor* on <62 h of unlabeled robot data; then plans via MPC in latent space, **zero-shot in unseen labs (65–80 % success).** "The separation of vision (frozen encoder) and dynamics (learned predictor) modularizes generalization and control." This is the canonical two-phase recipe.
- **DINO-WM** (arXiv:2411.04983): **frozen DINOv2 patch features** + a learned latent causal-ViT/MLP dynamics model; **no pixel reconstruction, no inverse model**; zero-shot MPC planning that **beats task-specific learned latents** across mazes/push/particles. Direct evidence that frozen-perception + learned-latent-dynamics ≥ end-to-end for planning.
- **I-JEPA / V-JEPA frozen evaluation**: the headline protocol is a **frozen encoder + attentive probe** (small cross-attention read-out), competitive with fine-tuning and, crucially, **more robust under corruption/OOD** — frozen V-JEPA retains its advantage over fine-tuned baselines on corruption robustness.
- **Kumar, Raghunathan, Jones, Ma & Liang** (ICLR 2022, arXiv:2202.10054), *Fine-Tuning can Distort Pretrained Features and Underperform OOD*: **the theorem behind all of the above.** When features are good and the shift is large, full fine-tuning *distorts* the pretrained features and underperforms a linear probe OOD; **LP-FT** (probe first, then gentle fine-tune) is ~1 % better ID and ~10 % better OOD than full FT. This is exactly the failure mode of our joint objective — the supervised heads are *distorting the trunk*.
- **LeCun**, *A Path Towards Autonomous Machine Intelligence* (2022): the architecture *itself* separates **Perception (encoder)** from **World Model (predictor)** from **Actor/Cost**. The encoder is not meant to be a dynamics decoder.
- **LeJEPA** (Balestriero & LeCun, 2025, arXiv:2511.08544 — our SIGReg) plus **"When Does LeJEPA Learn a World Model?"** (Klindt et al., 2026, arXiv:2605.26379): the SSL objective *alone* **linearly recovers the world's latent variables up to rotation** when latents are Gaussian/stationary; for O(n)-invariant costs, planning in the learned latent equals planning in the true latent. I.e. a *pure* LeJEPA encoder can be a sufficient world-model substrate that you **decode/plan on top of**, without supervised dynamics reshaping it.

**Verdict on the lineage question:** yes — the lineage two-stages representation vs. action-conditioned dynamics, and the generalization advantage (OOD robustness, un-distorted features, modular control) is the *stated reason*. Our symptoms (`vision_use` 12 %, `route` command-echo, yaw-R²-0.89 re-encode, grounding dominance growing) are precisely the pathologies decoupling is designed to remove.

### 3.2 The metric tension — the reason to be careful

The literature's generalization case runs into **one hard, in-house data point**: the pre-grounding JEPA latent had an **oracle in-distribution decode ceiling of 1.65 m** (vs. trivial CV 0.28 m and MLP probe 3.89 m — `metric_dynamics.py` docstring). The 0.033 m metric exists **because end-to-end grounding reshaped the encoder.** So a *frozen pure-SSL trunk + readout* risks capping the metric near that 1.65 m ceiling rather than 0.033 m — **unless** (i) phase-1 SSL is on *our driving corpus* (so the trunk actually contains driving ego-motion structure, which V-JEPA 2/DINO-WM get from massive in-domain-relevant pretraining) and (ii) the action-conditioned predictor + adapter/readout is expressive enough to carry the metric that the frozen trunk does not. V-JEPA 2-AC and DINO-WM *do* get metric-quality control from a frozen trunk + learned dynamics, so it is plausible — but our 1.65 m number says **v3 must validate the metric survives the freeze, not assume it.**

### 3.3 The in-house test: flagship (E2E) vs REF-A (frozen adapter) — and why it is currently *confounded*

We already run a partial instance of Part B: **REF-A = frozen DINOv2 + adapter + learned dynamics** (the DINO-WM analog) vs **flagship = end-to-end trained ViT encoder**.

Current in-house score (HYPOTHESIS_LEDGER / memory, 30 k):
- **flagship E2E:** `ade_0_2s` **0.4522 m** — first arm below every trivial bar (best-of-3 floor 0.5005, CTRV 0.523, ridge ego-status ceiling 0.5735).
- **REF-A frozen DINOv2:** **plateaued ≈ 2.14 m** (frozen ceiling).

Taken at face value this says *end-to-end wins*. **But it is a confounded test of "frozen,"** for the same reason DINO-WM emphasizes feature quality: REF-A froze **generic web-image DINOv2**, which — our own ledger states — **"can't decode ego-motion."** That is not a test of "own-SSL-encoder, frozen"; it is a test of "off-the-shelf-encoder, frozen," which the lineage would *also* predict to underperform on a metric the frozen features don't contain. (Consistently, the H26 refinement found REF-A's frozen encoder *co-adapts to a large constant intent offset* — a symptom of a trunk that can't supply the needed signal.) So REF-A shows **"generic-frozen underperforms our metric"** — expected and *not* a falsification of Part B. The clean v3 test is **own-data-SSL encoder, frozen, vs flagship E2E, on BOTH in-distribution metric AND OOD/`vision_use`** — which we have **not** run.

### 3.4 VERDICT and recipe

**Warranted — as a measurement-gated v3 experiment, with an own-data phase-1, not a wholesale replacement and not "freeze DINOv2."**

**Recipe:**
1. **Phase 1 — SSL encoder on our driving corpus.** Action-conditioned JEPA predict-future-latent + SIGReg/LeJEPA (our existing core), **with the dynamics-decode gradient kept off the trunk** (i.e. Part A taken to its limit: grounding present only as detached probes, or only the (b) path lightly attached). Deliverable: a driving-appropriate encoder. **Gate before proceeding:** measure this encoder's *oracle* metric-decode ADE. If it is not ≪ current (and specifically far below the old 1.65 m / REF-A 2.14 m regime), **stop** — freezing it will cap the metric.
2. **Phase 2 — freeze (or LP-FT) + decode.** Freeze the encoder (or LP-FT per Kumar: adapter/linear probe first, then a *very* low-LR unfreeze), and train the action-conditioned predictor + hierarchical grounding readouts + planning heads on top (this is the V-JEPA 2-AC / DINO-WM stage-2). Attentive-probe-style read-outs (I-JEPA/V-JEPA) rather than a single linear layer.

**Costs / risks:**
- ~2× pipeline (two phases, a new phase-1 checkpoint, re-tuning), plus a real risk the frozen trunk caps the metric (the 1.65 m warning). This is a *weeks*, not *days*, commitment.
- Upside if it lands: the generalization the literature promises — higher `vision_use`, OOD/corruption robustness, route-from-vision, and the *modular* control story (swap dynamics/heads without disturbing perception).

### 3.5 Synthesis — Part A de-risks Part B

Part A and Part B test the **same hypothesis** (H25: decouple dynamics from the trained encoder) at two price points. **Part A is the ~1%-cost ablation of Part B.** Run Part A first:
- If soft-detaching grounding term (a) **raises `vision_use` while holding 0.033 m**, that is strong in-house evidence the encoder does *not* need the static-probe reshaping and the full two-phase v3 freeze will work — greenlight v3 phase-1.
- If the metric **softens the moment you relieve encoder grounding**, that is a direct warning that a frozen v3 trunk will land near the 1.65 m ceiling — and v3 must first solve "phase-1 encoder that contains driving ego-motion" before any freeze.

Either way, Part A returns the single most decision-relevant number for Part B, cheaply and safely, on the current run.

---

## 4. Strongest citations (annotated)

1. **Kumar, Raghunathan, Jones, Ma, Liang (2022), *Fine-Tuning can Distort Pretrained Features and Underperform OOD*, ICLR — arXiv:2202.10054.** The theorem for *both* parts: supervised heads distort a good trunk and hurt OOD; LP-FT (probe-then-gentle-FT) is the fix. Justifies Part A (detach the head from the trunk) and Part B (frozen/LP-FT phase-2).
2. **V-JEPA 2 / V-JEPA 2-AC (Meta, 2025) — arXiv:2506.09985.** Canonical two-phase: frozen SSL video encoder + post-trained action-conditioned predictor → zero-shot planning. The reference architecture for a v3.
3. **DINO-WM (2024) — arXiv:2411.04983.** Frozen perceptual features + learned latent dynamics beat task-specific latents for zero-shot planning — and the in-house REF-A is its (confounded, generic-encoder) analog.
4. **Du et al. (2018), *Adapting Auxiliary Losses Using Gradient Similarity* — arXiv:1812.02224.** Principled, convergence-guaranteed way to keep an auxiliary (grounding) loss from harming the main (SSL) objective — the Part A #2 fallback.
5. **Kendall, Gal, Cipolla (2018), *Multi-Task Learning Using Uncertainty…*, CVPR — openaccess.thecvf.com.** The uncertainty-weighting baseline — cited to explain *why we are NOT using it here* (the `1/σ²→∞` failure on well-fit grounding heads).
6. **Balestriero & LeCun (2025), *LeJEPA* — arXiv:2511.08544** + **Klindt et al. (2026), *When Does LeJEPA Learn a World Model?* — arXiv:2605.26379.** Our SIGReg's source, plus the identifiability conditions under which a *pure* SSL encoder is a sufficient world-model substrate to decode/plan on top of.

Supporting: Chen et al. GradNorm (ICML 2018, arXiv:1711.02257); Yu et al. PCGrad (NeurIPS 2020, arXiv:2001.06782); Chen & He SimSiam (2021); Stooke et al. *Decoupling Representation Learning from RL* (2021); LeCun *A Path Towards Autonomous Machine Intelligence* (2022).

## 5. Sources

- https://arxiv.org/abs/2202.10054 — Fine-Tuning can Distort Pretrained Features and Underperform OOD (Kumar et al., 2022)
- https://arxiv.org/abs/2506.09985 / https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/ — V-JEPA 2 / V-JEPA 2-AC (2025)
- https://arxiv.org/abs/2411.04983 / https://dino-wm.github.io/ — DINO-WM (2024)
- https://arxiv.org/abs/1812.02224 — Adapting Auxiliary Losses Using Gradient Similarity (Du et al., 2018)
- https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html — Uncertainty weighting (Kendall et al., 2018)
- https://arxiv.org/abs/1711.02257 — GradNorm (Chen et al., 2018)
- https://arxiv.org/abs/2001.06782 — PCGrad / Gradient Surgery (Yu et al., 2020)
- https://arxiv.org/abs/2511.08544 — LeJEPA (Balestriero & LeCun, 2025)
- https://arxiv.org/abs/2605.26379 — When Does LeJEPA Learn a World Model? (Klindt et al., 2026)
- https://cis.temple.edu/tagit/presentations/A%20Path%20Towards%20Autonomous%20Machine%20Intelligence.pdf — LeCun position paper (2022)
- https://ar5iv.labs.arxiv.org/html/2009.08319 — Decoupling Representation Learning from RL (Stooke et al., 2021)
- https://learnopencv.com/simsiam/ — SimSiam stop-gradient (Chen & He, 2021)
- https://arxiv.org/pdf/2404.08471 — V-JEPA / feature-prediction frozen evaluation

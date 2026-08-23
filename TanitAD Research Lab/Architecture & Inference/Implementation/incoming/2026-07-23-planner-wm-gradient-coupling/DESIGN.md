# Planner↔WM coupling by gradient surgery — a smarter lever than the scalar floor

**Date:** 2026-07-23 (Berlin) · **Stream:** GradCouple (`a4fde3c6`, pod-free) · **Status:** DESIGN +
tested reference implementation, staged. **Nothing here launches a run; Sayed owns the go.**

**Evidence discipline (CLAUDE.md).** Every quantity is tagged `MEASURED` (with its artifact),
`PUBLISHED` (cited), `INHERITED` (another doc/agent, not re-verified), `ESTIMATED`, or `HYPOTHESIS`.
No GPU-day decision rests on an INHERITED claim. Read `Project Steering/RETRACTION_LOG.md` first — this
design touches classes **C3** (mechanism-as-fact) and **C6** (confounded comparison), and §7 addresses
both head-on.

---

## 0. TL;DR

The blunt scalar `lam_mult` floor shrinks the planner→trunk gradient **in every direction at once**.
The conflict it is trying to manage lives in **one shared activation** (`states`, `[B,8,2048]`), because
the planner reaches the trunk through exactly that one seam. So we can replace the scalar with a
**one-sided PCGrad projection at that seam**: let the world-model gradient flow untouched, and remove
from the planner gradient only the component that **opposes** it. The planner then shapes the trunk only
in directions that do not hurt prediction. It is a **single-backward, ~zero-cost drop-in** for the
existing `grad_scale(states, λ)` seam (§6 cost), it **preserves the canary controller and the gate as
backstops**, and it changes **exactly one thing** vs v4.2 (the coupling *operator*), so it is maximally
attributable. The reference implementation (`grad_surgery.py`, 9 tests green) proves the math is
one-sided, a strict no-op when the tasks agree, and finite. **§7 states plainly where it fails:** if the
degradation is driven by the warm trunk's own re-optimisation at `lr_trunk` (a single-task effect with
no conflict to project) or if the WM and planner want *fundamentally opposed* representations (cosine
≈ −1, projection removes almost all planner signal → v4.1-style starvation), surgery cannot save it and
the answer is from-scratch. The pre-registered experiment (`PRE_REGISTRATION.md`) is built to
**discriminate exactly those cases**, both outcomes committed in advance.

---

## 1. The problem, measured

v4 warm-starts v1's prediction-converged world model and couples a new anchored-diffusion planner's
gradient into the shared ViT trunk. That coupling **degrades the world model**, tracked by the plan-free
operative-rollout **canary** (roll the operative predictor under TRUE actions → ADE@2s; v1 baseline
**0.452**, `train_flagship_v16.py:271-303`). Three arms, three mistunings of the **one scalar knob**:

| arm | `lr_trunk` | coupling knob | canary (MEASURED) | planner | source |
|---|---|---|---|---|---|
| **v4** | 3e-4 | naive halve-to-**zero** controller | runaway **0.42 → 1.30+**; *rose even at λ_plan=0* | — | LOOP_STATE §hist (v4→v4.1 restart); brief |
| **v4.1** | 3e-5 | controller clamped λ to **1.5e-5** | **0.4599 PASS** (WM healthy) | **STARVED**: ade@2s **0.8522** ≫0.60 FAIL | `a938e1c0` / `flagship-v4.1-10k.json`; LOOP_STATE CURRENT #2 |
| **v4.2** | 1e-4 | cap-and-hold **floor 0.25** | **0.86@2k / 0.72@4k / 0.77@5k** DEGRADE | — | LOOP_STATE v4.2b stream (pre-reg rule) |
| v4.2b | 1e-4 | floor **0.15** | Phase-A 0.495; Phase-B tell pending | — | LOOP_STATE (live pod2) |

The scalar knob has no good setting: too high degrades the WM (v4, v4.2), too low starves the planner
(v4.1). That is the signature of a **wrong instrument**, not a wrong value — it is trying to resolve a
*directional* conflict with a *magnitude* control.

**⚠️ The one confound this design must respect (class C6, and it is in the table above).** v4 at
`lr_trunk 3e-4` **degraded the canary even with `λ_plan` pinned at 0** (LOOP_STATE §hist: *"it kept
degrading with the planner gradient fully clamped … the trunk fine-tuning, not the planner, is the
culprit"*, `MEASURED`). So *some* of the degradation is the warm trunk re-optimising under its **own** WM
loss at a warm-start-incompatible LR — a **single-task effect with no gradient conflict to project**.
Gradient surgery only touches the *planner's* gradient; it **cannot** fix trunk-LR re-optimisation. The
attribution at `lr_trunk 1e-4` (v4.2) is exactly what the pre-registered §PR experiment measures, via a
`λ_plan=0` control, before spending a full run. This document does **not** assume the planner is the sole
cause; it builds the instrument that finds out.

---

## 2. The one architectural fact that makes surgery cheap

`MEASURED` by source-read (`stack/scripts/train_flagship_v4.py`, `stack/tanitad/models/flagship_v4.py`,
`flagship_losses.py`, `flagship_v4.1_config.json`):

**The planner touches the shared trunk through exactly ONE tensor — the encoder-readout output
`states` — and the WM loss reads the same tensor.**

```
# train_flagship_v4.v4_loss_step (lines 89, 93, 107, 134)
states = world.encode_window(batch["frames"])            # the ONE shared trunk output [B,8,2048]
wm_total, ... = flagship_loss(world, grounding, batch, states, fut, ...)   # WM reads states
out           = head(states, v0, lambda_plan=lam, ...)   # planner reads states (through the λ seam)
strat         = goal_head(states[:, -1])                 # strategic head reads states too
```
```
# flagship_v4.FlagshipV4Head.forward (line 211)  — THE seam
states_p = grad_scale(states, lambda_plan)   # fwd-identity; bwd multiplies ∂L_plan/∂states by λ
```
`grad_scale` (`metric_dynamics.py:59`) is a straight-through **scalar** on the planner→trunk gradient.
Two facts make the seam the *complete* locus of the conflict:

- **`cond_imagination: false`** in the shipping head config (`flagship-v4.1-config.json` head_cfg) → the
  planner does **not** consume any predictor rollout, so it sends **no gradient into the predictor**. Its
  entire trunk footprint is `∂L_plan/∂states`. (This is *why* one scalar sufficed.)
- The WM loss's `states`-path gradient `∂L_predict/∂states` is where the WM and planner **meet**. The WM
  *also* reaches the trunk through `fut = encode(future)` and the predictor rollout — those never pass
  through `states`, so they are the WM's private directions and must stay untouched.

So the whole WM↔planner conflict is captured by two vectors of identical shape:

```
g_wm   = ∂L_predict / ∂states      # how the WM wants the encoder output to move
g_plan = ∂L_plan    / ∂states      # how the planner wants it to move
```

De-conflict these two and you have de-conflicted the planner's **entire** influence on the trunk — at
the cost of a few elementwise ops on a `[16,8,2048]` tensor. No second backward, no full-trunk PCGrad.

---

## 3. The mechanism — one-sided PCGrad at the seam, in a single backward

Replace the scalar fork `states → grad_scale(states, λ) → head` with a **projecting fork**:

```
s_wm, s_plan = seam_project(states)          # forward-identity on BOTH branches
wm_total, ... = flagship_loss(world, grounding, batch, s_wm,  fut, ...)
out           = head(s_plan, v0, lambda_plan=1.0, ...)   # λ seam now a no-op (or a scalar backstop)
strat         = goal_head(s_plan[:, -1])
total = wm_total + plan_l + fac + sm + strat
total.backward()                             # ONE backward; the seam de-conflicts inside it
```

`seam_project` is a custom autograd Function (`_SeamProject` in `grad_surgery.py`). Forward returns two
views of `states` (identity). In the **single** backward it receives `(g_wm, g_plan)` — the two branches'
gradients arrive **separately** because they came from two graph outputs — and returns, as the gradient
w.r.t. `states`:

```
combined = g_wm + deconflict(g_plan, g_wm)
```

`deconflict` is **one-sided (asymmetric) PCGrad** — the asymmetry is the whole point, because the two
tasks are **not** peers here: the WM is protected, the planner is the supplicant.

```
# per sample i (rows of the batch):
if  <g_plan_i, g_wm_i>  <  0:                     # the tasks conflict on this example
     g_plan_i ← g_plan_i − (<g_plan_i, g_wm_i> / ‖g_wm_i‖²) · g_wm_i   # drop the opposing component
# else: keep g_plan_i unchanged (no conflict → the planner contributes fully)
# g_wm is NEVER modified.
```

Properties (all `MEASURED` by `grad_surgery.py` smoke + `tests/test_grad_surgery.py`, 9 green):

- **Forward-identical** on both branches (`fwd_identical: true`) — the fitted `states` value never moves.
- **One-sided**: the WM gradient into the trunk is **byte-identical** with/without the seam
  (`wm_grad_untouched_maxabs: 0.0`). The WM always descends at full strength.
- **Strict no-op when the tasks agree** (`noop_when_aligned_maxabs: 0.0`): when
  `⟨g_plan, g_wm⟩ ≥ 0` the seam reduces to today's plain summed backward, so the default path is provably
  unchanged — the safety contract for an unattended multi-day run (exactly `grad_scale`'s `α==1`
  short-circuit contract).
- **Acts only under conflict**, and stays finite.

**Why this is strictly better than the scalar.** The scalar multiplies `g_plan` by `α∈[floor,1]` — it
attenuates the planner in *all* directions, including the ones that help the WM or are neutral. The
projection removes **only** the component of `g_plan` that points against `g_wm`, and keeps the full
magnitude of everything orthogonal. So where the scalar buys WM safety by starving the planner globally
(v4.1), the projection buys WM safety by **spending only the conflicting fraction** of the planner
gradient — and the diagnostic `seam_frac_removed` (§PR) measures exactly how much that is.

The seam generalises `λ_plan`: at `α=1` `grad_scale` is identity and the projection is a no-op whenever
the tasks agree, so **`seam_project` with a disabled controller is a superset of today's seam**, not a
replacement of the mechanism family. It is a refinement of O-20's lever, not a new lever (§5).

---

## 4. Which published method this adapts — and why not the others

`PUBLISHED` (cited; the mission named these five):

| method | what it does | fit to *our* asymmetric, WM-protecting problem |
|---|---|---|
| **PCGrad** — *Gradient Surgery for Multi-Task Learning*, Yu, Kumar, Gupta, Levine, Hausman, Finn, NeurIPS 2020 (arXiv:2001.06782) | drop each task-gradient's component that conflicts with another's | ⭐ **ADAPTED.** We take its projection and make it **one-sided** (project `g_plan` against `g_wm`; never the reverse) — exactly *"the planner may shape the trunk only where it doesn't hurt prediction."* Standard PCGrad is symmetric (peers); our tasks are not peers. |
| **GradVaccine** — Wang, Tsvetkov, Firat, Cao, ICLR 2021 (arXiv:2010.05874) | de-conflict **and** nudge cosine to a small positive target φ | **EXTENSION (off by default).** If pure PCGrad under-protects (canary in 0.55–0.60), raising φ>0 actively *aligns* the planner toward the WM. One extra scalar; implemented in `deconflict(target_cos=φ)` and pre-registered as the first fallback. |
| **OGD** — Farajtabar et al., AISTATS 2020 (arXiv:1910.07104) | project new-task grads orthogonal to stored reference grads | the **one-sided** idea in continual learning; our seam projection is OGD restricted to a single reference (`g_wm`) and computed live, not stored. |
| **CAGrad** — Liu et al., NeurIPS 2021 (arXiv:2110.14048) / **MGDA** — Sener & Koltun, NeurIPS 2018 (arXiv:1810.04650) | solve a min-norm QP over task weights to find a descent direction for **all** tasks | ❌ **rejected.** Both are engineered to keep making progress on **every** task, planner included. We explicitly do **not** want a guaranteed-planner-progress direction — we want the WM held and the planner subordinate. Their bias is the wrong sign for us, and the QP costs more than the whole seam op. |
| **GradNorm** — Chen et al., ICML 2018 (arXiv:1711.02257) | tune per-task loss **weights** so gradient magnitudes balance | ❌ **rejected as the target.** It is a smarter **scalar** — precisely the family the floor already belongs to. It rebalances *magnitude*; it does **nothing** about *direction*, which is the conflict. Moving beyond scalars is the mission. |

**Verdict: one-sided PCGrad at the seam is the adaptation; GradVaccine(φ) is the one-knob fallback;
GradNorm/CAGrad/MGDA are the wrong bias for an asymmetric protect-the-WM objective.**

---

## 5. Where it hooks in — and what it deliberately leaves intact

**One new module + one splice.** No change to the WM stack, the planner head internals, parity, or the
gate protocol.

1. **`stack/tanitad/train/grad_surgery.py`** (staged here as `grad_surgery.py`): `seam_project`,
   `deconflict`, `_SeamProject`, diagnostics. Pure-torch, unit-tested. Import into the trainer.
2. **`stack/scripts/train_flagship_v4.py :: v4_loss_step`** (the §3 splice): fork `states` via
   `seam_project` before it is handed to `flagship_loss` and `head`/`goal_head`. Feed the head
   `lambda_plan=1.0` (the projection now owns the coupling). ~8 lines. The head's internal
   `grad_scale(states, lambda_plan)` (`flagship_v4.py:211`) **stays** and becomes the **scalar backstop**
   — see below.
3. **`stack/scripts/train_flagship_v4.py :: _training_loop`**: log `_SeamProject.last_diag`
   (`seam_cos_mean`, `seam_frac_conflict`, `seam_frac_removed_mean`) every `log_every` next to
   `gnorm_encoder`/`gnorm_predictor`. This is the pre-registered instrument (§PR); it costs nothing.
4. **CLI** (`build_parser`): `--coupling {scalar,seam,seam+floor}` (default `scalar` = today, byte-
   identical), `--seam-per-sample/--seam-global`, `--seam-target-cos φ`. `preflight_asserts` gains one
   line: `seam` coupling requires `cond_imagination:false` OR a second seam (§7.3).

**What is deliberately preserved (so this stays inside GATE_PROTOCOL and the closed lever door):**

- **The canary controller stays** (`CanaryController`, `v4_curriculum.py`), as a **backstop**, not the
  primary. With `--coupling seam+floor` the controller's down-only floored `lam_mult` multiplies the
  *already-projected* planner gradient at the head's `grad_scale` seam — belt-and-suspenders. With
  `--coupling seam` the controller is inert (floor 1.0) and the projection is the sole coupling control;
  that is the clean discriminating configuration (§PR). Either way O-14 (*"the canary may only move it
  down"*) is preserved: projection also only ever *subtracts* from the planner gradient.
- **The step-10k gate is untouched** (`flagship-v4.card.json`, 8-KILL/5-report). A canary-triggered
  mid-run kill still never happens; only the pre-registered gate may stop a run (GATE_PROTOCOL §1).
- **Lever accounting.** v4's encoder-touching structural levers are `λ_plan` + `strategic` = **2 of 2,
  door CLOSED** (RETRACTION_LOG 07-21 C4). Swapping the coupling *operator* (scalar→projection) at the
  **same seam** is a **modification of lever #1 (O-20), not a third lever** — it changes *how* the
  existing planner→trunk gradient is shaped, adds no new module to the trunk, and touches no new
  activation. This must be stated in the launch card so the door stays provably closed.

---

## 6. The design-space questions, answered

**Which gradients to de-conflict — full-trunk vs last-k-blocks vs predictor-only?**
De-conflict **at the `states` seam**, which *is* the whole encoder's planner-facing gradient (the
encoder is the planner's only trunk path). Therefore:
- **Predictor-only: moot.** With imagination off the planner sends **zero** gradient to the predictor;
  there is nothing to project there. (The canary still degrades because the *encoder's* `states` drift
  off the manifold the predictor was fit to — which is exactly the `g_wm` direction the seam protects.)
- **Full-trunk θ-space PCGrad: unnecessary and expensive.** Running two *separate* full backward passes
  to get `g_wm_θ` and `g_plan_θ` over all trunk parameters, projecting, and recombining is faithful
  PCGrad but costs **~+1 full backward (~+40–50 % step)**. The seam version is the same projection
  pushed to the single activation the planner actually flows through, at ~0 cost. Offer full-trunk only
  as an **ablation reference** if the seam approximation is ever in doubt.
- **Last-k-blocks: a cost variant of the θ-space form**, not needed for the seam (which already covers
  all blocks for free). Keep in pocket only if a second seam for imagination (§7.3) makes θ-space
  attractive.

**Per-layer vs global (per-sample) projection?** At the seam the natural granularity is **per-sample**
(project each window-row's `g_plan` against its own `g_wm`; `--seam-per-sample`, default) — finer, and it
matches per-example PCGrad. **Global** (flatten the whole micro-batch, one projection) is the cheaper,
coarser ablation (`--seam-global`). Per-**layer** projection is a property of the θ-space variant
(GradVaccine does it per parameter-tensor); it does not apply to the single-tensor seam. Both seam
granularities are implemented and finite (`test_global_vs_per_sample_shapes_and_finiteness`).

**Cost per step.** `MEASURED` (math) + `ESTIMATED` (wall-clock):
- Extra compute = the projection arithmetic on `states` `[16,8,2048]`: a handful of reductions and an
  axpy per micro-batch. **≈ 0.1–0.3 % of a step** — negligible. **No extra encoder or predictor pass.**
- vs the alternatives it replaces: the scalar floor is free but blunt; full-trunk PCGrad is +40–50 %;
  the two-backward seam variant (if one ever needed `g_wm` without `retain_graph`) is +1 encoder
  backward (~+10–35 %). The single-backward `_SeamProject` avoids all of that.

**Interaction with grad-checkpointing (F-5) + eff-batch-64.**
- **Grad-checkpointing:** `_SeamProject` sits at the encoder **output**; checkpointing is *inside* the
  encoder blocks. The single backward through the seam triggers the normal checkpointed encoder
  backward — **one recompute, same as today**. Surgery adds **no** extra recompute. (v4.1 ran
  `grad_checkpoint:false`; this holds whether it is on or off.)
- **Effective batch 64 = micro-batch 16 × accum 4:** `states` is per-micro-batch, so the seam projects
  **per micro-batch** and the de-conflicted encoder gradient **accumulates** into `.grad` across the 4
  micro-steps — well-defined and consistent (projection is per-micro-batch, never across differing
  samples). No change to the accumulation loop beyond calling `seam_project` inside it.

---

## 7. Where this might fail — stated plainly (both are pre-registered controls in §PR)

**7.1 If the degradation is the trunk's own re-optimisation, not the planner (class C6).** v4 at
`lr_trunk 3e-4` degraded the canary at `λ_plan=0` — a **single-task** effect (the warm WM re-fitting its
own loss off the v1 optimum) with **no conflict to project**. If, at `lr_trunk 1e-4`, the `λ_plan=0`
control already pushes the canary past ~0.55, then surgery is aimed at the wrong target and the lever is
**lower `lr_trunk` or from-scratch**, not gradient geometry. The §PR `λ_plan=0` control measures this
*first*; the surgery PASS bar is defined **relative** to it.

**7.2 If the conflict is fundamental, not just mis-weighted.** If the WM and planner want **genuinely
opposed** `states` representations, the per-sample cosine sits near −1, projection removes **almost all**
of `g_plan`, and the planner starves exactly as in v4.1 — but now *honestly*, because the directions
truly cannot coexist. The seam does not manufacture a non-conflicting direction that isn't there. The
diagnostic that tells this apart from success is `seam_frac_removed`: **surgery only helps if a large
fraction of `g_plan` survives projection** (a real orthogonal subspace exists). If the canary holds
**and** `seam_frac_removed` is near 1.0, the planner is being starved geometrically → treat as FAIL and
go from-scratch. This is why §PR reads *both* the canary and the fraction removed.

**7.3 If imagination is turned on later.** The one-seam completeness relies on `cond_imagination:false`.
If a future arm enables imagination with `--probe-grad one/all`, the planner also gradients the
**predictor** (through the grad-carrying probe), and a **second seam** at the imagined-latent input to
the head is required to cover that path; otherwise the predictor's planner-gradient is un-projected.
`preflight_asserts` must block `--coupling seam` + imagination-on until the second seam lands. Named, not
silent.

**7.4 Seam-space ≈ θ-space, not =.** Projecting in activation space then applying the shared encoder
Jacobian is not identical to projecting the trunk parameter gradients directly (they coincide only if
the Jacobian is an isometry). This is an **approximation**, justified because (a) the planner's trunk
path is exactly this one activation and (b) the alternative (θ-space PCGrad) costs a full extra backward.
The experiment tests the approximation *empirically* (does the canary hold?), not in theory; if it holds,
the approximation is good enough, and if it doesn't, §7.1/§7.2 discriminate why before we escalate to the
expensive θ-space form.

**Honest bottom line.** Gradient surgery is the right lever **iff** the v4.2 degradation is a
*directional planner-vs-WM conflict with a real orthogonal subspace*. It is **not** a fix for trunk-LR
re-optimisation (→ lower LR / from-scratch) or for a fundamentally opposed objective (→ from-scratch).
The pre-registered experiment is designed to return which of these three worlds we are in, cheaply, with
all three verdicts written down before it runs.

---

## 8. Deliverable manifest

| artifact | where it lives | status |
|---|---|---|
| `DESIGN.md` (this doc) | repo, staged: `…/Architecture & Inference/Implementation/incoming/2026-07-23-planner-wm-gradient-coupling/` | ✅ staged |
| `PRE_REGISTRATION.md` (cheap discriminating experiment, both outcomes committed) | same dir | ✅ staged |
| `grad_surgery.py` (reference impl: `seam_project` / `deconflict` / `_SeamProject` / diagnostics; CPU smoke) | same dir → merges to `stack/tanitad/train/grad_surgery.py` | ✅ staged; smoke green |
| `tests/test_grad_surgery.py` (9 cases: identity, one-sidedness, no-op-when-aligned, conflict removal, per-sample/global, GradVaccine, diagnostics) | same dir → merges to `stack/tests/` | ✅ staged; **9 passed** (venv `C:/Users/Admin/venvs/tanitad`, torch 2.11.0) |
| `INTAKE.md` (integration note for the merge) | same dir | ✅ staged |

**Integration escalation (per the Agent Operating Standard — do not bury a "please merge" in a doc).**
The two-file merge (`grad_surgery.py` → `stack/tanitad/train/`, splice into `v4_loss_step` +
`_training_loop` logging + CLI) is a small, self-contained change gated behind `--coupling seam`
(default `scalar` = byte-identical to today). It must **not** be applied to the pod while v4.2b is
training off that file; it is queued for the **next** v4.x launch that Sayed approves. Priority order if
this stream is cut short: (1) the design + failure analysis (§3, §7) is the durable deliverable; (2) the
tested `deconflict` math; (3) the trainer splice.

**Nothing in this deliverable was launched, and no training pod was touched.**

> ⚠️ ✅ **RE-CONFIRMED STILL TRUE 2026-08-16 — THE MERGE NEVER HAPPENED. 24 DAYS OPEN.**
> Probed at HEAD, twice: (1) `stack/tanitad/train/` contains `__init__.py, ckpt_io.py, decorr.py,
> flagship_losses.py, heldout_gate.py, heldout_goal.py, train_worldmodel.py, v4_curriculum.py` —
> **no `grad_surgery.py`**; (2) by symbol rather than filename, `grep -rn "seam_project\|deconflict\|
> grad_surgery" stack/ --include="*.py"` returns **zero files**. The tested implementation and its
> 9 green tests are still only in this `incoming/` folder.
> **The gating condition this doc names is also void**: the merge was queued for *"the next v4.x launch
> Sayed approves"*, and the programme has since moved past v4 entirely (v5f → v6). Waiting for a v4.x
> launch that will not come is what turns a queued merge into the 10-day-orthogonality-instrument
> failure — so this is escalated in-channel, not left here.
> Code involved: `…/2026-07-23-planner-wm-gradient-coupling/grad_surgery.py` (+ `tests/test_grad_surgery.py`)
> → intended destinations `stack/tanitad/train/grad_surgery.py` and `stack/tests/`.
> Swept by the 2026-08-16 stale-blocker sweep.

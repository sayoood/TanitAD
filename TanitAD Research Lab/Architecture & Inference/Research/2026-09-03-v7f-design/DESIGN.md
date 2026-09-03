<title>DESIGN — v7f: a trainable DINOv3 trunk, shaped by LDAD, on a bounded gradient chain</title>

# DESIGN — v7f

`TanitAD Research Lab · Architecture & Inference FlyWheel · 2026-09-03 · 0 GPU · no pod / no Thor contact`
`Pre-registration: Project Steering/PREREG_V7F.md · SPEC: ./SPEC.md · numbers: ./raw/EVIDENCE_TABLE.md`
`Tier: T0 for every diagnostic below. No claim here is a driving claim; L4/T1 is the only tier at which`
`the word "effective" is admissible, and no v7 arm has ever reached it.`

**The PI's directive this design answers, verbatim:** *"at v7 we ARE allowed to train the trunk —
consider the previous findings, taking DINO as init."*

---

## 1. The design in one table — every choice, its source, its class

| # | choice | traced to | class |
|---|---|---|---|
| 1 | **Trunk = DINOv3 ViT-B/16, TRAINABLE** | PI directive 2026-09-03; `D-TRUNK-FREEZE` (v7 line: TRAINABLE (D1) stands) | PI + MEASURED |
| 2 | **…initialised from the real DINOv3 weights**, not a distillation | `E-DEC-68`: DINOv3 wins **9/9** point estimates over V-JEPA2, 6/9 paired-separated ⇒ DINOv3 is the v7r distill/fallback teacher. `C164`: `--init-from` is **NOT a lever** — `postrain30k` and `splitp30k` share the same `distill_init.pt` and sit **0.47** apart on drift | MEASURED |
| 3 | **ViT-B/16 (768×12) specifically** | config E's encoder is **already** 768×12 (`HANDOVER_TO_LOCAL_2026-08-15.md:76-79`) ⇒ the swap is parameter-neutral at ≈ 85 M | INHERITED (config E) + derived |
| 4 | **Discriminative LR (`trunk_lr_scale` 0.1) + 2,000-step warmup** | Observer Effect `2602.12218`: full fine-tune drives a frozen-feature linear probe from ρ **0.91 → 0.05**; last-layer-only 0.65; damage concentrated in **B5–B10** (δ(l) > 0.10, CKA < 0.2) | PUBLISHED-PRIMARY |
| 5 | **Distillation ANCHOR to a frozen copy of the same weights** | Latent-WAM `2603.24581` Table 4: distillation **INTO the trunk 89.3** > concatenating frozen features 88.0 > none 88.3 | PUBLISHED-PRIMARY |
| 6 | **⛔ NOT LoRA** | Latent-WAM Table 5: **Base-LoRA 68.5** is the WORST row — below Small-LoRA 84.7, Small full 86.3, Base full **89.3** | PUBLISHED-PRIMARY |
| 7 | **Observer-Effect probe as an in-training MONITOR** with a no-information control | `2602.12218`'s own instrument rule: judge a trunk by **LINEAR probes on FROZEN features** with a raw-input baseline and time-dependent probes for dynamic variables; an *invasive* probe hallucinates competence (MAPE 140 % → 18 %) | PUBLISHED-PRIMARY |
| 8 | **LDAD as the ONE new anti-collapse term**, `Δz → (a, κ)` | Delta-JEPA `2606.31232`: `Δz` beats `concat[z_t, z_{t+1}]` on **all four** envs (Table 2); λ sweep {0…1000} with λ = 0 near-collapse and **λ = 50 best** (Fig. 3); Table 5 `Δx` from `Δz` r **0.992** vs LeWM 0.765 | PUBLISHED-PRIMARY |
| 9 | **…decoding `(a, κ)` and EXCLUDING `v`** | `D-V-EXCLUDED` (ours): `(a, κ)` is clean **only for encoder-side decoders**. Independently, Delta-JEPA Table 3 (Reacher): raw action **81.33**, Δjoint 80.47, both 76.40, realised end-effector **Δfinger 64.93** — the target closest to the commanded quantity wins by 16 points | MEASURED + PUBLISHED-PRIMARY |
| 10 | **The action NEVER enters the trunk** (conditioning stays at the predictor's `_lift3`) | `2606.07687` Table 6, the **action-conditioning paradox**: feeding the action into the trunk drops action-relevant R² **0.26 → −0.32**. The published form of our MM-E17/E18 (attenuation 56×/128× located in FiLM; gain converged, not starved) | PUBLISHED-PRIMARY + MEASURED |
| 11 | **Forward horizon `o5_k = 60` (the 6 s contract) with a BOUNDED gradient chain** | ours: k = 60 diverged at gnorm **2.1e9**, killed at 9,000; the clip-0.5 successor spiked **1.24e10** at 18k. Field: V-JEPA 2-AC `2506.09985` §3.1 differentiates through **ONE** recurrent step (T = 2); What-Drives-Success `2512.24497` back-props only through the last prediction and finds **K = 2** optimal in sim / **K = 6** on DROID with a decline beyond | MEASURED + PUBLISHED-PRIMARY |
| 12 | **Gradient depth 4, not the inherited 15** | 4 = 2× the largest published gradient depth; 15 = 7.5× it, and is the **refav1** value carried into the flag suggestion by `D-V7-WIRING`. Decided by rung R1, not by inheritance | derived from PUBLISHED-PRIMARY |
| 13 | **Checkpoint selection: sensitivity filter, then one-step embedding loss** | Dueling WM `2608.06706`: val-loss selection picks action-collapsed checkpoints **17/36**; separation collapses **1.28 → 0.002** *while validation loss improves*. What-Drives-Success Tables 13–16: the **one-step** embedding loss is the best proxy for planning success (mean −ρ 0.47–0.86) | PUBLISHED-PRIMARY |
| 14 | **Anchored (zero-action) actdiv as the gate metric** | `GS-8`; ActSWM `2607.26712` Table 5 / Delta-JEPA Fig. 6 / AD-JEPA form — the only formulations with published before/after effect sizes. Our banked ratio has **no published counterpart** and floats with its scene denominator | PUBLISHED-PRIMARY + MEASURED |
| 14b | ⛔ **…with the denominator PINNED to a named reference arm, co-primaries reported, a WORSE branch and an interval** | `PREREG_MM_E19_K60_HORIZON.md:74-105` — **DEFECT 1** (the bar floated: `scene_factor` 1.7126 turned a 10× bar into **17.13×**, *"a criterion whose threshold is a function of the result is not a pre-registration"*, and it is sign-blind the other way — a doubled sensitivity would report **+17 %** and be written up INERT); **DEFECT 2** (the outcome set admitted no worsening; the arm fell **0.48×**, off the table); and the four co-primaries the same page mandates — action side alone, own scene spread (it carried **73.7 %** of MM-E19's ratio fall), the **h ≥ 2 floor** (~1e-05 on every arm, horizon-independent), and the **training-window count** (`o5_k` also sets `o4_n`: **415,002 → 319,002** = 76.9 %). Plus backlog **L-13**: the actdiv probe has **no interval** — *"a decision statistic without an interval is a gap, not a virtue"* | MEASURED (programme's own prereg-defect finding) |
| 15 | **Trivial-profile and `baseline_won_frac` before any family row** | `D-REFAV1-PAIRED-READ-VOID`: two checkpoints differing by a full recipe produced **bit-identical driving on 140/140** windows because the deployed plan was the CV baseline; `baseline_won_frac` 0.7571 / 0.7500 | MEASURED |
| 16 | **A `ha0` control (straight line at `v0`) beside hold-action** | `AMENDMENT TO D-REFAV1-STEP1000-READ`: hold-action drifts **0.12 m** even where the human drives straight and **0.76 m** on curved windows, while the straight line reads 0.035 / 0.496 — so a control weaker than a trivial baseline manufactured a "lateral planning gain" | MEASURED |
| 17 | **Everything above the operative level: v7r unchanged** | `V7R_DESIGN_PROPOSAL.md` (PI-closed 2026-08-27): Alpamayo meta-action tactical vocabulary, latent-subgoal matching, anchored-diffusion planner in `UnicycleAccelCurvatureActionSpace`, v1.7 decoder recipe, X3 isolation | PI-closed |

---

## 2. Why a trainable trunk is the right answer to a question the frozen trunk cannot answer

Three facts, and they compose.

**(a) Every intervention we have tried was a PREDICTION lever, and the field measures that prediction
levers do not move action sensitivity.** ActSWM `2607.26712` Table 5: with context H 3 → 32 and
multi-step training the 31-step rollout cosine goes **0.239 → 0.972** — a 4× improvement in
prediction — and the zero-action **gap stays ≤ 0.002**. Only an explicit sensitivity term moves it
(0.002 → 0.760). Our k, O14, EMA and τ-ramp arms are that same class of lever, and MM-E10's ratio has
sat at 0.004–0.006 across all of them.

**(b) The published fixes PARTITION, and the encoder-shaping half is the half a frozen trunk forbids.**
LDAD/Delta-JEPA, SMWM and EB-JEPA's IDM all decode the action from quantities that are **encoder
outputs**; on a frozen trunk they have nothing to shape. ACID, ActSWM's hinge and AD-JEPA's anchoring
act on the predictor and survive a frozen trunk. `D-SOTA-PASSES-2026-09-03` confirmed the partition
from the PDFs, and found **no falsifier**: no banked primary shows a predictor-side consistency loss
FAILING under a frozen encoder.

**(c) Our own strongest evidence for the mechanism came from freezing.** `postrain30k_freeze` — the
one-variable freeze cell — is **DEGENERATE** on prediction (held-out nrmse 0.9301 vs 0.8115, **+14.6 %**,
E-DEC-64) and, measured at the trained lift by the leak audit, its **command channel is dead**:
steer/accel ratio **0.00078** against the trainable incumbent's **0.00561** (7.2× lower; paired
command-channel factor **0.45 [0.28, 0.72]**), with its scene spread 3.2× higher. **Freezing made the
action pathway deader.** That is `GS-10`'s encoder-shaped-sensitivity hypothesis read from the other
side, and it is the single most direct argument in our own data for the PI's directive.

---

## 3. …and why that does not license a naive fine-tune

**The Observer Effect is measured, large, and specifically about the variables we care about.**
`2602.12218`: a linear probe on **frozen** features reads ρ **0.91**; the same trunk **fully
fine-tuned** reads **0.05**; last-layer-only reads **0.65**. On a *kinematic invariant* the correlation
goes **0.94 → −0.03**. The damage is concentrated in the **deep blocks** and it destroys **time-varying**
invariants (Speed, Radius) while sparing static ones (Mass). Our own L2 pattern is the same shape:
`splitp30k` gains on `n_agents` (**+0.3881**) and **loses** on `lead_range_m` (−0.1611) and `d_ego`
(−0.0399) — static kept, dynamic/metric erased.

**And the driving-level version exists too.** FROST-Drive `2601.03460`: a **frozen** 14 B VLM encoder
scores RFS **8.17** (ADE@3 s **1.04**); the **same encoder fully fine-tuned** scores 8.13 (ADE **1.47**).
A frozen ImageNet ViT scores 7.39 and its fine-tuned self 7.79. ⇒ **fine-tuning HURTS a strong trunk
and HELPS a weak one** — and DINOv3 ViT-B/16 is on the strong side of that line.

⇒ **The design must be the middle, and the middle is not LoRA.** Latent-WAM `2603.24581` Table 5 puts
**Base-LoRA at 68.5** — the worst row in a table whose best is a **full fine-tune** of the same Base
trunk at **89.3**, with **distillation INTO the trunk** as the thing that made the full fine-tune work
(Table 4: 89.3 with distillation vs 88.0 for concatenated frozen features vs 88.3 for none).

**v7f's middle, therefore:** full trainability, a **low trunk LR with a warmup**, a **distillation
anchor to the frozen copy of the same weights**, and — the part that makes it a design rather than a
hope — **the Observer-Effect probe running as a MONITOR at every checkpoint**, with a committed floor
(0.70 × step-0), its own no-information control (a time-shuffled feature stream that must read the
constant value), and a **deliberate-regression arm** (the naive `full` fine-tune) that the monitor must
trip. *If the monitor does not trip the naive arm, the monitor is void and the anchored arm's pass
means nothing.*

⚠️ **One honest caveat on the anchor.** JEPA-x `2608.24044`'s `DISTILL` control arm read **0.516** on
its drift metric against a baseline of 0.361 — *worse*. A distillation term is not automatically
benign. This is exactly why `w_trunk_anchor` lives inside the `anchored` arm and the `full` arm
(anchor = 0) is its one-variable control, rather than the anchor being added everywhere as hygiene.

---

## 4. The one new loss term: LDAD on `Δz`, targets `(a, κ)`, `v` excluded

### 4.1 Why LDAD and not SMWM or EB-JEPA's IDM

All three are encoder-shaping and all three are unlocked by a trainable trunk. They differ in **shape**
and in **how much their λ can be trusted**:

| | LDAD (Delta-JEPA `2606.31232`) | SMWM `2606.20104` | EB-JEPA IDM `2602.03604` |
|---|---|---|---|
| input to the decoder | **`Δz_t = z_{t+1} − z_t`** | `[z_t; z_{t+1}]` concat | `MLP(z_t, z_{t+1})` concat |
| is the shape ablated? | ⭐ **YES** — `Δz` beats concat on **all four** envs: +4.07 / +1.07 / +12.60 / +0.67 (Table 2) | no | no |
| λ evidence | sweep {0…1000}; **λ = 0 near-collapse, λ = 50 best** (Fig. 3); λ = 10 in the main tables | ⚠️ **environment-specific over 300×**: Two-Room 0.1, Reacher 5, Push-T 30, Cube 1 — and **λ = 0.1 collapsed Reacher** | ablation only: ω = 0 → **1 ± 1 %** (total collapse) |
| our leak exposure | the `Δz` form is the one `GS-1` does **not** flag | ⛔ the **endpoint-concatenation** shape `GS-1` was raised about | ⛔ same concat shape |

⇒ **LDAD.** It is the only one whose form was tested against the alternative, its λ has an in-paper
optimum rather than a per-environment lottery, and its shape is the one our own leak audit does not
flag. ⚠️ **The λ caveat travels anyway**: SMWM's 300× spread is the field's own warning that λ is
environment-specific, so ours is **swept on the tiny rig on the FIT split** (R2), never assumed.

### 4.2 What it decodes, and the proof it is not decoding the target from itself

- **Decoder input:** `Δz_t = z_{t+1} − z_t`, both terms **encoder outputs** on images alone. The
  encoder's forward signature is `[B, C, H, W] → [B, N, D]` (`encoder.py:97`, `:319`); no action, no
  ego channel, no speed reaches it. Conditioning enters at the **predictor** via `_lift3`
  (`train_v6_staged.py:3604-3646`).
- **Decode target:** `(a_long, κ)`. ⛔ **`v` is excluded at every level — level AND Δv** (`D-V-EXCLUDED`).
- **Therefore the self-decoding hazard is structurally excluded**, and this is precisely the
  distinction `D-V-EXCLUDED` draws: `(a, κ)` is clean *for encoder-side decoders* (`idm_head`,
  `InverseDynamicsHead`, LDAD/`Δz`) and dirty for **predictor-output** decoders (O13), whose input
  already consumed the action — *"any decode on the predictor's output must be action-free by
  construction or dropped."*
- ⚠️ **The residual confound, named rather than hidden.** `(a, κ)` is the ego's **realised motion**
  (`E-DEC-57`: `v·tan(steer)/L` reproduces the measured yaw-rate at r **0.9988** on 20 clips,
  **0.9664** on 129). Two consecutive images of a moving camera contain that motion, so
  `Δz → (a, κ)` could be **visual odometry** rather than action representation. That is **not a leak**
  (VO from vision is legitimate, and inference stays vision-only) but it *is* a confound for the
  claim. Hence three mandatory controls on every LDAD panel:
  1. **shuffled-action** — `Δz_t` paired with `(a, κ)` from another window; must read the
     no-information value;
  2. **endpoint-shuffled** `Δz′_t = z_{π(t)+1} − z_t` — the `H-LEAK-3` control, which reproduced
     **102 % / 100 %** of what the programme was calling "drift"; if LDAD survives on `Δz′`, it is
     arithmetic, not a transition property;
  3. **raw-pixel-difference floor** — the VO check. A learned representation that does not beat the
     raw input has added nothing.
- ⭐ **And the whole line can be refuted at zero cost before an arm is spent.** `GS-2 (2)`, committed:
  **fit the LDAD decoder at step 0 on the DINOv3-init trunk.** If the loss is already at the floor,
  the objective is **tautological on our channel** and the line is REFUTED for TanitAD. That is rung
  **R0** in the prereg and it runs before anything else.

### 4.3 Where LDAD attaches, and the slot that keeps the successor reachable

LDAD proper is encoder-side and needs `z_true_steps`, which the trainer already holds per window. In
parallel, `D-M_T-SLOT` reserves the **predictor-side** attachment at zero cost: keep
`cfg.residual = True`, expose `delta` beside `out[k]` (`predictor.py:285-289`), keep the
`(win_s[:, -1], z_hat)` pairs (`metric_dynamics.py:279`). `m_t` and LDAD **compose rather than
compete** — H-RANK-17's additive motion latent makes the displacement explicit, LDAD supervises it —
and the reservation is what keeps the line reachable on refav1, where `Δz` is a constant of the
feature cache and only the `m_t` side exists.

---

## 5. Horizon: a 6 s forward contract on a short gradient chain

| | |
|---|---|
| **the contract we keep** | `--o5-k 60` — the 6 s rollout, because the corpus' median manoeuvre is **12.5 s** and the trained horizon at k = 8 is **0.8 s**. Shortening the *forward* horizon abandons the hierarchy's premise |
| **the thing we cut** | the **gradient chain**. `--bptt-truncate N` detaches the carried WINDOW every N steps inside `rollout_transitions`; the forward pass is **bit-identical**, only the back-prop path is cut, and the last step is never detached (`D-V7-WIRING`, property-tested) |
| **why the field agrees** | **every** published multi-step rollout loss is truncated and short: V-JEPA 2-AC `2506.09985` §3.1 *"only differentiate the predictor through one recurrent step"* (T = 2); What-Drives-Success `2512.24497` §C back-props through the **last** prediction only, with K = 2 optimal in sim and K = 6 on DROID and a **decline** beyond (Fig. 3b), and proves the mechanism — Remark 1: the K-step loss lowers the effective Lipschitz constant along the rollout **at the price of one-step accuracy**; EB-JEPA K = 8 with a Pareto at **4** |
| **why 4 and not 15** | 15 is the **refav1** value carried into `D-V7-WIRING`'s v7f flag suggestion. It is **7.5×** the largest published gradient depth. 4 is 2× it — deliberate headroom rather than inheritance. `--bptt-truncate` must be `< --o5-k` (preflight-enforced) |
| **what R6 does and does not block** | `--bptt-truncate` reaches the **O5/O11** rolls, not the O1 stage-A rolls. ⭐ **MEASURED from source: it does not matter for v7f** — `train_v6_staged.py:3691` guards the entire O1 block on `if w.o1_ctrl or w.o1_fact or w.o1_scene:`, and v7f runs all three at **0**, so the six `stage_a_losses` rollouts are never reached. R6 becomes a precondition only if O1 is ever turned on |
| ⚠️ **the confound to hold constant** | the window census depends on `max_horizon`, derived from the stage's live loss horizon (`train_v6_staged.py:3188-3214`) — an episode shorter than `window + max_horizon` contributes **ZERO** windows. Changing `o5_k` changes the **training-window count** (the L-10 confound). `o5_k` is fixed at 60 for every rung except R1, and R1 reports its per-arm window census |

---

## 6. Evaluation: the refav1 lesson, transferred

refav1's first real T1 read produced a **VOID** paired comparison: two checkpoints differing by a full
recipe drove **identically to 1e-9 on 140/140 windows**, because the deployed plan was the
**constant-velocity baseline** — a function of the measured `v0` alone, not of any weight.
`baseline_won_frac` 0.7571 vs 0.7500; every closed-loop trajectory had `y ≡ 0` and constant speed on
140/140. And the read that *looked* like planning skill — cross-track −18 cm, heading −2° over
hold-action — was an artefact of a **control weaker than a trivial baseline**: hold-action drifts
0.12 m even where the human drives straight and 0.76 m on curved windows, while the straight line
reads 0.035 / 0.496.

**Every v7f T1 read therefore reports, BEFORE any family row:**

1. `trivial_profile_fraction` (`tools/straight_line_probe.py`) — `== 1.0` ⇒ the read is **VOID**;
2. `baseline_won_frac` and the `cem` share from the planner's own provenance — not below the refav1
   reference **0.7571** ⇒ no planning claim is available;
3. the paired **bit-identity** check (`tools/paired_dump_compare.py`) — arms identical on the window
   grid are stamped **VOID**, never reported as "no difference";
4. a **`ha0`** arm (straight line at `v0`) beside hold-(a, κ), because the strongest trivial baseline
   is the only honest floor.

Only then: the **four families**, per family, each with the paired episode-cluster bootstrap
(`taniteval/ci.py`, `full_set` mean — never `overlapping_holdout_se`), on the **v7.2 labelled eval
split** (141 clips with pixels), never val40.

⚠️ **A precondition inside the instrument:** `C-REFAV1-KIN-CONTRACT-LAT` — the open-loop replay of the
RECORDED `(a, κ)` through the refav1 adapter's unicycle misses the human's lateral position by
**0.716 m** on curved windows, *more than the human's own excursion* (0.496 m). Whether the v7 T1
adapter shares that defect is **UNVERIFIED** and must be checked **before** G-DRIVE. Until it is,
no lateral row would be quotable.

---

## 7. ⛔ WHAT WE ARE **NOT** DOING, AND WHY

### 7.1 NOT a frozen trunk (for the v7 line)

- Our own one-variable cell is **DEGENERATE**: `postrain30k_freeze` = `postrain30k + --freeze-encoder`
  only, held-out nrmse **0.9301 vs 0.8115 (+14.6 %)** (E-DEC-64).
- And it **killed the command channel**: steer/accel ratio **0.00078** vs **0.00561** (7.2× lower;
  paired factor **0.45 [0.28, 0.72]**), which is the opposite of what v7f needs.
- ⚠️ **Two honest qualifications.** (i) The *drift* half of E-DEC-64's verdict is now uninformative:
  `H-LEAK-3` showed an endpoint-shuffled control reproduces **102 %** of "drift", so the 0.3905 that
  made freezing look good measures mean-reversion of a change score on its own start point. The
  **nrmse** half is a different instrument (`meanpred.py`) and is untouched. (ii) Our freeze arms froze
  a **distilled tiny** trunk, not a large pretrained one — the field's frozen-trunk successes are on
  14–78 B VLMs / ViT-g / ViT-L, so *"REF-A's frozen ceiling"* is **not** evidence against freezing a
  strong trunk. That is why the **`frozen` arm stays in R3 as a control**, and why the design is a
  *test*, not a foregone conclusion.
- ⇒ **Frozen stays the fallback**, and the fallback has a route with numbers: a per-token
  bisimulation-style adapter on frozen DINOv2 patch tokens raised PointMaze success under
  lighting+colour+geometry shift **0.48 → 0.78**, encoder-agnostic (iBOT 0.72), at **0 trunk compute**
  (`2602.18639`).

### 7.2 NOT a naive full fine-tune

Section 3. The naive arm exists **only** as the deliberate-regression arm the Observer-Effect monitor
must trip (ρ 0.91 → 0.05). ⛔ And **not LoRA either**: Base-LoRA is the worst row in Latent-WAM's own
table (68.5 vs 89.3 full).

### 7.3 NOT a flat k = 60

- Ours: gnorm **2.1e9**, killed at 9,000; the clip-0.5 successor spiked **1.24e10** at 18k.
- Field: no primary back-propagates a rollout loss through more than one recurrent step at K > 6.
- ⇒ 6 s **forward**, gradient depth **4**, and a pre-registered fallback to 2.
- ⚠️ And the horizon was never the action lever anyway: MM-E19's k = 60 arm did not clear the bar
  (ratio 0.82×), and at the **trained** lift the leak audit reversed only the *attribution*
  (steer/accel spread **1.41× [1.03, 1.78]**, scene spread 1.72×), not the verdict. `SMAS-1`'s *"the
  action response fell 17 %"* is **withdrawn**; the k = 60 verdict stands.

### 7.4 NOT the levers the leak audit invalidated or re-classified

| lever | why it is out |
|---|---|
| **O11 counterfactual-action InfoNCE** | ⛔ **class-(i) positive by construction**: the positive action sequence **is** the target's realised motion (r 0.966–0.999), the degenerate solution `ẑ = f(z) + λa` is named in the loss's own docstring (`train_v6_staged.py:1370-1375`) and was **measured** on `o11p30k` (`shuffle_all` d_out 0.51 vs latent control 0.43; `zero_v` **1.64** — the response is dominated by `v`). The "breakout" is not evidence of learned dynamics. The discriminating actdiv read is **QUEUED**; until it clears, O11 stays at 0 |
| **the banked `actdiv` ratio as a gate** | ⛔ **two independent defects.** (1) `H-LEAK-1`: the instrument fed `v/30` where every arm trained on `v/10`, and it **rolls the speed-STATE channel with the actions**, so the legacy ratio is partly a speed-sensitivity read and moves **3–5×** with the lift. (2) `PREREG_MM_E19_K60_HORIZON.md` **DEFECT 1**: the ratio's denominator is the arm's **own** scene spread, so the threshold is a function of the result — the k = 60 arm's `scene_factor` 1.7126 turned a written-down 10× bar into a **17.13×** one, and an arm that genuinely doubled its sensitivity would read **+17 %** and be written up INERT. v7f gates on the **anchored** read (holds `v` fixed, immune to (1)) **with the denominator PINNED to a named reference arm** (immune to (2)), reports the four mandated co-primaries, carries a **WORSE** branch, and reads an **interval** (backlog L-13) |
| **drift as a gate (P3)** | ⛔ `H-LEAK-3`: an endpoint-shuffled control reads **102 % / 100 %** of "drift". Readiness decision **G.1** (*accept P3 as met by `postrain30k_freeze`, drift 0.3905*) is **not to be adopted on that metric**. Drift stays a **diagnostic**, and the field agrees it is not the capability-limiting quantity (What-Drives-Success: the best proxy for planning success is the **one-step** embedding loss) |
| **the L3 t-statistics** | ⛔ `H-LEAK-2`: `envpred.loeo` scores a pooled 12-clip second half per iteration and divides by √24 as if the scores were independent (~92 % overlap). **L3 has never been passed or failed on an admissible read.** ⇒ v7f does **not** gate on L3 as currently instrumented; the estimator fix (0 GPU) is a precondition for any L3 claim |
| **`--init-from` as a lever** | ⛔ **C164**: `postrain30k` and `splitp30k` share the same `distill_init.pt` and sit 0.47 apart on drift. Harmless to keep; it buys nothing it was credited with |
| **O13 ego-dynamics decode on `ẑ`** | ⛔ `D-V-EXCLUDED`: the head sits on the **predictor's** output, which consumed the actions; `(a, κ)` as a target there re-creates the echo. Action-free by construction, or dropped |
| **UWM-JEPA counterfactual targets** | needs a **simulator**; not available on our corpus. Its *diagnosis* is ours (teacher-forced target ⇒ ‖H1‖/‖H0‖ ≈ 0.03) but its fix is not portable |
| **the EMA τ-ramp** | gate CLOSED **NEUTRAL** 2026-08-30: every delta ~20× smaller than the recipe's only seed spread. τ stays **fixed at 0.996**; `--ema-decay-ramp` is DROPPED |

### 7.5 NOT gating on rank alone, and NOT on the bare 8.56

⛔ **Rank is NECESSARY, NOT SUFFICIENT (C131)**: flagship v1-era had the highest rank ever measured in
this programme and no environment interpretation — it was conditioned on future GT points and fails
closed-loop. And ⛔ **the 8.56 floor is not reproducible**: through `spectrum_report` itself, frozen
DINOv3 ViT-L/16 reads **5.756** on the 12 clips 8.56 is sourced to and **20.228** on the 130-clip
corpus 40.77 is sourced to (`MODEL_REGISTRY.md` §13.3, `v6.py:1517-1552`, pinned by
`test_participation_floor_provenance.py`). G-RANK is re-specified as a **matched-corpus,
matched-episode-count, matched-`d`** comparison against a reference this run measures itself.

### 7.6 NOT quoting a number outside its scope

Two live examples this design deliberately avoids: `E-DEC-49`'s echo-cleared **r 0.326** is a 20-clip
number (**0.63–0.79** corpus-wide), and `E-DEC-57`'s **0.9988** is **0.9664** on 129 clips. Where this
document quotes either, it quotes both with their `n`.

---

## 8. What v7f does NOT attempt, and what it hands to the next run

- **Strategic ARGS are unsupervised** — the v7.2 labels carry a named dict and no slot encoder
  (BACKLOG R9, Data FlyWheel). The strategic head trains on ids only.
- **The tactical CE term on the factored v7.2 labels is NOT enabled.** The labels land in the batch
  (`tac_lat_id` / `tac_lon_id` / `tac_valid`) and **no loss reads them yet**; adding one is a NEW loss
  term and goes through `TanitAD_ValidateAIDesign` as its own one-variable arm (BACKLOG R7), never as
  a silent addition inside v7f.
- **h ≥ 2 heads.** Only head 1 is trained (MM-E14), and the predictor collapses horizons ≥ 2
  (`max|h1−h2| = 1.211` vs `max|h2−h4| = 0.0018`). ⛔ **This is a design constraint on every
  predictor-side fix in the literature** — ActSWM's hinge needs K ≥ 2, ACID's cost is H-step — so if
  G-ACT fails and the programme moves to the predictor-side family, **a K ≥ 2 rollout is that
  family's precondition** and the curriculum is how it is reached.
- **REF-D** does not exist; the L5 dominance comparison is against REF-C and the frozen-DINOv3
  fallback WM only.

---

## 9. Evidence-class ledger for this document

| block | class |
|---|---|
| §1 rows 1–3, 8–9, 11, 14–16; §2(c); §7.1–7.4 | **MEASURED** (ours) — register rows / registry / RESULT paths in `raw/EVIDENCE_TABLE.md` |
| §1 rows 4–7, 10, 12–13; §2(a)(b); §3; §4.1; §5 | **PUBLISHED-PRIMARY** — banked PDFs, `lib:<key>` + table, read in full by the Research Lab's four SOTA passes (`D-SOTA-PASSES-2026-09-03`) |
| config E = 336.5 M; v7-tiny ≈ 19 M; refav1 = frozen DINOv3 patch tokens | **INHERITED** — `HANDOVER_TO_LOCAL_2026-08-15.md:76-79`; registry §13 scope note; readiness report §D. refav1 has **no registry row** |
| existence of a DINOv3 ViT-S/16; the DINOv3→`ViT5Encoder` weight mapping; per-arm cost of a ViT-B trunk on the tiny rig; whether the v7 T1 adapter shares `C-REFAV1-KIN-CONTRACT-LAT` | ⛔ **UNVERIFIED** — flagged in the PREREG as decisions or preconditions, never used as premises |

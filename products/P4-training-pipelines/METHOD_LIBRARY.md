# P4 — FRONTIER METHOD LIBRARY (RL · preference optimisation · imitation)

`Written 2026-08-23 by the P4 method-library agent for TanitAD_TrainingFlyWheel.
This is a SPEC + SURVEY, not an implementation. Nothing here was trained; 0 GPU spent.`

**Reading rules in force.** Model facts come only from `Project Steering/MODEL_REGISTRY.md`
or raw eval JSON. Every claim carries an evidence class ∈ **MEASURED / PUBLISHED /
INHERITED / ESTIMATED / HYPOTHESIS**. Every paper cited here is **banked**
(`TanitAD Research Lab/Library/papers/`, sha256-verified — `kb_add.py --verify`:
*67 entries, 0 problems*, 2026-08-23); the manifest is §6.

> ## ⛔ ESCALATION — ONE DECISION IS NEEDED, AND IT IS NOT "WHICH OPTIMISER"
>
> **The mandate asks which of PPO / GRPO / DPO / GSPO we should carry. The honest answer
> from our own measurements is: none of them changes anything today, and the reason is
> structural rather than a matter of effort.** RL exists to handle a reward that is
> **black-box / non-differentiable** or a state distribution that is **policy-induced**.
> At every site where we currently have a usable signal we have **neither** condition:
> our cost is a differentiable function of our own emitted trajectory (§2.2 proof), and
> our states are logged. Where the conditions *would* hold — a faithful closed loop with
> non-differentiable safety outcomes — **we have no environment** (§1.3).
>
> ⇒ **The decision the PI/Master Mind must take is not an algorithm choice. It is whether
> to fund a SIGNAL: (B4) a preference-collection instrument, (B5) the `obstacle.offline`
> ingest that makes a safety reward constructible, or (B6) TanitSim.** Until one of those
> lands, the whole RLHF/GRPO/DPO family is `REJECT` or `ADOPT-LATER (blocked)` — and
> saying otherwise would be inventing a mapping to make a method look adoptable.
> The three items that are **`ADOPT-NOW` and need no new signal** (B1, B2, B8) are all
> *audits and cheap structural fixes*, not new machinery.

---

## 1. OUR STACK, AS THIS SURVEY NEEDS IT

### 1.1 The five sites every row below must name

| id | site | code | what it optimises today |
|---|---|---|---|
| **S1** | **trunk / world model** (stage **S-W**) | `stack/scripts/train_v6_staged.py:2149 v6_loss_step`; `stack/tanitad/models/v6.py` | self-supervised latent prediction, terms **O1–O6** (`o1_ctrl`, `o1_fact`, `o1_scene`, `o2_nearfield`, `o3_masked`, `o5_rollout`, `o6_sigreg`). **Emits no action. Has no reward.** |
| **S2** | **planner / candidate scoring** (stage **S-T**) | `v6.py:5097 V6Stack.emit` → fan `plan["waypoints"] [B,N,60,2]`; scorer `GoalDistanceScorer` (`v6.py:2323`, **+267 params**) or `MLPCandidateScorer` (`v6.py:2422`, **+33 801 params**); losses `lambda_plan` (WTA) + `w_select` (`softade`) at `train_v6_staged.py:2416-2478` | ⭐ the only site with a per-candidate cardinal cost |
| **S3** | **tactical decision / vocabulary head** (S-T) | `goal_head_tac` / `act_head_tac`, factored `e_a_tac = [LAT‖LON]`; `_s2_family` (`:1312`), `t2_contrastive_loss` (`:1454`), `anchor_goal_loss` (`:1171`) | discrete token + args, CE / metric-aware objectives |
| **S4** | **strategic layer** (stage **S-S**) | `goal_head_str`/`act_head_str`, `s2_goal_loss` (`:1389`), `s1_rollout_loss` (`:2039`) | discrete strategic token, 0.5 Hz, `stride_str=20` |
| **S5** | **deployment-time search** (no training) | `taniteval/planner_p2.py` (⛔ uncommitted, lived on the terminated eval pod), `MpcRefiner` (`--mpc-refine`) | CEM / MPC over the frozen predictor |

A **"stage"** here is one of `S-W → S-T → S-S → S-J` (`STAGE_PRECONDITION`,
`train_v6_staged.py:335`), each with a frozen-module set, a `STAGE_MAY_INTRODUCE`
allowlist and a `STAGE_GATE_SPEC`. A **"loss"** is a keyed term in `V6LossWeights`
(`:186`) whose weight is **0.0 by default** unless it is an incumbent term — so any
method below that adds an objective adds *one weight, defaulting to zero*, and its
units must be declared (the file enforces this: metres vs nats vs m/s² are never
interchangeable).

### 1.2 ⭐ THE SIGNAL LEDGER — what could play the role of "reward" or "preference"

This is the crux the whole survey turns on. **Have-it** is judged against the episode
contract, the registry, and the fleet — not against what a method wishes for.

| # | candidate signal | what it actually is | have it? | evidence |
|---|---|---|---|---|
| R1 | **expert action** `(steer, accel)` @10 Hz | `actions [T,2]` in the episode contract | ✅ **2 376 eps / 406 099 windows**, parity key `physicalai-train-e438721ae894`, skip-hash `f09e44db` | MEASURED — registry §0.1 |
| R2 | **expert future trajectory** | derived from `poses [T,4]` = (x,y,yaw,v) | ✅ — this is `batch["plan_target"] [B,60,2]` | MEASURED — registry §0.1 |
| R3 | ⭐ **per-candidate cardinal cost** `err[b,i] = mean‖wp_i − gt‖` | computed **offline, exactly, for every candidate in the fan** | ✅ **already computed in the live trainer**, `train_v6_staged.py:2423` | MEASURED (code at HEAD) |
| R4 | **comfort / progress reward** (accel², jerk², steer-rate², progress) | a function of a candidate's **own kinematics** — needs no environment | ✅ constructible (v1.6's barrier terms; the P2 cost) | MEASURED — registry §5, §1.10 |
| R5 | **safety reward** (collision / off-road / infraction) | needs a counterfactual outcome | ❌ | episode contract is *frames / actions / poses / episode_id / maneuvers only* (`tanitad/data/_contract.py:8-12`, quoted in `build_obstacle_join.py`) |
| R6 | **lead gap / TTC** | `obstacle.offline` **exists in the raw dataset (96.90 % of our corpus)** but **was never ingested**; a join script exists and a **26 k-record** join was produced; distance-keeping was computed on **2 846 lead windows** of the oodval corpus | ⚠️ **PARTIAL — offline join only, not in the trainer's batch** | MEASURED — `stack/scripts/lead_state_gate.py` docstring; `build_obstacle_join.py`; registry §1.9 |
| R7 | **safety signal read from OUR OWN latent** | a learned reward head on `z` | ⛔ **NO** — **LF0: in 81.4 % (encoded) / 92.3 % (predicted) of windows where GT has a lead vehicle in the ego corridor, the decoded BEV has no occupied cell there at all**; when it fires the read is off by **26.85 m / 42.65 m** on a 60 m grid | MEASURED 2026-08-12, registry §LF0 (n = 129 labelled) |
| R8 | **human preference pairs over trajectories** | two trajectories + a human ranking | ❌ **none exist anywhere in the programme** | absence probed: `grep` over repo (§6 note) + no instrument in `taniteval/` |
| R9 | **AI (VLM) preference / semantic labels** | `Sayood/tanitad-alpamayo2-augmentation`: **23 644 rows = 4 729 clips × 5 tasks**; PH1-fused hierarchical layer | ⚠️ **PARTIAL and measured-weak**: the fused layer ships a **named 57.2 % perception hole**; **G1 sign-OCR closed at 0/31 verifiable** | MEASURED — registry §11.1, §11.2; `G1_RESULT.md` |
| R10 | **on-policy rollouts in a FAITHFUL environment** | a simulator our policy can act in | ❌ **not today** — see §1.3 | MEASURED |
| R11 | **on-policy rollouts in IMAGINATION** | `taniteval/closedloop.py`, the T1 harness | ⚠️ **exists, and MEASURED HARMFUL as a training signal** — §1.4 | MEASURED |

> ### ⭐⭐ THE ONE-PARAGRAPH ANSWER TO "WHAT PLAYS THE ROLE OF THE PREFERENCE PAIR?"
>
> **Nothing does, and the reason is structural: a demonstration corpus contains exactly one
> policy's output per state.** Preference learning (DPO, IPO, KTO, ORPO, SimPO, CPO, RLHF)
> requires *two* candidate outputs and a judgement of which is better. Our corpus has the
> human's trajectory and nothing to compare it against — **there is no negative class in a
> corpus of expert driving.** A negative *can* be manufactured by sampling our own model's
> fan, but then the label "which is better" is exactly R3, the **cardinal** GT distance —
> and reducing a cardinal cost to a binary preference is **MEASURED to cost us money**:
> E-OBJ-1 swapping the ranking objective from metric-aware `softade` to a one-hot CE was
> **+0.0974 m (base) / +0.1670 m (XL) separated WORSE**, and *softening* the CE target was
> **separated worse at every τ** (MEASURED, LOEO, paired cluster bootstrap —
> `V6F_PLANNER_DESIGN.md` §2.2). ⇒ **the pairwise reduction is strictly lossy on the one
> signal we actually own.** For **reward**, the only thing we can compute for an arbitrary
> trajectory without a simulator is **distance-to-expert (R3) plus own-kinematics comfort
> (R4)** — i.e. an *imitation* reward and a *smoothness* reward. Safety, interaction and
> rule-compliance rewards are **not producible today** (R5–R7).

### 1.3 Is there a closed-loop environment? — probed, two ways, per the absence rule

| probe | finding | class |
|---|---|---|
| **AlpaSim on the fleet** | ran **once**, x86 A40 eval pod, 2026-07-22, full bare topology (renderer :6011 · physics :6006 · controller-MPC :6007 · driver :6789 · runtime). **That pod is terminated.** On **Thor**: `alpasim_grpc/utils/wizard` import; **`alpasim_runtime`, `alpasim_controller`, `alpasim_physics`, `utils_rs` do NOT**, and `uv` is absent ⇒ **no AlpaSim collision/offroad/scene score is obtainable today**. `cargo` is present ⇒ finishing the runtime is *bounded, not blocked*. | MEASURED — `PROGRAM_OVERVIEW.md` §5.0.3 + §5 host-stamped note |
| **the one AlpaSim result we own** | REF-C NuRec suite, **n = 12**, at-fault collision 33.3 % both arms — ⛔ **RECONSTRUCTION-OOD CONFOUNDED**: REF-C's open-loop ADE *on the reconstructions* is **1.52 m = 3.21× its real-footage 0.4728** ⇒ the numbers measure model × reconstruction fidelity, not the model (`RETRACTION_LOG` C6) | MEASURED — registry §4.4 |
| **CARLA** | **CARLA 0.9.16 IS installed on the dev box** (`C:\Users\Admin\carla\CarlaUE4.exe`) and `stack/scripts/carla_work_zone.py` drives a live build — but its own docstring is explicit: *"the two policies are still SCRIPTED archetypes … this is not our model driving yet — that needs camera rendering"*. **No camera-rendering path from CARLA into our 9-ch 256×640 cylindrical input exists.** | MEASURED — file at HEAD |
| **TanitSim** | a **future product**, not an asset | VOCABULARY.md |

⇒ **RL-online verdict substrate: we cannot roll out a policy in a faithful environment
today.** Assets that shorten the path exist (NuRec scenes are open gzip+msgpack; `map.xodr`
extracted; `cargo` present on Thor) — see B6.

> ### ⛔ A CORRECTION THIS SURVEY OWES — the "gsplat at 492 FPS" asset does not say what it is quoted as saying
>
> `Project Steering/RESEARCH_AGENDA.md:34` states *"we have gsplat at 492 FPS on Thor
> already"*, and I nearly carried that number into the B6 feasibility case. **The owning
> experiment's own FINDINGS contradict the reading:**
> `stack/experiments/nurec-gsplat/FINDINGS.md:156` — *"**225 ms/frame at 1920×1080 with
> 3.1 M gaussians is ~4.4 FPS**, not the 492 FPS measured on the **20 k-gaussian synthetic
> probe**. Closed loop at 10 Hz needs work."* The same file classes *"4.4 FPS is the
> deployment rate"* as ⛔ **NOT ESTABLISHED** (one unoptimised full-1080p frame, sky off,
> first-call overhead included) and notes that front-camera-only at a lower raster is a
> large reduction. ⇒ **the honest statement for B6 is "renderer throughput is UNMEASURED at
> the closed-loop operating point", not "492 FPS".** Anyone sizing B6 must re-measure at
> front-camera / 256×640 first. `MEASURED` (4.4 FPS at 1080p) ·
> `INHERITED — MISLEADING IN CONTEXT` (492 FPS).

### 1.4 ⛔ The one on-policy training experiment we ran — and it HURT

`…/incoming/2026-07-23-dagger-closedloop-aware/` (VERDICT.md + `dagger_result.json`),
MEASURED 2026-07-23, local RTX 4060, n = 265 windows / 12 held-out eps, cross-fit, paired
episode-cluster bootstrap. Verdict string in the artifact: **`DAGGER_HURTS`**.

| DAgger − baseline | Δ | 95 % CI | separated |
|---|---|---|---|
| closed-loop ADE@2s | **+0.266 m** | [0.008, 0.550] | ✓ **worse** |
| divergence >5 m @2s | **+0.166** (0.22→0.39) | [0.030, 0.313] | ✓ worse |
| lateral off-road proxy@2s | **+0.548 m** | [0.155, 0.994] | ✓ worse |
| open-loop head ADE@2s | +0.107 m | [−0.120, 0.320] | ✗ **matched** (so it is not a capacity/effort story) |
| **DAgger − BC-FT control** (isolates the on-policy data itself) | **+0.092 m** | [0.015, 0.187] | ✓ worse |

**Root cause (the artifact's own HYPOTHESIS, not isolated):** the harness is
**self-referential** — the on-policy states are the world model's own *imagined,
off-manifold* latents, so recovery targets train the head to over-react to imagination
artefacts. ⇒ **this refutes the cheap harness as a proving ground, not the method.**
It is also the reason **B8** (an admission gate for imagination-in-the-loop) exists.

### 1.5 The measurement that constrains every reward-based method

**C101, MEASURED 2026-08-18, [TIER T1 — PRIMARY], paired, from banked data:** the P2 CEM
planner over the frozen v1 world model is **+0.2585 m [+0.0869, +0.4309] WORSE than
constant velocity, closed-loop** (p(δ>0)=0.9975) — **35.8 % worse** — while the same
predictor rolled under **true** actions is **−0.3151 m better than CV**. Per family:
**LONGITUDINAL 1.9062 vs CV 1.6705 m**; TACTICAL and STRATEGIC are **N/A with reasons**
(the CEM emits no manoeuvre class; the cost carries no route/goal term).

The registry reads this as *"the loss is in the action search, not in the world model."*
For the RL question **either reading blocks the reward-based family**, which is why the
verdicts below do not depend on adjudicating it:

- if it is **search**: a stochastic policy-gradient learner is a *worse* optimiser of the
  same cost than a batched CEM with elites — RL would be adopted to lose ground;
- if it is the **cost**: RL inherits the misspecification exactly. Evidence for this
  reading is in the same artifact — the CEM **tracks its own minted `v_target` to
  1.03 m/s, better than the GT log tracks it (1.54 m/s)** and still loses on the
  longitudinal family. `HYPOTHESIS` — this survey does not adjudicate it, it flags that
  a discriminating experiment is worth more than an optimiser swap.

---

## 2. ⭐ THE TWO STRUCTURAL RESULTS THAT DECIDE MOST ROWS

### 2.1 Our shipped `w_select` loss is ALREADY the exact form of the GRPO/RLOO estimator

`stack/scripts/train_v6_staged.py:2459,2466-2467` — **verbatim at HEAD**:

```python
score = out["plan"]["sel_score"].float()          # [B, N]
...
p = score.softmax(dim=-1)
lsel = (p * err.detach()).sum(dim=-1).mean()
```

Write `c_i = err_i` (detached), `s` = scorer logits, `p = softmax(s)`. Then

```
∂/∂s_k  Σ_i p_i c_i  =  Σ_i c_i p_i (δ_ik − p_k)  =  p_k ( c_k − E_p[c] )
```

and for a categorical policy with reward `r_i = −c_i` and **any** baseline `b`, the
REINFORCE gradient is

```
∇_{s_k} E_p[r] = Σ_i p_i (r_i − b) ∇_{s_k} log p_i = p_k ( r_k − E_p[r] )      # b cancels identically
```

**These are the same gradient.** Consequences, and they are the survey's load-bearing
finding.

> ⚠️ **Evidence class, stated precisely rather than stretched:** this is a **DERIVATION over
> code at HEAD**, not an experiment. It is not `MEASURED` (no run produced it) and not
> `PUBLISHED` (no paper states it about our loss); it is checkable by inspection and by a
> two-line gradient test, and **that test should be written** (it is part of B2). Treat it
> as a *structural* claim of the same standing as a type-check, and refute it with algebra
> or a failing gradient test, not with a benchmark.

| GRPO / RLOO ingredient | what it does there | what it does HERE |
|---|---|---|
| **group of G sampled outputs** | Monte-Carlo the expectation | ⛔ unnecessary — the group is the fan, `N` is small and **enumerable**, so we take the expectation **exactly** |
| **group-mean / leave-one-out baseline** | reduce sampling variance | ⛔ **provably vacuous** — the baseline cancels identically, and there is no sampling variance to reduce |
| **std normaliser** (GRPO) | scale the advantage | ⛔ absent from `softade` — **which is what Dr. GRPO (2503.20783) argues is correct**: the std divisor biases toward low-variance groups |
| **length normaliser** (GRPO) | per-token averaging | ⛔ absent — same Dr. GRPO argument |
| **clipped ratio + KL-to-ref** (PPO/GRPO/GSPO) | keep multiple gradient steps on **stale** samples trusted | ⛔ inapplicable — the fan is **re-emitted every step**; there are no stale samples and no off-policy correction to make |
| **sequence-level ratio** (GSPO) | match the unit of the ratio to the unit of the reward | ⭐ the **principle** is already enforced and independently MEASURED here — see §2.3 |

⇒ **"Adopt GRPO at the selector" is not a change; it is a strictly higher-variance
re-implementation of a term we already ship.** This is the single most important row in
the document.

### 2.2 Our candidate generator is DIFFERENTIABLE to the metric cost — so the score-function estimator is dominated

`UnicycleEmission.forward` (`stack/scripts/train_v58f_unicycle_head.py:208-222`) maps the
per-candidate feature to `(a, κ)` through a bounded squash and calls
`unicycle_rollout(a_ctl, kappa, v0, dt=self.dt)` (`:221`) — an **in-graph torch
integration**. `plan["waypoints"]` therefore carries gradient to the emission MLP, and the
incumbent plan loss consumes it directly and **undetached**:
`fan = out["plan"]["waypoints"].float()` (`train_v6_staged.py:2422`). *(Class: DERIVATION
over code at HEAD — see the note in §2.1.)*

**RL's score-function (REINFORCE) estimator exists because the environment is a black box.
Ours is not — at the proposal site we *are* the environment** (a differentiable unicycle
integrator plus a known target). When the reward is a differentiable function of the
action, the **pathwise / reparameterised** gradient is available and is not an estimator at
all — it is the exact derivative — whereas the score-function estimator is unbiased but
noisy. ⇒ policy-gradient RL at S2's proposal head is **dominated, not merely unnecessary**,
for as long as the objective stays metric. ⚠️ *"Dominated" here is an argument from the
estimator's form, not a measured variance ratio on our fan; nobody has run the comparison
and nobody should need to.*

⚠️ **Where this argument stops.** It fails the moment the objective becomes
non-differentiable — collision, off-road, rule compliance, a human judgement, a
simulator's verdict. **That boundary is exactly the unblock list (B4/B5/B6), and it is
the honest scope of "RL is not useful here".**

### 2.3 GSPO's principle and E-S1-0 are the same finding, arrived at independently

GSPO (2507.18071, PUBLISHED) argues the importance ratio must be defined on the **same
object the reward scores** (the sequence), because a token-level ratio against a
sequence-level reward is a unit mismatch that injects variance. Our stack MEASURED the
identical principle in trajectory space: **supervision must sit at the ranked object.**
E-S1-0: a supervised `t=0` confidence selects **0.4728**, while **the same weights'**
unsupervised refined readout selects **1.3100** — a **2.8× penalty purely for scoring
off-distribution**; reproduced on the XL fan at **0.4714 vs 1.3901 (2.95×)**
(MEASURED, `V6F_PLANNER_DESIGN.md` §2.2). This is a clean cross-discipline
confirmation, and it is already binding on v6f's design ("the selector reads the
**emitted** trajectory, and is trained on it").

---

## 3. THE METHOD LIBRARY

Verdict key: **`ADOPT-NOW`** · **`ADOPT-LATER (blocked on …)`** · **`REJECT (reason)`**.
Cost: XS ≤ ½ d · S ≤ 2 d · M ≤ 1 wk · L ≤ 1 mo · XL > 1 mo.

### 3.1 RL — ONLINE (PPO family)

| method | objective, written out | original setting | our site | data it needs · **do we have it** | integration cost & main risk | verdict |
|---|---|---|---|---|---|---|
| **PPO** (1707.06347) | `max_θ E[ min( r_t Â_t , clip(r_t,1−ε,1+ε) Â_t ) ]`, `r_t = π_θ(a\|s)/π_θold(a\|s)` | on-policy continuous/discrete control, MuJoCo & Atari | S2 (policy = emission head), S5 | **online rollouts in an environment** + a per-step reward. ❌ **R10 absent** (§1.3) | XL. Needs an env, a value head, a rollout buffer, and a reward. Risk: with no env it can only be run in *imagination*, which is §1.4's measured failure | **ADOPT-LATER (blocked on B6 — a faithful closed loop).** The published existence proof that this works for driving is **RAD** (2502.13144): large-scale 3DGS closed-loop RL, IL-regularised |
| **TRPO** (1502.05477) | `max E[r_t Â_t]` s.t. `E[KL(π_old‖π_θ)] ≤ δ` | same, with a hard trust region | — | as PPO, plus a conjugate-gradient/Fisher solve | XL | **REJECT (superseded).** PPO's clip is the practical form; carrying both buys nothing |
| **GAE** (1506.02438) | `Â_t = Σ_l (γλ)^l δ_{t+l}`, `δ_t = r_t + γV(s_{t+1}) − V(s_t)` | advantage estimation inside any actor-critic | — | a value function + a reward stream. ❌ | — | **REJECT-until-PPO.** A *component*, not a method; it re-enters with B6 |
| **SAC** (1801.01290) | `max E[Σ_t r_t + α H(π(·\|s_t))]`, off-policy, twin critics | continuous control, sample-efficient off-policy | — | replay buffer of `(s,a,r,s′)` with **rewards**. ❌ R5 | XL | **REJECT (superseded + no reward).** Entropy-max exploration is actively wrong for a safety-critical policy trained on a single expert mode |
| **DDPG** (1509.02971) | `∇_θ J = E[ ∇_a Q(s,a)\|_{a=μ(s)} ∇_θ μ(s) ]` | deterministic continuous control | — | as SAC | XL | **REJECT (superseded by SAC/PPO; same block).** ⚠️ Note its *deterministic policy gradient* is the pathwise estimator — §2.2 says we already have that, without a critic |
| **DreamerV3** (2301.04104) | actor-critic trained **entirely on imagined rollouts** of a learned latent model; symlog two-hot return prediction; fixed hyper-parameters across domains | 150+ tasks from one config, incl. Minecraft | S1 + S2 | a **reward head on the latent** + imagined rollouts. ⚠️ **R11 available, R7 measured absent** | L. Risk is precisely §1.4's: training on the WM's own off-manifold latents. Our WM **cannot locate a lead vehicle in its own decode (LF0: 81.4 % censoring)**, so a latent safety-reward head has nothing to read | **ADOPT-LATER (blocked on B8 — a decodability admission gate the WM must pass first).** It is the only route to a closed loop with no renderer, which is why it stays on the list |

### 3.2 RL — OFFLINE (CQL / IQL family)

**One block applies to this entire family and it should be read before the rows:** offline
RL needs `(s, a, r, s′)` — the **`r` does not exist** (R5). The only reward we can build
from the corpus is **R4 (comfort + progress)**, which is *exactly* the P2 CEM cost, and
**C101 MEASURED that a near-oracle optimiser of that cost loses to constant velocity by
+0.2585 m [+0.0869, +0.4309] closed-loop, and loses on the longitudinal family it was
built for** (§1.5). ⇒ adding a *learned* optimiser for a cost we have already optimised
into a measured loss is not a lever.

| method | objective | original setting | our site | data · have it | cost & risk | verdict |
|---|---|---|---|---|---|---|
| **CQL** (2006.04779) | `min_Q α( E_{a∼μ}[Q(s,a)] − E_{a∼D}[Q(s,a)] ) + Bellman`, i.e. push OOD action-values **down** | D4RL offline benchmarks | S2 | `(s,a,r,s′)`. ❌ **no r** | L; needs a critic over a 60×2 action | **REJECT (no reward signal exists; the constructible one is MEASURED misspecified — §1.5)** |
| **IQL** (2110.06169) | expectile regression `L_τ(u)=\|τ−1(u<0)\|u²` on `u = Q(s,a) − V(s)`; policy by advantage-weighted BC `exp(β·A)`. **Never queries π on OOD actions** | offline D4RL | S2 | as CQL. ❌ | L | **REJECT (same).** ⭐ *If* a reward ever exists (B5), IQL is the **right first offline method** — it is the only one in this family that never evaluates the critic off-support, which matters because our fan is generated, not logged |
| **AWAC** (2006.09359) | `π ← argmax E_{s,a∼D}[ log π(a\|s) · exp(A^π(s,a)/λ) ]` | offline→online fine-tuning | S2 | reward + a critic. ❌ | M | **REJECT (no reward)** — but see AWR |
| **AWR** (1910.00177) | same advantage-weighted regression with a TD(λ) value baseline, fully offline | simple scalable off-policy RL | S2 | reward + value. ❌ | **XS given a reward** | ⭐ **REJECT-NOW, but it is the CHEAPEST DOOR.** Advantage-weighted BC is literally *"multiply the existing WTA/plan loss by `exp(A/λ)`"* — no new module, no `STAGE_MAY_INTRODUCE` entry, no state-dict key. **Keep the plan loss in a shape where that is a 3-line change** (backlog B7) |
| **TD3+BC** (2106.06860) | `max E[ λ Q(s,π(s)) − (π(s)−a)² ]`, `λ = α / (N⁻¹Σ\|Q\|)` | minimalist offline RL | S2 | reward. ❌ | M | **REJECT (no reward).** Structurally it is *our incumbent plus a critic* — with `λ=0` it **is** our BC plan loss, which is a useful way to see how little the family adds without `r` |
| **Decision Transformer** (2106.01345) | supervised `p(a_t \| R̂_t, s_t, a_{<t}, …)` with return-to-go `R̂_t = Σ_{t′>t} r_{t′}`; at test time **condition on the desired return** | offline RL as sequence modelling | S2 + S3 | a return computed **in hindsight** from the logged future — **no reward model, no rollouts, no preference pairs**. ✅ **constructible today** | M | ⭐ **ADOPT-LATER (blocked on B1's leak audit).** See the RvS row — this is the sleeper |
| **RvS** (2112.10751) | conditional BC on an **outcome** (goal or return): `max E[log π(a\|s, ω)]` with `ω` read off the same trajectory's future | shows outcome-conditioning + capacity, not TD learning, is what matters | **S3 (already!)** | hindsight outcome. ✅ | **XS — we already do it** | ⭐⭐ **ADOPT-NOW (as a naming + audit, B1).** **We already ship RvS and do not call it that:** `SPEED_BAND(v_lo,v_hi)` and `ANCHOR_GOAL(anchor_id, t_reach_s)` are hindsight-derived outcome conditioners, and P2's `v_target` = *the 85th percentile of future speed over the next 10–20 s* is a **return-to-go** by another name. ⛔ **THE RISK IS THE PROGRAMME'S OWN NAMED FAILURE MODE:** conditioning on an outcome derived from the ego's own future is exactly the shape of the **nav-echo** (route head scored 1.0000 as a bijection of its input), the **T1 action echo** (97.9 % open-loop → 0.0 % hold-action), and the **P1 speed echo** (R² 0.995 → **−0.72** under the v0 shuffle). The RvS literature's own caveat (achievability of the conditioning; failure under stochastic dynamics) and our anti-echo doctrine are the same warning. ⇒ adopt the *name and the controls*, not new machinery |

### 3.3 IMITATION — BC and the DAgger family

| method | objective | original setting | our site | data · have it | cost & risk | verdict |
|---|---|---|---|---|---|---|
| **Behaviour cloning** | `min E_{(s,a)∼D}[ −log π(a\|s) ]` (or L1/L2 on continuous actions) | supervised imitation | **S2 — the incumbent** | R1/R2. ✅ | 0 — it is what runs | **ADOPT-NOW (already shipped).** The v6 form is *ε-relaxed winner-take-all* over the fan (`lambda_plan`, `plan_wta_eps`), and the ε exists for a measured reason: **under pure WTA the N−1 losing candidates get exactly zero gradient and nothing bounds the fan's mean — MEASURED on the banked REF-C-XL fan, oracle 0.1639 m against a fan mean 13.9564 m, 85×** (`train_v6_staged.py:2430-2440`) |
| **BC's known ceiling** (1904.08980) | — | *Exploring the Limitations of Behavior Cloning for Autonomous Driving* | — | — | — | **PUBLISHED, and it is our own diagnosis too:** compounding covariate shift. Our measurement of it: **`cl − ol` = +9.0039 m [6.3659, 11.8487] separated, and the divergence is ~99 % LONGITUDINAL** (LON 9.2655 vs LAT 0.7446), [T1, 6 844 windows / 40 eps] |
| **DAgger** (1011.0686) | round *i*: roll out `π_i`, **query the expert on the visited states**, `D ← D ∪ {(s, π*(s))}`, retrain. No-regret ⇒ `O(T)` error vs BC's `O(T²)` | structured prediction / imitation | S2 | **on-policy states + a queryable expert at those states.** ❌ both: the human drove once, and R10 is absent | M in a sim; **already implemented against the cheap harness** | ⛔ **ADOPT-LATER (blocked on B6), and REFUTED on the cheap harness — MEASURED (§1.4): +0.266 m [0.008, 0.550] closed-loop, separated WORSE, and CI-worse than a matched-budget BC control.** The programme's own decision stands: *deferred, not refuted*, re-enters only as an AlpaSim/TanitSim-validated curriculum |
| **ChauffeurNet-style perturbation** (1812.03079) | BC **+ synthesised perturbations of the past state, with the logged future kept as target** + auxiliary collision/on-road losses | the *offline* substitute for on-policy exposure | S2 | perturbable state + logged future. ⚠️ **PARTIAL — see risk** | M | ⭐ **ADOPT-NOW as a pre-registered v7-tiny experiment (B3), not as a trainer default.** ChauffeurNet perturbs a *raster*, which re-renders for free; **we take camera frames, so we cannot perturb the viewpoint** — but we **can** perturb `v0` and the action history while keeping the logged 6 s target, which yields a **longitudinal-only recovery curriculum**, and longitudinal is **~99 % of the measured `cl − ol` divergence**. ⛔ **RISK, and it is sharp:** a v0-perturbed input is *visually inconsistent*, so the head may learn to **distrust `v0` entirely** — which would undo the programme's validated speed-channel fix (REF-A **3.73 → 0.83 m**, speed R² **0.61 → 0.965**). ⇒ **mandatory arms: speed-R² regression arm + the existing `--speed-echo-control` v0-shuffle + the hold-v0 baseline.** `HYPOTHESIS` — this is a design proposal, nothing here is measured |
| **Learning by Cheating** (1912.12294) / **Roach** (2108.08265) | train a *privileged* agent on GT layout (BC, or PPO for Roach), then distil into a sensorimotor student on the privileged agent's **conditional action distribution** at all states — a queryable expert makes DAgger cheap | CARLA | S2 | a **privileged, queryable** expert. ❌ for logged real data | L | **REJECT (not applicable — no queryable expert exists for logged real-world episodes).** ⚠️ The *fan-oracle* version of this pattern is already what our selector's supervision **is** (`sel_norm_err_rank` measures exactly the gap to the oracle-over-the-fan), so there is nothing left to import. Re-enters only with B6, where the privileged agent is a sim-side planner |
| **GAIL** (1606.03476) | `min_π max_D E_π[log D(s,a)] + E_{π_E}[log(1−D(s,a))] − λH(π)` — occupancy-measure matching | imitation without a reward function | S2 | **on-policy rollouts** (the outer min is an RL problem) + a discriminator | XL | **REJECT (needs R10; same block as PPO).** Additionally the discriminator would read our latents, and **R7/LF0 says those latents cannot see agents** — so the discriminator would grade on appearance, not interaction |
| **MILE** (2210.07729) | model-based imitation: learn a latent BEV world model from expert video+labels and imitate **in latent imagination** | urban driving, CARLA | S1+S2 | expert video + BEV labels. ⚠️ our BEV decode is weak (P8 absolute IoU ~0.02; the admissible claim is the *retention ratio*) | L | **ADOPT-LATER (blocked on B8, same gate as DreamerV3).** Carried because it is the closest published architecture to ours that trains a policy inside a learned latent |

### 3.4 PREFERENCE OPTIMISATION — the RLHF / DPO / GRPO family

⛔ **The family-wide block, stated once (§1.2):** every method here needs either a
**preference pair** (R8 ❌) or a **reward over generated sequences** (R5 ❌ / R3 ✅ but
cardinal). A demonstration corpus has **one policy's output per state**, so there is no
negative class; and the synthesizable negative reduces the cardinal R3 to a binary, which
**E-OBJ-1 MEASURED to be separated WORSE** (+0.0974 / +0.1670 m).

| method | objective, written out | original setting | our site | data · have it | cost & risk | verdict |
|---|---|---|---|---|---|---|
| **RLHF from preferences** (1706.03741) | Bradley-Terry RM `L = −E[log σ(r(y_w) − r(y_l))]` over compared **trajectory segments**, then RL against `r` | Atari/MuJoCo from ~1 % human feedback | S2/S3 | human comparisons + an RL loop. ❌ R8, ❌ R10 | XL | **REJECT-NOW.** It is the *ancestor* of every row below and its two ingredients are both absent |
| **InstructGPT / RLHF pipeline** (2203.02155) | SFT → BT reward model → **PPO on `r(x,y) − β·KL(π_θ‖π_SFT)`** | LLM instruction following | — | as above | XL | **REJECT-NOW (both ingredients absent).** The KL-to-SFT term is the one transferable idea and we already have its analogue: the trunk-freeze + declared detaches |
| **DPO** (2305.18290) | `L = −E[ log σ( β log π(y_w\|x)/π_ref(y_w\|x) − β log π(y_l\|x)/π_ref(y_l\|x) ) ]` — the RLHF optimum reparameterised, no RM, no rollouts | LLM alignment | S2 | **preference pairs.** ❌ R8 | M *if* pairs existed | **REJECT-NOW (no negative class; and the manufacturable pair is a lossy reduction of R3 — MEASURED).** ⇒ becomes testable only via **B4** |
| **IPO** (2310.12036) | `L = E[(h_π(y_w,y_l) − τ⁻¹/2)²]`, `h` = log ratio-of-ratios — a **bounded** Ψ-PO objective that does not blow up when preferences are deterministic | fixes DPO's overfitting | S2 | pairs. ❌ | M | **REJECT-NOW (same block).** ⭐ It is the **right variant if B4 lands**, because preferences *derived from a metric* are deterministic by construction — exactly the regime DPO degenerates in and IPO was built for |
| **GRPO** (2402.03300; at scale 2501.12948) | sample `G` outputs per prompt, `Â_i = (r_i − mean r)/std r`, clipped PPO surrogate + KL-to-ref; **no value network** | LLM math reasoning with verifiable rewards | S2 | a group of scored candidates. ✅ **R3 gives exactly this** | S | ⛔ **REJECT (already present in exact form) — §2.1.** Our `softade` is the *exact-expectation* version of this estimator over an enumerable fan; the group baseline **cancels identically**, and sampling `G` would only add variance. Adopting GRPO here would be a strictly worse re-implementation |
| **Dr. GRPO** (2503.20783) | removes GRPO's `1/\|o\|` length normaliser and the `std` divisor → an **unbiased** policy gradient; fixes length/difficulty bias | critical re-analysis of R1-zero training | S2 | — | **XS (an audit)** | ⭐ **ADOPT-NOW (B2, 0 params).** Its prescription is *"do not divide by std, do not divide by length"* — **our `softade` already has neither.** Adopt as a **pinned audit test** so a future contributor cannot add them |
| **DAPO** (2503.14476) | decoupled clip (`ε_low`≠`ε_high`), **dynamic sampling — drop groups whose rewards are all identical**, token-level PG loss, overlong reward shaping | open large-scale LLM RL | S2 | — | **S** | ⭐ **ADOPT-NOW, one clause only (B2).** *Dynamic sampling* transfers: **drop windows whose fan is degenerate** (all candidates ≈ equal error), because they contribute gradient noise and no ranking information. We already do two cousins of this — the **reachability prefilter deletes 72.08 % of the fan for a paired ΔADE of exactly 0.0000 at a 3.5× compute saving**, and `sel_ce_reach` normalises the ranking objective over the **admissible** set — so this is the third clause of a pattern we independently found. The clip/token clauses are inapplicable (no ratios, no tokens) |
| **GSPO** (2507.18071) | sequence-level importance ratio `s_i(θ) = (π_θ(y_i\|x)/π_old(y_i\|x))^{1/\|y_i\|}`, clipped at the **sequence** level — matches the ratio's unit to the reward's unit | stabilises long-sequence / MoE RL where token-ratio variance breaks GRPO | S2 | — | — | **REJECT the algorithm (no importance ratios exist here — §2.1); ADOPT-NOW the principle, which is ALREADY ENFORCED.** ⭐ GSPO's principle and our **E-S1-0** are the same finding: *score the object you rank.* Ours is MEASURED at **0.4728 vs 1.3100 (2.8×)**, reproduced **0.4714 vs 1.3901 (2.95×)** on the XL fan. Record the convergence in the Lab's cross-discipline log — it is the transfer the Research Agenda Field 2.4 asks for |
| **RLOO** (2402.14740) | REINFORCE with a **leave-one-out** baseline over `k` samples: `Â_i = r_i − (k−1)⁻¹ Σ_{j≠i} r_j`; whole generation treated as one action; no value net, no clipping | shows PPO's machinery is unnecessary for RLHF | S2 | a group of scored samples. ✅ R3 | S | ⛔ **REJECT (the baseline is provably irrelevant here — §2.1 shows `b` cancels identically under exact enumeration).** ⭐ Its *thesis* — "the PPO machinery is unnecessary" — is however the correct reading of our situation, one step further: **the REINFORCE machinery is unnecessary too** |
| **KTO** (2402.01306) | prospect-theoretic value of `(implicit reward − reference point)`, from **unpaired binary desirable/undesirable** labels | alignment without pairs | S2/S3 | an **undesirable** class. ❌ | M | **REJECT (no undesirable class in an expert-only corpus).** ⭐ KTO is the *most* nearly-applicable of the family precisely because it drops the pairing requirement — so it becomes the **cheapest** member to adopt the day any binary "bad trajectory" label exists (a sim infraction from B6, or a human thumbs-down from B4) |
| **ORPO** (2403.07691) | `L = L_SFT + λ · −log σ( log [odds(y_w)/odds(y_l)] )` — monolithic, **no reference model, no separate preference stage** | one-stage LLM alignment | S2 | pairs. ❌ | M | **REJECT (no negative class).** Its structural lesson does transfer: a preference term **fused into the supervised loss** avoids a second stage — which matters here because §1.14 of the registry MEASURED that **keeping a planner across a trunk repair moved a frozen selector 0.7933 → 4.4159**, i.e. our stage seams are expensive |
| **SimPO** (2405.14734) | reference-free reward `r(x,y) = (β/\|y\|) log π(y\|x)` (length-normalised average log-prob) + target margin γ | removes DPO's reference model | S2 | pairs. ❌ | M | **REJECT (no negative class).** ⚠️ And note its length normalisation is the **opposite** of Dr. GRPO's prescription; for us the analogue (normalise by horizon) would silently re-weight the 0–2 s operative band against the 2–6 s tactical band, which `cfg.split_bands` exists to keep separable — so it would be **actively harmful** if imported |
| **CPO** (2401.08417) | contrastive preference loss approximated **without** the reference model + a BC regulariser | memory-efficient DPO for MT | S2 | pairs. ❌ | M | **REJECT (no negative class).** Structurally it is *"DPO + BC"*, and the BC half is our incumbent |
| **RLAIF** (2309.00267) | replace human preference labels with an off-the-shelf model's judgements; otherwise the RLHF pipeline | scaling RLHF | S3 | a **trustworthy** AI judge. ⚠️ **R9 partial and measured-weak** | L | **ADOPT-LATER (blocked on label quality, not on the optimiser).** We have a VLM-labelled corpus (**23 644 rows / 4 729 clips × 5 tasks**) but the fused hierarchical layer ships a **named 57.2 % perception hole** and **G1 closed at 0/31 verifiable sign reads**. ⇒ an AI judge built on that today would optimise the labeller's errors |

### 3.5 PREFERENCE / RL **applied to driving** — the three published bridges

| paper | what it did | what it tells US |
|---|---|---|
| **TrajHF** (2503.10434, *Learning Personalized Driving Styles via RLHF*) | reward model over **trajectories** from human preference, used to finetune a **generative trajectory model** | ⭐ **The single strongest case for preference learning in our stack, and it is the one case where R3 genuinely cannot substitute.** Style/comfort is *unspecifiable but comparable*: two trajectories can both be safe and legal, and only a human can rank them. Distance-to-expert (R3) cannot express this, because it declares the human's single realisation the unique optimum. ⇒ **ADOPT-LATER, blocked on B4**, and B4 is cheaper than it looks because the *rendering* half already exists in the standing viz shape (camera + metric-BEV pane + text panel): `stack/scripts/ph0_rich_overlay.py`, `stack/scripts/p8_bev_reel.py`, `taniteval/probe_overlay.py`. ⚠️ **VERIFIED BY CONTENT, and the popular name is wrong: there is no `corpus_overlay.py` in this repo** — what is missing is a *fan-comparison* renderer (two candidates side by side) and the ranking UI, not the overlay itself |
| **AlphaDrive** (2503.07608) | **GRPO** + rule-based planning rewards (action accuracy, planning consistency) on a driving **VLM** emitting meta-actions | the closest published analogue to our **S3**. Its reward is *rule-based over discrete meta-actions* — i.e. it needed no simulator, only a labelled meta-action. **We have the site and the token vocabulary but not the label quality** (R9). ⇒ **ADOPT-LATER, blocked on the same label work as RLAIF.** ⚠️ Note that with a *label* in hand, plain CE is the right tool; GRPO earns its place only when there is a **verifier but no label** — which is not our situation |
| **RAD** (2502.13144) | end-to-end driving policy trained by **large-scale 3DGS-based closed-loop RL**, IL-regularised | ⭐ **the existence proof for B6, and architecturally reachable for us**: NuRec scenes are open (gzip+msgpack, MEASURED), gsplat renders them natively on aarch64 Thor (MEASURED), `map.xodr` was extracted. ⚠️ **but throughput at the closed-loop operating point is UNMEASURED — see the correction in §1.3; do not size this on "492 FPS".** This is the paper the TanitSpear/TanitSim line should be measured against |

### 3.6 Sites where NO method in this survey applies — clean negatives

| site | why nothing here applies | class |
|---|---|---|
| **S1 — trunk / world model (S-W)** | S-W's objective is self-supervised latent prediction; it **emits no action and consumes no reward**, so there is no policy for a policy-gradient to differentiate and no return for a value function to estimate. The only bridge would be DreamerV3's latent reward head, and **R7/LF0 MEASURED that our latent cannot locate a lead vehicle in 81.4 % of the windows where one exists** ⇒ a latent safety reward has nothing to read | MEASURED |
| **S4 — strategic layer (S-S)** | the strategic layer's measured problem is **label existence, not optimisation**: `route_acc_follow` **0.8031 == `majority_straight_rate` 0.8031** with prediction distribution **{left 0, straight 1737, right 0}** — a constant predictor with **no vision-only route skill at all**; `LANE_TARGET` is emitted by **no path today** (`required = None` for **801/801** clips, because PhysicalAI-AV ships no lane context). An RL objective over an 8–30 s return would be fitted on a channel that carries no signal | MEASURED — registry §1.9; `HIERARCHY_VOCABULARY.md` §3 |
| **S5 — deployment-time search** | CEM/MPC is a *search*, not a learner; the methods here train parameters. C101 already MEASURED this search losing to CV (§1.5) | MEASURED |

---

## 4. ⭐ RANKED BACKLOG (highest value first)

`For merge into products/P4-training-pipelines/BACKLOG.md — I did not write that file.`

| # | item | method | site | what unblocks it | cost | why it ranks here |
|---|---|---|---|---|---|---|
| **B1** | **Name and leak-audit the return-conditioning we already ship.** Declare `SPEED_BAND`, `ANCHOR_GOAL` and P2's `v_target` as **RvS/DT outcome conditioning**; then run the anti-echo controls **on the conditioner itself**: hold-v0 baseline + `--speed-echo-control` v0-shuffle + a corpus-marginal control | **RvS** (2112.10751), **Decision Transformer** (2106.01345) | S3, S2 | **nothing** — banked data, existing machinery, 0 GPU | **S** | Highest value/cost in the list. It costs almost nothing and it tests a term that is *already training*; the programme has been fooled by this exact shape **three times** (nav-echo 1.0000, T1 action echo 97.9 %→0.0 %, P1 speed echo R² 0.995→−0.72) |
| **B2** | **Dr. GRPO / DAPO audit of the `w_select` objective.** (a) pin a test that `softade` carries **no std and no length normaliser**; (b) add DAPO **dynamic sampling** — drop windows whose fan is degenerate (all candidates ≈ equal `err`); (c) report `sel_norm_err_rank` **and** lower-tail hit rate, never ρ | **Dr. GRPO** (2503.20783), **DAPO** (2503.14476) | S2 | **nothing** — 0 params, 0 new state-dict keys | **S** | Turns two frontier results into a regression test on a term we already ship, and the degenerate-fan drop is the third clause of a pattern we independently found (reachability prefilter: **72.08 % deleted, ΔADE 0.0000, 3.5×** compute) |
| **B3** | **ChauffeurNet-style LONGITUDINAL recovery curriculum.** Perturb `v0` / action history, keep the logged 6 s target. **Pre-registered on the v7-tiny ladder** with a deliberate-regression arm; mandatory controls: speed-R² arm, v0-shuffle, hold-v0 | **ChauffeurNet** (1812.03079) | S2 | **nothing** (dev-box RTX 4060; v7-tiny is a 29-min rig) | **M** | The only method in the survey that attacks the **~99 %-longitudinal `cl − ol` divergence (+9.0039 m [6.3659, 11.8487])** *without* a simulator. ⛔ Real risk of undoing the validated speed fix (REF-A **3.73→0.83 m**) — the regression arm is not optional |
| **B4** | **Build the preference-collection instrument.** A *fan-comparison* renderer (two candidates side by side) on top of the existing overlay stack (`ph0_rich_overlay.py` / `p8_bev_reel.py` / `probe_overlay.py`) → pairwise ranking, human and/or VLM. Ship it as a P7/P5 asset | unlocks **TrajHF** (2503.10434), **IPO** (2310.12036), **KTO** (2402.01306), **DPO** (2305.18290) | S2 | **nothing technical** — a PI decision on whose time labels it | **M** | ⭐ **The single item that converts an entire REJECTED family into a testable one.** And it is the one objective R3 provably cannot express: *style/comfort is comparable but unspecifiable*. Recommend **IPO** as the first estimator (metric-derived preferences are deterministic — DPO's degenerate regime) |
| **B5** | **Reward-existence work package: ingest `obstacle.offline` for the parity corpus.** The join script exists (`build_obstacle_join.py`, schema pinned by test); gate the spend on the **already pre-registered** lead-state premise test | unlocks **IQL** (2110.06169), **AWR** (1910.00177), **CQL** (2006.04779) | S2, S3 | a PI decision on ~12.4 GB + 2–3 eng-days | **L** | Without it the offline-RL family has **no `r`** and the survey's `REJECT`s cannot be revisited. R7/LF0 proves the reward must come from **GT**, not from our latent. ⚠️ Run the pre-registered gate first — a `FAIL` there is a *cheaper* outcome than an ingest |
| **B6** | **Closed-loop environment (TanitSim, 3DGS/NuRec-class, RAD-shaped).** Either finish `alpasim_runtime` on x86 (`cargo` is present ⇒ bounded) or build the gsplat path | unlocks **PPO** (1707.06347), **GRPO/DAPO** in their real form, **GSPO**, **DAgger** (1011.0686), **GAIL** (1606.03476), **RAD** (2502.13144) | S2, S5 | funding + a host; assets in hand: gsplat renders NuRec natively on Thor, NuRec open msgpack, `map.xodr`, CARLA 0.9.16 local | **XL** | The **only** thing that makes the online-RL half of the mandate real. ⚠️ **Step 0 is a throughput re-measure at front-camera / 256×640 — the "492 FPS" figure is a 20 k-gaussian synthetic probe; the real scene read 4.4 FPS at 1080p (§1.3).** ⚠️ Carry the **OOD lesson**: our one AlpaSim result is confounded at **3.21× reconstruction-OOD** — any new sim must ship an open-loop-on-sim control **before** any closed-loop verdict |
| **B7** | **Keep the plan loss AWR-ready.** A code-shape note + a test, so that `plan_loss × exp(A/λ)` is a 3-line change the day B5 lands | **AWR** (1910.00177) | S2 | nothing | **XS** | Cheapest possible option value: advantage-weighted BC adds **no module, no state-dict key, no `STAGE_MAY_INTRODUCE` entry** |
| **B8** | **Imagination-in-the-loop admission gate.** A pre-registered criterion (decodability of agents + a `cl − ol` ceiling) the WM must pass **before** any on-policy training inside imagination is permitted | gates **DreamerV3** (2301.04104), **MILE** (2210.07729), and any re-run of DAgger on the cheap harness | S1→S2 | nothing | **S** | Encodes §1.4's measured failure as a rule instead of a memory. Without it, the next agent re-runs a curriculum that is **MEASURED separated-worse** |

**Not on the list, deliberately:** TRPO, SAC, DDPG, GAE, ORPO, SimPO, CPO, RLOO, GAIL,
Learning-by-Cheating/Roach. Each is either superseded within its own family for our shape,
or blocked by the *same* missing signal as a higher-ranked row, and carrying it separately
would inflate the backlog without adding an experiment. Their rows in §3 record the
reason so the decision is not re-litigated.

---

## 5. WHAT THIS SURVEY DOES **NOT** CLAIM

1. ⛔ **Nothing here is measured on a v6 fan.** As of `V6F_PLANNER_DESIGN.md`'s stamp
   (2026-08-15/16) v6 had never emitted one — S-W stopped at step 6 250/30 000 at the pod
   stop and resumed on Thor at ~6 400. ⚠️ **I did not re-probe the live training state**
   (`LOOP_STATE.md` is stale at 2026-07-29); the S-W step count may have advanced. Every
   `err`/fan number quoted here is from banked **REF-C** fans at 2 s / 4-waypoint
   resolution, and the `σ*`-type absolutes must be re-measured at 6 s, never extrapolated.
2. ⛔ **§2.1's equivalence is algebra over code at HEAD, not an experiment.** It says
   GRPO's *estimator* collapses to `softade` under exact enumeration. It does **not** say
   `softade` is the best selection objective — that is an open question, and the primary
   endpoints for it are already fixed (`sel_norm_err_rank`, lower-tail hit rate).
3. ⛔ **§1.5's "the cost may be misspecified" is `HYPOTHESIS`**, offered beside the
   registry's "the loss is in the action search". This survey does not adjudicate; it
   shows the RL verdicts hold under either reading.
4. ⛔ **B3 is a design proposal.** No perturbation curriculum has been run in this
   programme. Its risk (teaching the head to distrust `v0`) is real and unmeasured.
5. ⛔ **The absence claims (R8 human preferences; R10 a faithful environment) are
   two-probe absences** — repo search plus the host-stamped fleet reads of §1.3 — not
   single-probe. Should a preference set or a live sim appear, §3.4 must be re-decided.
6. ⚠️ **Two secondary-source corrections were made while writing this** and are recorded so
   they are not re-inherited: the **"gsplat 492 FPS"** feasibility figure (§1.3 — contradicted
   by its own experiment's FINDINGS) and **`corpus_overlay.py`**, which does not exist in this
   repo (§3.5 — the real overlay scripts are named there). Both were caught by opening the
   primary; both would have been carried by quoting the steering doc.

---

## 6. BANKED-SOURCE MANIFEST

All 39 papers cited above were banked by `tools/kb_add.py … --cited-by
"products/P4-training-pipelines/METHOD_LIBRARY.md"` and **verified by content**
(`--verify` → *67 entries, 0 problems*, 2026-08-23). PDFs live in
`TanitAD Research Lab/Library/papers/`; metadata + sha256 in
`TanitAD Research Lab/Library/library.json`; human index `…/Library/LIBRARY.md`.

| arXiv | banked title | tag |
|---|---|---|
| 1707.06347 | Proximal Policy Optimization Algorithms | rl-online |
| 1502.05477 | Trust Region Policy Optimization | rl-online |
| 1506.02438 | High-Dimensional Continuous Control Using Generalized Advantage Estimation | rl-online |
| 1801.01290 | Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor | rl-online |
| 1509.02971 | Continuous control with deep reinforcement learning | rl-online |
| 2301.04104 | Mastering Diverse Domains through World Models (DreamerV3) | rl-model-based |
| 2005.01643 | Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems | rl-offline |
| 2006.04779 | Conservative Q-Learning for Offline Reinforcement Learning | rl-offline |
| 2110.06169 | Offline Reinforcement Learning with Implicit Q-Learning | rl-offline |
| 2006.09359 | AWAC: Accelerating Online Reinforcement Learning with Offline Datasets | rl-offline |
| 2106.06860 | A Minimalist Approach to Offline Reinforcement Learning (TD3+BC) | rl-offline |
| 2106.01345 | Decision Transformer: Reinforcement Learning via Sequence Modeling | rl-offline |
| 1910.00177 | Advantage-Weighted Regression: Simple and Scalable Off-Policy RL | rl-offline |
| 2112.10751 | RvS: What is Essential for Offline RL via Supervised Learning? | rl-offline |
| 1011.0686 | A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger) | imitation |
| 1904.08980 | Exploring the Limitations of Behavior Cloning for Autonomous Driving | imitation-ad |
| 1812.03079 | ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst | imitation-ad |
| 1912.12294 | Learning by Cheating | imitation-ad |
| 2108.08265 | End-to-End Urban Driving by Imitating a Reinforcement Learning Coach (Roach) | imitation-ad |
| 1606.03476 | Generative Adversarial Imitation Learning | imitation |
| 2210.07729 | Model-Based Imitation Learning for Urban Driving (MILE) | imitation-ad |
| 1706.03741 | Deep reinforcement learning from human preferences | preference |
| 2203.02155 | Training language models to follow instructions with human feedback | preference |
| 2305.18290 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | preference |
| 2310.12036 | A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO) | preference |
| 2402.03300 | DeepSeekMath (introduces GRPO) | preference |
| 2501.12948 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL | preference |
| 2507.18071 | Group Sequence Policy Optimization (GSPO) | preference |
| 2402.01306 | KTO: Model Alignment as Prospect Theoretic Optimization | preference |
| 2403.07691 | ORPO: Monolithic Preference Optimization without Reference Model | preference |
| 2405.14734 | SimPO: Simple Preference Optimization with a Reference-Free Reward | preference |
| 2401.08417 | Contrastive Preference Optimization (CPO) | preference |
| 2402.14740 | Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback (RLOO) | preference |
| 2503.14476 | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | preference |
| 2503.20783 | Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO) | preference |
| 2309.00267 | RLAIF vs. RLHF: Scaling RLHF with AI Feedback | preference |
| 2503.10434 | Learning Personalized Driving Styles via Reinforcement Learning from Human Feedback (TrajHF) | rl-ad |
| 2503.07608 | AlphaDrive: Unleashing the Power of VLMs in Autonomous Driving via RL and Reasoning | rl-ad |
| 2502.13144 | RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based RL | rl-ad |

### Primary sources used for OUR facts (never a summary)

| fact | source |
|---|---|
| corpus / parity / episode contract | `Project Steering/MODEL_REGISTRY.md` §0.1 |
| closed-loop T1, `cl − ol`, action echo | `MODEL_REGISTRY.md` §1.12; `V6F_PLANNER_DESIGN.md` §2.2 |
| AlpaSim n=12 + the 3.21× OOD confound (C6) | `MODEL_REGISTRY.md` §4.4 |
| C101 — CEM planner vs CV, paired, T1 | `MODEL_REGISTRY.md` §5 (UPDATE 2026-08-18) |
| LF0 — decoded BEV cannot read the lead gap | `MODEL_REGISTRY.md` §LF0 (line ~1805) |
| strategic constant-predictor result | `MODEL_REGISTRY.md` §1.9 |
| E-OBJ-1 `softade` vs CE; E-S1-0; reachability prefilter; §1.14 selector-across-repair | `Project Steering/V6F_PLANNER_DESIGN.md` §2.2 |
| DAgger `DAGGER_HURTS` | `TanitAD Research Lab/Architecture & Inference/Implementation/incoming/2026-07-23-dagger-closedloop-aware/VERDICT.md` + `dagger_result.json` |
| loss/stage/selector code shapes | `stack/scripts/train_v6_staged.py`, `stack/tanitad/models/v6.py`, `stack/scripts/train_v58f_unicycle_head.py` (all at HEAD) |
| obstacle.offline coverage + never-ingested | `stack/scripts/build_obstacle_join.py`, `stack/scripts/lead_state_gate.py` (docstrings, at HEAD) |
| fleet / sim availability | `Project Steering/PROGRAM_OVERVIEW.md` §5.0.3 + §5; `stack/scripts/carla_work_zone.py` |
| renderer throughput (and the 492-FPS correction) | `stack/experiments/nurec-gsplat/FINDINGS.md:156,168` vs `Project Steering/RESEARCH_AGENDA.md:34` |
| overlay/render assets for B4 | `stack/scripts/ph0_rich_overlay.py`, `stack/scripts/p8_bev_reel.py`, `taniteval/probe_overlay.py` (⚠️ no `corpus_overlay.py` exists) |
| VLM label volume + the 57.2 % hole; G1 0/31 | `MODEL_REGISTRY.md` §11.1, §11.2; `Project Steering/G1_RESULT.md` |
| tactical / strategic vocabulary | `…/incoming/2026-08-07-hierarchical-wm-redesign/HIERARCHY_VOCABULARY.md` |

# Angle 3 of 3 — Alternatives to RL, hybrids, and the honest "do we even need RL?" verdict

**Author:** TanitAD research subagent (web-research only, no pod use)
**Generated:** 2026-07-23 ~05:36 Europe/Berlin (UTC+2)
**Scope:** Non-RL and hybrid closed-loop methods for a sub-300M latent driving world model (WM),
open-loop-IL-trained + warm-started from supervised v1, **logged data** (comma2k19 + PhysicalAI-AV),
**no fast live sim** (NuRec is slow + ~3× OOD — *internal, per brief, INHERITED*), MEASURED failure =
off-road departure under compounding error, mostly off-highway.

**Evidence-class convention (binding):** every external number is `PUBLISHED (cite)`; every
inference to OUR stack is `HYPOTHESIS`; internal facts I did not re-measure are `INHERITED (brief)`.
No claim here is `MEASURED` by me — I ran no eval. The ranking in §7 is `HYPOTHESIS`; treat it as a
pre-registration to be settled by the experiment in §8, not as a decision already made.

---

## 0. TL;DR verdict

For OUR regime (logged data, no fast sim, safety-critical, a WM + anchored-diffusion planner that
**already** gets a MEASURED closed-loop win from imagination-in-the-loop re-planning), the published
evidence says **reward-driven RL is the *last* lever to reach for, not the first.** The two things RL
is uniquely good at — killing compounding error, and handling the safety-critical tail — are **both
substantially obtainable without a reward function**:

- **Compounding error** is killed by *closed-loop-aware imitation* (CAT-K, RoaD, DAgger): pure
  supervised losses on the policy's own rollout distribution. `PUBLISHED`: a 7M CAT-K model beats a
  102M open-loop model (2412.05334); RoaD lifts an **AlpaSim + VLM policy** — our exact stack family —
  by **+41% driving score, −54% collisions** (2512.01993).
- **The off-road tail** is most cheaply attacked at *inference* by **cost-guided diffusion**
  (training-free collision/off-road energy in the denoiser — 2501.15564) and **wrapped by a safety
  filter** (CBF / HJ-reachability / RSS-style envelope — 2309.05837, 1708.06374), neither of which
  needs a reward, a sim, or a policy update.

RL **genuinely wins** in exactly two places, both narrower than they first look: (a) the hardest
safety-critical scenarios where demonstrations are sparse (Waymo: **>38%** fewer failures on the
hardest slice — 2212.11419), and (b) large-scale planning *when you have a fast simulator or an
aligned model to roll out in* (CaRL, CarPlanner, Raw2Drive). Even then, on real-data nuPlan, RL only
**ties** the 2023 rule-based baseline PDM-Closed — it does not decisively beat hand-written rules.

**The single cheapest discriminating experiment (§8):** front-load the *free* floor (training-free
guidance + a road-boundary safety clamp) and measure off-road departure on our 40 val episodes; if it
collapses to ~0, the RL question is closed at ~zero training cost. Only a surviving off-highway
residual justifies the (still sim-free) matched-compute reward-flag A/B.

---

## 1. Closed-loop-aware imitation — the strongest RL alternative

The core idea: train on the **policy's own rollout distribution** to eliminate covariate shift /
compounding error, **without a reward function**. This is the single most important family for us
because it fixes the failure at the *training* level (a permanent policy fix), not just at inference.

### 1.1 The problem it solves (`PUBLISHED`)
Covariate shift, a.k.a. "the DAgger problem": small policy errors push the agent into states absent
from the expert data; from there predictions are unreliable and error compounds into a failure spiral
(Ross et al. DAgger lineage; restated in 2412.05334, 2508.07029). Open-loop BC minimizes one-step
error but **can have low one-step error and still diverge in long-horizon closed loop** — demonstrated
directly in 2508.07029 (below). This is exactly our MEASURED off-road-departure mode (`INHERITED`).

### 1.2 DAgger and why it doesn't transplant cleanly (`PUBLISHED`)
Classic **DAgger** (Ross, Gordon, Bagnell 2011) queries an expert on the *policy's* visited states and
aggregates. For driving this needs an interactive expert that can answer "what would you do from this
off-distribution state?" — impractical with logged human data (2512.01993 calls repeated expert
relabeling "impractical in autonomous driving"). Our internal "DAgger proof in flight" (`INHERITED`)
therefore almost certainly means a *model-based* DAgger — relabel using the WM / a rule expert — which
is the right adaptation.

### 1.3 CAT-K — Closest-Among-Top-K (`PUBLISHED`, Zhang et al., **CVPR 2025 Oral**, arXiv:2412.05334, NVLabs)
Recipe (exact):
1. Pre-train the policy with plain BC (32 epochs in the paper).
2. **Closed-loop fine-tune (10 epochs):** at each step, take the policy's **top-K** action tokens
   `{ξ₁…ξ_K}`, simulate each one step `f(s_t, x_c)`, and **keep the one whose next state is closest to
   ground truth** `argmin d(f(s_t,x_c), ŝ_{t+1})`. Step forward with it; the visited states stay near
   the expert manifold *while being the policy's own*.
3. Loss is ordinary **cross-entropy** toward the recovery-action target — **no reward, no RL, no GAIL.**

Results (Waymo Open Sim Agents, RMM realism metric): **7M CAT-K = 0.7702 > 102M open-loop SMART-large
= 0.7614** (a 14× smaller model wins); BC→CAT-K local val 0.7581→0.7616. GAIL is dismissed as
"prone to mode collapse." **Limitations that matter to us:** needs **discrete action tokens**, a
**deterministic simulator with known inverse dynamics**, and only works "when the top-K rollouts cover
the GT mode" (i.e. the BC prior must already be decent — we have a warm-started v1, so this holds).

### 1.4 RoaD — Rollouts as Demonstrations (`PUBLISHED`, arXiv:2512.01993, 2025) — *our closest analog*
RoaD removes CAT-K's three restrictive assumptions: **no discrete actions**, **no deterministic /
known-inverse dynamics**, and it uses the **policy's closed-loop rollout directly as the BC target**
(with linear interpolation back to GT as a recovery when all K samples diverge). Needs a **stochastic
simulator**, **no reward**.

Why this is the most relevant single paper for us:
- It is validated on **AlpaSim with a VLM-based policy** — i.e. the *exact* gold-standard sim
  (AlpaSim) our program has deferred, driving a VLM policy of the family we run.
  `PUBLISHED` E2E numbers: **Driving Score 0.4443 (base) → 0.6300 (RoaD) = +41%**;
  **Collision 0.0525 → 0.0239 = −54%.** On Waymo sim-agents it matches CAT-K (RMM 0.7847 vs 0.7846,
  base 0.7591).
- **Data efficiency (decisive for a slow sim):** a **one-off** rollout dataset scores 0.7664 vs 0.7673
  for continuously-refreshed rollouts — i.e. you do **not** need a fast sim in the training loop; one
  batch of (even slow) closed-loop rollouts is ~as good. This directly answers our "no fast sim"
  constraint for this method (`HYPOTHESIS`: our slow NuRec is usable here in one-off mode).
- **Limitations:** needs a "high-fidelity simulator such as AlpaSim"; **sim2sim degrades** (0.75→0.58
  driving score across reconstruction methods) — a red flag given NuRec's ~3× OOD gap (`HYPOTHESIS`:
  the OOD gap threatens the "rollout ≈ good demonstration" assumption off-highway, exactly where we
  fail).

### 1.5 Offline value learning as a closed-loop-IL cousin (`PUBLISHED`, arXiv:2508.07029, 2025)
"From Imitation to Optimization" on WOMD (1,000 unseen scenarios) is the cleanest published statement
of why closed-loop competence ≠ one-step accuracy: **Conservative Q-Learning (CQL) vs the strongest
BC-Transformer: success 54.4% vs 17.3% (3.2×), collision 4.1% vs 31.1% (7.4× lower).** BC-T had *low
one-step error but failed long-horizon*; CQL "learns a conservative value function that enables it to
recover from minor errors." This is offline RL (needs a reward), so I file it under §5, but it is the
sharpest evidence that **the training distribution, not the loss family, is the lever.**

---

## 2. Diffusion-policy closed-loop control (we already have an anchored-diffusion planner)

### 2.1 Cost/energy **guidance at inference — training-free** (`PUBLISHED`, Diffusion-Planner, Zheng et al., **ICLR 2025 Oral**, arXiv:2501.15564)
This is a near-drop-in upgrade for our anchored-diffusion planner. Mechanism: **diffusion posterior
sampling** — at each denoising step subtract the gradient of hand-specified **differentiable energies**
from the score: `∇log p_t ≈ ∇log q_t − ∇E(μ_θ(x_t,t,C))`. Energies used, all training-free and
**composable**:
- **Collision:** signed distance ego↔neighbours per timestamp.
- **Drivable-area / off-road:** distance the plan leaves the lane per step ← *this is our exact failure
  cost.*
- **Comfort:** amount state exceeds accel/jerk limits.
- **Speed:** deviation from a target speed.

nuPlan closed-loop (`PUBLISHED`): Val14 NR **89.87** / R 82.80; **with a refinement head Test14-NR
94.80** (surpasses rule-based PDM's 90.05). Inference **~20 Hz, 0.04 s/step on an A6000.**
`HYPOTHESIS` for us: adding a drivable-area + collision energy to our planner's denoising is the
**cheapest possible off-road intervention** — zero training, no reward *model*, no sim; the "reward"
is a hand-written geometric cost we already have from maps. It composes with the safety filter (§4).

### 2.2 Closed-loop **fine-tuning** of a diffusion policy (`PUBLISHED`, DPPO, Ren et al., ICLR 2025, arXiv:2409.00588)
DPPO treats the denoising chain as an inner MDP and runs **PPO** over the whole two-layer diffusion
MDP; it reports the strongest, most stable RL fine-tuning of diffusion policies (Furniture-Bench
57→97 / 12→87 / 1→86% from sparse reward, better sim2real). **But this is RL** (needs reward + env
interaction) — so the "tuned" half of "guided/tuned diffusion" belongs in the RL tier (§5–6), while the
"guided" half (§2.1) is essentially free. I keep them separate in the ranking for this reason.

---

## 3. Planning-as-control over the WM (CEM / MPPI / MPC at inference — *partly what we already do*)

We already re-plan every step on the imagined latent (MEASURED paired **Δ ADE@2s −0.213** — *INHERITED,
brief*). The published generalization of this is sampling-based MPC in a learned latent WM:

- **MPPI** (`PUBLISHED`, Williams et al.; AutoRally 1707.05303 / 1806.00678): derivative-free, samples
  thousands of action sequences through (possibly learned) dynamics, path-integral soft-min weighting;
  proven on aggressive real-car driving with learned models.
- **TD-MPC2** (`PUBLISHED`, Hansen et al., ICLR 2024, arXiv:2310.16828): short-horizon **MPPI/CEM in
  the latent space** of an implicit WM (joint-embedding + reward + TD, no pixel decode); a single 317M
  agent solves 80+ tasks. Crucially it **amortizes** the plan with a learned **value function + policy
  prior** that (a) seeds the sampler and (b) provides a terminal value so the short horizon isn't
  myopic. This is the principled fix for the two weaknesses of pure online planning: **horizon** and
  **per-step compute.**

**When does online planning beat a trained closed-loop policy?** (`PUBLISHED`+`HYPOTHESIS`) When the
cost/goal can change at test time, when the model is more trustworthy than a not-yet-closed-loop-trained
reactive policy, and when you can afford the sampling. Its cost is **per-step compute** (K rollouts ×
horizon through the WM), mitigable by the TD-MPC2 amortization. For us this is the *natural extension of
the −0.213 win*: replace 1-shot re-planning with an MPPI/CEM search over the WM using the **same
off-road/collision/progress cost** as §2.1 — no reward learning, no sim.

---

## 4. Safety-filtered / shielded learning — the pragmatic industry pattern

A safety filter **wraps** a nominal (possibly learned) controller and **minimally modifies** its action
to keep the system inside a safe set — the "least-restrictive" principle. It does not improve nominal
driving; it **bounds the catastrophe**, which is precisely our failure mode.

Families (`PUBLISHED`):
- **Hamilton-Jacobi reachability** (2312.15347; survey 2309.05837): computes a safe set + least-restrictive
  filter offline; overrides are *infrequent but extreme*. Needs a dynamics model + disturbance bound;
  classic conservatism problem.
- **Control Barrier Functions** (many; refined by HJ in IEEE 9982203): enforce forward-invariance of a
  safe set with a **minimal, smooth** action correction solved as a small QP each step. Road-boundary
  CBFs for arbitrary road geometry are now real-time (2505.02395).
- **Predictive / MPC shielding** (2405.13863): a model-predictive safety monitor that provably keeps
  provably-safe RL inside the safe set.

**What production-ish stacks actually deploy** (`PUBLISHED`): **Mobileye RSS** (Shalev-Shwartz et al.
2017, arXiv:1708.06374) and **NVIDIA Safety Force Field** (Nistér et al. 2019) — an **add-on runtime
module in parallel to the main stack** that ingests perception + the planner's decision and acts as an
**upper-level restrictor** that can override, giving a *provable minimum safety envelope* even against
adversarial agents. "In modern autonomy stacks the safe state is enforced by a runtime safety governor
supervising downstream planners and learned components." This is the dominant industry answer to
"learned planner + safety": **wrap it, don't trust it.**

`HYPOTHESIS` for us: a road-boundary CBF / RSS-style lateral-offset clamp, using the **short-horizon WM
we already have** as the model and the **drivable-area from maps** as the safe set, is the
**highest-safety-per-GPU-hour** item on the board and targets our exact MEASURED off-road failure. It is
a *floor*, not a driver — pair it with §1–3 for nominal competence.

---

## 5. When RL genuinely wins (the honest counter-evidence)

### 5.1 The safety-critical tail (`PUBLISHED`, Waymo "Imitation Is Not Enough", arXiv:2212.11419)
On 100k+ miles of urban data, IL + RL with **simple rewards** cuts failures by **>38% on the hardest
scenarios**, while conceding IL "can perform well in low-difficulty scenarios well-covered by the
demonstration data." Read precisely: **RL earns its keep on the sparse-demonstration tail, not on the
bulk.** For us the tail is off-highway off-road departure.

### 5.2 Large-scale planning RL now matches/edges rule-based — *if* you can roll out (`PUBLISHED`)
- **CaRL** (Jaeger et al., CoRL 2025, arXiv:2504.17838): PPO + a **single simple reward** (route
  completion × soft penalties − terminal); complex shaped rewards *fail to scale* (mini-batch
  256→1024 collapses Roach reward 34→2 DS). Cost: **300M samples CARLA / 500M nuPlan, 8×A100 for a
  week**, needs a **speed-optimized simulator + privileged inputs**. Result: CARLA longest6-v2
  **64 DS** (best learning method; still < rule-based PDM-Lite 73); nuPlan Val14 **91.3 NR / 90.6 R** —
  **below** rule-based PDM-Closed (92.8 / 92.1). So even the best-scaled RL only *approaches* hand-rules.
- **CarPlanner** (Zhang et al., CVPR 2025, arXiv:2502.19908): "first RL planner to beat IL **and**
  rule-based on nuPlan." PPO + expert-guided reward (−displacement + collision/off-road penalties),
  trained over a **learned non-reactive world model on logged nuPlan data — no simulator** (regime =
  ours!), **2×3090**. Test14-Random CLS-NR **94.07** (PDM 90.05, PLUTO-IL 91.92) but Val14 **91.45 vs
  PDM 91.21 — a +0.24 tie.** So the "RL beats rules" headline is **scenario-set-dependent.**
- **Raw2Drive** (arXiv:2505.16394, 2025): model-based RL **in imagination** over aligned privileged/raw
  WMs — the closest to *our* WM-RL. Cost **40–64 H800-GPU-days**; Bench2Drive **DS 71.36** (> IL
  DriveTransformer 63.46) but far under the **privileged** Think2Drive 91.85. Confirms WM-RL beats IL
  **but is expensive and model-limited.**

### 5.3 The honest asterisk on all nuPlan "wins" (`PUBLISHED`, 2306.07962 / nuPlan-R 2511.10403)
Rule-based **PDM-Closed** (2023) still ties or tops learned planners on standard nuPlan; closed-loop
metrics reward conservatism. On the harder **reactive** nuPlan-R, PDM *drops* (Test14-Random 91.64→90.62,
hard 75.19→67.33) while learned methods hold — so rule dominance is **partly a benchmark artifact**, and
learned closed-loop methods look better the more realistic the eval. Net: nobody has shown RL is
*necessary*; they've shown closed-loop-aware learning (RL **or** CAT-K/RoaD-style IL) closes the gap.

---

## 6. Model-based / imagined-rollout RL (highest ceiling, highest cost/instability)

Dreamer-lineage (DreamerV3, Think2Drive, DreamerAD, Raw2Drive — `PUBLISHED`): learn a WM, run RL
**inside imagined rollouts** (free rollouts, no live sim — attractive for us), generating dense reward
from the latent. Think2Drive solved CARLA-v2 leaderboard; DreamerAD/Raw2Drive report strong CARLA
closed-loop. **Two dangers for our regime** (`HYPOTHESIS`): (a) cost (Raw2Drive 40–64 H800-days), and
(b) **the policy exploits WM inaccuracy** — and our WM has a ~3× OOD gap *exactly off-highway where we
fail*, so imagined-rollout RL is most likely to hallucinate competence precisely in the failure region.
This is the strongest argument for preferring inference-time guidance + a safety filter (which cannot
"learn to game" the model) over reward-driven imagination RL, until the WM's off-highway fidelity is
independently verified.

---

## 7. HYPOTHESIS ranking for OUR stack — cheapest-and-safest first

Ranked by (training cost + sim dependence + instability + sim2real/OOD risk), with expected benefit and
the single biggest risk. **All rows are `HYPOTHESIS`; the published support is cited.**

| # | Method | Cost / sim need | Reward? | Targets our off-road failure? | Biggest risk | Key published support |
|---|--------|-----------------|---------|-------------------------------|--------------|-----------------------|
| **1** | **Safety filter / shield** (road-boundary CBF or RSS-style clamp over the WM) | ~free; inference-time; no training/sim | No | **Directly** (bounds the departure) | Only a floor; conservatism/false overrides | RSS 1708.06374; SFF; HJ 2309.05837/2312.15347; CBF 2505.02395 |
| **2** | **Guided diffusion at inference** (training-free off-road/collision/comfort energy in our denoiser) | ~free; inference-time; no reward *model*/sim | No (hand cost) | **Directly** (drivable-area energy) | Cost must be well-shaped; small per-step compute | Diffusion-Planner 2501.15564 |
| **3** | **WM-planning / MPC** (upgrade our −0.213 re-planner to MPPI/CEM over the WM) | low; inference compute only; no reward/sim | No (hand cost) | Yes (progress + off-road cost) | Per-step compute; horizon myopia (fix via value prior) | TD-MPC2 2310.16828; MPPI 1707.05303 |
| **4** | **Closed-loop-aware IL** (RoaD / CAT-K / model-based DAgger — *already in flight*) | med; **needs closed-loop rollouts** (one-off OK) | No | Yes (permanent policy fix) | Our slow/OOD NuRec ⇒ rollout≈demo may break off-highway | RoaD 2512.01993; CAT-K 2412.05334 |
| **5** | **Offline RL** (CQL/IQL on logged data) | med; **no sim** (logged-native) | Yes | Yes (recovers from errors) | Instability + OOD value overestimation; departs from warm-start | 2508.07029; survey 2503.23650 |
| **6** | **Imagined-rollout RL** (Dreamer/Raw2Drive-style in our WM) | **high** (10s of GPU-days); reward design | Yes | Yes, esp. the tail | **Policy games WM's off-highway OOD gap**; instability | Raw2Drive 2505.16394; CaRL 2504.17838; Waymo 2212.11419 |

Notes: **#1–3 are inference-time and should be built regardless** — they are the free/cheap floor and
each directly attacks the MEASURED failure. **#4 is the highest-ceiling non-RL option** (fixes the
policy itself) and we're already partway (DAgger in flight); its only gate is rollout throughput, which
RoaD's one-off-data result (§1.4) largely dissolves. **#5–6 are the RL tier**; they win published
comparisons only on the sparse tail or with a fast/aligned model, and #6 carries a specific WM-gaming
hazard given our OOD gap. The "tuned diffusion" (DPPO) half of the diffusion bucket lives at #6-cost.

---

## 8. The single cheapest discriminating experiment: "do we even need RL?"

**Design goal:** decide RL-vs-not by isolating the *one* variable — the reward — while sharing
everything else, and by front-loading the outcome that can end the question for free.

### Primary test (Gate 0 — near-zero cost, can *close* the RL question)
Add to the **existing** warm-started anchored-diffusion planner, with **no training**:
(a) training-free **drivable-area + collision energy guidance** in the denoiser (§2.1), and
(b) a **road-boundary safety clamp** (§4). Evaluate **closed-loop off-road departure rate + ADE@2s on
the 40 val episodes**, scored with the house **episode-cluster bootstrap**, **paired** vs the
un-guided planner (`taniteval/ci.py`).
**Pre-registered read:** if off-road departures collapse to ~0 (no significant residual on the
off-highway slice), **RL is not needed for our MEASURED failure** — you cannot beat ~0, and the whole
capability came from inference-time cost + a shield, no reward. Cost: a few eval GPU-hours, no retrain.

### Escalation only if a residual survives (Gate 1 — matched-compute reward flag, still **no live sim**)
Fine-tune the same planner two ways over **identical imagined rollouts from our WM** (so compute is
matched and no fast sim is used), differing **only** in the training signal:
- **IL-CL arm:** RoaD/CAT-K-style — supervise the rollout toward the logged expert continuation
  (no reward).
- **RL arm:** optimize a **simple** off-road/collision/route-completion reward over the same rollouts
  (CaRL/CarPlanner-style simple reward — §5.2).
Same data, horizon, and rollout machinery — a single harness with a **reward flag on/off**. Primary
metric: **off-highway off-road departure rate**, paired episode-cluster bootstrap on the 40 val
episodes. **Pre-commit both outcomes:** if the RL arm's reduction over IL-CL is **inside the CI**, the
verdict is *"no RL — closed-loop-aware IL + guidance + shield is the stack"*; if RL **significantly**
beats IL-CL **on the off-highway tail**, RL earns a **scoped tail-specialist** role (consistent with
Waymo 2212.11419), not a wholesale adoption.

**Why this is the cheapest true discriminator:** Gate 0 spends ~zero training and, in the outcome our
own −0.213 re-planning win makes plausible, *decides RL-vs-not outright*. Gate 1 escalates only on a
real residual, still avoids the live sim (rolls out in imagination), matches compute across arms, and
puts the entire decision on the single bit that defines "RL": the presence of a reward. It also
surfaces the §6 hazard — if the RL arm wins *in imagination* but the gain doesn't survive on held-out
episodes, that is the WM-gaming signature, and the honest verdict stays "no RL until WM off-highway
fidelity is verified."

---

## 9. Sources (all `PUBLISHED` unless noted)

- CAT-K — Closed-Loop SFT of Tokenized Traffic Models, CVPR 2025 Oral — https://arxiv.org/abs/2412.05334 · code https://github.com/NVlabs/catk
- RoaD — Rollouts as Demonstrations for Closed-Loop SFT of AD Policies, 2025 — https://arxiv.org/abs/2512.01993
- From Imitation to Optimization (BC vs CQL, WOMD), 2025 — https://arxiv.org/abs/2508.07029
- Diffusion-Planner — Diffusion-Based Planning with Flexible Guidance, ICLR 2025 Oral — https://arxiv.org/abs/2501.15564 · code https://github.com/ZhengYinan-AIR/Diffusion-Planner
- DPPO — Diffusion Policy Policy Optimization, ICLR 2025 — https://arxiv.org/abs/2409.00588
- TD-MPC2 — Scalable, Robust World Models for Continuous Control, ICLR 2024 — https://arxiv.org/abs/2310.16828
- MPPI / AutoRally aggressive driving — https://arxiv.org/abs/1707.05303 · https://arxiv.org/abs/1806.00678
- The Safety Filter: A Unified View (Hsu, Hu, Fisac), Annu. Rev. Control 2024 — https://arxiv.org/abs/2309.05837 (Annual Reviews page paywalled/403)
- On Safety and Liveness Filtering via HJ Reachability — https://arxiv.org/abs/2312.15347
- Real-time CBF safety filter, arbitrary road boundaries — https://arxiv.org/abs/2505.02395
- Dynamic Model Predictive Shielding — https://arxiv.org/abs/2405.13863
- Mobileye RSS — On a Formal Model of Safe and Scalable Self-Driving Cars — https://arxiv.org/abs/1708.06374 · NVIDIA Safety Force Field (Nistér et al. 2019)
- Imitation Is Not Enough (Waymo) — https://arxiv.org/abs/2212.11419
- CaRL — Learning Scalable Planning Policies with Simple Rewards, CoRL 2025 — https://arxiv.org/abs/2504.17838 · code https://github.com/autonomousvision/CaRL
- CarPlanner — Auto-regressive Trajectory Planning for Large-scale RL, CVPR 2025 — https://arxiv.org/abs/2502.19908
- Raw2Drive — RL with Aligned World Models for E2E AD, 2025 — https://arxiv.org/abs/2505.16394
- Parting with Misconceptions about Learning-based Motion Planning (PDM-Closed), CoRL 2023 — https://arxiv.org/abs/2306.07962
- nuPlan-R — reactive closed-loop benchmark, 2025 — https://arxiv.org/abs/2511.10403
- Survey of RL-based motion planning for AD, 2025 — https://arxiv.org/abs/2503.23650

*Internal facts used (INHERITED from brief, not re-measured here): imagination-in-the-loop re-planning
MEASURED paired Δ ADE@2s −0.213; NuRec slow + ~3× OOD; off-road departure failure, mostly off-highway;
DAgger proof in flight; anchored-diffusion planner warm-started from supervised v1.*

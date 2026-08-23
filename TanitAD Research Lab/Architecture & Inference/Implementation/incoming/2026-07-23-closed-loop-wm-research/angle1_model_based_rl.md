# Angle 1 — Learning a policy THROUGH / INSIDE a world model (model-based-RL family)

**Closed-loop world-model research sweep, angle 1 of 3.**
Author: TanitAD research subagent · Compiled 2026-07-23, ~16:20 Europe/Berlin (UTC+2).
Web research only (WebSearch + WebFetch); no pod use.

**Evidence-class legend** (per `Operating standard`): `PUBLISHED` = external paper, cited with
year + venue/arXiv id · `HYPOTHESIS` = my inference for our setup · `INHERITED` = a TanitAD
fact taken from our registry/memory, NOT re-verified by me here.

---

## TL;DR (read this if nothing else)

- The entire "policy through a world model" canon splits on one axis that decides everything for
  us: **online (the agent collects fresh data by acting) vs offline (logged data only)**. Dreamer,
  DayDreamer, TD-MPC, MBPO and Dyna are all **fundamentally online** — they interleave imagination
  with real environment steps and a real environment *reward*. **We have neither** (no fast sim, no
  environment reward, logged comma2k19 + PhysicalAI only). `HYPOTHESIS`
- The methods that actually match our constraints are the **offline / differentiable-model-imitation**
  corner: SVG-style analytic gradients backpropagated **through a differentiable world model** with
  an **imitation-divergence cost** (not an environment reward), guarded by **offline-MBRL pessimism**
  (MOPO/MOReL/COMBO). Two driving papers do almost exactly the recommended experiment on logged Waymo
  data and report the **exact failure we measured getting fixed** (off-road/collision under compounding
  error). `HYPOTHESIS`
- **Recommendation:** train the anchored-diffusion planner in closed loop by **differentiating a short
  imagined rollout of our own WM** (analytic/"dynamics-backprop" gradient, i.e. what DreamerV3 itself
  reduces to for continuous control), cost = imitation divergence + a shaped drivability cost, with a
  learned terminal **value** (TD-MPC's contribution) to see past the short horizon. **Single biggest
  risk: world-model exploitation** — the planner learns to drive on hallucinated dynamics that are
  wrong exactly where our data is thin (off-highway). `HYPOTHESIS`

---

## 0. The framing that reorganizes the whole family (online vs offline)

`PUBLISHED` Dreamer (v1/v2/v3), DayDreamer, TD-MPC(2), MBPO and Dyna are **online** model-based RL:
each alternates *(a)* collect real transitions with the current policy, *(b)* fit/refresh the world
model, *(c)* improve the policy inside the model, and every one of them consumes an **environment
reward** `r_t` emitted by the real task. (Sutton, *Dyna*, ML 1990/SIGART 1991; Janner et al., NeurIPS
2019, arXiv:1906.08253; Hafner et al., arXiv:2301.04104; Wu et al., CoRL 2022, arXiv:2206.14176;
Hansen et al., ICML 2022, arXiv:2203.04955.)

`INHERITED` (our registry/memory) Our regime is the opposite corner: **logged data, no fast sim**
(AlpaSim/NuRec runs ~0.3–1× real-time and is ~3× OOD), and **no environment reward** — we are trained
by open-loop imitation, and the measured failure is that the planner *drives off-road under its own
compounding errors, worse off-highway* (open-loop ADE ~0.45 m collapses to ~1.69 m closed-loop).

`HYPOTHESIS` So the literature is useful to us **mechanistically** (how to optimize a policy against a
learned model) but **not as drop-in recipes** — every method below has to be re-derived with two
substitutions: *(i)* the environment reward `r_t` → an **imitation + shaped-drivability cost** we can
evaluate inside the WM; *(ii)* "collect fresh real data" → **stay anchored to the log** (DAgger-style
resets / pessimism), because we cannot safely act online. Sections 1–4 cover the canon; §5 covers the
driving papers that already made these substitutions; §6 is the reward/cost menu; §7 is the call.

---

## 1. The Dreamer line (DreamerV2 / V3, DayDreamer)

**What it is.** `PUBLISHED` Learn a recurrent latent world model (RSSM), then train an **actor-critic
entirely inside it** on **imagined latent rollouts** — no real environment steps during behavior
learning. From replay states, roll the RSSM prior + current actor forward `H` steps (DreamerV3 default
`H=15`), predict reward and a continue flag at each latent, and train the actor to maximize the critic
and the critic to regress a **λ-return** over the imagined trajectory
`R^λ_t = r_t + γc_t[(1−λ)v(ẑ_{t+1}) + λR^λ_{t+1}]`. (Hafner et al., *Mastering Atari with Discrete
World Models* / DreamerV2, ICLR 2021, arXiv:2010.02193; *Mastering Diverse Domains through World
Models* / DreamerV3, arXiv:2301.04104, 2023, later in *Nature* 2025.)

**The actor gradient — the exact mechanism (directly relevant to us).** `PUBLISHED` DreamerV3's actor
loss interpolates two estimators with a single knob ρ, plus entropy:
`−ρ · ln p(â_t|ẑ_t) · sg(V^λ_t − v(ẑ_t))` (REINFORCE / score-function) `− (1−ρ) · V^λ_t`
(**dynamics backpropagation** — differentiate the return *through* the differentiable RSSM dynamics
and reward, i.e. the analytic/pathwise gradient) `− η · H[a_t|ẑ_t]`. **For continuous control they use
ρ=0 (pure dynamics-backprop) with η=1e-4; for discrete Atari they use ρ=1 (pure REINFORCE) with
η=1e-3.** (arXiv:2301.04104, robustness section; confirmed against the DreamerV3 loss write-up.)
DreamerV1 was pure analytic value-gradient through dynamics (Hafner et al., *Dream to Control*, ICLR
2020, arXiv:1912.01603); V2 added categorical latents with **straight-through** gradients in the WM.
`HYPOTHESIS` **This is the single most important fact for us: for a continuous trajectory planner, the
Dreamer recipe *is* analytic backprop-through-the-model (Angle-1 #3) — the "RL" framing collapses onto
the differentiable-WM framing.** DreamerV3's robustness stack (symlog obs, two-hot symexp reward/value,
KL balancing + free bits, 1% unimix, **percentile return normalization**) is what lets one hyperparameter
set span domains — worth borrowing wholesale for numerical stability.

**Sample efficiency.** `PUBLISHED` DreamerV3 reaches strong scores across >150 tasks with one config
and is highly sample-efficient; **DayDreamer** (Wu, Escontrela, Hafner, Goldberg, Abbeel, CoRL 2022,
arXiv:2206.14176) is the headline proof it works from *small real-world* data: a physical quadruped
learned to stand and walk **in ~1 hour of real interaction, no simulator**, and arms learned pick-place
from sparse reward, beating model-free baselines. `HYPOTHESIS` DayDreamer is the closest existence-proof
that imagined-rollout learning tolerates limited real data — **but it still requires online interaction
on the real robot**, which is exactly the affordance we lack for cars.

**Known failure modes.** `PUBLISHED` (1) **World-model exploitation / hallucinated reward**: the actor
optimizes the *model's* predicted return and will find and exploit regions where the model is wrong
(unvisited states), producing behavior that looks great in imagination and fails in reality — the
classic MBRL "objective mismatch" (Lambert et al., L4DC 2020, arXiv:2002.04523) and, recently
formalized, essentially unavoidable over large policy sets (*Imperfect World Models are Exploitable*,
arXiv:2605.15960, 2026 preprint — treat as recent/unreplicated). (2) **Non-controllable dynamics**
swamp the imagination in driving (other agents move independently of ego action), degrading the model
(Iso-Dream, §5). (3) Long imagined horizons compound model error.

**Applied to DRIVING (yes, several).** `PUBLISHED`
- **Think2Drive** (Li et al., ECCV 2024, arXiv:2402.16720): first model-based RL for AD; DreamerV3
  trains the planner *inside* a learned latent world model acting as a neural simulator — **expert-level
  on CARLA v2 in ~3 days on one A6000**, learning the transition/reward/termination models then
  maximizing predicted reward. **But**: privileged BEV state input and a CARLA reward — a simulator,
  not logged sensor data.
- **CarDreamer** (Gao et al., 2024, arXiv:2405.09111): open-source DreamerV3 (18M) driving platform in
  CARLA.
- **DreamerAD** (arXiv:2603.24587, 2026 preprint — recent, unverified by me): DreamerV3-style imagined-
  rollout RL trained on **real-world driving data** (explicitly to avoid the cost/safety of online RL),
  with an "autoregressive **dense reward model** operating on latents" — reports **87.7 EPDMS on NavSim
  v2 (SOTA)**. Notable because NavSim is a **log-based** benchmark like ours; it is the strongest recent
  signal that latent imagined-rollout learning transfers to logged AV data, and its "learned dense
  reward on latents" is one concrete answer to our missing-reward problem.
- **Iso-Dream** (Pan et al., NeurIPS 2022, arXiv:2205.13817; ++ journal arXiv:2303.14889): splits
  controllable vs non-controllable latent dynamics and plans on the decoupled controllable part —
  directly targets the driving-specific WM failure mode above.

---

## 2. TD-MPC / TD-MPC2 (latent MPC + learned value)

**What it is.** `PUBLISHED` TD-MPC (Hansen, Wang, Su, ICML 2022, arXiv:2203.04955) learns a
**decoder-free, task-oriented latent dynamics model** ("TOLD") by joint-embedding + reward + TD
learning, and at every step does **short-horizon trajectory optimization by MPPI/CEM in latent space**
(sample action sequences, roll them through the latent model, score them), then **bootstraps past the
short planning horizon with a learned terminal value** `Q/V` appended to the end of each rollout. It
also seeds the MPPI samples with a **learned policy prior** to speed convergence. TD-MPC2 (Hansen, Su,
Wang, ICLR 2024, arXiv:2310.16828) makes it scale/robust: SimNorm latent normalization, LayerNorm/Mish,
an **ensemble of Q-functions**, discrete (log-transformed) reward/value regression, and simplified
momentum-free MPPI — enabling a **single 317M-parameter agent to solve 80 tasks** across domains/embodiments
with one hyperparameter set (~300× the params of prior latent-MPC world models).

**What the learned value + closed-loop training buys (the crux for us).** `HYPOTHESIS` Our current stack
(action-conditioned imagination + anchored-diffusion planner scoring rollouts) **is already ~"TD-MPC
without the value and without closed-loop targets"** — we plan by imagination but score with an
open-loop-trained head. TD-MPC's two additions are exactly the two things we're missing:
1. **A terminal value → the horizon decoupling.** `PUBLISHED` A short planning horizon (single-digit
   steps in both papers) keeps model error from compounding, while the learned value carries the
   *long-horizon* consequence (does this cruise lead to a safe merge 8 s later?). `HYPOTHESIS` This is
   precisely our measured pathology: open-loop ADE does not predict closed-loop drivability, i.e. our
   planner has no long-horizon consequence signal — a learned value is the cheapest structural fix.
2. **TD targets = closed-loop training signal without unrolling the full horizon.** The value is
   trained by bootstrapping (its own prediction one step ahead), so you get a long-horizon objective
   from short rollouts — sidestepping the exploding-gradient problem of §3.

`HYPOTHESIS` **Caveat for our regime:** TD-MPC/TD-MPC2 as published are **online with an environment
reward**; the value is a *return* predictor. To use it we must define what the value estimates (§6) —
an imitation-consistency value or a shaped-cost value — because we have no reward to bootstrap from.

---

## 3. Analytic-gradient / differentiable-world-model policy learning (SVG, backprop-through-model, SHAC)

**What it is.** `PUBLISHED` If the dynamics are differentiable, the return/cost of a rollout is an
end-to-end differentiable function of the policy parameters, so you can get the **exact pathwise
(analytic) policy gradient by backpropagating the cost through the model** — no score-function
estimator. **Stochastic Value Gradients** (Heess et al., NeurIPS 2015, arXiv:1510.09142) made this work
for stochastic control via the **reparameterization trick**: write `a = π(s,η;θ)` and `s' = f(s,a,ξ;φ)`
as deterministic functions of exogenous noise, then differentiate the (stochastic) Bellman value.
SVG(1) uses a one-step model + a value; SVG(∞) differentiates the full trajectory. A key SVG insight
(Heess 2015; Amos et al., *On the model-based SVG*, L4DC 2021, arXiv:2008.12775): use the model **for
its gradients on real data**, not for generating long fantasy rollouts — this sidesteps compounded
model error and is stable with **short horizons**. This is the same object as DreamerV3's "dynamics
backprop" (ρ=0, §1).

**Stability — the central difficulty, and the mitigations.** `PUBLISHED` Backprop-through-time over a
rollout multiplies a chain of state-Jacobians; when the dynamics are stiff/chaotic the **product's
spectrum blows up or vanishes**, giving high-variance, poorly-conditioned gradients — even when the
objective is smooth. This is the thesis of *Gradients are Not All You Need* (Metz, Freeman, Schoenholz,
Kachman, 2021, arXiv:2111.05803): the failure is diagnosable from the **Jacobian spectrum** of the
iterated map. Standard mitigations that work:
- **Short / truncated horizons + a learned terminal value** to smooth the loss landscape — **SHAC**
  (Xu et al., *Accelerated Policy Learning with Parallel Differentiable Simulation*, ICLR 2022,
  arXiv:2204.07137): split the task into short sub-windows, backprop analytic gradients only within a
  window, and let a **model-free-trained critic** carry value across window boundaries. This is the
  analytic-gradient twin of TD-MPC's horizon decoupling.
- **Gradient detachment at state boundaries** (one-step / truncated BPTT), gradient clipping.
- **Smoothing non-differentiable events** (contacts, collisions) into continuous surrogates.
- Interpolating with a low-variance biased estimator (exactly DreamerV3's ρ knob).

**Applied to DRIVING — this is where our exact experiment already exists.** `PUBLISHED`
- **Nachkov, Paudel, Van Gool, *Autonomous Vehicle Controllers From End-to-End Differentiable
  Simulation*** (2024, arXiv:2409.07965). APG through **Waymax** (differentiable JAX sim) on **~500k
  logged WOMD scenarios**. Objective is **pure imitation** — trajectory divergence
  `L = (1/T)Σ‖ŝ_t − s_t‖` — with **gradients flowing through the dynamics, not any reward**; 1 s history,
  **~8 s (80-frame) rollout**. They explicitly hit the §3 problem ("vanishing or exploding gradients …
  stemming from the spectrum of the Jacobian") and fix it with **gradient detachment at state
  boundaries, curriculum resets to the log every n steps (reduced over training), smoothed dynamics
  (ε in sqrt, arctan2 for mod), and dense per-step supervision.** Results vs behavior cloning: **collision
  0.0800 vs 0.2475, off-road 0.0282 vs 0.2673** (single-agent planning), and multi-agent **ADE 1.8096 vs
  4.1123**. **Honest tension:** single-agent **minADE is *worse* than BC (2.0083 vs 1.4157)** — closed-loop
  analytic training **trades open-loop displacement accuracy for drivability/robustness.**
- **Nachkov et al., *Model-Based Vehicle Control Using Analytic World Models* ("Dream to Drive")** (2025,
  arXiv:2502.10012). Extends the above from a *given* sim to a **learned "analytic world model"** trained
  *through* the differentiable sim, and to three task types (relative-odometry prediction, optimal
  planners via inverse-kinematics, inverse optimal-state estimation as a confidence measure). The
  differentiable-sim training signal **halves prediction error** (relative-odometry ADE **0.3475 m with
  DiffSim vs 0.7900 m without**); a **6M-param** agent runs real-time; MPC with 20-step rollouts.

`HYPOTHESIS` The Nachkov line is the **most direct precedent for what the brief proposes** (differentiate
an imagined closed-loop rollout into the planner, on logged data, no environment reward) and it reports
**our exact failure mode — off-road/collision under compounding error — being fixed.** The one gap: they
backprop through **Waymax's known kinematic bicycle model** (exact, non-exploitable), whereas we would
backprop through a **learned** WM — which is the whole ballgame (§7 risk).

---

## 4. MBPO / Dyna (short model rollouts to augment real data)

**What it is.** `PUBLISHED` **Dyna** (Sutton, 1990/1991) is the ancestor: use the learned model to
generate extra imagined transitions and train a model-free policy on real + imagined data. **MBPO**
(Janner, Fu, Zhang, Levine, *When to Trust Your Model*, NeurIPS 2019, arXiv:1906.08253) makes it work
at scale with the key trick: **short, branched rollouts started from *real replay-buffer states*** —
not long rollouts from the initial state. Short branches keep model error bounded and **decouple the
model horizon from the task horizon**; because each real sample spawns many synthetic ones, MBPO can
take **20–40 policy-gradient steps per real environment step** (far more than is stable model-free),
giving big sample-efficiency wins while matching model-free asymptotic performance. Their theory ties
the safe rollout length `k` to an empirical estimate of **model generalization error**.

**Sample-efficiency vs model-bias.** `PUBLISHED` The entire method is a bias/efficiency dial: longer
`k` = more data leverage but more compounded model bias; MBPO keeps `k` short (often 1–a few steps) and
grows it only as the model improves. `HYPOTHESIS` For us MBPO is the **least directly applicable** of
the four: it presupposes an **online model-free RL loop with an environment reward** (SAC on the mixed
buffer), and it uses the model as a *data generator*, discarding exactly the differentiable structure
our WM+planner make available. Its **transferable idea** is the discipline: **keep imagined horizons
short and tie their length to measured model accuracy** — which we should adopt regardless of which
method we pick. The **offline** cousins (MOPO/MOReL/COMBO, §6) are the version that fits our data.

---

## 5. The driving-specific closed-loop-on-logged-data cluster (our true peer group)

`PUBLISHED` These already made the two substitutions from §0 (env-reward → imitation/adversarial cost;
online → stay-on-log). They are our real reference class:

| Work | venue / id | Model used how | Where the training signal comes from | Data / loop |
|---|---|---|---|---|
| **MILE** (Hu et al., Wayve) | NeurIPS 2022, arXiv:2210.07729 | BEV latent world model + policy, jointly learned | **Imitation** (behavior cloning) of expert actions inside the model | **Offline** logged urban driving; +31% driving score in an unseen CARLA town |
| **Urban Driver** (Scheel et al., Woven/L5) | CoRL 2021, arXiv:2109.13333 | **Differentiable data-driven simulator built from logs** | **Closed-loop policy gradients** on imitation objective (unroll policy in the diff-sim, match expert) | 100 h logs; closed-loop training; **deployed on a real AV** |
| **MGAIL for AD** (Bronstein et al., Waymo) | 2022, arXiv:2210.09539 | Model-based **adversarial** imitation, hierarchical for routes | **Learned discriminator** = dense reward everywhere (incl. off-distribution recovery); **mix closed-loop MGAIL + open-loop BC** | 100k+ miles SF; closed-loop eval with reactive agents; best policy ≈ expert |
| **Nachkov APG** | 2024, arXiv:2409.07965 | **Analytic gradient through diff-sim** | **Imitation divergence, gradient through dynamics** (no reward) | 500k WOMD logs; 8 s rollout; collision/off-road ≪ BC |
| **Dream to Drive** | 2025, arXiv:2502.10012 | **Learned analytic world model** through diff-sim | Diff-sim gradients supervise prediction/planning | WOMD logs; halves prediction error |
| **Think2Drive** | ECCV 2024, arXiv:2402.16720 | DreamerV3 policy inside learned latent WM | **Simulator reward** (CARLA) | Online in CARLA v2 (not logged sensors) |
| **DreamerAD** | arXiv:2603.24587, 2026 pre | DreamerV3-style imagined-rollout RL | **Learned dense reward model on latents** | Real-world data; 87.7 EPDMS NavSim v2 |

`HYPOTHESIS` The pattern across the *deployed-on-real-cars* ones (Urban Driver, MGAIL) is identical and
worth stating as a near-law: **closed-loop training through a (differentiable or adversarial) model,
mixed with an open-loop BC anchor, on logged data.** That mixture — not pure closed-loop, not pure BC —
is the recurring winning recipe.

---

## 6. Where does the training signal come from? (no environment reward in imitation)

This is the load-bearing design question. Menu, cheapest→richest, each with a citation for precedent:

1. **Imitation divergence as the cost (differentiable).** `PUBLISHED` Roll the planner forward in the
   differentiable WM, penalize `Σ‖ŝ_t − s^{expert}_t‖` (position/heading/speed), backprop through the WM
   (Urban Driver, arXiv:2109.13333; Nachkov, arXiv:2409.07965). `HYPOTHESIS` **Cheapest for us — reuses
   the signal we already train on**, just unrolled closed-loop. Weakness: only supervises where the
   expert went; off-distribution recovery comes *only* through the dynamics prior, so it needs (3)/(5).
2. **Learned discriminator / adversarial cost.** `PUBLISHED` A discriminator scores "expert-like vs
   agent-like" states in closed loop, giving a **dense reward everywhere including off-distribution** —
   the mechanism that lets MGAIL recover from its own errors (arXiv:2210.09539). `HYPOTHESIS` Best
   answer to the *recovery* problem; cost is GAN-style training instability.
3. **Hand-designed shaped drivability cost.** `PUBLISHED` Collision, off-road, lane-keeping, progress,
   comfort — the Think2Drive/TD-MPC route (arXiv:2402.16720; arXiv:2203.04955). `HYPOTHESIS` Requires
   the WM to expose queryable predicates (decoded BEV occupancy / drivable-area / agent boxes). Directly
   attacks our measured off-road failure; needs a decodable state, which our BEV/decoder path can give.
4. **Learned dense reward model on latents.** `PUBLISHED` Train a reward head on the latent to emit
   per-step credit (DreamerAD, arXiv:2603.24587). `HYPOTHESIS` Flexible but adds a second learned object
   that can *itself* be exploited (reward hacking) — highest-variance option.
5. **Pessimism / uncertainty penalty (the guard-rail, not a reward by itself).** `PUBLISHED` Offline MBRL
   subtracts a penalty for model uncertainty so the policy cannot exploit OOD model error: **MOPO** (Yu
   et al., NeurIPS 2020, arXiv:2005.13239) soft-penalizes by an ensemble error estimate; **MOReL**
   (Kidambi et al., NeurIPS 2020, arXiv:2005.05951) builds a **pessimistic MDP** with a hard penalty via
   an unknown-state detector; **COMBO** (Yu et al., NeurIPS 2021, arXiv:2102.08363) penalizes value on
   OOD model samples (CQL-style, no explicit uncertainty). `HYPOTHESIS` **Not optional for us** — this is
   the mechanism that keeps (1)–(4) from steering the planner into hallucinated regions off-highway.

`HYPOTHESIS` **Concrete proposed cost** for the planner rollout in imagination:
`J = Σ_t [ w_im·‖ŝ_t − s^{exp}_t‖ + w_dr·c_drivable(ẑ_t) + w_col·c_collision(ẑ_t) ] + γ^H·V(ẑ_H)
     − w_pess·Disagreement_ensemble(ẑ_t)`,
optimized by analytic backprop through the WM over a **short truncated horizon** (grow with measured WM
accuracy, MBPO-style), with `V` a learned terminal value (TD-MPC) and the pessimism term an ensemble/
dropout disagreement over WM predictions (MOPO/MOReL). Keep an **open-loop BC anchor** in the loss
(MGAIL/Urban-Driver law).

---

## 7. HYPOTHESIS — the call for our setup

**Most promising method to train the planner in closed loop via imagination:**
`HYPOTHESIS` **Analytic-gradient closed-loop imitation through our differentiable WM (Angle-1 #3, in
SVG/SHAC form), upgraded with a TD-MPC-style learned terminal value, and hard-guarded by offline-MBRL
pessimism.** Concretely: unroll the anchored-diffusion planner inside the WM for a short horizon,
score with the §6 cost, backprop the cost through the WM into the planner (reparameterized/pathwise),
truncate BPTT at short windows with the value carrying the tail, and subtract an ensemble-disagreement
penalty.

**Why this over the alternatives (settled by matching assets + the cheapest discriminating precedent):**
- It **matches what we already have** — a differentiable WM and a differentiable planner — and needs
  **no fast sim** (the constraint that rules out naive Dreamer/TD-MPC/MBPO online loops). `HYPOTHESIS`
- For a **continuous** trajectory, **DreamerV3 itself uses ρ=0 = pure dynamics-backprop** (§1) — so the
  "should we do Dreamer-actor-critic or analytic gradients?" question is partly a false choice; the
  Dreamer recipe *reduces to* analytic gradients in our regime. Adopting DreamerV3's numerical stack
  (symlog, two-hot value, percentile normalization, KL-balance/free-bits) gives the stability without
  the online loop. `PUBLISHED` mechanism (arXiv:2301.04104) · `HYPOTHESIS` the reduction.
- **Nachkov (arXiv:2409.07965) already ran essentially this experiment on logged Waymo data and reports
  our exact failure mode fixed** (collision 0.08 vs 0.25, off-road 0.028 vs 0.27 vs BC) — the cheapest
  discriminating precedent points here. `PUBLISHED`
- **TD-MPC2's learned value is the single cheapest structural upgrade** to our current imagination+planner
  (adds the long-horizon consequence signal our open-loop head lacks) and can be added test-time-only
  first (MPPI over imagination + value) before any closed-loop training — a low-risk first step.
  `HYPOTHESIS`
- **MBPO/Dyna is the weakest fit** (online, model-free, reward-dependent, discards differentiability);
  keep only its "short horizon tied to measured model error" discipline. `HYPOTHESIS`

**Single biggest risk: WORLD-MODEL EXPLOITATION.** `HYPOTHESIS` The planner will drive on the *gradients
and predictions of a learned WM*, and it will find the places where that WM is wrong — precisely the
thin-data off-highway regime where our closed-loop failure already lives. Unlike Urban Driver / Nachkov
(who backprop through an **exact kinematic sim** that cannot be exploited), our dynamics are learned, so
we inherit the full MBRL exploitation problem (`PUBLISHED` objective mismatch, Lambert et al. L4DC 2020,
arXiv:2002.04523; formalized in arXiv:2605.15960, 2026 pre) **amplified by covariate shift off the log**.
A planner that looks perfect in imagination can be worse on the road than the open-loop baseline — and
because open-loop ADE does not predict closed-loop behavior for us (`INHERITED`), we could not even see
it in our current metrics.
**Mitigation bundle (mandatory, not optional):** (i) **offline-MBRL pessimism** — ensemble/dropout
disagreement penalty on WM predictions, MOPO/MOReL-style, so the planner is repelled from OOD latents;
(ii) **short truncated horizons grown with measured WM fidelity** (MBPO discipline) + a learned value for
the tail (SHAC/TD-MPC); (iii) **keep the log anchor** — mix open-loop BC + curriculum resets to logged
states (Urban Driver / Nachkov / MGAIL); (iv) **gate on closed-loop drivability in AlpaSim, never on
open-loop ADE**, and **pre-register that open-loop ADE may regress** (Nachkov saw minADE worsen even as
drivability improved — for us that trade is *desirable*, since our failure is drivability, not
displacement).

**Cheapest pre-registered discriminating experiment** (per operating-standard rule 5, both outcomes
banked in advance): `HYPOTHESIS`
- **Arm A (test-time only, ~1 day):** add a learned terminal **value** and do MPPI/CEM over our existing
  imagination at inference — *no planner retraining*. Isolates "does horizon-decoupling via a value fix
  closed-loop drift?"
- **Arm B (the real bet):** train the planner by **analytic backprop through a short imagined rollout**
  with the §6 cost + pessimism, BC-anchored.
- **Pre-registered reads:** primary = **closed-loop drivability in AlpaSim** (off-road/collision rate),
  secondary = open-loop ADE (expected flat-or-worse, and that is acceptable).
  - If **A alone** closes most of the open→closed gap → the deficit was *long-horizon value*, and we may
    not need risky closed-loop backprop at all (bank: "value, not closed-loop training, was the lever").
  - If **B beats A on drivability** → closed-loop analytic training is justified (bank: "differentiable
    closed-loop imitation is the lever; pay the WM-exploitation tax with pessimism").
  - If **B underperforms A / the open-loop baseline** → WM exploitation dominates at our data scale
    (bank: "learned-WM closed-loop training is premature; fix WM fidelity / grow data first").

---

## Sources

- Sutton. *Dyna, an Integrated Architecture for Learning, Planning, and Reacting.* ML 1990 / ACM SIGART 1991.
- Heess, Wayne, Silver, Lillicrap, Tassa, Erez. *Learning Continuous Control Policies by Stochastic Value Gradients (SVG).* NeurIPS 2015, arXiv:1510.09142. https://arxiv.org/abs/1510.09142
- Janner, Fu, Zhang, Levine. *When to Trust Your Model: Model-Based Policy Optimization (MBPO).* NeurIPS 2019, arXiv:1906.08253. https://arxiv.org/abs/1906.08253
- Hafner, Lillicrap, Ba, Norouzi. *Dream to Control (DreamerV1).* ICLR 2020, arXiv:1912.01603.
- Lambert, Amos, Yadan, Calandra. *Objective Mismatch in Model-based RL.* L4DC 2020, arXiv:2002.04523. https://arxiv.org/abs/2002.04523
- Yu et al. *MOPO: Model-based Offline Policy Optimization.* NeurIPS 2020, arXiv:2005.13239.
- Kidambi, Rajeswaran, Netrapalli, Joachims. *MOReL: Model-Based Offline RL.* NeurIPS 2020, arXiv:2005.05951.
- Hafner, Lillicrap, Norouzi, Ba. *Mastering Atari with Discrete World Models (DreamerV2).* ICLR 2021, arXiv:2010.02193. https://arxiv.org/abs/2010.02193
- Amos, Stanton, Yarats, Wilson. *On the Model-Based Stochastic Value Gradient for Continuous RL.* L4DC 2021, arXiv:2008.12775.
- Metz, Freeman, Schoenholz, Kachman. *Gradients are Not All You Need.* 2021, arXiv:2111.05803. https://arxiv.org/abs/2111.05803
- Yu et al. *COMBO: Conservative Offline Model-Based Policy Optimization.* NeurIPS 2021, arXiv:2102.08363.
- Scheel, Bergamini, Wolczyk, Osiński, Ondruska. *Urban Driver: Learning to Drive from Real-world Demonstrations Using Policy Gradients.* CoRL 2021, arXiv:2109.13333. https://arxiv.org/abs/2109.13333
- Xu, Makoviychuk, Narang, Ramos, Matusik, Garg, Macklin. *Accelerated Policy Learning with Parallel Differentiable Simulation (SHAC).* ICLR 2022, arXiv:2204.07137. https://arxiv.org/abs/2204.07137
- Hansen, Wang, Su. *Temporal Difference Learning for Model Predictive Control (TD-MPC).* ICML 2022, arXiv:2203.04955. https://arxiv.org/abs/2203.04955
- Hu, Corrado, Griffiths, Murez, Gurau, Yeo, Kendall, Cipolla, Shotton (Wayve). *Model-Based Imitation Learning for Urban Driving (MILE).* NeurIPS 2022, arXiv:2210.07729. https://arxiv.org/abs/2210.07729
- Wu, Escontrela, Hafner, Goldberg, Abbeel. *DayDreamer: World Models for Physical Robot Learning.* CoRL 2022, arXiv:2206.14176. https://arxiv.org/abs/2206.14176
- Bronstein et al. (Waymo). *Hierarchical Model-Based Imitation Learning for Planning in Autonomous Driving (MGAIL).* 2022, arXiv:2210.09539. https://arxiv.org/abs/2210.09539
- Pan et al. *Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models.* NeurIPS 2022, arXiv:2205.13817 (++ arXiv:2303.14889).
- Hafner, Pasukonis, Ba, Lillicrap. *Mastering Diverse Domains through World Models (DreamerV3).* arXiv:2301.04104, 2023 (later *Nature*, 2025). https://arxiv.org/abs/2301.04104
- Hansen, Su, Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR 2024, arXiv:2310.16828. https://arxiv.org/abs/2310.16828
- Li et al. *Think2Drive: Efficient RL by Thinking in Latent World Model … (CARLA-v2).* ECCV 2024, arXiv:2402.16720. https://arxiv.org/abs/2402.16720
- Gao et al. *CarDreamer: Open-Source Learning Platform for World Model based Autonomous Driving.* 2024, arXiv:2405.09111.
- Nachkov, Paudel, Van Gool. *Autonomous Vehicle Controllers From End-to-End Differentiable Simulation.* 2024, arXiv:2409.07965. https://arxiv.org/abs/2409.07965
- Nachkov et al. *Model-Based Vehicle Control Using Analytic World Models ("Dream to Drive").* 2025, arXiv:2502.10012. https://arxiv.org/abs/2502.10012
- *Imperfect World Models are Exploitable.* arXiv:2605.15960, 2026 preprint (recent; not independently verified here).
- *DreamerAD: Efficient RL via Latent World Model for Autonomous Driving.* arXiv:2603.24587, 2026 preprint (recent; not independently verified here).

**Provenance note:** all external claims sourced via web search/fetch 2026-07-23. arXiv ids for the
canonical pre-2023 papers (SVG, MBPO, MOPO/MOReL/COMBO, SHAC, Dreamer v1/v2) are cited from established
record with venue+year as the primary anchor; the driving-specific and 2024+ ids were confirmed against
fetched arXiv pages this session. Two 2026 ids (2603.24587, 2605.15960) are recent preprints surfaced in
search and are flagged as unverified.

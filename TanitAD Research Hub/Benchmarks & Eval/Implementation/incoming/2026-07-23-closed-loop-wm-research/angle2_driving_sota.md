# Angle 2 of 3 — Closed-loop training for DRIVING: the driving state of the art (≈2022–2026)

**Compiled:** 2026-07-23 (Europe/Berlin, UTC+2) · **Method:** web research only (no pod use, no fan-out).
**Evidence-class legend:** `PUBLISHED` = external paper, cited (year + arXiv/venue). `HYPOTHESIS` = my inference for TanitAD.
Every quantitative result below is the *source paper's own* measurement (i.e. PUBLISHED); it is not re-verified by us.

**Our context (recap):** sub-300M latent driving WM (ViT encoder + action-conditioned imagination + anchored-diffusion
planner), open-loop-imitation-trained today, warm-started from supervised v1. Data = **LOGGED** driving
(comma2k19 + PhysicalAI-AV). No fast live sim; our NuRec closed-loop is slow + ≈3× OOD. Measured failure:
the open-loop planner **departs the road under compounding error, mostly off-highway.**

**One-line answer to the brief:** across every driving family surveyed, the lever that moves *closed-loop*
metrics is **training on executed rollouts (RL, closed-loop IL, or offline-RL), not open-loop imitation** —
and for a **logs-only, no-live-sim** shop like us the method with the strongest published closed-loop
track record is **offline RL with conservatism (CQL/IQL)**, *not* world-model-imagined RL, which as of
mid-2026 has essentially **no published driving track record** (its wins are in robotics).

---

## 1. CARLA RL / model-based leaders

| Method | How the policy is trained | Sensor regime | Closed-loop metric it moved (source's numbers) |
|---|---|---|---|
| **Roach** (Zhang et al., **ICCV 2021**, arXiv 2108.08265) | **Model-free RL coach** (PPO-family) trained with **privileged BEV** → emits dense **on-policy + off-policy** supervision → **camera IL student distilled via DAgger** | Coach: privileged BEV. Student: monocular camera | Coach: **85 DS** CARLA leaderboard, **95%** NoCrash-dense. Camera student (RL-coached): **78%** success on NoCrash-dense, generalizing to new town + weather |
| **MILE** (Hu et al., **NeurIPS 2022**, arXiv 2210.07729) | **Model-based imitation:** latent-dynamics **world model + policy learned jointly**, purely on **offline logs, no online interaction** | Camera (high-res video) | **+31% driving score** over prior SOTA when deployed in a *new town + new weather* (CARLA) |
| **Think2Drive** (Li et al., **ECCV 2024**, arXiv 2402.16720) | **Model-based RL, DreamerV3-style:** policy trained **entirely by imagined rollouts inside a learned latent world model** (RSSM, actor-critic in latent space) | **Privileged BEV semantic masks** (GT objects/map/lights) — planning only, "eliminates perception" | **100% route completion** + **89.0 weighted DS** on the 39 CARLA-**v2** hard scenarios (first reported success on v2); **3 days on 1×A6000**. Official test routes: 56.8 DS / 98.6% RC |
| **CaRL** (Jaeger et al., **CoRL 2025**, arXiv 2504.17838) | **Model-free RL (PPO)** with a **single route-completion reward** (infractions terminate or multiplicatively discount). Scales to **300M samples (CARLA) / 500M–1B (nuPlan)** on one 8-GPU node | Privileged planner (BEV/vector state) | **64 DS / 82 RC** on CARLA **longest6 v2** (beats IL planner **PlanT 62** by +2, beats prior RL "by a large margin"). nuPlan below. |
| **TransFuser** (Chitta et al., **PAMI 2022**, arXiv 2205.15997) | **Open-loop IL (behavior cloning)**, transformer camera+LiDAR fusion | Camera + LiDAR | SOTA-at-submission CARLA driving score (closed-loop *evaluated*, BC *trained*) |
| **InterFuser** (Shao et al., **CoRL 2022**, arXiv 2207.14024) | **Open-loop IL** with interpretable safety-aware fusion + a safety controller | Camera(+LiDAR) | **76.18 DS** on the public CARLA leaderboard (rank-1 at the time) |

**Reading of Section 1.**
- **Two of the four "leaders" that actually top CARLA are RL, not IL** (Roach's coach, CaRL), and a third
  (Think2Drive) is **model-based RL inside a learned WM**. The pure-IL sensor-fusion stacks (TransFuser,
  InterFuser) are **open-loop-trained** and, while strong, are consistently beaten in closed loop by the
  RL/coach approaches — `PUBLISHED` (the papers above) `HYPOTHESIS` (the cross-paper ranking is my synthesis).
- **The recurring recipe is "RL/expert in the loop → distill to a deployable student"** (Roach), or
  **"RL directly on a cheap privileged state"** (CaRL, Think2Drive). Every one of these leaders **needs an
  interactive simulator (CARLA) to generate the on-policy experience.** None is a logs-only method.
- Think2Drive is the **architecturally closest** to us (latent WM + imagination + actor-critic) but its WM
  consumes **privileged GT BEV**, not pixels, and it still needs CARLA to collect experience.
- *(Note, lower-confidence secondary source: emergentmind reports a recent "TransFuser v6" reaching ~95 DS on
  Bench2Drive and doubling prior Longest6-v2 numbers — flagged as recent/secondary, not a primary cite.)*

---

## 2. nuPlan / Waymo closed-loop planning — the field's uncomfortable finding

**PDM-Closed / PDM-Hybrid** (Dauner et al., **CoRL 2023**, arXiv 2306.07962 — *"Parting with Misconceptions
about Learning-based Vehicle Motion Planning"*, won the **nuPlan 2023 challenge**):
- **Rule-based** at its core: centerline from a lane-graph search + **IDM proposals scored by a
  simulation-based (forward-model) cost**, pick the best; **PDM-Hybrid** adds a light learned correction for
  long-horizon ego-forecasting. `PUBLISHED`
- Achieves **≈92 CLS-R** (closed-loop score, reactive) on Val14. `PUBLISHED`
- **The uncomfortable finding:** planning and ego-forecasting are **fundamentally misaligned**; **open-loop
  (imitation) accuracy does not predict closed-loop driving**, and a *simple rule-based planner beats all
  learned planners in closed loop.* `PUBLISHED`

**Do learned planners beat rule-based in *closed* loop? As of the solid record — no.** `PUBLISHED` `HYPOTHESIS`
- **CaRL** (best **learned** planner on nuPlan, arXiv 2504.17838): **91.3** (non-reactive) / **90.6** (reactive)
  CLS on Val14 — explicitly **below rule-based PDM-Closed (>92)**, and above prior learned methods
  (e.g. Diffusion Planner 89.6/82.7). So even a 1-billion-sample RL planner is **state-of-the-art *among
  learned methods* but still loses to the rule-based prior in closed loop.** `PUBLISHED`

**The 2025–2026 caveat (the benchmark itself is partly the culprit):** `PUBLISHED` (recent) `HYPOTHESIS`
- The original CLS-R uses **simplistic IDM reactive agents**, which several works argue **over-credit
  rule-based planners** and hide learned planners' interaction skill. With **more realistic learned reactive
  agents** (e.g. *nuPlan-R*, arXiv 2511.10403; "how learned reactive agents shift nuPlan," arXiv 2510.14677;
  *Mosaic* hybrid, arXiv 2604.13853) the **learned planners' relative standing improves**, and hybrids can
  lead. Net: the "rule-based wins closed-loop" verdict is **real on the canonical benchmark but partly an
  artifact of easy reactive agents.** *(These 2026 IDs are beyond my Jan-2026 cutoff; surfaced by search,
  not deeply verified — treat as directional.)*

**Reading of Section 2 for us.** The durable, benchmark-independent lesson is PDM's **propose-then-score-with-a-
forward-model** recipe: it is the *cheapest* closed-loop win in the whole nuPlan record and it is **exactly the
shape of our anchored-diffusion fan** (we already generate proposals). We are one component short of PDM —
**a forward-model scorer that ranks the fan** — and that component needs **no policy retraining**. `HYPOTHESIS`

---

## 3. Offline RL for driving — **the section that matters most for us (we have LOGS, not a live sim)**

**Foundations (the OOD-overestimation problem):** offline RL's core failure is **value overestimation from
distribution shift** — the learned policy queries **out-of-distribution actions** whose Q-values the critic
extrapolates too high, and the policy then chases those phantom values.
- **CQL** (Kumar et al., **NeurIPS 2020**, arXiv 2006.04779): explicitly **penalizes Q-values of OOD actions**
  (pessimism). `PUBLISHED`
- **IQL** (Kostrikov et al., **2021**, arXiv 2110.06169): **never queries OOD actions at all** (implicit,
  expectile value learning) — sidesteps both the BC constraint and the conservative penalty. `PUBLISHED`
- **Decision Transformer** (Chen et al., **NeurIPS 2021**, arXiv 2106.01345) and **Trajectory Transformer**
  (Janner et al., **NeurIPS 2021**, arXiv 2106.02039): recast offline RL as **return-conditioned sequence
  modeling / beam search over a learned trajectory model** — the ancestors of the transformer planners now
  used on nuPlan (PlanTF, PlanFormer, Diffusion Planner). `PUBLISHED`

**Does offline RL beat plain imitation on *closed-loop* driving? Yes — decisively, in the one head-to-head
study on logged driving:** `PUBLISHED`
- *"From Imitation to Optimization: A Comparative Study of Offline Learning for Autonomous Driving"*
  (2025, **arXiv 2508.07029**). Setup: **Waymo Open Motion Dataset v1.1 (~200k scenarios)** + the **Waymax**
  **log-replay closed-loop simulator**, 1,000 unseen val scenarios. Compares three BC variants (MLP, structured,
  **transformer BC-T = strongest BC**) vs **CQL**:

  | | BC-K | BC-S | **BC-T** (best BC) | **CQL** |
  |---|---|---|---|---|
  | **Success rate** | 5.2% | 11.5% | **17.3%** | **54.4%** |
  | **Collision rate** | 45.8% | 39.2% | **31.1%** | **4.1%** |
  | **Off-road rate** | 32.1% | 15.6% | 0.7% | **0.0%** |

  → **CQL delivers 3.1× the success rate and 7.6× lower collisions than the strongest BC**, from **the same
  offline logs, with no new interaction.** The authors' conclusion is precisely our failure mode:
  *low one-step BC error is not sufficient — a conservative value function is what lets the agent recover from
  minor errors and avoid OOD (off-road) states.* `PUBLISHED`

**Reading of Section 3 for us.** This is the **closest published analog to TanitAD's regime**: offline logs +
a replay-style closed-loop evaluator, and the failure it fixes (**off-road departure under compounding error**)
is **the exact failure we measured.** The mechanism — conservative value learning that suppresses OOD-action
overestimation — is the single best-evidenced logs-only lever in the driving literature. Caveat: it is **one
2025 study on Waymax** (a log-replay sim, benign reactive agents), so treat the *magnitude* as indicative, not
guaranteed to transfer to our NuRec/PhysicalAI closed loop. `PUBLISHED` (the study) `HYPOTHESIS` (transfer to us).

---

## 4. World-model-as-training-ENVIRONMENT (2024–2026) — do driving WMs actually train a policy by RL?

**Short answer: the well-established driving WMs do NOT.** They are used for **scenario/data generation,
action-conditioned prediction, reward *scoring*, and closed-loop *evaluation* — not as an RL environment that
optimizes a policy inside them.** `PUBLISHED` (per each paper below) `HYPOTHESIS` (the cross-cutting claim).

| WM | What it is | Is a policy trained (by RL) *inside* it? |
|---|---|---|
| **GAIA-1** (Wayve, 2023, arXiv 2309.17080) | Generative WM (video+text+action, discrete next-token) | **No.** Scenario/data generation + learned simulator; no policy optimized inside it. `PUBLISHED` |
| **GAIA-2** (Wayve, 2025, arXiv 2503.20523) | Multi-view **latent-diffusion** controllable generative WM | **No.** Controllable *simulation for evaluation + data*; reduces real-data reliance. Not a policy RL trainer. `PUBLISHED` |
| **Vista** (Gao et al., **NeurIPS 2024**, arXiv 2405.17398) | Generative video WM, action-conditioned | **No — reward *scoring* only.** Provides a **test-time reward** = exp(−avg conditional variance) over M denoising passes to **rank candidate actions without GT**. Authors state it does **not** optimize a policy. `PUBLISHED` |
| **DriveDreamer** (Wang et al., **ECCV 2024**) | Generative WM: video generation + action prediction | **No.** Two-stage video synthesis + action prediction; primarily data/scene generation. `PUBLISHED` |
| **DriveArena** (Yang et al., **ICCV 2025**, arXiv 2408.00415) | **Closed-loop generative simulation *platform*** (Traffic Manager + World Dreamer) | **No — closed-loop *evaluation testbed*.** Feeds generated images to an **externally-trained** agent and measures closed-loop performance; it is not an RL policy-training loop. `PUBLISHED` |
| **Think2Drive** (§1) | **Learned *latent-state* WM (DreamerV3)** | **Yes — but** on **privileged compact BEV state, not pixels**, and it **needs CARLA to gather experience.** The one driving case of "policy trained inside a learned WM," and it is *not* generative-video and *not* logs-only. `PUBLISHED` |

**Where WM-as-RL-environment IS being made to work: robotics/VLA, not driving (yet).** `PUBLISHED` (recent) `HYPOTHESIS`
- Search surfaced the *actual* "policy trained by RL inside a generative video WM" results in **robotic
  manipulation**: e.g. **WMPO** (arXiv 2511.09515), **World-VLA-Loop** (arXiv 2602.06508), **WoVR** (arXiv
  2602.13977) — all on LIBERO / real manipulators, and all reporting that the hard part is **rollout stability**
  (they add tricks like **keyframe-initialized short rollouts** and **world-model↔policy co-evolution** to bound
  compounding WM error). *(2026 IDs beyond my cutoff; directional only.)* The takeaway that IS solid: **using a
  generative WM as the RL environment for a driving policy has no established published track record**, and the
  domains that do it bolt on explicit machinery to stop the imagined rollouts from diverging.

---

## HYPOTHESIS — what the driving field's evidence says actually works in closed loop, for OUR no-live-sim regime

`HYPOTHESIS` throughout this section (grounded in the `PUBLISHED` results above; not re-verified on our stack).

**H2.1 — Open-loop imitation is the thing to stop doing; the universal closed-loop lever is training on
executed rollouts.** Every closed-loop leader trains on rollouts (Roach's on-policy DAgger, CaRL/Think2Drive RL,
CQL on Waymax), and two independent papers state open-loop accuracy is *misaligned* with closed-loop driving
(PDM's "misconceptions," and the offline study's "low one-step error, 31% collisions"). This is **exactly our
measured symptom** (open-loop planner departs road). Confidence: **high** — convergent across CARLA, nuPlan, and
Waymax.

**H2.2 — For a logs-only shop with no fast sim, the best-evidenced lever is OFFLINE RL with conservatism
(CQL/IQL) on our own logs — not world-model-imagined RL.** The single head-to-head on logged driving
(arXiv 2508.07029) shows **CQL 54.4% vs best-BC 17.3% success, 4.1% vs 31.1% collisions, 0% off-road**, from the
same logs with no new interaction, and the mechanism it fixes (recover from OOD/off-road states via a
conservative value function) *is* our failure mode. This should be the **first** closed-loop experiment we run.
Confidence: **medium-high** (one 2025 study, benign sim; mechanism is well-founded across offline-RL theory).

**H2.3 — Use our latent WM as a SCORER before we ever use it as an RL ENVIRONMENT.** The two cheapest,
best-precedented closed-loop wins both *score proposals with a forward model* rather than RL-training inside a
WM: **PDM** (propose IDM trajectories → simulation-cost scoring → pick best; still #1 on nuPlan CLS-R) and
**Vista** (WM scores candidate actions by predictive variance, no policy training). We already generate proposals
(the anchored-diffusion fan); adding a **forward-model / WM-variance scorer over the fan** is a **no-retraining**
lever that copies the best-evidenced recipe in the field. Confidence: **medium-high**.

**H2.4 — World-model-imagined RL (Think2Drive-style, using OUR WM as the environment) is the highest-ceiling but
worst-fit-today option, and its known failure mode is precisely our WM's weakness.** Think2Drive's 100%-CARLA-v2
result is on **privileged compact state with a live simulator to keep the WM in-distribution**; every driving WM
that generates *pixels* is used for data/eval/reward, not RL. Our WM is measured **≈3× OOD and slow** — the exact
regime where imagined-rollout RL diverges and the policy **reward-hacks WM artifacts.** The robotics groups now
making WM-as-RL-env work only do so with explicit rollout-bounding tricks (short keyframe-initialized rollouts,
WM↔policy co-evolution). **Recommendation: do not stake a GPU-week on WM-imagined RL for the flagship yet**;
if pursued, pre-register it *against* the CQL-from-logs baseline and bound rollout length hard. Confidence:
**medium** (strong on "unproven for driving + matches our failure mode"; the ceiling claim is genuinely high).

**H2.5 — Keep a rule/optimization prior in the loop.** Even a 1B-sample RL planner (CaRL) loses to rule-based
**PDM-Closed** in closed-loop nuPlan. A hybrid — learned proposals + a rule/forward-model scorer + a safety
fallback — is the top of the closed-loop record and is robust to our WM being imperfect. Confidence: **high** on
nuPlan; **medium** on transfer to our highway-heavy PhysicalAI regime.

### Cheapest discriminating experiment (pre-registerable, both outcomes committed in advance)
On our logged corpus, evaluated on the **NuRec closed loop** (slow but decision-grade) with the **episode-cluster
bootstrap** (`taniteval/ci.py`), compare four arms holding the encoder/warm-start fixed:
- **A — current open-loop BC planner** (baseline).
- **B — + CQL/IQL conservative value head re-ranking the anchored-diffusion fan** (H2.2; logs-only, no sim).
- **C — + WM/forward-model scoring of the fan** (H2.3; Vista/PDM-style, no policy retraining).
- **D — short-horizon WM-imagined-RL fine-tune, rollout hard-capped** (H2.4; the ambitious arm).

**Pre-registered outcomes:** if **B or C** closes a decisive fraction of our open-loop→closed-loop off-road gap,
the field's logs-only recipe transfers and **D is deprioritized**. If **B and C both fail** but **D** succeeds,
we have positive evidence that our WM is good enough to be an RL *environment* (a genuinely novel driving result)
— and if **D** diverges/reward-hacks, that is the pre-committed signal that our WM's ~3× OOD must be fixed
(NuRec fidelity) *before* any imagined-RL flagship bet. This settles the "offline-RL-from-logs vs
WM-imagined-RL" question with an experiment, not deference.

---

## Sources (PUBLISHED)
- Roach — Zhang et al., *End-to-End Urban Driving by Imitating a Reinforcement Learning Coach*, ICCV 2021 — arXiv:2108.08265
- MILE — Hu et al., *Model-Based Imitation Learning for Urban Driving*, NeurIPS 2022 — arXiv:2210.07729
- Think2Drive — Li et al., *Think2Drive: Efficient RL by Thinking with Latent World Model (CARLA-v2)*, ECCV 2024 — arXiv:2402.16720
- CaRL — Jaeger et al., *CaRL: Learning Scalable Planning Policies with Simple Rewards*, CoRL 2025 — arXiv:2504.17838
- TransFuser — Chitta et al., *TransFuser: Imitation with Transformer-Based Sensor Fusion*, IEEE PAMI 2022 — arXiv:2205.15997
- InterFuser — Shao et al., *Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer*, CoRL 2022 — arXiv:2207.14024
- PDM — Dauner et al., *Parting with Misconceptions about Learning-based Vehicle Motion Planning*, CoRL 2023 — arXiv:2306.07962
- nuPlan reactive-agent caveats — nuPlan-R (arXiv:2511.10403), "learned reactive agents shift nuPlan" (arXiv:2510.14677), Mosaic (arXiv:2604.13853) — recent, directional
- Offline-vs-imitation driving study — *From Imitation to Optimization*, 2025 — arXiv:2508.07029 (WOMD + Waymax; CQL vs BC)
- CQL — Kumar et al., NeurIPS 2020 — arXiv:2006.04779 · IQL — Kostrikov et al., 2021 — arXiv:2110.06169
- Decision Transformer — Chen et al., NeurIPS 2021 — arXiv:2106.01345 · Trajectory Transformer — Janner et al., NeurIPS 2021 — arXiv:2106.02039
- GAIA-1 — Wayve 2023 — arXiv:2309.17080 · GAIA-2 — Wayve 2025 — arXiv:2503.20523
- Vista — Gao et al., NeurIPS 2024 — arXiv:2405.17398 · DriveDreamer — Wang et al., ECCV 2024 · DriveArena — Yang et al., ICCV 2025 — arXiv:2408.00415
- WM-as-RL-env in robotics (adjacent domain, directional) — WMPO (arXiv:2511.09515), World-VLA-Loop (arXiv:2602.06508), WoVR (arXiv:2602.13977)

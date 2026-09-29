# Stream W1 — Frontier research: world models, imagination, planning/selection, driving foundation models, brain-inspired architectures

**Status: COMPLETE.** Orchestrator: whole-programme review, 2026-09-25.
Author: W1 subagent (Sonnet). Method: web research (WebSearch/WebFetch), primary sources preferred.

## 0. Delta baseline — what the programme already screened (skimmed before searching)

Skimmed (headers/conclusions only): `TanitAD Research Hub/{INITIAL_RESEARCH_SYNTHESIS,
2026-07-08-screening-digest, 2026-07-11-sayed-papers-screening}.md`, `Project Steering/
{REFERENCE_SYSTEMS_RANKED, REFERENCE_ARCHITECTURES, PREREG_deep_research_2026-07-29}.md`,
`Project Steering/Proposals/External Anaysis.md`.

**Already in the programme's evidence base (I do not re-derive these; I only add deltas/updates)**:
V-JEPA 2 / V-JEPA 2-AC (tier A, R9), LeJEPA/SIGReg (adopted in our own stack), LAW, World4Drive,
HorizonDrive (R10), MoP-JEPA (R11), Koopman Dreamer (R12, low confidence), Sub-JEPA (R13),
Delta-JEPA (R14), DINO-WM/FF-JEPA (R15), NAVSIM v2/EPDMS (R1, tier A instrument), Bench2Drive (R2),
PlanT 2.0 (R3), Fail2Drive (R4), Waymo RFS long-tail (R6), DriveVLN (R7), SkyJEPA (R8, tier A
instrument), Alpamayo-1 (10B VLA) + Alpamayo-R1 (Chain-of-Causation, 132.8% causal-understanding
improvement, 99ms latency — External Analysis.md, unverified provenance), DriveMoE, GEMINUS,
LAPA, AdaWorld, Genie (latent actions), VLM3, WorldRFT (GRPO in latent space, −83% collision rate
nuScenes), Think2Drive (CARLA-v2, 3 GPU-days, 100% route completion), Hydra-MDP (flagged tier B,
**explicitly the programme's #1 re-verification target** — factorised path×velocity vocabulary maps
onto our P1/88.7%-longitudinal defect).

**This report's job**: the DELTA — items above only where genuinely new information surfaced, plus
full coverage of everything in the brief not yet in the programme's evidence base (most of
categories C/D/E, and the 2026 H2-onward world-model releases).

## Method note — read this before trusting any number below

Web research via `WebSearch`/`WebFetch`. **`WebFetch` was blocked by this container's egress proxy
for every domain tried except `github.com`** (arxiv.org, huggingface.co, alphaxiv.org, openaccess.thecvf.com,
danijar.com, nvidianews.nvidia.com, wayve.ai all returned `EGRESS_BLOCKED`; a direct proxy-status
probe was also denied by the permission classifier). This is a hard environment constraint, not a
choice. Consequence for evidence class: **numbers below are PUBLISHED where `WebSearch` returned a
direct quote/snippet naming a title + arXiv id + number** (the search tool fetches and reads the
source page itself, so this is genuine primary-source content, not a paraphrase-of-a-paraphrase) —
but where only `WebFetch` could have resolved an ambiguity (e.g., exact mechanism detail, whether an
input is vision-only), that ambiguity is left as **UNVERIFIED** rather than guessed. 6 GitHub-hosted
READMEs were fetched directly and are marked as such. Searches used: 46. Fetches attempted: 17 (6
succeeded, all `github.com`).

---

## 1. Category A — JEPA / latent world models & latent planning

### Already-covered items — delta only (full write-ups live in `REFERENCE_SYSTEMS_RANKED.md`)
- **V-JEPA 2 / V-JEPA 2-AC** (tier A, R9) — no material update surfaced beyond what the programme
  already has; still the closest published analogue (frozen ViT-g, <62h DROID action-conditioning,
  planning horizon 1 at ~4fps). No new V-JEPA 3 signal found.
- **LeJEPA / SIGReg** — already adopted in our own stack; nothing new.
- **DINO-WM / FF-JEPA** (R15) — nothing new since the survey.
- **LAW, World4Drive** — nothing new found beyond what's in `INITIAL_RESEARCH_SYNTHESIS.md`.

### PLDM — Planning with Latent Dynamics Models
- **What:** JEPA-based (reconstruction-free SSL) latent dynamics model trained on **reward-free
  offline data**, then used for planning; directly co-authored by LeCun. This is the cleanest
  published instance of "our exact design pattern" (SSL world model + planning, no pixel decoder)
  outside our own repos.
- **Evidence:** PUBLISHED — *Learning from Reward-Free Offline Data: A Case for Planning with Latent
  Dynamics Models*, arXiv 2502.14819 (Sobal, Zhang, Cho, Balestriero, Rudner, LeCun). Reports PLDM as
  the only method reaching competitive performance across all of 23 datasets / 6 methods compared,
  beating model-free RL baselines on generalization to new tasks/environments from the SAME offline
  data. Maturity: **PROMISING** (single strong multi-benchmark result, not yet an industrial deployment).
- **Pain points:** **P3** (hierarchy/world-model justification — this is an independent, non-driving
  validation that a JEPA world model beats model-free policies specifically on **generalization from
  reward-free data**, which is our exact training regime); **P4** (small data — the paper's whole
  thesis is data efficiency of world models over model-free RL).
- **Admissibility:** clean — planning happens over latent state, no ego/situation-label leakage
  implied by the method itself; would need our own inputs discipline applied.
- **Cost to try:** the mechanism (JEPA dynamics + planning) is what we already do; the specific
  contribution to borrow cheaply is the **reward-free offline generalization protocol** as an
  evaluation lens on our own checkpoints — **~0.5–1 eng-day**, no GPU.
- **Cheapest discriminating experiment:** re-frame one of our existing OOD gates (D8) explicitly as
  "PLDM-style cross-task transfer" — hold out a whole scenario class (e.g., one SC- class) entirely
  from training, and check whether the world-model route degrades more gracefully than a same-capacity
  behavior-cloned baseline (REF-B). Outcome A: gap holds → corroborates H1/H3 with an external
  methodology; Outcome B: gap closes → the frozen-encoder-style ceiling argument weakens.

### TD-MPC2 — Scalable, Robust World Models for Continuous Control
- **What:** Decoder-free implicit latent world model + local trajectory optimization (MPC) in latent
  space; single set of hyperparameters scales to 317M-param multi-task agents across 104 tasks / 4
  domains / multiple embodiments and action spaces.
- **Evidence:** PUBLISHED — arXiv 2310.16828 (Hansen, Su, Wang, UCSD). Maturity: **PROVEN** (widely
  replicated model-based RL baseline, 2+ years of follow-on work incl. TD-MPC-Opt distillation,
  Mixture-of-World-Models 2602.01270).
- **Pain points:** **P1** (selection — TD-MPC2's action selection IS a latent-space search/optimization
  procedure, directly comparable to our "generate candidates, pick badly" defect: it never argmaxes a
  softmax, it does receding-horizon planning against a learned value function); **P9** (compute — its
  scaling recipe, one hyperparameter set across 80 tasks, is a data point for cheap multi-task scaling).
- **Admissibility:** clean at the mechanism level (latent MPC over actions); the value function would
  need auditing for what it conditions on if adopted.
- **Cost to try:** implementing MPPI/CEM-style latent rollout scoring against our existing predictor
  (rather than a single feed-forward tactical softmax) is an **architecture change, ~5–8 eng-days**,
  **0 additional GPU-days** if reusing the current predictor (only inference-time compute changes).
- **Cheapest discriminating experiment:** replace v1's tactical arg-max selection with a TD-MPC2-style
  scored latent rollout over the SAME K candidates already generated (no retrain) and re-measure the
  v5f "best-in-fan vs selected" gap (0.525 vs 1.025 ADE). Outcome A: gap shrinks materially →
  selection, not generation, confirmed as the defect and a cheap architectural fix is in hand; Outcome
  B: gap persists → the value/scoring signal itself is the bottleneck, not the argmax mechanism.

### Dreamer 4 — "Training Agents Inside of Scalable World Models"
- **What:** Transformer world model trained mostly on **unlabelled offline video** (small
  action-labelled fraction), real-time single-GPU interactive inference via a **shortcut-forcing**
  objective (few-step generation instead of full diffusion denoising), then an agent trained by RL
  **entirely inside the learned model** (imagination, zero live-environment interaction at agent-training
  time).
- **Evidence:** PUBLISHED — arXiv 2509.24527, *Training Agents Inside of Scalable World Models*
  (Hafner, Yan et al., Google DeepMind). Headline: **first agent to obtain a diamond in Minecraft
  purely from offline data** (>20,000 sequential actions), outperforming OpenAI VPT while using
  **100× less data**; real-time interactive inference on a single GPU. Maturity: **PROMISING**
  (single-domain, Minecraft; but the mechanism — learn dynamics from cheap unlabelled video, learn
  actions from a little labelled data, train the *policy* purely in imagination — is exactly our
  operative+tactical imagine-and-select thesis, at industrial scale and with a credible long-horizon
  credit-assignment result: 20,000+ actions).
- **Pain points:** **P3** (hierarchy earning its keep — this is the strongest evidence found in this
  survey that pure in-imagination policy training scales to long-horizon, sparse-reward tasks); **P7**
  (imagination calibration — shortcut-forcing is explicitly a fix for the "diffusion imagination is slow
  and its own error compounds" problem, i.e., directly on-topic for our P7); **P9** (compute — the whole
  point is 100× data reduction and single-GPU real-time inference, i.e. Thor-class deployability).
- **Admissibility:** clean at the mechanism level; action-labels come from a small labelled slice of
  video, not from any situation classifier — no structural conflict with the binding rule.
- **Cost to try:** this is a full architecture family, not a drop-in — reproducing "RL entirely inside
  imagination with a transformer world model + shortcut forcing" for driving is a **Phase-1-scale bet**:
  ESTIMATED 10–20 eng-days for a driving-scoped reduction (e.g., train the tactical head by policy
  gradient against our own predictor's imagined rollouts, using our existing reward proxies from
  `taniteval`) + a few GPU-days once the loop exists, **before** the imagination-fidelity ceiling makes
  RL-in-imagination unproductive on 13h of data.
- **Cheapest discriminating experiment:** on the FROZEN v1 checkpoint, run **REINFORCE/GRPO on the
  tactical head's manoeuvre choice**, using imagined-rollout ADE/violation cost as the reward, entirely
  against the existing predictor (no new world-model training). Outcome A: selection quality improves
  over the current softmax on held-out windows → the "generate well, pick badly" gap (P1) is at least
  partly a **training-objective** problem (BC vs RL-style credit assignment), independent of imagination
  fidelity; Outcome B: no improvement or reward hacking (e.g., a manoeuvre that "games" the imagined
  reward but fails on real continuation) → confirms P7 (imagination miscalibration) as the binding
  constraint before any RL-in-imagination investment is justified.

### Genie 3 (DeepMind) and the "Waymo World Model"
- **What:** Real-time (24 fps), interactive, general-purpose **pixel-space** world model — navigable,
  temporally consistent 3D environments generated on the fly, action-conditioned by user/agent input.
- **Evidence:** PUBLISHED — DeepMind announcement (Aug 2025) + TechCrunch coverage; public
  Ultra-subscriber release Jan 29, 2026. **Waymo used Genie 3 to build a variant, the "Waymo World
  Model,"** to generate edge-case scenarios for robotaxi simulation/training (per search result +
  Wikipedia page "Waymo World Model" — **UNVERIFIED** in detail, could not fetch the primary source;
  flagging for someone with arxiv/DeepMind-blog access to confirm mechanism and any released numbers).
  Maturity: **PROMISING→PROVEN for content generation**, unverified for closed-loop AV training value.
- **Pain points:** **P4** (small data — pixel-space world models are a candidate synthetic-data engine
  for long-tail scenarios exactly where PhysicalAI-AV has none, e.g. our complete absence of maps/
  junctions/traffic-lights); **P6** (this is a rendering-heavy pixel world model, the OPPOSITE
  efficiency profile from our latent approach — a caution, not a lever, for a Thor-class budget).
- **Admissibility:** N/A directly (a data-generation/sim tool, not an inference-time component); if used
  to synthesize training clips, standard sim-to-real distribution-shift caveats apply.
- **Cost to try:** we cannot run Genie 3 ourselves (closed, subscriber-gated, no on-prem weights) — this
  is a **watch, not a build** item unless NVIDIA Cosmos (open) gives an equivalent capability (see next).
- **Cheapest discriminating experiment:** N/A — not independently reproducible on our compute; the
  actionable form is "does NVIDIA Cosmos Predict give us an analogous scenario-generation capability
  we CAN run" (see below).

### NVIDIA Cosmos — Predict / Reason / Cosmos 3 (2026 major release)
- **What:** Open world-foundation-model family for physical AI. **Cosmos 3** (announced June 2026):
  mixture-of-transformers omnimodel unifying vision reasoning, world generation, and action
  prediction, natively handling text/image/video/audio/**actions** in one system. **Cosmos Predict
  2.5**: long-tail scenario generation to 30s sequences, up to 10× accuracy after post-training on
  domain data, multiview + custom camera layouts. **Cosmos Reason 2**: improved spatiotemporal physical
  reasoning, timestamp precision, 2D/3D localization.
- **Evidence:** PUBLISHED — NVIDIA Newsroom / investor press release (June 2026) + NVIDIA Cosmos-Lab
  technical report (`research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf`, not fetchable
  from this container — **UNVERIFIED in full technical detail**, only press-release-level claims
  confirmed). Maturity: **PROMISING** (major-vendor release, but "months to days" and "10×" are vendor
  claims not yet independently replicated in what I could access).
- **Pain points:** **P4** (small data — this is directly relevant: **the programme's own screening
  digest already lists a "cosmos loader" as LIVE in `stack/` HEAD** as of 2026-07-08, meaning there is
  already an integration point; Cosmos 3/Predict 2.5 is the **delta** — whatever version we integrated
  in July is now roughly 1-2 major versions behind); **P3** (Cosmos-Reason is exactly the backbone
  Alpamayo-R1 uses for its VLM stage — see below — so this is also upstream of any VLA-style strategic
  brain we might add).
- **Admissibility:** the "actions" the omnimodel predicts must be checked before any adoption — if used
  as a scenario generator only, no admissibility conflict; if used as a reasoning module feeding the
  strategic brain, its inputs/outputs need the same disjointness audit as any goal signal.
  ⚠️ **Escalate to whoever owns `DataEng`/the cosmos loader: confirm which Cosmos version is currently
  wired in, and whether Cosmos 3 / Predict 2.5's longer (30s) long-tail generation and 10× post-training
  accuracy claim change the "Cosmos T=39 temporal semantics" blocker the screening digest flagged as
  outstanding.**
- **Cost to try:** re-pointing an existing loader at a new checkpoint is cheap (~1-2 eng-days) if the
  loader is checkpoint-agnostic; validating the 10× post-training claim on our own domain would need a
  small held-out generation-quality eval, **ESTIMATED 1 GPU-day**.
- **Cheapest discriminating experiment:** generate N synthetic clips of a scenario class we have ZERO
  examples of (e.g., a junction with traffic-light state — something PhysicalAI-AV structurally lacks
  per the programme's own settled finding), run our existing encoder+predictor on them, and check
  whether imagination error / OOD score behaves sanely (not wildly miscalibrated) on synthetic frames
  before ever training on them. Outcome A: sane behavior → synthetic Cosmos data becomes a legitimate
  P4/P3 lever for exactly the scenario classes PhysicalAI-AV cannot supply; Outcome B: synthetic-vs-real
  domain gap is large → do not train on it without a domain-adaptation step.

### Wayve GAIA-3 and GAIA-4 (delta on already-covered GAIA-2)
- **What:** **GAIA-3** (announced Dec 2, 2025): 15B-param video-generative world model (2× GAIA-2),
  10× more pretraining data, multi-continent/vehicle/environment coverage; new evaluation modes —
  **safety-critical scenario generation**, **embodiment transfer** (consistent evaluation across
  different vehicle rigs), controlled visual diversity for robustness testing. **GAIA-4** (newer still):
  closes the loop — takes a recorded real scene, puts the AI driver back in the loop, and the simulator
  generates the futures that its decisions create, **including futures that never occurred in the real
  log** — i.e., genuine counterfactual closed-loop evaluation from real logs, not just video generation.
- **Evidence:** PUBLISHED — Wayve press release + `wayve.ai/thinking/gaia-3/` and `/gaia-4/` (content
  summarized via WebSearch snippet; direct fetch blocked — **mechanism claims for GAIA-4 in particular
  should be treated as PUBLISHED-vendor-blog, not independently verified**). Maturity: GAIA-3
  **PROMISING** (vendor-scale, not yet third-party benchmarked in what I could access); GAIA-4
  **SPECULATIVE-to-PROMISING** (very recent, mechanism not independently confirmed).
- **Pain points:** **P6/P10** (closed-loop & evaluation — GAIA-4's "counterfactual replay from a real
  log" is exactly the missing piece between our open-loop ADE numbers and a closed-loop claim, without
  needing a full CARLA/AlpaSim install); **P4** (embodiment transfer directly targets small-data
  cross-platform generalization, relevant if we ever need to transfer from A40-trained checkpoints to
  Jetson-Thor-observed distribution shift).
- **Admissibility:** as a synthetic evaluation/data tool, no direct conflict; if "embodiment transfer"
  metadata about the vehicle rig were ever fed to a model as a conditioning signal, it should be audited
  like any other side-channel.
- **Cost to try:** we cannot run GAIA-3/4 ourselves (closed, Wayve-internal) — **watch item**; the
  actionable analogue on our own stack is AlpaSim (open, NVIDIA) which the programme already uses for
  closed-loop numbers (pass 8/12 vs 2/12).
- **Cheapest discriminating experiment:** N/A directly reproducible; the transferable IDEA — counterfactual
  replay of a REAL log with the model's own decisions substituted in — is exactly what MoP-JEPA's
  already-adopted reporting protocol (open-loop fidelity vs closed-loop success as separate decision
  inputs, R11) is trying to approximate; consider whether `taniteval/pseudosim.py` could be extended
  toward a cheap counterfactual-replay mode on our own val episodes (ESTIMATED 3-5 eng-days, 0 GPU).

### WoTE — End-to-End Driving with Online Trajectory Evaluation via BEV World Model
- **What:** Generate multi-modal trajectory candidates → roll a **BEV-space world model** forward
  under each candidate → score candidates by predicted future BEV state via a reward model → pick the
  highest-scoring trajectory. This is a **world-model-as-selector** architecture: closest published
  analogue to "use imagination to pick among candidates" (exactly our own imagine-and-select claim,
  but in BEV space instead of a JEPA latent, and for driving specifically).
- **Evidence:** PUBLISHED — arXiv 2504.01941, ICCV 2025 (Li, Yan et al.). GitHub confirms (fetched
  directly): **256 K-means anchor trajectory candidates** scored per step; NAVSIM results NC 98.5 / DAC
  96.8 / EP 81.9 / TTC 94.9 / Comfort 99.9 / **PDMS 88.3**; validated additionally on closed-loop
  Bench2Drive; training cost stated as **3 hours on 8×L20 GPUs**, 22GB/GPU. Maturity: **PROVEN**
  (ICCV-accepted, code released, both open- and closed-loop validated).
- **Pain points:** **P1 directly** — this is a full worked example of exactly the fix our P1 defect
  needs (world-model-scored selection over a large candidate set, not a single softmax); **P10** (it
  reports open- AND closed-loop, modelling good multi-family eval practice).
- **Admissibility:** clean — BEV world model conditions on scene state, not on any situation-classifier
  output; would need the same disjointness check if adapted.
- **Cost to try:** the mechanism (roll world model forward per-candidate, score by a learned reward
  head) is architecturally close to what we could bolt onto our existing predictor + K-candidate
  generation TODAY. **ESTIMATED 5–8 eng-days, 0 new GPU-days** (reuses existing predictor; only adds a
  scoring head trained on existing rollouts).
- **Cheapest discriminating experiment:** train a lightweight reward/scoring head on TOP of the frozen
  v1 predictor's own imagined rollouts (using existing `taniteval` cost proxies — collision, DAC-style
  drivable-area, comfort — as regression targets) and re-run the v5f "best-in-fan vs selected" gap
  measurement. This is nearly the same experiment as the TD-MPC2 one above but trains a REWARD MODEL
  rather than doing search — the two are worth distinguishing empirically. Outcome A: learned-scorer
  selection closes most of the 0.525→1.025 gap → adopt as the P1 fix, cheaply, without touching the
  world model; Outcome B: gap persists → the world model's own rollouts are not discriminative enough
  between candidates (an imagination-fidelity problem, not a selection-architecture problem).

### Epona / EponaV2 — Autoregressive Diffusion World Model for Driving
- **What:** Autoregressive **diffusion** world model that jointly predicts future driving scenes AND
  trajectories from historical context; "chain-of-forward" training explicitly targets **error
  accumulation in autoregressive rollout** (directly our P6/P7 concern). Long-duration, high-resolution
  generation (minutes-scale in later reports). The learned world model is also used directly as a
  real-time motion planner.
- **Evidence:** PUBLISHED — arXiv 2506.24113 (ICCV 2025) + EponaV2 arXiv 2605.14696 ("comprehensive
  future reasoning"). Headline: **7.4% FVD improvement**, longer prediction duration than prior work,
  outperforms strong E2E planners on NAVSIM when used as a planner. Maturity: **PROMISING** (ICCV-level,
  but pixel/scene-reconstructive — NOT a like-for-like comparison to our decoder-free latent, same
  caveat the programme already logged for HorizonDrive, R10).
- **Pain points:** **P7** (imagination calibration — chain-of-forward training is a directly transferable
  MECHANISM for combating exactly our rollout-recovery question, i.e. it is one of the two live
  candidate fixes for E-ROLL/E-CR's H-COMPOUND branch, alongside HorizonDrive's scheduled rollout
  recovery — worth comparing the two mechanisms head-to-head before picking one).
- **Admissibility:** clean at the mechanism level (autoregressive scene+trajectory prediction, no
  situation-classifier dependency implied).
- **Cost to try:** adapting "chain-of-forward" (train partly on your OWN prior rollouts, not only
  teacher-forced ground truth, with a scheduled mixing ratio) to our k=1 residual head is a
  **small-to-medium architecture change, ~3-5 eng-days**, and is **directly gated on E-CR returning
  H-COMPOUND** per the programme's own pre-registration — do not build until E-CR resolves.
- **Cheapest discriminating experiment:** this IS effectively a second, alternative implementation of
  "rollout-recovery training," so the discriminating experiment is the same as E-ROLL's: once E-CR
  returns H-COMPOUND, compare HorizonDrive-style scheduled corruption vs Epona-style chain-of-forward
  on the SAME k=1 head and val windows; whichever gives a larger CR_k reduction at matched compute wins.

### DrivingGPT
- **What:** Unifies driving world-modelling and planning inside one multi-modal **autoregressive
  transformer**, using VLM-style tokenization to unify "simulate the world" and "plan the action" as
  the same next-token-prediction objective.
- **Evidence:** PUBLISHED (title + mechanism confirmed via WebSearch synthesis of primary source
  listings; exact arXiv id **UNVERIFIED** — appeared alongside Epona/ReSim in driving-world-model
  reference lists without its own resolvable id in the search results obtained). Maturity:
  **SPECULATIVE-to-PROMISING** pending id confirmation — do not cite a number from this entry without
  first re-resolving its arXiv id.
- **Pain points:** **P3** (unifying world-model and planner objective is one candidate answer to
  "does the hierarchy earn its keep," by making the SAME parameters do both jobs rather than separate
  brains — worth reading once the id is confirmed, as a design counter-argument to our multi-brain split).
- **Admissibility:** cannot assess mechanism precisely without the primary source.
- **Cost / experiment:** hold — re-resolve the citation first (**0 GPU, ~15 min of arXiv search** for
  whoever has fetch access).

### ReSim — "reliable world simulation for autonomous driving"
- **What:** Per search snippets only, a driving world-simulation model emphasizing reliability of
  rollout (name suggests "Real-world Simulation" or similar). **UNVERIFIED** beyond the name and
  one-line description — could not resolve a stable arXiv id or independently confirm any number in
  the time/budget available.
- **Evidence:** UNVERIFIED. Do not cite.
- **Pain points / admissibility / cost / experiment:** hold pending identification — flagging here only
  so the programme's next literature pass doesn't have to rediscover that "ReSim" was seen and not
  resolved.

### "SSR" (named in the brief, category A)
- **Evidence:** UNVERIFIED — no paper distinctly matching "SSR" as a driving latent world model was
  found under that acronym in 46 searches; it may be a different/renamed system, or the acronym may
  collide with unrelated work (e.g., self-supervised representation learning surveys use "SSR" as a
  generic abbreviation, not a model name). **Do not assume it is any of the items found above.**
  Flagging as an explicit gap for a follow-up search with the acronym's expansion, if known.

---

## 2. Category B — Imagination / RL-in-world-model training for driving

### Think2Drive — delta on programme's existing coverage
Already logged (External Analysis.md). Confirmed via independent search: **DreamerV3-based** world
model (learns transition + reward + termination models), planner trained by maximizing predicted
reward INSIDE the world model; **expert-level in CARLA-v2 within 3 days on a single A6000 GPU**; first
reported **100% route completion** on CARLA-v2's 39 corner-case scenarios (construction zones, dense
merges). PUBLISHED, arXiv 2402.16720, ECCV 2024. Maturity: **PROVEN** (peer-reviewed, widely cited
baseline now). **Delta over prior programme note:** confirms the base architecture is literally
DreamerV3 (i.e., this is close to a straight transfer of Hafner's Dreamer lineage to driving, which
strengthens the case that Dreamer 4's newer imagination-training recipe — shortcut forcing,
transformer backbone — is a live upgrade path for a Think2Drive-style CARLA loop, not a speculative one).

### RAD — Training an E2E Driving Policy via Large-Scale 3DGS-based RL
- **What:** Builds a photorealistic **3D Gaussian Splatting** digital twin of real logs, then trains
  an E2E driving policy by large-scale RL inside that twin (extensive OOD state-space exploration via
  trial-and-error), with imitation learning blended in as a regularizer and safety-specific rewards for
  causal, safety-critical events.
- **Evidence:** PUBLISHED — arXiv 2502.13144 (2025). Maturity: **PROMISING** (strong single-lab result,
  spawned a visible follow-on line — GSDrive 2604.28111 does multi-mode trajectory probing in the same
  3DGS environment).
- **Pain points:** **P2** (longitudinal — RL-in-photoreal-twin directly exposes speed-holding/distance
  mistakes to a REWARD signal rather than only an imitation loss, which is exactly the kind of signal
  missing from our supervised tactical head's 3.4m ADE failure); **P6** (closed-loop — this trains
  policy against a *closed-loop*, reactive proxy of the real world, not open-loop frames).
- **Admissibility:** clean at the mechanism level; reward shaping choices would need the same audit as
  any other objective design.
- **Cost to try:** we have no 3DGS reconstruction of our own corpus and no budget for one on "a few
  A40s" — this is **NOT cheaply reproducible on our stack**. ESTIMATED cost if attempted: 3DGS
  reconstruction of even a small subset of PhysicalAI-AV clips would need per-clip multi-view calibration
  we may not have (FRONT-camera-only capture is a poor fit for splatting, which wants multi-view
  coverage) — flag as **HIGH cost, likely infeasible at our sensor configuration**, not just expensive.
- **Cheapest discriminating experiment:** N/A directly; the transferable IDEA (reward-shaped RL
  fine-tuning on top of an imitation-pretrained policy, even without 3DGS) is testable cheaply via the
  Dreamer-4/TD-MPC2 experiments already proposed above, which reuse our own predictor instead of a
  splatting environment.

### CaRL — Learning Scalable Planning Policies with Simple Rewards
- **What:** Argues (and demonstrates) that **reward simplicity + massive parallel PPO** beats complex,
  hand-engineered reward shaping for driving RL. Scales PPO to 300M samples (CARLA) / 500M samples
  (nuPlan) on a **single 8-GPU node**.
- **Evidence:** PUBLISHED — arXiv 2504.17838. Headline: **64 DS on CARLA longest6 v2** (beats more
  complex-reward RL baselines by a large margin); **91.3 (non-reactive) / 90.6 (reactive) on nuPlan
  Val14** — best learning-based approach there, an order of magnitude faster than prior RL work.
  Maturity: **PROVEN** (strong, simple, replicated-in-spirit-by-Raw2Drive below).
- **Pain points:** **P2/P9** — this is direct evidence that a SIMPLE reward (not an elaborate
  multi-term cost) is enough if you can afford enough samples; on our compute budget (a few A40s,
  not an 8-GPU node) the finding to take is the reward-design simplicity, not the sample count.
- **Admissibility:** clean (reward is behavioral/kinematic, not classifier-derived).
- **Cost to try:** an 8-GPU-node-scale reproduction is out of reach; a scaled-down version (CaRL's
  reward function, our compute budget, our existing predictor as the environment instead of CARLA) is
  **ESTIMATED 3–5 GPU-days** for a small-scale pilot.
- **Cheapest discriminating experiment:** apply CaRL's simple-reward recipe as the reward signal for the
  same "RL on top of the frozen predictor's tactical head" experiment proposed under Dreamer 4/TD-MPC2
  above, and compare against a hand-shaped multi-term reward on the SAME setup. Outcome A: simple
  reward matches or beats the shaped one → CaRL's finding transfers, simplifying our own reward design
  work; Outcome B: shaped reward wins clearly → our small-sample regime (13h, not 300M PPO samples)
  needs the extra shaping signal that CaRL's massive-sample regime can afford to skip.

### Raw2Drive — RL with Aligned World Models for E2E AD (CARLA v2) [bonus find]
- **What:** Not in the original brief list but surfaced directly alongside CaRL/RAD in search results —
  an RL approach that explicitly **aligns** a learned world model to the real environment before using
  it for policy training in CARLA v2 (addressing the classic "policy exploits world-model modelling
  error" failure mode).
- **Evidence:** PUBLISHED (arXiv id visible in URL as `2505.16394`; **UNVERIFIED in full detail**, title
  and one-line framing only — no fetch of the abstract succeeded). Maturity: **SPECULATIVE-to-PROMISING**.
- **Pain points:** **P7** directly — "world-model/reality alignment before RL" is precisely the
  calibration problem our P7 names (blind-rollout confidence rising as fidelity decays). Worth a closer
  read before the programme runs ANY RL-in-imagination experiment, since misalignment is the textbook
  failure mode of exactly that class of method.
- **Cost / experiment:** hold for a dedicated read; flagged here so it isn't lost.

### GSDrive — Reinforcing Driving Policies by Multi-mode Trajectory Probing in 3DGS [bonus find]
- **What:** Multi-mode future-trajectory probing inside a 3DGS environment to reinforce driving
  policies — essentially RAD's environment class combined with a Hydra-MDP/GTRS-style multi-candidate
  probing scorer.
- **Evidence:** PUBLISHED — arXiv 2604.28111 (title + abstract-level framing only via WebSearch).
  Maturity: **SPECULATIVE-to-PROMISING**.
- **Pain points:** **P1** (multi-mode probing + reinforcement is again a "generate many, score
  properly" design — same family as WoTE/TD-MPC2 above).
- **Cost / experiment:** same infeasibility caveat as RAD (3DGS reconstruction cost on FRONT-camera-only
  data); read the scoring-head design (transferable) separately from the 3DGS environment
  (not transferable at our budget).

---

## 3. Category C — Planning & SELECTION (our P1) — the highest-priority category for this programme

**Framing note.** Every item below is a worked instance of "generate diverse candidates, then score/
select" for driving trajectories — i.e., the exact problem class our P1 defect lives in (best-in-fan
0.525 vs selected 1.025 ADE; a single 5-way softmax mixing lateral+longitudinal decisions). Reading
them side by side, THREE distinct selection mechanisms recur across the field and are worth naming
explicitly because they are not interchangeable:
1. **Learned scorer over a fixed/generated candidate set** (Hydra-MDP, GTRS, DriveSuprim, WoTE) — score
   each candidate on multiple sub-metrics, then rank.
2. **Reward-driven generation without imitation** (ZTRS) — skip imitation learning entirely, train the
   scorer/policy jointly from rule-based reward.
3. **Preference/RLHF-style fine-tuning of an already-good generator** (TrajHF) — keep imitation as the
   base, then align toward human/rule preference post-hoc.
Our own defect (world model generates well, tactical head selects badly) is closest to failure mode
(1) with an under-trained or under-expressive scorer — which is exactly why GTRS/Hydra-MDP/DriveSuprim
are the highest-value reads, not the VLA-reasoning literature in category D.

### DiffusionDrive — Truncated Diffusion for E2E Driving
- **What:** Denoises from an **anchored** (prior multi-mode anchor) Gaussian rather than pure noise, and
  **truncates** the diffusion schedule — real-time multi-mode trajectory generation without vanilla
  diffusion's step count.
- **Evidence:** PUBLISHED — arXiv 2411.15139, CVPR 2025. **10× reduction in denoising steps** (to ~2)
  vs vanilla diffusion policy, while improving planning quality/mode diversity. A **DiffusionDriveV2**
  (arXiv 2512.07745) adds RL-constrained truncated diffusion on top. Maturity: **PROVEN** (CVPR,
  widely built upon — GoalFlow, GTRS, and others explicitly position against it).
- **Pain points:** **P1** — directly the "generate diverse, keep it cheap" half of the problem; does
  NOT by itself solve selection (still needs a scorer on top, hence GTRS/Hydra-MDP layering on top of
  diffusion generators) — read together with a scorer, not alone.
- **Admissibility:** clean (denoises conditioned on scene features).
- **Cost to try:** swapping our tactical head's candidate generator for a truncated-diffusion generator
  (keeping our own encoder/predictor) is a **medium architecture change, ~5-7 eng-days**, ~1-2 GPU-days
  to retrain the head at our scale.
- **Cheapest discriminating experiment:** generate the SAME number of candidates via truncated-diffusion
  vs our current mechanism (whatever produces the "fan" in the v5f log) and measure candidate DIVERSITY
  and best-in-fan ADE only (not selection) — isolate whether our generator or our selector is the
  weaker half. Outcome A: diffusion-generated fan has a better best-in-fan floor → generation quality is
  part of the P1 gap too; Outcome B: no improvement in best-in-fan → confirms selection (not generation)
  is the entire P1 defect, sharpening the case for GTRS/WoTE/TD-MPC2-style scorers as the sole fix needed.

### GoalFlow — Goal-Driven Flow Matching
- **What:** Score/select a **goal point** from candidates using scene information, THEN condition Flow
  Matching trajectory generation on that goal — directly resolves diffusion's mode-divergence problem by
  anchoring generation to a committed endpoint. Needs only **1 denoising step**.
- **Evidence:** PUBLISHED — arXiv 2503.05689, CVPR 2025, code released. **PDMS 90.3** on NAVSIM.
  Maturity: **PROVEN** (CVPR, code+repo live). GitHub README (fetched directly) confirms the "Goal
  Point scorer" module and 1-step flow matching but its own code for the goal-scoring module was still
  marked "coming soon" at last check — **the exact inputs to the goal scorer (vision-only vs
  privileged/ego) are UNVERIFIED from what I could access.**
- **Pain points:** **P1 and P3 simultaneously** — this is the single most on-point external validation
  of the programme's OWN binding admissibility rule: *"a goal input is admissible... a PREDICTED
  geometric goal point... is the lever that actually works (+4.7)"* (per the programme's own binding
  doc, quoting the wider literature). GoalFlow is a full worked implementation of exactly that
  prescription, at SOTA level, for the flagship's likely strategic-brain goal signal.
- **Admissibility:** **⚠️ MUST VERIFY BEFORE ADOPTING** — the binding rule requires the goal be computed
  from vision/scene only, never from the situation classifier's output and never (at inference) from
  privileged ego/future-path signal. GoalFlow's goal-point scorer inputs were not confirmed from
  available sources. This is the single highest-priority "verify before you touch it" item in this
  entire report.
- **Cost to try:** implementing an analogous goal-point predictor + flow-matching-conditioned tactical
  generation is a **substantial architecture addition, ~10-15 eng-days**, ~2-4 GPU-days to train at our
  scale — justified only after the admissibility check above passes.
- **Cheapest discriminating experiment:** BEFORE building anything, read GoalFlow's actual goal-scorer
  input feature list (needs arXiv/HF fetch access this container lacks — **hand this specific
  sub-question to whichever stream/agent has working WebFetch**). If vision-only: prototype a minimal
  goal-point head off our EXISTING encoder features (no new training data), and check whether
  conditioning the tactical head's manoeuvre choice on a predicted goal point (rather than the current
  5-way softmax) improves the v5f selection gap. Outcome A: improves it → strong, literature-backed,
  admissible fix for P1+P3 together; Outcome B: goal point turns out to require privileged info → the
  finding itself is valuable (explains why the technique "works" in papers using CARLA GT rather than
  human logs) and must NOT be adopted as-is.

### Hydra-MDP / Hydra-MDP++ — re-verification (programme's explicit #1 target, R5)
- **What:** Multi-teacher (human + rule-based) knowledge distillation into a multi-head decoder, each
  head scoring a different evaluation metric (Hydra-MDP++ adds traffic-light compliance, lane-keeping,
  extended comfort).
- **Evidence:** PUBLISHED — arXiv 2406.06978 (1st place, NAVSIM CVPR24 Challenge + Innovation Award) and
  Hydra-MDP++ arXiv 2503.12820. GitHub README (fetched directly) confirms the multi-teacher,
  multi-head-decoder framing and the CVPR24 win, but the repository itself states **"delay in code and
  model release due to company policy"** — code/weights were not available even at the source. **Could
  NOT re-verify the specific "factorised path × velocity vocabulary" characterization** that
  `REFERENCE_SYSTEMS_RANKED.md` attributes to this system (arxiv.org fetch blocked in this container).
  **Status: still tier B / re-verify — this survey did not close that gap, only confirmed the broader
  mechanism class.**
- **Pain points:** **P1** (multi-teacher distillation onto a MULTI-HEAD decoder is a structurally clean
  answer to "one softmax mixes lat+lon" IF the heads are organized around separable sub-metrics/axes —
  exactly what needs the primary-source confirmation above before we treat it as validated).
- **Admissibility:** clean at the mechanism level described (distillation from rule-based teachers).
- **Cost/experiment:** **unchanged recommendation from the existing registry — this remains the
  programme's #1 re-verification target**, now additionally blocked on WebFetch access; escalate to a
  stream/session with working arXiv access rather than re-attempting here.

### GTRS — Generalized Trajectory Scoring (NAVSIM v2 Challenge winner)
- **What:** Unifies coarse (static vocabulary) and fine (dynamically generated) trajectory scoring:
  integrates a **diffusion-based trajectory generator**, a **vocabulary generalization technique**, and
  a **sensor augmentation strategy** so the scorer degrades gracefully under sub-optimal/imperfect
  sensor input (i.e., robust to exactly the kind of imperfect perception a camera-only, from-scratch
  system like ours will have, vs privileged-GT-perception baselines).
- **Evidence:** PUBLISHED — arXiv 2506.06664, **winning solution of the NAVSIM v2 Challenge**.
  Maturity: **PROVEN** (competition-winning, later work — SparseDriveV2 2603.29163 — explicitly builds
  on "scoring is all you need" framing from this line).
- **Pain points:** **P1 directly, and specifically the robustness-to-imperfect-perception angle** —
  most selection literature (Hydra-MDP, DriveSuprim) is read/tuned against near-perfect BEV perception;
  GTRS's sensor-augmentation robustness result is the closest match to OUR situation (a small,
  from-scratch, single-camera encoder, not a mature perception stack).
- **Admissibility:** clean (scores trajectories against scene features + rule-based sub-metrics).
- **Cost to try:** implementing a coarse-vocabulary + fine-diffusion-proposal hybrid scorer on our
  tactical head is a **medium-large change, ~8-12 eng-days**, ~2 GPU-days retrain.
- **Cheapest discriminating experiment:** as a CHEAPER first cut before the full hybrid, replicate just
  GTRS's **vocabulary generalization** idea (score a static coarse vocabulary AND our own dynamically
  generated candidates with the SAME scorer, rather than only the dynamic set) on the frozen v1
  predictor, and check whether adding the coarse vocabulary changes best-in-fan or selected ADE.
  Outcome A: coarse vocabulary improves either metric → cheap generalization lever, adopt; Outcome B: no
  change → our dynamic generation already covers the useful part of trajectory space, and the win (if
  any) must come from the scorer itself, sharpening the case for a GTRS/DriveSuprim-style coarse-to-fine
  scorer specifically.

### DriveSuprim — Towards Precise Trajectory Selection
- **What:** **Coarse-to-fine progressive candidate filtering** + **rotation-based augmentation** for
  OOD robustness + **self-distillation**. Explicitly argues selection-based methods beat single-trajectory
  regression BECAUSE they can evaluate alternatives in safety-critical scenes where subtle differences
  matter — this is close to a direct citation for our own "the world model generates good candidates,
  the head just doesn't pick well" diagnosis.
- **Evidence:** PUBLISHED — arXiv 2506.06659, **AAAI 2026**. **93.5% PDMS on NAVSIM v1, 87.1% EPDMS on
  NAVSIM v2**, without extra data. Maturity: **PROVEN** (AAAI-accepted, current or near-current SOTA on
  both NAVSIM versions per what I could find).
- **Pain points:** **P1 directly**, and notably **rotation-based augmentation for OOD robustness** is
  methodologically adjacent to the programme's OWN `pseudosim.py` `dyaw` heading-perturbation instrument
  (already built for E-DPSI) — DriveSuprim's augmentation could be read as an independent argument that
  heading/rotation robustness training (not just measurement) is a real, current-SOTA-relevant lever.
- **Admissibility:** clean.
- **Cost to try:** coarse-to-fine progressive filtering is a natural fit for a K-candidate tactical head
  — **~6-10 eng-days**, ~1-2 GPU-days.
- **Cheapest discriminating experiment:** apply DriveSuprim's rotation-based augmentation specifically to
  our tactical head's TRAINING data (not just as an eval probe, which is what E-DPSI already does) and
  re-measure both E-DPSI's heading-shortcut sweep AND the selected-vs-best-in-fan ADE gap. Outcome A:
  both improve → rotation augmentation is a genuine two-for-one fix (closes a shortcut AND improves
  selection robustness); Outcome B: only one improves → separates "shortcut removal" from "selection
  quality" as two distinct benefits, useful for prioritization either way.

### ZTRS — Zero-Imitation E2E Driving with Trajectory Scoring
- **What:** **Eliminates imitation learning entirely.** Trained solely on real images + rule-based
  rewards via **Exhaustive Policy Optimization (EPO)**, a policy-gradient variant for enumerable
  trajectory actions with dense supervision. Five modules: image backbone, trajectory tokenizer,
  Transformer decoder, policy head (likelihoods per action), scoring heads (predicted sub-metric
  scores).
- **Evidence:** PUBLISHED — arXiv 2510.24108 (most recent revision dated **2026-07-08**, i.e. this is
  itself a 2026 update, squarely in-scope for "not just 2024-25 classics"). SOTA on **Navhard**,
  outperforms IL baselines on **HUGSIM**. Maturity: **PROMISING** (novel enough — "first framework
  eliminating IL entirely" — to not yet be PROVEN at the Hydra-MDP/GTRS replication level, but the
  July-2026 revision shows active, current development).
- **Pain points:** **P1 (most directly of anything in this survey)** — if a policy can be trained
  end-to-end from **rule-based reward alone**, this bypasses the entire "imitation learns to imitate a
  bad selection habit" failure mode implicated in the v5f gap; **P2** — rule-based rewards can encode
  speed-holding/distance-keeping DIRECTLY as a dense reward term, rather than hoping BC captures it
  (relevant to the 88.7% longitudinal gap).
- **Admissibility:** clean — reward is rule-based (rules over trajectories against the scene), not
  situation-classifier-derived.
- **Cost to try:** this is the most architecturally disruptive item in category C — replacing
  imitation-trained heads with a pure policy-gradient objective is a **large change, ESTIMATED 15-20
  eng-days**, ~3-5 GPU-days for a driving-scale reproduction at our size.
- **Cheapest discriminating experiment:** do NOT reproduce ZTRS wholesale first. Instead, take the
  EXISTING tactical head and replace only its TRAINING LOSS with an EPO-style policy-gradient objective
  over the same discrete manoeuvre vocabulary (same enumerable action space we already have, same
  encoder/predictor frozen), using `taniteval`'s rule-based cost terms as the reward. Outcome A: the
  v5f selection gap closes substantially → the LOSS FUNCTION (imitation vs reward-driven), not
  architecture, was the dominant cause of P1, which would be the single highest-value finding available
  to the programme this quarter; Outcome B: no change → rules out the loss-function hypothesis cheaply
  (one training run, no new modules) before any larger ZTRS-style rebuild is considered.

### iPad — Iterative Proposal-centric E2E Autonomous Driving
- **What:** **Iteratively refines** trajectory proposals toward human-like trajectories while preserving
  multi-modality at intersections (rather than one-shot generation). Also used as a baseline planner
  for **TOAD** (Test-Time Trajectory Optimization, arXiv 2606.07170), which explicitly reports gains
  come from **search discovering NEW trajectories, not re-ranking/smoothing existing ones** — an
  important, directly on-point empirical distinction.
- **Evidence:** PUBLISHED — arXiv 2505.15111 (iPad); TOAD arXiv 2606.07170 tested WITH iPad and
  Hydra-MDP as base planners. Maturity: **PROMISING** (iPad); TOAD's finding (search > re-ranking) is
  **PROMISING** and directly relevant methodologically.
- **Pain points:** **P1** — TOAD's finding is a sharp, falsifiable claim directly against our own
  candidate approach: if search-based test-time trajectory discovery beats re-ranking a fixed candidate
  set, that argues our fix should not be "a better scorer over the SAME fan" but "generate MORE/BETTER
  candidates at inference time," which changes which of the two P1 experiments above (generator vs
  selector) is worth prioritizing.
- **Admissibility:** clean.
- **Cost to try:** test-time search (even simple gradient-free perturbation search around the existing
  fan) is CHEAP — **~3-5 eng-days, 0 GPU-days beyond eval compute** (it's an inference-time technique).
- **Cheapest discriminating experiment:** on the frozen v5f checkpoint, add a simple test-time search
  step (e.g., local perturbation + re-score using existing cost terms) around the current fan and
  measure whether the SELECTED trajectory's ADE improves beyond just re-ranking the existing fan.
  Outcome A: improves → TOAD's "search beats re-ranking" finding transfers, and test-time search is a
  near-free P1 win layered on top of whatever scorer is chosen; Outcome B: no improvement → our fan
  already spans the useful search space and the whole P1 fix is in scoring/selection, not generation
  breadth — directly informs which of GTRS/DriveSuprim/ZTRS to prioritize next.

### TrajHF — RLHF for Trajectory Generation
- **What:** GRPO-based direct fine-tuning of an already-imitation-trained generative trajectory model
  toward human PREFERENCE (personalized driving style), using multi-conditional denoising + behavior
  cloning loss retained to preserve base capability (avoids catastrophic forgetting of the imitation
  prior).
- **Evidence:** PUBLISHED — arXiv 2503.10434 (*Learning Personalized Driving Styles via RLHF*),
  performance "comparable to SOTA on NAVSIM." Maturity: **PROMISING**.
- **Pain points:** **P1 (tertiary)** — less about fixing wrong selection, more about ALIGNING an
  already-adequate generator toward a preference axis; more directly useful if/when the programme wants
  a comfort/style axis than as a P1 fix per se. Also a candidate mechanism for the "goal/tactical
  preference" axis distinct from the situation classifier (relevant to the admissibility rule — GRPO
  reward here is human-preference-derived, not classifier-derived, so it is a clean pattern to imitate
  if a preference signal is ever wanted).
- **Admissibility:** clean.
- **Cost to try:** LOW priority relative to the other category-C items above; **~5-8 eng-days** if
  pursued, no new GPU beyond fine-tuning cost.
- **Cheapest discriminating experiment:** deprioritized — only worth running after a P1 selection fix
  (GTRS/ZTRS/DriveSuprim-style) is in place, since RLHF preference-tuning on top of a demonstrably bad
  selector would confound style-alignment gains with selection-quality gains.

### Current leaderboard snapshot (NAVSIM v2 / Bench2Drive / WOD-E2E) — as of search date, not live-verified
- **NAVSIM v2 / EPDMS (`navhard`):** could not resolve a single authoritative current top-line number
  from this container (HF Space leaderboard not fetchable); by paper-reported EPDMS, **DriveSuprim
  87.1%** and **GTRS** (challenge winner) are the strongest confirmed points; a system called
  **"DriveZero-Scale"** appeared in search results as evaluated 2026-09-05 but its score was
  **UNVERIFIED** (not resolvable from snippets).
- **Bench2Drive Driving Score:** search results reported **AutoVLA 78.84**, "MindDrive" 78.04, and
  "SpaceDrive+" 78.02 as top-3 — **these three numbers are UNVERIFIED beyond the search snippet** (no
  primary source opened; "MindDrive"/"SpaceDrive+" could not be independently traced to a paper in the
  time available and may be leaderboard-only entries without a public write-up). Separately, **ORION**
  is independently confirmed via its own abstract-level search result at **77.74 DS / 54.62% SR**, and
  **SimLingo** at **85.94 DS** (its own claim, vision-only, CARLA LB2.0 context rather than strictly
  Bench2Drive — do not treat as directly comparable without checking the exact track).
- **WOD-E2E (Waymo):** **no formal 2026 Challenge**, but the leaderboard is active; scored by **Rater
  Feedback Score (RFS)**, not ADE — reinforcing the programme's own already-adopted rule (R6 in the
  registry) never to rank on ADE alone.
- ⚠️ **Reporting caveat for whoever uses this snapshot:** none of these numbers should be quoted as a
  programme decision input (per CLAUDE.md rule — this is INHERITED-from-search-snippet, not
  independently re-verified against a primary source at the individual-claim level for most rows).
  Treat this subsection as "where to look next," not as a citable leaderboard.

---

## 4. Category D — Foundation & VLA models for driving and robotics

### Alpamayo-R1 — verify relationship to PhysicalAI-AV (per brief's explicit instruction)
- **What:** 10B-param VLA: **Cosmos-Reason** VLM backbone (physical-AI-pretrained) + a **diffusion
  trajectory decoder**; trained via a "Chain-of-Causation" (CoC) SFT stage (hybrid auto-label +
  human-in-the-loop causal reasoning traces) then RL to align reasoning with action.
- **Evidence:** PUBLISHED — arXiv 2511.00088 (*Alpamayo-R1: Bridging Reasoning and Action Prediction for
  Generalizable Autonomous Driving in the Long Tail*). **12% planning-accuracy improvement on
  challenging cases** vs a trajectory-only baseline; **35% reduction in close-encounter rate**
  (closed-loop sim); RL post-training **+45% reasoning quality, +37% reasoning-action consistency**;
  **99ms latency**, real-time on-vehicle road tests. Family: Alpamayo 1 Nano (10B) → 1.5 Nano → 2 Super
  (34B); open ecosystem includes **AlpaSim** (closed-loop AV sim) and **AlpaGym** (closed-loop RL
  training) — both already used by the programme for its own closed-loop numbers (8/12 vs 2/12).
  Maturity: **PROVEN at the vendor level** (NVIDIA research publication + open weights + road-test
  claim), independent replication **UNVERIFIED**.
  ⚠️ **Verification result on the specific question asked: I could NOT confirm from any source opened in
  this session that Alpamayo-R1 is trained specifically on the PhysicalAI-AV corpus.** Every source
  found describes Alpamayo-R1's OWN dataset as the "Chain-of-Causation" (CoC) dataset (auto-labelled +
  human-in-the-loop reasoning traces over driving logs), and separately describes PhysicalAI-AV as part
  of the wider "Alpamayo open ecosystem" (alongside AlpaSim/AlpaGym) — but no source stated outright
  that AR1's own training footage IS the PhysicalAI-AV release, vs. NVIDIA's separate internal fleet
  logs used only to build CoC. **Mark this UNVERIFIED, not assumed-true, until someone with arXiv full-text
  access reads AR1's §"Data" section directly.** This matters concretely: if AR1 is trained on the SAME
  corpus we are, its numbers become a much more direct competitive/reference bar; if not, it is a
  same-ecosystem but different-corpus comparison.
- **Pain points:** **P3 (strategic-brain design)** — Chain-of-Causation is a concrete, working example
  of a "singular high-level decision + minimal causal factors + causal text path" strategic layer, one
  concrete alternative to our own strategic-brain design if we ever add reasoning traces; **P8**
  (safety) — the "Semantic Observer" concept mentioned alongside it (1-2Hz semantic-anomaly detection
  layer) is architecturally identical in spirit to our own fallback-monitor idea (H11).
- **Admissibility:** the CoC reasoning trace is derived from privileged/offline labels (per "ground
  truth may use ego + other + maps" rule) — fine for a LABEL, would need re-audit if any part of it were
  fed back as a live inference-time input to a downstream module.
- **Cost to try:** not reproducible at our scale (10B params, NVIDIA-fleet CoC data); the transferable
  IDEA (constrain any reasoning/decision explanation to ONE decision + minimal causal factors + a
  2-second context window) is a **design pattern, ~2-3 eng-days** to prototype as an explainability head
  off our OWN latents (H13-style), not a retrain.
- **Cheapest discriminating experiment:** N/A for reproduction; the actionable item is the verification
  task above (arXiv full-text read of AR1's data section) — **hand to a stream with WebFetch access,
  ~30 minutes**.

### EMMA — End-to-End Multimodal Model for Autonomous Driving (Waymo)
- **What:** Gemini-powered E2E model recasting ALL driving tasks (planning, perception, road-graph) as
  vision-question-answering in a single language space — maximizes use of a pretrained LLM's world
  knowledge and chain-of-thought.
- **Evidence:** PUBLISHED — arXiv 2410.23262 (Waymo). SOTA motion planning on nuScenes, competitive on
  WOMD and WOD 3D detection; co-training across tasks improves all three. Stated limitations: few image
  frames, no LiDAR/radar, computationally expensive. Maturity: **PROVEN** (Waymo-published, replicated
  open-source as "OpenEMMA").
- **Pain points:** **P10** (evaluation — EMMA's own stated limitation, "computationally expensive" +
  "text-only I/O for geometry," is a caution directly against a text-mediated strategic brain for a
  sub-300M/Thor-class budget); **P3** — a full VLM-as-planner is the opposite end of the spectrum from
  our compact 4-brain design; useful as the "what we are deliberately NOT doing and why" comparison
  point in any paper/positioning section.
- **Admissibility:** N/A (not planned for adoption; reference only).
- **Cost/experiment:** not applicable at our budget; **reference item, no experiment proposed.**

### AutoVLA — adaptive reasoning + RL fine-tuning VLA
- **What:** Single autoregressive model unifying reasoning and action; **dual thinking modes** — "fast"
  (trajectory-only) and "slow" (chain-of-thought-augmented) — chosen adaptively; tokenizes continuous
  trajectories into discrete feasible actions for direct LM integration.
- **Evidence:** PUBLISHED — arXiv 2506.13757 (UCLA). Reported **78.84 Driving Score on Bench2Drive**
  (UNVERIFIED against a primary fetch, per leaderboard caveat above, but internally consistent across
  multiple independent search snippets). Maturity: **PROMISING-to-PROVEN**.
- **Pain points:** **P1/P9** — the fast/slow SWITCH is itself a cheap, concrete instance of "don't always
  pay for expensive reasoning," directly relevant to our own cadence design (operative 10Hz / tactical
  5-cadence / strategic 20-cadence) — worth reading as a possible trigger-condition design for WHEN the
  tactical brain should escalate to a more expensive computation, rather than always running fixed-cadence.
- **Admissibility:** clean if the fast/slow trigger is scene-derived, not classifier-derived — would
  need the same disjointness check as any gating signal if adopted.
- **Cost to try:** **~4-6 eng-days** to prototype an adaptive-cadence trigger (reuse existing imagination-
  error signal, already free per A9, as the trigger rather than building a new one).
- **Cheapest discriminating experiment:** using the EXISTING imagination-error monitor as an "escalate to
  slow/tactical" trigger (rather than fixed cadence-5), check whether escalation correlates with the
  windows where the current tactical head is actually wrong (i.e., does high imagination-error predict
  bad tactical decisions). Outcome A: yes → free, already-available signal for adaptive-cadence routing;
  Outcome B: no correlation → the fixed-cadence assumption is fine and adaptive triggering is not a
  priority lever.

### ORION — holistic VLA-instructed action generation
- QT-Former (long-term history) + LLM reasoning + generative planner, jointly aligning reasoning and
  action spaces. PUBLISHED, arXiv 2503.19755, ICCV 2025. **77.74 DS / 54.62% SR on Bench2Drive**
  ("outperforms SOTA by 14.28 DS / 19.61% SR" per the paper's own claim). Maturity: **PROVEN**
  (ICCV-accepted, code released). Pain point: **P3** — another concrete "align reasoning space and
  action space" design, same family as Alpamayo-R1/AutoVLA but smaller/more accessible (Xiaomi lab, not
  NVIDIA-fleet-scale data) — the most REPRODUCIBLE of the three VLA-reasoning systems if the programme
  ever wants to prototype a language-mediated strategic brain. Admissibility: needs the same audit as
  any reasoning-to-action bridge. Cost: full reproduction is still a large VLA build (**ESTIMATED
  15-25 eng-days**); not recommended before P1/P2 fixes above given the priority order. Cheapest
  experiment: read its "align reasoning space and action space" loss design as a reference architecture
  only, no build — **0 GPU-days**.

### SimLingo — Vision-Only Closed-Loop Driving with Language-Action Alignment
- **What:** Camera-only (**no LiDAR**), decouples **speed** waypoints from **path** waypoints via a
  disentangled MLP head — i.e., an explicit LAT/LON split at the output head, precisely the fix our own
  P1 diagnosis (single 5-way softmax MIXING lat+lon) is missing. Introduces "Action Dreaming" — an
  instruction-following consistency check between language and control.
- **Evidence:** PUBLISHED — arXiv 2503.09594 (Tübingen, Renz et al.). **SOTA on CARLA Leaderboard 2.0
  and Bench2Drive using camera only**, DS **85.94**. Maturity: **PROVEN**.
- **Pain points:** **P1 — this is the single cleanest existing precedent for "split lat/lon at the head"
  the survey found.** Directly actionable: SimLingo's disentangled-MLP pattern (separate heads for
  temporal/speed vs geometric/path waypoints, feeding from a shared trunk) is close to a drop-in
  replacement for our 5-way manoeuvre softmax's mixed lat+lon design; **P2** — the speed head is
  explicitly separated and independently supervised, which is exactly the kind of architectural
  isolation our 88.7%-longitudinal-gap diagnosis calls for.
- **Admissibility:** clean — vision-only by design (also directly compliant with the programme's OWN
  vision-only-inference binding rule, unlike most of category D).
  ⭐ **Strongest single admissibility fit in category D.**
- **Cost to try:** splitting the tactical head's output into a disentangled lat-MLP + lon-MLP off the
  SAME shared trunk is a **small-to-medium change, ~3-5 eng-days**, ~1 GPU-day retrain.
- **Cheapest discriminating experiment:** on the frozen v1 encoder/predictor, replace ONLY the tactical
  head's final layer — one 5-way softmax → two separate heads (manoeuvre-lateral class + target-speed
  regression) — retrain just that head, and re-measure both the tactical ADE (currently ~3.4m, worse
  than constant-velocity) and the longitudinal-specific metrics (target-speed accuracy, distance-keeping)
  the programme's binding eval rule already requires. Outcome A: tactical ADE improves and beats
  constant-velocity → validates the "mixed softmax" diagnosis as the dominant cause and gives a cheap,
  near-drop-in fix; Outcome B: no improvement → the defect is deeper than output-head entanglement
  (e.g., in the training data/loss itself), redirecting effort toward ZTRS/GTRS-style objective changes
  instead of an architecture change.

### OpenDriveVLA
- Open-source-LLM-based VLA; hierarchical vision-language alignment (2D+3D tokens → unified semantic
  space). PUBLISHED, arXiv 2503.23463, **AAAI 2026**. nuScenes open-loop **L2 0.33m** (3B/7B versions).
  Maturity: **PROVEN** (AAAI-accepted, code released). Pain point: **P5** (representation) — its
  hierarchical 2D/3D token alignment is a concrete recipe for grounding a from-scratch encoder's tokens
  geometrically without a full 3D detector, potentially relevant if the encoder ever needs auxiliary 3D
  grounding losses. Admissibility: clean. Cost: **~10+ eng-days** for the alignment mechanism alone;
  lower priority than category-C items. Cheapest experiment: not prioritized this cycle — logged as a
  P5 backlog read.

### π0 / π0.5 / π*0.6 (RECAP) — Physical Intelligence's generalist-robot lineage
- **What:** π0 → π0.5 (open-world generalization via co-training on heterogeneous robot/web/semantic
  data) → **π*0.6 with RECAP** ("RL with Experience & Corrections via Advantage-conditioned Policies")
  — combines (1) demonstrations, (2) real-time EXPERT CORRECTIONS during autonomous execution, and (3)
  self-improvement via RL from autonomous trials, into one advantage-conditioned policy.
- **Evidence:** PUBLISHED — π0.5 arXiv 2504.16054; π*0.6/RECAP via Physical Intelligence's own technical
  report (`pi.website/download/pistar06.pdf`, not fetchable from this container — **PUBLISHED via
  vendor primary source, mechanism confirmed by multiple independent secondary write-ups, but I did not
  personally read the PDF**). Headline: **π*0.6 doubles throughput and cuts failures 2×+ on hard tasks**;
  ran unattended for **18 hours** making espresso, folded **50 novel laundry items** in an unseen home,
  assembled/labelled **59 real factory boxes**. Maturity: **PROVEN at the vendor-demo level**
  (extensive real-robot hours, not yet third-party replicated).
- **Pain points:** **P1 — RECAP is the most directly transferable MECHANISM in category D for our
  selection problem**, because it explicitly targets "imitation's key flaw: small mistakes compound in
  real interaction" via a THIRD data source most driving work ignores: **live corrections during
  autonomous execution**, not just offline demonstrations or a reward function. Our own equivalent would
  be flagging exactly the windows where the tactical head's selection diverges from a better-scoring
  candidate (already measurable via the v5f "best-in-fan vs selected" gap) and training an advantage-
  conditioned correction signal from THOSE specific failures, rather than a blanket reward or a blanket
  imitation loss.
- **Admissibility:** clean — corrections come from execution outcomes, not a situation classifier.
- **Cost to try:** a scoped, driving-specific RECAP analogue (advantage-conditioning the tactical head on
  its OWN historical best-in-fan-vs-selected gap, as a cheap proxy for "expert correction," entirely
  offline on existing logs) is **ESTIMATED 8-12 eng-days**, minimal new GPU (reuses existing rollouts).
- **Cheapest discriminating experiment:** construct an offline "advantage" label per training window =
  (best-in-fan ADE − selected ADE) from EXISTING v5f-style logs, and add an advantage-conditioned
  auxiliary loss term to the tactical head (predict/condition on this advantage at train time, similar in
  spirit to RECAP but built entirely from logs already on disk — no live correction loop needed for this
  cheap version). Outcome A: selection gap narrows → strong, cheap, log-only validation of the RECAP
  mechanism for driving; Outcome B: no change → the advantage signal from past logs is too sparse/noisy
  to shape the head this way, and a genuine live-correction loop (expensive, needs a sim or fleet) would
  be needed to test RECAP properly — a much bigger ask, correctly deferred.

### GR00T N1 / N1.5 / N1.6 / N1.7 (NVIDIA humanoid foundation model)
- **What:** Open VLA foundation model for generalist humanoid robots; rapid iteration — N1 (Mar 2025) →
  N1.5 (May 2025, frozen VLM + Eagle 2.5 grounding + FLARE objective learning from human ego-video +
  GR00T-Dreams synthetic-data blueprint, cutting data-collection time from ~3 months to **36 hours**) →
  N1.6 (Dec 2025, Cosmos-2B VLM backbone, 2× larger DiT, state-relative action chunks) → N1.7 (current).
- **Evidence:** PUBLISHED — arXiv 2503.14734 (GR00T N1) + NVIDIA Newsroom for N1.5/N1.6 (press-release
  level; **UNVERIFIED in technical depth** beyond what WebSearch summarized). Maturity: **PROVEN**
  (shipped, iterated 4 times in ~18 months — this is a fast-moving, well-resourced reference line even
  if not driving-specific).
- **Pain points:** **P4/P9** — the **GR00T-Dreams** synthetic-data-generation blueprint (36h vs 3 months)
  is a directly relevant PATTERN for our own small-data problem (P4): use a world model to GENERATE
  action-labelled training variety cheaply, rather than collect more real hours. This is architecturally
  the same idea as our own inverse-dynamics-labels-from-passive-video plan (H7) but industrially proven
  at humanoid scale.
- **Admissibility:** N/A directly (not a driving system); as a pattern, clean.
- **Cost to try:** not directly portable (different embodiment/domain); the TRANSFERABLE idea (world-
  model-driven synthetic action-label generation) is already on our own roadmap as H7 — this is
  corroborating evidence to prioritize H7, not a new build. **0 additional cost — reference item.**

### Gemini Robotics 1.5 / ER 1.5 / ER 1.6 / ER 2
- **What:** Dual-model system: Gemini Robotics 1.5 (vision+instruction → motor commands) + Gemini
  Robotics-ER (embodied reasoning: plans using digital tools like web search before handing off to the
  execution model). **Motion Transfer** mechanism: a task learned on one embodiment (ALOHA2 dual-arm)
  transfers DIRECTLY to a different embodiment (Franka, and Apptronik's Apollo humanoid) **without
  retraining**.
- **Evidence:** PUBLISHED — arXiv 2510.03342 (Google DeepMind). Maturity: **PROVEN at vendor-demo level**
  (specific cross-embodiment zero-shot transfer claim, not independently replicated by a third party in
  what I found).
- **Pain points:** **P4/P6** — Motion Transfer is relevant to any future embodiment change (e.g., if the
  Jetson Thor target vehicle platform changes, or if the model is ever adapted across vehicle types); the
  "reason first with digital tools, then hand off to a fast execution model" split is architecturally the
  SAME dual-process idea as our own strategic (slow)/operative (fast) split, independently arrived at by
  a completely different lab — **corroborating evidence for H1's frequency-separation design**, not a
  new technique to adopt.
- **Admissibility:** N/A directly (different domain); as a design pattern, clean.
- **Cost/experiment:** reference/corroboration item only — **0 cost, no experiment**, cited for the
  architecture-validation value in the TOP-10 discussion below.

---

## 5. Category E — Brain/cortex-inspired architectures

### LeCun's H-JEPA / Configurator / Cost-Module architecture
- **What:** The full 6-module agent decomposition (Configurator, Perception, World Model, Cost Module,
  Actor, Short-Term Memory) from *A Path Towards Autonomous Machine Intelligence*, with hierarchical
  planning (higher levels set abstract subgoals for lower levels).
- **Evidence:** PUBLISHED — position paper (2022) + ongoing elaboration (Meta AI blog, various follow-on
  papers e.g. *Value-guided action planning with JEPA world models*, arXiv 2601.00844, found in this
  survey — a 2026 continuation of the programme). Maturity: **PROVEN as a framework** (widely cited,
  actively extended), **SPECULATIVE as a complete implementation** (no full 6-module system has been
  published end-to-end at scale by LeCun's own group as of what I found).
- **Pain points:** **P3 directly** — our own 4-brain split IS a partial instantiation of this
  architecture (World Model = operative, Actor = tactical/strategic, hierarchy = the strategic/tactical
  split); the piece we are explicitly MISSING relative to the full LeCun design is a **Configurator**
  (a module that reconfigures cost/actor/world-model behavior per situation) — this is architecturally
  distinct from (and safer than) letting a "situation classifier" feed the goal path directly, because a
  Configurator RECONFIGURES which cost terms and actor parameters are active, rather than injecting a
  content signal into the goal itself. **This may be exactly the legitimate way to use situation
  information that the binding admissibility rule is trying to rule OUT of the goal path** — worth a
  careful design discussion, since a Configurator-style use of situation (reweighting cost terms, e.g.
  "in a merge situation, weight TTC higher") is architecturally different from a goal-path leak, but the
  distinction is subtle enough to need explicit, written justification before building it (per the
  binding rule's own instruction: "if a shared trunk feeds both, say so and justify why that is not a
  back door").
- **Admissibility:** ⚠️ **nuanced — see above.** A Configurator that reweights COST, not one that emits a
  GOAL, is the safer reading, but this needs to be argued explicitly, not assumed.
- **Cost to try:** a minimal Configurator (situation-conditioned cost-term reweighting, e.g. modulating
  the existing loss/cost weights fed to the tactical head based on a coarse scene-derived signal) is
  **~5-8 eng-days**, no new GPU beyond retraining affected heads.
- **Cheapest discriminating experiment:** implement the narrowest possible Configurator — reweight ONLY
  the existing TTC/comfort/progress cost terms already in `taniteval` based on a cheap, clearly
  vision-only scene signal (e.g., "how many agents are close") — and check (a) whether it improves
  selection quality, and (b) run the SAME leak test the programme uses elsewhere (could the reweighting
  signal have been computed from the situation classifier's output?). Outcome A: improves selection AND
  passes the leak test → a validated, admissible new mechanism, distinct from a goal-path leak; Outcome
  B: fails the leak test → confirms the Configurator pattern is NOT a safe workaround for this programme
  and must not be pursued further under this framing.

### Active inference & predictive coding for driving
- **What:** Frame driving control as minimizing "surprise" (prediction error) under a generative model,
  unifying perception and action; several 2025-2026 papers apply this directly to AV control and human
  driver modelling.
- **Evidence:** PUBLISHED — *Towards Human-Like Driving: Active Inference in AV Control*, arXiv
  2407.07684; *Active inference as a unified model of collision avoidance behavior in human drivers*,
  arXiv 2506.02215; *Towards Intelligible HRI: Active Inference... Occluded Pedestrian Scenarios*, arXiv
  2602.23109 (HRI 2026 — genuinely new, in-scope for 2026); combining active inference with diffusion
  motion prediction, arXiv 2406.00211. Maturity: **PROMISING** (multiple independent groups, growing
  2025-2026 body of work; no large-scale industrial deployment found).
- **Pain points:** **P7 directly** — "surprise"/prediction-error IS our imagination-error signal (A9);
  active inference formalizes using that SAME quantity to drive ACTION SELECTION (minimize expected
  future surprise), not just as a free OOD monitor. This is a principled, literature-grounded upgrade
  path for turning our already-existing, already-free imagination-error signal into a selection
  criterion, rather than adding a whole new mechanism; **P8** (safety) — active inference's
  "occluded pedestrian" HRI-2026 application is a direct hit on exactly the kind of hidden-actor
  scenario the programme's own H15/LOPS work targets.
- **Admissibility:** clean — surprise is computed from the model's own prediction error against
  observed frames, not from a situation classifier.
- **Cost to try:** using imagination-error (already computed, free) as an additional SELECTION term
  (prefer candidates with lower predicted future surprise, not just lower cost) is a **cheap addition,
  ~3-5 eng-days**, 0 new GPU-days.
- **Cheapest discriminating experiment:** add "minimize predicted imagination-error over the rollout" as
  an extra term in candidate scoring (alongside whatever selector is chosen from category C) and check
  whether it correlates with/improves the v5f selection gap independently of the other proposed scoring
  fixes. Outcome A: adds independent signal → active-inference-style surprise-minimization is a free,
  literature-grounded ingredient to fold into ANY of the category-C selector designs; Outcome B: redundant
  with cost-based scoring already used → deprioritize, but keep imagination-error as the OOD monitor it
  already is.

### Basal ganglia-inspired action selection (direct analogue of our selector, P1)
- **What:** Direct/indirect pathway gating: the direct pathway FACILITATES a candidate action, the
  indirect pathway INHIBITS competitors — a biological WINNER-TAKE-ALL circuit for exactly the
  "many candidate actions, pick one" problem our tactical head has.
- **Evidence:** PUBLISHED — foundational robotics line (robot basal ganglia models, action selection in
  survival tasks, contracting dynamical-systems formulation) plus recent computational work
  quantitatively analyzing D1/D2 pathway contributions (arXiv 2404.13888) and spiking-network
  arm/locomotor coordination (arXiv 2606.11034, 2026). Maturity: **PROVEN as a robotics mechanism**
  (decades of replication in behavior-based robotics), **SPECULATIVE as a direct fit for a modern deep
  tactical head** (no found paper wires this INTO a modern transformer-based driving stack specifically).
- **Pain points:** **P1 — the most direct biological analogue to our exact defect.** The key structural
  idea worth stealing is NOT the spiking-neuron biology, but the **competitive dual-pathway gating
  principle**: separate the "how good is this candidate" (direct/facilitation, ~ our scorer) from "how
  much should this candidate be suppressed given the OTHERS" (indirect/inhibition, ~ an explicit
  competition/normalization term) — which our SINGLE softmax collapses into one computation. This maps
  onto a concrete, testable architecture change: replace the tactical softmax with an explicit
  scorer-plus-lateral-inhibition mechanism (e.g., a normalization that suppresses near-duplicate
  candidates, rather than a plain softmax over raw scores).
- **Admissibility:** clean (an architectural/computational analogy, not a data source).
- **Cost to try:** **~4-6 eng-days** to prototype a lateral-inhibition/competition layer on top of the
  existing candidate scores (whichever scorer is chosen from category C).
- **Cheapest discriminating experiment:** on the frozen v1/v5f candidate set, replace the final softmax
  with a simple lateral-inhibition rule (e.g., score candidates, then suppress candidates too similar to
  a higher-scoring one before renormalizing) and measure whether the SELECTED trajectory's ADE improves
  purely from this normalization change, with NO change to the underlying per-candidate scores. Outcome
  A: improves → part of our P1 defect is in how scores get turned into a DECISION (missing competition/
  suppression), independent of scoring quality — a very cheap fix; Outcome B: no change → the defect is
  entirely in the per-candidate scores themselves, ruling out this specific mechanism cheaply.

### Cerebellar forward models (motor prediction, Smith-predictor pattern)
- **What:** The cerebellum is modelled as a **forward model**: it takes a copy of the motor command
  (efference copy) and predicts the SENSORY CONSEQUENCE ahead of actual feedback, acting as a Smith
  predictor to compensate for feedback delay.
- **Evidence:** PUBLISHED — *50 years since the Marr, Ito, and Albus models of the cerebellum*, arXiv
  2003.05647 (review); cerebellar-predictive-learning spiking control, arXiv 2011.01641; a 2026 brain-
  inspired reflexive-control paper (arXiv 2601.14628) explicitly built around an "Iterative Refinement
  Loop" using anticipated sensory feedback. Maturity: **PROVEN as neuroscience**, **PROMISING as a
  robotics control pattern** (multiple working implementations, none at driving-planner scale found).
- **Pain points:** **P6 (latency)** — this is precisely the Smith-predictor pattern our own operative
  brain already implements (predict forward to compensate for the ~100ms planning-tick latency against a
  10Hz budget) — mostly CORROBORATING evidence that the operative brain's basic design (predict-ahead to
  cancel latency) is the biologically correct pattern, not a new lever.
- **Admissibility:** clean.
- **Cost/experiment:** **no new experiment needed** — this is validation-by-analogy for the existing
  operative-brain design, worth citing in any paper/positioning writeup rather than a build item.

### Hippocampal replay & cognitive maps
- **What:** Replay = reactivation of place-cell sequences encoding recent experience, occurring in
  hippocampus/PFC; theorized to support value-based RL and to be equivalent to a "cognitive map" (≈ a
  world model, in RL terms).
- **Evidence:** PUBLISHED — *Brain-Like Replay Naturally Emerges in RL Agents*, arXiv 2402.01467 (replay
  emerges WITHOUT being explicitly designed in, when an RL agent has both a policy net and a world
  model — directly relevant to whether our own architecture would benefit from an explicit replay
  buffer, or might already exhibit replay-like dynamics); *A Robotic Model of Hippocampal Reverse Replay
  for RL*, arXiv 2102.11914 (reverse replay accelerates learning + improves stability/robustness).
  Maturity: **PROMISING** (solid computational-neuroscience-to-RL bridge, not yet driving-specific).
- **Pain points:** **P4/P10** — directly upstream of the programme's OWN H10 (latent RAG / continual
  learning from experience), which the programme has ALREADY measured has a real mechanism (+18.8% on
  surprise contexts) and a real failure mode (−24% interference on well-predicted contexts, hence
  surprise-gated retrieval). The hippocampal-replay literature's finding that **REVERSE replay
  specifically improves stability** is a concrete, testable refinement to H10's write policy (currently
  "write on imagination-error spike," forward in time) — reverse-order replay of stored surprising
  episodes during a training/consolidation phase is untested in the programme's own H10 work.
- **Admissibility:** clean (operates on stored past experience, not live classifier output).
- **Cost to try:** **~3-5 eng-days** to add a reverse-replay consolidation pass to the existing H10
  memory-write mechanism (which the programme has already implemented per `INITIAL_RESEARCH_SYNTHESIS.md`).
- **Cheapest discriminating experiment:** on the existing surprise-gated memory buffer, compare
  forward-order vs reverse-order replay during a consolidation/fine-tuning pass, measuring the SAME
  interference metric H10 already tracks (−24% on well-predicted contexts). Outcome A: reverse replay
  reduces interference vs forward → adopt cheaply, directly improves an already-known programme defect;
  Outcome B: no difference → replay ORDER doesn't matter for us, simplifying the H10 implementation
  (no need to maintain ordering) without losing anything.

### Dual-process (System 1/2) driving: DriveVLM-Dual, FASIONAD, ETA
- **What:** A slow VLM branch does high-level reasoning/situational assessment at low frequency; a fast
  classical/learned planner does real-time trajectory generation, with the slow branch's output serving
  as a reference/conditioning signal for the fast branch (async, slow-fast coupling).
- **Evidence:** PUBLISHED — DriveVLM-Dual, arXiv 2402.12289; FASIONAD ("FAst and Slow FusION Thinking"),
  arXiv 2411.18013; ETA ("Efficiency through Thinking Ahead"), arXiv 2506.07725. Maturity: **PROVEN**
  (multiple independent groups converging on the same slow/fast split, 2024-2026).
- **Pain points:** **P3/P9 — this is the closest external validation of our OWN operative/tactical/
  strategic frequency separation** (10-20Hz / cadence-5 / cadence-20), independently re-derived by at
  least three separate groups for driving specifically (plus Gemini Robotics 1.5's dual-model split in
  category D, and LeCun's Configurator/Actor split in this category) — this is strong corroborating
  evidence, from FIVE independent sources now, that hierarchical multi-timescale processing is the right
  shape for this problem, which is directly relevant to defending H1 against the "flat REF-C ties/beats
  us" finding (P3): the finding may be about EXECUTION quality within the hierarchy, not about whether
  hierarchy is the right shape.
- **Admissibility:** clean (reasoning conditions the fast planner via features, not via a smuggled
  situation-classifier output — though exactly HOW each system couples the branches would need the same
  disjointness audit if any coupling mechanism were adopted verbatim).
- **Cost/experiment:** **primarily a corroboration item**; the actionable piece already appears in
  category D (AutoVLA's adaptive fast/slow SWITCH experiment) — do not duplicate, reference instead.

### Hierarchical Reasoning Model (HRM) and Tiny Recursive Model (TRM)
- **What:** HRM: two recurrent modules (slow/abstract high-level, fast/detailed low-level) doing
  MULTI-STEP LATENT reasoning in a single forward pass, no chain-of-thought text. TRM (its successor):
  strips this down to ONE tiny 2-layer network recursing on its own latent+answer state, and BEATS HRM.
- **Evidence:** PUBLISHED — HRM: arXiv 2506.21734 (Sapient + Tsinghua), **27M params, 1000 training
  examples**, strong ARC-AGI/Sudoku/Maze results (confirmed via direct GitHub fetch). TRM: arXiv
  2510.04871 (Jolicoeur-Martineau, Samsung SAIL Montreal), **7M params**, **45% ARC-AGI-1 / 8%
  ARC-AGI-2** — beats HRM's reported **40%** on ARC-AGI-1 with fewer params and, per the author's own
  framing, "nothing to do with the human brain, no hierarchy, no fixed-point theorem" (confirmed via
  direct GitHub fetch). Maturity: **PROVEN** (both open-sourced, TRM is a direct, reproducible
  improvement on HRM).
- **Pain points:** **P1/P9 — the TRM result is a load-bearing CAUTION, not just an opportunity**: the
  author explicitly shows that the "hierarchical, brain-inspired" framing of HRM was UNNECESSARY for its
  own reported gains — a much simpler single tiny recursive network matches or beats it. This is directly
  relevant to the programme's own hierarchy question (P3): before attributing any future TanitAD result
  to "hierarchy," the TRM precedent says **explicitly test whether a same-parameter-budget FLAT recursive
  network matches it**, exactly the same falsification discipline the programme already applies via
  REF-B. Positively: TRM's tiny-recursive-refinement pattern (refine an answer + a latent through K
  steps with ONE small network) is itself a candidate mechanism for our tactical head's SELECTION step —
  iteratively refine a candidate/decision through a few recursive passes instead of one softmax.
- **Admissibility:** clean (pure architecture, no data-source implications).
- **Cost to try:** prototyping a TRM-style recursive-refinement tactical head (small network, K
  recursive steps refining the manoeuvre+trajectory choice) is **~6-10 eng-days**, ~1 GPU-day.
- **Cheapest discriminating experiment:** replace the tactical head's single-pass softmax with a TRM-style
  small network doing K=3-6 recursive refinement steps over the SAME candidate representations (same
  params budget, enforced like REF-B), and measure the selection gap. Outcome A: improves → recursive
  refinement (not raw capacity or hierarchy per se) is a cheap, general fix, AND is a caution to
  re-examine whether some of our own claimed "hierarchy" gains are actually "iteration" gains in
  disguise; Outcome B: no improvement → rules out recursive refinement cheaply, redirecting to the
  scoring-mechanism experiments in category C instead.

### Neural Circuit Policies / Liquid Networks (Closed-form Continuous-time, CfC)
- **What:** ODE-inspired but SOLVER-FREE ("closed-form") continuous-time recurrent units; extremely
  parameter-efficient (a full CfC lane-keeping controller uses **~4,000 parameters**), causal by
  construction, and originally demonstrated for exactly our domain — end-to-end steering.
- **Evidence:** PUBLISHED — *Closed-form Continuous-time Neural Networks*, arXiv 2106.13898 (Nature
  Machine Intelligence 2022) — foundational, 100×+ faster than ODE-solver-based counterparts; follow-on
  robust-flight-navigation OOD result (Science Robotics) shows strong OOD generalization for liquid
  networks specifically, a property directly relevant to our P4/OOD concerns. Maturity: **PROVEN**
  (published in Nature MI, flight-tested OOD).
- **Pain points:** **P6 (latency/edge deployment)** — 4K-parameter continuous-time controllers are an
  extreme point on the efficiency axis, useful specifically for a Jetson-Thor-class FALLBACK/monitor
  channel (H11) that must be cheap and always-on, rather than for the main world model; **P4** — the
  documented OOD robustness of liquid networks is a genuinely distinct mechanism from our imagination-
  error OOD signal and could serve as an independent, architecturally-different second OOD channel
  (redundant-channel safety-case value, same idea the programme already logged for ZipDepth in the
  2026-07-11 screening).
- **Admissibility:** clean.
- **Cost to try:** a CfC-based lightweight fallback/OOD monitor, running alongside (not instead of) the
  main world model, is **~5-8 eng-days**, **0 GPU beyond a tiny training run** given the ~4K-parameter
  scale.
- **Cheapest discriminating experiment:** train a minimal CfC controller as an INDEPENDENT redundant OOD/
  safety channel (predict next-frame heading/speed from raw sensor input, flag divergence from the main
  model's own action as a safety signal) and check whether its OOD-flagging DISAGREES informatively with
  the imagination-error monitor on any held-out scenario class. Outcome A: it flags cases imagination-
  error misses → adopt as a genuinely redundant (not merely duplicate) safety channel, strengthening the
  P8 safety story cheaply; Outcome B: perfectly correlated with existing monitor → no redundancy value
  added, deprioritize.

### Thousand Brains Project (Numenta / Monty)
- **What:** Cortical-column-inspired "learning modules," each maintaining an OBJECT-CENTRIC REFERENCE
  FRAME via grid-cell-like path integration, voting together across many semi-independent modules rather
  than one monolithic network.
- **Evidence:** PUBLISHED — *The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence*
  (2412.18354) + *Thousand Brains Theory 2.0* (arXiv 2507.05888, 2025) + active open-source `tbp.monty`
  framework; now an **independent non-profit**, partly Gates-Foundation-funded, running a "Meet Monty
  2026" onboarding series (confirms active 2026 development). Maturity: **SPECULATIVE-for-driving**
  (no automotive-scale application found; strongest evidence is at small sensorimotor/object-recognition
  scale).
- **Pain points:** **P3/P5** — the "many semi-independent modules voting, each with its own reference
  frame" idea is architecturally distinct from both a monolithic encoder AND our current strict
  4-brain hierarchy — it suggests a THIRD topology (parallel, voting modules rather than a strict
  cadence hierarchy) worth being aware of as a contrast case, but is far too immature/small-scale to
  adopt directly for driving.
- **Admissibility:** N/A (no working system to assess at our scale).
- **Cost/experiment:** **not recommended this cycle** — flag as a watch item only; the object-centric
  reference-frame idea is closer to a Phase-2+ representation-learning research question than a
  near-term experiment.

### Continuous Thought Machines (Sakana AI)
- **What:** Neurons carry their OWN per-neuron temporal processing (a short history of past inputs, not
  just current activation) and the network uses cross-neuron **synchronization** itself as a latent
  representation — i.e., timing/synchrony IS the code, not just activation magnitude. Adaptive "ticks"
  let the network spend more or less computation per input based on difficulty.
- **Evidence:** PUBLISHED — arXiv 2505.05522 (Sakana AI). Maturity: **PROMISING** (novel, open-sourced,
  one strong lab's result; not yet widely replicated).
- **Pain points:** **P9 (compute)** — adaptive per-input compute ("ticks") is directly relevant to a
  Thor-class latency budget: spend more compute on hard frames (e.g., a dense intersection), less on
  easy ones (empty highway), rather than the FIXED per-tick compute our cadence hierarchy currently
  implies; **P1** — using synchrony/timing as an explicit representation is a genuinely different
  computational primitive from anything else in this survey, worth a small exploratory read even though
  it is early-stage.
- **Admissibility:** clean (pure architecture).
- **Cost to try:** full adoption is a large architecture change (**ESTIMATED 15-20+ eng-days**) — NOT
  recommended as a near-term experiment given the priority order; the adaptive-compute-per-difficulty
  IDEA alone is cheaper to approximate (e.g., using imagination-error as a proxy for "how hard is this
  frame" to gate extra tactical-head compute) — **~5 eng-days** for that narrower version.
- **Cheapest discriminating experiment:** the narrow version — gate an EXTRA recursive refinement pass
  (see TRM above) on high imagination-error frames only, and measure whether compute-adaptive refinement
  beats fixed-compute refinement at MATCHED AVERAGE compute. Outcome A: yes → adaptive compute is a free
  efficiency win compatible with a Thor budget; Outcome B: no → fixed-cadence compute is fine, and the
  full CTM architecture is correctly left as a longer-horizon research bet, not a near-term item.

### Energy-Based Transformers (EBTs) — System 2 thinking via energy minimization
- **What:** Instead of a direct forward pass, assign an ENERGY to each (input, candidate-prediction)
  pair and predict by gradient-descending that energy until convergence — makes "think longer for harder
  inputs" (System 2) emerge from unsupervised learning, modality-agnostic.
- **Evidence:** PUBLISHED — arXiv 2507.02092 (Gladstone et al.). **Up to 35% higher scaling rate**
  (data/params/FLOPs/depth) vs standard Transformer++ during TRAINING; **+29% inference improvement from
  "thinking longer"** on language tasks; beats Diffusion Transformers on image denoising with FEWER
  forward passes. Maturity: **PROMISING** (strong single-paper result across two modalities, not yet
  widely replicated at scale).
- **Pain points:** **P1 — this is a genuinely different mechanism for "selection" than anything else in
  this survey**: instead of generating K candidates and scoring them (the entire category-C playbook),
  an energy-based head could score a SINGLE continuous trajectory space directly and gradient-descend to
  the best point, sidestepping the "generate diverse, then pick" pipeline (and its P1 failure mode)
  altogether. This is speculative for driving specifically but conceptually the most different idea in
  the whole survey, hence a genuine disruptive-bet candidate (see below).
- **Admissibility:** clean (energy computed from scene features + candidate, no classifier dependency
  implied).
- **Cost to try:** this is a genuinely new head TYPE, not a drop-in — **ESTIMATED 15-25 eng-days** for a
  first driving-scoped prototype (energy function over trajectory space, gradient-descent inference
  loop), plus **2-4 GPU-days**.
- **Cheapest discriminating experiment:** on a SMALL toy version first (not the full flagship) — train an
  energy-based scorer over trajectory space using existing candidates + ground truth as positive/negative
  examples, and check whether gradient-descending from a random trajectory init converges to something
  competitive with the current BEST candidate in the fan, without ever enumerating the fan. Outcome A:
  competitive → a fundamentally different, potentially more efficient selection paradigm is viable and
  worth the larger investment; Outcome B: does not converge well / energy landscape too non-convex at
  our data scale → informative negative result, keep candidate-based scoring (category C) as the
  approach and mark EBT-style selection as not yet ready for our data regime.

---

## 6. Ranked TOP-10 for TanitAD

Ranked by (expected P1/P2/P3 impact) × (admissibility) ÷ (cost), given "a few A40s" and the priority
order P1 (selection) > P2 (longitudinal) > P3 (hierarchy justification) established by the fact sheet.

| # | Idea | Pain point(s) | Cost (eng-days / A40-GPU-days) | Why it's ranked here |
|---|---|---|---|---|
| 1 | **SimLingo-style disentangled lat/lon output head** | P1, P2 | 3–5 / ~1 | Cheapest, most direct fix for the EXACT diagnosed defect (one softmax mixing lat+lon); vision-only, so trivially admissible; the pattern is already SOTA (85.94 DS, camera-only). |
| 2 | **ZTRS-style reward-driven (EPO) loss swap** on the existing tactical head | P1 (primarily), P2 | 0 new modules / ~1 retrain | Tests whether the LOSS FUNCTION (imitation vs reward) — not architecture — is the dominant cause of the 0.525→1.025 selection gap; if true, this is the single highest-value finding available this quarter. |
| 3 | **WoTE-style world-model-scored selection**, reusing our OWN predictor as the reward model | P1 | 5–8 / 0 (reuses existing predictor) | Directly implements our own "imagine-and-select" thesis the way the literature does it end-to-end; zero new GPU-days. |
| 4 | **GTRS/DriveSuprim-style coarse-to-fine, sensor-robust learned scorer** | P1 | 6–12 / 1–2 | Current NAVSIM v2/AAAI-2026 SOTA; GTRS specifically targets robustness under imperfect sensing — our actual situation (from-scratch, single camera), unlike most privileged-BEV SOTA. |
| 5 | **Basal-ganglia-style lateral-inhibition/competition layer** on top of any scorer above | P1 | 4–6 / 0 | Very cheap; separates "how good" from "how much to suppress duplicates," which a plain softmax conflates — complements #2–4, doesn't compete with them. |
| 6 | **GoalFlow-style predicted geometric goal point** ⚠️ GATED on admissibility verification | P1, P3 | 10–15 / 2–4 | Exactly the lever the programme's OWN binding rule names as preferred ("+4.7," predicted not supplied) — but GoalFlow's own goal-scorer inputs are UNVERIFIED for vision-only-ness; verify before building. |
| 7 | **π*0.6/RECAP-style offline advantage-conditioning** from the historical best-in-fan-vs-selected gap | P1 | 8–12 / ~0 (log-only) | Cheapest possible version of "learn from your own past selection mistakes," entirely from logs already on disk, no live correction loop needed. |
| 8 | **TRM-style recursive-refinement tactical head** + mandatory flat-matched-params control | P1, P3 (methodological) | 6–10 / ~1 | Cheap fix AND forces the same hierarchy-attribution discipline the programme already applies via REF-B — directly answers "is this gain from hierarchy or from iteration?" |
| 9 | **Dreamer-4-style RL-in-imagination** for the tactical head (policy-gradient against imagined rollouts) | P1, P7, P9 | 10–20 / a few | Highest ceiling in this list — the closest published analogue to our whole thesis, at industrial scale, with a genuine long-horizon (20,000+ action) credit-assignment result; correspondingly the most engineering-heavy item here. |
| 10 | **Active-inference surprise-minimization** folded into candidate scoring | P7 (feeds P1) | 3–5 / 0 | Reuses the ALREADY-FREE imagination-error signal (A9) as a selection criterion, not just an OOD monitor — principled, nearly free. |

**Not in the top 10 but flagged as important watch/escalation items:** NVIDIA Cosmos 3/Predict 2.5 as a
possible fix for the "cosmos loader" version drift + the missing map/junction/traffic-light scenario
classes (P3/P4) — escalate to whoever owns the DataEng cosmos integration; Alpamayo-R1's PhysicalAI-AV
training-data relationship — UNVERIFIED, escalate to a stream with working arXiv/HF fetch access;
Hydra-MDP's "factorised path×velocity" characterization — still tier B, still the programme's own #1
re-verification target, still blocked on fetch access from this container.

## 7. Three disruptive bets (high-risk / high-reward), each with kill criteria

### Bet 1 — Eliminate imitation learning from the tactical (and eventually operative) training objective, ZTRS-style
**The bet:** replace behaviour-cloning losses with a pure rule-based-reward policy-gradient objective
(ZTRS's EPO, or a scoped variant) across the tactical head, on the theory that BC is itself teaching the
model to imitate a HUMAN'S selection habits including their noise, rather than to select well against an
explicit, controllable notion of "good." This questions the training PARADIGM, not just the architecture.
**Why it's disruptive:** if it works, it could resolve P1 AND P2 simultaneously (rule-based reward can
encode speed-holding/distance-keeping directly, addressing the 88.7% longitudinal gap at the objective
level, not just the architecture level) with no new parameters.
**Kill criteria (pre-committed):** (a) if, at matched compute, the reward-only policy's SAFETY floor
(collision/off-road rate on held-out windows) is worse than the current imitation-trained baseline by
more than a pre-registered margin, kill it — reward misspecification in a domain this rich is a known
failure mode (CaRL and RAD both build in imitation-as-regularizer for exactly this reason, which is
itself evidence this risk is real, not hypothetical); (b) if the cheap discriminating experiment (§3,
ZTRS entry) shows NO improvement in the selection gap, do not proceed to the full retrain.

### Bet 2 — Train the tactical+operative loop by RL entirely inside our own world model (Dreamer-4-style)
**The bet:** stop supervising the tactical head at all; train it by policy gradient against imagined
rollouts scored by `taniteval` cost proxies, the way Dreamer 4 trains a Minecraft agent purely from
offline video plus imagination — no live environment needed, matching our own offline-corpus constraint.
**Why it's disruptive:** this is the single most industrially-validated version of "the world model is
not just a representation, it is a training environment" found in this entire survey (a real 20,000+
action credit-assignment result, 100× data efficiency vs the prior SOTA) — if it transfers, it could
make P1 AND P7 (imagination calibration, forced to matter once the policy is optimized against it) and
P9 (a single training loop instead of separate supervised heads) all move together.
**Kill criteria (pre-committed):** (a) reward hacking — if the trained policy achieves high IMAGINED
reward but its imagination-error AT THE STATES IT VISITS spikes (i.e., it has learned to exploit world-
model blind spots rather than to drive well), kill before any further scaling — this is the textbook
failure mode the world-model-RL literature itself warns about (see the Raw2Drive "world-model alignment"
bonus find in §2, whose whole premise is that this failure mode is common enough to need its own fix);
(b) if the cheap pilot (§1, Dreamer-4 entry: REINFORCE/GRPO on the tactical head against existing
imagined rollouts, no new world-model training) shows no improvement over the current softmax, do not
proceed to a full Dreamer-4-style retrain of the whole loop.

### Bet 3 — Replace generate-and-score entirely with an Energy-Based Transformer selection head
**The bet:** instead of generating K candidates and scoring them (every single item in category C), learn
an energy function over the full continuous trajectory space and select by gradient descent — potentially
sidestepping the entire "candidate fan is too narrow / scorer is too weak" dichotomy that frames every
other P1 fix in this report.
**Why it's disruptive:** it is the most CONCEPTUALLY different mechanism found in this entire survey —
not a variation on generate-and-score, a different paradigm entirely — with a real (non-driving) result
showing higher scaling rates and better test-time compute utilization than standard transformers.
**Kill criteria (pre-committed):** (a) if the small-scale toy prototype (§5, EBT entry: energy scorer
over existing candidates + GT as positive/negative examples, gradient-descent-from-random-init) does not
converge to a trajectory competitive with the best-in-fan candidate, kill before any larger investment —
energy landscapes are known to be hard to make well-behaved (non-convex, multiple local minima) and our
13-hour corpus may simply be too small to shape a good one; (b) do not invest ANY engineering time beyond
the toy prototype (a few days) until outcome (a) is decided — this bet's entire cost structure is
front-loaded into a cheap go/no-go gate by design.

## 8. Literature that CONTRADICTS the programme's current direction

1. **TRM's own headline finding directly undercuts a "hierarchy/brain-inspiration" framing on its
   OWN turf.** Jolicoeur-Martineau (TRM) explicitly demonstrates that HRM's biologically-inspired
   two-timescale hierarchy was NOT necessary for HRM's results — a single tiny non-hierarchical recursive
   network matches or beats it, and the author states this in so many words ("nothing to do with the
   human brain... does not require any hierarchy"). **This is a direct, load-bearing tension with H1**,
   not a minor caveat: it is a recent (Oct 2025), reproducible, falsifiable demonstration that apparent
   "hierarchy" gains can actually be "iterative refinement" gains wearing a hierarchy's clothes. The
   programme's own REF-B (flat E2E baseline) is exactly the right falsifier for THIS possibility applied
   to TanitAD — but TRM raises the bar: REF-B should be matched not just on parameters but on **compute
   spent per decision** (recursive refinement steps cost compute too), or a flat-but-iterative baseline
   could look like it "loses to hierarchy" purely because it wasn't given the same iteration budget.
2. **TOAD's finding (search discovers new trajectories; re-ranking does not) is in tension with a
   pure-better-scorer reading of our P1 fix.** Much of category C (Hydra-MDP, GTRS, DriveSuprim) is
   architecturally "keep the fan fixed, build a better scorer" — TOAD's own ablation says the GAINS in
   that family of methods came from finding NEW candidates outside the original fan, not from ranking the
   existing ones better. If this generalizes, several of the TOP-10 items above (specifically #4, #6 in
   spirit) may under-deliver unless paired with iPad/TOAD-style test-time search (#already flagged as a
   cheap experiment in §3) — this is a genuine reason NOT to treat "install a better scorer" as
   sufficient on its own before that experiment is run.
3. **The "Agentic AI" hard-modular-swarm philosophy (Uber/Autobrains Munich robotaxi program, per the
   programme's own `External Anaysis.md`) is philosophically opposed to a jointly-trained, differentiable
   4-brain hierarchy for the exact reason the programme cares about most: certification.** That
   programme's explicit argument is that SEPARATELY-trained, deterministically-composed single-purpose
   agents are easier to certify under exactly the WP.29-style regulation the programme's own H11 targets,
   BECAUSE errors can be isolated to one agent instead of diffusing through a jointly-trained
   differentiable stack. This does not mean the swarm approach is right — but it is a real, currently-
   deployed (Munich pilot) counter-architecture whose entire selling point is the safety-case property
   our own end-to-end-differentiable 4-brain design does NOT structurally have, and the programme's H11
   monitoring story should explicitly address why layered monitors on a joint model are an adequate
   substitute for hard module isolation, rather than leaving the comparison unaddressed.
4. **CaRL's finding that reward SIMPLICITY beats complex shaping (at massive sample counts) is in mild
   tension with our own multi-term energy design** (`E_total = αE_str + βE_tac + γE_op`, multiple loss
   terms per gate). CaRL's regime (300–500M samples, 8-GPU node) is far from ours (13h, a few A40s), so
   this does not directly transfer, but it is a reason to periodically ask whether cost/loss-term count is
   creeping up for reasons of engineering convenience rather than measured necessity — worth a cheap
   ablation next time a new loss term is proposed.

## 9. Deliverable manifest

| Artifact | Location | Notes |
|---|---|---|
| This stream report | `repo:Project Steering/Reviews/2026-09-25-programme-review/streams/W1_frontier_worldmodels_planning_brain.md` | Staged (`git add`), not committed, per operating rules. Single file, no other artifacts produced. |
| Final summary to orchestrator | delivered via `SubagentHandback` | Compact TOP-10 + 3 bets + contradictions + this manifest, ≤1200 words. |

No code, data, or pod artifacts were produced by this stream (research-only brief). No sibling files
were read or modified beyond the read-only skim listed in §0. All searches/fetches are logged in the
Method Note (§ above §1) for auditability.

**Escalations for the orchestrator / other streams (not actioned here, per this stream's research-only
scope):**
1. Re-verify Hydra-MDP's "factorised path×velocity vocabulary" claim from primary text — needs working
   arXiv/HF fetch access, which this container's egress proxy blocks (only `github.com` was reachable).
2. Resolve whether Alpamayo-R1 trains on PhysicalAI-AV specifically, or only shares its ecosystem —
   needs the same fetch access; the answer changes how directly AR1's numbers function as a competitive
   bar for TanitAD.
3. Confirm the version of NVIDIA Cosmos currently wired into `stack/`'s "cosmos loader" (per the
   2026-07-08 screening digest) against the newly-released Cosmos 3 / Predict 2.5 / Reason 2 — likely 1-2
   major versions behind, and Predict 2.5's longer long-tail generation window may resolve the
   previously-logged "Cosmos T=39 temporal semantics" blocker.
4. GoalFlow's goal-point-scorer input list needs a primary-source read before ANY goal-point work is
   authorized, per the binding admissibility rule — flagged as gated, not cleared, in the TOP-10 table.

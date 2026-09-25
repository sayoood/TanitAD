# Stream W1 — Frontier research: world models, imagination, planning/selection, driving foundation models, brain-inspired architectures

**Status: IN PROGRESS — banking incrementally.** Orchestrator: whole-programme review, 2026-09-25.
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

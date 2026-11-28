# LAB RUN 001 — Opponent Analysis — literature pass + ideation

`TanitAD Research Lab, daily run 001. Run label 2026-08-22 (Master Mind trigger);
executed 2026-08-23 wall-clock. Literature only, no GPU. Agenda targets:
RESEARCH_AGENDA.md Field 4, items 1 (teardown series), 3 (capability-claim
verification), 4 (trend detection).`

## Findings

### F4.1 — NVIDIA Alpamayo-R1 / "Alpamayo 1": the first fully inspectable big-league opponent ⭐ banked
- **Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable
  Autonomous Driving in the Long Tail** — arXiv **2511.00088** (Oct 2025, rev
  7 Jan 2026; renamed *Alpamayo 1* at CES 2026). **PUBLISHED, BANKED (library
  key `2511.00088`, tag `opponent-alpamayo`)**. Weights (10B) on Hugging Face,
  inference code on GitHub (`NVlabs/alpamayo`) — licence class to be stated by
  the teardown before any use.
- **Finding:** a reasoning VLA = **Cosmos-Reason** VLM backbone + a
  **diffusion trajectory decoder** ("dynamically feasible trajectories in real
  time"), trained on a **Chain-of-Causation** dataset (hybrid auto-labelling +
  human-in-the-loop, causally linked reasoning traces) in three stages
  (SFT → RL for consistency → RL for reasoning quality). Claims: **+12 %**
  planning accuracy on hard cases vs trajectory-only, **−35 %** close-encounter
  rate in closed-loop sim, +45 % reasoning quality from RL post-training, +37 %
  reasoning–action consistency, **99 ms** deployment latency, gains consistent
  from 0.5B to 7B.
- **Impact:** architecture, data recipe AND weights are open — the teardown
  series (agenda item 1) can be *measured*, not inferred; and its 0.5B→7B
  scaling statement is the closest published curve to our sub-300M thesis.
- **What it would change:** the leaderboard gets AR1 rows marked
  *claimed-not-reproduced*; the teardown's first question is what fraction of
  the gain survives without the reasoning tokens (H-OPP-1 below).

### F4.2 — Wayve GAIA-3 / GAIA-4: the world model moves from data generator to SAFETY EVALUATOR
- **GAIA-3** (15B — 2× GAIA-2, video tokenizer also 2×, now offered to third
  parties for validation) and **GAIA-4** ("Multimodal World Models Powering
  Closed-Loop Simulation", Wayve, 2026). **PUBLISHED (company technical blog
  posts, verified); NOT banked — no arXiv primary found today; secondary.**
- **Finding (GAIA-4):** the AI Driver's own actions drive the simulation
  forward — position, viewpoint, sensor stream — so the model's decisions
  change what it perceives next ("world-on-rails": background road users stay
  faithful to the recording, only ego varies); first driving simulator to
  generate coherent **radar + camera** with realistic Doppler; used for
  counterfactual replay of operator interventions and scenario-library testing
  at scale.
- **Impact:** the field's largest world-model shop has restated the WM's
  purpose as *measuring end-to-end safety* — exactly the TanitSim thesis, at
  15B. Our differentiator cannot be "a GAIA"; it must be the radically cheaper
  proof (NuRec/gsplat at 492 FPS on Thor + the pseudo-simulation protocol, see
  Benchmarks & Evals run 001).
- **What it would change:** TanitSim's SPEC positions against GAIA-4 on
  cost-per-counterfactual, and "world-on-rails" (log-faithful others, free ego)
  is adopted as the first TanitSim fidelity tier.

### F4.3 — Trend detection (from F4.1, F4.2 and the self-play line)
Three convergences in 2026: (a) **reasoning VLAs with trajectory decoders**
(AR1; DriveVLM-class), (b) **world-model-in-the-eval-loop** (GAIA-4; NAVSIM v2
pseudo-simulation), (c) **RL post-training / self-play** (AR1's RL stages;
*Scaling Self-Play for End-to-End Driving*, arXiv 2606.19641 — policies
competitive on HUGSIM/NAVSIM-v2 with NO human trajectories, from a 50k-steps/s
simplified renderer). **Contrarian opening for sub-300M:** none of the three is
scale-bound a priori — each has a small-scale proof design, which is the Lab's
mandate (charter: "prove the concept at SMALL scale").

## Ideation — our own hypothesis

**H-OPP-1 (OPEN, proposed).** *AR1's hard-case planning gain is mostly
decoder-and-data, not reasoning.* Most of Alpamayo-R1's +12 % hard-case planning
gain is attributable to the diffusion trajectory decoder and the CoC data
distribution rather than to the reasoning tokens consumed at inference: an AR1
run with reasoning generation suppressed (empty/minimal chain before the
trajectory tokens) will retain the majority of the open-loop hard-case gain.

- **Cheapest discriminating check (agenda item 3):** weights are open. Run
  AR1-10B on a small hard-case slice with and without reasoning generation;
  compare trajectory accuracy. ⚠️ 10B does not fit the 4060 in FP16 — needs
  INT4 (itself a confound; report it) or a metered HF GPU burst → **PI consent
  required before any metered run; spec only today.** Licence class of weights
  to be stated first.
- **Outcome A (gain mostly retained):** reasoning is a data-labelling lever
  more than an inference lever — strengthens our no-VLM-at-inference,
  sub-300M position; CoC-style *labels* become a DataFlyWheel item.
- **Outcome B (gain collapses without reasoning):** inference-time reasoning is
  load-bearing; Field 2's "test-time compute" scan gets a concrete target.

## Transfer note (charter §7)

→ **TanitAD_EvalFlyWheel:** add AR1 (claimed) and GAIA-4 (no public metric) to
the leaderboard as claimed-not-reproduced rows; the AR1 open weights make a
reproduced row feasible once compute is consented. → **Master Mind:** teardown
series starts with AR1 (weekly cadence per agenda); TanitSim SPEC to cite
GAIA-4's world-on-rails as tier 1. Accept/reject with reason requested.

## Evidence discipline

AR1 numbers are the authors' claims (PUBLISHED, abstract-verified) and are
labelled *claimed* wherever they are reused; GAIA figures are from company
posts (secondary). The banked key is the only registry-admissible citation.

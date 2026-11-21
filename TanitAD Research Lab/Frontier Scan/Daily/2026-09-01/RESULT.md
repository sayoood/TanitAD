<title>Frontier Scan 2026-09-01 — Waymo's own stack argues our thesis, and three of our records were wrong</title>

# FRONTIER SCAN — 2026-09-01

`TanitAD Research Lab · daily pass under DAILY_RESEARCH_CHARTER.md + v2 AMENDMENT.`
`Band D x2 primaries · Band A x2 · Band B x3 (B2, B12, B13) · Band C sweep · 16 queries, 5 empties named.`
`⚠️ This pass is the FRONTIER SCAN ONLY. Today's four domain packages already existed at trigger time and were NOT redone.`

---

## THE HEADLINE

⭐⭐⭐ **We read Waymo's two technical primaries, and their production architecture is a fast/slow
hierarchy with learned-embedding interfaces and a large-teacher → small-student deployment split.
Stated plainly: our largest opponent's shipped stack is closer to TanitAD's thesis than to the
"pure end-to-end" story that has been attributed to it — including by our own register.**

⛔ **And the pass corrected THREE of our own records** (D-1's numbers, D-2's convergence framing,
B2's "25 % budget" claim). All three were `RELAYED` entries carried as though settled.
**Per V-6, the contradictions are the value of the day; they are reported before the confirmations.**

---

## F1 ⭐⭐⭐ — The Waymo World Model is a SIMULATOR, not a driver — and it concedes our hardest problem

`PUBLISHED-BLOG · waymo.com/blog/2026/02/... · retrieved 2026-09-01 · Band D · D-3 debt DISCHARGED`

The highest-value unread document known to the programme is now read.

1. **It is a simulator.** *"the component that is responsible for generating hyper-realistic
   simulated environments"* — one of *"three key pillars of our approach to demonstrably safe AI"*.
   It is **not** claimed to drive the car.
2. **It is Genie 3 post-trained.** *"built upon Genie 3—Google DeepMind's most advanced
   general-purpose world model"*, *"adapted for the rigors of the driving domain"* via
   *"specialized post-training"*.
3. ⭐ **THE CONCESSION** (amendment §7.2 — look hardest for the sentence that costs the author
   something): *"longer scenes...is harder to do because the longer the simulation, the tougher it
   is to compute and maintain stable quality."*
4. ⛔ **ZERO ablations, zero metrics.** The only number in the document is the 200 M-mile exposure
   figure, which is not about the world model at all.

| dim | analysis |
|---|---|
| **RELEVANCE** | ⭐⭐⭐ — direct. It is the reference point for what a frontier world model is *for*. |
| **CONSEQUENCE** | Waymo's WM and TanitAD's WM are **different products**: theirs renders futures for training/testing, ours predicts futures inside the driving loop. ⇒ Their announcement does **not** refute our thesis, and we should stop treating it as competitive pressure on the same axis. |
| **COMBINATION** | Their concession is **our measured problem**. TanitAD's drift (0.4531 → 0.36–0.40 under EMA, MM-E1) and today's own long-horizon-BPTT-stability package are the same failure. **With Genie 3 and Google-scale compute, Waymo has not solved long-horizon rollout stability either.** This is the B13 geometry-ceiling reframing repeating: *a property of the field, not our bug.* |
| **CHANCES / RISKS** | ⭐ Upside: it de-risks our roadmap — we are not behind on a solved problem. ⚠️ Risk: a *simulator*-grade WM has no latency budget, so their quality bar is reachable in ways ours is not; do not read their fidelity as a target for an in-loop predictor. |
| **EXPERIMENT** | *(proposed, unranked)* Score our own rollouts on **WorldRoamBench's segment-based drift metric** (F7) instead of start-vs-end drift. **Discriminating outcome committed in advance:** if our drift is *monotone*, start-vs-end is adequate and the instrument stands; if it is **non-monotone with mid-sequence collapse**, every drift number we hold is a lower bound and the instrument must change. |

## F2 ⭐⭐⭐ — Waymo's production architecture is a hierarchy, and it deploys SMALL

`PUBLISHED-BLOG · waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/ · Band D · D-2 SETTLED`

> *"leverages the full expressibility of learned embeddings as a rich interface between model
> components and supports full end-to-end signal backpropagation"*

- **Architecture:** a **Sensor Fusion Encoder** (rapid reactions) + a **Driving VLM** (semantic
  reasoning), both feeding a **World Decoder**. Three pillars: **Driver · Simulator · Critic**.
- ⭐ **Distillation at PRIMARY, no longer relayed:** *"these Teacher models are too big to run on
  vehicles for real-time decision making"* — teachers distilled into students for onboard use.
- **Claims:** *">ten-fold reduction in crashes with serious injuries compared to human drivers"*,
  *">100 million fully autonomous miles"*. ⛔ **No ablations.**
- **No limitations admitted anywhere in the document** — worth noting beside F1, which did concede.

| dim | analysis |
|---|---|
| **RELEVANCE** | ⭐⭐⭐ — it settles register debt D-2 and re-opens W-3. |
| **CONSEQUENCE** | ⛔ **D-2's framing is REFUTED as stated.** *"Waymo's model is end-to-end just like Tesla and Wayve"* is not what the primary says. It describes an explicitly **hybrid** design — modular components, learned-embedding interfaces, end-to-end backprop — which their own text presents as better than pure-E2E **and** better than modular. ⇒ The register row is corrected, and W-1's framing weakens with it. |
| **COMBINATION** | ⭐⭐ This is **TanitAD's hierarchy thesis in an opponent's production stack**: a fast reactive path, a slow semantic path, a shared decoder, differentiable interfaces between named components. Combined with the distillation concession it is also **external support for sub-300M deployment** — they *train* large and *deploy* small, exactly the W-3 self-undercut, now at primary rather than relayed. |
| **CHANCES / RISKS** | ⭐ Upside: the programme's two most-questioned commitments (inspectable hierarchy; small deployed model) are both instantiated by the largest operator in the field. ⚠️ Risk — ⛔ **this is advocacy, not evidence.** There is no ablation showing the hybrid beats either alternative. Adopting it *because Waymo does it* would be the exact `INHERITED` failure the operating standard bans. It raises our prior; it decides no GPU-day. |
| **EXPERIMENT** | *(proposed, unranked)* Our v7 already has the seam. Register a **three-way at matched params on the v7-tiny ladder**: differentiable-embedding interface (Waymo-shaped) vs our current interface vs a pure-E2E control. **Committed outcome:** if the embedding-interface arm does not beat the pure-E2E control on rank *and* decodability, the architectural convergence is cosmetic and we say so. |

## F3 ⛔ CORRECTION — the B2 debt's "25 % budget" claim does not exist

`PUBLISHED lib 2601.22156 · abstract read · B2 debt DISCHARGED`

`2601.22156` is **"Hybrid Linear Attention Done Right: Efficient Distillation and Effective
Architectures for Extremely Long Contexts"** (Chen et al., 2026-01-29) — proposing **HypeNet** with a
**HyPE** position encoding.

⛔ **Its efficiency claim is about TRAINING DATA, not compute budget.** The conversion pipeline needs
*"just 2.3B tokens, less than 0.01% of their pre-training data"* against *">10B tokens"* for prior
distillation methods. **There is no 25 % compute/memory figure.** `TRACKS.md` and injected row **I-2**
both carry the 25 % number as `RELAYED`; it is now **UNSUPPORTED-AS-STATED** and is struck.

⭐ **But the paper is MORE useful than the claim it was filed under, in a different direction.** I-2
was scoped as *"train a Mamba-class predictor at matched params and compare"*. This paper says the
cheaper move is **conversion**: take an already-trained transformer and distil it into a hybrid
linear-attention model for a very small token budget. **For a programme with a trained v7 predictor
and a fixed fleet, converting is a fundamentally cheaper experiment than retraining.** I-2's method
should be re-scoped accordingly — a change to a live row, not a note.

## F4 ⚠️ TENSION WITH A LIVE P0 ROW — variable-length latent WMs argue against composed h=1

`PUBLISHED lib 2606.21775 · abstract-only, DECLARED · B12 debt DISCHARGED`

**"Beyond the Next Step: Variable-Length Latent World Models for Long-Horizon Planning"** (Du, Zhang,
Wang, Wang; 2026-06-19). Verbatim:

> *"existing latent world models typically rely on one-step prediction and must be recursively rolled
> out for long-horizon planning, which leads to compounding errors and a mismatch between training
> objectives and downstream planning tasks."*

Claim: **+13 % average over LeWM**, larger gains on tasks needing extended planning.

⛔ **This directly tensions `LAB_BACKLOG` row 5 (P0, Arch): "drop the h≥2 predictor heads for composed
h=1".** Our evidence for that row (H-PROOF-6d: heads never trained, 1e-3 init; H-PROOF-7: rolled h=1
predicts to 6 steps) shows the heads are **vestigial under the objectives we currently run** — which
is *not* the same as *"multi-step prediction is the wrong design"*. VLWM argues the opposite
direction and reports a number for it.

⚠️ **These are compatible, and that is the point:** *vestigial-because-untrained* ≠ *useless in
principle*. **Dropping the heads is cheap; re-adding them after the v7 geometry is frozen is not.**
The Lab does not overrule a P0 row — it escalates, with a concrete cheap ordering: run the
composed-h=1 simplification, but **keep the head slots in the config** so the VLWM arm stays
reachable. `ESTIMATED` cost of keeping them: near zero (unused parameters at 1e-3 init).

## F5 ⛔ CORRECTION — D-1's numbers do not support what they were cited for

`PUBLISHED lib 2505.06113 · abstract read · Band D · D-1 debt DISCHARGED`

The register carried **"85.1 % / 92.1 % camera-vs-lidar parity figures"** as *"the most
decision-relevant numbers in the W-1 debate"*, `RELAYED`. The primary says:

> *"achieving up to 85% road segmentation accuracy and 85-90% vehicle detection rates when compared
> against LiDAR ground truth, with average positional errors limited to 1.2 meters"*

Four defects, each independently disqualifying for the use it was put to:

1. ⛔ **There is no camera-vs-lidar comparison in this paper.** LiDAR is the **ground truth**, not a
   competing arm. It measures camera-vs-lidar **agreement**, and agreement with a reference cannot
   establish **parity of capability**. *(Same family as our own `df`/scope traps: a true number quoted
   outside its scope.)*
2. ⛔ **The relayed version dropped the 1.2 m positional error** — the one number that decides the
   question. Our deployed v1 is **0.452 m** `fwd_ade` (MODEL_REGISTRY, `flagship4b-speedjerk-30k`).
   A perception front-end with 1.2 m average positional error cannot support planning at that scale.
3. **"92.1 %" does not appear in the abstract at all.**
4. **Single author, no named venue, not peer-reviewed, no ablation reported.**

⇒ **W-1 stays `CONTESTED`, but for the opposite reason than recorded:** we no longer hold the
counter-evidence we thought we held. **V-5 applies exactly** — the unit/scope error was the
corruption path.

## F6 ⭐⭐ MEASURED — banking has been substituting for reading

`MEASURED (this pass, kb_add.py output) · process finding`

**5 of the 6 primaries this pass needed were ALREADY BANKED.** Only WorldRoamBench (`2606.31672`) was
new. On 2026-08-31 the same ratio was **2 of 10**.

⛔ **The Library holds 310 entries / 2,115.6 MB, and today's four highest-value findings all came from
papers we already had and had not read.** Every one of B2/B12/B13's debts was a *banked-but-unread*
row. **The `kb_add` ritual has been standing in for the reading, and the debt list is the receipt.**
V-1 said *"a re-find is a free finding"*; the sharper statement after two passes is:

> ⭐ **A banked primary with no five-dimension analysis is not an asset, it is a debt with a hash.**

⇒ Proposed as a backlog row: `TRACKS.md` should carry a **banked-but-unread count per track**, so the
debt is visible without a full pass to rediscover it.

## F7 ⭐⭐ — An independent line arrives at our action-echo finding

`PUBLISHED lib 2606.31672 · abstract-only, DECLARED · A5 · banked this pass (the only new bank)`

**WorldRoamBench** (Xu et al., 2026-06-30). Verbatim: *"existing benchmarks evaluate action following
only at trajectory level and ignore memory and interaction physics"*; it introduces a **per-frame
action metric** *"exposing failures hidden by trajectory"* and a **segment-based drift metric**
*"capturing non-monotonic mid-sequence collapse missed by start-vs-end comparisons"*. Over
**10+ models: *"None reliably satisfies all dimensions."***

| dim | analysis |
|---|---|
| **RELEVANCE** | ⭐⭐⭐ — A5 has had no dedicated DEEP since the charter began, and this lands on `EVAL_DOCTRINE`'s T0/T1 split. |
| **CONSEQUENCE** | *"failures hidden by trajectory[-level metrics]"* is **our action-echo finding, found independently**: open-loop S-curve reproduction 97.9 % vs hold-action 0.0 % vs closed-loop ~5 % (MEASURED, `EVAL_DOCTRINE` §1.12). A further independent line now agrees that trajectory-level open-loop scoring conceals action-following failure. |
| **COMBINATION** | It supplies **two instruments we lack**: a per-frame action-fidelity metric, and a segment-based drift metric. Our drift is currently read start-vs-end — which by their construction **cannot see mid-sequence collapse**. |
| **CHANCES / RISKS** | ⭐ Upside: an external, citable instrument for the tactical/lateral families rather than a bespoke one. ⚠️ Risk: it is built for *open-world interactive* world models, not driving; the physics/memory dimensions may not port. **Abstract-only — no degradation curves or per-model numbers were readable, so nothing here may decide a GPU-day yet.** |
| **EXPERIMENT** | *(proposed, unranked)* Port **only** the segment-based drift metric to `taniteval` and re-score the banked v7 rollout dumps — **0 GPU**, the dumps exist. **Committed outcome** as in F1. |

## F8 ⭐ — The UNECE online-learning ban is unfound at two probes; I-3's real constraint is different

`Band C / C3 · two independent probes · primary 403-BLOCKED and UNREAD`

Injected row **I-3** carries a `CONSTRAINT-TO-VERIFY`: UNECE GRVA *"reportedly discussed the ban of
online learning"* (RELAYED, trade summary). Two probes today:

- GRVA **adopted a draft ADS regulation at its 19–23 Jan 2026 session**, submitted to WP.29 for the
  23–26 June 2026 session. `PUBLISHED-BLOG`.
- ⛔ **Neither probe found any online-learning, in-vehicle-learning, or post-deployment-model-update
  prohibition.** The second source states explicitly that these are **not mentioned**.
- ⚠️ **The primary GTR text returned HTTP 403 and remains UNREAD.** ⇒ The correct status is
  **"unsupported at two probes"**, *not* "refuted". Absence at two locations is stronger than at one,
  but the document that would settle it has not been read.

⭐ **What the probes DID find is a real constraint the relayed claim obscured:** the regulation
requires **In-Service Monitoring and Reporting (ISMR)** and a **Data Storage System for Automated
Driving (DSSAD)**. These bind on *reporting and traceability of deployed behaviour* — which an
online-adapting encoder makes materially harder to satisfy, without being banned.

⇒ **I-3's in-vehicle path is NOT closed.** It is re-scoped: the binding question is
*auditability under ISMR/DSSAD*, not legality. **The GRVA primary stays an open debt.**

## F9 — Band C sweep: JetPack 7.2.1, and a documented-vs-measured conflict on Thor

`PUBLISHED-RELEASE-NOTE / RELAYED · resolves empty E3 from 2026-08-31`

- **JetPack 7.2.1 released 2026-08-12**; a Jetson T3000 emulator for AGX Thor shipped with it.
- **TensorRT support matrix: Jetson AGX Thor = compute capability 11.0**; **FP4 requires CC ≥ 10.0**,
  **FP8 requires CC ≥ 8.9**. Thor's Transformer Engine *"dynamically switches between FP4 and FP8 at
  runtime"*. T3000: 865 FP4 TFLOPS.
- ⛔ **This CONFLICTS with our own MEASURED finding** (D-B1-GATE, backlog row 1): TRT on Thor `sm_110`
  **silently falls back to FP32** under FP8/FP4 flags (issue #4590 re-verified OPEN). The vendor
  matrix says supported; our measurement says silently unsupported.
- ⚠️ **The measurement wins, and the conflict is exactly why `quant_gate.py` is P0.** A support-matrix
  row is `PUBLISHED-RELEASE-NOTE` and **may not overturn a MEASURED fallback.** Worth re-testing on
  JetPack 7.2.1 specifically, since our measurement predates it.

## F10 — B13: SparseOcc++ discharged, but its load-bearing question is still open

`PUBLISHED lib 2607.04732 · abstract-only, DECLARED`

**SparseOcc++** (Tang et al., 2026-07-06) decouples scene completion (signed-distance regression on
sparse anchor voxels) from semantic segmentation: **+2.3 IoU and 3.9x faster than SparseOcc on
nuScenes**, **5.9x over OccFormer on SemanticKITTI**.

⛔ **The rotation asked this paper one question — "establish supervision requirements before any
design" — and the abstract does not answer it.** Whether it needs dense occupancy labels, lidar, or
is camera-only is **not stated**. Since PhysicalAI-AV has **no map/lane-graph/occupancy labels** and
our only 3D agent source is `obstacle.offline` (10 dynamic-agent classes, 87,481 cuboids), the
supervision requirement is the *only* thing that decides whether this is reachable for us at all.
⇒ **Full-text read carried forward as a debt. No design work until it is answered.**

## F11 — A3 second probe: E1 narrows but does not close

`Band A3 · E1's second probe, in a different phrasing`

Frozen-DINOv3 as a universal encoder is **strong and general**: SOTA dense prediction with a frozen
backbone (**COCO detection mAP 66.1**, **ADE20k mIoU 63.0**), plus competitive frozen performance in
robotic manipulation. ⛔ **But DINOv3 x BEV-driving specifically returned nothing at a second probe.**

⚠️ **Read against our own history this is informative, not merely empty.** Our frozen-encoder ceiling
(REF-A: 2.14 m plateau, speed R² 0.61) has been treated as an encoder-quality verdict. The general
result says frozen SSL features are *not* the weak link for dense prediction — which points back at
the **v6 readout geometry** (4 azimuth bins over 120°, 2.1–7.8x too coarse) as the binding
constraint, consistent with W-7 and the B13 reframing. **E1 stays open at two probes.**

---

## Opponent strengths recorded (amendment §7.2 — a Band-D package listing none is INCOMPLETE)

1. **Waymo's simulation stack is a genuine structural advantage.** A Genie-3-derived simulator plus a
   Critic plus RL-in-sim is a closed-loop training loop we do not have and cannot cheaply build.
2. **Their closed-loop evaluation culture is ahead of ours** and is stated as architecture, not
   aspiration (Driver / Simulator / Critic *"fueled by the same underlying AI"*).
3. **They have solved the compression problem in production** — teacher→student distillation running
   on-vehicle in a commercial fleet. We have the thesis; they have the deployment.
4. **>100 M autonomous miles and a claimed >10x serious-injury-crash reduction.** Unablated, but an
   exposure record no research programme can match.

## What this changes for TanitAD — 3 recommendations (cap respected)

1. ⛔ **Correct the register and I-2 in the same turn.** D-1's numbers, D-2's convergence framing and
   B2's 25 % claim are all wrong as recorded. *(Done this turn — register appended, backlog updated.)*
2. ⭐ **Re-scope I-2 from "train a Mamba-class predictor" to "CONVERT the trained v7 predictor"**
   (F3). Same question, far cheaper, and it is the method the literature actually supports.
3. ⚠️ **Do not drop the h≥2 head slots while executing backlog row 5** (F4). Keep them at 1e-3 init so
   the VLWM direction stays reachable; the simplification is cheap to do and expensive to undo.

## Load-bearing items this pass (V-2)

**Full primaries read:** the two Waymo documents (F1, F2) — they carried every doctrinal finding.
**Abstract-only, declared and not decision-grade:** `2606.31672`, `2606.21775`, `2607.04732`,
`2603.09086`, `2505.06113`.

## Completeness self-report (charter §6, fail-loudly clause)

| requirement | status |
|---|---|
| Band D adjudicated | ✅ 2 primaries, 5 claims, counter-search recorded per claim |
| ≥2 full-text deep reads | ✅ both Waymo primaries |
| Band A | ⚠️ **PARTIAL** — A1/A5 covered (`2603.09086`, `2606.31672`), A3 probed. ⛔ **A2 and A4 SCANNED-ONLY, no DEEP this pass.** |
| ≥3 Band B deep-reads | ✅ B2, B12, B13 (all three were standing debts) |
| Band C sweep | ✅ C1/C2/C3 — E3 resolved, C3 second probe |
| Ledger appends | ✅ A1, A5(new), B2, B12, B13, C2(new), C3 |
| Every empty named | ✅ 5 empties (E1 carried, E5, E6 new) |
| Debts discharged | ✅ D-1, D-2, D-3, B2, B12, B13 |
| **New debts opened** | ⛔ **3** — UNECE GTR primary (403), SparseOcc++ supervision, A2/A4 DEEP |

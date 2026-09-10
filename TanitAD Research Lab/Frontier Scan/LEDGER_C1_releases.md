<title>LEDGER C1 — frontier-lab and AV-company releases</title>

# LEDGER C1 — Releases, model families, announcements

⛔ **APPEND-ONLY.** Band-C items carry `PUBLISHED-BLOG` / `PUBLISHED-RELEASE-NOTE` / `RELAYED` —
never bare `PUBLISHED` — and **may never decide a GPU-day alone**. Vendor cards are the primary;
news summaries about them are secondary and are verified against the card before use.

---

## Entry 2026-08-31-01 — ⭐⭐⭐ NVIDIA Alpamayo: an open reasoning-VLA family trained on OUR corpus

**Announced 2026-01-05 at CES** (NVIDIA newsroom, fetched today).
⚠️ **A search summary dated this "August 10, 2026" — off by seven months.** The primary corrects it.
Recorded as a live demonstration of why Band-C claims are verified against the primary.

**Why this matters more than a normal release:** `AlpaSim`, which this programme already runs, is
**part of this release** — and Alpamayo's training data includes
**`nvidia/PhysicalAI-Autonomous-Vehicles`**, the corpus family we train on. We adopted the simulator
and never registered the model family shipped beside it.

| | Alpamayo-1.5-10B | Alpamayo2-Super |
|---|---|---|
| architecture | Cosmos-Reason2 **8.2 B** backbone + **2.3 B** action expert; *"diffusion-based trajectory decoder"* | **32 B** VLM backbone + **2.3 B** action expert |
| licence | **OpenMDW-1.1** — *"ready for non-commercial use. Commercial licensing available upon request."* | **OpenMDW-1.1**, *"permits commercial deployment globally"*; source Apache-2.0 |
| inputs | image/video (multi-camera, multi-timestep) · text — *"user commands and navigation guidance"* · **egomotion history** | + 3D translation, 9D rotation |
| training data | **`PhysicalAI-Autonomous-Vehicles`** + **`-NuRec`** + 16 named public sets + internal; **3.0 M** Chain-of-Causation traces; **RL post-trained** | ~**115 000 h** video; **3.7 M** CoC traces |
| benchmarks | LingoQA 74.2 · AlpaSim 1.37 ± 0.10 · minADE₆@6.4 s 0.916 m | LingoQA 79.2 · AlpaSim 1.50 ± 0.13 · minADE₆@6.4 s 0.911 m |

**Also in the family (not yet examined):** Alpamayo 1 (10 B, Chain-of-Causation trajectory
prediction, `NVlabs/alpamayo`), Alpamayo 1.5 (`NVlabs/alpamayo1.5`), 1 700+ h Physical AI open
driving dataset.

### ⚠️ Two open verification items

| # | item | status |
|---|---|---|
| **V-1** | The two cards state **different commercial terms under the same licence name** (OpenMDW-1.1). | Immaterial for us — research use is sanctioned by the PI's binding rule — **material the moment anything ships.** |
| **V-2** | *"without restrictions on commercial use"* appeared in a **secondary** (news/aggregator), contradicting the 1.5 card's own *"non-commercial use"*. | ⛔ **RELAYED. Do not repeat.** The card is the primary. |

### What it changes for TanitAD

1. ⛔ **Inference-input admissibility:** Alpamayo consumes **egomotion history at inference**. Under
   the binding **vision-only** rule an Alpamayo-style ego-input arm is **inadmissible as our
   deployable arm**. Usable as an **opponent** and as an **offline labeller** (where ego is
   explicitly permitted), never as our inference recipe.
2. **Deployment:** 10.5 B / 34.3 B are **orders outside Thor's 100 ms loop** — today's own
   measurement puts a 37.8 M model at 203 ms for a 60-step rollout. Reference and teacher, never a
   deployment target.
3. **Labels:** their 3.0 M CoC traces are themselves *"VLM-generated"* / *"VLM-based auto-labeling"* —
   the same untimed-CoT provenance problem `D-LABEL-GT` is stuck on. Adopting them imports their
   errors as ground truth.
4. **Command channel:** *"navigation guidance"* as free text is a **declared instruction**, not
   recoverable from the driven path — the property our Data package prized in L2D's `turn_signal`.

`Next in this track: enumerate Alpamayo 1 / 1.5 / 2 cards fully; determine whether their eval split intersects parity split e438721ae894; probe C3 (regulatory) and C2 (NVIDIA docs) which were not covered today.`

---

## Entry 2026-08-31-02 — context: the AV market moved, our tracking did not

**Evidence class: RELAYED (news aggregators, not verified against primaries).** Recorded as
*leads to verify*, not as facts:

- Waymo World Model announced 2026-02 (generative driving simulation); Waymo custom ASIC >1000 TOPS.
- Wayve raised $1.2 B (NVIDIA, Uber, three automakers), 2026-02.
- Nevada approved Tesla / Uber / Waymo robotaxi permits, 2026-08-20.
- DeepRoute.ai showed a 40 B VLA foundation model at GTC 2026.

⚠️ **None of these is banked or primary-verified.** They are here so the next pass has leads, and
because **the programme had zero Band-C coverage before today** — which is how a release built on
our own corpus went unregistered for ~8 months.


---

## Entry 2026-09-01-01 - Waymo Foundation Model: a HYBRID hierarchy that deploys SMALL (settles debt D-2)

**Source:** `waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/`
**Evidence class:** PUBLISHED-BLOG (primary fetched 2026-09-01). **Discharges register debt D-2.**

> *"leverages the full expressibility of learned embeddings as a rich interface between model
> components and supports full end-to-end signal backpropagation"*

**Architecture:** **Sensor Fusion Encoder** (rapid reactions) + **Driving VLM** (semantic reasoning),
both feeding a **World Decoder**. Ecosystem pillars: **Driver / Simulator / Critic**, "all fueled by
the same underlying AI".

⭐ **Distillation now at PRIMARY:** *"these Teacher models are too big to run on vehicles for real-time
decision making"* - teachers distilled to students for onboard use, with a separate onboard
validation layer verifying the generative model's trajectories.

**Claims:** ">ten-fold reduction in crashes with serious injuries compared to human drivers";
">100 million fully autonomous miles". ⛔ **No ablations. No limitations admitted anywhere.**

⛔ **CORRECTION TO OUR RECORD (register debt D-2):** the framing *"Waymo's model is end-to-end just
like Tesla and Wayve"* is **UNSUPPORTED-AS-STATED**. The primary describes an explicitly **hybrid**
design - modular components with learned-embedding interfaces, trained with end-to-end backprop -
which the document presents as superior to *both* pure-E2E and modular.

**What it changes for us:** ⭐⭐ this is **TanitAD's hierarchy thesis instantiated in an opponent's
production stack** (fast reactive path + slow semantic path + shared decoder + differentiable
interfaces), and, with the distillation concession, **external support for sub-300M deployment**.
⚠️ **But it is advocacy, not evidence** - no ablation shows the hybrid beats either alternative.
It raises our prior; it decides no GPU-day.

## Entry 2026-09-01-02 - Band C sweep: JetPack 7.2.1, and a documented-vs-measured conflict on Thor

**Evidence class:** PUBLISHED-RELEASE-NOTE / RELAYED (vendor docs + trade coverage, 2026-09-01).
**Resolves empty E3 from 2026-08-31** ("no Aug-2026 TensorRT/JetPack notes surfaced").

- **JetPack 7.2.1 released 2026-08-12**, shipping a Jetson T3000 emulator for AGX Thor.
- TensorRT support matrix: **Jetson AGX Thor = compute capability 11.0**; **FP4 requires CC >= 10.0**,
  **FP8 requires CC >= 8.9**. Thor's Transformer Engine "dynamically switches between FP4 and FP8 at
  runtime". T3000: 865 FP4 TFLOPS; T2000: 400 FP4 TFLOPS.

⛔ **CONFLICT WITH OUR OWN MEASUREMENT.** D-B1-GATE (backlog row 1) records that TRT on Thor `sm_110`
**silently falls back to FP32** under FP8/FP4 flags (issue #4590 re-verified OPEN). The vendor matrix
says the precisions are supported; our measurement says the flags are silently ignored.

⚠️ **The MEASUREMENT wins.** A support-matrix row is `PUBLISHED-RELEASE-NOTE` and may not overturn a
measured fallback - this is exactly the evidence-class ordering the operating standard exists for, and
exactly why `quant_gate.py` is P0. ⇒ **Worth one re-test on JetPack 7.2.1**, since our measurement
predates this release; a fail-closed gate must not inherit a stale platform assumption either.

`Next in this track: re-run the FP8/FP4 precision census on JetPack 7.2.1 once quant_gate.py exists.`


---

## C4 entry 2026-09-02-01 — the first leaderboard-adjacent numbers the programme holds

`Band C / C4 · PUBLISHED (paper claims) · ⛔ NOT read off a live leaderboard — see empty E7`

**C4 (community signals / leaderboards) had never received a dedicated DEEP since the charter began.**
This entry opens it, with an explicit caveat about what these numbers are.

| system | score | split |
|---|---|---|
| **DrivoR** | **56.3 EPDMS** | NAVSIM v2 |
| **PDM-Closed** (privileged, ground-truth perception) | **56.6 EPDMS** | NAVSIM v2 |
| **CLOVER** | **48.3 EPDMS** | navhard-two-stage |
| **RAP-DINO** | **36.9 EPDMS** | NAVSIM v2 |

Context: NAVSIM v2 introduces **two-stage EPDMS**, adding Traffic Light Compliance, Driving Direction
Compliance, Lane Keeping and Extended Comfort. `navhard` is a nuPlan subset — **450 stage-1 and 5,462
stage-2 observations**. Release line: **NAVSIM v2.1.2** shipped `navhard_two_stage` and the updated EPDMS
for the HuggingFace warmup leaderboard.

### ⭐⭐ Why this sharpens the efficiency wedge

**DrivoR is within 0.3 EPDMS of a PRIVILEGED planner that consumes ground-truth perception.** Combined
with backlog row 32's ~40 M parameter figure for DrivoR, the wedge must be restated: **"sub-300M" is not
merely community-demonstrated — a ~40 M camera-only model sits at parity with a GT-perception planner on
this benchmark.** ⛔ Our efficiency claim has to beat *that*, not a 32 B model. Backlog row 32 is
strengthened and should be re-read with these numbers.

### ⚠️⛔ V-5 guards — both binding, both reasons these may not enter a table yet

1. **The splits differ.** NAVSIM v2 overall and navhard-two-stage are **not one ranking** and must never be
   pooled. Three of the four numbers above are not mutually comparable.
2. **EPDMS is comparable only within one scoring-basis era**, and **none of these has been cross-checked
   against our four-family definitions** — D-EPDMS-FAM is still open (EPDMS measures compliance and
   outcome; nothing in it maps to `tac.manoeuvre_decision` or `strat.route_goal`).
3. ⚠️ **These are paper claims, not a leaderboard read.** The only leaderboard snapshot surfaced was
   **March 2026** (empty E7).

⇒ **No number here may enter a TanitAD comparability table until backlog row 3 (portfolio approval) and
D-EPDMS-FAM land.** Recorded now so the wedge argument can be written, not so the numbers can be quoted.


---

## Entry 2026-09-05-01 — C1: Wayve GAIA-4, and a correction to our own LEADERBOARD row

`Cited by: Frontier Scan/Daily/2026-09-05/RESULT.md F1. Full Band-D adjudication lives in`
`Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md rows V-1…V-4 — this entry records the RELEASE facts only.`

| item | fact | class |
|---|---|---|
| **GAIA-4** | `wayve.ai/thinking/gaia-4/`, **2026-08-03**, 13 named authors. *"the world model at the core of Simulation 2.0"* | PUBLISHED-BLOG |
| architecture | multimodal generative world model producing **camera video AND radar from the same model** — *"These detections are generated by the same world model that produces the video, rather than being added separately afterward."* Claimed an industry first for AV simulators | PUBLISHED-BLOG |
| loop semantics | *"as the model makes different driving decisions, the sensor inputs it receives change accordingly"* — ego-reactive | PUBLISHED-BLOG |
| ⛔ constraint | *"every other agent in the scene keeps the exact behavior it showed in the real-world log"*; *"the evaluation stays conservative: no vehicle, pedestrian, or cyclist changes its behavior in response to the AI Driver"* | PUBLISHED-BLOG (**the concession**) |
| parameter count | ⛔ **NOT STATED** (empty E6) | — |
| validity number | ⛔ **NOT STATED** — the post's own question 1 (*"Do the simulated runs reproduce the outcomes we observed in the real world?"*) is never answered numerically | — |
| only controlled number | *"Training GAIA for this task improves how faithfully it preserves the recorded world by **2.5x**"* — metric unnamed, no absolute values | PUBLISHED-BLOG |
| **GAIA-3** | `wayve.ai/press/wayve-launches-gaia3/`, **dated 2 December 2025 on the page**. 15 B params (2× GAIA-2), video tokenizer 2× larger, 10× pre-training data. *"reduced synthetic-test rejection rates fivefold"* | PUBLISHED-BLOG |
| deployment | Wayve + Uber London robotaxi service launched **2026-09-03**; no operator has yet completed VCA registration for unsupervised service | RELAYED (secondary) |
| external validation | **DriveSafeSim**, a UK-government-funded project with Warwick Manufacturing Group, *"to validate the use of generative world models such as GAIA-3 in safety evaluation"* | PUBLISHED-BLOG |

### ⛔ Correction to a TanitAD record

`Benchmarks & Eval/LEADERBOARD.md:1370` reads *"Wayve GAIA-3 | 15 B | **offline** generative world model |
Opponent profiles, 2026-07"*. **GAIA-4 supersedes it and changes the CATEGORY** — GAIA-3's own page calls it
offline; GAIA-4's entire claim is that it is not. ⇒ update the row, and mark both **capability-only**:
⛔ **no GAIA-3/GAIA-4 number may enter a TanitAD comparability table** (guideline T-5) — the 2.5× has no
named metric, the fivefold measures test throughput not validity, and there is no parameter count at all.

### C2 sub-entry — engineering blogs / release notes

⚠️ **EMPTY at one probe (E3b).** No Sept-2026 JetPack/TensorRT release notes surfaced; the newest NVIDIA
developer-blog item found is **2026-03-12**. That item does state that **Alpamayo-1 on DRIVE Thor uses
FP8-accelerated Vision Transformers at *"production-viable latencies"*** — ⛔ **no number, so it neither
confirms nor disturbs our MEASURED silent-FP32-fallback finding (#4590)**; FS-4 stands as written.
**One probe is not absence.** Next route: `docs.nvidia.com` directly, not a search engine.

---

## 2026-09-09-01 - C1: Alpamayo 1.5 (the day's Band-D item); C2 empty again; C4 commercial-only

`Retrieved 2026-09-09. Class PUBLISHED-RELEASE-NOTE (C1) / PUBLISHED-BLOG (C4).`

**C1 - `NVlabs/alpamayo1.5`.** Open-weights 10B reasoning VLA: Cosmos-Reason backbone plus an action
expert; **RL post-trained**; explicit navigation inputs; VQA; flexible multi-camera. Two inference
modes: *"the VLM generates chain-of-causation reasoning, then a diffusion expert produces trajectory
predictions"*, and text-only for VQA. **Hardware: minimum 24 GB VRAM single-sample, up to 60 GB with
CFG.** Limitations stated plainly: *"not a fully fledged driving stack... lacks access to critical
real-world sensor inputs, does not incorporate required diverse and redundant safety mechanisms"*, and
*"model accuracy may degrade with fewer cameras"*.

**Full seven-step adjudication (claims A15-1 to A15-4) is in `../Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md`.**
Headline: the **teacher-student concession is theirs and it is our thesis** (A15-2, CONFIRMS-US), while
**verbose natural-language CoT is contradicted by two independent groups** (A15-3,
UNSUPPORTED-AS-STATED).

**C2 - empty at a second probe (E-C2b).** Newest JetPack remains **7.2.1 (2026-08-12)**; nothing newer
for Thor. vLLM v0.28.0 (2026-08-26); TensorRT-LLM tracking torch 2.9.0. **FS-4 stands unchanged for a
second consecutive pass** - the FP8/FP4 precision census re-run is still gated on `quant_gate.py`, not
on a platform release.

**C4 - commercial expansion, no architectural content.** Waymo opened San Diego, Denver and Tampa on
2026-09-01 and reports more than 500,000 robotaxi rides per week, targeting 1 million by year end. Zoox
began supervised testing in Houston and San Diego. Tesla's purpose-built Cybercab has carried public
riders in part of Austin since 2026-09-04. Mobileye plans about 100 robotaxis in one US city by 2027,
scaling toward 17,000 over five years. **Class RELAYED / PUBLISHED-BLOG; none of it decides anything
for us, and it is recorded so the C4 rotation is not re-run on the same ground.**

## 2026-09-10-01 — C2 empty at a THIRD probe; C1/C4 quiet; C3 declared un-probed

**C2 (engineering blogs / release notes).** ⛔ **EMPTY E-C2c, third consecutive probe.** Newest Thor-relevant stack is still **JetPack 7.2.1 / Jetson Linux 39.2.1 / CUDA 13.2.1 / TensorRT 10.16.2** (2026-08-12). ⭐ Three probes across three passes now agree: **no Thor-relevant JetPack release in four weeks.** Recorded as an absence with its probe count, not as a gap in reading.

⭐ **New C2 signal with a live consequence.** `PUBLISHED-RELEASE-NOTE`: TensorRT **Edge-LLM** (open-source C++ SDK for LLM/VLM edge inference) now supports Thor; and the Isaac-ROS release notes record that **DOPE ONNX → TensorRT plan conversion FAILS on Jetson AGX Thor on unsupported layers.** ⇒ a third-party instance of the silent-fallback / unsupported-layer class that backlog row 1's `quant_gate.py` exists to catch (`D-B1-GATE`). ⚠️ A different model and a different failure mode from ours; carried as a corroborating instance of the class, **not** as evidence about our engines.

**C1 (lab + AV releases).** NVIDIA Alpamayo 2 Super released for commercial use; Alpamayo 1.5 (10 B) post-training path documented → the Band-D item adjudicated this pass. Wayve GAIA-3 (2025-12) / GAIA-4 (2026-08-03) already registered 09-05; **no new Wayve doctrine post surfaced.**

**C4 (community signals).** No new leaderboard movement probed beyond the navhard anchors extracted for `LEDGER_A5_benchmarks.md` 2026-09-10-01.

**C3 (regulatory).** ⛔ **NOT SCANNED — declared, not concealed.** D-4 stands at five failed routes and is with the PI; no new route was attempted and none is claimed.

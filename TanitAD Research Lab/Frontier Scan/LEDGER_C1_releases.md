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

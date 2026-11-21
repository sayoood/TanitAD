<title>QUOTES — Waymo 10-lessons adjudication, evidence layer</title>

# QUOTES — evidence layer

`Every passage the RESULT relies on, with its source, retrieval method and evidence class.`
`⛔ Tiers are never merged in a sentence. A RELAYED item is a verification target, not a claim.`

---

## Tier 1 — THE TARGET DOCUMENT (`PUBLISHED-BLOG`)

**Source:** Srikanth Thirumalai, *"10 AI Lessons from Driving 200+ Million Fully Autonomous Miles"*,
`waymo.com/blog/2026/08/10ailessons/`, **published 2026-08-26**.
**Retrieval:** WebFetch of the URL supplied by the PI, 2026-08-31. Full article extracted.
**Evidence class:** `PUBLISHED-BLOG` — vendor advocacy. **No controlled ablation, no interval, no
`n`, no metric definition appears anywhere in the document.**

| L | verbatim passage relied on |
|---|---|
| L1 | lidar gives *"3D geometry with millimeter precision"*; radar *"sees through"* obscuring weather; cameras deliver *"semantic overlay"* |
| L2 | maps are *"another input—like sensors, but acting as a mental memory"*; *"helpful in poor visibility and complex thoroughfares"* |
| L3 | modules become *"unmaintainable at scale"*; consolidate to *"fewer, high-capacity, specialized foundation models"*; *"data, rather than brittle human priors, determine what is relevant"*; ⭐ *"teacher-student models to optimize onboard compute"* |
| L4 | *"an independent onboard validation layer"* that *"monitors every trajectory"* against *"hard physics-based constraints and traffic laws"*; a *"hard backstop"* |
| L5 | ⭐⭐⭐ open-loop replay: surrounding traffic *"moves, but it's completely indifferent to your actions"*; closed-loop *"mimic[s] real-world cause and effect"* |
| L6 | an *"AI critic"* preventing the system from *"grading its own homework"*; *"millions of road miles traveled each week (and tens of billions in simulation)"* |
| L7 | ⭐⭐⭐ VLMs give *"high-level semantic hints"* but are *"too slow for real-time control"* and *"lack sufficient spatial awareness"*; *"thinking fast and slow"* |
| L8 | *"a rigorous, quantitative data engine"*; *"multiple complementary evaluation methodologies"* |
| L9 | *"an automated data flywheel—a virtuous cycle of continuous improvement"*; *"exabytes of data"* |
| L10 | L2→L4 progression is *"a false summit"*; L4 needs *"a purpose-built system... hardened by the uncompromising experience of driving without a human in the car"* |

**Sources the article links but which I did NOT read** (recorded so no claim is made about them):
*"Demonstrably Safe AI for Autonomous Driving"* (2025-12) · Waymo Foundation Model (2025-12) ·
*"The Waymo World Model"* (2026-02).

---

## Tier 2 — FRAMING EVIDENCE (`RELAYED`, used only to date the comms context)

| item | source | retrieval |
|---|---|---|
| *"Waymo says there's no AI shortcut to self-driving"* — Axios exclusive, **2026-08-26**, same day as the blog | axios.com | WebSearch result title/date, **page not fetched** |
| *"Waymo takes a shot at Tesla's self-driving: it's a 'false summit'"*, **2026-08-27** | electrek.co | WebSearch result title/date, **page not fetched** |
| Waymo co-CEO on camera-only, **2026-08-04** | electrek.co | WebSearch result title/date, **page not fetched** |

⚠️ **These date the publication context and nothing more.** The "coordinated comms push" reading in
RESULT §1 is an **inference from timing**, explicitly labelled as such there.

---

## Tier 3 — CONTRADICTING / CONFIRMING EVIDENCE

| # | claim used | source | class |
|---|---|---|---|
| C1 | camera-only BEV ≈ **85.1 %** of lidar segmentation, **92.1 %** of lidar detection | `2505.06113` **banked**, ⛔ **NOT READ** | ⚠️ **RELAYED** — attributed by a search summary. **T-3 exists to verify this in the primary.** May not decide anything. |
| C2 | planning from **raw pixels** end-to-end *"rival[s] state-of-the-art multimodal systems"* | `2507.17596` PRIX, **banked**, not read | ⚠️ RELAYED |
| C3 | ⭐ *"Waymo vehicles are controlled by a foundation model trained in an end-to-end fashion — just like Tesla and Wayve vehicles"* | understandingai.org | RELAYED — **the architectural-convergence observation; the single most load-bearing counter to L1's framing, and it is NOT primary-verified** |
| C4 | *"the self-driving problem is not a sensor problem, it's an AI problem"* — Elluswamy, ScaledML **2026-01-29** | conference talk via search summary | RELAYED |
| C5 | lidar detected a pedestrian in a Phoenix dust storm invisible to camera, Google I/O 2026 | search summary | RELAYED — **an anecdote, treated in RESULT as an existence proof only** |
| C6 | learned critics: optimisation *"can raise the learned reward while external quality falls"*; a critic *"may optimize toward evaluator artifacts... and the policy may inherit the same misalignment"* | `2606.03238`, `2604.13602` — **both banked**, abstract/summary level | PUBLISHED, **abstract-only — declared** |
| C7 | mapless / online lane-graph literature is active (CGNet, LGmap, online lane-graph extraction); manual HD-map annotation *"slow and expensive"* and a scaling bottleneck | search summaries | RELAYED |
| C8 | Wayve raised **$1.2 B** (NVIDIA, Uber, three automakers), 2026-02 | techcrunch via search | RELAYED |

---

## Tier 4 — OUR OWN MEASUREMENTS USED IN THE ADJUDICATION (`MEASURED`, in-programme)

| # | fact | where it lives |
|---|---|---|
| M1 | action echo — S-curve reproduction **97.9 %** open-loop, **0.0 %** hold-action, **~5 %** closed-loop | `EVAL_DOCTRINE.md` §1.12 |
| M2 | rollout latency **linear in K**; **203 ms at 37.8 M params, K=60**; controls clean | `Deployment & Optimization/Research/2026-08-31-rollout-depth-latency/` |
| M3 | v6 readout geometry: 16×40 tokens → 4×4 = **4 azimuth bins over 120° (30°/bin)**, **2.1–7.8×** too coarse for BEV localisation | v6 readout-geometry finding |
| M4 | PhysicalAI-AV has **no map, lane graph, junction annotation, traffic-light feature or route/goal signal** (card: *"we do not include open maps data"*); `egomotion` has **no lat/lon/GNSS** | settled at **five independent probes** |
| M5 | NuRec scenes ship **`map.xodr`**; `volume.nurec` is gzip+msgpack, not an opaque blob; gsplat **492 FPS** on Thor | NuRec finding |
| M6 | the ridge-probe post-mortem — **four distinct failures in one afternoon**, each caught only by a control that had to read a known value | `CLAUDE.md` traps preflight |

## Tier 5 — VENDOR SELF-REPORT USED AGAINST L3 (`PUBLISHED`, vendor cards)

| model | params | LingoQA | AlpaSim closed-loop | open-loop minADE₆ @6.4 s |
|---|---:|---:|---:|---:|
| `nvidia/Alpamayo-1.5-10B` | 8.2 B + 2.3 B | 74.2 | 1.37 ± 0.10 | 0.916 m |
| `nvidia/Alpamayo2-Super` | 32 B + 2.3 B | 79.2 | 1.50 ± 0.13 (913 scen.) | 0.911 m (1434 samples) |

⛔ **`minADE₆` is best-of-6, not ADE — never comparable to our `fwd_ade`.** Used here **only** to
compare Alpamayo to Alpamayo. **This is vendor self-report on their own protocol, not a controlled
scaling study**, and RESULT §6.3 says so.

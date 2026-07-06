# OPPONENT PROFILES

> One page per opponent. Updated **deltas only** by the Opponent Analyzer agent; each keeps a
> **"What would beat them"** section current. Labels: FACT / CLAIM / INFER (G-O1).
> Created 2026-07-17 (v1). Last full sweep: **2026-07-17**.

---

## Wayve  (UK · E2E foundation model · robotaxi + consumer L2+/L3)
- **Approach (FACT):** AV2.0 — a single end-to-end **camera+radar, mapless** foundation model; GAIA
  generative world models for synthetic data + **offline evaluation** (GAIA-3 = 15 B latent diffusion);
  LINGO VLA for natural-language rationales. Qualcomm production-ADAS partnership (Mar 2026).
- **Business (FACT):** **Series D $1.2 B (Feb 2026), $8.6 B** post-money; **$1.5 B** total secured incl.
  Uber milestone capital; investors Microsoft/NVIDIA/Uber + Mercedes/Nissan/Stellantis. London robotaxi
  trials 2026; supervised consumer autonomy from 2027; 10+ markets targeted.
- **Strengths (INFER):** talent + capital density; UK/EU regulatory access; genuine multi-country
  generalization data; strong generative-WM data factory.
- **Exploitable weaknesses:** generative-**pixel** WM is compute-hungry and used **offline**, not in the
  loop (W-05); on-car driver is monolithic E2E — **no hierarchy, no imagination-in-the-loop, no
  self-monitoring guarantee** (W-04 adjacent); L2+/robotaxi split spreads focus.
- **What would beat them:** ship the WM *as the on-car reasoning substrate* (not an offline data factory)
  at 1–2 orders less compute (H1/H3/H5, CNCE), with in-loop imagination (H15) + guaranteed self-monitoring
  (H11) that a monolithic E2E net structurally lacks.

## Waymo  (US · modular+learned, HD maps · robotaxi at scale)
- **Approach (FACT):** modular+learned stack, HD maps, large multi-sensor fleet; freeway autonomy since
  Nov 2025.
- **Business (FACT):** best-capitalized Western operator; 2026 plan = 20+ new cities + London/Tokyo intl.
- **Safety record (FACT):** **recall of 3,871 vehicles (2026-06-18)** for freeway construction-zone entry
  (freeway autonomy suspended, expansion frozen); NTSB **HWY26FH008** school-zone pedestrian braking-late
  case; separate probe into **school-bus stop-arm** passing; NHTSA SGO **697 incidents** (1 fatality, 23
  hospitalizations) in the Jun'25–May'26 window; CA DMV **19,234 mi/disengagement**.
- **Strengths (INFER):** scale, safety-engineering process, war chest, brand.
- **Exploitable weaknesses:** **construction/work-zone brittleness** (W-01, headline), **occlusion/VRU
  anticipation** (W-02), **rule-compliance edges** (W-03); map-dependence; cost/vehicle + geofence
  economics; no data-efficiency story.
- **What would beat them:** own the exact edge cases their recalls expose — work-zone imagination (H15) +
  inherent rule compliance (H9) + occlusion permanence (LOPS/OKRI, D9) — at hobbyist compute, framed for
  the new WP.29 regulation (H11/H9).

## Pony.ai  (China + intl · robotaxi · NASDAQ)
- **Approach (FACT):** multi-sensor robotaxi stack; rapid geographic expansion.
- **Business (FACT):** Q1'26 total rev **$34.3 M (+145% YoY)**, robotaxi rev **$8.6 M (+395% YoY)**; 2026
  fleet target **3,500+**, 20+ cities; live in **Croatia (first EU commercial robotaxi)** + **Dubai**;
  9 countries.
- **Strengths (INFER):** fleet scale, China + Middle East + first-mover EU footprint, steep growth.
- **Exploitable weaknesses:** **thin unit economics** vs fleet (W-06); same compute-heavy multi-sensor
  stack (W-05); geopolitics limits Western data/market access.
- **What would beat them:** the data-efficiency + cost-per-vehicle story (H3+H7) they have no answer to;
  a Western/EU-clean data and compliance posture.

## Momenta  (China · two-leg: mass L2++ + robotaxi · HK IPO)
- **Approach (FACT):** two divisions — production L2++ software for OEMs, and robotaxi; GM + Tencent backed.
- **Business (FACT):** **HK IPO ~$752 M**, ~**$9 B** valuation, trading 2026-07-08; 60% proceeds→R&D,
  20%→robotaxi; approvals Suzhou/Shanghai; **Abu Dhabi + Munich 2026**; Uber L4 pilot.
- **Strengths (INFER):** OEM data flywheel, China scale, GM/Tencent + public capital.
- **Exploitable weaknesses:** **strategy split** now locked by public-market scrutiny; opaque safety case;
  geopolitics.
- **What would beat them:** a single coherent efficiency+safety thesis (vs their L2++/L4 straddle) with a
  transparent, regulation-native safety case (H9/H11).

## Autobrains  (Israel · L2+/ADAS · "Liquid AI")  ⚠ narrative overlap
- **Approach (FACT):** liquid neural networks + **modular agentic AI**, marketed as **edge-cases with less
  compute on standard sensors**; Skills product line (Oct 2024).
- **Business (FACT):** **$140 M+** funding; BMW/Toyota/Continental/Temasek.
- **Strengths (INFER):** low-compute narrative overlaps ours; strong Tier-1/OEM channel; real mass-market
  ADAS focus.
- **Exploitable weaknesses:** **sub-L3 / ADAS, not L4**; "liquid" = runtime adaptivity, **not** a
  hierarchical latent world model, no imagination, no self-monitoring-with-guarantees, no
  action-free-video data-efficiency claim.
- **What would beat them:** own the L4 world-model + safety-case ground they don't play on, and pre-empt
  their efficiency messaging with **compute-normalized (CNCE) proof** on L4-grade edge cases.

## NVIDIA Alpamayo  (US · open ecosystem · frenemy / supply chain)
- **Approach (FACT):** open models + sim + data; **Alpamayo 2 Super = 32 B reasoning VLA** on Cosmos,
  Chain-of-Causation traces; open dataset **1,700+ h / 25 countries / 2,500+ cities**.
- **Role (INFER):** **supply chain, not competitor** — their data/sim (Cosmos-Drive-Dreams, PhysicalAI-AV)
  feed our training mix; their **32 B on-car VLA is our foil** on efficiency.
- **Exploitable weaknesses:** 32 B/vehicle = anti-efficiency (W-05); Chain-of-Causation is *post-hoc*
  interpretability vs our *inherent* fallback + self-monitoring.
- **What would beat them (as a narrative):** ~261 M-on-Orin at comparable causal efficacy (CNCE), inherent
  (not traced) safety. Keep consuming their open assets.

## Tesla  (US · camera-only E2E · robotaxi)  — emerging player
- **Approach (FACT):** camera-only end-to-end FSD; unsupervised robotaxi (Miami launch 2026-07-03; 5
  territories; TX fleet ~42 vs Waymo 577).
- **Safety (FACT/CLAIM):** **NHTSA engineering analysis (Mar 2026)** — camera-only FSD fails under
  **degraded visibility (glare/obscurants)**, pre-recall step; Austin ~**14 crashes / 800 k mi** (CLAIM,
  ~4× US-driver rate by Tesla's metric); scale deferred to unreleased FSD v15.
- **Strengths (INFER):** fleet-data scale, vertical integration, cost focus.
- **Exploitable weaknesses:** **no calibrated epistemic uncertainty** → confident-when-blind (W-04);
  camera-only sensing; monolithic E2E, no self-monitoring guarantee.
- **What would beat them:** H11 self-monitoring (degraded-visibility AUROC) + H15 epistemic σ + H2
  sensor-modality steering (radar fallback) — exactly the axis their open NHTSA case is about.

---

### Cross-field one-liner (INFER)
Nobody occupies our Pareto point — **hierarchical latent world model, ~261 M params, data-efficient,
real-time on Orin, in-loop imagination + guaranteed self-monitoring, regulation-native.** Mid-2026
made the case *stronger*: competitors' public failures are our target edge cases, and the research
field's convergence on latent WMs means our differentiation is now **hierarchy + efficiency +
imagination + self-monitoring**, not "world model" alone.

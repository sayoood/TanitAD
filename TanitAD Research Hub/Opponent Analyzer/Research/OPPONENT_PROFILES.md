# OPPONENT PROFILES

> One page per opponent. Updated **deltas only** by the Opponent Analyzer agent; each keeps a
> **"What would beat them"** section current. Labels: FACT / CLAIM / INFER (G-O1).
> Created 2026-07-17 (v1). Last full sweep: **run #4, real wall-clock 2026-07-11** (deltas tagged
> **Δ 07-11**; narrative-clock artefact — run #2's deltas are tagged Δ 07-24 but run #4 is most
> recent). New this run: **Zoox** profile. Note: run #3 (874f78e, unmerged) added the Waymo-6th-recall /
> Wayve-Stellantis / Pony-4-cities / AlpaGym deltas + FMVSS-135 — merge it before this branch.

---

## Wayve  (UK · E2E foundation model · robotaxi + consumer L2+/L3)
- **Approach (FACT):** AV2.0 — a single end-to-end **camera+radar, mapless** foundation model; GAIA
  generative world models for synthetic data + **offline evaluation** (GAIA-3 = 15 B latent diffusion);
  LINGO VLA for natural-language rationales. Qualcomm production-ADAS partnership (Mar 2026).
- **Business (FACT):** **Series D $1.2 B (Feb 2026), $8.6 B** post-money; **$1.5 B** total secured incl.
  Uber milestone capital; investors Microsoft/NVIDIA/Uber + Mercedes/Nissan/Stellantis. London robotaxi
  trials 2026; supervised consumer autonomy from 2027; 10+ markets targeted.
- **Δ 07-11 (FACT):** decisive pivot to **L4 robotaxi** — **Stellantis + Wayve + Uber** partnership
  (June 2026, jointly develop/deploy L4 across Europe/NA) + **Wayve × Nissan** prototype on **NVIDIA
  DRIVE Hyperion** for Uber **Tokyo** pilots (late-2026); London commercial trials 2026. Wayve is
  converting its foundation-model/GAIA-3 story into a **multi-OEM L4 deployment on NVIDIA compute.**
  (INFER) This raises Wayve's exposure to the exact L4 edge cases in our scenario DB while the on-car
  driver stays monolithic E2E and GAIA-3 stays an *offline* edge-case generator (not in-loop).
- **Strengths (INFER):** talent + capital density; UK/EU regulatory access; genuine multi-country
  generalization data; strong generative-WM data factory; now a **multi-OEM (Stellantis/Nissan) +
  Uber distribution** channel.
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
- **Safety record (FACT):** **recall of 3,871 vehicles (2026-06-18, NHTSA 26E035)** for freeway
  construction-zone entry (freeway autonomy suspended, expansion frozen); NTSB **HWY26FH008** school-zone
  pedestrian braking-late case; separate probe into **school-bus stop-arm** passing; NHTSA SGO **697
  incidents** (1 fatality, 23 hospitalizations) in the Jun'25–May'26 window; CA DMV **19,234
  mi/disengagement**.
- **Δ 07-24 (FACT):** the construction-zone recall was Waymo's **second in ~one month**; it **pulled all
  robotaxis from highways on 2026-05-19** and the fix is still "under development." Its own filing names
  the mechanism (mis-prioritizing hazard-avoidance / not recognizing the work zone). **New Dallas market
  trouble:** a Waymo recorded **running a red light** (Irving Blvd/Inwood Rd) amid a **new federal
  investigation** there → widens the rule-compliance surface (SC-04/SC-14, W-03).
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
- **Δ 07-24 (FACT):** robotaxi fleet now **exceeded 1,700 units** (toward the raised 3,500 target); added
  **Guangzhou**; **raised** 2026 robotaxi-revenue and fleet targets on record Q1. Growth real but the
  revenue-vs-fleet gap (W-06) is unchanged — $8.6 M robotaxi rev against a 1,700+ (→3,500) fleet.
- **Δ 07-11 (FACT):** 2026 fleet goal lifted to **3,500** (now **>1,700**); markets add **Singapore,
  South Korea, Qatar**. Q1'26 robotaxi rev **$8.7 M** → the revenue-vs-fleet gap (**W-06**) is
  unchanged — growth is geographic, not margin. — https://www.bloomberg.com/news/articles/2026-05-26/pony-ai-lifts-2026-robotaxi-fleet-goal-to-3-500-on-fast-growth
- **Strengths (INFER):** fleet scale, China + Middle East + first-mover EU footprint, steep growth.
- **Exploitable weaknesses:** **thin unit economics** vs fleet (W-06); same compute-heavy multi-sensor
  stack (W-05); geopolitics limits Western data/market access.
- **What would beat them:** the data-efficiency + cost-per-vehicle story (H3+H7) they have no answer to;
  a Western/EU-clean data and compliance posture.

## Momenta  (China · two-leg: mass L2++ + robotaxi · HK IPO)
- **Approach (FACT):** two divisions — production L2++ software for OEMs, and robotaxi; GM + Tencent backed.
- **Business (FACT):** **HK IPO ~$752 M**, ~**$9 B** valuation, trading 2026-07-08; 60% proceeds→R&D,
  20%→robotaxi; approvals Suzhou/Shanghai; **Abu Dhabi + Munich 2026**; Uber L4 pilot.
- **Δ 07-24 (FACT):** listed 2026-07-08 at HK$295.6 (~HK$69.6 B ≈ **$8.9 B** cap, rose on debut);
  cornerstones incl. **Mercedes-Benz + BYD + GIC/Fidelity**. Shipped its own **R7 Reinforcement-Learning
  World Model (Apr 2026)** and first self-developed chip **X7** (in SAIC-VW ID.ERA 9X) → Momenta now also
  has a "world model," reinforcing that *"world model" is table stakes, not a differentiator*. **Uber's
  Munich robotaxi plan appears to have shifted from Momenta to Autobrains+NVIDIA** (see Autobrains) — a
  competitive-loss signal for Momenta's international robotaxi leg.
- **Strengths (INFER):** OEM data flywheel, China scale, GM/Tencent + public capital, own WM + own silicon.
- **Exploitable weaknesses:** **strategy split** now locked by public-market scrutiny; opaque safety case;
  geopolitics; R7 is a *generative RL* WM (compute/data-heavy) with no hierarchy/self-monitoring claim.
- **What would beat them:** a single coherent efficiency+safety thesis (vs their L2++/L4 straddle) with a
  transparent, regulation-native safety case (H9/H11).

## Autobrains  (Israel · L2+/ADAS · "Liquid AI")  ⚠ narrative overlap
- **Approach (FACT):** liquid neural networks + **modular agentic AI**, marketed as **edge-cases with less
  compute on standard sensors**; Skills product line (Oct 2024).
- **Business (FACT):** **$140 M+** funding; BMW/Toyota/Continental/Temasek.
- **Δ 07-24 (FACT):** **Uber + Autobrains (+ NVIDIA) Munich robotaxi pilot announced 2026-06-02**,
  apparently displacing/paralleling Uber's earlier Momenta-Munich plan → Autobrains is **stepping up from
  ADAS toward an L4 pilot.** This is the sharpest watch-list escalation this run: their "edge-cases with
  less compute" message now rides an L4 deployment. — https://www.electrive.com/2026/06/02/uber-and-autobrains-to-partner-on-munich-robotaxi-pilot-project/
- **Strengths (INFER):** low-compute narrative overlaps ours; strong Tier-1/OEM channel; now an Uber L4
  pilot partner.
- **Exploitable weaknesses:** still **no public hierarchical latent WM, no in-loop imagination, no
  self-monitoring-with-guarantees, no action-free-video data-efficiency claim**; "liquid" = runtime
  adaptivity. The L4 pilot raises their exposure to exactly the L4 edge cases (work-zone, occlusion,
  rule-barrier) our scenario database is built on.
- **What would beat them:** own the L4 world-model + safety-case ground they don't play on, and pre-empt
  their efficiency messaging with **compute-normalized (CNCE) proof** on L4-grade edge cases.

## NVIDIA Alpamayo  (US · open ecosystem · frenemy / supply chain)
- **Approach (FACT):** open models + sim + data; **Alpamayo 2 Super = 32 B reasoning VLA** on Cosmos,
  Chain-of-Causation traces; open dataset **1,700+ h / 25 countries / 2,500+ cities**.
- **Role (INFER):** **supply chain, not competitor** — their data/sim (Cosmos-Drive-Dreams, PhysicalAI-AV)
  feed our training mix; their **32 B on-car VLA is our foil** on efficiency.
- **Δ 07-24 (FACT):** the **Mercedes-Benz CLA** becomes the **first production vehicle to ship NVIDIA's
  entire AV stack** (US, this quarter) → Alpamayo goes from reference model to shipped product. Family now
  spans **10 B (Alpamayo 1 Nano / 1.5 Nano) → 32 B (2 Super)** — i.e. NVIDIA has a *smaller* tier that
  partially answers the efficiency critique, but 10 B on-car is still ~40× our ~261 M active envelope.
  **AlpaSim** (open-source closed-loop sim) is on GitHub → a *usable asset* for our CARLA-alternative
  closed-loop eval (flag to Tools&DevEnv). Our CNCE wedge holds; watch whether a Nano-tier CNCE number
  ever gets published.
- **Exploitable weaknesses:** 10–32 B/vehicle = anti-efficiency (W-05); Chain-of-Causation is *post-hoc*
  interpretability vs our *inherent* fallback + self-monitoring.
- **What would beat them (as a narrative):** ~261 M-on-Orin at comparable causal efficacy (CNCE), inherent
  (not traced) safety. Keep consuming their open assets.

## Tesla  (US · camera-only E2E · robotaxi)  — emerging player
- **Approach (FACT):** camera-only end-to-end FSD; unsupervised robotaxi (Miami launch 2026-07-03; 5
  territories; TX fleet ~42 vs Waymo 577).
- **Safety (FACT/CLAIM):** **NHTSA engineering analysis (2026-03-18)** — camera-only FSD fails under
  **degraded visibility (glare/obscurants)**, pre-recall step; Austin ~**14 crashes / 800 k mi** (CLAIM,
  ~4× US-driver rate by Tesla's metric); scale deferred to unreleased FSD v15.
- **Δ 07-24 (FACT):** the EA covers **~3.2 M vehicles / 9 crashes / 1 fatality + 2 injuries** and names
  the failed **"degradation-detection"** feature (doesn't flag impaired cameras until immediately
  pre-crash). Separately, Tesla **unredacted its 17 Austin robotaxi ADS incidents** (Jul'25–Mar'26; 13
  property-only, 1 hospitalization, **2 involving teleoperators**). Miami robotaxi launched into rain
  (2026-07-03) — a live stress of exactly the open case. This is the strongest single validation of our
  H11/H15/H2 axis in the field.
- **Δ 07-11 (FACT):** (1) **Houston fatality (2026-06-21)** — a Model 3 under automated driving crossed
  a lawn and **rammed a house, killing 76-yo Martha Avila in her living room**; NHTSA special crash
  investigation opened 06-23 (46 Tesla ADS/ADAS special investigations over the decade, "more than a
  dozen" with a fatality). (2) A **second, distinct NHTSA docket** clarified: **EA26002** (opened
  2025-10-07, ~**2.88 M** FSD vehicles) is the **traffic-law-violation** EA — **80 incidents** (from
  58) of **red-light running / illegal turns / oncoming-traffic entry**, **14 crashes / 23 injuries**,
  fines ≤ $139.4 M — SEPARATE from the **visibility EA** (3.2 M, degradation-detection). So Tesla now
  spans **two** open weaknesses: W-04 (visibility) *and* **W-03/SC-14 rule compliance.**
- **Strengths (INFER):** fleet-data scale, vertical integration, cost focus.
- **Exploitable weaknesses:** **no calibrated epistemic uncertainty** → confident-when-blind (W-04);
  **rule-compliance failures at scale** (red-light/illegal-turn, W-03/SC-14); camera-only sensing;
  monolithic E2E, no self-monitoring guarantee.
- **What would beat them:** H11 self-monitoring (degraded-visibility AUROC) + H15 epistemic σ + H2
  sensor-modality steering (radar fallback) for W-04; **H9 inherent rule-barrier (violation-rate = 0,
  SC-14)** for the red-light docket — exactly the axes their two open NHTSA cases are about.

## Avride  (US/intl · Uber robotaxi partner · Yandex SDG lineage)  — emerging player (new 2026-07-24)
- **Approach (FACT/INFER):** self-driving stack spun out of Yandex's SDG group, deployed via **Uber** in
  US pilot markets (incl. Dallas). Modular AV stack (INFER — architecture not publicly detailed).
- **Business (FACT):** Uber robotaxi partner; scaling in Dallas and other US pilots alongside Uber's
  multi-vendor strategy (also Waymo, Momenta, Autobrains, NVIDIA).
- **Safety (FACT):** **NHTSA ODI investigation opened 2026-05-08** — **16 crashes + 1 minor injury**; ODI
  says all concern **"the competence of"** the system: **lane-changing, same-lane vehicle response, and
  stationary-object response.** — https://techcrunch.com/2026/05/08/uber-partner-avride-is-under-investigation-for-self-driving-crashes/
- **Strengths (INFER):** Uber distribution + demand aggregation; Yandex AV heritage.
- **Exploitable weaknesses:** the ODI list is a **basic-competence** indictment (the cheapest, broadest
  surface) → **W-08 / SC-13**. Our H15 consequence-forward-model targets stationary-object/same-lane
  response directly (no detection prior to be wrong about).
- **What would beat them:** prove excellence on the *mundane* longitudinal/lateral tasks (SC-13) that
  their ODI flags — the least glamorous but most damning ground — at ~261 M params.

## Zoox  (US · Amazon · purpose-built driverless robotaxi)  — emerging player (new 2026-07-11)
- **Approach (FACT):** Amazon-owned; **purpose-built, bidirectional, steering-wheel-free** robotaxi
  (not a retrofit) → requires an **FMVSS exemption** to operate commercially (petition ≤2,500
  vehicles, under NHTSA review late-June 2026). Multi-sensor stack.
- **Business (FACT):** Amazon-funded (acquired 2020); pre-commercial (no paid driverless rides
  pending the exemption); test/early-rider ops in Las Vegas, SF, Austin, etc.
- **Safety (FACT):** **recalled 332 robotaxis (2026-12-23 filing)** — software could **cross the
  yellow centre line and stop in front of oncoming traffic near intersections** (bug surfaced
  2025-08-26 on a wide right turn into the opposing lane; **63 crossing instances** by Dec-5);
  **3rd software recall in ~8 months.** — https://techcrunch.com/2025/12/23/zoox-issues-software-recall-over-lane-crossings/
- **Strengths (INFER):** Amazon capital + logistics; clean-sheet vehicle designed for autonomy;
  no legacy-driver UX compromise.
- **Exploitable weaknesses:** **wrong-side / oncoming-lane entry + bad intersection stop-placement**
  (2nd FACT source for **SC-11**, and SC-08 family) — a directional/rule-barrier failure; recall
  cadence signals stack instability; FMVSS-exemption dependence = a distinct regulatory-risk surface.
- **What would beat them:** **H9 directional/lane barrier** (contra-flow excursion bounded by
  imagined oncoming occupancy, SC-11) + **H15** imagined oncoming risk before any lane commit, proven
  at ~261 M — the oncoming-lane class their recall exposes.

---

### Cross-field one-liner (INFER)
Nobody occupies our Pareto point — **hierarchical latent world model, ~261 M params, data-efficient,
real-time on Orin, in-loop imagination + guaranteed self-monitoring, regulation-native.** Run #4
(2026-07-11) hardens the case: the failures keep multiplying, moving **down-market to basic
competence** and **across operators** — a Tesla **fatality** (Houston) + a **second Tesla docket
(EA26002, 2.88 M veh, red-light/illegal-turn)**, **Zoox's third recall** (oncoming-lane), on top of
Avride's **basic-competence ODI** and Waymo's construction-zone recalls. The same two hard classes now
have **two major-operator FACT sources each** — red-light/rule-barrier (Waymo + Tesla → SC-14) and
oncoming-lane (Waymo + Zoox → SC-11) — which is exactly what turns a scenario into a public
excellence claim. Meanwhile "world model" spread to *everyone* (Momenta R7, Metis, GigaWorld-Policy,
shipped Alpamayo). Our moat is unambiguously **hierarchy + compute-normalized efficiency (CNCE) +
in-loop imagination (H15) + guaranteed self-monitoring (H11)** — none of which any tracked opponent
demonstrates together — proven on *their own* documented edge cases.

# OPPONENT PROFILES

> One page per opponent. Updated **deltas only** by the Opponent Analyzer agent; each keeps a
> **"What would beat them"** section current. Labels: FACT / CLAIM / INFER (G-O1).
> Created 2026-07-17 (v1). Last sweep: **run #3, narrative 2026-07-31 (real wall-clock 2026-07-17)** —
> deltas tagged **Δ 07-31**; prior **Δ 07-24** kept for history.

---

## Wayve  (UK · E2E foundation model · robotaxi + consumer L2+/L3)
- **Approach (FACT):** AV2.0 — a single end-to-end **camera+radar, mapless** foundation model; GAIA
  generative world models for synthetic data + **offline evaluation** (GAIA-3 = 15 B latent diffusion);
  LINGO VLA for natural-language rationales. Qualcomm production-ADAS partnership (Mar 2026).
- **Business (FACT):** **Series D $1.2 B (Feb 2026), $8.6 B** post-money; **$1.5 B** total secured incl.
  Uber milestone capital; investors Microsoft/NVIDIA/Uber + Mercedes/Nissan/Stellantis. London robotaxi
  trials 2026; supervised consumer autonomy from 2027; 10+ markets targeted.
- **Δ 07-31 (FACT):** Series-D extended with **+$60 M from AMD, Arm, Qualcomm** (multi-compute-platform
  breadth — a "plug-and-play across SoCs" play); **Tokyo pilot late 2026 (Nissan LEAF)** added to London.
  Still no signal that GAIA moves *in-loop* — it stays an offline data/eval factory (W-05 intact).
- **Δ 08-07 / real 07-20 (FACT):** **$85 M employee tender (2026-07-01)** — a **liquidity event, not
  new capital**; no technical or deployment delta in-window. W-05 unchanged. — https://wayve.ai/press/series-d/
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
- **Δ 07-31 (FACT/INFER):** now carries the **NHTSA first-responder directive (2026-07-08, W-09)** — ≥6
  incidents where responders had to physically move Waymo vehicles; a June natural-gas-explosion case;
  fix due end-July. Press frames a **"robotaxi ultimatum": Waymo fighting NHTSA and its distributor (Uber)
  at once** as Uber diversifies to Avride/Autobrains/Momenta/Wayve/Nuro. The failure surface is now
  **broad + federal** (W-01/W-03/W-09 all live), not a single recall. — https://businessmodelanalyst.com/nhtsa-robotaxi-ultimatum-waymo-uber/
- **Δ 08-07 / real 07-20 (FACT):** **the 2026-07-04 San Francisco breakdown** — dozens of vehicles
  stalled in post-fireworks gridlock at the **Presidio**; **64 vehicles** retrieved by staff or tow
  truck, several with **depleted batteries**; **unplanned road closures** a named contributor; one
  **occupied** vehicle **drove over a lit firework**; the SF mayor is calling for stricter rules. This
  is a **different failure class from every prior entry** — not perception, not a rule edge, but
  **mission-scale infeasibility and fleet self-interference** → **new W-10**, and it upgrades SC-08's
  evidence from a 2022 Cruise anecdote to a fresh large-N FACT. **Correction (P8):** the NHTSA
  first-responder deadline is for **presenting fixes in meetings**, not deployed fixes.
  — https://sfstandard.com/2026/07/05/waymo-sf-gridlock-fourth-of-july-2026/ , https://www.axios.com/2026/07/15/waymo-accountability-emergencies-nhtsa
- **Strengths (INFER):** scale, safety-engineering process, war chest, brand.
- **Exploitable weaknesses:** **construction/work-zone brittleness** (W-01, headline), **occlusion/VRU
  anticipation** (W-02), **rule-compliance edges** (W-03), **emergency-scene interference** (W-09),
  **fleet-scale mission/energy blindness** (W-10, new); map-dependence; cost/vehicle + geofence
  economics; no data-efficiency story.
- **What would beat them:** own the exact edge cases their recalls expose — work-zone imagination (H15) +
  inherent rule compliance (H9) + occlusion permanence (LOPS/OKRI, D9) — at hobbyist compute, framed for
  the new WP.29 regulation (H11/H9). **New, and honestly two-edged:** W-10 is the one weakness where we
  have **no counter either** — a strategic layer that reasons about mission feasibility (energy, network
  disruption) is *designed into* the 4-brain hierarchy and *implemented nowhere*. Either scope it or
  drop the claim; do not narrate it as a differentiator until something is measured.

## Pony.ai  (China + intl · robotaxi · NASDAQ)
- **Approach (FACT):** multi-sensor robotaxi stack; rapid geographic expansion.
- **Business (FACT):** Q1'26 total rev **$34.3 M (+145% YoY)**, robotaxi rev **$8.6 M (+395% YoY)**; 2026
  fleet target **3,500+**, 20+ cities; live in **Croatia (first EU commercial robotaxi)** + **Dubai**;
  9 countries.
- **Δ 07-24 (FACT):** robotaxi fleet now **exceeded 1,700 units** (toward the raised 3,500 target); added
  **Guangzhou**; **raised** 2026 robotaxi-revenue and fleet targets on record Q1. Growth real but the
  revenue-vs-fleet gap (W-06) is unchanged — $8.6 M robotaxi rev against a 1,700+ (→3,500) fleet.
- **Δ 07-31 (FACT):** **Q2'26 — 200+ Gen-7 robotaxis produced, revenue +76%**; weekly paid orders **+119%
  vs January**, Labor-Day daily orders **+544% YoY**. But Q1 **net loss widened to $50.4 M** — the
  scale-up is real, the **unit economics (W-06) are not improving**: order/fleet growth outruns margin.
- **Δ 08-07 / real 07-20 (FACT):** 2026 guidance reaffirmed — **>3,500 robotaxis across 20+ cities**,
  robotaxi revenue **>3.5× 2025**; driverless **light truck** launched (Apr'26); Uber/Verne **Croatia**
  and Stellantis **Luxembourg** e-Traveller vans progressing. No safety-docket delta in-window.
  **W-06 unchanged — fleet targets keep outrunning revenue.** — https://ir.pony.ai/news-events/press-releases
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
- **Δ 07-31 (FACT):** the reason for Uber's Munich switch is now explicit — **EU political resistance to
  sensitive Chinese key-tech** blocked the Momenta plan. Confirms an **EU-market-access weakness** for
  Momenta (and Pony) that our **Western/EU-clean data + compliance posture** turns into a wedge.
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
- **Δ 07-31 (FACT):** CLA ships as **MB.Drive Assist Pro** — L2++ point-to-point urban under supervision
  on **10 cameras / 5 radars / 12 ultrasonics**; Alpamayo positioned *explicitly* to solve the "long tail
  / rare weird edge cases." Family unchanged (10 B Nano → 32 B Super); **still no Nano-tier CNCE number**
  → the efficiency critique is unanswered on the metric that matters. "Solve the long tail" is the same
  claim we make — but at 10 B on-car vs our ~261 M; the CNCE contrast is the whole argument.
- **Δ 08-07 / real 07-20 (FACT):** no in-window delta. Confirmed on the product page: **Alpamayo 1 = a
  10 B chain-of-thought reasoning VLA with open weights**; **Alpamayo 2 Super = 32 B, "expected this
  summer"** (inference code on GitHub, weights on HF); **AlpaSim** fully open on GitHub. **Still no
  Nano-tier compute-normalized number** — our CNCE wedge stays open. **Watch item:** if the 2-Super
  release lands with a params-vs-benchmark table, the W-05 wedge narrows and we should publish our CNCE
  contrast first. — https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/
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
- **Strengths (INFER):** fleet-data scale, vertical integration, cost focus.
- **Exploitable weaknesses:** **no calibrated epistemic uncertainty** → confident-when-blind (W-04);
  camera-only sensing; monolithic E2E, no self-monitoring guarantee.
- **What would beat them:** H11 self-monitoring (degraded-visibility AUROC) + H15 epistemic σ + H2
  sensor-modality steering (radar fallback) — exactly the axis their open NHTSA case is about.

## Avride  (US/intl · Uber robotaxi partner · Yandex SDG lineage)  — emerging player (new 2026-07-24)
- **Approach (FACT/INFER):** self-driving stack spun out of Yandex's SDG group, deployed via **Uber** in
  US pilot markets (incl. Dallas). Modular AV stack (INFER — architecture not publicly detailed).
- **Business (FACT):** Uber robotaxi partner; scaling in Dallas and other US pilots alongside Uber's
  multi-vendor strategy (also Waymo, Momenta, Autobrains, NVIDIA).
- **Safety (FACT):** **NHTSA ODI investigation opened 2026-05-08** — **16 crashes + 1 minor injury**; ODI
  says all concern **"the competence of"** the system: **lane-changing, same-lane vehicle response, and
  stationary-object response.** — https://techcrunch.com/2026/05/08/uber-partner-avride-is-under-investigation-for-self-driving-crashes/
- **Δ 07-31 (FACT):** PE opened **2026-05-06**; 16 crashes span **Dec'25–Mar'26** (**≥9 Dallas**, rest
  Austin) in Hyundai **Ioniq 5** robotaxis; NHTSA video shows "unsafe lane changes into the path of other
  cars, **failing to avoid slow-moving vehicles ahead, and striking stationary objects.**" Damning: **all
  16 ran with a safety monitor in the seat, and only *one* attempted to intervene** → the failures are
  fast + systematic. Directly reinforces **SC-13** (this run's measured experiment).
- **Strengths (INFER):** Uber distribution + demand aggregation; Yandex AV heritage.
- **Exploitable weaknesses:** the ODI list is a **basic-competence** indictment (the cheapest, broadest
  surface) → **W-08 / SC-13**. Our H15 consequence-forward-model targets stationary-object/same-lane
  response directly (no detection prior to be wrong about).
- **What would beat them:** prove excellence on the *mundane* longitudinal/lateral tasks (SC-13) that
  their ODI flags — the least glamorous but most damning ground — at ~261 M params.

## Zoox  (US · Amazon · purpose-built robotaxi)  — emerging player (new 2026-07-31)
- **Approach (FACT):** purpose-built, no-manual-controls robotaxi (bidirectional "toaster"); multi-sensor.
- **Business (FACT):** unveiled a **production-intent** vehicle (Jun 2026); large-scale Bay-Area production
  starting; free rides in Las Vegas + SF, select Austin/Miami, testing in 6 more cities.
- **Status (FACT):** **gated on NHTSA approval** to operate up to **2,500** no-manual-controls vehicles
  commercially — its bottleneck is **regulatory** (FMVSS exemption), not (publicly) capability.
- **Δ 08-07 / real 07-20 (FACT) — first hard failure evidence, and it lands on our thesis:** Zoox
  **recalled 105 vehicles** (NHTSA notified **2026-07-08**, public **2026-07-17**) after a Las Vegas
  robotaxi **drove into thick smoke from an active fire** (**2026-06-20**), **failed to recognize the
  smoke**, then **suddenly braked, tried to turn, and halted** — inside the scene. The trace is the
  documented failure mode in one line: *drove in → failed to recognize → panic brake → stopped in the
  way.* — https://www.cnbc.com/2026/07/17/amazon-zoox-recalls-robotaxi-smoke.html
- **Exploitable weaknesses:** compute-heavy multi-sensor stack (W-05); **degraded-visibility /
  obscurant response (W-04) and emergency-scene interference (W-09) — now FACT-documented at recall
  grade**, which makes Zoox the **second operator** in the W-09 class and turns it from a company story
  into a class story; no public efficiency / self-monitoring / imagination story; commercial timing at
  the mercy of the FMVSS exemption decision.
- **What would beat them:** the CNCE + safety-case wedge, **plus SC-06/SC-05 directly** — a scene-level
  OOD flag that fires on smoke *as uncertainty* rather than waiting for an object to be classifiable.
  Honesty check (P8): that detector is **ours to prove** — SC-05's D8 probe has not yet cleared its bar,
  so this is a targeted opportunity, not a current advantage. **Promoted from "not a scenario-DB
  priority" to a primary SC-06 evidence source.**

## WeRide  (China/intl · robotaxi via Uber · NASDAQ)  — emerging player (new 2026-07-31)
- **Approach (FACT):** multi-sensor L4 robotaxi; heavy **Middle-East** footprint via Uber.
- **Business (FACT):** fully-driverless fare-charging via Uber in **Dubai (2026-03-31)**, plus Abu
  Dhabi/Riyadh; **1,200+ vehicle Middle-East commitment** by ~2027.
- **Exploitable weaknesses (INFER):** same W-05/W-06 (compute-heavy, thin economics); geopolitics limits
  Western data/market access (shared with Pony/Momenta).
- **What would beat them:** data-efficiency + cost-per-vehicle (H3/H7); Western/EU-clean posture.

## Nuro  (US · L4 stack supplier · Uber+Lucid)  — emerging player (new 2026-07-31)
- **Approach (FACT):** shifted from delivery pods to **licensing its L4 driver**; supplies the stack for
  the **Uber+Lucid** robotaxi (Lucid builds the car, Nuro the driver).
- **Business (FACT):** Uber deal expanded to **≥35,000 Lucid vehicles** (from 20 k, Jul 2025); Uber
  investment ~$500 M; first SF-Bay service later 2026.
- **Exploitable weaknesses (INFER):** supplier model = margin squeezed between Uber + Lucid; compute-heavy
  multi-sensor (W-05); no public efficiency/self-monitoring story.
- **What would beat them:** own the efficient + safety-case ground; our stack is a *driver* too — CNCE is
  the licensing-pitch differentiator.

---

### Cross-field one-liner (INFER)
Nobody occupies our Pareto point — **hierarchical latent world model, ~261 M params, data-efficient,
real-time on Orin, in-loop imagination + guaranteed self-monitoring, regulation-native.** Run #3 hardens
the case further and **from the regulator's own mouth**: NHTSA (Administrator Morrison) declared that
**"emergency scenes are not rare or extreme edge cases"** and failing them is a **"functional
insufficiency"** — i.e. the federal regulator now states the scenario-database thesis verbatim, gives
*every* operator a July deadline (W-09/SC-06), and calls AVs a "danger to the public." Meanwhile Waymo
fights NHTSA *and* Uber at once; the L4 field is a **multi-vendor Uber marketplace** (Waymo/Avride/
Autobrains/Momenta/Wayve/Nuro/WeRide + Zoox pending) where the **distribution moat is Uber's, not any
stack's**; "world model" is commoditized and **hierarchy is starting to appear** (SGDrive). Our moat is
unambiguously **hierarchy + compute-normalized efficiency (CNCE) + in-loop imagination (H15) + guaranteed
self-monitoring (H11)** — none demonstrated together by any tracked opponent — proven on *their own*
FACT-documented failures, at ~261 M params where they run 10–32 B.

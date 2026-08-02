# OPPONENT PROFILES

> One page per opponent. Updated **deltas only** by the Opponent Analyzer agent; each keeps a
> **"What would beat them"** section current. Labels: FACT / CLAIM / INFER (G-O1).
> Created 2026-07-17 (v1). Last sweep: **run #5, 2026-08-02 (real date — the discipline's narrative
> clock is retired, see STATE.md)** — deltas tagged **Δ run #5 / 2026-08-02**. Earlier tags **Δ 08-07**
> (= run #4, real 2026-07-20), **Δ 07-31** (= run #3, real 2026-07-17) and **Δ 07-24** are narrative-clock
> dates kept verbatim for history; **order them by run number, not by date**.
>
> ⛔ **Run #5 contains a RETRACTION** (Momenta / "EU market-access weakness"). Read that entry before
> reusing any geopolitical framing.

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
- **Δ run #5 / 2026-08-02 — FULL DEEP-DIVE: `Research/2026-08-02-wayve-deep-dive.md`.** Read that note
  before quoting anything here; the summary below is compressed from it.
- **Architecture, corrected (FACT) — Wayve is TWO stacks, not one.**
  **(a) On-car:** a *flat* end-to-end policy, camera-first + radar, mapless, licensed to OEMs (they do
  not operate fleets). **LINGO-2** (Apr'24) = the first closed-loop **vision-language-action** driving
  model tested on public roads — emits a path *and* a running natural-language commentary, and accepts
  language instruction. **Gen 3** runs on **NVIDIA DRIVE AGX Thor, up to 2,000 FP4 TFLOPS**, targeting
  **eyes-off L3 + driverless L4**. ⚠️ **The on-car parameter count has NEVER been disclosed** (checked
  four ways); only the *offline* models' sizes are public. A **75 W** deployable figure exists as
  **CLAIM** (secondary source only).
  **(b) Offline factory:** FIERY (2021) → MILE (2022, model-based imitation w/ latent WM — their closest
  ancestor to *our* design) → **GAIA-1** 9 B (2023) → **Ghost Gym** (Dec'23, **closed-loop neural
  simulator**: neural renderer + simulated robot car + vehicle dynamics, action fed back) → **PRISM-1**
  (Jun'24, 4D photorealistic reconstruction **from cameras only, no LiDAR**) → **GAIA-2** (Mar'25,
  multi-view, arXiv 2503.20523) → **Rig3R** (Oct'25, **NeurIPS Spotlight**, multi-camera geometry
  conditioned on **rig metadata**, +17–45 % over baselines, best on **unseen rig configs**) → **GAIA-3**
  (**2 Dec 2025**, 15 B latent diffusion) → **LA-Pose** (**30 Apr 2026**, arXiv 2604.27448).
- **★ THE LOAD-BEARING FACT (FACT, Wayve's own page):** GAIA-3 is **explicitly for offline evaluation and
  safety validation, NOT real-time in-vehicle deployment.** Our long-standing "no signal that GAIA moves
  in-loop" is no longer an inference from absence — **they say so.** ⇒ the honest framing is **"their
  world model VALIDATES; ours DRIVES."** GAIA-3 detail: video tokenizer 2× GAIA-2's, **5× compute, ~10×
  data, 9 countries / 3 continents**; conditioned on ego action, agents' 3D boxes, weather/time, road
  attributes; **"World-on-Rails" perturbation** (move ego, hold the scene); claims **"reduced
  synthetic-test rejection rates fivefold"** and that it **"reliably predicts relative policy
  performance."** ⚠️ **No paper — blog + press release only.** — https://wayve.ai/thinking/gaia-3/
- **★ LA-Pose = OUR H7, PUBLISHED, AT 10.2 M CLIPS (FACT).** An **inverse-dynamics model** on **10.2
  million unlabelled driving clips** learns **latent actions** (never told speed or heading); they
  cluster into straight/left/right/stopped **with zero pose labels**; a light head then reads **camera
  pose incl. field-of-view and metric scale** in one forward pass. **>10 % over feed-forward SOTA** on
  Waymo + **PandaSet (unseen → zero-shot)**. **A 50-d latent bottleneck beat a 1,536-d one despite worse
  video reconstruction.** Known limit: degrades in reverse motion. — https://arxiv.org/abs/2604.27448
- **Multi-country generalization, their own numbers (FACT):** UK→US needed **500 h** of incremental US
  data over 8 weeks to reach UK-equivalence (100 h → "5×", 500 h → "40×"); **Germany zero-shot "3×
  better"** than the initial US deployment; a new vehicle platform → "8×" after **100 h**. ⚠️ **All
  relative multipliers off an undisclosed baseline — no miles, no intervention rate** (→ W-11).
- **Δ run #5 — deployment reality check (FACT):** first commercial launch = **Ford Mustang Mach-E on Uber
  London WITH A SAFETY DRIVER**, pending regulator go-ahead (waitlist open); UK fully-driverless targeted
  **2027**; **Waymo targets a London service Q3 2026, possibly driverless from the start.** Nissan ADAS
  from 2027; **Wayve+Nissan+Uber Tokyo pilot late 2026** (DRIVE Hyperion). ⚠️ **Do not quote the
  TechCrunch "$1.8 B" headline** — its own body says $1.2 B + $300 M conditional = **$1.5 B at $8.6 B
  post**, which is our figure. The **NVIDIA $500 M LOI is 2025-09-18** and is superseded.
- **Strengths (INFER, sharpened):** ① **a closed-loop camera-only evaluation stack that actually works**
  (Ghost Gym + PRISM-1 + GAIA-3) — *the capability we are most blocked on*; ② real multi-domain
  generalization with a measured data cost; ③ research depth (NeurIPS Spotlight, active arXiv output);
  ④ distribution + capital + **multi-SoC optionality** (NVIDIA/AMD/Arm/Qualcomm).
- **Exploitable weaknesses:** ① **the WM is an evaluator, not a driver** — no hierarchy, **no
  imagination at decision time**, no separable strategic layer on-car (W-05); ② **no denominators
  anywhere** — no safety metrics, thresholds, runtime-monitor or OOD methodology on their safety page;
  generalization quoted only as multipliers (**W-11**, their most technical exemplar); ③ **LINGO
  introspection is narrative, not numeric** — a commentary track is not a runtime monitor with a
  threshold (H11 gap); ④ they bet **explicitly against scenario enumeration** ("Safety 2.0"), which
  NHTSA's W-09 *"functional insufficiency"* finding cuts against; ⑤ **behind Waymo in their own home
  market**, safety-driver-first; ⑥ camera+radar with no published epistemic-uncertainty mechanism (W-04
  exposure — **INFER from absence at one source class; probe again before it carries weight**);
  ⑦ **an internal tension**: their own 50-d LA-Pose result argues reconstruction fidelity is *not* what
  driving needs, against their 15 B reconstructive flagship.
- **What would beat them:** the WM **as the on-car reasoning substrate at decision time** (H15 in-loop
  imagination) inside a **planning-time hierarchy** (H1), at 1–2 orders less compute with a **published
  CNCE** number they do not report — plus the one thing nobody in the field publishes: **self-monitoring
  with a threshold (H11)**. ⚠️ **P8 honesty:** H11 is our *widest opportunity*, **not** a current
  advantage — our own D8 AUROC bar (>0.85) is unmet. State it that way.

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
- **Δ run #5 / 2026-08-02 (FACT) — the pressure moves from regulator to legislature:** Rep. **Kevin
  Mullin (D-Calif.)** introduced the **"AV Emergency Response Coordination Act"** (week of 2026-07-28) —
  first-responder protocols, a **24 h hotline** for public officials, NHTSA **minimum standards**, and
  **city authority to geofence AVs during emergencies**. SF Fire Chief **Dean Crispen** cites robotaxis
  blocking fire stations and ambulance facilities. Same coverage adds a **second** fleet-scale stall: a
  **December power outage stranded dozens of Waymo vehicles** (→ W-10 is now two instances, two
  different triggers). — https://techcrunch.com/2026/07/28/waymo-robotaxi-operators-face-fresh-scrutiny-over-emergency-response-failures/
- **Δ run #5 / 2026-08-02 (FACT/CLAIM) — the economics, for the W-06 contrast:** ~**3,700** Jaguar I-PACE
  (Feb 2026); ~**500,000 paid autonomous rides/week** (May 2026, up from 250 k in Apr 2025); **~$355 M
  annualized revenue** (Feb 2026, Sacra estimate — **CLAIM**, third-party modelling, not a Waymo filing);
  ~**$15–17** average fare, ~**20 trips/vehicle/day**. **INFER:** at ~$355 M/yr against Alphabet-scale
  capital and a 3,700-vehicle fleet, the revenue *per vehicle* (~$96 k/yr gross) is real but thin against
  a multi-sensor, map-maintained, remote-assisted stack — W-06 is not only a Chinese-operator story.
  — https://sacra.com/c/waymo/
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
- **⛔ Δ run #5 / 2026-08-02 — RETRACTION: the "EU-market-access weakness" INFER above is FALSIFIED.**
  On **2026-07-29** Momenta received a **Germany-wide Level-4 testing approval from the KBA**
  (Kraftfahrt-Bundesamt) — cleared for urban autonomous operation **across the whole country**, and it
  says it is the **first Chinese firm** to hold such an authorization. It underpins the **Munich**
  robotaxi deployment. In the same week **Uber increased its stake** in Momenta. Two of the three planks
  of the 07-31 read collapse: the EU did not block Momenta on key-tech grounds, and Uber did not walk
  away. **Do not reuse "EU market access is a Chinese-vendor weakness" in any deck.**
  — https://cnevpost.com/2026/07/29/momenta-cleared-test-robotaxis-across-germany/
  , https://cleantechnica.com/2026/07/29/momenta-to-test-robotaxis-across-germany-uber-invests-more/
  **Root-cause class (for RETRACTION_LOG): single-source geopolitical INFER promoted to a market-structure
  conclusion.** One vendor-switch datum was read as a policy regime. The switch was real; the *reason*
  was inferred, never sourced to a regulator or to either company, and it was then reused as a standing
  wedge. **Rule reinforced (Operating Standard #2): an inference about *why* a competitor lost a deal
  needs a second, independent probe before it becomes a strategic asset — and a competitor's regulatory
  posture must be re-checked at the regulator, not at the press.**
- **Δ run #5 / 2026-08-02 (FACT) — platform breadth:** Momenta confirmed it is running **robovans in
  Suzhou** for delivery (2026-07-27), extending its world-model platform across **robotaxi + robovan +
  robotruck**. **INFER:** this is the same "one platform, many form factors" story we tell about a
  hierarchy that separates strategic/tactical/operative — from a company with a chip, an OEM channel and
  a listing. Breadth is no longer a differentiator; the *safety case* and the *compute envelope* are.
  — https://cnevpost.com/2026/07/27/momenta-confirms-robovan-entry/
- **Exploitable weaknesses:** **strategy split** now locked by public-market scrutiny; opaque safety case;
  R7 is a *generative RL* WM (compute/data-heavy) with no hierarchy/self-monitoring claim. **Geopolitics
  is REMOVED from this list** — the Germany-wide permit is direct evidence against it.
- **What would beat them:** a single coherent efficiency+safety thesis (vs their L2++/L4 straddle) with a
  transparent, regulation-native safety case (H9/H11). Now with **less** room to lean on market access:
  they are inside the EU with a national-scale permit, so the contest is technical, not political.

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
- **Δ run #5 / 2026-08-02 (FACT) — checked at NVIDIA's own text, not press:** the Alpamayo 2 technical
  post (launch **GTC Taipei 2026-06-01**) describes a **32 B VLM backbone, "3× the number of parameters
  as prior Alpamayo models"**, adds **full 360° surround perception** and **Meta-Action** outputs, and
  claims "state-of-the-art performance in multiple aspects including reasoning quality, trajectory
  accuracy, alignment" — **with no benchmark table, no latency number, no compute figure, and no Nano
  tier reported.** Weights/inference code "**coming summer 2026**". **W-05 wedge re-verified OPEN at the
  primary source** (third consecutive run). Two new companion assets: **AlpaGym**, an open-source
  high-throughput **closed-loop RL** framework (GRPO, default reward functions, single-GPU→multi-node,
  "release before mid-June"), and **quantization scripts "coming soon"** — **INFER:** shipping
  quantization tooling alongside a 3× parameter jump is NVIDIA conceding the deployment-cost problem in
  engineering while not conceding it in the benchmark table. **AlpaGym is the actionable item — but it
  is NOT new to us:** Tools & DevEnv logged *"AlpaSim/AlpaGym = Phase-1 cloud (40–60 GB VRAM)"* on
  **2026-07-06** (`PROJECT_STATE.md` §5). What changed is that it is **released, open-source, GRPO-based,
  and stated to scale from a SINGLE GPU** — which contradicts the 40–60 GB figure on record and makes it
  re-testable on the **A40 48 GB we already have**. Relevant because AlpaSim was a NO-GO on the eval pod
  and CARLA pixels are host-blocked → **Tools & DevEnv: re-check the VRAM figure before any further
  spend on a graphics-capable host.** — https://huggingface.co/blog/nvidia/nvidia-alpamayo-2
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
- **Δ run #5 / 2026-08-02 (FACT) — THE REGULATORY BOTTLENECK IS GONE, AND THE CAPABILITY GAP IS NOT.**
  NHTSA **granted the commercial exemption on 2026-07-30** (Federal Register **2026-07-31**): Zoox may
  **charge for rides** in a vehicle with **no steering wheel, no pedals, no driver's seat** — the **first
  such US authorization for a purpose-built AV** — for **up to 2,500 vehicles over two years**. It landed
  **the same day** the NHTSA first-responder deadline expired **with no public resolution**, and **six
  weeks after** Zoox's own smoke recall. **INFER, and it is the strategic read of this whole run:** the
  gate on scaling is not the unfixed capability. Our thesis cannot be "the regulator will stop them" —
  it has to be that **the capability itself is worth more than the exemption**, demonstrated on the
  scenarios where they are documented to fail. — https://fortune.com/2026/07/31/zoox-robotaxi-steering-wheel-safety-data-gap/
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
- **Exploitable weaknesses (INFER):** same W-05/W-06 (compute-heavy, thin economics). ⛔ **The clause
  "geopolitics limits Western data/market access (shared with Pony/Momenta)" is WITHDRAWN (run #5)** —
  Momenta's Germany-wide KBA L4 permit (2026-07-29) is direct counter-evidence to the shared premise.
  It was never separately sourced for WeRide; it was inherited from the same single-source INFER that
  §2.2 of the run-#5 note retracts. **Do not restate it without a WeRide-specific regulator source.**
- **What would beat them:** data-efficiency + cost-per-vehicle (H3/H7). *(The "Western/EU-clean posture"
  half of this line is withdrawn with the clause above — it rested on the same retracted premise.)*

## Waabi  (Canada/US · simulation-first L4 · trucks + robotaxi)  — emerging player (new run #5, PARTIAL)
> ⚠️ **Stub, deliberately thin.** Surfaced late in run #5's sweep from a single search pass; sourcing is
> **one step short** of the bar the other profiles meet. Completing it is a run-#6 backlog item. Do not
> quote this entry in a deck until it carries per-fact primary links.
- **Business (FACT/CLAIM):** raised **~$1 B**, including a **$750 M Series C co-led by Khosla Ventures
  and G2 Venture Partners**, for self-driving **trucks** and robotaxis. Amount and co-leads are FACT
  from reporting; the split and current valuation are unverified here.
  — https://www.barchart.com/story/news/37268532/waabi-secures-us1-billion-in-funding-as-it-pushes-self-driving-trucks-robotaxis
- **Why it matters to us (INFER):** Waabi is the most **architecturally adjacent** unprofiled player —
  a **simulation-first** thesis ("Waabi World" generative simulator) that claims capability from
  *simulated* rather than fleet-scale real miles. That is the closest public analogue to our own
  data-efficiency argument (H3/H7), and it targets the same weakness (W-06) we attack. Trucks-first also
  sidesteps the urban emergency-scene surface (W-09) that dominates the robotaxi field.
- **What to check next run:** parameter counts / compute disclosure (W-05, CNCE comparability); whether
  Waabi World is used **in the loop** or only as a data/eval factory (the same question that separates
  us from Wayve's GAIA); any published sim-to-real transfer number.

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

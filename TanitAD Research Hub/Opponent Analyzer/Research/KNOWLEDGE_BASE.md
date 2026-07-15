# KNOWLEDGE_BASE — Opponent Analyzer

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`
> Labels: FACT / CLAIM / INFER (G-O1). Full analysis: dated notes in this folder.
> **Ordering note:** the entries dated **2026-07-15 are run #3** and are the *newest* (the hub's
> narrative clock runs ahead of wall-clock, so run #3's real date sorts below run #2's 07-24 stamp).

- [2026-07-15] [Opponent Analyzer] INFER (design-oracle, P8) — **SC-13 Stationary-Lead** shipped
  (forward-simulated oracle, real kinematics, **16/16 tests**): in the documented late-classification
  regime **`detection_reactive` COLLIDES** (min-TTC 0.09 s, LAL-v2 **−0.50 s**) while **`imagination`
  anticipates** (LAL-v2 **+2.30 s**, min-TTC 1.83 s, no collision, **OKRI −73 %**); **collision rate
  0.60 (reactive) / 0.00 (imagination)** over the classification-range sweep, invariant to the
  competence knob — impact: **W-08 / SC-13 → H15/A9** — see `2026-07-15-opponent-sweep-run3.md`
- [2026-07-15] [Avride/NHTSA] FACT — ODI language sharpened (vs the 07-24 entry): the crashes involve
  *"responding to **stationary objects partially obstructing the lane ahead**"* and NHTSA flags
  **"excessive assertiveness and insufficient capability … may also constitute traffic safety
  violations,"** all **under a safety monitor** (Dallas + Austin, Jan–Mar) — impact: validates SC-13
  geometry (partial in-lane obstruction); **W-08** — https://thenextweb.com/news/avride-uber-robotaxi-crashes-nhtsa-investigation
- [2026-07-15] [NHTSA/Waymo/Tesla] FACT — NHTSA letter (**2026-07-08**) warns Waymo **and** Tesla over a
  **"clear pattern"** of **first-responder / emergency-scene interference**, ~one-month fix deadline;
  trigger included Waymo robotaxis **stalling in SF's July-4 gridlock** (some towed, dead batteries) —
  impact: **new W-09 / SC-15** (emergency-scene interference); fresh SC-08 evidence; A9/D8 + H1 fallback
  — https://www.benzinga.com/markets/tech/26/07/60351872/nhtsa-warns-autonomous-vehicle-companies-over-clear-pattern-of-first-responder-interference
- [2026-07-15] [Pony.ai] FACT — reports **break-even operations in one city (Guangzhou)**, raised goal to
  **10,000+ vehicles**, **Bolt** EU ride-hail partnership — impact: **honesty delta on W-06** (partially
  blunts "thin unit economics"; shift the argument to CNCE + data-efficiency slope, P8) — https://thenextweb.com/news/pony-ai-lifts-3500-robotaxi-fleet-target-2026
- [2026-07-15] [Tesla/NHTSA] FACT — new federal probe into a **fatal Model-3 Autopilot crash** (Katy, TX;
  76-yo killed); Tesla also in the first-responder-interference pattern — impact: **W-04 field weight;
  W-09** — https://www.cnbc.com/2026/06/22/tesla-nhtsa-model-3-crash-autopilot-katy-texas.html
- [2026-07-15] [arXiv] FACT/INFER — latent-WM surge continues: **Latent-WAM** (2603.24581),
  **DriveWorld-VLA** (2602.06521), **DriveFuture** (2605.09701), **World Models survey** (2606.00133).
  Reinforces "latent WM = table stakes"; none reports hierarchy + in-loop imagination + self-monitoring +
  a compute-normalized metric together — impact: moat read unchanged; deep-read Latent-WAM/DriveWorld-VLA
  next — https://arxiv.org/abs/2603.24581

- [2026-07-24] [Avride/NHTSA] FACT — NHTSA ODI opened an investigation (**2026-05-08**) into **Avride**
  (Uber robotaxi partner, Yandex SDG lineage): **16 crashes + 1 minor injury**, all tied to **"the
  competence of"** the system — lane-changing, same-lane vehicle response, stationary-object response —
  impact: **new opponent; W-08 → H15/A9; SC-13** stationary-object/same-lane spec — https://techcrunch.com/2026/05/08/uber-partner-avride-is-under-investigation-for-self-driving-crashes/
- [2026-07-24] [Waymo/NHTSA] FACT — the construction-zone recall (26E035) was Waymo's **2nd in ~1 month**;
  it **pulled all robotaxis from highways 2026-05-19**; filing names the mechanism (mis-prioritizing
  hazard-avoidance / not recognizing the work zone). Separately a Waymo **ran a red light in Dallas** amid
  a new federal probe — impact: **W-01 enrich; W-03 family → SC-14** red-light barrier — https://techcrunch.com/2026/06/18/waymo-recalls-nearly-4000-robotaxis-to-stop-them-driving-into-highway-construction-zones/
- [2026-07-24] [Tesla/NHTSA] FACT — FSD probe **upgraded to Engineering Analysis 2026-03-18**, **~3.2 M
  vehicles**, **9 crashes / 1 fatality / 2 injuries**; the **"degradation-detection" feature** fails to
  flag impaired cameras until immediately pre-crash. Tesla also **unredacted 17 Austin robotaxi ADS
  incidents** (2 with teleoperators); Miami launched into rain 2026-07-03 — impact: **W-04 → H11/H15/H2
  strongest field validation** — https://electrek.co/2026/03/19/nhtsa-upgrades-tesla-fsd-visibility-investigation-3-2-million-vehicles/
- [2026-07-24] [arXiv] FACT/INFER — **Metis** (2606.15869, Fudan/HKU/Tongji/Li Auto, subm. 06-14):
  "efficient world-action model" — Mixture-of-Transformers with separate video-gen + action experts and
  an **asymmetric attention mask that lets the action head skip generative rollout at inference** (its
  efficiency lever ≈ our latent path). SOTA NAVSIM navhard/navtest + CityWalker. **But no hierarchy, no
  in-loop imagination, no self-monitoring, and NO param count / compute-normalized metric** → not a true
  CNCE competitor — impact: sharpens H1/H3/H5/H15/H11 wedge; publish params+CNCE it doesn't — https://arxiv.org/abs/2606.15869
- [2026-07-24] [Momenta/Autobrains/NVIDIA] FACT — **Momenta listed HK 2026-07-08** (~$8.9 B cap; Mercedes+
  BYD cornerstones); shipped **R7 RL World Model (Apr'26)** + **X7** chip (SAIC-VW ID.ERA 9X). **Uber's
  Munich robotaxi pilot shifted to Autobrains+NVIDIA (2026-06-02)** → Autobrains stepping ADAS→L4. **NVIDIA
  Mercedes-Benz CLA** ships the full Alpamayo stack (US, this quarter); family **10 B (Nano) → 32 B
  (Super)**; **AlpaSim** open-source — impact: "world model" now table stakes (H0/H6); Autobrains watch
  escalated; AlpaSim = usable sim asset (Tools&DevEnv) — https://www.electrive.com/2026/06/02/uber-and-autobrains-to-partner-on-munich-robotaxi-pilot-project/
- [2026-07-24] [Opponent Analyzer] INFER (design-oracle, P8) — **Stop-Arm Gate** scenario (SC-04, W-03)
  shipped: H9 **violation rate rule_barrier 0.0 vs soft_prior 1.0** over the free-path temptation sweep;
  the barrier is invariant to temptation while the soft prior's line-crossing speed grows 3.0→9.6 m/s;
  OKRI toward the occluded child 80% lower at 4 B vs 15 B params (**11/11 offline tests**) — impact:
  **H9/H15**, first violation-rate contrast — see `2026-07-24-opponent-sweep-w3.md`
- [2026-07-17] [Waymo/NHTSA] FACT — Waymo recalled **3,871 robotaxis (2026-06-18)** for driving into
  freeway **construction zones** (unrecognized ramp-closure signs; drove between lane-closure cones);
  freeway autonomy suspended, 20+-city expansion frozen — impact: **W-01 → H15/H9/H1**; drives the new
  Work-Zone Phantom scenario — https://www.cnbc.com/2026/06/18/waymo-nhtsa-voluntary-recall-robotaxis-entered-freeway-construction-zones.html
- [2026-07-17] [NTSB] FACT — HWY26FH008 (2026-01-23, Santa Monica): Waymo I-Pace struck a 9-yo
  pedestrian in a school zone; ADS **detected + braked heavily but late**; NTSB examining sudden-VRU
  anticipation. Distinct from the separate school-bus stop-arm probe (CLAIM: one case = human error)
  — impact: **W-02/W-03 → H15/H9 (LOPS/OKRI/LAL, D9)** — https://www.ntsb.gov/investigations/Pages/HWY26FH008.aspx
- [2026-07-17] [NHTSA SGO] FACT — Jun'25–May'26 window: 825 ADS incidents; **Waymo 697** (1 fatality,
  23 hospitalizations, 51 minor, 613 property-only) — impact: weakness-evidence corpus, H0/H6 — https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting
- [2026-07-17] [CA DMV] FACT — Dec'24–Nov'25: Waymo 19,234 mi/disengagement (3.35 M mi), Zoox 60,682;
  DMV **proposes to replace the disengagement metric in 2026** — impact: **W-07 narrative**, aligns with
  Benchmarks&Eval open-loop⊥closed-loop — https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/disengagement-reports/
- [2026-07-17] [Tesla/NHTSA] FACT — NHTSA engineering analysis (Mar 2026, pre-recall): camera-only FSD
  **fails under degraded visibility (glare/airborne obscurants)**; robotaxi in Miami (2026-07-03), TX
  fleet ~42 vs Waymo 577 — impact: **W-04 → H11/H15/H2** — https://www.automotiveworld.com/news/tesla-robotaxi-fleet-hits-25-as-musk-defers-scale-to-fsd-v15/
- [2026-07-17] [Wayve] FACT — Series D **$1.2 B (Feb'26), $8.6 B** post-money ($1.5 B total); **GAIA-3 =
  15 B latent-diffusion WM for offline eval**; on-car AV2.0 = monolithic E2E cam+radar, mapless
  (corrects kickoff "$2.8 B") — impact: **W-05 → H1/H3/H5 (CNCE)** — https://wayve.ai/press/series-d/ , https://wayve.ai/thinking/gaia-3/
- [2026-07-17] [NVIDIA] FACT — **Alpamayo 2 Super = 32 B reasoning VLA** on Cosmos (Chain-of-Causation);
  open dataset **1,700+ h / 25 countries / 2,500+ cities** — impact: **frenemy/supply chain; W-05 foil**;
  32 B on-car = anti-efficiency — https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development
- [2026-07-17] [Pony.ai] FACT — Q1'26 total rev $34.3 M (+145% YoY), robotaxi rev $8.6 M (+395% YoY) vs
  3,500-vehicle target; Croatia (first EU commercial robotaxi) + Dubai — impact: **W-06 → H3/H7** (thin
  unit economics) — https://mlq.ai/news/v2/pony-ai-q1-revenue-more-than-doubles-to-343m-as-robotaxi-sales-surge-nearly-fivefold/
- [2026-07-17] [Momenta] FACT — HK IPO ~$752 M, ~$9 B valuation (trading 2026-07-08), GM+Tencent;
  two-leg L2++/robotaxi; Abu Dhabi + Munich 2026, Uber L4 pilot — impact: strategy-split weakness locked
  by public markets — https://technode.com/2026/06/30/momenta-launches-hong-kong-ipo-with-gic-fidelity-and-blackrock-as-cornerstone-investors/
- [2026-07-17] [Autobrains] FACT — $140 M+ funding (BMW/Toyota/Continental/Temasek); "Liquid AI" +
  agentic, edge-cases-with-less-compute on standard sensors; L2+/ADAS only — impact: **narrative-overlap
  watch → own L4/WM ground + pre-empt CNCE** — https://autobrains.ai/about-us/
- [2026-07-17] [arXiv] FACT/INFER — 2026 **latent-WM/JEPA-for-driving surge** (survey 2603.09086;
  Drive-JEPA 2601.22032; Metis "efficient world-action model" 2606.15869; GraphWorld 2606.16274; IDOL
  2605.31476; +more) — impact: H3 externally validated but **"latent WM" no longer differentiating** →
  moat = hierarchy+efficiency+imagination+self-monitoring; deep-read Metis next run — https://arxiv.org/abs/2603.09086
- [2026-07-05] [kickoff] Initial research baseline for all hypotheses established; discipline agenda
  seeds defined — impact: all — see `../../INITIAL_RESEARCH_SYNTHESIS.md`

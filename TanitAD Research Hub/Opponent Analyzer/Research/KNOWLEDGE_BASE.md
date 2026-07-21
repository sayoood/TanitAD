# KNOWLEDGE_BASE — Opponent Analyzer

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`
> Labels: FACT / CLAIM / INFER (G-O1). Full analysis: dated notes in this folder.

- [2026-08-07 · run #4, real 2026-07-20] [Zoox/NHTSA] FACT — **Zoox recalled 105 vehicles** (NHTSA
  notified **2026-07-08**, public **2026-07-17**): on **2026-06-20** a Las Vegas robotaxi **drove into
  thick smoke from an active fire**, **failed to recognize the smoke**, then **suddenly braked, tried
  to turn, and halted** — inside the scene — impact: **W-09 becomes CROSS-OPERATOR** (Waymo + Zoox +
  federal directive = a *class*, not a company story) and **fuses W-09 with W-04** (smoke is obscurant
  *and* emergency cue → **one shared OOD head**); drove the SC-06 authoring this run — https://www.cnbc.com/2026/07/17/amazon-zoox-recalls-robotaxi-smoke.html
- [2026-08-07 · run #4] [Waymo/SF] FACT — **2026-07-04 San Francisco breakdown**: dozens of Waymos
  stalled in post-fireworks gridlock at the **Presidio**; **64 vehicles** retrieved by staff/tow, some
  with **depleted batteries**; **unplanned road closures** a named contributor; one **occupied** car
  **drove over a lit firework**; SF mayor demanding stricter rules — impact: **new W-10** (fleet-scale
  mission/energy/network-disruption blindness, marked **`no-counter-yet`**); **SC-08 evidence upgraded**
  from the 2022 Cruise anecdote to a fresh large-N FACT — https://sfstandard.com/2026/07/05/waymo-sf-gridlock-fourth-of-july-2026/
- [2026-08-07 · run #4] [TanitAD / eval pod] **MEASURED (not oracle) — NEGATIVE RESULT (P8)** — first
  SC-13 test on our own checkpoint, and the **pre-registered falsifier FIRED on replication**.
  flagship-30k, future actions **withheld**, speed confound controlled two ways. **In-domain
  (PhysicalAI, 3,241 anchors, n=23 events):** braking starting **2–3 s out (outside the 2 s rollout)**
  detected at **AUROC 0.72–0.74** vs reactive floor **0.43**. **Cross-corpus (comma2k19, 8,384 anchors,
  n=45): held 0.54–0.61 ≈ vision-blind 0.55–0.61 ≈ reactive 0.55–0.59 — indistinguishable.** Confounds:
  comma2k19 is out-of-domain and **CV beats the model there** (1.302 vs 1.874 m ADE), plus it is
  highway (29.1 m/s cruise) where CV is near-unbeatable — a failed replication, not a clean refutation
  — impact: **H15 evidence moves AGAINST the open-loop anticipation claim**; **SC-13 → live-measured
  (falsifier fired)**; the oracle collision-rate contrast is now **unsupported** and must stay out of
  external narrative; next test = in-domain volume + an arm that beats CV on the target corpus — see
  `2026-08-07-opponent-sweep-w5.md` §1, archive `Implementation/sc13-real-probe/`
- [2026-08-07 · run #4] [arXiv] FACT/INFER — **HWM, "Hierarchical Planning with Latent World Models"**
  (**2604.03208**, Zhang/Terver/Zholus et al., Apr'26 rev Jun'26): world models at **multiple temporal
  scales in one latent space**, long-horizon predictions used as **subgoals for the short-horizon model
  via latent matching**, no rewards/hierarchical policy, **up to 3× less planning compute**.
  **Planning-time hierarchy — our H1 claim — is now published**, though on **manipulation/maze, not
  driving**, with **no param count** and **no self-monitoring/OOD guarantee** — impact: H1 must be
  positioned as hierarchy+efficiency+in-loop-imagination+self-monitoring *on driving*; also the closest
  published relative of the **v3** direction (DINO-WM lineage) → **Architecture deep-read, top
  priority** — https://arxiv.org/abs/2604.03208
- [2026-08-07 · run #4] [Opponent Analyzer] INFER (design-oracle, P8) — **Emergency-Scene** scenario
  (SC-06, W-09) shipped, **16/16 tests**: corridor **incursion rate 0.0 (yield) vs 0.2 (rule-literal)**;
  **blockage 0.0 s vs 2.54 s** (12.7 s at thick smoke); **detection lead time +5.70 s vs +2.84 s**
  (−0.10 s at thick smoke). Mechanism: the obscurant collapses **object** range **90→13.5 m** while
  **scene**-level OOD range falls only **80→68 m**. **The failure is a CLIFF not a slope** → graded
  obscurant sweeps are mandatory — impact: **H11/H15/A9**; **blocked on SC-05's D8 detector**, which is
  currently failing — see `2026-08-07-opponent-sweep-w5.md` §2
- [2026-08-07 · run #4] [NHTSA/Wayve/Pony/NVIDIA] FACT (deltas) — the first-responder deadline is for
  **presenting fixes in meetings, NOT deployed fixes** (correction, do not overstate); **Wayve $85 M
  employee tender (07-01)** = liquidity, not new capital; **Pony** reaffirms **>3,500 robotaxis / 20+
  cities** 2026 (**W-06 unchanged** — fleet targets still outrun revenue); **NVIDIA**: Alpamayo 1 =
  **10 B** open weights, **2 Super = 32 B** "expected this summer", AlpaSim open — **still no Nano-tier
  CNCE number**, our W-05 wedge stays open — https://www.axios.com/2026/07/15/waymo-accountability-emergencies-nhtsa
- [2026-07-31 · run #3, real 2026-07-17] [NHTSA] FACT — ODI issued a formal **ADS-developers letter
  (2026-07-08)** demanding every AV developer fix, **by end of July 2026**, a **"clear pattern"** of
  robotaxis interfering with first responders (driving into emergency scenes; blocking ambulances/fire;
  failing to recognize **flashing lights, flares, smoke, fire, cones**). ≥6 incidents through Mar 2026
  needed responders to **physically move Waymo vehicles**. Morrison: **"functional insufficiency"**;
  **"Emergency scenes are not rare or extreme edge cases"** — impact: **new W-09; SC-06 elevated →
  H15/H11/A9/H9**; strongest external endorsement of the scenario-DB thesis (H0/H6) — https://techcrunch.com/2026/07/08/feds-demand-autonomous-vehicle-companies-stop-interfering-with-first-responders/
- [2026-07-31 · run #3] [Opponent Analyzer] INFER (design-oracle, P8) — **Stationary-Lead** scenario
  (SC-13, W-08) shipped: over the classification-ambiguity sweep {0…1}, **collision rate imagination 0.0
  / detection-reactive 0.4**; **braking-onset lead time +1.20 s vs −1.26 s**; forward model **invariant
  to ambiguity** (min-TTC 2.88 s) while reactive degrades to a collision (drops the lead ≥ 0.75); OKRI
  ~3.2× lower (**14/14 tests**) — impact: **H15/A9**, first collision-rate + lead-time contrast for the
  consequence-forward-model thesis — see `2026-07-31-opponent-sweep-w4.md`
- [2026-07-31 · run #3] [Avride/NHTSA] FACT (delta) — PE opened **2026-05-06**; the 16 crashes span
  Dec'25–Mar'26 (**≥9 Dallas**, rest Austin), Hyundai **Ioniq 5** on Uber; NHTSA video: "unsafe lane
  changes into the path of other cars, **failing to avoid slow-moving vehicles ahead, and striking
  stationary objects**"; **all 16 under a safety monitor, only 1 attempted to intervene** — impact:
  reinforces **W-08/SC-13** (systematic, not marginal) — https://www.nbcdfw.com/news/local/robotaxi-operator-under-investigation-for-crashes-in-dallas/4023503/
- [2026-07-31 · run #3] [Wayve/Pony/NVIDIA/Uber-field] FACT — **Wayve +$60 M** (AMD/Arm/Qualcomm) Series-D
  extension + **Tokyo pilot late'26 (Nissan LEAF)**; **Pony Q2'26 200+ Gen-7 built, rev +76%**, orders
  +119% vs Jan (net loss Q1 $50.4 M — W-06 unchanged); **NVIDIA/Mercedes CLA = MB.Drive Assist Pro** L2++
  (10cam/5radar/12us), Alpamayo pitched to solve the "long tail," **no Nano-tier CNCE number**; Uber is now
  a **multi-vendor L4 marketplace** (Waymo/Avride/Autobrains/Momenta/Wayve/Nuro/WeRide) — impact: W-05/W-06
  hold; distribution moat is Uber's → technical-moat premium rises (H0) — https://blogs.nvidia.com/blog/drive-av-software-mercedes-benz-cla/
- [2026-07-31 · run #3] [emerging: Zoox/WeRide/Nuro] FACT — **Zoox** production-intent robotaxi unveiled
  (Jun'26), gated on NHTSA approval for up to **2,500** no-manual-controls vehicles (regulatory, not
  capability, bottleneck); **WeRide** driverless-fare via Uber in **Dubai (Mar 31'26)** + Abu Dhabi/Riyadh
  (1,200+ by ~2027); **Nuro+Lucid+Uber** expanded to **≥35,000 Lucid** vehicles, SF-Bay later'26 — impact:
  L4 field widening; all compute-heavy multi-sensor (W-05) — https://www.cnbc.com/2026/06/24/amazons-zoox-unveils-redesigned-robotaxi-ahead-of-upcoming-expansion.html
- [2026-07-31 · run #3] [arXiv] FACT/INFER — AD latent-WM surge continues; **hierarchy now surfacing** —
  **SGDrive** (2601.05640) "scene-to-goal *hierarchical* world cognition"; **DriveFuture** (2605.09701,
  1st NAVSIM-v2 navhard Apr'26); **EponaV2** (2605.14696); **Latent-WAM** (2603.24581) — impact: H1
  differentiator being explored (not yet with our combination); deep-read SGDrive next (Architecture) — https://arxiv.org/abs/2601.05640
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

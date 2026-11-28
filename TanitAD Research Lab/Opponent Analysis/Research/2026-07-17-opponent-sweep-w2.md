# Opponent Analyzer — Weekly Sweep 2026-07-17 (first live run)

**Agent:** Opponent Analyzer (Friday). **Run:** #1 (discipline was seeded-only at kickoff).
**Loop budget used:** 12 web searches / ~1 iteration (well under the 25-search / 3-iteration cap).
**Labeling (G-O1):** every line is tagged **[FACT]** (verifiable, sourced), **[CLAIM]** (reported /
self-reported, not independently verified), or **[INFER]** (our own inference).

> Scope note: this is the baseline build of `WEAKNESS_CATALOG.md` and `OPPONENT_PROFILES.md`
> (both created this run) plus the first monthly weak-spot **scenario feed** (intake package
> `2026-07-17-work-zone-phantom-scenario/`). Later runs record deltas only.

---

## 0. One-paragraph strategic read (mid-July 2026)

The field has split into two motions that both help us. (a) The scale-and-capital robotaxi operators are
hitting a **safety ceiling at exactly the edge cases our hypotheses target** — Waymo recalled 3,871
vehicles for driving into freeway **construction zones** and is under NTSB scrutiny for **school-zone
pedestrian anticipation** and **school-bus stop-arm** compliance; Tesla's camera-only robotaxi has an
open NHTSA engineering analysis for failing in **degraded visibility**. (b) The research world has
**converged on latent world models** (a 2026 arXiv surge + Wayve GAIA-3 + NVIDIA Alpamayo/Cosmos),
which *validates our core thesis but commoditizes it* — so our moat can no longer be "world model"; it
must be the parts nobody else ships: **hierarchy (H1), on-vehicle efficiency (H1/H3/H5), imagination of
unobserved area (H15), self-monitoring with guarantees (H11), regulation-native design (H9/H11)**.
[INFER] Net: the weaknesses got *more concrete and more public* this quarter, and the "we're just doing
what everyone else is doing" risk got *higher* — both point to the same response: weaponize the specific
failure modes into eval scenarios and lead every comparison with compute-normalized efficiency + safety.

---

## 1. Per-opponent deltas since kickoff (2026-07-05 baseline)

### Wayve — **correction + technical delta**
- [FACT] **Series D: $1.2 B (Feb 2026), $8.6 B post-money**, led by financial investors with
  Microsoft, NVIDIA, Uber + automakers Mercedes-Benz, Nissan, Stellantis; **$1.5 B total** secured
  incl. Uber milestone capital for robotaxi rollout to **10+ markets**; London commercial trials 2026,
  supervised consumer autonomy from 2027. *(Corrects the kickoff synthesis's "$2.8 B mid-2026" — the
  verifiable figure is the $1.2 B Series D / $1.5 B total-secured.)*
  — https://wayve.ai/press/series-d/ , https://www.otpp.com/en-ca/about-us/news-and-insights/2026/wayve-secures-funding-to-deploy-global-autonomy-platform/
- [FACT] **GAIA-3** launched: a **15 B-param latent-diffusion world model**, ~5× the compute and ~10× the
  data of GAIA-2, 9 countries / 3 continents — positioned for **offline evaluation/validation**, not the
  on-car driver. — https://wayve.ai/thinking/gaia-3/ , https://www.autonomousvehicleinternational.com/news/ai-sensor-fusion/wayves-gaia-3-generative-world-model-now-available-for-autonomous-driving-validation.html
- [FACT] On-car product is **AV2.0**: a single end-to-end camera+radar foundation model, **mapless**;
  LINGO VLA gives natural-language rationales (~60% human on reasoning); **Qualcomm** production-ADAS
  partnership (Mar 2026). — https://wayve.ai/technology/av2-0/ , https://www.qualcomm.com/news/releases/2026/03/qualcomm-and-wayve-advance-production-ready----end-to-end-ai-for
- [INFER] Wayve's world model is **generative-pixel + offline** (GAIA), while the driver is a monolithic
  E2E net — no explicit hierarchy, no imagination-in-the-loop, no self-monitoring guarantee. Their WM is
  a data/eval factory, not the reasoning substrate. That is our whitespace, not their moat.

### Waymo — **failure evidence (headline)**
- [FACT] **Recall of 3,871 robotaxis (2026-06-18)** after **13 incidents driving into freeway
  construction zones**: 6 in Phoenix (Apr 11 & 19) — **failed to recognize ramp-closure signs** and
  entered pre-planned freeway work zones; 7 in the SF Bay Area (May 18) — **drove between lane-closure
  cones** in an active construction stretch. Freeway autonomy suspended pending a validated OTA patch;
  **freezes the 20+-city 2026 expansion** that freeway capability was meant to power.
  — https://www.cnbc.com/2026/06/18/waymo-nhtsa-voluntary-recall-robotaxis-entered-freeway-construction-zones.html , https://techcrunch.com/2026/06/18/waymo-recalls-nearly-4000-robotaxis-to-stop-them-driving-into-highway-construction-zones/
- [FACT] **NTSB HWY26FH008** (2026-01-23, Santa Monica): a Waymo Jaguar I-Pace **struck a 9-year-old
  pedestrian** crossing midblock in a school zone. NTSB **preliminary**: the ADS **detected the child and
  applied heavy braking** before impact (minor injuries, no medical transport); no evidence of speeding or
  intentional law violation; NTSB is **examining how the ADS anticipates sudden pedestrian movement in
  school zones**. — https://www.ntsb.gov/investigations/Pages/HWY26FH008.aspx , https://santamonicanext.org/2026/03/ntsb-report-absolves-waymo-in-crash-near-school/
  - ⚠️ **G-O1 correction of a common conflation:** some coverage framed this as "Waymo struck a child" or
    merged it with a **separate** NTSB/NHTSA probe into **illegal school-bus stop-arm passing** (Austin
    ISD). These are two distinct threads; the Santa Monica case is the braking-but-late one above.
    [CLAIM] The school-bus thread: NTSB reportedly attributed one illegal-passing case to human error.
    — https://techcrunch.com/2026/01/23/waymo-probed-by-national-transportation-safety-board-over-illegal-school-bus-behavior/ , https://www.schoolbusfleet.com/news/ntsb-determines-human-error-led-to-waymos-illegal-school-bus-passing
- [FACT] **NHTSA Standing General Order** (window Jun 16 2025 – May 15 2026): **825 ADS incidents** total;
  **Waymo = 697** (613 property-only, 51 minor injury, 23 requiring hospitalization, **1 fatality**).
  Volume tracks Waymo's fleet-mile dominance. — https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting , https://www.notateslaapp.com/news/3999/nhtsa-releases-crash-data-on-robotaxi-and-teslas-competitors
- [FACT] **CA DMV disengagement report** (Dec 1 2024 – Nov 30 2025): Waymo **19,234 mi/disengagement**
  over 3.35 M mi; Zoox 60,682 mi over 1.21 M mi; Pony.ai ranks #3; 9 M+ industry test miles. The DMV
  **proposes to replace the disengagement metric in 2026** with "safety-relevant event" reporting.
  — https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/disengagement-reports/ , https://www.eetimes.com/waymo-dominates-california-av-test-data/

### Pony.ai — **growth, thin unit economics**
- [FACT] **Q1 2026**: total revenue ¥236 M (**$34.3 M, +145% YoY**); robotaxi revenue ¥59.1 M
  (**$8.6 M, +395% YoY**). Raised 2026 fleet target **3,000 → 3,500+** across **20+ cities**; live in
  **Croatia (first commercial robotaxi in Europe)** and **Dubai** (driverless); presence in **9
  countries** by May. — https://autonews.gasgoo.com/articles/news/ponyai-q1-2026-financial-results-robotaxi-quarterly-revenue-hits-record-high-raises-full-year-targets-2059600479305392128 , https://mlq.ai/news/v2/pony-ai-q1-revenue-more-than-doubles-to-343m-as-robotaxi-sales-surge-nearly-fivefold/
- [INFER] Robotaxi revenue of **$8.6 M against a 3,500-vehicle target** = still deeply pre-profitable
  unit economics on a compute-heavy multi-sensor stack — the cost-per-vehicle / data-efficiency gap we
  target (H3+H7) is unaddressed.

### Momenta — **HK IPO, two-leg strategy confirmed**
- [FACT] **HK IPO ~$752 M**, ~**$9 B** valuation, cornerstones GIC/Fidelity/BlackRock; GM + Tencent
  backed; trading from **2026-07-08**. Two divisions: **mass-produced L2++ software for OEMs** and
  **robotaxi**; **60% of proceeds → R&D, 20% → robotaxi**. Approvals in Suzhou/Shanghai; international
  launches **Abu Dhabi + Munich 2026**; L4 pilot with **Uber**. — https://www.bloomberg.com/news/articles/2026-06-29/gm-backed-self-driving-firm-momenta-seeks-752-million-from-hong-kong-listing , https://technode.com/2026/06/30/momenta-launches-hong-kong-ipo-with-gic-fidelity-and-blackrock-as-cornerstone-investors/
- [INFER] The two-leg model is now capital-locked by public-market scrutiny — the strategy-split
  weakness (attention divided between L2++ OEM cost pressure and L4 burn) becomes harder to reverse.

### Autobrains — **the narrative-overlap competitor**
- [FACT] **$140 M+ total funding**; backers BMW i Ventures, Toyota Ventures, Continental, Temasek.
  **"Liquid AI"** (liquid neural networks) + **modular agentic AI**, marketed as handling **edge cases
  with less compute on standard sensors + automotive-grade compute**; **Skills** product line (Oct 2024).
  — https://autobrains.ai/about-us/ , https://www.acnnewswire.com/press-release/english/93447/
- [INFER] **The one competitor whose *pitch* overlaps ours** (low-compute, edge-case, adaptive). But
  they are **supervised L2+/ADAS mass-market, not L4**; the "liquid" story is **runtime adaptivity**,
  not a hierarchical latent world model, no imagination, no self-monitoring-with-guarantees, no L4
  regulation-native safety case, no data-efficiency-from-action-free-video claim. **Watch:** if they
  publish compute-normalized numbers, that is a direct CNCE comparison we must pre-empt.

### NVIDIA Alpamayo — **frenemy / supply chain (delta)**
- [FACT] Launched CES 2026. **Alpamayo 2 Super = 32 B-param reasoning VLA** with **Chain-of-Causation**
  reasoning traces, built on **Cosmos**; open ecosystem = models + sim + data; **open dataset now
  1,700+ h across 25 countries / 2,500+ cities**. — https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development , https://techcrunch.com/2026/01/05/nvidia-launches-alpamayo-open-ai-models-that-allow-autonomous-vehicles-to-think-like-a-human/
- [INFER] Confirms the kickoff thesis: **their data/sim is our supply chain, their 32 B on-car VLA is our
  foil.** A 32 B reasoning model per vehicle is the anti-thesis of our ~261 M-on-Orin efficiency claim —
  CNCE is precisely the axis that separates us. Chain-of-Causation is their interpretability play; our
  answer is *inherent* (H1 fallback + H11 self-monitoring), not a post-hoc trace.

### Tesla — **emerging player, camera-only robustness gap**
- [FACT] Robotaxi launched in **Miami 2026-07-03** (unsupervised, first outside TX/CA); 5 territories;
  authorized TX fleet ~**42** vs Waymo's **577**. **NHTSA engineering analysis (Mar 2026)** — the final
  step before a possible recall — found camera-only FSD **"fails to detect and/or warn … under degraded
  visibility such as glare and airborne obscurants."** [CLAIM] Austin: **14 crashes** Jun 2025 – mid-Jan
  2026 over ~800 k paid mi (~1/57 k mi, ~4× the US-driver rate by Tesla's own metric); large-scale
  expansion deferred to the unreleased FSD v15. — https://www.automotiveworld.com/news/tesla-robotaxi-fleet-hits-25-as-musk-defers-scale-to-fsd-v15/ , https://www.techtimes.com/articles/319711/20260704/
- [INFER] Tesla is the cleanest illustration of the **no-epistemic-uncertainty** failure: a camera-only
  policy that keeps confidence when the sensing degrades. Our H11 (self-monitoring w/ guarantees) + H15
  (epistemic σ) + H2 (sensor-modality steering incl. radar) is the direct counter-story.

---

## 2. Technical / research-front sweep (arXiv, citation-graph)

- [FACT] A pronounced **2026 latent-world-model / JEPA-for-driving surge**: Drive-JEPA (2601.22032),
  Self-Supervised JEPA World Models (2602.12540), DriveFuture (2605.09701), GraphWorld — long-horizon
  planning WM (2606.16274), IDOL — inverse-dynamics-guided prediction (2605.31476), **Metis — a
  "generalizable and *efficient* world-action model"** (2606.15869), EponaV2 (2605.14696), a **Latent
  World Models survey** (2603.09086), CoWorld-VLA (2605.10426), DriveWorld-VLA (2602.06521).
  — https://arxiv.org/abs/2603.09086 (survey; entry point to the cluster)
- [INFER] Two consequences: (1) **our anchors (LeJEPA/V-JEPA-2/LAW/World4Drive) are now mainstream** —
  external validation of H3, cheap to cite. (2) **The "latent WM" label is no longer differentiating.**
  The nearest competitive threat to our *efficiency* framing is **Metis (efficient world-action model)** —
  flagged for a deep-read next run; if it reports FLOPs/param-normalized planning quality, it is the first
  academic head-to-head for CNCE. The survey (2603.09086) is the fastest way to keep the citation graph
  current — adopt it as a standing anchor.

---

## 3. What changes for TanitAD (analysis → actions)

| # | Attack surface (opponent) | Mechanism hypothesis | Our counter (H / gate) | Action this run |
|---|---|---|---|---|
| A1 | **Work-zone brittleness** (Waymo recall) | prior/expected road topology diverges from posterior reality (closed lanes, cone taper, ramp-closure sign); reactive perception can't reason about the *changed* drivable area | **H15** (imagine changed/unobserved area) + **H9** (sign/rule compliance) + **H1** fallback | **New scenario "Work-Zone Phantom"** shipped as intake pkg → OKRI/LOPS/LAL + blocked-lane compliance signal |
| A2 | **Degraded-visibility** (Tesla NHTSA) | camera-only, no epistemic uncertainty → confident when sensing degrades | **H11** self-monitoring (D8 AUROC>0.85 under glare/rain/obscurant) + **H15** σ + **H2** radar routing | Recommend a **degraded-visibility D8 stressor** to Benchmarks&Eval + Data Eng (Cosmos weather corpus already in mix per D-014) |
| A3 | **Compute-hungry WMs** (Alpamayo 32 B; Wayve GAIA-3 15 B offline) | scale-first reasoning/generation; on-car 32 B VLA | **H1/H3/H5** efficiency; **CNCE** | Feed competitor **param counts** to `LEADERBOARD.md` efficiency block (261 M vs 32 B vs 15 B-offline) |
| A4 | **Disengagement metric deprecated** (CA DMV) | the field's headline safety metric is being abandoned as not decision-relevant | **H0 / narrative**; aligns with Benchmarks&Eval "open-loop ⊥ closed-loop" | Adopt as a **story beat**: our closed-loop, regulation-native metrics were designed for exactly this gap |
| A5 | **Autobrains efficiency pitch** + **Metis** (academic) | narrative/technical overlap on "efficient" | **H1/H3/H5** must own compute-normalized proof | **Watch-list**; deep-read Metis (2606.15869) next run |

**Actionable recommendations (G-B):**
1. **[H6/H15/H9 — done this run]** Add **Work-Zone Phantom** to the weak-spot eval set (intake
   `2026-07-17-work-zone-phantom-scenario/`). It converts the Waymo recall into a repeatable gate.
2. **[H11/H2]** Add a **degraded-visibility robustness stressor** to D8 self-monitoring — directly targets
   Tesla's open NHTSA deficiency and is our sharpest live differentiator. Owner: Benchmarks&Eval (+Data Eng
   for the weather corpus). *Recommendation logged for Thursday agent; no cross-boundary write made.*
3. **[H1/H3/H5]** Keep a **compute-normalized (CNCE) leaderboard column** with competitor param counts so
   every comparison leads with efficiency. Owner: Benchmarks&Eval.
4. **[Narrative/H0]** Use the disengagement-metric deprecation in the vision story.
5. **[Watch]** Deep-read Metis (2606.15869) + monitor Autobrains for compute-normalized claims.

---

## 4. Self-critique vs quality gates

- G-A ✅ every claim carries a source link or repo path. G-O1 ✅ FACT/CLAIM/INFER labeled throughout.
- G-B ✅ five actionable recs, each tied to H / a gate. G-O2 ✅ every weakness entry names the exploiting H
  (no `no-counter-yet` entries this run).
- G-C ✅ KB updated (deltas, newest first). G-D ✅ ledger H0/H6 change-log row added (no status upgrade —
  nothing measured on our stack, P8).
- G-E ✅ implementation increment = the Work-Zone Phantom intake package with an **offline passing test**
  (telemetry-contract + discriminative-structure test; live CARLA wiring is the explicit next step).
- G-F ✅ done in the session-end ritual (STATE + PROJECT_STATE row + commit + push).
- **Boundaries:** no writes into `stack/` or another discipline's owned files; the degraded-visibility and
  leaderboard asks are *recommendations* to Benchmarks&Eval, not edits. `Mission Plan.md` untouched.

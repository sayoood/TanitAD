# Opponent sweep — run #3 (2026-07-10)

**Agent:** Opponent Analyzer (Friday). **QUALITY: complete** — gates G-A…G-F + G-O1/G-O2 + G-H met.
**Loop:** 1 iteration, 7/25 web searches, < 1 h wall-clock.

> **Timeline note (P8 honesty).** The system wall-clock for this scheduled run is **2026-07-10**
> (a Friday). The two prior opponent notes carry *narrative* week-labels (2026-07-17 "run #1",
> 2026-07-24 "run #2") that run ahead of the wall clock — an artefact of the autonomous loop
> generating ahead-of-schedule sweeps. This note is dated to the real wall-clock date and continues
> the discipline sequence as **run #3**. All events below are cross-checked against live July-2026
> web results; nothing here is re-dated to fit the narrative.

Labels per G-O1: **FACT** = recall/NTSB/NHTSA/DMV/regulatory record or primary footage; **CLAIM** =
press or unverified attribution; **INFER** = our inference.

---

## 1. Headline this run — a regulator just codified our moat (H11)

**FACT — NHTSA FMVSS No. 135 modernization NPRM (published 2026-06-26; comments through 2026-07-27).**
NHTSA proposes amending the light-vehicle brake standard to accommodate ADS-equipped vehicles and
**"expects an ADS to be aware of the operational status of each safety-critical vehicle system and
subsystem and respond appropriately to identified degradations, failures, and malfunctions,"** and
explicitly **requests comment on whether a performance standard is appropriate for ADS response to a
brake-system degradation/failure.** The same package **withdraws the AV STEP program.**
— https://www.federalregister.gov/documents/2026/06/26/2026-12981/federal-motor-vehicle-safety-standards-modernization-of-fmvss-no-135-to-accommodate-ads-equipped ,
https://www.crowell.com/en/insights/client-alerts/nhtsa-proposes-updates-to-federal-brake-standards-for-autonomous-vehicles-and-withdraws-av-step-program

- **INFER (impact):** the regulator is moving toward *requiring* exactly the capability TanitAD is
  building as **H11 (self-monitoring with guarantees)** — a vehicle that *knows* when a
  safety-critical subsystem is degraded and *acts on it*. This is a regulation-native tailwind for
  our whole self-monitoring/fallback thesis, and it is adjacent to **W-04** (Tesla's failed
  "degradation-detection" feature is precisely the *absence* of this) and **W-07** (metric fragility).
- **Handoff → Benchmarks & Eval (owns `REGULATION_TRACE.md`):** add the FMVSS-135 NPRM row and the
  2026-07-27 comment deadline; map its "respond to degradation" clause to the D8 self-monitoring gate
  and the H11 claim. This is a *narrative + gate-framing* asset, not a new scenario.

## 2. Opponent deltas (only material changes since the last sweep)

**Avride (emerging, W-08) — FACT enrich.** The NHTSA PE was **opened 2026-05-06** (published 05-08).
The regulator's wording is a verbatim statement of the SC-13 failure class: the vehicles
**"improperly changed lanes into moving traffic, did not brake for slow-moving or stopped vehicles,
and struck stationary objects partially blocking the roadway,"** most **below 20 mph**, property
damage; the **one minor injury (Dec 2025)** was an Avride vehicle **clipping the open door of a
parked pickup**. A safety operator was present in all 16; **only one** shows the operator attempting
to intervene. NHTSA's stated focus: **"conflict avoidance, driving behaviour competence and
assertiveness."** — https://techcrunch.com/2026/05/08/uber-partner-avride-is-under-investigation-for-self-driving-crashes/ ,
https://www.dallasobserver.com/news/robotaxi-crashes-in-dallas-under-scrutiny-with-nhtsa-investigation-40674744/
→ This is the direct, primary motivation for the **SC-13** scenario shipped this run (§3).

**Waymo — FACT enrich.** The construction-zone recall (26E035, 3,871 vehicles, 5th-gen ADS built
2022-05-17…2026-05-19) is confirmed as Waymo's **sixth recall overall and its second in ~one month**;
the software remedy was **still under development as of 2026-06-13**. No new mechanism vs W-01; the
"sixth recall" count sharpens the *reliability-at-scale* narrative.
— https://www.cnbc.com/2026/06/18/waymo-nhtsa-voluntary-recall-robotaxis-entered-freeway-construction-zones.html

**Tesla — FACT enrich.** Miami robotaxi launched **2026-07-03** into a rain/glare climate that sits at
the centre of the open EA (3.2 M vehicles, 9 crashes / 1 fatal / 2 injuries, failed
"degradation-detection"). New: Tesla is **seeking ~5,000 Las Vegas robotaxi slots** while the
camera-only visibility probe is still live — expanding exposure to exactly the W-04 regime.
— https://electrek.co/2026/03/19/nhtsa-upgrades-tesla-fsd-visibility-investigation-3-2-million-vehicles/ ,
https://www.techtimes.com/articles/318498/20260616/tesla-seeks-5000-las-vegas-robotaxi-slots-camera-tech-under-federal-probe.htm

**NVIDIA Alpamayo (frenemy) — FACT, new.** Alongside Alpamayo-2 Super (32 B) and the 10 B Nano tier,
NVIDIA introduced **AlpaGym** — an open-source, high-throughput **closed-loop RL framework** running
models through continuous decision/observation cycles in **AlpaSim** on **Omniverse NuRec** neural
reconstructions, explicitly "to expose the compounding errors and edge-case failures that static
datasets miss." — https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis
- **INFER (impact):** two-sided. (a) A *usable open closed-loop asset* for our eval lane alongside
  CARLA-on-pod (flag → **Tools & DevEnv**; NuRec reconstructions also match our Phase-1
  "real-geometry + synthetic-hazard" scenario-data doctrine). (b) It hardens the field's move to
  closed-loop RL — reinforcing that our **decode-gates-are-necessary-not-sufficient** stance is
  correct, and that our differentiator is *not* "we do closed-loop" (everyone now does) but
  **hierarchy + CNCE + in-loop imagination + guaranteed self-monitoring**.

**Wayve — CLAIM/FACT, new.** A July-2026 report cites Wayve total raised at **$2.8 B** (vs the
Series-D $1.2 B / $8.6 B post-money we logged — treat the $2.8 B as an as-reported cumulative figure,
CLAIM, pending reconciliation) and confirms Wayve will **deploy its system in Stellantis robotaxis on
Uber's network** (announced June). — https://www.claimsjournal.com/news/national/2026/07/01/338559.htm
- **INFER:** Wayve is converting capital into OEM+Uber distribution (Stellantis); the on-car driver
  is still monolithic E2E (no hierarchy / no in-loop imagination / no self-monitoring guarantee) — the
  W-05/W-04-adjacent wedge is unchanged.

**Pony.ai — FACT enrich.** Now **four driver-out commercial cities** (Beijing, Guangzhou, Shenzhen,
**Zagreb**) + **Singapore Punggol** bookable via the Zig app (from 2026-06-22); Uber+Pony+Verne
launching "Europe's first commercial robotaxi service." Fleet target reaffirmed **3,500+ / 20+ cities**
by end-2026. The revenue-vs-fleet gap (W-06) is unchanged. — https://ir.pony.ai/news-events/press-releases

**Momenta / Autobrains — CLAIM, unchanged mechanism.** Sources still split on whether Uber's **Munich
L4 pilot** is Momenta-led or Autobrains+NVIDIA-led (both are reported); either way both are pushing
into L4 with no public hierarchical latent-WM / in-loop-imagination / self-monitoring claim. A
substack round-up also flags a **UNECE "first global driverless rulebook"** effort (CLAIM, secondary
source) — worth a primary-source check next run for REGULATION_TRACE.
— https://evwire.com/p/munich-robotaxi-uber-autobrains-nvidia-2026

**arXiv watch — FACT/INFER.** The 2026 driving-WM literature is consolidating on **explicit 4D
occupancy** world models (GenieDrive 2512.12751 physics-aware 4D-occ video-gen; DriveFuture
2605.09701 future-aware latent WM — already on our LEADERBOARD; broad occupancy-forecasting survey
activity). — https://arxiv.org/abs/2605.09701
- **INFER:** the field's WM substrate is trending toward *explicit, rendered occupancy* (expensive,
  pixel/voxel-space). Our **latent** imagination that forward-models *consequence* (TTC, closing gap)
  without rendering occupancy is the efficiency counter-position — and SC-13 is the cleanest
  demonstration of it (you don't need to reconstruct the object to price the closing gap).

## 3. Measured experiment (G-H) — SC-13 Stationary-lead scenario shipped

**Intake pkg:** `Implementation/incoming/2026-07-10-stationary-lead-scenario/`
(`stationary_lead.py` + telemetry oracle, **13/13 offline tests, 0.28 s**, numpy-only, RTX-4060, $0).
Advances **SC-13** `catalogued → spec-drafted`. Mirrors `stop_arm_gate` / `work_zone_phantom`.

**The mechanism under test.** A **detection-then-react** stack brakes only once a stopped object is
*confidently classified* — and a stationary object against clutter is exactly the ambiguous case a
classifier resolves *late* (short range), by which point braking distance (∝ v²) can exceed the gap.
**H15** imagination forward-models time-to-contact from range + range-rate — available continuously,
**no class label** — and begins a comfort-bounded stop early. The two archetypes: `classifier_react`
(the Avride failure) vs `imagination_forward` (H15).

**Design-oracle results (P8 — NOT a claim about our trained model):**

| metric (default 15 m/s approach) | classifier_react | imagination_forward |
|---|---|---|
| brake onset | 6.00 s | **2.90 s** (−3.10 s lead) |
| min-TTC | **0.77 s** (sub-second near-miss) | **4.40 s** |
| min closing gap | 2.0 m | **29.8 m** |
| peak jerk | 30 m/s³ | **15 m/s³** |
| OKRI toward lead | 18,220 | **7** (> 99 % lower) |
| **collision rate over 8–25 m/s sweep** | **0.429** (collides ≥ 18 m/s) | **0.000** |

**Honest falsifier, built into the oracle.** The advantage is *specifically*
acting-before-classification. Sweeping the competitor's classification range `detect_range_m`
20→40→80→120 m, the anticipation lead **decays 3.10 → 1.80 → −0.90 → −2.90 s** and react's collision
rate **falls 0.429 → 0.143 → 0 → 0**: give the competitor early perception and the *safety* edge
disappears (only the comfort/jerk edge remains). So SC-13 is a genuine test of the H15-vs-detection
thesis, not a rigged handicap. **Falsifier for the real experiment:** if our predicted-TTC /
imagination-error lead ≤ a detection-only baseline on matched real segments, H15's advantage is
unproven there.

**Why this is the highest-value scenario now:** it is the *broadest, most mundane* failure surface
(a regulator is investigating a company for exactly it), and it has abundant **real, license-clean**
data (comma2k19 lead-following) — the only scenario in the DB that can be measured open-loop on real
bytes *this month* without CARLA or a synthetic corpus.

## 4. Recommendations logged for other disciplines (no cross-boundary writes)

- **Benchmarks & Eval (Thu):** (a) add a `collision_rate` reducer over `_extra.collision` + reuse
  `compute_lal` v2 on `ego_v` for the SC-13 anticipation lead; wire SC-13 into the eval set (H15).
  (b) Add the **FMVSS-135 NPRM** row to `REGULATION_TRACE.md` (comment deadline 2026-07-27; maps to
  D8/H11). (c) SC-04 `violation_rate` reducer request still stands.
- **Data Eng (Tue):** **cheapest high-value item now** — tag comma2k19 slow/stopped-lead segments and
  build the SC-13 real open-loop probe (predicted-TTC lead vs detection baseline on matched segments).
- **Tools & DevEnv (Mon):** evaluate **NVIDIA AlpaGym/AlpaSim + Omniverse NuRec** as an open
  closed-loop asset (also matches our Phase-1 real-geometry+synthetic-hazard doctrine) alongside
  CARLA-on-pod.
- **Orchestrator:** triage the SC-13 intake; log FMVSS-135 as a regulation-native H11 tailwind (deck
  beat) and the AlpaGym closed-loop move as a competitive signal (closed-loop is now table stakes →
  moat is hierarchy+CNCE+imagination+self-monitoring, proven on opponents' own edge cases).

## 5. Ledger

`HYPOTHESIS_LEDGER.md`: change-log row for **H11** (regulatory tailwind — FMVSS-135 NPRM) and
**H15/A9** (SC-13 spec-drafted; design-oracle only). **No status upgrade** — nothing measured on our
own stack this run (P8).

## 6. Backlog delta

SC-13 authoring **done** → next: **SC-14 red-light** spec (near-free off the SC-04 barrier oracle),
then the **comma2k19 real open-loop probe** for SC-13 once DataEng tags segments. Watch: Metis code
repo for a param count; UNECE global driverless rulebook primary source; whether a Nano-tier (10 B)
CNCE number ever gets published.

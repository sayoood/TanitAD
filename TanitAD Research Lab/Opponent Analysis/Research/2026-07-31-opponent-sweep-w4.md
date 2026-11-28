# Opponent sweep — run #3 (narrative 2026-07-31; real wall-clock 2026-07-17, Fri)

**Agent:** Opponent Analyzer. **Budget used:** 11 web searches/fetches of 25, 1 iteration, ~1.4 h.
**Quality:** complete (G-A…G-F + G-O1/G-O2 + G-H met; measured experiment executed with numbers).
**Evidence labels (G-O1):** FACT = recall/NTSB/NHTSA/DMV record or primary footage; CLAIM = press or
unverified attribution; INFER = our inference. **P8:** the Stationary-Lead numbers are a *design
oracle*, not a claim about our trained model.

> **Wall-clock / narrative-clock note (honesty).** The discipline's own clock runs ~2 weeks ahead of
> real time (a known loop artefact — see memory `narrative-clock-ahead-of-wallclock`). This run fires on
> the real scheduled Friday **2026-07-17** but continues the established run/Friday sequence: STATE
> LAST_RUN was run #2 (2026-07-24), so this is **run #3**, dated **2026-07-31** to keep the package
> folder, module docstrings and STATE handoff internally consistent. **Orchestrator dedup flag:** an
> unmerged off-schedule branch (`agent/opponent-20260715`) already authored SC-13/SC-14 speculatively;
> this canonical scheduled branch (`agent/opponent-20260717`) re-does SC-13 cleanly. Pick one at merge.

Prior run: `2026-07-24-opponent-sweep-w3.md` (Stop-Arm Gate SC-04, Avride entrant, Metis deep-read).
This is a **deltas-only** sweep plus the scheduled **Stationary-Lead** scenario (backlog P0.1) and a new
FACT-grade federal enforcement action that elevates an existing scenario.

---

## 1. Headline deltas (only what is material)

### 1.1 NHTSA "first-responder ultimatum" — a federal, all-operator enforcement action (FACT) → **new W-09 / SC-06 elevated** ★ headline
On **2026-07-08** NHTSA's Office of Defects Investigation issued a formal **ADS-developers letter**
(`nhtsa.gov/sites/nhtsa.gov/files/2026-07/ADS-developers-letter-july-2026.pdf`) demanding every AV
developer present, **by the end of July 2026**, fixes for a documented **"clear pattern"** of robotaxis
interfering with first responders: driving into active emergency scenes, blocking ambulances and fire
crews, and **failing to recognize flashing lights, flares, smoke, fire, and traffic cones.** In **≥6
incidents through March 2026** first responders had to **physically take control of Waymo vehicles** to
move them out of an emergency scene; a June 2026 case had an officer move a Waymo blocking responders at
a **natural-gas explosion.** Administrator **Jonathan Morrison**: *"the inability to detect and
appropriately respond to such situations represents a functional insufficiency"* and — the line that
matters for us — **"Emergency scenes are not rare or extreme 'edge cases.'"** NHTSA framed AVs as a
"danger to the public"; the letter names no company but Waymo has the largest fleet and the longest
incident list.

**Why this is the run's centerpiece (INFER).** This is the first time the *entire scenario-database
thesis* — that opponents fail the situations any competent driver handles — is stated, verbatim, by the
federal regulator: emergency scenes are **not edge cases**, and failing them is a **"functional
insufficiency."** It (a) creates a **new non-scenario-specific + scenario-specific weakness W-09**
(first-responder / emergency-scene interference), and (b) **elevates SC-06** (emergency-vehicle
interaction) from a 2023-Cruise anecdote (catalogued, "data gap") to a **live, FACT-documented,
all-operator federal action with a hard July deadline.** Our counter is direct: **H15** imagines the
emergency-scene actors and hazard field (flares/cones/personnel) before classification; **H11**
self-monitoring flags the non-nominal scene as OOD → **A9 fallback** degrades to yield/clear-the-corridor
rather than rule-literal continuation; **H9** exception-handling permits the rule-breaking a cleared
corridor needs. Sources: https://techcrunch.com/2026/07/08/feds-demand-autonomous-vehicle-companies-stop-interfering-with-first-responders/
, https://thenextweb.com/news/nhtsa-autonomous-vehicles-first-responders-interference

### 1.2 Waymo — now fighting the regulator *and* its biggest distributor at once (FACT/INFER) → W-01/W-03/W-09
The construction-zone recall (26E035, 3,871 vehicles) stands; on top of it Waymo now carries the
first-responder directive (§1.1), the Dallas red-light (SC-14), the school-bus stop-arm probe (SC-04)
and NTSB HWY26FH008 (SC-02). Press frames it as a **"robotaxi ultimatum": Waymo fighting a regulator
(NHTSA) and its biggest distributor (Uber) simultaneously** — Uber has diversified to **Avride,
Autobrains+NVIDIA, Momenta and Wayve**, eroding Waymo's channel leverage. **Read-through (INFER):**
Waymo's failure surface is now **broad and federal**, not a single recall; every entry maps to a
scenario we already own or are adding (W-01/W-03/W-09). Source:
https://businessmodelanalyst.com/nhtsa-robotaxi-ultimatum-waymo-uber/

### 1.3 Avride — richer FACT detail that directly reinforces SC-13 (FACT) → W-08/SC-13
NHTSA's PE (opened **2026-05-06**) covers **16 crashes** (Dec 2025–Mar 2026; **≥9 in Dallas**, rest
Austin) in **Hyundai Ioniq 5** robotaxis on Uber. NHTSA video shows the ADS **"executing unsafe lane
changes into the path of other cars, failing to avoid slow-moving vehicles ahead, and striking
stationary objects partially blocking the roadway."** Damningly, **all 16 happened with a safety monitor
in the driver's seat, and only *one* monitor even attempted to intervene** — i.e. the failures are fast
and systematic, not marginal. This is exactly the SC-13 axis (stationary-object / same-lane lead) and
the measured experiment below (§4) is its design oracle. Source:
https://techcrunch.com/2026/05/08/uber-partner-avride-is-under-investigation-for-self-driving-crashes/
, https://www.nbcdfw.com/news/local/robotaxi-operator-under-investigation-for-crashes-in-dallas/4023503/

### 1.4 Opponent business/deployment deltas (FACT, one line each)
- **Wayve** — extended Series D with **+$60 M from AMD, Arm, Qualcomm** (compute-platform breadth), on
  top of the $1.5 B secured; **Tokyo pilot late 2026 (Nissan LEAF)** added to the London 2026 plan.
  GAIA-3 stays an **offline** data/eval factory (W-05). — https://wayve.ai/press/series-d/
- **Pony.ai** — **Q2 2026: 200+ Gen-7 robotaxis produced, revenue +76%**; weekly paid orders **+119% vs
  January**, Labor-Day daily orders **+544% YoY**; fleet **>1,700** (toward the raised >3,500 target).
  Growth real; the **thin-economics gap (W-06) unchanged** — Q1 net loss widened to **$50.4 M**. —
  https://www.stocktitan.net/news/PONY/pony-ai-inc-accelerates-gen-7-robotaxi-production-with-over-200-hh9t413ag1af.html
- **Momenta→Autobrains (Munich)** — the reason for Uber's switch is now explicit: **EU political
  resistance to sensitive Chinese key-tech.** Sharpens Pony/Momenta's **EU-market-access** weakness and
  our **Western/EU-clean data + compliance** wedge. — https://www.electrive.com/2026/06/02/uber-and-autobrains-to-partner-on-munich-robotaxi-pilot-project/
- **NVIDIA / Mercedes CLA** — ships as **MB.Drive Assist Pro** (10 cameras / 5 radars / 12 ultrasonics,
  L2++ point-to-point urban under supervision); Alpamayo positioned *explicitly* to "solve the long tail
  / rare weird edge cases." Family unchanged: **10 B Nano → 32 B Super**; still ~40× our ~261 M envelope;
  **no Nano-tier CNCE number published.** — https://blogs.nvidia.com/blog/drive-av-software-mercedes-benz-cla/

### 1.5 Emerging players (FACT) — Uber's multi-vendor L4 field widens → W-06/W-05
- **Zoox** (Amazon) — unveiled its **production-intent** purpose-built robotaxi (Jun 2026); large-scale
  Bay-Area production starting; free rides in Las Vegas/SF, select Austin/Miami; testing in 6 more
  cities. **Gated on NHTSA approval** to run up to **2,500** no-manual-controls vehicles commercially —
  a *regulatory* bottleneck, not a capability one. — https://www.cnbc.com/2026/06/24/amazons-zoox-unveils-redesigned-robotaxi-ahead-of-upcoming-expansion.html
- **WeRide** — fully driverless fare-charging via Uber in **Dubai (Mar 31 2026)**, plus Abu Dhabi/Riyadh;
  **1,200+ vehicle Middle-East commitment** by ~2027.
- **Nuro + Lucid + Uber** — deal expanded to **≥35,000 Lucid vehicles** (Nuro supplies the L4 stack;
  Uber investment ~$500 M); first SF-Bay service later 2026. **Read-through (INFER):** the L4 field is
  now a **multi-vendor Uber marketplace** (Waymo, Avride, Autobrains, Momenta, Wayve, Nuro, WeRide) — the
  distribution moat is Uber's, not any stack's, which *raises* the premium on a defensible technical moat
  (efficiency + safety-case) that none of them holds.

## 2. Field / arXiv recency scan (D-028) — latent-WM surge continues; hierarchy now appearing (FACT, external)
Not opponents; they move the H1/H3/H15 story (P8: external corroboration, no status change). Newest AD
world-model work since the last sweep:
- **SGDrive** (2601.05640) — **"scene-to-goal *hierarchical* world cognition"** for driving. Notable: a
  competitor academic line is now explicitly **hierarchical** — our H1 differentiator is being explored,
  not yet with our combination (efficiency + in-loop imagination + self-monitoring). **Watch item.**
- **DriveFuture** (2605.09701) — future-conditioned latent WM, **1st on NAVSIM-v2 navhard** (Apr 2026).
- **EponaV2** (2605.14696) — driving WM with "comprehensive future reasoning."
- **Latent-WAM** (2603.24581) and **DriveWorld-VLA** (2602.06521) — more latent WM + VLA momentum (H3).
- **Taxonomy anchor** (2603.09086) unchanged — "latent WM" is a *field*, not a differentiator.

**Read-through (INFER):** the "latent world model" label is fully commoditized and **hierarchy is
starting to appear** (SGDrive). Our moat compresses to the *combination none ship together*:
**hierarchy + compute-normalized efficiency (CNCE) + in-loop imagination (H15) + guaranteed
self-monitoring (H11)** — proven on opponents' own FACT-documented failures. Next-run action: deep-read
**SGDrive** for whether its hierarchy is planning-time or representation-only (Architecture handoff).

## 3. Actionable recommendations (G-B; each tied to a hypothesis / work package)
1. **Benchmarks & Eval (Thu):** (a) add a **`min_ttc` reducer** + a **`collision_rate` reducer** over
   the SC-13 `_extra`, and reuse the **LAL-v2** lead-time metric over `_extra.brake_onset_lead_time_s`;
   wire SC-13 into the eval set (H15). (b) **W-09/SC-06:** define an **emergency-scene metric** — corridor-
   clear time + a "non-nominal-scene detected" flag (proxy for OOD) — so the first-responder failure
   becomes measurable before CARLA (H11/H15/H9). [Prior recs — SC-04 violation-rate reducer, SC-05
   degraded-visibility D8 stressor, competitor-param CNCE block — still stand.]
2. **Data Engineering (Tue):** **top cheap win** — tag **stopped/slow-lead + stationary-object segments
   in comma2k19** (real, license-clean) for the SC-13 open-loop lead-time probe (the falsifier needs
   matched segments). Also screen dashcam corpora for **emergency-scene / flashing-light events** (W-09).
3. **Tools & DevEnv (Mon):** for the **W-09/SC-06** CARLA build, use emergency-vehicle + light-pattern
   assets (visual-only proxy, no audio Phase-0); evaluate **AlpaSim** as the closed-loop harness (still
   open from run #2). (D5/D6, G0.5)
4. **Architecture & Inference (Wed):** deep-read **SGDrive** (2601.05640) — is its hierarchy at
   planning time (our claim) or representation-only? Feeds the H1 differentiation story. (H1)
5. **Orchestrator:** (a) triage the Stationary-Lead intake (**dedup vs the unmerged `agent/opponent-20260715`
   SC-13**, pick one); (b) log the **NHTSA first-responder directive** as a strategy-grade signal —
   "emergency scenes are not edge cases" is the strongest external endorsement of the scenario-database
   thesis and belongs in the vision deck / weekly report (H0); (c) note the Uber multi-vendor marketplace
   as a distribution-moat observation (nobody's stack owns the channel → technical moat premium rises).

## 4. Measured experiment (G-H) — Stationary-Lead scenario (SC-13 / W-08)
**Backlog P0.1 delivered.** Intake pkg `Implementation/incoming/2026-07-31-stationary-lead-scenario/`
(`stationary_lead.py` + telemetry oracle, `tests/test_stationary_lead.py`). Mirrors the integrated
`work_zone_phantom` / `stop_arm_gate` structure: an ego cruising at 15 m/s toward a **stationary lead
object in its lane**, parameterized by **`classification_ambiguity` ∈ [0,1]** (how hard the object is to
*classify* — weak class prior or a degraded sensing channel, à la the Tesla EA lead-loss finding). Two
archetypes — `detection_reactive` (brakes on a TTC threshold that **slips later** as ambiguity rises, and
**drops the lead** past a threshold — the documented failure) and `imagination_forward` (H15 forward-
models the **closing-gap consequence** and eases off early, **independent of classification**, holding an
A9 latent lead estimate).

**Result — offline design oracle (`C:/Users/Admin/venvs/tanitad` py3.13, numpy 2.5.1; RTX-4060 box,
CPU-only; <1 s; cost $0; 14/14 tests pass):**

| Metric (ambiguity sweep {0, .25, .5, .75, 1}) | imagination_forward (H15) | detection_reactive (failure) |
|---|---|---|
| **Collision rate** | **0.0** | **0.4** (contacts at ambiguity ≥ 0.75) |
| **Mean braking-onset lead time (LAL-v2)** | **+1.20 s** | **−1.26 s** |
| Lead time vs ambiguity | **invariant** (+1.20 s at every level) | **−0.50 → −2.00 s** (monotone worse) |
| min-TTC vs ambiguity | **invariant 2.88 s** | **1.91 → 1.48 → 1.11 → 0.00 s** |
| min-gap vs ambiguity | **invariant 10.75 m** | **11.5 → 7.0 → 4.0 → 0.4 m** |
| Lead dropped (wm→NaN in-range) | never | at ambiguity ≥ 0.75 |
| OKRI toward lead @ ambiguity 0.5 (lower safer) | **4,612** | **14,570** (~3.2×) |
| params / latency | 4 B / 18 ms | 15 B / 40 ms |

**Interpretation.** The scenario cleanly separates the two mechanisms along the axis the Avride ODI and
Tesla EA name: the **consequence of the closing gap is knowable before the object is classified.** The
forward model is **invariant to classification ambiguity** (nothing about the class prior enters its
decision) and **never contacts** the lead; the detection-reactive policy **degrades monotonically** and
**collides once it drops the lead** (min-TTC → 0, min-gap → 0.4 m at ambiguity ≥ 0.75). The primary H15
metric here is the **braking-onset lead time** (+1.20 s vs −1.26 s) with a hard **collisions == 0** bar.
**SC-13 advances `catalogued → spec-drafted`. Pre-registered falsifier (for the live run):** if our
trained checkpoint's imagination-error lead time is **≤** a detection-only baseline on matched real
stopped-lead segments (comma2k19), the H15-vs-detection advantage is **unproven here** — record as a
negative result (P8), do not claim SC-13 excellence. Next: DataEng tags the comma2k19 segments;
Benchmarks & Eval adds the min-TTC + collision-rate reducers; then live-measure on CARLA-on-pod.

## 5. Ledger / catalog updates
- `HYPOTHESIS_LEDGER.md` — change-log row (run #3): **H15/A9** gain a shipped scenario (SC-13, 14/14
  tests, first collision-rate + lead-time contrast for the consequence-forward-model thesis); **H0/H6/H11**
  reinforced by the NHTSA first-responder directive. No status *upgrade* (design-oracle only, P8).
- `WEAKNESS_CATALOG.md` — **new W-09** (first-responder / emergency-scene interference, FACT); **W-08**
  SC-13 → spec-drafted; W-01/W-03 enriched (first-responder + Waymo-vs-Uber); watch-list adds the
  hierarchy field-signal (SGDrive).
- `OPPONENT_PROFILES.md` — Δ 07-31 deltas on Waymo/Wayve/Pony/Momenta/Autobrains/NVIDIA/Avride; **new
  Zoox / WeRide / Nuro** emerging-player entries; cross-field one-liner refreshed.
- `SCENARIO_DATABASE.md` — **SC-13 → spec-drafted** (with numbers + handoffs); **SC-06 evidence upgraded**
  (2026-07-08 federal directive, priority ↑) + linked to new W-09; coverage matrix + scoreboard updated.
- `KNOWLEDGE_BASE.md` — 6 new dated findings, newest first.
- `BACKLOG.md` — re-prioritized (SC-13 done → SC-06/W-09 emergency-scene spec is the new P0; SGDrive
  deep-read added).

## 6. Loop / gate self-check
- G-A (sources): every claim linked or repo-path referenced. ✓
- G-B (actionable): §3, five recs tied to H/WP. ✓
- G-C (KB deltas, deduped, newest first). ✓
- G-D (ledger updated). ✓
- G-E (implementation increment verifiable): Stationary-Lead pkg, 14/14 tests. ✓
- G-H (measured experiment with numbers): §4. ✓ BACKLOG re-prioritized (see `BACKLOG.md`).
- G-O1 (FACT/CLAIM/INFER labeled). ✓  G-O2 (every weakness names its H or is `no-counter-yet`). ✓

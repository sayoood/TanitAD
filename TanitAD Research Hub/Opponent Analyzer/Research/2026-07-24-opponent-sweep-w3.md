# Opponent sweep — run #2 (2026-07-24, Fri)

**Agent:** Opponent Analyzer. **Budget used:** 9 web searches/fetches of 25, 1 iteration, ~1.5 h.
**Quality:** complete (G-A…G-F + G-O1/G-O2 met; measured experiment executed with numbers).
**Evidence labels (G-O1):** FACT = recall/NTSB/NHTSA/DMV/primary footage; CLAIM = press/unverified;
INFER = our inference. **P8:** the Stop-Arm Gate numbers are a *design oracle*, not a claim about our
trained model.

Prior run: `2026-07-17-opponent-sweep-w2.md` (built the v1 catalogs). This is a **deltas-only** sweep
plus the scheduled **Stop-Arm Gate** scenario (backlog P0.1) and the **Metis** deep-read (backlog P1).

---

## 1. Headline deltas (only what is material)

### 1.1 A new emerging player under investigation — Avride (FACT) → **W-08 / SC-13**
NHTSA ODI opened an investigation (**2026-05-08**) into **Avride** (Uber's robotaxi partner, Yandex SDG
lineage) after **16 crashes + 1 minor injury**. ODI states all relate to **"the competence of"** the
driving system — **lane-changing, responding to other vehicles in the same lane, and responding to
stationary objects.** This is the most useful new find of the run: it's a *basic-competence* indictment,
the broadest and least glamorous failure surface, and it maps cleanly onto **H15** (imagination
forward-models time-to-contact on a stopped/slow lead *before* classification, so there is no detection
prior to be wrong about). New profile added; **W-08** in the catalog; **SC-13** (stationary-object /
same-lane lead) catalogued — a cheap real-data spec on comma2k19.
Source: https://techcrunch.com/2026/05/08/uber-partner-avride-is-under-investigation-for-self-driving-crashes/

### 1.2 Waymo — second recall, highways pulled, and a Dallas red-light (FACT) → W-01 / W-03
The construction-zone recall (NHTSA **26E035**, 3,871 vehicles) was Waymo's **second in ~one month**; it
**pulled all robotaxis from highways on 2026-05-19** and the fix remains "under development." Waymo's own
filing names the mechanism: the AV **"inappropriately prioritiz[es] the avoidance of other freeway
hazards and/or fail[s] to recognize the construction zone"** — i.e. a hazard-avoidance objective
overrides the (unrecognized) closure. Separately, a Waymo was recorded **running a red light** in Dallas
(Irving Blvd/Inwood Rd) amid a **new federal investigation** there → a hard-rule-compliance failure
(W-03 family) catalogued as **SC-14**, which reuses the Stop-Arm Gate barrier oracle almost for free.
Sources: https://techcrunch.com/2026/06/18/waymo-recalls-nearly-4000-robotaxis-to-stop-them-driving-into-highway-construction-zones/
, https://www.dallasobserver.com/news/robotaxi-crashes-in-dallas-under-scrutiny-with-nhtsa-investigation-40674744/

### 1.3 Tesla — probe now an Engineering Analysis over 3.2 M vehicles (FACT) → W-04 / SC-05
The FSD probe was **upgraded to an Engineering Analysis on 2026-03-18**, covering **~3.2 M vehicles**
(MY2016–2026), the final phase before a recall. It found FSD's **"degradation-detection" feature** — the
component meant to recognize impaired cameras and alert the driver — **"did not detect common roadway
conditions that impaired camera visibility … until immediately before the crash."** **9 crashes**, incl.
**1 fatality + 2 injuries**, under glare/fog/dust. Tesla also **unredacted its 17 Austin robotaxi
incidents** (Jul'25–Mar'26; 13 property-only, 1 hospitalization, **2 involving teleoperators**), and the
Miami robotaxi launched into rain on 2026-07-03. This is the sharpest field validation of our **H11**
(self-monitoring/OOD, D8 AUROC) + **H15** (epistemic σ throttle) + **H2** (modality steering) axis: the
*named* failing feature is exactly a self-monitoring component without calibrated uncertainty.
Source: https://electrek.co/2026/03/19/nhtsa-upgrades-tesla-fsd-visibility-investigation-3-2-million-vehicles/

### 1.4 "World model" is now table stakes — Momenta R7, NVIDIA ships Alpamayo, Autobrains → L4 (FACT)
- **Momenta** listed in Hong Kong (2026-07-08, ~$8.9 B cap; Mercedes-Benz + BYD cornerstones) and had
  already shipped its own **R7 Reinforcement-Learning World Model (Apr 2026)** + first in-house chip
  **X7** (SAIC-VW ID.ERA 9X). So a mass-market OEM-supplier opponent now *also* has a "world model."
- **NVIDIA** — the **Mercedes-Benz CLA** becomes the **first production car to ship NVIDIA's entire AV
  stack** (US, this quarter). The Alpamayo family spans **10 B (1 Nano / 1.5 Nano) → 32 B (2 Super)**;
  the Nano tier is a *partial* answer to the efficiency critique, but 10 B on-car is still ~40× our
  ~261 M active envelope. **AlpaSim** (open-source closed-loop sim) is on GitHub → a usable asset for our
  closed-loop eval (handoff: Tools&DevEnv).
- **Autobrains** — **Uber + Autobrains (+ NVIDIA) Munich robotaxi pilot (2026-06-02)**, apparently
  displacing/paralleling Uber's earlier Momenta-Munich plan → Autobrains is **stepping from ADAS toward
  L4.** The watch-list escalates: their "less compute, standard sensors" pitch now rides an L4 pilot.

**Read-through (INFER):** the generative/latent "world model" label has diffused to essentially every
tracked player (Wayve GAIA-3, NVIDIA Alpamayo/Cosmos, Momenta R7, plus the academic WAM cluster). It is
no longer a differentiator. Our moat is the *combination* none of them ships together: **hierarchy +
compute-normalized efficiency (CNCE) + in-loop imagination (H15) + guaranteed self-monitoring (H11)**.

## 2. Deep-read: Metis (arXiv 2606.15869) — the nearest CNCE head-to-head (backlog P1)
Fudan/HKU/Tongji/Li Auto, submitted 2026-06-14. Metis is a **generalizable, efficient world-action
model**. Two design moves matter for us:
1. **Decoupling via Mixture-of-Transformers** — dedicated experts for *video generation* and *action
   prediction*, preserving each task's distribution (they argue tight video↔action coupling causes
   representational mismatch and hurts generalization).
2. **Asymmetric attention mask** — enables joint training of both experts while letting the **action
   model bypass explicit video generation at inference**, cutting the high test-time latency of WAMs
   that roll out future frames. Claims SOTA on **NAVSIM navhard/navtest** and **CityWalker**.

**Assessment (INFER).** Its efficiency lever — *skip the generative rollout at inference* — is
conceptually our latent/no-pixel path, so Metis validates the "don't render pixels to plan" thesis. But
against TanitAD it is **flat** (a two-expert MoT, **no hierarchy**), has **no in-loop imagination used
for planning** (video gen is a *training* auxiliary it discards at test time — the opposite of using an
imagination field to price candidate actions), and **no self-monitoring / OOD guarantee**. Critically,
the abstract reports **no parameter count and no compute-normalized metric** — so it is *not yet* a true
CNCE competitor, and the comparability gap is ours to define. **Action:** track its code
(github.com/LogosRoboticsGroup/Metis) for a param disclosure; when we publish, lead with params + a
compute-normalized causal-efficacy number Metis omits.

## 3. Field scan — new latent-WM / hierarchy / imagination papers (FACT, external support)
Not opponents, but they move the H1/H3/H15 story (P8: external corroboration, no status change):
- **Hierarchical Planning with Latent World Models** (2604.03208, upd. 2026-06-16) — direct support for
  **H1** horizon factorization via hierarchy.
- **FF-JEPA: Long-Horizon Planning with Latent Planners** (2606.09311) and **Variable-Length Latent
  World Models for Long-Horizon Planning** (2606.21775) — latent long-horizon planning momentum (H1/H3).
- **Sparse Imagination for Efficient Visual World-Model Planning** (2506.01392) — efficiency of the
  *imagination* step itself (H15 relevance; possible Architecture backlog read).
- **Adjacent-domain (monthly sweep):** **SkyJEPA** (2606.23444) — long-horizon JEPA world models for
  zero-shot sim-to-real quadrotor control. Confirms the JEPA-WM recipe transfers to aviation autonomy;
  worth a cross-domain read for the sim-to-real transfer protocol (Tools&DevEnv/Architecture).

## 4. Measured experiment (G-H) — Stop-Arm Gate scenario (SC-04 / W-03)
**Backlog P0.1 delivered.** Intake pkg `Implementation/incoming/2026-07-24-stop-arm-gate-scenario/`
(`stop_arm_gate.py` + telemetry oracle, `tests/test_stop_arm_gate.py`). Mirrors the integrated
`work_zone_phantom` structure: a stopped school bus with a deployed stop-arm, a legal stop line, an
**occluded child crossing in front of the bus**, and a *tempting free path* in the ego lane. Two
archetypes — `soft_prior` (treats the stop-arm as a soft cost; the documented Waymo failure) and
`rule_barrier` (H9 hard barrier + H15 latent child estimate).

**Result — offline design oracle (`venvs/tanitad` py3.13; 0 s; cost $0; 11/11 tests pass):**

| Metric | rule_barrier (H9) | soft_prior (failure) |
|---|---|---|
| **H9 violation rate** (sweep {0,2,4,6,8,10,12} m clearance) | **0.0** | **1.0** |
| Stop margin before line @8 m clearance | +0.4 m (v=0, full stop) | never halts (rolls through) |
| Speed at the stop line @8 m | 0.0 m/s | 8.28 m/s |
| Speed-at-line vs temptation (0→12 m) | **invariant** (~0) | **3.0 → 9.6 m/s** (monotone) |
| OKRI toward the occluded child | 10.5 k | 51.4 k (**80% higher**) |
| params / latency | 4 B / 18 ms | 15 B / 40 ms |

**Interpretation.** The scenario cleanly separates the two failure modes along the axis the Waymo probe
exposes: a **barrier term is invariant to the "apparent free path" temptation** (it stops regardless),
whereas a **soft prior's line-crossing speed rises with the temptation** — the mechanistic essence of
"rule as barrier vs rule as cost." The violation rate is the primary H9 metric and its bar is exactly 0.
**Falsifier (for the eventual live run):** if our trained checkpoint's violation rate is > 0 on this
scenario, the RMFM/barrier term is not actually a barrier and H9 is unproven — escalate. **SC-04**
advances `catalogued → spec-drafted`.

## 5. Actionable recommendations (G-B; each tied to a hypothesis / work package)
1. **Benchmarks & Eval (Thu):** add a `violation_rate` reducer over the scenario `_extra.stop_arm_violation`
   field (a rate, not a soft score) alongside `scenario_metrics`; it is the H9 metric home. Then wire
   SC-04 into the eval set. (H9)
2. **Data Engineering (Tue):** (a) source a school-bus + stop-arm asset for CARLA-on-pod and screen US
   dashcam corpora for stop-arm/red-light events (SC-04/SC-14); (b) **cheap win** — tag **stopped/slow-lead
   segments in comma2k19** for SC-13 (Avride's competence failure) as a license-clean real open-loop probe.
   (H15/H9)
3. **Tools & DevEnv (Mon):** evaluate **AlpaSim** (NVIDIA, open-source, GitHub) as a closed-loop sim asset
   alongside the CARLA-on-pod plan — it is now a usable supply-chain asset. (D5/D6, G0.5)
4. **Orchestrator (Fri PM):** triage the Stop-Arm Gate intake; note the Autobrains ADAS→L4 escalation and
   the Avride entrant as strategy-relevant competitive signals; AlpaSim availability may de-risk closed-loop.
5. **Narrative/paper (H0):** the 07-24 evidence — Waymo's 2nd recall + Dallas red-light, Tesla's 3.2 M-vehicle
   EA naming a self-monitoring failure, a new entrant (Avride) under a *competence* probe, and "world model"
   diffusing to everyone — is the strongest "own their edge cases at 1–2 orders less compute" beat yet.

## 6. Ledger / catalog updates
- `HYPOTHESIS_LEDGER.md` — change-log row 2026-07-24 (H0/H6/H9 evidence + wedge sharpening; no status
  upgrade, P8).
- `WEAKNESS_CATALOG.md` — W-01/W-03/W-04 enriched; **W-08 (Avride competence)** added; watch-list updated
  (Autobrains→L4, Metis deep-read done).
- `OPPONENT_PROFILES.md` — Δ 07-24 deltas on Waymo/Pony/Momenta/Autobrains/NVIDIA/Tesla; **new Avride
  profile**; cross-field one-liner refreshed.
- `SCENARIO_DATABASE.md` — SC-04 → **spec-drafted** (with numbers + handoffs); **SC-13** (Avride) and
  **SC-14** (red-light) added; coverage matrix + excellence scoreboard updated.
- `KNOWLEDGE_BASE.md` — 6 new dated findings, newest first.

## 7. Loop / gate self-check
- G-A (sources): every claim linked or repo-path referenced. ✓
- G-B (actionable): §5, five recs tied to H/WP. ✓
- G-C (KB deltas, deduped, newest first). ✓
- G-D (ledger updated). ✓
- G-E (implementation increment verifiable): Stop-Arm Gate pkg, 11/11 tests. ✓
- G-H (measured experiment with numbers): §4. ✓ BACKLOG re-prioritized (see `BACKLOG.md`).
- G-O1 (FACT/CLAIM/INFER labeled). ✓  G-O2 (every weakness names its H or is `no-counter-yet`). ✓

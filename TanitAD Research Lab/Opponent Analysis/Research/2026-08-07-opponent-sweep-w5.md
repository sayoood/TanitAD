# Opponent sweep — run #4 (narrative 2026-08-07 · **real wall-clock 2026-07-20**)

> **Clock note (P8 honesty).** This discipline's narrative clock runs ~2.5 weeks ahead of wall-clock
> (a known loop artefact; see `STATE.md` and the `narrative-clock-ahead-of-wallclock` record). Notes and
> packages are dated on the narrative clock so folder names, module docstrings and STATE stay mutually
> consistent; runs are keyed by **run number**, not by note date. The real calendar date of this run is
> **2026-07-20**, i.e. only **3 days** of real-world news delta since run #3. The weight of this run is
> therefore deliberately on the **experiment**, not on the news sweep — and the news that did land in
> those 3 days turned out to be the most important of the quarter.

**Evidence labels (G-O1):** FACT = recall/NTSB/NHTSA/DMV record or primary footage · CLAIM = press or
unverified attribution · INFER = our own inference.

---

## 0. Headline

1. **First MEASURED (non-oracle) SC-13 number on our own checkpoint — and the pre-registered falsifier
   FIRES on the cross-corpus replication.** Every scenario number this discipline has produced so far
   was a *design oracle*. This run put the SC-13 thesis on **real held-out windows with flagship-30k**.
   On the **in-domain PhysicalAI val** the result looked good: for braking that begins 2–3 s out —
   beyond the model's own 2 s rollout — the imagined-slowdown signal scored **AUROC 0.72–0.74** against
   a reactive kinematic floor at **0.43**. **It did not replicate on comma2k19**: after the same speed
   control, **held 0.54–0.61 vs blind 0.55–0.61 vs reactive 0.55–0.59 — mutually indistinguishable**.
   **Verdict: the consequence-forward-model advantage is NOT established.** One corpus is not evidence,
   and the corpus where it worked is the one the model was trained on. Recorded as a **negative result
   (P8)**, with the two live confounds named (out-of-domain + highway regime, where CV is near-
   unbeatable — on comma2k19 constant velocity beats the model outright on ADE, 1.30 m vs 1.87 m).
2. **W-09 (emergency-scene interference) is now a TWO-OPERATOR, FACT-documented class**, not a Waymo
   anecdote: **Zoox recalled 105 vehicles** after a robotaxi **drove into thick smoke from an active
   fire**, failed to recognize it, panic-braked and **halted inside the scene**. SC-06 authored and
   shipped as an intake package this run (16/16 tests).
3. **New weakness W-10 — fleet-scale mission/energy blindness.** Waymo's **2026-07-04 San Francisco
   breakdown** (**64 vehicles** retrieved by staff or tow truck, battery depletion, unplanned road
   closures, one occupied car **drove over a lit firework**) is a *strategic-layer* failure with no
   counterpart in any competitor's published architecture — and, uncomfortably, none in ours either
   beyond a design intent. Marked partially `no-counter-yet`.

---

## 1. MEASURED EXPERIMENT (G-H / G-I) — SC-13 on our own checkpoint

**Backlog item executed:** P0.1 ("Stationary-Lead ON OUR FLAGSHIP — first non-oracle number, EVAL POD").

### 1.1 What was actually run

`sc13_real_probe.py` on the **dedicated eval pod** (`ssh tanitad-eval`, A40 48 GB), against
**flagship-30k** (`/root/models/flagship-30k/ckpt.pt`, step 29999, the v1 FINAL decision-grade
checkpoint) over the **canonical 40-episode held-out PhysicalAI val**, window 8, stride 2, 2 s
horizon → **3,241 anchors**.

**Proxy honesty (INFER, not FACT):** the canonical val has **no object labels**, so a "stationary lead"
cannot be labelled directly. We label the **observable consequence** instead — a sustained ego
deceleration — which is exactly the response SC-13 measures (braking-onset lead time). The scenario
*identity* is therefore inferred; the *behaviour* is measured.

Anchors, all requiring v(t) ≥ 5 m/s:

| label | definition | n |
|---|---|---|
| `BRAKE_NEAR` | speed drops ≥ 2.0 m/s within the next 2 s (inside the model's horizon) | 157 |
| `BRAKE_FAR` | drops ≥ 1.5 m/s in the **2–3 s** window **and** < 0.75 m/s inside 0–2 s — braking has **not started** and lies **outside** the rollout: the pure anticipation test | **23** |
| `CRUISE` | \|v(t+k) − v(t)\| ≤ 0.5 for all k ≤ 3 s | 1,283 |

Signal in every model arm: **D = CV_forward(2 s) − pred_forward(2 s)** (positive = the imagined future
is *shorter* than constant velocity = an imagined slowdown). Four arms on identical anchors:

- **informed** — true future actions fed to the rollout. **This leaks the braking command**; reported
  only as an upper bound. *(A first pass scored AUROC 1.00 here and I nearly recorded it as the
  result — it is leakage, not anticipation. Recorded as a caught error, per P8.)*
- **held** — future actions = the last observed action repeated. **The real test.**
- **blind** — actions held **and** vision replaced by a constant mean frame: isolates how much of
  `held` comes from vision rather than the ego-state channels.
- **reactive** — −(v(t) − v(t−0.5 s))/0.5. No model, no detection: the "react to what already
  happened" floor.

### 1.2 The speed confound — caught and controlled

The first pass looked strong (`held` AUROC 0.821 on BRAKE_FAR). Then the anchor statistics showed
**braking anchors sit at median v0 8.94 m/s while cruise controls sit at 17.34 m/s** — a signal that
merely grows at low speed would score high while anticipating nothing. Re-scored two independent ways
on the same saved windows: **matched** (per event, only cruise anchors within ±1 m/s of its v0) and
**stratified** (AUROC within v0 bins, pooled by event count).

**BRAKE_FAR — the anticipation test (n = 23 events vs 1,283 cruise):**

| arm | raw AUROC | bootstrap 95 % CI | speed-**matched** | speed-**stratified** |
|---|---|---|---|---|
| informed *(leaks — upper bound)* | 0.869 | [0.799, 0.919] | 0.680 | 0.670 |
| **held (the claim)** | **0.821** | **[0.702, 0.917]** | **0.723** | **0.740** |
| blind (no vision) | 0.686 | [0.546, 0.824] | 0.654 | 0.685 |
| gt_oracle (true 2 s trajectory) | 0.627 | [0.472, 0.766] | 0.633 | 0.668 |
| reactive (kinematic floor) | 0.424 | [0.249, 0.606] | 0.434 | 0.450 |

**BRAKE_NEAR — the near-term control (n = 157):**

| arm | raw | matched | stratified |
|---|---|---|---|
| held | 0.968 | 0.963 | 0.948 |
| blind | 0.945 | 0.955 | 0.936 |
| reactive | 0.938 | 0.956 | 0.931 |

2 s ADE on the same anchors: informed 0.899 m · held 1.195 m · blind 1.264 m · **CV 1.742 m**.

### 1.3 Verdict on the in-domain corpus (PhysicalAI)

Falsifier (registered before the run): *on BRAKE_FAR, AUROC(held) ≤ AUROC(reactive) + 0.02, **or**
AUROC(held) ≤ AUROC(blind) + 0.02 ⇒ no vision-driven anticipation advantage.*

**On PhysicalAI val the falsifier does not fire** — but see §1.4: it fires on the replication, which
supersedes this section as the run's verdict.

- **vs reactive: CLEARS decisively.** 0.723 vs 0.434. The reactive baseline is *below chance* — by
  construction nothing has happened yet at 2–3 s out, so a "react to what already happened" detector
  is not merely weak here, it is **anti-informative**. This is the cleanest result of the run and it
  is the exact axis the Avride ODI docket names ("responding to stationary objects").
- **vs blind: clears numerically (+0.069 matched, +0.055 stratified) but NOT statistically.** The
  bootstrap CI on `held` alone spans ±0.11; the vision increment sits well inside that noise at
  **n = 23**. **We may not claim a vision-driven advantage.**
- **A genuinely interesting side-finding (INFER):** `held` (0.72–0.74) **exceeds `gt_oracle`**
  (0.63–0.67) — the model's imagined 2 s future indicates an upcoming 2–3 s braking *better than the
  actual 2 s trajectory does*. Since BRAKE_FAR windows have no braking inside 0–2 s by construction,
  the true near trajectory is nearly uninformative. So the model is **not** simply tracking the near
  future; it is discounting its own forward prediction on cues the near future does not contain. That
  is the shape a consequence-forward-model should have.
- **The lead-time gradient (INFER, and the most useful structure here):** vision adds **+0.008–0.023**
  at 0–2 s but **+0.055–0.069** at 2–3 s. Kinematics dominate what is about to happen; whatever vision
  contributes shows up **only at longer lead**. If it survives more data, this is the H15 story in one
  curve — and it is also precisely why open-loop ADE (a 0–2 s metric) has never predicted closed-loop
  behaviour for us (`flagship-closed-loop-gap`).

### 1.4 CROSS-CORPUS REPLICATION — the falsifier fires

Because n=23 was the binding constraint, the same probe was run on the **comma2k19 held-out val**
(64 episodes, **8,384 anchors**, 22.6 min) — a corpus with abundant real lead-following. It roughly
doubled the event count (**n = 45** BRAKE_FAR, 209 BRAKE_NEAR) and **contradicted the PhysicalAI
result.**

**BRAKE_FAR, comma2k19 (n = 45 vs 4,640 cruise):**

| arm | raw AUROC | bootstrap 95 % CI | speed-**matched** | speed-**stratified** |
|---|---|---|---|---|
| informed *(leaks)* | 0.869 | [0.804, 0.925] | 0.681 | 0.690 |
| **held (the claim)** | 0.776 | [0.689, 0.850] | **0.538** | **0.605** |
| blind (no vision) | 0.684 | [0.596, 0.761] | **0.608** | **0.549** |
| gt_oracle | 0.640 | [0.536, 0.745] | 0.801 | 0.579 |
| reactive | 0.550 | [0.454, 0.654] | **0.588** | **0.549** |

**BRAKE_NEAR, comma2k19 (n = 209):** held 0.952 / blind 0.944 / reactive 0.923 (matched) — same story
as PhysicalAI: near-term braking is **kinematic**, and vision adds ~nothing.

**Verdict — the pre-registered falsifier FIRES.** After speed control, `held` (0.538 matched / 0.605
stratified) is **at or below** `blind` (0.608 / 0.549) and **indistinguishable from** `reactive`
(0.588 / 0.549). All three sit within each other's noise. **The PhysicalAI advantage does not
replicate, and the corpus where it appeared is the corpus the model was trained on.**

**Two confounds that keep this from being a clean refutation (INFER, stated so nobody over-reads it
in either direction):**
1. **Out-of-domain.** flagship-30k was trained on PhysicalAI; comma2k19 is a domain shift. The ADE
   numbers make this stark: **constant velocity beats the model outright on comma2k19** (CV **1.302 m**
   vs held 1.874 m, informed 1.755 m) whereas on PhysicalAI the model beat CV (1.195 m vs 1.742 m).
   A model that is worse than CV on a corpus has an unreliable "deficit vs CV" signal there **by
   construction** — this may be measuring domain shift, not absence of anticipation.
2. **Regime.** comma2k19 cruise anchors sit at **29.1 m/s** vs PhysicalAI's 17.3 m/s: it is highway-
   dominated, exactly the regime where CV is near-unbeatable and where our own prior work located the
   longitudinal weakness.

**What is nonetheless settled:** we may **not** claim a measured consequence-forward-model advantage.
A single-corpus, in-domain, n=23 positive that fails to replicate is a hypothesis, not a result.

### 1.5 What this changes

- **SC-13 advances `spec-drafted` → `live-measured (falsifier fired)`** — the first row in this database
  with a number from our own checkpoint rather than a design oracle, and the number is **negative**.
  The oracle's headline contrast (collision rate 0.0 vs 0.4) remains oracle-only and is now explicitly
  **unsupported** by the one real-data test we have run. It must not appear in any external narrative.
- **The right next experiment changed.** Before the replication, the obvious follow-up was "get 10×
  more events". After it, that is the *wrong* first move: the question is no longer statistical power,
  it is **whether the PhysicalAI positive was domain-specific or an artefact**. The decisive tests are
  (a) **in-domain, more events** — more PhysicalAI val episodes + stride 1, and (b) an **arm that is
  actually good on comma2k19** (REF-B v2 / REF-C-XL, or a comma-trained checkpoint), so the deficit
  signal is not being read off a model that loses to constant velocity on that corpus.
- **A cheap decisive control now exists:** run the probe on an arm whose ADE **beats CV** on the target
  corpus. If anticipation appears exactly when the model beats CV and vanishes when it does not, the
  signal is a competence artefact, not a capability.
- **Honest limits.** (a) Deceleration is a proxy for "stationary lead" — some events are curves,
  junctions or traffic lights, not leads. (b) `blind` uses a constant mean frame; a stronger control is
  a *shuffled* real frame (same statistics, wrong scene) — next run. (c) One checkpoint, one seed;
  two corpora that disagree. (d) n = 23 / 45 events: both corpora are under-powered, and the negative
  is as noisy as the positive was.

### 1.6 Resource declaration (G-I)

| item | value |
|---|---|
| Resources | **Eval pod A40 48 GB** (`tanitad-eval`) for the probe; local RTX-4060 box (CPU) for the SC-06 oracle + tests |
| Wall-clock | 309 s + 349 s (PhysicalAI, two passes) + **1,359 s** (comma2k19, 8,384 anchors) + speed-matched re-analyses + < 1 s (oracle tests); ~1 h including authoring/iteration |
| Cost | **$0** (standing pod, no new spend) |
| Why not bigger | The eval pod **was** the resource used, per the mandate. Nothing here needs training compute; the constraint is *event count in the val corpus*, not FLOPs. |
| Coordination | `results/LOCK.opponent-analyzer` touched; GPU was idle (0 MiB) at start; training pods untouched. |

Artefacts on the pod: `/root/taniteval/sc13_real_probe.py`, `sc13_speedmatch.py`; results
`results/sc13_flagship30k{,_comma}{,_speedmatched}.json` + `*_windows.pt` (raw substrate — every
re-analysis is free, no model re-run). Archived in-repo with a README that states how to read the arms:
`Opponent Analyzer/Implementation/sc13-real-probe/`.

---

## 2. SECOND EXPERIMENT (G-E/G-H) — SC-06 emergency-scene scenario authored

Intake package `Implementation/incoming/2026-08-07-emergency-scene-scenario/` — `emergency_scene.py`
+ telemetry oracle, **16/16 offline tests** (venvs/tanitad py3.13, numpy 2.5.1; RTX-4060 box, CPU-only,
0.18 s, $0). **SC-06 `catalogued` → `spec-drafted`.**

**The mechanism it isolates.** The Zoox docket is the whole thesis in one sentence: *drove in → failed
to recognize → panic brake → halted inside the scene.* A stack that must **classify an object** before
acting is **range-limited by the obscurant itself** — smoke shrinks the distance at which anything
becomes classifiable, so reaction distance collapses **exactly when the hazard is greatest**.

Design-oracle numbers (**P8 — not a claim about our model**), over the obscurant sweep {0…1}:

| metric | `imagine_and_yield` (H11+H15) | `rule_literal` (documented failure) |
|---|---|---|
| corridor incursion rate | **0.0** | 0.2 |
| mean corridor blockage | **0.0 s** | 2.54 s (**12.7 s** at thick smoke) |
| mean detection lead time | **+5.70 s** | +2.84 s (**−0.10 s** at thick smoke) |
| at a = 1.0 | stops before the boundary | penetrates **15.6 m**, `halted_in_corridor=True` |

**Mechanism in one number:** the obscurant collapses **object**-classification range **90.0 → 13.5 m**
while the **scene**-level OOD range falls only **80.0 → 68.0 m**.

**Two findings worth carrying forward:**

- **The failure is a CLIFF, not a slope (INFER).** Rule-literal incursion is 0 m at ambiguity ≤ 0.75
  and 15.6 m at 1.0 — a panic brake keeps succeeding right up until trigger range drops below stopping
  distance, then the outcome flips. **This predicts an operator can pass ordinary fog/rain testing and
  still fail catastrophically at a real fire**, which is what the Zoox docket describes. Graded
  obscurant sweeps are therefore mandatory; a pass/fail at one weather level proves nothing. This is a
  concrete recommendation for our own eval design, not just an opponent observation.
- **This scenario's core assumption is asserted, not measured, and the module says so.** The
  object-range/scene-range asymmetry **is** the H11 claim. Its falsifier already exists and is already
  failing: **SC-05's D8 probe measures this exact detector** and scored AUROC **0.34–0.59 unpaired**
  (falsifier fired) with only a **+1.60 median paired shift (p ≈ 0.047)** on matched pairs. **SC-06
  must not be scored as an excellence row until the SC-05 detector clears its own bar.** Recorded in
  the INTAKE as a blocking condition rather than left as an optimistic spec.

---

## 3. Opponent deltas (3-day real window — small, but two of them are large)

### 3.1 Zoox — smoke recall (FACT) ★ new evidence, new operator

**Zoox issued a voluntary software recall for 105 vehicles**, notifying NHTSA **2026-07-08** (public
**2026-07-17**). On **2026-06-20** a Zoox robotaxi in Las Vegas **drove into thick smoke from an active
fire**, **failed to recognize the smoke**, then **suddenly applied its brakes and tried to turn**,
coming to a halt. — https://www.cnbc.com/2026/07/17/amazon-zoox-recalls-robotaxi-smoke.html ,
https://techcrunch.com/2026/07/17/zoox-issues-software-recall-after-a-robotaxi-got-confused-by-heavy-smoke/

**Why this matters more than one more incident (INFER):** it makes W-09 and W-04 **cross-operator**.
Waymo alone is a company story; Waymo **and** Zoox, with a federal all-operator directive between them,
is a **class** story — which is exactly the claim our scenario database makes. It also **fuses W-04
and W-09**: smoke is simultaneously an obscurant (SC-05) and an emergency-scene cue (SC-06), so the
same OOD head serves both — noted in the SC-06 intake so we do not build two detectors.

### 3.2 Waymo — the July 4 San Francisco breakdown (FACT) ★ new weakness W-10

Dozens of Waymo vehicles stalled in post-fireworks gridlock around the **Presidio, 2026-07-04**;
**64 cars** had to be retrieved by staff or tow truck, several with **depleted batteries**;
**unplanned road closures** around the Golden Gate Bridge show contributed; one **occupied** vehicle
**drove over a lit firework**, and a separate empty vehicle reportedly caught fire after doing the
same. The SF mayor has called for stricter rules. — https://sfstandard.com/2026/07/05/waymo-sf-gridlock-fourth-of-july-2026/ ,
https://abc7news.com/post/waymo-fleet-clogs-presidio-july-4-fireworks-leaving-vehicles-stranded-towed/19455862/

This also **upgrades SC-08** (fleet stall / frozen vehicle) from a 2022-Cruise anecdote to a **fresh,
large-N FACT** — and it is the proximate trigger for NHTSA's end-July deadline being framed as urgent.

### 3.3 Waymo/NHTSA — the first-responder directive is a *meetings* deadline (FACT, correction)

Run #3 recorded the end-of-July deadline. Refined this run: it is a deadline for **companies to present
fixes in meetings**, **not** to have fixes deployed. — https://www.axios.com/2026/07/15/waymo-accountability-emergencies-nhtsa
**Do not overstate this in the vision deck.** The regulatory pressure is real; a July-31 capability
change is not implied.

### 3.4 The rest (FACT, no material change)

- **Wayve** — **$85 M employee tender (2026-07-01)**; liquidity event, not new capital. Series D
  ($1.2 B, $8.6 B post) + the $60 M AMD/Arm/Qualcomm extension stand. GAIA-3 (15 B, offline eval)
  unchanged → **W-05 intact**. — https://wayve.ai/press/series-d/
- **Pony.ai** — 2026 target reaffirmed: **>3,500 robotaxis across 20+ cities**, robotaxi revenue
  >3.5× 2025; driverless light truck (Apr'26); Uber/Verne Croatia + Stellantis Luxembourg vans.
  **W-06 (thin unit economics) unchanged** — fleet targets keep outrunning revenue.
- **Momenta / Autobrains** — no delta in-window; Uber's Munich L4 (Autobrains + NVIDIA) and the Momenta
  HK listing stand from run #3.
- **NVIDIA Alpamayo** — no delta. **Alpamayo 1 = 10 B** params with open weights; **Alpamayo 2 Super
  = 32 B** "expected this summer" on GitHub/HF; **AlpaSim** open-source. **Still no Nano-tier
  compute-normalized number** — the CNCE gap we intend to fill is still open. (Watch: if the 2-Super
  release lands with a params-vs-benchmark table, our W-05 wedge narrows.) — https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/

### 3.5 Field scan (FACT/INFER) — hierarchy is no longer only ours

- **HWM — "Hierarchical Planning with Latent World Models"** (arXiv **2604.03208**, Zhang, Terver,
  Zholus et al.; subm. 2026-04-03, rev. 2026-06-16). **Deep-read done this run.** Learns world models
  at **multiple temporal scales in a shared latent space**; the long-horizon model's predictions serve
  as **subgoals for the short-horizon model via latent matching**, with no task rewards and no
  hierarchical policy. Reports **up to 3× less planning compute** than single-level planning.
  **This is planning-time hierarchy — our H1 claim — published.** Mitigations for our differentiation
  (INFER): it is **robot manipulation and maze navigation, not driving**; it reports **no parameter
  count**; and it has **no self-monitoring / OOD guarantee** and no in-loop imagination for a safety
  case. But "hierarchy at planning time" can no longer be presented as unclaimed ground.
  **This is also directly relevant to the v3 direction** (frozen encoder + feature-prediction + a
  CEM/MPC planner) — it is the closest published relative of what v3 proposes, from the DINO-WM
  lineage. **Architecture & Inference deep-read: top priority.** — https://arxiv.org/abs/2604.03208
- **WorldRFT** — latent world-model planning with **RL fine-tuning** for AD (2026); new to the watch
  list, not yet read.
- Standing: **SGDrive** (2601.05640) hierarchy read still open from run #3.

---

## 4. Ledger / hypothesis impact

| H | change | evidence |
|---|---|---|
| **H15** | **First real-data test — NEGATIVE on replication.** In-domain (PhysicalAI) the anticipation signal beat a reactive floor 0.72 vs 0.43; on comma2k19 it collapsed to 0.54–0.61, indistinguishable from both the vision-blind and reactive controls ⇒ **falsifier fired**. Status stays OPEN; evidence moves **against** the open-loop version of the claim, with two named confounds (out-of-domain; CV-unbeatable highway regime). | §1.3–1.4 |
| **H11** | SC-06 now depends explicitly on the scene-level OOD claim; SC-05's D8 result is its falsifier and is currently **failing**. Recorded as a blocking condition. | §2 |
| **H1** | **Differentiation pressure:** planning-time hierarchy published (HWM 2604.03208) — outside driving, without params or a safety case. | §3.5 |
| **H0/H6** | Scenario-DB thesis strengthened: W-09 is now cross-operator (Waymo + Zoox) with a federal action. New W-10 (fleet/mission-energy) is a coverage **gap**, incl. for us. | §3.1–3.2 |
| **A9** | The `D = CV − pred` deficit is still the natural monitor feature, but it is **corpus-sensitive** and inverts where the model loses to CV — any monitor built on it needs a competence guard. | §1.4 |

No hypothesis status upgrades (P8). Nothing here is closed-loop; the in-domain positive is
under-powered (n=23) and the cross-corpus test **contradicted** it (n=45). The net movement of
evidence this run is **against** the open-loop form of the H15 anticipation claim — which is precisely
the case for prioritizing the closed loop over more open-loop probing.

---

## 5. Recommendations for other disciplines (no cross-boundary writes)

- **Benchmarks & Eval (Thu):** (a) `D = CV_forward − pred_forward` is a cheap label-free monitor
  feature and beat a kinematic floor at 2–3 s lead **in-domain only** — if you adopt it, adopt it
  **with a competence guard**: it is unreliable on any corpus where the model loses to CV (it did on
  comma2k19). **Do not wire it as an unconditional monitor**; (b) add the
  **blockage-duration + incursion-rate** reducers over SC-06 `_extra` and **unify SC-06's
  `non_nominal_detected` with the SC-05 OOD head — one detector, not two**; (c) the SC-05 D8 detector
  is now a **blocker for SC-06 scoring** — please treat its bar as gating, not informational;
  (d) [standing] min-TTC + collision-rate reducers over SC-13 `_extra`.
- **Data Eng (Tue):** the stopped-lead tagging ask **changed shape** after the replication failed. Raw
  event count is no longer the top need — **in-domain** event count is. Priority order: (1) more
  **PhysicalAI** val episodes / denser sampling (in-domain, where the model is competent); (2) tagged
  **true stopped-lead** segments so the label stops being "any deceleration" (curves and traffic lights
  currently pollute it); (3) only then cross-corpus volume. Also: screen for flashing-light / flare /
  **smoke** events (W-09) — the Zoox recall makes smoke the highest-value visual cue in the corpus.
- **Architecture & Inference (Wed):** **deep-read HWM (2604.03208) as top priority** — planning-time
  hierarchy over multi-timescale latent world models, 3× less planning compute, from the DINO-WM
  lineage. It is both the closest published competitor to H1 **and** the closest published relative of
  the **v3** direction. Second: SGDrive (2601.05640), still open.
- **Tools & DevEnv (Mon):** CARLA emergency-vehicle / flashing-light / flare / cone assets **plus a
  smoke volumetric or photometric overlay** for SC-06's `carla_recipe()`; **AlpaSim** evaluation still
  open from run #2.
- **Orchestrator:** (a) triage the SC-06 intake; **the SC-13 dedup verdict from run #3 is still
  outstanding** and now blocks a `live-measured` row from being wired cleanly; (b) **W-10 is a
  strategy gap marked `no-counter-yet`** — the 4-brain has a strategic layer *by design* but nothing
  measured; decide whether mission-feasibility/energy is in Phase-0 scope or explicitly deferred;
  (c) narrative: "emergency scenes are not edge cases" is now backed by **two operators**, but the
  end-July deadline is for **meetings, not fixes** — do not overstate it.

---

## 6. Self-critique (gate check)

- **G-A** every claim sourced (link or repo path) ✔ · **G-B** actionable recs tied to H ✔ ·
  **G-C** KB updated ✔ · **G-D** ledger row ✔ · **G-E** intake pkg 16/16 ✔ ·
  **G-H** measured experiment with numbers, hardware, falsifier verdict ✔ · **G-I** resource
  declaration §1.5 ✔ · **G-O1** labels throughout ✔ · **G-O2** W-10 marked `no-counter-yet` ✔.
- **Weakest points, stated plainly:** (1) both corpora are under-powered (n = 23 and 45) — the negative
  is as noisy as the positive was, and neither settles H15; (2) the comma2k19 test is confounded by
  domain shift *and* by the model losing to CV there, so it is a failed replication, **not** a clean
  refutation; (3) the `blind` control uses a mean frame, not a shuffled real frame, so it may understate
  vision; (4) deceleration is a proxy for "stationary lead" — curves and traffic lights pollute the
  label; (5) one checkpoint, one seed; (6) the news window was 3 real days — breadth is thin by
  construction.
- **Process note worth keeping:** the run's headline **reversed** between the first result and the
  replication. Had the session ended after the PhysicalAI pass, this note would have claimed a positive
  measured result for H15. The replication cost 23 minutes of idle pod time. **Single-corpus results
  should not leave this discipline.**
- **Caught error worth recording (P8):** the first version of the probe fed **true future actions** to
  the rollout and scored AUROC 1.00. That is command leakage, not anticipation. It was caught by asking
  why a result was implausibly perfect. Any future "our model anticipates" claim must state its action
  conditioning explicitly.

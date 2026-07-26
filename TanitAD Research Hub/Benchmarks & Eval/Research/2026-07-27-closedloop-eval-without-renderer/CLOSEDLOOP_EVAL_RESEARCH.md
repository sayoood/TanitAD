# Evaluating closed-loop driving WITHOUT a photorealistic renderer — what the field actually does, and what TanitAD should do

**Date:** 2026-07-27 (Europe/Berlin) · **Stream:** Benchmarks & Eval · **Mode:** CPU / web only. **NO pod contacted.**
**Answers:** the open wound left by `…/2026-07-26-p1-envelope-revalidation/P1_REVALIDATION.md` — our closed-loop
instrument cannot produce a MEASUREMENT at any admissible horizon.
**Pre-registration:** `PREREGISTRATION.md` in this directory, written and **staged before any source was read**.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited, specific document) ·
`INHERITED` (another agent/doc, **not** re-verified) · `ESTIMATED` · `HYPOTHESIS` · `UNVERIFIED`.
Within `PUBLISHED` I additionally separate **DEMONSTRATED** (a statistic, an n, an external criterion) from
**ASSERTED** (stated confidently with no supporting statistic). *This separation is the point of the document.*

---

## 0. Headline

| # | Result | Class |
|:--:|---|---|
| **1** | ⭐⭐⭐ **The field already solved our exact problem, and the solution is a PROTOCOL CHANGE, not a renderer.** *Pseudo-Simulation* (NAVSIM v2, CoRL '25) evaluates by **pre-generating a bounded grid of perturbed observations offline and never rolling out sequentially**. Deviation is therefore bounded **by construction** and can never leave a validated envelope. It reports **R² = 0.80 (Pearson r = 0.89), n = 83 planners** against nuPlan closed-loop, vs **R² = 0.70 (r = 0.83)** for the best open-loop approach. | `PUBLISHED` **DEMONSTRATED** |
| **2** | ⭐⭐⭐ **Their perturbation grid is NARROWER than our measured envelope.** Pseudo-Sim samples lateral **every 0.5 m up to ±2.0 m** and filters heading mismatch at **20°**. Our envelope is **\|dlat\| ≤ 3.0 m, \|dpsi\| ≤ 12°** (usable yaw edge measured at **15.47°**). ⇒ **A pseudo-simulation protocol built on our OWN warp would be 0 % out-of-envelope — a MEASUREMENT, not an extrapolation.** This is the repair, and it needs no renderer, no licence, and no new corpus. | `PUBLISHED` + `INHERITED` |
| **3** | ⭐⭐ **NOBODY in this literature demonstrates correlation with REAL driving.** Every "closed-loop correlation" claim found — NAVSIM v1, Pseudo-Sim, the 2026 cross-benchmark study — correlates a simulator against **another simulator**. Pseudo-Sim says so explicitly: *"We do not yet demonstrate or claim direct correlation with … real-world vehicle deployment."* **Pre-registered prediction P-a CONFIRMED.** | `PUBLISHED` |
| **4** | ⭐⭐ **No benchmark publishes an ego-deviation validity bound. Our envelope is MORE rigorous than published practice.** nuPlan acknowledges closed-loop non-reactive *"quickly diverges"* but never bounds it; Pseudo-Sim does **not** quantify maximum reliable synthesis distance beyond its sampling grid. **Pre-registered prediction P-b CONFIRMED ⇒ condition R3 FIRES.** Our wound is a **disclosure advantage**, and publishable. | `PUBLISHED` |
| **5** | ⚠️⚠️ **LICENCE LANDMINE — Waymax forbids exactly what TanitAD is.** The *Waymax License Agreement for Non-Commercial Use* states you may not use it *"to train or otherwise develop or improve (directly or indirectly) an artificial intelligence foundation model."* TanitAD is a latent world model. **Recommend NOT adopting Waymax without legal review.** Short name "open-source Waymo simulator" would have hidden this — the third instance of the licence-from-short-name class (after ZOD, nuScenes). | `PUBLISHED` (licence doc fetched) |
| **6** | ⚠️ **Bench2Drive is CC-BY-NC-ND — No Derivatives, on CODE AND ASSETS.** *"All assets and code are under the CC-BY-NC-ND unless specified otherwise."* Same derivative prohibition as AlpaSim's NGC container licence. Plus 400 GB–4 TB and ~5 DS points run-to-run variance. **Rank low.** | `PUBLISHED` (repo licence fetched) |
| **7** | ⛔ **Our own reactive-agent option is already dead, and it was measured, not assumed.** The AlpaSim `trafficsim` reactivity probe returns **NOT REACTIVE**: GO-vs-STOP agent displacement Δ = **0.0229 m**, CI **[−0.0117, 0.0625]**, `separated: false`; near-ego-50 m Δ = **0.0044 m**. It is functionally log-replay. | `INHERITED` (raw artifact read directly) |
| **8** | ⚠️ **The strongest open reactive-agent models cannot be run out of the box.** CAT-K (CVPR '25 Oral, 7 M params, WOSAC #1): *"We cannot share pre-trained models according to the terms of the Waymo Open Motion Dataset."* SMART likewise. **Checkpoints are gated by dataset registration, not by licence text.** | `PUBLISHED` |
| **9** | ⭐ **A metric-saturation warning we can use immediately.** In the cross-benchmark study, **Comfort saturates at ≥ 99.9 %** contributing *"essentially zero discriminative information"*, TTC is near-saturated, and a **3-metric formula matches the full 5-metric PDMS at the same ρ = 0.90**. And **Ego Progress (ρ = 0.83) predicts closed-loop better than the collision metric NC (ρ = 0.45)**. | `PUBLISHED` **DEMONSTRATED** (n = 8) |
| **10** | ⚠️ **A substrate asymmetry that constrains the repair, found in our own code.** Our yaw warp is depth-independent and exact; the **lateral arm is NOT** — `yaw_geometry.py` states the flat-road assumption *"plays no role on this arm, **unlike the lateral arm**"*. ⇒ we can synthesise **heading** perturbations exactly and **lateral** ones only under flat-road. Pseudo-Sim's 3DGS handles both. **This is the one real gap between us and them, and it is the cheapest discriminating experiment.** | `MEASURED` (code) |

### 0.1 The verdict in one sentence

**Repair our instrument by changing the PROTOCOL (bounded pre-generated perturbation grid instead of
sequential rollout) — this converts our numbers from EXTRAPOLATION to MEASUREMENT at zero licence cost —
and in parallel adopt NAVSIM on OpenScene for a comparable, publishable number that no internal
instrument can ever give us.** The pre-registration binds this ordering: conditions **R3 and R4 both fire**.

---

## 1. Pre-registration verdict — did my own committed conditions fire?

`PREREGISTRATION.md` §1 committed, before any source was read, to four conditions that would make me
recommend **repairing** over **adopting**. Scored honestly, including where they did not fire.

| # | Condition | Fired? | Evidence |
|:--:|---|:--:|---|
| **R1** | No external benchmark publishes evidence that its score predicts real or full closed-loop driving | **PARTIAL** | They **do** publish correlation evidence — but **only against other simulators**, never against real driving (§3). So the condition fires in its *real-driving* form and fails in its *closed-loop* form. **I report both sides and do not score this as a win for repair.** |
| **R2** | Every external benchmark requires an asset our corpus cannot supply | **NO** | All require an HD map / lane graph, which PhysicalAI-AV lacks (5 probes, `INHERITED`) — **but all can be run on their own bundled data**, which is the escape clause I pre-committed to. **Prediction P-c CONFIRMED: the answer is "run on THEIR data", not "port our corpus".** |
| **R3** | Our out-of-envelope problem is shared by the field, which simply does not report it | ⭐ **YES** | §4. No published ego-deviation validity bound exists. nuPlan concedes divergence qualitatively; Pseudo-Sim explicitly declines to quantify it. Our envelope is more rigorous than the field's practice. |
| **R4** | A cheap, bounded repair exists converting EXTRAPOLATION → MEASUREMENT | ⭐ **YES, decisively** | §2. Pseudo-simulation is exactly that design, and its perturbation grid is **narrower than our measured envelope**. |
| **A1** | An external benchmark **demonstrates** rank-consistency with a fuller criterion | **YES** | Pseudo-Sim R² = 0.80, n = 83; cross-benchmark ρ = 0.90, n = 8. |
| **A2** | Runnable on CPU / one non-training GPU, on its own data, under a workable licence | **YES, with a copyleft caveat** | NAVSIM code Apache-2.0; **data CC BY-NC-SA 4.0 + nuPlan non-commercial agreement** (§5). |
| **A3** | Yields a number comparable to a public leaderboard | **YES** | NAVSIM navtest/navhard leaderboards. |

**Pre-registered decision rule, applied as written:** *"If any of R1–R4 holds, 'repair ours' outranks 'adopt
external'."* **R3 and R4 hold.** ⇒ **Repair ranks first.** A1–A3 also hold ⇒ **adopt is not refused, it is
ranked second and pursued in parallel.** I am not permitted by my own pre-registration to lead with adoption,
and I do not.

---

## 2. ⭐ THE DECISIVE FINDING — pseudo-simulation, and why it dissolves our envelope problem

### 2.1 What it is

*Pseudo-Simulation for Autonomous Driving* — Cao, Hallgarten, Li, Dauner, Gu, Wang, Miron, Aiello, Li,
Gilitschenski, Ivanovic, Pavone, Geiger, Chitta (Tübingen / NVIDIA / Bosch / Shanghai AI Lab / Toronto /
Stanford), **arXiv 2506.04218, CoRL 2025**. It is the paper behind **NAVSIM v2** and the `navhard` leaderboard.

Verbatim from the abstract:

> "Pseudo-simulation operates on real datasets, similar to open-loop evaluation, but augments them with
> synthetic observations generated **prior to evaluation** using 3D Gaussian Splatting."

and

> "Our key idea is to approximate potential future states the AV might encounter by generating a diverse set
> of observations that vary in position, heading, and speed."

**Mechanism** (`PUBLISHED`): real observations → MTGS (Multi-Traversal Gaussian Splatting) scene
reconstruction → neural rendering at perturbed poses → rejection sampling → synthetic multi-view images with
plausible motion history. A **proximity-based weighting scheme** assigns higher importance to synthetic
observations that best match the AV's likely behaviour.

### 2.2 ⭐ Why this is the answer to OUR wound, specifically

Our instrument fails because **sequential rollout lets the ego walk arbitrarily far from the logged pose**,
and the further it walks the less the substrate can honestly render. Out-of-envelope fractions:
K = 20 → **12.3 %**, K = 60 → **50.7 %**, K = 185 → **90.2 %** (`INHERITED`, P1 revalidation).
That fraction is a **monotone consequence of unrolling**. Widening the envelope cannot reach zero — P1 proved
that arithmetically at yaw = ∞ (lateral clause alone leaves 3.75 % at K = 20).

Pseudo-simulation removes the mechanism instead of fighting it:

| | our closed loop | pseudo-simulation |
|---|---|---|
| how deviation arises | **emergent** — accumulates over K steps of rollout | **chosen** — a fixed grid sampled before evaluation |
| upper bound on deviation | **none** (that is the wound) | **the grid** — ±2.0 m lateral, 5.0 m longitudinal steps, 20° heading filter |
| can it leave the validated envelope? | **yes, at every admissible K** | **no, by construction** |
| requires a renderer at eval time? | yes | **no** — observations are pre-generated offline |
| recovers error-recovery / causal-confusion signal? | yes | **yes** — that is the paper's explicit claim |

Verbatim on the last row:

> "This enables evaluating error recovery and the mitigation of causal confusion, as in closed-loop
> benchmarks, **without requiring sequential interactive simulation**."

### 2.3 ⭐⭐ The number that makes this actionable for us

| axis | Pseudo-Simulation's grid (`PUBLISHED`) | TanitAD's envelope (`INHERITED`, P1) | headroom |
|---|---|---|---|
| lateral | sampled every **0.5 m up to 2.0 m each side** | `ENV_LAT_MAX` = **3.0 m** | **1.5×** |
| heading | matching filter rejects mismatch **> 20°** | `ENV_YAW_MAX` = **12°** shipped; **15.47 °** [12.14, 17.88] measured usable edge | **0.6–0.77×** ⚠️ |
| longitudinal | every **5.0 m**, bounded by ±4.0 m/s² reachability | not an envelope axis for us | — |

**Read this carefully — it cuts both ways and I report both.**

- On the **lateral** axis our validated envelope is **wider** than theirs. A grid at ±2.0 m is comfortably
  inside `ENV_LAT_MAX = 3.0 m`. ✅
- On the **heading** axis theirs is **wider** than ours (20° filter vs our 15.47° usable edge). Their 20° is
  a *rejection filter on matched motion history*, not a synthesis bound, so the two are **not strictly
  commensurable** — but I will not stretch that into a claim that we match them. **We would run a narrower
  heading grid than NAVSIM v2 does.** That is a real, disclosable limitation, not a defect: our 15.47° is
  *measured*, theirs is *chosen*.

**Consequence, and it is arithmetic rather than a prediction:** a perturbation grid chosen at
`|dlat| ≤ 2.0 m, |dpsi| ≤ 12°` has an out-of-envelope fraction of **exactly 0** — because the grid *is* the
deviation and we pick it. `ood.verdict` returns `MEASUREMENT` iff zero windows are outside. **⇒ the protocol
change converts our headline closed-loop number from EXTRAPOLATION to MEASUREMENT, with no renderer, no
licence, and no corpus migration.**

### 2.4 What it costs, and the one thing it does not give us

- **Preprocessing:** *"approximately 1–2 hours per scene on current hardware … limits scalability for
  extremely large datasets"* — but that is **their** 3DGS reconstruction cost. **Ours is a homography warp
  that already runs**, so this cost largely does not transfer. ✅
- **Traffic realism:** *"relatively simple, rule-based traffic models"*; agents *"strictly follow
  road-centerline paths during Stage 2 evaluation."* Even the SOTA design does **not** have reactive traffic.
  ⇒ our lack of a reactive `trafficsim` (§6) is **not** the gap we thought it was.
- **What it does NOT give us:** comparability. A pseudo-simulation on our own corpus is still an internal
  number. That is precisely what Option 2 is for.

### 2.5 ⚠️ The honest gap — our substrate is not their substrate

`MEASURED` (our code, `…/2026-07-26-p1-envelope-revalidation/scripts/yaw_geometry.py:12-17`):

> "a pure camera rotation induces `H = K R K^-1`, exact for ANY scene geometry" … "the flat-road assumption
> plays no role on this arm, **unlike the lateral arm**"

So:

| perturbation axis | our substrate | fidelity |
|---|---|---|
| **heading (yaw)** | `K R K⁻¹` homography | **exact for arbitrary depth** (`max\|ΔH\| = 0.000e+00`, 30 conditions) — only FOV fabrication degrades it |
| **lateral** | homography under a **flat-road plane assumption** | **approximate** — error grows with scene non-planarity |
| **longitudinal / speed** | not implemented as a warp axis | ❌ absent |

Pseudo-Sim's 3DGS handles all three without a planarity assumption; our warp does not. **This is the single
substantive technical gap and it is the cheapest discriminating experiment** (§7.4). It is *not* a reason to
prefer their renderer — theirs costs 1–2 h/scene and, in AlpaSim's packaging, carries a
derivative-forbidding licence.

---

## 3. ⭐ The NAVSIM correlation question — what is DEMONSTRATED vs ASSERTED

The brief asked whether NAVSIM's claim — that non-reactive, short-horizon simulation with a well-designed
metric correlates with full closed-loop — actually holds. **It holds, but against simulators, not reality**,
and the evidence tiers differ sharply.

| study | claim | statistic | n | **criterion it correlates against** | class |
|---|---|---|:--:|---|---|
| **NAVSIM v1** (NeurIPS '24 D&B) | PDMS correlates with closed-loop better than open-loop | Spearman + Pearson, *"consistently … better closed-loop correlation for PDMS"* — **values shown in Fig. 3(a) graphically, not stated numerically in the text I could read** | **151 planners** (37 rule-based, 12 PDM-Closed variants, 114 learned) on navmini, 396 scenarios | **nuPlan Closed-Loop Score (CLS)** — a simulator | `PUBLISHED` **DEMONSTRATED** (design) / ⚠️ **numeric value UNVERIFIED** |
| **Pseudo-Simulation** (CoRL '25) | pseudo-sim correlates with closed-loop better than the best open-loop method | **r = 0.89, R² = 0.80** vs open-loop r = 0.83, R² = 0.70 | **83 planners** (37 rule-based, 46 learned), 244 Stage-1 + 4 164 Stage-2 observations | **nuPlan simulator** | `PUBLISHED` **DEMONSTRATED** ⭐ strongest |
| **Cross-benchmark study** (arXiv 2605.00066, Bosch, Apr 2026) | NAVSIM v2 PDMS predicts CARLA closed-loop Driving Score | **Spearman ρ = 0.90 (p = 0.002), Kendall τ = 0.79 (p = 0.005)** | **n = 8 methods** with complete paired data (from 15 surveyed) | **Bench2Drive Driving Score** — CARLA, a different simulator | `PUBLISHED` **DEMONSTRATED but WEAK TIER** ⚠️ |
| same study, control | classical open-loop L2 predicts nothing | **ρ = −0.36 (p = 0.43)** — *"no significant correlation"* | n = 7 | Bench2Drive DS | `PUBLISHED` **DEMONSTRATED** |

### 3.1 ⚠️ Why the ρ = 0.90 headline is the weakest of the three, despite being the most quotable

Read the abstract's own method statement:

> "By systematically **cross-referencing published results** from 15 state-of-the-art methods across NAVSIM
> (open-loop) and Bench2Drive (closed-loop), we compile a paired dataset…"

This is a **meta-analysis of leaderboard numbers produced by different authors under different
configurations** — not a controlled experiment where one team ran the same checkpoints through both. The
paper lists this itself: *"cross-architecture comparisons introduce configuration drift confounders"*,
*"NAVSIM v1 vs. v2 inconsistencies across methods"*, *"Bench2Drive variance (~5 DS points from run-to-run
variability)"*, and *"purely correlational … no causal claims"*, at **n = 8**.

**This program has a standing rule against exactly this shape of number** (an exponent or a coefficient
quoted bare, without its window and n). **A ρ = 0.90 at n = 8 from pooled third-party leaderboard entries is
not decision-grade for a GPU-day.** The **Pseudo-Simulation R² = 0.80 at n = 83, produced by one team running
one pipeline, is.** I rank them accordingly and I do not lead with the bigger number.

### 3.2 The negative findings, reported at equal prominence (pre-committed in §2 of the pre-registration)

- **Ranking inversions exist.** "SafeDrive" ranks **3rd** in PDMS but **5th** in Driving Score. The
  correlation is explicitly *"strong positive but **non-monotonic**"*.
- **The safety/progress trade-off flips sign between paradigms.** *"methods that maximize safety at the
  expense of progress rank highly in NAVSIM but underperform in closed-loop due to timeout and slow-driving
  penalties."* ⇒ **a policy tuned to a non-reactive safety metric can be actively worse closed-loop.**
- **NAVSIM's own authors agree.** From the NAVSIM limitations: *"A high PDMS does not always imply a high
  CLS, since our framework does not consider reactiveness or the compounding accumulation of errors in
  closed-loop simulation."* And: *"We strongly encourage the use of graphics-based closed-loop simulators,
  such as CARLA, as complementary benchmarks to NAVSIM."* **NAVSIM does not claim to replace closed-loop.**
- **The residual gap has a named mechanism:** the *"snowball effect"* — small open-loop deviations compounding
  into closed-loop failures. This is the same compounding our rollout suffers from.

### 3.3 ⛔ The claim NOBODY makes

Across every source fetched, **no benchmark demonstrates that its score predicts real on-road driving.**
- Pseudo-Simulation, verbatim: *"We do not yet demonstrate or claim direct correlation with performance
  metrics from real-world vehicle deployment."*
- SimNet (ICRA '21), which trains reactive simulation from 1 000 h of logs and is the closest thing to a
  sim-vs-real study, presents **realism and reactivity** metrics — **no correlation statistic against on-road
  outcomes** in the material I could read (`UNVERIFIED` beyond the abstract).
- Sim2Val (CoRL '25) *assumes* sim/real correlation as a **variance-reduction control variate**; it does not
  establish that sim scores predict real driving (`UNVERIFIED` beyond the abstract).

**⇒ Pre-registered prediction P-a CONFIRMED.** Adopting an external benchmark buys **comparability and a
validated-against-simulation score**. It does **not** buy validity against reality. Anyone who tells us
otherwise is quoting a reputation, not a paper.

---

## 4. ⭐ The replay-validity literature — the exact question our envelope was trying to answer

**This was the highest-value negative result of the study.** I probed four ways (rule 2: absence at one
location is not absence): the nuPlan benchmark paper, the Pseudo-Simulation paper, targeted searches for
divergence thresholds, and the industry re-simulation literature.

### 4.1 What the field says

| source | on ego deviation vs replay validity | bound published? |
|---|---|:--:|
| **nuPlan** (arXiv 2403.04133) | Three modes: open-loop, **closed-loop non-reactive** (log replay), closed-loop reactive (IDM). Log-replay agents *"propagate agents according to the logged data"* giving *"a near-perfect recreation of the recorded data"*, **however** *"closed-loop simulation quickly diverges if the planner decides to take different actions from what is recorded in the log."* | ❌ **acknowledged qualitatively, never quantified** |
| **Pseudo-Simulation** (2506.04218) | Imposes constraints — history matching filters velocity > 1.0 m/s, acceleration > 1.0 m/s², **heading > 20°**; rejection sampling on EPDMS constraint violations; scenes discarded below 5 valid synthetic observations; visual-quality filtering | ⚠️ **operating grid published (±2.0 m lat, 5.0 m long); "does not quantify maximum reliable synthesis distance from logged poses beyond the sampling grid"** |
| **nuPlan-R** (arXiv 2511.10403) | Attacks the *agent-model* half of the problem, not the deviation half: rule-based IDM agents *"lack behavioral diversity and fail to capture realistic human interactions, leading to oversimplified traffic dynamics"* | ❌ (different question) |
| **Bench2Drive** (NeurIPS '24 D&B) | Sidesteps replay entirely — full CARLA synthesis, so there is no logged path to deviate from | n/a |
| **Industry re-simulation practice** (Applied Intuition engineering blogs) | Defines *pose divergence* / *simulation drift*, measures **Log Divergence = L2 between simulated and logged poses**, transforms actor positions to the simulated ego pose, and states divergence *"may … even invalidate results in high divergence cases"* | ❌ **no published threshold; not peer-reviewed** `UNVERIFIED` |

### 4.2 ⭐ The conclusion, and it reframes our wound

**There is no principled published treatment of "how far off the logged path can you go before replay is
meaningless."** The field's actual practice is one of three things:

1. **Ignore it** (nuPlan CL-NR: concede divergence, publish the score anyway);
2. **Design it away** (Pseudo-Sim: never roll out, so deviation is chosen not accumulated);
3. **Escape it** (CARLA/Bench2Drive: synthesise the whole world, pay 400 GB–4 TB and CC-BY-NC-ND for it).

**Nobody does what P1 did — measure the edge and then refuse to call the number a measurement.**

**⇒ Pre-registered prediction P-b CONFIRMED. Condition R3 FIRES.** Our envelope work is not a wound relative
to the field; it is **an instrument the field does not have**. The correct response is (a) adopt design
route 2, and (b) **publish the envelope methodology** — a measured deviation-validity bound for
warp-based closed-loop evaluation, with the C-GEO model-free geometry (fabricated-pixel fraction, FOV
half-angle 25.70°) and the destroyed-observation dynamic-range controls, is a genuine contribution to a
literature that currently has none.

⚠️ **Caveat I must state against my own conclusion:** P1 also found that the warp is geometrically exact and
that *"roughly half of what the envelope measures is our own arm's OOD sensitivity"* (`INHERITED`). A bound
that is half a property of one checkpoint is **not** a substrate constant, and must not be published as one.
The publishable object is the **method** (how to measure such a bound, with dynamic-range controls), plus an
arm-conditioned number — not "the envelope of warp-based simulation."

---

## 5. Per-benchmark table — simulates / does NOT simulate / validity evidence / licence

⚠️ **Licence = the document, not the short name.** Every row below was fetched from the actual licence text or
repository licence statement, and **code and data licences are separate fields**. Where I could not fetch the
document, the field says `UNVERIFIED` rather than a guess.

| benchmark | renders images? | what it SIMULATES | what it does NOT simulate | validity evidence (DEMONSTRATED vs ASSERTED) | **code licence** | **data licence** |
|---|:--:|---|---|---|---|---|
| **NAVSIM v1** (NeurIPS '24 D&B) | ❌ no — replays recorded sensor data, no rendering | Non-reactive 4 s unroll @10 Hz of a **fixed** planned trajectory; kinematic bicycle model + LQR; collision / drivable-area / comfort metrics | **Reactive agents** (ego actions do not affect others); **sensor updates during the 4 s window**; long-horizon compounding; traffic lights, stop signs; rear-end-into-ego as at-fault | **DEMONSTRATED** vs nuPlan CLS, 151 planners / 396 scenarios — *values graphical, `UNVERIFIED` numerically*. **ASSERTED**: nothing about real driving | **Apache 2.0** — *"All assets and code in this repository are under the Apache 2.0 license unless specified otherwise."* | ⚠️ **OpenScene: CC BY-NC-SA 4.0 + nuPlan Dataset License Agreement for Non-Commercial Use** — non-commercial **and copyleft** |
| **NAVSIM v2 / Pseudo-Sim** (CoRL '25) | ⚠️ **offline only** — 3DGS/MTGS pre-generates perturbed views before evaluation | Two-stage pseudo-simulation on `navhard`; Stage 1 nominal driving, **Stage 2 corrective behaviour from perturbed states**; EPDMS adds Traffic-Light Compliance, Driving-Direction Compliance, Lane Keeping, Extended Comfort | Sequential interaction; **reactive traffic** (*"relatively simple, rule-based"*, agents *"strictly follow road-centerline paths"*); real-world validity (explicitly disclaimed) | ⭐ **DEMONSTRATED**: **r = 0.89 / R² = 0.80, n = 83 planners** vs nuPlan CL, beating best open-loop r = 0.83 / R² = 0.70 | **Apache 2.0** (same repo) | same as above ⚠️ |
| **nuPlan** (arXiv 2403.04133) | ❌ no — state-space only | Three modes: OL, closed-loop **non-reactive** (log replay), closed-loop **reactive** (IDM background traffic); bicycle model with PID/LQR tracking; CLS = weighted progress + TTC + speed-limit + comfort, zeroed by hard penalties | Sensor/image simulation; behavioural diversity in agents (IDM only); *"quickly diverges"* under planner deviation | **ASSERTED** for real-driving validity. **DEMONSTRATED against it**: nuPlan-R shows IDM *"substantially overestimates planner robustness and conceals failure modes"* ⚠️ (specific numbers `UNVERIFIED` — abstract only) | **nuplan-devkit: Apache 2.0** | ⚠️ **nuPlan Dataset License Agreement for Non-Commercial Use**; derived redistributions **CC BY-NC-SA**. Free for academic use; commercial licensing via Motional |
| **Waymax** (arXiv 2310.08710) | ❌ **no rendering** — trajectory/state-space, runs in-graph on TPU/GPU | Multi-agent scenes initialised or played back from Waymo Open Motion Data; **learned and hard-coded behaviour models** for reactive interaction; supports in-graph RL training | Images/sensors entirely | **ASSERTED**. Benchmarks IL/RL algorithms and shows *"the ability of RL to overfit against simulated agents"* — a negative result about its own agents, honestly reported | ⛔⛔ **"Waymax License Agreement for Non-Commercial Use"** (2023-10-17): prohibits real-world vehicle operation, validation, commercial scenario simulation, Production Systems, **and** *"to train or otherwise develop or improve (directly or indirectly) an artificial intelligence foundation model"*; may not convey unmodified materials | **Waymo Open Motion Dataset:** non-commercial; *"any machine-learning model, software, or algorithm, including architectures, weights, and parameters"* trained on it **is a derivative work** subject to the terms; redistribution only to registered users |
| **GPUDrive** (arXiv 2408.01584) | ❌ no rendering — state-space, Madrona engine, >1 M steps/s | Multi-agent driving from WOMD scenarios; C++/CUDA observation, reward and dynamics functions; BVH collision; goal-reaching RL agents trainable in minutes–hours | Images/sensors; anything outside WOMD scenario geometry | **ASSERTED** (throughput and RL results, not evaluation validity) | ✅ **MIT** | ⚠️ **Waymo Open Motion Dataset** (non-commercial, derivative-work clause as above) |
| **MetaDrive** (arXiv 2109.12674) | ⚠️ lightweight 3D rendering (not photorealistic) | Procedurally generated + real-data-imported scenarios; physics up to 300 FPS on a standard PC; single- and multi-agent RL; safe-exploration benchmarks | Photorealistic sensors; real-log fidelity when procedurally generating | **ASSERTED** — designed for RL *generalisation*, not for predicting real driving | ✅ **Apache 2.0** — the cleanest licence in the field | ✅ **none required** for procedural scenarios (real-data import inherits that dataset's licence) |
| **Bench2Drive / CARLA** (NeurIPS '24 D&B) | ✅ full synthesis (CARLA v2 / 0.9.15) | Fully closed-loop E2E driving; **44 interactive scenarios × 23 weathers × 12 towns = 220 routes**; reactive traffic; Driving Score + Success Rate | Real-log fidelity — it is synthetic throughout; sim-to-real transfer | **ASSERTED** for real driving. It **DEMONSTRATES** its own critique of prior practice: fixed-route CARLA scores are *"known for high variance"*; the 2026 study measures **~5 DS points run-to-run variance** | ⛔ **CC-BY-NC-ND** — *"All assets and code are under the CC-BY-NC-ND unless specified otherwise."* **No Derivatives** | ⛔ same CC-BY-NC-ND; **~4 GB mini / ~400 GB base / ~4 TB full** |
| **CARLA** itself | ✅ | full simulator | real-log fidelity | n/a | ✅ **MIT** (CVC/UAB, 2017) | ⚠️ asset/content licence **UNVERIFIED** — repo LICENSE covers code; a separate asset licence is commonly claimed but I could not fetch the document |
| **TanitAD / AlpaSim NuRec** (ours) | ✅ gsplat | reconstruction-based closed-loop | — | envelope MEASURED but **verdict = EXTRAPOLATION at every admissible K** | ⛔ **NGC-DL-CONTAINER-LICENSE — forbids derivatives** (`INHERITED`) | reconstruction **3.21× OOD** (`INHERITED`) |
| **TanitAD warp closed loop** (ours) | ⚠️ homography warp | yaw-exact, lateral flat-road | reactive agents; longitudinal/speed perturbation | ⛔ K = 20 **12.3 %** out-of-envelope ⇒ never a MEASUREMENT | ours | ours |

### 5.1 ⚠️ The licence findings that would have been missed from short names

1. **"Waymax is Waymo's open-source simulator"** → in fact a bespoke non-commercial licence that
   **prohibits using it to develop or improve an AI foundation model, directly or indirectly.** TanitAD is a
   sub-300 M latent world model. Whether a world model for driving is a "foundation model" is a legal
   question, not an engineering one. **Do not adopt Waymax without a decision from Sayed.**
2. **"Bench2Drive is an open benchmark"** → **CC-BY-NC-ND on code and assets.** ND means we may not
   distribute a modified evaluator. Same class of restriction as the NGC container licence we already hit.
3. **"NAVSIM is Apache-2.0"** → **the code is; the data is not.** OpenScene is **CC BY-NC-SA 4.0** —
   non-commercial **and share-alike**, the identical trap as nuScenes. ShareAlike can attach to derived
   datasets we build on it.
4. **"GPUDrive is MIT"** → the **engine** is MIT; every scenario comes from WOMD, whose terms declare
   **trained model weights a derivative work**.

---

## 6. Reactive agents we could actually run

### 6.1 ⛔ Our own option is already measured dead

`INHERITED` (raw artifact read directly, not re-run):
`…/Architecture & Inference/Implementation/incoming/2026-07-26-trafficsim-wheelbase/artifacts/ts_reactivity.json`

| stratum | GO-vs-STOP agent displacement | GO-vs-GO2 floor | **Δ** | 95 % CI | separated? | verdict |
|---|---|---|---|---|:--:|---|
| all (197 agents, 15 103 samples) | 0.5349 m | 0.5120 m | **0.0229 m** | [−0.0117, 0.0625] | ❌ | **NOT REACTIVE** |
| near-ego ≤ 50 m (155 agents) | 0.3096 m | 0.3052 m | **0.0044 m** | [−0.0186, 0.0298] | ❌ | **NOT REACTIVE** |

Estimator: `paired_episode_cluster_bootstrap`, unit = AGENT, B = 2000 — the decision-grade estimator, not
`overlapping_holdout_se`. The control confirms the intervention was real (`intervention_reached_model: true`;
ego GO-vs-STOP mean 6.16 m, max 21.50 m) — **the ego moved 6 m and the agents moved 2 cm.**

⇒ **AlpaSim's `trafficsim` is functionally log-replay even when enabled.** Enabling it buys nothing.

### 6.2 The strongest open reactive-agent models — and why we cannot just run them

| model | what it is | strength | **licence** | ⛔ **can we run it?** |
|---|---|---|---|---|
| **SMART** (NeurIPS '24) | Decoder-only transformer, next-token prediction over tokenised map + agent trajectories | **#1 on WOMD Sim Agents**; SMART-Planner SOTA among learning-based on nuPlan closed-loop; >1 B motion tokens collected | ✅ **Apache 2.0** (code) | ⚠️ **Checkpoints withheld** — authors will release only *"model parameters of a medium-sized model not trained on Waymo data"*; availability `UNVERIFIED`. Needs WOMD scenario-protocol data |
| **CAT-K** (CVPR '25 **Oral**, NVlabs) | Closed-loop supervised fine-tuning: at each step take top-K action tokens, choose the one landing closest to ground truth — keeps rollout states near GT, killing covariate shift **without RL or GAIL** | **7 M-param model outperforms a 102 M model of the same family**; top of WOSAC at submission | `UNVERIFIED` — LICENSE file exists, text not fetched | ⛔ *"We cannot share pre-trained models according to the terms of the Waymo Open Motion Dataset."* Obtainable by emailing authors with proof of Waymo registration. Training: **SMART-tiny-7M = 8×A100 (80 GB) for a few days**; SMART-nano-1M on a single A100 but *"significantly worse"* |
| **IDM** (in nuPlan, AlpaSim) | Rule-based car-following | trivial to run, no licence issue | permissive | ✅ runnable — **but** nuPlan-R: IDM agents *"lack behavioral diversity and fail to capture realistic human interactions"* and (per secondary sources) *"substantially overestimates planner robustness and conceals failure modes, particularly for learning-based approaches"* ⚠️ exact numbers `UNVERIFIED` |
| **nuPlan-R diffusion agents** (2025) | Noise-decoupled diffusion reactive agents + interaction-aware agent selection | realism + diversity | `UNVERIFIED` | needs nuPlan |

### 6.3 ⭐ The finding that de-prioritises this whole axis

**Reactive agents require a map / lane graph. PhysicalAI-AV has none** (five probes, `INHERITED`: no map, no
lane graph, no junction annotation, no traffic-light feature, no route/goal signal; `obstacle.offline`'s enum
over 87 481 cuboids is 10 classes, **all dynamic agents**). Every model above consumes *vectorised map +
agent* tokens. **We cannot run any of them on our corpus.**

**And — crucially — we do not need to yet.** NAVSIM v2, the design we should copy, uses *"relatively simple,
rule-based traffic models"* whose agents *"strictly follow road-centerline paths."* The state of the art in
non-photorealistic closed-loop evaluation **does not have reactive traffic either.** ⇒ **Deprioritise
reactive agents. They are not the blocker; the protocol is.**

---

## 7. Metrics — what discriminates, and what saturates

We already learned the hard way that a saturating metric is worthless (`RATIO_EXTRAPOLATION_X` clause 1,
arithmetically dead at `sup(ratio) = 1.298888 < 1.5`). The field has measured the same thing.

### 7.1 PDM-Score composition (`PUBLISHED`, NAVSIM)

```
PDMS = ( ∏  score_m ) × ( Σ w_x · score_x  /  Σ w_x )
       m ∈ {NC, DAC}     x ∈ {EP, TTC, C}
```
- **Multiplicative gates:** NC (No at-fault Collision) → 0 on hard violation, 0.5 for at-fault static
  collisions; DAC (Drivable Area Compliance) → 0 on violation.
- **Weighted average:** EP (Ego Progress) w = 5, TTC w = 5, **C (Comfort) w = 2**. All sub-scores ∈ [0, 1].
- **EPDMS (v2)** adds Traffic-Light Compliance, Driving-Direction Compliance, Lane Keeping, Extended Comfort.

### 7.2 ⭐ What actually discriminates — DEMONSTRATED, n = 8

| sub-metric | Spearman ρ vs Bench2Drive Driving Score | reading |
|---|:--:|---|
| **Ego Progress (EP)** | **0.83** | ⭐ strongest single predictor of closed-loop success |
| Drivable Area Compliance (DAC) | 0.71 | strong |
| Time-to-Collision (TTC) | 0.59 | weak; near-saturated |
| **No at-fault Collision (NC)** | **0.45** | ⛔ **lowest of the five** — the safety-critical metric is the *worst* predictor |
| **Comfort (C)** | saturated at **≥ 99.9 %** | ⛔ *"essentially zero discriminative information"* |
| aggregate **PDMS** | **0.90** (τ = 0.79) | strong but **non-monotonic**, with ranking inversions |
| classical **L2 / ADE** | **−0.36** (p = 0.43) | ⛔ **no significant correlation** |

> *"a much simpler 3-metric formula matches the predictive power of the full 5-metric PDMS at the same
> Spearman ρ = 0.90 … where TTC and Comfort approach saturation, these two sub-metrics add little marginal
> information for closed-loop ranking."*

### 7.3 ⚠️ Three consequences for TanitAD, stated plainly

1. **Our headline metric is the field's worst predictor.** ADE/L2 correlates at **ρ = −0.36, p = 0.43** with
   closed-loop driving quality. Our program's entire leaderboard is denominated in `ade_0_2s`. This is
   consistent with our own long-standing finding that open-loop ADE 0.45 m → closed-loop 1.69 m and *"open-loop
   does not predict closed-loop"* (`INHERITED`). **The literature agrees, at n = 8 with a published p-value.**
2. **Progress, not safety, is the discriminating axis** among current SOTA. A metric suite that gates hard on
   collisions and rewards conservatism will saturate and stop ranking policies. We should weight an
   **ego-progress** term heavily — and we already know our arm's failure is **longitudinal** (83 % of 2 s
   error along-track, +0.66 m/s speed over-prediction, `INHERITED`). **These line up.**
3. **Check every proposed sub-metric for dynamic range before adopting it** — the C13 gate, which we already
   apply. Comfort at ≥ 99.9 % is exactly a dead clause 1 in someone else's suite.

### 7.4 ⭐ The cheapest discriminating experiment, pre-registered here with both outcomes committed

**Question:** is our flat-road lateral warp faithful enough to carry a pseudo-simulation grid, or must the
grid be **heading-only**?

**Design:** on the same 40 val episodes and the same estimator (`episode_cluster_bootstrap`, B = 2000, paired),
compare the C-GEO/C-RT model-free fidelity of the **lateral** warp against the **yaw** warp across the grid
`|dlat| ∈ {0.5, 1.0, 1.5, 2.0} m`, with the destroyed-observation controls (`dead_black` +1.5619,
`dead_shuffle` +0.3388, `dead_noise` +0.1442) as the dynamic-range scale — **the useful headroom is only
0.1442 m** and every number must be read against it.

| outcome | what we write |
|---|---|
| **L-OK** — lateral round-trip residual stays within the yaw arm's band over 0–2.0 m | the pseudo-sim grid is **2-D** (lateral × heading), matching NAVSIM v2's axes. Full adoption of the protocol. |
| **L-BAD** — lateral degrades materially faster than yaw (as the flat-road assumption predicts on non-planar scenes) | the grid is **heading-only + longitudinal-by-resampling**. We publish a *narrower* protocol than NAVSIM v2 and **say so**, rather than shipping an unvalidated lateral axis. A publishable, decision-relevant negative. |

**No stretching a marginal L-BAD into L-OK.** This is a CPU/1-GPU experiment on an eval pod; it touches no
training pod.

---

## 8. ⭐ THE RANKED RECOMMENDATION

Ranked per the pre-registration's binding rule (R3 + R4 fired ⇒ repair outranks adopt), with cost, licence,
corpus requirement, and — the question that matters — **what it lets us claim that we currently cannot.**

### 🥇 Option 1 — REPAIR BY PROTOCOL: replace sequential rollout with a bounded pseudo-simulation grid

**What:** stop unrolling K steps. Pre-generate a fixed grid of perturbed observations at
`|dlat| ≤ 2.0 m` (pending §7.4) and `|dpsi| ≤ 12°`, score the policy **once** from each perturbed state over a
short horizon, and aggregate with a proximity weighting toward the ego's likely behaviour — the NAVSIM v2
design, on our own substrate.

| | |
|---|---|
| **cost** | **Engineering only.** No new renderer — `taniteval.clhorizon.sampling_homography` already exists and is what the closed loop calls. Runs on the eval pod. **No training pod touched.** `ESTIMATED` days, not weeks. |
| **licence** | **None.** Our code, our corpus, our warp. |
| **corpus requirement** | **None new.** ⚠️ But map-based metrics (DAC, Lane Keeping, Driving-Direction Compliance, Traffic-Light Compliance) are **impossible** on PhysicalAI-AV — we need a **map-free metric set** built on the axes we can compute: at-fault collision against `obstacle.offline` cuboids (87 481, 10 dynamic classes), ego progress, TTC, comfort. |
| ⭐ **what it lets us claim** | **"MEASUREMENT, not extrapolation."** Out-of-envelope fraction becomes **0 by construction** instead of 12.3 % at K = 20. **This is the only option that removes the permanent EXTRAPOLATION label from our closed-loop numbers.** It also gives us *error-recovery and causal-confusion* signal — the thing open-loop ADE provably does not measure (ρ = −0.36). |
| **what it does NOT give** | comparability to anyone. Still an internal number. |
| **risk** | §7.4's L-BAD outcome narrows the grid to heading-only. **Run §7.4 first.** |

### 🥈 Option 2 — ADOPT NAVSIM (navtest + navhard) on OpenScene, in parallel

**What:** run our planner through the public NAVSIM v1/v2 harness on their data, and report PDMS/EPDMS.

| | |
|---|---|
| **cost** | OpenScene download + inference GPU. ⚠️ **The real cost is an input-adapter**: NAVSIM expects **8 cameras at 1920×1080 + merged 5-LiDAR**, 1.5 s history at 2 Hz. Our arm consumes a **single 256×256 front-wide crop**. This is a genuine porting effort, not a config flag. `ESTIMATED` weeks. |
| **licence** | ✅ code **Apache 2.0**. ⚠️ data **CC BY-NC-SA 4.0 + nuPlan Dataset License Agreement for Non-Commercial Use** — non-commercial **and share-alike**. Fine for research publication; **must be flagged to Sayed if TanitAD has any commercial horizon**, and ShareAlike may attach to anything we derive from it. |
| **corpus requirement** | ✅ **runs entirely on THEIR data.** Our parity corpus (`physicalai-train-e438721ae894`, 2376 eps, skip-hash `f09e44db`) is untouched — **no parity violation.** |
| ⭐ **what it lets us claim** | **A number on a public leaderboard, comparable to 15+ published SOTA methods, validated against nuPlan closed-loop at R² = 0.80 / n = 83.** **No internal instrument can ever give us this.** It is the difference between "our model scores 0.452 m on our harness" and "our model scores X PDMS, here is the leaderboard." |
| **honest limit** | It is **still not real-driving validity** (§3.3), and PDMS is non-monotonic with true closed-loop (SafeDrive 3rd → 5th). |

### 🥉 Option 3 — Publish the envelope methodology as a contribution

**What:** write up C-GEO (fabricated-pixel fraction, FOV half-angle 25.70°, exactness of `K R K⁻¹`), the
destroyed-observation dynamic-range controls, and the out-of-envelope arithmetic, as *a method for bounding
the validity of warp-based closed-loop evaluation.*

| | |
|---|---|
| **cost** | Writing. Data already exists and is staged. |
| **licence** | none |
| **corpus** | none |
| ⭐ **what it lets us claim** | **That we measured something the field does not measure at all** (§4.2). Currently no benchmark publishes an ego-deviation validity bound. |
| ⚠️ **constraint** | Publish the **method** plus an arm-conditioned number — **not** "the envelope", since ~half of it is our v1 arm's OOD sensitivity. |

### 4️⃣ Option 4 — MetaDrive as a map-carrying tactical/strategic sandbox

Apache 2.0 (**the cleanest licence in the field**), no dataset licence needed for procedural scenarios,
300 FPS on a standard PC, and it **has the map/lane-graph/junction topology our corpus provably lacks** —
which is the exact asset the strategic brain needs. ⚠️ But it renders only lightweight 3D, so it exercises a
**state-space** policy, not our vision encoder. **Recommend only for the tactical/strategic planners**, not
as our headline closed-loop instrument.

### 5️⃣ Option 5 — GPUDrive

MIT engine (✅), 1 M FPS, but **WOMD data** (non-commercial; **trained weights are a derivative work**), no
rendering, and no published evaluation-validity evidence. Useful for reactive-agent research later.

### ⛔ Option 6 — Bench2Drive / CARLA. **NOT RECOMMENDED now.**
CC-BY-NC-ND (**No Derivatives, code and assets**), 400 GB–4 TB, **~5 DS points run-to-run variance**, and it
requires a CARLA-compatible sensor stack we do not have. Revisit only if a fully-synthetic closed-loop number
becomes a publication requirement.

### ⛔⛔ Option 7 — Waymax. **DO NOT ADOPT WITHOUT A DECISION FROM SAYED.**
The licence prohibits use *"to train or otherwise develop or improve (directly or indirectly) an artificial
intelligence foundation model."* **This is a question about what TanitAD is, and I am not the one to answer
it.** Escalated in §9.

### ⛔ Option 8 — Widen the envelope and keep rolling out. **REFUSED ON EVIDENCE.**
Already settled arithmetically by P1: at yaw = ∞ the lateral clause alone leaves 3.75 % of K = 20 windows
outside, and MEASUREMENT requires zero. **This option cannot succeed and must not be re-proposed.**

### 8.1 What I would do first, concretely

1. **§7.4 lateral-vs-yaw fidelity experiment** (CPU/1 GPU, eval pod only) — decides whether the grid is 1-D or 2-D.
2. **Build the map-free metric set**, weighting **ego progress** heavily (ρ = 0.83) and **not** relying on
   collision-rate alone (ρ = 0.45); C13-gate every clause for dynamic range before it ships.
3. **Re-label every existing closed-loop number EXTRAPOLATION** in `MODEL_REGISTRY.md` until Option 1 lands.
4. **Scope the NAVSIM input-adapter** in parallel — it is the long pole for the only comparable number we can get.

---

## 9. ⭐ ESCALATION — decisions that are not mine to make

1. ⛔ **Waymax licence vs TanitAD's identity.** Is a sub-300 M hierarchical latent world model for driving
   an *"artificial intelligence foundation model"* under the Waymax licence? If yes, Waymax and anything
   derived from it are closed to us. **Needs Sayed, not an agent.**
2. ⚠️ **OpenScene/nuPlan is CC BY-NC-SA — non-commercial AND share-alike.** If TanitAD has any commercial
   horizon, adopting NAVSIM's data has downstream consequences. **This is the third time this program has hit
   a licence that a short name concealed** (ZOD → nuScenes → now Waymax/Bench2Drive). Recommend a standing
   rule: *no external corpus or simulator is adopted until its licence document is fetched and its code/data
   terms recorded separately.*
3. ⚠️ **`MODEL_REGISTRY.md` must be updated** to label every closed-loop number EXTRAPOLATION. This crosses
   streams and will not happen by itself.
4. ⚠️ **The §7.4 experiment needs an eval-pod slot.** It is CPU/1-GPU and touches no training pod.

---

## 10. Self-refutations and open gaps, recorded not hidden

| # | what | status |
|:--:|---|---|
| 1 | I initially treated the cross-benchmark **ρ = 0.90** as the headline correlation evidence. **Corrected**: it is n = 8, pooled from third-party leaderboards with acknowledged configuration drift. **Pseudo-Sim's R² = 0.80 at n = 83, single-pipeline, is the stronger result despite the smaller coefficient.** | corrected in §3.1 |
| 2 | I nearly wrote that our envelope "matches" NAVSIM v2's grid. **It does not**: we are wider laterally (3.0 vs 2.0 m) but **narrower in heading** (15.47° usable vs their 20° filter), and the two heading numbers are not strictly commensurable. | corrected in §2.3 |
| 3 | NAVSIM v1's correlation coefficients are **shown graphically in Fig. 3(a)**, not stated numerically in the text I could read. Marked `UNVERIFIED` numerically rather than quoted. | open |
| 4 | nuPlan-R's specific IDM-overestimation numbers: **abstract only, results section not fetched.** Claim marked `UNVERIFIED`. | open |
| 5 | CARLA **asset** licence not fetched (repo LICENSE is code MIT). CAT-K repo licence text not fetched. | open |
| 6 | SimNet and Sim2Val assessed from abstracts only — I state they present **no** sim-vs-real correlation statistic *in the material I could read*, which is weaker than stating none exists. | open, flagged |
| 7 | The AlpaSim `trafficsim` result is `INHERITED` — I read the raw JSON directly but did not re-run it. | disclosed |

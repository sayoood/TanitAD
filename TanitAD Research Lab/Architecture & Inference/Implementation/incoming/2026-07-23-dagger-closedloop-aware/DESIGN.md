# `dagger_planner_ft` — closed-loop-AWARE planner fine-tune: design + cheap no-renderer proof

**Date:** 2026-07-23 (Europe/Berlin, UTC+2) · **Author:** architecture subagent · **Compute:** local
RTX 4060 (no pod touched, pod-free per brief) · **Type:** design doc + MEASURED mechanism proof.

**Number discipline (CLAUDE.md).** Evidence class on every claim: **MEASURED** (ours + artifact path) ·
**PUBLISHED** · **INHERITED** · **ESTIMATED** · **HYPOTHESIS**. Intervals carry their estimator
(episode-cluster bootstrap / paired, `taniteval/ci.py`). RETRACTION_LOG C1–C6 read first; the C5/C6
closed-loop entries of 07-22/07-23 are honoured (every AlpaSim number below is flagged WITHIN-SIM
RELATIVE / ~3.2× reconstruction-OOD; the harness here is flagged SELF-REFERENTIAL throughout).

---

## 0. TL;DR — verdict: **DAGGER_HURTS on the cheap harness → do NOT promote it on this evidence**

I designed `dagger_planner_ft` (a DAgger / rollout-in-the-loop fine-tune that exposes the planner to its
OWN compounding-error states with recovery labels) and ran the pre-registered cheap proof on the
no-renderer kinematic harness (task #21). **MEASURED, at matched open-loop ADE, the closed-loop-aware
fine-tune makes closed-loop drift WORSE, not better** — robustly, across two fine-tune scopes and two
DAgger rounds, and CI-worse than a matched-budget behaviour-cloning control (so the *on-policy data
itself* is the culprit, not training effort).

This maps onto the pre-registered **NEUTRAL/HURTS** branch (§5.1 of `2026-07-23-planner-is-the-bottleneck.md`):
**on-policy state coverage is not the cheaply-demonstrable lever; the lever stays v4.2's schedule fix.**
The most likely mechanism (HYPOTHESIS) is that the harness is **self-referential** — the on-policy states
are the world model's own *imagined, off-manifold* latents, and training the head to emit aggressive
logged-GT recovery corrections in response to them makes the closed loop **overcorrect** (divergence rises
0.22 → 0.39). That **refutes the cheap harness as a DAgger proving ground, not DAgger in a faithful sim.**
A decision-grade DAgger test needs AlpaSim (faithful, non-self-referential on-policy states) + a gentler /
consequence-aware recovery objective.

**Headline numbers (MEASURED, `dagger_result.json`, head-only regularised, 12 held-out eps, cross-fit):**

| paired, DAgger − baseline | delta | 95% CI (paired episode-cluster boot) | separated |
|---|---|---|---|
| closed-loop ADE@2s | **+0.266 m** | [0.008, 0.550] | ✓ (worse) |
| divergence>5 m @2s | **+0.166** | [0.030, 0.313] | ✓ (worse) |
| lateral-dev@2s (off-road proxy) | **+0.548 m** | [0.155, 0.994] | ✓ (worse) |
| open-loop head ADE@2s | +0.107 m | [−0.120, 0.320] | ✗ → **MATCHED** |

---

## 1. Motivation — the measured failure this lever targets

Three MEASURED anchors (verified by reading the artifacts):

- **flagship-v1's tactical head and REF-C are BOTH open-loop-trained, and BOTH fail closed-loop by off-road
  DEPARTURE, not collision.** AlpaSim n=12 paired: flagship offroad **8/12**, plan_dev **1.12 vs REF-C 0.34**,
  collisions **tied** (`…/incoming/2026-07-22-alpasim-closedloop-evalpod/flagship_vs_refc_suite_NOTE.md`).
- **The deficit is concentrated OFF-highway** — flagship ties REF-C on highway (identical 1/3) but goes
  off-road 6/8 on straight/urban/rural geometry (`scenario_stratified_results.json`). Off-road departure
  from compounding heading/position error is the exact distribution shift **open-loop training omits**.
- **Imagination-in-the-loop re-plan** already helps modestly but is NOT closed-loop-*trained*
  (paired Δ ade@2s −0.213, `…/incoming/2026-07-22-imagination-closedloop-proof/`).

The hypothesis under test (well-motivated, live): a fine-tune that trains the planner on its **own
on-policy compounding-error states** (DAgger) closes that distribution-shift gap → better road-keeping.

---

## 2. The design — `dagger_planner_ft`

### 2.1 Algorithm

Take a healthy-WM checkpoint (flagship-v1, `flagship4b-speedjerk-30k`, step 29999). **Freeze the world
model + strategic policy** (isolate the planner). Three arms that differ ONLY in the tactical-policy weights:

| arm | what it is |
|---|---|
| **A0 baseline** | v1's tactical policy, unmodified (open-loop-trained). |
| **A1 BC-FT** *(control)* | v1 tactical fine-tuned `total_steps` on the **open-loop demonstrations only** (real encoded states → GT waypoints). Controls for "just more training at this LR". |
| **A2 DAgger-FT** | v1 tactical fine-tuned `total_steps` on the **union** of the demonstrations AND **on-policy** imagined states (the compounding-error distribution) labelled with the **recovery expert** (§2.2), aggregated over R=2 DAgger rounds. Same lr/batch/total_steps as A1. |

**DAgger rounds (aggregation).** Round r rolls the round-(r−1) policy closed-loop in the harness, collects
the imagined latent windows it actually visits + their recovery labels, adds them to the aggregate, and
retrains the tactical head from the v1 init on `demos ∪ Σ on-policy` (textbook DAgger: retrain on the
aggregate each round). The A2−A1 paired contrast (identical params/budget; A2 adds on-policy data) isolates
the **on-policy contribution** from raw training effort (RETRACTION_LOG C6: name every difference).

**Loss.** Mean Euclidean point error over horizons {0.5, 1, 1.5, 2} s — directly ADE. **Pure DAgger:**
expert = open-loop GT recovery; states = on-policy closed-loop visited states.

### 2.2 The recovery expert (validated)

At closed-loop tick *i* the ego has drifted to bicycle pose **Q_i** (in the window's frame-0 ego coords).
The expert target at relative horizon *h* is **the logged GT ego pose at ABSOLUTE future tick i+h,
re-expressed in the CURRENT drifted ego frame at Q_i** — "from where you actually are, here is the trajectory
back onto the demonstrated path." Uses the harness's own `_ego` rotation (`driving_diagnostic.py`); the
frame-0 GT path is collected to 2×K ticks so the target is defined for all i∈[0,K).

**Geometry validated (MEASURED, `dagger_probe.py`):** at Q=origin the recovery expert reproduces
`gt_ego_waypoints` exactly (max|err| = 0.000000).

### 2.3 Harness integration + protocol

- **Harness:** `taniteval/taniteval/closedloop.py` (task #21) — the flagship WM is its own neural simulator;
  plan on the imagined latent → pure-pursuit + kinematic-bicycle control → operative predictor imagines the
  next latent → 20 ticks (2 s @ 10 Hz). Reused verbatim; the imagination proof ran it in 32.4 s on this same
  4060/ckpt/valsub. Eval reuses the stock `collect`; the on-policy collector is a recording variant that
  matches `closed_loop_rollout` tick-for-tick.
- **Trained params:** the tactical policy only (WM + strategic frozen). Two scopes reported (§3):
  **head-only** = the 4 waypoint-output heads (~2 K params) over the frozen trunk — the honest, low-overfit
  "planner head" test (**primary**); **full-head** = the whole 22.7 M-param tactical policy (**ablation**).
- **Cross-fit (uses all 12 local val eps; every eval window OUT-OF-SAMPLE):** 2-fold by episode parity —
  train on ODD eps / eval on EVEN, and vice-versa; pool both held-out halves → 12 episodes, each scored by a
  planner that never trained on it. A0 is weight-identical across folds. Per-window arrays align across arms
  (window enumeration is weight-independent) → valid pairing.
- **Estimator:** episode-cluster bootstrap / paired (`taniteval/ci.py`, 2000 boot) — the program's
  decision-grade interval, NOT the deprecated overlapping-holdout.
- **Hyperparameters:** head-only lr 3e-5 / 150 steps; full-head lr 1e-4 / 400 steps; batch 64 (A2 = 32 demo
  + 32 on-policy / step); AdamW wd 1e-4; R=2 rounds; seed 0.

### 2.4 Pre-registered decision predicate (both outcomes committed IN ADVANCE)

- **HELPS** ⇒ DAgger-FT has a materially lower divergence / off-road rate (or lower closed-loop ADE)
  **at matched open-loop ADE** (paired CI-separated below 0; open-loop paired CI includes 0) ⇒ closed-loop-
  aware training is a real lever → promote to an AlpaSim confirmation; fallback if v4.2's schedule fix alone
  underperforms.
- **NEUTRAL/HURTS** ⇒ paired Δ includes 0 (ties) or is CI-separated the WRONG way ⇒ on-policy coverage is
  not the bottleneck at this fidelity; the lever is the schedule (v4.2) / a consequence-aware objective; the
  no-renderer harness has hit its ceiling.
- **Off-road proxy** (kinematic harness has no real road boundary): large lateral deviation@2s; thresholds
  {1.5, 2.0, 3.0} m pre-registered, **2.0 m primary**.

---

## 3. Measured result

### 3.1 PRIMARY — head-only (regularised, open-loop MATCHED) — `dagger_result.json`

n = 265 windows / 12 held-out eps (cross-fit). Sanity: **A0 reproduces the imagination proof exactly**
(closed_bike ADE@2s 1.720 vs 1.7196; open-loop head ADE 3.147 vs plan_direct 3.1469) → harness wired
correctly.

| arm | closed ADE@2s [CI] | divergence>5m | open-loop ADE@2s | lat-dev@2s | off-road@2.0 m |
|---|---|---|---|---|---|
| **A0 baseline** | **1.720** [1.444, 2.040] | **0.223** | 3.147 | 1.302 | **0.200** |
| A1 BC-FT | 1.893 [1.546, 2.300] | 0.343 | 3.229 | 1.629 | 0.272 |
| **A2 DAgger-FT** | **1.985** [1.607, 2.412] | **0.389** | 3.254 | 1.850 | 0.359 |

**Paired (episode-cluster bootstrap; "+"=worse):**

| contrast | closed ADE@2s | divergence | open-loop ADE@2s | lat-dev@2s |
|---|---|---|---|---|
| **A2 DAgger − A0 baseline** | **+0.266** [0.008, 0.550] ✓ | **+0.166** [0.030, 0.313] ✓ | +0.107 [−0.120, 0.320] ✗ **(matched)** | **+0.548** [0.155, 0.994] ✓ |
| A1 BC − A0 baseline | +0.174 [−0.079, 0.476] ✗ | +0.121 [0.004, 0.259] ✓ | +0.082 [−0.155, 0.317] ✗ | +0.326 [0.042, 0.793] ✓ |
| **A2 DAgger − A1 BC** *(isolates on-policy)* | **+0.092** [0.015, 0.187] ✓ | **+0.045** [0.004, 0.087] ✓ | +0.025 [−0.025, 0.086] ✗ | +0.222 [−0.039, 0.524] ✗ |

Reading: **open-loop ADE is statistically MATCHED across all three arms** (every open-loop paired CI includes
0). At that matched open-loop accuracy, **DAgger significantly WORSENS every closed-loop metric** — ADE,
divergence, and the lateral off-road proxy — and is **CI-worse than the BC-FT control** on closed ADE and
divergence, so the harm is attributable to the **on-policy recovery data specifically**, not to fine-tuning
effort. This is the clean opposite of the pre-registered HELPS condition.

### 3.2 ABLATION — full-head (22.7 M params) — `dagger_result_fullhead.json`

Fine-tuning the whole tactical policy on the 12-ep budget **overfits**: even the BC-only control degrades
open-loop ADE (3.147 → 4.587) and closed-loop (1.720 → 2.266); DAgger is worst (open 6.036, closed 2.602;
A2−A0 closed **+0.882** separated; A2−A1 **+0.335** separated). This documents *why the head-only scope is the
honest test* — and corroborates the direction: **more capacity fine-tuned on-policy → more closed-loop harm.**

### 3.3 DAgger-round progression (closed ADE@2s, head-only)

baseline **1.720** → 1 round **1.925** → 2 rounds **1.985**. **Monotonically worse** with more on-policy
data — the opposite of the "compounding improvement" a working DAgger lever would show.

---

## 4. Interpretation — why it hurts (MEASURED fact vs HYPOTHESIS mechanism)

**MEASURED fact:** on the no-renderer harness, at matched open-loop ADE, closed-loop-aware DAgger fine-tuning
increases closed-loop drift / off-road proxy, robustly (2 scopes, 2 rounds, CI-worse than BC).

**HYPOTHESIS (mechanism, well-grounded but not isolated):** the harness is **self-referential** — the
on-policy states are the WM's OWN imagined latents, which drift **off the WM's training manifold** under
self-rollout (the same self-rollout degradation the imagination proof's `closed_grnd` block shows:
closed−open grounded Δ@2s +4.44 m, MEASURED). The recovery targets, meanwhile, demand **aggressive lateral
cut-backs** toward the logged path (e.g. re-close a 2 m lateral drift within 0.5 s). Training the head to
emit those corrections in response to off-manifold imagined features makes it **over-react**, and through the
harness's simple pure-pursuit → bicycle controller the closed loop **overcorrects and destabilises**
(divergence 0.223 → 0.389; lat-dev +0.548 m). This is a property of **the cheap harness (self-referential +
kinematic + a fixed harness controller)**, not necessarily of DAgger in a faithful sim where on-policy states
are real renders.

**Therefore the honest boundary:** this result **refutes the cheap no-renderer harness as a DAgger proving
ground** (its on-policy signal is corrupted by the WM's own imagination), and — because the harness cannot
separate "self-referential artefact" from "real DAgger failure" — it **cannot license investing GPU-days in a
DAgger curriculum**. It does **not** refute DAgger in a faithful photoreal sim.

---

## 5. Decision — what this means for the program

- **Do NOT promote DAgger / rollout-in-the-loop as the next planner lever on this evidence.** The cheapest
  discriminating experiment argues AGAINST it (operating standard #5: pre-registered, both outcomes committed;
  the committed NEUTRAL/HURTS branch fired).
- **The next planner lever stays v4.2's schedule fix** (the λ_plan cap-and-hold controller that keeps the
  planner gradient alive on a healthy WM — `2026-07-23-planner-is-the-bottleneck.md` §4, exp #2/#5). DAgger is
  **deferred, not refuted**: it re-enters only as an AlpaSim-validated curriculum (exp #4), never on the
  self-referential proxy.
- **This is a data/harness-ceiling finding, consistent with the program's split:** the cheap harness proved
  the **inference** mechanism (imagination-in-the-loop, needs no training) but **cannot** prove a **training**
  mechanism (DAgger, needs faithful on-policy states + more data). Same ceiling the pre-registration flagged.

---

## 6. Honest caveats (stated, not hidden)

1. **SELF-REFERENTIAL** — the WM is both simulator and state estimator; this proves/refutes the DAgger
   MECHANISM in-harness, NOT the real off-road/safety rate. AlpaSim (photoreal, external) is required for the
   rate (exp #4).
2. **KINEMATIC** — no real road boundary; off-road is APPROXIMATED as large lateral deviation@2s (proxy
   thresholds pre-registered; 2.0 m primary).
3. **n = 12 local val eps** (not the full 40). The cross-fit paired design (every eval window out-of-sample,
   shared difficulty cancelled) is the robust part; absolute rates are wide.
4. **The bicycle + pure-pursuit are a fixed HARNESS controller** shared by all arms; part of the overcorrection
   is the controller's response to aggressive recovery waypoints. A learned/consequence-aware controller or a
   gentler rejoin horizon is the indicated next variant — but that is beyond "pure DAgger" and only meaningful
   on a faithful sim (would be p-hacking to tune here for a HELP).
5. **Pure DAgger, one recovery-label family** — logged-GT re-expression. A consequence-aware objective
   (imagined-offroad penalty in *selection*, not imitation) is the pre-registered redirect and is untested.

---

## 7. Escalations (need a decision/owner, not a paragraph)

1. **Prioritisation:** the "closed-loop-aware training is the next planner lever" thesis is **not supported by
   the cheap proof** — it argues against it. Recommend the planner-lever budget stay on **v4.2's schedule
   fix**; DAgger re-enters only post-AlpaSim. (Owner: PI / planner-track lead.)
2. **Harness scope note for the eval owner:** task-#21 (no-renderer) is validated for **inference-mechanism**
   proofs (imagination) but is **refuted as a DAgger/training proving ground** (self-referential on-policy
   states corrupt the recovery signal). Future closed-loop-*training* experiments should route to AlpaSim.

---

## 8. Deliverable manifest

| artifact | where | status |
|---|---|---|
| this design doc | `…/incoming/2026-07-23-dagger-closedloop-aware/DESIGN.md` | **STAGED** |
| verdict (one-page) | `…/incoming/2026-07-23-dagger-closedloop-aware/VERDICT.md` | **STAGED** |
| MEASURED result — primary (head-only) | `…/incoming/2026-07-23-dagger-closedloop-aware/dagger_result.json` | **STAGED** |
| MEASURED result — ablation (full-head) | `…/incoming/2026-07-23-dagger-closedloop-aware/dagger_result_fullhead.json` | **STAGED** |
| driver (design + proof, reusable) | `…/incoming/2026-07-23-dagger-closedloop-aware/dagger_planner_ft.py` | **STAGED** |
| geometry probe (recovery-expert validation) | `…/incoming/2026-07-23-dagger-closedloop-aware/dagger_probe.py` | **STAGED** |

**Inputs read (in-repo, not modified):** `Project Steering/RETRACTION_LOG.md`;
`Architecture & Inference/Research/2026-07-23-planner-is-the-bottleneck.md` §4/§5;
`…/incoming/2026-07-22-alpasim-closedloop-evalpod/{flagship_vs_refc_suite_NOTE.md,scenario_stratified_results.json}`;
`…/incoming/2026-07-22-imagination-closedloop-proof/{README.md,closedloop_flagship-30k_imagination-proof.json,run_proof_local.py}`;
`taniteval/taniteval/{closedloop.py,ci.py,loaders.py,data.py,hierarchy.py}`;
`stack/tanitad/models/fourbrain.py`; `stack/scripts/driving_diagnostic.py`.

**Compute:** local RTX 4060, venv `C:\Users\Admin\venvs\tanitad` (torch 2.11+cu128). Same ckpt (HF
`Sayood/tanitad-flagship-4b-speedjerk` step 29999) + 12-ep held-out val subset the imagination proof used.
**No GPU pod touched. No commit, no push (Agent Operating Standard: stage, never push).**

# Is our anti-echo control suite stricter than the field's? — precedent audit for a methodological claim

**Package** `Benchmarks & Evals/Research/2026-08-30-anti-echo-control-precedent` · **Author**
Research Lab agent (daily run 004, 2026-08-30) · literature only, 0 GPU.
Seed: the current best published closed-loop / T1-equivalent numbers for camera-only driving
world models, and **what control THEY use to exclude action echo**. Thesis under test: *our
anti-echo suite (hold-action · hold-v0 · copy detector) is stricter than standard practice, and
if so that is a publishable methodological point.*

---

## VERDICT — the thesis holds, but it is NARROWER and SHARPER than "nobody runs controls"

**Position it precisely or a reviewer will find the precedents.** Two halves of our doctrine
have very different novelty, and conflating them would be the claim's weakest point:

| our claim | status | why |
|---|---|---|
| *"T0 / open-loop must never be called driving performance"* | ⛔ **NOT NOVEL — and repeatedly re-established** | `1809.04843` (2018) · `2306.07962` (2023) · `2505.05638` (2025) · `2605.00066` (2026). Claiming this as our finding would be an easy and correct reviewer objection. |
| *"there is no standing ACTION-SIDE control suite in the field"* | ⭐ **SUPPORTED at five probes** | see F1–F3 |

**The defensible claim, stated for the paper:** *the field has established the open-loop /
closed-loop dissociation and has published one-off ego-status shortcut ablations, but there is
**no standing action-side control suite** — no hold-action floor a model is required to beat,
and no published echo metric with a calibrated known value. Both purpose-built action-fidelity
benchmarks (ACT-Bench, WorldLens) were verified to run no control arm at all.*

---

## F1 — Per-control precedent audit

[PUBLISHED, banked] Each of our three controls, against the strongest thing the field has:

| our control | strongest published precedent | how far it actually goes |
|---|---|---|
| **(a) HOLD-ACTION** (freeze the action, roll out; the model must beat it) | ⛔ **NONE FOUND in driving world models.** Nearest: `2506.09981` (ReSim) masks trajectories at p = 0.5 — but that is a **training** device, not an evaluation control | The field measures how *well* an action is followed; it never establishes what a model scores for **not using the action at all**. **This is the open lane.** |
| **(b) HOLD-V0** (constant initial velocity) | ⭐ **WELL PRECEDENTED** — `2406.15349` NAVSIM ships a **`ConstantVelocityAgent`** as a standing blind baseline; `1903.07933` is the general lesson (constant velocity **beats** SOTA neural pedestrian predictors); `2305.12032` runs it as a leaderboard entry | ⛔ **Claim no novelty here.** The claimable part is that we apply it to *every* closed-loop number rather than once. |
| **(c) COPY DETECTOR** (`echo_index`, GT-calibrated at 0.2113) | `2312.03031` (BEV-Planner) for the *shortcut-ablation ladder*; `2010.14876` / `2207.09705` for the *copycat concept*; `2406.03877` for the empirical demolition | ⭐ **No published echo METRIC found.** BEV-Planner **ablates the input away**; we **quantify how much of the output *is* the input**. Different instruments — and ours is the finer one. The **GT-calibrated known value** is the part with no precedent at all. |

## F2 — ⛔ The two purpose-built action-fidelity benchmarks were verified to carry NO control arm

[PUBLISHED, banked, verified at source by the search pass] `2412.05337` (**ACT-Bench**, ICLR
2025) reports IEC (instruction–execution consistency) and TA (ADE/FDE between intended and
estimated trajectory) — **no frozen-action rollout, no shuffled/mismatched action, no trivial
floor, no oracle ceiling**; its tables give no reference for what constitutes adequate
alignment. `2512.10958` (**WorldLens**, CVPR 2026 Oral) scores 24 dimensions including an
Action-Following family — **no null/adversarial baseline**; it compares generated video against
real video, not model against trivial control.

⚠️ `2601.01528` (DrivingGen, ICLR 2026) adds controllability w.r.t. ego conditioning; whether
it carries a null control is **UNCONFIRMED** — not verified from the primary. Do not cite it
either way without opening it.

⭐ The dual of our finding exists and is worth citing: `2511.20325` (**AD-R1**) identifies an
*"optimistic bias"* — conditioned on an **unsafe** trajectory, standard driving world models
**hallucinate a safe future** (obstacles vanish). That is a world model **ignoring the action**,
diagnosed narratively and fixed with counterfactual synthesis — **no formal echo/no-echo
metric**. It is the closest the field comes to treating action-insensitivity as a first-class
defect, and it corroborates this run's Architecture package (`H-ARCH-ACTINS`).

## F3 — The strongest empirical warrant for our doctrine, and it is someone else's number

[PUBLISHED, banked, verified from the primary] `2406.03877` (**Bench2Drive**, Table 3) runs
**AD-MLP** — the ego-state-only, no-camera, no-LiDAR model that matches perception-based
planners *open-loop* — as a **closed-loop** entry:

> **AD-MLP: Driving Score 18.05, Success Rate 0.00 %.**

⭐ **The near-SOTA open-loop blind model scores EXACTLY ZERO closed-loop.** That single row is
the field's own demonstration of why T0 cannot be quoted as capability, and it is a stronger
citation for our doctrine than anything we have measured ourselves. Companion numbers from the
same table: TCP 40.70 / 15.00 · VAD 42.35 / 15.00 · UniAD-Base 45.81 / 16.36 · ThinkTwice
62.44 / 31.23 · DriveAdapter 64.22 / 33.08.

The shortcut line behind it: `2305.10430` (AD-MLP, ego state only, avg L2 **0.23 m**, corrected
to **0.29** after a training-data bug) and `2312.03031` (BEV-Planner) — the latter carrying the
**full ego-status ablation ladder** (Table 1, VERIFIED): UniAD **without ego status anywhere
1.03 m**, ego status in BEV only 0.66, both 0.46. **That ladder is the closest published
analogue to our echo finding**, and it is the template for how to present ours.

## F4 — Published closed-loop numbers: what is quotable and what is not

[PUBLISHED — **VERIFIED from primary**] Camera-only closed-loop SOTA on Bench2Drive:
**SimLingo 85.07 ± 0.95 DS / 67.27 ± 2.11 % SR** (`2503.09594`, Table 2). CARLA Leaderboard
2.0 official: **SimLingo-BASE DS 6.87 / RC 18.08 / IS 0.42** (same paper, Table 1) — note the
leaderboard closed June 2024, so the official figure is frozen. NAVSIM v1 navtest:
**DiffusionDrive 88.1 PDMS** with ResNet-34, +1.6 over Hydra-MDP (`2411.15139`).

⛔ **UNCONFIRMED — must not be quoted without opening the primary:**
- **NAVSIM v2 EPDMS SOTA.** Sources give **DriveSuprim 87.1**, **HAD 88.6** and **RAP-DINO
  36.9** all as "SOTA on NAVSIM v2". ⚠️ **These cannot be the same quantity** — the ~37 figure
  is almost certainly **navhard** and the ~87–89 figures a navtest-style split. **This is a
  split/scope error waiting to happen** (the `df` / `step_s` family in benchmark costume);
  establish the split before any EPDMS enters a document.
- nuPlan Val14 (PDM-Closed CLS-NR 92.84 / CLS-R 92.12), Bench2Drive TF++ w/ VLAAD-MIL, and the
  2025 WOSAC leaderboard values — all repeated widely, none verified here.

⛔ **AND THE DISCIPLINE POINT THAT BINDS US: none of these is comparable to our T1 number.**
Our `D-T1-V7-READ` figures are **ADE/FDE in metres on our own 40-episode / 6,924-window
PhysicalAI split** — a different corpus, a different metric family, and a different simulator
regime from PDMS / DS / EPDMS. ⛔ **Placing 14.069 m beside "85.07 DS" in any table would be a
category error**, and a reviewer would be right to reject it. Cross-programme comparability
requires running a community benchmark, which is the standing `TanitEval` work item — not a
prose bridge.

## F5 — The open-loop↔closed-loop correlation evidence, with its estimator

[PUBLISHED, banked] `2605.00066` measures it directly: **ADE/FDE vs Bench2Drive Driving Score
Spearman ρ = −0.36, p = 0.43**; NAVSIM **PDMS ρ = 0.90, p = 0.002** but with ranking
inversions. ⚠️ **BINDING ON HOW WE QUOTE IT: n = 8 paired methods.** p = 0.43 establishes
**ABSENCE of correlation**, not negative correlation — quoting "ρ = −0.36" bare, as though
open-loop *anti-predicts* closed-loop, is exactly the estimator slip our own operating standard
forbids. Always carry **n = 8 and p = 0.43**. Older members of the same line: `1809.04843`
(*"two models with identical prediction error can differ dramatically in driving performance"*),
`2306.07962` (open-loop and closed-loop sub-tasks *"fundamentally misaligned"*; the best
open-loop result used **only the centerline**), `2505.05638` (models with 86 % fewer parameters
matched or beat larger ones closed-loop).

Also useful and independent of us: `2605.10858` finds planners reading *isolated* generated
frames succeed ~**79 %**, but under **closed-loop** feedback route completion collapses to
~**13.51 %** — the same open→closed collapse shape, measured on the simulator side.

---

## What this changes for TanitAD (≤3)

1. ⭐ **Write the methodological claim as an ACTION-SIDE claim, and drop the doctrine half.**
   The paper should assert the absence of a standing hold-action floor and of a GT-calibrated
   echo metric (F1, F2) — and should **cite `1809.04843`/`2306.07962`/`2605.00066` as prior art
   for the open-loop/closed-loop dissociation rather than claiming it** (VERDICT table).
2. **Adopt `2312.03031`'s ablation-ladder presentation for our echo result.** It is the field's
   accepted format for "the model was using a shortcut", it is peer-reviewed (CVPR 2024), and
   presenting `echo_index` beside a ladder makes our finer instrument legible to reviewers who
   already know that paper.
3. ⛔ **Add a comparability guard to the benchmark portfolio**: no table may place a TanitAD
   ADE/FDE beside a PDMS/DS/EPDMS score (F4), and no EPDMS number enters any document without
   its split named. Both are cheap checks that prevent a category error and a scope error whose
   costs we have already paid in other costumes.

## Named empty searches

- **A hold-action (frozen-action) control used as a required-to-beat baseline in a driving
  world model**: NOT FOUND at five probes (keyword sweeps on action-ablation phrasings; both
  dedicated action-fidelity benchmarks read at source; the counterfactual/intervention-fidelity
  literature; the copycat literature; the 2026 position papers).
- **A published echo/copy metric on predicted trajectories with a calibrated known value**:
  NOT FOUND. The copycat concept is named (`2010.14876`) and fixed architecturally
  (`2207.09705`) — verified that the latter publishes **no diagnostic metric**.
- ⚠️ **INFERRED, unverified:** `2603.12864` reportedly removes the action branch at sampling
  time to check ego motion is not leaked from structure cues — read from a search summary, not
  the primary. Even if exact, it is a **generator ablation**, not a baseline the model must
  beat. Do not cite until opened.
- The 2026 position papers `2606.15032` (and `2604.22748`, unbanked) **argue for** interventional
  action fidelity and closed-loop rollout validity as the decisive evidence class, with no
  experiments. ⭐ **They are calling for the instrument we already built** — which is the
  citation that frames our contribution as timely rather than idiosyncratic.

## Banked primaries

NEW this package: `2312.03031` · `2305.10430` · `2406.03877` · `2406.15349` · `1903.07933` ·
`2306.07962` · `1809.04843` · `2605.00066` · `2412.05337` · `2512.10958` · `2511.20325` ·
`2505.05638` · `2207.09705` · `2605.10858` · `2503.09594` · `2606.15032`.
⚠️ **Read-depth:** `2412.05337`, `2512.10958`, `2406.03877` (Table 3), `2312.03031` (Table 1),
`2503.09594` (Tables 1–2) and `2605.00066` were verified at source; the rest were read from
search summaries and are **PUBLISHED-SECONDARY until their banked PDF is opened**.
`2601.01528`, `2603.12864` and `2604.22748` are **NOT banked and NOT quotable**.
**PARA-Drive has no arXiv ID** (CVPR 2024 open-access, Weng et al.) — bank the CVF PDF via
`--local` if it is ever cited.

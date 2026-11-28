# The 4-brain dominance program — an executable proof, not an essay

**Date:** 2026-07-26 (Europe/Berlin) · **Author:** chief architect, 4-brain dominance stream
**Commissioned by:** Sayed (PI) · **Extends:** `Project Steering/Reviews/2026-07-25-independent-chief-scientist-review/01_EXECUTION_PLAN.md` **Part A** (PC1–PC4, HP-1…HP-6)
**Parts:** §1 → `STRATEGIC_TACTICAL_PROBLEM_SPEC.md` · §2 → `DATA_STRATEGY.md` · §3–§4 → this file
**Compute used:** dev box only. **Zero GPU. No pod touched. Nothing staged, committed or pushed.**

**Evidence classes** (CLAUDE.md operating standard 1): `MEASURED` · `PUBLISHED` · `INHERITED` ·
`ESTIMATED` · `HYPOTHESIS` · `PLANNED`. **This document will drive GPU-months; every number below
names its class, and no `INHERITED` claim decides a GPU-day.**

---

## 0. Executive summary

**The commission.** *"Develop a plan how to test and prove the dominance of the 4-brain architecture…
including the formulation of strategic and tactical problems… and the necessary data strategy and data
building for it. We can try to apply it if flagship is ready."*

**The position.** The hierarchy thesis is not on trial — the measurement apparatus is. That is a
binding PI ruling and a logged retraction (`RETRACTION_LOG` 07-25, class **C6**), not deference: six
independently `MEASURED` confounds each make a hierarchy effect **undetectable rather than absent**.
The correct response to "untested" is to build the test.

**Where the test stands today.** All four pre-conditions **FAIL**, each for a fixable instrument
reason, and the program is closer than it looks:

| PC | State | What closes it | Cost |
|---|---|---|---|
| **PC1** route input works | 🟥 `route_skill = 0.0000` by construction (target is a lookup of the input); `L 0 / S 240 / R 0` on 3 artifacts | **two CONFIG FLAGS that already exist** — `--labels-v2` + `--v2-route-from-vision --route-vis-weight 0.3` — carried to a scored checkpoint | **a launch decision, not engineering** |
| **PC2** hierarchy in the scored loop | 🟥 `rollout_decode` takes no `intent`/`ctx`/`nav` **and is fed the expert's true future actions** | forward-hook assertion in `rollout.collect`/`runner`; score only surfaces where actions are chosen | S |
| **PC3** instrument can see it | 🟡 **substantially closed since 07-25** — `corridor.py` lands `corridor_departure_rate` at arbitrary K with the cluster bootstrap; `GATE_PROTOCOL` §0 now **refuses** K≤20 | wire `corridor`/`lateral` into `driving.tier0` (2 lines); one `rollout.collect` re-run per arm for `pred_dense` | S |
| **PC4** corpus contains decisions | 🟥 decision stratum **n ≈ 13** and **no option set exists on PhysicalAI at any n** | ⭐ **AlpaSim's `trajdata.VectorMap` (130–472 lane polygons/scene) + its unused in-tree reactive-agent model** | data build, §2 |

**⭐ The fact that changes this plan.** The program has repeatedly (and correctly) said *"we have no
map, no lane graph, no junction annotation"* — true of PhysicalAI-AV, whose card says so outright. But
**AlpaSim's scenes ship a full `trajdata.VectorMap` per scene USDZ**, and the same bundle contains a
**reactive-agent model (`trafficsim/catk/smart/`, SMART + CAT-K, Apache-2.0) that we have never once
enabled** (both `MEASURED`). A strategic problem needs a topology to choose *over*; a tactical problem
needs an agent to choose *against*. **We have owned both for weeks and switched on neither.** That is
the difference between a proof program and a wish.

**What this plan delivers.** Nine measurable decision problems (§1) whose targets survive an absolute
circularity bar; a three-corpus composition — **train** on PhysicalAI, **prove** on AlpaSim, **publish**
on Cosmos-Drive-Dreams (§2); an eight-prediction pre-registered battery extending HP-1…HP-6 with two
new structural predictions (§3); an ablation ladder priced at **~34 pod-days for the minimum
decision-grade contrast**, staged so a killed program still yields value (§3.4); and a critical path
whose **first six items need zero GPU and can start today** (§4).

**The single result that would falsify dominance is stated in §5 and is committed in advance.**

---

## 1. What is already true — the honest baseline

Everything in this section is `MEASURED` with an artifact path, and it is the ground the ladder is
built on.

### 1.1 Evidence FOR the hierarchy that survives the estimator correction

| # | Result | Value | Status |
|---|---|---|---|
| **E1 / H18** | **Grounding dominance** — grounded operative rollout vs the ungrounded supervised tactical head | paired Δ **+2.9568 m** (corrected UP from +2.6979); un-separating needs an **8.65×** interval widening against a worst-ever-measured **2.06×** | ✅ **admissible and decisive**; the hierarchy's strongest single positive |
| **E4 / IMP-2** | Planning over the WM beats supervised heads | G1 open-loop **0.893 vs 3.150**; closed-loop divergence >5 m **8.7 % vs 22.2 %** | ✅ direction safe. ⚠️ the **+2.257 magnitude is NOT a deployable margin** — the P2 cost uses a **future-derived** `v_target` (input asymmetry, same class as the 07-24 C6) |
| **E5** | The levels cohere | maneuver↔trajectory agreement 0.872, **kappa 0.612** | ✅ consistency, independent of causality |
| **E6** | The strategic pathway is wired end-to-end | `route_acc_zeronav` **0.2167** vs chance 0.3333 — removing the input makes it **worse than guessing** | ✅ *the pipe works; the water is wrong* |
| **O1** | Operative level validated | `ade_0_2s` **0.4271 [0.3675, 0.4871]** (full-set, cluster bootstrap) | ✅ ⚠️ but this is a **`wm_fidelity`** number (§PC2), not a driving number |

### 1.2 Evidence that has been RETRACTED and must not be re-cited

- **E2 / H26 "`ctx→tactical` is load-bearing, Δ +0.0439 CI-separated"** — **RETRACTED 2026-07-25.**
  `overlapping_holdout_se` **manufactured the effect**: it moves the point estimate, not just the
  interval (**×2.97** here; **×4.29 with a sign flip** synthetically). True full-set paired Δ **+0.0148**,
  which fails the practical floor (`MIN_ACC` 0.02) **on the point estimate alone**. ⇒ **0 of 3 seams
  load-bearing, not 1 of 3.**
  ⚠️ **This does not weaken the thesis.** All three seams were measured under **PC1-violated** conditions
  with a **biased estimator**. *0/3 under a broken instrument is what an untested hypothesis looks like.*
- **"REF-C ties/beats us ⇒ flat wins"** — void: **`nav_cmd=None`** in `refc_eval.py:78`,
  `refc_rerank.py:262`, `plan_fan.py:549`. Logged as C6 **twice** (07-21, 07-25).
- **"AlpaSim Δ −0.43, REF-C 8/12 vs 2/12"** — superseded by the **balanced n=37**: **Δ −0.1228
  [−0.2079, −0.0412]**, ~3.5× smaller, **roundabout and highway TIED**, and **both models collapse at
  uncontrolled intersections (flagship 0/7)**.
- **"v1.6 is best ADE in the program"** — retracted (C4 propagation); paired Δ vs v1 **+0.0104
  [−0.0888, +0.1147]**, not separated.

### 1.3 ⚠️ Three eval-path defects that must be closed before any comparison

`MEASURED` (`HPP0_CONFOUND_AUDIT.md` §1.4) — **the program has three mutually exclusive route-input
bugs, and no arm has ever been scored with a produced, non-oracle, non-constant route:**

| class | arms | defect |
|---|---|---|
| **Echo** | v1 / REF-A / REF-B | input present, target is a lookup of the input |
| **Withheld** | REF-C (3 files) | `nav_cmd=None` — the input is never exercised |
| **Oracle** | v1.5 / v1.6 / **v4 MODE B** | route/`vt` tokens fed **from GT labels minted off the ego's own future** |

`V4_FLAGSHIP_DESIGN.md:558-560` forbids exactly the third (*"No leaderboard number may come from a
GT-derived plan or a GT-derived goal"*). ⇒ **v1.6's 0.4375 and every v4 MODE-B number are goal-oracle
numbers, and neither the registry nor the leaderboard says so.** The fix is a **disclosure**, not a
re-run, and it should reach the registry owner today.

---

## 2. The problems and the data — pointers

- **§1 — the nine decision problems** (S1–S4 strategic, T1–T4 tactical, O1 operative), each with input,
  option set, non-circular ground truth, metric, and the structural failure argument:
  → **`STRATEGIC_TACTICAL_PROBLEM_SPEC.md`**
  Headline: **four of the nine have no ground truth on our parity corpus at all**, and every missing
  piece is supplied by an asset we already own and have never switched on.
- **§2 — the corpus verdict and the data-building work package** (label schemas, decision-point miner,
  option-set construction, counterfactual GT, coverage targets, power arithmetic):
  → **`DATA_STRATEGY.md`**
  Headline: **no single corpus can carry the proof** ⇒ **train** PhysicalAI · **prove** AlpaSim ·
  **publish** Cosmos-Drive-Dreams. AlpaSim *evaluation* is **~1.25 pod-days for a 200-scene, 3-arm
  suite** — the expensive thing in this program is arms, not scenes.

---

## 3. THE PROOF PROGRAM — the experiment ladder

### 3.0 Admissibility rules, binding on every row below

1. **Matched parameters ±5 %**, matched training steps, matched corpus, matched labels available,
   matched schedule. Parameter matching is achieved by **widening the flat arm's decoder**, and the
   final counts are recorded in the registry before launch.
2. ⭐ **The flat control receives the IDENTICAL working route/goal input.** Any comparison that
   withholds it reproduces the C6 confound and is void on arrival. *This is the single most important
   line in the ladder.*
3. **Paired episode-cluster bootstrap** (`taniteval/ci.py`, B=2000), estimator named, resampling unit
   named (val episode / AlpaSim scene). **`overlapping_holdout_se` appears nowhere** — it biases the
   *mean*, not only the interval.
4. **≥4 seeds where a curve is claimed**; a single-seed arm-vs-arm result is a **screen with no verdict
   authority** and is labelled as such.
5. **Every metric names its horizon and its surface** (`open-loop-choice` / `closed-loop` / `probe`).
   `GATE_PROTOCOL` §0 refuses K≤20 and K>190.
6. **Both outcomes pre-registered before the run**, and — the 07-26 **C10** lesson —
   ⚠️ **the evaluator must be executed against a synthetic case that MUST fail each registered
   guardrail, and confirmed to render the failing verdict, BEFORE the deciding run.** *A guardrail
   written in a document and absent from the code is a comment.*
7. **AlpaSim numbers are paired-only**, and carry the qualifier *"on NuRec reconstructions — a
   within-sim RELATIVE comparison, not a real-world rate"* plus the **3.21× OOD** ratio.

### 3.1 The arms

| Arm | Structure | Route/goal input | Params | Purpose |
|---|---|---|---|---|
| **A0-FLAT** | encoder + WM + **one** trajectory decoder. No strategic head, no tactical head, no intent conditioning path | ✅ **identical working goal input**, FiLM into the decoder | matched to A2 **±5 %** by widening the decoder | the control |
| **A1-TAC** | A0 **+ explicit tactical level** — single-axis discrete manoeuvre vocabulary + anchor prior, conditioning the operative decode | ✅ identical | matched ±5 % | isolates the tactical rung |
| **A2-HIER** | full — strategic (vision+goal → branch over the option set) → tactical → operative, each planning through the WM | ✅ identical | reference | the treatment |

**Shared config, all arms:** corpus **v2-balanced** (junction 61.4 %, step-weighted turns 28.0 %);
labels **v2.1/v3** (`--labels-v2`, coverage 80.43 %, circularity broken); **LEVER A on**
(`--v2-route-from-vision --route-vis-weight 0.3`); PC2 forward-hook assertion armed; identical seeds.

> ⚠️ **A1's vocabulary must be single-axis.** `MEASURED` root cause: the v1 5-way softmax mixes lateral
> and longitudinal classes, which is one mechanism behind **0/881 accelerate predictions**. A tactical
> rung that cannot say "turn left **and** brake" is not a tactical rung; it is a bug being ablated.

### 3.2 The eight discriminating predictions — pre-registered, both outcomes committed

HP-1…HP-6 are Part A's; **HP-7 and HP-8 are new** and are the only two *structural* arguments in the
set (§5 of the problem spec).

| # | Prediction | Metric · surface · horizon | Estimator | Power need | ⛔ **Falsifier** | GPU |
|---|---|---|---|---|---|:--:|
| **HP-1** | **Advantage grows with horizon.** ≈ tie at 2 s, separates by 10–20 s | paired Δ corridor-departure & ADE at **K ∈ {20, 60, 120, 185}** · closed-loop | paired cluster bootstrap; **interaction test across K** | n ≥ 200 (2-arm) | **Δ flat across horizon** ⇒ no hierarchical benefit | eval-only |
| **HP-2** | **Advantage concentrates at decision points**, not cruise | junction/multi-option stratum vs straight-cruise, paired | same, stratified | ≥200/stratum | **uniform Δ across strata** ⇒ the gain is capacity, not strategy | eval-only |
| **HP-3** | **Route-conditionality**: same scene, different goal ⇒ different **and correct** trajectory | `cross_track_2s_m` + p90 divergence **and** `counterfactual_route_correctness` vs chance `1/\|options\|` · probe | paired bootstrap; direction score must clear chance | ≥40 decision clusters | **A0 passes HP-3 too** ⇒ conditionality is not hierarchical; or **A2 fails** ⇒ PC1 regression, not a model verdict | **0** |
| **HP-4** | **Compositional generalisation to unseen junction TOPOLOGIES** (not just unseen episodes) | Δ degradation on held-out topology classes | paired, by topology | ≥40 clusters/class | **equal degradation** ⇒ no compositional benefit | eval-only |
| **HP-5** | **Structure substitutes for data** | matched-param learning curves at data fractions {25, 50, 100} % | ≥4 seeds; **no exponent without window + R² + n**; R²<0.80 ⇒ use the matched-step ratio | 4 seeds × 3 points | **identical or worse slope** ⇒ structure buys no sample efficiency | **20.2 pd** |
| **HP-6** | **Recovery / re-planning after perturbation** | route re-acquisition rate after a lateral offset (reuses E2a machinery: offset perceivable **R²=0.72**, loss **91 % downstream**) | paired | ≥200 | **no difference in re-acquisition** ⇒ no hierarchical control benefit | eval-only |
| ⭐ **HP-7** | **NEW — branch-mean collapse.** A unimodal marginal policy converges to the **centroid of the option set**, i.e. off-road *between* branches | `between_branch_rate` = P(pred closer to option-centroid than to any single option) at \|options\|≥2 · closed-loop | paired | ≥200 | **equal `between_branch_rate`** ⇒ the mode-collapse mechanism is wrong and is withdrawn | **0** |
| ⭐ **HP-8** | **NEW — decision persistence.** A strategic level *holds* a branch; a marginal model re-decides each frame | `branch_flip_rate` over the last 10 s of approach; `plan_stability` on `pred_dense` | paired | ≥200 | **equal flip rate** ⇒ no persistence benefit; H20 stays parked | **0** |

**HP-7's `MEASURED` corroboration — and why it is not proof.** On the balanced n=37 AlpaSim suite the
flagship's offroad rate by category is **intersection 0.86 · roundabout 0.62 · traffic-light 0.50 ·
straight 0.25**, with a wide-swerve signature (**plan_dev 0.91 vs REF-C 0.33**). That is the *shape*
HP-7 predicts — concentrated exactly where the corridor branches. It is **not evidence for HP-7**:
plan-deviation is not between-branch distance, and the two arms differ in more than one respect
(class C6). `between_branch_rate` is the discriminating measurement and it does not exist yet.

**Six of the eight predictions need ZERO new training.** They need the *instrument* and the *decision
set*, both of which are §2 work. That is the cheapest discrimination available and it is why §4 puts
them first.

### 3.3 The ablation ladder — experiments, costs, dependencies

**Cost basis, `MEASURED`:** flagship v1 trained at **10.888 s/step** (`sum(step_s)` = 326,638.2 s over
29,999 steps = 90.73 h). ⇒ **30 k = 3.78 pod-days · 15 k = 1.89 · 10 k = 1.26.** pod1's v2-corpus arm
runs at ~11 s/step, so the figure transfers.

| ID | Experiment | Arms × seeds × steps | Pod-days | Depends on | What it licenses |
|---|---|---|---:|---|---|
| **X0** | **Rung screen** | A0/A1/A2 × 1 × 10 k | **3.8** | PC1 flags, PC2 assert | wiring de-risked; effect sizes for power; **PC1 gate answered** (`route_skill > 0` CI-separated at 10 k). ⚠️ **NO verdict authority — 1 seed** |
| **X0b** | ⭐ **Seed-variance probe** | A2 × 2 extra seeds × 10 k | **2.5** | X0 | ⚠️ **the program has never measured flagship seed-to-seed variance.** If between-seed σ exceeds the expected between-arm effect, the 4-seed design is underpowered and **X2 must be resized before 30 pod-days are spent.** Cheapest insurance in the plan |
| **X1** | **PC1 gate arm** | — | 0 | folded into X0 | `route_skill > 0` CI-separated **and** `nonav_route_beats_majority` PASS ⇒ **PC1 met**; below that, **the ladder does not launch** |
| **X2** | ⭐ **THE DECISION CONTRAST** | A0 vs A2 × **4** × 30 k | **30.2** | X0, X0b, PC1–PC4 | the proof — or a *fair* negative, which is real information |
| **X3** | Middle rung | A1 × 4 × 30 k | **15.1** | X2 separates | **which level** carries the effect (tactical vs strategic) |
| **X4** | HP-5 data efficiency | {A0,A2} × 4 × 10 k × {25 %, 50 %} | **20.2** | X2 (100 % cell amortised from X2's 10 k checkpoints) | the data-efficiency claim — H3's *raison d'être*, unmeasured for the program's whole life |
| **X5–X8** | HP-3, HP-4, HP-6, HP-7, HP-8 | eval-only on X2's checkpoints | **0** | decision set + instrument | six of eight predictions |
| **XE** | AlpaSim paired eval | 200 scenes × up to 3 arms | **~1.25** (eval pod) | suite scale-up | the powered closed-loop surface |

**Budget summary:**

| tier | contents | pod-days | with 3 pods |
|---|---|---:|---|
| **Instrument + data** (§4 wave 0–1) | probes, miner, firewall, suite scale-up, eval-only predictions | **~1.3** (eval pod) + 0 GPU | days, in parallel with training |
| **Minimum decision-grade** | X0 + X0b + X2 + all eval-only + XE | **~37.8** | **~13 calendar days** |
| **Full program** | + X3 + X4 | **~73** | **~25 calendar days** |

⚠️ **Do not truncate X2 to 15 k to save 15 pod-days.** `MEASURED` counter-evidence: v1 went **0.6152
@19 k → 0.4271 @30 k**, H18's grounding dominance **grew** (Δ 2.82 @19 k → 2.9568 @30 k), and the
ctx→tactical seam moved 0/3 @19 k → 1/3 @30 k (before its own retraction). **Hierarchy effects appear
with training**, so a truncated ladder buys a cheap false null — the exact failure this whole program
exists to avoid.

### 3.4 Priority order — so a killed program still yields value

1. **HP-3 pre-fix null** (today, 0 GPU) — the baseline the probe exists to establish.
2. **The four §7 gating probes** (VectorMap connectivity · `trafficsim` · circularity firewall ·
   Cosmos-DD count) — each independently decides whether a whole problem class is buildable.
3. **X0 + X0b** — PC1 answered and the 30-pod-day decision correctly sized, for **6.3 pod-days**.
4. **X2** — the contrast.
5. **X5–X8** on X2's checkpoints — six predictions for zero GPU.
6. **X3, X4** — mechanism and data-efficiency.

Every stage **banks incrementally**: X0 alone answers PC1 and retires or confirms the whole HPP-1 work
list; X2 alone renders the dominance verdict; X3/X4 explain it.

---

## 4. Sequencing against reality

### 4.1 The fleet, and the honest read on *"if flagship is ready"*

| Resource | State | `MEASURED`/`INHERITED` | Frees |
|---|---|---|---|
| **pod2** | flagship-v4-fromscratch, step ~24,300/30 k | `INHERITED` (`LOOP_STATE` drumbeat) | **~7 h** → then ckpt backup + the formal 8-metric gate |
| **pod1** | flagship-v2corpus, ~85 h into a ~90 h run | `INHERITED` | ~5 h + eval |
| **pod3** | E1c (held-out-gated CL-SFT) | `INHERITED` | ~1 day |
| **eval pod** | ⭐ **free** | brief | **now** |
| **dev box** | all instrument work, no GPU | — | **now** |

⚠️ **"Flagship is ready" is not established, and the plan must not assume it.** `MEASURED` at the first
decision-grade eval of the v4 arm (step 15,000, cluster bootstrap, harness pinned by reproducing v1's
0.4271 exactly): **ADE@2s 0.5839 [0.4962, 0.6821]**, paired vs v1 **Δ +0.1568 [+0.0630, +0.2504],
CI-SEPARATED BEHIND v1**. The trainer's "~0.48 and descending" was a **different reduction** (dense-20
vs the 4 gate waypoints) — retracted 07-25 as class C1. **The 30 k gate decides, and the ladder's arms
are launched on the trainer config that wins that gate, not on an assumption.**

### 4.2 🟥 A gate reconciliation that must happen in the next ~7 hours

`nonav_route_beats_majority` is a **KILL secondary** of the v4 30 k gate
(`V4_FLAGSHIP_DESIGN.md:1305,1376`) and it reads **0 / FAIL** — **by construction**, because
`route_target = _NAV_TO_ROUTE[nav_cmd]` makes `route_skill` exactly 0. The dry-run's
`verdict_from_kill_only` reads **CONTINUE** (`INHERITED`, `v1_g1_dryrun_gate_FIXED.json`), so the two
readings need reconciling **before** the real gate renders.

> **Recommendation (PI/orchestrator decision, not agent-autonomous):** adjudicate
> `nonav_route_beats_majority` as **INSTRUMENT-FAIL, not MODEL-FAIL**, and record that adjudication on
> the gate card with a pointer to `HPP0_CONFOUND_AUDIT.md` §1.5. Otherwise the program risks **killing
> a healthy arm for a label bug** — and the label bug is fixed by two flags that already exist.

### 4.3 ⭐ What can start TODAY — zero GPU, eval pod free

| # | Item | Owner-type | Effort | Why it is first |
|:--:|---|---|:--:|---|
| **1** | ⭐ **VectorMap CONNECTIVITY probe** on an AlpaSim scene — does the lane graph carry `next_lanes`/`prev_lanes`, or only polygons? **Two probes** (trajdata API + raw USDZ prim) | eval pod, read-only | ~1 h | **Gates S1, S2, S4 and HP-4 — four of the nine problems.** `gate0_prereq_probe.json` measured *counts only* and its trajectory read **errored**. Highest leverage $0 probe in the program |
| **2** | ⭐ **`trafficsim` (SMART/CAT-K) one-scene rollout** — and verify non-ego agents **deviate from their logged tracks** | eval pod | 1–3 d | Gates T1–T4's `Y_outcome`. Apache-2.0, in-tree, on the pod, **never once enabled** |
| **3** | ⭐ **The two CONFIG-ONLY PC1 fixes**, staged into the next flagship launch config: `--labels-v2` and `--v2-route-from-vision --route-vis-weight 0.3` | launch decision | **0** | Both already exist in `train_flagship4b.py`. `MEASURED`: the aux loss does **not** collapse (last-20 mean `route_vis` 0.608–0.763), i.e. route-from-vision is genuinely hard, not degenerate. **Only two abandoned arms ever ran it** |
| **4** | ⭐ **HP-3 zero-training probe** on `flagship-30k` + `refc-base-30k` (`taniteval/strategic_probes.py`, exists, 17 tests, `INVOCATION` asserted by a test) | eval pod | minutes | Establishes the **pre-fix null**, both outcomes pre-registered. `MEASURED` on fixtures: the FLAT arm scores **exactly 0** divergence while echoing the command **1.0** — the probe separates *"the command reached the logits"* from *"the command reached the trajectory"* |
| **5** | **Kill `nav_cmd=None`** in `refc_eval.py:78`, `refc_rerank.py:262`, `plan_fan.py:549`; **disclose the eval-time route oracle** in `eval_flagship_v{15,16,4}.py` | dev box | S | Three one-line edits close a confound logged **twice**; the oracle fix is a **disclosure**, not a re-run, and it touches three shipped headline numbers |
| **6** | **PC2 forward-hook assertion** in `rollout.collect` / `runner.py` — any arm claiming a hierarchy whose scored pass leaves a counter at 0 **fails loud** | dev box | S | Turns PC2 from an inspection into a test |
| **7** | **`blind_conditioning_baseline` firewall** + regression test that a synthetic echo label is REFUSED; run against every existing label | dev box, CPU | S | Would have caught the v1 route label for **CPU-minutes** instead of months |
| **8** | **$0 arithmetic:** AlpaSim category frequencies over the **356 banked screened labels** → project onto the 1,606-scene pool; and the Cosmos-DD junction/roundabout count from cached metadata | dev box | ~1 h | Decides whether **S4 is powerable at all** and whether a **publishable twin exists** — before any download or scale-up is scheduled |
| **9** | **Wire `lateral.block` + `corridor.from_windows` into `driving.tier0`** (2 lines, both guarded, both already pass `assert_no_deprecated_estimator`) | dev box | S | 🟠 flagged in `HPP1_UNBLOCK_REPORT` §5 as needing an owner **"or it becomes the next 10-day orphan"** |

### 4.4 What waits, and on what

| Waits for | Item |
|---|---|
| **pod2 at 30 k (~7 h)** | ckpt backup to HF + the **formal 8-metric gate** with the §4.2 adjudication recorded |
| **PC1 PASS (X0/X1)** | the entire X2 ladder. **`route_skill > 0` CI-separated is a hard launch gate** — running X2 before it reproduces the confounded null this program is correcting |
| **VectorMap connectivity (item 1)** | S1/S2/S4 option sets ⇒ HP-3's *real* counterfactuals, HP-4, HP-7 |
| **`trafficsim` validated (item 2)** | T1–T4 |
| **Suite scale-up to n≥200** | any *powered* per-category closed-loop claim. Today per-category n=6–8 is **directional, not powered** |
| **`pred_dense` re-run per arm** | `corridor.from_windows` on any archived arm — 🟠 **PC3 is unblocked in code but unmeasured on any real arm** |
| **PI decision** | the Cosmos-DD publishable twin (2–3 eng-days) — **schedule it from the start or the result is `gated-confidential` twice over** |

### 4.5 The critical path, in one line

**VectorMap connectivity → decision-point miner → PC1 flags on the next launch → X0 (PC1 gate + rung
screen + seed variance, 6.3 pod-days) → X2 (30.2 pod-days) → the eval-only battery on X2's
checkpoints.** Everything before X0 is **zero GPU** and runs today in parallel with three busy pods.

---

## 5. ⛔ The falsifier — committed in advance

> ### Dominance is FALSIFIED if:
>
> With **PC1–PC4 all met** (route input CI-separated above the blind baseline · PC2 asserted **in code**
> · horizon ≥ 18 s with the corridor co-primary · decision-rich set at **n ≥ 200 decision
> episode-clusters per stratum**), the **paired episode-cluster bootstrap** (B=2000, resampling unit
> named) of **Δ(A2-HIER − A0-FLAT)** on the pre-registered co-primary —
> **junction / multi-option `route_compliance_rate` @ K = max horizon, closed-loop** —
> has a **CI containing 0, or favouring A0**, across **≥4 matched seed pairs at matched parameters
> (±5 %) and matched steps**, **while A0 also passes HP-3** (route-conditional divergence in the
> commanded direction, CI above chance `1/|options|`).
>
> **That combination says the structure buys nothing that the same capacity, with the same inputs, does
> not already buy. It is a real result and it will be reported as one — in the registry, in the
> retraction log with its root-cause class, and to the PI — with the same prominence as a positive.**

**Secondary falsifiers, each already tabled in §3.2:** HP-1 flat across horizon · HP-2 uniform across
strata · HP-4 equal degradation on unseen topologies · HP-5 identical or worse slope · HP-6 equal
re-acquisition · HP-7 equal `between_branch_rate` · HP-8 equal flip rate.

**What a falsification would NOT license.** It would not license *"the hierarchy is wrong"* in general —
only *"on this corpus, at this scale, with this implementation, the explicit decomposition adds nothing
measurable"*. Six of the eight predictions are measured on **NuRec reconstructions at 3.21× OOD**, and
that qualifier travels with the verdict in both directions.

**And the honest note kept in the record** (Part A's, restated): *the reframe buys the hypothesis a
fair test, not immunity.* The difference between this and the withdrawn "drop the claim" proposal is
that we now measure the thing under conditions where it **could** show up.

---

## 6. Escalations — do not let these live only in this file

1. 🟥 **The v4 30 k gate's `nonav_route_beats_majority` KILL secondary reads 0/FAIL by construction.**
   Adjudicate as **INSTRUMENT-FAIL** on the gate card, or a healthy arm may be killed for a label bug.
   **Window: ~7 hours.** Owner: orchestrator / gate owner.
2. 🟥 **The two PC1 fixes are config-only and already in `train_flagship4b.py`.** The next flagship
   launch can carry them at **zero engineering cost**. This needs a **launch decision**, not a work
   package. Owner: PI / launch owner.
3. 🟥 **Three shipped headline numbers are goal-ORACLE numbers** (v1.5, v1.6 0.4375, every v4 MODE-B).
   The fix is a **disclosure**. Owner: registry owner, today.
4. 🟠 **`driving.tier0` still does not call `lateral.block` / `corridor.from_windows`** — two guarded
   lines, flagged as at risk of becoming the next 10-day orphan. Needs a named owner.
5. 🟠 **No archived arm has `pred_dense`**, so PC3 is unblocked in code and **unmeasured on any real
   arm**. One `rollout.collect` re-run per arm (GPU-minutes each).
6. 🟠 **The AlpaSim NRE image-pull procedure exists only as prose** in `LOOP_STATE.md`; the transcribed
   script in `BUILD_AND_USE.md` §3 has **never been re-run**. A pod reset costs the single hardest step.
7. 🟡 **`schema.py` may mis-register nuScenes' licence** (`share_alike=False` vs CC-BY-NC-**SA**-4.0).
   Escalate to Data Engineering; do not fix in passing.
8. 🟡 **The Cosmos-DD twin must be scheduled from the start**, not after the internal result — the
   derivative rule makes a PhysicalAI+AlpaSim result `gated-confidential` twice over.

---

## 7. Deliverable manifest

| Artifact | Where it lives |
|---|---|
| **`4BRAIN_DOMINANCE_PROGRAM.md`** (this file) | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-4brain-dominance-program/` — repo **working tree, NOT staged** (per brief) |
| **`STRATEGIC_TACTICAL_PROBLEM_SPEC.md`** (§1) | same directory |
| **`DATA_STRATEGY.md`** (§2) | same directory |
| Primary sources read | `01_EXECUTION_PLAN.md` Part A · `HPP0_CONFOUND_AUDIT.md` · `HPP1_UNBLOCK_REPORT.md` · `ALPASIM_STATE.md` · `TANITSIM_FORK_RECOMMENDATION.md` · `MODEL_REGISTRY.md` §6 + §7 · `RETRACTION_LOG.md` (all 107 lines) · `GATE_PROTOCOL.md` §0 · `CORPUS_PROFILE.md` · `DATA_STRATEGY_FOR_HIERARCHY.md` · `H2_EXTERNAL_DATA_SURVEY.md` · `E1B_RESULTS.md` · `gate0_prereq_probe.json` · `LOOP_STATE.md` (fleet line) |
| Code inspected (unmodified, existence + size verified) | `taniteval/taniteval/{corridor,lateral,strategic_probes,hierarchy}.py` |
| Pods touched · GPU used · training or eval run · files staged/committed/pushed | **none / none / none / none** |

**Verification for a reader:** every `MEASURED` number above resolves to a file named in the row or in
the manifest. Where a number is `INHERITED` (the fleet state, the v4 dry-run's `verdict_from_kill_only`)
it is labelled, and **no `INHERITED` claim decides a GPU-day in this plan.**

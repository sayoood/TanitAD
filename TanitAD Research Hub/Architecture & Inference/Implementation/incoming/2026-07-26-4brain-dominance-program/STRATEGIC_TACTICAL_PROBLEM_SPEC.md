# The strategic and tactical problems — a measurable specification

**Date:** 2026-07-26 (Europe/Berlin) · **Author:** chief-architect, 4-brain dominance stream
**Part §1 of** `4BRAIN_DOMINANCE_PROGRAM.md` · **Companion:** `DATA_STRATEGY.md`
**Compute used:** dev box only. Zero GPU, zero pod SSH, nothing staged, nothing committed.

**Evidence classes** (CLAUDE.md operating standard 1): `MEASURED` (ours + artifact path) ·
`PUBLISHED` (cited) · `INHERITED` (another of our docs, **not** re-verified here) · `ESTIMATED` ·
`HYPOTHESIS` · `PLANNED` (does not exist yet; this document is the specification for building it).

> **PI commission (binding):** *"Develop a plan how to test and prove the dominance of the 4-brain
> architecture. This includes the formulation of strategic and tactical problems related to autonomous
> driving, including the necessary data strategy and data building for it."*
>
> **Prior binding ruling:** the hierarchy thesis is **not in question**. The question is how to prove it
> **consequently**. `RETRACTION_LOG.md` 07-25 (C6) records the withdrawal of the "drop the claim"
> recommendation. This document does not re-litigate the thesis; it makes it *provable*.

---

## 0. The admissibility bar — read this before any problem definition

A "decision problem" the program cannot score honestly is worse than no problem at all. Three of the
four HPP pre-conditions failed for label/instrument reasons (`HPP0_CONFOUND_AUDIT.md`), and the single
most expensive one was a **circular target**. Every problem below is defined against these four bars.

### 0.1 ⚠️ The circularity bar — absolute

`MEASURED` (`HPP0_CONFOUND_AUDIT.md` §1.1, `stack/scripts/refb_labels.py:172-175`):

```
route_target(nav_cmd)  ->  return _NAV_TO_ROUTE[nav_cmd]        # the target IS the input
```

Consequence, `MEASURED` on three independent artifacts
(`hierarchy_flagship-{30k,v4.2b-dryrun}.json`, `hierarchy_flagship-30k_v1.json`):
route CE reaches **exactly 0.0** by step ~14.5 k, `route_acc_nav = 1.0000`, and with the command
withheld the head answers `straight` on **240/240** valid windows ⇒ **`route_skill = 0.0000`, by
construction**. Months of "the strategic seam is load-bearing" rested on a lookup table.

> **RULE — inadmissible target.** *A target that is computable from the model's inference-time inputs
> by a fixed rule is not a target. It is a copy.*

**The firewall — `blind_conditioning_baseline`, and it is a REQUIRED pre-flight, not a review step.**
For every (input set `X`, target `Y`) pair proposed below:

1. Enumerate `X` = exactly what the model receives at inference (not at training).
2. Train a **blind** predictor `Y ← X_cond` on the *conditioning channels only*, no pixels, no
   history: a 2-layer MLP, CPU, minutes.
3. Record `acc_blind` and the majority-class rate `acc_major`.
4. **The label is admissible only if a scene-blind predictor cannot reach ceiling**, and the reported
   capability is always `skill = acc_model − acc_blind`, never `acc_model` alone, with the **paired
   episode-cluster bootstrap** (`taniteval/ci.py`, B=2000) separating it from `acc_blind`.
5. `acc_blind ≥ 0.98 · acc_ceiling` ⇒ **the label is REFUSED and does not enter the program.**

The v1 route label fails step 5 with `acc_blind = 1.0000`. It would never have shipped under this rule,
and the rule costs CPU-minutes. **This is the single cheapest defect-prevention in the whole program.**

**A caution that is not obvious.** `skill > 0` is necessary and not sufficient. A goal input that
*names a direction* ("turn left in 100 m") makes the blind baseline strong but non-trivial, so the
label survives step 5 while still being mostly an echo. The problems below therefore also specify a
**hard variant** whose goal names no direction (§2.1 H, §2.4 H) — the branch is only recoverable by
matching route geometry to junction geometry *in the scene*. That is where a strategic level should
earn its existence.

### 0.2 The in-the-loop bar (PC2)

`MEASURED` (`stack/tanitad/models/metric_dynamics.py:220-244`; callers `taniteval/rollout.py:126-136`,
`bench.py:291-298`): the scored rollout calls `predictor(win_s, win_a)` with **no `intent`, no `ctx`,
no `nav`**, and is fed **the expert's true future actions**. The 0.4271 headline is a dynamics decode
of a control sequence the model was told. **No policy of any shape can differ on that surface except
through dynamics fidelity.**

> **RULE.** A problem is only scored on a surface where the model **chooses**. Every metric below names
> its surface: `open-loop-choice` (the model emits the trajectory), `closed-loop` (the model's output
> is executed and the observation updates), or `probe` (a controlled counterfactual forward pass).
> The true-action rollout is renamed `wm_fidelity_ade_2s` and is never a problem metric.

### 0.3 The horizon bar (PC3)

`MEASURED` (E1a, `e1a_horizon_heldout44_K185.json`, paired common-start, 43 identical windows,
`episode_cluster_bootstrap` B=2000): corridor departure **0.0035 @ K=20 → 0.5877 @ K=185**; junction
stratum **0.025 → 0.8414**; peak XTE **0.35 m → 38.94 m**. A **168×** change in the failure rate
between the standing horizon and the event's horizon.

> **RULE.** Every problem declares its horizon, and its metric is measured **at that horizon**.
> `GATE_PROTOCOL.md` §0 (2026-07-26) already enforces this: `K ≤ 20` is refused as the blind horizon,
> `K > 190` as structurally impossible on PhysicalAI (`corridor.horizon_ceiling(T) = T−W−1`, `MEASURED`).

### 0.4 The power bar (PC4)

Single-arm proportion: **n ≥ 40 episode-clusters per stratum**. Two-arm comparison: **n ≥ 200 per
stratum** (the program's standing bars; the 40-cluster figure is the resampling unit of
`taniteval/ci.py`). Today's decision stratum on canonical val is `MEASURED` at **n ≈ 13**
(240 judgeable windows = exactly 6 per episode over 40 episodes, ~13 of them turns) — i.e. **3× short
of even the single-arm bar**, before any comparison is attempted.

---

## 1. The taxonomy — what each brain is *for*, as a decision problem

The three levels are separated by **what varies while the level is deciding**, not by a time constant
alone. The horizons follow from that.

| Level | The quantity chosen | What is held fixed | Horizon | Output space | Failure signature when absent |
|---|---|---|---|---|---|
| **Strategic ①** | **which branch of the road network** | the agents; the lane-level tactics | **10–30 s** | discrete over the scene's **option set** (map-derived, variable arity) | wrong road taken; or the between-branch average (§5.1) |
| **Tactical ②** | **whether/when to take a gap against other agents** | the branch already chosen | **5–15 s** | discrete over `{yield, proceed}` × timing, or gap index | collision, or frozen-robot; wrong right-of-way |
| **Operative ③** | **the trajectory that executes the above** | branch and gap already chosen | **0–2 s** | continuous `[K,2]` | jerk, corridor departure, dynamics error |
| **(WM ④)** | *not a decision level* — the imagination each of ① ② ③ plans through | — | — | latent | — |

**Why this decomposition and not another.** `MEASURED`, and it is our own root-cause finding: the v1
5-way maneuver softmax **mixes lateral and longitudinal classes** (`lane_keep, turn_left, turn_right,
accelerate, brake_stop`), which is one mechanism explaining **0/881 accelerate predictions**, the speed
fan, and why no arm beats hold-v0 at cruising
(`memory: longitudinal-blindness-root-cause`). A level whose output space cannot represent
"turn left **and** brake" cannot decide a junction. **Every option set below is therefore
single-axis**, and where two axes must be chosen jointly the problem says so explicitly.

---

## 2. STRATEGIC problems (10–30 s)

Each block is the executable spec. `Ⓐ` marks the **admissibility argument** required by §0.1.

---

### **S1 — Branch selection at a multi-option junction**

> *Given the scene and a goal, which of the K roads leaving this junction does the ego take?*

| Field | Specification |
|---|---|
| **Horizon** | decision at **t_dp**; correctness scored over **t_dp → t_dp + 20 s** (K=200 @10 Hz; on PhysicalAI capped at **K ≤ 190**, `MEASURED` ceiling) |
| **Decision point `t_dp`** | the last frame at which the ego is on a lane whose lane-graph successor set has **\|succ\| ≥ 2** and the junction entry is within `d ∈ [15, 60] m`. Derived from the **map**, never from the ego's motion. |
| **Input `X`** | front-camera window (8 frames, canonical crop) · ego state (`v0`, yaw-rate) · **goal**: the route polyline in the **ego frame**, truncated at 30 m and downsampled to 6 points *(easy variant `E`)* **or** a single goal point at 150–200 m *(hard variant `H`)* |
| **Option set** | `succ(lane(ego, t_dp))` from `trajdata.VectorMap` — the map's own successor lanes, arity **variable (2…K)**. ⚠️ Depends on the connectivity probe, §7 item 1. |
| **Ground truth `Y`** | the index of the successor lane the ego's realized future path **first enters**, by polygon containment over `t_dp → +20 s`. |
| **Ⓐ non-circularity** | `Y` is a **map ∩ realized-future** fact. `X`'s goal is a **map ∩ destination** fact. They are related *only through the junction's geometry*, which lives in the pixels and in the map — **not in each other**. In variant `H` the goal point carries no branch name at all: two different junction layouts with the *same* goal point require *different* branches. **Blind baseline `acc_blind` MUST be measured and reported for both variants;** `skill = acc_model − acc_blind`, paired bootstrap. |
| **Metric** | `branch_accuracy` (top-1 over the option set) and **`route_compliance_rate`** = fraction of decision points where the *executed* path (closed-loop) stays inside the commanded branch's corridor to +20 s. Surface: `open-loop-choice` for the first, **`closed-loop`** for the second. |
| **Estimator** | paired episode-cluster bootstrap, B=2000, resampling unit = **scene** (AlpaSim) or **val episode** (real footage). Never `overlapping_holdout_se`. |
| **Why a flat model fails** | (a) its output space is a 2 s trajectory — the branch commitment is **not representable** in it; (b) trained on L2 it learns `E[traj \| obs]`, which at a K-option junction is the **branch-set centroid** — an off-road path (§5.1, and see the MEASURED corroboration there); (c) with variable-arity option sets, a fixed 3-class `{L,S,R}` head cannot even *enumerate* the alternatives at a 5-arm junction or a roundabout. |
| **Corpus** | ✅ **AlpaSim** (VectorMap, 130–472 lane polygons/scene, `MEASURED`) · 🟡 nuScenes / Cosmos-DD (HD map, needs a lane-graph adapter) · ❌ **PhysicalAI-AV — no map exists, so the option set does not exist and S1 has no ground truth on it.** |

---

### **S2 — Lane selection for an upcoming maneuver**

> *Which lane must the ego be in, by when, to take the branch S1 chose?*

| Field | Specification |
|---|---|
| **Horizon** | **10–25 s** before the junction; scored on whether the ego is in a lane whose successors contain the target branch, at the junction-entry frame. |
| **Decision point** | first frame at which the ego is ≥ 2 lanes' lateral distance from *some* lane that admits the target branch, and the junction is within 25 s of travel at current speed. |
| **Input `X`** | as S1, plus the lane count / lane polygons in the ego's forward corridor. |
| **Option set** | the set of parallel lanes at `t_dp`, arity 1…N (map-derived). |
| **Ground truth `Y`** | the lane the ego occupies at junction entry — **and, decisively, whether that lane admits the branch S1 chose** (a pure map fact: `target_branch ∈ succ(lane)`). |
| **Ⓐ non-circularity** | `Y`'s admissibility component is `map(lane) → succ`, entirely independent of both `X` and the ego's motion. The **failure label** (`in a lane that cannot reach the goal at junction entry`) is a map fact the model was never told. |
| **Metric** | `lane_feasibility_rate` = fraction of decision points where the executed path reaches junction entry in a **branch-admitting** lane; plus `lane_change_lead_time_s` (how early the commitment was made). Surface: **`closed-loop`**. |
| **Why a flat model fails** | this is a **multi-step commitment**: the correct action at `t_dp` (start moving right) is only correct because of an event 20 s later. A 2 s-horizon marginal policy has no term that prefers it; the lane change is *free* in its loss and *fatal* in the outcome. This is the cleanest "structure substitutes for horizon" case in the set. |
| **Corpus** | ✅ AlpaSim (lane polygons + connectivity) · 🟡 L2D (`observation.state.lanes` = lane **count**, not index — `MEASURED`; lane index is explicitly flagged untrustworthy) · ❌ PhysicalAI. **PhysicalAI has 1,172 lane-change episodes (`MEASURED`, H2 substrate) but no lane geometry, so the *execution* is observable and the *feasibility target* is not.** |

---

### **S3 — Maneuver-initiation timing (`time-to-maneuver`)**

> *How far ahead does the ego commit, and does it commit at the right moment?*

| Field | Specification |
|---|---|
| **Horizon** | 5–25 s; the target is a **scalar time**. |
| **Input `X`** | as S1 (goal + scene). |
| **Option set** | continuous, but scored as a **band** (`≤2 s`, `2–5 s`, `5–10 s`, `>10 s`) so a regression-to-mean is visible as a band error. |
| **Ground truth `Y`** | `ttm` — already minted at **20.48 % coverage** (`MEASURED`, `labels_train_v4_provenance.json`), and at **62.92 %** as `dist_target`. |
| **Ⓐ non-circularity** | `ttm` is derived from the ego's **future** poses and is **not fed** at inference. ⚠️ It is a *kinematic* target, so it inherits the "description of what the ego did" limit (§6). Admissible for S3 specifically, because *when* is a kinematic fact even when *which branch* is not. |
| **Metric** | `ttm_band_accuracy` + `ttm_MAE_s`, `open-loop-choice`. |
| **Why a flat model fails** | not structurally impossible — a flat model **can** regress `ttm`. **S3 is deliberately included as a NEAR-CONTROL**: if the hierarchy's advantage is uniform across S1/S2/S3 rather than concentrated in the option-set problems, the advantage is capacity, not structure. **S3 is where the hierarchy is predicted NOT to win by much.** Registering a problem we expect to tie is what makes the battery a test rather than a demonstration. |
| **Corpus** | ✅ PhysicalAI (labels exist today, 20.5 % / 62.9 %) · ✅ AlpaSim. |

---

### **S4 — Roundabout exit ordering**

> *Which exit, counted from entry?*

| Field | Specification |
|---|---|
| **Horizon** | 10–30 s (a traversal is ~10–15 s, `INHERITED`, `DATA_STRATEGY_FOR_HIERARCHY.md`). |
| **Option set** | the ordered exit list from the roundabout's lane cycle — **arity 3–6, and the correct answer is an ordinal, not a direction.** |
| **Ground truth `Y`** | the exit polygon the ego's realized path leaves through. |
| **Ⓐ non-circularity** | *"3rd exit"* is not recoverable from any `{left, straight, right}` command, and a goal point 200 m past the roundabout does not name it. This is the **strongest anti-echo problem in the set** — the option set has no direction semantics at all. |
| **Metric** | `exit_ordinal_accuracy`; `route_compliance_rate` closed-loop. |
| **Why a flat model fails** | a 3-class lateral vocabulary **cannot express the answer**. `MEASURED` on our own labels: 4 of 9 v3 ROUTE tokens are **never minted** — `straight`, `exit_left`, `exit_right`, `merge` — each because it *"asserts a junction exists = a MAP fact"*. The vocabulary gap is not an implementation detail; it is the thing S4 measures. |
| **Corpus** | ✅ **AlpaSim — 8 roundabout scenes in the balanced suite, `MEASURED`, and roundabout is the one category where flagship and REF-C are a DEAD HEAT (Δ +0.002)** · ✅ L2D (3,532 roundabout episodes, instructions name the exit, `MEASURED`) · ❌ PhysicalAI (**19 strict roundabout episodes ⇒ descoped**, `MEASURED`, H2 substrate audit). |

---

## 3. TACTICAL problems (5–15 s, agent-relational)

Every tactical problem requires **another agent whose behaviour is not pre-recorded**. On replayed logs
the other agents do not react to us, so "yield vs proceed" has no counterfactual: the log tells us what
happened when the ego did what it did, and nothing about what happens if it does otherwise.

> ⭐ **This is why the unused reactive-agent model matters.** `MEASURED` (`ALPASIM_STATE.md` §7,
> `TANITSIM_FORK_RECOMMENDATION.md` §3): `src/trafficsim/alpasim_trafficsim/catk/smart/` — SMART
> trajectory-tokenisation + **CAT-K** closed-loop fine-tuning, **Apache-2.0, in-tree, on the pod, and
> disabled in every run we have ever made.** Turning it on is the difference between a tactical problem
> and a tactical *description*. **⚠️ Untested by us — no rollout has proven it works. Cost to find out
> is ~1–3 eng-days and it is item #1 of §7.**

---

### **T1 — Yield vs proceed at an unprotected conflict**

> *A crossing/oncoming agent's path intersects ours. Do we go first?*

| Field | Specification |
|---|---|
| **Horizon** | **5–15 s** (from first mutual visibility to conflict-point clearance). |
| **Decision point** | first frame at which a tracked agent's predicted path intersects the ego's chosen branch corridor within 15 s, with `TTC_ego` and `TTC_agent` both finite. |
| **Input `X`** | camera window + ego state + the chosen branch (from ①). **No agent boxes are fed** — the tactical level must read the agents from pixels; feeding boxes would make T1 a geometry exercise. |
| **Option set** | `{yield, proceed}` × commitment time. (Binary by design: three-way `{yield, proceed, creep}` is a v2 extension, not the first test.) |
| **Ground truth `Y`** | ⚠️ **Two distinct constructions, and they must never be pooled.** <br>**(a) `Y_expert` (imitation target, replay-safe):** did the human ego reach the conflict point first? A pure geometry+track fact. Admissible but **weak** — the expert's choice is one sample of a bimodal decision. <br>**(b) `Y_outcome` (the real target, requires reactive agents):** roll the policy in AlpaSim with `trafficsim` ON and score the **outcome**: `at_fault_collision`, `conflict_clearance_margin_s`, `progress`. This needs no yield/proceed label at all — **the environment adjudicates.** |
| **Ⓐ non-circularity** | `Y_outcome` is **not a label**; it is a simulated consequence. It cannot be circular with any input by construction. This is the strongest admissibility position available anywhere in this document, and it exists **only** because we own a reactive simulator. |
| **Metric** | `at_fault_collision_rate`, `conflict_margin_s` (p10 and median), `frozen_robot_rate` (progress < 20 % of nominal with no obstruction) — the last one is essential or "always yield" wins. Surface: **`closed-loop`**. |
| **Why a flat model fails** | (a) the decision is **bimodal with a shared observation** — go-first and yield are both feasible, and an L2/marginal head averages them into "creep into the conflict point", which is the worst option; (b) `MEASURED` corroboration: **at uncontrolled intersections BOTH arms collapse — flagship 0/7, REF-C 1/7, flagship offroad 6/7** on the balanced suite. Neither model has a tactical level in the scored loop; this is the category where the program's models are *at their worst*, which is exactly where a tactical brain must show up or it does not exist. |
| **Corpus** | ✅ **AlpaSim + trafficsim ON** (only source) · 🟡 PhysicalAI `obstacle.offline` for `Y_expert` only (3D tracks on **96.90 %** of the corpus, `MEASURED`; **2 of 36 features currently ingested**) · ❌ replay-only sources for `Y_outcome`. |
| ⚠️ **Pre-registered caution** | `RETRACTION_LOG` 07-21 (C3): the *"`obstacle.offline` unblocks agent-relational tactics"* argument was **falsified** by its own gate (**+1.16 % [−0.92, +3.19]** on the 2 s longitudinal target). **That retraction is about agent tracks as an INPUT FEATURE for trajectory regression. T1 uses them for TARGET/option-set construction on a different surface and horizon — a different claim, and it gets its own pre-registered gate (§7 item 6). Do not read the old null as covering this, and do not spend the 12.4 GB ingest before that gate passes.** |

---

### **T2 — Gap acceptance for a merge or lane change**

| Field | Specification |
|---|---|
| **Horizon** | 5–12 s. |
| **Decision point** | ego must change lane (from S2) and the target lane contains ≥1 tracked agent within ±40 m. |
| **Option set** | the **ordered set of gaps** in the target lane: `{gap_0 (ahead of lead), gap_1, …, abort}` — arity variable, and *the option set is composed of other agents*, which no map provides. |
| **Ground truth `Y`** | **(a)** the gap the expert entered (`Y_expert`); **(b)** `Y_outcome` = merged without an at-fault event and without forcing a following agent below a 1.5 s headway (**requires reactive agents to be meaningful** — on a replay the follower cannot brake for us). |
| **Metric** | `merge_success_rate`, `induced_deceleration_on_follower` (m/s², reactive only), `abort_rate`, `time_to_merge_s`. Surface: **`closed-loop`**. |
| **Why a flat model fails** | the option set is **agent-indexed and variable-arity**; a fixed action head has no slot for "gap 2 of 4". A marginal model can only produce a lateral profile, and the *choice of which gap* is precisely the tactical quantity. |
| **Corpus** | ✅ AlpaSim + trafficsim · 🟡 PhysicalAI 1,172 lane-change episodes for `Y_expert` (`MEASURED`) · ✅ **L2D's ego turn-signal is a free, dense, 5.5–6.7 s-early declaration of this decision** (`MEASURED`: non-zero on 18.36 % of frames, direction correct on 79.9 % of the 358 onsets followed by a real manoeuvre). |

---

### **T3 — Overtake vs follow**

| Field | Specification |
|---|---|
| **Horizon** | 8–15 s. |
| **Decision point** | a lead vehicle is present with closing speed > 0 and `gap < 3 s` headway. `MEASURED` lead presence: **38.51 % of windows / 66.1 % of clips** corpus-wide (26 chunks / 25 countries / 614 clips). |
| **Option set** | `{follow (match lead speed), overtake-left, overtake-right, abort}`. |
| **Ground truth `Y`** | `Y_expert` = did the ego's lateral offset from the lead's path exceed a lane width while passing; `Y_outcome` = completed the pass with no at-fault event and no oncoming conflict (reactive). |
| **Metric** | `overtake_completion_rate`, `following_headway_p10`, `oncoming_conflict_rate`. Surface: `closed-loop`. |
| **Why a flat model fails** | this is the **joint lateral+longitudinal** case — the answer is a *pair* (`turn out` **and** `accelerate`). Our v1 5-way softmax cannot represent it (§1), and `MEASURED` 0/881 accelerate predictions is the observable consequence. T3 is the sharpest test of the *output-space* half of the hierarchy claim. |
| **Corpus** | ✅ AlpaSim + trafficsim · 🟡 PhysicalAI for `Y_expert`. |

---

### **T4 — Right-of-way resolution at an uncontrolled intersection**

| Field | Specification |
|---|---|
| **Horizon** | 5–15 s. |
| **Decision point** | ≥2 agents (incl. ego) arrive at an uncontrolled junction within a 3 s window. |
| **Option set** | the **arrival order** over the conflicting set — a *permutation*, arity `n!`, scored as "was the ego's position in the realized order correct under the applicable rule". |
| **Ground truth `Y`** | ⚠️ **rule-dependent and therefore geography-dependent** (`right-before-left` in DE vs `first-to-stop` in the US). Do **not** mint a single global rule label. Use `Y_outcome` (no at-fault event, no deadlock, positive progress) as the primary and treat the rule label as a **diagnostic only**, stratified by `country` (PhysicalAI ships `country`, `MEASURED`). |
| **Metric** | `deadlock_rate`, `at_fault_rate`, `clearance_margin_s`. Surface: `closed-loop`, reactive. |
| **Why a flat model fails** | the decision is **about the other agent's decision** — a joint fixed point. A marginal policy conditioned on the current frame has no representation of "it is yielding to me", which is why both of our arms collapse here (`MEASURED`: 0/7 and 1/7). |
| **Corpus** | ✅ **AlpaSim + trafficsim only.** `SIGNAL` (other agents' indicators) and `LIGHTSTATE` remain **unmintable from every real source we hold** (`MEASURED`, verified against the actual label schemas of nuScenes / AV2 / Waymo / ZOD — none annotates vehicle light state). |

---

## 4. OPERATIVE problem (0–2 s) — largely validated

| Field | Specification |
|---|---|
| **O1** | execute the chosen branch and gap as a dynamically feasible trajectory. |
| **Status** | ✅ **the one level with a decision-grade positive.** flagship v1 `ade_0_2s` **0.4271 [0.3675, 0.4871]** (full-set mean, episode-cluster bootstrap, `MODEL_REGISTRY.md` §6, re-emitted 2026-07-26). |
| **The hierarchy-supporting result, and it is admissible** | **H18 grounding dominance:** grounded operative rollout vs the ungrounded tactical head, corrected paired Δ **+2.9568 m** (was +2.6979 under the deprecated estimator). Un-separating it would require an **8.65×** interval widening; the worst ever measured in this program is **2.06×** and the migration fixture measures **~1.6×**. `MEASURED`, `HPP1_UNBLOCK_REPORT.md` §1.4. |
| ⚠️ **Caveat that must travel with 0.4271** | it is scored on `rollout_decode` **under the expert's true future actions** ⇒ it is a **world-model-fidelity number, not a driving number** (§0.2). Rename to `wm_fidelity_ade_2s` everywhere it is quoted as "drives". |

---

## 5. Why a flat model fails — the two *structural* arguments, stated as testable predictions

Most "a flat model can't do X" claims are false: a flat model with the same inputs and the same
capacity usually can. Only two arguments here are genuinely structural, and both are **new
pre-registered predictions** extending HP-1…HP-6.

### 5.1 ⭐ **HP-7 — branch-mean collapse at multi-option decision points**

**The claim.** A unimodal policy trained on `E[traj | obs]` at a K-option junction converges to the
**centroid of the option set**, which lies *between* the branches and therefore *off the road*. A
level that commits to a branch does not.

**The metric (new, zero training, computable from `pred_dense` once the decision set exists):**

```
between_branch_rate = P( d(pred_path, centroid(options)) < min_k d(pred_path, option_k) )
```

measured at decision points with `|options| ≥ 2`, paired between arms, episode-cluster bootstrap.

**⚠️ Status: `HYPOTHESIS`, with a `MEASURED` corroboration that is NOT proof.** On the balanced n=37
AlpaSim suite the flagship's failure is **offroad, concentrated where the corridor branches** —
offroad by category: intersection **0.86**, roundabout 0.62, traffic-light 0.50, straight **0.25** —
driven by a wide-swerve signature (**plan_dev 0.91 vs REF-C 0.33**). That is the *shape* HP-7 predicts.
It is not evidence for HP-7, because plan-deviation is not between-branch distance and the two arms
differ in more than one respect. **The discriminating measurement is `between_branch_rate`, and it does
not exist yet.**

**Falsifier.** `between_branch_rate` equal (paired CI ∋ 0) between the flat and hierarchical arms ⇒
the mode-collapse story is wrong and must be withdrawn.

### 5.2 ⭐ **HP-8 — decision persistence on approach**

**The claim.** A strategic level *holds* a branch commitment across the approach; a marginal model
re-decides every frame and flips.

**The metric:** `branch_flip_rate` = mean number of argmax changes over the option set across the last
10 s before `t_dp`, and `plan_stability` = mean L2 between consecutive replans of the dense path.
Newly computable: T3-11 landed `pred_dense`/`gt_dense` `[N,20,2]` persistence in `rollout.collect`
(`MEASURED`, `rollout.py:108-152`, working tree).

**Falsifier.** Equal flip rate ⇒ no persistence benefit; H20 (BridgeAD plan-persistence) stays parked.

### 5.3 What is *not* a structural argument — stated so nobody re-derives it

| Tempting claim | Why it is not admissible |
|---|---|
| "REF-C ties us, so flat wins" | ⛔ **`nav_cmd=None` in `refc_eval.py:78`, `refc_rerank.py:262`, `plan_fan.py:549`** — the route input is never exercised. Logged twice as C6 (07-21, 07-25). **The flat control in every experiment below receives the identical working route input.** Any comparison that withholds it is void. |
| "A 2 s ADE tie shows no hierarchy benefit" | ⛔ class **C9** (horizon-blind instrument, `RETRACTION_LOG` standing consequences). The 2 s window hid a dominant failure mode by **168×**. |
| "0/3 seams load-bearing refutes the hierarchy" | ⛔ all three seams were measured under **PC1-violated** conditions with a **biased estimator** (`_jack` moved the point estimate by up to **×2.97**; up to **×4.29 with a sign flip** synthetically). 0/3 under a broken instrument is what **untested** looks like. |
| "The flat model can't be route-conditional" | ⛔ **false, and the HP-3 fixture proves it**: `strategic_probes.py`'s ROUTE-CONDITIONAL synthetic arm passes HP-3 with no hierarchy at all. The claim that survives is narrower: a flat model cannot be conditional on an option set it **cannot enumerate** (S1/S4 variable arity, T2 agent-indexed gaps). |

---

## 6. The honest limit of every label we can mint today

`MEASURED` (`labels_train_v4_provenance.json`, 2,376 eps / 406,099 windows):

| label | coverage | limit |
|---|---:|---|
| `route_valid` (v2.1/v3 curvature-relative) | **80.43 %** (val 79.35 %) | **circularity broken** ✅ — but it is a *description of the ego's arc*, not an instruction |
| `route_token` (9-token vocab) | 80.70 % | **4 of 9 tokens never minted** — `straight`, `exit_left`, `exit_right`, `merge`; each *"asserts a junction exists = a MAP fact"* |
| `dist_target` | 62.92 % | ✅ usable for S3 stratification |
| `strat_scalars.ttm` | 20.48 % | ✅ S3 target |
| `lat_target` / `lon_target` | 100 % | operative only |
| `follow_lead` / `close_gap` / `open_gap` | **0 %** — `lead_state` is a `None` stub | T1–T3 unmintable from labels today |
| `TACPOINT` | names stay `unknown` | — |

> **The one sentence.** *Every route label the program can mint today is a description of what the ego
> did, never an instruction about what it should do — because an instruction requires alternatives, and
> alternatives require a map.* **This is the whole reason the AlpaSim VectorMap finding changes the
> plan.**

---

## 7. What must be true before any of this is buildable — the gating probes

Ordered so that a killed agent still yields value. All are **0 GPU** and the eval pod is free.

| # | Probe | Why it gates | Two-probe requirement (CLAUDE.md rule 2) |
|:--:|---|---|---|
| **1** | ⭐ **Does `trajdata.VectorMap` in the AlpaSim USDZ carry lane CONNECTIVITY (`next_lanes`/`prev_lanes`/`adj_*`), or only polygons?** | **Everything strategic depends on it.** `gate0_prereq_probe.json` `MEASURED` **counts** (`n_lanes` 130–472, `n_road_edges` 130–393) — it did **not** read successors, and its trajectory read errored (`'utils_rs.Trajectory' object has no attribute 'poses'`). **Without connectivity there is no option set, and S1/S2/S4 collapse to kinematic descriptors.** | (a) `trajdata` API on a loaded scene; (b) raw USDZ prim inspection. Do not conclude absence from one. |
| **2** | **Does `trafficsim` (SMART/CAT-K) actually run a rollout?** | Gates every tactical problem's `Y_outcome`. Apache-2.0, in-tree, **never once enabled** | (a) one scene with `trafficsim` on; (b) verify non-ego agents deviate from their logged tracks — otherwise it is replay with extra steps. |
| **3** | **`blind_conditioning_baseline` implemented and run on every existing label** | §0.1 firewall; would have caught the v1 route label for CPU-minutes | code + a regression test that a synthetic echo label is REFUSED |
| **4** | **HP-3 baseline on `flagship-30k` + `refc-base-30k`** (`strategic_probes.py`, exists, 17 tests) | establishes the **pre-fix null** with both outcomes pre-registered; `INVOCATION` is asserted by a test so it cannot rot | run on the free eval pod, minutes |
| **5** | **Cosmos-DD roundabout/junction count from already-cached metadata** | decides whether a **publishable** twin exists at all (§`DATA_STRATEGY.md`) | **$0**, no download |
| **6** | **Pre-registered gate for `obstacle.offline` as option-set/target machinery** | the 07-21 C3 retraction priced a 12.4 GB ingest on an unmeasured mechanism. **Do not repeat it.** Gate on a 1-chunk sample first | 1.1 GB, 0 GPU (the shape of the gate that caught it last time) |

---

## 8. Summary table — the nine problems

| ID | Problem | Horizon | Option set | GT non-circular via | Primary metric | Corpus that can carry it |
|---|---|---|---|---|---|---|
| **S1** | branch selection | 10–30 s | map successors (2…K) | map ∩ realized path; goal carries no branch in variant H | `route_compliance_rate` (CL) | **AlpaSim**; Cosmos-DD/nuScenes w/ adapter |
| **S2** | lane for the manoeuvre | 10–25 s | parallel lanes (1…N) | `target_branch ∈ succ(lane)` — pure map | `lane_feasibility_rate` (CL) | **AlpaSim** |
| **S3** | manoeuvre timing | 5–25 s | time bands | ego future, not fed | `ttm_band_accuracy` | **PhysicalAI today** (20.5 %/62.9 %) |
| **S4** | roundabout exit ordinal | 10–30 s | ordered exits (3–6) | exit polygon ∩ realized path | `exit_ordinal_accuracy` | **AlpaSim (n=8)**, L2D (3,532 eps) |
| **T1** | yield vs proceed | 5–15 s | {yield, proceed} × t | **simulated outcome — not a label** | `at_fault` + `frozen_robot` (CL, reactive) | **AlpaSim + trafficsim ON** |
| **T2** | gap acceptance | 5–12 s | agent-indexed gaps | outcome + expert gap | `merge_success` + `induced_decel` | **AlpaSim + trafficsim** |
| **T3** | overtake vs follow | 8–15 s | {follow, OT-L, OT-R, abort} | outcome | `overtake_completion` | **AlpaSim + trafficsim** |
| **T4** | right-of-way | 5–15 s | arrival permutation | outcome (rule label = diagnostic only) | `deadlock` + `at_fault` | **AlpaSim + trafficsim only** |
| **O1** | trajectory execution | 0–2 s | continuous | — | ✅ validated 0.4271 | PhysicalAI |

**Read in one line:** *four of the nine problems have no ground truth on our parity corpus at all, three
more have only a weak imitation target on it, and every one of the missing pieces is supplied by an
asset we already own and have never switched on.*

---

## 9. Deliverable manifest

| Artifact | Where it lives |
|---|---|
| **This spec** | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-26-4brain-dominance-program/STRATEGIC_TACTICAL_PROBLEM_SPEC.md` (repo working tree, **NOT staged** — per brief) |
| Companions | `4BRAIN_DOMINANCE_PROGRAM.md`, `DATA_STRATEGY.md` (same dir) |
| Primary sources read | `01_EXECUTION_PLAN.md` Part A · `HPP0_CONFOUND_AUDIT.md` · `HPP1_UNBLOCK_REPORT.md` · `ALPASIM_STATE.md` · `TANITSIM_FORK_RECOMMENDATION.md` · `MODEL_REGISTRY.md` §6 · `RETRACTION_LOG.md` · `GATE_PROTOCOL.md` §0 · `CORPUS_PROFILE.md` · `DATA_STRATEGY_FOR_HIERARCHY.md` · `H2_EXTERNAL_DATA_SURVEY.md` · `E1B_RESULTS.md` · `gate0_prereq_probe.json` |
| Code inspected (unmodified) | `taniteval/taniteval/{corridor,lateral,strategic_probes,hierarchy}.py` (existence + size verified) |
| Nothing staged, committed or pushed · no pod touched · no GPU used | ✅ |

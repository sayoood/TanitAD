# TanitAD — Program Overview (living)

> **Whole-program strategic briefing**: the vision & the bet, the three undeniable claims, the four
> edges + the plan to build each, the hypothesis ledger, the phases/timeline, the agent ecosystem,
> current state & achievements, the honest position (P8), and the critical path.
>
> Distinct from the operational `Project Steering/Reports/*-program-report.md` (which tracks the live
> training/gate cadence). **This is the canonical format for every future whole-program briefing** —
> refreshed at each phase boundary and on request.
>
> **Model lineage lives in [`MODEL_REGISTRY.md`](MODEL_REGISTRY.md)** — every version's architecture,
> exact training command, data key, code state, results, and reconstruction gaps, plus the decision log
> behind them. This document is the *strategy*; the registry is the *record*.
>
> **Last refreshed:** 2026-08-03 · Phase 0 (~day 30 / 42) · **Next refresh:** Phase-0 exit, or the
> v5f / v1arch 30 k verdicts.
>
> **Refresh convention:** §5.0 carries the current state and is rewritten each refresh. §§5.1–5.4 are
> the **2026-07-25 snapshot**, retained because their tables are still the provenance for the
> open-loop bake-off; where 5.0 and 5.1–5.4 disagree, **5.0 is newer**. §5.0.5 lists what is
> **RETRACTED** and may not be re-quoted from any of them.
>
> **Evidence class on every load-bearing number** (CLAUDE.md operating standard): `MEASURED` (ours +
> artifact) · `PUBLISHED` (cited) · `INHERITED` · `ESTIMATED` · `HYPOTHESIS`. Unmarked numbers in the
> tables below are MEASURED and traceable to the registry row or raw eval JSON named beside them.
> **Intervals are the episode-cluster bootstrap** (`taniteval/ci.py`) unless a row says otherwise; the
> legacy `± CI95` column that this document used to carry is `overlapping_holdout_se` and is
> **1.28–2.06× too narrow** — it is retained only where it is the only figure on record, and labelled.

---

## 0. The bet, in one sentence
Prove — with falsifiable gates and *recognizable* metrics, not self-defined claims — that a
**sub-300 M hierarchical latent world model (the "4-brain")**, trained on **orders of magnitude less
data with zero perception labels**, drives **better per unit of compute** than incumbent stacks
15–120× larger (Alpamayo-2 32 B, GAIA-3 15 B), then scale only the proven mechanism. Declared
opponents to beat: **Waymo, Wayve, Pony AI, Momenta, Autobrains**. First hard evaluation:
**2026-10-05 (P7)**. Target hardware: Jetson **Orin / Thor**. Focus: **L3/L4** (L2++ is a degradation
of L4, not a goal). North-star framing: win on **hierarchy + imagination + self-monitoring +
per-scenario excellence**, not on scale.

## 1. The three claims we must make undeniable
| Claim | Goal | Proof artifact | Where it stands (**2026-08-03**) |
|---|---|---|---|
| **C1 — It drives** | Goal 1 | Closed-loop success + latency/FLOPs ledger (Orin/Thor envelope) | 🔶 Open-loop **cleared** (0.4271 full-set, below every trivial floor). Closed-loop **not** — now measured four ways: 1.685 m self-referential in-house, **1.488 m [1.329, 1.647] on the n=40 real-footage low-OOD instrument where REF-C base scores 0.564**, 2/12 pass on the reconstruction suite, and — new — **closed-loop on a NuRec reconstruction rendered on the Jetson Thor, where REF-C beats flagship v1** (§5.0.1). ⛔ **The "ENTIRELY LATERAL / ADE does not separate" reading is RETRACTED (R-2026-08-03-C)**: re-measured on the improved render the shipped videos use, ADE separates at **+7.164 [+5.265, +8.966]**, both longitudinal metrics and strategic corridor departure separate too, and all four lateral separations survive and widen — because **flagship v1's driven path moves 9.05 m mean under the render change while REF-C's moves 0.43 m (21×)**. Determinism control exactly 0.0 on 450/450 windows; current panel `stack/experiments/alpasim-gsplat/results/closedloop-hq-render/`. **The deficit is measured LONGITUDINAL, not lane-keeping.** Latency side is now measured **end-to-end on real weights**, not composed: **60.3 ms p50 / 63.1 ms p95 on Thor against a 100 ms budget** (§5.0.2) |
| **C2 — It needs magnitudes less data** | Goal 2 (prio) | Data-efficiency slope vs supervised baseline at matched params | 🔶 Still not a slope — but the **enabling mechanism is now measured**: pseudo-label WM pretraining captures **~96 %** (proxy, 8 seeds) / **109 % speed, 71 % yaw** (parity target, 4 seeds) of real-label value, and an 80-clip CC YouTube pilot reaches **≈92 % of ceiling** (directional). Corpus baseline now exact: **13.13 h, 4.73 epochs at 30 k** |
| **C3 — It is inherently safe & compliant** | Goal 2 (prio) | Fallback brain, self-monitoring AUROC, rule-violation rates, UN-ADS regulation trace | 🔶 Instruments built; D8 separation still preview-only (p≈0.047 paired). **New: SC-14 traffic-light scenario + TLC metric** (`red_entry_gate × stop_quality × green_flow`; a single red-run zeroes it) — design oracle **rule_barrier 1.0 vs soft_prior 0.0**, the H9 hard-barrier claim made scorable. Model-side TLC/LAL/OKRI/LOPS **renderer-gated** |

Efficiency (inference) is embedded in all three — every experiment reports params, FLOPs/decision,
latency, and the **CNCE** metric (first real value: **median 210,551** on the deployed 262.8 M
architecture, comma val, 30 eps).

## 2. The four edges and the plan to build each
| Edge | Core hypotheses | How it's built | Status (**2026-08-03**; ③ and ④ updated, ① and ② are the 07-25 reading) |
|---|---|---|---|
| **① Planning / Hierarchy** | H1 (4-brain: strategic/tactical/operative + fallback-MRC), H26 (cross-level alignment) | Three E2E abstraction layers at different clock rates + a fallback that forces a Minimal-Risk Condition on collapse | 🔶 **Built and in its first real training round.** The v3 design shipped as **flagship-v4** (three *planners* over the WM, ≈247.9 M — 30 M smaller than v1). Four **warm-start** arms failed (best 10 k `ade_0_2s` **0.8522 [0.75, 0.98]** vs a 0.60 bar); a ~0-GPU **cosine pre-probe** (seam cos **+0.0043**) refuted gradient surgery and selected **co-evolution from random init**, which is working (canary **15.674 → ~1.4** under full coupling, ADE ~0.48 at 40 % of training, 10 k gate CONTINUE). Standing evidence unchanged: heads are a lossy readout (P2 planner 0.893 vs head 3.150 open-loop; 1.038 vs 1.685 closed-loop) |
| **② Data efficiency** | H3 (LeJEPA/SigReg world model), H4 (frozen vs trained encoder), H7 (1000× data via IDM) | Latent world model, no perception labels; inverse-dynamics to mine action-free video; focal-length canonicalization | ✅→🔶 **H4 is now SPLIT** (the flat "closed-negative" was unsafe — see the ledger): the ceiling is **static decode** off a frozen JEPA latent (3.65 m, the REF-A band), not freezing — a planner reading the *same* frozen WM through its dynamics reaches **0.599 m**. ⇒ **H4a** (frozen + supervised regression) refuted; **H4b** (frozen + feature-prediction + planning) **OPEN and positive**. **H7 has its first end-to-end evidence** (pseudo-label pretraining ≈96 % / 109 % of real-label value; YouTube pilot ≈92 % of ceiling). The C2 **slope** is still unmeasured — now the clearest single gap |
| **③ Inference efficiency** | H5 (efficient decode/inference transfer), H2 (modality steering), H8 (MoE) | 263 M vs 15–120× larger; imagine-and-select instead of generative rollout | ✅ **Closed for the deployed arm, and now proven ON THE TARGET SILICON.** ⚠️ the old "deploy tick 11.16 ms" was a *different tick* (retracted). **MEASURED end-to-end on Jetson Thor with real step-29999 weights: 60.3 ms p50 / 63.1 ms p95 against a 100 ms budget** (60 windows / 12 eps, K=20, 9-candidate fan) — §5.0.2. **FP16/bf16 is the deployment precision, INT8 REJECTED** (no latency win; W+A INT8 collapses the readout head to cos 0.566 and costs +0.0215 m over 20 steps — **on real weights**; ⛔ the random-weight precision table is RETRACTED, §5.0.5). ⚠️ **Scope: latency + tactical-decision, not four-family accuracy** — `TacticalSelector` has **no production caller today**, so this makes the *designed* path affordable rather than speeding up what runs now. H2/MoE are Phase-1 |
| **④ Safety / self-knowledge** | H11 (self-monitoring), H9 (rule compliance), H15 (imagination), H14 (physical grounding) | Per-level monitors + hard rule barriers + ImaginationField (advection + epistemic σ) + kinematic/Kamm grounding | 🔶 Instruments built and **now partly real**: first MEASURED beyond-ADE numbers on the deployed architecture (decision-tick p50 **14.33 ms**, TMS **0.0435** = expert-log band, CNCE **210,551**) + **SC-14/TLC**. **H15 `vision_use` still flat at ~12 %**; D8 separation still preview-only; σ-dissipation stands (0.357 → 0.011 by k=4). ⭐ **The renderer half of the gap is now OPEN, not blocked** — gsplat renders NuRec natively on aarch64 and a full closed-loop panel ran on Thor (§5.0.3). ⛔ **But the SAFETY half is still missing and for a NEW reason**: it is the **renderer wire contract, not `alpasim_runtime`**, so there is **no collision / offroad / scene score** — TLC/LAL/OKRI/LOPS closed-loop still need the runtime finished (bounded: `cargo` is present) |

## 3. Hypothesis ledger

> **⛔ This section no longer carries a status table.** The single quotable source for hypothesis
> status is **[`TanitAD Research Hub/HYPOTHESIS_LEDGER.md`](../TanitAD%20Research%20Hub/HYPOTHESIS_LEDGER.md)**.
>
> **Why:** two ledgers had diverged — that file's table was frozen at **2026-07-05** while this one
> was live, and H4 / H25–H28 existed only here while H20–H24 and IMP-1…IMP-8 existed in neither.
> A reader trusting the wrong one got a stale answer (`R3_hypothesis_portfolio.md` §1). They were
> merged on **2026-07-25** into one living ledger with mandatory columns
> (`status · DoA % · evidence-class · deciding-artifact-path or "untested" · gate/falsifier ·
> action · last-retested · owner`) and an explicit **PARKED {reason, revisit-trigger}** state.
> **A status with neither a MEASURED artifact path nor an explicit "untested" is inadmissible there.**
>
> Duplicating any row here would recreate the divergence. Quote the ledger, not this page.

**Live summary (2026-07-25) — the portfolio's true active surface, 41 rows across 37 audited
hypotheses.** Counts and statuses are the ledger's; see it for every per-row artifact path.

| Action class | Rows | The live surface |
|---|---:|---|
| **PROVE** — build the decisive test | 3 | **H1b** (hierarchy edge), H18, H26 → the **Hierarchy Proof Program** (`01_EXECUTION_PLAN` PART A) |
| **MEASURE** — instrument exists, just run it | 8 | H3, H7 *(the C2 slope)*, H9, H11, H15 *(D9)*, H27, IMP-4 *(E1b)*, IMP-8 |
| **FIX** — broken instrument before retesting | 4 | H6, H9, H25 *(decorr never-on)*, H26 |
| **PARK** — explicit, with a revisit-trigger | 10 | H2, H8, H10, H12, H16, H17, H20, H21, H23, H24 |
| **UN-PARK** — targets a MEASURED failure | 1 | **H22** (σ-dissipation to chance by k=4 caps both H15 and H11) |
| **RETIRE** — settled, stop spending attention | 11 | H0, H4a, H5, H13, H19, H28, IMP-1, IMP-3, IMP-5, IMP-6, IMP-7 |

**Three splits applied** (a settled negative may not hide an open sub-claim):

- **H4 → H4a** (frozen + supervised regression) **Confirmed-negative, 2.9196, RETIRE** ·
  **H4b** (frozen + feature-prediction + planning) **OPEN and reads POSITIVE at 0.599 m** — the real
  v3 question. ⚠️ *The flat "H4 CLOSED NEGATIVE" this section used to carry is UNSAFE and is
  superseded by the split.*
- **H1 → H1a-operative** (validated, 0.452 m) · **H1b-hierarchy-edge (UNTESTED — the D5/D6 topology
  gates have never run)** · **H1c-planner-coupling** (= H27, in-flight).
- **H14 → H14a-narrow** (kinematic + Kamm, 95.9 % physically-shaped, done) · **H14b-broad**
  (physical-law / ethics / culture injection — **untouched, no gate written**).

**~11 hypotheses genuinely need work.** Ten are retire-able today and ten were presented as "open"
while being un-owned — which is what made the program feel spread thin.

<!-- The per-hypothesis status table that stood here 2026-07-05 → 2026-07-25 was merged into
     HYPOTHESIS_LEDGER.md on 2026-07-25 (Wave-1 workstream E). Do not reintroduce it. -->

## 4. Phases & timeline
| Phase | Window | Goal | Where we are |
|---|---|---|---|
| **Phase 0** — foundation & edge proofs | 07-05 → ~08-15 (6 wks) | Running 4-brain WM, single front cam, open-loop + first closed-loop; gates D1–D6 | **~day 21.** Open-loop bar **cleared**; the bake-off has a verdict; the planner architecture is **built and training**; closed-loop is now **measurable at low OOD** (n=40) though not yet *good*. Binding constraint: **closed-loop longitudinal control + generalization** |
| **Phase 1** — boost & breadth | ~08-15 → 09-20 | Real data at scale + the C2 data-efficiency slope headline; H2 modality steering; H9/H15/H12; NAVSIM/Bench2Drive entries; AlpaSim | Gated on Phase-0 exit — but **two Phase-1 assets landed early**: the balanced **50 h v2 corpus** (built, QA pending) and the **IDM video-pretraining mechanism** (de-risked; scale-up is a licensing decision, now taken) |
| **Phase 2** — scaling & external proof | ~09-20 → 10-05 (P7 eval) | Scale along the measured slope; multi-cam+radar; closed-loop at benchmark scale; Orin/Thor TensorRT; final safety case | Not started. **Deployment export is de-risked ahead of schedule** (ONNX→TRT-FP16 path proven, per-chip precision map fixed, INT8 rejected on evidence); real Orin/Thor silicon remains the only hard blocker |

**Phase-0 exit is NOT "gates measured"** — it is: (1) open-loop beats constant-velocity AND go-straight
on **both** straight and curve strata; (2) **closed-loop** route completion with imagine-and-select;
(3) held-out ADE within a factor of the oracle ceiling. **(1) is met in-distribution. (3) is met.
(2) is no longer *blocked* — it is now measured** on a real-footage low-OOD instrument (n=40) and on a
reconstruction suite (n=12), and the honest reading is that **the flagship's deployed head does not yet
pass it**. What remains genuinely blocked is the *safety-grade* half — off-road and collision rates need
a map + reactive agents, i.e. a renderer, and every renderer we have sits at ~3.2× observation-OOD.
*Only then do more cameras/sensors/the H-stack proceed.*

## 5. Current state & latest achievements

### 5.0 — THE CURRENT STATE (2026-08-03)

> §§5.1–5.4 below are the **2026-07-25 snapshot**, kept for their provenance tables. Where they and
> this section disagree, **this section is newer**. §5.0.5 is the do-not-re-quote list.

#### 5.0.1 ⭐⭐ Closed-loop on a neural reconstruction, on the edge device — and **ADE saw nothing**

`MEASURED` — run dir **`stack/experiments/alpasim-gsplat/results/`**
(`metrics_empty.json`, `metrics_objects.json`, `contract_test.json`, `actor_map.json`); code
`stack/experiments/alpasim-gsplat/`; videos
`TanitAD Research Hub/Evaluation/Videos/alpasim-closedloop/` (4 × 18.0 s, 1800×850, each verified by
**decoding it back** and md5-matched to Thor).

**9 rollout starts × 50 ticks in a NuRec reconstruction rendered ON THE JETSON THOR**, paired over
**437 shared windows**, episode-cluster bootstrap. Paired Δ = flagship v1 − REF-C base (positive =
flagship worse), empty-road:

| family | metric | paired Δ [CI95] | sep |
|---|---|---|:--:|
| **ADE** | `ade_0_2s` | **+0.7885 [−0.8653, +2.7282]** | ❌ |
| **LONGITUDINAL** | `abs_target_speed_err_ms` | +1.1242 [−0.1008, +2.5657] | ❌ |
| **LONGITUDINAL** | `along_track_ade_m` | +0.6498 [−1.0166, +2.5903] | ❌ |
| **LATERAL** | `cross_track_abs_m` (= `dist_to_gt_traj_m`) | **+1.1705 [+0.0296, +2.2438]** | ✅ |
| **LATERAL** | `heading_err_rad` | **+0.0838 [+0.0278, +0.1750]** | ✅ |
| **LATERAL** | `curvature_err_1pm` | **+0.0050 [+0.0008, +0.0130]** | ✅ |
| **LATERAL** | `yawrate_err_rads` | **+0.0378 [+0.0201, +0.0565]** | ✅ |
| **TACTICAL** | `manoeuvre_plan_eq_logged` | +0.0709 [−0.1241, +0.2600] | ❌ |
| **STRATEGIC** | `route_corridor_departure_rate` | +0.2037 [−0.0023, +0.3982] | ❌ |

⭐ **REF-C base beats flagship v1 closed-loop and the whole separation is LATERAL. An ADE-only table
would have reported NO DIFFERENCE** on a comparison where four lateral measures separate cleanly and
in the same direction. **This is why the four-family rule is binding** — and it reproduces the
07-23 native-1080 n=12 suite on **different hardware, a different renderer and a different scene**,
so the doctrine is not an artifact of one harness.

**Three defects only the families expose:**
1. **The arms fail longitudinally in OPPOSITE directions** — flagship `target_speed_err_ms`
   **−2.0412 m/s** (too slow), REF-C **+1.3307 m/s** (too fast). **A pooled score cancels this.**
2. **The flagship does not execute what it selects** — `manoeuvre_exec_eq_plan` **0.4481**, a genuine
   5-class agreement rate (executed classes span lane_keep 83 / turn_right 30 / accelerate 33 /
   brake_stop 124 of 270). ⛔ **REF-C's 0.8741 is NOT the comparator it looks like.** REF-C's executed
   class is `lane_keep` on **270/270** windows, so the metric degenerates to *"how often was the plan
   lane_keep"* = 236/270 = **0.8741 exactly** — a **constant-predictor tie**, the same collapse as
   defect 3, not evidence REF-C executes well. **The metric cannot discriminate on a single-class
   arm; the two values are not comparable.** *(Verified against the primary artifact
   `confusion_planned_x_executed` in `metrics_empty.json`, adversarial pass 2026-08-03.)*
3. 🔴 **REF-C's 5-way manoeuvre head NEVER emits a longitudinal class** — closed-loop
   `head_class_share` lane_keep 0.627 / turn_right 0.373 / **accelerate 0 / brake_stop 0**, while
   **41.9 %** of logged windows are `brake_stop`. **The programme's top defect**, the documented
   lat+lon-mixing softmax, seen in CLOSED LOOP for the first time rather than inferred open-loop.

**Bounds that travel with the result:**
- ⛔ **Within-sim relative** — REF-C open-loop 1.5157 on these reconstructions vs 0.4728 on real
  footage (**3.21× OOD**). Orderings survive; **absolute rates do not**.
- ⚠️ **The clusters are disjoint segments of ONE clip**, not independent episodes — stated in the
  JSON so it is never mistaken for the 40-episode val bootstrap.
- ⚠️ **STRATEGIC is degenerate on a junction-free 20 s clip.** Flagship `route_head_eq_logged`
  **1.0000** is a constant-predictor tie; `route_label_valid_rate` 0.3778/0.3867, the rest
  `gray_zone`. **A junction scene is required before any strategic-accuracy claim.** TTC reported as
  **n = 0 with its reason**, not dropped.
- **Objects vs empty road is NULL for both arms** (19/20 CIs contain zero) — bounded honestly to
  *distant* traffic (0.02–0.4 % of frame at 40–45 m, ~2.8 s gap), **not** claimed as vision being
  ignored. A close-following / cut-in scene is the discriminating follow-up.
- 🔴 **Instrument property that constrains every future sim number: the renderer is a STEP FUNCTION
  of pose.** Identical pose is bit-exact, but 1e-9 → 1e-4 rad all give the same mean pixel delta with
  no growth — discrete blend-order ties among 3.1 M semi-transparent gaussians, **not** float
  precision. Decision-level cost on one identical window: **0.0000 m** in-process, **4.59 m** through
  the gRPC float32 pose round-trip, **6.65 m** under a 0.1 px camera rotation. ⇒ **All production
  numbers must come from ONE numerical path**, and **bit-identical is the wrong acceptance criterion**
  for a splat renderer.

#### 5.0.2 ⭐ The Thor planning tick is UNDER budget — measured end-to-end on real weights

`MEASURED` — artifact **`TanitAD Research Hub/Production & Optimization/Implementation/incoming/2026-08-03-thor-batch9-engine/thor_d6_tick_intent_K20.json`**;
engines at `thor:~/trt_deploy/`; runbook `TanitAD Research Hub/Production & Optimization/THOR_DEPLOYMENT_RUNBOOK.md`.

First tick measured **end-to-end** (encoder + strategic head + tactical head + 9-candidate fan +
`step_readout` decode + SE(2) + scoring) rather than composed arithmetically. **60 real held-out
windows / 12 episodes, real step-29999 weights, K = 20, budget 100 ms:**

| configuration | p50 | p95 | vs budget |
|---|---:|---:|---|
| fp32 eager, serialised fan | 764.1 | 768.4 | 764 % |
| bf16 + engine, **caller not fixed** | 372.0 | 380.1 | 372 % |
| bf16 + eager batched, no engine | 204.4 | 205.7 | 204 % |
| **bf16 + dynamic 1..9 engine, BATCHED fan** | **60.3** | **63.1** | **60 % / 63 % — PASSES** |

**6.17× from the batching fix alone** (`speedup_B_to_C` 6.169), **12.7×** over fp32 eager.
⚠️ **"Rebuild the engine at batch 9" was an INSUFFICIENT instruction** — `TacticalSelector` LOOPS
over candidates, so the rebuilt engine with the unchanged caller measures **272.8 ms, worse** than
the batch-1 engine's 265.7. The batching had to be implemented in `stack/`
(`propose_and_score(..., batch_fan=True)`). ⇒ **an optimisation stated as an artifact change must
name the CALLER shape it requires.**

⚠️ **Scope stated plainly: LATENCY + tactical-decision, NOT a four-family accuracy claim.** Two
probes confirm `TacticalSelector` has **no production caller today** (the only closed-loop driver is
heads-only, ~24 ms) ⇒ this makes the **designed** path affordable rather than speeding up what runs
now. **NOT DONE:** the four-family panel was not re-run (argued unnecessary via 1.57e-4 b9-vs-b1
numerics — *argued, not measured*).

#### 5.0.3 ⭐ The renderer blocker is gone — and the scope limit is exact

`MEASURED` — `stack/experiments/nurec-gsplat/` (+ `FINDINGS.md`, `isp_report.json`) and
`stack/experiments/alpasim-gsplat/`.

- NVIDIA's NRE renderer is amd64-only, **but it is not required**: `volume.nurec` is
  **gzip + MessagePack**, and **gsplat 1.5.3 renders it natively on aarch64 including the f-theta
  camera model** at **16–28 ms / 1920×1080 frame** with the scene GPU-resident; whole closed loop
  **0.09–0.21 s/step = 5–11 Hz**. *(The banked 224.98 ms / 4.4 FPS was a **first-call** number and is
  superseded; steady state at 3.1 M gaussians is 0.10–0.17 s/frame.)*
- Validation is by **gradient-NCC** against the scene's own shipped reference video. **PSNR and plain
  NCC are INADMISSIBLE here** — §5.0.5.
- ⭐ Also extracted from the USDZ: **`map.xodr`, `clipgt/lane.parquet`, `clipgt/obstacle.parquet`** —
  the **strategic-map material the programme has been missing**, since PhysicalAI-AV ships none
  (its card says verbatim that open maps data is not included).
- ⛔ **SCOPE LIMIT, binding on every sim number: this is AlpaSim's renderer WIRE CONTRACT satisfied by
  our renderer, driven by a TanitAD closed-loop harness. It is NOT `alpasim_runtime.simulate`.**
  MEASURED on Thor: `alpasim_grpc`, `alpasim_utils`, `alpasim_wizard` import; **`alpasim_runtime`,
  `alpasim_controller`, `alpasim_physics`, `utils_rs` do not**, and `uv` is absent ⇒ **there is NO
  AlpaSim collision / offroad / scene score** for these runs; the four TanitAD families are what is
  measured instead. `cargo` IS present ⇒ finishing the runtime is **bounded, not blocked**.

#### 5.0.4 REF-C's route pathway is LIVE — the defect is the ARCHITECTURE, not the label

`MEASURED` — run dir **`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-lan-refc-e0/`**
(REF-C-base 104.192 M, **859 windows / 39 val episodes**, **256 px square raster asserted before any
forward pass**, paired episode-cluster bootstrap n_boot 2000). Pre-registration:
`PREREG_lan_refc.md`.

- **E0 verdict RESPONSIVE, not INERT** — sweeping `nav_cmd` over the label-reachable commands
  {follow, left, right} moves the trajectory **0.2416 m**; the bit-identical-input control is
  **exactly 0.0** (tol 1e-6). ⇒ every published REF-C number held a **LIVE** input constant, not a
  dead one. Different defect, different fix.
- **The pre-registered fallback is refuted too.** Handing REF-C the **oracle** route degrades lateral
  (`cross_mae` +0.0031 [+0.0001, +0.0063] **separated worse**) and buys nothing anywhere; its own
  produced route is worse still (ADE +0.0118 [+0.0011, +0.0227]). ⇒ **do not re-score the REF-C rows
  expecting a gain.**
- **The 5-way defect reproduces open-loop at n=859** — accelerate **0/93**, brake_stop **7/78** — and
  is **INVARIANT to `nav_cmd`**, because both aux heads read the pooled feature only.
  ⛔ **LAN as specified CANNOT fix it**: `lan_emb`/`lan_dir` reach only the decoder, never the
  manoeuvre or route head. Naming that now saves the GPU-days a LAN arm would have cost.
- **A 4-way `nav_cmd` sweep is really a 3-way sweep** — `ROUTE_TO_NAV` maps the 3-class label onto
  nav {0,1,2}; index 3 is an untrained embedding row that dominates the raw sweep (1.5126 vs 0.2416)
  and would have overstated route sensitivity **7.3×**. An OOD probe mislabelled as a route response.
- ⚠️ `ep_00028.pt` was a truncated transfer ⇒ **39/40 episodes, NOT window-comparable** to the
  published 881-window rows.

#### 5.0.5 ⛔ RETRACTED — may not be re-quoted from this document or any older one

Full entries and root-cause classes in [`RETRACTION_LOG.md`](RETRACTION_LOG.md).

| retracted | what is true instead |
|---|---|
| **PSNR / plain NCC on the NuRec night clip** | A **WRONG** reference frame beats the correct one under both (PSNR 17.457 > 16.758; NCC 0.782 > 0.704) — every frame is a dark night street, so ~17 dB measures "both images are dark". ✅ **grad-NCC discriminates** (argmax = frame 0); on the corrected `wxyz` quaternion layout **0.3802** vs best wrong **0.2110**, margin **+0.1692**. The mapping is validated by **STRUCTURE, not photometry**. ⇒ *on a low-dynamic-range corpus the negative control runs FIRST and CHOOSES the metric.* |
| **The ISP / per-frame-photometry lead** | **Dead.** PPISP found (`.post_processings.0.ppisp.*`, 3594 views): exposure **exactly 0** for all 3594, colour **identical** (std == 0), vignetting max \|α\| 0.0047 ⇒ **combined 0.18 %**, because `per_frame_ppisp_enabled: false`. **The scene ships no per-frame photometry.** ⭐ The real residual is **COVERAGE, not colour** — 79–81 % of the absolute error is in pixels **no gaussian covers**; the "near-equal ~0.45 per-channel gain" was an averaging artifact (masked to covered pixels the channels spread 1.55–3.4×). ⚠️ Enabling the sky env-map made the render **worse**. ⚠️ Standing risk: the `f0` appearance-basis choice was selected **on PSNR** and therefore rests on a retracted metric. |
| **The Thor precision / quantisation table** (*"precision gate PASS, error does not compound"*) | **Measured on a RANDOMLY-INITIALISED model fed `torch.randn`** — no `torch.load` in any of the five scripts. Quantisation error is a function of the *trained* distribution; a random net has no outlier channels. 🔴 **We measured the OPPOSITE on real weights** (paper §7.10): W+A INT8 collapses the post-pool `readout_head` to cos 0.566, costing **+0.0215 m ADE@2s** — past the 0.02 m pre-registered falsifier, degradation growing 27× from 0.5 s to 2 s. ✅ **Latency survives** (weight-independent). ⚠️ The CUDA-graph "bit-exact" row is near-tautological (a *static* input replayed must reproduce itself). Also: **`thor:~/trt/predictor_fp16.plan` was itself built from random weights** and was never deployable — superseded by `thor:~/trt_deploy/`. |
| **The ego-only `sitclf` swap** — **and, since 2026-08-03 12:39, score-level image+ego fusion as well** | **The ego-only swap was REJECTED by the PI ("no ego heads") and must not be proposed again.** ⛔ **Then a THIRD PI position superseded BOTH candidates: `LABELS MAY USE EGO; INFERENCE IS VISION-ONLY`** (CLAUDE.md, binding) — verbatim *"for ground truth data of scenario classification you can use both ego and other label, for inference only vision."* **`late_fuse_scores` is therefore ALSO out**, because score-level fusion is still ego-at-inference. **The deployable arm is `head_img` (image-only).** ⚠️ **And the anomaly that started this is now itself suspect:** the situation labels are derived from **ego dynamics** (`stack/tanitad/data/situations.py`), so a classifier *given* ego at inference is partly reading **the label's own source** — the banked ranking `head_ego` 0.0697 > `head_img_ego` 0.0525 > `head_img` 0.0376 may be **LEAK MAGNITUDE, not capability**, and it makes `situations.py:19`'s *"vision adds nothing over ego state"* unfalsifiable as stated. **Flagged to verify from source (two probes, file:line), not assumed.** The `sc_train.py:143` scale bug is real but is no longer the fix. **Guardrail: "vision scores worse" is NEVER a reason to reopen ego at inference.** |
| **"REF-C's route pathway is INERT"** | **REFUTED** by §5.0.4. Every premise was individually MEASURED; the *conclusion* was an **inference** that travelled through six documents wearing its premises' evidence class. **A chain of MEASURED links does not make the conclusion MEASURED.** |
| **Every four-family ABSOLUTE RATE published before 2026-08-03** | Wrong by **5×–25×** — `four_families.py` hard-coded `DT_S = 0.1` while its inputs are the sparse 4-waypoint **0.5 s** grid. `speed_*` **/5.00**, `accel_*` **/25.00**, `yaw_rate_*` /6.48, `curvature_*` **/8.36**, `heading_*` /1.90 (the last two via the `MIN_DS` mask alone — dt-invariant and *still* wrong). Ground truth: ego speed 12.4565 m/s vs the instrument's 62.9789. ✅ **Every cross-arm comparison, rank and paired delta SURVIVES** (common factor). **FIXED + tested** (`infer_dt`, `prefer_dense`, `MIN_DS_MPS`, a `_grid` provenance block, 12 tests). |
| **`overlapping_holdout_se`** (the block once labelled *"8-split episode-disjoint jackknife"*) | Neither a jackknife nor a valid SE, **and it biases the point estimate** (−6.67 % to +11.69 %, bidirectional, 27 arms). It manufactured the programme's one "load-bearing" hierarchy seam: `ctx→tactical` +0.0439 → true **+0.0148**. |
| **`flagship4b-phase0-30k` as "the deployed v1"** | It is the **no-speed ablation control** (2.918 m). The deployed v1 is **`flagship4b-speedjerk-30k`**. |

⚠️ **Also standing:** REF-A I-JEPA's val number is **unusable** (~80 % val leakage); **no
learning-curve exponent may be quoted bare** (window + R² + n, and nothing below R² 0.80).

#### 5.0.6 Fleet (2026-08-03)

| Host | Run | State |
|---|---|---|
| `tanitad-thor` | Jetson AGX Thor (aarch64, Blackwell sm_110) — edge inference, four-family evals, **and the renderer** | 🟢 the programme's only non-pod compute; two venvs **never mixed** (`tanitad-edge` vs `tanitad-train`) |
| `tanitad-new` | **v5f** (429-token, 176×624) | 🟢 training — ⛔ **DO NOT TOUCH** |
| `tanitad-pod4` | **`flagship-v1arch-v2bal-30k`** | 🟢 training — ⛔ **DO NOT TOUCH** |
| `tanitad-pod2` | — | ⚪ evacuated, idle; rescue payload at `_pod_backup/pod2-2026-08-03/` |

`INHERITED` (from `LOOP_STATE.md`; not re-verified in this refresh). ⚠️ **flagship v1 + REF-C are
256 px SQUARE; v5f is 429-token (176×624) — NEVER score an arm off its own raster** (that error
produced a published `speed_mae` of 3.06 m/s for an arm whose registry ADE@2s is 0.4728).

---

### 5.1 The open-loop bake-off (2026-07-25 snapshot) — settled, and the top is a three-way tie
All numbers: TanitEval, physicalai val (`physicalai-val-0c5f7dac3b11`), **881 windows / 40 episodes**.
⚠️ The `±` column is the **deprecated** `overlapping_holdout_se`. Updated 2026-07-26: it is
**1.107–3.100× too narrow, median 1.499×** over 27 arms (the old "1.28–2.06×" was under-sampled at 10),
**and its `mean` is a split-mean that shifts the point estimate −6.67 % to +11.69 %, bidirectionally.**
Read the `full-set / bootstrap` column; the `±` column is retained only for traceability. Provenance:
[`MODEL_REGISTRY.md §6`](MODEL_REGISTRY.md) (re-emitted 2026-07-26 — **the ranking moves in 10 of 27
positions**, and the two REF-C rows below are swapped relative to the legacy order for that reason).

| Arm | Step | Params | full-set mean · [episode-cluster bootstrap] | `legacy_split_mean ±` (DEPRECATED) | Beats CV (full-set 0.8377)? |
|---|---:|---:|---|---:|:--:|
| **1= Flagship v1 (4-brain WM, trained ViT) — DEPLOYED** | 29 999 | 263.4 M | **0.4271** · [0.3675, 0.4871] | *0.4522 ± 0.0312* | ✅ |
| **1= REF-C-XL** (anchored diffusion) | 29 999 | 251.9 M | **0.4714** · [0.3896, 0.5556] | *0.4577 ± 0.0572* | ✅ |
| **1= REF-C-base** (anchored diffusion) | 29 999 | **104.2 M** | **0.4728** · [0.3835, 0.5699] | *0.4523 ± 0.0497* | ✅ |
| *best-of-3 kinematic floor 0.5005 · CTRV oracle 0.523 · no-vision ego ceiling 0.5735 — **all full-set means by construction**, so the legacy column never applied to them* | | | | | |
| REF-C-small (separated 3rd rung) | 29 999 | 54.7 M | 0.5261 · [0.4295, 0.6262] | *0.5007 ± 0.0671* | ✅ |
| **REF-B v2** (BC + time-anchored decoder) | 29 999 | 271.6 M | 0.5913 · [0.4766, 0.7131] | *0.5921 ± 0.0685* | ✅ |
| Flagship v1, 19 k relay | 19 000 | 263.4 M | 0.6152 · [0.5422, 0.6951] | *0.6277 ± 0.0551* | ✅ |
| REF-B speed | 10 000 | 262.8 M | 0.8372 · [0.6753, 1.0218] | *0.8255 ± 0.0992* | ⚠️ **TIE** (was ✗ — flipped under the correction, paired test not separated) |
| **Constant velocity (the floor)** | — | 0 | **0.8377** | *0.8248* | — |
| P2 CEM planner over frozen v1 | n/a | 0 trained | 🟥 not recomputable (no raw JSON, no windows dump) | *0.893 ± 0.114* | ✗ |
| Flagship **v3enc** 10 k (RESTART) | 10 000 | 272.9 M | 1.9654 · [1.6556, 2.2859] | *2.1072 ± 0.2020* | ✗ |
| REF-A DINOv2 4B (frozen encoder) | 29 999 | — | 2.1675 · [1.9081, 2.4212] | *2.1322 ± 0.1821* | ✗ |
| Flagship **no-speed** (ablation control) | ~22 000 | 263.4 M | 3.0175 · [2.5450, 3.5444] | *2.9176 ± 0.3558* | ✗ |
| REF-A dyn-in 4B (frozen, every remedy applied) | 29 999 | — | 3.0471 · [2.4984, 3.6878] | *2.9196 ± 0.3937* | ✗ |
| Flagship v1 **tactical head** (not the rollout) | 29 999 | — | 🟥 no windows dump | *3.38* | ✗ |
| Flagship v2 (killed) | 6 000 | 272.9 M | 5.9396 · [4.3273, 7.6249] | *6.179 ± 1.2845* | ✗ |

**Three readings.** (1) Every **trained-encoder** arm sits above CV; both **frozen-encoder** arms sit far
below — H4 in one table. (2) **Rank 1 is a genuine three-way tie no paired test can order** (base − XL
Δ +0.0013 [−0.0281, +0.0316]) — held by a 263 M world model, a 252 M diffusion arm **and a 104 M
diffusion arm**, so *scale bought nothing above 104 M on this corpus*. (3) The flagship's own supervised
**tactical head is worse than CV** (3.38 m) while the same model's operative rollout is **0.4271 m**
(legacy split-mean 0.452) — the head is a lossy readout of a good world model. ⚠️ Both sides of that
comparison are still legacy statistics (no windows dump for the head); the ratio is far too large for the
≤ 11.7 % bias to touch, but neither scalar is decision-grade on its own.

### 5.2 What moved the program in the round to 2026-07-25

**(a) ⭐ The crux: a planner CAN be coupled to the world model — the failures were a warm-start artifact.**
The v3 design shipped as **flagship-v4**: v1's trunk + a strategic planner in a 128-d subspace + tactical
and operative **anchored-diffusion planners** (256 anchors), λ_plan curriculum, **≈247.9 M trainable —
~30 M smaller than v1** (three planners cost less than three supervised heads). The attributability
instrument is the **plan-free WM-integrity canary** (the operative rollout with the planner removed;
v1 = 0.452).

| arm | one lever | held-out ADE@2s | WM canary | read |
|---|---|---|---|---|
| v4 | hot trunk lr 3e-4 | killed ~3.5 k, never gated | 0.452 → ~1.3 *(in-loop)* | the trunk LR alone degrades the WM |
| **v4.1** | lr_trunk 3e-5 | **0.8522 [0.7468, 0.9800]** vs 0.60 bar 🟥 | **0.4599 PASS** | WM healthy, **planner starved**; loss is longitudinal (speed vs CV −0.366 sep) while path geometry *beats* CV (+0.115 sep) |
| v4.2 | canary floor 0.25 | 0.9869 [0.880, 1.109] @4 k | **0.7222 FAIL** | protecting the planner costs the WM |
| v4.2b | canary floor 0.15 | not gated | 0.697 @4 k | floor-tuning exhausted |

⭐ **The ~0-GPU decision.** Before buying gradient surgery we measured the seam's geometry (n=512 windows,
clean val, the `states` seam): **cos(g_wm, g_plan) = +0.0043** (sd 0.064), 47.9 % of windows negative,
PCGrad `frac_removed` = **0.0224**. Surgery would strip ~2 % of the planner gradient — a **no-op** — and
the floor had already cut g_plan to 15 % with the canary still degrading. **Neither the direction nor the
magnitude of the planner gradient is the cause.** That selected **from-scratch co-evolution** (v1's own
recipe) for ~0 GPU-hours instead of ~1.3 A40-days on a refuted lever.

🟢 **`flagship-v4-fromscratch-30k` (LIVE, pod2).** Canary baseline recalibrated to **15.674** at random
init — the warm-start ≤0.55 bar is meaningless there, so the pre-registered read is the **descent
trajectory**. At **full coupling (λ_plan = 1.0)** the canary **descends** 15.674 → 2.59@7 k →
**1.371@9 k** where every warm-start arm rose; held-out ADE@2s 0.531@9 k → **0.4788@11.5 k**
(miss@2 m 0.169, oracle-in-fan 0.242). **10 k gate = CONTINUE**, restarts 0.
⚠️ **These are the trainer's in-loop evaluation on the clean split, not `eval_flagship_v4.py`** — the
formal 8-metric gate is **deferred** behind an HF-relay quota block. Under C1 a trainer number is not an
eval number (v1.6's in-loop read ~10 % optimistic). **Quote the trend, not the level**; the run is at
~40 % of 30 k.

**(b) D1 — frozen WM + learned planner: a fallback, and a re-localization of H4.** A **3.77 M** planner
trained *only* by backpropagating ADE **through** the frozen v1 reaches **0.5989 [0.374, 0.854]** —
paired-beats CV (−0.2474 [−0.505, −0.034]), hold-v0, and action-BC (−0.4012 [−0.717, −0.128]) — and is
**not separated** from the WM's own oracle-action ceiling **0.4045** (+0.1944 [−0.045, +0.448]). Decoding
waypoints *statically* off the same frozen latent gives **3.649 m** — the REF-A band. **So the frozen
ceiling is a ceiling of static decode, not of freezing.** Capacity is not the lever (11× scaling: 0.599 →
0.601 → 0.599, none separated; query-decoder variants overfit to 0.82–0.86) — the residual is
**aleatoric**. 🔴 **The "CEM search 0.132 = 4.5× planner headroom" claim is RETRACTED (C6): it peeks at
the expert's realized future.** The deployable version (learned value, no GT) scores **1.0162**,
+0.4173 [+0.237, +0.605] separated **worse** than feedforward. Verdict: a **~0.60 m degradation-free
fallback** (canary untouched by construction), not a contender.

**(c) D2 — the closed-loop recovery lever closes honestly, and leaves two measurement rules.** A fair
lane-tolerance metric (`band_ade2d(1.0)`) showed the apparent ADE cost was largely a **knife-edge-L2
artifact** (vanishes CI∋0 for 3/4 configs, −74 % for the fourth). But the departure *benefit* **reverses
at full power**: n=12 held-out **+0.0089** → n=40 2-fold cross-fit **−0.0302 [−0.0595, −0.0088]** — the
fine-tune departs **3.3× more**. 🔴 **Retracts the "halves departures + generalizes" durable-positive
(C5).** Durable: the machinery, and **REF-C's encoder is safely fine-tunable** (feat_cos 0.9658 at a
material move, canary holds — *not* the v4 WM hazard).

**(d) ⭐ Closed-loop is now measurable at low OOD — and the flagship's head loses.** A new
**real-footage log-replay** instrument warps recorded frames to the on-policy ego deviation: measured
on-policy observation-OOD **1.02–1.20×** (longitudinal 1.018/1.004 ≈ OOD-free) against a photoreal
reconstruction's flat **3.75×**; the flagship's prediction is statistically flat out to **2.0 m** lateral
and separates on yaw only at **3°**. At **n = 40 eps / 881 windows**, paired, identical windows:

| | flagship v1 (deployed head) | REF-C base | paired Δ | sep |
|---|---|---|---|---|
| closed-loop ADE@2s | **1.488** [1.329, 1.647] | **0.564** [0.452, 0.676] | +0.924 [+0.781, +1.065] | ✅ |
| corridor departure @1.75 m | 0.0318 [0.0152, 0.0531] | 0.0134 [0.0059, 0.0223] | +0.0184 [+0.0077, +0.0328] | ✅ |
| peak cross-track error | 0.764 m | 0.442 m | +0.321 [+0.193, +0.495] | ✅ |

**Triple-confirmed** across independent instruments: n=1 (retracted as a lucky scene) → n=12 paired
reconstruction suite (pass 8/12 vs 2/12, Δ −0.430 [−0.646, −0.215], sign-test p=0.008, **collisions tied
1–1**) → this n=40 real-footage run. **The decomposition is the real prize:** in longitudinal scenes both
arms keep the lane nearly perfectly (0.4 % vs 0.04 % departure) yet the flagship's ADE is **4×** — its
deficit is **longitudinal, not lane-keeping**, matching the registered 89 %-along-track signature. In
junctions it departs **~2.3× more** (peak XTE 2.372 vs 1.458 m): a **high-deviation planner** whose
failure mode is off-road, not collision. ⚠️ **Map-free / agent-free ⇒ lane-keeping drift, NOT off-road or
collision.** The low-OOD-vs-safety-metric gap is ~fundamental without a lower-OOD reactive renderer.

**(e) ⭐ The data thesis (H7) gets its first evidence.** Direct pseudo-label accuracy is honestly modest
(speed R² 0.62–0.66 cross-domain, longitudinal-traj 0.60, **yaw ≈ 0** ⚠️*STALE-PENDING — that cell is
comma2k19 with `heading_repair` OFF, a broken label, not a transfer result (C29); on repaired labels the
same head reads comma yaw **+0.3308**, retrained **+0.679**; see
`…/incoming/2026-07-27-comma-yaw-reissue/`. 🔴 **AMENDED 2026-07-27 (C43): the `+0.3308` is
WITHDRAWN** — 2 of its 22 comma val episodes are, BY CONTENT, in that head's own comma TRAINING set;
without them it reads **−0.746**. `+0.679` stands (no leak) at **+0.3038 [+0.054, +0.479]** on the 20
content-clean episodes. ⇒ comma yaw is **testable, and this head does not do it**; the `yaw ≈ 0` cell
stays STALE-PENDING either way. See `…/incoming/2026-07-27-anchor-settlement/`*, accel dropped) — but what matters
downstream is structure, not precision: pseudo-label WM pretraining captures **~96 %** of real-label
pretraining value (8 seeds, 2 proxy domains) and **109 % speed / 107 % traj / 71 % yaw** on the actual
parity target (4 seeds, all CI-separated from the floor). The **80-clip Creative-Commons YouTube pilot**
lifts downstream parity-val **speed R² −0.520 → +0.563** (3 seeds, clip-cluster CI excludes 0 on *every*
seed), yaw 0.55 → 0.75, ADE **halved** 12.82 → 6.31 m ⇒ **≈92 % of the real-label ceiling — the YouTube
domain transfers.** ⚠️ **DIRECTIONAL** (80 clips, 3 seeds, unknown intrinsics; the fraction-of-ceiling is
the substantive claim, not the R² delta). Counterweight: **Branch B FAILED** — from-scratch GAIA-2
camera-conditioning gives cross-rig speed R² **−0.667** vs frozen v1's **+0.657** (paired CI excludes 0
on 3/4 arms). The cheap substrate (frozen v1 + a multi-domain IDM head) beat the expensive one.

**(f) The corpus, exactly — and a balanced 50 h successor.** MEASURED: the parity set is **13.13 h /
472,627 frames / 2,376 clips × 19.9 s / 406,099 windows**, so **30 k steps = 4.73 epochs**. Mix:
lane_keep **59.6 %** · accel 13.2 · brake_stop 12.9 · turn_right 7.4 · turn_left 6.9; **only 42.6 % of
clips contain ANY turn**; **semantic scenarios (lights, roundabouts, merges) 0 %-labeled.** A **v2 50 h
corpus** was designed and built *inside the same source* by **selection-balancing** (not synthetic
perturbation): 9,000 clips, turns **14.25 → 28.0 %**, junction-clip presence **37.7 → 61.3 %**, key
`physicalai-v2bal-4b7eeeac222d`, stored JPEG-compressed **982 GB → ~25 GB** with frames **bit-identical**
to the parity decode path. **Breaks parity by design** — the running arm finishes on the 13 h set.
⚠️ Kinematic selection cannot buy semantic scenarios; those need the VLM labeling track.

**(g) Standing findings that still carry the program** — the **speed fix** (v0 as a 3rd action channel:
REF-A fwd-ADE 3.73 → 0.83, no-speed 2.918 vs speed 0.452 causally, +2.21 m [2.04, 2.39]); **H4 closed
negative** on a monotone 5 k→30 k curve (3.755 → 2.920, best is last — a capability ceiling, not
overfitting); and **P2**, the training-free CEM planner over frozen v1 that beats the tactical head by
+2.257 ± 0.329 m open-loop and drifts 38 % less closed-loop.

### 5.3 What was running on 2026-07-25 — ⛔ **SUPERSEDED by §5.0.6, do not read as current**
| Pod | Run | State |
|---|---|---|
| `tanitad-pod2` | ⭐ **flagship-v4 from-scratch → 30 k** | 🟢 ~step 11.9 k, λ_plan 1.0, canary in-band, restarts 0; ~2 days to 30 k, auto-continues. ⚠️ 3.2 GB ckpt on **pod2 disk only** — HF backup 403-blocked (mid-checkpoint-loss risk). **NEVER eval here** |
| `tanitad-pod` (pod1) | **YouTube-IDM non-CC harvest/label** (scale-up) | 🟢 Sayed committed to the licensing 07-25; v2-corpus shard **DONE** (4953/4953) |
| `tanitad-pod3` | **v2-corpus build** → then IDM scale-up pretrain | 🟢 finishing the second shard; consolidate by clip-id union, then **QA** |
| `tanitad-eval` | **GeoCalib** (per-video intrinsics — removes the pilot's fixed-HFOV approximation) | 🟢 |
**0 idle GPUs.** 🟡 **`Sayood/` HF storage is FULL (403)** — blocks the flagship ckpt backup, the formal
v4 gate relay, and a REF-C arm. Minimal safe unblock ≈13 GB (superseded v4.1/v4.2/v4.2b + the val-leaked
refa-ijepa) — **Sayed's click; irreversible deletes are not run autonomously.**

### 5.4 Stack & tooling maturity (2026-07-25)
Train pipeline ✅ · gate runner ✅ (`run_gate.py`, `estimator` field now mandatory) ·
**TanitEval ✅ PRODUCTIONIZED + IN-REPO** — one canonical `runner.py` CLI (20 subcommands, closed-loop
wired in), **episode-cluster bootstrap as the default CI** with `overlapping_holdout_se` deprecated
read-only, **the 78 %-leaking val split `physicalai-val-f1b378f295ae` now HARD-REFUSED in code**,
off-pod reproducible tests (153 passing, was 0) · **TanitResim ✅ PRODUCTIONIZED** — decoded-intent HUD
(maneuver + route + ADE + v) per the standing viz standard, BEV-only fallback for uncalibrated corpora,
one-command demo; a real bug fixed (SPA nav labels were mis-indexed vs canonical `NAV_COMMANDS`) ·
**beyond-ADE suite** LAL/TMS/OKRI/CNCE/LOPS + **new SC-14 traffic-light scenario & TLC metric** ✅ built,
first REAL numbers on the deployed architecture ✅, closed-loop half **renderer-gated** · **low-OOD
closed-loop harness** ✅ new (corridor departure, band-ADE, on-policy OOD envelope) · data recipe ✅ ·
TanitDataSet v1 lake ✅ · **v2 compressed-cache loader** ✅ (byte-identical windows; `--v2-cache` flag) ·
deployment export ✅ (ONNX→TRT-FP16 proven, INT8 rejected) · CARLA ✅-narrow · AlpaSim ⚠️ usable but
**~3.2× reconstruction-OOD**. 🔴 **Correction to the previous refresh: TanitEval and the lake modules
were NEVER stranded** — that claim came from a truncated mtime-sorted file listing (RETRACTION_LOG C8);
`git ls-files` shows main has every module and is the newest copy.

## 6. The agent ecosystem (the research flywheel)
Seven disciplines, each a folder + a **weekly post-doc-grade agent** (Mon→Fri), doing theory +
implementation, each with a knowledge base, BACKLOG, and ≥1 measured experiment per run:

| Agent (day) | Owns | Recent output |
|---|---|---|
| **Tools & DevEnv** (Mon) | sim, replay, CI, compute | **TanitResim productionized** (decoded-intent HUD, BEV fallback, nav-label bugfix); OOD corpus provisioning; the renderer question researched to a decision |
| **Data Engineering** (Tue) | datasets, loaders, training flow | **exact corpus profile** (13.13 h / 4.73 epochs / the 0 %-semantic gap); the **balanced 50 h v2 corpus** designed + built + a byte-identical compressed loader; **YouTube-IDM pipeline** (harvest → privacy-blur → pseudo-label → downstream lift) |
| **Architecture & Inference** (Wed) | the stack, efficiency | **v4 planner architecture + the cosine pre-probe that redirected it**; **D1 frozen-WM planner** (+ the value-model crux); Branch-B encoder refutation; Orin/Thor export + the **INT8 rejection** |
| **Benchmarks & Eval** (Thu) | metrics, gates, leaderboard, regulation | **TanitEval productionized** (canonical CLI, cluster-bootstrap default, leak split refused in code); **low-OOD closed-loop instrument + the n=40 comparison**; **SC-14 + TLC**; first real beyond-ADE numbers; the v4 eval harness + gate emitters |
| **Opponent Analyzer** (Fri) | opponent intel, weakness catalog, scenario DB | weekly competitor sweeps → scenarios (Stationary-Lead SC-13, first-responder W-09, **SC-14 red-light running** now oracle-tested) |
| **Project Steering** (Fri) | plans, reports, resource control | 3×/day program reports, **`RETRACTION_LOG.md`** (the self-learning mechanism), the model registry, **this document + the living paper** |
| **Production & Optimization** | ONNX/TRT/quant, latency, compliance | the corrected **two-tick** latency doctrine; composed tick 100.29 → **18.75 ms**; per-layer FP16-vs-INT8 benchmark |

Cross-cutting: the living paper (`Paper/TANITAD_PAPER.md` — now **v0.8**, §7.12 = the closed-loop
result in §5.0.1), `LEADERBOARD.md`,
`SCENARIO_DATABASE.md` (SC-01…SC-14), `GATE_PROTOCOL.md`, `RETRACTION_LOG.md`.

**Honest agent-health note — the standing risk has changed shape.** Git hygiene is **materially better**:
on 2026-07-25, 924 staged files landed in three clean commits (stack · taniteval · research+steering),
and the previously-stranded TanitEval harness is in-repo and productionized. Two cautions replace it.
(1) **Checkpoints, not code, are now the single-disk risk** — the live from-scratch flagship's 3.2 GB
checkpoint exists only on pod2 because the HF account is over quota. (2) **Premature certainty is the
recurring failure mode, not sloppiness**: `RETRACTION_LOG.md` records four "this direction is closed"
claims in one session that a *zero-cost* follow-up reopened, plus three n=1/n=12 headlines that reversed
at power. The countermeasure is now doctrine — evidence class on every claim, two probes before an
absence, the cheapest metric-or-power check *before* declaring closure.

## 7. Honest position (P8)

> **Amended 2026-08-03.** The five bullets below are new; the 07-25 body that follows them stands
> except where §5.0 supersedes it.

- ⭐ **Newly settled — the four-family doctrine paid for itself.** On the strongest closed-loop test
  the programme has run (§5.0.1), **ADE reported nothing** while four lateral measures separated
  cleanly and in the same direction. The rule is no longer a methodological preference; it is the
  only reason that comparison has a verdict.
- ⭐ **Newly settled — the edge device is not the constraint.** The full designed planning tick is
  **60.3 ms p50 / 63.1 ms p95 on Jetson Thor with real trained weights**, against a 100 ms budget
  (§5.0.2), and Thor also **renders** the reconstruction (§5.0.3). ⚠️ It is a **latency** result:
  `TacticalSelector` has no production caller yet.
- 🔴 **Newly localised — the top defect is REF-C's 5-way manoeuvre head**, which **never emits a
  longitudinal class** while 41.9 % of logged windows are `brake_stop`; reproduced closed-loop and
  open-loop (n=859), **invariant to `nav_cmd`**, and **not fixable by LAN as specified** (§5.0.4).
  Factorising lat × lon is the work item.
- **Newly refuted — "REF-C's route pathway is inert."** It is **RESPONSIVE** (0.2416 m, control 0.0),
  and supplying the **oracle** route makes lateral *worse*. **The architecture is the defect, not the
  label** — so the cheap eval-side fix that was pre-registered as the remedy is dead.
- ⚠️ **Newly bounded — the safety half of closed-loop is still missing, for a NEW reason.** We have a
  renderer; we do **not** have `alpasim_runtime`, so there is **no collision / offroad / scene
  score**. That is bounded work (`cargo` present), not a blocker — but nothing safety-grade may be
  claimed until it lands.

  > ⚠️ **RE-CONFIRMED-WITH-A-CORRECTION 2026-08-16 — the bullet is TRUE ONLY FOR THOR, and as written
  > it is a C2 single-host absence-claim.** Probed two hosts, not one.
  > **STILL TRUE on Thor (aarch64):** `alpasim_runtime`, `alpasim_controller`, `alpasim_physics` and
  > `utils_rs` do not import there and `uv` is absent — §5.0.3 above, which *is* correctly
  > host-stamped. The Thor closed-loop panels therefore genuinely carry no AlpaSim safety score.
  > ⏹ **BUT "we do not have `alpasim_runtime`" is FALSE program-wide, and AlpaSim safety scores
  > ALREADY EXIST in this repo (MEASURED).** On the **x86 A40 eval pod**, 2026-07-22, all of
  > `alpasim_runtime`, `alpasim_controller`, `alpasim_physics` imported and a full bare topology ran
  > (renderer :6011 · physics :6006 · controller-MPC :6007 · driver :6789 · runtime) —
  > `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-22-alpasim-closedloop-evalpod/RUN_RECIPE.md`.
  > The banked result summaries in that same directory carry **`scene_score_enabled: true`** and real
  > safety fields: `M2_results-summary.json` → `collision_any 0.0`, `collision_at_fault 0.0`,
  > `offroad 0.0`, `offroad_or_collision 0.0`, `min_distance_to_obstacle_m 1.4279`,
  > `img_is_black 0.0` (real frames), `dist_traveled_m 39.17` of `gt_dist_traveled_m 73.77`, scene
  > `score_criteria = {collision_at_fault == 0, offroad == 0, progress_score …}`; likewise
  > `REFC_base_results-summary.json` and `Flagship_v1_results-summary.json`.
  > ⇒ **The honest statement is: the safety half is missing ON THE CURRENT FLEET** (Thor + dev box)
  > **because the host that had `alpasim_runtime` — the eval pod — is terminated**, not because the
  > program never had it. ⚠️ Those banked scores are **n=1 scene, DIRECTIONAL only**; they do not
  > license a safety-grade claim, which is the bullet's actual point and it stands.
  > Swept by the 2026-08-16 stale-blocker sweep.
- ⛔ **New binding doctrine (PI, 2026-08-03): LABELS MAY USE EGO; INFERENCE IS VISION-ONLY.** Label
  derivation may use ego state, other agents, maps, future poses — anything, offline. **Inference may
  use vision only.** It supersedes both sitclf candidates (ego-only swap *and* score-level fusion)
  and generalises: **for ANY head, ask whether its inference inputs include something the label was
  derived from.** If they do, the score measures **leak magnitude, not capability** — the same family
  as the C6 confound and the REF-A I-JEPA val leak. Full text in `CLAUDE.md`.

- **Proven:** the 4-brain latent world model **beats every trivial floor open-loop** (0.452 m vs
  best-of-3 0.5005, CTRV 0.523, no-vision ego ceiling 0.5735, CV 0.825), and it does so *causally* —
  ablating the scene inverts a +0.796 m oracle-beating margin to −0.529 m (vision effect +1.325 m,
  CI-separated). Upcoming-curvature decodes from the pooled latent at R² 0.254 vs 0.031 ego-only.
  **The world model is real** — and a *frozen* copy of it now behaves as a good differentiable
  simulator: a 3.77 M planner driving it lands within bootstrap noise of feeding it perfect actions.
- **Settled:** **H4 — the frozen encoder has a capability ceiling** (monotone improvement to 30 k, still
  2.92 m). **Sharpened this round:** the ceiling belongs to *static decode off the frozen latent*
  (3.65 m), not to freezing — the same frozen latent supports 0.599 m *through its dynamics*.
- **Settled:** **supervised heads are a lossy readout of a good world model** — a training-free planner
  beats them 72 % open-loop and 38 % closed-loop, and an anchored-diffusion reference arm now beats the
  flagship's head **closed-loop at n=40 on a low-OOD instrument** (0.564 vs 1.488).
- **Newly settled (the crux):** **planner–WM coupling failure was a warm-start artifact.** The seam is
  near-orthogonal (cos +0.0043) so no projection surgery can help; co-evolving from random init
  reproduces v1's behaviour with the canary *descending* under full coupling. ⚠️ in-flight, in-loop
  evidence, formal gate deferred — **the coupling question is answered; the level is not.**
- **Newly settled (negatives worth as much):** the frozen-WM route is a **~0.60 m fallback, not a
  contender** (aleatoric wall; the 4.5×-headroom claim was hindsight-privileged, retracted); the
  closed-loop **recovery-augmentation lever is not promotable** (benefit reverses at n=40); **Branch B
  camera-conditioning is refuted** (−0.667 vs frozen v1's +0.657); **INT8 is rejected** for deployment.
- **Binding constraints, in order:**
  1. **Longitudinal control.** Now the most-triangulated weakness in the program: the only above-floor
     open-loop stratum (1.785× floor, 89 % along-track, +0.66 m/s over-prediction at speed); the failure
     axis of *every* v4 planner arm; and — measured on-policy, n=40, through an independent instrument —
     **the whole of the flagship's closed-loop deficit, while lane-keeping is nearly perfect.**
  2. **Closed-loop competence.** 0.452 m open-loop → 1.488 m on the low-OOD instrument; the deployed head
     is a **high-deviation planner** (junction peak XTE 2.372 m, ~2.3× REF-C's departure rate).
     **Open-loop does not predict closed-loop.**
  3. **Generalization.** In-distribution 0.427 vs floor 0.523 ✅; comma2k19 0.849 vs floor 0.372 ✗
     (17.5 % win-rate); path feasibility collapses 97.8 % → 62.8 % on OOD sharp curvature.
  4. **The safety-metric instrument gap.** Off-road and collision rates need a **map + reactive agents**
     ⇒ a renderer; every renderer we have is ~**3.2× observation-OOD**; low OOD needs real footage, which
     has no agents. **Resolving both at once needs a lower-OOD reactive renderer** — this is a build, and
     it also gates the closed-loop half of TLC/LAL/OKRI/LOPS.
  5. **Data.** 13.13 h, 4.73 epochs, **42.6 % of clips with no turn, 0 % semantic scenarios.** The v2
     corpus fixes the kinematic half; the semantic half needs the VLM track.
- **Top risks:** ✅ *the "checkpoint on a single pod disk with HF backup 403-blocked" risk is
  CLEARED* — everything has a home as of 2026-08-03 (`Sayood/tanitad-archive-pod2-2026-08`, 11.76 GB,
  verified from HF's side; parity survived the round trip on both corpus sha256s). Replaced by:
  **provider-console state no probe can see** (a "critical error on this machine" banner plus
  scheduled maintenance 2026-08-06→08 that a live run will not survive in place); **cgroup limits
  read at the wrong scope** (`free` shows the host, not the 50 GB container limit — the same trap
  `df` sets for disk); **premature certainty** (four closure claims reopened by zero-cost checks in
  one session — the countermeasure is doctrine, not vigilance); the **v4 gate cannot render a
  complete formal verdict**
  (3 of 8 kill secondaries still have no emitter); open-loop ⊥ closed-loop; PhysicalAI-AV license
  firewall (never in public claims); **YouTube-IDM scale-up now carries a licensing/privacy obligation**
  (pointers + pseudo-labels only, face/plate blur pre-downscale, never raw bytes).

## 8. The critical path from here

### 8.0 As of 2026-08-03 — this supersedes 8.1–8.6 where they conflict

1. **Factorise REF-C's manoeuvre head into lateral × longitudinal.** The single most localised defect
   in the programme (§5.0.4), now observed in **both** loops, invariant to the route input, and
   **provably not addressed by LAN**. Everything else in the tactical family is downstream of it.
2. **Make the STRATEGIC family scoreable.** It is currently degenerate — a junction-free 20 s clip
   ties any constant predictor, and `route_label_valid_rate` is ~0.38. **A junction scene is a
   precondition for any strategic claim**, and the hierarchy is the programme's thesis. The
   USDZ-extracted `map.xodr` / `lane.parquet` (§5.0.3) is the material for it.
3. **Finish `alpasim_runtime` on Thor** to convert the renderer wire contract into a **safety-grade**
   closed-loop score (collision / offroad). Bounded — `cargo` is present.
4. **Re-scope `sitclf` to `head_img` (image-only) and establish label provenance from source.**
   ⛔ *Superseded 2026-08-03 12:39:* this item previously read "wire `late_fuse_scores`" — that is
   **out**, because score-level image+ego fusion is still ego-at-inference. The binding rule is
   **labels may use ego; inference is VISION-ONLY**. First deliverable is **label provenance from
   source (two probes, file:line)**, because the banked ranking may be measuring a leak.
5. **Wire distance-keeping into LONGITUDINAL.** The reader exists and is not wired; a family with a
   built-but-unwired instrument is a **work item, not a pass**.
6. **Re-quote nothing from §5.0.5** — including in the paper, the registry and the reports. Several
   of those numbers are still live in older documents.

### 8.1–8.6 (2026-07-25)
1. **Finish the co-evolved flagship to 30 k and gate it properly.** Two prerequisites, both instrument
   work: run the **canonical `eval_flagship_v4.py`** rather than the trainer's in-loop val (C1), and
   build emitters for the **3 kill secondaries that have none** — a gate that cannot complete is an
   instrument failure, not a model result. Acceptance = the **OOD panel**, not the in-distribution one we
   already pass. *(Unblocked by the HF cleanup, item 6.)*
2. **Attack longitudinal control as one problem, not three.** It is the same mechanism in the open-loop
   stratum, the v4 planner arms, and the n=40 closed-loop gap. The instruments to gate it already exist
   (`pathspeed.py`, the driving panel's speed-MAE-vs-hold-v0 test, corridor + band-ADE).
3. **Build the lower-OOD reactive-agent instrument.** The single dependency for a safety-grade closed-loop
   number, for D5/D6, and for the renderer-gated half of the beyond-ADE suite. This is the largest
   remaining *build* on the board and the cheap experiments around it are exhausted.
4. **Land the v2 corpus** (consolidate the two shards by clip-id union → QA the balanced distribution and
   cache integrity → launch the next generation on it via the `--v2-cache` wrapper).
5. **Scale YouTube-IDM from pilot to decision-grade** (~300+ clips, 4+ seeds, GeoCalib per-video
   intrinsics replacing the fixed-HFOV approximation) — this is the direct path to the **C2
   data-efficiency slope**, the one headline claim still entirely unmeasured.
6. **HF-storage cleanup (Sayed's click, ~13 GB).** Small, but it currently blocks a checkpoint backup, a
   formal gate, and a benchmark arm simultaneously.

**Bottom line:** Phase 0 has a 4-brain world model that **clears the open-loop bar against honest
floors**, two clean publishable negatives (frozen encoder; camera-conditioning), and — new this round —
an **answer to its own crux question**: a planner *can* be coupled to the world model, provided the two
co-evolve rather than one being grafted onto the other. What it has not got is closed-loop competence:
measured properly for the first time, at low observation-OOD and at n=40, the deployed head **loses to a
104 M reference arm**, and the deficit is **longitudinal**. That gap — plus the renderer that would let
us measure safety rather than drift — is the honest distance between "the world model works" and "the
edge is proven."

> **Amended 2026-08-03.** Two of the three sentences above have moved. (i) The REF-C-beats-flagship
> result now **replicates closed-loop on a NuRec reconstruction rendered on the Jetson Thor**, on
> different hardware and a different renderer — and there **the separation is entirely LATERAL while
> ADE sees nothing** (§5.0.1), which is a sharper and more uncomfortable finding than "the deficit is
> longitudinal": *both* are true of different arms and different loops, which is exactly what a
> pooled score would have hidden. (ii) **The renderer is no longer the blocker** — we render NuRec
> natively on aarch64 (§5.0.3). What remains is **the simulator runtime**, so the honest distance is
> now: *a factorised manoeuvre head, a junction scene, and a collision score.*

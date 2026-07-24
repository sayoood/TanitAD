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
> **Last refreshed:** 2026-07-25 · Phase 0 (~day 21 / 42) · **Next refresh:** the coupled-flagship 30 k
> verdict, or Phase-0 exit.
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
| Claim | Goal | Proof artifact | Where it stands (2026-07-25) |
|---|---|---|---|
| **C1 — It drives** | Goal 1 | Closed-loop success + latency/FLOPs ledger (Orin/Thor envelope) | 🔶 Open-loop **cleared** (0.452 m, below every trivial floor). Closed-loop **not** — and now measured three ways: 1.685 m self-referential in-house, **1.488 m [1.329, 1.647] on the new n=40 real-footage low-OOD instrument where REF-C base scores 0.564**, 2/12 pass on the reconstruction suite. **The deficit is measured LONGITUDINAL, not lane-keeping.** Latency side is done: composed planning tick **18.75 ms p50**, 10 Hz at p99 with 5.3× headroom, FP16 |
| **C2 — It needs magnitudes less data** | Goal 2 (prio) | Data-efficiency slope vs supervised baseline at matched params | 🔶 Still not a slope — but the **enabling mechanism is now measured**: pseudo-label WM pretraining captures **~96 %** (proxy, 8 seeds) / **109 % speed, 71 % yaw** (parity target, 4 seeds) of real-label value, and an 80-clip CC YouTube pilot reaches **≈92 % of ceiling** (directional). Corpus baseline now exact: **13.13 h, 4.73 epochs at 30 k** |
| **C3 — It is inherently safe & compliant** | Goal 2 (prio) | Fallback brain, self-monitoring AUROC, rule-violation rates, UN-ADS regulation trace | 🔶 Instruments built; D8 separation still preview-only (p≈0.047 paired). **New: SC-14 traffic-light scenario + TLC metric** (`red_entry_gate × stop_quality × green_flow`; a single red-run zeroes it) — design oracle **rule_barrier 1.0 vs soft_prior 0.0**, the H9 hard-barrier claim made scorable. Model-side TLC/LAL/OKRI/LOPS **renderer-gated** |

Efficiency (inference) is embedded in all three — every experiment reports params, FLOPs/decision,
latency, and the **CNCE** metric (first real value: **median 210,551** on the deployed 262.8 M
architecture, comma val, 30 eps).

## 2. The four edges and the plan to build each
| Edge | Core hypotheses | How it's built | Status (2026-07-25) |
|---|---|---|---|
| **① Planning / Hierarchy** | H1 (4-brain: strategic/tactical/operative + fallback-MRC), H26 (cross-level alignment) | Three E2E abstraction layers at different clock rates + a fallback that forces a Minimal-Risk Condition on collapse | 🔶 **Built and in its first real training round.** The v3 design shipped as **flagship-v4** (three *planners* over the WM, ≈247.9 M — 30 M smaller than v1). Four **warm-start** arms failed (best 10 k `ade_0_2s` **0.8522 [0.75, 0.98]** vs a 0.60 bar); a ~0-GPU **cosine pre-probe** (seam cos **+0.0043**) refuted gradient surgery and selected **co-evolution from random init**, which is working (canary **15.674 → ~1.4** under full coupling, ADE ~0.48 at 40 % of training, 10 k gate CONTINUE). Standing evidence unchanged: heads are a lossy readout (P2 planner 0.893 vs head 3.150 open-loop; 1.038 vs 1.685 closed-loop) |
| **② Data efficiency** | H3 (LeJEPA/SigReg world model), H4 (frozen vs trained encoder), H7 (1000× data via IDM) | Latent world model, no perception labels; inverse-dynamics to mine action-free video; focal-length canonicalization | ✅→🔶 **H4 stays closed-negative but is now correctly *localized***: the ceiling is **static decode** off a frozen JEPA latent (3.65 m, the REF-A band), not freezing — a planner reading the *same* frozen WM through its dynamics reaches **0.599 m**. **H7 has its first end-to-end evidence** (pseudo-label pretraining ≈96 % / 109 % of real-label value; YouTube pilot ≈92 % of ceiling). The C2 **slope** is still unmeasured — now the clearest single gap |
| **③ Inference efficiency** | H5 (efficient decode/inference transfer), H2 (modality steering), H8 (MoE) | 263 M vs 15–120× larger; imagine-and-select instead of generative rollout | ✅ **Closed for the deployed arm.** ⚠️ the old "deploy tick 11.16 ms" was a *different tick* (retracted). Real **planning tick** 100.29 ms eager → **18.75 ms p50 composed (5.35×)**, 10 Hz at **p99** with 5.3× headroom; levers are **sequenced, not additive**. **FP16 is the deployment precision, INT8 REJECTED** (no latency win; W+A INT8 collapses the readout head to cos 0.566 and costs +0.0215 m over 20 steps). CEM is **not** latency-blocked (K=8 fan 20.82 ms, ~0.3 ms/candidate). H2/MoE are Phase-1 |
| **④ Safety / self-knowledge** | H11 (self-monitoring), H9 (rule compliance), H15 (imagination), H14 (physical grounding) | Per-level monitors + hard rule barriers + ImaginationField (advection + epistemic σ) + kinematic/Kamm grounding | 🔶 Instruments built and **now partly real**: first MEASURED beyond-ADE numbers on the deployed architecture (decision-tick p50 **14.33 ms**, TMS **0.0435** = expert-log band, CNCE **210,551**) + **SC-14/TLC**. **H15 `vision_use` still flat at ~12 %**; D8 separation still preview-only; σ-dissipation stands (0.357 → 0.011 by k=4). The **binding gap is the renderer**: closed-loop TLC/LAL/OKRI/LOPS cannot be computed without one |

## 3. Hypothesis ledger
| H | Claim | Status (2026-07-25) |
|---|---|---|
| H0 | E2E > rule-based (settled) | ✅ confirmed |
| H1 | 4-brain hierarchy | 🔶 operative validated; heads-as-decision-makers falsified → planning (D-033). **The planner architecture is now built and training (§5.2a); coupling it is no longer the open question — the level it reaches is** |
| H2 | Attention-based modality steering | supported (DriveMoE/GEMINUS); Phase-0 exit demo |
| H3 | Latent world model (LeJEPA/SigReg) | ✅ **strengthened.** The WM is a good *differentiable simulator*: a 3.77 M planner backpropagating ADE **through the frozen WM** reaches 0.599 m, **not separated** from the WM's own oracle-action ceiling 0.4045 (+0.194 [−0.045, +0.448]) |
| H4 | Frozen vs trained encoder | ✅ **CLOSED NEGATIVE — and re-localized.** The ceiling is **static decode off the frozen latent** (3.65 m), not freezing: through the frozen *dynamics* the same latent supports 0.599 m. The trained encoder remains the data-efficient path; "frozen + regress" is what fails |
| H5 | Efficient inference transfer (CNCE moat) | ✅ supported + **first real CNCE** (median 210,551) and a defined, met tick budget (18.75 ms p50, p99 < 100 ms) |
| H6 | Opponent weak-spot corpus | actionable — scenarios shipped (Stop-Arm, Work-Zone Phantom, Stationary-Lead) + **SC-14 traffic light w/ an oracle-tested metric** |
| H7 | 1000× data via IDM + focal canonicalization | 🟢→✅ **first end-to-end evidence.** Pseudo-label WM pretraining = **~96 %** of real-label value (8 seeds, 2 proxy domains) and **109 % speed / 107 % traj / 71 % yaw** on the parity target (4 seeds); **80-clip CC YouTube pilot ≈92 % of ceiling**, CI excludes 0 every seed. ⚠️ DIRECTIONAL (80 clips / 3 seeds / unknown intrinsics); direct label accuracy is modest (speed R² 0.62–0.66, **yaw ≈ 0**, accel dropped) |
| H8 | MoE beyond sensors | prio-2, interface ready |
| H9 | Inherent rule compliance (hard barriers) | ✅→🔶 **now scorable, not just argued**: TLC design oracle **rule_barrier 1.0 vs soft_prior 0.0** on SC-14. Model-side number renderer-gated |
| H10 | Latent-RAG continual learning | validated-toy w/ known failure mode |
| H11 | Self-monitoring w/ guarantees | validated-toy; D8 AUROC > 0.85 not yet reached. **New instrument in daily use: the plan-free WM-integrity canary** — it is what made the v4 line attributable |
| H12 | Text as part, not core | supported (1 B LLM bridge) |
| H13 | Extraction heads (probes) | settled pattern |
| H14 | Physical grounding | Track 1 adopted (kinematic + Kamm) |
| H15 | Imagination of unobserved areas | 🔶 live in training; `vision_use` flat ~12 %; capped at 1-step self-monitor |
| H16 | Active depth interrogation (σ-triggered) | open — Phase-1 |
| H17 | Unified-FOV masked-periphery training | open |
| H18 | Hierarchical action grounding | 🔶 shipped; grounding dominance grew Δ 2.70 m at 30 k |
| H19 | Maneuver → anchor prior (anchored decoders) | ✅ validated by the REF-C anchor-prior graft; now the flagship's own proposal mechanism in v4 (256 anchors, tactical + operative) |
| H25 | Vision-decoupling — the encoder redundantly re-encodes ego dynamics | 🔶 open. The v3enc lever family spent 1 of 2 restarts and did **not** recover the speed probe (0.393 vs v1's 0.861); **decorr was measured never-on** during the gate window, so the family is under-tested, not refuted |
| H26 | Hierarchical cross-alignment = the core-goal proof | 🔶 **1 of 3 seams load-bearing** at 30 k (ctx→tactical +0.044 CI-sep). The v4 line replaces the seam question with a *planning* question — G1 (chosen plan beats non-chosen on realized outcome) is the successor instrument |
| **H27** *(new, 2026-07-23)* | **Planner–WM coupling failure is a warm-start artifact, not an intrinsic objective conflict** | 🟢 **supported, in-flight.** Seam gradients are near-**orthogonal** (cos +0.0043, 47.9 % negative, PCGrad removes 2.2 %) so the conflict is neither directional nor magnitude-borne; four warm-start arms degrade the WM, the random-init co-evolved arm's canary **descends** under full coupling. ⚠️ in-loop evidence, formal gate deferred, run at ~40 % |
| **H28** *(new, 2026-07-24)* | **The frozen-WM planner's residual is aleatoric, not a capacity or search deficit** | ✅ **settled negative for the contender question.** 11× planner scaling is flat (0.599 → 0.601 → 0.599, none separated); the deployable learned-value search is **worse** (1.016, separated). Frozen-WM = a ~0.60 m degradation-free **fallback** |

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

## 5. Current state & latest achievements (2026-07-25)

### 5.1 The open-loop bake-off — settled, and the top of the table is a three-way tie
All numbers: TanitEval, physicalai val (`physicalai-val-0c5f7dac3b11`), **881 windows / 40 episodes**.
⚠️ The `±` column is the **deprecated** `overlapping_holdout_se` (1.28–2.06× too narrow); the
`[bootstrap]` figures are the decision-grade episode-cluster bootstrap. Provenance:
[`MODEL_REGISTRY.md §6`](MODEL_REGISTRY.md).

| Arm | Step | Params | ADE@2s (heldout ± legacy SE) | full-set / bootstrap | Beats CV 0.825? |
|---|---:|---:|---:|---|:--:|
| **1= Flagship v1 (4-brain WM, trained ViT) — DEPLOYED** | 29 999 | 263.4 M | **0.4522 ± 0.0312** | 0.4271 · [0.3675, 0.4871] | ✅ |
| **1= REF-C-base** (anchored diffusion) | 29 999 | **104.2 M** | **0.4523 ± 0.0497** | 0.4728 · [0.3835, 0.5699] | ✅ |
| **1= REF-C-XL** (anchored diffusion) | 29 999 | 251.9 M | **0.458 ± 0.057** | 0.4714 · [0.3896, 0.5556] | ✅ |
| *best-of-3 kinematic floor 0.5005 · CTRV oracle 0.523 · no-vision ego ceiling 0.5735* | | | | | |
| REF-C-small (separated 3rd rung) | 29 999 | 54.7 M | 0.5007 ± 0.0671 | 0.5261 | ✅ |
| **REF-B v2** (BC + time-anchored decoder) | 29 999 | 271.6 M | 0.5921 ± 0.0685 | — | ✅ |
| Flagship v1, 19 k relay | 19 000 | 263.4 M | 0.6277 ± 0.0551 | 0.6152 | ✅ |
| **Constant velocity (the floor)** | — | 0 | **0.8248** | 0.8377 | — |
| P2 CEM planner over frozen v1 | n/a | 0 trained | 0.893 ± 0.114 | — | ✗ |
| REF-A DINOv2 4B (frozen encoder) | 29 999 | — | 2.1322 ± 0.1821 | 2.1675 | ✗ |
| Flagship **no-speed** (ablation control) | ~22 000 | 263.4 M | 2.9176 ± 0.3558 | 3.0175 | ✗ |
| REF-A dyn-in 4B (frozen, every remedy applied) | 29 999 | — | 2.9196 ± 0.3937 | 3.047 | ✗ |
| Flagship v1 **tactical head** (not the rollout) | 29 999 | — | 3.38 | — | ✗ |
| Flagship v2 (killed) / v3enc 10 k (RESTART) | 6 k / 10 k | 272.9 M | 6.179 ± 1.28 / 1.9654 [1.656, 2.286] | — | ✗ |

**Three readings.** (1) Every **trained-encoder** arm sits above CV; both **frozen-encoder** arms sit far
below — H4 in one table. (2) **Rank 1 is a genuine three-way tie no paired test can order** (base − XL
Δ +0.0013 [−0.0281, +0.0316]) — held by a 263 M world model, a 252 M diffusion arm **and a 104 M
diffusion arm**, so *scale bought nothing above 104 M on this corpus*. (3) The flagship's own supervised
**tactical head is worse than CV** (3.38 m) while the same model's operative rollout is 0.452 m — the
head is a lossy readout of a good world model.

### 5.2 What moved the program this round

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
(speed R² 0.62–0.66 cross-domain, longitudinal-traj 0.60, **yaw ≈ 0**, accel dropped) — but what matters
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

### 5.3 What is running right now (fleet ~2026-07-25)
| Pod | Run | State |
|---|---|---|
| `tanitad-pod2` | ⭐ **flagship-v4 from-scratch → 30 k** | 🟢 ~step 11.9 k, λ_plan 1.0, canary in-band, restarts 0; ~2 days to 30 k, auto-continues. ⚠️ 3.2 GB ckpt on **pod2 disk only** — HF backup 403-blocked (mid-checkpoint-loss risk). **NEVER eval here** |
| `tanitad-pod` (pod1) | **YouTube-IDM non-CC harvest/label** (scale-up) | 🟢 Sayed committed to the licensing 07-25; v2-corpus shard **DONE** (4953/4953) |
| `tanitad-pod3` | **v2-corpus build** → then IDM scale-up pretrain | 🟢 finishing the second shard; consolidate by clip-id union, then **QA** |
| `tanitad-eval` | **GeoCalib** (per-video intrinsics — removes the pilot's fixed-HFOV approximation) | 🟢 |
**0 idle GPUs.** 🟡 **`Sayood/` HF storage is FULL (403)** — blocks the flagship ckpt backup, the formal
v4 gate relay, and a REF-C arm. Minimal safe unblock ≈13 GB (superseded v4.1/v4.2/v4.2b + the val-leaked
refa-ijepa) — **Sayed's click; irreversible deletes are not run autonomously.**

### 5.4 Stack & tooling maturity
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

Cross-cutting: the living paper (`Paper/TANITAD_PAPER.md` — now **v0.6**), `LEADERBOARD.md`,
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
- **Top risks:** the live flagship's **checkpoint on a single pod disk with HF backup 403-blocked**;
  **premature certainty** (four closure claims reopened by zero-cost checks in one session — the
  countermeasure is doctrine, not vigilance); the **v4 gate cannot render a complete formal verdict**
  (3 of 8 kill secondaries still have no emitter); open-loop ⊥ closed-loop; PhysicalAI-AV license
  firewall (never in public claims); **YouTube-IDM scale-up now carries a licensing/privacy obligation**
  (pointers + pseudo-labels only, face/plate blur pre-downscale, never raw bytes).

## 8. The critical path from here
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

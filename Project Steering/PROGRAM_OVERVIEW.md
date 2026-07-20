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
> **Last refreshed:** 2026-07-20 · Phase 0 (~day 16 / 42) · **Next refresh:** v3enc verdict or Phase-0 exit.

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
| Claim | Goal | Proof artifact | Where it stands |
|---|---|---|---|
| **C1 — It drives** | Goal 1 | Closed-loop success + latency/FLOPs ledger (Orin/Thor envelope) | 🔶 Open-loop **cleared** (0.452 m, below every trivial floor). Closed-loop **not** — 1.685 m in-house, external sim blocked |
| **C2 — It needs magnitudes less data** | Goal 2 (prio) | Data-efficiency slope vs supervised baseline at matched params | 🔶 Not yet measured as a slope. The Phase-1 headline |
| **C3 — It is inherently safe & compliant** | Goal 2 (prio) | Fallback brain, self-monitoring AUROC, rule-violation rates, UN-ADS regulation trace | 🔶 Instruments built; D8 separation still preview-only (p≈0.047 paired) |

Efficiency (inference) is embedded in all three — every experiment reports params, FLOPs/decision,
latency, and the **CNCE** metric.

## 2. The four edges and the plan to build each
| Edge | Core hypotheses | How it's built | Status (2026-07-20) |
|---|---|---|---|
| **① Planning / Hierarchy** | H1 (4-brain: strategic/tactical/operative + fallback-MRC), H26 (cross-level alignment) | Three E2E abstraction layers at different clock rates + a fallback that forces a Minimal-Risk Condition on collapse | 🔶 **Reframed.** Operative is excellent (0.452 m). The *supervised heads* are the weak link: tactical head 3.38 m (worse than CV), strategic seam is a pure command echo (`route_skill_vs_chance` 0.0), intent→operative actively harmful (cos −0.238). **P2 proved planning-over-the-WM recovers the decision** (0.893 vs 3.150 open-loop; 1.038 vs 1.685 closed-loop, both CI-separated) → **v3 pivot (D-033)** |
| **② Data efficiency** | H3 (LeJEPA/SigReg world model), H4 (frozen vs trained encoder), H7 (1000× data via IDM) | Latent world model, no perception labels; inverse-dynamics to mine action-free video; focal-length canonicalization | ✅→🔶 **H4 CLOSED NEGATIVE** with a clean ceiling proof (REF-A improves monotonically to 30 k and still plateaus at 2.92 m — a *capability* ceiling, not overfitting). The trained encoder is the data-efficient path. The C2 **slope** itself is still unmeasured |
| **③ Inference efficiency** | H5 (efficient decode/inference transfer), H2 (modality steering), H8 (MoE) | 263 M vs 15–120× larger; imagine-and-select instead of generative rollout | ✅ On track — 263 M envelope; deploy tick **11.16 ms / 89.6 Hz** fp16 (1.59× vs fp32), predictor CUDA-graph 2.57×. H2/MoE are Phase-1 |
| **④ Safety / self-knowledge** | H11 (self-monitoring), H9 (rule compliance), H15 (imagination), H14 (physical grounding) | Per-level monitors + hard rule barriers + ImaginationField (advection + epistemic σ) + kinematic/Kamm grounding | 🔶 Instruments built. **H15 `vision_use` sits flat at ~12 %** — the imagination edge is live but small. D8 clean-vs-degraded separation still preview-only. σ-dissipation measured (fidelity 0.357 → 0.011 by k=4) |

## 3. Hypothesis ledger
| H | Claim | Status |
|---|---|---|
| H0 | E2E > rule-based (settled) | ✅ confirmed |
| H1 | 4-brain hierarchy | 🔶 operative validated; **heads-as-decision-makers falsified** → planning (D-033) |
| H2 | Attention-based modality steering | supported (DriveMoE/GEMINUS); Phase-0 exit demo |
| H3 | Latent world model (LeJEPA/SigReg) | ✅ the WM is real — a training-free planner over it beats every supervised head |
| H4 | Frozen vs trained encoder | ✅ **CLOSED NEGATIVE** — frozen-DINO ceiling proven by a monotone milestone curve |
| H5 | Efficient inference transfer (CNCE moat) | supported |
| H6 | Opponent weak-spot corpus | actionable — scenarios shipped (Stop-Arm, Work-Zone Phantom, Stationary-Lead) |
| H7 | 1000× data via IDM + focal canonicalization | supported; real (steer, accel) pairs in hand |
| H8 | MoE beyond sensors | prio-2, interface ready |
| H9 | Inherent rule compliance (hard barriers) | supported, concrete math (0.0 vs 1.0 violation) |
| H10 | Latent-RAG continual learning | validated-toy w/ known failure mode |
| H11 | Self-monitoring w/ guarantees | validated-toy; D8 AUROC > 0.85 not yet reached |
| H12 | Text as part, not core | supported (1 B LLM bridge) |
| H13 | Extraction heads (probes) | settled pattern |
| H14 | Physical grounding | Track 1 adopted (kinematic + Kamm) |
| H15 | Imagination of unobserved areas | 🔶 live in training; `vision_use` flat ~12 %; capped at 1-step self-monitor |
| H16 | Active depth interrogation (σ-triggered) | open — Phase-1 |
| H17 | Unified-FOV masked-periphery training | open |
| H18 | Hierarchical action grounding | 🔶 shipped; grounding dominance grew Δ 2.70 m at 30 k |
| H19 | Maneuver → anchor prior (anchored decoders) | ✅ validated by the REF-C anchor-prior graft; now also in flagship v2/v3enc and REF-B v2 |
| H25 | Vision-decoupling — the encoder redundantly re-encodes ego dynamics | 🔶 open — motivated the v2/v3enc encoder-grounding levers |
| H26 | Hierarchical cross-alignment = the core-goal proof | 🔶 **1 of 3 seams load-bearing** at 30 k (ctx→tactical +0.044 CI-sep); the other two are null or harmful |

## 4. Phases & timeline
| Phase | Window | Goal | Where we are |
|---|---|---|---|
| **Phase 0** — foundation & edge proofs | 07-05 → ~08-15 (6 wks) | Running 4-brain WM, single front cam, open-loop + first closed-loop; gates D1–D6 | **~day 16.** Open-loop bar **cleared**; the bake-off has a verdict; the binding constraint moved to **closed-loop + generalization** |
| **Phase 1** — boost & breadth | ~08-15 → 09-20 | Real data at scale + the C2 data-efficiency slope headline; H2 modality steering; H9/H15/H12; NAVSIM/Bench2Drive entries; AlpaSim | Gated on Phase-0 exit. **TanitDataSet v1** + the v3 goal vocabulary are being built now |
| **Phase 2** — scaling & external proof | ~09-20 → 10-05 (P7 eval) | Scale along the measured slope; multi-cam+radar; closed-loop at benchmark scale; Orin/Thor TensorRT; final safety case | Not started |

**Phase-0 exit is NOT "gates measured"** — it is: (1) open-loop beats constant-velocity AND go-straight
on **both** straight and curve strata; (2) **closed-loop** route completion with imagine-and-select;
(3) held-out ADE within a factor of the oracle ceiling. **(1) is now met in-distribution. (2) is blocked
on simulator infrastructure. (3) is met.** *Only then do more cameras/sensors/the H-stack proceed.*

## 5. Current state & latest achievements (2026-07-20)

### 5.1 The bake-off has a verdict — the trained encoder wins, and the floor is beaten
All numbers: TanitEval, physicalai val, **881 windows**, 8-split episode-disjoint jackknife,
`heldout mean ± CI95`. Full detail and provenance in [`MODEL_REGISTRY.md §6`](MODEL_REGISTRY.md).

| Arm | Step | ADE@2s | Beats CV 0.825? |
|---|---:|---:|:--:|
| **Flagship v1 (4-brain WM, trained ViT) — DEPLOYED** | 29 999 | **0.452 ± 0.031** | ✅ |
| *best-of-3 kinematic floor 0.5005 · CTRV oracle 0.523 · no-vision ego ceiling 0.5735* | | | |
| **REF-C-XL** (DiffusionDrive anchored diffusion, 252 M) | ~16 000 | 0.565 ± 0.045 | ✅ |
| **REF-B v2** (from-scratch BC + time-anchored decoder, 272 M) | 29 999 | 0.592 ± 0.069 | ✅ |
| Flagship v1, 19 k relay | 19 000 | 0.628 ± 0.055 | ✅ |
| **Constant velocity (the floor)** | — | **0.825** | — |
| REF-A DINOv2 4B (frozen encoder) | 29 999 | 2.132 ± 0.182 | ✗ |
| Flagship **no-speed** (ablation control) | ~22 000 | 2.918 ± 0.356 | ✗ |
| REF-A dyn-in 4B (frozen, every remedy applied) | 29 999 | 2.920 ± 0.394 | ✗ |

**Reading it:** every trained-encoder arm is above CV; every frozen-encoder arm is far below. That is
**H4 in one table.** The flagship also clears the *honest* floors — the best-of-3 kinematic baseline and
the no-vision ego-status ceiling — which is the bar that distinguishes "drives" from "extrapolates".

### 5.2 The three findings that actually moved the program

**(a) The speed fix — a real root-cause diagnosis (07-14).** Actions are derivatives `[steer, accel]`;
absolute displacement needs `v0`. Feeding `v0` as a **3rd action channel** took REF-A's operative
fwd-ADE **3.73 → 0.83 m** and speed-decodability **R² 0.61 → 0.965** *in isolation, before committing to
the retrain*, then held up causally on the flagship: no-speed **2.918** vs speed **0.452** on identical
data and architecture (paired A/B +2.21 m [2.04, 2.39]). All three arms were reset from scratch to get it.

**(b) H4 closes negative, cleanly.** The frozen encoder plateaus **while still improving** — 5 k 3.755
→ 15 k 3.694 → 20 k 3.016 → 30 k **2.920** (best is last; held-out error never rises). Not overfitting:
a capability ceiling. Every remedy was tried (speed, yaw, `[v0,yr0]` dyn-input, ego-dropout, temporal
adapter, full 4 brains, I-JEPA features). REF-A is now **accepted as a reference arm**, not a candidate.
A publishable negative that motivates the trained-encoder flagship.

**(c) The heads are the bottleneck, not the world model — and planning proves it.** The flagship's own
supervised tactical head scores **3.38 m** (worse than CV) while the *same model's* operative rollout
scores 0.452 m. A **training-free CEM planner** over the frozen v1 world model — 64 samples, 3
iterations, a hand-built longitudinal+comfort cost, nothing fit — beats that head by **+2.257 ± 0.329 m
open-loop (0.893 vs 3.150, a 72 % error reduction)** and drifts **38 % less closed-loop (1.038 ± 0.202
vs 1.685 ± 0.098)** with **2.5× fewer divergences (8.7 % vs 22.2 %)**. Both gates CI-separated; robust
across a 4× weight band. **This is the evidence base for the v3 pivot (D-033).**

### 5.3 What is running right now
| Pod | Run | State |
|---|---|---|
| `tanitad-pod` | **flagship v3enc** (`--v2 --staged-levers`) | 🟢 relaunched 2026-07-20 05:27 UTC from step 0, after the pod2 copy died at step 1,950 on a full-disk checkpoint write |
| `tanitad-pod3` | **REF-C-XL** 30 k | 🟢 ~26,250 / 30,000 |
| `tanitad-pod2` | — | idle; **overlay 98 % full** (the v3enc killer) |
| `tanitad-eval` | TanitEval + AlpaSim investigation | evals on demand |

### 5.4 Stack & tooling maturity
Train pipeline ✅ · gate runner ✅ · **TanitEval** (jackknife CI, A/B, imagination, hierarchy, OOD
generalization, closed-loop, path/speed decomposition, REF-C path, P2 planner) ✅ *but uncommitted* ·
TanitResim replay ✅ · data recipe (rebuild-from-origin, parity-gated, proven on 3 pods) ✅ ·
TanitDataSet v1 lake pipeline (schema/filtering/dedup/curation/minting) ✅ shipping ·
CARLA harness ✅-narrow · AlpaSim 🔴 NO-GO on current infra.

## 6. The agent ecosystem (the research flywheel)
Seven disciplines, each a folder + a **weekly post-doc-grade agent** (Mon→Fri), doing theory +
implementation, each with a knowledge base, BACKLOG, and ≥1 measured experiment per run:

| Agent (day) | Owns | Recent output |
|---|---|---|
| **Tools & DevEnv** (Mon) | sim, replay, CI, compute | TanitResim/TanitScena; OOD corpus provisioning (comma + Cosmos val caches, f-theta-canonical) |
| **Data Engineering** (Tue) | datasets, loaders, training flow | TanitDataSet v1 strategy + non-VLM lake pipeline; per-clip principal-point crop (the two-rig fix); VLM pilot → Cosmos-Reason1-7B |
| **Architecture & Inference** (Wed) | the stack, efficiency | REF-A deep analysis + DINO-WM literature comparison; **P2 planner**; blind-rollout σ-dissipation; v3 hierarchical-planning design + goal vocabulary |
| **Benchmarks & Eval** (Thu) | metrics, gates, leaderboard, regulation | flagship-v2 6 k diagnostic (the restart evidence); REF-C eval path; open-loop L2 + ego-status shortcut ceiling; AlpaSim closed-loop v1 |
| **Opponent Analyzer** (Fri) | opponent intel, weakness catalog, scenario DB | weekly competitor sweeps → scenarios (Stationary-Lead SC-13, first-responder W-09) |
| **Project Steering** (Fri) | plans, reports, resource control | weekly Progress Reports, roadmap, decision log, **this document + the model registry** |
| **Production & Optimization** | ONNX/TRT/quant, latency, compliance | clean-GPU fp16 latency; H15 logvar NaN clamp; milestone-archive corruption fix |

Cross-cutting: the living paper (`Paper/TANITAD_PAPER.md`), `LEADERBOARD.md`, `SCENARIO_DATABASE.md`
(SC-01…SC-14). **Honest agent-health note:** research output is strong and continuous; **git hygiene is
the standing structural risk** — three of the highest-value artifacts of the last week (REF-B v2's
architecture, the whole TanitEval harness, the P2 planner) exist **only on pods** and are not in this
repo. See [`MODEL_REGISTRY.md §7`](MODEL_REGISTRY.md) for the full gap register.

## 7. Honest position (P8)
- **Proven:** the 4-brain latent world model **beats every trivial floor open-loop** (0.452 m vs
  best-of-3 0.5005, CTRV 0.523, no-vision ego ceiling 0.5735, CV 0.825), and it does so *causally* —
  ablating the scene inverts a +0.796 m oracle-beating margin to −0.529 m (vision effect +1.325 m,
  CI-separated). Upcoming-curvature decodes from the pooled latent at R² 0.254 vs 0.031 ego-only.
  The world model is real.
- **Now settled:** **H4 — the frozen encoder has a capability ceiling** (monotone improvement to 30 k,
  still 2.92 m). A clean, publishable negative.
- **Newly settled:** **supervised heads are a lossy readout of a good world model.** A training-free
  planner beats them by 72 % open-loop and 38 % closed-loop. This is why v3 exists.
- **Binding constraints, in order:**
  1. **Closed-loop.** 0.452 m open-loop → 1.685 m closed-loop; 22 % of windows diverge > 5 m; 59 % of
     steps exceed the 2.0 m/s² comfort bound. **Open-loop does not predict closed-loop.**
  2. **Generalization.** In-distribution 0.427 vs floor 0.523 ✅; comma2k19 0.849 vs floor 0.372 ✗
     (17.5 % win-rate); path feasibility collapses 97.8 % → 62.8 % on OOD sharp curvature.
  3. **Longitudinal at speed.** The only above-floor stratum (1.785× floor); 89 % of squared error is
     along-track; +0.66 m/s speed over-prediction at high speed. More training fixed *lateral*, not this.
  4. **Simulator infrastructure.** D5/D6 blocked: AlpaSim needs nested docker in an unprivileged
     container; CARLA pixels host-blocked. One shared fix — a graphics-capable GPU host.
- **Top risks:** pod disk/memory ceilings (already cost one run); **uncommitted pod-only code** (§6);
  open-loop ⊥ closed-loop; stale gate evidence (D2/D3 last measured pre-reset at 27 k, in camera-frame
  units); PhysicalAI-AV license firewall (never in public claims).

## 8. The critical path from here
1. **Commit the pod-only code** — REF-B v2's architecture, TanitEval, the P2 planner, the flagship v1
   trainer diff. Highest value per hour on the board; three of our best artifacts are one pod-loss from
   gone. ([`MODEL_REGISTRY.md §7`](MODEL_REGISTRY.md))
2. **v3enc to the 10 k gate** — with the pre-registered falsifier (no improvement in same-step
   forward-consistency vs v1 ⇒ restart again) and the **OOD panel as the acceptance gate**, not the
   in-distribution one we already pass.
3. **P3 — the lateral goal in the planning cost.** P2 localized 66 % of its residual as lateral and
   untouched; the strategic ROUTE/LANEOBJ term is the direct lever, and curved windows (2.114 m today
   vs 0.484 m with true actions) are where it should pay.
4. **Unblock closed-loop.** A graphics-capable GPU host is the single dependency for D5/D6 and for the
   Phase-0 exit arbiter.
5. **REF-C-XL to 30 k**, then decide whether the 3-size scaling study (small 55 M / base 104 M /
   XL 252 M) is still worth the compute — today only XL exists, so the scaling claim is unsupported.
6. **C2 data-efficiency slope** — the Phase-1 headline that makes "1000× less data" measurable.

**Bottom line:** Phase 0 now has a 4-brain world model that **clears the open-loop driving bar on real
data against honest floors**, a settled negative on the frozen encoder, and a training-free result that
redirects the architecture (planning > heads). It has **not** cleared closed-loop or generalization —
and those two, plus the simulator infrastructure that gates them, are the honest distance between
"the world model works" and "the edge is proven."

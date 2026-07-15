# TanitAD — Program Overview (living)

> **Whole-program strategic briefing**: the vision & the bet, the three undeniable claims, the four
> edges + the plan to build each, the H0–H18 hypothesis ledger, the phases/timeline, the agent
> ecosystem, current state & achievements, the honest position (P8), and the critical path.
>
> Distinct from the operational `Project Steering/Reports/*-program-report.md` (which tracks the live
> training/gate cadence). **This is the canonical format for every future whole-program briefing** —
> refreshed at each phase boundary and on request.
>
> **Last refreshed:** 2026-07-15 · Phase 0 (~day 11 / 42) · **Next refresh:** flagship@30k verdict or Phase-0 exit.

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
| Claim | Goal | Proof artifact |
|---|---|---|
| **C1 — It drives** | Goal 1 | Closed-loop success + latency/FLOPs ledger (Orin/Thor envelope) |
| **C2 — It needs magnitudes less data** | Goal 2 (prio) | Data-efficiency slope vs supervised baseline at matched params |
| **C3 — It is inherently safe & compliant** | Goal 2 (prio) | Fallback brain, self-monitoring AUROC, rule-violation rates, UN-ADS regulation trace |

Efficiency (inference) is embedded in all three — every experiment reports params, FLOPs/decision,
latency, and the **CNCE** metric.

## 2. The four edges and the plan to build each
| Edge | Core hypotheses | How it's built | Status |
|---|---|---|---|
| **① Planning / Hierarchy** | H1 (4-brain: strategic/tactical/operative + fallback-MRC) | Three E2E abstraction layers at different clock rates (thinking fast/slow) + a fallback that forces a Minimal-Risk Condition on collapse | 🔶 In-training. Operative strong (0.11 m in-train); tactical/strategic maturing. REF-A now carries the same 4 brains → encoder is the only diff vs flagship. Verdict at flagship@30k. |
| **② Data efficiency** | H3 (LeJEPA/SigReg world model), H4 (frozen vs trained encoder), H7 (1000× data via IDM on unlabeled video) | Latent world model, no perception labels; inverse-dynamics to mine action-free video; focal-length canonicalization | 🔶 The bake-off's core. H4 answered (§5): frozen-DINO ceiling; the trained encoder is the data-efficient path. C2 slope experiment is the Phase-1 headline. |
| **③ Inference efficiency** | H5 (efficient decode/inference transfer), H2 (attention-based modality steering), H8 (MoE) | 261 M vs 15–120× larger; imagine-and-select instead of generative rollout; tactical layer picks camera/sensor only when needed | ✅ On track — 261 M envelope; clean-GPU latency fp16 13.4 ms / 74.6 Hz (proxy); CNCE from day 1. H2/MoE are Phase-1. |
| **④ Safety / self-knowledge** | H11 (self-monitoring w/ guarantees), H9 (inherent rule compliance), H15 (imagination of unobserved areas), H14 (physical grounding) | Per-level monitors + hard rule barriers (rule_barrier 0.0 vs soft-prior 1.0 violation) + ImaginationField (advection + epistemic σ) + kinematic/Kamm-circle grounding | 🔶 Instruments built (D8 AUROC, LAL/OKRI/LOPS, violation-rate). H15 trains inside the WM; D8 clean-vs-degraded separation still blocked. |

## 3. Hypothesis ledger (H0–H18)
| H | Claim | Status |
|---|---|---|
| H0 | E2E > rule-based (settled) | ✅ confirmed |
| H1 | 4-brain hierarchy | validated-toy (5× lift, ALPS-4B) → now in the live bake-off |
| H2 | Attention-based modality steering | supported (DriveMoE/GEMINUS); Phase-0 exit demo |
| H3 | Latent world model (LeJEPA/SigReg) | validated-toy; D1–D3 |
| H4 | Frozen vs trained encoder | **answered** — frozen-DINO ceiling (§5) |
| H5 | Efficient inference transfer (CNCE moat) | supported |
| H6 | Opponent weak-spot corpus | actionable — scenarios shipped (Stop-Arm, Work-Zone Phantom) |
| H7 | 1000× data via IDM + focal canonicalization | supported; real (steer,accel) pairs in hand |
| H8 | MoE beyond sensors | prio-2, interface ready |
| H9 | Inherent rule compliance (hard barriers) | supported, concrete math (0.0 vs 1.0 violation) |
| H10 | Latent-RAG continual learning | validated-toy w/ known failure mode |
| H11 | Self-monitoring w/ guarantees | validated-toy (3 mechanisms); D8 AUROC>0.85 |
| H12 | Text as part, not core | supported (1 B LLM bridge) |
| H13 | Extraction heads (probes) | settled pattern |
| H14 | Physical grounding | Track 1 adopted (kinematic + Kamm) |
| H15 | Imagination of unobserved areas | in Phase 0 — ImaginationField live; D9 |
| H16 | Active depth interrogation (σ-triggered) | open — Sayed idea; Phase-1 |
| H17 | Unified-FOV masked-periphery training | open — Sayed idea |
| H18 | Hierarchical action grounding | open — operative grounding ships now (behind the current gates) |

## 4. Phases & timeline
| Phase | Window | Goal | Where we are |
|---|---|---|---|
| **Phase 0** — foundation & edge proofs | 07-05 → ~08-15 (6 wks) | Running 4-brain WM, single front cam, open-loop + first closed-loop; gates D1–D6 (D1–D3 pass, D5/D6 show hierarchy edge); two encoder arms compared honestly | ~day 11, in progress. Bake-off training; the reset relocated the binding constraint (§5). |
| **Phase 1** — boost & breadth | ~08-15 → 09-20 | Real data at scale + the C2 data-efficiency slope headline; H2 modality steering; H9/H15/H12; NAVSIM/Bench2Drive entries; AlpaSim | Gated on Phase-0 exit |
| **Phase 2** — scaling & external proof | ~09-20 → 10-05 (P7 eval) | Scale along the measured slope; multi-cam+radar; closed-loop at benchmark scale; Orin/Thor TensorRT; final safety case | Not started |

**Phase-0 exit is NOT "gates measured"** — it is: (1) open-loop beats constant-velocity AND go-straight
on **both** straight and curve strata; (2) **closed-loop** route completion with imagine-and-select;
(3) held-out ADE within a factor of the oracle ceiling. *Only then do more cameras/sensors/the H-stack proceed.*

## 5. Current state & latest achievements (2026-07-15)
**The 3-arm bake-off (the H1/H4 evidence engine, strict same-data parity):**

| Arm | Isolates | Step | Decision-grade gate (held-out, vs CV 0.83 m) |
|---|---|---|---|
| **Flagship** 4-brain, trained ViT | the full thesis | 6.35k/30k (21%) | 5k: **2.34 m**, but beats REF-A@30k at the 1 s horizon on every stratum — rising |
| **REF-A** 4-brain, frozen DINO | the encoder axis (H4) | **30k DONE** | **2.14 m** — halved old 3.73 m, but **plateaued = frozen-encoder ceiling** |
| **REF-B** from-scratch BC | hierarchy vs flat (H1/D4) | 3.6k/30k (12%) | pending (~7 d) |

**Headline achievement — a real root-cause fix (the "speed/scale reset").** The models lacked the
**current-speed input**: actions are derivatives `[steer, accel]`, so absolute displacement needs `v0`,
which a frozen encoder can't recover from pixels. Feeding `v0` as a 3rd action channel **halved REF-A's
error (3.73 → 2.14 m)** and lifted speed-decodability 0.61 → 0.965 R². All three arms restarted from
scratch; **REF-A given the full 4 brains by hand**; two speed-aware gate harnesses + a trajectory-overlay
pipeline built; the reset archived to the repo (`stack/experiments/reset-speed4b/`). **H4 effectively
answered:** the frozen encoder plateaus above CV; the trained encoder is the path and is already sharper
per-step at ⅐ the training.

**Stack & tooling maturity:** train pipeline ✅, gate runner ✅, TanitResim replay ✅-advancing, eval
suite (LAL/TMS/OKRI/CNCE/LOPS, 22 tests) ✅, data-recipe (rebuild-from-origin, 2 pods) ✅, CARLA harness
✅-narrow, TanitScena prototype.

## 6. The agent ecosystem (the research flywheel)
Seven disciplines, each a folder + a **weekly post-doc-grade agent** (Mon→Fri), doing theory +
implementation, each with a knowledge base, BACKLOG, and ≥1 measured experiment per run:

| Agent (day) | Owns | Recent output |
|---|---|---|
| **Tools & DevEnv** (Mon) | sim, replay, CI, compute | TanitResim/TanitScena; MetaDrive front-cam perturbation |
| **Data Engineering** (Tue) | datasets, loaders, training flow | comma2k19 + PhysicalAI-AV + Cosmos-Drive-Dreams loaders; realmix recipe |
| **Architecture & Inference** (Wed) | the stack, efficiency | gate runner, spectral-sizing, K-step bake-off, JEPA-WM deltas |
| **Benchmarks & Eval** (Thu) | metrics, gates, leaderboard, regulation | metric suite; LAL-v2; eval-metric-suite |
| **Opponent Analyzer** (Fri) | opponent intel, weakness catalog, scenario DB | weekly competitor sweeps → scenarios (Waymo/Tesla/Avride recalls) |
| **Project Steering** (Fri) | plans, reports, resource control | weekly Progress Reports, roadmap, decision log |
| **Production & Optimization** | ONNX/TRT/quant, latency, compliance | INT8 curves, fail-loud windowing, compliance reviews |

Cross-cutting: the living paper (`Paper/TANITAD_PAPER.md`), `LEADERBOARD.md`, `SCENARIO_DATABASE.md`
(SC-01…SC-14). **Honest agent-health note:** research output is strong and continuous; git hygiene lags
(intake packages sit uncommitted; the main-stream MVP work — like the reset — is carried by the operator
loop, not the discipline branches). Standing, tracked risk.

## 7. Honest position (P8)
- **Proven:** the reset corrected a real design bug and cut REF-A's driving error nearly in half —
  validating the speed/scale diagnosis. Tooling, gates, and 4-brain wiring are real and running on
  identical data (strict parity).
- **Now settled:** **H4 — the frozen encoder has a ceiling** (REF-A plateaus at 2.14 m, above CV even
  fully trained + speed + 4 brains). A clean, publishable negative that motivates the trained-encoder flagship.
- **Binding constraint:** **neither arm beats constant-velocity yet.** Highway CV is brutally strong
  (straight CV@1 s = 0.15 m). The decisive verdict is the **flagship at 30k (~4 days)**, then the real
  Phase-0 exit test: **closed-loop route completion** (open-loop ADE ⊥ closed-loop score, arXiv 2605.00066).
- **Top risks:** pod memory/throughput cap; open-loop⊥closed-loop; agent git-hygiene; PhysicalAI-AV
  license firewall (never in public claims).

## 8. The critical path from here
1. **Flagship 15k re-gate (~1.4 d)** — does the 2 s rollout tighten (under-training) or stay (k=4 recipe)?
2. **Flagship → 30k (~4 d)** — the decisive C1/H1/H4 verdict vs CV and vs REF-A.
3. **Closed-loop route completion** — the actual Phase-0 exit arbiter (CARLA-on-pod).
4. **REF-B → 30k** — completes the hierarchy-vs-flat (D4) comparison.
5. **C2 data-efficiency slope** — the Phase-1 headline that makes "1000× less data" measurable.

**Bottom line:** Phase 0 has a running 4-brain world-model stack on real driving data, a strict
three-way bake-off, a full custom eval/metric suite, and a genuine mechanistic finding
(frozen-encoder ceiling + the speed-input fix). It has **not yet** cleared the driving bar (beat CV,
closed-loop) — that is the next ~4 days to two weeks, and it is the honest gate between "promising
foundation" and "edge proven."

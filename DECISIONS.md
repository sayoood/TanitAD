# TanitAD — Decision Log (ADR style)

> Append-only. Every entry: ID, date, status, decision, rationale, consequences.
> Statuses: `proposed` → `accepted` | `rejected` | `superseded by D-xxx`.
> Decisions must never contradict `Project Steering/Mission Plan.md` (the constitution).
> Changes to the constitution itself are only proposals → `Project Steering/Proposals/`.

---

## D-001 — Repo layout: runnable code lives in `stack/` (2026-07-05, accepted)

**Decision.** All runnable MVP code lives in a top-level `stack/` folder (Python package `tanitad`),
because the discipline folders ("Benchmarks & Eval", "TanitAD Research Hub") contain spaces and mixed
content and are unsuitable as Python package roots. Discipline folders keep documents, research and
per-area state; `stack/` keeps code, configs, tests, scripts.

**Rationale.** Clean imports, CI-ability, one place to run/test; matches Mission Plan structure without
changing it.

## D-002 — Phase 0 data strategy: sim-first, then real data with free actions (2026-07-05, accepted)

**Decision.** Phase 0 uses a staged data ladder:
- **Stage A (weeks 1–4):** MetaDrive procedural driving sim (BEV first, then front camera). Purpose:
  prove the edge hypotheses cheaply and falsifiably (topology/generalization experiments are impossible
  to run cleanly on real logs).
- **Stage B (weeks 5–10):** comma2k19 (33 h real driving, CAN steering/speed + GNSS — real actions with
  zero annotation cost) as first real dataset; **NVIDIA PhysicalAI-AV** (1 727 h, multi-camera + radar +
  lidar, loader already prototyped in `DataEng/AVDataSetLoader`) as the multi-sensor dataset feeding H2
  (modality steering); nuScenes-mini for OOD probes.

**Rationale.** Follows the validated ALPS-4B transfer program (`ALPS-4B/docs/AD_TRANSFER_RESEARCH.md` §9–10);
maximizes proof-per-GPU-dollar (P5); PhysicalAI-AV is the only open dataset with the camera+radar+lidar
diversity H2 needs. License review of PhysicalAI-AV before public claims is an open task.

## D-003 — Architecture baseline: from-scratch 4B latent world model; frozen-encoder as H4 comparison arm (2026-07-05, accepted)

**Decision.** The main track is the 4B hierarchical latent world model trained from scratch with
SIGReg-only anti-collapse (LeJEPA), residual/delta prediction, change-weighted latent loss, inverse-dynamics
grounding, spatial grid readout, and frozen imagination-calibrated probes for trajectory decode.
A parallel low-cost arm trains the same predictor stack on top of a **frozen DINOv3/DINOv2 encoder (+LoRA)**
to answer H4. Decision between arms is made on measured gate results, not preference.

**Rationale.** Every component of the main track is individually validated in ALPS-4B (assets A1–A10) or
externally (LAW, IDOL, ResWorld, V-JEPA-2-AC). The from-scratch arm is what makes the data-efficiency
claim disruptive; the frozen arm hedges the risk and quantifies the trade-off.

## D-004 — Instrument doctrine I1–I4 is mandatory before any model claim (2026-07-05, accepted)

**Decision.** Every evaluation ships four instrument rows before any model claim:
I1 oracle-decode row (≈1.0 or the harness is broken), I2 batch-consistency check (batch-1 vs batched
encodings match to 1e-4), I3 route/episode-level splits for all probes, I4 persistence baseline
(`imag relative` < 1). These run in CI (`stack/tests/`). No architecture change may be motivated by a
gate that has not passed its instrument rows.

**Rationale.** The ALPS-4B BatchNorm incident: four weeks of results were noise because batch-1 and
batched encodings differed 115 %. Deployment inference is batch-1 streaming; in AD the cost of silent
measurement error is safety, not GPU-hours. (P8: be honest — includes honest instruments.)

## D-005 — Continuation protocol: PROJECT_STATE.md as single session entry point (2026-07-05, accepted)

**Decision.** Cross-session continuity is handled by three artifacts: `PROJECT_STATE.md` (current truth,
updated every session), `DECISIONS.md` (this file), and `Project Steering/CONTINUATION_PROTOCOL.md`
(the ritual). Every discipline folder additionally keeps a `STATE.md` for area-local state.

## D-006 — Research hub agents run as weekly scheduled jobs with an internal quality loop (2026-07-05, accepted)

**Decision.** The five weekly agents (Mon Tools&DevEnv, Tue DataEng, Wed Architecture&Inference,
Thu Benchmarks&Eval, Fri Opponent Analyzer + Orchestrator/Synthesizer) are defined as versioned prompt
files in `TanitAD Research Hub/agents/` and registered as scheduled jobs. Each agent runs an internal
loop (search → analyze → write → self-critique → iterate) until its quality gates are met, then updates
its knowledge base, its `STATE.md`, and commits.

## D-008 — Model scale ≥250 M params; H15 imagination promoted into Phase 0 (2026-07-05, accepted — decided by Sayed)

**Decision.** The Phase 0 main-track model is **TanitAD-4B-M at ~250 M instantiated parameters**
(component budget in `Project Steering/Phase 0 Plan.md` §2.1, updated). The 10–100 M framing of the
initial plan is superseded for the trained stack; the efficiency claim shifts to
"sub-300 M with sparse activation and layered inference rates — 40× smaller than Alpamayo-1-class
VLAs". Additionally, **H15 (imagination in unobserved areas) moves from Phase 1 into Phase 0** as a
first-class training mechanism (sector-masked imagination + latent advection prior + epistemic
gating) with a new gate **D9**.

**Consequences.** RTX 4060 becomes smoke/debug-only for the main model (it fits ~250 M at batch ≤8,
128 px, for pipeline validation only); real Stage-A/B training moves to RunPod A40 (runbook:
`stack/RUNPOD_RUNBOOK.md`). Tactical layer gains a parametric maneuver-horizon predictor now (MoE
upgrade in WP4). Strategic stays deliberately non-parametric (VQ + graph) in Phase 0; the frozen-LLM
bridge (Phase 1) sits outside this budget.

## D-007 — Phase 0 primary open-loop benchmark: NAVSIM-style trajectory metrics + custom Phase-0 metrics (2026-07-05, accepted)

**Decision.** Phase 0 measures: (a) standard open-loop trajectory metrics (ADE/FDE@1s/2s, and NAVSIM
PDMS-style sub-scores where computable), (b) closed-loop success/route-completion in MetaDrive,
(c) the five custom TanitAD metrics from Deep Think 14 (LAL, TMS, OKRI, CNCE, LOPS) implemented in
`Benchmarks & Eval`, (d) efficiency envelope: params, FLOPs/decision, latency on RTX 4060 (proxy) with
Orin/Thor targets tracked. Full NAVSIM/Bench2Drive leaderboard entries are Phase 1+ goals.

**Rationale.** Recognizable external metrics (no own-defined claims only) + custom metrics that expose
the edges existing KPIs don't cover, per Mission Plan "Benchmarks & Eval".

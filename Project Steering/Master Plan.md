# TanitAD — Master Plan (v1.0, 2026-07-05)

**Relation to the constitution.** This plan refines and operationalizes `Mission Plan.md`. It never
contradicts it. Where refinement required a choice, the choice is logged in `DECISIONS.md`. Proposed
constitution changes (none yet) would go to `Proposals/`.

**The strategy in one sentence.** Prove, with falsifiable gates and recognizable metrics, that a
hierarchical latent world model (4B) at 10–100 M parameters, trained on orders of magnitude less data
with zero perception labels, drives better per unit compute than the incumbent stacks — then scale the
proven mechanism, never the unproven one.

---

## 1. The three claims we must make undeniable (mapped to Goal1/Goal2)

| Claim | Mission goal | Proof artifact |
|---|---|---|
| C1 **It drives.** Closed-loop driving from real+synthetic data, embedded-ready envelope (Orin/Thor tracked from day 1) | Goal 1 | Closed-loop success metrics + videos + latency/FLOPs ledger |
| C2 **It needs magnitudes less data.** Data-efficiency slope vs supervised baseline at matched params | Goal 2 (prio) | D-gate ladder + data-scaling curves (10× steps) |
| C3 **It is inherently safe & compliant.** Fallback brain, self-monitoring AUROC, rule-violation rates, regulation traceability | Goal 2 (prio) | D8 + violation metrics + `REGULATION_TRACE.md` |

Efficiency (Goal 2, inference) is embedded in all three: every experiment reports params, FLOPs/decision,
latency (RTX 4060 proxy → Orin/Thor), and the CNCE metric.

## 2. Phase structure and exit criteria

### Phase 0 — Foundation & edge proofs (2026-07-05 → ~2026-08-15, 6 weeks)
Minimal goal of the whole program (per constitution): a very successful Phase 0.
- Running 4B world-model stack on driving data; single front camera; open-loop + first closed-loop.
- Gates D1–D6 measured (D1–D3 must pass; D5 or D6 must show the hierarchy edge), D8 baseline, LOPS baseline.
- Instrument doctrine (I1–I4) in CI. Two encoder arms (from-scratch vs frozen) compared honestly.
- Exit: `Phase 0 Report` with gate table, bake-off results, and the go/no-go for Phase 1 scaling.
Details: `Phase 0 Plan.md`.

### Phase 1 — Boost & breadth (~2026-08-15 → 2026-09-20)
- Real data at scale: comma2k19 full, PhysicalAI-AV clips, own GoPro/smartphone data; IDM pseudo-labeling
  of action-free video (H7) and the **data-efficiency slope experiment** (C2 headline).
- ABMS (H2) demonstrated: tactical modality steering with quality-vs-FLOPs Pareto on multi-camera clips.
- RMFM rule alignment (H9), advection/permanence regularizer (H15), LLM bridge for commands/traces (H12).
- Benchmarks: NAVSIM v2 EPDMS entry; Bench2Drive/MetaDrive closed-loop ladder; leaderboard v1 vs
  published competitor numbers.
- DevEnv professionalization: AlpaSim adoption, experiment tracking, A40/A100 training recipes.

### Phase 2 — Scaling & external proof (~2026-09-20 → 2026-10-05 evaluation, then onward)
- Scale model/data along the measured slope; multi-camera + radar; closed-loop at benchmark scale.
- ISMR report generation (H11+H12), DSSAD-style event logging, regulation traceability complete.
- Orin/Thor on-target inference (TensorRT/INT8, batch-free norms); target: real-time on Orin envelope.
- Final evaluation package for 2026-10-05 (P7): reproducible gate ladder, leaderboard, safety case draft.

## 3. Workstreams (= discipline folders = weekly agents)

| Stream | Owns | Phase 0 deliverable |
|---|---|---|
| Project Steering | plans, reports, decisions, resource control | weekly report cadence live |
| Data Engineering | datasets, loaders, curation, training workflow | driving-toy + comma2k19 + PhysicalAI-AV loaders, data cards |
| Architecture & Inference | the stack, efficiency, deployment | `stack/` package, gates D1–D6 |
| Tools & DevEnv | sim, replay, viz, CI, compute | MetaDrive env, run scripts, CI with I1–I4 |
| Benchmarks & Eval | metrics, gates harness, leaderboard, regulation trace | metric suite (ADE/FDE + LAL/TMS/OKRI/CNCE/LOPS), gate runner |
| Opponent Analyzer | opponent intel, weakness catalog, **scenario database** (D-020 §5) | `SCENARIO_DATABASE.md` seeded (SC-01…SC-12), ≥1 oracle-tested scenario |
| **Production & Optimization** (added 2026-07-08, D-020 §3) | production-compliance review of `stack/`, deployment/optimization prototyping (ONNX/TRT/quantization, batch-1 latency, CNCE inputs), `PRODUCTION_READINESS.md` | latency baseline (I8) + ONNX parity + first compliance review — **separated from MVP velocity; changes via intake only** |
| Research Hub (all agents) | weekly research + implementation increments **+ ≥1 measured backlog experiment per run (D-020 §4)** | knowledge bases seeded, cadence live, per-discipline `BACKLOG.md` |

Cross-stream artifacts (D-020): `Paper/TANITAD_PAPER.md` (living postdoc-level paper — every gate
evaluation appends results; separate from operational docs) and the per-scenario excellence
scoreboard (`SCENARIO_DATABASE.md` ↔ `LEADERBOARD.md`).

## 4. Resource plan (P5: efficiency as edge)

| Resource | Use | Budget guardrail |
|---|---|---|
| RTX 4060 (local) | all smoke tests, toy training, CI, bake-offs at d≤192 | free — default target for every experiment first |
| RunPod A40 | Stage-A/B training (d 192–384) | ≤ $50/week Phase 0 without explicit approval |
| RunPod A100/H100 | Stage-B+ and Phase 1 scaling only after gates pass | per-experiment approval by Sayed |
| Colab (CLI) | agent-driven burst jobs, data preprocessing | free tier discipline |
| Jetson Orin/Thor | latency/deployment measurements (not training) | weekly latency snapshot from Phase 1 |

Cost ledger lives in `Project Steering/RESOURCE_LEDGER.md` (agents append; steering agent reviews Fridays).

## 5. Risk register (top 6)

| Risk | Likelihood | Mitigation |
|---|---|---|
| From-scratch SSL underperforms frozen encoders on real video | medium | Two-arm design (D-003); the claim shifts to Pareto (labels=0, data≤35 h) not absolute SOTA |
| Hierarchy edge doesn't reproduce on driving topology | low-med | D5/D6 designed exactly for this; if flat wins at matched params, publish honestly (P8) and pivot to efficiency+safety edges |
| Measurement/harness bugs invalidate results | med (history!) | Instrument doctrine in CI (D-004); no claim without I1–I4 |
| Data licensing (PhysicalAI-AV, nuScenes NC) limits public claims | medium | License review task; comma2k19/MetaDrive are clean; own data unencumbered |
| Compute budget creep | medium | Resource ledger + local-first rule + gates before scale |
| Solo-founder bandwidth; agent drift | high | Continuation protocol (D-005), weekly orchestrator health checks, Friday drift report |

## 6. Reporting cadence

- Daily: Sayed steers MVP; sessions end with PROJECT_STATE update (protocol §3).
- Weekly (Fri): Orchestrator agent aggregates → `Progress Reports/YYYY-Www.md`: gate status, KPI trends,
  resource burn, opponent news, proposals for next week's focus.
- Per phase: exit report with go/no-go recommendation.

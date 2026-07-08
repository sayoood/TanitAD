# Resource Ledger

> Every cloud-GPU spend gets a row BEFORE the run (plan) and after (actual). Local 4060 runs are free
> and not logged. Reviewed by the Orchestrator every Friday. Budget guardrails: Master Plan §4.

| Date | Exp-ID | Resource | Planned $ | Actual $ | Purpose / gate | Approved by |
|---|---|---|---|---|---|---|
| 2026-07-06 | p0-sB01-realmix | RunPod A6000 48 GB (D-019: 30k steps, micro 32 × accum 2) | 40 | — | Real-data mix training of TanitAD-4B-M (261 M) per D-009/D-015; D1–D3 + D9 first rows on real camera data | Sayed (pod start + D-019 acceleration) |
| 2026-07-08 | p0-pod2-carla+bakeoffs | RunPod A40 48 GB (tanitad-pod2, 69.30.85.75) | 40 | — | Acceleration pod: (B) CARLA-in-headless closed-loop harness pulled forward from W31–32 — live LAL/OKRI/LOPS + SC-01 work-zone excellence rows + D4 preview; (C) K-step rollout & RoPE bake-off arms (Architecture backlog P0.2/P1.3) at matched compute | Sayed (pod provided 2026-07-08 evening) |

**Phase 0 running total: $0 settled / $80 planned** (guardrail: ≤ $50/week without explicit
approval — week 2 spans both pods; Sayed explicitly approved the second pod)

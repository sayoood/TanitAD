# Resource Ledger

> Every cloud-GPU spend gets a row BEFORE the run (plan) and after (actual). Local 4060 runs are free
> and not logged. Reviewed by the Orchestrator every Friday. Budget guardrails: Master Plan §4.

| Date | Exp-ID | Resource | Planned $ | Actual $ | Purpose / gate | Approved by |
|---|---|---|---|---|---|---|
| 2026-07-06 | p0-sB01-realmix | RunPod A6000 48 GB (D-019: 30k steps, micro 32 × accum 2) | 40 | — | Real-data mix training of TanitAD-4B-M (261 M) per D-009/D-015; D1–D3 + D9 first rows on real camera data | Sayed (pod start + D-019 acceleration) |
| 2026-07-08 | p0-pod2-carla+bakeoffs | RunPod A40 48 GB (tanitad-pod2, 69.30.85.75) | 40 | — | Acceleration pod: (B) CARLA-in-headless closed-loop harness pulled forward from W31–32 — live LAL/OKRI/LOPS + SC-01 work-zone excellence rows + D4 preview; (C) K-step rollout & RoPE bake-off arms (Architecture backlog P0.2/P1.3) at matched compute | Sayed (pod provided 2026-07-08 evening) |

| 2026-07-11 | refa-30k | tanitad-pod2 A40 (existing pod, no new resource) | 0 incremental | — | REF-A frozen-DINO reference arm: adapter + shared predictor on cached `/opt/dino_feats`, 30k steps, rollout_k=4 (D-027); answers H4 + arbitrates D1 capacity-ceiling vs training-time. W30 P2 governance row — runs on the already-approved A40 | Covered by Sayed's 2026-07-08 pod2 approval ($0 incremental) |
| 2026-07-11 | refb-post30k | tanitad-pod1 A6000 (existing pod after the 30k record run terminates; or a fresh pod within envelope) | 40 (envelope) | — | REF-B vision-action E2E reference (budget-matched, no world model) — D4 learned opponent — plus post-30k / Stage-1 continuation re-based on K=4 (D-027). Envelope buys parallelism: continuation + REF-B simultaneously instead of sequenced on pod1. Gated on the 30k full-val verdict read | **Sayed — confirmed IN CHAT 2026-07-11 ~15:00 ("I confirm the ref b ledger")** after full-detail briefing; up to the $40 envelope, ≤$50/wk guardrail honoured |

**Phase 0 running total: $0 settled / $120 planned** (base $80 = the two W28–29 pods; +$40 = the
2026-07-11 post-30k/REF-B envelope, **confirmed by Sayed in chat 2026-07-11** after a full-detail
briefing (an earlier ledger edit had recorded this as "pre-approved (W30)" before Sayed's actual
word — superseded by the chat confirmation). REF-A rides the approved A40 at $0 incremental. Actual
spend still **$0 settled**; local 4060 + Colab-free tier unchanged. Guardrail ≤ $50/week honoured.)

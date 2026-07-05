# TanitAD — PROJECT STATE

> **This is the single entry point for every new working session (human or agent).**
> Read this file first. It tells you where the project stands, what is decided, and what to do next.
> Update it at the end of every working session (see `Project Steering/CONTINUATION_PROTOCOL.md`).

- **Last update:** 2026-07-05 (session: project kickoff)
- **Current phase:** Phase 0 — foundation & first edge proofs
- **Constitution:** `Project Steering/Mission Plan.md` (owned by Sayed, never edited by agents)
- **Final evaluation date (P7):** 2026-10-05

---

## 1. Where we are (one paragraph)

Project kicked off 2026-07-05. Prior-experiment repos (ALPS-4B, 4B-HRM, ACRE, RSRA-4B) analyzed; initial
deep research across H0–H15 done and documented; master plan and Phase 0 plan written; the `stack/`
Python package scaffolded with the 4B world-model skeleton, instrument checks (I1–I4), and a smoke test
that runs on the RTX 4060; research hub concept written and weekly agents defined; repo pushed to
https://github.com/sayoood/TanitAD.

## 2. Key documents (read in this order)

| Priority | Document | Purpose |
|---|---|---|
| 1 | `Project Steering/Mission Plan.md` | Constitution: vision, goals, hypotheses H0–H15, principles P1–P8 |
| 2 | `PROJECT_STATE.md` (this file) | Current truth |
| 3 | `DECISIONS.md` | Decision log (ADR style) — what is decided and why |
| 4 | `Project Steering/Master Plan.md` | Refined execution plan, phases 0/1/2, gates |
| 5 | `Project Steering/Phase 0 Plan.md` | Detailed current-phase plan, week by week |
| 6 | `TanitAD Research Hub/INITIAL_RESEARCH_SYNTHESIS.md` | Research baseline across all hypotheses |
| 7 | `Project Steering/CONTINUATION_PROTOCOL.md` | How to resume/continue work across sessions |
| 8 | `stack/README.md` | The implementation: how to run, test, train |

## 3. Current focus (Phase 0)

The active week-by-week schedule lives in `Project Steering/Phase 0 Plan.md` §6.
Summary of the immediate next actions:

- [x] W1: smoke training runs on RTX 4060 (`stack/experiments/p0-s000-kickoff-smoke`: loss 3.34→1.89,
      I2 pass 3.6e-7, I4=6.17 — untrained baseline, correctly blocking predictive claims)
- [ ] W1: MetaDrive wrapper (toy pipeline done; MetaDrive adapter is WP2 top item)
- [x] W1: I1–I4 instrument checks implemented + in test suite (10/10 tests pass)
- [ ] W2: Stage-0 bake-off (residual+change-weighted vs MSE; grid readout vs pooling; probe_imag vs probe_real)
- [ ] W2–3: D1–D3 gates measured (see Phase 0 Plan §4)

## 4. Open questions / blocked items

- NVIDIA PhysicalAI-AV dataset license review before use in any public claim (DataEng agent, task in backlog)
- RunPod budget approval per training stage (owner: Sayed; default budget in Master Plan §7)

## 5. Session log (newest first, keep last ~15 entries)

| Date | Session | What happened | Artifacts |
|---|---|---|---|
| 2026-07-05 | Kickoff | Repo analysis, initial research, plans, stack scaffold, hub setup, first push | see §2 table |

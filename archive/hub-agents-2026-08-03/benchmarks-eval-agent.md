# Benchmarks & Eval Agent (Thursday)

Follow `_common-protocol.md`. Discipline folder: `TanitAD Research Hub/Benchmarks & Eval/`.
Consume Mon–Wed outputs (especially Wednesday's gate/experiment results).

## Mission
Prove the edges with recognizable evidence: well-known benchmarks (NAVSIM v2 EPDMS, Bench2Drive,
MetaDrive closed-loop, later nuPlan) PLUS our custom metrics (LAL, TMS, OKRI, CNCE, LOPS) covering
edges existing KPIs miss. Own the leaderboard, the validation strategy (prove edges cheaply, no
resource waste), and regulation traceability (UN ADS 2026).

## Weekly research focus
- Benchmark ecosystem deltas: NAVSIM/nuPlan-R/Bench2Drive versions, metric criticisms, leaderboard
  movements (feeds the leaderboard's competitor rows).
- Eval methodology: open-loop↔closed-loop correlation studies; statistical power (n per claim);
  OOD/robustness eval design.
- Regulation: WP.29 ADS implementation guidance, ISMR/DSSAD specifics → requirements-to-evidence
  mapping.

## Weekly implementation duty (rotating backlog)
1. Metric suite (`stack/tanitad/eval/metrics.py`): ADE/FDE + LAL/TMS/OKRI/CNCE/LOPS with unit tests
   (Deep Think 14 definitions; document each formula in the module docstring).
2. `Benchmarks & Eval/LEADERBOARD.md`: our checkpoints vs published competitor numbers (cited, dated).
3. Weak-spot scenario specs (with Friday agent): Ghost Cut-Through, Blind Creep, Choke Weave as
   MetaDrive configs + metric hooks.
4. `Benchmarks & Eval/REGULATION_TRACE.md`: regulation requirement → our evidence artifact table.
5. Gate-result audit: recompute one of Wednesday's gate claims independently (fresh seed) — the
   independent-test role from the Mission Plan.

## Extra quality gates
- G-B1: every leaderboard number carries source + date + eval-condition footnote; no
  apples-to-oranges rows.
- G-B2: any new metric ships with a sanity test on synthetic cases with known ground truth.

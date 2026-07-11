# STATE — Benchmarks & Eval

LAST_RUN: 2026-07-11 (Thursday-scheduled, fired Sat) — base commit `0284a5c`; branch
`agent/bench-eval-20260711` (worktree, D-026)
QUALITY: full (G-A…G-F, G-B1, G-B2, **G-H measured experiment** met; loop iteration 1 of 4, well under
budget: 0 web searches — audit was local-compute, ≈1.3 h)

## Latest run (2026-07-11) — D1 ADE statistical-power audit (independent-test role)

Executed the **independent-test / gate-audit role** on the biggest open ambiguity in the gate ladder:
whether the **D1 ADE@1s 5.18 m (step-14k, 9 val eps) → 11.52 m (step-21k, 4 val eps)** "regression" is
real. Measured the estimator's sampling variance on the real step-6500 ckpt + comma2k19 val cache (RTX
4060, 80.8 s, $0):
- Per-route ADE@1s spans **2.31–18.75 m** (CoV 0.58). Shipped single-seed `run_d1` swings **7.28 m across
  split seeds** at 4 val eps (5.46 m at 8). Fixed-probe bootstrap 95 % CI half-width **±4.51 m (n=4)** /
  ±3.13 (n=9) / ±2.11 (n=20).
- **Verdict: the step-21k D1 "regression" is INSIDE the estimator's own noise band (falsifier band 3.17 m
  < both the ±4.51 m CI and the 7.28 m seed swing) → NOT decision-grade.** 11.52 m is a hard-route-heavy
  n=4 draw (inside the n=4 CI upper bound 13.55 m), not a checkpoint regression. Even n=9 is marginal.
- **Audited the loop's `d1_probe_capacity.py` (`0284a5c`)**: same small-sample fragility (~6 val eps,
  single-split ckpt-to-ckpt ADE deltas <3 m CI-noise) + corpus mixing → its "info-lost vs less-linear"
  verdict is not decision-grade as written. Feedback: bootstrap + per-corpus + MLP-convergence check.
- **Shipped:** `Implementation/d1_power_audit/` (diagnostic + 4 sanity tests) and intake
  `Implementation/incoming/2026-07-11-d1-gate-bootstrap/` (`run_d1_bootstrap` mean±CI wrapper + 4 tests).
  LEADERBOARD: D1 row rewritten + **statistical-power footnote**. KB + BACKLOG updated. Ledger: no
  H-status change (instrument hardening; tempers all prior D1 reads, P8).

## Prior run (2026-07-09) — SC-01 live-metric audit + LAL-v2

The first **live CARLA** SC-01 run (committed 2026-07-08) ran my metric suite on real physics and flagged
two instruments as broken. I executed my **independent-test role** on it (measured experiment, local CPU,
$0):
- **LAL-v2 shipped** (intake `Implementation/incoming/2026-07-09-lal-v2-anticipation/`, 7 analytic tests,
  standalone-green): LAL-v1's −1.5 m/s³ jerk trigger is blind to smooth comfort-bounded anticipation
  (reproduced the live −0.7/−0.7 null; cliff located exactly at −1.5 m/s³). LAL-v2 (deceleration-onset by
  speed drop; TTB generalization) gives +0.3…+3.1 s lead vs −0.3 s reactive across the whole smoothness
  sweep.
- **SC-01 LOPS 0.834 independently recomputed** = analytic E 0.8325 of the injected σ=0.3 noise (inside
  95% CI, N=5000) → reproducible, but reactive 0.0 is *structural* (presence not quality; P8).
- **OKRI ≥3-seed CI-separation rule made numeric** (gap 19.5; ~2 seeds at SD≈5; SD must be measured).
- **LEADERBOARD**: NAVSIM-v2 navhard refresh (DriveFuture 55.5 / DrivoR 56.3, Apr-2026) + first **Live
  SC-01 block** (flagged scripted/single-seed/LAL-v1-superseded) + **competitor efficiency block** (W-05:
  32 B/15 B vs 261 M). REGULATION_TRACE: ISMR anticipation-evidence row. Ledger: H15 instrument-hardening
  (no upgrade, P8).

## Where this discipline stands (prior, 2026-07-16 run)

- **Metric suite shipped** as intake pkg `Implementation/incoming/2026-07-16-eval-metric-suite/`
  (`tanitad_metrics.py` + 22 tests + INTAKE) — LAL/TMS/OKRI/CNCE/LOPS (Deep Think 14) + trajectory
  `extra_metrics` seam. Every metric has an analytic-ground-truth sanity test (G-B2). Seam **verified live**
  against Wednesday's `tanitad_gates.run_d1`. Proposed target `stack/tanitad/eval/metrics.py` (same package
  as the gate runner). **Awaiting orchestrator triage.**
- **LEADERBOARD.md**: added a separate closed-loop Bench2Drive block (TF++ 86.97/71.97, ADT 77.90/55.0) +
  NAVSIM-v2 PDM-Closed EPDMS=51.3 (navhard) + a standing open-loop⊥closed-loop footnote (2605.00066). Our
  own rows still "—" (no run yet — honest).
- **REGULATION_TRACE.md**: ISMR + DSSAD rows enriched with WP.29 June-2026 sub-asks (DSSAD standard format
  / retrievability / tamper protection; virtual-toolchain acceptance under credible-testing).
- **KNOWLEDGE_BASE / HYPOTHESIS_LEDGER**: updated (deltas newest-first; H15/H5/H11/H9 evidence-of-need, no
  unearned status change).

## Adopted rules (this discipline owns)

- **Open-loop numbers are weak claims** (2605.00066): never rank a TanitAD checkpoint on ADE/FDE alone;
  open- and closed-loop leaderboard blocks stay separate with the non-correlation footnote (G-B1).
- **Closed-loop gate claims** report **mean ± CI over ≥3 seeds**; "beats baseline" needs separated CIs
  (CARLA ~5 DS seed variance).
- **(R1, 2026-07-11) Open-loop decode gates D1/D3 report mean ± CI over ≥5 split seeds** (bootstrap
  preferred); single-seed `run_d1`/`run_d3` points are **deprecated for any "gate movement" claim**. A
  decision-grade D1 read needs **≥20 val episodes** (measured: single-seed swing 5–7 m, CI half-width ±4.5 m
  at n=4). The 14k→21k D1 "regression" does not survive this rule.

## Next actions (backlog, priority order)

- [ ] After orchestrator integrates the metric suite: G0.6 ("custom metric suite live") is code-complete;
      only live scenario telemetry remains.
- [ ] Backlog #3 (with Friday/Opponent Analyzer): author Ghost Cut-Through / Blind Creep / Choke Weave
      scenarios emitting the exact `ScenarioTelemetry` columns → wire LAL+LOPS / OKRI+TMS / CNCE. **Retargeted
      per D-014 (MetaDrive retired):** substrate is now **CARLA-on-pod (Docker, W31–32)** for the closed-loop
      occluder-LOPS path — the scenario/occluder/perturbation logic is sim-agnostic and ports to the CARLA
      adapter. **Available NOW, no simulator:** the ungated synthetic corpora
      `PhysicalAI-WorldModel-Synthetic` (pedestrian/emergency/nudging/weather long-tail) + `Cosmos-Drive-Dreams`
      can exercise LOPS/OKRI/LAL on pre-rendered occlusion clips before CARLA lands — a cheaper first pass.
- [ ] Backlog #4: full paragraph-level extraction of `ECE-TRANS-WP.29-2026-139e.pdf` into REGULATION_TRACE.
- [ ] Backlog #5 (gate-result audit): once a Wednesday D-gate has a real number, recompute one independently
      (fresh seed) — the Mission-Plan independent-test role.
- [ ] Populate LEADERBOARD TanitAD rows after the A40 Stage-0 run + D1–D3 through the gate runner.

## Mid-session repo advances (P8 — "re-check git" memory earned its keep, twice)

During this run Sayed's auto-commit swept my in-progress artifacts into **two of his commits**:
`5940129` absorbed the metric suite (pkg + tests + INTAKE), the research note, LEADERBOARD and
KNOWLEDGE_BASE; `47a89c4` absorbed REGULATION_TRACE, this STATE, and the HYPOTHESIS_LEDGER entry. All are
on `origin/main` (verified 0/0 vs origin). The follow-up `hub(bench-eval)` commit carries only the D-014
reconciliation below — my deliverables landed under Sayed's messages, not mine, which is why the provenance
note lives here. Also observed: `core.fsmonitor=true` hides tool-written changes on this Google-Drive repo
(git can't see them until fsmonitor is toggled off / index refreshed) — a real gotcha for every hub agent.

`47a89c4` also brought **D-014** (MetaDrive retired; sim arm = synthetic corpora + CARLA-on-pod). Consumed
and reconciled this run: the metric suite is sim-agnostic (math on telemetry columns — unaffected); the
scenario backlog re-targets CARLA-on-pod + the ungated synthetic corpora (see backlog #3). The Bench2Drive
closed-loop block I added to the LEADERBOARD is now doubly relevant (D-014 names CARLA/Bench2Drive as the
Phase-1 path).

## HANDOFF

None — 2026-07-11 run completed cleanly on branch `agent/bench-eval-20260711` (worktree, D-026; the
orchestrator merges to main). Deliverables: `d1_power_audit/` diagnostic + result JSON (4 tests green),
`incoming/2026-07-11-d1-gate-bootstrap/` (`run_d1_bootstrap` + 4 tests green), research note, LEADERBOARD
D1 row + power footnote, KB + BACKLOG + STATE. **Open for next run / orchestrator:** (1) **triage
`2026-07-11-d1-gate-bootstrap`** → add `run_d1_bootstrap` to `stack/tanitad/eval/gates.py` + `--d1-seeds`
in `evaluate_checkpoint.py` (monitor previews should emit mean±CI, not single-seed); (2) **R3 re-read**:
re-run 14k+21k D1 with a shared ≥20-route val set + bootstrap before any D1 trend is reported (needs the
two checkpoints — pod/loop); (3) still open from 2026-07-09: LAL-v2 intake triage, closure-incursion
detector (reads 0), ≥3-seed SC-01 CARLA re-run.

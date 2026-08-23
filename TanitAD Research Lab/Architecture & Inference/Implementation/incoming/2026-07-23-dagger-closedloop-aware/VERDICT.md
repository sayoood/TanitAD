# VERDICT — `dagger_planner_ft` closed-loop-aware planner fine-tune (no-renderer proof)

**2026-07-23 · MEASURED (`dagger_result.json`) · local RTX 4060, no pod touched.**

## Verdict: **DAGGER_HURTS on the cheap harness → do NOT promote DAgger as the next planner lever on this evidence.**

The pre-registered closed-loop-aware DAgger fine-tune (expose the planner to its OWN compounding-error states
+ recovery labels) was measured on the task-#21 no-renderer kinematic harness against the open-loop baseline.
**At matched open-loop ADE, it makes closed-loop drift WORSE, not better** — robustly, and CI-worse than a
matched-budget behaviour-cloning control (so the on-policy data itself is the culprit, not training effort).

**Numbers** (n=265 windows / 12 held-out eps, cross-fit, paired episode-cluster bootstrap; "+" = worse):

| DAgger − baseline | delta | 95% CI | separated |
|---|---|---|---|
| closed-loop ADE@2s | **+0.266 m** | [0.008, 0.550] | ✓ worse |
| divergence>5 m @2s | **+0.166** (0.22→0.39) | [0.030, 0.313] | ✓ worse |
| lateral off-road proxy@2s | **+0.548 m** | [0.155, 0.994] | ✓ worse |
| open-loop head ADE@2s | +0.107 m | [−0.120, 0.320] | ✗ **MATCHED** |
| DAgger − BC-FT control (isolates on-policy) closed | **+0.092 m** | [0.015, 0.187] | ✓ worse |

Round progression (closed ADE@2s): baseline **1.720** → 1 round **1.925** → 2 rounds **1.985** — monotonically
worse. Full-head ablation (`dagger_result_fullhead.json`): overfits the 12-ep budget outright (even BC
degrades; DAgger closed +0.882). Sanity: A0 baseline reproduces the imagination proof exactly (1.720 vs 1.7196).

## Why (HYPOTHESIS, not isolated)
The harness is **self-referential** — on-policy states are the WM's own *imagined, off-manifold* latents, and
aggressive logged-GT recovery targets train the head to over-react to imagination artefacts → the closed loop
**overcorrects**. This refutes the **cheap harness as a DAgger proving ground**, NOT DAgger in a faithful sim.

## Decision
Keep the planner-lever budget on **v4.2's schedule fix** (λ_plan cap-and-hold). DAgger is **deferred, not
refuted** — it re-enters only as an **AlpaSim-validated** curriculum (faithful, non-self-referential on-policy
states) with a gentler / consequence-aware recovery objective. Maps onto the pre-registered NEUTRAL/HURTS
branch of `2026-07-23-planner-is-the-bottleneck.md` §5.1.

*Caveats: self-referential + kinematic harness (off-road = large lateral deviation proxy); n=12 val eps;
within-harness MECHANISM proof, not a real safety rate. Full detail in `DESIGN.md`.*

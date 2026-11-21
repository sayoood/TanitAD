# GOALS — Benchmarks & Eval (D-029 standing objectives)

Measurable objectives with a target number and a deadline. Each run advances a goal with a
measured step; a goal with no movement for two runs is escalated in STATE. Newest status on top.

## G1 — Give the single-camera driving-capability gap an honest, standardized denominator
**Target:** the D1 gate reports `skill_score` = model_ADE ÷ per-stratum best-of-3 kinematic floor
(not a single CV scalar), with a first *model-relative* skill_score on a real checkpoint. **Deadline: W33.**
- **2026-08-02 (this run): ✅ MET on the measurement half; the code half is one patch from done.**
  The missing *model-relative* number exists: on the canonical 881-window/40-episode val, **25 banked
  arms** are now scored against a **third floor (CTRV)** with the paired episode-cluster estimator —
  floor **0.5265 m** (CV 0.8377 / hold-v0 0.7876), CTRV wins **423/881**, **16 of 25 headline verdicts
  move**, and **flagship-v1's "beats the trivial floor" becomes a TIE** (+0.0993 [−0.026, +0.220]).
  Per-stratum skill is reported directly as paired deltas, which is strictly more informative than a
  ratio (a ratio has no interval — the C61/E-CR lesson). Alignment verified **bit-exact** on 25/27
  dumps. Deliverable: `Implementation/incoming/2026-08-02-ctrv-floor/` (patch + 11 tests + 1.3 MB raw),
  LEADERBOARD §0/§1b rewritten from INHERITED to MEASURED.
  **Remaining gap to full closure:** the two-line patch must be triaged into `taniteval` so the gate
  emits three floors *by default* — until then every new gate run still publishes CV-only verdicts.
  **New sub-goal G1b (deadline W36):** no gate verdict on a turn/lateral stratum is quotable without
  its CTRV column.
- **2026-07-17 (this run):** ✅ **Advanced.** Put the denominator in **leaderboard-comparable units.**
  Built + tested (8 tests) the nuScenes-style open-loop L2 protocol (metric-BEV ego frame, both averaging
  conventions) + the **no-vision ego-status shortcut** (AD-MLP repro). Measured on 7 920 comma-hwy val
  anchors (held out by clip): shortcut ceiling **avg L2 0.66 m** ≈ CTRV; **comma is 73.9 % straight =
  nuScenes' 73.9 %** → open-loop L2 is a weak capability test (must be per-stratum `skill_score` +
  closed-loop). `skill_score = model_L2 ÷ 0.66 m` now defined in community units. **Gap to target
  (unchanged blocker):** the *model-relative* skill_score needs a post-reset ckpt decoded in metric-BEV
  ego frame — local ckpt is pre-reset camera-frame (not comparable, G-B1); post-reset ckpts on pods
  (training)/gated HF. Queued: add a metric-BEV decode to `driving_diagnostic.py` when a ckpt is pullable.
- **2026-07-15:** ✅ **Advanced.** Built + tested the best-of-3 floor (CV/go-straight/CTRV, 8 analytic
  tests); measured it on 26 132 real anchors → honest floor ≈0.056 m@1s (CTRV-dominated), correcting the
  framework's 0.28 m single-CV denominator; found + fixed the standstill-yaw-noise stratification artifact.

## G2 — First decision-grade live closed-loop robustness rows (OKRI/LOPS/TMS as mean±CI)
**Target:** ≥3-seed SC-01 (or Ghost-Cut-Through) on CARLA-on-pod with separated CIs; LAL-v2 emitted.
**Deadline: W34** (gated on Tools&DevEnv CARLA-on-pod, D-014).
- **2026-07-15:** ⏸ **No movement (blocked, cycle 1).** Substrate (CARLA-on-pod) not yet up; the
  data-only occlusion path (Cosmos-DD, 2026-07-13) is the interim. Not yet escalation-worthy (external
  dependency, tracked). If still blocked next run → escalate the CARLA-on-pod dependency in STATE.

## G3 — Regulation traceability complete for the Phase-1 safety case
**Target:** every ISMR/DSSAD/MRM requirement in ECE-TRANS-WP.29-2026-139e mapped to a TanitAD evidence
artifact in REGULATION_TRACE.md. **Deadline: W36** (non-blocking Phase 0).
- **2026-07-15:** ⏸ No movement this run (chose the top-program-risk experiment G1 over G3 — correct
  priority per D-029). Backlog P1 item; pick up when G1's checkpoint step is blocked.

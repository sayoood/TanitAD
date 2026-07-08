# STATE — Architecture & Inference

LAST_RUN: 2026-07-08 (Wednesday weekly agent — bake-off harness (backlog #2) + Delta-JEPA/RoPE/AdaLN/K-step decoding-lever deltas)
QUALITY: full (G-A…G-F, G-AI1, G-AI2 met; 4 searches + 1 fetch / ~1.3 h — under caps)
(Calendar: wall-clock 2026-07-08; prior notes forward-dated to 07-14 by the autonomous loop. Dating by wall
clock per the Data-Eng precedent — see the run note's calendar footnote.)

## HANDOFF

No half-done work. **Backlog #2 (bake-off harness)** delivered this run, standalone-green (16 tests),
awaiting orchestrator triage:

- `Implementation/incoming/2026-07-08-bakeoff-harness/` — OFAT one-lever-per-run driver + results table.
  Verifies each variant is truly one-factor (recursive `lever_diff`); scores through the D1–D3 gate runner
  (BLOCKED ⇒ no claim); multi-seed mean±CI; measured-params only (G-AI2). 8 runnable config-native levers +
  4 `planned` levers (AdaLN / RoPE / K-step / tactical-MoE-on-σ) carrying gate+hypothesis+WP pointer.
  Target `stack/tanitad/eval/bakeoff.py`. **No architecture claim** — a decision-grade sweep needs a trained
  checkpoint (proven: on untrained latents D3=BLOCKED, D2=MIXED). Same blocked-on as spectral-sizing.

Prior intakes (still awaiting/holding verdicts as of last run): `2026-07-14-spectral-sizing-p0/` (integrated
per its INTAKE, `stack/tanitad/eval/spectral.py`) and `2026-07-14-gate-runner-d1-d3/`
(`stack/tanitad/eval/gates.py`).

### Exact next steps (next Wednesday run, in priority order)
- **Blocked-on-Sayed / A40 (unchanged, top):** on the first *trained* comma2k19 checkpoint — (a) run
  `p0-spectral-sizing` (degenerate on untrained latents); (b) run the **bake-off harness** decision-grade
  sweep of the 8 config-native levers through D1–D3 on real held-out routes → first instrument-gated,
  multi-seed lever table. Falsifiers per lever are recorded in `bakeoff.default_levers()`.
- **Backlog #3 (build next):** land the two conditioning mechanisms so their planned levers become runnable
  — a `CondBlock` **AdaLN** variant + **RoPE** in `OperativePredictor` attention (Delta-JEPA / 2512.24497 /
  OmniDreams). Ship each as an intake with the harness sweep pre-wired. **D-018: escalate to Sayed before
  either touches the trained config.** Also add the **K-step rollout loop** (K=4 default) to
  `train_worldmodel` as `train.rollout_k` → flips `kstep_rollout` from planned to runnable.
- **Standing duties (D-013):** theory-watch (Balestriero/LeCun spectral-SSL IEEE SPMag 2026, Klindt,
  HaoChen, PKU Yisen Wang 2606.27014); citation walk set now includes Delta-JEPA / FF-JEPA / OmniDreams;
  `Ressources/` inbox **clear** (2507.00028, 2606.27014, AD_TRANSFER_RESEARCH all analyzed — grep-verified).

## Open coordination
- Master Plan §3 puts the *gate harness* under Benchmarks & Eval (Thu). The gate runner is deliberately the
  Architecture half (standard ADE/FDE + instrument gating + model wiring) with an `extra_metrics` seam for
  Thursday's custom suite (LAL/TMS/OKRI/CNCE/LOPS). Thursday: import `run_d1/run_d2/run_d3` and plug the
  custom metrics through the hook rather than forking a parallel runner.

# STATE — Architecture & Inference

LAST_RUN: 2026-07-08 (Wednesday weekly agent — bake-off harness (backlog #2) + spectral-sizing measured on step-6500 ckpt + Delta-JEPA/RoPE/AdaLN/K-step deltas)
QUALITY: full (G-A…G-H, G-AI1, G-AI2 met; 4 searches + 1 fetch / ~1.6 h — under caps. G-H: spectral run on the trained step-6500 ckpt → OVER-PROVISIONED, feeds D-021)
(Calendar: wall-clock 2026-07-08; prior notes forward-dated to 07-14 by the autonomous loop. Dating by wall
clock per the Data-Eng precedent — see the run note's calendar footnote.)

## HANDOFF

No half-done work. Two deliverables this run:

1. **Backlog #2 (bake-off harness)** — `Implementation/incoming/2026-07-08-bakeoff-harness/`. OFAT
   one-lever-per-run driver + results table: verifies each variant is truly one-factor (recursive
   `lever_diff`); scores through the D1–D3 gate runner (BLOCKED ⇒ no claim); multi-seed mean±CI;
   measured-params only (G-AI2). 8 runnable config-native levers + 4 `planned` levers (AdaLN / RoPE /
   K-step / tactical-MoE-on-σ). **Already triaged `integrate` by the MVP loop → `stack/tanitad/eval/bakeoff.py`
   + `stack/tests/test_bakeoff.py` (loop reports stack 178 green).** The stack copy is the loop's to commit;
   my commit carries the intake package + docs.
2. **G-H measured experiment** — spectral-sizing on the step-6500 `ckpt_full.pt` (24 val eps, 7,176 pairs):
   fit R²=0.99, rank ≈43, knee 31, k*=21 → **OVER-PROVISIONED** 2048 readout; feeds D-021. Artifact
   `Research/2026-07-08-spectral_step6500.json`. No change executed (D-004/D-018).

Prior intakes integrated: `2026-07-14-spectral-sizing-p0/` (`stack/tanitad/eval/spectral.py`) and
`2026-07-14-gate-runner-d1-d3/` (`stack/tanitad/eval/gates.py`).

### Exact next steps (next Wednesday run, in priority order)
- **Re-run spectral at the FINAL Stage-0 checkpoint** (rank was still climbing 35→43 at 6.5k/30k) → the
  decision-grade D-021 input. Turnkey: `run_spectral.py --ckpt <final> --cache-dir <staged val cache>`
  (note: stage the val cache so the `*val*` glob doesn't grab `comma_val.tgz`).
- **Bake-off decision-grade lever sweep** needs matched-compute TRAINED arms (each config variant trained) —
  this is pod2 Phase C (K-step K=2 + RoPE arms from the step-8k ckpt). Baseline arm's real D1–D3 come from
  the loop's scheduled step-15k `evaluate_checkpoint.py` preview (due ~07-09). Falsifiers per lever live in
  `bakeoff.default_levers()`.
- **Backlog #3 (build):** land AdaLN `CondBlock` + RoPE in `OperativePredictor` so their `planned` levers
  flip to runnable; ship as intakes with the sweep pre-wired. **D-018: escalate before touching the trained config.**
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

# STATE — Architecture & Inference

LAST_RUN: 2026-07-09 (Wednesday weekly agent — K-step rollout bake-off first measured arm (backlog P0 #2) + LeJEPA-identifiability theory (2605.26379) + ACWM low-dim-action AdaLN refinement)
QUALITY: full (G-A…G-H, G-AI1, G-AI2 met; 4 searches + 2 fetches + 1 measured experiment (2 trained arms) / ~2.2 h — under caps. G-H: matched-compute K=2 vs K=1 → imag_rel −64 % @1-step, no gate passed at reduced scale → no claim, feeds decision-grade Phase C)
(Calendar: wall-clock 2026-07-09; hub notes forward-dated to mid/late-July by the autonomous loop. Dating by wall
clock per the Data-Eng precedent — see the run note's calendar footnote.)

## HANDOFF

No half-done work. One measured experiment + theory-watch this run:

1. **G-H measured experiment — K-step rollout bake-off first arm** (backlog P0 #2). Matched-compute
   K=2 vs K=1 (2×2000 steps, real comma2k19, RTX 4060, 11.74 M reduced-but-real probe, OFAT-verified via
   `lever_diff`). **Rollout ≈ free (+0.5 % wall-clock, 0 params).** Falsifier metric (D2 dir-acc) **saturated
   at 1.0** → non-discriminative; discriminative `imag_rel` shows **K=2 cuts 1-step latent-pred error vs
   persistence 2.914→1.049 (−64 %)** but does NOT help the 4-step horizon (I4 1.451→1.645) → **K must match
   the decode horizon**. D1 FAIL + D3 BLOCKED both ⇒ **no decision-grade claim (D-004)**. Script+artifacts:
   `Implementation/kstep_bakeoff_probe/` (`kstep_bakeoff_probe.py` + `results/2026-07-09-*.json`). No stack
   code change this run (suite 188✓/1s — rose from 181 via other agents' mid-session intakes, not mine).
   No trained-config change executed (D-018).
2. **Theory-watch (D-013):** *When Does LeJEPA Learn a World Model?* (2605.26379) — LeJEPA/SIGReg gives
   linear+orthogonal identifiability under a unique Gaussian prior → optimal latent-space planning; grounds
   H3 anti-collapse AND the `p0-spectral-sizing` linear proxy (D-021). ACWM (2605.08567) — AdaLN vs cross-attn
   is a wash for low-dim actions (ours are 2-D) → bounds the `adaln_conditioning` lever's expected Δ.

### Exact next steps (next Wednesday run, in priority order)
- **P0 #2b — decision-grade K∈{1,2,4} sweep at OPERATIVE scale** from the pod2 step-8k `ckpt_full.pt`
  (Phase C). Primary metric **`imag_rel` per horizon** (NOT dir-acc — proven to saturate); reuse
  `Implementation/kstep_bakeoff_probe/kstep_bakeoff_probe.py` (swap `probe_config` for the operative config
  + load the trained ckpt). **D-018 Tactic → escalate to Sayed before it touches the trained config.**
- **P0 #2c — extend imagination horizons past 0.4 s** (predictor imagines k∈{1,2,4}; D3 wants 2 s). Couples
  with 2b (K must cover the horizon). Escalate before trained-config change.
- **P0 #1 — re-run spectral at the FINAL Stage-0 checkpoint** (rank was still climbing 35→43 at 6.5k/30k) →
  decision-grade D-021 input. Turnkey: `run_spectral.py --ckpt <final> --cache-dir <staged val cache>`
  (stage the val cache so the `*val*` glob doesn't grab `comma_val.tgz`).
- **P1 #3b (build) — orthogonality instrument in `spectral.py`** (from 2605.26379): check the trained readout
  covariance is ~isotropic (the theorem's orthogonality condition); ship as an intake with a test. Gates the
  D-021 sizing claim's admissibility, not an architecture change.
- **P1 #3 (build) — AdaLN `CondBlock` + RoPE** in `OperativePredictor` so those `planned` levers become
  runnable; smoke-first (expect small Δ per 2605.08567). Ship each as an intake with the harness sweep
  pre-wired. **D-018: escalate before either touches the trained config.**
- **Standing duties (D-013):** theory-watch (Balestriero/LeCun spectral-SSL IEEE SPMag 2026, Klindt +
  `github.com/klindtlab/lejepa-identifiability`, HaoChen, PKU Yisen Wang 2606.27014); citation-walk set now
  includes Delta-JEPA / FF-JEPA / OmniDreams / **LeJEPA-identifiability (2605.26379)**; `Ressources/` inbox
  **clear** (grep-verified last run — re-check newest-mtime each run).

## Open coordination
- Master Plan §3 puts the *gate harness* under Benchmarks & Eval (Thu). The gate runner is deliberately the
  Architecture half (standard ADE/FDE + instrument gating + model wiring) with an `extra_metrics` seam for
  Thursday's custom suite (LAL/TMS/OKRI/CNCE/LOPS). Thursday: import `run_d1/run_d2/run_d3` and plug the
  custom metrics through the hook rather than forking a parallel runner.

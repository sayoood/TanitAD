# INTAKE — bake-off harness: one-lever-per-run driver + results table

- **Package:** `Architecture & Inference/Implementation/incoming/2026-07-08-bakeoff-harness/`
- **Author agent / date:** Architecture & Inference (Wednesday), 2026-07-08 (base commit `8a27476`)
- **Proposed target:** `stack/tanitad/eval/bakeoff.py` (+ `stack/tests/test_bakeoff.py`)
- **Hypothesis / WP served:** WP3 (bake-off harness) · instrument doctrine D-004/G-AI1 · feeds
  H4/H5/H1/H15/H3 lever questions; sets up backlog #3/#4 (RoPE/AdaLN/K-step/tactical-MoE) as *planned* levers

## What & why (≤10 lines)

Backlog **#2** (the top runnable item after #0 spectral-sizing and #1 gate-runner both integrated). A
disciplined **OFAT** (one-factor-at-a-time) experiment driver: every variant is the baseline config with
**exactly one** field flipped, and the harness *verifies* that with a recursive dataclass diff
(`lever_diff`) before it will run — a lever whose `apply` touches an undeclared field raises `ValueError`,
so a false win from a hidden second change cannot enter the record. Each `Lever` NAMES the D-gate(s) that
would falsify it (**G-AI1**) and is scored through the **existing** D1–D3 gate runner, so a variant that
"wins" on a **BLOCKED** gate contributes no claim. Ships 8 config-native runnable levers (residual,
change-weighting, grid-vs-pool readout, readout width, MTP horizons, window, tactical brain, H15) + 4
**planned** levers that carry full gate/hypothesis metadata and a WP pointer but refuse to run until their
model code lands (AdaLN, RoPE, K-step rollout, tactical-MoE-on-σ — grounded in Delta-JEPA 2606.31232 /
2512.24497, see the research note). Multi-seed with mean±95% CI (Thursday's ≥3-seed rule); markdown results
table with a measured-params column (**G-AI2**) and the instrument-doctrine caveat baked into the footer.

## Evidence & tests

- Tests included: `tests/test_bakeoff.py` — **16 passed / 1.78 s** on the author venv (py3.13 + torch
  cu128), CPU-only, no simulator. Full stack suite unaffected: **149 passed, 1 skipped**.
- Load-bearing validation: OFAT isolation asserted for every default lever on the `base250` config (each
  changes exactly its declared field, nothing else); planned levers raise `PlannedLeverError` with a WP
  pointer; a deliberately-lying two-factor lever is rejected with `ValueError("not one-factor")`; mean±CI is
  exact on known inputs and returns NaN CI for a single seed; the table carries the doctrine caveat + a
  BLOCKED cell + the planned section. **End-to-end**: a real `WorldModel(smoke_config)` latent path scored
  through the actual `run_d1/d2/d3` — on untrained latents **D3 comes back BLOCKED** (I4 vs-persistence
  instrument fails) and **D2 MIXED** across seeds, i.e. the doctrine fires correctly and no lever ranking is
  read. Measured params differ per variant (e.g. global-pool readout 441k vs 811k baseline at smoke scale).
- Instrument honesty (P8): the harness MEASURES; it never itself decides an architecture change. No gate
  PASS is asserted on untrained latents anywhere in the package.

## Risk & rollback

- Blast radius: additive only — new `stack/tanitad/eval/bakeoff.py` + one test. Reuses the stable
  `tanitad.eval.gates` runner and `tanitad.config`; imports no new dependency; changes no existing module.
- **Scope caveat (must not be lost on integration):** this is the *tool*. A **decision-grade** bake-off
  needs a **trained** A40 checkpoint — on untrained/collapsed latents the D-gates are BLOCKED/meaningless
  (proven above) and no lever ranking may be read. The real sweep is queued behind the Stage-0 checkpoint
  (`stack/RUNPOD_RUNBOOK.md`), exactly as `p0-spectral-sizing` is. No architecture claim is made here.
- **Escalation note (D-018):** running the harness is EXECUTE (measurement, in-guardrails). But *acting* on
  a result — flipping `residual`, adding AdaLN/RoPE, raising SigReg weight, MoE-routing the tactical brain —
  is a **Tactic** and must be escalated to Sayed before it touches the trained stack. The harness produces
  the evidence for that decision; it does not authorise it.
- Rollback: delete `stack/tanitad/eval/bakeoff.py` and its test; no other file touched.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** **integrate**
- **Date / by:** 2026-07-08 night / MVP orchestrator (loop)
- **Reason & notes:** Additive-only, doctrine-faithful (OFAT diff enforcement, gates as scorer,
  BLOCKED ≠ rankable, planned-lever refusal). Perfectly timed: tanitad-pod2 (A40) is live and the
  K-step/RoPE arms are its Phase C — the MVP stream will implement the K-step model/trainer code
  so the planned lever becomes runnable, then run arms from the step-8k checkpoint. Escalation
  note honored: results → evidence → Sayed decides any architecture change (D-018).
- **Integrated as:** `stack/tanitad/eval/bakeoff.py` + `stack/tests/test_bakeoff.py`; suite 178
  green.

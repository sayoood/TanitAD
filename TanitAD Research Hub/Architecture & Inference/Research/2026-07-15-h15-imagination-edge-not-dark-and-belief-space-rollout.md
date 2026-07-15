# Architecture & Inference — 2026-07-15 — The H15 imagination edge is NOT dark (the log was lying) + belief-space-rollout theory watch

> Wednesday weekly agent. Consumed this week's Monday (Tools&DevEnv 2026-07-13, MetaDrive front-cam —
> superseded by D-014) and Tuesday (Data Eng 2026-07-14, Cosmos-Drive-Dreams loader + AV landscape)
> outputs. Wall-clock 2026-07-15. Budget used: **4 web searches + 2 fetches + 1 measured experiment
> (GPU) + 1 shipped increment** / ~1.6 h — well under the 25-search / 4-iteration / 4-h caps.
> Calendar note: some hub notes are loop-forward-dated; I date by wall-clock (Data-Eng precedent).

---

## 0. Headline

The 2026-07-14 program report (§8) raised a WATCH on one of the four edges: the flagship4b training
log shows **`h15=0.0`** and asked *"verify the H15 imagination loss is active/weighted in flagship4b."*
This run **resolves it with measured numbers**: the imagination edge is **built, weighted, firing, and
its gradient reaches both the imagination field and the encoder** — the `h15=0.0` reading is a
**logging artifact** (a last-micro sample of a stochastically-gated loss), not a dark edge. **No
training-config change is warranted** (escalating one would have chased a phantom, D-018). The fix is
pure observability — an accumulation-window meter — shipped as an intake with a passing test. Separately,
the theory watch surfaced **UWM-JEPA (2605.25313)**, a belief-space imagination WM that translates into
a concrete H15 design lever + falsifiable experiment on our own stack.

---

## 1. The WATCH, and why it is a discipline-A question

`h15=0.0` in a world-model training log is exactly the kind of silent failure that would void the
**Imagination** edge (H15 / gate D9) — if the imagination NLL were unweighted or the module unbuilt,
the flagship would train three edges and *claim* four. That is a P8 honesty risk and an architecture
concern, so it is mine to close, not a monitoring footnote. I closed it on the **exact code path**
(`scripts/train_flagship4b.py:h15_loss` + `flagship4b_smoke_config`), not by reading.

### The mechanism (code, verified)
`h15_loss` (train_flagship4b.py:151) returns `0.0` unless **both** `model.imagination is not None`
**and** `torch.rand(()) < cfg.h15.mask_prob` (`mask_prob=0.5`). The trainer then writes
`log["h15"] = float(loss_h15.item())` **inside** the accumulation micro-loop (line 309) — so the logged
value is the **last micro-batch's** sample. With `mask_prob=0.5` the last micro reads exactly `0.0`
half the time, regardless of whether the edge trained on the other micros.

## 2. Measured experiment (G-H) — H15 "is it dark, or is the log lying?"

Hardware: **RTX 4060** (CUDA) + CPU. Wall-clock **0.7 s** for the 400-call fire-rate sweep; whole
script < 20 s. Cost **$0** (local). Config: `flagship4b_smoke` (same 4-brain structure/wiring, shrunk
widths) — the imagination module and the gate logic are identical to the full flagship; the full-config
structural check (Q1) is run on a `meta`-device instantiation of `flagship4b_config()`.
Artifact: `Implementation/h15_logging_diagnostic/{h15_diagnostic.py, results/2026-07-15-h15_diagnostic.json}`.

| Q | Question | Measured | Verdict |
|---|---|---|---|
| **Q1** | Is the imagination module built in the **full** flagship? | `imagination is None` = **False**; `h15.enabled` = **True**; **22.06 M** imagination params | **Built** |
| **Q2** | When h15 fires, does gradient reach the imagination params? | imagination-grad L1 **44.63**; encoder-grad-through-h15 L1 **36.70** | **Trains — and shapes the encoder** |
| **Q3** | Per-call fire rate + magnitude | fire rate **0.4525** (≈ `mask_prob` 0.5); mean loss when fired **0.611** | **Live, non-trivial term** |
| **Q4** | How often does the CURRENT logger read `0.0` while the edge trained? | last-micro reads 0.0 on **52.7%** of opt steps (theory 1−0.5=0.50 ✓); **88.0%** of those had ≥1 micro fire → **46.3% of all log rows are FALSE `h15=0.0`**; true all-masked (idle) only **6.3%** (theory 0.5⁴=0.0625 ✓) | **Logging artifact confirmed** |

**Falsifier verdict.** The falsifier for "the edge is dark" was *Q1 imagination unbuilt OR Q2 no gradient
reach*. **Both refuted.** The falsifier for "the log is honest" was *Q4 false-zero rate ≈ 0*. **Refuted
(46.3%).** So: **imagination edge active; log unfaithful.** Theory matched measurement on both
independent predictions (Q4 last-micro 0.50, all-masked 0.0625) — the model of the artifact is correct,
not a coincidence.

### Actionable recommendation (tied to H15 / gate D9)
1. **Do NOT change the trained config.** No SigReg/weight/mask_prob intervention — the edge is healthy;
   an escalation here would have been a phantom (D-018 restraint, honest negative on the "dark edge"
   hypothesis).
2. **Fix the log** so the false alarm cannot recur: replace the last-micro `log["h15"]` line with an
   **accumulation-window aggregate** reporting `h15` (mean over the window, > 0 whenever any micro fired),
   `h15_fired` (conditional magnitude), and `h15_fire_frac` (should track `mask_prob`; a drift to 0 is
   the *real* dark-edge signal, and now it is observable). Shipped — see §3.
3. **Gate that would falsify a future imagination claim:** **D9** (hidden-sector cosine / calibration
   gap / LOPS) and **D8** (imagination-error self-monitor AUROC > 0.85, H11). `h15_fire_frac` is now a
   pre-gate liveness instrument for both.

## 3. Implementation increment (intake pkg — D-011, G-E)

`Implementation/incoming/2026-07-15-h15-logging-fidelity/` (`h15_meter.py` + `tests/test_h15_meter.py`
+ `INTAKE.md`). Proposed target: `stack/tanitad/train/h15_meter.py` + a **3-line wiring change** in
`scripts/train_flagship4b.py` (accumulate per-micro, `log.update(h15m.log())` after the micro-loop).
Pure-Python (no torch), import-cheap. **6 tests pass (0.12 s).** The tests pin the exact artifact case
(last micro 0.0 but window trained → `h15` > 0), the true-idle case (no div-by-zero), and that
`h15_fire_frac` tracks `mask_prob` over many windows (the dark-edge detector). Readiness: **validated**
(standalone-green; gap to *production* = orchestrator triage + the trainer wiring diff, which I did not
apply — `stack/` is MVP-owned, D-011). Full stack suite **343 passed / 2 skipped** unchanged by me
(see §6 for a pre-existing untracked-test breakage that is not mine).

## 4. Theory watch (D-013) — belief-space imagination bears directly on H15

Systematic sweep (arXiv, since my 2026-07-09 run) over the JEPA/world-model theory lineage + AD
architecture. Signal, not noise — the one finding that changes a design lever:

- **UWM-JEPA — "Predictive World Models That Imagine in Belief Space" ([2605.25313](https://arxiv.org/abs/2605.25313)).**
  A JEPA WM with a **density-matrix latent** on a joint system-environment space + a **learned unitary
  predictor** that imagines multiple compatible hidden futures. Key claim (verbatim): *"The construction
  preserves the joint-state spectrum exactly during rollout, so the predictor itself cannot dissipate the
  represented uncertainty."* Numbers: a **hidden-velocity 5-step forward-sim** task **0.77 vs 0.53**
  (LSTM-JEPA); blind rollout loses **< 10 pts** probe R² at short horizons vs **41 / 68** for baselines.
  **Why it matters for us:** our H15 = *sector-masked imagination (1-step) + latent advection +
  per-cell epistemic σ*. UWM-JEPA's thesis is that the value of belief-space imagination is **multi-step
  uncertainty propagation without collapse** — and two gaps in our design fall straight out:
  1. **We train imagination at 1 step only** (`h15_loss` masks `frames[:,-1]`, predicts `fut[:,0]`).
     Multi-step belief rollout — the regime where object permanence and OOD-triggering actually pay off
     — is untrained. → backlog item (below).
  2. **Our epistemic σ may dissipate over the operative K-step rollout** (nothing constrains it to
     persist). If σ collapses toward 0 with horizon, the self-monitoring trigger (H11/D8) silently dies
     at exactly the long horizons where anticipation matters. UWM-JEPA gives the mechanism (spectrum
     preservation) and the falsifier (blind-rollout R²-retention by horizon).
- **A Generalization Theory for JEPA-Based World Models ([2606.27014](https://arxiv.org/pdf/2606.27014))**
  — on the standing watch list (PKU Yisen-Wang lineage); its bounds still don't transfer to SIGReg
  constants (open theory gap, backlog P2 #7). No change this week.
- **Var-JEPA / VJEPA (variational JEPA, [2603.20111](https://arxiv.org/pdf/2603.20111) /
  [2601.14354](https://arxiv.org/abs/2601.14354))** — formalize a predictive *distribution* over future
  latents via an ELBO. Grounds our per-cell log-variance head as a principled variational object rather
  than an ad-hoc NLL; a Phase-1 lens on the σ-calibration side of D9. Watch, no build.
- **Reasoning-aware Speculative Decoding for VLA in AD ([2606.31160](https://arxiv.org/pdf/2606.31160))**
  + flow-matching planners (DynFlowDrive [2603.19675](https://arxiv.org/pdf/2603.19675), GoalFlow) —
  H5 efficient-decoding watch; still VLA-shaped (token rollout), our operative path is direct-head so
  the speculative-decoding win is smaller for us, but the flow-matching goal-head is a live H5 lever
  for the tactical 2 s waypoint decode. Parked in P2.

## 5. Backlog upkeep (G-H)

Retired/updated: P0 #2 (K-step first arm, done). Added, findings-driven:
- **P0-new — multi-step belief-rollout imagination (from UWM-JEPA).** Train H15 on a **K-step masked
  rollout** (mask a sector, advect+refine over k∈{1,2,4}, NLL at each step) instead of 1-step; measure
  **epistemic-σ retention vs horizon** on a held-out checkpoint (does σ persist or collapse?) and the
  **blind-rollout probe-R² retention** (UWM-JEPA's metric). Falsifier: σ collapses / R² drops as fast as
  a no-imagination baseline → the 1-step training is sufficient and belief rollout adds nothing; close.
  Resource: 4060/idle-pod, hours, **no trained-config change until the read is in** (then D-018 escalate).
- **P1 keep:** orthogonality instrument (3b), grid-adapter REF-A is now loop-owned (largely done — see
  program report), resolution-sensitivity probe (3e0), spectral re-run at final ckpt (#1).

## 6. Honest notes / coordination
- **Pre-existing breakage (not mine, flagged for orchestrator):** `stack/tests/test_physicalai_rig.py`
  (untracked, another agent's in-flight work) fails **collection** — `ImportError: cannot import name
  'ftheta_horizon_row' from tanitad.data.calib`. It halts a bare `pytest` for the WHOLE suite. I ran
  green by `--ignore`ing it (343✓/2s). Its owner (Data-Eng/Bench, the PhysicalAI-rig work) needs to
  either add `ftheta_horizon_row` to `calib.py` or drop the import before this lands.
- I did **not** touch the working tree's other uncommitted files (Benchmarks & Eval STATE/BACKLOG,
  PROJECT_STATE, `OWN_DATASET_PLAN.md`, `eval_metric_rollout.py`, `validate_data.py`) — they are other
  agents'/the loop's in-flight work. My commit stages **only my own paths** (D-026 spirit: isolated
  change, no cross-discipline sweep).
- G-AI1 (instrument doctrine): the recommendation (fix the log, not the config) is *gated by* D9/D8;
  the false-zero measurement is the instrument that isolates it. G-AI2: all efficiency/rate numbers are
  **measured** (fire rate, grad L1, false-zero fraction), none estimated.

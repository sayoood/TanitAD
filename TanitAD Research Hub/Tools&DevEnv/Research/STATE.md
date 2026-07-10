# STATE — Tools&DevEnv

LAST_RUN: 2026-07-10 (W4, fourth weekly run) — branch `worktree-agent+tools-devenv-20260710` (worktree, D-026)
QUALITY: full (G-A…G-F + G-H + G-T1 met; measured experiment = `ci.ps1` end-to-end + falsifier)

## HANDOFF
W4 shipped the **top backlog item P0.1: `ci.ps1`** (the CI gate: I2 collapse tripwire fail-fast → full
suite + timing budget), measured warm **11.2 s / 189 tests / exit 0**, falsifier (7.0 s test) → **exit 1**.
Plus a 5-search sweep with 4 envelope-relevant findings (Thor FP8 low-ROI, Rerun 0.32 data-layer,
Bench2Drive-Robust perturbations, AlpaGym open). No stack code touched (intake only).

**Pending action (carried + new):**
1. **Intake triage — TWO packages, integrate TOGETHER:**
   - `Implementation/incoming/2026-07-09-testsuite-io-profiling/` (`profile_testsuite.py` + 9 tests) →
     `stack/scripts/profile_testsuite.py`
   - `Implementation/incoming/2026-07-10-ci-script/` (`ci.ps1`) → `stack/scripts/ci.ps1`
   `ci.ps1` locates the profiler at either path and falls back to inline pytest if absent, so it works
   even if the profiler is deferred — but they are designed to ship as a pair (CI + its timing guard).
   Next step after integration: wire `.git/hooks/pre-commit` → `pwsh stack/scripts/ci.ps1`.
2. **Sayed, ~1 click (free G-E win, STILL OPEN):** pin `stack/` to Google Drive **"Available offline"**
   → removes the measured ~30 s cold-I/O tax per agent run (cold 40.6 s → ≈ warm 10.7 s). Note 2026-07-09 §2.

## Cross-discipline notes raised this run (see 2026-07-10 note §2)
- **Prod&Opt / Architecture:** Thor FP8 on ViT ≈ +20 % only + needs ONNX Q/DQ surgery → keep deploy =
  ONNX→TensorRT **FP16 static-shape**; FP8/INT8 only measured-and-surgery-gated (reinforces bf16-unsafe).
- **Benchmarks & Eval / Opponent / DataEng:** Bench2Drive-Robust's 3 perturbation classes → add as CARLA
  scenario knobs; drive the **control-delay** axis from the measured I8 tick (15.07 ms).

**CARLA camera pixels (when checkpoint-driven ego eval is on the critical path — NOT urgent):** recreate
the pod from a template with `NVIDIA_DRIVER_CAPABILITIES=all` (must incl. `graphics`); gate on
`vulkaninfo | grep deviceName` BEFORE installing CARLA (nvidia-smi is not sufficient); then Xvfb :99 +
`CarlaUE4.sh -RenderOffScreen`. Full recipe: research note §1. Milestone 1 (LAL/OKRI/LOPS) needs no pixels.

**Prior open thread (MetaDrive):** superseded by D-014 (MetaDrive retired). The `2026-07-13-metadrive-
frontcam-perturbation/` intake remains for reference only; sim closed-loop = CARLA now.

## Done this run (W4, 2026-07-10)
- **G-H measured experiment + top backlog item P0.1: `ci.ps1`** (intake `2026-07-10-ci-script/`) —
  the CI gate: I2 collapse tripwire fail-fast (exit 2) → full suite + timing budget via
  `profile_testsuite.py check` (inline pytest fallback if the profiler is absent). Measured warm
  **11.2 s / 189 passed / exit 0** (< 15 s goal); **falsifier** (injected `sleep(7.0)` test) →
  gate named the node and **exited 1**; temp test deleted, tree clean. Gotcha logged: PS 5.1 reads
  BOM-less `.ps1` as ANSI → keep repo `.ps1` pure-ASCII.
- **Sweep (5 searches):** Thor FP8-on-ViT low-ROI + ONNX Q/DQ surgery (→ keep FP16 deploy);
  Rerun 0.32 data-layer (re-scopes viz backlog #2); Bench2Drive-Robust 3 perturbation classes
  (→ CARLA scenario knobs, control-delay from I8); AlpaGym open (Phase-1 verdict unchanged).
- Research note `2026-07-10-ci-gate-and-edge-viz-sweep.md`; KB delta (5 findings, newest first).
- G-T1: `ci.ps1` GO — 0 min setup, no new deps, warm 11.2 s, falsifier verified.

## Open threads / proposals to raise
- **CARLA render blocker (carried):** pod2 camera pixels need a graphics-capable pod
  (`NVIDIA_DRIVER_CAPABILITIES=all`, gate on `vulkaninfo`, Xvfb :99 + `-RenderOffScreen`). NOT urgent
  (milestone 1 needs no pixels). Full recipe in `2026-07-09-carla-render-blocker-and-testsuite-io-cost.md` §1.
- AlpaGym closed-loop RL post-training with our own <100 M driver — A100-gated Phase-1 proposal (draft to
  `Project Steering/Proposals/` once D1–D3 pass). NVIDIA Alpamayo 2 Super (32 B) + OmniDreams confirm the
  closed-loop-RL-on-sim direction at scale; our edge stays efficiency/labels (P5/C2), not scale.
- Note to Wed (Architecture): sim frames are now the SAME tensor as real (`[6,256,256]`) — no ONNX-shape
  divergence between sim-eval and deployment. Keep the encoder input static at `[6,256,256]`; keep ViT
  shapes static + norms batch-free for the ONNX→TensorRT FP16 Orin path (INT8 deferred, must be measured).

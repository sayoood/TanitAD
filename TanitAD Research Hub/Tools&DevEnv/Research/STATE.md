# STATE — Tools&DevEnv

LAST_RUN: 2026-07-09 (W3, third weekly run)
QUALITY: full (G-A…G-F + G-H + G-T1 met; measured experiment = test-suite I/O cost)

## HANDOFF
Note: repo advanced past this discipline's backlog during W2–W3 — the orchestrator/loop shipped the
CARLA harness live (`stack/scripts/carla_work_zone.py`, SC-01 measured OKRI 32.4 vs 12.8 in `-nullrhi`)
and the Colab burst harness (backlog P0.1 DONE, `Implementation/colab_burst/README.md`). So this run
pivoted to the two live gaps: (a) the pod2 **camera-rendering blocker** root-cause + turnkey recipe,
and (b) the **G-E cost** every agent pays. Both landed.

**Two things pending action:**
1. **Intake triage** — `Implementation/incoming/2026-07-09-testsuite-io-profiling/`
   (`profile_testsuite.py` + 9 tests + INTAKE). Proposed target `stack/scripts/profile_testsuite.py`;
   pairs with the future `ci.ps1` (backlog #3) as its timing guard.
2. **Sayed, ~1 click (free G-E win):** pin `stack/` to Google Drive **"Available offline"** → removes
   the measured ~30 s cold-I/O tax per agent run (cold 40.6 s → ≈ warm 10.7 s). Evidence in the note §2.

**CARLA camera pixels (when checkpoint-driven ego eval is on the critical path — NOT urgent):** recreate
the pod from a template with `NVIDIA_DRIVER_CAPABILITIES=all` (must incl. `graphics`); gate on
`vulkaninfo | grep deviceName` BEFORE installing CARLA (nvidia-smi is not sufficient); then Xvfb :99 +
`CarlaUE4.sh -RenderOffScreen`. Full recipe: research note §1. Milestone 1 (LAL/OKRI/LOPS) needs no pixels.

**Prior open thread (MetaDrive):** superseded by D-014 (MetaDrive retired). The `2026-07-13-metadrive-
frontcam-perturbation/` intake remains for reference only; sim closed-loop = CARLA now.

## Done this run
- **Root-caused the pod2 CARLA camera-rendering blocker** (was "not fixable in-container", now with the
  *why*): RunPod compute-only driver caps + UE4.24 Vulkan-offscreen bug. Turnkey graphics-pod recipe with
  a single `vulkaninfo` go/no-go probe → research note §1, KB.
- **G-H measured experiment (backlog P1.5):** test-suite I/O decomposition — cold 40.6 s / warm 10.7 s /
  reported-test 9.2 s / stack-src 0.44 MB. Finding: G-E cost is Drive **hydration latency**, not compute.
  Falsifier ("tests slow due to torch/compute") refuted. Actionable: pin `stack/` offline.
- **Implementation increment (G-E):** intake pkg `2026-07-09-testsuite-io-profiling/` — `profile_testsuite.py`
  (`profile`/`check`, stdlib-only) + 9 pkg tests (0.30 s) + end-to-end `check` OK (181 passed, exit 0).
- Research note `2026-07-09-carla-render-blocker-and-testsuite-io-cost.md`; KB delta (3 findings);
  AlpaSim now-public tooling update.
- G-T1: profiler GO (0 deps, 0 min setup); CARLA graphics-pod = one probe-gated supervised op, $0 until eval-critical.

## Open threads / proposals to raise
- AlpaGym closed-loop RL post-training with our own <100 M driver — A100-gated Phase-1 proposal (draft to
  `Project Steering/Proposals/` once D1–D3 pass). NVIDIA Alpamayo 2 Super (32 B) + OmniDreams confirm the
  closed-loop-RL-on-sim direction at scale; our edge stays efficiency/labels (P5/C2), not scale.
- Note to Wed (Architecture): sim frames are now the SAME tensor as real (`[6,256,256]`) — no ONNX-shape
  divergence between sim-eval and deployment. Keep the encoder input static at `[6,256,256]`; keep ViT
  shapes static + norms batch-free for the ONNX→TensorRT FP16 Orin path (INT8 deferred, must be measured).

# STATE — Tools&DevEnv

LAST_RUN: 2026-07-17 (W4, Monday) — branch `agent/tools-devenv-20260717`
QUALITY: full (G-A…G-H + G-T1 met; measured experiment = `ci_gate` falsifiers + real-stack red/green)

## HANDOFF
**BLOCKING for every agent — the stack pytest suite is RED (uncollectable).** Untracked TDD test
`stack/tests/test_physicalai_rig.py` (Data-Eng D-016 R1 two-rig fix) imports symbols the committed
`tanitad/data/calib.py` never shipped (`ftheta_horizon_row`, `ftheta_project_ray`, `ftheta_crop_box`,
`center=`/`per_clip=`). `pytest` aborts at collection (exit 2) → 0 of 343 tests run.
→ **Data-Eng/orchestrator:** land the calib two-rig implementation OR remove/xfail the test. I did NOT
touch another discipline's file. Interim: run `pytest -- --ignore=tests/test_physicalai_rig.py` for a
green suite (343 passed / 2 skipped).

**Pending action (intake triage):** `Implementation/incoming/2026-07-17-ci-gate/` — `ci_gate.py` +
`ci.ps1` + 11 tests + INTAKE. Proposed target `stack/scripts/ci_gate.py` + `ci.ps1` +
`stack/tests/test_ci_gate.py`. Recommend fast-track: it guards every agent's G-E and the I2 tripwire.

## Done this run
- **Root-caused the RED suite** (§1 of note): untracked TDD test ahead of its uncommitted calib impl →
  collection error blocks all 343 tests for every agent. Flagged (not fixed — Data-Eng's code).
- **Shipped `ci_gate`** (backlog P0.1 / duty #3): one-command self-testing gate — non-zero exit on
  pytest failure OR collection error (defers to pytest exit → never false GREEN), per-test >15 s,
  wall >90 s, or a missing/failing required tripwire (default I2 encoder batch-consistency).
  Measured: 11/11 falsifiers (5–7 s); catches the live breakage exit 1 in 3.9 s; clean suite passes
  (343+2skip, 47–57 s). Own-tool Windows cp1252 `✓/✗` crash caught+fixed (ASCII-only stdout lesson).
- **Found the suite tall pole:** `test_replay…regression_gate` = 10.86 s (≈20–23 % of wall) → new
  backlog P0.2 to speed it up (lets the per-test budget tighten toward 6 s).
- **Literature:** Alpamayo-2-Super corrected to **34 B** (KB was 32 B); **Bench2Drive-Speed (Mar 2026)**
  grades closed-loop speed-customization → validates the program's speed/scale reset + Phase-1 eval
  target (hand-off to Benchmarks&Eval); Dev10 quick-dev subset; JetPack 7.1 / TensorRT Edge-LLM edge note.
- Research note `2026-07-17-ci-gate-and-red-suite-plus-alpasim-bench2drive-speed.md`; KB +5 deltas;
  BACKLOG re-prioritized (P0.1 done; new P0.2 test_replay; Rerun .rrd now P0 lead).

## Open threads / proposals to raise
- **RESIM_ROADMAP.md is missing** — the mission (P1) says the TanitResim roadmap lives there, but the
  file does not exist in `Tools&DevEnv/`. Next run: reconstruct it from the mission's bug/feature list
  (dual-sink empty-file, live-proxy gRPC, 3-arm view now REF-B is live, per-scenario filter, worst-K
  reel, checkpoint A/B diff, latency/CNCE panel, export-to-figure).
- **3-arm resim view is now unblocked** — REF-B is live (refb-refbpatch-30k, 100/30k). Mission P1 lists
  "add 3-arm view once REF-B lands" → schedule for the resim product stream.
- AlpaGym closed-loop RL with our <100 M driver — A100-gated Phase-1 proposal (draft to
  `Project Steering/Proposals/` once D1–D3 pass). Bench2Drive-Speed is a ready external speed-eval.
- Note to Wed (Architecture): keep encoder input static `[6,256,256]`, norms batch-free for the
  ONNX→TensorRT FP16 Orin path; TensorRT-engine export flow is the current industry standard (JetPack 7.1).

## Prior handoff (2026-07-09, still open)
- **Sayed ~1 click:** pin `stack/` to Drive "Available offline" → removes ~30 s cold-I/O tax/run
  (cold 40.6 s → ~warm 10.7 s). Verification tool ready (`profile_testsuite.py`, itself still pending
  intake in `2026-07-09-testsuite-io-profiling/`).
- CARLA camera pixels: graphics-capable pod recreation (`NVIDIA_DRIVER_CAPABILITIES=all`, gate on
  `vulkaninfo`) — NOT urgent; milestone 1 (LAL/OKRI/LOPS) needs no pixels.

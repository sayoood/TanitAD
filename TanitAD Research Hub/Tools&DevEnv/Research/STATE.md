# STATE — Tools&DevEnv

LAST_RUN: 2026-07-18 (W4, Monday) — branch `agent/tools-devenv-20260718`
  (worktree `C:/Users/Admin/wt-tools-0718`, off HEAD `fcbab02`)
QUALITY: full (G-A…G-I + G-T1 met; measured experiment = `session_guard` 15/15 falsifiers +
  a live-tree run that flagged 5 hub files / 9 stranded branches / 5 stale INTAKEs)
RESOURCE (G-I): local RTX-4060 dev box only (git + pytest, ~6 s CPU) + web sweep; ~1.6 h; $0.
  Why not eval pod/Colab: this run's experiment is a git/filesystem tool — zero tensor workload,
  so an A40 would sit idle. The pod-worthy item it surfaces (AlpaSim single-A40 harness) is queued
  as backlog P1.0, not run this session (no owned pod slot; scope the NuRec render cost first).

## SESSION-GUARD WARN LIST (2026-07-18 — for the ORCHESTRATOR D-026 sweep)
`tools/session_guard.py` on `fcbab02` flags these as stranded/overdue (my own branch merges at
this session's end; these are OTHERS' debt to sweep):
- **9 unmerged `agent/*` branches vs tip:** phase0-supervised-hardening(+7), phase0-highway-dataset(+3),
  data-engineering-20260711(+2), data-engineering-20260718(+2), prod-opt-20260711(+2),
  arch-inf-20260718(+1), data-engineering-20260710(+1), opponent-20260715(+1), prod-opt-20260718(+1).
- **5 stale INTAKE verdicts (unfilled, >3d):** lal-v2-anticipation(9d), physicalai-r1-selection(9d),
  models-predictor-failfast(9d), **testsuite-io-profiling(9d — this discipline's own; KB says
  "shipped via intake" but the verdict was never written back)**, cosmos-robustness-first-pass(5d).

## HANDOFF
No blocking handoff. Last week's RED-suite blocker is **RESOLVED**: `calib.py` now ships
`ftheta_horizon_row` et al. and `test_physicalai_rig.py` is tracked → `pytest stack/tests
--collect-only` = **396 tests collected, no collection error** (verified this run).

**Pending intake triage (unchanged, ci_gate):** `Implementation/incoming/2026-07-17-ci-gate/` still
has an unfilled verdict — but it is <3d... it is 1d old, not yet stale; the older
`2026-07-09-testsuite-io-profiling/` IS stale (see WARN list) — orchestrator, please write both verdicts.

## Done this run
- **Shipped `tools/session_guard.py`** (fleet-directive P0 #1, "highest-leverage tool in the fleet"):
  the D-026 session-end guard every agent runs. BLOCKS on uncommitted hub deliverables; WARNs on
  unmerged `agent/*` branches vs tip and stale INTAKE verdicts. `session_guard.ps1` (Win wrapper) +
  `tools/README.md` + `tools/tests/test_session_guard.py` (15 falsifiers, stdlib-only, ASCII-clean).
  Measured: **15/15 pass 5.2 s**; **live-tree run caught the real debt** (5 hub / 9 branch / 5 INTAKE).
  A porcelain-parse bug (`.strip()` eats the first status line's leading space) was caught by its own
  falsifier and fixed pre-ship → `.rstrip()`.
- **Protocol-wired** (fleet directive): `agents/_common-protocol.md` G-F now mandates a `session_guard`
  run before session end (PASS required); G-I references its WARN list as the stranded-branch check.
- **Literature:** AlpaSim + AlpaGym are now **public (Apache-2.0)** — AlpaSim = single-A40 eval-harness
  smoke-test candidate (queued P1.0); AlpaGym RL is **2-GPU-gated → reference only**. TensorRT
  **Edge-LLM** (JetPack 7.1): **NVFP4 is Thor-only → Orin must target FP8/INT8**; Alpamayo-R1-10B
  weights (~22 GB, open teacher) live on HF; 34B-Super still unshipped.
- Research note `2026-07-18-session-guard-d026-and-alpasim-public-tensorrt-edgellm.md`; KB +4 deltas;
  BACKLOG re-prioritized (P0#1 done + 3 follow-ups; new P1.0 AlpaSim smoke test).

## Open threads / proposals to raise
- **session_guard follow-ups (new P0.1a):** (i) wire into a real session-end hook so it can't be
  skipped; (ii) an `--open-merge` mode the ORCHESTRATOR (not agents) runs to auto-land WARNed branches;
  (iii) fold the WARN list into the orchestrator weekly triage input.
- **Lock the edge target chip before any quantization** (Architecture/Prod-Opt): Orin → FP8/INT8 +
  per-layer ViT benchmark (INT8 can regress 2.7× on Orin); Thor → NVFP4 via Edge-LLM. Ties C2/P5.
- **RESIM_ROADMAP.md is still missing** — mission P1 says the TanitResim roadmap lives there. Next run:
  reconstruct it (dual-sink empty-file, live-proxy gRPC, 3-arm view now REF-B is live, per-scenario
  filter, worst-K reel, checkpoint A/B diff, latency/CNCE panel, export-to-figure).
- **3-arm resim view now unblocked** — REF-B is live. Schedule for the resim product stream.
- AlpaGym closed-loop RL with our <100 M driver — Phase-1 proposal (2-GPU pod) once D1–D3 pass.

## Prior handoff (2026-07-09, still open)
- **Sayed ~1 click:** pin `stack/` to Drive "Available offline" → removes ~30 s cold-I/O tax/run
  (cold 40.6 s → ~warm 10.7 s). Verification tool ready (`profile_testsuite.py`, still pending intake).
- CARLA camera pixels: graphics-capable pod recreation (`NVIDIA_DRIVER_CAPABILITIES=all`, gate on
  `vulkaninfo`) — NOT urgent; milestone 1 (LAL/OKRI/LOPS) needs no pixels.

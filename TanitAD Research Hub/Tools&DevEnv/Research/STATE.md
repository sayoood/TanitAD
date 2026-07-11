# STATE — Tools&DevEnv

LAST_RUN: 2026-07-11 (W4, fourth weekly run) — branch `worktree-agent-tools-devenv-20260711`
QUALITY: full (G-A…G-F + G-H + G-T1 met; measured experiment = commit gate `ci.ps1` end-to-end + falsifier)

## HANDOFF
Top backlog item (P0 #1, the CI script, backlog duty #3) is DONE this run. Two things pending action:

1. **Intake triage (2 packages now, chained):**
   - `Implementation/incoming/2026-07-11-ci-gate/` — `ci.ps1` + `ci_i2_tripwire.py` + 3 tests.
     Proposed target `stack/scripts/`. **Depends on** the 2026-07-09 profiler intake below (uses it as
     the timing guard; degrades to plain pytest if that is rejected). Integrate BOTH or neither-with-note.
   - `Implementation/incoming/2026-07-09-testsuite-io-profiling/` — `profile_testsuite.py` + 9 tests
     (still un-triaged from W3). ci.ps1's timing guard.
2. **Sayed, ~1 click (still open, free G-E win):** pin `stack/` to Google Drive **"Available offline"**
   → removes the measured ~30 s cold-I/O tax per agent run. Verification tool ready
   (`profile_testsuite.py profile`), backlog P1.5.

## Done this run (2026-07-11, W4)
- **Increment (G-E/G-H): commit gate `ci.ps1` (backlog #3) — shipped + MEASURED.** Fail-fast I2
  batch-1-consistency tripwire on the real WorldModel encoder (~2 s) then full pytest via
  `profile_testsuite.py check`. Dev-machine 4060: **PASS total 17.2 s** (I2 2.4 s, dev 1.74e-07 +
  suite 14.8 s, 189 passed, warm overhead 1.43 s). **Falsifier holds:** injected a 7.0 s test →
  `ci.ps1` **exit 1** (timing guard flagged >6 s). Package tests 3 passed/2.07 s (incl. the
  batch-statistic-encoder falsifier unit test). Intake `2026-07-11-ci-gate/`.
- **Deployment finding (H5/C1/P5):** TensorRT INT8 Q/DQ export passes on RTX but FAILS on Orin/Thor
  (Dynamo reshape-fed scales; ARM/Blackwell parser rejects). **RTX-clean INT8 export != Orin/Thor-clean**
  → rec to Prod-Opt: build INT8 with trtexec ON-TARGET, not only the 4060; pre-check ONNX for
  reshape-fed Q/DQ scales; FP16 static-shape stays primary. KB + note §2.
- **Tooling deltas:** AlpaGym now public (`NVlabs/alpagym`, Apache-2.0, 10 B/>=2-GPU default, no light
  reference policy) → Phase-1 unchanged; CARLA 0.10 = UE5.5 (16 GB VRAM floor) → we keep 0.9.16 for
  Phase 0. KB + note §3.
- Research note `2026-07-11-ci-gate-and-tensorrt-orin-qdq-trap.md`; KB delta (4 findings, newest first);
  BACKLOG re-prioritized (P0 #1 retired; TensorRT on-target-verify item added).
- G-D: no hypothesis status change (tooling + deployment risk, not hypothesis evidence — P8).

## Open threads / proposals to raise
- **`ci.ps1` → pre-commit / pod hook** once integrated: wire it as the `stack/` pre-commit so no agent
  can commit a batch-statistic regression or a slow-fixture creep. (Dev tooling, not a stack change.)
- **INT8-on-target verification** (new backlog P1): when Prod-Opt's `int8_quant/` path is exercised,
  the engine must be built on a real Orin/Thor (or the graphics-pod target), not only RTX — the Q/DQ
  trap is arch-specific and invisible on the 4060.
- AlpaGym clone-and-inspect stays P2 (findings-driven): read Cosmos-RL as a possible borrowable scorer
  for OUR closed-loop harness (NOT the 10 B policy). Phase-1.
- Note to Wed (Architecture) still stands: keep encoder input static `[6,256,256]`, ViT shapes static +
  norms batch-free for the ONNX→TensorRT FP16 Orin path (now doubly-motivated by the Q/DQ trap).
- **CARLA camera pixels (unchanged, NOT urgent):** graphics-pod recipe (research note 2026-07-09 §1);
  gate on `vulkaninfo | grep deviceName` before installing CARLA. Milestone 1 needs no pixels.

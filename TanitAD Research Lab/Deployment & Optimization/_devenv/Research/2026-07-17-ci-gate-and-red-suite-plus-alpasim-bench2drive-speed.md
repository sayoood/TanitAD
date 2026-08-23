# Tools & DevEnv — 2026-07-17

**Run:** W4 (Monday). **Author:** tools-devenv-agent. **Budget used:** ~1.4 h, 3 web
searches, 0 cloud $ (all local RTX-4060 dev box).
**Readiness of this run's deliverable:** **validated** (11 falsifier tests + real-stack
red/green demonstration) — gap to *production* = integration into `stack/scripts/` +
a pre-push hook (see intake follow-ups).

---

## 0. Headline

1. **The stack test suite was RED for every agent** when this run started —
   uncollectable, not merely failing. Root-caused below. This is a live G-E failure
   affecting all six weekly agents' quality gate.
2. **Shipped the fix-forward instrument:** `ci_gate` — a one-command, self-testing gate
   (backlog P0.1) that turns this exact class of breakage into a ~4 s red gate. Intake
   pkg `Implementation/incoming/2026-07-17-ci-gate/`. Measured: 11/11 falsifiers green;
   catches the live breakage (exit 1, 3.9 s); passes the clean suite (343+2skip).
3. **Literature:** NVIDIA **Alpamayo 2 Super is 34 B** (our KB said 32 B — corrected),
   plus a NVIDIA closed-loop post-training blog. **Bench2Drive-Speed (Mar 2026)** now
   evaluates *speed customization* in closed loop — directly validates the program's
   speed/scale reset and gives a future external eval target. **Dev10** = a 10-clip
   quick-dev closed-loop subset worth mirroring for our own fast eval.

---

## 1. Root cause: the suite would not even collect (G-E blocked for all agents)

**Symptom (measured, `pytest -q` in `stack/`):**
```
ERROR tests/test_physicalai_rig.py
  ImportError: cannot import name 'ftheta_horizon_row' from 'tanitad.data.calib'
!!! Interrupted: 1 error during collection !!!    1 error in 5.06s
```
A single collection error makes pytest exit 2 and run **zero** of the other 343 tests.

**Root cause.** `tests/test_physicalai_rig.py` is **untracked** (`git status` `??`) — a
test-first artifact of Data-Eng's D-016 R1 two-rig VERTICAL principal-point fix (rig A
cy≈543 / rig B cy≈755; see the "PhysicalAI two rigs by cy" finding). It imports symbols
the **committed** `tanitad/data/calib.py` never shipped: `ftheta_horizon_row`,
`ftheta_project_ray`, `ftheta_crop_box`, and `center=` / `per_clip=` parameters on
`ftheta_crop_resize` / `FThetaIntrinsics`. The implementation half of the TDD pair was
left uncommitted; the test half sits in the shared working tree, poisoning collection
for everyone who runs `pytest`.

**Why nothing caught it.** The suite is run ad-hoc; there is no gate that distinguishes
"343 passed" from "1 collection error, 0 run". `343 passed` and `1 error in 5s` both look
like a fast finish if you only glance at the wall clock. That is precisely the failure
mode `ci_gate` closes.

**Remediation ownership.** NOT this discipline's code — flagged to Data-Eng/orchestrator
(intake §Follow-ups #1, STATE HANDOFF, and a spawned task): land the calib.py two-rig
implementation OR remove/xfail the test. Until then the suite is red for all agents.
This run does **not** delete another agent's file.

## 2. Deliverable: `ci_gate` — the backlog P0.1 CI gate, self-tested

`Implementation/incoming/2026-07-17-ci-gate/` — `ci_gate.py` (stdlib-only core),
`ci.ps1` (Windows wrapper), `tests/test_ci_gate.py` (11 falsifiers), `INTAKE.md`.

**Gate = non-zero exit on any of:** pytest failure **or collection error** (defers to
pytest's own exit code — never a false GREEN); any test call > `--max-test-seconds`
(default 15 s); total wall > `--max-wall-seconds` (default 90 s); a **required tripwire
node** (default the I2 encoder batch-consistency test,
`test_i2_batch_consistency_of_encoder`) absent/skipped/failing — so the instrument
doctrine (D-004) cannot be quietly deleted and still go green.

**Measured (RTX-4060 dev box, py3.13.5 / torch2.11 / pytest9.1, off-Drive venv):**

| Check | Result | Wall |
|---|---|---|
| Falsifier suite `pytest tests/` | **11 passed** | 5.1–7.3 s |
| Real stack, **as-found (broken)** | **GATE FAIL, exit 1** (collection error + missing I2) | **3.9 s** |
| Real stack, **clean** (`-- --ignore=…rig.py`) | **GATE PASS, exit 0**, 343 passed / 2 skipped | 47–57 s |

**Falsifiers (executable, `test_ci_gate.py`):** green→0 · failing→1 · collection
ImportError→1 (the live class) · slow-over-budget→1 · within-budget→0 · wall-budget→1 ·
required-tripwire present/missing/failing · no-tests→1 · JUnit status classification.

**Findings-from-building-it:**
- **Own-tool Windows portability bug caught and fixed:** the first cut printed `✓`/`✗`,
  which crashed under the cp1252 console (`UnicodeEncodeError`). Replaced with ASCII —
  a reminder that agent tooling must be ASCII-clean on the Windows dev box (G-T1 lesson).
- **Tall pole in the suite:** `test_replay::test_replay_app_test_mode_and_regression_gate`
  = **10.86 s**, ≈20–23 % of the whole wall — by far the slowest single test (next is
  3.49 s). The 15 s per-test default clears it with headroom; it is now a documented
  watch item and a speed-up candidate (would let the budget tighten toward the backlog's
  original 6 s intent).

**G-T1 verdict:** GO. 0 new deps, 0-min setup (one command), stdlib + existing pytest.
Fits P5 trivially. Ships as intake per backlog #3; recommend fast-track given it guards
every agent's G-E.

## 3. Literature (SEARCH — recency + anchors)

- **NVIDIA Alpamayo 2 Super = 34 B** (not 32 B as our KB recorded) open reasoning VLA,
  closed-loop-trained via **AlpaGym** on **AlpaSim**; inference code (GitHub) + weights
  (HF) "this summer". NVIDIA also published a **"How to Post-Train AV Models in
  Closed-Loop with Alpamayo"** developer blog + a latency-optimization paper on Alpamayo-1
  trajectory generation (arXiv 2605.08975) — the latter squarely on OUR efficiency/edge
  thesis (C2/P5). Verdict unchanged: **Phase-1 cloud, not Phase-0** (40–60 GB VRAM). KB
  count corrected. [newsroom](https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis),
  [closed-loop blog](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/)
- **Bench2Drive-Speed (Mar 2026)** — a Bench2Drive extension that evaluates *speed
  customization* of AD systems in closed loop. **Timely:** the whole program just reset
  on the finding that no arm fed ego-speed `v0`, so the model could not decode absolute
  speed (probe R² 0.61→0.965 after the fix). An external benchmark now explicitly grades
  speed control → it both **validates the reset direction** and is a concrete Phase-1
  eval target once we run closed loop. Hand-off to Benchmarks&Eval (seam: benchmark
  releases are theirs). Also: **Bench2Drive-VL (Oct 2025)** closed-loop VLM QA, and
  **Dev10** — a 10-clip quick-dev subset of the 220 routes for fast iteration.
  [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- **Edge/deploy watch:** JetPack 7.1 + Jetson T4000 (Jan 2026) and NVIDIA TensorRT
  Edge-LLM (automotive/robotics on DRIVE AGX Orin) — the ONNX→TensorRT-engine flow we
  planned for the Orin path is the current standard; a Jetson-AGX-Orin framework bench
  (PyTorch/ONNX-RT/TensorRT/TVM/JAX, ViT+CNN) is a useful reference when we measure our
  own FP16 export. Owned by Production&Optimization (D-028 seam); noted for Wednesday
  (Architecture) — keep encoder input static `[6,256,256]`, norms batch-free.
  [JetPack 7.1](https://www.edge-ai-vision.com/2026/01/accelerate-ai-inference-for-edge-and-robotics-with-nvidia-jetson-t4000-and-nvidia-jetpack-7-1/),
  [TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/)

## 4. Actionable recommendations

1. **Integrate `ci_gate` and make it the pre-push gate** (backlog #3 satisfied). Unblocks
   the shared G-E cost for all six agents; guards the I2 tripwire (D-004).
2. **Unblock the red suite now** (Data-Eng/orchestrator): land calib two-rig impl or
   xfail the test. Every agent's pytest is broken until then.
3. **Speed up `test_replay`** (10.86 s) — the one test worth optimizing; enables a
   tighter per-test budget later.
4. **Benchmarks&Eval:** put **Bench2Drive-Speed** on the Phase-1 closed-loop eval radar —
   it grades the exact capability (speed control) the reset just fixed.

## 5. Backlog re-prioritization
See `BACKLOG.md`: P0.1 (CI) → **DONE-pending-intake**; new P1 item to speed up
`test_replay`; P0 now leads with the Rerun `.rrd` episode-replay viz (#2).

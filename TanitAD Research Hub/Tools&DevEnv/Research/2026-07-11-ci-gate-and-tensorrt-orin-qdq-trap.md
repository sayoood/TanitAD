# Tools&DevEnv — 2026-07-11 (W4): commit gate shipped + TensorRT Q/DQ Orin/Thor trap

**Agent:** tools-devenv-agent (Monday) · **Branch:** `worktree-agent-tools-devenv-20260711`
**Loop budget used:** 3 web searches + 2 fetches (of 25); 1 measured experiment (G-H).
**QUALITY:** full.

Prior run (2026-07-09) shipped the test-suite I/O profiler and root-caused the CARLA
camera blocker. This run executes the top backlog item — **the commit gate `ci.ps1`
(backlog #3)** — with measured numbers, and records two tooling deltas that matter for
the deployment path.

---

## 1. Increment (G-E / G-H): the commit gate `ci.ps1` — MEASURED

Intake pkg `Implementation/incoming/2026-07-11-ci-gate/` (`ci.ps1` + `ci_i2_tripwire.py`
+ 3 tests + INTAKE). One command an agent / pre-commit hook runs before touching
`stack/`. Two fail-fast gates:

1. **I2 tripwire first (~2 s).** `ci_i2_tripwire.py` builds the real
   `WorldModel(smoke_config())` and asserts `encode(batch=1) == encode(batch=B)` to
   1e-4 (`i2_batch_consistency`, D-004). Deployment is **batch-1 streaming on Orin**;
   a batch-statistic layer (BatchNorm, a stray `x-x.mean(0)`, running stats) is silent
   in training — loss still falls — but makes the exported TensorRT engine return
   different latents than the trained model. Running this *before* the slow suite fails
   the whole class in ~2 s.
2. **Suite + timing guard.** The full pytest suite (G-E) run *through*
   `profile_testsuite.py check` (the 2026-07-09 intake), which fails the commit on any
   test failure, warm overhead > budget, or any single `call` test > budget.

**Measured (RTX-4060 host, dev machine, venv `tanitad` py3.13/torch 2.11, 2026-07-11):**

| run | result | numbers |
|---|---|---|
| `ci.ps1` (clean tree) | **PASS, exit 0** | total **17.2 s** (I2 2.4 s + suite 14.8 s); I2 dev **1.74e-07**; suite **189 passed**; warm overhead **1.429 s** |
| package tests standalone | **3 passed, 2.07 s** | incl. the falsifier unit test (batch-mean encoder rejected, dev > 1e-4) |
| **gate falsifier** (7.0 s test injected into `stack/tests/`) | **FAIL, exit 1** | timing guard flagged `test_deliberately_slow_fixture 7.0s > 6.0s`; temp test removed |

- **Falsifier verdict (backlog stated):** *"a newly-added slow fixture must make ci.ps1
  exit nonzero"* → **holds** (exit 1, correct reason). The I2 falsifier
  *"a batch-statistic layer must be rejected"* → **holds** (unit test, dev > 1e-4).
- Cost story: warm wall 17.2 s is dominated by the suite (14.8 s), which is itself
  Drive-hydration-bound (the still-pending "pin `stack/` offline" fix would cut the
  cold tax ~30 s — see 2026-07-09 note §2). The gate adds only the ~2.4 s I2 front-run.
- **G-T1: GO** — 0 new deps (pwsh/powershell present on dev box + pods), 0 min setup,
  catches two real bug classes at commit. Robustness: auto-detects `stack/`, prefers the
  off-Drive venv, degrades to plain `pytest` if the profiler intake is rejected.
- **Encoding pitfall (recorded):** Windows PowerShell 5.1 reads a no-BOM `.ps1` as the
  system codepage → non-ASCII punctuation (em-dash) broke parsing on first run. `ci.ps1`
  is now pure ASCII; keep it so, or save `.ps1` as UTF-8-BOM.

## 2. Deployment finding (H5/C1/P5): TensorRT Q/DQ export passes on RTX, FAILS on Orin+Thor

NVIDIA dev-forum thread (TRT 10.13.3 Thor / 10.3 Orin): a PyTorch **Dynamo**-exported
**INT8-quantized** ONNX model builds a TensorRT engine fine on desktop **RTX** but
**segfaults on Thor** and **errors on Orin**. Root cause: TensorRT wants Q/DQ scales as
"strictly initialized float32 constants," but Dynamo wraps the scales behind **reshape**
ops. x86/Ada (RTX) silently *constant-folds* and masks it; the ARM/Blackwell parser
(Thor sm_110, Orin) rejects it — Thor segfaults after Q/DQ-fusion, Orin throws
`input tensor shape misaligns with the input kernel shape`, both spraying
`has invalid precision Int8, ignored`. **Workaround: emit Q/DQ scales as direct
float32 initializers, not reshape-derived, during ONNX export.**

**Why this matters for us (actionable):**
- Prod-Opt validated our ONNX export **on the RTX 4060** (opset 17/18 clean, parity
  8.8e-6 — 2026-07-08 run). This finding shows **an RTX-clean INT8 export does NOT imply
  an Orin/Thor-clean engine** — the failure only surfaces on the target arch. Our
  earlier KB row ("INT8 on small ViTs can regress latency 2.7×"; INT8 deferred, must be
  measured) is now joined by an INT8-*export* trap, not just a latency trap.
- There is **live Prod-Opt work-in-progress on this exact surface**: an uncommitted
  `Production & Optimization/Implementation/int8_quant/` folder is stranded in `main`
  (flagged by the orchestrator 2026-W29). Recommendation → Prod-Opt/Sat: when the INT8
  path is exercised, **build the engine with `trtexec` on an actual Orin/Thor (or the
  graphics-pod recipe target), not only RTX**, and pre-check the exported ONNX for
  reshape-fed Q/DQ scales. Keep FP16 static-shape as the primary Orin path (unchanged);
  INT8 stays deferred until measured *on-target*.

## 3. Tooling landscape delta (P5/opponent): AlpaGym now public; CARLA 0.10 = UE5.5

- **AlpaGym is now a public repo** (`NVlabs/alpagym`, Apache-2.0) — was announced-only at
  my 2026-07-09 run. RL closed-loop post-training: AlpaSim (envs) + **Cosmos-RL**
  (distributed training). Default policy = **Alpamayo 1.5, 10 B params, needs ≥2 GPUs**;
  requires HF + W&B + `uv`; **no lightweight reference policy** shipped. Also live: an
  "AlpaSim E2E Closed-Loop Challenge 2026" (HF Space). **Verdict unchanged: Phase-1
  cloud, not Phase-0** (2-GPU/10 B is 100×+ our envelope, P5). It is now *clonable* — the
  Phase-1 self-play target once we have a <100 M driver; our Phase-0 closed-loop path
  stays **CARLA-on-pod** (D-014). Cosmos-RL is the interesting sub-component to watch as a
  scorer/orchestrator we could borrow, not the 10 B policy.
- **CARLA 0.10.0 = Unreal Engine 5.5** (Lumen/Nanite, remodeled Town10, InvertedAI
  traffic, native ROS). **Min spec: RTX 3000-series, ≥16 GB VRAM, Ubuntu 22.04/Win11.**
  Our pod2 runbook pins **0.9.16** (UE4.24) deliberately — 0.10's 16 GB-VRAM floor + the
  UE5 rebuild are a Phase-1 consideration, not a Phase-0 switch. The 0.9.16 nullrhi
  telemetry path (SC-01 measured) is unaffected. Watch 0.10 for when photoreal pixels
  become eval-critical (pairs with the graphics-pod recipe, P1.3).

## 4. Ledger / gates

- **G-D (HYPOTHESIS_LEDGER): no status change.** The TensorRT Q/DQ finding is an
  engineering risk on the H5/C1 deployment path (efficient inference), not evidence
  for/against a hypothesis — recorded as a KB/deployment note, not a ledger row (P8:
  don't inflate a tooling gotcha into hypothesis evidence).
- G-A: every claim above links a source or repo path. G-B: actionable recs to Prod-Opt
  (on-target INT8 build) and to the orchestrator (integrate `ci.ps1`; pin `stack/`
  offline still pending). G-E/G-H: increment shipped + measured + falsified.

## 5. Sources

- TensorRT export fails on Orin/Thor, passes on RTX — NVIDIA Developer Forums:
  https://forums.developer.nvidia.com/t/tensorrt-export-fails-on-both-jetson-agx-orin-and-thor-but-passes-on-rtx/363379
- AlpaGym (Apache-2.0): https://github.com/NVlabs/alpagym · AlpaSim: https://github.com/NVlabs/alpasim
- CARLA 0.10.0 release (UE5.5): https://carla.org/2024/12/19/release-0.10.0/
- Repo: intake `Implementation/incoming/2026-07-11-ci-gate/`; instrument `stack/tanitad/instruments/checks.py` (I2); profiler intake `2026-07-09-testsuite-io-profiling/`.

# Tools&DevEnv — CI gate `ci.ps1` (measured) + edge/viz/robustness sweep

**Agent:** tools-devenv-agent (Monday) · **Date:** 2026-07-10 · **Run:** W4, fourth weekly run
**Loop:** 1 iteration · 5 web searches · worktree `agent/tools-devenv-20260710`
**QUALITY: full** (G-A…G-F + G-H + G-T1 met; measured experiment = `ci.ps1` end-to-end + falsifier)

---

## 0. TL;DR

- **Shipped the top backlog item (P0.1): `stack/scripts/ci.ps1`** — one command (I2 collapse tripwire
  fail-fast → full suite + timing budget). **Measured warm: 11.2 s, exit 0, 189 tests**; falsifier
  (a 7.0 s test) **correctly forced exit 1**. Intake pkg `2026-07-10-ci-script/`.
- **Sweep, 4 deep findings actionable for our envelope (P5):**
  1. **Jetson Thor FP8 on ViT ≈ +20 % only, and needs ONNX Q/DQ surgery** → do NOT bank latency on
     FP8 for the ViT tower; keep the ONNX→TensorRT **FP16 static-shape** path as the deploy default.
  2. **Rerun 0.32** became a full "data layer" (MCAP/rrd/LeRobot ingest, dataset-review UI) → re-scopes
     backlog #2 (episode→`.rrd`) toward *also* using the review UI for D3 imagined-vs-oracle triage.
  3. **Bench2Drive-Robust** (2605.18059) defines 3 deployment-perturbation classes (camera-stream
     failure, ego-state error, **compute-induced control delay**) → these are exactly what our CARLA
     harness + perturbation policy should inject; the control-delay axis ties directly to our I8 latency.
  4. **AlpaGym is open** (`NVlabs/alpamayo-recipes`) → Phase-1 closed-loop-RL verdict unchanged (P5).

---

## 1. Experiment (G-H): the CI gate `ci.ps1` — measured

**Goal (backlog P0.1 / agent-file duty #3):** one command for agents & pre-commit — pytest + I2
tripwire + timing budget, warm wall < 15 s, and a falsifier where a newly-added slow fixture forces a
nonzero exit. **Hardware:** dev machine (venv off-Drive at `C:\Users\Admin\venvs\tanitad`, stack on
Google Drive File Stream). **Cost:** $0 (local CPU).

**Design.** `ci.ps1` runs two stages, stopping at the first failure, with distinct exit codes so a git
hook can branch:
1. **I2 collapse tripwire** `tests/test_instruments.py::test_i2_batch_consistency_of_encoder` — the
   BatchNorm-in-inference canary (D-004). Fail-fast (~1.5 s) → **exit 2** if it trips.
2. **Full suite + timing budget** via `profile_testsuite.py check` (warm-overhead ≤ `-MaxWarmOverhead`
   4 s, no single `call` > `-MaxTest` 6 s) → **exit 1** on any test failure or budget breach.
   If the profiler is absent, an **inline pytest --durations fallback** enforces the same budgets, so
   `ci.ps1` has no hard dependency on the (still-pending-triage) profiler intake.
Python resolves `$env:VIRTUAL_ENV` → dev venv → PATH; `StackDir` defaults to the script's parent.

**Measured (2026-07-10, warm):**

| Path | I2 tripwire | Full suite | ci.ps1 total | Exit |
|---|---|---|---|---|
| Green | `1 passed 1.57 s` | `189 passed, overhead 1.112 s, wall 8.712 s` | **11.2 s** | **0** |
| Falsifier (`sleep(7.0)` test) | `1 passed 1.64 s` | `slow test ...7.0s > 6.0s` | 18.1 s | **1** |

**Falsifier verdict:** REFUTED the null "a slow test slips through" — the injected 7.0 s `call`
tripped the `-MaxTest` budget and the gate exited nonzero, naming the offending node. Temp test
deleted; `git status` clean. **Goal met:** warm 11.2 s < 15 s; falsifier fires.

**Gotcha logged (reusable):** PS 5.1 reads a BOM-less `.ps1` as the ANSI codepage; an em-dash in a
string mangled to `â€"` and broke the parser. Rule for all future repo `.ps1`: **pure ASCII** (or save
UTF-8-with-BOM). Added to the path/encoding-audit backlog item (P2.6).

**G-T1:** GO — 0 min setup, no new deps, measured 11.2 s, falsifier verified. Recommend wiring as
`.git/hooks/pre-commit` → `pwsh stack/scripts/ci.ps1` once integrated (with `profile_testsuite.py`).

---

## 2. Sweep — findings that change something (G-A: each claim sourced)

### 2.1 Jetson Thor / TensorRT: FP8 on ViT is a low-ROI trap without ONNX surgery
NVIDIA dev-forum + TensorRT issue #4599 report **~20 % latency reduction only** for base ViT under FP8
vs FP16 on Jetson Thor, and that reaching even that requires **ONNX "surgery" — inserting Q/DQ nodes at
specific locations to trigger the right Multi-Head-Attention FP8 fusion** ([forum][thor-fp8],
[TensorRT#4599][trt4599]). This triangulates our 2026-07-06 KB (Orin: INT8 on small ViTs can *regress*
2.7×; DLA has no ViT attention) and the Prod&Opt 2026-07-09 result (**bf16 decision-unsafe** at 67.2 %
selection agreement; fp16 safe at 95.3 %). **Actionable (H5/C1/P5):** deploy target stays
**ONNX→TensorRT FP16, static shapes**; treat FP8/INT8 on the ViT tower as an *optional, measured-only*
optimization gated behind a Q/DQ-surgery step and a decision-safety re-check like the bf16 audit —
never a default. Note to Prod&Opt (owns the export/precision path) + Architecture (keep ViT shapes
static, norms batch-free — already the standard).

### 2.2 Rerun 0.32 is now a "data layer", not just a viewer — re-scopes viz backlog #2
The Rerun 0.32 SDK reframes it as a unified physical-AI data layer: ingest from **MCAP, `.rrd`,
LeRobot**, a **chunk-processing API**, and a **dataset-review UI** over columnar Arrow `.rrd`
([rerun releases][rerun], [rerun what-is][rerun-what]). Our pick (2026-07-06 KB) is unchanged
(MIT/Apache, `pip install rerun-sdk`, ROS-free). **Actionable (WP-viz/D3/H5):** scope backlog #2 as
*episode → `.rrd`* with predicted-vs-actual trajectory + BEV overlay (complement to the already-shipped
`viz_trajectory_fan.py`, not a dup), and additionally use the **dataset-review UI** as the human triage
surface for D3 imagined-vs-oracle rollouts. Measure setup cost + one real episode `.rrd` size/time next
run (G-T1).

### 2.3 Bench2Drive-Robust: a ready-made deployment-perturbation taxonomy for our CARLA harness
Bench2Drive-Robust (arXiv 2605.18059) is the first *device-centric* closed-loop robustness benchmark;
it perturbs three deployment axes: **camera-stream failures** (frame drop, partial observation),
**ego-state estimation errors** (GPS noise, speed/odometry error), and **compute-induced control
delay** (model inference delay) ([abs][b2dr]). This is directly reusable: our CARLA-on-pod harness
already has a *perturbation policy* seam (retired MetaDrive port + `carla_work_zone.py`), and the
**control-delay axis is our I8 latency made closed-loop** — a slow decision tick literally becomes a
control delay in sim. **Actionable (D5/D6/H15/H11):** add these three perturbation classes as CARLA
scenario knobs; the control-delay one should be driven by the *measured* I8 tick (15.07 ms p50 fp32),
so eval degradation is tied to real latency, not a guess. Cross-note to Benchmarks & Eval (metric
seam), Opponent (maps onto Waymo/Tesla degraded-visibility scenarios), DataEng (perturbed episodes).

### 2.4 AlpaGym open-sourced — Phase-1 verdict unchanged
NVIDIA's **AlpaGym** (high-throughput closed-loop RL over AlpaSim + Omniverse NuRec) is open, with an
`NVlabs/alpamayo-recipes` repo to adapt the closed-loop post-training recipe ([NVIDIA blog][alpa-blog],
[newsroom][alpa-news]). Confirms the direction we already logged: closed-loop RL post-training of OUR
<100 M driver is a **Phase-1 cloud** target (40–60 GB VRAM, P5) — not Phase-0. Watch `alpamayo-recipes`
for a *lighter reference harness* we could adapt rather than build from scratch (backlog P2.7).

### 2.5 CARLA housekeeping
No CARLA 0.10 GA confirmed in results; the **Leaderboard 2.1** exists with a changed infraction score
(explicitly **not comparable to 2.0**) ([leaderboard eval v2.1][carla-lb]). Keep our closed-loop
scenario scoring self-defined (Benchmarks & Eval suite) rather than pinned to a moving CARLA leaderboard
revision; note the 2.0↔2.1 incomparability if we ever cite CARLA-leaderboard numbers.

---

## 3. Actionable recommendations (G-B)

1. **Integrate `ci.ps1` + `profile_testsuite.py` together** into `stack/scripts/`, then wire a
   `pre-commit` hook / add to the agents' session-end ritual (`pwsh stack/scripts/ci.ps1`). — CI/G-E.
2. **Keep deploy precision = TensorRT FP16 static-shape.** FP8/INT8 on the ViT tower only as a
   measured, Q/DQ-surgery-gated, decision-safety-rechecked option (→ Prod&Opt). — H5/C1/P5.
3. **Add Bench2Drive-Robust's 3 perturbation classes to the CARLA harness**, with control-delay driven
   by the measured I8 tick. — D5/D6/H11/H15 (→ Benchmarks & Eval, Opponent, DataEng).
4. **Scope viz backlog #2 to Rerun 0.32** episode→`.rrd` + dataset-review UI for D3 triage. — WP-viz/D3.

## 4. Hypothesis ledger (G-D)

No hypothesis status change this run (infra + tooling week). Evidence notes only: H5/C1 export-precision
(§2.1) reinforced; H11/H15 robustness scenario surface widened (§2.3). No ledger row upgrade (P8).

## 5. Sources

- [thor-fp8] https://forums.developer.nvidia.com/t/low-vit-performance-gain-on-jetson-thor-using-fp8-vs-fp16/349329
- [trt4599] https://github.com/NVIDIA/TensorRT/issues/4599
- [rerun] https://github.com/rerun-io/rerun/releases
- [rerun-what] https://rerun.io/docs/overview/what-is-rerun
- [b2dr] https://arxiv.org/abs/2605.18059
- [alpa-blog] https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/
- [alpa-news] https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis
- [carla-lb] https://leaderboard.carla.org/evaluation_v2_1/
- Repo refs: `stack/scripts/profile_testsuite.py` (intake), `stack/tests/test_instruments.py`,
  `stack/scripts/carla_work_zone.py`, `stack/scripts/viz_trajectory_fan.py`,
  KB `2026-07-06`/`2026-07-09` rows, Prod&Opt `2026-07-09-half-precision-and-models-failfast.md`.

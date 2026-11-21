# Tools&DevEnv research note — 2026-07-09 (W3, Monday run)

Focus: (1) root-cause the pod2 CARLA **camera-rendering blocker** and turn it into a
turnkey graphics-pod recipe; (2) measured experiment — **test-suite I/O cost** (G-H);
(3) tooling sweep (AlpaSim now public). Loop: 6 web searches, 1 fetch; wall ~55 min;
cost $0 (all local RTX-4060 host + web).

---

## 1. CARLA camera rendering on pod2 — root cause found (was: "not fixable in-container")

**Symptom (PROJECT_STATE Phase B):** CARLA 0.9.16 runs fine in `-nullrhi` (~1400 ticks/s,
sync stepping) but camera RENDERING is host-blocked: `vulkaninfo`/GIPA returns NULL while
`nvidia-smi` works; all userspace libs verified present/matching → declared not fixable
in-container, needs a graphics-capable pod (Sayed decision, not urgent).

**Root cause is now pinned — two independent causes stack, and both are host/template level:**

1. **RunPod GPUs are exposed compute-only.** Standard RunPod pods launch with
   `NVIDIA_DRIVER_CAPABILITIES=compute,utility`. That capability set installs the CUDA
   runtime into the container but NOT the graphics/display driver libraries (Vulkan WSI,
   the NVIDIA Vulkan ICD, EGL device nodes). Result: `nvidia-smi` works (compute) but
   `vulkaninfo` finds no ICD → `GetInstanceProcAddr`/GIPA returns NULL — **exactly our
   symptom**. This is set by the NVIDIA Container Toolkit at container-creation from the
   host; **it cannot be changed from inside a running container** (the driver is the
   host's, per RunPod docs) → confirms "not fixable in-container" and says *why*.
   A RunPod-specific EGL report shows the same failure mode: `eglinfo` lists devices
   outside RunPod, none inside → silent CPU fallback [answeroverflow].
2. **Unreal Engine 4.24 (CARLA's engine) cannot render Vulkan offscreen.** UE4 crashes
   when Vulkan runs off-screen; Epic never fixed it. CARLA's own docs: on a machine with
   no display, **OpenGL needs no config; Vulkan needs extra steps** (an X server to talk
   WSI to) [carla docs; issue #3800]. So even on a graphics-capable pod, pure
   `-RenderOffScreen` over Vulkan is fragile.

### Turnkey recipe for the eventual graphics-pod recreation (Sayed, ~15–30 min, supervised)

Prepared so the supervised window is a single go/no-go probe, not debugging:

1. **Launch the pod from a template with `NVIDIA_DRIVER_CAPABILITIES=all`** (must include
   `graphics`; `video`/`display` help). This is a template/GPU-type property — pick a
   RunPod GPU type that advertises graphics; not all do.
2. **Probe BEFORE installing CARLA — this is the gate:**
   `vulkaninfo | grep -i deviceName` must print the GPU. If it still says GIPA/NULL, the
   host driver for that GPU type is compute-only → **no in-container fix exists**; try a
   different GPU type or provider. `nvidia-smi` alone is NOT a sufficient check.
3. Confirm the ICD is visible: `ls /usr/share/vulkan/icd.d/nvidia_icd.json`
   (`VK_ICD_FILENAMES` can point at it if mounted elsewhere).
4. **Belt-and-braces for the UE4-Vulkan-offscreen fragility:** run under a virtual X
   display — `Xvfb :99 -screen 0 1280x720x24 & export DISPLAY=:99` — then
   `./CarlaUE4.sh -RenderOffScreen -quality-level=Low`. The Xvfb path renders to a
   Pbuffer and sidesteps the UE4 no-display Vulkan crash.
5. Re-run the existing `stack/scripts/carla_work_zone.py` but with the checkpoint-driven
   ego (camera path) instead of the scripted archetypes — the pixels milestone.

**G-T1 verdict:** the recreation is a small supervised op gated by ONE probe
(`vulkaninfo`). Cost = one graphics-capable pod's hourly rate for the eval window; $0
until then. **NOT urgent** — milestone 1 (work-zone LAL/OKRI/LOPS, already measured live:
SC-01 OKRI 32.4 vs 12.8) needs no pixels. Do it only when checkpoint-driven ego eval in
CARLA is on the critical path (post-D1–D3). Recorded here so it's turnkey when it is.

## 2. Measured experiment (G-H) — test-suite I/O cost, and the real G-E lever

Backlog P1.5. Ran on the dev machine (RTX-4060 host; **venv off-Drive**, repo/tests/
fixtures **on Google Drive File Stream**). Suite = 181 passed / 1 skipped.

| Measurement | Time | Note |
|---|---|---|
| **cold** first run of the day (wall) | **40.6 s** | fresh process, cold Drive cache |
| **warm** subsequent runs (wall) | **10.7 s** | ×2 runs, 10.75 / 10.69 s — stable |
| pytest-reported test time | 9.2 s | compute only |
| `pytest --collect-only` | 4.9 s | imports torch once across 28 modules |
| `import torch` | 1.9 s | |
| `import tanitad` (pkg) | 0.08 s | lazy; no torch at package top |
| warm full read of `stack/` src | 0.13 s | 87 files, **0.44 MB** |
| slowest test `test_smoke_training` | 3.02 s | 1 fwd/bwd smoke train |
| slowest test `test_base250_parameter_budget` | 1.09 s | instantiates the 261 M model |

**Interpretation.** The ~30 s cold penalty is **Google-Drive hydration latency**, not byte
volume (0.44 MB) and not compute (9.2 s). Drive File Stream fetches per-file metadata +
content on first touch in a fresh session; a scheduled agent runs cold once/day → pays
the tax every run. The P1.5 target (<60 s local) is already met even cold, so the goal
**reframes** from "make tests faster" to "kill the cold-I/O tax and guard against creep".

**Falsifier check:** hypothesis was "tests are slow because of compute/torch import."
Falsified — warm compute is 9.2 s and import is 1.9 s; the variable cost is I/O locality.

### Actionable recommendation (G-B)

- **Pin `stack/` to Google Drive "Available offline"** (Drive UI, one click, ~1 MB source
  + fixture footprint). Expected effect: cold ≈ warm (~10.7 s), i.e. **remove ~30 s per
  cold agent run** — free G-E speedup for all six weekly agents. No code, no risk.
  (Alternative: an off-Drive `git worktree` for the stack; heavier, deferred.)
- Guard against regressions with the shipped `profile_testsuite.py check` (below).

## 3. Implementation increment (G-E) — `profile_testsuite.py` (intake)

Intake pkg `Implementation/incoming/2026-07-09-testsuite-io-profiling/` — stdlib-only
profiler: `profile` (cold/warm decomposition + slowest-test ranking → JSON) and `check`
(CI/agent regression guard: nonzero exit if warm overhead or any single test exceeds a
budget). Parsers are unit-tested on canned pytest text (no pytest-in-pytest).
**9 pkg tests pass (0.30 s); end-to-end `check` → OK, 181 passed, overhead 1.38 s, exit 0.**
Proposed target `stack/scripts/profile_testsuite.py`; pairs with the future `ci.ps1`
(backlog #3) as its timing guard.

## 4. Tooling sweep

- **AlpaSim is now PUBLIC on GitHub** — `NVlabs/alpasim` (open-source end-to-end AV sim)
  + AlpaGym (closed-loop RL) are live repos, not just an announcement. Alpamayo-2 Super
  (32 B VLA) inference code + HF weights "this summer". Moves AlpaSim from "announced" to
  "clonable" → a concrete Phase-1 adoption path now exists. Still 40–60 GB VRAM / Docker
  / HF-gated driver models → unchanged verdict: **Phase-1 cloud, not Phase-0** (P5). Our
  edge stays efficiency/labels vs 32 B scale (C2). Watch for a lighter reference policy in
  the repo that could seed our closed-loop harness.
- CARLA remains the Phase-0 closed-loop tool (D-014); the rendering recipe above is the
  only thing between us and checkpoint-driven ego eval in CARLA.

## Sources

- [CARLA rendering options (headless: OpenGL needs no config, Vulkan needs an X server)](https://carla.readthedocs.io/en/latest/adv_rendering_options/)
- [CARLA #3800 — headless Vulkan on a server; UE4 Vulkan-offscreen crash](https://github.com/carla-simulator/carla/issues/3800)
- [CARLA #8079 — RenderOffScreen Vulkan driver problems, Ubuntu 22.04 + NVIDIA](https://github.com/carla-simulator/carla/issues/8079)
- [CARLA #6234 — Vulkan ICD issues from the carla Docker image](https://github.com/carla-simulator/carla/issues/6234)
- [RunPod: driver is host-side, cannot change in-container](https://docs.runpod.io/get-started)
- [RunPod EGL headless falls back to CPU — no EGL devices in-container](https://www.answeroverflow.com/m/1220555454518923364)
- [NVIDIA: minimal Docker Vulkan offscreen setup (ICD + NVIDIA_DRIVER_CAPABILITIES)](https://forums.developer.nvidia.com/t/minimal-docker-vulkan-offscreen-setup/242883)
- [NVlabs/alpasim (public repo)](https://github.com/NVlabs/alpasim)
- [NVIDIA Alpamayo 2 Super 32 B — June 2026](https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis)

# KNOWLEDGE_BASE — Tools&DevEnv

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`

- [2026-07-15] [built+measured] **CI commit gate `ci.ps1`** (backlog P0 #1): one command = I2
  tripwire (fail-fast, exit 2) → per-test latency budget (slowest `call` > 6 s → exit 1, the
  registered falsifier — **CONFIRMED**, a 7 s test blocks the full gate) → suite-green + opt-in
  warm-wall budget. Logic in unit-tested `ci_check.py` (12 tests/0.20 s), `ci.ps1` thin
  venv-resolving wrapper. Measured warm wall: **quick pre-commit gate 8.3 s** (curated safety
  subset, 38 tests), **full gate 24.4 s** (363 tests). 0 new deps. NOTE: the old "full suite < 15 s"
  target is stale — suite grew 181→351 tests, ~21 s warm, driven by breadth not slow tests (slowest
  single `call` 1.97 s) → split into fast-quick + full gates — impact: G-E/CI/all-agents —
  `2026-07-15-ci-commit-gate-and-latency-budget.md`
- [2026-07-15] [tooling] **AlpaGym** = NVIDIA's new high-throughput **closed-loop RL** framework
  (Alpamayo family, open on GitHub); AlpaSim open E2E sim; Alpamayo-2 Super 32 B weights "this
  summer". Verdict unchanged: **Phase-1 cloud, not Phase-0** (40–60 GB VRAM). AlpaGym is the
  reference for our own Phase-1 closed-loop-RL harness over a <100 M driver — impact: P5/H1/opponent —
  [newsroom](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development)
- [2026-07-15] [reference] **Bench2Drive** closed-loop protocol = **220 routes (5 × 44 scenarios ×
  weather × location, ~150 m each)**, task-disentangled to cut seed variance. This is the structure
  our **CARLA-on-pod closed-loop runner (P1.3) should mirror** — short scenario-attributed routes,
  not one monolithic drive — impact: P1.3/H1/H11 — [OpenReview](https://openreview.net/forum?id=y09S5rdaWY)
- [2026-07-09] [root-cause] CARLA camera-rendering on pod2 (GIPA/vulkaninfo NULL) = TWO stacked
  host-level causes: (1) RunPod pods launch `NVIDIA_DRIVER_CAPABILITIES=compute,utility` → no Vulkan
  ICD / EGL device in-container (nvidia-smi works, vulkaninfo NULL) — set by NVIDIA Container Toolkit
  at creation, unchangeable in a running container; (2) UE4.24 can't render Vulkan offscreen (Epic
  bug) → needs OpenGL or an X server. Turnkey fix = pod template with `NVIDIA_DRIVER_CAPABILITIES=all`
  (must incl. `graphics`), gate on `vulkaninfo | grep deviceName` BEFORE installing CARLA, then Xvfb
  `:99` + `CarlaUE4.sh -RenderOffScreen`. NOT urgent (milestone 1 needs no pixels) — impact: D-014/Phase-B —
  `2026-07-09-carla-render-blocker-and-testsuite-io-cost.md` §1
- [2026-07-09] [measured] Test-suite G-E cost is dominated by **Google-Drive hydration latency**, not
  compute: cold 40.6 s vs warm 10.7 s (same 181-pass suite; reported test time 9.2 s; stack src is only
  0.44 MB / 87 files). Fix = pin `stack/` to Drive "Available offline" → cold≈warm, ~30 s saved per cold
  agent run (all 6 weekly agents), zero code. Regression-guard shipped (`profile_testsuite.py check`) —
  impact: G-E/CI/backlog#3 — `2026-07-09-carla-render-blocker-and-testsuite-io-cost.md` §2–3
- [2026-07-09] [tooling] AlpaSim is now a PUBLIC GitHub repo (`NVlabs/alpasim`) + AlpaGym closed-loop RL;
  Alpamayo-2 Super 32 B inference/weights "this summer". Moves from announced→clonable, but still
  40–60 GB VRAM/Docker/HF-gated → verdict unchanged: **Phase-1 cloud, not Phase-0** (P5); watch for a
  lighter reference policy to seed our closed-loop harness — impact: P5/H1/opponent —
  [NVlabs/alpasim](https://github.com/NVlabs/alpasim)
- [2026-07-13] [built] MetaDrive front-camera RGB path unblocks the D-010 sim arm: sim episodes were
  `[T,1,64,64]` BEV, real (comma2k19) is `[T,6,256,256]`; `MixedWindowDataset._check_contract` rejected
  the mismatch → sim arm was structurally dead. New intake pkg renders 6ch/256 2-frame RGB stacks
  (comma2k19-identical geometry/alignment) + perturbation policy + occluder/blocked-route scenarios;
  17 tests pass, 0 new deps, import 1.38 s — impact: D-010/WP2/A8 — `2026-07-13-metadrive-frontcam-rgb-and-perturbation.md`
- [2026-07-13] [api] MetaDrive front camera: `image_observation=True`, `sensors={"rgb_camera":(RGBCamera,W,H)}`,
  `vehicle_config.image_source="rgb_camera"`; `obs["image"]` = `(H,W,3,stack_size)`, newest at `[...,-1]`,
  [0,1] float32; `image_on_cuda`≈10× (pod, VRAM). Caveat: BGR/row-flip on some backends — verify vs PNG —
  impact: WP2 — [sensors](https://metadrive-simulator.readthedocs.io/en/latest/sensors.html)
- [2026-07-13] [opponent] NVIDIA Alpamayo 2 Super (GTC Taipei 2026-06-01): 32 B open VLA reasoning model,
  closed-loop-trained via AlpaGym on AlpaSim; GitHub/HF "this summer". ~100–300× our envelope → reinforces
  the prove-the-mechanism-not-scale thesis (C2/P5). OmniDreams = photorealistic closed-loop world-model sim
  (AlpaSim+Omniverse NuRec) — Phase-1 watch, not Phase-0 adopt — impact: P5/H15/opponent —
  [newsroom](https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis)
- [2026-07-06] [measured] Orin DLA does NOT support ViT attention (GPU-only, JetPack 6.2); INT8 on small
  ViTs can regress latency 2.7× (ViT-S+DPT, Orin Nano). Plan ONNX→TensorRT FP16 static-shape first, INT8
  only with measured calibration — impact: C1/P5 — see `2026-07-06-metadrive-adoption-and-alpasim-verdict.md` §5
- [2026-07-06] [pick] Rerun.io (MIT/Apache, `pip install rerun-sdk`) = ROS-free, PyTorch-native replay/viz;
  Arrow-columnar `.rrd`, published nuScenes AV example. Chosen for backlog #2 (episode→overlay) — impact:
  WP-viz/D3/H5 — [rerun.io](https://rerun.io/docs/overview/what-is-rerun)
- [2026-07-06] [verdict] AlpaSim/AlpaGym (NVIDIA Alpamayo, Apache-2.0, Jan 2026) = closed-loop microservice
  sim + closed-loop RL. Driver models need ~40–60 GB VRAM, Docker/SLURM + HF-token gated → NO-GO on RTX
  4060; Phase-1 cloud target for self-play with OUR <100 M driver — impact: C1/H1/H11, P5 — see note §3
- [2026-07-06] [verdict] MetaDrive install: PyPI `metadrive-simulator` NO-GO on py3.13 (pins unbuildable
  gym 0.19); native blocker cleared (panda3d 1.10.16 + gymnasium 1.3.0 have cp313 wheels, <1 min). GO path =
  source install (GitHub main, gymnasium) in a supervised session — impact: WP2 — see note §2
- [2026-07-05] [kickoff] Initial research baseline for all hypotheses established; discipline agenda
  seeds defined — impact: all — see `../../INITIAL_RESEARCH_SYNTHESIS.md`

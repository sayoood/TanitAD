# KNOWLEDGE_BASE — Tools&DevEnv

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`

- [2026-07-10] [measured] **CI gate `ci.ps1` shipped** (backlog P0.1): I2 collapse tripwire fail-fast
  → full suite + timing-budget (`profile_testsuite.py check`, inline fallback if absent). Warm **11.2 s
  / 189 tests / exit 0**; falsifier (7.0 s test) → **exit 1** naming the node. Distinct exits 0/1/2 for
  hook branching. Gotcha: PS 5.1 reads BOM-less `.ps1` as ANSI → **keep repo `.ps1` pure-ASCII** —
  impact: CI/G-E/D-004 — intake `2026-07-10-ci-script/`, `2026-07-10-ci-gate-and-edge-viz-sweep.md` §1
- [2026-07-10] [edge] Jetson **Thor FP8 on base ViT ≈ +20 % only, and requires ONNX Q/DQ surgery** to
  trigger the MHA FP8 fusion (dev-forum + TensorRT#4599). Triangulates Orin INT8-regression (2026-07-06)
  + bf16-unsafe (Prod&Opt 2026-07-09). Deploy default stays **ONNX→TensorRT FP16 static-shape**; FP8/INT8
  on the ViT tower = measured-only, surgery-gated, decision-safety-rechecked option — impact: H5/C1/P5 —
  [forum](https://forums.developer.nvidia.com/t/low-vit-performance-gain-on-jetson-thor-using-fp8-vs-fp16/349329),
  [TRT#4599](https://github.com/NVIDIA/TensorRT/issues/4599)
- [2026-07-10] [robustness] **Bench2Drive-Robust** (2605.18059) = first device-centric closed-loop
  robustness benchmark; 3 deployment-perturbation classes: camera-stream failure (frame drop / partial
  obs), ego-state error (GPS/speed/odometry), **compute-induced control delay (inference delay)**.
  Reusable as CARLA scenario knobs; control-delay axis = our I8 tick made closed-loop (drive it from the
  measured 15.07 ms) — impact: D5/D6/H11/H15 — [abs](https://arxiv.org/abs/2605.18059)
- [2026-07-10] [tooling] **Rerun 0.32** = unified physical-AI *data layer* (MCAP/`.rrd`/LeRobot ingest,
  chunk-processing API, **dataset-review UI**, columnar Arrow). Pick unchanged (MIT/Apache, ROS-free).
  Re-scopes viz backlog #2: episode→`.rrd` overlay + use the review UI for D3 imagined-vs-oracle triage —
  impact: WP-viz/D3/H5 — [releases](https://github.com/rerun-io/rerun/releases)
- [2026-07-10] [opponent] **AlpaGym open-sourced** (`NVlabs/alpamayo-recipes`): high-throughput
  closed-loop RL over AlpaSim + Omniverse NuRec. Verdict unchanged — closed-loop RL post-training of our
  <100 M driver = **Phase-1 cloud** (40–60 GB VRAM, P5); watch the recipes repo for a lighter reference
  harness to adapt — impact: P5/H1/opponent —
  [blog](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/)
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

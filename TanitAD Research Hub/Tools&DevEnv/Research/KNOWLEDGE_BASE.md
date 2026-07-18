# KNOWLEDGE_BASE — Tools&DevEnv

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`

- [2026-07-18] [built] `tools/session_guard.py` — the D-026 session-end stranded-work guard every
  agent runs (protocol-wired G-F/G-I). BLOCKS on uncommitted hub deliverables; WARNs on unmerged
  `agent/*` branches vs tip (`rev-list --count tip..branch`; current branch info-only) and on
  `incoming/*/INTAKE.md` with an unfilled `ORCHESTRATOR VERDICT` older than 3d. Tip defaults to HEAD
  (origin/main is diverged, `0f93b98`). Stdlib-only, 15 falsifiers 5.2 s. **Live-tree run flagged the
  real debt: 5 uncommitted hub files / 9 stranded branches / 5 stale INTAKEs (9d/5d)** — impact:
  G-F/G-I/D-026 — note `2026-07-18-session-guard-…-edgellm.md` §1
- [2026-07-18] [bug-lesson] `git status --porcelain` parsing: a global `.strip()` on git stdout eats
  the leading status-column space of the **first** line (` M path` → `M path`) → fixed-offset `[3:]`
  path parse breaks silently for the first modified file. Use `.rstrip()` (preserve leading). Caught
  by the guard's own falsifier before ship — impact: all-tooling/G-T1 — note §1
- [2026-07-18] [tooling] **AlpaSim (`NVlabs/alpasim`) + AlpaGym (`NVlabs/alpagym`) are now PUBLIC,
  Apache-2.0.** AlpaSim = microservice closed-loop AV sim (NuRec neural renderer, gRPC services,
  ready policies), eval data `PhysicalAI-AV-NuRec` HF. **AlpaGym RL default 10B policy needs 2 GPUs**
  → no-go single-4060/A40; reference-only until a 2×A40 pod or a <100M policy swap. AlpaSim is a
  single-A40 eval-harness smoke-test candidate (GPU-heavy NuRec → not a 4060 job) — impact:
  P5/H1/H11/closed-loop — [alpasim](https://github.com/NVlabs/alpasim) [alpagym](https://github.com/NVlabs/alpagym)
- [2026-07-18] [deployment] **TensorRT Edge-LLM (JetPack 7.1)** = current ViT/VLM edge-export path
  (HF→quantize→ONNX→engine; `--visual_quantization fp8` for ViT). **NVFP4 (4× mem win) is Thor/SM110+
  (Blackwell) ONLY — Orin cannot run NVFP4, only FP8/INT4.** Rule: lock the target chip first (Orin→
  FP8/INT8; Thor→NVFP4). Alpamayo-R1-10B weights (~22GB) live on HF (open teacher); 34B-Super still
  unshipped — impact: C2/P5/Orin-path — [edge-llm](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/)
- [2026-07-17] [built] `ci_gate` — one-command self-testing pytest gate (backlog P0.1). Fails on
  pytest failure OR **collection error** (defers to pytest exit code → never a false GREEN), per-test
  >15 s, wall >90 s, or a missing/failing required tripwire (default the I2 encoder batch-consistency
  test). Stdlib-only, OS-agnostic (`ci.ps1` Win wrapper / `python ci_gate.py` on pod). Measured:
  11/11 falsifiers; catches the live broken suite in 3.9 s; clean suite 343+2skip 47–57 s — impact:
  G-E/CI/D-004 — intake `2026-07-17-ci-gate/`, note `2026-07-17-…-bench2drive-speed.md` §2
- [2026-07-17] [root-cause] The stack suite was **RED for every agent** on 2026-07-17: an untracked
  TDD test `tests/test_physicalai_rig.py` (Data-Eng D-016 R1 two-rig fix) imports `ftheta_horizon_row`
  / `ftheta_project_ray` / `ftheta_crop_box` + `center=`/`per_clip=` that committed `calib.py` never
  shipped → `pytest` exit 2 at collection, 0 of 343 tests run. Fix-forward = `ci_gate` makes it a hard
  gate; remediation (land calib impl or xfail) is Data-Eng/orchestrator — impact: G-E-all-agents —
  note §1
- [2026-07-17] [tooling] Agent tooling on the Windows dev box must be **ASCII-clean stdout**: `ci_gate`
  v1 crashed with `UnicodeEncodeError` printing `✓/✗` under the cp1252 console. Use ASCII markers (or
  force UTF-8) in any script an agent/CI runs on this box — impact: G-T1/all-tooling — note §2
- [2026-07-17] [opponent] NVIDIA **Alpamayo 2 Super = 34 B** (corrects prior KB "32 B"), closed-loop via
  AlpaGym on AlpaSim; GitHub/HF "this summer". NVIDIA shipped a "post-train AV in closed-loop with
  Alpamayo" dev blog + an Alpamayo-1 trajectory-latency paper (arXiv 2605.08975, on our C2/P5 edge
  thesis). Verdict unchanged: Phase-1 cloud (40–60 GB) — impact: P5/H1/opponent —
  [newsroom](https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis)
- [2026-07-17] [benchmark] **Bench2Drive-Speed (Mar 2026)** grades closed-loop *speed customization* —
  directly validates the program's speed/scale reset (v0-as-action-channel, probe R² 0.61→0.965) and is
  a Phase-1 closed-loop eval target. **Dev10** = 10-clip quick-dev subset (fast iteration); Bench2Drive-VL
  (Oct 2025) = closed-loop VLM QA. Seam: Benchmarks&Eval owns — impact: H4/speed/eval —
  [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
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

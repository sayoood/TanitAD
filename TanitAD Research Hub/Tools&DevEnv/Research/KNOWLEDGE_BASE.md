# KNOWLEDGE_BASE — Tools&DevEnv

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`

- [2026-07-11] [built] Commit gate `ci.ps1` (backlog #3) shipped (intake `2026-07-11-ci-gate/`):
  fail-fast (1) I2 batch-1==batch-B tripwire on the real WorldModel encoder (~2 s, catches the
  BatchNorm/batch-stat class that silently breaks the batch-1 Orin engine) then (2) full pytest
  suite via `profile_testsuite.py check` (timing guard). Measured dev machine 4060: PASS total
  17.2 s (I2 2.4 s + suite 14.8 s, 189 passed, warm overhead 1.43 s); falsifier holds — a 7.0 s
  test → exit 1 flagged >6 s. Pure-ASCII (PS 5.1 reads no-BOM .ps1 as system codepage). 0 new deps,
  G-T1 GO — impact: G-E/D-004-I2/CI — `2026-07-11-ci-gate-and-tensorrt-orin-qdq-trap.md` §1
- [2026-07-11] [gotcha] TensorRT INT8 Q/DQ export **passes on RTX, FAILS on Orin(TRT 10.3)+Thor
  (10.13.3)**: PyTorch Dynamo wraps Q/DQ scales behind reshape ops; x86/Ada silently constant-folds
  and masks it, ARM/Blackwell parser rejects (Thor segfault post-Q/DQ-fusion; Orin "input shape
  misaligns", both spam "invalid precision Int8, ignored"). Fix = emit Q/DQ scales as direct fp32
  initializers, not reshape-derived. **An RTX-clean INT8 export ≠ Orin/Thor-clean** → build with
  trtexec on-target, not only 4060 (rec to Prod-Opt `int8_quant/`). FP16 static-shape stays the
  primary Orin path — impact: H5/C1/P5 — [forum](https://forums.developer.nvidia.com/t/tensorrt-export-fails-on-both-jetson-agx-orin-and-thor-but-passes-on-rtx/363379)
- [2026-07-11] [tooling] **AlpaGym now PUBLIC** (`NVlabs/alpagym`, Apache-2.0): RL closed-loop
  post-training = AlpaSim envs + Cosmos-RL orchestration; default policy Alpamayo-1.5 **10 B, ≥2
  GPUs**, HF+W&B+uv, **no lightweight reference policy**. **VRAM floor UPDATED (lit sweep): 2×24 GB**
  (model card tested 3090/A100/H100/B200, not "40–60 GB single") + a documented local smoke config
  `experiment=alpamayo_1_5_local_2gpu_smoke`. Our single A40 (48 GB) still can't meet the 2-GPU
  default un-reconfigured; 4060 out. Verdict unchanged: **Phase-1 cloud, not Phase-0** (100×+ our
  envelope, P5) — now clonable w/ an official smoke path (scoped 2×GPU spike), useful as an
  oracle/data source not an adoptable policy; Phase-0 closed-loop stays CARLA-on-pod (D-014). Watch
  Cosmos-RL as a borrowable scorer/orchestrator — impact: P5/H1/opponent —
  [alpagym](https://github.com/NVlabs/alpagym)
- [2026-07-11] [tooling] **CARLA 0.10.0 = Unreal Engine 5.5** (Lumen/Nanite, remodeled Town10,
  InvertedAI traffic, native ROS). **Min spec RTX 3000 / ≥16 GB VRAM / Ubuntu 22.04|Win11.** We pin
  **0.9.16 (UE4.24)** deliberately; 0.10's 16 GB floor + UE5 rebuild = Phase-1 only, not a Phase-0
  switch. nullrhi SC-01 telemetry path unaffected; revisit 0.10 when photoreal pixels are
  eval-critical (pairs w/ graphics-pod recipe P1.3) — impact: P5/D-014 — [release](https://carla.org/2024/12/19/release-0.10.0/)
- [2026-07-11] [tooling] **Rerun 0.34.1 (2026-07-07)** — fresh `rerun-sdk` for episode replay/viz
  (backlog P0 #2 pick, already chosen 2026-07-06). New in 0.34: an **MCP that lets an LLM agent
  see/drive the Viewer** (dovetails with our agentic loop), a `VoxelGridMap` archetype, gamepad 3D
  nav. Pure-Python, no server infra. **G-T1 GO, setup ~15 min** (`pip install -U rerun-sdk`) —
  greenlights the episode→.rrd replay increment on the 4060 — impact: WP-viz/D3/H5 —
  [release](https://github.com/rerun-io/rerun/releases)
- [2026-07-11] [tooling] **Trackio** (HF, beta) — local-first **W&B drop-in**: `import trackio as
  wandb` (init/log/finish compatible), <1000 LOC, SQLite→Parquet backend, optional free sync to a
  private HF Space. No vendor lock-in, $0. Also **shims AlpaGym's W&B dependency** (finding above).
  Caveat: no artifact mgmt / advanced viz yet. **G-T1 GO, setup ~10 min** — candidate experiment
  tracker for our training runs (replaces ad-hoc JSON logs) — impact: G-E/dev-tooling —
  [trackio](https://github.com/gradio-app/trackio)
- [2026-07-11] [paper] **ZipDepth** (2607.08771, 07-09) — **6.1 M-param** monocular depth, real-time
  server→edge, distilled from a foundation model to near the zero-shot accuracy of ~50× larger nets.
  Squarely in our <100 M edge envelope → candidate cheap geometry / auxiliary perception signal on
  Orin-class targets (H16 active-depth banked, `7cd9d7b`). Was on the D-028 "historically-missed"
  list — now captured — impact: edge-efficiency/C1/H16 — [arXiv](https://arxiv.org/abs/2607.08771)
- [2026-07-11] [benchmark] **Bench2Drive-Robust** (2605.18059) — first closed-loop E2E-AD benchmark
  that perturbs **compute-induced control delay (model inference delay)** + camera-stream failure +
  ego-state noise: it scores exactly the degradations a driver hits on modest edge hardware, so our
  low-latency constraint becomes a **rewarded advantage** rather than a penalty. Companions
  Bench2Drive-VL / -Speed. → handoff to Benchmarks&Eval (Thu): add the inference-delay axis to the
  eval plan — impact: eval/edge-efficiency/C2 — [arXiv](https://arxiv.org/abs/2605.18059)
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

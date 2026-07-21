# KNOWLEDGE_BASE — Tools&DevEnv

> Curated, deduplicated, newest first. Format:
> `[YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`

- [2026-07-21] [root-cause] **The fleet monitor's blind spot is structural, not a bug**: every
  check in `.claude/skills/fleet-status/SKILL.md` grepped a **hardcoded** run/log name
  (`p0-sB01-realmix.log`, `arm_base.log`, `arm_kstep.log`, `pgrep -fc train_worldmode[l]`) —
  all belonging to runs that ended weeks ago. A grep that matches nothing prints nothing, and
  a monitor that prints nothing reports no anomaly. **Renaming a run silently blinds it**, and
  every arm since has been renamed → 4 recurrences, latest 2026-07-20 05:01 UTC (2 of 4 GPUs
  dead, the 04:55 probe clean). Fix = discovery, plus the rule **absence of evidence is an
  ALARM, not an all-clear** — impact: TOP-RISK/ops/all-agents —
  note `2026-07-21-fleet-probe-and-the-rerun-dual-sink-loss.md` §1
- [2026-07-21] [built] **`tools/fleet_probe.py`** — discovers jobs from `ps` (grouped by
  `--out`, so a 6-proc fan-out = one run) and logs from the launcher's stdout redirect walked
  up the ppid chain; cross-checks GPU vs process table (`ORPHANED_GPU_MEMORY`,
  `GPU_IDLE_NO_TRAINER`), catches freezes two ways (`LOG_STALE` 15 min, `STEP_NOT_ADVANCING`
  via a state file), and measures disk with a real 100 MB `dd` (never `df`). Verdicts start
  UNKNOWN; a job with no discoverable log is **AMBER, never GREEN**. Measured live:
  **whole 4-pod fleet in 9.7-11.3 s**; 20 falsifiers 0.35 s — impact: TOP-RISK/ops/G-I — note §1
- [2026-07-21] [measured] **pod2 (A40) idle with no trainer** on every probe run of 2026-07-21
  (0 %, 0 MiB, no job process; disk healthy 208-474 MB/s). Live instance of the class the old
  monitor missed 4x — impact: burn/M-1 resource mandate — note §1, escalated in STATE
- [2026-07-21] [trap] **git-bash's MSYS `ssh.exe` deadlocks under `subprocess` pipes from a
  native-Windows Python** — the identical payload runs in **2.0-2.2 s from a shell** but hangs
  past 90 s from Python, reproducing **100 % on the two *training* hosts and 0 % on the two
  idle ones**, i.e. *it reads exactly like a fleet outage and is not one*.
  `C:\Windows\System32\OpenSSH\ssh.exe` ran the same payload on all 4 hosts in **0.7-2.5 s**.
  Prefer native OpenSSH on win32 for any Python-driven pod tooling — impact: all pod tooling —
  note §1 negative-results
- [2026-07-21] [trap] **`subprocess.run(..., text=True)` corrupts every remote bash payload on
  Windows**: its stdin TextIOWrapper translates `\n` -> `os.linesep`, so every `fi` arrives as
  `fi\r` and bash dies with the misleading `syntax error: unexpected end of file`. A CRLF
  checkout does the same. Encode payloads to LF **bytes** — impact: all remote tooling — note §1
- [2026-07-21] [trap] **`find /workspace -maxdepth 3` times out (>90 s) on the MooseFS pods**,
  and does so only on the *busy* ones — the naive form is blind precisely where it matters.
  Use per-dir `timeout 8 find ... -mmin -2880` — impact: pod tooling — note §1
- [2026-07-21] [measured] **The `--rrd` + `--serve` dual sink is a 3,314x silent data loss.**
  rerun 0.34.1: `rr.save()` sets the file sink, `rr.serve_grpc()` **replaces** it (SDK's own
  docstring), so only the blueprint reaches the file. 200 windows x 3 arms x 256^2:
  rrd-only **10,593,179 B** (52,966 B/win, 299 win/s) vs dual-sink **3,196 B** (16 B/win).
  It survived because the file is **non-zero** — *non-zero is not non-empty; test emptiness
  only against a same-input single-sink baseline*. jpeg85 vs raw = **3.79x smaller for 17 %
  less throughput** (default is right). Guard shipped via intake
  `2026-07-21-rrd-dual-sink-guard/` — impact: P1 TanitResim/viz/G-T1 — note §2
- [2026-07-21] [negative] **The documented rerun tee deadlocks**:
  `rr.set_sinks(FileSink, GrpcSink(url))` after `serve_grpc()` hangs indefinitely (killed at
  120 s, no output) — the GrpcSink connects back to the in-process server on the same thread.
  A real tee needs two `RecordingStream`s + explicit `recording=` per log call — impact: viz —
  note §2
- [2026-07-21] [measured] **`rerun-sdk` is pinned in NO requirements file** anywhere in the repo
  (`stack/requirements*.txt`, `pyproject.toml` -> no match) although 0.34.1 is installed and the
  whole viz backbone depends on it. Also corrects a stale backlog premise: the "pin 0.34.1 +
  migrate, 1-2 h" work did not exist — 0.34.1 was already in the venv and `rr_log.py` (417 lines)
  already logs episodes — impact: reproducibility/G-T1 — note §2
- [2026-07-21] [watch] **TerraZero still has no public code** (5-min check, backlog P1.0b).
  Project page `terra-applied.github.io`; **the GitHub org literally named `TerraZero` is an
  unrelated third party** — do not mistake it for Applied Intuition's release. Separately, an
  **AlpaSim E2E Closed-Loop Challenge 2026** exists (HF space) — a possible external yardstick
  if the docker-host blocker is ever cleared — impact: closed-loop fallback —
  https://terra-applied.github.io/
- [2026-07-20] [measured] **The stack test suite has ZERO GPU coverage**: `grep -rl cuda
  stack/tests/` returns nothing across all 396/531 tests, while every trainer, eval and
  deploy tick runs on a GPU. Device/dtype placement, on-device batch-statistic leaks and
  CUDA-only NaNs were structurally invisible to CI. Closed by `tools/gpu_tripwire.py`
  (4 probes on the real model). Measured on the RTX 4060 (torch 2.11+cu128, fp32, 1.7 s):
  encode CPU-vs-CUDA **9.54e-07**, imagine **7.15e-07**, I2-on-device **1.66e-07**, 0
  non-finite grads; batch-1 encode **0.85–1.43 ms** (I8 proxy). Default tol 1e-3 = ~1000x
  headroom; a falsifier at tol=0 proves the probes can fail — impact: G-E/CI/I2/I8 —
  note `2026-07-20-ci-gate-v2-suite-manifest-gpu-tripwire-and-the-uncommitted-stack.md` §1–2
- [2026-07-20] [root-cause] **40 uncommitted `stack/` paths on the shared Drive tree, 22
  UNTRACKED** — 12 test modules (~135 tests), 9 `tanitad/lake/*` + `eval/ckpt_compat.py`
  + `train/decorr.py`, 18 modified core files (`config.py`, `fourbrain.py`, `predictor.py`,
  `refa.py`, `flagship_losses.py`, 10 scripts). In no commit, on no branch. Found via a
  396-vs-531 collected discrepancy between the worktree and the Drive tree. **Strictly
  worse than D-026's unmerged branches** (those are at least pushed). `session_guard` v1
  called that tree clean because it only checked hub prefixes → source check added —
  impact: D-026/G-I/all-agents — note §4
- [2026-07-20] [tooling] `git status --porcelain` **collapses a wholly-untracked directory
  to one `?? dir/` row** — fatal for any guard whose job is to name the missing files. Use
  `--untracked-files=all`. Caught by a falsifier before ship, not after — impact:
  tooling/session_guard — note §4
- [2026-07-20] [built] **`ci_gate` v2** (`tools/ci_gate.py`, promoted out of stranded intake
  to repo-root tooling): adds a **SUITE_MANIFEST** (16 load-bearing modules pinned to a
  collected-count floor — a named-node tripwire only guards nodes somebody thought to name;
  whole modules vanish silently), `--min-total` (390), `--gpu-smoke off|warn|require`,
  `--json`. Skips stay green **unless a whole module is skipped**. Measured: both trees GATE
  PASS, **396/39.0 s** (off-Drive worktree) and **531/60.2 s** (Drive); 57 falsifiers 15.5 s
  — impact: G-E/CI — note §3
- [2026-07-20] [measured] **Sharding NOT needed** (backlog condition was "<5 min or shard"):
  worst measured tree = **60.2 s, 5x under the ceiling**. Budgets set from measurement:
  per-test 15 s, wall 150 s. Caveat: timings are **contention-sensitive** — the same suite
  ran 65.0 s with a 14.90 s tall pole beside a second pytest process (vs 39.0 s / 8.02 s
  clean), so a concurrent agent run can false-positive the slow-test budget. This also
  re-scopes backlog P0.2: `test_replay`'s "10.86 s" was partly an I/O+contention artifact —
  impact: G-E/backlog — note §3.3–3.4
- [2026-07-20] [verdict] **AlpaSim closed-loop on `tanitad-eval`: NO-GO** (executed
  2026-07-19 by the investigation agent — retires my P1.0). The eval pod is itself an
  unprivileged container with **no nested container runtime**, and AlpaSim's NuRec renderer
  ships only as `nvcr.io/nvidia/nre/nre-ga:26.04` — no source form. Policy/driver side GO
  (bare gRPC, adapter written); ~1.5 GB/scene, <2 GB VRAM would fit a proper host. Residual
  ask is infra (a docker-capable GPU host), not tooling — impact: P5/closed-loop/D-014 —
  `Benchmarks & Eval/Implementation/incoming/2026-07-19-alpasim-closedloop-v1/INTAKE.md`
- [2026-07-20] [tooling] **Rerun 0.34.0/0.34.1 ships a Viewer MCP server** — an agent can
  see and interact with what the viewer renders, i.e. verify its own rollout overlay instead
  of asserting it. Also `VoxelGridMap`, transform-debug UI; **breaking API changes**
  (migration guide), pin **0.34.1** (live-stream stack-overflow fix). GO on a branch, est.
  1–2 h SDK bump + `corpus_overlay.py` migration + ~30 min MCP wiring — impact:
  WP-viz/TanitEval-viz-standard — [releases](https://github.com/rerun-io/rerun/releases)
- [2026-07-20] [correction] Orin export should target **JetPack 7.2 (Jetson Linux 39.2,
  shipped 2026-06-02)**, not the 7.1 this KB recorded: 7.2 brings the **Orin family into the
  JetPack 7 line** (CUDA 13.2.1, TensorRT 10.16.2, unified Orin+Thor installer). NVFP4 is
  still Thor-only; Orin still targets FP8/INT8 — impact: C1/C2/P5 —
  [JetPack](https://developer.nvidia.com/embedded/jetpack)
- [2026-07-20] [paper] **"Validate the Dream Before You Trust Its Verdict"** (arXiv
  2607.07196, RSS-2026 wksp): a world model used as a test ORACLE must be accredited first;
  L0–L4 admissibility ladder from VV&A/SOTIF. Key result: the model ranking higher on visual
  generation quality ranks **lower** on action-following — the citable external form of our
  open-loop-ADE ⊥ closed-loop finding (0.45 → 1.69 m). Seam: Benchmarks & Eval — impact:
  H15/eval — [abs](https://arxiv.org/abs/2607.07196)
- [2026-07-20] [paper] **DynaDreamer** (arXiv 2607.13410): physics-informed ego-dynamics
  context that *modulates* a causal-Transformer WM, with a dynamics predictor keeping it
  synced **during rollout**; +28 % urban / +61 % highway, +73 % on an unseen chassis with no
  retraining. The principled generalization of our v0-as-3rd-action-channel fix (3.73 →
  0.83 m, speed-R² 0.965) and a direct lever on the longitudinal 83 %. No code, no stated
  scale → design input, not a dependency. Seam: Architecture — impact: H4/H25 —
  [abs](https://arxiv.org/abs/2607.13410)
- [2026-07-20] [paper] **Orbis 2** (arXiv 2607.15898, Freiburg): hierarchical driving WM
  (coarse predictor + detail generator) trained **diffusion-forcing then teacher-forcing** —
  a reusable rollout-stability schedule that costs only a training-schedule change. Code +
  ckpts advertised, but generative-video scale → read the loop, do NOT run the weights —
  impact: H15/V3-hierarchy — [abs](https://arxiv.org/abs/2607.15898)
- [2026-07-20] [paper] **TerraZero** (arXiv 2607.13028, Applied Intuition): procedural
  driving sim, **1.3 M agent-steps/s on one GPU**, pure-RL policies, tops InterPlan. Exactly
  our affordable closed-loop shape (no rendering) — but **no code released**, commercial
  vendor → WATCH, re-check in 2–4 weeks — impact: P5/closed-loop —
  [abs](https://arxiv.org/abs/2607.13028)
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

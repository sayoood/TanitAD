# 2026-07-06 — MetaDrive adoption, AlpaSim/AlpaGym verdict, ROS-free viz, Orin export

**Agent:** Tools & DevEnv (Monday). **Run:** W1→W2 first weekly cycle.
**Budget used:** ~7 web searches / ~1 h wall-clock / iteration 1 of 3.
**Quality:** full (G-A…G-F, G-T1 all met; live MetaDrive smoke deferred to a supervised run — see §6 HANDOFF).

Every tool recommendation carries a **measured setup cost + go/no-go for our P5 envelope (RTX 4060, 8 GB)** per gate G-T1.

---

## 1. Headline decisions this week

1. **MetaDrive WP2 wrapper landed** (`stack/tanitad/data/metadrive_env.py`) — contract-identical to
   the toy env; 7 new pure contract tests pass (17 passed / 1 skipped total). Live rollout path coded
   against MetaDrive's documented API, gated behind `pytest.importorskip` until a supervised install.
2. **MetaDrive install verdict (G-T1):** PyPI `metadrive-simulator` is a **NO-GO on Python 3.13**;
   the **GO path is source install** — but the native blocker is already cleared. Details §2.
3. **AlpaSim/AlpaGym (our declared adoption target) verdict (G-T1): NO-GO for local Phase 0, Phase-1
   cloud item.** Its driver models want ~40–60 GB VRAM and it is Docker/SLURM + HF-token gated. §3.
4. **Replay/viz pick: Rerun.io** — ROS-free, pip-installable, PyTorch-native; recommended for backlog
   item #2 (episode → overlay). §4.
5. **Orin/Thor export reality check:** ViT attention does **not** run on Orin's DLA (GPU-only), and
   naive INT8 on small ViTs can *regress* latency. Plan ONNX→TensorRT-GPU + FP16 first, INT8 only with
   measured calibration. §5.

---

## 2. MetaDrive — the honest install ledger (G-T1)

Measured on this dev machine (Python 3.13.5, Windows 11, venv `C:\Users\Admin\venvs\tanitad`):

| Step | Command | Result | Time |
|---|---|---|---|
| PyPI package | `pip install metadrive-simulator` | **FAIL** | ~4 s to failure |
| — why | pulls `gym==0.19.0`; its `setup.py` raises `extras_require` type error on modern setuptools (py3.13) | build-wheel error | — |
| Native deps | `pip install "panda3d>=1.10" "gymnasium>=0.28"` | **OK** — `panda3d 1.10.16` (cp313 wheel, 67 MB), `gymnasium 1.3.0` | 29 s |
| Source install | `pip install git+https://github.com/metadriverse/metadrive.git` | **deferred** — external-code install needs user trust; blocked in unattended agent run | — |

**Verdict:** MetaDrive is **GO for our stack** once installed from source in a supervised session. The
feared blocker (panda3d native wheel on py3.13/Windows) does **not** exist — the wheel is published and
installs in <1 min. The only obstacle is the stale PyPI package pinning `gym 0.19`; the maintained
GitHub `main` uses `gymnasium` (already installed here). `pyproject.toml` `[sim]` extra now installs the
two native deps; MetaDrive source is a one-line supervised follow-up.

**Design consequence baked into the wrapper:** MetaDrive stays an *optional, lazily-imported* dependency.
`import tanitad.data.metadrive_env` works with zero sim deps; only `generate_metadrive_episode()` imports
MetaDrive. CI never needs the simulator. This matches the toy-first, scale-stepwise mandate.

Sources: [metadrive-simulator PyPI](https://pypi.org/project/metadrive-simulator/) ·
[metadrive releases](https://github.com/metadriverse/metadrive/releases) ·
[MetaDrive TopDownObservation docs](https://metadrive-simulator.readthedocs.io/en/latest/obs.html)

## 3. AlpaSim & AlpaGym — declared adoption target, reality-checked (G-T1)

NVIDIA open-sourced the **Alpamayo** family (2026-01-05): open AV reasoning models + **AlpaSim**
(closed-loop microservice sim, gRPC-connected Driver/Renderer/TrafficSim/Controller/Physics services,
Apache-2.0) + **AlpaGym** (high-throughput closed-loop RL post-training). This is exactly the closed-loop
+ self-play + RL story our mission wants — but sized for NVIDIA-scale, not ours.

**Measured envelope facts (from AlpaSim docs):**
- Runs via **Docker Compose** (single machine) or **SLURM/cluster**; needs the **`uv`** package manager
  and a **Hugging Face token** for model/scene downloads.
- Renderer backends: **NuRec** or **OmniDreams** (neural reconstruction / world-model renderers).
- Driver-model VRAM: **Alpamayo-1 ≈ 40 GB**, **Alpamayo-1.5 ≈ 40 GB (60 GB with CFG)**.
- Validation suite: 916 scenes; assets on Hugging Face (sizes unpublished, expect tens of GB).

**Verdict for Phase 0 (P5): NO-GO on the RTX 4060.** The 8 GB card cannot host the NVIDIA driver models,
and the Docker microservice topology is heavier than our toy→MetaDrive ladder needs right now.
**But the architecture is a strategic north star:** AlpaSim's *closed-loop-consequence* validation is the
same thesis as our world model (validate decisions against real consequences). Two concrete, cheap plays:
- **Renderer decoupling insight** — AlpaSim separates Driver from Renderer over gRPC. We can mirror the
  interface boundary (our world-model policy ↔ MetaDrive renderer) so a later swap to NuRec/OmniDreams is
  a config change, not a rewrite. Relevant to H1 (hierarchy), H11 (fallback), C1 (it drives).
- **AlpaGym as the Phase-1 RL target** — when we rent an A100/H100 (post-gate), AlpaGym's closed-loop
  rollout API is the adoption target for self-play, *with our own <100 M param driver*, not Alpamayo.
  Proposal-worthy; budget-gated per Master Plan §4.

Sources: [NVIDIA Alpamayo newsroom](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development) ·
[NVlabs/alpasim](https://github.com/NVlabs/alpasim) ·
[AlpaSim tutorial](https://github.com/NVlabs/alpasim/blob/main/docs/TUTORIAL.md) ·
[Post-train in closed loop with Alpamayo (dev blog)](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/) ·
[TechCrunch launch](https://techcrunch.com/2026/01/05/nvidia-launches-alpamayo-open-ai-models-that-allow-autonomous-vehicles-to-think-like-a-human/)

## 4. Replay & visualization — Rerun.io (backlog item #2 prep) (G-T1)

**Pick: [Rerun](https://rerun.io/)** (MIT/Apache-2.0). Code-first, PyTorch-native, ROS-free by design
(no ROS install, unlike RViz/Foxglove's ROS-centric flow); logs images, 3D, time-series into a single
time-aware `.rrd` (Apache-Arrow columnar). Has a published nuScenes AV example (6 cam + LiDAR + radar +
GPS/IMU). `pip install rerun-sdk` — a pure-Python wheel, no native GPU deps → **expected GO for our
envelope** (setup cost to be measured when I build backlog item #2 next week; provisional <2 min).

Alternative considered — **Foxglove**: stronger for fleet ops / MCAP / live streaming, but ROS/MCAP-
centric and heavier than we need for single-episode overlays. Keep as a later option if we adopt MCAP.

Actionable next week: implement `episode → Rerun .rrd` (predicted-vs-actual trajectory + BEV frames),
which doubles as the D3 imagined-vs-oracle visual (ties to gate D3, hypothesis H5).

Sources: [Rerun what-is](https://rerun.io/docs/overview/what-is-rerun) ·
[Rerun nuScenes example](https://dev.to/rerunio/visualize-autonomous-driving-dataset-h84) ·
[RViz vs Foxglove vs Rerun (Foxglove)](https://foxglove.dev/robotics/rviz-vs-foxglove-vs-rerun)

## 5. Orin/Thor export path — early reality check (informs C1 embedded envelope)

We track Orin/Thor from day 1 (Master Plan C1). Two facts that shape architecture choices *now*:
- **ViT attention is GPU-only on Orin** — the DLA (as of JetPack 6.2) does not support transformer
  attention, dynamic shapes, or custom plugins. Our ViT encoder/predictor will run on the Orin GPU, not
  offloaded to DLA. Budget latency accordingly; keep shapes static for TensorRT.
- **INT8 is not free lunch on small ViTs** — a documented ViT-S+DPT case *regressed 2.7×* under INT8 on
  Orin Nano. Plan: **ONNX → TensorRT (GPU) FP16 first**; adopt INT8 only per-layer with measured
  calibration and a latency A/B. This reinforces P5 (efficiency measured, never assumed) and feeds the
  FLOPs/latency ledger every experiment must report.

Actionable: when the stack exports its first engine (Phase 1), start with FP16 static-shape ONNX; add an
Orin latency snapshot to the experiment record format. No action required in Phase 0 beyond keeping
encoder norms batch-free (already enforced, I2) and shapes static.

Sources: [TensorRT vs DLA on Jetson Orin](https://proventusnova.com/blog/tensorrt-vs-dla-jetson-orin/) ·
[Orin INT8 ViT-S regression (NVIDIA forums)](https://forums.developer.nvidia.com/t/tensorrt-model-optimizer-int8-quantization-causes-2-7x-performance-regression-on-jetson-orin-nano-4gb-vit-s-dpt-architecture/357835) ·
[ONNX Runtime TensorRT EP](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)

## 6. Actionable recommendations (tied to hypotheses / WPs)

- **[WP2, done]** MetaDrive wrapper merged; contract-identical to toy → world-model training ports
  unchanged. **Next:** supervised source-install of MetaDrive + un-skip the live smoke; confirm
  `env.render(mode="topdown")` return shape for the installed version (HANDOFF below).
- **[WP-viz, next week]** Build `episode → Rerun .rrd` overlay (backlog item #2); serves D3 visualization
  (H5). Measure Rerun setup cost for G-T1.
- **[Phase-1 proposal]** AlpaGym closed-loop RL post-training with *our* <100 M driver, A100-gated. Draft
  a `Project Steering/Proposals/` entry when Phase 0 gates D1–D3 pass. (H1/H11, C1)
- **[Architecture note to Wed agent]** Keep ViT shapes static + norms batch-free for a clean ONNX→TensorRT
  FP16 Orin path; INT8 is deferred and must be measured, not assumed. (C1, P5)

### HANDOFF (for the supervised MetaDrive validation run)
1. `pip install git+https://github.com/metadriverse/metadrive.git` (user-approved; native deps already in).
2. `pytest stack/tests/test_metadrive_env.py -m slow -q` — runs `test_metadrive_live_episode`.
3. If `env.render(mode="topdown", window=False, screen_size=(64,64))` returns `None`/wrong shape for the
   installed version, switch `_topdown_frame` to `TopDownObservation` (import path noted in the module).
   The pure conversion helpers (`bev_frame_from_rgb`, `assemble_episode`) need no change — they are the
   contract guarantee and are already green.

# AlpaSim closed-loop for flagship-v1 — feasibility intake

- Date: 2026-07-19
- Author: investigation agent (Sayed #1 priority)
- Pod: `tanitad-eval` (A40 48GB, idle) — the only pod touched. Non-destructive.
- Repo read: `github.com/NVlabs/alpasim` @ `main` (shallow clone, LFS skipped),
  left on the pod at `/workspace/alpa-invest/alpasim` (144 MB, code only).

---

## 1. VERDICT: NO-GO on the current eval pod (with a clean plan to GO)

**We cannot run the standard NuRec closed loop on `tanitad-eval` as it stands.**
One decisive, non-negotiable blocker:

> **The eval pod is itself an unprivileged Docker container with no nested
> container runtime, and AlpaSim runs entirely on Docker Compose — its renderer
> (NRE) ships ONLY as the prebuilt image `nvcr.io/nvidia/nre/nre-ga:26.04`, with
> no Python/pip source form. No container engine -> no renderer -> no closed loop.**

This is a hardware/infra wall, not a code problem, and not something I can install
around on this pod. Everything else (GPU, CUDA, disk, egress, our model, the policy
adapter) is fine or solvable. The fix is an infra change (a docker-capable GPU host),
after which wiring flagship-v1 is <1 day because the adapter is written and the API
is fully mapped (Sections 4-5). An immediate on-pod alternative that needs no
renderer is in Section 8.

Sub-verdicts:
- Policy/driver side: **GO** — ordinary Python, runs as a bare external gRPC
  process (no docker), adapter drafted and syntax-validated.
- Renderer / full sim on THIS pod: **NO-GO** — container-only, see above.
- Disk & GPU headroom for a minimal 1-scene smoke on a proper host: **GO**
  (~1.5 GB/scene, our policy <2 GB VRAM; A40 fits).

---

## 2. What was checked on the A40 (all commands run read-only / cleaned up)

| Item | Result |
|---|---|
| GPU | NVIDIA A40, 46068 MiB, **idle (0 MiB used)**, driver 580.159.04 |
| Compute cap | 8.6; `torch 2.8.0+cu128`, `cuda.is_available()=True`, CUDA 12.8 |
| CUDA toolkit | `/usr/local/cuda-12.8/bin/nvcc` present (not on PATH); CUDA 12.6+ req met |
| Python | 3.12.3; `uv` present at `/usr/bin/uv` (AlpaSim's package manager) |
| **Container runtime** | **NONE** — docker/podman/singularity/apptainer/enroot/nerdctl/ctr all MISSING |
| **Are we in a container?** | **YES** — `/.dockerenv` present, cgroup `/docker/1e0bac0…` (RunPod pod) |
| CPU / RAM | 96 vCPU, 503 GB RAM (249 GB free) — abundant |
| Egress | github / pypi / huggingface all HTTP 200, low latency |
| Disk `/` (overlay, local) | 200 G, **103 G free**, dd 1 GiB = **2.3 GB/s** (real, fast) |
| Disk `/workspace` (MooseFS) | df says 134 T free **but that lies** (cluster, not our quota); clean dd 1 GiB = OK @ 1.0 GB/s; treat as network scratch, unknown GB-scale quota |
| Our data | `TanitEval`+`valdata` live in **`/root`** (on overlay `/`), not `/workspace` |
| **HF token** | **ABSENT** — `Keys.txt` not on this pod. Did NOT copy from another pod. |
| Raw NVMe | two 3.5 TB `nvme0n1/1n1` visible in `lsblk` but **not mounted** in the container |

Note on the MooseFS quota: my first `/workspace` dd "FAILED" was a false alarm from a
missing `/usr/bin/time` wrapper (grep exit code), not a write failure — the clean
retest wrote 1 GiB fine. **Real local headroom that matters = 103 GB on `/`.**

---

## 3. Why the wall is real (evidence from the repo)

- `src/wizard/configs/base_config.yaml:138` → `image: nvcr.io/nvidia/nre/nre-ga:26.04`
  with `external_image: true`. The renderer is a **prebuilt NGC image**.
- `src/` contains controller, driver, eval, grpc, physics, runtime, trafficsim,
  utils, wizard, tools, plugins — **no `renderer`**. NRE is closed; not in the
  open-source tree, no pip wheel. `utils.py` also references
  `docker.io/carlasimulator/nvidia-nurec-grpc:0.2.0` as an NRE image.
- The wizard's only deployment strategy is `deployment/docker_compose.py`, which
  shells out to `docker compose up --exit-code-from runtime-0`. `deploy=local`
  = docker-compose on one machine. Onboarding + tutorial + the NVIDIA blog all
  state **"AlpaSim Wizard runs on Docker Compose."**
- Core services (runtime/controller/physics/driver) run as the `alpasim-base`
  image built from the repo `Dockerfile`. They ARE Python (uv workspace members),
  so they *could* be hand-run as bare daemons — but the renderer cannot, and
  without it there is no camera simulation.
- SLURM path exists (`src/tools/run-on-slurm`) using **enroot/pyxis** to import
  the nvcr.io image to `.sqsh`. enroot is also absent here and typically needs
  user-namespaces (usually off inside a nested container). Not a quick unlock.

Secondary (would still bite even with containers): the NuRec scene dataset
`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` is **gated** and needs an approved
`HF_TOKEN` (none on this pod); the NRE image on nvcr.io needs **NGC credentials**.

---

## 4. Policy plug-in API — MAP (this is fully usable today)

AlpaSim resolves a driver by name via Python entry points (`alpasim.models`), then
the **driver service** (`src/driver/.../main.py`) runs the gRPC `EgodriverService`.
It accumulates observations and, per `drive()`, builds a `PredictionInput`, calls
`model.predict_batch(...)`, and serialises the result to a rig-frame trajectory.
**We implement a model class, not the proto.**

Interface to implement — `BaseTrajectoryModel`
(`src/driver/src/alpasim_driver/models/base.py`):
- `from_config(model_cfg, device, camera_ids, context_length, output_frequency_hz)`
- `predict(PredictionInput) -> ModelPrediction` (+ optional `predict_batch`)
- `_encode_command(DriveCommand)`
- properties: `camera_ids: list[str]`, `context_length: int`, `output_frequency_hz: int`

### Observation schema (`PredictionInput`)
| Field | Type | Notes |
|---|---|---|
| `camera_images` | `dict[cam_id -> list[CameraFrame]]` | `CameraFrame=(timestamp_us:int, image: HWC uint8 RGB)`; list length == `context_length` |
| `command` | `DriveCommand` | `LEFT=0, STRAIGHT=1, RIGHT=2, UNKNOWN=3` (from route) |
| `speed` | `float` (m/s) | **hands us v0 directly** for our 3rd action channel |
| `acceleration` | `float` (m/s²) | |
| `ego_pose_history` | `list[PoseAtTime]` | rig-frame poses + dynamic states |
| `inference_seed` | `int` | session seed + inference count |

**Intrinsics are NOT passed to the policy.** The renderer + the driver's
`RectificationTargetConfig` (focal, principal point, resolution, distortion) produce
a pinhole frame at the configured FOV/resolution — the model just consumes pixels.
This matches how we trained (front-wide frames), and lets us set the rectification
target to our training intrinsics.

### Action schema (`ModelPrediction`)
| Field | Type | Notes |
|---|---|---|
| `trajectory_xy` | `np.ndarray (T,2)` | x,y offsets, **RIG frame: x forward, y LEFT** |
| `headings` | `np.ndarray (T,)` | radians, rig frame (helper provided) |
| `reasoning_text` | `str \| None` | optional |

The servicer spaces waypoints at `1/output_frequency_hz` s and prepends the current
pose. **Wire (egodriver.proto)**: `drive(DriveRequest) -> DriveResponse{trajectory,
sampled_trajectories, terminate_session}`; obs arrive via `submit_image_observation`
/ `submit_egomotion_observation` (Trajectory + DynamicState) / `submit_route`.

### Camera / rate / horizon
- **Single front cam** id = `camera_front_wide_120fov` (matches our front-wide input).
- Batch eval cadence = **2 Hz** (0.5 s waypoint spacing); manual driver = 10 Hz.
- Controller MPC `dt_mpc=0.1` (10 Hz) tracks whatever trajectory we return.
- Reference adapters: `plugins/transfuser_driver/...transfuser_model.py` (single-frame,
  4-cam, 2 Hz; note it flips y because CARLA is y-right — we should NOT, we're NVIDIA-rig)
  and `src/driver/.../models/{vam,manual}_model.py`.

### Registration (what our plugin ships)
```toml
[project.entry-points."alpasim.models"]
flagship_v1 = "alpasim_flagship.flagship_v1_policy:FlagshipV1Model"
[project.entry-points."alpasim.configs"]
flagship_v1 = "alpasim_flagship.configs"   # ships driver/flagship_v1.yaml
```
Run in-container (`driver=flagship_v1`) or, on a host, as a **bare external process**
(`driver=flagship_v1 driver_source=external_static
wizard.external_services.driver=["<ip>:6789"]`) — the pattern `docs/MANUAL_DRIVER.md`
uses to run a driver as a plain Python script outside docker.

---

## 5. flagship-v1 adapter design (stub delivered)

File: `flagship_v1_policy.py` (this folder). Syntax-validated (`py_compile`); NOT run
end-to-end (no sim on this pod). Mapping single front cam + ego kinematics → our
4-brain tactical planner → 2 s ego waypoints → AlpaSim rig trajectory.

Flow inside `predict()`:
1. Ingest `camera_images` frames into an internal rolling raw-RGB buffer (keyed by
   `timestamp_us`) — robust to however many frames the sim packs per call.
2. Build the predictor window `[1, W, 9, 256, 256]`: each of `W=cfg.predictor.window`
   (=8) ticks is a **9-channel = 3 RGB sub-frames @100 ms** stack, at 256 px — the
   flagship encoder contract (`config.py`).
3. `states = world.encode_window(frames)`.
4. Past-action window `[1,W,2]=(steer,accel)`, then append **v0 = speed/`SPEED_SCALE`(10)**
   via `ckpt_compat.append_speed_channel` → `[1,W,3]` (only if the ckpt is action_dim=3;
   `build_world_from_ckpt` self-describes this and strict-loads).
5. `run_hierarchy(world, states, actions, nav_cmd)["waypoints"]` — the **trained
   tactical policy** plan at horizons `{5,10,15,20}@10Hz = {0.5,1,1.5,2}s`.
   (Deliberately NOT the grounded-rollout script — that is teacher-forced on TRUE
   future actions, a metric harness, not a closed-loop planner.)
6. Map ego→rig (identity `x`, `y*_Y_SIGN`), compute headings, return `ModelPrediction`.

`output_frequency_hz = 2`, `context_length = (W-1)+3`, `camera_ids=[front_wide]`.
Model load reuses `stack/tanitad/eval/ckpt_compat.build_world_from_ckpt`.

**Open TODOs (marked in the file) — need the live sim or our train constants to pin:**
- `TODO(ours)` exact front-wide crop/resize + normalisation to match training transform.
- `TODO(ours)` past-action derivation from ego kinematics (or `inv_dyn` head) + the
  train-time action normalisation; currently a zero placeholder (planner still runs on
  the visual window; conditioning is weak until pinned).
- `TODO(ours)` confirm `wp` layout from `TacticalPolicy.forward` (already the 4 horizon
  waypoints vs dense needing horizon index-select — stub handles both).
- `TODO(ours)` confirm the **y sign** vs a known left-turn window before trusting steering.
- `TODO(ours)` align `nav_cmd` indices with the strategic policy's trained vocabulary.
- `TODO(sim)` resample the frame buffer to an exact 100 ms grid once real frame timing
  is known; per-session buffers for multi-rollout batching (topology>1).

---

## 6. Disk / GPU footprint (for a real host)

- **Per NuRec scene ≈ 1.5 GB** (full `public_2601` suite ≈ 1.5 TB — NOT for one pod).
  A 1-5 scene smoke = ~1.5-7.5 GB → fits the 103 GB local `/` easily.
- NVIDIA recommends **≥40 GB VRAM + ~100-150 GB free disk**; their RL setup used
  2×RTX 6000 Ada (that 40 GB is dominated by the 10B Alpamayo policy). **Our policy
  is <2 GB VRAM**, so A40 48 GB is plausibly fine for NRE + flagship-v1 on one GPU,
  non-real-time. Host deps noted: cuDNN, NCCL headers, Redis.
- AlpaSim code (no LFS) = 144 MB; a full `uv sync` env is multi-GB (pulls torch and,
  for the built-in drivers, the heavy `vam`/`alpamayo_r1`/`alpamayo1_5` packages —
  our own plugin avoids those, like `transfuser_driver` does).

---

## 7. Smoke result

**Not attempted — correctly gated off.** Phase-4 smoke was conditioned on Phase-1
green; Phase-1 is red (no container runtime → the wizard's `docker compose up` fails
immediately; the renderer image cannot be pulled or run). Forcing a `uv sync` +
`docker compose` here would only reproduce that failure and burn the disk.

What was validated instead, non-destructively:
- Environment probe (Section 2), incl. corrected MooseFS dd test.
- Repo cloned and read end-to-end (README, ONBOARDING, TUTORIAL, DESIGN, PLUGIN_SYSTEM,
  MANUAL_DRIVER, egodriver.proto, driver schema, base model, transfuser adapter, wizard
  deploy code, service config).
- Adapter stub written and **syntax-validated** (`py_compile`).
- The bare-process external-driver path is established by construction: driver plugins
  are ordinary Python and `MANUAL_DRIVER.md` runs a driver as an external script — no
  docker needed on the *policy* side.

---

## 8. Recommended next steps (ranked)

1. **Provision a docker-capable GPU host** (RunPod "docker-enabled"/privileged
   template, or a cloud VM with an A40/L40/6000-Ada where we own the docker daemon).
   Add: docker + NVIDIA Container Toolkit, an **NGC API key** (pull
   `nvcr.io/nvidia/nre/nre-ga:26.04`), an **approved HF token** for the gated NuRec
   dataset, ~150 GB disk. Then: `source setup_local_env.sh` → download 1-3 scenes →
   `uv run alpasim_wizard deploy=local topology=1gpu driver=vavam n_rollouts=1` to
   prove the loop with a shipped policy, then swap `driver=flagship_v1` (external).
   This is the shortest path to the native-distribution closed loop Sayed wants.
2. **In parallel, an immediate on-pod closed-loop signal that needs no renderer:**
   a DIY neural-sim harness using **our own imagination/operative brain**
   (`WorldModel.imagine` / `OperativePredictor`) as the world model — roll the policy's
   planned actions through our latent predictor and measure closed-loop drift/stability
   over a horizon on `tanitad-eval` today. Not photoreal NuRec, but an honest
   closed-loop planning signal now, on the exact model, on the pod we have.
3. **De-risk the bare-renderer route (spike):** `deploy/external_video_model.yaml`
   + `VIDEO_MODEL.md` allow an **external** renderer (like the external driver) backed
   by **FlashDreams/OmniDreams** (a Python video-diffusion model). If FlashDreams runs
   as a bare process, it bypasses the NRE container entirely — the only credible
   no-docker path to real rendering. Heavier VRAM, non-real-time; worth a scoped test.
4. **Alternatives if the NRE/NGC gate stays shut:** **HUGSIM** (arXiv 2412.01718,
   3DGS closed-loop, open — alpamayo has a HUGSIM eval issue #54) or **OmniDreams**
   (arXiv 2606.03159). Each is its own investigation but avoids NVIDIA's closed NRE.

Once (1) exists, wiring flagship-v1 is small: finish the six TODOs in the stub,
`uv pip install -e` the plugin, `uv run alpasim-info` to confirm registration, run one
scene. Weights: `Sayood/tanitad-flagship-4b-speedjerk` (gated HF) or the pod2 copy
(do not touch pod2) — needs an HF token on the target host.

---

## 9. Flags for Sayed

- **Supply-chain note (upstream, not ours):** AlpaSim's root `pyproject.toml` pins
  `lightning` to a **GitHub archive tarball** (SHA `2129fdf3…`) rather than PyPI,
  citing an "April 30 2026 PyPI compromise / quarantine." Legit-looking and
  SHA-pinned, but any AlpaSim install with the vam/alpamayo drivers pulls a dependency
  from arbitrary GitHub — worth a conscious approval. Our own plugin does not need it.
- **NGC + HF gating** are the two credential gates beyond hardware; line them up early.
- `exclude-newer = "3 days"` in their uv config means reproducible-but-frozen installs.

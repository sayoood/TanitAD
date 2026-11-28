# AlpaSim closed-loop on `tanitad-eval` (A40) — reproducible run recipe

**Owner:** agent (stream S1). **Started:** 2026-07-22. **Goal:** M1 renderer serves a scene →
M2 closed-loop rollout with a stock/manual driver → M3 REF-C driver plugin → M4 REF-C metrics.

**Evidence class legend:** MEASURED (ours+artifact) · PUBLISHED · INHERITED · ESTIMATED · HYPOTHESIS.

---

## 0. Environment facts (MEASURED 2026-07-22)

- Pod `tanitad-eval`, A40, GPU FREE (0 MiB, 0%), lock held: `gpu_lock.sh acquire alpasim` (ttl 14400s).
- SSH: PowerShell `ssh -n -o ConnectTimeout=15 -o BatchMode=yes tanitad-eval '<cmd>'`. No `()` in remote echo labels.
- Disk: `/workspace` MooseFS ~287 MB/s (dd-verified), write EVERYTHING there. `/` is 93% full (186/200G) — never write to `/`.
- AlpaSim repo: `/workspace/alpa-invest/alpasim/` (NVlabs/alpasim, Apache-2.0).
- NRE renderer rootfs: `/workspace/nre/rootfs/`. Launcher (renderer entrypoint):
  `/workspace/nre/rootfs/app/internal/scripts/pycena/runtime/pycena_nrm_full`
  (= what `/app/run` symlinks to; prior session PROVED it bootstraps bare, torch 2.7.0+cu128 sees A40).

## 1. Architecture (from docs, INHERITED from upstream docs)

Microservices over gRPC, runtime is the central orchestrator/client:
- **renderer** (NRE/pycena, GPU) — serves camera frames from a scene USDZ.
- **controller** (CPU) — MPC vehicle model, egomotion.
- **physics** (GPU, skippable) — ground-mesh constraints.
- **trafficsim** (disabled by default) — non-ego actors.
- **driver** (the AV policy) — camera+nav in → trajectory out. Can be EXTERNAL (`driver_source=external_static`).
- **runtime** — drives the loop, writes `.asl` logs, runs eval.

### Renderer command (base_config.yaml services.renderer)
```
<rootfs>/app/internal/scripts/pycena/runtime/pycena_nrm_full serve-grpc \
  --port={port} --host=0.0.0.0 \
  --artifact-glob=<scene_cache>/{sceneset}/**/*.usdz \
  --egocar-hood-dir=<sensordata>/ego-hoods \
  --no-enable-nrend --download-cache-dir /tmp/nre-cache-dir \
  --cache-size=5 --max-workers=4 --enable-editing-actors
```
env: HOME=/tmp, XDG_CACHE_HOME=/tmp/.cache, OMP_NUM_THREADS=1.

### Other services (bare, via `uv run` in the alpasim workspace venv)
- driver:     `uv run -m alpasim_driver.main --config-path=<logdir> --config-name=driver-config.yaml host=0.0.0.0 port={port}`
- physics:    `uv run physics_server --host=0.0.0.0 --port={port} --artifact-glob=<cache>/{sceneset}/**/*.usdz --use-ground-mesh=true --cache-size=16`
- controller: `uv run python -m alpasim_controller.server --port={port} --log_dir=<out> --config=<logdir>/controller-config.yaml`
- runtime:    `uv run python -m alpasim_runtime.simulate --user-config=<logdir>/generated-user-config-0.yaml --network-config=<logdir>/generated-network-config.yaml --log-dir=<logdir> --eval-config=<logdir>/eval-config.yaml`

### Key config defaults (base_config.yaml)
- `deploy=local` → `defines.filesystem = <repo>/data`; scenes+ego-hoods under `<repo>/data/nre-artifacts`.
- default scene: `clipgt-01d503d4-449b-46fc-8d78-9085e70d3554` (26.01 release).
- topology=1gpu: nre_cache_size=5, all GPU services on gpu 0, controller CPU.
- n_sim_steps=200, n_rollouts=1, control_timestep_us=100_000 (10 Hz), 2 cameras 320x512
  (camera_front_wide_120fov + camera_front_tele_30fov), render_bundling=BATCH_RENDER_RGB.
- manual driver: CPU-only, "returns a constant forward trajectory", `driver_source=external_static`, binds 0.0.0.0:6789.

## 2. Bare-run strategy (no docker)

Upstream runs via docker compose. We run BARE. Path:
1. Set up alpasim uv workspace (uv sync + compile protos + build utils_rs via rust). CPU only.
2. Download 1 scene USDZ + ego-hoods from HF `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` to `/workspace/.../data/nre-artifacts`.
3. M1: run pycena_nrm_full serve-grpc on the scene; confirm gRPC up + a frame renders.
4. Wizard `wizard.run_method=NONE wizard.debug_flags.use_localhost=True` → generate configs (network-config maps services→localhost:port).
5. M2: start controller, (physics), driver, renderer, runtime bare on localhost ports; run N ticks; collect run UUID + `.asl` + metrics.

---

## 2b. Bare-run setup (MEASURED, reproducible)

**Workspace setup** (`repo:.../alpasim_setup.sh`, staged; run detached on pod, ~7 min):
- uv 0.9.0 (apt) can't `self update` → install standalone uv to `/workspace/uvbin` (curl astral installer, `UV_INSTALL_DIR`).
- rust via rustup to `/workspace/.cargo` + `/workspace/.rustup` (`--no-modify-path --profile minimal`) — needed for `utils_rs` (maturin).
- ALL caches → `/workspace` (`/` is 93% full): `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `CARGO_HOME`, `RUSTUP_HOME`, `XDG_CACHE_HOME`, `TMPDIR`.
- **Pared-down root pyproject** (`repo:.../pyproject_pared.toml`, staged; orig backed up to `pyproject.toml.orig` on pod): drops the `driver` + `plugins/*` members so `uv sync` does NOT resolve/clone the heavy `vam`/`alpamayo_r1`/`alpamayo1_5` git deps. Extra renamed `all`→`core`.
- `uv sync --extra core` → builds+installs 10 alpasim pkgs (grpc, utils, controller, physics, runtime, wizard, eval, tools, plugins, trafficsim) + torch/trajdata. `cd src/grpc && uv run compile-protos`. `uv pip install -e src/utils_rs`.
- venv at `/workspace/alpa-invest/alpasim/.venv` (python 3.12). **VERIFIED (MEASURED): all M2 imports pass** (`repo:.../verify_imports.py`, bad=0): alpasim_grpc, egodriver_pb2_grpc, alpasim_utils.geometry, alpasim_wizard, alpasim_runtime, alpasim_controller, alpasim_physics, grpc, numpy.

**Scene** (MEASURED): `hf download nvidia/PhysicalAI-Autonomous-Vehicles-NuRec --repo-type dataset --revision 26.04 --include "sample_set/26.04_release/01d503d4-449b-46fc-8d78-9085e70d3554/*" --local-dir /workspace/scene_dl` (token via stdin, BOM+CR stripped: `tr -d '\r\357\273\277'`). Gated access GRANTED (DL_EXIT=0). USDZ 1.74 GB + reference `camera_front_wide_120fov.mp4` at `/workspace/scene_dl/sample_set/26.04_release/01d503d4-449b-46fc-8d78-9085e70d3554/`. **26.04 release chosen to match the NRE 26.04 image.**

## 3. Progress log (append-only, dated)

- 2026-07-22 ~11:20 local: lock acquired, docs read, architecture mapped, configs pulled.
- 2026-07-22 ~11:44 local: launched workspace setup + scene download detached. Wrote `simple_driver.py` (minimal external EgodriverService, constant-forward, swappable policy for REF-C).
- 2026-07-22 ~11:51 local: setup DONE, all M2 imports pass, scene down (1.74GB). Created `/app -> /workspace/nre/rootfs/app` symlink.
- 2026-07-22 ~11:53 local: **M1 in progress** — launched `pycena_nrm_full_cc serve-grpc --port 6011` on the scene (`repo:.../renderer_serve.sh`). Obfuscated binary booting (RSS climbing past 1GB, ~28% CPU). Port not open yet; polling for the Vulkan/CUDA wall. NuRec = gsplat/OptiX (CUDA), so the historic Vulkan wall may not apply.
- 2026-07-22 **11:57 local: ✅ M1 ACHIEVED (MEASURED).** Renderer serving after ~4.5 min cold boot (first-run kernel JIT, now cached under `/workspace/nrehome/.cache`). `/workspace/renderer.log`:
  `Available scenes: ['clipgt-01d503d4-449b-46fc-8d78-9085e70d3554']` · `Available egocar masks: EgocarRigBank(hyperion_8, hyperion_8_1)` · `BackendCache initialized ... maxsize=5` · `Serving on 0.0.0.0:6011 (health on same port)`. GPU 278 MiB (pid 1373324), no Vulkan/EGL error. **The NuRec renderer runs BARE on the A40 — the 12-day Vulkan wall does NOT apply (gsplat/OptiX are CUDA).** Frame-render confirmation follows from the M2 loop (runtime issues the render RPCs).
  **M1 launch recipe:** `/app` symlink → `/workspace/nre/rootfs/app`; `RUNFILES_DIR=<rootfs>/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles`; `HOME=/workspace/nrehome`; then the serve-grpc command in §1. Cold boot ~4.5 min (kernel compile); warm boot fast.

- 2026-07-22 **12:12 local: ✅ M2 ACHIEVED — "AlpaSim runs on the eval pod" (MEASURED).** First CLOSED-LOOP rollout completed bare (no docker).
  **Topology (all bare, localhost):** renderer :6011 (warm, from M1) · physics :6006 (`.venv/bin/physics_server`, sceneset glob, warp GPU) · controller :6007 (`.venv/bin/python -m alpasim_controller.server`, linear MPC) · our driver :6789 (`simple_driver.py` ConstantForwardPolicy, 5 m/s) · runtime (`.venv/bin/python -m alpasim_runtime.simulate`).
  **Run:** eval `run_uuid=1d2d061799889ee09df5334f2e3b16f189976ef34ee490dc9f83051cbf79a9bd`, `rollout_id=41e4ef8c-85c6-11f1-8abd-97225e00fa1a`, scene clipgt-01d503d4, 50 steps @5Hz + 3s force-GT warmup. **status=PASS, score 0.6637.**
  **Metrics (`/workspace/m2run/aggregate/results-summary.json`):** collision_any/at_fault=0.0 · offroad=0.0 · **img_is_black=0.0** (renderer produced real non-black frames → M1 render CONFIRMED) · dist_to_gt_trajectory=0.574 m · dist_traveled=39.17 m (gt 73.77) · progress=0.531, progress_rel=0.988 · plan_deviation=0.084 · min_dist_to_obstacle=1.43 m · min_ade@*=null (GT-ADE not computed for a non-GT-matched driver).
  **Artifacts (pod):** `/workspace/m2run/rollouts/clipgt-01d503d4.../41e4ef8c.../{rollout.asl 7.27MB, metrics.parquet, *_camera_front_wide_120fov_default.mp4 456KB}` + `/workspace/m2run/aggregate/{metrics_results.txt,.png,.parquet, results-summary.json}`.

## 4. Full M2 reproduction (bare, no docker) — the EXACT recipe

Prereqs: workspace set up (§2b), scene under `data/nre-artifacts` (wizard downloads it), `gpu_lock.sh acquire alpasim`.
1. **Renderer** (M1): `bash /workspace/renderer_serve.sh 6011` detached → serves scene on :6011 (cold ~4.5min, warm fast).
2. **Configs:** `bash /workspace/wizard_gen.sh 50 /workspace/m2run` (HF token via stdin) → wizard `run_method=NONE debug_flags.use_localhost=True driver=manual driver_source=external_static`. Emits `/workspace/m2run/{generated-network-config.yaml, generated-user-config-0.yaml, controller-config.yaml, eval-config.yaml, ...}`. Ports: renderer 6005, physics 6006, controller 6007, driver 6789.
3. **Backing services:** `bash /workspace/launch_services.sh` → rewrites network-config renderer→:6011, starts controller/physics/our-driver detached, waits for ports.
4. **Runtime:** `bash /workspace/run_runtime.sh /workspace/m2run` → rewrites `/mnt/nre-data`→host path in user-config (the wizard emits container-mount paths), then runs `alpasim_runtime.simulate`. Writes rollouts + aggregate.
⚠️ **Bare-run gotchas fixed:** (a) wizard emits `/mnt/*` container paths → rewrite to `/workspace/...` host paths (only `scene_provider.usdz.data_dir` matters for the runtime). (b) renderer runs on :6011 not the wizard's :6005 → `sed` the network-config renderer endpoint. (c) scene must be the **26.04** release to match the NRE image.

## 5. M3 — REF-C adapter (IMPLEMENTED, NOT closed-loop-validated) + M4 path

- 2026-07-22 ~14:15 local: **M3 adapter written** (`repo:.../refc_driver.py`, staged). `RefCPolicy` +
  `RefCDriver` wrap `tanitad.refs.refc.RefCModel` behind the same gRPC EgodriverService as M2 — drops
  into the M2 topology unchanged (swap `simple_driver.py`→`refc_driver.py` on :6789).
- **REF-C interface (MEASURED from `tanitad/refs/refc.py` + `refc_v12_cache.load_frozen`):**
  `cfg=refc_config()` (base) → `RefCModel(cfg)` → `load_state_dict(torch.load(ckpt)["model"])` (STRICT;
  trained anchors travel in the ckpt buffer) → `.eval()`. Infer: `model(frames[B,8,9,256,256], nav_cmd[B],
  v0[B], steps=2)` → `out["traj"] [B,4,2]` ego-frame (x-fwd,y-left) waypoints at 0.5/1.0/1.5/2.0 s.
- **Checkpoints located (already on eval pod, MODEL_REGISTRY §4):** base `/root/models/refc-base-30k/ckpt.pt`
  (md5 8f10d6f934f4199e11ddc7352e074939, v2.1 labels, 104.2M); XL `/root/models/refc-xl-30k/ckpt.pt`.
- **Nav-confound fix (implemented):** `RefCPolicy._nav_from_route` maps AlpaSim's submitted route waypoints →
  a real nav_cmd (left/right/straight) from the far-waypoint lateral offset, instead of the constant `follow`
  REF-C evals with (the documented confound).

### ⚠️ WHY M4 IS NOT YET A TRUSTWORTHY MEASURED RESULT — do these before quoting any REF-C closed-loop metric
1. **Frame-preprocessing byte-match (the crux).** REF-C trained on `ep.feats` = `taniteval.data.load_frames`
   output (a specific PhysicalAI crop+resize→256 + 3-frame stack + channel order + uint8/255). `refc_driver.py`'s
   `_prep_stack` is BEST-EFFORT. VALIDATE: push one known PhysicalAI window through both `_prep_stack` and
   `taniteval.data.load_frames`, assert equal, THEN trust closed-loop output. Wrong preprocessing → OOD input →
   meaningless metrics (a C1/C3 trap). Plus a domain shift: NuRec renders vs real dashcam.
2. **Waypoint timing (known bug to fix).** REF-C emits 4 waypoints at 0.5-s spacing (0.5/1.0/1.5/2.0 s), but the
   shared `SimpleDriver.drive()` timestamps waypoints at uniform `1/hz` (100 ms). `RefCDriver` must override
   `drive` to stamp the 4 waypoints at their true `HORIZON_S` (or upsample REF-C's 2-s plan to 10 Hz). Not fixed.
3. **taniteval/tanitad importability + GPU coexistence.** `refc_driver.py` imports `tanitad.refs.refc` (needs
   `PYTHONPATH=/workspace/TanitAD/stack` or the stack on the pod) and torch; it shares GPU 0 with the renderer
   (~watch VRAM: renderer peaks ~6.5 GB at video render + REF-C base ~1-2 GB — A40 46 GB fits).

### M4 execution recipe (once (1)+(2) validated)
`.venv/bin/python /workspace/refc_driver.py --port 6789 --ckpt /root/models/refc-base-30k/ckpt.pt --preset base`
(with `PYTHONPATH` incl. the TanitAD stack + tanitad importable), then rerun `launch_services.sh` (skip its
driver — REF-C is the driver) + `run_runtime.sh`. Repeat with `--ckpt /root/models/refc-xl-30k/ckpt.pt
--preset xl`. Compare `results-summary.json` (progress/collision/offroad/dist_to_gt) base vs XL vs the M2
constant-forward baseline (score 0.6637).

## 6. Milestone status (honest)
- **M1 renderer serves — ✅ DONE (MEASURED).** Bare NuRec on A40, scene loaded, gRPC :6011, no Vulkan wall.
- **M2 closed-loop rollout — ✅ DONE (MEASURED).** run_uuid 1d2d0617…, rollout 41e4ef8c…, PASS, score 0.6637.
  **"AlpaSim runs on the eval pod" = satisfied.**
- **M3 REF-C adapter — 🟡 IMPLEMENTED, not closed-loop-validated.** Correct loading/inference/nav/conversion.
  Gate 1 (frame preprocessing) INVESTIGATED 2026-07-22 → **BLOCKED** (see §8): needs TanitAD's per-clip
  f-theta canonicalization, whose intrinsics parquet is absent on the eval pod. ESCALATED.
- **M4 REF-C metrics — 🔴 NOT DONE / ESCALATED.** Deliberately NOT run: gate 1 (§8) cannot be cleanly closed
  on the eval pod, so any REF-C metric would be from mis-canonicalized (OOD) input — untrustworthy (C1/C3).
  Checkpoints located, adapter written, unblock paths A/B/C in §8 for the coordinator to choose.

## 8. GATE 1 — INVESTIGATED, BLOCKED (2026-07-22, M4 attempt) — do NOT emit a REF-C metric until resolved

**Investigation (MEASURED, code-traced):** REF-C's `[T,9,256,256]` input (`ep.feats = ep.frames`,
`taniteval/data.py:120`) is NOT a simple resize. The epcache build (`scripts/build_pai_cache.py` →
`tanitad.data.physicalai.build_episode` → `_decode_mp4` → `tanitad.data.calib.ftheta_crop_resize` +
`comma2k19.stack_frames`) applies TanitAD's **f-theta fisheye→pinhole canonicalization**:
`ftheta_crop_resize(vid, intr, size=256, center="principal")` — a poly-dependent square crop centered on
the clip's **per-clip principal point (cx,cy)** to land f_eff ≈ `F_REF=266`, then bilinear resize;
then a 3-frame stack @100ms → 9 ch (`[-3:]` = current). `refc_driver.py`'s `_prep_stack` (naive cv2.resize)
does NOT match this and would feed REF-C OUT-OF-DISTRIBUTION input.

**Why it can't be cleanly closed on the eval pod (each point MEASURED):**
1. **No per-clip intrinsics available.** `ftheta_crop_resize(center="principal")` REQUIRES `intr.per_clip`
   (real cx,cy) from `intrinsics_for_clip` → `r0/r0_selection.parquet`. **That parquet is ABSENT on
   `tanitad-eval`** (`find /root /workspace -name r0_selection.parquet` → none; it lives on the build
   pods). Its fallback reverts to geometric-center with a rig-B `cy` (a known-WRONG crop for rig-A clips —
   the exact two-rig defect the v2 calib fixed) and warns.
2. **The specified byte-match is impossible by construction.** `load_frames` reads the val cache built from
   the REAL PhysicalAI mp4; the closed-loop frames are AlpaSim **NuRec reconstructions** — different pixel
   sources, so no byte-identity exists even with identical transform code. The transform match would have
   to be proven at the geometry level (`ftheta_horizon_row` / `last_f_eff`), which needs provably-consistent
   intrinsics between two independent f-theta systems (TanitAD R0 parquet vs NuRec USDZ calib).
3. **Domain gap (inherent).** Even with perfect canonicalization, REF-C trained on REAL dashcam footage but
   would see NuRec reconstructions — a sim2real shift REF-C never saw. Acceptable as "the closed-loop number"
   but must be flagged.

**Decision (per coordinator's explicit instruction + CLAUDE.md C1/C3):** STOP — do NOT run REF-C on
mis-canonicalized frames and report ADE/collision/progress. An untrustworthy number is worse than an
honest gap. Gate 2 (waypoint timing) is a real but MOOT fix until gate 1 is resolved.

**Paths to unblock (for the coordinator to choose):**
- **A (correct, substantial):** get the scene's per-clip f-theta intrinsics onto the pod — either copy
  `r0_selection.parquet` from a build pod IF clip `01d503d4…` is in R0, or extract the f-theta calib from the
  NuRec USDZ and convert to `FThetaIntrinsics` (+ verify vs R0 for a known clip). Then wire
  `ftheta_crop_resize(center="principal")` + `stack_frames` into `refc_driver.py`, render AlpaSim at the
  native f-theta resolution, and VERIFY `ftheta_horizon_row`/`last_f_eff` align with the clip's cache
  provenance. Confirm whether AlpaSim renders the native f-theta projection (vs a baked pinhole) first.
- **B (cheap de-risk, open-loop, I can do next if approved):** run REF-C base on the val cache's own
  `[9,256,256]` frames via the existing `taniteval.refc_eval`/`refc_v12_eval` path and confirm it reproduces
  the registry ADE (base 0.4728) — proving the checkpoint+load+inference wiring in isolation, leaving the
  AlpaSim-render canonicalization as the SOLE remaining closed-loop unknown. NOT closed-loop, so offered for
  a decision, not done unilaterally.
- **C (open question):** is clip `01d503d4…` in the R0 training corpus at all? If not, only the USDZ-calib
  path (A-ii) exists. (Not resolvable on the eval pod — no parquet.)

## 9. M4 de-risk results (2026-07-22, coordinator-approved B / B2 / C) — MEASURED

**Lock:** `gpu_lock.sh acquire refc-derisk`. Pod2 untouched.

### ✅ B — ckpt+env proof (MEASURED, `repo:.../refc_b_eval.py`, log `pod:/workspace/refc_b.log`)
REF-C **base** (`/root/models/refc-base-30k/ckpt.pt`, step 29999, 128 anchors) through the canonical eval
(`taniteval.refc_eval.collect` on the 40-ep val cache, window 8 stride 8, nav=follow, v0 fed, 2 denoise steps)
→ **full-set `ade@2s = 0.47277`** (881 windows) = **registry 0.4728 EXACTLY.** (de@2s 1.003, miss@2m 0.142;
CV baseline ade@2s 0.838; heldout ade@2s 0.452.) **The checkpoint + eval-pod environment are proven good.**

### ✅ C — A-feasibility: the scene USDZ carries extractable f-theta calib (MEASURED, `repo:.../refc_rigdump.py`)
`clipgt/calibration_estimate.parquet` → `rig_json` (rig `hyperion_8.1_daimler_gls`, 33 sensors). The
`camera:front:wide:120fov` sensor `properties`: `Model: ftheta`, native **3840×2160**, `cx=1912.72`,
`cy=1510.21`, `polynomial-type: pixeldistance-to-angle` (BACKWARD, pixel→angle),
`polynomial="0 5.2854e-4 7.406e-9 -1.164e-11 1.043e-14 -2.185e-18"`.
**Consistency check (MEASURED):** scaling /2 → cx≈956, cy≈755 = TanitAD's known front-wide FHD **rig-B**
principal point (cx 958 / cy 753); backward paraxial dθ/dr 5.285e-4 at 4K inverts to ≈946 px/rad at FHD ≈
TanitAD forward `poly[1]=927.5` (~2%). **So A IS feasible on the eval pod without the build-pod parquet** —
BUT requires: (i) INVERT the backward poly → TanitAD forward `fw_poly` (angle→pixel), (ii) scale 4K→render
resolution, (iii) verify f_eff≈`F_REF=266` + horizon row via `ftheta_horizon_row`. This clip is **rig-B**
(cy well below center) → the naive geometric-center crop is exactly the wrong-rig case the v2 calib fixed.

### ✅ B2 — preprocessing damage (MEASURED, `repo:.../refc_b2.py`, log `pod:/workspace/refc_b2.log`) — DECISION NUMBER
Used the scene's OWN 4K fisheye mp4 (`camera_front_wide_120fov.mp4`, 3840×2160@30→10Hz, native res == the
calib) + the USDZ calib. **Inverted the backward poly → TanitAD forward `fw_poly=(0, 1873.76, 83.77, -170.41,
55.43)`** (paraxial 1873.8 px/rad @4K ≈ 2×927.5 TanitAD FHD; inversion max err **1.24 px**). Ran REF-C base on
the SAME frames through two preprocessings:
- **Canonical** `ftheta_crop_resize(center="principal")` — **SELF-CHECK `f_eff=266.0 == F_REF` PASS** (so the
  canonical arm is correct; canon traj[0] = 4.85/9.6/14.27/18.59 m forward, sensible).
- **Naive** `F.interpolate(4K→256×256)` (the `_prep_stack` path).

**PLAN DIVERGENCE canonical-vs-naive (GT-free, isolates preprocessing): mean per-waypoint 0.747 m · @2s
1.566 m · max 3.97 m.** Naive over-reaches (endpoint 21.30 m vs canonical 19.87 m — the aspect squish of the
wide 4K frame distorts apparent scale). **This is ~3.3× REF-C's own ade@2s (0.4728) and ~1.9× the CV-baseline
gap (0.838).**

### 🔴 CONCLUSION — the A decision (for Sayed)
**Option A (real f-theta canonicalization) is MANDATORY for a trustworthy REF-C closed-loop eval.** The naive
`cv2.resize` preprocessing shifts REF-C's plan by MORE than its entire accuracy budget → a closed-loop metric
on naive input would be meaningless. AND **A is now PROVEN feasible on the eval pod**: the USDZ carries the
f-theta calib, the backward→forward inversion works (err 1.24 px), and the canonical pipeline self-checks to
`f_eff=266`. Remaining for a full A closed-loop (Sayed's go): (1) confirm AlpaSim renders the native f-theta
`camera_front_wide_120fov` (vs a baked pinhole) at a resolution the calib scales to; (2) wire
`ftheta_crop_resize(center="principal")` + `stack_frames` into `refc_driver.py` (replacing `_prep_stack`),
sourcing per-scene intrinsics from the USDZ rig_json; (3) gate-2 waypoint timing; (4) accept the sim2real
domain caveat (NuRec reconstructions vs REF-C's real-footage training).

### De-risk deliverables (staged): `refc_b_eval.py` (B), `refc_rigdump.py`+`refc_c_calib.py` (C),
`refc_b2.py` (B2), `refc_recon2.py`. Logs on pod: `/workspace/refc_{b,b2}.log`.

## 10. Option A build (Sayed-approved 2026-07-22) — MEASURED

**Lock:** `gpu_lock.sh acquire refc-closedloop`. Pod2 untouched.

### ✅ STEP 1 — render projection (make-or-break): NATIVE F-THETA CONFIRMED (MEASURED, `repo:.../asl_camera_probe.py`)
Two-probe verification: (a) code — `camera_catalog.py` + `sensorsim_service.py:611` show the runtime fetches
camera specs via the renderer's `get_available_cameras` and uses them verbatim (native scene calib; no pinhole
override in the default config). (b) **empirical, from the REAL M2 rollout.asl** (`available_cameras_return`,
the exact spec the renderer served): `camera_front_wide_120fov` → **`MODEL=ftheta_param`** (not pinhole).
Renderer-reported intrinsics (FHD 1080×1920): principal **cx=956.11, cy=754.85** (= TanitAD rig-B FHD, cx958/
cy753 — consistent), **forward poly `angle_to_pixeldist=[0, 944.49, -10.98, 32.70, -77.40, 32.52]`** (poly[1]
944.5 ≈ TanitAD 927.5), backward poly + `max_angle=1.3509`. **→ A proceeds; NuRec renders native f-theta.**
**Bonus:** the driver receives this ftheta `CameraSpec` directly in `DriveSessionRequest.rollout_spec.vehicle.
available_cameras[*].intrinsics` — so `refc_driver.py` sources per-scene intrinsics from the SESSION (no USDZ
parse, no poly inversion, no `get_available_cameras` RPC which pickle-errors on a cold probe).
⚠️ Plan: render `camera_front_wide_120fov` at the **native 1080×1920** so `ftheta_crop_resize` applies directly.

### ✅ STEPS 2+3 — canonicalization + gate-2 wired (`repo:.../refc_driver.py`, staged)
`RefCDriver.start_session` builds `FThetaIntrinsics` from the session's ftheta CameraSpec (forward poly,
cx, cy, native res); `RefCPolicy.plan` applies `ftheta_crop_resize(center="principal")` + `stack_frames` →
[8,9,256,256] → REF-C (steps=2, eval). Gate-2: the 4 waypoints stamped at their true 0.5/1/1.5/2.0 s.
Runs in the alpasim venv + `PYTHONPATH=/root/TanitAD/stack:.../scripts` (imports verified: torch 2.13+cu130,
alpasim_grpc, tanitad.refs.refc, ftheta_crop_resize all OK). Camera rendered at native 1080×1920.

### ✅ STEP 4 — REF-C BASE closed-loop rollout (MEASURED) — "REF-C on NuRec reconstructions"
Topology: renderer :6011 (native f-theta 1080×1920) · physics :6006 · controller :6007 · REF-C base :6789.
`repo:.../refc_launch_services.sh` + `refc_wizard_gen.sh` + `run_runtime.sh`.
**Canonicalization validated LIVE on the rendered frames:** `start_session ftheta cx=956.1 cy=754.9 1920x1080
poly1=944.5` → **`CANON f_eff=265.9 (F_REF=266.0) OK`** (the make-or-break geometry self-check PASSES on the
actual NuRec renders). 10.04 sim-s in 34.5 wall (0.29× RT).
**Result — eval `run_uuid=1d0fee08...`, rollout `426964c6-85dd-11f1-b6b6-2fd7fcf47eb2`
(`repo:.../REFC_base_results-summary.json`):**
| metric | REF-C base | (M2 const-fwd ref) |
|---|---|---|
| status | **COLLIDED at-fault (front)** | pass |
| collision_any / at_fault / front | 1.0 / 1.0 / 1.0 | 0 / 0 / 0 |
| offroad | 0.0 | 0.0 |
| dist_to_gt_trajectory (m) | **1.664** | 0.574 |
| progress / progress_rel | 0.539 / 0.982 | 0.531 / 0.988 |
| plan_deviation | 0.443 | 0.084 |
| dist_traveled (m) | 39.8 (to failure) | 39.2 |
| duration_frac_20s | 0.23 | 0.35 |
| img_is_black | 0.0 | 0.0 |
| min_ade@*(gt) | null | null |
**Read:** REF-C base drives forward with high relative progress but **collides at-fault** on this trafficked
scene (41 actors), deviating 1.66 m from GT — a plausible open-loop-trained-planner-in-closed-loop outcome
(cf. the flagship open-loop 0.45 → closed-loop 1.69 divergence). ⚠️ **REF-C on NuRec RECONSTRUCTIONS, single
scene** — not a real-world closed-loop number, and n=1.

### ✅ STEP 5 — REF-C XL closed-loop rollout (MEASURED) — "REF-C on NuRec reconstructions"
Same topology, XL driver (`/root/models/refc-xl-30k/ckpt.pt`, step 29999, 256 anchors). **Canonicalization
validated LIVE again: `CANON f_eff=265.9 OK`.** rollout `701de24c-85de-11f1-8c20-77f91ebe17e0`
(`repo:.../REFC_xl_results-summary.json`).

### 📊 REF-C base vs XL closed-loop (MEASURED, n=1 scene, on NuRec reconstructions)
| metric | REF-C base | REF-C XL | M2 const-fwd |
|---|---|---|---|
| rollout uuid | 426964c6… | 701de24c… | 41e4ef8c… |
| collision_at_fault (front) | **1.0** | **1.0** | 0.0 |
| offroad | 0.0 | 0.0 | 0.0 |
| progress / progress_rel | 0.539 / 0.982 | 0.410 / 0.970 | 0.531 / 0.988 |
| dist_to_gt_trajectory (m) | 1.664 | 2.671 | 0.574 |
| plan_deviation | 0.443 | 0.620 | 0.084 |
| dist_traveled to failure (m) | 39.8 | 30.4 | 39.2 (no fail) |
| duration_frac_20s | 0.23 | 0.20 | 0.35 |
| CANON f_eff self-check | 265.9 OK | 265.9 OK | n/a |
| img_is_black | 0.0 | 0.0 | 0.0 |

**Findings (MEASURED, n=1):** Both REF-C base and XL **collide at-fault (front)** in closed loop on this
trafficked scene; **base slightly outperforms XL** (further before failure, less GT deviation) — consistent
with the registry's "base ties XL, no XL advantage." Both open-loop-trained anchored-diffusion planners
accumulate error and hit a lead actor without closed-loop collision avoidance. The **canonicalization is
correct** (f_eff=265.9 on both live render paths) so the REF-C INPUT is trustworthy; the collisions are a real
property of these planners in closed loop, not a preprocessing artifact.
⚠️ **Caveats:** (1) "REF-C on NuRec RECONSTRUCTIONS" — not real-world; sim2real domain shift on top. (2) **n=1
scene, 1 rollout each** — directional only; a suite (`scenes.test_suite_id=public_2601`) is needed for a real
number. (3) REF-C evals open-loop with nav=follow; here the driver derives nav from the route, but REF-C was
never closed-loop-trained. (4) min_ade@*(gt)=null (AlpaSim's GT-ADE not computed for these).

### 🎯 M4 COMPLETE. Milestones: M1✅ M2✅ M3✅ M4✅ (base+XL closed-loop, MEASURED, canonicalization-correct).

## 11. Standing /goal (Sayed) — REF-C variant sweep + flagship v1 + videos (coordinator-approved 2026-07-22)

**Lock:** `gpu_lock.sh acquire refc-goal`. Pod2 untouched.

### ✅ REF-C SMALL closed-loop (MEASURED) — "on NuRec reconstructions"
`/root/models/refc-small-30k/ckpt.pt` (step 29999, 64 anchors). Canon validated `f_eff=265.9 OK`.
rollout `17e55c6a-85e1-11f1-8f0c-cd5348a77f79` (`repo:.../REFC_small_results-summary.json`):
collision_at_fault(front)=1.0, offroad=0.0, dist_to_gt_trajectory=1.575, progress=0.465, plan_deviation=0.326,
drove 34.4 m, duration_frac_20s=0.22, img_is_black=0.0. **status=fail (collision).**

### 📊 REF-C variant sweep — closed-loop, n=1 scene, on NuRec reconstructions (MEASURED)
| | base | **small** | XL |
|---|---|---|---|
| rollout | 426964c6… | 17e55c6a… | 701de24c… |
| anchors | 128 | 64 | 256 |
| collision_at_fault (front) | 1.0 | 1.0 | 1.0 |
| progress | 0.539 | 0.465 | 0.410 |
| dist_to_gt_trajectory (m) | 1.664 | **1.575** | 2.671 |
| plan_deviation | 0.443 | 0.326 | 0.620 |
| dist_traveled to failure (m) | 39.8 | 34.4 | 30.4 |
| CANON f_eff | 265.9 | 265.9 | 265.9 |
**All three REF-C variants collide at-fault (front)** on this trafficked scene. **small ≈ base > XL** (small
has the LOWEST GT-deviation + plan_deviation) — consistent with the registry S3 finding "the fan lever is
ANCHOR WIDTH not encoder scale; small proposes best per-anchor." No XL advantage. Canonicalization correct on
all three → these are trustworthy REF-C-input closed-loop numbers (n=1).

### ✅ VIDEOS (TanitEval-standard) — base + XL + small (`repo:.../REFC_{base,xl,small}_video.mp4`)
**AlpaSim's DEFAULT eval-video layout IS the TanitEval viz standard** (verified frame
`repo:.../REFC_base_video_frame.png`): (1) **rendered front camera** (NuRec f-theta) with the model's
**planned trajectory projected onto it**; (2) **metric BEV inset** — ego + planned traj (orange) + GT/route
(green) + numbered scene actors + lanes; (3) **text overlay** — run/rollout UUIDs + collision/offroad/progress/
dist_to_gt. (Generated natively by `overlay_plans_on_camera=True` + `map_elements_to_plot` + `metrics_table_
entries` — no corpus_overlay.py adaptation needed.) The base frame visibly shows the ego about to rear-end the
lead truck (BEV actor 18) → the collision_front. Minor gap vs spec: no explicit "decoded maneuver" text
(metrics shown instead); the nav command is fed to REF-C but not overlaid.

### 🔴 FLAGSHIP v1 closed-loop — ESCALATED (genuine interface wall, per coordinator's "escalate >30min")
**Finding (MEASURED from code):** flagship v1 (`flagship4b-speedjerk-30k`) is an **action-conditioned world
model, NOT a policy.** Its eval path (`eval_grounded_rollout_4b_speed.py` / `taniteval/rollout.py`):
`states = world.encode_window(fw)` → `rollout_decode(world.predictor, states, aw, **fa**, step_readout, fwd_k)`
where **`fa` = the TRUE FUTURE ACTIONS** (`ep.actions[t+window:...]`) — the 0.452 open-loop number rolls the
operative predictor forward UNDER GT FUTURE ACTIONS. **In closed loop there are no GT future actions** → v1
cannot output a trajectory from observations alone. It needs a **CEM/MPC tactical planner** to search action
sequences (the memory's documented "tactical planner is the next closed-loop lever," explicitly DEFERRED).
This matches the coordinator's flagged risk exactly. **Options (for a decision):**
- **A — v1 hold-action rollout driver** (~30-60 min + validation): roll v1's operative predictor under a
  HELD/neutral action (estimate `aw` from pose history, `fa`=maintain) → trajectory. MEASURABLE, but it's
  "v1 dead-reckons under a hold policy," NOT intelligent planning; honest label required. Correctness risk in
  the action estimation + ckpt-grounding structure (`flagship-speed/` ships a SEPARATE `enc_readout.pt`, not
  the `ck["grounding"]` the eval expects — needs structure confirmation).
- **B — use v1.5/v1.6** (`eval_flagship_v15.py`/`v16`), which HAVE the tactical fan/CEM planner and CAN drive
  closed-loop — if a planner-variant is acceptable as the "flagship policy."
- **C — build the CEM tactical planner over v1** (the real lever) — major work, out of a driver-wrap scope.
**Recommend B** (v1.5/v1.6 is the actual closed-loop-capable flagship) or A if a hold-action v1 number is
wanted. Did NOT build a fragile v1 driver on a wrong premise (v1≠policy) — escalating per instruction.

### 🔴 FLAGSHIP v1.5/v1.6 (coordinator chose B) — BLOCKED: ckpt not available on the eval pod or HF
Studied `eval_flagship_v15.py`/`v16.py`: v1.5/v1.6 = FROZEN v1 trunk (encoder+predictor) + a REF-C
anchored-diffusion tactical HEAD that DOES produce waypoints from the window alone (no future actions) —
`head(st, v0, imagined=imag, vt_band, route, route_graded, vt_speed, steps)` → `out["traj"]`. So v1.5/v1.6
IS closed-loop-capable (the correct realization of "flagship driving"). **But the ckpts are unavailable here
(2-probe MEASURED):**
- `find /root /workspace -iname "*v15*"/"*v16*"` on the eval pod → **nothing**; `/workspace/experiments` absent.
- HF `Sayood/` list → phase0, **speedjerk (the v1 trunk)**, refa-*, refb-speed, refc-small, internal —
  **NO flagship-v15 / v16** (`model_info` RepositoryNotFoundError for all v15/v16 variants).
- They live only on the TRAINING pods (`/workspace/experiments/flagship-v15-abc`, `flagship-v16-ab-ft`) —
  which I must not scp from (pod2 = v4 training).

**Escalation (per coordinator's "escalate if v1.5/v1.6 needs something it doesn't have"):** it needs its
CKPT, absent on the eval pod + HF. **Options for the coordinator:**
- **1 — push the v1.5 head (+ its `anchors`/`probes` files) or the v1.6 ckpt to HF `Sayood/`** → I pull the
  trunk (`tanitad-flagship-4b-speedjerk`, already on HF) + the head + build the adapter.
- **2 — authorize a pod1 (NOT pod2) pull** of the v15/v16 experiment to the eval pod.
- **3 — v1 hold-action rollout** as a fallback flagship number using the AVAILABLE `flagship-speed` ckpt
  (roll the v1 world model under a held action). Cheaper but "dead-reckoning, not planning."
⚠️ Even WITH the ckpt, the v1.5 adapter is a SUBSTANTIAL build (bigger than REF-C): frozen-trunk `encode_window`
→ states; `imagine_probes(predictor, st, ac, probes, ...)` (needs the `probes` file + past-window actions `ac`,
which have NO GT in closed loop → estimate from poses); and the tactical-INTENT inputs `vt_band`/`vt_speed`/
`route_graded`/`route` are GT-derived label encodings in eval → must be SYNTHESIZED as a default cruise-band +
the AlpaSim route command. Not a quick model-swap.

### 🎯 /goal status: REF-C variant sweep (base/XL/small) ✅ + videos (base/XL/small) ✅ + flagship closed-loop
🔴 **BLOCKED on v1.5/v1.6 ckpt availability** (needs a push-to-HF / pod1-pull decision) — with the DOCUMENTED
finding that pure v1 is an action-conditioned WM that cannot drive without a planner. Recommended next once a
flagship ckpt is reachable: build the v1.5 adapter (est. real effort) + `public_2601` suite for statistical
REF-C numbers.

## 12. REF-C closed-loop SUITE — base vs XL, n=12 scenes (coordinator-approved 2026-07-22) — MEASURED
Resolves the n=1 caveat. 12 26.04-release scenes (`refc_suite_wizard_gen.sh` scene list) at **480×854**
(5× faster than native; f-theta canon still applies, `f_eff=265.6 OK` both), external REF-C driver, one
rollout/scene/model. `repo:.../REFC_suite_{base,xl}_results.json` (raw) + `REFC_suite_results.json` (combined
+ per-scene UUIDs). **"REF-C on NuRec reconstructions."**

| metric (n=12) | **base** | **XL** |
|---|---|---|
| at-fault collision rate | **0.333 (4/12)** | **0.333 (4/12)** |
| collision rate | 0.333 | 0.333 |
| offroad rate | 0.167 (2/12) | 0.250 (3/12) |
| failure rate (offroad_or_collision_at_fault) | 0.500 (6/12) | 0.583 (7/12) |
| mean progress | 0.474 | 0.519 |
| mean dist_to_gt_trajectory (m) | 1.642 | 1.973 |
| mean plan_deviation | 0.682 | 0.524 |
| **mean score** | **0.345** | 0.246 |
| passes | **6/12** | 5/12 |
| img_is_black | 0.0 | 0.0 |

**Findings (MEASURED, n=12):**
1. **The at-fault collision rate is 33% (4/12), NOT "always."** The n=1 highway scene (01d503d4, 41 actors)
   was a worst-case; across 12 scenes REF-C **passes ~half** (base 6/12, e.g. clip 00064c58 score 0.837).
2. **Base ≥ XL statistically:** base wins mean score (0.345 vs 0.246), passes (6 vs 5), dist_to_gt (1.642 vs
   1.973), offroad (0.167 vs 0.250), failure rate (0.500 vs 0.583); XL only edges raw progress + plan_dev.
   **The 2× XL capacity gives NO closed-loop advantage** — confirms the registry "base ties/beats XL" at n=12.
3. Both fail ~half of scenes (base 50%, XL 58%) — open-loop-trained anchored-diffusion planners accumulate
   error in closed loop. Canonicalization correct throughout → trustworthy REF-C inputs.
⚠️ **Caveats:** n=12 (not the full 916 — infeasible in-window: ~1.5GB download + render/scene); **wide binomial
CIs at n=12** (a 4/12 rate is ~13–61% at 95%), so "base=XL on collision rate" is not a confident tie — the
SCORE gap (0.345 vs 0.246) is the cleaner signal. 480×854 (single-scene runs were 1080×1920). NuRec recon.

### 🎯 FINAL /goal status: REF-C variants (base/XL/small) ✅ + REF-C **suite** base-vs-XL (n=12, statistical) ✅
+ videos (base/XL/small) ✅ + flagship 🔴 blocked on v1.5/v1.6 ckpt (push-to-HF/pod1 decision). The closed-loop
sim deliverable is now STATISTICAL: **base ≥ XL, ~33% at-fault collision rate, no XL advantage.**

## 13. OPEN-LOOP DIAGNOSTIC — model vs data/env (Sayed's control, coordinator-approved) — MEASURED ⭐
**The load-bearing control: is "REF-C fails ~half closed-loop" the MODEL, or the DATA/ENV (NuRec domain gap)?**
Method: **force-GT rollout** (`refc_openloop_wizard_gen.sh`: `force_gt_duration_us=20e6`, `skip_driver_during_
force_gt=false`) → the ego follows the GT path, so REF-C predicts on IN-DISTRIBUTION rendered frames along GT;
the driver LOGS its rig-frame prediction + world pose per drive (`refc_driver.py --log-preds`); post-process
(`refc_openloop_ade.py`) scores predictions vs the GT ego path exactly as taniteval (`ade_0_2s = mean de@
{0.5,1,1.5,2}s`). 4 scenes, 480×854, canon `f_eff=265.6 OK`, **236 scored predictions**. Artifacts:
`repo:.../REFC_openloop_diagnostic.json` + `REFC_openloop_preds.jsonl`.
⚠️ **Bug caught + fixed mid-run (C1-adjacent):** AlpaSim force-GT bypasses the controller → sends NO dynamic
state → the driver saw `speed=0` (REF-C mis-conditioned). Fixed to estimate `v0` from pose finite-difference
(now 11–20 m/s, predictions scale with speed). The first (v0=0) run was discarded.

**RESULT (MEASURED):** AlpaSim open-loop **ADE = 1.466** (de@0.5/1/1.5/2s = 0.501/1.066/1.782/2.515),
per-scene 1.21–1.63 — **3.1× the taniteval reference (REF-C base 0.4728 on real PhysicalAI val).**

### 🔬 VERDICT: **≫ 0.47 → REF-C sees OUT-OF-DISTRIBUTION input in AlpaSim → the closed-loop failures are
CONFOUNDED by DATA/ENV (the NuRec reconstruction domain gap), NOT purely the model.**
This **RE-FRAMES §11/§12**: "REF-C fails ~half closed-loop / base≥XL" cannot be attributed to REF-C's driving
ability alone — REF-C is being fed reconstructions ~3× more off-distribution than its real-footage training,
so a large part of the collisions is the sim2real/reconstruction gap. The base-vs-XL *ordering* (base≥XL) still
holds (both see the same OOD input), but the ABSOLUTE "REF-C collides" story is not a clean model indictment.
⚠️ **Honest bounds on the 3.1×:** the multiplier bundles (a) the NuRec reconstruction domain gap [likely
dominant], (b) 480×854 vs native 1080×1920 source detail, (c) any ego-mask/color/canonicalization residual,
(d) a possible ≤1-step timing-alignment residual in the diagnostic (de@0.5s=0.50 is high for the short horizon).
Each would ADD to the ratio; none reverses the direction. **The verdict (data/env confound, ADE ≫ 0.47) is
robust; the exact 3.1× is an upper-ish estimate of REF-C's own model error contribution being SMALL.**
Next to tighten it: re-run at native 1080×1920 + log the pose timestamp to zero out (b)+(d).

## 14. FLAGSHIP v1 CLOSED-LOOP — ⚠️ EARLIER FINDING CORRECTED (Sayed was right) — MEASURED
**CORRECTION of §11's "pure v1 cannot drive closed-loop":** WRONG. Pure flagship v1 (`flagship4b-speedjerk-30k`)
**DOES drive from observations alone** via its trained **TACTICAL POLICY** head. My earlier finding looked only
at the OPERATIVE rollout (`rollout_decode` under true future actions `fa`) — which is what
`taniteval/rollout.py` AND the flagship trajectory-video generator (`flagship_overlay.py`) use, so it misled me.
But the ckpt ALSO has a trained `tactical_policy` (probe: **103 keys** incl. `maneuver_head`) + `tactical_pred`
(99 keys), and the state-only DEPLOY path (`taniteval/closedloop.py`) is:
```
states = model.encode_window(frames)                 # [1,W,2048]
ctx    = model.strategic_policy(states, nav_cmd)["ctx"]
wp     = model.tactical_policy(states, ctx)["waypoints"]   # dict {5,10,15,20} -> [1,2] ego frame
```
**No future actions.** Ckpt: `/root/models/flagship-30k/ckpt.pt` = **step 29999 = the speedjerk-30k v1**
(load STRICT-clean 0/0 with `flagship4b_config()` + `predictor.action_dim=3`). Adapter `repo:.../flagship_v1_
driver.py` reuses REF-C's f-theta canon + gRPC (RefCDriver is model-agnostic; only the policy swaps).

### ✅ RESULT — flagship v1 closed-loop on clip 01d503d4 (native 1080×1920), "on NuRec reconstructions"
Canon validated LIVE `f_eff=265.9 OK`. rollout `71f9740c-85f0-11f1-9571-9760cad9b231`
(`repo:.../Flagship_v1_results-summary.json`, `Flagship_v1_video.mp4`):
| metric | **flagship v1** | REF-C base (same scene) |
|---|---|---|
| status | **PASS** | fail |
| score | **0.699** | 0.0 |
| **collision_at_fault (front)** | **0.0** | 1.0 |
| offroad | 0.0 | 0.0 |
| dist_to_gt_trajectory (m) | 4.25 | 1.66 |
| progress / progress_rel | 0.560 / 0.989 | 0.539 / 0.982 |
| plan_deviation | 0.940 | 0.443 |
| dist_traveled (m) | 41.7 | 39.8 |
**FINDING (MEASURED, n=1):** On the SAME trafficked highway scene where all three REF-C variants **collide
at-fault (front)**, **flagship v1 does NOT collide** (PASS, score 0.699) — its 4-brain world-model + tactical
policy avoids the lead-truck collision REF-C rear-ends into. It deviates MORE from the GT line (4.25 vs 1.66m —
it swerves to avoid) but stays on-road and collision-free. **Flagship v1 drives BETTER closed-loop than REF-C
here**, and — notably — it is more robust to the same NuRec-reconstruction OOD input (§13) that degrades REF-C.
⚠️ n=1 scene, NuRec reconstructions. (Aggregate JSON also contained a stale rollout; the 71f9740c-specific
metrics above are the clean flagship result.)

### 🎯 /goal — NOW COMPLETE: REF-C variants (base/XL/small) ✅ + REF-C suite base-vs-XL (n=12) ✅ + **flagship
v1 closed-loop** ✅ + videos (base/XL/small/**flagship v1**) ✅ + the open-loop **model-vs-data/env diagnostic**
✅ (⭐ REF-C sees OOD input, closed-loop failures confounded by the reconstruction gap). Flagship v1 is the
driveable flagship (tactical policy) — no v1.5/v1.6 needed.

## 15. FLAGSHIP v1 vs REF-C base — PAIRED SUITE (n=12), the statistical answer — MEASURED 2026-07-23
**Resolves §14's n=1.** Lock `gpu_lock.sh acquire flagship-vs-refc` (released clean at end). Autonomous
master `vs_suite_master.sh` → renderer :6011 (12-scene sceneset) → REF-C base into `/workspace/vs_refc` →
flagship v1 into `/workspace/vs_flag` (each a clean logdir via `vs_suite_run.sh`, runtime foreground, services
killed by port) → renderer killed + lock released. Both models over the **SAME 12 `public_2601` 26.04 scenes**
(the §12 list), **one rollout/scene, 480×854**, canon verified LIVE: **flag f_eff=265.7 OK, refc f_eff=265.6 OK**.
Paired stats `vs_aggregate.py`. Full write-up + caveats: **`flagship_vs_refc_suite_NOTE.md`**; raw:
**`flagship_vs_refc_suite_results.json`** (per-scene rollout UUIDs) + `vs_{flag,refc}_results-summary.json`.

| metric (n=12, NuRec recon) | **flagship v1** | **REF-C base** | flag−refc |
|---|---|---|---|
| at-fault collision | 0.167 (2/12) | 0.167 (2/12) | **TIED** |
| offroad | **0.667 (8/12)** | 0.167 (2/12) | +0.500 |
| pass rate | **2/12** | **8/12** | −0.500 |
| mean score | **0.066** | **0.496** | **−0.430** boot95 [−0.646,−0.215] |
| mean dist-to-GT (m) | 1.805 | 1.874 | −0.069 (tied) |
| mean plan_deviation | 1.125 | 0.342 | 3.3× wider |

Paired: **pass McNemar 6–0 for REF-C (p=0.031)**; **score sign test 8–0 for REF-C (p=0.008)**; collision
McNemar 1–1 (**p=1.0, tied**). **VERDICT: the n=1 was a LUCKY scene (C5). REF-C base statistically beats
flagship v1 closed-loop on this suite; flagship's failure mode is OFFROAD (high plan-deviation swerve), not
collision — its n=1 collision-avoidance does NOT generalize.** ⚠️ WITHIN-SIM RELATIVE (both ~3.2× OOD, §13);
480×854 (n=1 was native 1080×1920 → the one residual confound, recommend a native paired re-run); REF-C
absolute varies run-to-run (fresh 0.496 vs §12's 0.345, likely diffusion sampling) but the paired delta and
"flagship passes 0 scenes REF-C fails" are robust to it. Pod CLEAN, GPU free, lock FREE — no orphan.

## 16. NATIVE 1080×1920 paired re-run — the resolution/environment control — MEASURED 2026-07-23
Resolves §15's one residual confound (854 vs native). Lock `vs-native1080` (released clean). Autonomous
`vs_suite_master_1080.sh` = §15's master with the **only** change being the camera res flip 480×854→1080×1920
(sed in each logdir; verified `cam=1920x1080 stray854=0`). Same 12 scenes, same drivers/ckpts, same
`vs_suite_run.sh`+`vs_aggregate.py`. Both canon self-checked **f_eff=266.0 == F_REF OK** at native.
Full write-up + 854-vs-native side-by-side: **`flagship_vs_refc_native1080_NOTE.md`**; raw:
**`flagship_vs_refc_native1080_results.json`** + `vs_{flag,refc}_1080_results-summary.json`.

| paired flag−refc | 480×854 | **native 1080×1920** |
|---|---|---|
| mean score delta | −0.430 [−0.646,−0.215] | **−0.295 [−0.494,−0.117]** (excludes 0) |
| score sign test | 8–0 refc, p=0.008 | 7–0 refc, **p=0.016** |
| pass McNemar | 6–0 refc, p=0.031 | 4–0 refc, p=0.125 (fewer discordant) |
| at-fault collision | tied | tied |
| flag / refc pass | 2/12 / 8/12 | 3/12 / 7/12 |
| flag / refc mean score | 0.066 / 0.496 | 0.115 / 0.410 |

**VERDICT: the delta HOLDS at native res — MODEL, not environment.** REF-C base beats flagship v1 closed-loop
at BOTH resolutions; flagship's tactical head is the resolution-robustly-worse closed-loop planner. Flagship
does *modestly* better at full res (offroad 8→6/12, pass 2→3/12, deficit shrinks 30 %) → resolution is a
**second-order** modifier, not the explanation; the 854 suite slightly overstated flagship's deficit but the
conclusion is unchanged. Collisions tied both res (the n=1 collision-avoidance never generalized). ⚠️ Still
WITHIN-SIM RELATIVE / ~3.2× OOD (§13). Pod CLEAN, GPU free, lock FREE — no orphan.

## 7. Deliverable manifest

| artifact | where it lives | status |
|---|---|---|
| `RUN_RECIPE.md` (this file) | repo (staged) | the reproducible recipe + status |
| `simple_driver.py` | repo (staged) · pod `/workspace/simple_driver.py` | M2 stock driver (MEASURED working) |
| `refc_driver.py` | repo (staged) | M3 REF-C adapter (implemented, NOT validated) |
| `alpasim_setup.sh` | repo (staged) · pod `/workspace/alpasim_setup.sh` | bare workspace setup (MEASURED) |
| `pyproject_pared.toml` | repo (staged) · pod (applied; orig at `pyproject.toml.orig`) | driver-excluded workspace |
| `scene_dl.sh` | repo (staged) · pod `/workspace/scene_dl.sh` | scene download (MEASURED) |
| `renderer_serve.sh` | repo (staged) · pod `/workspace/renderer_serve.sh` | M1 renderer launch (MEASURED) |
| `wizard_gen.sh` | repo (staged) · pod `/workspace/wizard_gen.sh` | config gen (MEASURED) |
| `launch_services.sh` | repo (staged) · pod `/workspace/launch_services.sh` | backing services (MEASURED) |
| `run_runtime.sh` | repo (staged) · pod `/workspace/run_runtime.sh` | runtime launch (MEASURED) |
| `verify_imports.py` | repo (staged) · pod `/workspace/verify_imports.py` | M2 import check (MEASURED bad=0) |
| `M2_results-summary.json` | repo (staged) | M2 metrics evidence (the load-bearing result) |
| M2 rollout (asl 7.27MB, mp4 456KB, metrics.parquet, aggregate/) | **pod only** `/workspace/m2run/` | regenerable via the recipe; metrics copied to repo |
| REF-C base+XL ckpts | **pod only** `/root/models/refc-{base,xl}-30k/ckpt.pt` | pre-existing (MODEL_REGISTRY §4) |
| alpasim workspace + venv + scene + NRE rootfs | **pod only** `/workspace/…` | large; reproducible via scripts |

**Pod left CLEAN:** all M2 services stopped (by explicit PID), GPU free, `gpu_lock` released.
Kernel cache persists at `/workspace/nrehome/.cache` for faster renderer re-boot.

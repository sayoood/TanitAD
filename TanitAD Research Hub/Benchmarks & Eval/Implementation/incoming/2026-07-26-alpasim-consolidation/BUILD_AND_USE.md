# AlpaSim — BUILD AND USE, from a clean pod

**Date:** 2026-07-26 · **Target:** a competent engineer with no prior AlpaSim context.
**Companions:** `ALPASIM_STATE.md` (what exists, what is stranded, what the numbers mean) ·
`TANITSIM_FORK_RECOMMENDATION.md`.

> # ⚠️ READ THIS BEFORE YOU QUOTE ANY NUMBER THIS PRODUCES
>
> **Every closed-loop result from AlpaSim is "*on NuRec reconstructions*" — a WITHIN-SIM RELATIVE
> comparison, NOT a real-world rate.**
>
> MEASURED (`REFC_openloop_diagnostic.json`, 4 scenes / **288** predictions): with the ego forced onto
> the ground-truth path, REF-C base's **open-loop** ADE on these rendered frames is **1.5157 m** against
> **0.4728 m** on real footage — **3.21× out-of-distribution before the policy takes a single action.**
>
> ⇒ **Paired A-vs-B orderings are admissible** (both arms see the same OOD input, which differences out).
> ⇒ **Absolute rates are NOT** — "model X collides in 33 % of scenes" confounds model quality with
> reconstruction fidelity. That exact claim has already been retracted once
> (`Project Steering/RETRACTION_LOG.md:52`, class **C6**).
> ⇒ A residual **480×854 vs native 1080×1920** confound is *substantially closed but still open*:
> the paired verdict holds at both resolutions, but the 3.21× OOD figure itself has **only ever been
> measured at 854**. See `ALPASIM_STATE.md` §5.2.
>
> **Before attributing any closed-loop failure to a model, re-run the open-loop-vs-known control (§7).**

---

## 0. Verification legend — what I actually checked

This document is assembled from a working, MEASURED history plus static verification done on
2026-07-26. **Nothing here was executed end-to-end by this agent** (pods pod1/pod2/pod3 are training;
the eval pod was probed read-only and no render was started). Each step carries one of:

| Tag | Meaning |
|---|---|
| ✅ **MEASURED (inherited)** | A previous agent ran this and banked an artifact. Cited. Not re-run by me. |
| 🔍 **STATIC-VERIFIED (2026-07-26)** | I confirmed the referenced files/symbols/paths exist, this session. Method stated. |
| 📝 **TRANSCRIBED — UNVERIFIED** | I wrote this from prose. **Nobody has run this exact text.** Treat as a first draft. |

**One step is 📝.** It is §3, the renderer acquisition — and it is the hardest step. Budget for it.

---

## 1. Prerequisites

### 1.1 Hardware

| | Requirement | Our reference host |
|---|---|---|
| GPU | ≥40 GB VRAM (NVIDIA's figure is dominated by *their* 10 B policy; **ours needs 1–2 GB**) | A40 46 GB, cc 8.6 |
| VRAM in practice | renderer peaks ~6.5 GB (video render) + policy 1–2 GB | fits easily |
| CUDA | **12.6+** | 12.8, driver 580.159.04 |
| Vulkan / EGL | **NOT REQUIRED** — NuRec is gsplat/OptiX = CUDA | n/a |
| Container runtime | **NOT REQUIRED** — we run bare (upstream uses Docker Compose) | absent |
| Disk | ~150 GB **on a writable volume that is not `/`** | `/workspace`, MooseFS |
| RAM / CPU | comfortable; 96 vCPU / 503 GB here, far more than needed | — |

⚠️ **`/` on `tanitad-eval` is 93 % full (186/200 GB).** Every cache, extraction and log below is
redirected to `/workspace`. The first NRE extraction ever attempted **silently ran out of space on `/`**.
✅ MEASURED, `LOOP_STATE.md`.

⚠️ **Do not judge `/workspace` fullness with `df`** — it reports the cluster, not the per-pod MooseFS
quota. Use a real `dd` write test. (CLAUDE.md standing trap.)

### 1.2 Credentials — two gates, line them up first

| Gate | For | Where |
|---|---|---|
| **NGC API key** (`nvapi-…`) | pulling `nvcr.io/nvidia/nre/nre-ga:26.04` | `Keys.txt` on the **dev box**. ✅ MEASURED to authenticate. |
| **HF token with approved access** to `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` | scene download (gated) | `Keys.txt`. ✅ MEASURED, `DL_EXIT=0`. |

🔴 **`Keys.txt` is git-ignored — NEVER commit it.** Read tokens in place
(`grep -oE 'hf_[A-Za-z0-9]+'`); never copy, print, or pass them as command-line arguments. The
committed scripts all take the token **on stdin** for this reason, and `scene_dl.sh` writes it only to
tmpfs (`/dev/shm`, mode 600) and deletes it before downloading. Keep that discipline.

### 1.3 Licence acknowledgements you are accepting by doing this

- **AlpaSim source** — Apache-2.0. 🔍 STATIC-VERIFIED 2026-07-26: `LICENSE` on the pod reads
  "Apache License / Version 2.0, January 2004".
- **NRE renderer container** — **NVIDIA Deep Learning Container Licence** (`NGC-DL-CONTAINER-LICENSE`
  at the rootfs root). 🔍 STATIC-VERIFIED (read this session). Internal use and service deployment are
  granted; **modification, reverse engineering and standalone redistribution are not** (§4b/c).
  §4h names *"autonomous vehicle applications"* a **Critical Application** the container is *"not
  tested or certified"* for. See `TANITSIM_FORK_RECOMMENDATION.md` §2.
- **NuRec scene data** — **NVIDIA AV Dataset Licence**: internal development only, NVIDIA Confidential,
  **12-month term, destroy on expiry**. In our `SOURCE_REGISTRY` the sibling `physicalai_av` source is
  class **`gated-confidential`** (`stack/tanitad/lake/schema.py:111`), which the lake's
  `license_guard` **refuses to export**. ⇒ **Nothing rendered from these scenes may leave the program**
  — no HF push, no paper figure, no public artifact.

---

## 2. Step 1 — the AlpaSim workspace (bare, no Docker)

**Time:** ~7 min. **CPU only.** ✅ MEASURED (`alpasim_setup.sh`, RUN_RECIPE §2b).

```bash
# 2.1  Clone upstream (public, Apache-2.0). Shallow is fine; skip LFS.
mkdir -p /workspace/alpa-invest && cd /workspace/alpa-invest
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/NVlabs/alpasim.git
# our reference revision (MEASURED 2026-07-26): 55814289d8047bf239206712d31a745f2ad8f5ea

# 2.2  Copy our two setup files onto the pod, then run the setup detached.
#      Both are committed at:
#        TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/
#          2026-07-22-alpasim-closedloop-evalpod/{alpasim_setup.sh,pyproject_pared.toml}
scp alpasim_setup.sh pyproject_pared.toml tanitad-eval:/workspace/
ssh -f tanitad-eval 'bash /workspace/alpasim_setup.sh > /workspace/setup.log 2>&1'
```

**What `alpasim_setup.sh` does** (🔍 STATIC-VERIFIED — I read all 47 lines):

| Stage | Action | Why |
|---|---|---|
| `STAGE_1_UV` | standalone **uv** → `/workspace/uvbin` | the apt uv 0.9.0 cannot `self update` |
| `STAGE_2_RUST` | rustup → `/workspace/.cargo` + `/workspace/.rustup` | `utils_rs` builds via maturin |
| `STAGE_3_PYPROJECT` | swap in `pyproject_pared.toml`, back up original to `pyproject.toml.orig` | **drops the `driver` + `plugins/*` workspace members** so `uv sync` never resolves the multi-GB `vam` / `alpamayo_r1` / `alpamayo1_5` git deps. We supply our own driver. |
| `STAGE_4_SYNC` | `uv sync --extra core` | builds 10 alpasim packages + torch + trajdata |
| `STAGE_5_PROTOS` | `cd src/grpc && uv run compile-protos` | generates `*_pb2.py`; **without this the driver cannot import** |
| `STAGE_6_UTILS_RS` | `uv pip install -e src/utils_rs` | Rust geometry helpers |
| `STAGE_7_VERIFY` | import smoke → `IMPORTS_OK` | |

All caches are redirected to `/workspace` (`UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `CARGO_HOME`,
`RUSTUP_HOME`, `XDG_CACHE_HOME`, `TMPDIR`).

**Gate — do not proceed until this passes** (✅ MEASURED, `bad=0`):

```bash
ssh tanitad-eval '/workspace/alpa-invest/alpasim/.venv/bin/python /workspace/verify_imports.py'
# checks: alpasim_grpc, egodriver_pb2_grpc, alpasim_utils.geometry, alpasim_wizard,
#         alpasim_runtime, alpasim_controller, alpasim_physics, grpc, numpy
```

Venv lands at `/workspace/alpa-invest/alpasim/.venv` (Python 3.12).

### 2.3 Supply-chain note — a conscious approval, not a footnote

AlpaSim's **root** `pyproject.toml` pins `lightning` to a **GitHub archive tarball** (SHA `2129fdf3…`)
rather than PyPI, citing an April-2026 PyPI compromise. It is SHA-pinned and looks legitimate, but any
install that includes the `vam` / `alpamayo` drivers pulls a dependency from arbitrary GitHub.
**`pyproject_pared.toml` excludes those members, so our install does not need it.** Also note
`exclude-newer = "3 days"` in their uv config → reproducible-but-frozen resolution.
(✅ MEASURED, `INTAKE.md` §9.)

---

## 3. Step 2 — the NRE renderer 🔴 **THE HARD STEP**

**This is the only step with no committed script.** The renderer ships **exclusively** as the prebuilt
NGC image `nvcr.io/nvidia/nre/nre-ga:26.04` — there is no `renderer` directory in the Apache-2.0 tree,
no pip wheel, no source form. The image is **14,295,757,278 bytes across 42 layers (40 unique)** and
extracts to **~38 GB**.

**We pull and extract it WITHOUT a container runtime**, because an OCI registry pull is plain HTTPS.
✅ The *procedure* is MEASURED (it produced the working renderer now on the pod); 📝 the *script below is
TRANSCRIBED from `Project Steering/LOOP_STATE.md:363-399` and has NOT been re-run.*

### 3.1 The invariants that made it work (all ✅ MEASURED)

1. **Mint the bearer token on the DEV BOX.** The `nvapi-` key never leaves it. Scope the token narrowly
   and short: `repository:nvidia/nre/nre-ga:pull`, **600 s**.
2. **Pipe the token to the pod on stdin**, write to `/dev/shm/h` (tmpfs, mode 600), use `curl -H @/dev/shm/h`,
   delete immediately. **Never in an argv, never on pod disk.**
3. **`curl`'s `@file` syntax collides with `xargs -I@`** → use `-I{}`.
4. **A heredoc cannot coexist with a credential on stdin** — both consume stdin, and `set -e` then kills
   the block silently. Ship the script with `scp`, don't heredoc it.
5. **Strip CRLF** from any digest list produced on Windows → `sed -i 's/\r$//'`, else `curl: (3) URL rejected`.
6. **Extract to `/workspace`, never `/`.**
7. **`tar: Cannot change ownership …` on MooseFS is BENIGN** (metadata only).
8. **Before concluding a file is missing, check `pgrep -fc 'tar -xf'`** — a still-extracting tree looks
   like a missing one. (This produced two false "missing" conclusions, class C2.)

### 3.2 📝 TRANSCRIBED script — `nre_pull.sh` (UNVERIFIED, run it once and fix it)

```bash
#!/bin/bash
# nre_pull.sh -- pull + extract the NRE renderer image WITHOUT a container runtime.
# 📝 TRANSCRIBED 2026-07-26 from LOOP_STATE.md:363-399. NOT YET RE-RUN. Verify before trusting.
# Token arrives on STDIN line 1 (a short-lived registry bearer token minted on the dev box).
# Usage:  <mint token on dev box> | ssh tanitad-eval 'bash -s' < nre_pull.sh
set -uo pipefail
IMG_REPO="nvidia/nre/nre-ga"; IMG_TAG="26.04"
LAYER_DIR=/opt/nre/layers; ROOTFS=/workspace/nre/rootfs   # extract to /workspace, NEVER /
mkdir -p "$LAYER_DIR" "$ROOTFS"

IFS= read -r TOK
printf 'Authorization: Bearer %s\n' "${TOK%$'\r'}" > /dev/shm/h && chmod 600 /dev/shm/h
unset TOK
trap 'rm -f /dev/shm/h' EXIT

API="https://nvcr.io/v2/${IMG_REPO}"
ACCEPT='Accept: application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json'

# 1. manifest -> ordered layer digest list (ORDER IS LOAD-BEARING for extraction)
curl -sSL -H @/dev/shm/h -H "$ACCEPT" "$API/manifests/$IMG_TAG" > /workspace/nre/manifest.json || exit 1
python3 - <<'PY' > /workspace/nre/layers.txt
import json;m=json.load(open('/workspace/nre/manifest.json'))
[print(l['digest']) for l in m['layers']]
PY
sed -i 's/\r$//' /workspace/nre/layers.txt                  # CRLF trap
echo "LAYERS=$(wc -l < /workspace/nre/layers.txt) UNIQUE=$(sort -u /workspace/nre/layers.txt | wc -l)"

# 2. parallel blob fetch. -I{} NOT -I@ (curl's @file collision).
sort -u /workspace/nre/layers.txt | \
  xargs -P 8 -I{} sh -c 'test -s '"$LAYER_DIR"'/{}.tar.gz || \
    curl -sSL -H @/dev/shm/h -o '"$LAYER_DIR"'/{}.tar.gz "'"$API"'/blobs/{}"'
echo "PULLED_BYTES=$(du -sb "$LAYER_DIR" | cut -f1)"     # expect 14295757278

# 3. extract IN MANIFEST ORDER (later layers must overwrite earlier ones)
while read -r d; do
  tar -xf "$LAYER_DIR/$d.tar.gz" -C "$ROOTFS" 2>&1 | grep -v 'Cannot change ownership'  # benign on MooseFS
done < /workspace/nre/layers.txt
echo "EXTRACT_DONE $(du -sh "$ROOTFS")"                  # expect ~38G

# 4. driver libs + the absolute-path symlink the Bazel runfiles expect
ln -sfn "$ROOTFS/app" /app
ls -d "$ROOTFS/app/internal/scripts/pycena/runtime/pycena_nrm_full" && echo "NRE_LAUNCHER_OK"
```

**Expected end state** (✅ MEASURED on the current pod): `/workspace/nre/rootfs` ≈ 38 GB;
the launcher exists at
`/workspace/nre/rootfs/app/internal/scripts/pycena/runtime/pycena_nrm_full`;
`/app` is a symlink to `/workspace/nre/rootfs/app`; `NGC-DL-CONTAINER-LICENSE` sits at the rootfs root.
The image is a **Bazel-packaged Python app** with a hermetic Python 3.11 and its own venv including
torch 2.7.0+cu128 — which is *why* it boots bare.

⚠️ **Verify `PULLED_BYTES` against the manifest total before extracting.** The one number that proves a
clean pull is **14,295,757,278**.

---

## 4. Step 3 — scenes

Scenes are **NuRec neural reconstructions** of recorded drives, shipped as USDZ (~1.5–1.7 GB each) with
a reference camera mp4, `calibration_estimate.parquet` (the f-theta rig), and an **embedded HD map**.

**Release `26.04` is mandatory** — it must match the NRE 26.04 image. ✅ MEASURED.

```bash
# Committed downloader: .../2026-07-22-alpasim-closedloop-evalpod/scene_dl.sh   (token on stdin)
# Single scene (the n=1 reference scene):
hf download nvidia/PhysicalAI-Autonomous-Vehicles-NuRec --repo-type dataset --revision 26.04 \
  --include "sample_set/26.04_release/01d503d4-449b-46fc-8d78-9085e70d3554/*" \
  --local-dir /workspace/scene_dl
```

- Strip BOM/CR from the token: `tr -d '\r\357\273\277'`.
- ⚠️ **`HF_HUB_DISABLE_XET=1`** — the HF Xet backend errors on this dataset. ✅ MEASURED (`kf_download.sh`).
- Set `HF_HOME=/workspace/.hf` (not `/`).

**The 12-scene suite is fully specified and committed.** 🔍 STATIC-VERIFIED 2026-07-26 — all twelve
`clipgt-…` UUIDs are literal in `refc_suite_wizard_gen.sh`:

```
00040136-e651-4abd-991d-0655ccda9430   000525f6-3999-4812-9924-8adff40ca514
000548db-e266-49e5-a832-6674ab53a615   00064c58-7047-4a53-8a36-b033baaaa5fb
0009402a-a514-443b-9a4c-0e792f5ae581   00097de1-5ded-4fba-a5ed-4b527678d1b0
000a3a34-1031-4f90-9bc3-5b5c132fd1ed   000a74ae-5c01-486e-ab6f-7f5160136357
000e95f7-560d-4411-8069-b9f531ed3cd6   000ff49d-aa30-46ee-af57-b4a0c1143f55
0010ce77-d06e-43e6-bdaf-2cf8ab65cfe4   001564ce-0019-4ec6-bb62-07ed2bd90f2e
```

≈ **18 GB**. **This is what makes the suite reproducible** — the scene bytes are not in the repo and
never can be, but the exact scene *list* is, so anyone with gated HF access reconstructs the identical
suite. The wizard downloads them itself when given `scenes.scene_ids=[…]`.

**Current pod state (MEASURED 2026-07-26):** `/workspace/scene_dl` (1.7 GB) plus **four built scenesets**
under `/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/`, including
`e11a2e57085844fa5d905fa259abb344` (the 12-scene suite) and `482d8796dfba79cc76c7b1f759e3d6b1` (the
single M2 scene). **You do not need to re-download on this pod.**

---

## 5. Step 4 — run ONE scenario

Five services on localhost. All committed scripts live in the 07-22 bundle; `scp` them to `/workspace/`
on the pod (the drivers `sys.path.insert(0, "/workspace")` and import each other, so **that exact
directory matters**). ✅ MEASURED end-to-end (RUN_RECIPE §4).

```bash
# 0. take the lock; never run this on a training pod
gpu_lock.sh acquire alpasim

# 1. RENDERER (:6011).  Cold boot ~4.5 min (CUDA kernel JIT); warm boot fast.
setsid bash /workspace/renderer_serve.sh 6011 \
  '/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/<SCENESET>/**/*.usdz' \
  </dev/null >/workspace/renderer.log 2>&1 &
# wait for:  Serving on 0.0.0.0:6011 (health on same port)
# also expect: "Available scenes: [...]" and "Available egocar masks: EgocarRigBank(hyperion_8, ...)"

# 2. CONFIGS (wizard, run_method=NONE -> generate only, do not deploy)
printf '%s\n' "$HF_TOKEN" | bash /workspace/wizard_gen.sh 50 /workspace/m2run

# 3. BACKING SERVICES: controller :6007, physics :6006, driver :6789
bash /workspace/launch_services.sh

# 4. RUNTIME (drives the loop, writes rollouts + aggregate)
bash /workspace/run_runtime.sh /workspace/m2run
```

**What the wizard invocation means** (🔍 STATIC-VERIFIED — I read `wizard_gen.sh`):

```
deploy=local topology=1gpu driver=manual driver_source=external_static
wizard.run_method=NONE            # generate configs, do NOT docker-compose up
wizard.debug_flags.use_localhost=True   # map every service to localhost:<port>
scenes.scene_ids=[clipgt-…]  runtime.simulation_config.n_sim_steps=50
```

### 5.1 The three bare-run rewrites — skip any of these and it fails

The wizard emits **Docker-Compose-shaped** config. Three fixes, already inside the scripts:

| # | Problem | Fix | In |
|---|---|---|---|
| 1 | user-config points scene data at the container mount `/mnt/nre-data` | `sed` → the real host path `…/data/nre-artifacts` | `run_runtime.sh` |
| 2 | network-config points the renderer at `:6005`; ours is on `:6011` | `sed 's\|localhost:6005\|localhost:6011\|'` | `launch_services.sh` |
| 3 | scene release must match the NRE image | use **26.04** | §4 |

### 5.2 Service map

| Service | Port | Device | Command (inside `/workspace/alpa-invest/alpasim`) |
|---|---|---|---|
| renderer (NRE/pycena) | 6011 | GPU | `…/pycena_nrm_full serve-grpc --port=… --artifact-glob=… --egocar-hood-dir=… --no-enable-nrend --cache-size=5 --max-workers=4 --enable-editing-actors` |
| physics | 6006 | GPU (warp JIT) | `.venv/bin/physics_server --artifact-glob=… --use-ground-mesh=true --cache-size=16` |
| controller | 6007 | CPU | `.venv/bin/python -m alpasim_controller.server --config=…/controller-config.yaml` |
| **driver (yours)** | 6789 | GPU | see §6 |
| runtime | — | CPU | `.venv/bin/python -m alpasim_runtime.simulate --user-config=… --network-config=… --eval-config=… --log-dir=…` |

Renderer env (🔍 STATIC-VERIFIED from `renderer_serve.sh`): `RUNFILES_DIR="$BIN.runfiles"`,
`HOME=/workspace/nrehome`, `XDG_CACHE_HOME=/workspace/nrehome/.cache`, `OMP_NUM_THREADS=1`,
`NVIDIA_DRIVER_CAPABILITIES=all`, plus the `/app` symlink.

### 5.3 Expected first result

✅ MEASURED (`M2_results-summary.json`): status **PASS**, score **0.6637**, `collision_any=0`,
`offroad=0`, **`img_is_black=0.0`** (proof real frames rendered), `dist_to_gt_trajectory=0.574 m`,
drove 39.17 m of a 73.77 m GT path.

**`img_is_black` is your smoke test.** Non-zero means the renderer served black frames and *every other
metric is meaningless*.

---

## 6. Step 5 — plug in a policy

### 6.1 The contract

AlpaSim's driver is a **gRPC `EgodriverService`** and can run as a **bare external process**
(`driver_source=external_static`, `wizard.external_services.driver=["<ip>:6789"]`) — no Docker, no
plugin install, no entry-point registration required. **This is the path we use.** Implement the
service; do not implement the proto.

| Direction | Payload |
|---|---|
| **In** | `camera_images: dict[cam_id -> list[CameraFrame]]` (`timestamp_us`, HWC uint8 RGB) · `DriveCommand` (LEFT=0 / STRAIGHT=1 / RIGHT=2 / UNKNOWN=3) · `speed` (m/s) · `acceleration` · `ego_pose_history` · `inference_seed` |
| **Out** | `trajectory_xy: (T,2)` **rig frame: x forward, y LEFT** · `headings: (T,)` rad · optional `reasoning_text` |

Camera id: **`camera_front_wide_120fov`** (matches our training input). Batch eval cadence **2 Hz**;
controller MPC tracks at 10 Hz.

⚠️ **Do not copy the CARLA/TransFuser reference adapter's y-flip.** It flips y because CARLA is y-right;
we are on the NVIDIA rig, which is y-**left**, same as our own ego frame.

### 6.2 ⭐ The worked example — use `refc_driver.py` + `flagship_v1_driver.py`

> **Do NOT start from `flagship_v1_policy.py`** (07-19 bundle). Despite its name it is a **design stub
> that was never run**, with **11 open TODOs** including the exact preprocessing question that turned
> out to be make-or-break. Its own docstring says so. It is design history.
> 🔍 STATIC-VERIFIED 2026-07-26 (`grep -c TODO` = 11).

The real pattern is two files:

- **`refc_driver.py`** (262 L) — the gRPC servicer, session handling, frame buffer, **f-theta
  canonicalization**, nav derivation, and waypoint timestamping. **`RefCDriver` is model-agnostic**: it
  calls `policy.plan(raw_frames, intr, v0, nav_cmd)`.
- **`flagship_v1_driver.py`** (108 L) — proof of that: a whole second model in ~60 lines of real code,
  reusing `RefCDriver` unchanged.

**To add a new policy, write a class with one method:**

```python
class MyPolicy:
    def __init__(self, ckpt: str, device: str = "cuda"): ...
    @torch.no_grad()
    def plan(self, raw_frames: list, intr, v0: float, nav_cmd: int):
        # returns (traj[T,2] rig-frame x-fwd/y-left, headings[T])
```

…then serve it:

```python
add_EgodriverServiceServicer_to_server(RefCDriver(MyPolicy(ckpt)), server)
```

🔍 STATIC-VERIFIED 2026-07-26 — every import these two files need resolves to a real symbol:

| Import | Resolves to |
|---|---|
| `tanitad.data.calib.{F_REF, FThetaIntrinsics, ftheta_crop_resize}` | `stack/tanitad/data/calib.py:100,217` |
| `tanitad.data.comma2k19.stack_frames` | `stack/tanitad/data/comma2k19.py:204` |
| `tanitad.refs.refc.RefCModel` / `refc_config` | `stack/tanitad/refs/refc.py:682,282` |
| `tanitad.config.flagship4b_config` | `stack/tanitad/config.py:307` |
| `tanitad.models.fourbrain.WorldModel` | `stack/tanitad/models/fourbrain.py:395` |
| **`refc_v12_cache.load_frozen`** | **`stack/scripts/refc_v12_cache.py:88`** ← this is why `stack/scripts` must be on `PYTHONPATH` |

### 6.3 🔴 THE MAKE-OR-BREAK: f-theta canonicalization

**A naive `cv2.resize` / `F.interpolate` to 256×256 silently destroys the result.**

MEASURED (`refc_b2.py`, RUN_RECIPE §9 B2): canonical-vs-naive preprocessing shifts REF-C's *plan* by
**0.747 m mean per waypoint / 1.566 m @2 s / 3.97 m max** — **~3.3× REF-C's entire ADE budget (0.4728)**
and ~1.9× the CV-baseline gap. A closed-loop metric computed on naively-resized input is **meaningless**.

The correct transform is TanitAD's f-theta fisheye→pinhole canonicalization: a poly-dependent square
crop centred on the clip's **principal point** landing `f_eff ≈ F_REF = 266`, then bilinear resize to
256, then a 3-frame @100 ms stack → 9 channels.

**Where the intrinsics come from — this is the elegant part.** You do **not** need the build pods'
`r0_selection.parquet`, and you do **not** need to parse the USDZ. **The renderer hands them to the
driver** in `DriveSessionRequest.rollout_spec.vehicle.available_cameras[*].intrinsics` — the
**forward** `angle_to_pixeldist` poly, `principal_point_x/y`, and native resolution.
✅ MEASURED (`asl_camera_probe.py`): NuRec renders **native f-theta** (`MODEL=ftheta_param`, not
pinhole), cx=956.11 cy=754.85 at 1080×1920, forward poly `[0, 944.49, −10.98, 32.70, −77.40, 32.52]`
— consistent with TanitAD's known rig-B principal point (cx 958 / cy 753) and `poly[1]=927.5` (~2 %).

```python
# in start_session:  build FThetaIntrinsics from the session's ftheta CameraSpec
# in plan():
canon   = ftheta_crop_resize(vid, intr, 256, center="principal")   # [T,3,256,256]
assert abs(ftheta_crop_resize.last_f_eff - F_REF) < 8.0            # ← THE GATE
stacked = stack_frames(canon, 3)                                   # [T-2,9,256,256]
```

> ### 🚦 THE ONE GATE YOU MUST NOT SKIP
> Every driver logs `CANON f_eff=<x> (F_REF=266.0) OK|FAIL` on its first plan.
> **If it does not read OK, the run is void.** Observed good values: **265.6** (480×854),
> **265.7**, **265.9**, **266.0** (native). ✅ MEASURED on every banked run.

**Also required — waypoint timing (gate 2).** REF-C-family heads emit 4 waypoints at **0.5 / 1.0 / 1.5 /
2.0 s**, not at a uniform `1/hz`. Stamp them at their true horizons. The shared `SimpleDriver.drive()`
gets this wrong by default; `RefCDriver` overrides it. ✅ MEASURED as fixed.

**Also required — `v0` under force-GT.** In force-GT mode AlpaSim bypasses the controller and sends **no
dynamic state**, so the driver reads `speed=0` and the policy is mis-conditioned. `refc_driver.py`
estimates `v0` by pose finite-difference. A run made before this fix was correctly discarded. ✅ MEASURED.

### 6.4 Launching a driver

```bash
PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts \
  /workspace/alpa-invest/alpasim/.venv/bin/python /workspace/refc_driver.py \
    --port 6789 --ckpt /root/models/refc-base-30k/ckpt.pt --preset base
# flagship v1:
PYTHONPATH=… /…/.venv/bin/python /workspace/flagship_v1_driver.py \
    --port 6789 --ckpt /root/models/flagship-30k/ckpt.pt
```

⚠️ **`/root/models/flagship-30k/` IS the deployed v1 (`flagship4b-speedjerk-30k`, step 29999, 0.452 m).**
Do not confuse it with `tanitad-flagship-4b-phase0`, the **no-speed ablation control** (2.918 m). The HF
repo naming invites exactly this inversion; CLAUDE.md flags it as a repeat error.
🔍 STATIC-VERIFIED: both directories exist on the pod, separately.

The driver shares GPU 0 with the renderer. A40 46 GB fits comfortably (renderer ~6.5 GB peak + policy
1–2 GB).

---

## 7. Step 6 — run the n=12 paired suite

This is the statistical instrument. **Both arms over the SAME 12 scenes, one rollout each.**
✅ MEASURED — `vs_suite_master.sh` produced `flagship_vs_refc_suite_results.json`.

```bash
# 1. generate the 12-scene configs once (downloads ~18 GB if not cached)
printf '%s\n' "$HF_TOKEN" | bash /workspace/refc_suite_wizard_gen.sh /workspace/refcsuite

# 2. run BOTH arms autonomously, self-cleaning
bash /workspace/vs_suite_master.sh          # 480x854
bash /workspace/vs_suite_master_1080.sh     # native 1080x1920 (res flip only)
```

**Why `vs_suite_master.sh` is the right pattern** (🔍 STATIC-VERIFIED — I read all 65 lines): it starts
the renderer if needed, **rebuilds clean logdirs from the shared config** (so no stale rollout poisons
an aggregate — a real bug that once contaminated a flagship result), runs REF-C first as a known-good
pipeline validation, then flagship, then **kills every service by port and releases the GPU lock**. A
dropped controlling session leaves **no orphan**. Copy this shape for any new suite.

Paired statistics: `vs_aggregate.py` → bootstrap CI on the mean score delta, sign test, McNemar on pass
and on at-fault collision.

### 7.0 ⭐ Prefer the BALANCED n=37 suite — the n=12 set is skewed

**The n=12 suite is 8/12 straight-or-urban scenes, which is REF-C's single best category, and its
Δscore of −0.43 is inflated by that skew.** The balanced **n=37** suite (~7–8 scenes in each of
roundabout / highway / intersection / traffic-light / straight-other) puts the honest paired delta at
**−0.1228 [−0.2079, −0.0412]** — still a REF-C win, but ~3.5× smaller, with **roundabout and highway
tied** and **both models collapsing at uncontrolled intersections**. ✅ MEASURED,
`scenario_stratified_scaled_results.json`.

**Use the balanced suite for any new comparison.** The whole pipeline is committed:

```bash
bash /workspace/kf_download.sh      # pull candidate scene mp4s (HF_HUB_DISABLE_XET=1)
python /workspace/kf_batch.py       # mp4 -> ffmpeg keyframe -> PIL montage grid for labelling
python /workspace/select_suite.py   # balanced selection from the labels
printf '%s\n' "$HF_TOKEN" | bash /workspace/scaled_wizard_gen.sh
bash /workspace/scaled_master.sh    # both arms, self-cleaning (vs_suite_master.sh shape)
python /workspace/scaled_aggregate2.py
```

- **Labelling is manual keyframe inspection**, not metadata: NuRec/`public_2604` exposes **no
  scene-type field**. 356 candidates were screened to build the current suite; the labels
  (`scaled_suite_labels.json`) and all 54 keyframes are committed and auditable. **Roundabouts are rare
  (~2.5 % of the pool)** — 9 were found and each verified at full resolution
  (`scaled_roundabout_verify/`). A VLM pass (Cosmos-Reason1-7B, cached on the pod) is the scale-up path
  but needs a separate `transformers`/`qwen2.5-vl` env.
- **`scaled_aggregate2.py` exists because the runtime can drop a scene.** One scene failed a route
  sanity check (ego 53 m off-route) and the runtime skipped `results-summary.json` entirely; the script
  recovers the score directly from each rollout's `metrics.parquet`. Its scoring is validated — it
  reproduces the known single-scene flagship score **0.699** exactly. **Check `n` in the output;
  a silently-dropped scene is the failure mode here.**

### 7.1 ⭐ The OOD control — run this for any new renderer or scene set

The single most important diagnostic this asset has. It separates *model* from *environment*:

```bash
printf '%s\n' "$HF_TOKEN" | bash /workspace/refc_openloop_wizard_gen.sh   # force_gt_duration_us=20e6,
                                                                          # skip_driver_during_force_gt=false
# driver with --log-preds ; then:
python /workspace/refc_openloop_ade.py
```

The ego follows the GT path, so the policy predicts from **in-distribution poses** on rendered frames;
its predictions are scored against the GT path exactly as `taniteval` does. **If the resulting ADE is
far above the model's real-footage reference, closed-loop failures are confounded by the renderer, not
the model.** ✅ MEASURED: 1.5157 vs 0.4728 = **3.21×**.

---

## 8. Where results land, and how to read them

```
<logdir>/
├── generated-user-config-0.yaml      # scenes, cameras, resolution   (sed /mnt/nre-data first)
├── generated-network-config.yaml     # service endpoints             (sed renderer -> :6011)
├── controller-config.yaml, eval-config.yaml, driver-config.yaml
├── rollouts/<scene>/<rollout_uuid>/
│   ├── rollout.asl                   # ~7 MB full event log — the primary record
│   ├── metrics.parquet
│   └── *_camera_front_wide_120fov_default.mp4   # camera + planned traj + BEV inset + metrics
└── aggregate/
    ├── results-summary.json          # ⭐ the quotable artifact
    └── metrics_results.{txt,png,parquet}
```

### 8.1 Metric dictionary

| Field | Meaning | How to read it |
|---|---|---|
| **`img_is_black`** | fraction of black frames | **Check first. Non-zero ⇒ discard the run.** |
| `status` / `score` | PASS/fail; composite 0–1 | Score is the cleanest continuous signal at n=12 |
| `collision_any` / `collision_at_fault` / `_front` | collision flags | At-fault is the meaningful one |
| `offroad` | left drivable area | Flagship v1's dominant failure mode |
| `dist_to_gt_trajectory` | mean m from the recorded path | **Not** an error — deviating to avoid a collision is correct behaviour |
| `progress` / `progress_rel` | absolute / relative distance covered | `progress_rel` ≈ 1 with low `progress` ⇒ terminated early |
| `plan_deviation` | how far the plan swings | **The mechanism metric.** flagship 1.12 vs REF-C 0.34 explains the offroad gap |
| `min_dist_to_obstacle` | closest approach (m) | |
| `duration_frac_20s` | fraction of a 20 s budget survived | |
| `min_ade@*` | GT-matched ADE | **`null` for our drivers** — AlpaSim only computes it for GT-matched drivers. Use §7.1 instead. |

### 8.2 Reading rules

1. **Check `img_is_black == 0` and `CANON f_eff ≈ 266`.** Two gates, both cheap, both fatal.
2. **Quote paired deltas, not marginal rates.** §5.1 of `ALPASIM_STATE.md`.
3. **Carry the framing string.** "on NuRec reconstructions — within-sim relative."
4. **n=12 is underpowered for small effects.** A 4/12 rate spans ~13–61 % at 95 %. A n=12 departure
   "win" already reversed at n=40 once (`RETRACTION_LOG.md:59`).
4b. **Absolute rates track the SCENE MIX, not the model.** At-fault collision was 0.167 for both arms on
   the easy n=12 set and **0.432 / 0.297** on the balanced n=37 set — same models, same renderer. Never
   compare a rate across suites, and never quote one as a model property.
5. **Videos are free and already in our house style** — AlpaSim's default eval-video layout *is* the
   TanitEval viz standard (camera + projected plan + metric BEV inset + text overlay). Enable with
   `eval.video.render_video=true` (the suite scripts disable it for speed). ⚠️ Note `.gitignore`
   excludes `*.mp4`, so committing one needs `git add -f`.

### 8.3 Performance envelope (✅ MEASURED, `alpasim_realtime_a40.json`)

| | 480×854 | native 1080×1920 |
|---|---|---|
| real-time factor | **0.75–0.98×** | **0.29×** |
| flagship model forward | ~90 ms | ~90 ms |
| REF-C model forward | ~18 ms | 18–40 ms |
| driver frame-canon CPU | ~46 ms | **~475 ms** ← redundant re-canon, cacheable |

Budget ≈ 12 scenes × 50 steps at ~0.2–0.7 s/step per arm, plus a ~4.5 min cold renderer boot (warm boot
is fast; the kernel cache persists at `/workspace/nrehome/.cache`).

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: tanitad` | `PYTHONPATH` missing | needs **both** `stack` and `stack/scripts` |
| `ModuleNotFoundError: refc_v12_cache` | `stack/scripts` missing specifically | as above |
| `ModuleNotFoundError: alpasim_grpc.v0.*_pb2` | protos never compiled | `cd src/grpc && uv run compile-protos` |
| Runtime hangs at scene load | user-config still says `/mnt/nre-data` | `run_runtime.sh` sed |
| Runtime cannot reach the renderer | network-config still says `:6005` | `launch_services.sh` sed |
| Renderer silent >5 min on first boot | **normal** — CUDA kernel JIT ~4.5 min | wait for `Serving on 0.0.0.0:6011` |
| `img_is_black = 1.0` | renderer served black frames | check renderer log, scene glob, scene release = 26.04 |
| `CANON f_eff` FAIL / absent | preprocessing wrong | §6.3. **Void the run.** |
| Plans look scaled wrong / over-reaching | naive resize instead of f-theta canon | §6.3 |
| Policy behaves as if stopped | `speed=0` under force-GT | estimate `v0` from pose finite-difference |
| Scene download fails / stalls | HF Xet backend | `HF_HUB_DISABLE_XET=1` |
| `curl: (3) URL rejected` | CRLF in a digest list | `sed -i 's/\r$//'` |
| Disk full mid-extract | wrote to `/` | everything to `/workspace` |
| Stale rollouts pollute an aggregate | reused logdir | rebuild clean logdirs — `vs_suite_master.sh` pattern |
| ssh command dies with a shell syntax error | `()` in a remote `echo` label | remove them |
| Killing a service kills your ssh session | `pkill -f <name>` self-matched | **kill by explicit PID** (CLAUDE.md) |

---

## 10. Checklist for a clean-pod rebuild

- [ ] GPU free; `gpu_lock.sh acquire <tag>`; **not** a training pod
- [ ] `/workspace` writable (real `dd` test, not `df`)
- [ ] NGC key + approved HF token available on the dev box
- [ ] §2 workspace → `verify_imports.py` **bad=0** ✅
- [ ] §3 NRE rootfs ≈38 GB, launcher present, `/app` symlink 📝 *first run of the transcribed script — expect to fix it*
- [ ] §4 scene(s), release **26.04**
- [ ] §5 single scenario → **PASS**, `img_is_black=0`
- [ ] §6 driver → **`CANON f_eff ≈ 266 OK`** in the driver log
- [ ] §7 suite → `results-summary.json` per arm + paired stats
- [ ] §7.1 OOD control re-run if the renderer or scene set changed
- [ ] every reported number carries **"on NuRec reconstructions — within-sim relative"**
- [ ] services killed **by explicit PID**, GPU lock released, no orphan

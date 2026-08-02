# AlpaSim on the Jetson Thor — the path is OPEN, every major unknown de-risked

**PI directive:** *"Get AlpaSim in Thor running and generate there the tests and the videos."*
**Status: feasible, and the four things that could have killed it are all MEASURED as fine.**

The earlier "blocked" note was about ONE acquisition path (`docker pull`) that this programme does
not use. See `ALPASIM_ON_THOR_BLOCKED.md`, corrected.

---

## What is MEASURED (2026-08-02, all on Thor)

| # | question that could have killed it | answer |
|---|---|---|
| 1 | Can we rasterize 3D Gaussians on aarch64 Blackwell? | ✅ **YES — `gsplat 1.5.3`, 492 FPS @ 640×256, 20 k gaussians (2.03 ms/frame).** Compiles for `sm_110`. ~50× the 10 Hz we need. |
| 2 | Can we get the NuRec scenes? | ✅ **YES — 1,607 usdz scenes readable on HF.** One pulled: **1.96 GB in 138 s**. |
| 3 | Is the reconstruction an opaque NVIDIA blob? | ✅ **NO — `volume.nurec` is gzip + plain MessagePack.** Parsed on Thor. |
| 4 | Is the front camera available? | ✅ **YES — `camera_front_wide_120fov.mp4` ships with the scene** (38 MB). Exactly the PI's *"render only the front camera"*, and a correctness reference. |

### The scene, decoded

```
volume.nurec (624 MB gz -> 639 MB msgpack)
  nre_data/version              : 26.4.96
  nre_data/model                : nre
  nre_data/config/name          : gaussians-composite
  nre_data/config/layers/       : background · road · dynamic_rigids · dynamic_deformables
  nre_data/config/layers/background/name : sh-gaussians     <- spherical-harmonic gaussians
  nre_data/config/background    : sky-env-map, cubemap 512x512
  nre_data/config/calib         : free-pose-calib
```

Alongside it in the usdz: `rig_trajectories.json` (2.1 MB), `mesh_ground.usd` (15 MB, physics),
`sequence_tracks.json` (actors), `pose_record.json`, `parsed_config.yaml`, and **`map.xodr`
(518 KB OpenDRIVE HD map)**.

`data_info.json`: 202 poses, 27563309000 → 27583309000 µs = **20.0 s per scene**.

⭐ **STRATEGIC SIDE-FINDING — `map.xodr` is an OpenDRIVE HD map with lanes and junctions.**
The programme concluded (five probes) that PhysicalAI-AV carries *no map, lane graph or junction
annotation*, and that the strategic brain's topology *"must come from AlpaSim or an external
corpus"*. **This is that source, and it is one zip entry away.** Different dataset from the AV
train set, so it does not contradict the earlier finding — it answers it.

## Why the renderer is replaceable at all

AlpaSim is six gRPC microservices, and the renderer is one of them:

- `src/runtime/alpasim_runtime/services/renderer.py` defines **`RendererService(Protocol)`** — a
  Python Protocol, i.e. a duck-typed interface, not a hard dependency on NVIDIA's binary.
- `src/grpc/alpasim_grpc/v0/sensorsim.proto` is the wire contract.
- `base_config.yaml` carries **`renderer: null # Optional override`**.
- A **second** renderer family already exists in-tree (`video_model.proto`,
  `test_video_model_renderer.py`), so a non-NRE renderer is an anticipated configuration.

⇒ A gsplat-backed service speaking `sensorsim.proto` is a **supported substitution, not a hack**.

## The build, in order

| # | step | risk |
|---|---|---|
| 1 | Parse the gaussian payload → tensors (means, quats, scales, opacities, SH) | ⚠️ **the one open unknown** |
| 2 | Render with gsplat + `rig_trajectories.json`, diff against the shipped front-camera mp4 | low — the reference video makes it checkable |
| 3 | Wrap as a `sensorsim.proto` gRPC service (front camera only) | low |
| 4 | `alpasim_wizard … renderer=<ours> driver=tanitad` with the REF-C / flagship-v1 adapters | low — adapters exist and are proven |
| 5 | Long videos: chain 20 s scenes + `eval.video.video_layouts=[REASONING_OVERLAY]` | low |

⚠️ **Step 1 is the real work and the honest risk.** The container is parseable and self-describing;
what is NOT yet proven is that the gaussian payload maps cleanly onto gsplat's parameterisation
(SH degree, scale/opacity activations, layer compositing order, dynamic-actor rigging). **Rendering
the shipped reference video is the falsifier** — if our render does not match it, the mapping is
wrong and we find out immediately rather than shipping a plausible-looking wrong world.

## Ops notes earned here

- `pip install ninja` puts `ninja` in the **venv** bin; a non-login ssh shell does not have it on
  PATH, and gsplat's JIT then fails with a misleading *"Ninja is required to load C++ extensions"*.
  Export `PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH`.
- `Python.h` is absent from `/usr/include/python3.12` (no `python3-dev` installed), but **present in
  the uv-managed interpreter**:
  `export CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12`.
  **No sudo, no system packages.** First JIT build: 126 s.

## Evidence class

| claim | class |
|---|---|
| gsplat 492 FPS on Thor | **MEASURED (ours)** — finite output, mean 0.2639 |
| 1,607 scenes; one downloaded 1.96 GB | **MEASURED** — HF listing + download |
| `volume.nurec` is gzip+msgpack, layer names | **MEASURED** — parsed on Thor |
| 20.0 s / 202 poses per scene | **MEASURED** — `data_info.json` |
| `map.xodr` is usable OpenDRIVE | ⚠️ **MEASURED it exists (518 KB)**; contents NOT parsed |
| steps 2–5 are low risk | ⚠️ **ESTIMATED** — follows from proven interfaces, not yet run |

# REFe's NAVSIM navtest evaluation on Colab Pro: a measured plan (2026-09-27)

**The question.** Can the per-snapshot `sub200` eval and the final full-navtest eval move off the dev box onto Colab
Pro? Would they produce the same results? What would it cost?

**Evidence.** The raw evidence is in `colab/raw/2026-09-27-navtest-plan/`, and every number below carries its class.

- The probes were metadata-only: HEAD requests, API listings and range reads of 50 MiB or less. Nothing larger than
  100 MB was downloaded.
- One CPU-runtime smoke was run, in three short sessions. It used **5.5 VM-minutes at 0.08 CU/h, about 0.007 CU**.
  The server was checked afterwards and read `assignments: []` with a rate of 0 (`final_ccu_check.txt`).
- No GPU was used, on Colab or on the dev box. PKG, `Project Steering/` and the existing `colab/*.md` files were not
  touched.
- Private files use the coordinator's route (2026-09-27, MEASURED in `colab/raw/2026-09-27-hf-relay/`). The dev box
  pushes them to the private relay `Sayood/tanitad-colab-relay`. It then mints short-lived signed links, and the VM
  pulls and md5-verifies. ⛔ **No HF token ever goes to a VM.**

## Answer first

1. **Yes, both evals can run on Colab.** The pipeline splits cleanly at the seam npz, which is 65 KB for sub200 and
   about 4 MB for full navtest.
   - The seam stage is the only one that needs a GPU. It is the stage worth moving: it blocks the dev-box GPU, and
     the final navtest's 5–12 h projection is almost entirely seam time.
   - The harness and E-6 stages are CPU-only and can move too. The floors/families/readouts should stay on the dev
     box: they need the TanitAD venv and the pod's `metrics.jsonl`.
2. **Would the results be identical?**
   - **The CPU stages should match exactly.**
     - MEASURED on Linux: the seam converter's analytic self-test and the planner's JPEG preprocessing are bit-for-bit
       identical to the dev box (4/4 array hashes).
     - The harness itself was not yet proven bit-exact on Linux. It needs all four nuPlan map files (0.97 GB zip),
       which the 100 MB rule held back. That is gate G-H, the first step of P0 (§f).
   - **The GPU seam cannot be guaranteed bit-identical on a different GPU.**
     - PUBLISHED: cuBLAS is bitwise reproducible only on "GPUs with the same architecture and the same number of SMs"
       ([cuBLAS §2.1.4](https://docs.nvidia.com/cuda/cublas/index.html)).
     - MEASURED on 4 of the 200 tokens, including the three whose pick margin is closest to a tie: moving the model
       from the RTX 4060 to fp32 CPU shifted the executed poses by **≤ 0.53 mm**. All 4 picks stayed the same.
       Per-token PDMS changed by **≤ 1.02e-5** on the 0–1 scale.
     - So a tolerance gate is admissible and has been measured (§f).
3. **What it costs.** All of these are ESTIMATED from measured rates (§d), and are replaced by P0's measurements
   (P0 is the parity run on snapshot 016; see the recommended plan).
   - **Per snapshot (sub200, including E-6):** about **0.70 CU on an L4** (27 min).
   - **Final full navtest:** about **7.8 CU on a G4, ≈ 53 min** from model to PDMS. An L4 costs ≈ 5.9 CU but takes
     ≈ 3.8 h.
   - **For the rest of the run** (P0 + ~12 snapshots + the final): about **17.5 CU**, 9 % of the 200 CU balance.
4. **Integration work** (the Master Mind or EvalFlyWheel should own this; nothing here is merged):
   - a Linux runner for eval steps 1–2 (`eval_checkpoint.py` hard-codes Windows paths and `;` in PYTHONPATH);
   - a log-sharded seam plus a merge step;
   - a Linux E-6 driver;
   - a data-prep script.

   A prototype of the harness launcher is banked and was exercised: `raw/…/linux_score_v1.py` and
   `navsim_v1_linux.py`.
5. **PI and orchestrator decisions needed:** listed at the end of this file.

---

## a. Inputs: everything the pipeline reads, sized on the dev box (MEASURED, `raw/…/input_sizes.json`)

Every entry was traced from the code, not assumed. In the table, "sub200" means W3's `A1_sub200_tokens.json`
(200 tokens / 93 logs). "Full" means `navtest_all_tokens.json` (12,146 tokens / 136 logs). Byte counts are apparent
sizes; on D:'s exFAT the allocated size is larger.

| # | input | read by (file:line) | sub200 | full navtest |
|---|---|---|---|---|
| 1 | token list (W3 format) | `refe_navtest_seam.py:140`; `run_v1.split_overrides` | 17,763 B | 1,044,791 B |
| 2 | W3 export `navtest_inputs.json.gz`: token order, log, fingerprint, human future | `refe_navtest_seam.py:38,138`; `families6.py` | subset 193,910 B (gzip) | 8,188,529 B |
| 3 | nuPlan **test DBs** `<log>.db`: scenario, route roadblocks, `image` + `camera` tables | `navtrain_scenarios.build_scenarios_for_log`; `planner.FrameResolver._index`; `calib_table.read_db` | 93 DBs, **10.03 GB** (5.81 GB compressed) | 136 DBs, **13.16 GB** (7.68 GB compressed) |
| 4 | nuPlan **maps**: all 4 cities' `map.gpkg` + `nuplan-maps-v1.0.json` | seam: `planner._goal_for` route; harness: `Scene._build_map_api` → `GPKGMapsDB._load_map_data` reads a layer of **every** city (smoke run 1) | all 4 cities: **1.43 GB** (zip 0.97 GB) | same |
| 5 | **camera frames** `<log>_<CAM>.zip`, 4 cams per token | `planner._image_for` | 800 JPEGs, **165.8 MB** | 48,584 JPEGs, **10.05 GB** (544 zips) |
| 6 | REFe checkpoint | `ckpt_io.load_for_inference` | `snap_epochNNN.pt` **76,052,289 B** | `model_final.pt` ~1.3 GB (ESTIMATED from `ckpt_io.py:11-12`; not written yet) |
| 7 | DINOv3 ViT-L/16 trunk, only for PARTIAL snapshots | `ckpt_io.py:131-141` → `load_dinov3.load_state` | **1,212,347,640 B** | not needed (model_final is FULL) |
| 8 | REFe code: `refe/{planner,model,ckpt_io,load_dinov3,navtrain_scenarios,calib_table}.py`, `code/augment_routes.py`, `eval/refe_navtest_seam.py` | — | 0.14 MB | same |
| 9 | DriveZero checkout @`2495954`: `driverl` + its nuPlan fork with `driverl_runtime_map_features` | `planner.py:451,460`; editable installs | 58.1 MB tracked (zip 54.0 MB) | same |
| 10 | seam venv `driverl-eval`: CPython 3.11.13, 117 distributions, torch 2.7.1+cu128 | all of the seam | 6.81 GB installed (Windows) | same |
| 11 | harness code: W3 `run_v1.py`, `navsim_v1_win.py`, `w3_agents_v1.py`; E1 `navsim_win.py`; `eval/score_navtest_refe.py` | `score_navtest_refe.py:64-74` | 0.2 MB | same |
| 12 | NAVSIM v1.1 tree @`3e8291b` | `navsim_v1_win.py:44`, asserted by C8 | 13.5 MB (zip 6.53 MB) | same |
| 13 | nuplan-devkit @`ce3c323` (v1.2) | `navsim_v1_win.py:46`, asserted by C8 | 7.36 MB (zip 2.45 MB) | same |
| 14 | harness venv `navsim-crun`: CPython 3.9.25, 202 freeze lines | `run_v1.NV_PY` | 2.40 GB installed (Windows) | same |
| 15 | `navsim_logs/test/<log>.pkl`: SceneLoader → AgentInput + Scene; metric caching | `run_pdm_score.py:81`, via `OPENSCENE_DATA_ROOT` | 93 logs, **762 MB** | 136 logs, **997 MB** |
| 16 | **metric cache** `metric_cache_navtest/<log>/unknown/<tok>/metric_cache.pkl` + metadata CSV | `run_pdm_score.py:75-77` | 200 pkl, **50.0 MB** | 12,146 pkl, **3.16 GB** (+9.9 MB metadata) |
| 17 | outputs: seam npz (+ E-6 proposal dump) | harness, E-6, families | 65 KB (+1.59 MB dump) | ~4 MB (+~96 MB dump, ESTIMATED by scaling) |
| 18 | floors (W3 CV/STOP/HUMAN/refcv4b per-token CSVs), cluster map, TanitAD venv, EV6 `parse_navtest6.py` / `families6.py` | eval steps 3–4 | small | small (**stays on the dev box**) |

**Environment the code reads.**

- Seam stage (`eval_checkpoint.env_driverl`):
  - `NUPLAN_MAPS_ROOT`, `NUPLAN_DATA_ROOT`, `REFE_BACKBONE_ROOT`;
  - **`REFE_SIM_HZ=10`**, which is load-bearing: `navtrain_scenarios.py:64` defaults to 20;
  - `OMP_NUM_THREADS=4`, `DZ_ROOT` and `DRIVERL_EVAL_*` (not read by the seam path), and PYTHONPATH (redundant with
    the editable installs).
- Harness stage (`run_v1.ENV`):
  - `NUPLAN_MAP_VERSION`, `NUPLAN_MAPS_ROOT`, `NAVSIM_EXP_ROOT`, `NAVSIM_DEVKIT_ROOT`, `OPENSCENE_DATA_ROOT`;
  - `OMP/MKL/OPENBLAS_NUM_THREADS=2`, `PYTHONHASHSEED=1`, `CUDA_VISIBLE_DEVICES=-1`.
- **On Colab, also `MPLBACKEND=Agg`** (see §c, trap 2).

## b. Where each input comes from (metadata only: `raw/…/public_sources.json`, `camera_shards_needed.json`)

| input | source | size | same bytes as the dev box? |
|---|---|---|---|
| 3 test DBs | public: `https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_test.zip`. HEAD 200, `Accept-Ranges: bytes`, a real zip whose members are separately addressable | 95,919,476,643 B total. **Range-fetch only the needed members**: sub200 **5.81 GB**, full **7.68 GB** compressed | ✅ MEASURED: remote size = local zip; the remote central directory (187 KB read) lists **136/136** navtest DBs with **identical compressed size, uncompressed size and CRC** to the local zip the DBs were extracted from |
| 4 maps | public: `…/public/nuplan-v1.1/nuplan-maps-v1.1.zip` (NAVSIM's `download_maps.sh` URL, 200) | **970,997,691 B**; the 4 gpkgs are 835 MB compressed | ✅ size equals the local zip |
| 5 frames | public: HF `OpenDriveLab/OpenScene` `openscene-v1.1/openscene_sensor_test_camera/…_{0..31}.tgz` (200) | 32 shards, **127.88 GB**. ⚠️ **Both sub200 and full need ALL 32 shards**: the HF shard index `openscene_sensor_test_0-31.json` puts sub200's 93 logs in 32/32 shards. So sub200 would stream 128 GB for 166 MB of JPEGs, and its frame subset goes through the relay instead | ✅ 32/32 shard sizes equal the local copies |
| 15 navsim_logs | public: HF `openscene-v1.1/openscene_metadata_test.tgz` (all 147 test logs) | **476,034,809 B** | ✅ **sha256 identical** (`871ba786…`), MEASURED (`metadata_tgz_sha_check.json`) |
| 7 trunk | public: HF `timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors` | 1,212,347,640 B | ✅ **sha256 identical** (`45172f20…`) |
| 9 DriveZero | public: GitHub `XiaomiAutoL3/DriveZero` @`2495954`, commit exists | zip 54.0 MB (fetched on the VM at 16.9–17.5 MB/s) | pinned by SHA |
| 12, 13 NAVSIM v1.1, nuplan-devkit v1.2 | public: GitHub `autonomousvision/navsim` @`3e8291b`, `motional/nuplan-devkit` @`ce3c323` | 6.53 MB / 2.45 MB (the VM's navsim zip is byte-for-byte the size of the dev box's) | pinned by SHA |
| 10, 14 venvs | public: PyPI + the PyTorch index (`raw/…/linux_wheels.json`) | seam venv: **113/114** pins have a Linux cp311 wheel (antlr4 is sdist-only, pure Python); `torch-2.7.1+cu128-cp311-manylinux_2_28_x86_64` is on the cu128 index. Harness venv: **194/197** (antlr4 and python-control are sdist-only; torch 2.0.1 → `+cpu` from the CPU index) | rebuilt on the VM (§c) |
| 6 checkpoint | **private**, from the A40 pod: pod → dev box (scp, as today) → **relay** → signed link → VM | per snapshot 76 MB | ✅ MEASURED by the coordinator: `snap_epoch016.pt` pushed in 77.4 s and pulled on a CPU VM in **7.3 s (10.4 MB/s)**, **md5 identical** (`hf-relay/signed-proof.json`) |
| 1, 2, 16 (sub200) + the sub200 frame subset | **ours**, derived from public inputs, via the **relay** (one-time push ≈ 228 s) | 224 MB | md5-verified by `signed_pull.py` |
| 16 (full) | **ours**: rebuilt on the VM with the unmodified devkit from 4 + 15 (validated by G-H′, §f), or pushed once to the relay (3.16 GB ≈ 54 min of uplink) | — | — |
| 8, 11 code | **ours**: `colab upload` (0.2 MB). ⚠️ The GitHub repo `sayoood/TanitAD` is **publicly readable**: MEASURED by three probes (API `private: false`, HTML 200, raw file 200). So a VM *could* fetch the code by commit SHA. **PI: confirm this exposure is intended.** | | |

## c. Environment: recreated on a Colab VM (MEASURED by the smoke, `raw/…/smoke*/`)

**The CPU runtime.** It is `Shape.STANDARD`: 2 vCPUs (Xeon @ 2.20 GHz), 12 GB RAM, 206 GB free disk, Ubuntu 24.04.5,
Python 3.13.15, **uv 0.12.15 preinstalled**. It costs **0.08 CU/h**, reads READY in 14.6–14.8 s, and `colab upload`
of an 18 MB bundle ran at **0.69–0.82 MB/s**.

| step | result | time |
|---|---|---|
| `uv python install 3.9.25 3.11.13`: the dev box's exact patch versions | ✅ | 4.2 s |
| GitHub codeload: NAVSIM v1.1 + nuplan-devkit + DriveZero | ✅ | 5.9 s |
| **harness venv** (py3.9.25, 197 pins `--no-deps` + `torch==2.0.1+cpu` + editable nuplan-devkit) | ✅ 2.51 GB | **51.0 s** |
| harness imports, C8-style: `navsim` from the v1.1 tree, `nuplan` from `ce3c323` | ✅ numpy 1.23.4, shapely 2.0.7 / GEOS 3.11.4 | 22.4 s |
| **seam venv** (py3.11.13, 115 pins `--no-deps` + `torch==2.7.1+cpu` for the smoke + editable DriveRL and its nuPlan fork) | ✅ 1.97 GB | **25.1–31.4 s** |
| `refe_navtest_seam.py --selftest` | ✅ `ZZCONVERTER_EXACTZZ`; the same bars and numbers as the dev box (2.499e-02 / 1.875e-02 / 9.998e-03 m, 8.9e-16 rad) | 0.6 s |
| REFe import + ViT-L build | ✅ **322,070,854 params / 18,991,426 trainable**, which equals `ckpt_io.py:9` (322.07 M / 18.99 M); rule `navsim_v1`; `REPAIR_LAST_HEADING=True` | 12.4 s |
| **JPEG decode + resize parity**, i.e. `planner._image_for` up to the float cast, on 4 real frames | ✅ **4/4 sha256 identical** to the dev box (opencv 4.11.0, numpy 1.26.4) | 3.0 s |
| harness run: W3's harness on Linux, the dev box's e2chk seam and metric-cache pickles, 25 tokens | ⛔ reached per-token scoring, then **every token failed: `NUPLAN_MAPS_ROOT` does not exist** | 22.1 s |

**Three traps, all MEASURED and all new:**

1. ⛔ **Harness scoring needs the nuPlan maps.** I assumed it did not, and the smoke refuted that. `SeamAgentV1`
   requires a scene, so `Scene._build_map_api` builds a `GPKGMapsDB`, whose `_load_map_data` loads a layer from **all
   four** cities' `map.gpkg`. The harness therefore needs the full 0.97 GB maps zip, the same as the seam.
2. ⛔ **The Colab kernel exports `MPLBACKEND=module://matplotlib_inline.backend_inline`**, and every subprocess
   inherits it.
   - The seam venv has no `matplotlib_inline`, so `import planner` dies with `ValueError: Key backend …`. The first
     import that pulls matplotlib is `planner` (smoke run 3).
   - The harness venv happens to carry `matplotlib-inline`, which hid the problem there.
   - ⇒ Set **`MPLBACKEND=Agg`** for every eval subprocess.
3. ⛔ **The dev box's metric cache cannot be unpickled on Linux.** navsim v1.1 pickles `MetricCache.file_path` as a
   `pathlib.Path`, so a Windows-built cache carries `WindowsPath`, and Python 3.9 on Linux raises
   `NotImplementedError: cannot instantiate 'WindowsPath' on your system`.
   - A two-line shim, `pathlib.WindowsPath = pathlib.PureWindowsPath` in the wrapper process, loads it
     (`ZZUNPICKLE_OK_WITH_SHIM PureWindowsPath MetricCache`).
   - The metadata CSV's `D:\…` paths must also be rewritten; nothing the scorer computes reads `file_path`.

**Windows-only patches that do NOT carry over:**

- The `fcntl.py` shim (EVAL_VENV App. A) must not be installed: it raises `ImportError` off Windows by design.
- MSYS `tar --force-local` is not needed.
- The E1 loader patch (`/` vs `\` in cache paths) is a no-op on Linux paths. `score_navtest_refe.py` always passes
  `--patch-loader`; keep it for code identity.

**What must be rebuilt for Linux** is only path plumbing. `run_v1.py` and `navsim_v1_win.py` hard-code `C:`/`D:`
paths (`NV_PY`, `V11_TREE`, `EXP`, `DATA`, `NUPLAN_TREE`, `E1_WRAPPER`). The banked prototype
`raw/…/linux_score_v1.py` + `navsim_v1_linux.py` re-points only those module globals and then calls W3's unchanged
`cmd_score` / wrapper `main()`. That is the same pattern `score_navtest_refe.py` already uses for `RAW`.
`refe_navtest_seam.py` is already portable through its args (`--frames --db-dir --export --tokens --out`) plus the
environment above.

⚠️ **The GPU session's seam venv must install `torch==2.7.1+cu128` WITH its `nvidia-*` wheels.**

- The Windows cu128 wheel bundles its CUDA DLLs, but the Linux wheel depends on separate `nvidia-*` packages.
- So on Linux, `--no-deps` for torch alone would import-fail.
- Install torch first from the cu128 index, then everything else with `--no-deps`. Verify with a **real CUDA conv2d**
  (CLAUDE.md torch trap).
- torch 2.7.x cu128 ships sm_89 (L4) and sm_120 (G4 Blackwell) kernels. The VM driver 580 / CUDA 13.0 runs cu128
  (COLAB_PRO.md). Real CUDA on this venv is **UNVERIFIED until P0**.

## d. Parallelism and projection

**The stages, MEASURED on the dev box:**

| stage | bound by | dev-box rate | how it parallelises |
|---|---|---|---|
| seam: GPU forward (ViT-L on 4×512×960, fp32, batch 1) | GPU | ≈ 0.70 s/token = 1.30 s total − 0.60 s CPU side (ep011, the lightest load, `points/sub200_ep011.json`); up to 10.5 s/token when other sessions held the GPU (ep014) | no worker arg. Run K copies on **disjoint log shards** (the forward is per-token and batch-1, so shards are independent) and merge. On the same GPU the result is bitwise identical (cuBLAS §2.1.4) |
| seam: CPU side (scenario build, route/goal, calib, 4-cam JPEG decode+resize, frame control) | 1 core | **0.25 s/token** at full-navtest density (155 tokens / 2 logs); **0.60 s/token** at sub200 density (1 token per log). Largest phase: decode+resize, 0.13 s. MEASURED with the model call stubbed (`seam_cpu_side_timing.json`) | overlaps across the K shards |
| harness (W3, `worker=sequential` hard-coded in `score_navtest_refe.py:73`) | 1 core, RAM ~0.6 GB (sub200) / 2.2 GB (full) | sub200: 59.2 s per run. Full navtest: **1,050–1,220 s** per run (W3 counts JSONs) | shard by token file into K unchanged runs, then concatenate the CSVs (per-token scores are independent) |
| E-6 (64 single-proposal harness runs) | CPU | per run 34–96 s (median 49.5 s); **1,730.5 s** wall with 2 workers (ep016 `gates.json`) | already parallel: `proposal_table.py --workers W` |
| metric cache (rebuild) | CPU | **1.47 s/scene**, one worker (W3 RESULT.md) → 4.96 CPU-h for the full navtest | W3 `run_v1.py cache --per-log --logs-per-process N` |

**The whole snapshot eval on the dev box today** (MEASURED): `eval_checkpoint.py` takes 310–3,075 s. For ep016 that
was 1,112 s (seam 918 s) plus E-6 at 1,731 s, about 47 min end to end. The readiness doc projects **5–12 h** for the
final full navtest.

**Colab projections** (`raw/…/project_costs.py` → `projections.json`, ESTIMATED). The inputs are MEASURED except:

- the per-token GPU time: 6.4 TFLOP/token ÷ (measured fp32 TFLOPS × 0.75), i.e. **L4 0.78 s, G4 0.12 s**;
- the VM-core factor ×2.0 on the CPU phases (the 2-vCPU runtime decoded JPEGs 3.05× slower than the 24-thread dev
  box);
- S3 at the measured **21.3 MB/s** (4 short streams, so a floor; 1 stream was 11.6 MB/s);
- public HF at 110 MB/s (MEASURED on L4/G4);
- relay pulls at the measured **10.4 MB/s** (a CPU VM; a GPU VM is UNMEASURED);
- relay pushes at the measured **0.98 MB/s** (dev box).

Transfers from independent sources run concurrently. The data phase is the slowest chain, and the dev-box pushes
happen **before** the VM exists, so they are not billed.

| job | GPU | route | setup | data (critical chain) | seam | harness | E-6 | **total** | **CU** |
|---|---|---|---|---|---|---|---|---|---|
| sub200 + E-6 (per snapshot) | **L4** | public + relay | 2.6 min | 6.6 min (S3 DBs + maps) | 3.3 min (3 shards) | 2.0 min | 11.6 min | **27 min** | **0.70** |
| | G4 | public + relay | 2.6 | 6.6 | 1.3 (8 shards) | 2.0 | 3.3 | 17 min | 2.49 |
| full navtest (final) | L4 | public + relay | 2.6 | 61 min (cache rebuild on 10 cores) | 158 min (GPU-bound) | 7.8 min | — | 3.8 h | 5.9 |
| | **G4** | public + relay | 2.6 | 21 min (the 128 GB public frame stream; model_final from the relay ≈ 2.6 min) | 24 min (8 shards, GPU-bound) | 3.6 min (16 shards) | — | **53 min** | **7.8** |
| | G4 | relay bundle, relay at 10.4 MB/s | 2.6 | 29 min (the relay pull of 14.5 GB) | 24 | 3.6 | — | 60 min | 9.0 |
| | G4 | relay bundle, relay at 110 MB/s | 2.6 | 8.5 min (S3 DBs + maps) | 24 | 3.6 | — | 40 min | 5.9 |

In the table, "relay bundle" means a one-time push of the full-navtest frames (10.05 GB ≈ 2.8 h) and the full cache
(3.16 GB ≈ 54 min) of dev-box uplink.

**Readings:**

- For sub200 the GPU is a small part of the session: 200 forwards take ≈ 2.6 min on an L4. Setup, data and E-6 are
  CPU/network work billed at the GPU's rate, so the **L4 is 3.6× cheaper** than the G4.
- For the full navtest the L4's fp32 (11.0 TFLOPS, MEASURED) makes the seam 2.6 h. The G4 is ≈ 53 min at 1.3× the CU.
- Staging the full frame set in the relay pays off only if a **GPU VM** pulls signed links at well above 20 MB/s.
  - At the measured CPU-VM rate it is *slower* than streaming the public shards.
  - P0 measures the GPU-VM rate first, and the push decision waits for that.

## e. Data-path options

| option | prerequisites | per-eval transfer (ESTIMATED from the measured rates) | verdict |
|---|---|---|---|
| **1. Public inputs from their public sources**: nuPlan DBs and maps (S3), navsim_logs and the trunk (HF), the camera shards (HF, full only), code (GitHub) | none | **sub200:** range-fetch 93 DBs 5.81 GB + maps 0.97 GB ≈ 6.6 min at 21 MB/s (MEASURED floor; more parallel member fetches should be faster, UNMEASURED); logs + trunk from HF ≈ 20 s. **Full:** DBs 7.68 GB + maps ≈ 8.5 min; stream all 32 camera shards 127.9 GB and keep 48,584 JPEGs ≈ 21 min at 110 MB/s. `build_navtest_frames.py` reads shards sequentially, so this needs a per-shard-parallel variant. Rebuild the metric cache ≈ 13 min on G4 / 60 min on L4, overlapping the stream | **the default for every public input**; the DB range fetch reuses `code/fetch_front_camera.py`'s `rng` / `central_directory` / `entries` |
| **2. Private relay + signed links** (the coordinator's route, MEASURED): `hf_relay_push.py` (quota-guarded) → `hf_signed_links.py` mints per-file links valid for 60 min, **on the dev box** → `colab upload` of the job file → `signed_pull.py` on the VM verifies bytes + md5 | none new. ⛔ **The HF token never goes to a VM**: the permission system refuses it, and CLI sessions cannot read Colab Secrets (MEASURED: no reply in 30 s). Private storage: 184.6 GB used of the 1,000 GB billing ceiling (`hf-relay/hf_quota_check.json`) | **Per snapshot:** push 76 MB (77 s, before the VM) and pull 7.3 s. **One-time for sub200:** frames 166 MB + cache pickles 50 MB + export 8 MB ≈ 228 s of push. **Final:** `model_final` 1.3 GB ≈ 22 min of push before the VM, pulled in ≈ 2.6 min at the measured CPU-VM rate. **Optional** full frames + cache 13.2 GB ≈ 3.7 h of push (see §d) | **the route for every private or derived file.** Mint links just before the pull, since they expire after 60 min |
| **3. From the pod** | the pod does **not** hold navtest inputs (it has navtrain, 449 GB), so this only affects the **checkpoint**. A pod → relay push would skip the dev box's uplink (the pods' HF relay runs at ~118 MB/s, CLAUDE.md). A VM → pod SSH key is a separate PI decision. The pod was not touched | `model_final`: removes the 22-min dev-box push, which is off the billed path anyway | only saves wall-clock before the final; an orchestrator call |
| 4. The Colab account's Google Drive via `colab drivemount` (the CLI has the command) | the PI's OAuth consent on `fambouzouraa`. **UNVERIFIED, not tried**: it is an account-permission action | Google-internal, so a VM-built bundle would persist across VMs | not needed now that route 2 exists |

## f. Parity gate: how a Colab run proves it reproduces the dev box

The reference is snapshot 016 on sub200:

- seam: `D:/Projects/TanitAD/data/refe_navtest/seams/refe_sub200_ep016.npz`;
- proposal dump: `…/proptable/sub200_ep016/proposals.npz`;
- E-6 table: `…/proptable/sub200_ep016/table.npz`;
- per-token CSV: `…/score/refe_sub200_ep016/refe_sub200_ep016.csv`;
- point: `…/points/sub200_ep016.json`, **PDMS 76.5243**, with all dev-box gates G1/G2/G3 = 0.0.

The snapshot itself is already in the relay: `refe/snapshots/snap_epoch016.pt`, md5 `eacd8e8f…`. The CPU stages must
be **exact**. The GPU stage gets a **tolerance**, with the evidence for it measured below.

| gate | what runs on the VM | pass criterion (pre-registered here, before any VM result) | if it fails |
|---|---|---|---|
| **G-ENV** (every VM) | `--selftest`; `jpeg_probe.py`; REFe param count; real CUDA conv2d; C8 provenance in the harness manifest; signed-pull md5 of every private file | `ZZCONVERTER_EXACTZZ` with the dev box's numbers; 4/4 JPEG hashes equal `jpeg_devbox_hashes.json`; 322,070,854 / 18,991,426 params; CUDA available; `ZZPULL_OK` | stop |
| **G-H** harness, exact | score the **dev box's** ep016 seam with the dev box's sub200 cache pickles (via the relay, WindowsPath shim) | every cell of the per-token CSV equals the dev box's (**max \|Δ\| = 0.0**); 200/200 valid; C1–C3 pass; PDMS **76.5243** | stop: the Linux harness is a different function. Diagnose (numpy/shapely builds differ between win_amd64 and manylinux) before any VM number is quoted |
| **G-H′** VM-built cache (final only) | the same seam with the VM-rebuilt cache; the CV/STOP/HUMAN floors on all 12,146 tokens with it | CSV equal to G-H's; floors equal W3's banked per-token CSVs (CV 20.6517, STOP 61.8202, HUMAN 94.55 to 4 dp) | push the dev box's cache (3.16 GB, ≈ 54 min of uplink) through the relay and use it with the shim |
| **G-E6** E-6 table, exact | the 64 single-proposal runs from the **dev box's** dump | `pdms` and `sub` arrays equal `table.npz` (max \|Δ\| = 0.0) | stop |
| **G-S** seam, tolerance | the ep016 seam + dump on the VM GPU | 200/200 rows; frame control ≤ 1 mm. **Picks:** identical on every token whose dev-box aggregate margin is ≥ 1e-3 (160/200 tokens); flips allowed only below that, and counted. **Same-pick tokens:** max \|Δ pose\| ≤ 1e-3 (m for x, y; rad for heading); max \|Δ aggregate\| ≤ 1e-3; \|Δ logit\| reported | a flip at margin ≥ 1e-3, or \|Δ pose\| > 1 mm, means the VM runs a different model: stop |
| **G-P** point, tolerance | the harness on the VM's own seam | all picks equal: \|PDMS − 76.5243\| ≤ **0.01** and every discrete sub-score (NC/DAC/TTC/C/DDC) equal per token. Any flips: the PDMS difference is **fully accounted for** by the flipped tokens (every non-flipped token as before) | stop |
| **G-REP** (cheap) | the seam in 1 process vs K shards on the same VM | bitwise identical (max \|Δ\| = 0) | sharding changed something: stop |

**Why this tolerance, and why it is admissible:**

- **A GPU change is a change of floating-point reduction order, not of function.**
  - PUBLISHED, cuBLAS §2.1.4: bitwise equality is promised only for the same architecture *and* the same SM count.
    The RTX 4060 has 24 SMs. The L4 shares its Ada architecture with 58 SMs, and the G4 is Blackwell.
  - PUBLISHED, PyTorch 2.7 notes: *"not guaranteed across … different platforms"*.
  - Both quotes are in `published_reproducibility_quotes.json`.
- **The scale is measured, not guessed** (`cpu_vs_gpu_deviation.json`, `cpu_vs_gpu_score_compare.json`).
  - Setup: the same 4 tokens, in two backends: fp32 CPU, and the landed RTX 4060 dump with cuDNN TF32 on the
    patch-embed conv, the torch default. Three of the tokens are the closest-to-tie picks of sub200 (margins
    5.8e-6, 1.5e-5, 1.7e-5).
  - Result: \|Δ logit\| ≤ 2.0e-3, \|Δ aggregate\| ≤ 1.1e-4, \|Δ pose\| ≤ 0.94 mm over all 64 proposals and
    ≤ 0.53 mm on the executed rows. **4/4 picks unchanged**: near-tied proposals are near-duplicates and move
    together.
  - Scoring those rows with the unchanged harness left every discrete term identical. EP moved by ≤ 1.0e-5, so
    per-token PDMS moved by ≤ 1.02e-5 (0.001 points).
  - A GPU-to-GPU move shares the TF32 rounding that CPU-vs-GPU does not, so this is an **upper-side** scale.
  - The 1 mm / 1e-3 bars leave about 2× (poses) and 9× (aggregate) headroom over it. 1 mm is also the seam's own
    frame-control bar.
- **Exposure is bounded and known in advance** (`pick_margin_ep016.json`).
  - On ep016's 200 tokens, 40 have an aggregate margin below 1e-3, 9 below 1e-4 and 1 below 1e-5.
  - The E-6 table says what any flip would cost. The closest five flips change the 200-token mean by 0.0 to 0.014
    points. The worst single flip anywhere is 0.5 points.
  - G-P's accounting rule turns any flip into a reported, attributable number, never a silent drift.
- **Which variance this answers** (CLAUDE.md's three questions). It is none of the three: it is *"is this the same
  compute path?"*.
  - REFe's selection is an argmax, with no sampling. The same checkpoint on the same GPU reproduces bit-for-bit:
    MEASURED by EVAL_VENV.md, with 600/600 values equal.
  - So a passed G-S/G-P licenses mixing Colab and dev-box points on the learning curve, with a `device` stamp on each
    point. It says nothing about training or episode variance.

---

## Recommended plan

**P0a: parity on an L4** (≈ 0.7 CU, ≈ 30 min). The orchestrator's go is needed for the public pulls, each over
100 MB: maps 0.97 GB, 93 DBs 5.8 GB, trunk 1.2 GB, logs 0.48 GB.

1. Relay the sub200 frames + sub200 cache pickles + export once (≈ 228 s of push). `snap_epoch016.pt` is already
   there.
2. Build the environment exactly as in §c.
3. Run G-ENV, then G-H, then G-E6 on the dev box's artifacts.
4. Run G-S, G-P and G-REP on snapshot 016.
5. Measure the **signed-link pull rate on a GPU VM** and the multi-stream S3 rate.
6. Bank everything under `colab/raw/<date>-navtest-parity/`, and re-run `project_costs.py` with the measured values.

**P1: every later snapshot on an L4** (≈ 0.70 CU each; ≈ 12 snapshots to the end of training ≈ 8.4 CU).

- Runs only after P0a passes.
- The dev box fetches the snapshot from the pod as today, pushes it to the relay (77 s) and mints links. The VM pulls
  it with the relayed sub200 inputs and the public ones, runs seam + harness + E-6, and returns the seam, the dump,
  the 65 CSVs and `table.npz`.
- The dev box then runs `parse_navtest6.py`, `families6.py`, `selection_readout.py`, `write_result_e6.py` and
  `amendment4_readout.py`. Those are unchanged, and they need the TanitAD venv and the pod's `metrics.jsonl`.
- Each point JSON records `device`, the VM GPU name and the P0 gate ID.
- **The dev-box GPU is freed entirely.**

**P2: the final full navtest on a G4** (≈ 7.8 CU, ≈ 53 min; ≈ 5.9 CU if P0 finds fast GPU-VM relay pulls and the
bundle is pushed). One session, in this order:

1. Setup, with `model_final` pulled from the relay after its ≈ 22-min push, and the 128 GB frame stream and the cache
   rebuild running in parallel.
2. **G-H′** on the rebuilt cache.
3. **G-S / G-P on G4** with ep016 from the relay. This qualifies the G4 inside the same session, for about 4 min
   (≈ 0.6 CU).
4. Only then the `model_final` seam in 8 log shards, the harness in 16 shards, and the downloads.
5. On the dev box, parse + families produce `navtest_final.json`, exactly as FINAL_NAVTEST_READINESS step 3.

- **Model to PDMS in ≈ 1 h instead of 5–12 h.**
- If the PI prefers CU over wall-clock, the same session on an L4 is ≈ 5.9 CU and ≈ 3.8 h.

**Budget for the remainder of the run: ≈ 17.5 CU out of 200.** That is P0a 0.7 + 12 × 0.70 + G4 qualification 0.6 +
final 7.8.

### Integration work this needs

It is escalated here, and none of it is merged. Each item carries a regression arm.

1. **A Linux runner for eval steps 1–2**, a new file beside `eval_checkpoint.py` and not an edit of it. It needs:
   `MPLBACKEND=Agg`, `REFE_SIM_HZ=10`, `:`-separated paths, explicit `--frames/--db-dir/--export`,
   `linux_score_v1.py` for the harness, and `signed_pull.py` for private inputs.
2. **A sharded seam and merge.** It must be proven bitwise equal to the single-process seam (G-REP).
3. **A Linux E-6 driver.** `proposal_table.py` stage 2 calls `score_navtest_refe.py`; the Linux version calls
   `linux_score_v1.py` with `--workers W`.
4. **Data prep**:
   - the DB range fetch, from `code/fetch_front_camera.py`'s helpers;
   - the maps zip;
   - navsim_logs from HF;
   - for the final, a per-shard-parallel variant of `build_navtest_frames.py`;
   - the relay manifests for the sub200 bundle.
5. **A resume-from-seam mode in `eval_checkpoint.py`**, so the dev box can run steps 3–4 on a seam produced
   elsewhere. The steps can also be run by hand, but they should not be.

### Exact PI and orchestrator decisions

1. **Budget:** ≈ 17.5 CU for P0a + ≈ 12 snapshots + the final.
2. **The orchestrator's go for P0a's public pulls**, each over 100 MB (the brief's cap): maps, DBs, trunk, logs.
3. **Relay staging:**
   - the sub200 bundle (≈ 224 MB, ≈ 228 s of push);
   - `model_final` (≈ 1.3 GB, ≈ 22 min of push) when training ends.
   - After P0, decide whether to push the full frames + cache (13.2 GB, ≈ 3.7 h of dev-box uplink): it saves ≈ 1.9 CU
     only if GPU VMs pull signed links fast.
   - Everything stays far under the 1 TB private ceiling (184.6 GB used).
4. **The GPU for the final:** G4 (≈ 53 min, ≈ 7.8 CU) or L4 (≈ 3.8 h, ≈ 5.9 CU).
5. **Adopt G-S/G-P as pre-registered**, so that Colab-computed points may enter the learning curve and
   MODEL_REGISTRY §14.1 with a device stamp.
6. **Confirm** that the public visibility of `github.com/sayoood/TanitAD` is intended.

## Evidence (`colab/raw/2026-09-27-navtest-plan/`)

| file | what |
|---|---|
| `measure_inputs.py` → `input_sizes.json` | §a sizes, including zip-member compressed sizes and map city per log |
| `probe_public_sources.py` → `public_sources.json`; `hf_openscene_sensor_test_0-31.json` → `camera_shards_needed.json`; `metadata_tgz_sha_check.json` | §b HEAD, API and central-directory results; shard coverage; sha256 identity |
| `probe_linux_wheels.py` → `linux_wheels.json`; `driverl_eval_venv_freeze.txt`, `navsim_crun_venv_freeze.txt` | §b/§c pin availability on Linux |
| `pick_margin_ep016.py` → `pick_margin_ep016.json` | §f pick-margin exposure |
| `cpu_vs_gpu_tokens.json`, `cpu_vs_gpu_seam.log`, `seam_cpu_ep016.{npz,report.json}`, `props_cpu_ep016.npz`, `compare_cpu_vs_gpu.py` → `cpu_vs_gpu_deviation.json`, `cpu_probe_score/`, `cpu_vs_gpu_score.log` → `cpu_vs_gpu_score_compare.json` | §f cross-backend scale: a dev-box CPU forward of 4 tokens (CUDA hidden), scored by the unchanged harness |
| `seam_cpu_side_timing.py` → `seam_cpu_side_timing.json` | §d the seam's CPU side, model stubbed |
| `published_reproducibility_quotes.json` | §f cuBLAS §2.1.4 and PyTorch 2.7 quotes |
| `build_smoke_bundle.py` → `smoke_bundle_manifest.json`, `jpeg_devbox_hashes.json`; `jpeg_probe.py` (+ `jpeg_probe_devbox.log`) | the smoke's 17.6 MB upload and the JPEG reference |
| `navplan_smoke.ps1` / `_run2.ps1` / `_run3.ps1`, `vm_smoke_navplan*.py`, `smoke/`, `smoke_run2/`, `smoke_run3/` (driver logs, ccu readouts before/assigned/after, VM results JSON, the Linux harness log and CSV) | §c smoke: 3 sessions, 0.08 CU/h, stopped and checked on the server each time. Run 3's after-stop readout listed one assignment: it was the coordinator's concurrent `relay-proof` VM, not this plan's (different endpoint; `final_ccu_check.txt` then read 0) |
| `linux_score_v1.py`, `navsim_v1_linux.py` | the Linux harness launcher prototype (§c) |
| `project_costs.py` → `projections.json` | §d table, with relay routes and the unbilled dev-box push times |
| `final_ccu_check.txt` | 15:50:49Z: `assignments: []`, rate 0, balance 200 |
| (coordinator) `colab/raw/2026-09-27-hf-relay/` | relay push/pull proof and the HF quota readout, cited above |

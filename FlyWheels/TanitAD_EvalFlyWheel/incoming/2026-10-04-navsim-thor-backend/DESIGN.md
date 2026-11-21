# NavSim CPU scoring backend on Thor — DESIGN (EvalFlyWheel, 2026-10-04)

**Why.** The dev box (31.8 GB) is RAM-bound for NavSim scoring: the step-50,400 navtest scorers
were RAM-guard-aborted repeatedly (MEASURED: `…/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/runner_50400b.out.txt`,
`ram_guard_abort=True` on R7_A1 / R7_A1_s1 / R7_CEILDECL_d), and D6 P1' fan scoring is held to 2
processes. Thor's GPU is busy (one job at a time; GPU sharing MEASURED net-negative), but at 22:06 local
13 of 14 cores were idle and ~50 GB MemAvailable (Master Mind measurement, INHERITED).
This backend moves **CPU-only** NavSim scoring to Thor without touching its GPU.

## Pre-registered acceptance bar (written before any Thor score was read)

| control | reference (dev box, banked) | Thor run | bar |
|---|---|---|---|
| **V-NH** navhard two-stage | `step50400/scores_navhard/score_R7_A1__navhard_two_stage_wrapper/R7_A1__navhard_two_stage_final_scores_frame.csv` (5,912 tokens, PASS) | same seam (sha256 `996411cd…`), same Hydra overrides + a stage-1 token filter selecting **9 whole two-stage groups = 204 tokens** (`raw/navhard_subset_200.json`, rule fixed in `code/select_navhard_subset.py`) | **max abs diff 0.0 AND zero text-differing cells on EVERY per-token column** of the final frame (and of the devkit CSV) |
| **V-NT** navtest v1.1 | `step30000/scores_navtest/r7s30000_R7_A1/r7s30000_R7_A1.csv` (12,146 tokens, PASS) | same seam — proven the scored one by comparing its rows with the dev-box hook's recorded agent poses, 200/200 bit-identical (`raw/navtest_val_seam_vs_hooks_s30000_R7_A1.json`) — on W3's own paired subset `A1_sub200_tokens.json` (200 tokens, chosen 2026-09-19 by W3, not by me) | same |

Why whole groups on navhard: `weight`, `two_frame_extended_comfort` and `score` come from the
`SceneAggregator` over a group; cutting a group would change them by construction. Why the
comparator also checks CSV **text**: pandas writes the shortest round-trip repr, so equal text ⇔
bit-identical float64 — a 1-ulp drift shows up even where `abs diff` would print `0.0` after rounding.
**If the bar is missed, production does not start**: the differing columns, magnitudes and a
diagnosis are reported instead.

## What runs where

| piece | Thor location | provenance |
|---|---|---|
| Python 3.9.25 (python-build-standalone, aarch64) | `/home/nvidia/venvs/navsim-cpu-python` (disk) | uv 0.12.1 |
| venv `navsim-cpu` (NEW; `tanitad-train` / `tanitad-edge` untouched) | `/home/nvidia/venvs/navsim-cpu` (disk, ~0.98 GB) | every package pinned to the dev-box venv's version (`raw/thor_env.json`), incl. **CPU torch 2.0.1** (its own venv; no shared torch touched) |
| navsim v2 devkit (`0a380a9` copy), nuplan-devkit (`ce3c323` copy), navsim v1.1 (`3e8291b`) | `/home/nvidia/navsim-thor/src/…` (disk) | **content tree-hash identical** to the dev-box trees (MEASURED, `raw/thor_env.json`) |
| E1 / E2 / W3 code (wrapper, seam agent, W3 driver + agents) | `/home/nvidia/navsim-thor/code/{e1,e2,w3}` | byte-identical (sha256 per file recorded) |
| Thor ports (this package's `code/`) | `/home/nvidia/navsim-thor/code/thor` | minimal diffs, every changed line marked `# THOR` |
| data: metric caches, navsim logs, synthetic scenes, maps (`.gpkg` + drivable-area npz) | `/dev/shm/navsim/…` (tmpfs, ≤ 7 GB at any time) | tree hashes identical to the dev-box sources (MEASURED) |

**Not copied:** `sensor_blobs` (12 GB). The seam agents declare `SensorConfig.build_no_sensors()`;
the file-open audit (`code/audit_site/sitecustomize.py`, `raw/opened_files_audit_navhard.json`) settles it: 0 sensor_blobs and 0 `.npz` opened; the scorer opens the cache metadata + one pickle per token, ALL 5,462 synthetic pickles and all 76 navhard logs (loader init), and the 4 map `.gpkg` + `nuplan-maps-v1.0.json`.

## The port — what had to change and why (each is a Windows artefact, none is a scoring change)

1. **Metric-cache pickles carry `pathlib.WindowsPath`** (`MetricCache.file_path`) — POSIX cannot
   instantiate it, so no cache file unpickles. `thor_compat` maps it to a `PureWindowsPath` subclass
   *at unpickle time only*. `file_path` is metadata; nothing scores from it.
2. **`MapParameters.map_root` is a Windows dir** (`C:/Users/Admin/navsim-crun/data/maps`) and the
   two-stage reactive IDM policy calls `get_maps_api(map_root, …)`. `thor_compat` substitutes
   `$NUPLAN_MAPS_ROOT` for any `X:/…` root (counted per run in the manifest). Map bytes identical.
3. **Metadata CSV rows are Windows absolute paths.** E1's loader patch already made the token
   split separator-agnostic; the Thor variant additionally maps each row onto the local copy of the
   same cache directory (`thor_compat.remap_cache_path`).
4. **`rasterio` has no cp39/aarch64 wheel** (uv: only a cp38 dev build). nuplan imports it at module
   level but only its RASTER-layer helpers call it; NavSim scoring reads vector layers via pyogrio.
   An import-only stub (`code/stubs/rasterio`) RAISES on any call, so a raster use would fail loud
   (invalid row ⇒ count guards refuse).
5. `CUDA_VISIBLE_DEVICES=""` (POSIX keeps an empty var; Windows needed `-1`).

### 6. The one difference that is NOT a Windows artefact: libm rounding in the AgentInput (found by V-NH run 1)

E2's / W3's seam agents refuse a token unless the SHA-1 of the AgentInput's float64 **bytes** equals
the fingerprint recorded at export on the dev box. On Thor that refused **6 of 204** navhard tokens in
the first validation run (all synthetic stage-two scenes). Bit-level diagnosis
(`code/diag_agent_input.py`, `raw/diag_agent_input_{devbox,thor}.json`): `np.cos(theta)` and
`normalize_angle = arctan2(sin, cos)` round differently in the Windows CRT and in glibc/aarch64 for a
few inputs — **1 ulp** — which the relative transform then carries into x/y (≤ 27 ulp). Over the whole
split (`code/fp_audit.py`): **481 / 5,912 navhard tokens (8.1 %)** differ, **all synthetic**, max
|Δ| **3.6e-15** (`raw/ulp_reference_check_navhard.json`). The fingerprint tables of 4 seams (3 arms
at 50,400 + one at 30,000) agree, i.e. the AgentInput depends only on the scene, never on the arm.

For a seam arm the AgentInput is **not a scoring input** — the agent returns the precomputed plan; the
fingerprint proves the row belongs to the token. **Candidate fix (NOT the default, needs Master Mind
acceptance):** `code/tanitad_seam_agent_ulp.py` / `code/w3_agents_v1_ulp.py` keep E2/W3's byte check
and, only when it fails, accept the row if a **dev-box reference** of that token's AgentInput (exact
float64 hex, `code/make_ulp_reference.py` run on the dev box, 6.6 s for navhard's 481) **re-hashes to the
seam's fingerprint** (so it *is* the exported input) **and** Thor's AgentInput is within **atol 1e-9**
of it element-wise. ⚠ The atol was chosen **after** seeing the 3.6e-15 measurement (≈2.8e5× margin);
the ulp form was dropped because ulp counts explode near zero (max 4,194,304 ulp on a value of ~1e-9).
Every waiver is logged per token (`fp_mode: "reference"`, measured `max_abs`, `max_ulp`). One
reference per split serves every arm and every checkpoint step.

## Safety on a box that is training

* Never the GPU (`CUDA_VISIBLE_DEVICES=""`, CPU torch); never `/home/nvidia/refcv7_post/thor_gpu.lock`.
* Every scorer under `nice -n 19 ionice -c3` (inherited by children via `setsid`).
* `code/thor_pool.py`: ≤ 4 scorers; launch only if MemAvailable ≥ 35 GB (and 45 s settle between
  launches); watchdog SIGTERMs **only the PIDs it started and their `/proc/*/task/*/children`
  descendants** if MemAvailable < 20 GB or `/home/nvidia/NAVSIM_YIELD` exists; opaque marker
  `ZZ<done>-<fail>-<alive>-<queued>-<availGB>ZZ`. No `pgrep`/`pkill`, no process-table patterns.
* Disk: ≤ 1.5 GB (venv + python + code); outputs live on tmpfs and are pulled to the dev box.
* Coordination: before any PRODUCTION arm, append `<split>:<arm> thor <UTC>` to
  `…/step50400/CLAIMED_ARMS.txt` and skip arms claimed by `devbox`. Validation runs need no claim.


_Pool limits tightened 2026-10-04 on the coordinator's instruction (was 6 / 30 GB / 15 GB) after another agent's process caused a global kernel OOM on Thor at 22:57:19._

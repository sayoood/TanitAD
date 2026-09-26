# EVAL_VENV — the `driverl-eval` interpreter for the REFe NAVSIM eval

**Interpreter:** `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe` (CPython 3.11.13, created by uv 0.12.5)
**Built:** 2026-09-26 09:09–09:11 Europe/Berlin (07:09–07:11 UTC) on the dev box, **fully offline** from the
uv cache (`--offline` on every uv call; no package was downloaded, no torch build was substituted).
**Replaces:** the damaged `…/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/bd7d00af-b98e-42f1-a53c-cb4113059b0f/scratchpad/driverl-venv`
(left untouched as evidence; it lives under `%TEMP%` and should be expected to disappear).

**Result (MEASURED 2026-09-26):** the banked check point `e2chk_ep002` (old venv, 2026-09-24) is reproduced
**exactly** by the new venv: seam poses max |Δ| = **0** (600/600 float32 values bytewise identical), `token`
arrays identical in order, PDMS **19.6473 = 19.6473** (full precision 0.19647299329340995 on both), and the
per-token score CSV is **byte-identical** (sha256 `3430fe1d…944c` on both).

## Use it

```bash
cd "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval"
REFE_DRIVERL_PY=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe \
  C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe eval_checkpoint.py --ckpt <snap.pt> --name <name> \
  --tokens <tokens.json> --frames <frames_dir>
```

⚠️ `eval_checkpoint.py` still **defaults** `DRIVERL_PY` to the dead `%TEMP%` venv, so `REFE_DRIVERL_PY` must
be set on every call until that default is changed. Ten more scripts hard-code the dead path (not env-overridable):
`refe/run_image_isolation.sh`, `code/finalize_mini.py`, `code/run_mini_teacher.sh`, `code/stage0_post.py`,
`code/run_headline_suite.sh`, `code/arrange_test_early.sh`, `code/arrange_watcher.sh`,
`code/fetch_scenario_cameras.sh`, `code/queue_camera_fetch.sh`, `raw/2026-09-21-review6/rank_instrument_attack.py`.
None was edited here (repointing them is an integration decision).

## 1. Why it was rebuilt

The old venv was created under `%TEMP%` (the session scratchpad) and was **partially deleted** — MEASURED from its
directory mtimes: 90 top-level entries of `site-packages` were modified 2026-09-24 **17:46:46–17:46:50** Berlin
(15:46 UTC), i.e. after the banked run (03:24–03:29 Berlin the same day). 3,675 files survive; `import torch`
fails with `ModuleNotFoundError: No module named 'typing_extensions'` (re-checked 2026-09-26 with `python -B`, so
no bytecode was written into it). What survived — `entry_points.txt` in most dist-infos, compiled `.pyc` files
whose `.py` source is gone (e.g. `__pycache__/typing_extensions.cpython-311.pyc`) — looks like a
last-access-age `%TEMP%` cleaner (INFERRED, mechanism not verified). The new venv lives outside `%TEMP%`.

## 2. Where the package set comes from (evidence, strongest first)

1. **`code/requirements_teacher_lock.txt`** was generated on 2026-09-23 05:42 UTC by `importlib.metadata`
   **from the old venv itself** (session transcript): 117 distributions = 113 pins + the excluded `torch`,
   `colorama` and the two editables. Running the **identical generator** against the new venv reproduces that
   file **byte-for-byte** (sha256 `892086644556e104541b6e741e189bdd123bac37f3993698a5cdcb8deff7ae69` on both).
2. **Hardlink identity.** uv installs by hardlinking the files of its unpacked cache archive
   (`cache/archive-v0/<id>/`). 491 of the old venv's surviving files still share an NTFS file id with a cache file;
   all 491 resolve to exactly one archive each — 50 distributions, every one at the version the lock names
   (incl. `torch-2.7.1+cu128` = archive `cDACEMiwg_25T-cv`, `shapely-2.1.2` = `VaVJypsDEnlf-rIc`,
   `pyogrio-0.13.0` = `MrNEKLoNdbpw0IYl`). The new venv installs **48/48** of the registry distributions among
   them from the **same archive**; the 2 others are the editable builds (rebuilt, see §6).
3. **The original commands**, recovered from the session transcript (2026-09-20, UTC; the two shim times are
   file mtimes): 08:16 `uv venv driverl-venv --python <PY311>` → 08:17
   `uv pip install --native-tls --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1` → ~08:22
   `uv pip install --native-tls -r requirements.txt -c constraints.txt` (DriveRL checkout, PyPI) → 08:23
   `-e .` and `-e nuplan-devkit`, both `--no-deps` → 08:25 the `fcntl.py` shim → 12:36 `sitecustomize.py` →
   13:10 `uv pip install --native-tls --no-deps huggingface_hub safetensors truststore`.
   The cache confirms the provenance split used below: the cu128-index pointers for torch's 8 dependencies were
   all written 2026-09-20 10:17:48–51 Berlin (the torch step) and none of their PyPI pointers during the
   requirements step, and `jinja2`/`networkx`/`sympy` are hardlink-matched to the **cu128-index** archives.
4. `colorama==0.4.6` (excluded from the lock as Windows-only): the version string in the surviving
   `colorama/__pycache__/__init__.cpython-311.pyc`; it is also the only colorama in the cache.

⚠️ **Correction to the task brief:** the brief listed `shapely-2.0.7`, `geopandas-1.0.1`, `pyogrio-0.11.1` as
cached wheels for this venv. Those are **cp39** wheels of the Python 3.9 NAVSIM venv
(`C:/Users/Admin/navsim-crun/venv` has exactly those three) and cannot install into 3.11. The old driverl venv
used **shapely 2.1.2 / geopandas 1.1.4 / pyogrio 0.13.0** (shapely + pyogrio by file-id identity; geopandas by
the lock and the surviving `geopandas/__pycache__/_version.cpython-311.pyc`, which holds `1.1.4`).

## 3. The recipe (exact commands, as run — Git Bash)

```bash
UV=/c/Users/Admin/AppData/Local/hermes/bin/uv                  # uv 0.12.5 (210d1f678 2026-08-14)
CACHE=C:/Users/Admin/AppData/Local/uv/cache
PY311=C:/Users/Admin/AppData/Roaming/uv/python/cpython-3.11-windows-x86_64-none/python.exe   # CPython 3.11.13
NEW=C:/Users/Admin/venvs/driverl-eval
VPY=$NEW/Scripts/python.exe
cd /c/Users/Admin/venvs                                        # neutral cwd: no pyproject/uv.toml above it

# 0. the venv (same pyvenv.cfg as the old one: home, uv = 0.12.5, version_info = 3.11)
$UV venv "$NEW" --python "$PY311" --offline --cache-dir "$CACHE"
# 1. torch + its 8 dependencies from the cu128 index, as originally (pins_cu128.txt, §4)
$UV pip install --offline --cache-dir "$CACHE" --python "$VPY" --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 -r pins_cu128.txt
# 2. everything else from PyPI (pins_pypi.txt, §4) -- --no-deps: nothing can re-resolve torch
$UV pip install --offline --cache-dir "$CACHE" --python "$VPY" --no-deps -r pins_pypi.txt
# 3. the two editable checkouts, --no-deps (built locally in an offline isolated build env)
$UV pip install --offline --cache-dir "$CACHE" --python "$VPY" --no-deps -e C:/Users/Admin/dz/DriveZero/DriveRL
$UV pip install --offline --cache-dir "$CACHE" --python "$VPY" --no-deps -e C:/Users/Admin/dz/DriveZero/DriveRL/nuplan-devkit
# 4. the Windows fcntl shim (nuplan's gpkg_mapsdb.py line 1 is `import fcntl`; without it the import fails
#    with ModuleNotFoundError -- MEASURED on this venv before the copy). As run: byte-copied from the damaged
#    venv's site-packages. For a future rebuild copy it from THIS venv, or write Appendix A with LF endings.
cp "<damaged venv>/Lib/site-packages/fcntl.py" "$NEW/Lib/site-packages/fcntl.py"
#    sha256 must be 5ccb9bbbd45c71c865baf272e3a086d3e72141b9e2c4bd611fa8e38cc4982cc1 (2,149 B)
```

uv output: step 1 `Resolved 9 packages … Installed 9 packages`, step 2 `Resolved 106 … Installed 106`, no
`Prepared`/download line in either; steps 3 each `Built … Installed 1 package`. Link mode is uv's Windows default
(**hardlink**): the venv's files are hardlinks into the uv cache (33,980 files right after the build, 1,908 of
them single-link, e.g. per-venv `RECORD`/`INSTALLER` and `fcntl.py`). Do not edit
files inside `site-packages` in place — that edits the cache copy too.

## 4. Pinned package list (what was installed — 117 distributions)

`pins_cu128.txt` (index `https://download.pytorch.org/whl/cu128`, 9):

```text
filelock==3.32.3
fsspec==2026.7.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
sympy==1.14.0
typing_extensions==4.16.0
torch==2.7.1+cu128
```

`pins_pypi.txt` (PyPI, 106 = the 113 lock pins minus the 8 above, plus `colorama==0.4.6`):

```text
absl-py==2.5.0
affine==3.0.1
aioboto3==15.5.0
aiobotocore==2.25.1
aiofiles==25.1.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aioitertools==0.13.0
aiosignal==1.4.0
annotated-doc==0.0.5
antlr4-python3-runtime==4.9.3
attrs==26.1.0
bokeh==2.4.3
boto3==1.40.61
botocore==1.40.61
cachetools==6.2.6
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.1
click==8.5.0
click-plugins==1.1.1.2
cligj==0.7.2
cloudpickle==3.1.2
contourpy==1.3.3
cycler==0.12.1
decorator==5.3.1
einops==0.8.2
Farama-Notifications==0.0.6
fonttools==4.65.0
frozenlist==1.8.0
geopandas==1.1.4
greenlet==3.5.6
grpcio==1.84.0
gymnasium==1.3.0
h11==0.16.0
huggingface_hub==1.32.0
hydra-core==1.3.2
idna==3.20
jmespath==1.1.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.1
Markdown==3.10.3
markdown-it-py==4.2.0
matplotlib==3.11.2
mdurl==0.1.2
msgpack==1.2.2
multidict==6.9.0
nest-asyncio==1.6.0
numpy==1.26.4
omegaconf==2.3.0
opencv-python-headless==4.11.0.86
outcome==1.3.0.post0
packaging==26.3
pandas==2.3.3
pillow==11.3.0
propcache==0.5.4
protobuf==7.36.2
psutil==7.2.2
py==1.11.0
pyarrow==25.0.1
pycparser==3.0
Pygments==2.21.0
pyogrio==0.13.0
pyparsing==3.3.2
pyproj==3.7.2
pyquaternion==0.9.9
PySocks==1.7.1
python-dateutil==2.9.0.post0
pytz==2026.3.post1
PyYAML==6.0.3
rasterio==1.4.4
ray==2.51.1
referencing==0.37.0
requests==2.34.2
retry==0.9.2
rich==15.0.0
rpds-py==2026.6.3
s3transfer==0.14.0
safetensors==0.8.0
scipy==1.17.1
selenium==4.49.0
setuptools==84.0.0
shapely==2.1.2
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
sortedcontainers==2.4.0
SQLAlchemy==2.0.54
tensorboard==2.21.0
tensorboard-data-server==0.7.2
tornado==6.5.10
tqdm==4.70.1
trio==0.34.0
trio-websocket==0.12.2
truststore==0.10.4
typer==0.27.2
tzdata==2026.4
ujson==6.0.0
urllib3==2.8.0
websocket-client==1.9.2
Werkzeug==3.1.8
wrapt==1.17.3
wsproto==1.3.2
yarl==1.25.1
colorama==0.4.6
```

Editable (`--no-deps`): `driverl==0.1.0` ← `C:/Users/Admin/dz/DriveZero/DriveRL` (`.pth` → `…\DriveRL\src`) and
`nuplan-devkit==1.2.2` ← `C:/Users/Admin/dz/DriveZero/DriveRL/nuplan-devkit` (meta-path finder). Plus one plain
file, `site-packages/fcntl.py` (Appendix A). `antlr4-python3-runtime` is sdist-only on PyPI; uv used its cached
build (`cache/sdists-v9/pypi/antlr4-python3-runtime/4.9.3`).

## 5. Verification (2026-09-26, all PASS)

| # | check | result |
|---|---|---|
| 1 | torch + REAL CUDA conv2d | `torch 2.7.1+cu128`, CUDA build 12.8, `is_available()` True, NVIDIA GeForce RTX 4060; `F.conv2d` (2,3,64,64)⊛(8,3,3,3) → (2,8,64,64), max \|GPU − CPU fp64\| = 5.16e-06 → `V1_CUDA_CONV2D_OK` |
| 2 | imports (PYTHONPATH **unset**, module files printed) | `NuPlanScenario` OK, nuplan ← `…\DriveRL\nuplan-devkit\nuplan`; driverl ← `…\DriveRL\src\driverl`; shapely 2.1.2, geopandas 1.1.4, pyogrio 0.13.0, hydra 1.3.2, omegaconf 2.3.0, numpy 1.26.4, pandas 2.3.3 (+ cv2 4.11.0, safetensors 0.8.0, yaml 6.0.3, einops 0.8.2, gymnasium 1.3.0, scipy 1.17.1, pyquaternion, fcntl shim), all from the new venv → `V2_IMPORTS_OK` |
| 3 | `python eval/refe_navtest_seam.py --selftest` | line 0.00e+00 m; arcs R=20/60/8 within sagitta (2.499e-02 / 1.875e-02 / 9.998e-03 m); heading across ±π 8.9e-16 rad → **`ZZCONVERTER_EXACTZZ`** |
| 4 | end-to-end `eval_checkpoint.py --name e2chk_newvenv` (ckpt `snap_epoch002.pt`, `e2_tokens.json`, `frames_e2`), run with the new python and `REFE_DRIVERL_PY` = new python | **`ZZPOINT_OK e2chk_newvenv PDMS=19.6473 n=25`**, started 09:13:29 Berlin, 137.0 s |
| 4a | seam `refe_e2chk_newvenv.npz` vs banked `refe_e2chk_ep002.npz` | `poses` (25,8,3) float32: **max \|Δ\| = 0**, 0/600 elements differ; `token` identical (same order); `fingerprint`, `sampling` identical; only `arm` differs (the run name, by construction) |
| 4b | seam report | frame control max 7.32e-15 m (n=25), interpolation residual max 0.12305 m, proposals (ADE selected 3.1612 / random 3.1597 / oracle 3.1148 m, M=64): all identical to banked |
| 4c | PDMS / terms | new = banked: NC 42.0, DAC 84.0, EP 25.5535, TTC 16.0, C 64.0, DDC 92.0, **PDMS 19.6473**; score CSV byte-identical (1,723 B); `C1_max_abs_delta` 0.0; 25/25 successful |
| 4d | steps 3–4 (TanitAD venv, unchanged) | `4_families.json` identical to banked (labels normalised); `3_parse.json`: CV/STOP/HUMAN/refcv4b floors identical — differences only (i) the banked run's extra `--prev e2_ep001_log1` arm, (ii) a new per-arm `device` key from `parse_navtest6.py` edited 2026-09-24 03:53, after the banked run |
| + | lock regeneration | byte-identical to `code/requirements_teacher_lock.txt` (117 distributions) |
| + | negative control | the damaged venv: `python -B -c "import torch"` → `ModuleNotFoundError: typing_extensions`, so the seam can only have come from the new venv |

Artifacts of the reproduction run (not in the repo): `D:/Projects/TanitAD/data/refe_navtest/seams/refe_e2chk_newvenv.{npz,report.json}`,
`…/points/e2chk_newvenv.json` + `…/points/e2chk_newvenv/`, `…/score/refe_e2chk_newvenv/`, and W3's run dir
`D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/runs/refe_e2chk_newvenv`.

## 6. Not restored, differences, caveats

* **`sitecustomize.py` NOT installed** — not needed here: it is a no-op unless `DRIVERL_EVAL_ROUTE_LANE_RANK` is
  set, which `eval_checkpoint.py` never sets. It IS needed for the Stage-1 route-override runs through DriveZero's
  harness (`code/run_mini_teacher.sh` with a non-zero rank), which spawns its own python. Its only other copy is in
  the doomed `%TEMP%` venv, so its exact text is preserved in Appendix B.
* **`uv pip check`: 2 incompatibilities**, same as the old venv by construction — `huggingface-hub` wants `hf-xet`
  and `httpx`; it was installed `--no-deps` on 2026-09-20 and neither is among the 117 distributions.
  `import huggingface_hub` still works (1.32.0).
* **Editables were rebuilt**, not reused: new cache archives (`lebtkO0DahyWSUwx`, `7uiDIwD6m6dkVZg3`), but the
  installed `__editable__.driverl-0.1.0.pth`, `__editable__.nuplan_devkit-1.2.2.pth` and
  `__editable___nuplan_devkit_1_2_2_finder.py` are byte-identical to the old ones, as are `_virtualenv.pth`,
  `_virtualenv.py`, `distutils-precedence.pth`, `_distutils_hack/__init__.py`. Side effect: the build regenerated
  the checkout's `src/driverl.egg-info` and `nuplan-devkit/nuplan_devkit.egg-info` (timestamps;
  `nuplan_devkit.egg-info/SOURCES.txt` +76 B = the one untracked file added since 2026-09-20,
  `nuplan/planning/script/config/common/scenario_filter/driverl_mini_bank.yaml`). No source file changed; the
  DriveRL checkout is otherwise clean (`git status`).
* The old venv's `Scripts/` held only `python.exe`/`pythonw.exe` after the deletion; the new one has uv's normal
  activation scripts and entry points — nothing in the eval uses them.
* Reproduction was exact on this box, this GPU, this checkpoint; wall time differed (137.0 s vs 280.0 s banked;
  seam 42.8 s vs 104.9 s), which is not a correctness signal.

## Appendix A — `site-packages/fcntl.py` (installed; LF endings, 2,149 B, sha256 `5ccb9bbb…82cc1`)

```python
"""Windows-only shim for the Unix `fcntl` module, for the DriveRL/nuplan-devkit venv.

WHY THIS EXISTS. nuplan-devkit imports `fcntl` in exactly one file,
`nuplan/database/maps_db/gpkg_mapsdb.py`, and uses it for exactly one thing: an advisory
`flock(fd, LOCK_EX)` / `flock(fd, LOCK_UN)` around map-DB access, i.e. a guard against
CONCURRENT PROCESSES racing on the same map file. `fcntl` does not exist on Windows, so the
import fails before any of that code runs (PREFLIGHT_ERROR simulation_import, 2026-09-20).

WHAT IT DOES. `flock` is a NO-OP that warns once. That is safe for our use -- a single
process, maps already present on disk at NUPLAN_MAPS_ROOT -- and it is NOT safe for
multi-process map DOWNLOADS into one directory. If that case ever arises, run one process
first to populate the maps, or replace this with a real `msvcrt.locking`-based lock.

WHAT IT REFUSES. On any non-Windows platform it raises ImportError so it can never shadow
the real stdlib module if this venv is ever copied somewhere it does not belong.
"""
import sys
import warnings

if sys.platform != "win32":
    raise ImportError(
        "fcntl.py in this venv is a WINDOWS-ONLY shim; on this platform the real stdlib "
        "fcntl must be used -- delete this file."
    )

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8

_warned = False


def flock(fd, operation):  # noqa: D401 - mirrors the stdlib signature
    """No-op advisory lock. Warns once so nobody mistakes it for a real lock."""
    global _warned
    if not _warned:
        warnings.warn(
            "fcntl.flock is a NO-OP on this Windows box (DriveRL shim). Safe for a single "
            "process with maps already on disk; NOT safe for concurrent map downloads.",
            RuntimeWarning,
            stacklevel=2,
        )
        _warned = True
    return None


def lockf(fd, operation, length=0, start=0, whence=0):
    return flock(fd, operation)


def fcntl(fd, cmd, arg=0):
    raise OSError("fcntl.fcntl is not available on Windows (DriveRL shim)")


def ioctl(fd, request, arg=0, mutate_flag=True):
    raise OSError("fcntl.ioctl is not available on Windows (DriveRL shim)")
```

## Appendix B — `site-packages/sitecustomize.py` (NOT installed; only for route-override runs; LF, 835 B, sha256 `38170d61…450f9`)

```python
"""Venv-wide startup hook. NO-OP unless DRIVERL_EVAL_ROUTE_LANE_RANK is set to a non-zero value.

Exists because DriveZero's harness spawns its own python process, so a Stage-1 route override has no
other injection seam. Deliberately silent and inert by default: every analysis script in this venv
also imports it.
"""
import os

_r = os.environ.get("DRIVERL_EVAL_ROUTE_LANE_RANK", "").strip()
if _r and _r != "0":
    import sys
    sys.path.insert(0, r"D:\Projects\TanitAD\TanitAD Research Lab\Architecture & Inference\Research\2026-09-20-refe-plan\code")
    try:
        import route_lane_rank_patch
        route_lane_rank_patch.install()
    except Exception as exc:                      # never break the interpreter over this
        print(f"[sitecustomize] route patch NOT installed: {type(exc).__name__}: {exc}", flush=True)
```

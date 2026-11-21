# BUNDLE_PMAP — running the registered P-MAP arms on any CUDA box

*WP-D, 2026-10-05. The Master Mind asked for this so that P-MAP can start the moment other compute is approved. The
arms, bars and must-fails are the registered ones (`PREREG_WPD_PROBES.md` §P-MAP, sha256 c054190b…). Nothing here
changes the science. The code changes since registration are portability and logging only, listed in §4.*

## 1. What is in the bundle

On Thor: `/home/nvidia/refcv8_wpd/bundle_pmap/`, 155 MB. This is a single copy; it moves by file ship, md5 verified.
The md5s are in `raw/pmap_bundle_manifests/BUNDLE.md5`.

| item | md5 | role |
|---|---|---|
| `code/pmap_arms.py` | `5c17c58d…` | the 9 arms; `--timing-steps N` for a timing-only run |
| `code/pmap_analyze.py` | `ce36fd03…` | bars, must-fails, paired bootstrap (CPU) |
| `code/diag_metrics.py` | `34982975…` | the diagnostics' metric definitions (`DIAG_CODE`) |
| `code/run_pmap_chain3.sh`, `code/run_pmap_timing.sh` | — | the Thor runners: the lock, ≤ 40-min jobs, resumable |
| `launch_tree_fec3a0dccf.tar.gz` | `05c1aef3…` | the refcv7 launch tree at fec3a0d (`stack/`, `taniteval/`, `tools/`); no `.git`, no pycache |

## 2. What the bundle needs beside it (the DATA): ~397 GB, of which the TRAIN cache is 364 GB

The md5s are in `raw/pmap_bundle_manifests/{RUN.md5, DATA_SMALL.md5, DATA_LARGE.txt}`. The layout is the loader's KIT
layout: set `REFCV6_KIT=<root>`, and every path below is `<root>/<path>`. `refcv7_loader.PATH_REMAP` maps each
run-config flag onto it.

| path under KIT | size | md5 / check | read by |
|---|---|---|---|
| `refcv7_run/runs/refcv7-r101-s0/ckpt.pt` + `config.json` | 1.1 GB | `d5f104ee…` / `e6512a01…` | `L.build_model(strict=True)`; `--run-dir` / `WPD_RUN_DIR` |
| `data/refcv6-b1-416x1024-train/` (4,372 files) | **364.3 GB** | `_v2manifest.pt` md5 `3c9f8bc8…`; per-file md5 AT TRANSFER | TRAIN draws (3,000 × 8 windows, with replacement, over all 4,369 clips) |
| `data/refcv6-b1-416x1024-eval139/` (141 files) | 11.9 GB | `_v2manifest.pt` `34433232…` | EVAL-DIAG |
| `refcv7_post/diag/cache/train_diag139/` | 139 SYMLINKS into the train cache + manifest | `_v2manifest.pt` `ad5c9c34…` | TRAIN-DIAG threshold fit. Rebuild the symlinks on the target; there are no extra bytes |
| `data/sam3_gt_v3/` (4,509 `.sam3mapgt.npz`) | 20.0 GB | count + bytes | the 10 cm map GT |
| 12 small files (anchors, v8 labels, speed-max sidecars, agent joins, extrinsics, clock sidecar, vis1 sidecar, nav tau, map class weights) | 0.93 GB | `DATA_SMALL.md5` | the loader's dataset build |

⚠️ **The 364 GB TRAIN cache is the whole transfer problem.** Price the transfer before choosing a box. From Thor:
- **pod direct SSH**, MEASURED 42 MB/s on another pod pair (C56): ~2.4 h;
- **HF relay**, ~118 MB/s when quota allows: ~1 h each way;
- **the dev-box uplink** is 1.2 MB/s, which makes it unusable for this.

Check first whether the target already holds the corpus. The refcv6-B1 corpus was built for pod training, so the A40
line may have a copy; verify by `_v2manifest.pt` md5, not by name.

A **600-clip TRAIN subset** (~50 GB, P-BOX's `default_rng(0)` rule, `pbox_prep.py`) would cut the transfer 7×. But
P-MAP is registered on the full TRAIN cache, so a subset needs a registered amendment and is NOT assumed here.

## 3. Run recipe on a non-Thor box

```
tar -xzf launch_tree_fec3a0dccf.tar.gz -C <KIT>/refcv7_run/            # -> <KIT>/refcv7_run/fec3a0dccf
export REFCV6_REPO=<KIT>/refcv7_run/fec3a0dccf REFCV6_KIT=<KIT> PYTHONPATH=<KIT>/refcv7_run/fec3a0dccf/stack
export DIAG_CODE=<bundle>/code WPD_DATA=<KIT>/data WPD_RUN_DIR=<KIT>/refcv7_run/runs/refcv7-r101-s0
export WPD_TRAIN_DIAG_CACHE=<KIT>/refcv7_post/diag/cache/train_diag139 OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1
python -c "import tanitad; print(tanitad.__file__)"     # MUST point into the extracted tree (editable-install trap)
python code/pmap_arms.py --arm PM4 --ckpt $WPD_RUN_DIR/ckpt.pt --out <out> --timing-steps 20   # timing + memory first
for n in PM0 PM0r PM1 PM2 PM3 PM4 R-s4zero R-s4shuf R-gtshift; do       # resumable; re-run until record.json exists
  python code/pmap_arms.py --arm $n --ckpt $WPD_RUN_DIR/ckpt.pt --out <out> --budget-s 1e9 --workers 6; done
python code/pmap_analyze.py --arms-dir <out> --out <package>/raw/pmap_result.json      # CPU, B = 1,000
```

Torch: the Thor venv's torch is pinned. On a pod, install torch from the pinned cu128 index LAST, and verify with a
real `conv2d` on CUDA (the `uv pip` torch-replacement trap). The same arm on a different GPU is the same experiment
only if PM0 and PM0r both run there: **never compare a pod arm against a Thor PM0.**

## 4. Code changes after registration (disclosed; none touches an arm's computation)

1. `opt.step()` added (RESULT §7.1, disclosure 8), plus control C-step.
2. `cuda_max_mem_gb` is logged per log line and in `record.json`.
3. `--timing-steps N`: a timing-only path that writes `<arm>_timing/timing.json` and never writes arm state.
4. The TRAIN / TRAIN-DIAG / run-dir paths read `WPD_DATA`, `WPD_TRAIN_CACHE`, `WPD_TRAIN_DIAG_CACHE` and
   `WPD_RUN_DIR`. With them unset, the paths are the Thor paths the arms run with.

## 5. GPU memory and speed

MEASURED on Thor (`raw/pmap_timing.json`, md5 `6d5f992c…`), 20 steps at batch 8 with 6 workers, median over steps
4–20. On Thor only `torch.cuda.max_memory_allocated()` is admissible.

| arm | s/step | of which data wait | `max_memory_allocated` |
|---|---|---|---|
| PM4 (s4 + FPN + soft lines, the heaviest) | **1.515** | 0.112 | **10.33 GB** |
| PM0 (the lightest) | **1.083** | 0.107 | **9.65 GB** |

⇒ **Need: a CUDA GPU with ≥ 12 GB at batch 8.** `max_memory_allocated` does not count the CUDA context or the
allocator's cache, so allow ~1–2 GB headroom. An A40 (48 GB) has ample room. **The dev-box RTX 4060 (8 GB) does NOT
fit batch 8**, and halving the batch would be a different experiment. It was also not free: at 02:10 local it held
6.9 of 8.2 GB for the REFe NavSim job (`refe_navtest_seam.py`), and it was left alone as instructed.

Data loading is not the bottleneck: 0.11 s of 1.1–1.5 s.

**GPU budget (ESTIMATED from these two):**
- 3,000 steps per arm is 0.9–1.3 h, plus fit and eval of ~10 min.
- Over the 9 arms (the s4 and FPN arms sit between PM0 and PM4) that is **~10–12 GPU-hours**.
- On Thor, under the 40-min rule, that is ~20–24 lock acquisitions.
- On a dedicated A40 it is one unattended run, likely faster than Thor (UNMEASURED there).

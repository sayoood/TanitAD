# ROUTE: which pixel route built v8F-mini's TRAIN cache, and can we reproduce it?

Written 2026-10-25 on export tip `190be80a`. Zero GPU, no box 2, no download, no existing file edited. Evidence classes: MEASURED (ours, artifact named) · INHERITED (another doc/log, not re-run) · ESTIMATED · UNKNOWN.
Probes in this directory: `probe_provenance.py`, `probe_pixels.py`, `probe_pixels2.py`, `probe_cast.py`, `repro_route.py` (+ `build_routes.sh`, `run_repro.sh`, `run_repro2.sh`), `compare_variants.py`, `fallback_cost.py`; each with its `.out` / `.json`.
`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini`.

## HEADLINE

1. **The premise is wrong: the production train cache was NOT built through `thor416`.** Every one of the 4,572 rows of box 2's train manifest (and all 141 eval rows) reads `source=local256, enc_dtype=bf16, device=cuda, pipeline=ra1a-pooled-v1`; **0 rows are thor416** (MEASURED, `probe_provenance.json`, from `box2_ship/logs/v8f_dscale/manifest_train.jsonl` and `.../v8f_grid/manifest_eval.jsonl`). The frames are the **native 256x640 cylinder rebuilt from the local HEVC mp4 on an x86 CPU** by `pod_build_b1_epcache.py`, encoded on box 2's GPU in pure bf16. `thor416` is `refav1_hj_cache/test_runscript`: a 38-clip dev-box smoke run of 2026-10-17 15:34 that D3_SCOPING calls "416-route test, not adopted"; its md5 and content_md5 differ from the production rows for both clips (MEASURED).
2. **So the identity control's 0.079 / 0.090 is a gap to a non-production reference, not the gap between the arm's frames and the cache's frames.** The same flaw sits on the eval side: E_FIELD / E_POOLED used the **Aug-30 Thor eval eps**, while box 2's eval cache was encoded from the dev-box **rebuild `eval141-256rb`** (`encode_B2s.log`: `dir=D:/tanitad/data/eval141-256rb`). Decomposed here: the 0.079-0.090 is **route 7.8-8.5 % (native256 vs thor416, both CPU fp32) + dtype ~3 %**.
3. **The frames DO reproduce, bit for bit.** Today's local rebuild of `8f387977` and `bf5ec76e` equals the S4b v2ep that box 2 encoded in every tensor and, re-saved under the production temp name, is **byte-identical** (sha256 `4d229504...` and `28ca420f...`, which are also the box-2 control hashes, `ctl_result.json` IDENTICAL x2). The eval clip `dca157a0` rebuilds byte-equal to the dev-box `eval141-256rb` file that was shipped to box 2, all 201 raw frames. (MEASURED here; that box 2 holds that file is INHERITED.)
4. **The committed tau_route = 1e-3 on CPU fp32 cannot be passed against the production cache by ANY route, however exact.** The production cache is GPU bf16; CPU fp32 vs bf16 on identical frames reads **rel-L2 0.030 / 0.035** (30-35x tau), measured two independent ways (thor416 frames: CPU fp32 vs GPU bf16, all 101 frames; production frames: CPU bf16 vs CPU fp32 = 0.029 / 0.033). The production cache's clips are also **not on this box** (box 2 only), so the per-clip PASS verdict you asked for is **not evaluable here**; what is evaluable is below.
5. **Escalation (PI / Master Mind).** (a) Amendment 1's tau_route needs a stated device: 1e-3 is only meaningful **same-device** (box 2 verify C5: 5/5 bit-exact same-device re-encodes, INHERITED `encode_A2s.log`). (b) The identity control must be re-run with the **native256** route against `cache_full` **on box 2, GPU bf16** (a box-2 go for minutes of GPU, not hours). (c) The existing `identity_control.json` must not be cited as the route evidence for the PREREG's "no identity-control file ... outside tau" refusal: its train and eval references are not the production caches (`identity_control.py:38-41`, `:212`).

## 1. The route, end to end

### 1.1 A TRAIN clip of the 4,512 (cache rows are 4,572 = 4,512 trained + the 60 RA-1 holdout, withheld by position)

| # | step | what exactly | where (file:line) |
|---|---|---|---|
| 0 | source | HF `nvidia/PhysicalAI-Autonomous-Vehicles` `camera_front_wide_120fov`, B1 bank (Sayood). HEVC **1920x1080 @ 30 fps, 605 frames** per clip (MEASURED with PyAV on the three clips). Local copy `tanitad-data/physicalai/camera/camera_front_wide_120fov/<clip>.mp4`; sha256-checked against HF's `camera_sha256.json`: 600/600 for the first 600 (`stage_report.json`: `mp4_sha_mismatch []`), 3,972/3,972 per chunk before each build | `stage_train600_native256.py`; `ra1_full_box2.py:188-190` |
| 1 | frame choice | timestamps parquet -> 10 Hz grid: `t_query = linspace(t0, t_end, n_target)`, `frame_idx = searchsorted(t_frames, t_query)` -> **T = 201 raw frames** (manifest `t_ep` 201); the cache keeps every 2nd = 101 steps (0.2 s) | `v2_compressed.py:108-126`; `physicalai.py:64` (`TARGET_HZ = 10`) |
| 2 | decode | PyAV, `to_ndarray(format="rgb24")` on the selected frames only (bit-identical to crop-all-then-select, per the docstring) | `v2_compressed.py:53-96` |
| 3 | intrinsics | per-clip f-theta intrinsics from `physicalai_front_wide_intrinsics.csv` (two rigs: cy 542.0 for both train clips, 756.8 for the eval clip; vs HF max diff 1.1e-13); a corpus-median fallback is refused | `physicalai.py:198`; `calib.py:960-994` (`require_per_clip`) |
| 4 | **rectify (the only resize)** | f-theta fisheye -> **cylinder 256x640, hfov 120 deg, `f_ref = 305.5774907364391`** (`CanonicalFrame.from_hfov(120, 256, 640, 'cylindrical')`) in **one** `F.grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False)` straight from the 1920x1080 frame, times the observed-mask; **no antialias / prefilter**; float -> uint8 by `clamp(0,255).to(uint8)` (**truncation**) | `pod_build_b1_epcache.py:83-88,108-109`; `calib.py:174-181,935-957,990-998` |
| 5 | store | lossless PNG per frame in `<clip>.v2ep.pt` (n_stack 3, codec png, 36.99 MB/clip) | `v2_compressed.py:129-161` (PNG at `:143`); `pod_build_b1_epcache.py:115-152` (`build_compressed(... size=256, frame=..., 'cylindrical', 'png')` at `:134-136`) |
| 6 | encode | `pull_encode_pooled.py --source local --src-dir <v2ep dir> --src-geom 256` -> manifest `kind local256`, **no `resize416_to_256`**; then `dinov3_fp8_encode_ship.encode_episode`: `decode_png` of every 2nd raw frame, `x = frames.to(device).to(bf16)/255`, `(x-mean)/std` (ImageNet constants from the model's processor), **DINOv3-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`, HF `AutoModel`, `dtype=bf16`)**, batches of 8, `last_hidden_state`, drop 5 special tokens, `.float()` | `pull_encode_pooled.py:225-235,352-364`; `dinov3_fp8_encode_ship.py:108-129`; `ra1_full_box2.py:95-98` |
| 7 | pool + quantise | fp32 average pool over equal 4x4-token cells -> `[T,40,1024]`, **then** `.to(float8_e4m3fn)` (pool-then-quantise); `mean` fp16 | `pull_encode_pooled.py:254-260,302-305` |

**Machine and date** (MEASURED from the logs unless marked):

* **First 600 clips** (the sha12-ordered prefix; includes `8f387977`, sha12 `001e7d520dd0`, and `bf5ec76e`, `003b73327...`). v2ep: **dev box, CPU only** (`CUDA_VISIBLE_DEVICES=-1`), `launch_build_train600.ps1` -> `pod_build_b1_epcache.py` unchanged, 6 workers (`build.log`), 82.3 min ending 2026-10-17 ~17:53 (the two clips' files are timestamped 17:25 and 17:38), into `C:/Users/Admin/tanitad-data/refav1-train600-256`. Shipped by `scp` to box 2 `D:\tanitad\data\train600-256` (`sync_train600.sh`). Encoded on **box 2 GPU, bf16**: `encode_A.log` 2026-10-17 18:11-18:36 (177 clips, rc 1 on C4 coverage only), `encode_A2s.log` 21:46-22:29 (423 clips, **5.01 s/clip**, `ZZVERIFY-OK`).
* **The other 3,972**. `ra1_full_box2.py` on **box 2**: C0 control at 12:16 on 2026-10-18 (the same two clips rebuilt on box 2 = **IDENTICAL** to the dev-box S4b sha, `ctl_result.json`), then 10 chunks of <=400 from 2026-10-18 12:14 to **2026-10-19 06:37** (`chain_ra1_full.out`): mp4 sha vs ledger -> `pod_build_b1_epcache.py` unchanged on box-2 CPU, 4 workers (`:89-92`) -> `pull_encode_pooled.py ... --source local --src-dir trainrest-256 --src-geom 256 ... --device cuda` (`:95-98`) -> verify (C1-C4; **C5 bit-exact redo was `--redo 0`, so it ran only on the 600**) -> v2ep and mp4 **deleted** (`:212-214`).
* Model weights sha256 `dcb2e451...` is the dev-box test cache's `meta.json` value (MEASURED); that box 2 used the same file is **INHERITED** (box 2's `meta.json` is not on this box).

### 1.2 An EVAL clip (141), e.g. `dca157a0`

Identical code path, steps 0-7. v2ep: **dev box, CPU**, `stage_eval141_native256.py` + `pod_build_b1_epcache.py`, 13.2 min (639 eps/h), 2026-10-17 ~18:03-18:15, into `refav1-eval141-256rb`; shipped to box 2 `D:/tanitad/data/eval141-256rb`; encoded on box 2 GPU bf16 `encode_B2s.log` 2026-10-17 21:36-21:46 (64 new + 77 already done), `kind=local256`. Manifest row: `local256 / bf16 / cuda`.

### 1.3 Where the routes differ

* **Train vs eval pixel route: nowhere** (same builder, same arguments, same `--src-geom 256`, same encoder, same dtype and device). They differ in bookkeeping only: build host (eval and the first 600 train: dev box; 3,972 train: box 2), parity role (`train` refuse / `eval` exclude), clip pick. Cross-host equality is MEASURED on 2 train clips only (C0); for the other 3,970 it is INHERITED.
* **Production eval frames vs the Aug-30 eval eps (what the identity control used as "eval frames"):** these ARE different pixel products (MEASURED, `probe_pixels.json`, clip `dca157a0`, 11 raw frames): MAE **0.747 / 255**, max 3 levels, 70.7 % of values differ; `check_report.json` of the rebuild says 0.730 over 141 clips. `probe_pixels2.json`: the difference is **one-sided and intensity-independent**: mean signed +0.69 / +0.90 / +0.65 (R/G/B), histogram {0: 29.3 %, +1: 66.6 %, +2: 4.0 %, +3: 0.015 %, negatives 0.06 %}, the same in every intensity bin. **Cause UNKNOWN.** Tested and NEGATIVE (`probe_cast.json`): it is not the float->uint8 cast; with the production decode and rectify and only the cast changed, floor reproduces production at 100.000 % of values over all 201 frames (the control) while round / ceil reach 54.2 % / 70.7 % equality with the Aug-30 frames. Candidates left (not tested): the Thor-side YUV->RGB conversion in PyAV/FFmpeg, or an older builder commit on Thor.
* **Production vs `thor416`:** thor416 = Thor's `physicalai-b1-w120-416x1024cyl` v2ep (a different, Thor-made product), `scp`'d and then `resize416_to_256`: `F.interpolate(size=(260,640), mode="bilinear", align_corners=False, antialias=False)`, rows 2:258, round, PNG (`pull_encode_pooled.py:392-420`). On the eval clip (`probe_pixels.json`): Thor-416-resized vs production MAE 1.01 / 255 with max 82 levels; **a local 416 rebuild resized vs Thor's 416 resized: MAE 0.747, max 3, 70 % differ**, i.e. the same Thor-vs-local signature as the Aug-30 eps. Thor's 416 train v2eps (scp'd for the test run) are no longer on this box (`src_pull` empty), so this is measured on the eval clip only.

## 2. Reproduction (CPU fp32, `dinov3_fp8_encode_ship.encode_episode` imported + `pool_field(f,4,10).to(float8_e4m3fn)`, = `payload_from_field`)

### 2.1 The frames (strongest link: PASS, exact)

| clip | local rebuild vs S4b (box 2 encoded this) | re-saved as `<clip>.v2ep.pt.tmp` sha256 | = S4b = box-2 control |
|---|---|---|---|
| `8f387977` | all tensors + scalars equal (`jpeg_buf`, `jpeg_len`, `poses`, `actions`, `frame`, ...) | `4d229504a8abe80c...` | yes |
| `bf5ec76e` | all equal | `28ca420fab214d56...` | yes |
| eval `dca157a0` vs `eval141-256rb` | `jpeg_buf`/`jpeg_len` bytes equal; 201/201 frames equal (`probe_cast.json` floor control, `probe_pixels.json`) | n/a | n/a |

(The raw file sha differs by 60 bytes because `torch.save` names its zip members after the file name: `build_v2ep.py` saves via `<clip>.v2ep.pt.build`; the production builder via `<clip>.v2ep.pt.tmp`. Re-saving under the production name removes it. `repro_native256_fp32.json` fields `v2ep_resaved_sha256_equal_S4b`.)

### 2.2 The pooled fp8 tokens

The production cache clips are **not on this box** (box 2 `cache_full`; D3_SCOPING found no copy; the md5 of the two local candidates differs from the manifest). The only stored caches for these clips are the two **thor416** dev-box test caches (`test_cpu_fp32/train`, CPU fp32; `test_runscript/train`, GPU bf16). pooled fp8 `[T,40,1024]`, 101 frames per clip for the full-clip rows; n = 4,136,960 values per clip.

| row | 8f387977 | bf5ec76e | note |
|---|---|---|---|
| **native256, CPU fp32 vs `test_runscript` (thor416, GPU bf16), 101 frames** | **0.0815** (max 0.25, bitwise 31.3 %) | **0.0887** (max 0.3125, 31.1 %) | FAIL vs 1e-3 (82x, 89x); wrong reference |
| native256, CPU fp32 vs `test_cpu_fp32` (thor416, CPU fp32), 101 frames | 0.0797 (32.0 %) | 0.0847 (32.2 %) | route only, device matched |
| **CONTROL: native256 CPU bf16 vs `test_runscript`, 12 frames** (the identity control's ROUTE case) | **0.07936** | **0.08961** | **reproduces** `identity_control.json` 0.07931 / 0.08960 |
| local 416 rebuild -> resize, CPU fp32 vs `test_cpu_fp32`, 21 frames | 0.0530 | 0.0717 | thor416 NOT reproducible from the local mp4s |
| local 416 rebuild -> resize, CPU fp32 vs `test_runscript`, 21 frames | 0.0553 | 0.0776 | FAIL |
| native256 vs local416 (both CPU fp32), 21 frames | 0.0660 | 0.0580 | the two LOCAL routes differ by 6 % themselves |
| **device/dtype floor, same frames:** thor416 CPU fp32 vs GPU bf16, 101 frames | **0.0297** (68.7 % bitwise) | **0.0354** (62.5 %) | the number tau_route=1e-3 is up against |
| dtype only, production frames: CPU bf16 vs CPU fp32, 12 frames | 0.0290 (69 %) | 0.0331 (65 %) | independent confirmation of the floor |

Reading:

* **Route vs device.** The 0.079 / 0.090 is route-dominated: native256 vs thor416 is 0.078 / 0.085 with the device matched; the dtype floor adds ~0.03 (sub-quadrature, 0.0797 -> 0.0815). A 0.75-level one-sided pixel offset alone moves pooled fp8 by ~3.7 % (the identity control's eval ROUTE row, `3.65e-2`), so a route that differs by Thor's offset **plus** the 416->256 resampling lands near 8 %. (ESTIMATED attribution; the five numbers are MEASURED.)
* **CPU fp32 vs a CPU fp32 cache of the SAME frames does pass:** E_POOLED, `identity_control.json`: **3.7e-5**, 99.9999 % bitwise (INHERITED, on the Aug-30 eval clip). So tau_route is a same-device tolerance, and it was calibrated on exactly that case.
* **PASS iff rel-L2 <= 1e-3 on every clip: not met by anything measurable here, and not evaluable against the production cache.** The route link of the chain passes at the strongest possible level (2.1); the numerics link cannot be tested on CPU.
* `repro_native256_fp32_full.pt` (8.3 MB, fp8 `[101,40,1024]` x 2 clips) is a **CPU-fp32 reference cache of the production route** for these two clips, usable as the CPU reference for a corrected identity control (the LoRA module at B=0 should read ~3.7e-5 against it, E_POOLED-style). It does NOT stand in for the production cache.

## 3. What would make it fully reproducible, and the cheapest alternative

**What is missing is exactly one thing: the numerics link, which needs the production device.** Frames: reproduced exactly. Encoder function: the arm's B=0 module equals the builder's `encode_episode` bit-for-bit at the same device and dtype (SAME_DEVICE case: rel-L2 0.0, max abs 0.0, 100 % bitwise; INHERITED `identity_control.json`). Cache: GPU bf16 on box 2, not on this box. A CPU run cannot close it (the floor is 3 %).

* **Cheapest reproducible alternative (run this first, ~minutes of box-2 GPU): the identity control on box 2 itself,** GPU bf16, native256 frames from `build_v2ep.py --frame 256`, the arm's B=0 encoder and the builder's own `encode_episode` against `cache_full`, >= 200 train + >= 200 eval frames, eval references = `cache/eval` (the `eval141-256rb` encode). Precedent: box 2's own C5 re-encoded 5 clips on the same device and got **5/5 bit-exact content_md5** (INHERITED, `encode_A2s.log`), so the expected reading is ~0, and tau_route = 1e-3 is then a real test. It needs (not edits to existing files, but a new driver): `identity_control.py` hard-codes the wrong references (`:38-41`: Aug-30 eps, `test_cpu_fp32`, `test_runscript`) and `--frame 416` for T_POOLED (`:212`).
* **Two traps to pre-register before that run:** (i) the arm must encode in the **builder's numerics** (pure-bf16 weights, batches of 8 from frame 0, `last_hidden_state`, 5 special tokens); bf16 **autocast over fp32 weights** is a different computation (PREREG allows autocast) and is UNMEASURED here: if it differs it is the same ~3 % effect, now a second lever; (ii) the arm must be fed **byte-identical v2ep**, i.e. rebuilt by `pod_build_b1_epcache`'s call on an x86 stack like S4b/box 2; a Thor-made or any other decoder's frames move the features by 3.7-8 %.
* **If the box-2 control fails the same-device test,** the PREREG's fallback applies. Cost, ESTIMATED from D3_SCOPING's MEASURED constants (`fallback_cost.out`):

| item | value |
|---|---|
| re-encode 4,572 train clips on box 2 (5.01 s/clip) | **6.36 h** box-2 GPU (4,512 only: 6.28 h); eval-141 re-encode (4.20 s/clip) 0.16 h |
| retrain M2b x 2 seeds (19,164 steps at 1.217 steps/s = 4.37 h/seed) | **8.75 h** box-2 GPU |
| frames must be rebuilt first: box 2 deleted the 3,972 v2ep and mp4 (`ra1_full_box2.py:212-214`), only `train600-256` survives | route A (dev CPU builds, 8.23 s/clip): 10.5 h dev CPU + 165 GB over the LAN (3.2 h, in chunks; box 2 `D:` held 85.9 GB free on 10-19) ; route B (box 2 builds, RA-1 FULL recipe 16.66 s/clip): 21.2 h box-2 + 0.98 h mp4 ship |
| **total** | **route A: 15.3 box-2 GPU h, ~19 h wall; route B: 30.1 box-2 h, ~31 h wall** |

Banked B (M2b seeds 0 and 1, F = 0.1207) is no longer the control after a re-encode: B' needs its own seeds and its own F'. That is the real price of the fallback.

## 4. Caveats and UNKNOWN

* The production train and eval pooled caches were **not read** (box 2 only; ssh forbidden). Everything about them is from box 2's manifests and logs pulled earlier by the chains (MEASURED as files, INHERITED as what box 2 did).
* GPU behaviour is **not measured** here (no GPU). The CPU-bf16 vs CUDA-bf16 residual is INHERITED from E_FIELD: 1.3e-2 on pooled-unquantised, Aug-30 frames.
* Cross-host byte-identity of the v2ep is MEASURED for 2 train clips (C0) and 1 eval clip; for the other 4,5xx it rests on the same unchanged builder plus per-chunk mp4 sha checks (INHERITED).
* The cause of the Thor-vs-local one-sided offset is UNKNOWN (cast mode excluded). It does not touch the production route: both production products are x86 rebuilds.
* The 21- and 12-frame rows use linspace subsets of the 101 steps; full-clip rows are marked.
* `subst R:` was used to stay under Windows MAX_PATH; it points at this directory and was removed afterwards. `work/` (195 MB of rebuilt v2ep, mp4 copies, egomotion zip) was deleted; `build_routes.sh` regenerates it in ~3 min. The four small pooled-fp8 dumps (`repro_{native256_fp32,local416_fp32,native256_bf16_control,native256_fp32_12}.pt`, 1-1.7 MB each) were deleted too; `run_repro.sh` and `run_repro2.sh` regenerate them and `compare_variants.json` keeps their numbers.

## Deliverable manifest (all in the export, nothing staged elsewhere, nothing pushed)

`PK/raw/2026-10-25-lora-route/`: `ROUTE.md` (this file); probes `probe_provenance.py/.out/.json`, `probe_pixels.py/.out/.json`, `probe_pixels2.py/.out/.json`, `probe_cast.py/.out/.json`; `build_routes.sh/.out`, `repro_route.py`, `run_repro.sh/.out`, `run_repro2.sh/.out`, `repro_native256_fp32.json/.out`, `repro_local416_fp32.json/.out`, `repro_native256_bf16_control.json/.out`, `repro_native256_fp32_12.json/.out`, `repro_native256_fp32_full.json/.out/.pt` (8.3 MB, the CPU-fp32 reference cache); `compare_variants.py/.out/.json`; `fallback_cost.py/.out`. The Lander should not put the 8.3 MB `.pt` in git unless a CPU reference cache is wanted there.

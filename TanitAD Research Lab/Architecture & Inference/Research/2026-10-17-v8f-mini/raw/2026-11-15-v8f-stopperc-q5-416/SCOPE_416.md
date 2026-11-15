# SCOPE_416: where 416x1024 frames and DINOv3-L tokens come from, how they meet the trainer, and what they cost

Design and scoping only (2026-11-15). Nothing was launched; the pod was read with read-only commands. Card: `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopperc/PREREG_V8F_STOPPERC.md` (Q5), PI approval `stopperc/PI_Q5_APPROVAL_2026-11-15.md` ("Approve reencoding": frames at 416x1024 and DINOv3-L tokens, pod and/or box 2).

Evidence classes: **MEASURED** (ours, path given), **INHERITED** (another doc, not re-run), **ESTIMATED** (rate named), **HYPOTHESIS**, **UNVERIFIED** (the probe that settles it is named in section 7).
Trees: `RT` = STOPBAL's read export `F:/TanitAD-scratch/stopbal_read/tree/stack` (code of record of the card); `E:` = working tree `E:/Projects/TanitAD` (branch agent/arch-inf-20260803; it can lag the mirror tip, so line numbers are for these trees).

## 0. Headline
1. **The eval side already exists at 416x1024, for exactly the 139 STOPCUE episodes.** `E:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl` holds 139 v2ep files, 11,539,188,325 B (MEASURED, its `MANIFEST.json`). Its two missing eval-141 clips are `204b15b4-…` and `38fe80c1-…`. Those are indices 15 and 30 of the eval join (`dagland/motion_c/dk141/ref/join_eval141.npz`, `clip` order is lexicographic, MEASURED). They are also exactly the two clips with no `has` window in `stopbal/read/out/stopdec/stopdec_BAL-NULL_s0.npz`, which leaves 139 clips, 2,365 `has` windows and 131 `stop_hit` (MEASURED). ⇒ A probe needs no eval frame build.
2. **The train side does not exist on the pod at 416.** The pod holds 4,572 train + 141 eval **256x640** v2ep (157 GB + 5.0 GB, `du`, MEASURED 2026-11-15). The only 416 train copy is Thor's banked B1 cache (4,713 clips, 392.97 GB, INHERITED from `Project Steering/Ops/disk_cleanup_2026-10-04/INVENTORY_THOR.md:73`; present today: UNVERIFIED). Thor is not a host the PI approval names.
3. **Same field of view.** Both frames are cylindrical with HFOV exactly 120°. The 256x640 frame is the 416x1024 frame scaled by 0.625, less 3.2 px of the 416 frame at the top and at the bottom. VFOV is 45.4556° against 46.0921°.
4. **Pooled to the trainer's `pool4x10`, a 416 cache needs no architecture change.** The patch grid is 26x64. An angle-matched area pool gives 6.4x6.4-patch cells, which are the 256 cache's 4x10 cells to the degree. `n_tok` stays 40, so `INIT_CKPT_MATCH` accepts M2b, G-DVB passes, and only `--cache-dir` changes.
5. **Cost.** The frame build dominates the probe (~4.3-5.4 h pod CPU, ESTIMATED); the encode dominates the full set (~10.4 h pod GPU, ESTIMATED). Section 5 has the table.

## 1. Raw camera source
| fact | where |
|---|---|
| Front-wide 120° camera mp4 + `timestamps.parquet` + egomotion, from the private HF dataset `Sayood/tanitad-v7-training-corpus` (`camera/<clip>.mp4`) | `C:/Users/Admin/lorapod/v2ep_pod.py:3-6`, `:30`; `E:stack/scripts/pod_build_b1_epcache.py:16-22` |
| 4,713 clips, 61,545,041,700 B of mp4 (mean 13.06 MB), summed from the corpus's own `camera/camera_sha256.json` | INHERITED, `E:TanitAD Research Lab/Data Engineering/Research/2026-09-23-b1-416x1024-cache/RESULT.md` §2.2 |
| The 416 builds pin revision `a0cf20dfb4eafa29b0ac1f3c05337f002bc33ca0` | `…/2026-09-23-b1-416x1024-cache/code/run_b1_416_build.py:36-37`; eval139 `MANIFEST.json` `source_revision` (MEASURED) |
| The pod's 256 route downloads **without** a pinned revision | `v2ep_pod.py:82-84` (`hf_hub_download(HF, path, …)`, no `revision=`) |
| The pod deletes each mp4 once its v2ep is built and verified, so a 416 build re-downloads the mp4s | `v2ep_pod.py:238`; pod `/root/data/_hf` is 2.0 GB (MEASURED `du`: metadata, not the 59 GB of train video) |
| The pod ledger has no `mp4_sha256` for the 139 eval clips (139 / 139 null; MEASURED by a read-only `grep` of `/root/data/v2ep_ledger.jsonl` against `…/2026-09-16-256x1024-cache/raw/fetch_receipt_eval139.json`) | so "same source video at 256 and 416" cannot be checked from the ledger. It is checked by pixels instead (Amendment 7 C6) |

## 2. The frame builders (v2ep = PNG frames + poses + actions, `n_stack` 3)
| builder | file:line | geometry | ran where (MEASURED rate) |
|---|---|---|---|
| **256x640 production**: `pod_build_b1_epcache.build_one` → `v2_compressed.build_compressed` | `E:stack/scripts/pod_build_b1_epcache.py:83-84` (HFOV 120.0, 256x640 **hard-coded**), `:108` (`CanonicalFrame.from_hfov`), `:115-152`; `E:stack/scripts/v2_compressed.py:129` | 256x640 cyl, PNG | LoRA pod, Linux CPU, through `C:/Users/Admin/lorapod/v2ep_pod.py` (imports `build_one` unedited, `:232-233`; ledger `/root/data/v2ep_ledger.jsonl` `:38`; log `/root/logs/v2ep_run.log`). Final line: 4,713 / 4,713, **1,703 clips/h at 13 workers, `mean_build_s` 25.89**, 166.0 min |
| **416x1024**: `build_v2ep_wide.py` (same `build_compressed` → `calib.cylindrical_rectify`, unmodified) | `E:TanitAD Research Lab/Data Engineering/Research/2026-09-16-256x1024-cache/code/build_v2ep_wide.py:69` (`from_hfov`), `:127` (`build_compressed`), `:171-172` (`--width`, `--height`) | `--height 416 --width 1024` | dev box: eval139, 139 clips, `build_s_per_clip_mean` **45.04 s** at 6 workers (MEASURED, `MANIFEST.json`); Thor: B1 4,713 at **3.5-3.8 s/clip wall at 12 workers** (INHERITED, `…/2026-09-23-b1-416x1024-cache/RESULT.md:16`), driven by `run_b1_416_build.py` (shard-staged HF download, pinned revision) |
| refav1 cache builder, Thor 416 source | `E:TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini/code/pull_encode_pooled.py:85-86` (`SRC416`), `:354-358` (the Thor 256 source was deleted), `:392-420` (`resize416_to_256`: x0.625 to 260x640, rows 2:258) | 416 resampled to 256 | the resampled frames match native 256 at MAE 0.59/255 (INHERITED, `:357`, `raw/validation.json`) |

⇒ For the pod, `v2ep_pod.py` cannot build 416 unedited, because `pod_build_b1_epcache.py:84` fixes 256x640. The 416 route on the pod is `build_v2ep_wide.py --height 416 --width 1024`, driven by a pod copy of `run_b1_416_build.py`'s shard pattern: stage a shard from HF at the pinned revision, build, verify, delete the mp4s.

## 3. Rectify parameters and the field of view
* **Projection:** cylindrical, so a column is linear in azimuth. `f_ref = (W/2) / radians(HFOV/2)`, solved from the field and never typed (`pod_build_b1_epcache.py:105-109`; `build_v2ep_wide.py:69-72`).

| | 256x640 | 416x1024 |
|---|---|---|
| `f_ref` (px) | 305.5774907364391 | 488.92398517830253 (MEASURED `MANIFEST.json`; `RESULT.md:15`) |
| HFOV, cylindrical `2·(W/2)/f` | 120.0000° | 120.0000° (`E:stack/tests/test_frame_416x1024.py:44-47`, `:56`) |
| HFOV by the pinhole formula `2·atan(W/2f)` (**wrong here**, TRAPS.md:368) | 92.64° | 92.64° (computed) |
| VFOV `2·atan(H/2f)` | 45.4556° | 46.0921° (`test_frame_416x1024.py:79-84`; `MANIFEST.json` 46.09213171161337) |
| px per degree of azimuth | 5.333 | 8.533 (x1.6) |
| one 16-px patch spans | 3.00° | 1.875° |

* **Containment (from `pull_encode_pooled.py:395-412`).** The 256 frame is the 416 frame resized by 0.625 to 260 rows and cropped to rows 2:258. In 416 pixels, the 256 field is rows 3.2-412.8 (edge coordinates) and all 1,024 columns. In patches, that is rows 0.2-25.8 of 26.
* **Intrinsics are per clip**, and this is load-bearing for rig B (`pod_build_b1_epcache.py:30-37`).
* **Black bottom strip, rig B (INHERITED, `RESULT.md` §6).** 30-33 of 416 rows are black on rig B (8 / 10 pilot clips, perfect rig separation). The 256 cache's own record gives rig B `masked_frac_mean` 0.0905 against rig A 0.00007. The strip decodes the rig at 0.899 balanced accuracy on eval-139. ⇒ The rig shortcut exists at both resolutions. The extra 3.2 px at the bottom of the 416 frame lie inside it for rig B. HYPOTHESIS: a paired 416-vs-256 difference cancels it to first order.

## 4. Token grid at 416x1024, and the trainer's interface
* **Grid:** 416/16 x 1024/16 = **26 x 64 = 1,664 patch tokens**, d 1,024, plus CLS and 4 registers, which are dropped (`E:stack/tanitad/models/trunk_shapes.py:97`; `test_frame_416x1024.py:72`; `pull_encode_pooled.py:7`). 416 and 1,024 are multiples of 16. DINOv3 uses RoPE, so the grid needs no new position table (INHERITED: FULLRES encoded 1,664 tokens with the banked model, `F:/TanitAD-scratch/v8proof/full/fullres_encode.py:121`).
* **The banked encode refuses 416.** `encode_episode` asserts 640 tokens (`E:stack/scripts/dinov3_fp8_encode_ship.py:127`), and `pool_field` hard-codes GH, GW = 16, 40 (`pull_encode_pooled.py:87`, `:255-260`). A 416 encode needs a sibling function. The banked files stay unedited (Amendment 7 C8 pins the sibling).
* **What the trainer accepts (RT):**

| fact | file:line |
|---|---|
| `--grid` ∈ {`pool4x10`, `pool8x20`} | `scripts/v8f_mini_train.py:247` |
| `n_tok` = 40 or 160 from the grid | `:323` |
| G-DVB: grid declared vs `n_tok` built | `:995-996` |
| `n_tok` and `d_feat` are in `INIT_CKPT_MATCH`, which refuses a mismatched init | `:662`, `:681-683` |
| `--cache-dir` (holds `train/` and `eval/`) | `:232` |
| `pos` is `[n_tok, d]`; the adapter reads `feats` | `tanitad/models/v8f_mini.py:701`, `:858` |
| The loader reads `o[self.grid]` and checks only the frame count; there is no pipeline or geometry check | `tanitad/data/v8f_mini_data.py:443`, `:478` |

⇒ A 416-derived cache that stores `pool4x10` `[T, 40, 1024]` fp8, with the same kinematics keys, trains with `--cache-dir` as the only change. The M2b init is accepted (`n_tok` 40, `d_feat` 1,024).

* **Mapping 26x64 onto 4x10 (cells of 6.5 x 6.4 patches do not divide):**

| option | cells (patches) | field | trainer | verdict |
|---|---|---|---|---|
| **P-A, angle-matched area pool** (fp32 fractional-overlap weights, normalised, then fp8 e4m3fn as the builder does) | 6.4 x 6.4 for 4x10, 3.2 x 3.2 for 8x20. Row edges at patch rows 0.2 + 6.4r, column edges at 6.4c | exactly the 256 field: each 4x10 cell spans 12.0° of azimuth and the same tan-elevation band as at 256 | `n_tok` 40 / 160; M2b's `pos` rows keep their meaning | **primary** |
| P-B, full-field area pool | 6.5 x 6.4 | the 416 field (+0.2 patch rows at each end, black on rig B) | as P-A | beside |
| P-C, integer crop to 24x60, 6x6 cells | 6 x 6 | loses 7.5° on each side plus one row at top and bottom | as P-A | rejected: it changes the field |
| P-D, `adaptive_avg_pool2d` | floor/ceil bins: unequal and overlapping | 416 field | as P-A | rejected: the cells are not equal-angle |
| P-E, native integer grids 13x32 (416 tokens) or 26x64 (1,664) | 2 x 2 or 1 x 1 | 416 field | `n_tok` 416 / 1,664: not a `--grid` choice, refused by `INIT_CKPT_MATCH` and G-DVB, so an architecture change | probe-only, beside |

With integer cells (16x40 → 4x10), P-A reduces to the builder's `reshape(…).mean` (`pull_encode_pooled.py:259`). Amendment 7 C7 pins that. ⚠️ A 4x10 cell at 416 averages ~41 patches against 16 at 256. HYPOTHESIS: pooling washes out what the extra pixels add. FULLRES measured that a one-token planted object survives unpooled (AUROC 1.0000) and is lost by the 4x10 pool (0.5030) (`F:/TanitAD-scratch/v8proof/full/VERDICT_FULLRES.md`, control f5). This is why V160 is read too, and why P-E sits beside.

## 5. Costs
**Rates (each estimate names its basis):**
| id | rate | class, source |
|---|---|---|
| R1 | 256 build on the pod: 25.89 s per clip-worker; 1,703 clips/h at 13 workers | MEASURED, `/root/logs/v2ep_run.log` final PROGRESS |
| R2 | 416 build: 45.04 s per clip-worker (dev box, 6 workers); 3.5-3.8 s/clip wall at 12 workers (Thor, about 42-46 s per clip-worker) | MEASURED eval139 `MANIFEST.json`; INHERITED `RESULT.md:16` |
| R3 | 416 build on the pod: **48-55 s per clip-worker, 3.7-4.2 s/clip wall at 13 workers** | ESTIMATED as R1 x (0.30 + 0.70 x 2.2-2.6). PNG encode is 70 % of a 256 build (INHERITED, `INVENTORY_THOR.md:148`); PNG bytes grow x2.2 (83.1 / 37.8 MB per clip) and pixels x2.6. Agrees with R2 |
| R4 | 256 encode on the pod's RTX 4000 Ada: **2.76 s/clip** (about 101 grid frames, so 27.3 ms per frame), 4x10 + 8x20 pooled | MEASURED, `/root/logs/cache_pod.log` (4,700 clips) |
| R5 | 416 encode on the pod: **8.2 s/clip, 81 ms per frame** | ESTIMATED as R4 x 2.98, the ViT-L/16 FLOP ratio at 1,669 vs 645 tokens (24·N·d² + 4·N²·d per block). Cross-check: FULLRES on the dev box's RTX 4060 encoded 416 and its 256 resample for 14,035 frames in 3,194 s (MEASURED, `v8proof/full/encode_job.log`). The 416 share (2.98/3.98) is ~171 ms/frame and the 256 share ~57 ms/frame, which is 2.1x R4, so ~82 ms/frame on the pod |
| R6 | sizes: 416 v2ep 83.1 MB/clip; 256 v2ep 37.8 MB/clip; pooled 4x10 + 8x20 + mean cache 19.5 MB/clip (pod 92 GB / 4,713); 4x10 + mean + kinematics only ~4.4 MB/clip; S0-P probe arrays 3.31 GB per 16,000 windows; eval probe npz 468 MB | MEASURED (`RESULT.md:15`; `INVENTORY_THOR.md:148`; pod `du`; `C:/Users/Admin/s0p_state/deletions.log`); 4x10-only ESTIMATED from 101 x 40 x 1,024 B + 101 x 2,048 B |
| R7 | S0-P fit (prep, 10 fits, read) at 2 CPU threads: 1.99 h | MEASURED, `C:/Users/Admin/s0p_state/chain.log` 06:59:24 → 08:58:48 |
| R8 | one 3,000-step v8f-mini run on box 2: 2,276 s (~40 min) | INHERITED, card §5 (`PREREG_V8F_STOPBAL.md:18`) |

**(a) Eval-141 + the S0-P probe sample only (Amendment 7 stage Q5-0)**
| item | amount | time (ESTIMATED) |
|---|---|---|
| eval frames | 0 (E: eval139 exists). For one uniform route, the pod rebuilds all 141 clips: 141 x 50 s / 13 workers | 0, or ~9 min pod CPU |
| train frames | The sample touches nearly every pool clip: 12,000 negatives over 4,512 clips leave ~4,196 distinct clips expected, plus the positives' clips, so ~4,200-4,512 clips (ESTIMATED; exact count from `s0p_extract_train.py --dry-run`) | **4.3-5.4 h pod CPU** (R3); ~55-59 GB of mp4 streamed from HF |
| encode | NOW frames: 16,000 (sample S1) + 16,000 (replicate S2) + 2,365 eval = 34,365 at 416, and the same 34,365 at 256 on the pod (the pod-vs-pod comparator) | ~47 min + ~16 min pod GPU (R5, R4), overlapped with the build |
| disk | transient: one shard of 416 v2ep (~120 clips ≈ 10 GB) + its mp4s (~1.6 GB). Kept: probe arrays 2 samples x 2 resolutions x 3.31 GB + eval npz ≈ 15 GB | |
| fits | ~20 real fits + 20 permutation fits | ~4-5 h CPU at 2 threads (R7 scaled by the fit count; V160 MLP is the long pole, ~50 min each) |
| **wall to a verdict** | | **~10-11 h**: ~5.5 h pod + ~0.5 h box-2 CPU (the S0-P re-extraction, Amendment 7 C0) + ~4-5 h fits |

**(b) Full 4,572-clip train set + eval-141, dense cache (Amendment 7 stage Q5-1, only on RES416-ADDS-40)**
| item | amount | time (ESTIMATED) |
|---|---|---|
| frames | 4,572 + 141 clips | 4.7-5.5 h pod CPU (R3) |
| encode | 4,713 x 8.2 s | **~10.7 h pod GPU** (R5). The build runs concurrently, so the wall is ~10.7-11 h |
| disk | output 89 GB train + 2.7 GB eval (R6, 4x10 + 8x20 + mean). The 416 v2ep are not kept: 380 GB, while the pod's `df` shows 224 GB free and `df` is not the authority on a pod (memory note: dd-probe the quota) | |
| ship to box 2 | 4x10 + mean + kinematics only, ~20 GB + 0.6 GB eval (R6) | rate UNVERIFIED |
| training | 2 runs (Q5_s0, Q5_s1) | ~80 min box 2 (R8) + T1-kin dumps + gates ≈ 2 h |
| Thor alternative for frames | 4,713 416 v2ep (393 GB) as of 2026-10-04 | skips R3; 380 GB over `tanitad-thor-wifi` (`pull_encode_pooled.py:80`) at a rate that is UNVERIFIED; Thor is outside the approval's hosts |

## 6. What PROOF-14 / FULLRES actually measured (read before relying on it)
Card `F:/TanitAD-scratch/v8proof/PREREG_V8F_PROOF_ADD9.md`. Code `F:/TanitAD-scratch/v8proof/full/fullres_encode.py`, `fullres_probe.py`. Definitions `full/FULL_DEFS.md`. Verdict `full/VERDICT_FULLRES.md` (MEASURED there, tier T0).
* **Frames.** E:'s eval139 416 cache (`fullres_encode.py:11`). The "LO" 256 set is the 416 frame resized with antialias to 260x640, rows 2:258 (`:109-111`), not the native 256 frames. f2 cosine to the cached 4x10 pool is 0.9964.
* **Sets.** L40 / L160 = LO pooled 4x10 / 8x20; U640 = LO unpooled; C640 = a 16x40 centre crop of the 416 grid (rows 5-20, columns 12-51); H1664 = the full 26x64 grid. PCA-128 int8 per token (FULL_DEFS R5).
* **Fit.** **5-fold, episode-grouped cross-fitting within the 139 eval episodes** (FULL_DEFS R2). No training clip was used, so each training fold holds ~105 of the 131 stop positives. S0-P had 4,000.
* **Label.** STOPFLAG = `stop_hit` on the `has` windows (FULL_DEFS R3), which is STOPCUE's label, n 2,365.
* **Result.** STOPFLAG AUROC: E 0.8308. Token sets: P3 0.70-0.77, P1 0.80-0.82, every one below E. Resolution pair H1664 − U640: P1 −0.0012 [−0.0066, +0.0043], P3 +0.0060 [−0.0303, +0.0412]. Outcome **CUE-ABSENT-FULLRES**.
* **What it did not measure:** (1) a probe fitted on training clips; (2) 416 tokens pooled to the trainer's 4x10 / 8x20 interface; (3) S0-P's per-token-PCA linear / MLP readout and CV; (4) native 256 frames as the low arm; (5) any replicate of the fitting sample.
* ⇒ It sets a **LOW prior** for Q5 and does not stand in for the S0-P-style read. HYPOTHESIS: every token set falling below E under P3 is the signature of a high-dimensional readout fitted on ~105 positives.

## 7. Open facts and the probe that settles each
| fact | status | probe (read-only unless stated) |
|---|---|---|
| Thor's 416 B1 cache still holds 4,713 v2ep | UNVERIFIED (inventory 2026-10-04) | `ls …/physicalai-b1-w120-416x1024cyl/*.v2ep.pt \| wc -l` on Thor |
| pod disk quota (df shows 500 G overlay, 224 G free) | `df` is not the authority on a pod | a `dd` write probe (a write: the MM's call) |
| HF revision of the pod's 256 frames = `a0cf20df` | UNVERIFIED | diff `/root/data/_hf/camera/camera_sha256.json` against the pinned table; the pixel route is Amendment 7 C6 |
| distinct clips in the S0-P sample (sha256 `843e4017…`) | ESTIMATED ~4,200-4,512 | `s0p_extract_train.py --dry-run` writes `sample.json` |
| box 2 free disk for a ~20 GB 4x10 cache | UNVERIFIED | `dir` on box 2's cache drive |
| `join_train.npz` and `native_tracks.pt` on the pod (needed if the S0-P extractor runs there) | `native_tracks.pt` present in `/root/data/cache_pod/` (MEASURED); join UNVERIFIED | `ls` |
| pod tokens vs box-2 tokens at 256 | MEASURED: pool4x10 rel-L2 0.0225-0.0263, 0.73-0.78 of fp8 values bitwise equal, kinematics equal (`/root/data/cache_pod/{train,eval}/compare_box2.jsonl`) | ⇒ The 416-vs-256 comparison must be pod against pod; box-2 tokens differ by a route term |

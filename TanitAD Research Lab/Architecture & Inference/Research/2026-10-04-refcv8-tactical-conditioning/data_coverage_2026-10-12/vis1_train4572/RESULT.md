# refcv8 VIS-1 sidecar over 4,572 train + 139 eval clips (2026-10-12, 0 GPU, Thor CPU + dev box)

Closes RESULT.md section 5.2 of `../` (the "+203 train clips" package: blocker 2, `require_clips` refusing at `vis1.py:277-283`).
Evidence class MEASURED unless tagged; artifact paths are in `raw/` and the manifest at the bottom. Layer key: SIDECAR = the VIS-1 npz,
JOIN = the 2-D / 3-D agent joins, VIEW = the 4,572-clip train symlink view.

## HEADLINE

| | |
|---|---|
| **Controls** | **ALL PASS** -- (a) 4,508/4,508 old clips unchanged, 0 differing; (b) builder re-run on 239 old clips (100 train + all 139 eval; 45,138 frames, 250,802 rows) reproduces the old entries exactly, 0 differing; (c) clip set == 4,572 train sha12 U 139 eval sha12 exactly; (d) 58 joined ADD clips have frames + finite, ordered values, 145 unjoined are well-formed zero-frame entries; (e) the trainer's own `require_clips` does not raise on the full 4,711-id list (+2 negative controls that must raise, and do); (f) end-to-end trainer-side lookup over all 11,063 joined ADD frames: 0 refusals |
| **New sidecar (Thor)** | `/home/nvidia/data/refcv8/vis1_sidecar_refcv8b1_train4572_eval139.npz` (189,489,856 B; 4,711 clips, 870,560 frames, 4,947,844 rows; 145 zero-frame clips) |
| **sha256** | `b50ab62ff7a49335cdc664a56fbfbb64c983bca6bd0563d0d7ab11e2f6e10c04` |
| **md5** | `da966d270c373fca7eb6f21ed255d2fe` |
| **INTEGRATION (escalated)** | `stack/scripts/launch_gate.py:572` pins the PATH `/home/nvidia/data/refcv7/vis1_sidecar_refcv6b1_train4369_eval139.npz` -> change to the path above; `:593-594` quotes `sha256 278443b3...` in the why-text -> `b50ab62f...`. Launch argv: `--vis1-sidecar /home/nvidia/data/refcv8/vis1_sidecar_refcv8b1_train4572_eval139.npz`. Nothing else about VIS-1 changes (`--slot-vis1`, n_stack 3 unchanged). |

## 1. Method: merge, not rebuild (and why)

The old sidecar took 2,585 s on 12 workers (859,497 frames, ~8 CPU-hours). Re-z-buffering 4,508 clips whose entries we already hold would only
re-derive bytes we can copy, on a GPU box that is training. So:

1. **Sub-build** (Thor CPU, `nice -n 19`, 4 workers, `OMP_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=`, 588 s wall, parent RSS ~0.55 GB, host available RAM 81-84 GB throughout):
   the **unmodified** `precompute_vis1_sidecar.py --mode full` (LF-normalised md5 `7b726a76b837b473d47482d5f435222b` == the builder md5 in the 2026-09-27 `full_record.json`)
   over the 203 ADD clips + a control sample of 100 OLD train clips (seeded random sample, seed 20261012) + all 139 eval clips (442 clips, 56,201 frames).
   Subsetting is done by the new `vis1_subbuild_driver.py`, which replaces only the builder's `load_split_clips` (it reads `_v2manifest.pt` directly with `torch.load`, never
   `load_or_build_manifest`, so no existing manifest can be rewritten) and REFUSES if a selected sha12 is absent.
2. **Merge** (`merge_vis1_sidecar.py`): per-clip entries of the old sidecar are copied unchanged, the 203 ADD entries come from the sub-build, and the whole is passed through
   the builder's own assembly block + `vis1.write_sidecar` (clips in sha12 order, i.e. the layout a full rebuild gives).
   **The assembly is proven before use:** re-assembling the OLD sidecar's own entries with the old meta reproduced all arrays, `content_sha256_of_arrays` `8ba9dafd...` **and the old FILE byte-for-byte (`278443b3...`)**
   (`raw/merge_record.json` -> `roundtrip`; the merge refuses otherwise).

**Inputs verified on Thor before the run** (`raw/preflight.json`): agent joins md5 `0c31a3a63d7205e9fa50e8ac4204ef46` / `401c9c3119a50632dafc35c595d9e051` == the md5s the old build recorded (joins unchanged);
view manifest `77dafb50...`; extrinsics `dba6da20...`; eval manifest `34433232...` (== the dev copy).

**Calibration -- one input differs from the old build and was checked.** The old build read `/home/nvidia/bx_0252/calib` (1,363 chunk files, exactly the old clips' chunks). The 203 ADD clips sit on 171 chunks, **48 of which are absent
from that dir** (the builder refuses a missing intrinsics file; a bounds-only mask would make `require_clips` refuse). Thor holds the full set at `/home/nvidia/data/physicalai-b1/calibration` (1,420 chunks); the new build reads that dir.
Control: all 1,363 old chunk files are md5-identical there, and the builder's own `md5_of_sorted_chunk_md5s` over them reproduces the old record's `dedb36c034b991d29ba4b74afe74dbcf`. All 1,411 chunks of the 4,711 clips are present (`n_missing 0`).

## 2. Representation of the 145 unjoined (zero-frame) clips -- NOT invented

The old sidecar has **no** zero-frame clip (0 of 4,508; every old clip is joined), so there was no precedent to copy; the representation is the builder's own:
`run_shard` emits `frames = ctx.frames_by_clip.get(cid, [])` -> empty frame/row arrays, `clip_frame_ptr` unchanged across the clip, and `clip_mask_ftheta = camera_for(...)[1].startswith("f-theta")` -- which depends only on the
chunk's intrinsics file, NOT on the join, so it is **True** for all 145 (control d). The reader accepts it: `VIS1Sidecar` pointer closure holds, and `require_clips` needs only presence + `mask True`.
Trainer side (read, then measured in control f): `_vis1_item` looks a frame up only `if bool(item["agent_label"]) and n > 0` (`refc_v3_train.py:4145`), and `_agent_item` returns `agent_label False` for a clip with no join record
(probed 3 NOW frames x 145 clips = 435 probes, 0 labelled) -- so a zero-frame clip is never asked for a frame. `vis1.py:305-` `vis1_block_for_rows` early-returns for `n == 0` as well.

## 3. Controls (numbers)

| # | control | result |
|---|---|---|
| a | OLD entries in NEW sidecar (split, f-theta mask, frame_f/n_zh/n_valid, all 8 row arrays; dtype + value equality, per clip) | **4,508 compared / 0 differing** (`raw/controls.json`) |
| a2 | same, by a DIFFERENT decode path on a different host (dev box, expands rows through `frame_clip`, not `clip_frame_ptr`) | 859,497 frames and 4,862,788 rows of the old file == the new file restricted to old clips, element-wise incl. order; clip mask + split equal; **PASS** (`devbox_independent_controls.out.json`) |
| b | the unmodified builder re-run on OLD clips (100 train + 139 eval) vs the old stored entries | **239 clips / 45,138 frames / 250,802 rows compared, 0 differing**; the three named: `00a9f72ef853` (198 fr, 1,054 rows), `02f7878099fa` (197, 295), `03a02abb8f1d` (198, 107) all equal. Proves builder + new calib dir + new extrinsics table + joins + dev-export trainer code (`_agent_item`, origin 58d413f) give the old bytes |
| b' | assembly no-op round trip of the old file | arrays equal, content hash equal, **file byte-identical** |
| c | clip set | 4,711 = 4,572 train + 139 eval; train == `sha12(clip_id)` of the 4,572-clip manifest, eval == the 139-clip manifest, disjoint, sorted-unique, split labels right; all 203 ADD present, `split=train` |
| d | 58 joined ADD clips | joined set derived INDEPENDENTLY from the 2-D join shard files == the set with frames (58 == 58); 11,063 frames, 85,056 rows; per-clip frames min/median/max 32/198/199; 0 clips with non-finite or mis-ordered values (`n_vis <= n_img <= n_full`, all >= 0); 99.56 % of rows have `n_full > 0` (old 99.99 %); mean `vis_frac` row-weighted 0.4247 (old 0.5452), per-clip median 0.634 (old train 0.721) |
| d' | 145 unjoined | 0 frames, 0 rows, mask True, split train, all 145 |
| e | `VIS1Sidecar(new).require_clips(<4,572 + 139 ids>)` | **no raise** (Thor venv; and again on the dev box with the dev-tree `vis1.py`); negative controls: an absent id raises, a flipped mask raises; `meta.n_stack == 3` |
| f | for every frame of every joined ADD clip: the trainer's `V3Dataset._agent_item` block -> `vis1_block_for_rows(new sidecar)` (what `_vis1_item` calls) | **11,063 frames, 0 refusals**; `known` rows 85,056 == in-store-scope rows with a 3-D label 85,056 |

## 4. Things the reader should not over-read (UNVERIFIED / caveats)

* **Not a from-scratch rebuild.** The ADD entries are the builder's own; the old entries are copied. Byte-equality of this file with a hypothetical full rebuild is NOT measured (it would need ~8 CPU-hours); what IS measured is that every ingredient reproduces the old bytes (b, b').
* **Meta differs from a native build:** `zbuffer.module_md5` is the dev tree's `bfd93ca2...` (old `a83fb5cb...`); the only change in `vis1_zbuffer.py` is an added `import hashlib` and the four audit function digests are identical (the builder checks them). `sources.extrinsics`, `sources.intrinsics`
  (`feeb84dd...`, 1,411 chunks), `splits.train` are the new ones; a `merge` block records the old sidecar's sha256 / content hash, the sub-build's sha256 and the round-trip result.
* **ADD `vis_frac` is lower than OLD** (row-weighted 0.425 vs 0.545; per-clip median 0.634 vs 0.721; 7.7 vs 5.7 stored rows per frame). The z-buffer code is the same function that reproduces 239 old clips exactly, so this reads as a population difference (the 58 joined ADD clips are the parent package's 'no-map' clips, and denser: one clip
  alone has 4,352 rows at mean 0.198). The cause is UNVERIFIED; it is stated so a later `vis1` positive-rate change is not mistaken for a sidecar defect.
* **5 of the 58 joined ADD clips store zero rows** (244302af6f29, 294b1e513004, 8f9a89b157ed, 994dba81cf4f, f3af899636c9; 40-2,212 labelled 3-D rows each, all outside `STORE_SCOPE`). The old file shows the same pattern in **295 of 4,508 clips (6.5 %)** vs 5/58 = 8.6 % here, so it is not anomalous; control (f) confirms the trainer would not look those rows up either.
* **Not run:** a full trainer `enable_vis1` on the real dataset / a launch smoke (needs the view's frames + all readers + GPU). Control (f) exercises the same `_agent_item` + `vis1_block_for_rows` pair on the real shards; `enable_vis1`'s own `require_clips` + `n_stack` checks were run as (e).
  `--join-defect-masks` was mined on the old 4,369+139; if a later re-mine lists an ego-footprint frame for an ADD clip, `vis1_rows_by_track` re-keys by track id and the sidecar's track + centre checks still apply (not exercised, no listed frame exists for these clips today).
* The 13 obstacle-bearing no-join ADD clips stay unjoined (RESULT.md section 4 of the parent package) -> they are zero-frame entries here, correctly.

## 5. Thor footprint (CPU only, nothing pre-existing touched)

New paths only: `/home/nvidia/vis1_train4572_build/` (493 MB: `tree/` = dev-export stack 58d413f (builder LF-normalised) + the three new scripts, `sub/`, `out/vis1_sub_add203_ctl.npz`, `merge/`, `logs/`) and `/home/nvidia/data/refcv8/` (the one sidecar).
Read-only use of: the old sidecar (re-hashed after the run: sha256 still `278443b3...`), both joins, the two manifests, extrinsics, `physicalai-b1/calibration`. The only `refcv6`-named path opened was READ-ONLY and required by the eval clip set: `/home/nvidia/data/refcv6-b1-416x1024-eval139/_v2manifest.pt` (md5 `3443323229279a18253691ff36ebffd3` before AND after, mtime 2026-09-23 unchanged); the old sidecar and the 1,363-file old calib are the `refcv7/` and `bx_0252/` copies. Nothing under `/home/nvidia/refcv8_ladder/` was opened or written -- one `ls` of that directory (entry names only) was issued while checking for ladder collisions at the end. No GPU, no git. The Thor builder PID exited (checked).

## 6. Deliverable manifest

See the final report; all files below `stage/` are staged for the Master Mind to land (stage, never push).

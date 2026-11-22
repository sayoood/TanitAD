# TELEGAP data-side kit: definitions, assumptions, commands

Kit for the card `PREREG_V8F_PROOF_ADD122.md` (sections 2, 5 and "Order of work"). Dev-box folder `F:/TanitAD-scratch/v8proof/telegap/enc/`; pod copy `/workspace/v8proof/telegap/kit/enc/`. The kit of card ADD99 (`cropread/enc`, pod `/workspace/v8proof/cropread/kit/enc/`) is used UNCHANGED: `enc_common.py` (frame plan, decode, window routines, `Encoder`, `Layout`) and `stage_clips.py` are imported, never edited. Written and tested on the dev box with synthetic data only; nothing of it has run on the pod. Evidence class of every number below: ESTIMATED unless it says otherwise.

## 0. FIRST: clauses that could fail by construction on real data

1. **t2 can fail with a correct calibration if the two cameras' optical axes differ by about 1 degree or more.** The card's wording is followed exactly: the wide W16 and T8 are both built at (azimuth 0, elevation -2) in each camera's OWN rectified frame, where elevation 0 is that camera's optical axis (the data kit builds no mount pitch, ENC_DEFS 0.3). The wide axis is pitched 0.07 to 1.7 degrees below horizontal (measured by ADD99); the tele mount pitch and yaw are not known to this kit. Two windows at "the same numbers" then see scenes that differ by the difference of the mounts, and the test turns the tele window by only 2 degrees. The record therefore carries an information field `extrinsics_variant` (the wide window pointed through both cameras' extrinsics at the direction of the tele centre); it is NOT part of the pass rule. A FAIL with a good `extrinsics_variant` share points to this clause, not to a wrong calibration row or time match. Parallax between the two mounts is negligible at the distances of interest but not zero for near scenes.
2. **t2 needs texture.** Pearson correlation of grey values in the central 8 degrees is undefined or noise on a flat scene (sky only, a wall, night); a frame with no variance counts as failed. The card has no texture condition, so a share below 0.9 can come from scene content alone.
3. **The 17 ms rule cannot discriminate at 30 Hz.** Both cameras run at about 30 fps (33.3 ms spacing), so the nearest tele frame is at most 16.7 ms away whenever the two clocks agree and the tele clip has no hole: nearly every pilot frame matches. The time match is protected by t2 only. If the tele and wide timestamp clocks have a constant offset (both are assumed to be the same clip-relative microsecond clock; the tele table of the one clip looked at starts at -147491) a part of the frames fails the rule, and a larger offset fails t1 (more than 5 percent with fewer than 0.95 of the pilot's frames matched) by construction.
4. **t1: "every pilot clip has a tele clip" was not looked at before the freeze.** A clip with no row in `clip_index.parquet`, no tele member in its chunk archive or no tele calibration row is a recorded failure of the stage and is listed in `ENCODE_FAILS.json` (`no_tele`); the 5 percent bound of t1 is the card's, not this kit's.
5. **Pilot frame time.** The card says "the frames of each clip that the pilot encoded" and gives no clock. This kit takes the time of the wide VIDEO frame that was decoded for each pilot frame, `t_frames[cache_idx[j]]` (see 2 below), not the nominal query time `t_query[2j]` (up to one wide frame interval earlier). If the card meant the query time, the matched frame can differ by one tele frame for some pilot frames.
6. **Fixed elevations -2 and -1 degrees are relative to the tele optical axis,** like ADD99's -6 and -2 for the wide one, while the card says "fixed in the rig frame". The two readings differ by the tele mount pitch, which the kit does not apply (no mount pitch is applied by the builder either).
7. **The tele camera is assumed to use the same f-theta model** as the wide one (`fw_poly_0..4` through `ftheta_project_rays`, intrinsics table with the same columns, 1920 x 1080). If the tele table's polynomial has another meaning (for example a pinhole focal), every tele window is wrong in scale and t2 fails; the synthetic tests cannot tell.

## 1. Files (all in `telegap/enc/`)

| file | job |
|---|---|
| `tele_stage.py` | stage the tele mp4 + timestamps of a clip list by range reads out of the chunk archives, tele calibration tables, ledger, record |
| `tele_encode.py` | frame match, the two tele windows, tokens, per-clip match files, ledger, manifest |
| `ctl_t2.py` | control t2 as the card words it |
| `pod_telegap_enc_queue.sh` | the queue (markers START / DONE / STOP) |
| `test_tele_enc.py` | 18 local tests (synthetic) |
| `helper_pytest.out` | output of the last full pytest run |

## 2. Definitions

* **Pilot frames of a clip** (the frames the pilot encoded, ADD99 ENC_DEFS section 2): cache frame j = episode frame 2j of the builder's plan; the video frame is `frame_idx[2j]`, `n_cache = ceil(n_target / 2)`. Taken from the WIDE stage root (the data kit's stage root, default `/workspace/v8proof/cropread/stage`, file `ts/<clip>.timestamps.parquet`) through the unchanged `enc_common.frame_plan`: `plan["cache_idx"]` (video frame per cache frame) and `plan["t_frames"]` (its microsecond clock). The time of pilot frame j is `t_frames[cache_idx[j]]` (`tele_encode.pilot_times`). The number of frames of a clip is `plan["n_cache"]`; with `--wide-tok-dir` (the pilot's `tok/D` folder) the first dimension of each pilot token file is compared with it and a clip that disagrees is refused.
* **Tele match** (`tele_encode.match_frames`): for each pilot time the tele frame with the nearest time on the tele timestamps table (column `timestamp`, int64 microseconds, row index = video frame index); a tie takes the earlier frame. Matched when the absolute difference is at most 17000 us (17000 itself is matched, 17001 is not). An unmatched pilot frame gets mask false and a zero token row (no window is built for it).
* **Tele windows** (card section 2): T8 = (8.0 deg, azimuth 0.0, elevation -2.0), T4 = (4.0 deg, azimuth 0.0, elevation -1.0), in that order; azimuth left positive, elevation up positive, elevation 0 = the tele optical axis. Built with the unchanged `window_frame(size)` (256 x 256, horizontal size exact: T8 32 px/deg, T4 64 px/deg), `window_grid` and `window_sample` on the RAW decoded 1920 x 1080 tele frame with the tele intrinsics. Constants `tele_encode.TELE_WINDOWS`, `TELE_SIZES_DEG`.
* **Tele intrinsics and extrinsics.** `tele_stage.py` writes, in the tele stage root, `calibration/physicalai_front_wide_intrinsics.csv` (the file NAME is fixed by the stack's `intrinsics_for_clip`; its CONTENT is the rows of `camera_front_tele_30fov`, columns clip_id, width, height, cx, cy, bw_poly_0..4, fw_poly_0..4) and `calibration/front_tele_extrinsics.csv` (clip_id, qx, qy, qz, qw, x, y, z of the same camera). `tele_encode.py` passes the tele stage root to `intrinsics_for_clip(clip, tele_root)` and asserts `per_clip` (a corpus-median fallback is refused). The extrinsics are used only by the information field of t2.
* **Token format** (ADD99): `tok_T/<clip>.npy` float16 `[n_frames, 2, 65, 1024]`, window order T8 then T4; token 0 = class token, 1..64 = the 16 x 16 patch tokens mean-pooled 2 x 2 to 8 x 8 row-major (the unchanged `Encoder.window_tokens`, bf16 weights, batch 8). Frame i of the file is frame i of the pilot's token file. `match/<clip>.npz`: `tele_idx` int32 (-1 = none), `dt_us` float64 (nan = none; tele time minus wide time), `mask` bool, `t_wide` float64. `tok_T/ledger.jsonl` is the resume record, `tok_T/MANIFEST.json` has n_frames, n_matched, bytes and sha256 per clip, `tok_T/ENCODE_FAILS.json` lists failures and clips with no staged tele clip. Files are written whole under a `.part` name and renamed once (the data kit's way; the kit deletes and moves nothing else).
* **Stage** (`tele_stage.py`): `clip_index.parquet` (HF dataset `nvidia/PhysicalAI-Autonomous-Vehicles`) gives the chunk; clips are grouped by chunk; one `HfFileSystem` file object per chunk (`block_size` 4 MiB), the zip directory is read, only the clip's mp4 and timestamps members are read (the blurred_boxes member is not); `zipfile` checks each member's CRC on read (`crc_ok` true = read without error). Worker threads run over chunks. Width and height are probed from the written mp4 (PyAV), `n_frames` is the row count of the timestamps table. The record `records/stage_<time_ns>.json` has per clip: chunk, bytes, width, height, n_frames, crc_ok, status. Resume: a clip with an ok ledger row, both files present, the mp4 at the ledger's size and a calibration row is skipped. The token is read in process, scrubbed from every error text, never printed; clip ids print as their first 8 characters.
* **Control t2** (`ctl_t2.py`): exactly the card's words; clips = the first `--n-clips` (100) of the evaluation list by ascending sha256 hex of the clip id that have a staged tele clip and at least 2 matched pilot frames; frames = the matched pilot frames at the 1/3 and 2/3 position; corr = Pearson correlation of grey values (0.299 R + 0.587 G + 0.114 B) between the T8 window sampled down to 128 px (2 x 2 mean) and the central 128 x 128 px of the W16 window of the wide frame of the same pilot frame (both at azimuth 0, elevation -2, unchanged routines); a frame passes when the correlation at the built direction is higher than at each of the four tele windows turned by 2 degrees (left az +2, right az -2, up el +2, down el -2); t2 passes when the share of passing frames is at least 0.9. The record JSON has pass, share, n_frames, per-frame numbers (no pixels), the skipped clips with the reason and the information field. Prints `t2 PASS|FAIL share <x> n <n>`; exit 0 on PASS.
* **Wide clips for t2.** The unchanged `stage_clips.py` puts a wide clip under its `--root` as `r0/camera/<clip>.mp4` and `ts/<clip>.timestamps.parquet` (plus ego, obs, calibration tables). `ctl_t2.py --wide-root <the ADD99 stage root>` reads them there; for chosen clips not yet staged there `--stage-wide` calls the unchanged `stage_clips.main(["--list", ..., "--root", <wide root>, "--kind", "eval"])`. On the pod the ADD99 full run has already staged the 141 evaluation clips in `/workspace/v8proof/cropread/stage`, so the call is normally a no-op.

## 3. Commands for the lead

```
# 1. push list (dev box -> pod folder /workspace/v8proof/telegap/kit/enc/), sha256 compare per file afterwards
#    tele_stage.py  tele_encode.py  ctl_t2.py  pod_telegap_enc_queue.sh  test_tele_enc.py  TELE_ENC_DEFS.md
#    (the ADD99 kit is already at /workspace/v8proof/cropread/kit/enc/ and is not pushed again)
# 2. dry run on the pod: prints "CMD: ..." lines and creates nothing
DRY=1 bash /workspace/v8proof/telegap/kit/enc/pod_telegap_enc_queue.sh
# 3. full queue, detached (stage 641 -> smoke 3 clips -> t2 -> full encode); markers in /workspace/v8proof/telegap
nohup setsid env WORKERS=3 bash /workspace/v8proof/telegap/kit/enc/pod_telegap_enc_queue.sh > /dev/null 2>&1 < /dev/null &
# 4. wait: until [ -e $R/TELEGAP_ENC_DONE.txt ] || [ -e $R/TELEGAP_ENC_STOP.txt ]; do sleep 20; done      (R=/workspace/v8proof/telegap)
# resume after a stop: TG_RESUME=1 (the START marker gets a resume line; every step is idempotent)
# single steps by hand (same arguments the queue uses), e.g. the smoke:
#   python tele_encode.py --list $R/lists/all_clips.txt --wide-root /workspace/v8proof/cropread/stage --tele-root $R/stage \
#          --out $R/smoke_enc --limit 3 --keep-windows --wide-tok-dir /workspace/v8proof/cropread/enc_out/tok/D
```
Queue facts: root `${TG_ROOT:-/workspace/v8proof/telegap}`; pilot lists copied from `/workspace/v8proof/cropread/enc_out/lists` (first token of each line; the queue stops unless they hold 641 clips); tele stage root `$R/stage`, outputs `$R/enc_out` (full) and `$R/smoke_enc` (smoke, private windows kept there), control record `$R/enc_out/ctl/T2.json`. Before each GPU step (smoke, full encode) it waits until fewer than 3 GPU processes run and more than 28 GB of container memory are free (`/sys/fs/cgroup/memory.max` minus the `anon` line of `memory.stat`). Stop rules: `TG_STOP_S` (14400 s since this start) and `TG_MAX_GB` (40 GB of new files under the root by `du`, against the size at the first start); a running step is killed with `kill` and a STOP marker is written. Refuses when less than `TG_MIN_FREE_GB` (60) GB are free on `TG_FREE_PATH` (/workspace) and when a START marker exists without `TG_RESUME=1`. A failing t2 writes the STOP marker and the full encode does not start. No `rm` and no `mv` anywhere in the queue.

## 4. Sizes and times (ESTIMATED)

* Staged tele clips: about 6.8 MB x 641 = 4.4 GB (card: about 5 GB), 405 chunk archives opened by range reads, none downloaded whole.
* Tokens: 266,240 bytes per frame (2 x 65 x 1024 x 2 bytes); at 101 pilot frames per clip (the ADD99 value) 26.9 MB per clip, 641 clips about 17.2 GB (card: about 17 GB). Match files under 10 KB per clip. Smoke: 3 clips, about 80 MB of tokens and about 120 MB of private windows.
* New files in all about 22 GB, inside the 40 GB rule.
* Time: tele decode of 1080p at one decode thread about 10 s per clip (the ADD99 wide rate), 3 workers and the GPU overlapped: about 35 to 45 min for the full encode; staging 0.3 h (card); t2 (decode of 2 clips-frames x 100 clips x 2 cameras, 4 workers) about 10 min.

## 5. Not done

No run on the pod; no test with a real mp4, a real chunk archive or the real encoder; the clock and the f-theta assumptions of clauses 3 and 7 are untested on real data. No part of the fit variant or the reader (the card's next step) is in this kit.

## 9. Lead's change after review (note N1 of card ADD122, before any encode)

Items 1 and 6 of section 0 are answered by a definition, not left as risks. The wide windows of card ADD99 are fixed in the wide camera's own
rectified frame (its optical axis, no mount pitch), not in the rig frame. The tele windows T8 and T4 are centred on the directions (0, -2) and
(0, -1) of that WIDE frame, carried into the tele camera's own frame by the extrinsics of both cameras of the clip (`tele_encode.tele_dirs`;
centre direction only, a roll between the cameras is not applied). The per-clip directions are stored in the match file (`tele_dirs_deg`).
Control t2 reads the tele window at that carried direction against W16 at (0, -2); the form with both windows in their own frame is printed
as `own_frame_variant`, for information. A clip with a missing extrinsics row is refused by the encode and skipped by t2. Item 2 (texture)
and item 3 (clock offset) stand as written: a t2 below 0.9 fails the card.

# ADD100 arm S data-side kit: definitions, commands, measured rates

Kit for card `PREREG_V8F_PROOF_ADD100.md` (sections 3 and 5: arm S, controls s2 second half, s3, s8). Dev-box folder `F:/TanitAD-scratch/v8proof/cropread/steer_enc/`; pod copy `/workspace/v8proof/cropread/kit/steer_enc/` (pushed by `push_steer_enc.sh`, sha256 compared per file). It imports the data kit `cropread/enc/` unchanged (`enc_common`: `frame_plan`, `decode_selected`, `window_frame`, `window_grid`, `window_sample`, `Encoder.window_tokens`, `Layout`; `ctl_c7.join_n`). Evidence class of every value below: MEASURED on the smoke (pod folders `s_smoke1`, `s_smoke2`) unless it says ESTIMATED.

## 0. FIRST: clauses that could fail by construction, and what the code cannot do as written

1. **s3 compares only frames with a lead, and that is where the routes agree.** Arm B on a frame with a lead centres W32 and W16 on one direction; arm S does the same with the steering direction, so those frames are the same computation. The differences that remain: (a) the steering file is float32 and the box file is float64 (the box files on the pod already hold float32-representable values: the cast moved no direction, `max_abs_float32_cast_deg` 0.0 on all 6 clips, so this adds no difference on the smoke); (b) arm B encodes only its lead frames in batches of 8 and arm S encodes every frame in batches of 8, so the windows of a batch differ and bf16 GPU kernels can differ in the last bits. MEASURED: cosine minimum 0.99995330 (one clip, frames near the end of the clip), the other 3 clips with a lead 0.9999999999999998; margin to the 0.9999 bar is 5.3e-5. The clause holds on the smoke with a small margin on one clip; it could fail on the full set from batch-composition noise. It is not changed.
2. **Frames without a lead cannot reproduce arm B or arm D in arm S.** Arm D has two different centres (W32 at elevation -6, W16 at -2); a steering file has one direction per frame, applied to both windows. So arm S can never equal arm D on any frame, with any steering file. Cosine against arm B on no-lead frames (information only, not in the pass): minimum 0.82 to 0.88 per clip (with the fill direction azimuth 0, elevation -4). The card's s3 names only frames with a lead, so this does not touch the control; it matters for reading arm S as "D plus a shift".
3. **s3 needs a clip with a lead.** 2 of the 6 clips (`998310a5`, `480d7910`) have no lead frame at all, so they contribute nothing; 310 lead frames over 4 clips are compared. The control passes when the total is above 0 and every compared frame is at or above 0.9999 (a clip with no lead frame is not a failure; the first smoke attempt, `s_smoke1`, stopped on exactly that reading, which was a defect of the control script, fixed in `ctl_s3.py` before the second attempt, with the threshold unchanged).
4. **Control s2 first half (reproduce the steering file from arm D tokens and LA within 1e-3 degrees) is not in this kit**: it needs the localiser. This kit writes the manifest half of s2 into `S8.json` (`s2_second_half`).
5. **Frame count rule.** The steering file's frame count is checked against `frame_plan(ts)["n_cache"]`, which equals the join's `n` (the c7 check of the data kit). s8 checks the token file's count against the join directly.
6. The code cannot encode a clip whose staged video or timestamps are missing; the precheck lists such a clip as refused and nothing starts.
7. **Kept camera pixels.** The first full-route smoke (`s_smoke2`) was started with the queue's old setting that kept the windows: `s_smoke2/win/` (76 MB, 2 clips, PRIVATE camera pixels) exists and was not deleted (no delete allowed). The queue now never passes `--keep-windows` (the file `pod_steer_enc_queue.sh` on both sides carries the change). It is the first thing to sweep.

## 1. Files (all in `cropread/steer_enc/`)

| file | job |
|---|---|
| `steer_encode.py` | steering check (`check_steering` :47), windows (`windows_from_video` :76, `make_steer_windows` :88), tokens (`encode_steer_tokens` :129), resumable `run_clips` :181, ledger, MANIFEST (:158), precheck of all files before the encoder starts (:169) |
| `ctl_s3.py` | control s3; `compare_s3` :40 (cosine per lead frame, bar `THRESH` :23) |
| `ctl_s8.py` | control s8 + the manifest half of s2; `check_clip` :20 |
| `fake_steer.py` | smoke only: made-up steering from the box files (box direction with a lead, azimuth 0 / elevation -4 elsewhere) |
| `pod_steer_enc_queue.sh` | queue: s3, then all clips, then s8; markers START / DONE / STOP; resumable; `SMOKE=1`; log; never deletes |
| `push_steer_enc.sh` | push with sha256 compare per file |
| `test_steer_enc.py` | 8 local tests |
| `tmp_pt/` | pytest basetemp (left in place) |

## 2. Definitions

* **Steering file** `<S_ROOT>/steer/<clip>.npy`: float32 [n_frames, 2], columns azimuth, elevation in degrees (left and up positive, the angles of note N1), every frame finite. Refused: other dtype, other shape, wrong frame count, NaN or inf (`check_steering`, tested).
* **Windows**: W32 then W16 of the data kit (`enc_common.window_frame`, `window_grid` with a per-frame centre as an array, `window_sample`), both centred on the frame's steering direction, cut from the raw decoded frame; the frame plan and decode are the data kit's.
* **Tokens**: `<S_ROOT>/tok/S/<clip>.npy` float16 [n_frames, 2, 65, 1024], the layout of arms D and B. `tok/S/ledger.jsonl` (resume record), `tok/S/MANIFEST.json` per clip: `n_frames`, `bytes`, `sha256` of the token file, `steer_sha256` of the steering file used. A token file written for a different steering file is not overwritten: the queue refuses (rc 6) and a new `S_ROOT` is needed. Output files are written under `.part` and renamed once.
* **s3**: clips = first 6 evaluation clips by sha256 hex of the clip id (the data kit's rule). Steering = box direction with a lead, else azimuth 0 / elevation -4, written as float32 into `<S_ROOT>/s3/steer`; tokens into `<S_ROOT>/s3/tok/S`; cosine per frame of the flattened [2,65,1024] against `enc_out/tok/B`, float64; `ctl/S3.json` with `pass`.
* **s8**: every clip of the list (141 evaluation + `train_final_ids.txt`) has a valid steering file and token file with the join's frame count; token sha256 re-hashed against the manifest; manifest steering sha256 present and equal to the file on disk (`s2_second_half`). `pass` needs all of them and 641 clips.

## 3. Commands (lead)

```
bash /f/TanitAD-scratch/v8proof/cropread/steer_enc/push_steer_enc.sh
# steering files first: /workspace/v8proof/cropread/s_out/steer/<clip>.npy for every clip of enc_out/lists/all_clips.txt (the localiser step)
# check memory.max / anon first: keep 28 GB free; at most one GPU process of yours
nohup setsid env WORKERS=5 bash /workspace/v8proof/cropread/kit/steer_enc/pod_steer_enc_queue.sh > /dev/null 2>&1 < /dev/null &
# markers in /workspace/v8proof/cropread/s_out: START.txt DONE.txt STOP.txt; log queue.log; controls ctl/S3.json ctl/S8.json
# wait: until [ -e .../s_out/DONE.txt ] || [ -e .../s_out/STOP.txt ]; do sleep 20; done      (one pssh call, <= 540 s)
# smoke again: SMOKE=1 (new folder s_smokeN each time).  S3_GATE=0 continues after a failed s3.  GPU_WAIT_S waits while 3 GPU processes are listed.
# local tests: PYCAP_MB=3000 bash /f/TanitAD-scratch/v8proof/ops/pycap.sh -m pytest test_steer_enc.py -q -p no:cacheprovider --basetemp <new dir>
```
The queue's s3 uses `enc_out` (lists, box, tok/B) and the stage root `/workspace/v8proof/cropread/stage`; s3 does not need the steering files, the encode step does (it refuses before the encoder starts when any is missing, listing them in `tok/STEER_REFUSED.json`).

## 4. Measured rates (A40 pod, CPU quota 7.65 cores, 2 to 3 workers)

| step | per clip (about 101 frames) |
|---|---|
| decode + windows (2 windows x 101 frames), in a worker | 9.8 to 13.1 s |
| encode (202 windows, GPU) | 1.1 to 1.6 s |
| s3 on 6 clips, workers 2 | 35 s wall = 6.98 s per clip (includes encoder load, about 7 s once) |
| full route on 2 clips, workers 3 | 14 s wall; 7.1 s per clip at the second clip |

Full run (ESTIMATED from the data kit's rule, decode bound): 641 x about 11 s / 5 workers = about 25 min, GPU encode 641 x 1.3 s = 14 min overlapped. Tokens 26.9 MB per clip: 17.2 GB for arm S (MEASURED size per clip in the data kit). Smoke test results: s3 `pass` true, cosine minimum 0.99995330, median of per-clip medians 1.0 (310 lead frames, 4 clips); s8 on the 2 smoke clips `pass` true with 2 of 2 steering sha256 equal.

## 5. Temporary files left in place (no delete was issued)

* dev box: `cropread/steer_enc/tmp_pt/` (pytest basetemp `t1`).
* pod: `/workspace/v8proof/cropread/s_smoke1` (the first attempt, s3 stopped on the control-script defect of item 3; `s3/` 154 MB of tokens of the 6 clips, no pixels), `/workspace/v8proof/cropread/s_smoke2` (second attempt: `s3/` 154 MB, `tok/` 52 MB, `steer/` 2 files, `win/` 76 MB PRIVATE camera pixels, logs, controls).

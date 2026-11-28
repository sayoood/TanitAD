# HANDOFF GAP48 (2026-11-07, dev-box clock 05:20). Built and rehearsed; NOT launched: box 2 was busy. CPU only, nothing staged or committed.

MEASURED = ours (artifact named). ESTIMATED = arithmetic from MEASURED stage-1 timings. UNVERIFIED = needs box 2.

## State
* Scene list: 62 - 14 = **48** (34 scenes2 + 13 scenes3 + 1 kit), derived by `scenes48.py` into `SCENES48.json`; every clip hashed against the release listing. A read-only listing of box 2 matched all 157 scene-file sizes and found `ref_frames.npz` for all 48 (MEASURED).
* Package `build/` (final): tar `t2v8f_gap48.tar` sha256 `d8b01785f0d42a773f1f55140287a3c87d84038e2195225d0415cb278f301f34` (110 KB: job, `jobs/gap48.cmd`, manifest, empty clip dirs), `MANIFEST_GAP48.json` sha256 `034ef81d...b19d5c`,
  job sha256 `8248fba0...a2fcd9410b7`. The 48 clips (1.97 GB) go by `put_mp4.sftp` from where they live. Earlier builds kept in `build_prev1_*`, `build_prev2_*`.
* Tests: `tests/test_gap48.py` **20 passed** (`logs/pytest_gap48.run8.log`; earlier red runs kept beside it with their cause in the name). They include the stub rehearsal on 2 scenes through the real builder and tar, resume, three refusals, the
  mutant (render replaced by the real frame: cosine 1.0 on every tick, `test_hook` recorded, reader status IDENTITY_DETECTED, nothing pooled; the control run is not flagged), and the chain's two refusal paths with a clobbering-chain mutant.
* Box 2 at 05:17: `D:\t2v8f\pull5_BPC\STAGE5.running` live (BPC, cell 97/248, deciding pass); dev-box marker `C:/Users/Admin/box2_ship/BOX2_BUSY.txt` = `owner=T2V8F stage5 BPC`, 04:29, eta 1.8-2.7 h. `D:/t2v8f/BOX2_BUSY.txt` on box 2 did not exist. Neither marker was touched.
  Note for the brief: the marker the existing chains use is the DEV-box file above; `gap48_chain.sh` honours it AND also claims/respects `D:/t2v8f/BOX2_BUSY.txt` on box 2, moving aside only the files it created.

## Exact next step (when `C:/Users/Admin/box2_ship/BOX2_BUSY.txt` is gone and no `STAGE5.running` is left on box 2)
```
ls C:/Users/Admin/box2_ship/BOX2_BUSY.txt        # must say: No such file
powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c E:\Projects\TanitAD-artifacts\dagland\motion_c\t2v8f\gap48\gap48_chain.cmd'}"
```
The chain refuses by itself if either marker or any `*.running` exists. Log: `gap48/logs/chain.out` (look for `ZZ GAP48 LAUNCHED`, `canary`, `ZZ GAP48 PULLED`). It claims, puts the tar and the clips, launches, waits on `GAP48.done|GAP48.fail`
(exact names), pulls to `C:/Users/Admin/t2v8f_pulls/gap48/pull_gap48/` (F: if C: is short), runs `read_gap48.py` into this folder (`RESULT_GAP48.md/.json`), moves its busy markers aside. Resume after a crash: same command with `GAP48_RESUME=1` in the environment.
Manual read: `python -I read_gap48.py --pull <pull>/pull_gap48`. Exit codes: 2 no package, 3 busy, 4 precondition, 5 hash, 6 verify, 9 wait timeout (box stays claimed).

## Wall time (ESTIMATED, from stage 1's MEASURED steps: preflight 145.7 s, gap 14.6 s for 2 scenes x 9 ticks incl. encoder load)
Job: preflight ~2.5 min (no 26 GB volume hash; sizes only) + 48 scenes x ~20 s (renderer 3 s, actor attach ~5 s, one-pass 4K decode ~3 s, 20 ticks x ~0.45 s) = **~16 min, range 10-28 min**. Chain: + clip put 1-6 min (2 GB; link speed UNVERIFIED) + pull 0.5 min
= **~25 min, range 15-45 min**. Footprint on box-2 D: ~2.1 GiB, free floor 5.1 GiB (D: had 12.8 GiB at 05:17). The first scene's marker is the canary; its `GAP48_RESULT.json` row carries the real per-scene seconds (replaces this estimate).

## Not done / open
* Nothing ran on box 2: `read_equiv_check`, actor attach with the real falsifier, DINOv3 tokens and the GPU timings are UNVERIFIED until the first scene lands. A refusal there ends in `preflight.fail` or `scene_<id>.fail` with a named reason.
* Volumes are checked by size, not sha256 (stages 4 and 5 hashed them on this box; INHERITED). `--hash-volumes` in `jobs/gap48.cmd` adds ~26 GB of hashing.
* Manifest of everything: see the deliverable list in the hand-back.

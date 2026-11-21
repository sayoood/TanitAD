# GAP48: the NuRec render gap on the 48 read scenes (descriptive; it decides nothing by itself)

Built 2026-11-07 on the dev box (CPU). Evidence classes: MEASURED = ours, artifact named; ESTIMATED = arithmetic from MEASURED stage-1 timings; UNVERIFIED = needs box 2.
Nothing here has run on box 2. The box was held by the BPC stage-5 run when this was built, so GAP48 was not launched.

## What it measures
For each scene that survives rule R-SCN, at every 10th tick (every tick if a scene has fewer than 20 ticks): render the front-wide camera at the LOGGED pose in the `objects` condition, read the real mp4 frame of the
same camera frame, encode both with the in-loop encoder (frozen DINOv3-L bf16, pool4x10), and bank both pooled token tensors (fp16 [40,1024]) with the cosine and the relative L2.
It is stage 1's `gap()` widened from 2 scenes x 9 ticks to 48 scenes x 20 ticks. `build_renderer`, `make_encoder`, `E.cosine`, `E.rel_l2` and `N.grad_ncc` are IMPORTED from the stage-1 package that is already on box 2
(pinned by sha256 in the manifest); `t2v8f_gap48.py` copies none of them.

**Two differences from stage 1's `gap()`, both on purpose and both recorded in the result:**
1. Stage 1 built the renderer WITHOUT actors (`objects=False`). GAP48 attaches them (`--condition objects`, the default; the replayed agents are drawn at the logged time, as in every T2 rollout). `--condition none` reproduces stage 1.
   A GAP48 number and a stage-1 number are therefore not the same quantity; the stage-1 rows in the yardstick are labelled "no actors attached".
2. Stage 1 read each real frame with `N.read_ref_frame`, which reopens the mp4 and decodes from frame 0 each time. The clips are 3840x2160 (MEASURED on the dev box: 6.2 s to reach frame 300).
   GAP48 decodes each clip ONCE and applies the same conversion (`rgb24`, then `N.resize_area` when the size differs). The first scene is checked against `N.read_ref_frame` bit for bit on the box (`read_equiv_check`),
   and the tests check it on a synthetic clip (with and without a resize) and on one real clip.

## The scene list: 62 - 14 = 48 (derived, never typed)
`scenes48.py` reads `stage4/SCENE_SOURCES_2p3.json` (62 distinct scenes: 41 in `scenes2`, 20 in `scenes3`, the kit scene 7c72937c) and `stage4/EXCLUSIONS4.json` (14 scenes excluded by R-SCN: 12 `ACTOR_REFUSED`, 2 `NO_DYNAMIC_LAYER`).
Result (MEASURED, `SCENES48.json`): **48 scenes = 34 in `scenes2` + 13 in `scenes3` + 1 kit**. All 14 exclusions are among the 62. The builder refuses any other count; every clip is hashed against the release listing's `lfs_sha256`.

## Files
| file | role |
|---|---|
| `scenes48.py` | the derivation; writes `SCENES48.json` (48 scenes with their files and their clip's path, size, sha256) |
| `make_gap48.py` | the builder: `build/t2v8f_gap48.tar` (the job, `jobs/gap48.cmd`, `MANIFEST_GAP48.json`, the empty clip directories), `put_mp4.sftp`, `expect_files.txt`, `BUILD_RESULT_GAP48.json` |
| `t2v8f_gap48.py` | the box-2 job (one process, strictly one scene at a time) |
| `gap48_chain.sh` / `.cmd` | the launch chain (claim, put, verify, launch, wait, pull, read, release) |
| `read_gap48.py` | the dev-box reader: `RESULT_GAP48.md` + `.json` |
| `tests/test_gap48.py` | the tests; `logs/pytest_gap48.*.log` the runs |
| `HANDOFF_GAP48.md` | state, the exact launch command, the wall-time estimate |

The clips (1.97 GB) are NOT copied into the tar: the chain `put`s them from where they live on the dev box (`put_mp4.sftp`), so the dev box (E: has 2.3 GB free) never holds a second copy. Box-2 layout:
`D:\t2v8f\gap48\{code,jobs,mp4\<scene>\camera_front_wide_120fov.mp4,MANIFEST_GAP48.json}`; output `D:\t2v8f\pull_gap48\`; log `D:\t2v8f\logs\gap48.log`.
It imports `D:\t2v8f\pkg\code\*` (stage 1's package, which stages 2 to 5 only added to) and reads the scenes where stages 2 to 4 left them (`D:\t2a\box2_kit\scenes`, `D:\t2v8f\scenes2`, `C:\t2v8f\scenes3`); it edits none of them.

## Markers in `D:\t2v8f\pull_gap48\` (exact names)
`GAP48.running` (liveness, rewritten after every scene: pid, done count, last scene), `preflight.done|.fail`, `scene_<scene8>.done|.fail`, `GAP48.done` (all 48 banked) | `GAP48.fail` (preflight refused or at least one scene not banked; the banked scenes stay).
A relaunch skips a scene whose marker, token file and recorded sha256 all agree (resume by content). Result: `GAP48_RESULT.json` (rewritten after every scene), `tokens/<scene8>.pt`.

## Preflight (refuses by name, before any scene)
The clips, the shipped code and the job file (size + sha256), the seven imported stage-1 files (sha256), `tree`, `v9`, the tool and the screen table, the kit files, every scene file's size, the sha256 of every small scene file,
`ref_frames.npz` per scene (the actor falsifier reads it), CUDA with at least 4 GiB free, `cl`/`nvcc`/`ninja`, DINOv3-L `model.safetensors` sha256, disk. The 26 GB of `volume.nurec` are checked by SIZE only unless the job is started with
`--hash-volumes`: stages 4 and 5 sha256-checked them on this box earlier today (INHERITED, not re-measured here). A read-only listing of box 2 on 2026-11-07 matched all 157 scene-file sizes of the manifest and found `ref_frames.npz` for all 48 (MEASURED).

## Reading the result (`read_gap48.py`)
Per scene and pooled: cosine (median, p10, p90) and relative L2 of render vs real, recomputed from the banked tokens; the cosine of the render tokens to the pinned eval-141 mean (`refs/blind_mean_eval141.pt` read out of
`stage1/t2v8f_stage1.tar`, tensor-bytes sha256 `cd6a7ab5...a2fcb9` asserted) against the real-frame-to-mean cosine, paired per tick; the yardstick of `gapscale/GAPSCALE.json` beside it. GAPSCALE could not give the render-to-mean cosine
("render tokens were not cached"); GAP48 banks them. The interval on a pooled median is a scene-cluster bootstrap: it answers "would another draw of scenes say this". The renderer is deterministic (stage-1 T5, bit-equal),
the encoder is frozen, and no policy or training run is in the loop, so the training-seed and inference-sampling variances do not enter. **Identity guard:** a scene whose render tokens equal its real tokens bit for bit on every tick
is flagged IDENTITY and excluded from the pooled figures; any such scene makes the run status IDENTITY_DETECTED. A stub or test-hook run is labelled and is not a measurement of the NuRec render.

## Launch and wall time
Launch: `HANDOFF_GAP48.md` has the exact command; the chain claims the box itself and refuses when `C:/Users/Admin/box2_ship/BOX2_BUSY.txt` or `D:/t2v8f/BOX2_BUSY.txt` exists or any `*.running` marker is live on box 2.
ESTIMATED from stage 1's MEASURED steps (preflight 145.7 s; gap 14.6 s for 2 scenes x 9 ticks including the encoder load): job ~16 min (range 10-28), chain ~25 min (range 15-45). The first scene's row in `GAP48_RESULT.json` replaces the estimate.

## Tests (MEASURED; see `logs/`)
Scene list = 62 minus R-SCN (count 48; changed inputs refuse); ticks; the one-pass read equals `read_ref_frame`; a CPU rehearsal with the stub renderer (`T2A_STUB=1`) on 2 fixture scenes through the real builder and the real tar
(markers, result, tokens, resume); the mutant that encodes the real frame as the render (cosine 1.0 on every tick, flagged by the reader); the refusals; the chain's refusal paths; prose.

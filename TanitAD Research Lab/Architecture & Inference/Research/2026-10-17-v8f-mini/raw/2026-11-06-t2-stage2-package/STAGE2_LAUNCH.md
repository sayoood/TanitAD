# STAGE2_LAUNCH: the deciding T2 rollouts on box 2, as an OVERLAY on the extracted stage-1 package (AGENT I, 2026-11-06)

Evidence: MEASURED = ours (CPU rehearsal, stage-1 pull); ESTIMATED = arithmetic from those; UNVERIFIED = needs box 2. Nothing here has been run on box 2. The frozen card
`motion_c/t2_prereg/PREREG_T2_CRAWL.md` is untouched. PI approval on file: "Approve T2 path" (box 2 runs ONE job at a time).

## 1. What ships
| item | value |
|---|---|
| overlay tar | `t2v8f/stage2/t2v8f_stage2_overlay.tar`, **81,920 bytes**, sha256 **8c94def60baa6140e701b15be553e5f67a100d8cafde2c55f511aaa5aafffe41** (`.sha256` beside it) |
| contents | NEW files only, under `pkg/`: `code/t2v8f_stage2.py`, `code/t2v8f_stage2_cell.py`, `code/t2v8f_exec_hold.py`, `jobs/stage2.cmd`, `MANIFEST2.json` (the plan: 42 scenes, 81 starts (28 launch, 53 moving: launch = logged v0 < 0.5 m/s), the size + sha256 of every scene file, the start classes). The 471 stage-1 files and their MANIFEST are untouched, so stage 1's preflight still verifies them |
| scenes | 41 scene folders, `volume.nurec` + `rig_trajectories.json` + `extracted/sequence_tracks.json` per scene (no mp4), copied file by file to `D:\t2v8f\scenes2\<id>\` with `sftp` (batch `stage2/scenes_put.sftp`, one explicit `put` per file; `make_stage2.py resume` writes the `reput` batch after an interruption). 7c72937c is already in box 2's kit from stage 1 and is verified there by sha256 (not copied) |
| cells | 588 = **420 deciding** (C0_s0, C0_s1, P10, ZERO, BLIND x {T2, RR} x 42 scenes) **then 168 beside** (DAGOACC_s0/_s1 x {T2, RR}); scene-major inside a pass, contrast-priority order inside a scene (X1 and X2 first). Executor `base` (the card). `hold` exists for a later dated amendment only and is NEVER run by this job file |
| records | stamped `deciding candidate` (no `--t2-plumbing`); never pooled with stage 1's plumbing rollouts |

## 2. Box-2 footprint on 1 MiB clusters (ESTIMATED from the MEASURED stage-1 cell of 6 MiB; the builder's exact figures)
| step | allocated | free after (ESTIMATED start: 35.23 GiB = stage 1's preflight 35.58 minus its install 160 MiB and pull 201 MiB; the gsplat JIT cache is UNVERIFIED and not counted) |
|---|---|---|
| 41 scenes copied (23.48 GB in bytes) | 22,556 MiB = 22.03 GiB | |
| overlay (tar 81,920 B + 5 files, kept) | 6 MiB | **13.19 GiB** after scenes + overlay |
| deciding pass (420 cells x 6 MiB + 168 scene directories + fixed 24 MiB) | 2,712 MiB = 2.65 GiB | **10.55 GiB** after the deciding pass: fits, margin 0.55 over the 10 GiB floor |
| beside pass (168 cells x 6 MiB + 2) | 1,010 MiB = 0.99 GiB | **9.56 GiB**: **SHORT by 0.44 GiB** against the floor. The driver refuses to start this pass if the real free space says so, and reports `DISK_LOW` with the shortfall |

Nothing is planned for deletion. PI candidates, all banked on the dev box: stage-1 tar 525 MiB (clears the shortfall by itself), stage-1 pull 201 MiB, kit scene 00040136.

The driver REFUSES TO START a pass that would end below the 10 GiB floor and names the shortfall (`passes_skipped_disk` in `STAGE2_RESULT.json`); it never dies mid-pass.
**If the beside pass refuses, that is reported, not worked around.** Freeing the 525 MiB stage-1 tar (or the 201 MiB stage-1 pull, or the kit's 00040136 scene) on box 2 is a PI decision; the Master Mind asks. All three are banked on the dev box. The launch step re-probes the real free space first.

## 3. Commands (dev box, git-bash)
```bash
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f

# a. the box must be free; claim it (a busy marker is the only lock)
ls C:/Users/Admin/box2_ship/BOX2_BUSY.txt 2>/dev/null && echo "BUSY: STOP"          # must print nothing
printf 'owner=T2V8F stage2\nstart=%s\neta=4-5 h\n' "$(date '+%F %T')" > C:/Users/Admin/box2_ship/BOX2_BUSY.txt

# b. folders, and the REAL free space (must exceed the footprint above plus the 10 GiB floor)
$SSH "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force D:/t2v8f/scenes2 | Out-Null; Get-PSDrive D | Select-Object Free\""
$SSH "powershell -NoProfile -Command \"(Get-ChildItem D:/t2v8f/pkg -Recurse -File | Measure-Object).Count\""        # 471 + stage-1 leftovers: stage 1 must be extracted

# c. the 41 scenes (~23.5 GB, ~16 min at the MEASURED 24 MB/s). Interrupted? Do NOT start over: list what is on box 2 and resume.
sftp -b $T/stage2/scenes_put.sftp tanitad-box2
$SSH "powershell -NoProfile -Command \"Get-ChildItem D:/t2v8f/scenes2 -Recurse -File | ForEach-Object { \$_.FullName + '|' + \$_.Length }\"" > C:/Users/Admin/box2_ship/scenes2_sizes.txt
python -I $T/code/make_stage2.py resume --remote-sizes C:/Users/Admin/box2_ship/scenes2_sizes.txt --out $T/stage2/scenes_resume.sftp     # complete files skipped, short ones reput, missing ones put
sftp -b $T/stage2/scenes_resume.sftp tanitad-box2                                                                      # repeat steps 'list' + 'resume' until it reports put 0 and reput 0

# d. the overlay (small): put, verify the bytes, extract OVER D:\t2v8f (adds files; replaces none)
sftp -b - tanitad-box2 <<EOF
put $T/stage2/t2v8f_stage2_overlay.tar /D:/t2v8f/t2v8f_stage2_overlay.tar
EOF
$SSH "certutil -hashfile D:\t2v8f\t2v8f_stage2_overlay.tar SHA256"      # 2nd line must equal: cut -d' ' -f1 $T/stage2/t2v8f_stage2_overlay.tar.sha256
$SSH "tar -xf D:\t2v8f\t2v8f_stage2_overlay.tar -C D:\t2v8f"
$SSH "powershell -NoProfile -Command \"Test-Path D:/t2v8f/pkg/MANIFEST2.json, D:/t2v8f/pkg/code/t2v8f_stage2.py, D:/t2v8f/pkg/jobs/stage2.cmd\""   # True True True

# e. launch, detached (the preflight hashes the 41 scenes + the kit scene by sha256 and names every failure; it does NOT need a separate verify step)
$SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\t2v8f\pkg\jobs\stage2.cmd -Log D:\t2v8f\logs\stage2.log"

# f. ONE background waiter, capped at 6 h (stops on STAGE2.done / STAGE2.fail, or at the cap); no model turns while it runs
( for i in $(seq 1 360); do $SSH "powershell -NoProfile -Command \"Get-ChildItem D:/t2v8f/pull2 -Filter 'STAGE2.*' -Name\"" 2>/dev/null | grep -qE 'STAGE2\.(done|fail)' && break; sleep 60; done ) 
$SSH "powershell -NoProfile -Command \"Get-Content D:/t2v8f/pull2/STAGE2.running, D:/t2v8f/logs/stage2.log -Tail 30\""        # live state: phase=... cell=k/588 pass=... last=<scene>|<T2|RR>|<arm>:<DONE|FAIL|SKIPPED_DONE>

# g. pull (588 cell dirs, ~3.6 GB allocated on box 2, far less in bytes) and release the box
mkdir -p $T/stage2/pull_box2
sftp -b - tanitad-box2 <<EOF
get -r /D:/t2v8f/pull2 $T/stage2/pull_box2
get /D:/t2v8f/logs/stage2.log $T/stage2/pull_box2/stage2.log
EOF
rm C:/Users/Admin/box2_ship/BOX2_BUSY.txt      # (move aside instead if you keep a record)

# h. score on the DEV BOX: pooled T2 score + every row per START CLASS (launch / moving), never pooled across classes
python -I $T/code/t2v8f_stage2_score.py --root $T/stage2/pull_box2/pull2/rollouts --sources $T/stage2/SCENE_SOURCES.json \
       --out $T/stage2/SCORE2.json --md $T/stage2/SCORE2.md        # also writes SCORE2_POOLED.json (the card's M1 contrast: `t2v8f_run.py score`, same arguments as stage 1)
```
Re-launching after any stop is safe: markers are renamed to `*.prev` (never deleted), a finished cell (`DONE`, with `executor=base` in its text) is skipped, a half-written `records.jsonl` is moved to `records.jsonl.prev`, and a `DONE` written under another executor REFUSES. Never copy a `.cmd` to box 2 while another queue job runs there.

## 4. The jobs and the markers (in `D:\t2v8f\pull2`)
| marker / file | meaning |
|---|---|
| `STAGE2.running` | heartbeat, rewritten every cell: `pid`, start time, `phase=<preflight|gate|rollouts> cell=k/588 pass=<deciding|beside> last=...` |
| `preflight.done|fail` | stage 1's preflight (471 files, CUDA, toolchain, DINOv3, kit) + the overlay files + MANIFEST2 bound to stage 1's MANIFEST + every scene file by size and sha256 + free disk. `fail` lists one named reason per failure: `MISSING`, `SIZE_MISMATCH`, `HASH_MISMATCH`, `DISK_WILL_RUN_LOW` |
| `gate.done|fail` | CPU, no render: the LOGGED plan played back by the T2 executor reproduces the logged future to 1e-6 m on every scene and start (card gate T2g); rig is 10 Hz, `n_ticks` equals the plan, `f0 >= 18` and `f0 + 60 < n_ticks`. `GATE_SCENES.json` has every row. A failed gate stops the run before any render |
| `rollouts.done|fail` | the cell loop. A FAILED cell never stops the others (`<cell>/FAIL`, its `error.txt` and the driver's `rollouts_<pass>.log`) |
| `STAGE2.done` / `STAGE2.fail` | `fail` = preflight refused or a job crashed (INFRASTRUCTURE_FAILURE). `done` carries the overall verdict: **COMPLETE** (588/588 DONE), **INCOMPLETE** (a cell failed, a pass refused on disk, or a scene was skipped), **GATE_FAILED** (a measured negative: no render ran). Only COMPLETE is a result to score for the card |
| `STAGE2_RESULT.json` | written after every job: gates, per-pass counts, `first_cells`, `passes_skipped_disk`, `scenes_skipped`, `executor`, the sha256 of MANIFEST2 |
| `rollouts/<scene8>/objects/<T2|RR>/<arm>/` | `DONE|FAIL`, `records.jsonl` (one start per line), `scene_<idx>.npz`, `actor_attach.json`, `rollouts_*.json` |

## 5. ESTIMATED wall time (stage 1: MEASURED 345 s / 28 cells = 12.3 s per cell at ONE start per scene)
| phase | estimate | basis |
|---|---|---|
| scene copy | ~16 min | 23.5 GB at the MEASURED 24 MB/s (549.65 MB in 23 s) |
| preflight | ~20 min | 24 GB of sha256 at stage 1's hashing rate (UNVERIFIED on box 2's D: for this volume) |
| CPU gate | ~5 min | 81 starts, no render |
| rollouts | **2.0 h up to 3.9 h** | 588 x 12.3 s = 2.0 h at one start per scene; x 1.93 starts per scene (81/42) = 3.9 h as the ceiling; BLIND renders nothing, so the real figure is lower |
| cold scene reads | +~15 min per pass | first touch of each 0.5 GB `volume.nurec` |
| **total, central** | **~4 h** (range 3 to 5.5 h) | the waiter cap is 6 h |

## 6. What is NOT covered (UNVERIFIED until box 2)
Real renders, the gsplat JIT, DINOv3 on the GPU, the 24 GB hash rate, `launch.ps1`, `tar`/`sftp` on this overlay, and the actor ROI thresholds on the 41 new scenes (the actor gate ran on 7c72937c only, in stage 1). Rehearsed on the CPU with the stub (`tests/test_t2v8f_stage2.py`): the whole control flow, the planted missing / changed scene file, DONE skipping, the executor-mismatch refusal, the gate going RED, and the resumable copy.

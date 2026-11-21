# STAGE2_RELAUNCH: put overlay 2 on box 2 and relaunch the SAME `stage2.cmd` (AGENT J, 2026-11-06)

Evidence: MEASURED = ours (dev-box CPU, or the first run's pull in `stage2/pull_box2/pull2`); ESTIMATED = arithmetic from those; UNVERIFIED = needs box 2. Nothing here has been run on box 2.
The frozen card `motion_c/t2_prereg/PREREG_T2_CRAWL.md` is untouched. PI approval on file: "Approve T2 path"; box 2 runs ONE job at a time.

## 0. READ FIRST: the relaunch will be REFUSED by the 10 GiB floor unless box 2 gets 0.8 to 1.6 GiB back (ESTIMATED; a PI decision)
| line | GiB | basis |
|---|---|---|
| D: free at the first run's preflight, after the scenes and overlay 1 | 13.55 | MEASURED, `STAGE2_RESULT.json` `free_gib_pull_after_scenes` |
| written by that run afterwards: 653 files + 757 directories in `pull2` (574 `FAIL` cells, 14 DONE) at 1 MiB each | -1.38 | ESTIMATED from the pull (`pull2` tree = 1,410 MiB allocated) |
| **free now** | **12.17** | ESTIMATED; the gsplat JIT cache and anything else written since are not counted |
| overlay 2: 41 npz (434,028,861 B = 432 MiB allocated) + the tar (434,155,520 B = 415 MiB) | -0.83 | MEASURED sizes, 1 MiB clusters |
| **free after overlay 2** | **11.35** (11.75 if the tar is removed after its extraction is verified) | ESTIMATED |
| the driver's need for the deciding pass: the manifest's 2,712 MiB x 410 cells left of 420 | 2.59 | `pass_need_mib` (new: a relaunch pays only for cells with no DONE) |
| the preflight needs free >= floor 10 + need | **12.59** | |
| **shortfall before the deciding pass** | **1.24 (tar kept) / 0.84 (tar removed)** | |
| and for the beside pass to start afterwards (free then = start free - 2.03 actual - floor 10 >= 0.96) | start free >= 12.99 | shortfall **1.64 / 1.24** |

The driver REFUSES, by its own rule, in SECONDS now (the disk check runs before the 24 GB sha256 pass; stage 2's first preflight spent 769 s). So a wrong guess costs nothing: put overlay 2, launch, read `STAGE2.fail`.
**Decide before the launch (PI):** EITHER free about 1.7 GiB on box 2, all banked on the dev box: stage-1 tar 525 MiB, stage-1 pull 201 MiB, the overlay-2 tar 415 MiB after `certutil` and extraction are verified, and the 574 `FAIL` stubs of run 1 (574 MiB; the driver would otherwise rename each to `FAIL.prev`, same cluster, and keep it); OR lower the floor for this run (`--min-free-gb 8` in `stage2.cmd`: it is hashed in MANIFEST2, so that is a rebuild of overlay 2 by this agent's `make_overlay2.py`, ask for it). Nothing is deleted by this document.

## 1. What ships (overlay 2, `t2v8f/stage2/overlay2/`)
| item | value |
|---|---|
| tar | `t2v8f_stage2_overlay2.tar`, **434,155,520 bytes**, sha256 **afa9dba7ff6c11b8f27c325fa829bf8ddd167431997536c13eb5f4aa5ecf2a50** (`.sha256` beside it) |
| members (43) | `pkg/MANIFEST2.json` (v2, replaces v1), `pkg/code/t2v8f_stage2.py` (the fail-closed driver, replaces v1), `scenes2/<id>/ref_frames.npz` x 41 (NEW; frames 0, 60, 120 of each scene's mp4, uint8 3840x2160x3). **No mp4 is shipped.** |
| not in the tar | the kit scene 7c72937c keeps the npz stage 1 installed (its sha256 is now in MANIFEST2 and checked by the preflight) |
| driver sha256 | 33b65b19a563b8fe275de5e46e0f569db6487bd7fdf40b7eab34e021094f4898 |
| MANIFEST2 v2 sha256 | 563b1d5753a8abba14944f560ad1630f0132d2b863f5914cc56fecdae28c9a9b (supersedes v1 07722a0c..., the one the first run recorded) |
| what the new preflight REFUSES, and names | `MISSING_REF_FRAMES`, `HASH_MISMATCH`, `SIZE_MISMATCH`, `MISSING_FRAMES` (an npz without f0/f60/f120), `NO_REF_FRAMES_IN_MANIFEST2`, one line per scene, for the 41 copied scenes and the kit scene. It does so whatever `--scene-policy` says, and before it hashes the 24 GB of scenes |

## 2. Commands (dev box, git-bash; the same conventions as `STAGE2_LAUNCH.md` section 3)
```bash
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f
O=$T/stage2/overlay2

# a. the box must be free; claim it
ls C:/Users/Admin/box2_ship/BOX2_BUSY.txt 2>/dev/null && echo "BUSY: STOP"          # must print nothing
printf 'owner=T2V8F stage2 relaunch\nstart=%s\neta=3-5 h\n' "$(date '+%F %T')" > C:/Users/Admin/box2_ship/BOX2_BUSY.txt

# b. the REAL free space; compare with section 0 (needs >= 12.99 GiB after the overlay for both passes, 12.59 for the deciding pass alone)
$SSH "powershell -NoProfile -Command \"[math]::Round((Get-PSDrive D).Free/1GB,2)\""

# c. put the overlay (434 MB, ~18 s at the MEASURED 24 MB/s), verify the bytes
sftp -b - tanitad-box2 <<EOF
put $O/t2v8f_stage2_overlay2.tar /D:/t2v8f/t2v8f_stage2_overlay2.tar
EOF
$SSH "certutil -hashfile D:\t2v8f\t2v8f_stage2_overlay2.tar SHA256"        # 2nd line must equal: cut -d' ' -f1 $O/t2v8f_stage2_overlay2.tar.sha256

# d. move the two files overlay 2 replaces ASIDE (nothing is deleted), then extract over D:\t2v8f
$SSH "powershell -NoProfile -Command \"Move-Item D:/t2v8f/pkg/MANIFEST2.json D:/t2v8f/pkg/MANIFEST2.json.ovl1; Move-Item D:/t2v8f/pkg/code/t2v8f_stage2.py D:/t2v8f/pkg/code/t2v8f_stage2.py.ovl1\""
$SSH "tar -xf D:\t2v8f\t2v8f_stage2_overlay2.tar -C D:\t2v8f"
$SSH "powershell -NoProfile -Command \"(Get-ChildItem D:/t2v8f/scenes2 -Recurse -Filter ref_frames.npz | Measure-Object).Count; (Get-FileHash D:/t2v8f/pkg/code/t2v8f_stage2.py).Hash\""   # 41, then 33B65B19...F4898
#    the stage-2 log of run 1 is kept: move it aside so the log of run 2 is its own file
$SSH "powershell -NoProfile -Command \"Move-Item D:/t2v8f/logs/stage2.log D:/t2v8f/logs/stage2.run1.log\""

# e. relaunch the SAME job file, detached. DONE cells (14 from run 1) are skipped; the 574 FAIL cells are renamed FAIL.prev and re-run.
$SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\t2v8f\pkg\jobs\stage2.cmd -Log D:\t2v8f\logs\stage2.log"

# f. CANARY (about 3 min after the launch, once the preflight has passed; ~14 min when it hashes the 24 GB): the first NEW-scene cell must be DONE, not FAIL.
$SSH "powershell -NoProfile -Command \"Get-Content D:/t2v8f/pull2/STAGE2.running; Get-Content D:/t2v8f/pull2/preflight.fail,D:/t2v8f/pull2/preflight.done -ErrorAction SilentlyContinue\""
#    phase=rollouts ... last=e848c843|T2|C0_s0:DONE   <- the failure of run 1 would read :FAIL here. A preflight.fail names every refused scene (section 1) and nothing ran.

# g. ONE background waiter, capped at 6 h (stops on STAGE2.done / STAGE2.fail); no model turns while it runs
( for i in $(seq 1 360); do $SSH "powershell -NoProfile -Command \"Get-ChildItem D:/t2v8f/pull2 -Filter 'STAGE2.*' -Name\"" 2>/dev/null | grep -qE 'STAGE2\.(done|fail)' && break; sleep 60; done )

# h. pull into its OWN folder (run 1's pull stays as the record), release the box
mkdir -p $T/stage2/pull_box2_run2
sftp -b - tanitad-box2 <<EOF
get -r /D:/t2v8f/pull2 $T/stage2/pull_box2_run2
get /D:/t2v8f/logs/stage2.log $T/stage2/pull_box2_run2/stage2.log
EOF
rm C:/Users/Admin/box2_ship/BOX2_BUSY.txt      # (move aside instead if you keep a record)

# i. score on the DEV BOX, per START CLASS (launch / moving), never pooled across classes. Only COMPLETE (588/588 DONE) is a result for the card.
python -I $T/code/t2v8f_stage2_score.py --root $T/stage2/pull_box2_run2/pull2/rollouts --sources $T/stage2/SCENE_SOURCES.json \
       --out $T/stage2/SCORE2.json --md $T/stage2/SCORE2.md        # and `t2v8f_run.py score` for the card's M1 contrast, same arguments as stage 1
```
A stop at any point is safe: markers become `*.prev`, a DONE cell is skipped, a half-written `records.jsonl` is moved to `.prev`, a DONE under another executor REFUSES.

## 3. ESTIMATED wall time (stage 1 MEASURED 345 s / 28 cells = 12.3 s per cell at ONE start per scene; stage 2's first run MEASURED preflight 769 s, gate 21 s)
| phase | estimate | basis |
|---|---|---|
| put + certutil + extract | ~1 min | 434 MB at 24 MB/s |
| preflight | ~13 min (seconds if it refuses) | MEASURED 769 s on the first run (603 s of it the 24 GB hash) |
| gate | 0.4 min | MEASURED 21 s |
| rollouts | **2.0 h to 3.8 h** | 574 cells left x 12.3 s = 1.96 h; x 1.93 starts per scene = 3.78 h as the ceiling (BLIND renders nothing, so the real figure is lower) |
| cold scene reads | +~0.5 h | ~15 min per pass, the first touch of each 0.5 GB `volume.nurec` |
| **total, central** | **~2.7 h** (range 2.5 to 4.5 h) | the waiter cap is 6 h. The 4,203 s of the first run was 574 cells that failed in ~7 s each, plus 14 that ran |

## 4. UNVERIFIED until box 2
The free-space figures in section 0 (a `Get-PSDrive D` before the put settles them); `tar -xf` replacing the two files on exFAT; that the real renderer's actor falsifier reads these npz exactly as the dev-box REAL reader does (`tests/test_t2v8f_stage2_refs.py::real_caller`: the kit's own `actor_map.falsify_actors`, a fake renderer, no mp4; stage 1's actor gate already read the same kind of npz for 7c72937c on box 2). The 4K frame is resized to the renderer's size on box 2 by `resize_area` (PIL BOX), as in stage 1.

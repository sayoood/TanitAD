# STAGE3_LAUNCH: the 20 guard scenes of PREREG_T2_CRAWL on box 2 (AGENT K, 2026-11-06)

Evidence: MEASURED = ours (dev-box CPU, files on disk); ESTIMATED = arithmetic from those and from stage 1 / stage 2's MEASURED figures; UNVERIFIED = needs box 2. Nothing here has been run on box 2.
The frozen card `motion_c/t2_prereg/PREREG_T2_CRAWL.md` is untouched. PI approval on file: "Approve T2 path". Box 2 runs ONE job at a time: **stage 2 must be finished (STAGE2.done, exact name) before this starts.**
Definitions of the guard draw: `DEFINITIONS_K.md` (written before the draw). Stage 2's driver (sha256 33b65b19...f4898) is NOT changed; stage 3 is a NEW file beside it.

## 0. What ships, and what it costs (all from `stage3/BUILD_RESULT3.json`, MEASURED on the dev box)
| item | value |
|---|---|
| overlay tar | `stage3/t2v8f_stage3_overlay.tar`, **51,200 bytes**, sha256 **bd96fca1c8d4c5c4cd1004c7256d54a85c3420e822a6c9297388b2be97dab840** (`.sha256` beside it) |
| members (3, all NEW; nothing on box 2 is replaced) | `pkg/code/t2v8f_stage3.py` (sha256 0688bb4732d8ef54f19e318de08008a54ce9276b66c815331e797b04019f733c), `pkg/jobs/stage3.cmd`, `pkg/MANIFEST3.json` (sha256 be113ea9c3592dc42f34520f08cb94d95bbd123ac94296358148a904f5c3e3c0) |
| scenes, by sftp (`stage3/scenes_put3.sftp`) | 20 scenes x {volume.nurec, rig_trajectories.json, extracted/sequence_tracks.json, ref_frames.npz} = 80 files, **11,583,891,171 B scene files + 192,554,391 B npz = 11,776,445,562 B (10.97 GiB)** into `C:\t2v8f\scenes3\<id>\` |
| starts | 40 (two per scene, none collapsed), `starts_sha256` 34c35cfe...f512, all class **moving** (0 launch): the guard draw lands on 10 to 30 m/s starts |
| cells | 200 deciding (20 scenes x 10) then 80 beside (DAGOACC), 280 in all, `--executor base`, `objects` agents |
| stage-2 files the driver needs (`requires`, sha256-pinned in MANIFEST3, checked by the preflight) | `code/t2v8f_stage2.py` 33b65b19...f4898, `code/t2v8f_stage2_cell.py` cd5bda37..., `code/t2v8f_exec_hold.py` c04ceb65... |

**Box-2 footprint per drive (ESTIMATED: stage 1's MEASURED 6 MiB per cell on 1 MiB-cluster exFAT; NTFS 4 KiB clusters on C:):**
| drive | what | MiB |
|---|---|---|
| **C:** (NTFS) | `C:\t2v8f\scenes3` (scenes + npz) | 11,231 (10.97 GiB). C: had ~29.5 GiB free: ~18.5 GiB left |
| **D:** (exFAT) | the overlay tar + its 3 extracted files | 4 |
| **D:** | pull, deciding pass | 1,304 |
| **D:** | pull, beside pass | 482 |
| **D:** | total | 1,790 |

**The floor is the PULL drive's (D:).** The driver refuses unless `D: free - (this pass's need) >= 10 GiB` (`--min-free-gb 10`, the default): **D: free must be >= 11.28 GiB for the deciding pass and >= 11.75 GiB for both passes** (these include the 4 MiB overlay). The driver never checks C: for the floor (it writes nothing there). `Get-PSDrive D` before the launch settles it; a refusal takes seconds and writes `preflight.fail` naming the shortfall. If D: is short, that is a PI decision (free space on D:, or lower the floor, which is a rebuild of `stage3.cmd`); nothing is deleted by this document.

## 1. Commands (dev box, git-bash; the conventions of `STAGE2_RELAUNCH.md`)
```bash
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f
S=$T/stage3

# a. stage 2 must be DONE and the box free; claim it
$SSH "powershell -NoProfile -Command \"if (Test-Path D:/t2v8f/pull2/STAGE2.done -PathType Leaf) {'MK_DONE'} elseif (Test-Path D:/t2v8f/pull2/STAGE2.fail -PathType Leaf) {'MK_FAIL'} else {'NOT_FINISHED'}; Test-Path D:/t2v8f/pull2/STAGE2.running\"" | tr -d '\r'   # MK_DONE (or a MK_FAIL you have read), then False
ls C:/Users/Admin/box2_ship/BOX2_BUSY.txt 2>/dev/null && echo "BUSY: STOP"          # must print nothing
printf 'owner=T2V8F stage3\nstart=%s\neta=2-3 h\n' "$(date '+%F %T')" > C:/Users/Admin/box2_ship/BOX2_BUSY.txt

# b. the REAL free space on both drives; compare with section 0 (D: >= 11.75 GiB for both passes; C: >= 11 GiB for the scenes plus a margin)
$SSH "powershell -NoProfile -Command \"[math]::Round((Get-PSDrive D).Free/1GB,2); [math]::Round((Get-PSDrive C).Free/1GB,2)\""

# c. the overlay (51 KB): put on D:, verify the bytes, extract (3 NEW files, nothing replaced)
sftp -b - tanitad-box2 <<EOF
put $S/t2v8f_stage3_overlay.tar /D:/t2v8f/t2v8f_stage3_overlay.tar
EOF
$SSH "certutil -hashfile D:\t2v8f\t2v8f_stage3_overlay.tar SHA256"        # 2nd line must equal: cut -d' ' -f1 $S/t2v8f_stage3_overlay.tar.sha256  (bd96fca1...)
$SSH "tar -xf D:\t2v8f\t2v8f_stage3_overlay.tar -C D:\t2v8f"
$SSH "powershell -NoProfile -Command \"(Get-FileHash D:/t2v8f/pkg/code/t2v8f_stage3.py).Hash; (Get-FileHash D:/t2v8f/pkg/code/t2v8f_stage2.py).Hash; Test-Path D:/t2v8f/pkg/MANIFEST3.json,D:/t2v8f/pkg/jobs/stage3.cmd\""   # 0688BB47...; 33B65B19... (unchanged); True True

# d. the 20 scenes, 11.78 GB (~8 min at the MEASURED ~24 MB/s), resumable: `put` of a whole file is restarted by `reput` below
sftp -b $S/scenes_put3.sftp tanitad-box2
#    interrupted? list what box 2 has, build the resume batch (complete files skipped, short ones `reput`), run it:
$SSH "powershell -NoProfile -Command \"Get-ChildItem C:/t2v8f/scenes3 -Recurse -File | ForEach-Object { \$_.FullName + '|' + \$_.Length }\"" | tr -d '\r' > $S/remote_sizes3.txt
python -I $T/code/make_stage3.py resume --remote-sizes $S/remote_sizes3.txt --out $S/scenes_resume3.sftp && sftp -b $S/scenes_resume3.sftp tanitad-box2

# e. launch the job (the preflight sha256-checks all 80 files, refuses by name, and checks the stage-2 heartbeat is gone)
$SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\t2v8f\pkg\jobs\stage3.cmd -Log D:\t2v8f\logs\stage3.log"

# f. CANARY (~6 min after the launch: the preflight hashes 11.8 GB): the first cell must be DONE, not FAIL
$SSH "powershell -NoProfile -Command \"Get-Content D:/t2v8f/pull3/STAGE3.running; Get-Content D:/t2v8f/pull3/preflight.fail,D:/t2v8f/pull3/preflight.done -ErrorAction SilentlyContinue\""
#    phase=rollouts ... last=829c55af|T2|C0_s0:DONE   <- a FAIL here: read D:/t2v8f/pull3/rollouts_deciding.log and the cell's error.txt before anything else
```

### g. The waiter: EXACT marker names, never a listing pattern (MEASURED trap, Master Mind, 2026-11-06 16:52)
The driver renames an old `STAGE3.done` to `STAGE3.done.prev` at start. A waiter that greps `STAGE3\.(done|fail)` in a `Get-ChildItem -Filter 'STAGE3.*'` listing MATCHES the `.prev` file and stops at once (stage 2's run-2 waiter did, and pulled a partial snapshot). Use `Test-Path <exact name> -PathType Leaf` and `grep -x`:
```bash
st=""
for i in $(seq 1 360); do                                      # cap 6 h, no model turns while it runs
  st=$($SSH "powershell -NoProfile -Command \"if (Test-Path D:/t2v8f/pull3/STAGE3.done -PathType Leaf) {'MK_DONE'} elseif (Test-Path D:/t2v8f/pull3/STAGE3.fail -PathType Leaf) {'MK_FAIL'}\"" 2>/dev/null | tr -d '\r' | grep -xE 'MK_(DONE|FAIL)' | head -1)
  [ -n "$st" ] && break
  sleep 60
done
echo "marker=${st:-TIMEOUT}"
```
**The control (MEASURED on the dev box, a directory holding only `STAGE3.done.prev`):** the WRONG pattern (`Get-ChildItem -Filter 'STAGE3.*' -Name | grep -E 'STAGE3\.(done|fail)'`) printed `STAGE3.done.prev` and matched; the exact-name expression printed nothing; with a real `STAGE3.done` beside it printed `MK_DONE`. Re-run it before trusting a waiter: create `STAGE3.done.prev` in a scratch folder, run the waiter's `if (Test-Path ...)` against that folder, and it must print nothing.

```bash
# h. pull into its OWN folder (nothing of stage 2 is touched), then release the box
[ -e $S/pull_box2 ] && echo "TARGET EXISTS: STOP"
mkdir -p $S/pull_box2
sftp -b - tanitad-box2 <<EOF
get -r /D:/t2v8f/pull3 $S/pull_box2
get /D:/t2v8f/logs/stage3.log $S/pull_box2/stage3.log
EOF
rm C:/Users/Admin/box2_ship/BOX2_BUSY.txt      # (move aside instead if you keep a record)
```
Markers in `pull3`: `STAGE3.running` (heartbeat; its text is `phase=... last=<scene8>|<T2|RR>|<arm>:<DONE|FAIL|SKIPPED_DONE>`), `preflight|gate|rollouts.done|fail`, `STAGE3.done` (all jobs ran) or `STAGE3.fail` (a job REFUSED or CRASHED), and `STAGE3_RESULT.json` (`overall`: COMPLETE, INCOMPLETE (a cell failed or a pass was skipped for disk), GATE_FAILED, INFRASTRUCTURE_FAILURE). **Only COMPLETE (280 of 280 cells DONE) is a result for the card.** A stop at any point is safe: markers become `*.prev`, a DONE cell is skipped, a half-written `records.jsonl` is moved to `.prev`, a DONE under another executor REFUSES.

## 2. The score (dev box): the 62 scenes, per start class, never pooled across classes
Card section 3: M1 is read over the LAUNCH scenes (stage 2); the four families over ALL 62 (stage 2 + stage 3). `t2v8f_stage3_score.py` writes two products and never mixes them:
```bash
python -I $T/code/t2v8f_stage3_score.py --root2 $T/stage2/pull_box2_run2/pull2/rollouts --root3 $S/pull_box2/pull3/rollouts \
       --sources2 $T/stage2/SCENE_SOURCES.json --sources3 $S/SCENE_SOURCES3.json --out-dir $S/score
#   M1_LAUNCH42.json/.md   = the stage-2 scorer on the 42 launch scenes alone (the card's M1 contrast X1, X2 and the lateral rows, per class)
#   FAMILIES_ALL62.json/.md = every row per class (launch / moving) over the pool of both stages; its *_POOLED.json sibling is the reader's all-start mean and is NOT a card quantity
```
Use the FINAL complete stage-2 pull for `--root2`. The pool refuses a scene idx in both roots (stage 2 is idx 0..41, stage 3 is 42..61), the same cell path in both, or an idx outside its stage's range. All 40 guard starts are `moving`, so the guard scenes add to the `moving` class only; a `launch` family row rests on the launch scenes alone. A class with n < 5 starts is a description, not an estimate. Each family row carries its estimator (paired scene-cluster bootstrap, B 2000, seed 0, the reader's own) and answers *another draw of scenes* only.

## 3. ESTIMATED wall time (stage 1 MEASURED 345 s / 28 cells = 12.3 s per cell at ONE start per scene; stage 2's first run MEASURED preflight 769 s of which 603 s hashed 24 GB, gate 21 s for 81 starts)
| phase | estimate | basis |
|---|---|---|
| put + certutil + extract | ~9 min | 11.78 GB at ~24 MB/s, + 51 KB |
| preflight | ~5 min | 11.8 GB x (603 s / 24 GB); seconds if it refuses |
| gate | ~0.2 min | 40 starts x (21 s / 81) |
| rollouts | **57 min to 1.9 h** | 280 cells x 12.3 s = 57 min; x 2 starts per scene = 1.9 h as the ceiling (BLIND renders nothing, so the real figure is lower) |
| cold scene reads | +~0.3 h | the first touch of each 0.5 GB `volume.nurec` |
| **total, central** | **~1.5 h** (range 1.3 to 2.6 h) | the waiter cap is 6 h |

## 4. UNVERIFIED until box 2
The free-space figures (section 0; `Get-PSDrive` settles them); `tar -xf` writing three new files on exFAT; the sftp path `/C:/t2v8f/scenes3/...` (the same syntax as stage 2's `/D:/...`); that the real renderer's actor falsifier reads these npz as the dev-box REAL reader does (`tests/test_t2v8f_stage3.py::check_real_guard`: the kit's own `falsify_actors`, a fake renderer, a real 4K guard npz resized by `resize_area`; stage 2 run 2's first new-scene cell is DONE on box 2, MM message); and the falsifier's verdict per guard scene (a scene whose frames 0, 60 and 120 fail the falsifier gets its actors switched off by the kit, `attach_actors_verified`: read `actor_attach.json` of each cell).

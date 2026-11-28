# STAGE4_LAUNCH: the composite arm C0C over the 62 scenes on box 2 (AGENT O, continuing N, 2026-11-06)

Evidence: MEASURED = ours (dev-box CPU, files on disk); ESTIMATED = arithmetic from those and from stage 1 / 2 / 3's MEASURED figures; UNVERIFIED = needs box 2. Nothing here has been run on box 2.
The frozen cards in `motion_c/t2_prereg/` are untouched (outcome 1 of `PREREG_T2_CRAWL.md` section 6, the read of `PREREG_T2_DOMINANCE.md`). Box 2 runs ONE job at a time: **stage 2 AND stage 3 must both be finished (`STAGE2.done`, `STAGE3.done`, exact names) before this starts.**
Frozen and NOT changed by this package: the stage-2 driver (sha256 33b65b19...f4898), the stage-3 driver (0688bb47...3c), the pinned read script `code/read_t2_dominance.py` (6345d4c7...2f5c). Stage 4 is a NEW driver beside them (it imports the stage-2 driver; the stage-2 cell and driver never import stage 4).
Gate C0C-ID (the arm is the T1-kin composite, three RED arms RED, off/off reproduces C0): PASS, `C0C_ID_RESULT.json` (sha256 c8df31eb...).

## 0. What ships, and what it costs (all from `stage4/BUILD_RESULT4.json`, MEASURED on the dev box)
| item | value |
|---|---|
| overlay tar | `stage4/t2v8f_stage4_overlay.tar`, **204,800 bytes**, sha256 **bbd4b8b04272d2d8da70909039c689b941ab17a76cf041bec24887bd337f3594** (`.sha256` beside it) |
| members (6, all NEW; nothing on box 2 is replaced) | `pkg/MANIFEST4.json` (sha256 b186fcffceaa05efec398ad69d0f4529e265e535b87112fa280cd1aea55cb3d2), `pkg/code/t2v8f_stage4.py` (588e764264cd...c87), `pkg/code/t2v8f_stage4_cell.py` (7a5c863cf9a2...5fa), `pkg/code/t2v8f_c0c.py` (1a78317951ae...62bc), `pkg/jobs/stage4.cmd` (c7aa6346296a...5bd), `pkg/refs/tool/v8f_t1_eval_ri.py` (**76,704 B**, a byte copy of the T1-kin RI tool, sha256 0417c838355a...61e7) |
| scenes | **NONE ships and none is put.** They are on box 2 already: the kit scene 7c72937c in `D:\t2a\box2_kit\scenes`, 41 scenes in `D:\t2v8f\scenes2`, the 20 guard scenes in `C:\t2v8f\scenes3`. The preflight sha256-checks 186 files (35,671,549,099 B = 33.22 GiB) against MANIFEST4 (which carries MANIFEST2's and MANIFEST3's file lists verbatim) and REFUSES, named, if `scenes3` (or `scenes2`) is absent or a byte differs |
| starts | 121 = stage 2's 81 + stage 3's 40, the SAME (scene idx, f0) keys P10(T2) ran on (the rehearsal asserts MANIFEST4's keys equal them; the pinned reader re-checks) |
| cells | **124 deciding** (T2 x {C0C_s0, C0C_s1} x 62 scenes) **then 124 beside** (RR x the same), 248 in all, `--executor base`, `objects` agents |
| stage-2 files the driver needs (`requires`, sha256-pinned in MANIFEST4, checked by the preflight) | `code/t2v8f_stage2.py` 33b65b19...f4898, `code/t2v8f_stage2_cell.py` cd5bda37..., `code/t2v8f_exec_hold.py` c04ceb65... |
| the arm | C0's checkpoint (`ckpt/C0_s0.pt` / `C0_s1.pt`, already in the stage-1 package), then on every replan `feasible_plan(smooth_plan(raw, prior, "sg72"), "kbox")` from the pinned RI tool (loaded by path, sha256-checked); the next frame is rendered at the EXECUTED pose |

**Box-2 footprint per drive (ESTIMATED: stage 1's MEASURED 6 MiB per cell on 1 MiB-cluster exFAT; a stub C0C cell costs exactly that, `tests/test_t2v8f_stage4.py::check_footprint`):**
| drive | what | MiB |
|---|---|---|
| **C:** (NTFS) | nothing new: `C:\t2v8f\scenes3` is READ only and already there | 0 |
| **D:** (exFAT) | the overlay tar + its 5 extracted files | 7 |
| **D:** | pull `D:\t2v8f\pull4`, deciding pass (124 x 6 + 62 x 4 + 24) | 1,016 |
| **D:** | pull, beside pass (124 x 6 + 2) | 746 |
| **D:** | total | 1,769 |

**The floor is the PULL drive's (D:).** The driver refuses unless `D: free - (this pass's need) >= 10 GiB` (`--min-free-gb 10`, the default): **D: free must be >= 11.00 GiB for the deciding pass and >= 11.73 GiB for both passes** (these include the 7 MiB overlay). A refusal takes seconds and writes `preflight.fail` naming the shortfall. If D: is short, that is a PI decision (free space on D:, or a lower floor, which is a rebuild of `stage4.cmd`); nothing is deleted by this document. If the beside pass alone refuses, that is REPORTED (the deciding cells stay valid), never worked around.

## 1. Commands (dev box, git-bash; the conventions of `STAGE3_LAUNCH.md`)
```bash
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f
S=$T/stage4

# a. stage 2 AND stage 3 must be DONE, neither still running, and the box free; claim it
for n in 2 3; do
  $SSH "powershell -NoProfile -Command \"if (Test-Path D:/t2v8f/pull$n/STAGE$n.done -PathType Leaf) {'MK_DONE'} elseif (Test-Path D:/t2v8f/pull$n/STAGE$n.fail -PathType Leaf) {'MK_FAIL'} else {'NOT_FINISHED'}; Test-Path D:/t2v8f/pull$n/STAGE$n.running\"" | tr -d '\r'
done
#    must print, for each of the two stages: MK_DONE then False   (a MK_FAIL you have read and accepted is a PI call; a True = a job is still running: STOP)
$SSH "powershell -NoProfile -Command \"Test-Path C:/t2v8f/scenes3, D:/t2v8f/scenes2\"" | tr -d '\r'                 # True True: the two scene roots stage 4 reads (the preflight refuses by name otherwise)
ls C:/Users/Admin/box2_ship/BOX2_BUSY.txt 2>/dev/null && echo "BUSY: STOP"                                       # must print nothing
printf 'owner=T2V8F stage4\nstart=%s\neta=1.6-2.5 h\n' "$(date '+%F %T')" > C:/Users/Admin/box2_ship/BOX2_BUSY.txt

# b. the REAL free space on D: (the pull drive); compare with section 0 (>= 11.73 GiB for both passes, >= 11.00 for the deciding pass alone)
$SSH "powershell -NoProfile -Command \"[math]::Round((Get-PSDrive D).Free/1GB,2)\""

# c. the overlay (205 KB; NO scene is put): put on D:, verify the bytes, extract (6 NEW files, nothing replaced)
sftp -b - tanitad-box2 <<EOF
put $S/t2v8f_stage4_overlay.tar /D:/t2v8f/t2v8f_stage4_overlay.tar
EOF
$SSH "certutil -hashfile D:\t2v8f\t2v8f_stage4_overlay.tar SHA256"        # 2nd line must equal: cut -d' ' -f1 $S/t2v8f_stage4_overlay.tar.sha256  (bbd4b8b0...)
$SSH "tar -xf D:\t2v8f\t2v8f_stage4_overlay.tar -C D:\t2v8f"
$SSH "powershell -NoProfile -Command \"(Get-FileHash D:/t2v8f/pkg/code/t2v8f_stage4.py).Hash; (Get-FileHash D:/t2v8f/pkg/code/t2v8f_stage4_cell.py).Hash; (Get-FileHash D:/t2v8f/pkg/code/t2v8f_c0c.py).Hash; (Get-FileHash D:/t2v8f/pkg/refs/tool/v8f_t1_eval_ri.py).Hash; (Get-FileHash D:/t2v8f/pkg/code/t2v8f_stage2.py).Hash; (Get-Item D:/t2v8f/pkg/refs/tool/v8f_t1_eval_ri.py).Length; Test-Path D:/t2v8f/pkg/MANIFEST4.json,D:/t2v8f/pkg/jobs/stage4.cmd\""
#    588E7642...C87; 7A5C863C...5FA; 1A783179...62BC; 0417C838...61E7; 33B65B19... (the stage-2 driver, UNCHANGED); 76704; True True

# d. NO scene put. (If a scene root is missing the preflight REFUSES, named: SCENES3_ABSENT / SCENES2_ABSENT. Do not re-put scenes without the PI: stage 3's `scenes_put3.sftp` is the way, 11.78 GB.)

# e. launch the job (the preflight sha256-checks all 186 scene files, refuses by name, and checks the stage-2 / stage-3 heartbeats are gone)
$SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\t2v8f\pkg\jobs\stage4.cmd -Log D:\t2v8f\logs\stage4.log"

# f. CANARY (~15 min after the launch: the preflight hashes 33.2 GiB, ESTIMATED from stage 2's 603 s for 24 GB): the first cell must be DONE, not FAIL
$SSH "powershell -NoProfile -Command \"Get-Content D:/t2v8f/pull4/STAGE4.running; Get-Content D:/t2v8f/pull4/preflight.fail,D:/t2v8f/pull4/preflight.done -ErrorAction SilentlyContinue\""
#    phase=rollouts ... last=e848c843|T2|C0C_s0:DONE   <- a FAIL here: read D:/t2v8f/pull4/rollouts_deciding.log and the cell's error.txt before anything else
#    a refusal that names MISSING_CKPT / HASH_MISMATCH_RI_TOOL / HASH_MISMATCH_STAGE2_FILE means the package on box 2 is not the one built here: re-run the hash line of step c.
```

### g. The waiter: EXACT marker names, never a listing pattern (MEASURED trap, Master Mind, 2026-11-06 16:52)
The driver renames an old `STAGE4.done` to `STAGE4.done.prev` at start. A waiter that greps `STAGE4\.(done|fail)` in a `Get-ChildItem -Filter 'STAGE4.*'` listing MATCHES the `.prev` file and stops at once (stage 2's run-2 waiter did, and pulled a partial snapshot). Use `Test-Path <exact name> -PathType Leaf` and `grep -x`:
```bash
st=""
for i in $(seq 1 360); do                                      # cap 6 h, no model turns while it runs
  st=$($SSH "powershell -NoProfile -Command \"if (Test-Path D:/t2v8f/pull4/STAGE4.done -PathType Leaf) {'MK_DONE'} elseif (Test-Path D:/t2v8f/pull4/STAGE4.fail -PathType Leaf) {'MK_FAIL'}\"" 2>/dev/null | tr -d '\r' | grep -xE 'MK_(DONE|FAIL)' | head -1)
  [ -n "$st" ] && break
  sleep 60
done
echo "marker=${st:-TIMEOUT}"
```
**The control (MEASURED on the dev box, 2026-11-06 by AGENT O, scratch folder `C:/Users/Admin/t2a_scratch/waiter_control4` holding only `STAGE4.done.prev`):** the WRONG pattern (`Get-ChildItem -Filter 'STAGE4.*' -Name | Select-String 'STAGE4\.(done|fail)'`) printed `STAGE4.done.prev` and matched; the exact-name expression (`Test-Path <folder>/STAGE4.done -PathType Leaf`, else `.fail`) printed nothing; with a real `STAGE4.done` beside it, it printed `MK_DONE`. Re-run it against that folder before trusting a changed waiter.

```bash
# h. pull into its OWN folder (nothing of stage 2 or 3 is touched), then release the box
[ -e $S/pull_box2 ] && echo "TARGET EXISTS: STOP"
mkdir -p $S/pull_box2
sftp -b - tanitad-box2 <<EOF
get -r /D:/t2v8f/pull4 $S/pull_box2
get /D:/t2v8f/logs/stage4.log $S/pull_box2/stage4.log
EOF
rm C:/Users/Admin/box2_ship/BOX2_BUSY.txt      # (move aside instead if you keep a record)
```
Markers in `pull4`: `STAGE4.running` (heartbeat; its text is `phase=... last=<scene8>|<T2|RR>|<arm>:<DONE|FAIL|SKIPPED_DONE>`), `preflight|gate|rollouts.done|fail`, `STAGE4.done` (all jobs ran) or `STAGE4.fail` (a job REFUSED or CRASHED), `STAGE4_RESULT.json` (`overall`: COMPLETE, INCOMPLETE (a cell failed or a pass was skipped for disk), GATE_FAILED, INFRASTRUCTURE_FAILURE; it records `manifest4_sha256` and the package file counts `pkg_files_ok == pkg_files`, which the reader's G2(a) reads). **Only COMPLETE (248 of 248 cells DONE) is a result for the card.** A stop at any point is safe: markers become `*.prev`, a DONE cell is skipped, a half-written `records.jsonl` is moved to `.prev`, a DONE under another executor REFUSES. Cells land at `pull4/rollouts/<scene8>/objects/<T2|RR>/C0C_s<0|1>/{DONE,records.jsonl,actor_attach.json,scene_<idx>.npz}`, the layout the pinned reader reads.

## 2. The read (dev box): ONE command, the pinned script, never edited
`code/read_t2_dominance.py` (sha256 6345d4c7...2f5c, pinned in `code/read_t2_dominance.py.sha256.txt`) reads PREREG_T2_DOMINANCE section 6: gates G4, G1, G2 (a) to (e), G3, then the 19 rows per seed and the outcome (DOMINANT-T2 / NOT-DOMINANT / UNDERPOWERED / GATE-FAIL). Every argument spelled out (the three pulls are the FINAL ones; the stage-2 path is the run-2 pull the relaunch writes, `STAGE2_RELAUNCH.md`):
```bash
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f
S=$T/stage4
python -I $T/code/read_t2_dominance.py \
  --root2 $T/stage2/pull_box2_run2/pull2/rollouts \
  --root3 $T/stage3/pull_box2/pull3/rollouts \
  --root4 $S/pull_box2/pull4/rollouts \
  --sources2 $T/stage2/SCENE_SOURCES.json \
  --sources3 $T/stage3/SCENE_SOURCES3.json \
  --g1 $T/C0C_ID_RESULT.json \
  --result1 $T/stage1/pull_box2/pull/STAGE1_RESULT.json \
  --manifest1 "$T/stage1/t2v8f_stage1.tar::pkg/MANIFEST.json" \
  --result2 $T/stage2/pull_box2_run2/pull2/STAGE2_RESULT.json \
  --result3 $T/stage3/pull_box2/pull3/STAGE3_RESULT.json \
  --result4 $S/pull_box2/pull4/STAGE4_RESULT.json \
  --out-dir $S/score                       # writes $S/score/DOMINANCE_T2.json and DOMINANCE_T2.md
```
Notes. (1) The overlay manifests for G2(a) default to the dev-box tars (`stage2/overlay2/...`, `stage3/...`, `stage4/t2v8f_stage4_overlay.tar::pkg/MANIFEST4.json`); the read refuses if a stage ran another manifest than the one on hand. (2) G2(b) cites `logs/pytest_full_M.log` for stage 4 and refuses a log OLDER than `tests/test_t2v8f_gates.py` or `tests/test_t2v8f_e5.py`: re-run the full suite (`logs/N_FINISHED.txt` shows the last green run) if either file was edited since. (3) The interval answers *another draw of SCENES* only (paired scene-cluster bootstrap, B 2000, seed 0) and both seeds must hold; there is no replicate arm (H-ESTIM-SEED-1). (4) Tier: T2 re-perception with agents replayed at log time: DOMINANT-T2 is not a reactive-traffic result. (5) Do not open the pulls by hand first; the read opens them.

**The contract was rehearsed end to end on the dev box** (`tests/test_t2v8f_stage4.py::check_contract`): the pinned reader's `g3_check`, `g2d_check`, `g2e_check`, `g2a_check`, `g2_check`, `scenes_from` and `arrays_for_seed` (19 rows per seed) accept the layout the stage-2, stage-3 and stage-4 DRIVERS wrote on the stub fixture (3 + 3 scenes); a removed DONE, a shifted f0, a stage-4 `overall` other than COMPLETE and a result without its `manifest4_sha256` each go RED, named. The full `main()` refuses a fixture by design (the 62-scene constants), and `read_blocks` (the bootstrap and the outcome) was NOT run on it: the stub fixture has no lead agent, so the three DK rows are empty at both arms (MEASURED, finite 0 of 11 windows) and the reader says "no valid pair"; the statistics are covered by `tests/test_t2v8f_dominance.py` on synthetic arrays. On the real pulls the DK rows depend on lead agents being in frame (per family, with the n, in the read's own output).

## 3. ESTIMATED wall time (MEASURED ~18 s per cell on stage 2's run; stage 2's first run MEASURED preflight 769 s of which 603 s hashed 24 GB, gate 21 s for 81 starts)
| phase | estimate | basis |
|---|---|---|
| put + certutil + extract | ~1 min | 205 KB, no scene |
| preflight | **~15 min** | 33.2 GiB x (603 s / 24 GB) = ~890 s; seconds if it refuses |
| gate | ~1 min | 121 starts x (21 s / 81), two child processes (one per scene root) |
| rollouts | **~74 min** | 248 cells x 18 s |
| cold scene reads | ~+0.1 to 0.5 h | the first touch of each volume after the preflight evicted the cache (stage 3's estimate was +0.3 h) |
| **total, central** | **~1.6 h** (range 1.4 to 2.5 h) | the waiter cap is 6 h |

## 4. UNVERIFIED until box 2
The free-space figure (`Get-PSDrive` settles it); `tar -xf` writing six new files on exFAT; the preflight time on 33 GiB (it spans two drives: D: and C:); that the C0C cell renders as the stub-rehearsed one does (the arm is proved on the dev box by `check_composite`: C0C_s0's tick-0 plan equals the composite of C0's raw plan and P10's prior bit for bit, in a fresh process, through the shipping cell entry, with the package's pinned RI copy, and composite-off / wrong-prior go RED); that the real renderer's actor falsifier reads the banked npz as the dev-box REAL reader does (`check_real_path`: the kit's own `falsify_actors` on a real stage-2 and a real guard npz that MANIFEST4 pins, no mp4; the Master Mind reported stage 2 run 2's first new-scene cell DONE on box 2, INHERITED, not re-verified here); the falsifier's verdict per scene (`actor_attach.json` of each cell: a scene whose frames fail it gets its actors switched off by the kit, which the reader tabulates per scene, G2(e)); and the wall time.

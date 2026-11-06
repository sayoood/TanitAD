# STAGE1_LAUNCH: the v8F T2 harness, stage 1 on box 2 (commands for the Master Mind; NOTHING HAS BEEN LAUNCHED)

Evidence class: MEASURED (ours) for the package, its hashes and the CPU stub rehearsal; ESTIMATED for box-2 wall time; UNVERIFIED for everything that needs box 2 or a real render.
Card: `motion_c/t2_prereg/PREREG_T2_CRAWL.md` section 4 (stage 1: plumbing only, never read for H1 or H2) and 6b. Memory: `box2-second-computer`, `devbox-drive-letter-and-detached-jobs`.
⛔ Box 2 execution needs the PI's explicit go in chat (box-2 history: two blue screens and one native crash under load on 2026-10-17). One job at a time; this job is one process tree, safe mode (`%RUNSAFE%`).

## 1. The package (built on the dev box, CPU)
| item | value |
|---|---|
| tar | `E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage1/t2v8f_stage1.tar` |
| bytes | **549,652,480** (524.2 MiB) |
| sha256 | `1785337f93d3625bd0ef3f040365759b40c4542af425a175c0d8795253d3421c` (also in `t2v8f_stage1.tar.sha256`, sha256sum format) |
| MANIFEST.json sha256 | `7c30471a519f97c0691b817474f0dd7f060d76d2f772e40ba63800d66b1f0291` (inside the tar: path, size, sha256 of the other 471 files) |
| contents | `pkg/code` (7 .py), `pkg/tree` (448, the tip tree 24b97ba9), `pkg/v9` (2), `pkg/ckpt` (C0_s0, C0_s1, DAGOACC_s0, DAGOACC_s1, all four distinct), `pkg/refs` (T1 tool, BLIND mean, SCREEN_TABLE_2, 2 mp4, 2 `ref_frames.npz`, 2 `sequence_tracks.json`), `pkg/jobs/stage1.cmd`, `pkg/MANIFEST.json` |
| asserted at build | BLIND mean tensor sha256 `cd6a7ab5...a2fcb9` = the eval-141 mean (the zeros stand-in is refused); both mp4 equal the `lfs_sha256` pinned in `RELEASE_LISTING.json` (rev 5fe7a149); the T1 tool is `949c4b7a...`; no CR in any .py / .cmd |
| rebuild | `python -I code/make_stage1.py` (the old tar must be moved aside first; the stage dir must be new). `tests/test_t2v8f_stage1.py` refuses a tar whose code differs from `t2v8f/code` |

**Footprint on box 2's D: (exFAT, 1 MiB clusters; each file rounds up to whole MiB, each directory takes one):**
tar 525 MiB + extracted `pkg` 1,019 MiB (472 files, 42 dirs) + additive install into the kit's scene dirs 160 MiB (mp4, `ref_frames.npz`, tracks) + pull folder about 201 MiB (160 files, MEASURED on the stub; the real run writes the same file set) = **about 1.9 GiB**. Last probe: 45.4 GB free, so about 43.5 GB stay free (the rule is >= 10 GB; the driver's preflight refuses below 10 GB). Not counted: torch/gsplat JIT build cache under `D:\tanitad\tmp` (UNVERIFIED size, hundreds of MB).

## 2. Commands (dev box, git-bash)
```bash
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f

# a. the box must be free; claim it
ls C:/Users/Admin/box2_ship/BOX2_BUSY.txt 2>/dev/null && echo "BUSY: STOP"          # must print nothing
printf 'owner=T2V8F stage1\nstart=%s\neta=45-75 min\n' "$(date '+%F %T')" > C:/Users/Admin/box2_ship/BOX2_BUSY.txt

# b. folders
$SSH "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force D:/t2v8f/logs | Out-Null; Get-PSDrive D | Select-Object Free\""     # Free must be well above 10 GB

# c. copy (resume an interrupted copy with `reput` instead of `put`) and verify the bytes
sftp -b - tanitad-box2 <<EOF
put $T/stage1/t2v8f_stage1.tar /D:/t2v8f/t2v8f_stage1.tar
EOF
$SSH "certutil -hashfile D:\t2v8f\t2v8f_stage1.tar SHA256"     # 2nd line must equal: cut -d' ' -f1 $T/stage1/t2v8f_stage1.tar.sha256

# d. extract (bsdtar is built into Windows) and count
$SSH "tar -xf D:\t2v8f\t2v8f_stage1.tar -C D:\t2v8f"
$SSH "powershell -NoProfile -Command \"(Get-ChildItem D:/t2v8f/pkg -Recurse -File | Measure-Object).Count\""      # 472

# e. launch, detached (the same launcher every box-2 job uses)
$SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\t2v8f\pkg\jobs\stage1.cmd -Log D:\t2v8f\logs\stage1.log"

# f. poll with ONE background wait, not turns (stops on DONE / FAIL):
until $SSH "powershell -NoProfile -Command \"Get-ChildItem D:/t2v8f/pull -Filter 'STAGE1.*' -Name\"" 2>/dev/null | grep -qE 'STAGE1\.(done|fail)'; do sleep 60; done
$SSH "powershell -NoProfile -Command \"Get-Content D:/t2v8f/logs/stage1.log -Tail 30\""     # what it is doing now; the live marker is D:\t2v8f\pull\STAGE1.running

# g. pull (about 3 MB real data in 160 files) and release the box
mkdir -p $T/stage1/pull_box2
sftp -b - tanitad-box2 <<EOF
get -r /D:/t2v8f/pull $T/stage1/pull_box2
get /D:/t2v8f/logs/stage1.log $T/stage1/pull_box2/stage1.log
EOF
rm C:/Users/Admin/box2_ship/BOX2_BUSY.txt      # (move aside instead if you keep a record)

# h. score on the DEV BOX (the scorer reads E: reader trees; plumbing only, never H1 or H2)
cd $T && CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I code/t2v8f_run.py score \
  --root stage1/pull_box2/pull/plumbing --out stage1/pull_box2/PLUMBING_SCORE.json \
  --tracks 0=E:/thor_archive/2026-09-14/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430/extracted/sequence_tracks.json \
  --tracks 1=E:/thor_archive/2026-09-14/nurec_scenes/sample_set/26.04_release/7c72937c-c620-4776-9555-d57222c0081f/extracted/sequence_tracks.json
```
Re-launching after a fix is safe: the preflight/install are additive and idempotent, old markers are renamed to `*.prev` (never deleted), finished plumbing cells (`DONE`) are skipped. ⚠️ Never copy a `.cmd` to box 2 while another queue job runs there. Optional guard for a long run (box-2 history): start `telemetry_long` first (`launch.ps1 -Job D:\tanitad\jobs\telemetry_long.cmd`, stop by writing `D:/tanitad/logs/telemetry.stop`); this job does not start it.

## 3. What the job does (one driver, jobs strictly in this order, one child process at a time)
| # | job | what | gate |
|---|---|---|---|
| 1 | `preflight` | every path; CUDA device and >= 4 GiB VRAM; `cl`, `nvcc`, `ninja`; imports `numpy torch transformers PIL gsplat msgpack`; free disk >= 10 GB on pkg / pull / kit; sha256 of all 471 shipped files; kit `KIT_MANIFEST.json` sha256 `44c59998...` and md5 of the kit's scene files and the overlay `closedloop_drive.py`; DINOv3-L `model.safetensors` sha256 `dcb2e451...718179`. Collects EVERY failure. | PASS / REFUSED (nothing later runs) |
| 2 | `install` | additive copy of mp4, `ref_frames.npz`, `sequence_tracks.json` into the kit's scene dirs (a different file already there is refused) | PASS / REFUSED |
| 3 | `t5` | one C0_s0 start (7c72937c, f0 = 22), `objects`, T2, rendered twice in two processes: plans, executed track and every frame md5 bit-equal | PASS / FAIL |
| 4 | `gap` | frames rendered at the LOGGED pose vs the real mp4 frames (9 ticks per scene): pooled-token cosine and rel-L2, grad-NCC. Beside; no threshold (floor for orientation: G2 encoder vs cache rel-L2 0.0264, cosine 0.99965) | REPORTED |
| 5 | `actor` | `objects` draws the replayed lead at 3 ticks (31, 44, 58; lead 9.2 to 12.6 m ahead, from the scene's own tracks). **MM ruling (card 6b):** the DECIDING planted no-lead read is condition `empty` (a renderer with no actors attached) at the SAME pose and tick as the `objects` read. The 30 s past-the-end clock read is reported beside and never decides. Thresholds as committed in the docstring of `t2v8f_stage1.py` before any box-2 render: PRESENT >= 200 changed pixels (> 8 levels) and >= 1 % of the ROI; ABSENT <= 5 | PASS / FAIL |
| 6 | `plumbing` | 7 arms x {T2, RR} x 2 scenes x 1 start = 28 cells, `objects`, one process per cell, records stamped PLUMBING ONLY (starts: 00040136 f0 = 60, 7c72937c f0 = 22; schedule `periodic`) | PASS (all 28 `DONE`) / FAIL |

## 4. Markers in `D:\t2v8f\pull` and what they mean
| file | meaning |
|---|---|
| `STAGE1.running` | the driver is alive (pid, start). Removed at the end. Present with a silent log and no process = a hard crash (native crash / reboot): no verdict exists |
| `<job>.done` / `<job>.fail` for `preflight install t5 gap actor plumbing` | `.done` = the job RAN TO COMPLETION; its text is the verdict (`PASS`, `REPORTED`, or a MEASURED `FAIL`). `.fail` = REFUSED (preflight / install: a named reason) or CRASH (exception) |
| `STAGE1.done` | no infrastructure failure. **Read `STAGE1_RESULT.json`**: `overall` is `ALL_GATES_PASS` or `GATE_FAILED` |
| `STAGE1.fail` | preflight or install refused, or a job crashed (`overall = INFRASTRUCTURE_FAILURE`); the text names the reason (`MISSING <path>`, `HASH_MISMATCH <path>`, `IMPORT_FAILED <module>`, `NO_CUDA_DEVICE`, `GPU_MEMORY_LOW`, `DISK_LOW`, `MISSING_TOOLCHAIN`) |
| `STAGE1_RESULT.json` | rewritten after every job (a crash keeps the earlier jobs' results): `gates`, `steps.*` with the numbers (T5 frame md5 diffs, gap rows, actor changed-pixel counts per tick, plumbing cell states) |
| `plumbing/<scene8>/objects/<T2|RR>/<arm>/{DONE,FAIL,records.jsonl,scene_N.npz,...}` | one cell; `DONE` text `rc=0 records=1/1` |
| `*.prev` | an earlier marker, renamed on a re-run |

After the run: `ALL_GATES_PASS` -> the plumbing score is a plumbing check, not a read; stage 2 (42 scenes, needs an exclusive slot and the PI's go) is the Master Mind's proposal. `GATE_FAILED` -> card section 6, outcome 5: the instrument is fixed first, no contrast is read. `INFRASTRUCTURE_FAILURE` -> fix the named reason and relaunch.

## 5. Box 2's venv imports: first checked by the preflight (UNVERIFIED until it runs)
`D:\tanitad\venv` is documented as torch 2.11.0+cu128 and transformers 5.15.0 only. **`gsplat==1.5.3`, `msgpack`, `ninja` are not documented there** (the kit's `t2a_preflight.cmd` has the pip step, which needs the network and new packages, and was never recorded as run). The preflight reports each as `IMPORT_FAILED <module>` or `MISSING_TOOLCHAIN` and stops. That would be a named blocker for the PI (install = new packages on his personal machine), not something this package does. `cv2` is not needed (`t2v8f_nocv2` replaces the two helpers `--condition objects` calls). The mp4 decoder is not needed either: the reference frames ship pre-decoded (`ref_frames.npz`, from the same mp4, hashes in the manifest); `info.mp4_decoders_here` records whether PyAV or imageio exist. A `HASH_MISMATCH kit:KIT_MANIFEST.json` means box 2's kit manifest is not the dev copy this package was built against (expected sha256 `44c599981d810b1f0c4b0b9de342a227e670fa5ffc6a92051cab215204b60657`): compare before overriding.

## 6. ESTIMATED wall time (no render timing has been measured by this harness on box 2)
`T2_SIZING_V8F.md` section 4 gives the 28 runs as 14 to 36 min (central 19) in one process plus 15 min of gates. This package runs each cell as its own process, so add a scene load (about 15 s) and imports per cell:
preflight 1 to 3 min (about 2.4 GB hashed from the bus-powered USB HDD) · install < 1 · T5 2 to 3 · gap 2 to 4 · actor 2 · plumbing 20 to 40 · first-use gsplat CUDA JIT build (if not cached) +3 to 10, once. **Total about 45 min central, 30 to 75 min range.** No download is involved.

## 7. What the CPU rehearsal covers, and what it cannot
MEASURED on the dev box (stub renderer and encoder, shadow kit with 16-byte placeholders; the real kit was only read): `tests/test_t2v8f_stage1.py` and `code/run_mutations_stage1.py` (logs in `logs/`).
* positive run of the shipped driver on the REAL tar: all six jobs, `ALL_GATES_PASS`, 28 cells DONE, 50 s;
* RED arms: planted missing `ckpt/DAGOACC_s1.pt` (`STAGE1.fail`, `MISSING ckpt/DAGOACC_s1.pt`, steps == [preflight]); one changed byte (`HASH_MISMATCH refs/SCREEN_TABLE_2.json`); shifted ROI (actor FAIL, 0 of 3 present); `empty` frame that draws the lead (actor FAIL, planted read 75,014 / 44,213 / 29,819 changed px); `check_score_e2e` with `drop_p10_record` (pairing guard) and `rows_missing`.
NOT covered (UNVERIFIED until box 2): real renders and the ROI thresholds (committed guesses), `grad_ncc` without cv2 (a re-statement, not compared with cv2, which is absent on both boxes), gsplat JIT, DINOv3 on the GPU (the float8 token cast in the gap job), `attach_actors_verified` on the real scene, `launch.ps1`, `tar` and `certutil` on box 2, the `KIT_MANIFEST.json` byte-equality across the copy, and whether the real `empty` renderer is pixel-identical to the `objects` renderer with its actors switched off (if it is not, the planted read reports FAIL for that reason, and the row shows it).

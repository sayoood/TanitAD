# Brief: screen and download the NuRec scenes for the v8F T2 crawl read (PI approval 2026-11-06)

**PI approval, in chat (2026-11-06):** "Approve T2 path". This was the answer to *"Approve the T2 path (about 26 GB scene download plus
box-2 jobs)?"*, and the sizing quoted 26 GB with 80 GB as the worst case. ⇒ This brief may download NuRec scene data from the
release named in the repo docs, up to **80 GB in total**. Nothing else may be downloaded.

Read first and follow: `E:/Projects/TanitAD/Project Steering/AGENT_OPERATING_STANDARD.md` (preamble).

**Repo and files**
* Never commit or push. Never delete files: move them aside. LF line endings.
* Read the repo via `git -C C:/Users/Admin/tanitad-push show 24b97ba9:<path>`.
* Do not touch other sessions' processes or files.

**Secrets**
* HF token: read it in place from `E:/Projects/TanitAD/Keys.txt` inside the Python process (the `hf_[A-Za-z0-9]+` pattern).
* ⛔ Never print it, never copy it, never put it in argv or into a file.
* Set `HF_HUB_DISABLE_XET=1`. Xet uploads and downloads hang on this box.

**Licence**
* ⛔ If the dataset is gated and its terms are not yet accepted on the account, STOP and report. Accepting terms is the user's to do.

**Compute and the clock**
* Dev-box CPU only (`CUDA_VISIBLE_DEVICES=-1`).
* Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`.
* Long transfers run detached via WMI `Win32_Process.Create` with `cmd /c ... < NUL`. ⛔ Do not wait on a transfer longer than about 4 minutes while alive.

**Context budget:** ≤ 150k tokens. Read by grep anchors.

## Sources to read (excerpts)
* The sizing: `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-06-t2-sizing-v8f/T2_SIZING_V8F.md`, sections 3, 5 and 6.
* The release docs it cites: `ALPASIM_STATE.md`, and the AlpaSim / NuRec lines of the feasibility docs (grep the tip for `NuRec`).
* The T2-A kit: `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-t2a-closedloop-prep/INVENTORY.md`, which says which members of a scene the renderer needs, and `code/scene_info.py`.
* The banked box-2 run: `.../2026-10-17-t2a-closedloop-box2/`.

## Step 1: freeze the screening rule BEFORE reading any scene log
Copy this rule verbatim into `t2_scenes/SCREEN_RULE.md`, adding only the concrete field names, the repo id and the revision. Write its
sha256 to `t2_scenes/SCREEN_RULE_SHA.txt` with a timestamp. Only then download screening logs.

* **Universe.**
  * Every scene of the NuRec release, at the revision pinned in step 1.
  * Minus every scene whose PhysicalAI clip id is in v8F-mini's TRAINING set: the 4,512 clips. Find the train clip list that built
    box 2's `cache_full` (v9 release 4,572 clips minus the `--exclude 540:600` holdout); cite where you found it.
  * Minus scenes the renderer cannot render (missing members). Record the exclusion counts.
* **Order.** Sort the remaining scene ids, then apply `numpy.random.default_rng(0).permutation`. Screen in that order.
* **Speed.** v(t) is the logged ego speed from the scene's own ego track: path length over a centred 1.0 s window.
* **Launch event (an M1-bearing scene).** An upward crossing of 5.2 m/s by v(t) at time t_c, with three conditions:
  * v ≤ 3.0 m/s at some time in [t_c − 5 s, t_c);
  * t_c ≥ the first renderable time + 2.0 s;
  * t_c ≤ the last renderable time − 6.0 s.
* **Lead stratum (recorded, never used to select).** At the last time with v ≤ 3.0 m/s before t_c, is there a vehicle-class agent
  0 < x ≤ 30 m ahead with |y| ≤ 2.0 m in the ego frame?
* **Guard scene.** No launch event, and v ≥ 5.2 m/s on at least 50 % of the renderable span.
* **Quotas.** The first 42 launch scenes and the first 20 guard scenes in permutation order. Stop screening when both are met, or after
  800 scenes. If 800 are screened without meeting a quota, report the counts: do not loosen the rule.

## Step 2: screen
* Download per scene only what the rule needs: the ego track, the agent tracks and the scene's time span (about 4 MB per scene, ESTIMATED).
* Write `t2_scenes/raw/SCREEN_TABLE.json`. Per scene: id, clip id, renderable span, launch events (t_c, lead stratum), guard flag, and
  any reason for exclusion.

## Step 3: download the 62 selected scenes' renderer members
* **Where.** First probe box 2 read-only (`ssh tanitad-box2`, see memory `box2-second-computer.md`): free space on `D:`, and whether it
  can reach HF with its own login (`hf auth whoami` or equivalent, never printing a token).
  * If box 2 has HF access and at least the needed size + 20 GB free, download there into `D:\tanitad\t2_scenes\`.
  * Otherwise download to the dev box under `F:/TanitAD-scratch/t2_scenes/`, and C: if needed.
  * ⛔ Never leave any drive below 10 GB free, and stop if the plan would.
  * Box 2 is idle. This copy is data staging, not a GPU job. Run nothing else there.
* **Integrity.** Verify every file by size, and by sha256 where the release publishes it. Write
  `t2_scenes/raw/DOWNLOAD_MANIFEST.json` (scene, member, bytes, sha256, location).
* **Running it.** Use a resumable detached script (it skips files already verified) with a done / fail marker:
  `t2_scenes/raw/T2_SCENES.done` or `T2_SCENES.fail`.

## End (token rule)
Once the screening is complete and the download is launched detached, write `t2_scenes/HANDOFF_T2_SCENES.md` (≤ 2k tokens): state,
counts, paths, the rerun command, caveats. Then END.

Hand back in ≤ 300 words: the rule sha, the screening counts (screened, excluded by reason, launch scenes with the lead stratum, guard
scenes), the bytes planned, the download location and launch time, the marker paths, and a deliverable manifest.

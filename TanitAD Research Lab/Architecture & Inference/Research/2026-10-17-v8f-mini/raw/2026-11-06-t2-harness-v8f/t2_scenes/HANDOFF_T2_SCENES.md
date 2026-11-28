# HANDOFF: T2 scene screen and download (written 2026-11-06 ~09:51Z; the download is still running)

Root: `E:/Projects/TanitAD-artifacts/dagland/motion_c/t2_scenes/` (call it `T`). Not in the git repo; single copy on E:. Nothing committed or staged.

## State
* Rule frozen BEFORE any scene log: `T/SCREEN_RULE.md`, sha256 `6c1daa86a6fab4e75a88328789e6955d33d48b82b6842842b252cbdeb31c8a9d`, frozen 2026-11-06T09:35:38Z (`T/SCREEN_RULE_SHA.txt`).
* Release pinned: `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`, branch `26.04`, commit `5fe7a149b40ee83a2d3e09f7fe5dc2f217cabe11` (`main` is now 26.10, not used). Gate: terms already accepted on this account (HEAD returned 302 not 403).
* Universe: 1,607 scenes, minus 16 in v8F's 4,512 train clips (list rebuilt from box 2 `cache_full/train/manifest.jsonl`; reproduces `train_sha256 37f4f9ee...`), minus 0 missing members, 0 directory read errors = 1,591.
* Screening (`T/raw/SCREEN_TABLE.json`, `SCREEN.done`): 263 screened, order sha `df863e93...`, numpy 2.5.1. Both quotas met: 42 launch scenes (lead stratum yes 19, no 23; one launch event each), 20 guard scenes (quota filled at scene 43), 75 neither, 0 unparseable. Traffic 884 MB.
* Download: detached since 2026-11-06T09:44:43Z (WMI pid 49280), 62 scenes x 3 members = 186 files, 35,671,549,099 B planned (33.2 GiB; launch 24.09 GB, guard 11.58 GB). F: `F:/TanitAD-scratch/t2_scenes/<scene>/` 38 scenes (21.6 GB), C: `C:/Users/Admin/TanitAD-scratch/t2_scenes/<scene>/` 24 scenes (14.0 GB). Both drives stay above 20 GB free by plan; the script refuses to go under 10 GB. Rate is bursty, 8 to 17 MB/s, so expect 40 to 70 min.
* Members per scene: `volume.nurec`, `rig_trajectories.json` (scene root), `sequence_tracks.json` (in `extracted/`, where `actor_map.py` reads it). Verified by size and CRC32 against the usdz central directory; sha256 is computed and recorded (the release publishes sha256 only for whole usdz files).

## Check / rerun
* Progress: `T/raw/DOWNLOAD_STATUS.json`, `T/raw/download.log`. Finished: `T/raw/T2_SCENES.done` (written only if all 186 files verified). Failed: `T/raw/T2_SCENES.fail` (lists failures).
* Resume (skips verified files, continues `.part` files):
  `powershell -NoProfile -File T\code\wmi_launch.ps1 -Script T\code\download.py -Log T\raw\download.log` (use backslash paths; check first that no `download.py` python process is alive).
* Result: `T/raw/DOWNLOAD_MANIFEST.json` (scene, member, bytes, crc32, sha256, location). Plan: `T/raw/DOWNLOAD_PLAN.json` (the 11 GB-floor first plan is kept as `.v1_floor11GB.json`).
* Re-run screening only if the rule changes (it must not): `code/screen.py` is resumable from `raw/screen_cache.jsonl`.

## Caveats
* Interpretations fixed at freeze (in the rule file): v(t) on the 10 Hz tick grid over ticks k-5..k+5; renderable span = first to last front-wide 10 Hz tick; guard share over all span ticks.
* 6 screened scenes carry a `train_or_tram_car` track. It is outside the frozen vehicle set, so a tram ahead does not count as a lead. This touches only the lead stratum, which never selects.
* Controls passed: synthetic self-test (`code/selftest_t2lib.py`, 9 checks incl. a mutation), and the two on-hand scenes read as expected (00040136 guard, 7c72937c launch with lead 12.35 m; 7c72937c is selected as `launch_23`).
* Box 2 (read-only probe): `hf auth whoami` works (user Sayood), but `D:` has 45.4 GB free, which fails "size + 20 GB"; so nothing was staged there. Moving all 62 scenes to `D:` would leave 9.7 GB (< 10 GB floor): ship launch scenes only (24.1 GB) or batches.
* NOT downloaded: the reference mp4 (about 39 MB per scene, about 2.4 GB for 62), needed by the render-replay gate in the sizing note. Outside this brief's "renderer members"; ask before fetching.
* `T/raw/box2_cache_full_train_manifest.jsonl` is a copy of a box-2 file (the original stays on box 2).

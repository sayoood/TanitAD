# refcv8 data audit — shared context for every stream (Master Mind, 2026-10-04)

PI, 2026-10-04: *"I need a big plan for refcv8, I need to understand all effects define measures and validate
them to retrain without loosing a lot of time. ... is our training data ok? are any thing missing or
uplausible, are we using them wrongly? Why the video is saying, no gt label in this frame?"*

This package answers the DATA half of that request. Five streams (D1–D5) write into their own sub-folder.

## What refcv7 trained on (the launch record — quote these, they are MEASURED from the config)

Source: `D:/refcv7_eval_kit/ckpt/config.json` (`argv` + `*_stats` blocks). Launch commit `fec3a0d`.

| channel | file (Thor path) | how it is applied |
|---|---|---|
| frames + ego poses | `--v2-cache /home/nvidia/data/refcv6-b1-416x1024-train` (symlinks into `physicalai-b1-w120-416x1024cyl/`), 4,369 clips, **746,946 windows** (~171 per clip) | per window |
| v8 labels | `--v7-labels /home/nvidia/data/v8labels/labels/s2_labels_v8_train.jsonl.gz` md5 `b45377a1f25263b5c0f3d318c126b1ac`, 4,572 records, ONE per clip | see below |
| tactical lat/lon + 22 goal tokens | from the v8 record | ONLY when the window NOW is within ±2.0 s of the record anchor (`v7_labels.window_in_band`), else IGNORE (-100 / weight 0) |
| nav input | `--nav-from-v7`: the record's `nav_command.token` (oracle, ego-future) | the SAME token on EVERY window of the clip (train: 2,752 follow / 779 left / 838 right clips) |
| max-speed input | `--max-speed-input-v6 --speed-max-sidecar-v6 refcv6_speed_max_v8_train.jsonl` = `g_tac.goals.SPEED_BAND.v_hi_ms` | ONE value per clip on EVERY window (`window_ceiling_frac 1.0`), 4-way one-hot {30,50,100,120} km/h |
| agent boxes (2D/agent head) | `--agent-join /home/nvidia/data/joins/b1_train_plus_eval_agents.jsonl.xz` | per window; 96.36 % of train windows labelled |
| 3D boxes | `--join3d /home/nvidia/data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz` | per window |
| 10 cm map | `--map-gt-root /home/nvidia/data/sam3_gt_v3` (SAM3 map GT) | per window |
| clock | `--clip-clock-sidecar /home/nvidia/data/refcv6_clip_clock_sidecar.jsonl` | `t_now = grid_start_s + (t + w - 1 + n_stack - 1) * dt_s`, raw offset 2 |
| eval | `--eval-cache .../refcv6-b1-416x1024-eval139` (139 clips, 23,772 windows), `--eval-labels s2_labels_v8_eval.jsonl.gz` md5 `eefc38d1453bd1c73802d44d45affced` | |

## Already MEASURED by the Master Mind today (do not redo; cite as "MM 2026-10-04, label file md5 b45377a1")

* **Every one of the 4,572 records has `t0_s = 8.0`** (raw recording seconds) and `bands.tactical_s = [2.0, 6.0]`.
  The clip cache covers raw ≈ [0.1, 20.1] s (`grid_start_s` median 0.113, `dt_s` 0.1007, 199–200 rows).
  ⇒ tactical/goal GT exists only for windows whose NOW ∈ [6.0, 10.0] s. ESTIMATED ≈ 40 of ~171 windows per
  clip (~23 %); D1 measures it exactly.
* `a_tac.lat`: LANE_KEEP 2,958 · NUDGE_R 592 · NUDGE_L 488 · TURN_L 275 · TURN_R 259.
  `a_tac.lon`: CRUISE 1,243 · ACCELERATE 998 · ADAPT_SPEED_FOR_CURVE 995 · BRAKE_TO 905 · FOLLOW 165 · CREEP 135 · HOLD 131.
* `a_tac.lat_args.lat_peak_m`: median |x| 19.7 m, p90 110.8, max 305.3; 70.2 % > 5 m. **Meaning/units UNVERIFIED** —
  a nudge does not move 145 m sideways, so either the field is not a lateral offset or it is broken. D2 settles it.
* Alpamayo (VLM) vs our geometric tactical label: lateral `agree` False on 1,537 / 4,275 scored records (36 %),
  longitudinal False on 1,655 / 4,567 (36 %).
* `nav_command` L/R (1,675 records): the turn starts a median **7.3 s** after the anchor; 52.7 % > 6 s, 42.0 % > 10 s
  (p90 26.2 s). `nav_30s.entries` carries every turn with `t_start_s/t_end_s` (anchor-relative) and arc-length
  `distance_m` — a time-localised nav IS in the file; refcv7 fed only the bare token.
* `speed_max_input` documents itself as an oracle: "max of the ego's REALISED speed over [anchor+2s, anchor+6s]";
  "v_hi<-v0 R^2 0.8789; v_hi<-(bin,v0) R^2 0.9702 — the bin still recovers 75.4 % of the future information";
  and "75 % of intersection clips get <= 30 km/h where a real map would say 50".
* `horizon.recording_span_s` median 139.7 s: the 20 s clip is cut from a ~140 s recording.
* strata: urban 2,965 · intersection 861 · highway 746; day 2,452 · night 2,120; ~25 countries, ~250–290 clips each.
* Route diagnosis (`TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-route-following/RESULT.md`):
  on 107 eval turn windows the tactical lateral label is IGNORE on 72 and agrees with the actual 6-s turn on 49 % where
  present; 193 of 291 left/right-nav eval windows are straight.

## Paths and environment

* Repo (working tree, read-only for you except your package folder): `D:/Projects/TanitAD`. ⛔ Never read or write `G:`.
* Launch-tree code extract used by the eval battery: `C:/Users/Admin/ev7/{stack,taniteval}` — run with
  `PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval"` (WINDOWS paths; MSYS `/c/...` entries are
  silently dropped and the editable install then serves an abandoned tree). Verify an imported module's blob against
  `GIT_DIR=/c/Users/Admin/tanitad-push/.git git show fec3a0d:<path>` before quoting behaviour.
  Trainer: `stack/scripts/refc_v3_train.py` (`V3Dataset.__getitem__` ~3853, `_now_s` 3798);
  labels lib `stack/tanitad/data/v7_labels.py` (`window_in_band` 678, `tactical_class_ids` 701, `tactical_goal_targets` 917).
* Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, set `PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=4`.
* Local copies (dev box): `D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz` (md5 b45377a1 verified),
  `D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz`, `D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl`,
  `D:/refcv6_eval_kit/data/a6/refcv6_speed_max_v8_train.jsonl`, `D:/refcv6_eval_kit/data/refcv6_speed_max_v8_eval.jsonl`,
  eval cache `D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/` (139 clips), a 140-clip train subset
  `D:/refcv6_eval_kit/data/refcv6-b1-416x1024-train-a6/`, SAM3 eval GT `D:/refcv6_eval_kit/data/sam3_gt_v3_eval/`,
  joins `D:/refcv6_eval_kit/data/joins/`, `join3d/`.
* Label builder history: `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/`
  and the v8 build copy `D:/Projects/TanitAD-artifacts/_s2build-copy-20260919/`.
* **Thor** (`ssh -n tanitad-thor-wifi`; ALWAYS `-n` inside scripts): IDLE (refcv7 finished). Full train cache, labels,
  joins and SAM3 GT live under `/home/nvidia/data/`. Venv `/home/nvidia/venvs/tanitad-train`. Work dir
  `/home/nvidia/refcv8_audit/<stream>/`. Disk is 94 % full (57 GB free): write < 5 GB. CPU only, ONE heavy job at a
  time across all streams (take `flock /home/nvidia/refcv8_audit/cpu.lock`). ⚠️ The PI's Thor disk cleanup REFUSES
  while any python process of user nvidia is alive — leave NO python process running when you finish, and say in your
  report when your last Thor job ended.
* **Dev box**: NavSim + battery evals hold the GPU; free RAM ~6.5 GB, free commit ~8.8 GB. CPU only, keep resident
  memory < 2 GB, never touch the GPU or `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock`.

## Binding rules

* Every number carries its evidence class (MEASURED + artifact path / INHERITED / ESTIMATED), its n and its split.
* Never print or write raw clip UUIDs — sha12 of the clip id only (`hashlib.sha256(clip_id.encode()).hexdigest()[:12]`).
* A cross-check must be derived INDEPENDENTLY of the value it checks (write thresholds as literals from road
  geometry, not by importing the builder's constants); include at least one analytic/known-value control that must
  read a known value, and report it.
* "0 hits" from a file you could not read is not absence — count unreadable files.
* Do not `git add`, commit or push. The Master Mind lands. Finish with `LANDING_READY_<stream>.txt` in this package
  (format: `## <STREAM-TAG>` heading, then one repo path per line with md5 and a note; a section for NOT-for-git
  files with size and location; a section for Thor-only state).
* Kill only by explicit PID. Never `pgrep -f`/`pkill -f`.

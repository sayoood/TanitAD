# HANDOFF -- refcv8 judgement videos (EvalFlyWheel, 2026-10-19).  NOT staged in git; qland hold `hold_r8video`.

Package (code + docs): `C:/Users/Admin/qland/hold_r8video/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-videos/`   Videos: `E:/refcv8_eval_kit/videos/`   Work cache: `E:/refcv8_eval_kit/videos/_cache/`
All CPU.  Nothing here takes the GPU lock; `dump_heads_r8.py --device cuda` must be run by a caller that HOLDS the dev-box GPU lock (as `chain_40k_r8.sh` does).

## (i) Render the full eval139 set at the final checkpoint 48,852 (CPU, ~45 min, resumable)
Precondition: the battery tree `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_unc/` exists with `panel_s0`, `s6_s0`, `dump_s0/{decisions,refcv6_extras.npz}`, `battery_summary.json` (`step`, `ckpt_md5`) -- i.e. the 48,852 battery is banked (expected night of 2026-10-20).
Check: `ls .../step48852ema_unc/battery_summary.json` and that `stages.roll_s0.n_windows == 4754`.
```
cd <package>/code
set OMP_NUM_THREADS=4 & set CUDA_VISIBLE_DEVICES=
python render_all_r8.py --tree step48852ema_unc --out-dir E:/refcv8_eval_kit/videos/final48852
        [--deploy-npz <DEPLOY8_POS_step48852ema_unc_s0.npz>]   # only if a 48,852 re-time table was built; default = the 30k EMA table, labelled "30k EMA layer" on screen
        [--heads-dir <dir with heads_ep###.npz>]                # fills the map-head / box-head / allocation layers (section ii)
```
* Resumable (skips existing mp4); waits for >= 5 GB free commit before each episode; one JSON line per episode in `<out-dir>/RENDER_LOG.jsonl` (check `ok`, `controls`).  Wall MEASURED 18 s (ep114, idle box) to 24-38 s per episode (smoke run beside a CPU dump, `render_all_r8.py --episodes 92,136,114`) -> **~45-90 min** for 139 episodes.
* Do NOT run it while the overnight CPU re-roll is at its peak if the log shows `waited_s` growing: it self-throttles on commit, it does not need babysitting.
* Reel of three rule picks at the new checkpoint: `python pick_episodes_r8.py --tree step48852ema_unc --out <raw>/PICKS_step48852ema.json`, render those three with `render_r8_video.py`, then `make_reel_r8.py` (it names files `refcv8_ema<K>k_ep<N>_<rule>.mp4`, K = step//1000).
* Per-episode failures are LOGGED (e.g. no map GT for episodes 92 and 136 renders with "no map GT", it does not fail).

## (ii) Dump the map-head / box-head / allocation / fan on the GPU (for the MAP + BEV-head panels)
```
(caller holds the GPU lock, job name e.g. refcv8-heads-step48852)
cd <package>/code
set R8_NO_POSE_SYNC=1   (default inside the script: the standalone clock == the banked `unc` trees)
python dump_heads_r8.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json ^
      --out-dir E:/refcv8_eval_kit/videos/heads_48852 --episodes all --device cuda
```
* ESTIMATED run time: **~41 min + ~2 min model build** for all 4,754 windows (the battery's os-only roll `roll_s1` took 2,446.8 s for 4,754 windows on this GPU, MEASURED in `step40000ema_unc/battery_summary.json`; the head capture adds a 1000x600 argmax per window, ms).  The three showcase episodes alone (~105 windows): ~2 min + build.  GPU memory 1.22 GiB peak for the os-only roll (MEASURED).  **CPU fallback is valid for the heads** (MEASURED: ep114, 34 windows, CPU fp32 trunk, see `keys.json` in `E:/refcv8_eval_kit/videos/heads_cpu_ep114/`): the map / box / tactical outputs are deterministic functions of the frames (CPU-vs-banked-GPU tactical logits <= 0.16, argmax equal); only plan / fan / selection are sampler draws.
* Output: `heads_ep###.npz` (+ `keys.json` listing captured / skipped keys and the v0 control vs the banked dump).  Disk ~ **0.57 MB per window MEASURED (ep114 CPU dump) -> ~2.7 GB for the 4,754 windows (ESTIMATED, linear)**.  CPU cost for the same job, MEASURED 8.5 s / window -> ~11 h: use the GPU.  `--save-logits` adds 6 MB per window: leave it off unless needed.
* After it: re-run (i) with `--heads-dir`.  The renderer then shows GT | refcv8 map head side by side, orange box-head boxes (presence >= 0.20, `--box-p-thr`) in the camera and BEV, and the allocation counts.
* The checkpoint md5 must be recorded: `keys.json` stores it (skipped above 4 GiB: then record it from the ckpt's `.md5` sidecar).

## Open items / escalations (also in the hand-back)
1. **nav_cmd is constant per episode in 139/139 episodes** (`raw/NAV_CONSTANCY_step40000ema.json`): a clip-level route command.  In the rule-picked sharp-turn episode (13) it is TURN_R for the whole clip while the first manoeuvre is a left turn.  Decision for the PI/MM: is a time-local nav intended?
2. The shipped re-timed layer exists for 30k only; a 48,852 deploy table is not built (nobody owns it yet).
3. Box-head presence is uncalibrated on EMA-40k: max p 0.399 / 0.289 / 0.324 over the three showcase episodes (never 0.5), so the display threshold is 0.20 (`--box-p-thr`), a display choice, not a detector threshold.  The 48,852 dump will show whether the final checkpoint calibrates.
4. Videos and head dumps live on the dev box only (see VIDEO_MANIFEST.md deliverable table): nothing is in git.

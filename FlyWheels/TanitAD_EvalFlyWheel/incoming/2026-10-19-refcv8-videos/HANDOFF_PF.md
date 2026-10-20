# HANDOFF_PF -- refcv8 videos on the PER-FRAME nav feed (step 48,852, tree `step48852ema_pf`)  (EvalFlyWheel, 2026-10-20, hold `hold_r8vidpf`)

Why: R8-EVAL-NAV-FEED-1 / R57 -- the banked `*_unc` rolls fed the CLIP-level v7 token and no `r8_nav`; PI decision (a) 2026-10-20 reads the 48,852 bars and SHIP-R8 on the per-frame feed. These videos must show that model. Package = the landed `2026-10-19-refcv8-videos/code/` + the files of this hold (same relative paths; `r8v_feed.py` and `test_r8v_feed_parity.py` are NEW).

## Precondition (all CPU checks, run first)
* `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/battery_summary.json` exists, `pf.all_seeds_fed == true`, `stages.roll_s0.n_windows == 4754`; the chain's clause-1b `feed_parity_pf` JSON for seed 0 reads PASS.
* Optional RT layer: `C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256` (64 hex) and `.../2026-10-20-refcv8-perframe-final/pfpkg/raw/retime_deploy/{DEPLOY8_POS_step48852ema_pf_s0.npz, refcv8_retime_deploy_gains_pf.json, FORWARD_DEPLOY8_step48852ema_pf.json}` (written by `pf_lane.py install` + `d8`). If any link is missing or does not chain to the registered sha256, the RT layer is OFF with an on-screen note. It never falls back to the clip-fed table `b208c9c1...`.

Env for every command: `PYTHONPATH=C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval`, `PYTHONIOENCODING=utf-8`, `HF_HUB_OFFLINE=1`, `R8_NO_POSE_SYNC=1`, python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8`, cwd `<package>/code`.

## (1) Heads dump: GPU job, run by a caller that HOLDS the dev-box GPU lock (e.g. job `refcv8-heads-step48852pf`)
```
python -X utf8 dump_heads_r8.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json \
    --out-dir E:/refcv8_eval_kit/videos/heads_48852_pf --episodes all --device cuda --feed perframe_full --banked-tree step48852ema_pf
```
* Default `--feed perframe_full` = `roll_seed_r8_pf.py`'s wiring (registered overlay into `r7_roll`, `--nav-feed perframe_full`). Every captured window's model-call token and `r8_nav` are asserted equal to the roll's own sidecar, and the pf sidecar census must be ok, before any `heads_ep###.npz` is written. Each npz carries `nav_feed`, `nav_cmd_fed`, `nav_cmd_clip`, `r8_nav`, `r8_nav_known`.
* Then `keys.json.nav_feed.parity` = `r8v_feed.check_feed_parity`. It checks the heads against the `step48852ema_pf/dump_s0` sidecar AND against the independent v9 lookup (`feed_parity_pf.V9`). **Exit 5 means parity is not PASS, and those heads must not be rendered.** The log ends `ZZHEADSFEEDPASSZZ` on success.
* Cost (from the original HANDOFF): ESTIMATED **~41 min + ~2 min build** for 4,754 windows. That figure scales from the MEASURED os-only roll `roll_s1` of 2,446.8 s (`step40000ema_unc/battery_summary.json`). GPU peak 1.22 GiB was MEASURED on the os-only roll. Disk is ~0.57 MB/window, MEASURED on the ep114 CPU dump, so ~2.7 GB ESTIMATED. The per-frame pre-pass adds v9 join lookups but no frame decode (ESTIMATED negligible, UNVERIFIED at 4,754 windows). CPU fallback is MEASURED at 8.5 s/window, ~11 h.

## (2) Independent re-check of the heads (CPU, seconds)
```
python -X utf8 r8v_feed.py check --heads-dir E:/refcv8_eval_kit/videos/heads_48852_pf --sha12-tree C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf \
    --pf-dump C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s0 --out <raw>/FEED_PARITY_heads_48852_pf.json
python -X utf8 r8v_feed.py check --tree-root C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf \
    --pf-dump C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s0 --out <raw>/FEED_PARITY_render_48852_pf.json
```
Both must print `ZZVIDFEEDPASSZZ`. The second one reads the feed exactly as the renderer does. Its PF half is the same artifact, so the V9 half is the informative check.

## (3) Render all 139 eval episodes (CPU, no lock, resumable)
```
set OMP_NUM_THREADS=4 & set CUDA_VISIBLE_DEVICES=
python -X utf8 render_all_r8.py --tree step48852ema_pf --out-dir E:/refcv8_eval_kit/videos/final48852_pf --heads-dir E:/refcv8_eval_kit/videos/heads_48852_pf
```
* The first printed JSON line states `nav_feed` and `rt` (ON with `pf table <sha8>` or the OFF reason). Each `RENDER_LOG.jsonl` line carries `nav_feed`, `n_windows_fed_token_differs_from_clip`, `rt`, `heads_feed`.
* On screen: the tactical panel shows **NAV FED (this frame)**, read from the roll artifact. Next to it is **clip nav (not fed)**. The title line says `nav feed: PER-FRAME v9 token + r8_nav (as trained)`. A heads dump with a different or unrecorded feed is REFUSED for that episode; the refusal is logged with `ok: false`.
* ⚠️ Resumability skips existing mp4s. Use a FRESH out-dir, never `final48852` (that dir holds clip-fed renders). If the RT table registers after a render started, delete the mp4s to re-render them with RT ON.
* Cost (from the original HANDOFF): render time was MEASURED at 18 s/episode on an idle box (ep114) and 24-38 s/episode beside a CPU dump, so **~45-90 min** for 139 episodes. Disk is ~8-14 MB/episode at CRF 18, ~1.5 GB (ESTIMATED in the original).
* Reel: `pick_episodes_r8.py --tree step48852ema_pf --out <raw>/PICKS_step48852ema_pf.json`, then `render_r8_video.py --tree step48852ema_pf ...` and `make_reel_r8.py` (both unchanged; `EvalBank` defaults to `auto`).

## Smoke-tested here (CPU only, the GPU lock untouched)
* `pytest test_r8v_feed_parity.py test_r8v_data.py test_r8v_geom.py`: **27 passed**. Mutation `R8V_MUTATE_FEED=clip`: **4 failed** (a, a, b, c), as designed. Logs: `raw/pytest_vidpf/`.
* The renderer still frame (`--still 15`, ep 0) on the pf CPU-smoke battery tree `C:/Users/Admin/refcv8_pf/smoke_batt/smoke40kema_pf` (EMA-40k, 2 clips x 3 windows) shows `NAV FED FOLLOW_ROAD (per-frame v9, args known)` and `clip nav (not fed) TURN_R (differs)`, plus the RT-OFF banner (no registered table yet). See `raw/smoke_vidpf/`.
* `dump_heads_r8.py` CPU smoke (`--device cpu`, EMA-40k, ep 0, 3 windows, `--banked-tree smoke40kema_pf`) was **EXIT 0 in 82.3 s** (MEASURED, `raw/smoke_vidpf/keys_heads_cpu_ep000_smoke40kema_pf.json`).
  * Wiring: `perframe_full`, patched refcv3_arm `64e6ffbb...`. That equals the pf battery's `roll_s0.json`.
  * Feed: 3/3 windows were fed token FOLLOW_ROAD while the clip token was TURN_R. All 3 had args known. The call matched the sidecar on every window.
  * Census ok; parity **PASS** (PF 0 fails, V9 0 fails, 3 discriminating). The v0 control diff was 0.0.
  * The traj diff vs the banked seed-0 roll is 5.75 m. That is informational: it is a sampler draw, os-only here versus the full arm set there.
* `render_all_r8.py` with `--heads-dir` from that dump on the smoke pf tree, ep 0: **ok**, 201 frames, 33.2 s wall, 6.8 MB mp4.
  * The map-head panel was filled from the per-frame heads. `RENDER_LOG` shows `nav_feed perframe_full`, `heads_feed perframe_full`, 3 fed tokens that differ from clip, and RT OFF.
  * Deliberate regression: the same heads with `nav_cmd_fed := nav_cmd_clip` were **refused** (exit 1, `fed nav / r8_nav differ from the tree's sidecar at ws [7, 12, 17]`).
* NOT tested: the GPU run, the full 4,754-window dump, the RT-ON path on a real per-frame npz (only on a synthetic chain in the tests; none exists yet), and a full 139-episode render.

# IMPL_CHOICES_MRATE (DRAFT for the Master Mind to freeze; nothing here is a result; the card `../PREREG_V8F_MRATE.md` is untouched)

## 1. What the arms are, in code
* W0 = `--wm-only --wm-k 1`: M2b's WM architecture (2,007,874 parameters, MEASURED equal to M2b `summary.json` WM 2,007,874), re-initialised under the run seed.
* W1 = `--wm-only --wm-k 3`: 2,139,202 parameters (+131,328: `enc1` and the three-frame level encoders). L1 reads the adapter tokens of frames t, t-1, t-2 (0.4 s); L2 reads L1 states 0.4 s apart (1.2 s); L3 reads L2 states 0.8 s apart (2.8 s). Same seed => every parameter that is not a level encoder is bit-equal to W0's (pytest-pinned).
* Both flags default OFF, so the four argvs below NAME them. Start state (the Log's reading): the adapter, readers and planner of the PLAIN `M2b_s{k}` (dev box `E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/M2b_s{k}/ckpt_last.pt`, s0 sha256 `601cb435...`, s1 `6104fb15...`) load strictly and bit-equal and are frozen; none of its WM tensors is loaded; hmax is not compared (16 vs 28).

## 2. Choices (proposed values)
| choice | value | basis |
|---|---|---|
| hmax | 28 | the card |
| data / windows | M2b's: `--exclude 540:600`, 4,512 clips, 306,566 train windows, `train_windows` start hist-1 = 3 | `window list does not depend on hmax`; the window list sha256 is recorded in `summary.json` |
| history before a clip's start | the data layer repeats the clip's first cached frame (`frame_ok` False); the WM losses do not read `frame_ok`; M2b's WM was trained the same way; W1 additionally repeats the window's first projection for the K-1 frames before it | unchanged data code; a window with t < 27 therefore trains on some repeated frames, equally for W0 and W1 |
| optimizer | AdamW, lr 3e-4, warmup 200, cosine to 0.1x, wd 1e-2, clip 1.0, batch 64 | M2b's argv defaults. MEASURED: the pre-clip gradient norm is about 5.4 on a synthetic batch (loss scale `w_wm` 0.1), so the clip of 1.0 is active at every step for BOTH arms alike |
| `w_wm` | 0.1 | M2b's; the only loss term, so it only rescales the gradient (clipped) |
| steps | 19,164 (`--steps-mult 4`, S = 4,791) | M2b's own step budget (`M2b_s0/summary.json`: steps 19164, steps_mult 4, batch 64); the gate asserts `steps_total` |
| seeds | 0 and 1; the run seed is the init checkpoint's seed | the card |

## 3. Time (one CPU WM-only step, batch 64, hmax 28, d 256, n_tok 40, d_feat 1024; `_time_step.py`, `logs/time_step_4thr.json`)
* MEASURED, 4 CPU threads, on a dev box SHARED with other sessions' jobs and a running pytest (so contended): W0 median 0.71 s (min 0.70), W1 median 1.16 s (min 0.74) per step; the adapter forward alone 0.26 to 0.28 s. Uncontended figures will be lower; treat min as the better estimate.
* ESTIMATED, box 2 (GPU), per run of 19,164 steps. Basis: MEASURED M2b on box 2 = 15,741 s / 19,164 = 0.82 s per FULL step (all heads, hmax 16, data loading included). WM-only has far less compute, so the step is bound by the window loader, which reads 28 instead of 16 frames: if loader-bound, 0.82 x 28/16 = 1.4 s => 7.6 h per run, 30.6 h for four sequential runs; if compute-bound, about 0.3 s => 1.6 h per run, 6.4 h for four. The range is wide because no box-2 WM-only step exists. FIRST STEP of the box-2 package: a 200-step pilot of W1 s0, read `sec` from `train_log.jsonl`, then fix the steps. If the pilot shows more than 4 h per run, the fallback is 9,582 steps (half), the same for all four runs.

## 4. The four argvs (box 2; paths from the C0/recft argv; `--steps-mult 4` gives `steps_total` 19164)
Common tail (every run): `--arm M2b --grid pool4x10 --hmax 28 --wm-only --batch 64 --steps-mult 4 --kin-mode yaw --exclude 540:600 --device cuda --cache-dir D:\tanitad\cache_full --v9-train D:\tanitad\data\v9\v9_labels_train.npz --v9-train-md5 83e30bd3791f0767707eade5f4671759 --lead-train D:\tanitad\data\v9\lead_train4572.npz --speed-train D:\tanitad\data\v9\refcv6_speed_max_v8_train.jsonl --join-train D:\tanitad\runs\v8f_mini_full\join_train.npz --p10 D:\tanitad\runs\v8f_mini_full\p10\p10_prior.pt --tracks D:\tanitad\cache_full\native_tracks.pt`
| run | per-run flags |
|---|---|
| W0 s0 | `--seed 0 --wm-k 1 --init-ckpt D:\tanitad\runs\v8f_mini_full\M2b_s0\ckpt_last.pt --out D:\tanitad\runs\v8f_mrate\W0_s0` |
| W0 s1 | `--seed 1 --wm-k 1 --init-ckpt D:\tanitad\runs\v8f_mini_full\M2b_s1\ckpt_last.pt --out D:\tanitad\runs\v8f_mrate\W0_s1` |
| W1 s0 | `--seed 0 --wm-k 3 --init-ckpt D:\tanitad\runs\v8f_mini_full\M2b_s0\ckpt_last.pt --out D:\tanitad\runs\v8f_mrate\W1_s0` |
| W1 s1 | `--seed 1 --wm-k 3 --init-ckpt D:\tanitad\runs\v8f_mini_full\M2b_s1\ckpt_last.pt --out D:\tanitad\runs\v8f_mrate\W1_s1` |
Lr, wd, warmup, clip and `--w-wm` are left at the parser defaults 3e-4, 1e-2, 200, 1.0 and 0.1 (= M2b's argv). To verify on box 2 before any run (do not ssh from here): the box-2 `M2b_s{k}\ckpt_last.pt` sha256 equals the dev-box file's (the Log says use the box-2 copy only if so), the paths and md5 above, free D: space, no python process, and the hmax-28 batch builds (`feats` 64 x 28 x 40 x 1024 fp16 = 147 MB per batch, prefetch depth 3).

## 5. The eval windows (T0, eval-141, 28 real frames, identical for W0, W1 and M2b's own WM)
* Windows: the eval-141 grid windows (`wmact_lib.open_windows(28)`: `V8FMiniWindows(hmax=28)`, NOW = grid step t, frames t-27 .. t), kept iff `frame_ok.all(1)` (all 28 frames real). The same list for every arm, its sha256 recorded.
* Start frame: window frame 14 for EVERY level (the earlier suggestion "start at 14" is exact only with this offset). Procedure: `z1 = wm.z1(adapt)` over the 28 frames, then `stack.encode(z1[:, 2:], A[:, 2:27])` (slice from frame 2). The sliced grids put an L2 state at every even and an L3 state at every 4th frame from 2, so frame 14 is a state at all three levels: L1 sliced index 12, L2 index 6, L3 index 3. All three are first fully clean: their reads reach back to window frame 0 and touch no repeated frame (pytest `check_history_extension`: L2 index >= 2 and L3 index >= 3 are unchanged when 4 older frames are prepended).
* Targets: L1 k 1/5/10 -> frames 15/19/24; L2 k 1/3 -> frames 16/20; L3 k 1/3 -> frames 18/26. The last real frame is 27, so L3 k 3 at frame 26 fits and 7 L3 states exist (indices 0..6).
* `level_pack` in `wmact_lib` starts at `enc.z[l-1][:, 0]` and uses `A[:, :H]`: the eval agent changes it to start at the indices above and to take the actions from there (not edited here).

## 6. Controls and where they are pinned (all CPU, logs in `mrate/logs/`)
* K = 1 identity, causality, reads, history extension, hmax-28 shapes, 4 mutations: `tests/test_v8f_mrate.py` (16 passed with the 45 hjepa tests = 61).
* G-LIVE (WM terms finite, every WM group non-zero gradient, adapter / readers / heads / planner exactly 0.0; `--w-wm 0` and a deleted adapter `.detach()` fail), planner identity (bit-equal after 3 steps and after a whole run), the start-state read-back, G-DVB refusals, resume: `tests/test_v8f_mrate_train.py` (13 passed).
* Not yet run on box 2: the chain's `--selfcheck-only` gate with the real files; the real-data planner identity on 64 eval windows (control 1 of the card), which the eval agent runs from the saved checkpoints with the earlier eval path.

## 7. Open points for the Master Mind
1. The step count (section 3) and whether the 200-step pilot is allowed before the freeze.
2. The pre-clip gradient norm is about 5.4 (clip 1.0 active every step, equally for both arms); say if `--clip-grad` should be raised for this arm (it would then differ from M2b's argv).
3. W1's `enc1` is built on a private RNG stream so W0 and W1 share their other parameters; its init is therefore not the one a plain `WMAux` would draw. Frozen by the pytest, noted here for the record.

## 8. Master Mind decisions (2026-11-07), and the freeze
1. **Steps:** 19,164 (`--steps-mult 4`), M2b's own budget, as section 2 proposes.
2. **Pilot (timing only):** before the four runs, box 2 runs 200 steps of W1 s0 (the slower arm) with its exact argv plus `--steps 200` (or the trainer's equivalent) into a throw-away output folder. Only the median s/step over steps 50 to 200, peak memory and the selfcheck are read; no WM metric is read or logged for a decision.
   * Rule, committed now: if (pilot median s/step) x 19,164 x 4 > 40 h, all four runs use 9,582 steps (`--steps-mult 2`) instead. The decision depends only on the step time.
3. **Clip:** 1.0 for both arms, M2b's value. The card requires the same optimizer as M2b; the clip is active at every step, equally for W0 and W1.
4. Everything else in sections 1 to 6 holds as written. The file is frozen with this section; its sha256 is logged in the card.

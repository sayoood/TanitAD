# r9clk — are the agent / 3-D box labels 0.2 s OLD relative to the image they are paired with?

**VERDICT: CONVENTION. There is NO physical lag.** The −0.1955 s that gate-2 measured is the **zero of the egomotion log's own clock**
(`ts0 = −0.1953 s` median, range −0.2000 … −0.1899 over 220 clips), not a 2-frame (`n_stack − 1`) shift. The join's `t_s` is on the **raw
clip clock** (camera / obstacle / egomotion timestamps); the clip-clock sidecar is on the **label timeline** (`ts − ts[0]`, zero at the
egomotion log's first sample, 0.195 s EARLIER than the raw clock's zero). The labels are aligned with the image row to **±1.4 ms**.
Nothing is wrong in the join, in refcv8's labels, or in any arm trained on them. **No label file, join or trainer was changed.**

Evidence class: **MEASURED (ours)**, Thor CPU read-only + dev-box CPU, artifacts under `raw/`. Tier: a data-label diagnosis; **no model
performance claim** (probe 3 is T0 teacher-forced detection and is corroboration only).

## 1. G2's number is reproduced, and it is not a probe defect

`clk_probe_thor.py` Stage A recomputes G2's formula verbatim (`t_s − (grid_start_s + frame·dt)`, 220 clips / 41,537 records): median
**−0.1953**, min −0.2006, max −0.1874 (G2: −0.1955 / −0.2002 / −0.1896, 107 clips). The arithmetic was right; the **interpretation** ("= 2 frames
= `n_stack − 1`") was the coincidence: `2·dt = 0.2013 s`.

Two hypotheses predict different per-clip numbers, and the data pick one outright:

| per clip (n = 220) | LAG by 2 frames predicts | CONVENTION predicts | measured |
|---|---|---|---|
| G2 error − (−2·dt) | 0 | not 0 | median **+6.05 ms**, MAD 2.5 ms, **0 of 220 within 1 ms** |
| G2 error − ts0 (egomotion log's first timestamp) | not 0 | 0 | median **−0.05 ms**, MAD 0.15 ms, **217 of 220 within 1 ms** (max 5.1 ms) |
| slope of per-clip G2 error on ts0 | 0 (const) | 1 | **0.988**, corr 0.988 |

The egomotion parquet of every clip starts 0.19–0.20 s before the raw clock's zero (e.g. first timestamp −196,667 µs). `build_obstacle_join.py:347`
keeps that raw clock (`t = timestamp / 1e6`, no zeroing); `egomotion_source.py:123-124` zeroes it (`ts = ts − ts[0]`), and
`build_clip_clock_sidecar.py:99` writes `grid_start_s = a − t[0]` on that zeroed clock. The bridge is exact:
**sidecar on the raw clock = `ts0 + grid_start_s + frame·dt_s`**, and it equals the camera-timestamp grid to **≤ 1.7e-7 s** on all 41,537 records.

## 2. Probe 1 — source trace (file:line, tip 513e012c)

| hop | what | where |
|---|---|---|
| image of a row | row `e` = stack of raw frames `e, e+1, e+2`; **newest = raw `e + n_stack − 1`**, pose = payload `poses[e+2]` | `comma2k19.py:633-641`, `v2_dataset.py:36,421` |
| image time | `t_query = linspace(t_frames[0], t_frames[-1], n_target)`, raw clock; poses = `signals_at(ego, t_query)` | `build_lead_block_b1.py:179-205` |
| join time | `t_s = register_poses_to_time(poses, ego.t, …)` → time of raw frame `i` on the **raw clock** | `lead_source.py:137-211`, `build_b1_agent_join.py:384` |
| join record | `frame = i` (RAW), `frame_idx = i − (n_stack−1)`, `t_s = t_s[i]`; agents composed `rig@sample → world → ego@pose[i]` | `build_b1_agent_join.py:419-421`, `build_obstacle_join.py:406-440` |
| trainer, 2-D | NOW row `f = t + w − 1`, looked up by `frame_idx` | `refc_v3_train.py:4767` → `_agent_item` 3895 |
| trainer, 3-D | key `f + n_stack − 1` = RAW `frame` | `refc_v3_train.py:3960-3985, 10522-10547` |
| label timeline (v7/v8) | `t_true = grid_start_s + (t + w − 1 + n_stack − 1)·dt_s`, zero = ego-log first sample | `refc_v3_train.py:4340`, `clip_clock.py:21` |
| **G-CLOCK** | `|t_trainer − t_true| ≤ 0.05` on the **label timeline**; it does **not** read the agent / 3-D join | `launch_gate.py:29-31, 187-188, job_clock 7327-7345` (rule_checked reads `V3Dataset._now_s` only) |

Row `e` ↔ image raw `e+2` ↔ join `frame_idx = e` (= raw `e+2`) ↔ 3-D `frame = e+2`: **one frame, consistent at every hop**. The trainer's join key
was already validated (`2026-09-17-refcv6-perception-training/raw/join3d_offset_sweep.json`: 2,955/2,955 agents exact at +2, none elsewhere).
The 3-D join is the 2-D join annotated: **120,000 / 120,000 lines identical** in key, `t_s`, `cx`, `cy`, `track_id`; 4,146,672 agents, all with `cz`/`h`.

## 3. Probe 2 — the decisive physical test (static labels; `clk_probe_thor.py` Stage B, `raw/clk_probe_thor_full220.json`)

First 220 train clips of `b1_train_plus_eval_agents.jsonl.xz` in file order; **173 clips / 2,597 static tracks / 206,957 label instances**
(raw tracks with world radius95 ≤ 0.30 m, ≥ 25 samples, ≥ 2.5 s, ego ≥ 3 m/s — thresholds fixed before the scan). Each label in the ego frame of
its record is put in the world with the ego pose at **(camera-grid image time `T_i` + δ)**; error = distance to the track's independent world position.

| δ (s) | −0.40 | −0.30 | **−0.20** | −0.10 | −0.05 | **0** | +0.05 | +0.10 | +0.20 |
|---|---|---|---|---|---|---|---|---|---|
| mean error (m) | 3.42 | 2.57 | **1.71** | 0.86 | 0.44 | **0.10** | 0.44 | 0.86 | 1.71 |

**δ\* = +0.0001 s, 95 % clip-bootstrap CI [−0.0001, +0.0004]** (1,000 resamples). Model-free dispersion statistic: δ\* = +0.003 s.
Known-answer controls (same 206,957 instances): labels shifted in software by **+0.10 s → +0.1001 [0.0999, 0.1004]**, −0.10 → −0.0999, −0.20 →
−0.1999; a deliberate **mis-key of +2 frames → −0.2009 [−0.2011, −0.2006]** (expected −0.2013). The probe is sensitive to 1 ms; the labels sit at δ = 0.
Supporting identities: the v2ep payload `poses[i]` equals egomotion at `T_i` to **1.0e-6 m**, and at `T_{i±1}` it is off by 0.827 m (220/220 clips, `n_stack` 3,
`n_poses == n_target`). `t_s − T_i` (join vs the camera-timestamp grid, same raw clock): median −0.04 ms, p05…p95 −0.42…+0.45 ms, min −1.35 ms, **max +10.9 ms (4 clips > 1.5 ms)**.
**Honest scope:** because `C0` pins `poses[i] = ego(T_i)`, δ\* = 0 certifies that the label's ego frame and agent time are anchored at the camera-grid time; it cannot see a
camera-vs-obstacle clock error inside the dataset (probe 4 addresses that, weakly).

## 4. Probe 3 — banked PERC8 packs (`perc8_lon_slope.py`, `raw/perc8_lon_slope.json`) — CORROBORATION ONLY, cannot decide

Held-out `perc_det.pkl` (refcv8 EMA-48,852): 14,215 matched TP pairs, 126 clips, ego speed mean 8.19 m/s, ego speed from the eval manifest at NOW = `ws + w − 1` (w = 8;
w = 6 and 10 give the same). Signed `pred_x − gt_x` on speed: **slope +0.0008 s [−0.0009, +0.0029]**, intercept −0.015 m, mean dx −0.008 m; 0–40 m and v > 3: −0.0008 [−0.0036, +0.0015].
Control: GT made 0.2 s old moves the slope to −0.1992 (exactly −0.2). A model that predicts where the object IS under 0.2 s-old GT would read −0.2; it reads 0. **⚠️ The model takes ego
speed as an input (PI ruling), so it could in principle learn a lag; this probe therefore excludes nothing alone** — it shows only that the matched boxes carry no speed-dependent residual.
The train packs (`train_diag.packs.pkl`) carry no window identity (`ep` only), so no slope is possible on them (not used). Pair reconstruction (min-distance among matched slots)
reproduces the pack's own `pair_err` multiset in 3,318 of 3,346 windows; the other 28 are included.

## 5. Probe 4 (added) — pixel truth (`pixel_overlay_thor.py`, `raw/overlay/`) — n = 3, by eye

Parked car at 14–17 m, ego 9–13 m/s: the banked cuboid (green) vs a 0.2 s-old label of the same car (red) projected into the SAME v2ep image `i` with the clip's own extrinsics.
The old-label variant moves 2.2 / 1.8 / 2.7 m deeper (= 0.2·v) and 35 / 14 / 37 px sideways. In `overlay_384cb23868d0_f155.jpg` green sits on the silver car and red is off it;
the other two are low-contrast (night; small displacement). **Illustrative, not statistical.**

## 6. What the residual offsets really are (all small, none 0.2 s)

* **Frame selection:** the image is the first camera frame AT OR AFTER `T_i` (`build_pose_sync_sidecar.py:1-8`): image newer than label by **0–33.4 ms, mean 16.5 ms, max 35.4 ms**
  (`refcv8_pose_sync_sidecar.jsonl.meta.json`, read on Thor) = 8 % of 0.2 s, ~0.2 m at 13 m/s. Already measured by X10; consumed via `--pose-sync-sidecar`.
* **Agent sample jitter:** the join labels each track with the sample nearest `t_s` within 0.06 s. Over 60 clips / 443,242 labels: `t_sample − t_s` mean **−0.02 ms**, median −0.17 ms,
  p05…p95 ±0.045 s, |max| 0.060 s; track period median 0.09999 s. **Zero-mean jitter, no bias.**

## 7. Consequences and the one thing to fix

* **refcv8's box AP@2 m (0.328) and every banked agent / box metric were scored on time-aligned GT.** No rescoring, no relabel, no shift.
* **G-CLOCK must keep the label timeline** (`ts − ts[0]`; `launch_gate.py:29-31` over `refc_v3_train.py:4340`). It is correct as written and does not touch the join.
* **A join-vs-sidecar time comparison (gate-2's 0.05 s bar) is a FALSE POSITIVE on every clip unless written on ONE clock.** Compare on the raw clock:
  `t_s_join − (ts0 + grid_start_s + frame·dt_s)`, `ts0` = first egomotion timestamp / 1e6 (observed |Δ| ≤ 1.4 ms, p05…p95 ±0.45 ms, 4 clips to 10.9 ms; a 0.02 s bar passes 220/220).
  Any future consumer that fuses join times with v7/v8 label times must add the bridge `label = raw − ts0` (**+0.1953 s** median).
* Suggested retraction-log class (MM decides): *two clocks with different zeros compared without the bridge, and the magnitude matched a plausible mechanism (`2·dt = 0.2013`)* —
  the discriminator that settles it is the **per-clip** relation to `ts0`, not the pooled median.

## 8. Not established / UNVERIFIED

* Dataset-level camera ↔ autolabel synchronisation inside PhysicalAI-AV is **assumed**; only probe 4 (n = 3, by eye) and probe 3 (non-discriminating) touch it.
* Sample = the first 220 train clips of the join in file order (173 with static tracks); the 26,394-line EVAL join was not scanned separately (same builder, `meta.json`).
* Static-radius sensitivity (0.15 / 0.60 m) was **not run** for Stage B (probe 4 used 0.15 m).
* Stage B pooled the clip-weighted mean; the per-clip curve is in `A_rows` / the JSON only for Stage A.

## 9. Reproduce

`scp clk_probe_thor.py tanitad-thor:/home/nvidia/r9clk/ && ssh tanitad-thor 'OMP_NUM_THREADS=1 /home/nvidia/venvs/tanitad-train/bin/python clk_probe_thor.py --n-clips 220 --n3d-lines 120000 --out out.json'`
(≈ 3 min, CPU). `perc8_lon_slope.py --packs <perc_det.pkl> --manifest <eval139 _v2manifest.pt>` runs on the dev box with the `tanitad` venv. No clip ids are stored: `sha12` only.

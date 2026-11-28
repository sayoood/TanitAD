# SFRESH_RESULT: PREREG_V8F_SFRESH executed exactly as written (2026-10-25)

Dev box, CPU only (`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=8`), ZERO training, box 2 read-only (sftp `get` only). Code at tip `583cfc02`, exported with `git archive` (not the E: tree). Every number is **MEASURED (ours)** from the artifact named; tier **T1-class open loop, DIAGNOSTIC: fresh ages are not the deployed schedule**; estimator **paired episode-cluster bootstrap, B = 2,000, seed 0, 141 episodes (`taniteval/taniteval/ci.py`, via `v8f_mini_eval.paired`)**, the same call `compare.json` used. The 50 m / 80 m checkpoint labels are valid on 2,319 / 2,294 of the 2,399 windows (137 / 136 episodes); the interval answers "would another draw of EPISODES say this?" and nothing about training runs, which is why both seeds ride side by side with their difference F. Inference is deterministic (no sampler), so no inference-seed floor applies.

`OUT` = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/sfresh`.

## 0. Outcome (the PREREG outcome table, applied verbatim)

**STALENESS ARTEFACT.** Both conditions hold:

* Q-LAYER <= 0 at the point at both seeds: **-2.4331 m (s0), -2.5260 m (s1)**.
* Q-STALE < 0 separated at both seeds: **-5.7157 [-6.4437, -5.0621] (s0), -5.7348 [-6.4604, -5.0657] (s1)**.

The other rows of the table do not fire: Q-LAYER > 0 and separated at both seeds is false (the sign is the opposite, at both seeds).

**Committed next step, quoted from the PREREG:** *"STRAT-1's failure was the metric. Clause 1 is re-read on fresh ages: it PASSES iff Q-LAYER < 0 separated at both seeds. Then the deployed-schedule cost (Q-STALE) becomes the strategic lever: a faster S tick or a re-anchored checkpoint."*

**Clause 1 re-read on fresh ages: PASSES.** Q-LAYER < 0 and separated at both seeds: s0 -2.4331 [-3.2220, -1.7394], s1 -2.5260 [-3.3325, -1.7945]. Reported as written, both ways: under the registered default (`--ages cycle`) clause 1 FAILED (learned S - baseline +3.2826 [+2.0633, +4.4673] / +3.2088 [+1.9640, +4.3805], reproduced below); the PREREG's table is what authorises the fresh-age re-read, and the re-read passes. The register row needs both facts.

## 1. The two decisive reads (S checkpoint position error at 50 m, metres, paired, n = 2,319 windows)

| read | definition | seed 0 | seed 1 |
|---|---|---|---|
| **Q-STALE** | fresh - banked cycle | **-5.7157 [-6.4437, -5.0621]**, separated | **-5.7348 [-6.4604, -5.0657]**, separated |
| **Q-LAYER** | fresh S - nav-only baseline | **-2.4331 [-3.2220, -1.7394]**, separated | **-2.5260 [-3.3325, -1.7945]**, separated |
| (reference) | banked cycle S - nav-only baseline | +3.2826 [+2.0633, +4.4673] | +3.2088 [+1.9640, +4.3805] |
| seed difference F = abs(s0 - s1) | | Q-STALE 0.0191 m | Q-LAYER 0.0929 m |

Levels (single arm, episode-cluster CI), 50 m: banked cycle **8.0122** [7.3116, 8.7866] / 7.9384 [7.2883, 8.6455]; fresh **2.2965** [1.7604, 2.9642] / 2.2037 [1.7471, 2.7400]; nav-only baseline **4.7297** [3.7182, 5.9036]. The effect is 26x the seed difference on Q-LAYER and 300x on Q-STALE. (`OUT/sfresh_numbers.json`, `OUT/sfresh_ade_paired.json`.)

## 2. Controls (all run before the reads were interpreted; all pass)

| control | result |
|---|---|
| Checkpoint identity | sha256 of `M2b_s0` / `M2b_s1` `ckpt_last.pt` = `601cb435...71dcc6f8` / `6104fb15...e50c2e6`, equal to `raw/2026-10-25-kind/run/ckpt_sha256.json` |
| Eval cache | 141 clips (2.947 GB), md5 of every file equal to box 2's `cache_full/eval/manifest.jsonl`; 0 bad (`OUT/cache_md5_verify.json`) |
| Code drift | `git diff 009d1cba 583cfc02` (the tip the banked KIND eval ran at) touches only `v8f_mini.py`, and only the MIXO `code_mix` training path (default `whole`); the eval path is unchanged |
| **Re-run of the banked `--ages cycle` eval on CPU** (`OUT/eval/eval_M2b_s{0,1}-sfresh-cycle.*`) | s0: S pos err mean 30/50/80 m = 7.023118 / **8.012241** / 11.205851 vs banked 7.023118 / 8.012241 / 11.205851; max abs per-window diff in S error 7.2e-5 m, in the 6 s plan 5.3e-5 m; ADE6 2.1753 = 2.1753, ADE2 0.3155 = 0.3155; paired rerun - banked at 50 m exactly 0.0 [0.0, 0.0]. s1: S pos err at 50 m 7.938449 vs banked 7.938448, max abs 9.5e-5 m (S error) / 9.9e-5 m (plan), ADE6 2.0546 = 2.0546. **Tolerance seen: <= 1e-4 m (CPU float32 vs box 2's GPU).** |
| Banked-side reproduction of `compare.json` (strat block) | recomputed s0 +3.2826 [2.0633, 4.4673] n 2,319, s1 +3.2088 [1.9640, 4.3805]; at 80 m +0.2107 [-1.6955, 1.9520] and +0.2190 [-1.6685, 1.9718], n 2,294: **4/4 match `compare.json`** (`OUT/controls_banked.json`) |
| Banked standard eval vs banked KIND own-code eval | `ck_pos_err`, `pred`, `ckpt` identical (max abs 0.0) at both seeds, so "the banked cycle reference" is one object |
| Re-expressing banked cycle checkpoints with `_ckpt_to_now` | reproduces the banked per-window S error exactly (max abs 0.0), so the heading rows below use the evaluator's own function |
| **Identity on age-0 windows** (`OUT/sfresh_extra.json`) | the 291 windows whose realised ages are (a_t, a_s) = (0, 0) under the cycle schedule feed the model what `--ages fresh` feeds it: `traj30`, `ckpt` and `goal_prob` are **bit-identical (max abs 0.0)** between the fresh and cycle runs at both seeds; on the other 2,108 windows `traj30` differs by 0.23 / 0.30 m on average. So the fresh arm changed the ages and nothing else |

## 3. Why Q-STALE is that large (mechanism, descriptive; decides nothing)

Banked cycle S error at 50 m by realised S age, against the same windows scored fresh (seed 0; seed 1 within 0.25 m on every row, `OUT/sfresh_extra.json`):

| S age (s) | n | cycle (m) | fresh (m) | ego travel S tick -> now (m) |
|---|---|---|---|---|
| 0.0 | 282 | 2.08 | 2.08 | 0.00 |
| 0.2 | 292 | 4.13 | 2.40 | 2.31 |
| 0.4 | 435 | 6.28 | 2.39 | 4.69 |
| 0.6 | 443 | 8.61 | 2.60 | 6.93 |
| 0.8 | 460 | 10.40 | 2.09 | 9.14 |
| 1.0 | 271 | 12.69 | 2.00 | 11.58 |
| 1.2 | 136 | 14.84 | 2.53 | 13.83 |

The cycle-minus-fresh gap grows by about 0.8-0.9 m per metre the ego travelled since the tick, and the fresh error is flat at 2.0-2.6 m at every age: the evaluator scores a checkpoint defined from S's tick against labels defined from NOW, so the metric pays for the distance driven since the tick. Q-STALE on the 2,037 aged windows only: -6.5070 [-7.3400, -5.7521] (s0), -6.5287 [-7.3566, -5.7603] (s1). Mean realised S age in the cycle schedule 2.81 steps (0.56 s), mean ego travel 6.44 m. Part of Q-STALE is therefore definitional (the frame the label lives in), which is exactly the confound the PREREG named; what is a model property is that the fresh checkpoint is accurate.

## 4. Beside rows (descriptive; the PREREG says the 80 m reading decides nothing)

S checkpoint position error, paired, metres, n = 2,322 / 2,319 / 2,294 windows at 30 / 50 / 80 m:

| | seed | 30 m | 50 m | 80 m |
|---|---|---|---|---|
| Q-STALE (fresh - cycle) | s0 | -6.0458 [-6.7665, -5.3836] | -5.7157 [-6.4437, -5.0621] | -5.3186 [-6.0479, -4.5984] |
| | s1 | -6.0512 [-6.7743, -5.3990] | -5.7348 [-6.4604, -5.0657] | -5.3177 [-6.0413, -4.5946] |
| Q-LAYER (fresh S - nav-only) | s0 | -1.0135 [-1.3608, -0.7007] | -2.4331 [-3.2220, -1.7394] | **-5.1079 [-6.5567, -3.7592]** |
| | s1 | -1.0619 [-1.4257, -0.7478] | -2.5260 [-3.3325, -1.7945] | **-5.0988 [-6.5914, -3.7021]** |
| banked cycle S - nav-only | s0 | +5.0322 [+4.1803, +5.9011] | +3.2826 | +0.2107 [-1.6955, +1.9520], not separated |

At 80 m the sign flips from "not separated" (banked) to separated-better (fresh) at both seeds; per the PREREG this decides nothing.

S checkpoint HEADING error (deg), level and paired (`OUT/sfresh_numbers.json`, `heading_deg`):

| | s0 30 / 50 / 80 m | s1 30 / 50 / 80 m |
|---|---|---|
| cycle (banked) | 4.20 / 6.69 / 13.05 | 4.04 / 6.72 / 13.60 |
| fresh | 3.67 / 6.06 / 12.11 | 3.44 / 6.00 / 12.52 |
| nav-only baseline | 7.64 / 10.78 / 16.83 | 7.64 / 10.78 / 16.83 |
| heading Q-STALE at 50 m | -0.6235 [-0.8730, -0.3941] | -0.7144 [-0.9660, -0.4661] |
| heading Q-LAYER at 50 m | -4.7170 [-6.1755, -3.3462] | -4.7756 [-6.1626, -3.3827] |

Heading Q-LAYER is below zero and separated at 30, 50 and 80 m at both seeds (80 m: -4.7143 [-6.3386, -3.0875], -4.3126 [-6.0977, -2.5702]).

## 5. The four families under fresh ages (descriptive; same code, same 2,399 windows; `OUT/eval/eval_M2b_s{0,1}-sfresh-fresh.json`, banked = `raw/full/eval/eval_M2b_s{0,1}.json`)

No family carries a bar in this PREREG, so none is PASS / FAIL here. Fresh vs banked cycle, seed 0 | seed 1:

| family | metric | s0 cycle -> fresh | s1 cycle -> fresh |
|---|---|---|---|
| ADE (paired, fresh - cycle) | ADE6 (m) | 2.1753 -> 2.1231; delta -0.0522 [-0.0782, -0.0288] | 2.0546 -> 1.9891; delta -0.0654 [-0.0895, -0.0427] |
| | ADE2 / FDE6 | 0.3155 -> 0.3149 (delta -0.0007, not separated); 6.281 -> 6.110 (delta -0.1711 [-0.2579, -0.0942]) | 0.2914 -> 0.2929 (delta +0.0015, not separated); 5.858 -> 5.627 (delta -0.2308 [-0.3162, -0.1556]) |
| LONGITUDINAL, speed | speed MAE (m/s) / accel MAE (m/s^2) | 1.067 -> 1.053 / 1.206 -> 1.186 | 0.911 -> 0.912 / 0.931 -> 0.939 |
| | along-track MAE (m) / within 1.0 m/s | 1.810 -> 1.775 / 0.645 -> 0.646 | 1.608 -> 1.586 / 0.696 -> 0.696 |
| LONGITUDINAL, distance keeping (6 s block, 595 at-risk, LEAD windows 743) | headway (m) / time-gap (s) / min-TTC (s) | 25.62 -> 25.57 / 4.022 -> 3.974 / 19.40 -> 19.29 | 25.54 -> 25.46 / 3.950 -> 3.942 / 19.52 -> 19.34 |
| | LEAD-VIOL (share of at-risk) | 0.0353 -> 0.0403 | 0.0605 -> 0.0588 |
| LATERAL | heading MAE (deg) / yaw-rate MAE (deg/s) | 3.977 -> 3.810 / 6.161 -> 6.063 | 3.715 -> 3.502 / 5.357 -> 5.364 |
| | curvature MAE (1/m) / cross MAE (m) | 0.0444 -> 0.0432 / 0.766 -> 0.738 | 0.0322 -> 0.0331 / 0.864 -> 0.804 |
| TACTICAL | lat / lon macro-F1 (heads) | 0.816 -> 0.835 / 0.499 -> 0.506 | 0.829 -> 0.848 / 0.510 -> 0.514 |
| | goal macro-F1; lateral-decision kappa (from trajectory) | 0.262 -> 0.262; 0.665 -> 0.676 | 0.262 -> 0.279; 0.632 -> 0.650 |
| CONSISTENCY | (a) any-rule violation rate; (b) self-consistency | 0.1596 -> 0.1622; 0.770 -> 0.743 | 0.2097 -> 0.2118; 0.857 -> 0.850 |
| STRATEGIC | S-checkpoint position error 30 / 50 / 80 m | 7.02 / 8.01 / 11.21 -> 0.98 / 2.30 / 5.89 | 6.98 / 7.94 / 11.21 -> 0.93 / 2.20 / 5.90 |

Distance keeping is `taniteval.lead_metrics` through the repo's own `v8f_dk_read.py` at the same tip, run on both the fresh and the banked cycle npz (`OUT/dk/dk_read_fresh.json`, `OUT/dk/dk_read_cycle.json`; in the fresh directory the files named `M2b_s*` ARE the fresh arms, and the `M1_s*` rows are the banked M1 files the reader insists on and are not part of this read). Only per-arm means are shown; no fresh-minus-cycle paired test was run on distance keeping, and the differences are small against the per-arm intervals (e.g. 6 s time-gap 3.974 [3.064, 5.087] fresh vs 4.022 [3.104, 5.269] cycle, s0). Tier T1-class open loop: the lead's logged future does not react to the arm. The STRATEGIC family is reported as its S-checkpoint reads, as the task specified; the layer pause of `PI_DECISION_QUEUE.md` ITEM 25 concerns the 4-layer rule for the programme's strategic layer, and this read does not lift or extend it.

## 6. What this does and does not establish

* **Establishes (MEASURED, two seeds, deterministic inference):** the STRAT-1 first clause's failure at 50 m was a staleness artefact of the default evaluation: with S and T ticking at NOW, the learned S checkpoint beats the nav-only kinematic baseline at 30, 50 and 80 m in position and in heading, separated at both seeds, and the age-0 windows of the default evaluation are bit-identical to the fresh run.
* **Does not establish:** deployed behaviour. Fresh ages are a diagnostic; the deployed schedule ticks S at up to 1.2 s old (here mean 0.56 s), and Q-STALE is the price of that schedule in this metric. The interval is blind to training-run variance; the two seeds agree to 0.02 m (Q-STALE) and 0.09 m (Q-LAYER) against effects of 5.7 m and 2.4 m, which is evidence on the training axis from n = 2 and no more. The windows without a valid RC-A label (77 / 80 / 105 at 30 / 50 / 80 m; 4 episodes carry no 50 m label) are excluded exactly as `compare.json` excluded them.
* **Not done:** M1 (the PREREG names M2b s0 / s1 only); no new prior, no training; no register or PREREG file was edited (the Master Mind lands this). `H-MINI-STRAT-1`'s row should record clause 1 as FAILED under the registered default ages and PASSING on fresh ages per this table, and its Q-STALE as the next lever.

## 7. Argv (the banked KIND own-code eval's, verbatim except paths and the one lever)

Banked (queue `:kindown`, `chain_v8f_kind.sh`): `run --ckpt <M2b_s*/ckpt_last.pt> --tag M2b_s*-kind-own --v9-eval v9_labels_eval139.npz --lead-eval lead_eval139.npz --speed-eval refcv6_speed_max_v8_eval.jsonl --kin-mode yaw --p10 p10_prior.pt --out-dir <eval> --eval-cache <cache_full> --tracks native_tracks.pt --project --families` (default `--ages cycle`). Run here: the same line with `--tag M2b_s*-sfresh-fresh --ages fresh` and local paths (`OUT/eval/argv_*.txt`, `OUT/scripts/run_eval.sh`). Inputs fetched from box 2 (`D:/tanitad/...`): the two checkpoints, `native_tracks.pt`, `p10_prior.pt`, `p10_eval_pred6.npy`, the 141 eval clips and their manifest, `v9_labels_eval139.npz` (md5 `6b5c7f207cffc3b7eb3cd527fd433599`), `lead_eval139.npz`, `refcv6_speed_max_v8_eval.jsonl`. Eval grid / kin / join come from the repo tip (`raw/eval_grid_windows.json`, `eval141_kin.npz`, `join_eval141.npz`), as on box 2.

## 8. Deliverable manifest (nothing staged, committed or pushed)

All under `OUT`:
* `SFRESH_RESULT.md`, `READ.json` (this report and the machine-readable verdict)
* `sfresh_numbers.json` (every Q read, levels, heading, families, re-run control), `sfresh_extra.json` (age-0 identity, age profile), `sfresh_ade_paired.json` (paired ADE, seed floors), `controls_banked.json`, `cache_md5_verify.json`
* `eval/` (4 evals: `eval_M2b_s{0,1}-sfresh-{fresh,cycle}.{json,npz,log}` + `_proj.npz`, `argv_*.txt`)
* `dk/` (`dk_read_fresh.json`, `dk_read_cycle.json`, `.out`, input npz copies)
* `scripts/` (`run_eval.sh`, `sfresh_analyze.py`, `sfresh_extra.py`, `sfresh_ade_paired.py`), `box2/` (sftp batches, listings, logs), `queue.cmd` (the tip's queue, source of the banked argv)
* `inputs/` (checkpoints, tracks, prior, v9 labels; 183 MB), `code/` (tip export, 55 MB)
* `cache/` (the 141 eval clips, 2,811 MB, left in place for reuse)

## 9. Deviations and incidents (reported, not hidden)

* **DISK RULE, partial breach by a tooling bug.** Before fetching, `df -k /c` read 8,242,284 KB free and the fetch needed about 3.14 GB, so 5.1 GB would remain: the rule's condition held. During the fetch OTHER processes consumed several GB of C: (free fell 5.17 -> 4.37 GB in 3 s at 10:05:27, then to a low of about 1.98 GB; it later recovered to 6.1 GB). My guard tripped at 4.37 GB and killed `sftp.exe`, but `pkill` did not stop the retry loop of `box2/fetch_eval.sh`, so it fetched the last 25 clips (~0.5 GB) while free was already below 4.5 GB. I then killed the loop. Nothing failed and nothing outside `OUT` was written, but the "stop below 4.5 GB" intent was not honoured for those 25 files. Free at hand-back is in the hand-back message. All 141 clips were md5-verified against the manifest afterwards.
* No `--basetemp`/pytest was used. No file outside `OUT` was created or deleted; box 2 saw only sftp `ls`/`get`.
* `cache/` is left in place (2,811 MB) for reuse by another read.

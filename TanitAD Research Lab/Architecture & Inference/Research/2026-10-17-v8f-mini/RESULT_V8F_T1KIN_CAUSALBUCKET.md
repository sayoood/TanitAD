# RESULT_V8F_T1KIN_CAUSALBUCKET: HACC's T1-kin margin over the floor does NOT need the future-derived speed bucket. Outcome: CAUSAL KEEPS THE MARGIN

`PREREG_V8F_T1KIN_CAUSALBUCKET.md` (`d2ed136d`, landed before any number with the causal bucket). Read 2026-10-26 by `raw/2026-10-26-t1kin-causalbucket/t1kc_read.py`, applied verbatim. Claim row `H-V8F-CAUSALBUCKET-1`. Zero training.

* **Tier T1 (T1-kin).** Self-action OPEN loop: at every 2 s replan (m = 10), the ego state the model reads is rebuilt from its own executed motion, and the camera frames stay logged. ⛔ Not closed loop.
* **The one change** against `RESULT_V8F_T1KIN_HACC`'s decisive run: `--limit causal`. At every tick j, the speed bucket is the sidecar's own containing-window rule (`tanitad.refs.refcv6_max_speed.speed_max_bin`) applied to the max MEASURED ego speed over rows [j − 20, j], the last 4 s. That speed is logged up to t0 and the model's own executed speed after it.
  * No future row, no label and no classifier output enters it.
  * The logged bucket it replaces is the max of the ego's REALISED speed over [t0 + 2 s, t0 + 6 s], one value per clip.
* **Data:** eval-141, 2,399 windows, 141 episodes; HACC checkpoints `M2b_s{0,1}_hacc.pt`. Dev-box CPU.
* **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000. It answers "another draw of EPISODES"; the two training seeds are the replicate. F = 0.1207 m, 2F = 0.2415 m.
* **Artifacts** (`raw/2026-10-26-t1kin-causalbucket/`):
  * `run/`: 4 run JSONs, the chain and its logs, the controls, and per-run float32 dumps with `dumps_manifest.json`.
  * The float64 originals: `E:/Projects/TanitAD-artifacts/v8f-mini-hacc/t1kin_causalbucket_dumps_f64_2026-10-26.tar` (564 files, 24,862,720 B, sha256 `0dbaf3a5…780871`, `tar -d` verified).
  * The tool the runs read is the `stack/scripts/v8f_t1_eval.py` of this commit (raw sha256 `d0ac8ce6…` in the export, which carried CRLF).

## The read (HACC with the causal bucket − P10, both in their own loop, m = 10; lower is better)

| row (family) | s0 | s1 |
|---|---|---|
| **ADE6, PRIMARY (m)** | **−0.7680 [−0.9177, −0.6258]** ✅ | **−0.7336 [−0.8776, −0.5984]** ✅ |
| ADE2 (m) | −0.0122 [−0.0164, −0.0074] better | −0.0107 [−0.0158, −0.0046] better |
| speed MAE (m/s), LONG | −0.3623 [−0.4564, −0.2683] better | −0.3876 [−0.4731, −0.3038] better |
| along-track MAE (m), LONG | −0.4739 [−0.6182, −0.3409] better | −0.5085 [−0.6266, −0.3974] better |
| accel MAE (m/s²), LONG | +0.0173 [−0.1468, +0.1799] n.s. | −0.2018 [−0.3638, −0.0556] better |
| heading MAE (deg), LAT | −0.6323 [−1.3106, +0.0256] n.s. | −0.9251 [−1.5349, −0.3448] better |
| **yaw-rate MAE (deg/s), LAT** | **+2.8851 [+1.3126, +4.5633] WORSE** | +1.1918 [−0.3093, +2.6964] n.s. |
| **curvature MAE (1/m), LAT** | **+0.0302 [+0.0188, +0.0442] WORSE** | **+0.0130 [+0.0057, +0.0217] WORSE** |
| cross-track MAE (m), LAT | −0.5572 [−0.6759, −0.4474] better | −0.4344 [−0.5677, −0.3085] better |
| TACTICAL / STRATEGIC | NOT COMPUTED: the T1 dumps carry no per-tick T code or S checkpoints (work item, as in `RESULT_V8F_T1KIN_HACC`) | same |
| distance keeping | NOT COMPUTED: `v8f_t1_eval` passes no lead track | same |

Mean ADE6 at m = 10: 2.675 / 2.710 m with the causal bucket, 2.614 / 2.649 m with the logged one, and P10 3.443 m.

**Outcome (the table, verbatim): CAUSAL KEEPS THE MARGIN.** PRIMARY holds at both seeds.
* ⇒ v8F's T1 result can be stated on an input built from the past alone.
* ⇒ `D-V8F-SPEEDBUCKET-T1` stops gating the headline.

## Controls (all read before the outcome)

| control | result |
|---|---|
| C-L1: `--limit logged` and `--limit off`, patched tool vs unpatched (`--max-windows 24`) | JSON and 24 dumps byte-identical, both modes |
| C-L2: rows after j never read | 12 new tests GREEN, together with the 23 of `test_v8f_t1.py`. **4 of 4 mutants RED**: reading row j + 1, a window from 0, the mode disabled, mean in place of max |
| C-L3: tick-0 bucket = an independent hand computation from the cache's logged speed (20 windows) | PASS, 20 / 20 |
| Reachability: causal ≠ logged on some ticks, and the plan moves > 1e-3 m | 1,527 / 1,536 of 7,197 model ticks differ; the tick-0 plan moves up to 12.5 / 11.5 m |
| The floor is untouched (P10 reads no bucket) | max \|floor − T1KIN floor\| = 0.0 m at both seeds |

⚠ **The first decisive launch was KILLED and re-run.**
* **What happened:** the C-L2 mutation proof was `sed`-editing the patched tool in the same tree for about 14 s, and both runs started at 01:19:00, inside that window. The outputs could have come from a mutant.
* **What was done:** they were killed by explicit PID, their partial output was banked in `run/killed_0119/`, and the chain was re-run. Every run now asserts the tool's sha256 (`d0ac8ce6…`) before and after.
* **The control runs:** the C-L1 runs started before the first mutant, and their outputs are byte-identical to the unpatched tool's anyway.
* Filed as a TRAPS entry.

## Beside (decides nothing)

* **What the causal bucket costs against the oracle one:** HACC(causal) − HACC(logged) ADE6 is +0.061 [+0.003, +0.120] / +0.061 [+0.019, +0.103] m. That is about F/2, separated but small. Yaw rate improves at s0 (−1.32 deg/s, separated).
* **The two buckets agree on 78.8 % of model ticks.** Confusion (logged × causal), s0:
  * the logged 30 km/h ticks: 1,843 read 30 and 526 read 50;
  * the logged 50 km/h ticks: 419 read 30 and 1,856 read 50.

  Most disagreements are one step either way. A causal bucket below the logged one means the ego later went faster than its past 4 s.
* ⭐ **This resolves the `--limit off` reading of `RESULT_V8F_T1KIN_HACC`.** There the margin vanished (+0.24 / +0.08 m). With a bucket that is PRESENT but causal, the margin is kept. So the collapse was mostly the model's response to an ABSENT input (out of its training distribution), not the oracle's information.
* **The lateral deficit is unchanged.** Curvature is separated-worse at both seeds and yaw rate at s0, exactly as with the logged bucket. It is `PREREG_V8F_HACC_LAT`'s target, being built.
* **`--limit causal_all`** (the whole past [0, j], instead of 4 s): HACC − P10 ADE6 is −0.7304 [−0.8711, −0.5979] / −0.6792 [−0.8221, −0.5457] m, separated. It agrees with the logged bucket on 71.6 % of ticks (5,153 / 5,152 of 7,197).
  * The 4 s window is the better proxy here, by 0.04 / 0.05 m.
  * It was fixed before data, and is not re-chosen.
  * The re-read with these runs (`read/T1KC_READ.json`) leaves every decision row identical to the decisive-only read (`read/T1KC_READ_decisive_only.json`).

## What the result is, and what it is not

* **It is v8F's first T1 result over the kinematic floor with no future-derived input.** The model living on its own executed motion for 6 s, with a speed bucket built only from the speed it has already driven, beats the floor driven the same way by 0.77 / 0.73 m of 6 s ADE at both seeds. It is separated-better on ADE2, speed, along-track and cross-track at both seeds, and on accel and heading at s1.
* ⚠ **Not dominance:** lateral curvature is worse at both seeds, and yaw rate at s0.
* ⚠ **Not closed loop:** the cameras are the human's. Their channel was bracketed at −0.79 / −0.71 m with frozen frames in `RESULT_V8F_T1KIN_HACC`. Frozen cameras together with the causal bucket is not measured here.
* ⚠ **The baseline is the programme's own floor,** not a published state of the art.
* ⚠ **Not a speed LIMIT:** the causal bucket is a past-speed proxy. PhysicalAI carries no limit sign (`D-MAXSPEED-ABSENT-ALL-LAYERS-1`).

## Next (committed by the PREREG's outcome row)

* `D-V8F-SPEEDBUCKET-T1` stops gating the headline. v8F's T1 statement is made on the causal bucket.
* When HACC-LAT's deployed arm exists, it is re-read with `--limit causal` as a beside run.
* **Beside the committed next:** the cameras-frozen + causal-bucket combination, a zero-training pair of runs, would bracket both open channels at once. It needs its own PREREG line before it is read.

# Band labels: schema (stream A)

Output root `F:/strat_out/band_labels/` (also the place the Master Mind banks from). Builders and tests are in this folder.
Evidence class of every statement below: MEASURED from the files named, unless marked otherwise.

## Common conventions

* **Clip universe.** `strategic_*` files: the v7.2 release clip lists (`_s2build-copy-20260919/release/v72/s2_labels_v7.2_{train,eval}.jsonl.gz`): train **4,572**, eval **147** (the 141-clip cache eval set is a subset: flag `clip__in_cache141`). `v10_labels_*`: the clips of the v9 manifests that were banked (train **4,369**, eval139 **139**), because the builder needs a v2 manifest per clip and none exists for the 203 other train clips (the `83e30bd3...` "4,572-clip v9 train" named in `v8f_mini_data.py:49-50` is not on this box).
* **Grid.** One row per 0.2 s cache step `j` of a clip: raw v2ep frame `k = 2 j`, log time `now_s = grid_start_s + k * dt_s` (the v9 clock rule: sidecar, else `clip_clock.pose_dt` of the v9 manifest poses, else nominal 0.1 s with start 0; `clip__clock_src` 0 / 1 / 2). Steps per clip `n = (T + 2 - 1) // 2 + 1` with `T` the v2 manifest rows (199 -> **101**); 101 when no manifest. Rows of all clips are concatenated; clip `c` owns rows `clip__off[c] : clip__off[c] + clip__n[c]`.
* **Join key.** `(sid, k)` with `sid = blake2b-8(clip_id) >> 1` (`clip__sid`). A v9/v10 row for the same clip and `k` carries the identical `now_s` (test `test_join_key_k_equals_2j`, bit-equal on 139 eval clips). The v9 rows start at `k = 2`; step 0 (`k = 0`) has no v9 row.
* **Frames.** Ego frame at NOW: x forward, y left, heading in radians relative to the NOW heading (wrapped to [-pi, pi]).
* **Ego privileged.** All labels are offline (ego future allowed for labels, inference stays vision only). `STRATEGIC` is reported PAUSED (PI item 25); these files carry labels only and say nothing about a pass.

## 1. `strategic_geom_{train,eval}.npz` (+ `.manifest.json`, schema `tanitad.band_labels.strategic_geom/1`)

| key | dtype, shape | meaning |
|---|---|---|
| `clip_ids` | str [C] | clip ids (clear text here; downstream joins by `clip__sid`) |
| `horizons_s` | float32 [9] | 6, 9, 12, 15, 18, 21, 24, 27, 30 |
| `k` | int16 [N] | raw frame, `2 * step` |
| `now_s` | float64 [N] | log time of the step |
| `ck_xyh` | float32 [N, 9, 3] | (x, y, heading) of the route point at NOW + h, ego frame at NOW. NaN where `ck_valid` is False |
| `ck_valid` | bool [N, 9] | strict mask: all rules below, gap rule at the v9 literal 1.0 s |
| `ck_valid_span` | bool [N, 9] | same without the gap rule (log gaps of any size tolerated, linear interpolation across them) |
| `ck_gap_s` | float32 [N, 9] | largest native-sample gap overlapping [now, now + h]; NaN beyond the log end. Lets a consumer pick its own gap threshold |
| `ck_why` | uint8 [N, 9] | first failing rule of the strict mask: 0 ok, 1 now before log start, 2 now+h beyond log end, 3 log gap > 1.0 s, 4 arc beyond the complete smoothing kernel, 5 chord sagitta > 0.25 m on the kernel footprint, 6 clip never moves (no path) |
| `clip__sid` int64, `clip__off` int64, `clip__n` int64, `clip__grid_start_s` float64, `clip__dt_s` float64, `clip__clock_src` int64, `clip__log_end_s` float64, `clip__in_v9_manifest` int64, `clip__in_cache141` int64 | [C] | clip table |

**Construction.** Same smoothing as the v9 `rc_*` checkpoints (`build_v9_labels.py` section 6): the 10 Hz log track resampled at 0.5 m in arc length, Gaussian sigma 8 m, kernel +-3 sigma. The only change is the index: arc `s = arc_of_ego_at(now + h)` (time-indexed), not a fixed look-ahead. NOW pose is the log interpolated at `now_s` (raw yaw, as v9).
**Differences from the rc validity rule (listed in full):** (a) the chord-sagitta check covers the kernel footprint of the point (`s +- 24 m`), the rc check covers the whole stretch NOW to the point; it never fired on the 4,572 train clips (`ck_why == 5` count is 0 over 461,539 x 9 train entries). (b) the gap rule is added (v9 `g_max_s`, applied to [now, now + h]).
**Heading convention.** The smoothed tangent, not the raw yaw at the point.

## 2. `v10/v10_labels_{train,eval139}.npz` (+ manifest, md5; schema `tanitad.v10_labels/1`)

Same fields, dtypes and row keys as `v9_labels_*` (`man["fields"]` equals the v9 manifest's list; test `test_v10_schema_pin_and_literals`), so the existing join in `v8f_mini_data.build_label_join` reads it unchanged. The only edits to the builder (`build_v9_verbatim.py` = byte copy of `stack/scripts/build_v9_labels.py`, md5 `128846cbd65e3268f980e6b2ed5aaa54`, which is the md5 in the v9 manifests; `build_v10_labels.py` = the edited copy). Every changed line:

| # | line | v9 | v10 |
|---|---|---|---|
| 1 | 34 `LITERALS["band_s"]` | `(2.0, 8.0)` | `(2.0, 6.0)` |
| 2 | 40 `LITERALS["h_abs_min_s"]` | `8.0` | `6.0` |
| 3 | 157 `h_obs(..., horizon=)` default | `8.0` | `LITERALS["band_s"][1]` |
| 4 | 800 `reversing_s(log, now, now + min(h, 8.0))` | `8.0` | `LITERALS["band_s"][1]` |
| 5 | 811 `b0, b1 = now + 2.0, now + min(h, 8.0)` | `8.0` | `LITERALS["band_s"][1]` |
| 6 | 844 `band_k` upper index `min(h, 8.0)` | `8.0` | `LITERALS["band_s"][1]` |
| 7 | 894-896 `v_end_ms` read at index 80 (`jh >= 80`) | 80 | `j_end = round(band_hi / 0.1)` = 60 |
| 8-10 | 1242, 1245, 1262 output names and schema string | `v9_labels_*`, `tanitad.v9_labels/1` | `v10_labels_*`, `tanitad.v10_labels/1` |

**Why more than two literals changed (disclosed).** `band_s[1]` is not read anywhere in the v9 builder (`grep band_s`: only `band_s[0]` at line 763); the 8 s horizon lives in the hard-coded `8.0` of lines 157, 800, 811, 844 and the index 80 of line 894. Changing only the two literals would have left every label on the 8 s horizon and changed the absence rule alone. Lines 3-7 make the literal act. Unchanged on purpose: the diagnostic `max_gap8_s` (line 780, its name says 8), all VLM windows (`t0 + off + 6.0`, line 761), `suppress_window_raw_s`, `h_min_any_s` 3.0.
**Reproducibility control.** `build_v9_verbatim.py` re-run on eval139 with the v9 inputs (md5s of lead, lc, labels, sidecar, manifest equal the v9 manifest's) gives `6b5c7f207cffc3b7eb3cd527fd433599`, the md5 of the banked `v9_labels_eval139.npz`. So every v9/v10 difference below comes from the lines above.

## 3. `strategic_tok_{train,eval}.npz` (+ manifest, schema `tanitad.band_labels.strategic_tok/1`)

Row layout identical to section 1 (same clip order, `clip__off`, `clip__n`; row `i` of both files is the same frame).

| key | dtype, shape | meaning |
|---|---|---|
| `g_tokens` | str [8] | `STRATEGIC_GOAL_TOKENS_V7` (`vocab_v7.py:62-71`): FOLLOW_ROUTE, TURN_LEFT_, TURN_RIGHT_, STOP_AT_, EXIT_LEFT_, EXIT_RIGHT_, LANE_CHANGE_L_, LANE_CHANGE_R_ (+ `_FOLLOW_ROUTE`) |
| `a_tokens` | str [7] | the 7 strategic action tokens |
| `g_str_id`, `a_str_id` | int8 [N] | index into the token lists; -1 = nothing claimed (NOW before log, or fewer than 3 s observed) |
| `by_time_s`, `within_m` | float32 [N] | seconds / arc metres from NOW to the first overnext event (NaN for FOLLOW_ROUTE) |
| `hold_for_s`, `v_target_ms` | float32 [N] | STOP hold duration; RESUME_CRUISE target speed (NaN otherwise) |
| `str_obs_s` | float32 [N] | observed horizon from NOW, capped at 30 s (log end and the first gap > 1.0 s end it) |
| `g_valid`, `a_valid` | bool [N] | positive tokens are valid; an absence claim (FOLLOW_ROUTE, HOLD_MAIN_ROAD) needs `str_obs_s >= 29.95` |
| `band_s` | float32 [2] | (6, 30) |
| `clip_ids`, `clip__n`, `clip__off` | | as above |

**Construction.** The v7 emitter's own `strategic()` on a 10 Hz track that starts at NOW (so NOW need not sit on the log's 10 Hz grid), 30 s long. Two runtime overrides of the emitter's module constants, nothing else: `STRATEGIC_S` (8, 30) -> **(6, 30)** and `GAP_S` (6, 8) -> **(6, 6)**. `TACTICAL_S` stays (2, 6). The 6-8 s span therefore belongs to the strategic head. Test `test_turn_in_the_6_to_8s_span_is_now_a_strategic_token`: a 90-degree turn wholly inside NOW + 6.2 to 7.8 s gives `TURN_LEFT_FOLLOW_ROUTE` under (6, 30) and `FOLLOW_ROUTE` under the v7.2 bands.

## Tests

`run_tests.cmd` runs `test_band_labels.py` (14) and `test_band_labels_v10.py`. Mutations that go red: NOW read 0.5 s late (closed form on a ramp-speed log); cache clock +0.2 s (speed error 10x or more); `band_s`/`h_abs_min_s` back to v9 (`test_v10_schema_pin_and_literals`, `test_band_end_is_enforced_in_v10_eval139`).

## 4. The train base (Master Mind, 2026-11-08)
Training reads the 4,572-clip v9 file (md5 `83e30bd3791f0767707eade5f4671759`, manifest copy `v9_labels_train_4572.manifest.json`, builder md5 `e6b2a7d4`), not the 4,369-clip banked copy under `v9labels/` (md5 `f63ece41...`). `v10_labels_train.npz` and `strategic_{geom,tok}_train.npz` are built on its inputs: manifest `add203/in/train4572_v2manifest.pt` (md5 `77dafb50`), sidecar `refcv8_clip_clock_sidecar_ts33.jsonl` (md5 `c81c2739`), lead `lead_train4572.npz` (md5 `d4336a44`). `build_v10_labels.py` carries the builder's one-point-grid guard (3 lines; reached only where a parked recording would raise). Control: the tip builder on these inputs reproduces md5 `83e30bd3...` exactly (`run_control4572.cmd`, MEASURED 2026-11-08). v10 train: 4,572 clips, 909,651 rows, md5 `733a4ad0d4431bfd430cfdab79ff9acf`.

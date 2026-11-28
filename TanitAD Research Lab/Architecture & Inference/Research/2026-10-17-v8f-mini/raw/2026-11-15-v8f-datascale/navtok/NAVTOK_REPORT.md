# NAVTOK: NVIDIA nav text as per-frame tokens, and how it compares with the nav the v8f-mini trainer feeds today (Data-Eng agent, 2026-11-16)

Layer: published repo FILE `navigation/navigation.parquet` (see `../navprobe/NAVPROBE.md`). Tip read: `ccec0f28c8f89b60bbe305c1d2fd87cab519df84` (remote `agent/arch-inf-20260803`, via `git ls-remote`). All numbers below are MEASURED from the artifacts named; nothing is re-quoted from prose.

## Headline

1. **The nav text is not an ego-future signal at its source.** The parquet's own schema metadata says: *"navigation instructions rendered from each drive's GPS trace matched to OpenStreetMap-based routing; street names and road numbers removed"*, and *"nav_text holds from timestamp (microseconds after the clip start; a clip's first row is at 0) until the clip's next row"* (MEASURED, `raw/nav_sub.parquet` schema metadata on the pod, read 2026-11-16). NAVPROBE did not record this. It is still computed along the driven route (GPS trace of the same drive), so it is optimistic by construction, but it is a map-routing product, not our geometry.
2. **The two navs agree on direction when both announce a turn, and disagree about whether a turn is announced.** Of 96,346 train windows where the trainer's token is a turn, the text announces the same-side turn (any distance) in 54,078 (56.1 %), a FOLLOW-type text in 36,682 (38.1 %), the opposite side in 4,537 (4.7 %), a U-turn in 1,049 (1.1 %). Section 4.
3. **Parser: 0 of 9,864 distinct texts unparsed.** 53 templates, 44 frozen classes (43 plus `UNPARSED`), 33 tests green including a deliberate-regression arm.
4. **Wiring decision for the MM:** a nav-text arm needs no new architecture if it is fed through the existing 5-channel `nav` vector (drop-in, arm N1 in the prereg draft). A richer vocabulary (44 classes) changes `e_nav = nn.Linear(5, d)` and so needs a G-DVB entry and a model change. Nothing here is wired into the trainer; it is staged only.

## 1. What the trainer feeds today (MEASURED, code at the tip)

Trace, every hop at file:line (`stack/...` at `ccec0f28`):

| hop | where |
|---|---|
| source labels | `--v9-train` (default `.../refcv8_hf_v9/release/v9_labels_train.npz`) `scripts/v8f_mini_train.py:229`; optional prebuilt `--join-train` `:233` |
| join build | `v8f_mini_train.py:2375-2380` calls `D.build_label_join(man, rel, ...)` |
| per-step tensor | `tanitad/data/v8f_mini_data.py:199-209` (`build_label_join`) |
| batch | `v8f_mini_data.py:521`, `:589` (`"nav": J["nav"][rows of the history frames]`), collated `:646` |
| model | `tanitad/models/v8f_mini.py:698` `self.e_nav = nn.Linear(5, d)`; read at `:802`, `:813`, `:836` (`self.e_nav(self._at(b["nav"], i))`) |
| row producer | `scripts/build_v9_labels.py:262-283` (`nav_fields`), literals `:44-45` (`nav_h_s = 6.0`, `nav_d_min_m = 30.0`) |

**The tensor** is 5 channels per cache step: `[FOLLOW, TURN_L, TURN_R]` one-hot, `d_next/100` clipped to [0, 3], `args_valid` (`v8f_mini_data.py:153-154,203-209`). Steps with no v9 row are all zeros.

**How it is computed** (`build_v9_labels.py:262-283`, and the manifest policy string `"nav": "allow_oracle_nav (ego-future)"`, `v9_labels_train.manifest.json`):

* The turns are the *sustained-yaw segments of the whole recording* (`turn_table`, `:233-257`; `turn_band_deg` 30, `is_turn`), minus clip-level suppression. They come from the ego's own future path, so the signal is an oracle (ego-future).
* `d_next = max(0, turn.s_start - s_now)` in arc metres to the turn's start.
* **Token** is TURN_L/TURN_R if a turn is in progress, or if `d_next <= d_ann = max(30 m, 6.0 s x v_now)`; else FOLLOW. So the token horizon is 6 s at current speed, floor 30 m.
* **Distance and validity** are set whenever any announced turn lies ahead (`args_valid = 1`), also beyond `d_ann`; the model sees `d_next/100` capped at 300 m.
* Classes: 3 (FOLLOW, TURN_L, TURN_R). No roundabout, ramp, merge or U-turn class.

**Where the per-window values live.** The join is one array per cache step (`J["nav"]`, `J["nav_d"]`, `J["now_s"]`; step j is v9 row k = 2j). Copy used here: pod `/root/runs/bprime/join_train.npz` (461,518 steps, 4,572 clips, 456,946 with a v9 row; 34,696,488 bytes) and, for eval-141, repo `.../Research/2026-10-17-v8f-mini/raw/join_eval141.npz` (14,237 steps, 141 clips, 13,896 with a row; 1,060,056 bytes). Box 2 holds its own copy for the trainer (path UNVERIFIED, not read). Time: `now_s = grid_start_s + k*dt_s` on the label clock (`clock_rule` in `v9_labels_train.manifest.json`; `clip_clock.py` docstring: label zero is the egomotion log's first sample).

## 2. Parser (`navtok.py`, `test_navtok.py`, `navtok_report.json`)

* Output `(class, distance_m | None, raw)`; `parse_full` also gives verb, degree, side, exit number. Closed grammar, `fullmatch` only; a text no rule matches is `UNPARSED`, never a default.
* **MEASURED, `navtok_report.json`:** 9,864 distinct texts, **0 unparsed (fraction 0.0)**, 35 classes used of the 43 real ones (44 with `UNPARSED`); the 8 unused (`CONTINUE_SHARP`, `CONTINUE_SHARP_STRAIGHT`, `CONTINUE_SLIGHT`, `CONTINUE_SLIGHT_STRAIGHT`, `KEEP_SLIGHT_STRAIGHT`, `MERGE_R`, `RAMP_SHARP_STRAIGHT`, `RAMP_SLIGHT_STRAIGHT`) are vocabulary the grammar admits but the corpus never emits; they stay in the frozen list so a future text does not shift ids. 26 distinct texts carry no distance (bare forms such as "Turn left"). Unparsed texts: none to list.
* Distinct texts per template: 53 once digits are masked; 251 distance values per `... in #m` template (0..250 m), 1,929 for "At roundabout, take exit # in #m" (exit number x distance).
* Tests: 33 passed (`python -m pytest test_navtok.py`, dev box, 0.29 s). The deliberate-regression arm monkeypatches the regexes (drops `slight`, drops the distance group) and requires the parser to turn `UNPARSED`, i.e. the corpus test to go red.
* Coarse map to the trainer's token space (`coarse()`): any `TURN_*` -> TURN_L/R by side, `UTURN` -> its own column, everything else (continue, keep, merge, ramp, roundabout) -> FOLLOW. The full 44-class table is in `agreement.json` so any other mapping can be derived.

## 3. Per-step table (`navtok_table.npz`, `build_table.py`, `navtok_table_report.json`)

* Rule: latest nav row with `ts <= round(now_s*1e6) + T0`, `T0 = -195,000 us` (egomotion t0 median, `../navprobe/navalign.json` `ego_t0_quantiles_us`: 5 % -199,566, 95 % -190,397; ESTIMATED per clip, +-5 ms).
* Arrays are aligned to the joins (`train_off`, `train_n`, `eval_off`, `eval_n` copied; `*_cls_id` index into `vocab`, `*_dist_m`, `*_text_i` into `raw/texts.json`, `*_age_s`, `*_flag`).
* **Flags (MEASURED):** train 461,518 steps: 456,111 ok, 36 before the clip's first row, 799 in the 8 clips with no nav rows, 4,572 with no v9 row (one per clip). Eval 14,237 steps: 13,794 ok, 2 before first row, 100 in the one clip without rows, 341 no v9 row.
* **sha256** `2836369e3e14a8c3c49a6b1a42116c36cda522ca42f98a6bcbc8c4e4d45c8992` (2,300,115 bytes).
* **Time-base check (MEASURED, `agreement.json` `t0_scan_*`):** scanning T0 from -2.0 s to +1.0 s, same-side turn agreement peaks at T0 between -0.3 s and -0.2 s (41,675 / 41,661 windows) and falls to 41,298 at T0 = 0; the curve is flat (about 1 % across 0.5 s), so the table is insensitive to the +-5 ms residual and the base is consistent with the -195,000 us estimate. The plain agreement rate is dominated by FOLLOW and is not used for this.

## 4. Agreement with the trainer's nav (`agreement.json`, `disagreements.csv`)

Windows = steps with a v9 row and a nav-text row (train 456,111; eval 13,794). Rows trainer token, columns text coarse class.

**Train (n = 456,111)**

| trainer \ text | FOLLOW | TURN_L | TURN_R | UTURN |
|---|---|---|---|---|
| FOLLOW (359,765) | 293,026 | 31,869 | 33,889 | 981 |
| TURN_L (48,611) | 18,741 | **26,571** | 2,500 | 799 |
| TURN_R (47,735) | 17,941 | 2,037 | **27,507** | 250 |

**Eval-141 (n = 13,794)**

| trainer \ text | FOLLOW | TURN_L | TURN_R | UTURN |
|---|---|---|---|---|
| FOLLOW (11,460) | 9,480 | 771 | 1,209 | 0 |
| TURN_L (575) | 237 | **251** | 87 | 0 |
| TURN_R (1,759) | 806 | 4 | **949** | 0 |

Text turn at any distance is far wider than the trainer's 6 s horizon (text announces at median 127 m, NAVPROBE), so the FOLLOW row's text-turn cells are mostly distant turns. With the text restricted to a turn within X m:

| X (m) | train agree (3-class) | trainer-turn windows with the same-side text turn | trainer FOLLOW with a text turn within X |
|---|---|---|---|
| 30 | 0.8504 | 0.326 | 3,277 of 359,765 |
| 60 | 0.8575 | 0.432 | 10,284 |
| 150 | 0.8038 | 0.476 | 39,006 |

Facts the table does not show:

* **Opposite side:** 4,537 train windows (trainer L / text R 2,500; trainer R / text L 2,037) = 4.7 % of trainer-turn windows. Eval: 91 (87 + 4).
* **Why text says FOLLOW while the trainer says turn (36,682 windows; top text classes):** `CONTINUE_STRAIGHT` 14,046 (7,222 L + 6,824 R), `RB_TAKE_EXIT` 10,321, `CONTINUE` 3,896, `RB_EXIT` 2,934, `CONTINUE_R` 1,146, `CONTINUE_L` 745 and so on. The two largest reasons are bends and roundabouts that our geometry labels a turn (a sustained yaw of 30 degrees or more) and the router labels continue or a roundabout exit; a roundabout exit is an announcement, but `coarse()` maps it to FOLLOW, so this row overstates the "text says nothing" share. The full-class confusion is in `agreement.json` `full_class_confusion_n`.
* **Distance where trainer and text both announce a same-side turn:** text distance at those windows, median **22 m** (L, n 26,571; p5 3, p25 10, p75 39, p95 109) and **19 m** (R, n 27,507; p5 2, p25 9, p75 36, p95 88). Text minus trainer `d_next`, same side, both valid: median **+7.7 m** (L, n 23,395) and **+7.2 m** (R, n 23,517); **92.0 % and 93.9 %** within +-30 m. The text counts to the manoeuvre node (junction), our `d_next` counts to the start of sustained yaw, which is consistent with a positive offset of one to a few seconds at urban speed.
* **Trainer FOLLOW but text announces a turn within 30 m:** 3,277 windows (0.9 % of trainer FOLLOW); a sample of each disagreement category (60 per category per split, seed 0) is in `disagreements.csv` with the clip, step, text and row age.

## 5. Caveats (stated, not hidden)

* T0 is a constant, not each clip's own egomotion t0 (not available on this machine); the scan above bounds the effect. UNVERIFIED per clip.
* The agreement uses the trainer's join as built for the v8f-mini run on the pod copy; whether box 2's current join file is byte-equal to the pod copy was not checked.
* Trainer-turn means "sustained yaw >= 30 degrees ahead", text-turn means "the router says turn": they are different definitions, so disagreement is not an error rate for either.
* The text is computed from the drive's GPS trace, so for any arm it is optimistic as a route input (the drive's own path), and it is a drop-in only for the token/distance channels.

## Manifest

See `MANIFEST.md`.

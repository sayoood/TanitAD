# TABTABLES_DEFS: the pod-side table tool of card ADD84 (TABCLOSE10K)

Tier T1-kin (kinematic rollout on the model's own actions) applies to the runs that read these tables; this tool is CPU only and makes tables and controls, no rollout.

## Files (folder `tabclose/`, copied to the pod beside `make_surrogate.py`)

* `make_tab_tables.py` the tool. `test_tab_tables.py` pytest (synthetic). `run_mutations_tt.py` + `MUTATIONS_TT.json` the mutation runner and its result.
* The tool imports `make_surrogate.py` of card ADD83 (maps, `surrogate_values`, `check_u1`, `u4_rows`, `check_u4`); it looks in `--surrogate-tool-dir`, `$SURROGATE_TOOL_DIR`, its own folder, then `../shufclose`. `closeread_fit.py` is found by `--fit-dir`.

## Pod call

```
/workspace/venv/bin/python make_tab_tables.py --dump-dir /workspace/v8proof/closeread/dump/both \
  --tk-dir /workspace/v8proof/tokclose/fits --fit-dir /workspace/v8proof/closeread \
  --rrc-run-dir <folder of the banked RRc run folders> --rrc-run-name I10RQ50cKF5P30Hb20 \
  --seeds 0,1 --out-dir /workspace/v8proof/closeread/est_tab
```
The out folder must not exist. `--selftest` runs the synthetic world (inside a new `tmp_selftt_*` folder beside the tool).

## What each table is

| file | content |
|---|---|
| `tab_L100_s<k>` | the closing input that form RRc feeds: label closing speed on the frames where the label has a slope (lead read present, trailing run of at least 3 frames), no value elsewhere |
| `tab_W50/W70/W90_s<k>` | on the frames of L100: m + r^2 (L - m) + s sqrt(r^2 (1 - r^2)) e, r = 0.50, 0.70, 0.90 |
| `tab_S70_s<k>` | as W70 with e smooth in time (autoregression 0.9 along the frames of a clip, restarted at every clip; the draw is made for every row of the table) |
| `tab_W00_s<k>` | m + 0.5 s e on the same frames |
| `tab_TK_s<k>` | the T25 table of card ADD81, the three arrays copied, values exactly equal |
| `tab_STK_s<k>` | its surrogate by the map of card ADD83 (the map of reader seed k from the presence share of the dump of seed k); the map itself is in `map_STK_s<k>.npz` (read by control a4_u1; not part of the reader contract) |

Rows of every table = the 14,237 rows of the dump of seed k, in that row order; arrays `clip_index` int64, `frame` int64, `est` float64 (NaN = no value). m and s are the mean and standard deviation (ddof 0) of L over the frames of L100. Random numbers: `default_rng(84000 + 10 a + k)`, a = 0..4 for W00, W50, W70, W90, S70, one standard normal per row.

## Build detail to put into a note to the card before the tool runs on data

The card says e is an independent standard normal per frame. The tool makes the drawn noise EXACT on the frames with a value: after the draw (for S70 after the autoregression) the noise is made mean 0, and for W50, W70, W90 and S70 also exactly uncorrelated with L (its least-squares projection on [1, L] is subtracted), then scaled to unit standard deviation (ddof 0). Control a1 then holds by construction (r, sd and mean read their targets to 1e-9) and does not depend on the luck of one draw. For W00 only mean 0 and unit sd are planted; its r against L is whatever the draw gives and a1 bounds it to [-0.03, +0.03] (with n valued frames the draw reads about 1/sqrt(n) in sd of r, so a small n makes a W00 failure of a1 possible; the card gives no remedy and none is built in).

## Definition of the valued frames, from the wrapper code

`v8f_t1_eval_reader_rule_rzgfhqet.py`: form RRc is mix (presence head, gap head, closing label) (line 1537); the closing input at a look is `c_src = where(present & ~lab_c & head_c, hc, lc)` (lines 863-864 at later looks; `v8f_t1_eval_rule_mix.py` lines 81-83 at the replan tick) with `lc` = third value of `LabelSource.read(ci, j)` (`v8f_t1_eval_rule.py` lines 92-112): the least-squares slope of the trailing run of at least 3 lead-valid frames (steps at most 3.0 m, run at most 5), else exactly 0.0, also exactly 0.0 where there is no lead at the frame or the frame lies outside the clip. For RRc `hc` is the zero array, so the substitution branch never fires on a finite `lc`, and `lc` is always finite. The logged `c_src` is therefore the label value at (clip, frame) at EVERY look, whatever the reader reads (present or not), 0.0 where the label has none. The dump columns reproduce it (`closeread_dump.py` lines 84-108: `lab_closing` is the slope where `lab_run >= 3`, 0.0 elsewhere). L100 has a value exactly where `lab_present` and `lab_run >= 3`; a valued slope that is exactly 0.0 stays a value (a constant gap).

## Controls

* a0: L100 against both rule logs (replan, mid-look) of the banked RRc runs of both seeds, key (`ci`, `j`). A look agrees when the log is finite and not 0 and the table is within 1e-6, or the log is exactly 0 and the table has no value or exactly 0. Judged: per (seed, log) the share of agreeing looks over ALL looks at least 0.999, and some looks with a nonzero log value. Reported beside: the same share over looks with `present` true, the number of looks with a nonzero log value where the table differs, and with log 0 where the table has a nonzero value.
* a1: per arm and seed from the files read back (r, sd, mean, lag-1 autocorrelation of the noise recovered from the table inside a clip, valued frames equal to L100).
* a4_u1, a4_u4: u1 and u4 of card ADD83 on the TK table with H = 25 (u3 of the card is a read-side control on rollouts and is not part of this tool).
* Input checks that REFUSE (not controls): label columns equal between the seeds' dumps; TK rows equal to the dump rows; RUN_MIN of closeread_fit equals 3.

## Record

`TAB_TABLES_RECORD.json`: `pass`, `failed`, `tables` (file name to sha256), `counts` (rows, valued per table), `rng`, `m_s` (m, s, valued per seed), `inputs` (path and sha256 of the dumps, TK tables and rule logs), controls `a0`, `a1`, `a4_u1`, `a4_u4` each with `PASS`, and the maps. The tables are written even when a control fails.

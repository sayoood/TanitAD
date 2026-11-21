# ROLLTRACE_DEFS -- kit for one figure of per-window traces (distance to the lead over rollout time, three forms of the lead rule)

Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s); the lead moves as logged; no re-perception; not driving performance.

## 1. What could fail by construction, and what was not checked

* Not run anywhere on real data: no pod access, no real dump opened. The format of the dump files and of the join is taken from the code (`read_longroll10k.load_dump_dir` reads `ep*.npz` with the arrays g, cl, ws, eid, clip_index, v0). The selftest runs the engine's own functions on synthetic windows (2 clips x 20 windows), not the real ones.
* The per-step series EXISTS in the engine: `headway_read_r2.window_cells` returns `_gm` [N,30] (the gap of the rolled-out ego to the lead, one value per rollout step). The three flags are computed from that same array in the same call. Nothing is re-implemented.
* The engine copy that the selftest uses on the dev box is `F:/TanitAD-scratch/v8proof/headway/headway_read_r2.py`; on the pod the export imports the engine through `R.engine(--code)` (the folder `podread/code`). Whether those two files are byte-identical was not compared. The control t0 (rates equal the banked rates) is what catches a different engine.
* Run folders: the script reads `<pull>/I10<run>_s<k>/dump_I10<run>_s<k>_m10_causal_heads` with `<pull>` = `/workspace/v8proof/integ10k/t1` (the folder the brief names). The reader itself read the same folders through the links of `podread/root/integ10k/pull`. If a dump is not under `t1`, set `ROLLTRACE_PULL=/workspace/v8proof/podread/root/integ10k/pull`. Not checked: that the banked RR and RLc dumps sit under `t1`.
* Window count 1,271 per model seed and 650 windows with a lead pooled are t0 clauses taken from the brief (the engine's constant `CARD_COUNTS` is 1271 for the 30-step grid). If the real counts differ, t0 fails and writes no trace; that is the intended behaviour.
* t0 compares two rates per form with the 1e-4 tolerance: the pooled mean of C over the lead windows of both seeds, and the mean of the two per-seed levels (what the engine's `levels` hold). They are equal while both seeds have the same number of lead windows.
* The window key text is `<clip id>/<window start>` with the full clip id (the join's `clip` column); the hash is taken over `<seed>|<key>`. The brief did not fix the key text; this is the choice.
* U is NaN (written as null) for a window where the logged lead is already unsafe: the engine does not score U there. C and F are scored for every window with a lead.
* The gap series is NaN (null) after the lead track ends (engine `build_track`); the number of scored steps is written per form.
* If fewer than 6 windows satisfy the rule, the script exits 3 with a FAIL line and writes nothing.
* The figure was drawn only from synthetic JSON. How the labels read on the 6 real windows (collisions, y range) is not checked; labels are placed only where they are at least 7.5 percent of the y range apart.
* Palette: aqua (slot 3) has 2.74:1 against the surface, below 3:1; the validator asks for relief by direct labels or a table. The figure carries direct labels where they fit, a legend, and ring markers at the contact step; where a label collides it is dropped and the legend names the line.

## 2. Where the series comes from (file:line)

| what | where |
|---|---|
| reader that produced the banked rates | `visrule/read_visrule10k.py:302` `measure`, line 322 `R.read_grid(...)`, line 323 mean of the per-seed levels |
| `R` = `read_longroll10k` (the engine's grid reader) | `integ10k/read_longroll10k.py:481` `read_grid`; dumps via `:490` `load_dump_dir(dump_dir(...))`, restriction `:184` `restrict_idx`, keys `:173` `grid_keys`, level per seed `:590` |
| the per-window cells | `headway/headway_read_r2.py:179` `window_cells(g, cl, L)` called at `read_longroll10k.py:513` |
| lead track, label frames | `headway_read_r2.py:148` `build_track`, `:167` `label_windows` (step bound 3.0 m, at least 10 lead steps) |
| gap series | `headway_read_r2.py:188-189`: `a_m = progress(g, cl)`, `gm = gl + a_l - a_m` (logged lead gap + logged arc - arc of the rolled ego), steps k = 1..30 of DT = 0.2 s (`:33`), stored as `out["_gm"]` (`:214`) |
| ego speed series | `headway_read_r2.py:136` `step_speeds(traj)` on `cl` (step length / 0.2 s); the engine floors it at 0.5 m/s for the time gap only |

Definitions as the engine codes them (per window with a lead, M = steps up to the end of the lead track):

* C contact: any step of M with `gm <= 0.0` (`headway_read_r2.py:196`); the rate is the mean of C over windows with a lead.
* U unsafe: any step of M with `gm < 2.0` m or time gap `gm / max(speed, 0.5) < 1.0` s (`:194`, constants `:38`); scored only where the logged lead is not itself unsafe.
* F falls behind: `gm - gl > 10.0` m at the last scored step (`:205`, constant `:39`).

## 3. What the scripts do

* `export_traces.py`: loads 3 forms x 2 seeds, runs control t0 (same windows in all six dumps; 1,271 windows; 650 windows with a lead pooled; pooled C rate RR 0.0954, RLc 0.0000, VD5c 0.0169 within 1e-4), selects, writes `ROLLTRACE.json` (6 windows x 3 forms, clip ids cut to 8 characters, no path, no pixels) and the line `ZZ EXPORT-TRACES t0 PASS windows 6`. Selection: windows (seed, key) where RR has contact and VD5c has none, sorted by sha256 hex of `<seed>|<key>`, first 6; the file also holds the 2 x 2 contact table per seed and pooled and the number of windows the rule chose from.
* `podrun_export_traces.sh`: the pod call (CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=4, `/workspace/venv/bin/python`), output `/workspace/v8proof/rolltrace/out`, refuses if it exists, `DRY=1` prints the command and makes nothing; it runs the selftest on the pod first.
* `make_fig_rolltrace.py`: reads `rolltrace/pod/ROLLTRACE.json`, writes `fig_rolltrace.png` (dpi 170) and `.pdf` into `paper/v11/figs` (or `--out-dir`).

## 4. The lead's commands, in order

1. `cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest rolltrace/test_rolltrace.py -q -p no:cacheprovider --basetemp /f/TanitAD-scratch/v8proof/rolltrace/tmp_pt/<new name>`
2. `bash rolltrace/push_rolltrace.sh` (expects `ZZ PUSH-ROLLTRACE bad=0`)
3. On the pod: `DRY=1 bash /workspace/v8proof/rolltrace/kit/podrun_export_traces.sh`, then `bash /workspace/v8proof/rolltrace/kit/podrun_export_traces.sh` (last line `ZZ PODRUN-ROLLTRACE DONE rc 0`, and in the log `ZZ EXPORT-TRACES t0 PASS windows 6`).
4. `bash rolltrace/pull_rolltrace.sh` (into `rolltrace/pod`; when t0 failed the pull holds the two logs and no ROLLTRACE.json).
5. `cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh rolltrace/make_fig_rolltrace.py`

## 5. Temp left in place (nothing deleted or moved)

* `rolltrace/tmp_probe.py` (an import probe), `rolltrace/tmp_pt/` (synthetic folders of the selftest and the tests; every selftest run adds two small folders), on the pod `kit/tmp_pt` after the pod selftest, and later `rolltrace/tmp_pull_rolltrace.txt` and `.tar`.

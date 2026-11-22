# FALLTRACE_DEFS -- a count of the windows in which a vision-fed arm falls behind and the label-fed form does not, and what the rule log says about them

Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s); the lead moves as logged; no re-perception; a diagnostic, not driving performance. No card, no name: the output is counts. The definitions below are fixed before the lead runs the kit. Nothing was run on real data: no pod access, no result file or dump opened; the selftest and the tests use tiny synthetic data.

## 0. FIRST: points of the brief that the code or the rule log cannot do as written, and the closest reading coded

1. **"The rule acted: its command is below the planner's own by more than a small tolerance."** The rule log holds no column with the planner's own speed or the rule's commanded speed (`LOG_FIELDS` of `distrule/v8f_t1_eval_rule.py:43-44`: ci, t0, j, k, tick, active, present, src_bad, sat, sat_exec, lift, changed, g_src, c_src, p_src, v0, v_l, v_L, a_l, a_m, g0; the later-look log adds q, excess, nchg, c_read). The plans themselves are not logged, except the plan after the rule at the first tick (`p1_first`). Coded instead: **rule acted at a look = `present` and `changed`**, where `changed` is the rule's own flag "the plan differs from the input" (`dist_rule.py:119`, set at `:183` by an exact comparison of the knots: an unchanged plan is returned bit for bit). So the tolerance is zero and it is the rule's own. A change in any knot counts, including a change that only lifts a plan that wants to slow faster than the braking floor; the log cannot tell these from a plain cap.
2. **"FALSE-PRESENCE" can be empty by construction for a label-presence arm.** For the arms of this count (the reader form RR with a camera-head gap file) presence comes from the reader (`present = reader probability >= 0.5 and a full history`, wrapper line 872 with `mix["presence"]` not `label`), so a step with a fed lead and no label lead can occur. It cannot occur for RLc (label presence): the count never classifies RLc, only compares its F flag. If an arm is ever run with label presence, class 2 reads zero for it; the export prints the number of acted rows without a label lead per arm and seed (`rule_log_coverage.acted_rows_without_label_lead`) so an empty class can be told from an unreachable one.
3. **"Fed gap minus label gap, recorded state or rolled state."** Taken at the **recorded state**: the label gap is the join row of the logged frame `j` of the look (`off[ci] + j`), the same frame the rule's source gap `g_src` is read at (head or label at frame j). The rule's own rolled-state gap is `g0 = g_src + a_l - a_m` (logged); the label's rolled-state gap would be `label + a_l - a_m`, and the difference of the two is again `g_src - label`, because `a_l - a_m` is the same on both sides. So the two readings are the same number.
4. **"The slope of the label gap over the same step."** Coded as the forward difference of the label gap from the logged frame `j` to `j + 1`, in m/s (step 0.2 s), defined only where both frames have a label lead (inside the clip, `lead_valid == 1`, finite, 0 to 60 m) and the step is at most 3 m (the engine's track bound). Where it is undefined it is NaN: such a step cannot be a SLOPE step. The closing input of a vision arm is a trailing least-squares slope over at most five frames (wrapper `vis_run_closing`), so the two numbers are not the same estimator; the class compares directions only.
5. **Sign of the closing input** (read from the wrapper, not assumed): `v_L = max(0, v_l + c_src)` (wrapper line 887) with `v_l` the logged speed, so `c_src` is the lead's speed minus the ego's logged speed = the gap RATE; **positive = the gap opens and the rule brakes less, negative = the gap closes**. "The closing input says the gap closes" is coded `c_src < -0.1 m/s`.
6. **SLOPE denominator.** The brief says "at 50 % or more of the rule-acted steps" (all acted steps), as for FALSE-PRESENCE; GAP-LOW is stated over acted steps WITH a label lead. Both coded as written. A SLOPE step needs a defined label slope, so the share is over all acted steps, of which the undefined ones count against it.
7. **Windows and rows.** Rule-log rows are matched to windows on (clip index, `t0`); `t0` is the window start the grid and the dump call `ws` (`grid_keys` returns `(clip index, t0)`). The rule log has one row per window and look, the tick log and the later-look log (`rule_log_mid_`) are concatenated; the pre-roll rows (`rule_log_pre_`) are not read. A lead window with no row gets zero steps and is classed NO-RULE-ACTION; the export counts such windows per arm and seed (`windows_without_rule_rows`) and the rows that match no window of the grid (`rule_rows_outside_windows`).
8. **Banked F.** The brief names contact values (RR 0.0954, RLc 0.0000, VD5c 0.0169, F2 0.0138, F3 0.0185, F4 0.0123, E5c 0.0138) and "the banked result files where they exist" for F. Those files are not on the dev box and were not opened. `--banked` takes a comma list of json files; in each the first dict named `levels` whose entries carry `F` is read (the layout `measure()` of `cropscale/read/read_cropscale10k.py:673` builds: `levels[arm] = {C, U, F}`, mean over the two model seeds). Where an arm has an entry, C and F are compared within 1e-4 with the mean of the two seed levels; where the file or the arm is absent, only the contact values above are checked and the record lists what was checked (`control.banked_checked`). The pod script passes `/workspace/v8proof/cropscale/read/CROPSCALE10K_RESULT.json` when it exists; whether that file nests `levels` where the search finds it was not checked. For G2, G3, G4, GEc, E3c there is no brief value for contact: they are checked against the banked file only, if it has them.
9. **Out of the 650.** "650 windows with a lead" and 1,271 windows per seed are control clauses taken from the brief (as in the rolltrace kit); if the real counts differ the control fails and nothing is written.
10. **Pooled over arms.** The pooled class counts add the arm-windows of all asked arms; one window can appear once per arm. They are counts of arm-windows, not of distinct windows.

## 1. Definitions as coded (`export_fall.py`)

Flags C, U, F per window and model seed: `headway_read_r2.window_cells` (the engine's own), as in `rolltrace/export_traces.py`: C contact = any scored step with gap <= 0; U unsafe; F falling behind = gap minus logged gap > 10 m at the last scored step. Windows with a lead: the engine's `lead`.

Runs (`RUNS`): RR `RQ50KF5P30Hb20`, RLc `RLcKF5P30Hb20`, VD5c `...vVD5c`, F2 F3 F4 `...fF<k>`, E5c `...eE5c`, G2 G3 G4 `...gG<k>`, GEc `...gGEc`, E3c `...gE3c`; folder `I10<run>_s<seed>`.

**Control** (first; on failure prints one line `ZZ FALLTRACE CONTROL FAIL <clause>`, writes nothing, exit 3): clauses `same-windows` (g, ws, clip index and lead identical in every dump), `n-windows` (1,271), `n-lead` (650 pooled), `rate-C-<form>` (pooled C and mean of the two seed levels each round to the brief's value at four decimals), `banked-C-<form>` / `banked-F-<form>` (section 0, point 8). The pooled and the seed-mean rates of C and F of RR, RLc and every asked arm are in the record.

**2 x 2** per arm X, per seed and pooled: over lead windows, F(X) against F(RLc) and against F(RR): `both`, `x_only`, `ref_only`, `neither`. An EXCESS window is `x_only` against RLc.

**Per window and arm** (from the arm's rule log; `numbers_of`):

| field | meaning |
|---|---|
| `n_steps` | rows of the window in the two logs |
| `n_fed` | rows with `present` (fed lead; `present` is already false where the source value is not finite) |
| `n_acted` | rows with `present` and `changed` |
| `n_acted_no_label_lead` | acted rows whose frame `j` has no label lead |
| `n_acted_label_lead` | acted rows with a label lead at `j` |
| `n_gap_low` | of those, rows with `g_src - label gap < -5 m` |
| `n_closing_fed` | acted rows with `c_src < -0.1 m/s` |
| `n_slope_mismatch` | acted rows with `c_src < -0.1` and a defined label slope >= -0.1 m/s |
| `gap_diff_median_m`, `closing_input_median_mps`, `label_slope_median_mps` | medians over the acted rows (with a label lead, all acted, defined slope) |

**Class** (`classify`, first match, constants at the top of the module: `NO_ACT_MIN = 5`, `SHARE_MIN = 0.5`, `GAP_LOW_M = 5.0`, `CLOSE_EPS_MPS = 0.1`):
1. `NO-RULE-ACTION`: `n_acted < 5`.
2. `FALSE-PRESENCE`: `n_acted_no_label_lead >= 0.5 * n_acted`.
3. `GAP-LOW`: `n_acted_label_lead > 0` and `n_gap_low >= 0.5 * n_acted_label_lead`.
4. `SLOPE`: `n_slope_mismatch >= 0.5 * n_acted`.
5. `MIXED`: none of these.

Printed per arm and pooled (`FALLTRACE CLASS <arm> excess ...` and `... neither ...`): counts and shares of each class among the EXCESS windows and among the windows where NEITHER arm X nor RLc falls behind, one line each, so the two shares can be read side by side. A class as frequent among the neither windows explains nothing.

`FALLTRACE.json`: tier, arms, control record, 2 x 2 tables, class counts per arm and pooled, constants, rule-log coverage, and the table of every lead window of every arm and seed (clip id, window start, seed, C, U, F of the arm, F of RLc and RR, group against RLc and RR, class, and the per-window numbers above). No camera data, no path. Last printed line `ZZ FALLTRACE DONE <n arms> <n excess windows>` (excess windows summed over arms and seeds).

## 2. What each class cannot see

* NO-RULE-ACTION says the rule did not touch the plan, so the fall-behind comes from the planner, not from the rule; it does not say why the rule stayed off (no fed lead, a gap above 60 m, or a plan already slower than the cap). `n_fed` is in the table to separate "no fed lead" from "fed and not needed".
* The class is a property of the window's rule log, not a cause proven by a change of the rule; a lever is ranked by the count, not confirmed by it.

## 3. The lead's commands, in order

1. `cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh falltrace/export_fall.py --selftest` (last line `ZZ SELFTEST-FALLTRACE ok: True`).
2. `PYCAP_MB=3000 bash ops/pycap.sh -m pytest falltrace/test_falltrace.py -q -p no:cacheprovider --basetemp falltrace/tmp_pt/<new name>`
3. `bash falltrace/push_falltrace.sh` (expects `ZZ PUSH-FALLTRACE bad=0`).
4. On the pod: `DRY=1 bash /workspace/v8proof/falltrace/kit/podrun_export_fall.sh` (prints what is missing), then `ARMS=VD5c,F2,F3,F4,E5c bash /workspace/v8proof/falltrace/kit/podrun_export_fall.sh` (last line `ZZ PODRUN-FALLTRACE DONE rc 0`); the G arms in a second call into a new root are not supported: `out` is refused if it exists, so ask for all arms in one `ARMS=` list.
5. `bash falltrace/pull_falltrace.sh` (into `falltrace/pod`).

## 4. Temp left in place (nothing deleted or moved)

`falltrace/tmp_pt/` (synthetic folders of the selftest and the tests; each selftest run adds three small folders); on the pod `falltrace/kit/tmp_pt` after the pod selftest and `falltrace/pull` (links to the run folders).

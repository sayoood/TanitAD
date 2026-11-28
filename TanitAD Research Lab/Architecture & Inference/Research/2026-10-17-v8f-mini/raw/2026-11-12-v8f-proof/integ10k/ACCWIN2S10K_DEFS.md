# ACCWIN2S10K_DEFS.md -- tool set of card PREREG_V8F_PROOF_ADD58 (written by the building agent; model output, grants nothing)

All files are in `integ10k/`. Nothing was run on a pod; nothing was rolled out. Checked here only: `ast.parse` of every Python file, `bash -n` of every shell script, a DRY=1 print of the queue, a diff count of the generated files against their parents, a word search.

## 1. Edits of the generator (`make_accwin2s10k.py`, asserted: each old text once, or the counted number of code lines)

Wrapper `v8f_t1_eval_reader_rule_rz.py` -> `v8f_t1_eval_reader_rule_rv.py` (sha256 1d26aad770daa5a8..., full value printed by the check; 83 changed lines, 14 hunks):
1. header text (new readout described; the parent header kept below it);
2. `STATE` gets `"exec": None` and `"w2log": None`;
3. new `W2_LEN = 10`, `commit_plan_with_state(orig_cp)` (sets `STATE["exec"] = self` while the pinned `WindowState.commit_plan` runs, resets it in `finally`) and `window2_steps(k_prev, knots, m)` (reads knots 0 .. j-1 of `ex.track` = rows `0:2*j:2`, keeps the trailing finite run, at most the last 11 knots, requires the last one to equal `k_prev` exactly, appends the block knots `knots[0..m]`, returns step speeds `s_ext`, `n_pre`, and `e_blk = j - t0 + m`);
4. in `causal_frames`: prelude for `acc == "window2"`, branch `v[2(i-1)] = s[i] - 0.5*DT*(s_ext[n_pre+i] - s_ext[n_pre+i-w]) / (DT*w)`, `w = max(1, min(10, i + n_pre))`; after the loop `STATE["w2log"] = (w at the last executed knot, the window value there, e_blk)`;
5. in `counted_frames`: dispatch `causal_v` -> `window2`; log lines for `causal_v` (count and sum of w, minimum w, calls below the cap, largest |tick acceleration - 2.0 s window value|, count and sum of |a| of the ticks whose whole reach lies inside the dump);
6. `save_rlog`: fields `window2_len_cap`, `window2_calls`, `window2_mean_len`, `window2_sum_len`, `window2_min_len`, `window2_calls_below_cap`, `window2_max_abs_acceleration_minus_window2`, `window2_dump_calls`, `window2_dump_sum_abs_acceleration_read` (zeros in the other readouts);
7. argparse choice `causal_v`; `orig_cp = T.WindowState.commit_plan`; install of the wrapped `commit_plan` for `causal_v` only; `T.hermite_frames = orig_hf; T.WindowState.commit_plan = orig_cp` at the three places where the frames function is restored.

Queue `pod_accwin10k_queue.sh` -> `pod_accwin2s10k_queue.sh`: markers `START_/ACCWIN2S10K_DONE/_STOP`; wait on `$W/$WAIT_DONE` / `$W/$WAIT_STOP` (defaults `RULE1S10K_DONE.txt` / `RULE1S10K_STOP.txt`, 8 h limit); the banked runs R0, R0M5, R0W5, R0Z5, R0W10 at both seeds must have `T1_DONE`; build files now include the rv and rz wrappers and `check_tick_readout_rv.py`; tool check step (marker `v9check`) before any rollout; rollouts replaced (below). Every other guard of the parent is unchanged (START noclobber, slot wait before each rollout, OUTPUT-EXISTS, pinned sha, memory probe, dump count 141, `done_step`).

Chain `chain_accwin10k.sh` -> `chain_accwin2s10k.sh`: names, file list (adds rv, rz wrappers and the check script), pulls `I10R0VID`, `I10R0V5`, `I10R0V10`, small files into `small_accwin2s` (adds the check output), poll limit 24 h (the queue itself first waits up to 8 h), the pull of `I10RRx` removed.

## 2. Run map (queue order)

| step | what | name |
|---|---|---|
| 0 | wait for the previous queue's end marker | |
| 1 | tool check on the pod, CPU, output `accwin2s_test/CHECK_TICK_READOUT_RV.txt` and a copy of the wrapper | control v9 |
| 2 | `I10R0VID_s0`: readout causal_w, 5 steps, model seed 0 | control v1 |
| 3 | `cmp_dumps.py` banked `I10R0W5_s0` vs `I10R0VID_s0`, marker `v1cmp`, output `small_accwin2s/CMP_R0VID.txt` | control v1 |
| 4 | `I10R0V5_s0`, `I10R0V5_s1` (causal_v, 5 steps) | R0V5 |
| 5 | `I10R0V10_s0`, `I10R0V10_s1` (causal_v, 10 steps) | R0V10 |

Output folders `t1/<run>_s<k>/` keep the tag ending `m10_causal_heads` (as the earlier cards; the readers find the dumps by it).

## 3. Pair list of the pod read (`podread_accwin2s.sh`)

`I10R0V5:I10R0M5` (SP), `I10R0V5:I10R0` (GC), `I10R0V10:I10R0` (PR), beside: `I10R0V5:I10R0W5`, `I10R0V5:I10R0Z5`, `I10R0V10:I10R0W10`. Control e1 first (card ADD46 pairs, output `e1v`).

## 4. Where each control is computed (`read_accwin2s10k.py`; names only if v1 to v9 all pass)

* v1: `CMP files=141 differing=0` in `small_accwin2s/CMP_R0VID.txt` (written by the queue).
* v2: readout log of each of the 5 new runs (readout, replan steps, calls > 0) and `replan_steps` in the result file (5 or 10).
* v3: log `max_abs_speed_minus_last_step` <= 1e-6 (the speed at the last executed knot equals the last executed step speed); for `causal_v` also `window2_max_abs_acceleration_minus_window2` <= 1e-6 (log, all committed plans), `window2_calls == calls`, mean window length recorded (> 5 steps required for R0V5), and an independent recomputation by the reader from the dump's executed tracks `cl` and `fl` of the ticks whose reach lies in the dump (e - 10 >= 1: e = 15, 20, 25, 30 at 5 steps; 20, 30 at 10 steps): tick count equal to the log's, mean absolute difference between the sum of the log and the reader's sum <= 1e-6. For R0VID: the speed condition and `max_abs_acceleration_minus_window` (the 5-step window of the parent) <= 1e-6.
* v4: first 5 executed steps (R0V5, R0VID) against R0M5 and first 10 steps of R0V10 against R0, model `cl` and prior `fl`, largest difference exactly 0.0.
* v5: 141 files in each of the five new dumps; 2,399 windows and 642 lead windows from the headway read.
* v6: sha256 of the 10 result files in `ACCWIN2S10K_REUSED.json` (see `ACCWIN2S10K_REUSED_FILES.txt`); RS of R0 and R0M5 and ADE6 / RS of their priors against the section 1 values of card ADD54.
* v7: `ALL_PASS` of the headway and the family reader.
* v8: checkpoint sha256 prefix in `ticks/run_<arm>.txt` of each new run against card ADD46 (UE_s0 2fd4b747..., UE_s1 f13df8ce...).
* v9: the check output ends `ZZ CHECK-TICK-READOUT-RV all hold: True`; wrapper at `build/` equals the copy the check ran with, equals the sha256 the check prints, and starts with `WRAPPER_SHA_PREFIX` of the reader.

Names (section 5 of the card): `ACCWIN2S10K-MODEL-STABILISES-1S | -SLOWS-CLIMB | -NO-EFFECT`, `ACCWIN2S10K-PRIOR-REMOVES-CLIMB | -REDUCES-CLIMB | -CLIMB-STAYS`, `ACCWIN2S10K-COSTS-AT-2S | -HELPS-AT-2S | -NEUTRAL-AT-2S` (rule of card ADD54 as in the parent reader: any pair-cell interval above 0 or family verdict WORSE = costs; below 0 or BETTER = helps).

## 5. Not checked here

The check script and the reader were only parsed; the wrapper was not imported (no pinned tool, no numpy job allowed on this box). The pod must run the check before the rollouts (the queue does). The reader never ran against real files (the layout of the readout log of R0V runs is new).

## 6. Guesses (all open to the lead's review)

1. History source: the knots before the window start are taken from `ex.track` rows 0, 2, .. 2(j-1) as the pinned tool holds them (logged native track up to t0); their step speeds are distances over 0.2 s, not the logged speed channel. The history is cut to the last 11 knots (the reach is 10 steps), so `n_pre <= 10`.
2. Number of "earlier steps available" at executed step i = i + n_pre, where `s[0]` (the step into the current knot) counts; `w = max(1, min(10, i + n_pre))`. At a window start with `j >= 11` this is 10 at every step. The hand formula of the check uses the same count from the track (`min(10, k - 1)`, k = knot index), not from `t0`.
3. The "window length at the last executed knot" in the log is w at i = m, over every committed plan (model and prior alone, 12 calls per window at 5 steps).
4. "Largest difference between the acceleration at the tick and the 2.0 s window value" is `|(v[-1]-v[-2])/0.1 - (s[m]-s[m-w])/(0.2 w)|` per call. The reader's independent recomputation covers only ticks whose reach lies in the dump, since the logged past before the window is not in the dump; the comparison is of sums (mean absolute difference per tick <= 1e-6), not of every tick.
5. Control v3 "mean window length above 5 steps in R0V5" is read as the mean of w over all calls; with a logged past this is about 10 and the bar is loose.
6. `STATE["exec"]` is set only while `commit_plan` runs; the frames function is also called by nothing else in the pinned tool (checked by reading `v8f_t1_eval.py` of commit cdbff971: `hermite_frames` has one caller, `commit_plan`).
7. `ACCWIN2S10K_REUSED.json` was filled from the dev-box pulled copies of the result files (the card says recorded before the run); the lead should confirm the pod copies have the same sha256 before the read.
8. The reader's 10-file requirement and the wrapper sha prefix (`1d26aad770daa5a8`) are fixed in the file; regenerate the wrapper and the prefix moves.
9. The pod read needs the reader, `ACCWIN2S10K_REUSED.json` and `podread_accwin2s.sh` pushed by hand (as in earlier cards: the chain pushes only queue files); `podread_accwin2s.sh` stops with a named reason if one is missing.

## 7. Things in the patterns that looked wrong

* `chain_accwin10k.sh` polled for 12 h although its queue may wait 6 h on the previous queue (I raised it to 24 h here).
* The parent chain's last cp line copies many files of earlier cards into the small folder (harmless, errors are discarded).
* The parent reader's `z5` counts `len(nfiles) == 7` by hand; here 5.
* In the parent wrapper `counted_frames` always computes the 5-step window difference, so the log field `max_abs_acceleration_minus_window` is meaningless for `causal_v` runs (it is not used by the reader of this card).

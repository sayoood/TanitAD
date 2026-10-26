# T1-HEADS: per-tick S / T head dumps for `v8f_t1_eval.py` (CPU only, nothing staged, committed or pushed)

Base: `635487e5:stack/scripts/v8f_t1_eval.py` (LF blob, sha256 baf16101...cbcb7d). New file sha256 61e140ed...7397; new test 83aaa1b9...c683.

## Two flags, both default OFF
* `--dump-ticks DIR` (carried from hlat) writes `DIR/ticks_<tag>.npz`, one row per (window, tick) of the MODEL rollout only. Rows are chunk-major, then tick-major.
* `--dump-heads` (new; REFUSED without `--dump-ticks`) adds the head columns below to that same file. Heads are read from the SAME `forward_infer` call that made the plan (no extra model call).
* Header gains `ticks_file`, `ticks_heads` only when `--dump-ticks` is given; flags absent => dumps and result JSON are byte-identical to the tip tool's.

## Schema added by `--dump-heads` (N = windows x 30/m rows; all finite; key | shape | dtype | frame)
| key | shape | dtype | frame / meaning |
|---|---|---|---|
| `s_ckpt` | N,3,3 | f32 | S checkpoints (x,y,psi) at the 30/50/80 m arcs, RAW, in S's tick frame (= executed pose at step j-age_s) |
| `s_ckpt_now` | N,3,3 | f32 | same, in the NOW frame (executed pose at the tick) via `transform_ckpt` = T0's `_ckpt_to_now` frame |
| `s_ckpt_w0` | N,3,3 | f32 | same, in the WINDOW-ORIGIN frame (logged pose at t0): shared by all ticks of a window, use this one to score against t0-frame RC labels |
| `s_pose`, `t_pose` | N,3 | f32 | world pose (x,y,yaw) of S's / T's tick frame, as fed to the model |
| `pose_w0` | N,3 | f64 | window-origin pose; `pose` (carried, N,4 f64) is the executed pose at the tick |
| `t_goal_prob` | N,22 | f32 | T goal tokens (sigmoid), `GOAL22` order |
| `t_lat_logits`, `t_lon_logits` | N,5 / N,7 | f32 | T lat / lon class logits |
| `t_cons_pred` | N,26 | f32 | T constraint head (de-standardised) |
Existing carried columns: `tau,P,plan,gt,gt_ok,v0,pose,fired,win,ci,t0,tick,j,age_t,age_s,clip` (+ split,tag,ckpt,replan_steps,n_windows,clip_names). `j` = the tick's row index; `win` = index into the run's window list.
SCORING RULE: S and T are sample-and-hold. The output at tick j belongs to label row `j-age_s` (S) / `j-age_t` (T), not row j. Tensors equal what `v8f_mini_eval.cmd_run` collects (`ckpt, goal_prob, lat_logits, lon_logits, cons_pred`), so `checkpoint_metrics` / `tactical_metrics` apply unchanged once rows are matched to labels.

## Carried from hlat, and not
Carried: `gt_in_frame`, `TickRecorder` (+ its tau/P/plan/v0/GT columns), `keep_aux` in `model_planner`, `--dump-ticks`, the tick-file save and header key, and the hlat C-T2 invariants as 3 tests. Not carried: `--windows train` and its 4 args (not needed by `--dump-ticks`), `hacc_gate_v` header (needs the gate in `v8f_mini.py`), the `speed_bucket` header edit (hlat's version predates CAUSALBUCKET; keeping the tip's), everything in `v8f_hlat.py`.
Inert leftover: column `fired` reads `hacc_gate_fired` if the model emits it; the tip model has no gate, so it is False everywhere (asserted). Kept for schema parity with the hlat consumers.

## Evidence (outputs in `out/`)
* Tests: `pytest_t1_all_with_tipref.out` = 49 passed (35 existing `test_v8f_t1*.py` + 14 new; baseline of the 35 on the untouched tip: `pytest_baseline_tip.out`). `pytest_t1_heads_no_tipref.out` = 13 passed, 1 skipped (the byte-identical-to-tip test needs `V8F_T1_TIP_TOOL=<tip blob file>`, sha256-pinned).
* Mutation (`mutation.out`, clean sequential pass, 5 scratch mutants, all RED): delete the T lat-logits line -> 3 red; delete the S ckpt line -> 4 red; w0 frame uses the NOW pose -> 1 red; drop the `--dump-heads` guard -> 1 red; leak a header key into flag-OFF -> 3 red (incl. the byte-identical control). `mutation_first_pass_overlapped.out` is an earlier pass that overlapped a second pytest of mine; superseded, kept for the record.
* Real-data smoke (`smoke.out`, `smoke_check.out`): 24 windows, M2b_s0_hacc, seed 0, replan 10, CPU, inputs from hlat's smoke. tip vs flags-OFF vs heads-ON: 24 episode dumps bit-identical, tip==OFF result JSON identical as bytes (205,860 B), heads JSON equal after the 2 keys; 72 rows x all columns finite; tick-0 ckpt x = 29.9/49.4/77.2 m.
* Wall time (whole process, shared CPU with hlat's chain, so noisy): tip 31 s / 23 s (two runs), OFF 22 s, heads-ON 21 s; rollout line 6 s / 6 s / 4 s / 5 s. Overhead of the flags is inside the run-to-run spread.
Not done: no full eval, no GPU, nothing landed. Land with `tests/test_v8f_t1_heads.py` and the tool together; hlat's own copy of the tool will conflict on the `--dump-ticks` hunks and must be rebased onto this one.

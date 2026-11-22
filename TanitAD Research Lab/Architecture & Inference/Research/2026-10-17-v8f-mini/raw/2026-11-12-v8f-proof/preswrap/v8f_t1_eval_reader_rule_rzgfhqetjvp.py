#!/usr/bin/env python
"""v8f_t1_eval_reader_rule_rzgfhqetjvp.py -- written by make_wrapper_p.py from v8f_t1_eval_reader_rule_rzgfhqetjv.py (study "preswrap").
  New flag --vis-pres-file NPZ (default empty; form RR only): a presence file of preshead/fit/pres_dump.py (arrays p, has_out, clips, off, n, seed; row = off[clip index] + frame).
  New flag --vis-pres-mode head|or (required with the file): the reader's probability p of a (clip, frame) is replaced by p2 before presence is tested: head = the file's value, or = max(p, file value);
    a frame without a finite file value keeps p (counted). The tick look, the later looks and the earlier frames of the closing run all test presence on p2.
  With the file absent every behaviour and output is the source's. Rule log meta gains vis_pres_file, vis_pres_file_sha256, vis_pres_mode, vis_pres_counts (only with the file); the side files
  carry p (the reader's own) and, only with the file, p_used (= p2); the column p_src of the logs is p2. Text of the source (rzgfhqetjv) follows.
v8f_t1_eval_reader_rule_rzgfhqetjv.py -- written by make_wrapper_v.py from v8f_t1_eval_reader_rule_rzgfhqetj.py (card ADD102, study "visrule").
  New flag --vis-gap-file NPZ (default empty; form RR only): where the reader marks a lead and the file (the fixed-window head's gap per clip and frame) has an output, the rule receives the file's gap.
  New switch --vis-closing (default off; form RR only): the closing input is the slope of the gap the rule receives, by the run rule of LabelSource (at most 5 frames, steps up to 3.0 m, at least 3 frames).
  With both off every behaviour is the source's. The rule log meta gains vis_gap_file, vis_gap_file_sha256, vis_closing, vis_counts. Text of the source (rzgfhqetj) follows.
v8f_t1_eval_reader_rule_rzgfhqetj.py -- written by make_wrapper_j.py from v8f_t1_eval_reader_rule_rzgfhqet.py (card ADD94, study "gaprule").
  New flag --rule-join NPZ (default empty): the label source of the lead rule is built from this join; without it, from --eval-join as in the parent. Only the rule reads that source.
  The scoring keeps --eval-join (the namespace handed to the pinned tool is not changed). The rule log meta gains rule_join and rule_join_sha256.
  A rule join has to equal the evaluation join in clip names, off, n and lead_valid, else the run is refused.
  Every other behaviour is the parent's. Text of the parent (rzgfhqet) follows.
v8f_t1_eval_reader_rule_rzgfhqet.py -- written by make_shuf_wrapper.py from v8f_t1_eval_reader_rule_rzgfhqe.py (card ADD83, study "shufclose").
  New flag --closing-table NPZ: admitted only with form RRe and exclusive with --closing-est. The table has the arrays clip_index (int), frame (int) and est (float, m/s; NaN where there is no
  value). With it the estimate at a look is the table value at (clip index of the window, frame of the look): the same clip index (s.ci) and the same frame (j for the tick look, j + q for a later
  look) that _est_at uses. A look whose (clip index, frame) has no row in the table, or whose row holds NaN or an infinite value, has est_ok False, c_est NaN and the closing input exactly 0.0
  (the behaviour of form RR). Nothing else of the form changes: the columns c_est / est_ok of the rule log, the later-look log and the pre-roll log, and the run-text line
  "est looks scored full <n> short <n> preroll full <n> short <n> file <sha256>" keep their format; the file hash in the line is that of the table. The table path reads no reader
  features and no estimator file; the reader of the rule (presence and gap) is unchanged.
  Every other form, and form RRe with --closing-est, behave exactly as in the parent.
Text of the parent (rzgfhqe) follows.
v8f_t1_eval_reader_rule_rzgfhqe.py -- written by make_estclose_wrapper.py from v8f_t1_eval_reader_rule_rzgfhqc.py (card ADD79, study "estclose"): the form RRe.
  RRe  the lead rule of form RR (presence = reader probability >= 0.5 and a full five-frame history; gap = min(g_reader, Q(g_reader)), Q the q50 map) with the closing-speed input of the
       rule ESTIMATED from the token reader's own reads over the last H frames. New flag --closing-est NPZ (required for RRe, refused for every other form): an estimator file of
       closest.py (ClosingEstimator; arm L = ridge or N = mean of small networks; H frames).
  At EVERY look of the rule (the replan tick and every later look of --rule-every) at frame j (a later look: j + q), in the main phase and in the pre-roll:
       X [H, 3] = the reader's reads on the five-frame windows that end at the frames j-H+1 .. j, oldest first, per frame (gap read before any map, in m; presence p; closing output
       times CLOS_SCALE_MS = 10 m/s): the quantities of closeread_dump.py; closing input = ClosingEstimator.apply(X), in m/s, not clipped (the rule forms v_L = max(0, v_l + closing)).
       The estimate needs the frames j - need_frames + 1 .. j of the clip (need_frames = H + 4). If they are not all in the clip the closing input is exactly 0.0 (the behaviour of form RR)
       and est_ok is False. The reads come from the logged tokens of the clip (cache steps at or before the look frame only), cached per (clip, frame).
  Logs: rule_log_<run>.npz gets the arrays c_est (float64, NaN where there is no full history) and est_ok (bool), row for row with the other arrays; rule_log_mid_<run>.npz the same two
       keys; rule_log_pre_<run>.npz the same two keys for the pre-roll rows; the side file rule_log_<run>_rd.npz carries them too. The run text run_<run>.txt (same folder) gets one line
       "est looks scored full <n> short <n> preroll full <n> short <n> file <sha256 of the estimator file>" (scored = tick and later looks of the main phase, preroll = those of the pre-roll).
  Every other form behaves exactly as in the parent: the new text is reached only when the form is RRe (the only unconditional additions are a no-op call at the end of main and an
  argparse flag that is refused with every other form).
Text of the parent (rzgfhqc) follows.
v8f_t1_eval_reader_rule_rzgfhqc.py -- written by make_midlook_closing.py from v8f_t1_eval_reader_rule_rzgfhq.py (study "midlook-closing"): the later looks of the rule for RRc and RRh.
  --rule-every below the replan interval is accepted for the forms RLb, RLc, RR (as in the parent) and now also RRc and RRh. The form RRd is still REFUSED (its closing speed is a gap rate
  that needs an earlier read; the later looks do not carry one). Nothing else changes: with --rule-every at or above the replan interval, and for every later look of RLb, RLc, RR, the
  arrays are those of the parent (the later looks of RR feed closing 0 as before).
  At a later look q (frame j + q) of the forms RRc and RRh the reader branch of mid_looks is used (presence, gap through the map, from the logged tokens of frame j + q, as RR does):
    RRc  closing speed = the label closing at the clip and frame j + q (the mix branch closing == "label"); where the label has none and a lead is present, exactly 0 (the reader value of
         the form is the zero array, as at the tick).
    RRh  closing speed = the reader's own closing output (third output of the token reader times CLOS_SCALE_MS) on the same token window as the gap read at j + q: cache steps
         j + q - hist + 1 .. j + q of the window's own clip, frames at or before j + q only (new helper _reader_closing_at). NaN where that window does not exist; presence needs it.
    RR   unchanged: closing 0 at every later look.
  The mid-look log (rule_log_mid_<run>.npz) has one more key, c_read: the reader's closing read at the later look, logged in every reader form (NaN in the label forms), fed only in RRh.
Text of the parent (rzgfhq) follows.
v8f_t1_eval_reader_rule_rzgfhq.py -- written by make_longroll10k_b.py from v8f_t1_eval_reader_rule_rzgfhp.py (the first build of the study "longroll"): three repairs, all for P above 0.
  With --preroll-steps 0 the wrapper is still the parent: the only new text that can be reached is `STATE.get("preroll", 0) > 0` in ReaderPlanner.__call__ (False), and the lookups
  of the first build. No hook wrapper, no rule shim, no segment loop, no new file.
  R1  FIRST TICK AFTER A RE-BASE. The rule code sets the own progress a_m of the first tick (k = 0) to 0 by definition. After a pre-roll the own position is not the logged pose at t0,
      so ReaderPlanner.__call__ (P above 0) computes for every state with j == t0 and a logged frame pose set (a re-based state) a_m0 = R.progress(d["gt"][t0], pos0)[:, 0], pos0 = the own
      position at t0 in the frame of the logged pose at t0 (floors_kin.gt_waypoints on the two rows [logged pose, own pose], the arithmetic the rule uses for k above 0); 0.0 exactly for every other
      state, for a state without frame pose (the first segment, and P = 0), and for a state whose own x, y equal the logged x, y bit for bit. A wrapper installed on R.distance_rule (P above 0 only,
      restored by finish_preroll) subtracts a_m0 from the g0 argument of exactly the NEXT call (the tick-level call of MixPlanner / RulePlanner) and clears it; the calls of the second look
      (mid_looks, q above 0) get no adjustment. The sign follows g0 = g_src + a_l - a_m: own ahead of the logged pose by d gives a_m0 = +d and g0 smaller by d.
      LOG COLUMNS: the rule-log columns a_m and g0 of first-tick rows (k = 0) still show the DEFINITION values (a_m = 0, g0 = g_src); the rule used g0 - a_m0. a_m0 of the main phase's first tick
      per (ci, t0) is saved in preroll_offsets_<run>.npz as a_m0_first_tick (aligned with its ci / t0 rows; computed by the runner at the re-base, with the same function the planner uses); the
      same for every later re-base under seg_*. preroll_log_<run>.json carries a_m0 (largest |a_m0| per kind and per run, main phase and segments) and a_m0_used_in_rule (what the shim
      subtracted: calls, non-zero entries, largest |a_m0|; empty for form R0, where no rule runs). The path extension of the rule (dist_rule._path_frame) is RAY = 1000 m straight on both sides
      of the 30-step logged path (back ray along the first non-degenerate segment, forward ray along the last); beyond 1000 m behind, a_m0 saturates at -1000; the projection is orthogonal, so a
      lateral offset does not change it; a degenerate logged path (all steps below 1e-9 m) extends along +x of the logged frame.
  R2  PRE-ROLLS LONGER THAN 30 STEPS. P must be a multiple of 30 (and of --replan-steps), else REFUSED. The pre-roll runs P / 30 consecutive segments, each ONE call of the original rollout with the
      module global H of the pinned tool set to 30 (restored in a finally), hook None, the chosen executor. After EVERY segment the state is re-based as before the main phase: st.t0 advances by 30,
      st.frame_pose5 is the logged pose at the new t0, st.clipped is added to the pre-roll count and reset; the rule therefore reads the arc tables of the 30-step logged path from the new t0. The first
      segment starts from the logged state with no frame pose (R1 adjustment exactly 0); R1 applies at the first tick of every later segment and of the main phase. The mutable stores are zeroed
      before each segment and kept per segment; the main-phase stores are restored after the last one (as in the first build). rule_log_pre_<run>.npz keeps its fields; its t0 column is the segment's
      t0 (t0 - P + 30 s) and tick counts inside the segment; a new column pre_segment (0 .. P/30 - 1) names the segment. A re-base at a step without a finite logged 30-step path stops the run.
      Limit (i) of the first build (the arc difference built up in the pre-roll is not carried) is repaired for the first tick by R1; the later ticks carry it through a_m, which projects the
      own position, in the logged frame, on the logged path. Limit (ii) (index clipping beyond step 30) is repaired by the segments.
  R3  TICK RECORDER FRAMES. The pinned TickRecorder expresses two recorded quantities in the window-origin frame: pose_w0 (= st.d["pose"][st.t0][:3], the world pose of the window origin) and
      s_ckpt_w0 (derived at save() by the pinned M.transform_ckpt from s_ckpt, s_pose and pose_w0). In the first build both were the OWN pose at t0 for P above 0. Now the main-phase call gets
      the hook through HookFrame, which runs the recorder and then replaces, for every state with a frame pose, the row just appended to rec.rows["pose_w0"] by the LOGGED pose at t0 ([:3], float64);
      s_ckpt_w0 then follows from the unchanged save(). MAPPED: pose_w0, s_ckpt_w0. LEFT, with the reason: pose (the world pose of the executed state at the tick: not a t0-frame quantity), s_pose and
      t_pose (world poses of the S / T tick frames), s_ckpt (raw, in S's tick frame), s_ckpt_now and plan (the NOW frame of the tick), gt and gt_ok (the logged future in the NOW frame), tau, P,
      v0, fired, win, ci, t0 (the logged t0 since the re-base), tick, j, age_t, age_s, t_goal_prob, t_lat_logits, t_lon_logits, t_cons_pred (no frame). Counters of the mapping are in the
      preroll log (tick_recorder_pose_w0_mapped). Assumes the hook has a rows dict of lists in which pose_w0 gets one row per state, in state order, per call (read in the pinned tool; not
      runnable on the dev box).
Text of the first build (rzgfhp) follows; where it differs from the three repairs, the repairs win (its item f ends with the two own-frame columns: that is repaired by R3).
v8f_t1_eval_reader_rule_rzgfhp.py -- written by make_longroll10k.py from v8f_t1_eval_reader_rule_rzgfh.py for the study "longroll" (pre-roll before every window): two more parameters.
  --preroll-steps P (integer, 0 or more, a multiple of --replan-steps; default 0)   --preroll-exec plan|logged (default plan; only with P above 0)
  P = 0: the wrapper is the parent. No pre-roll code runs and no file is written. The only new text that is reached is two lookups that return the parent's own values
         (getattr(s, "frame_pose5", None) is None in the second look of the rule, and STATE.get("preroll", 0) in the planner constructor), and a main() that calls the parent main.
  P above 0, for every call of the pinned tool's rollout (the model arm and the kinematic-prior arm each get their own pre-roll by their own planner):
    a. each window (ci, t0) of the grid gets its state built at t0 - P (refused with a clear error when t0 - P is below 3);
    b. pre-roll: P / m ticks through the ORIGINAL rollout (the same planner with rule, readout, bound and second look; hook None, so the tick recorder records nothing), the module
       global H of the pinned tool set to P for that call and restored in a finally. Read in the pinned tool: rollout uses H only for the tick count (H // m, and H % m); exec_in_ego
       uses H for the number of knots (the executed_ego that the pre-roll returns is discarded); gt_in_frame (tick recorder only) uses H and is not called in the pre-roll. Nothing
       else reached from rollout reads H. Executor plan (own plan executed), or logged with the full logged tracks (--preroll-exec logged: the carry-identity control);
    c. then the unmodified rollout for 30 steps from st.j = t0 (ages restart at step 0): st.t0 is set to t0, st.clipped is reset (the pre-roll count goes to the preroll log), and
       executed_ego() returns the executed steps t0+1 .. t0+30 in the frame of the LOGGED pose at t0 (row 2*t0 of the full logged native track; the arithmetic of floors_kin.gt_waypoints
       with the own pose at t0 replaced by the logged pose). plan0 (the tick-0 plan, in the frame of the own pose at t0) is mapped into the same logged frame. Ground truth g, v0 and
       the other values cmd_run takes from the store (d["gt"][t0], d["kin3"][t0]) are the logged values of window (ci, t0), unchanged. n_clip counts the main phase only;
    d. dead-reckoned nav is held from the pre-roll start t0 - P, its route progress from own motion (as the pinned tool does it);
    e. the lead rule (forms RLb, RLc, RR) in the main phase reads the own position in the logged frame at t0 (the planner's fk.gt_waypoints sees the logged pose at t0 for the own
       track; the second look does the same through frame_pose5). KNOWN LIMITS, not repaired here: (i) the rule's gap bookkeeping g0 = g_src + a_l - a_m restarts at t0 and does not
       carry the arc difference built up in the pre-roll; (ii) in the pre-roll the rule's arc tables are those of the 30-step logged path from t0 - P, so for pre-roll ticks beyond
       step 30 the index is clipped (np.clip(k - 1, 0, 29)). Both only matter for rule forms at P above 0;
    f. logs: every file a reader consumes (rule log, mid-look log, p1_first, readout log, accel log, tick recorder file, causal-bucket and mix counters of the header) holds main-phase
       rows only, in the shapes of a P = 0 run on the same grid: the mutable counters and row stores of the wrapper and of the pinned tool are snapshotted before the pre-roll and
       restored in place after it. Pre-roll rows go to rule_log_pre_<run>.npz (same format; the t0 column is the pre-roll start t0 - P, tick is the pre-roll tick), the counters
       of the pre-roll phase and the per-window offsets summary to preroll_log_<run>.json, the per-window offsets to preroll_offsets_<run>.npz. The tick recorder's pose_w0 and
       s_ckpt_w0 columns (pinned tool) are in the frame of the OWN pose at t0, not the logged one, when P is above 0.
Text of the parent below.
Header of the parent:
v8f_t1_eval_reader_rule_rzgfh.py -- written by make_headbound10k.py from v8f_t1_eval_reader_rule_rzgf.py for card PREREG_V8F_PROOF_ADD67 (HEADBOUND10K): one more parameter,
  --kappa-max K (1/m, finite, above 0; needs --readout, which may be pinned): after the chosen readout has returned (xy, psi, v) for the 2m frames of the executed steps of a commit,
            the heading column is passed through the bound, frame by frame, starting from psi0 (the heading before the block): with prev the previous output heading (psi0 for the
            first frame), xy_prev the previous frame position (knots[0], the current knot, for the first frame), d = wrap(psi[f] - prev) and lim = K x |xy[f] - xy_prev|:
            if |d| <= lim the output heading is psi[f] itself, else prev + sign(d) x lim (wrapped into [-pi, pi)). Positions and speeds are not changed. The psi returned here is what
            the pinned tool stores as the executed heading (track column 2) and passes back as psi0 at the next commit (v8f_t1_eval.py: commit_plan, pose[2]).
            Without the parameter the wrapper is the wrapper of card ADD65 bit for bit. The readout log gets the fields kappa_max, frames_total, frames_bound_acted, max_excess_rad
            (largest of |output step| - lim over all frames), max_abs_diff_where_not_acted, max_abs_diff_xy, max_abs_diff_v and the frame counts by the speed of the frame (below 2, 2 to 8,
            above 8 m/s). Nothing else is new.
Header of the parent:
v8f_t1_eval_reader_rule_rzgf.py -- written by make_cand10k.py from v8f_t1_eval_reader_rule_rzf.py for card PREREG_V8F_PROOF_ADD65 (CAND10K): the readout causal_g of card ADD63,
with two parameters, added to the wrapper of card ADD62 (--rule-every). Nothing else is new.
  --readout causal_g --acc-gain G : positions and headings as pinned; the speed column from executed steps only: at the knot of executed step i the step speed s[i], at the
            half step before it s[i] - 0.5 x G x (s[i] - s[i-1]). The one-frame acceleration read at every executed knot, and so at the tick, is G times the executed one.
            G = 1.0 goes through the formula of the readout causal and G = 0.0 through the formula of the readout causal_z, unchanged (so the dumps equal those of the two
            readouts array for array); every other G goes through the general formula above.
  --path-exec (only with causal_g): positions and headings from the pinned frames function given the executed knots only (as the readout backward does); the speed column as above.
The readout log gets three more fields: acc_gain, path_exec and max_abs_acceleration_minus_gain_executed (the largest absolute difference between the one-frame
acceleration read at the last executed knot and G times the executed acceleration, over all committed plans). Every other readout and the parameter --rule-every of the parent
are unchanged (bit for bit). Readout K of card ADD65 is --readout causal_g --acc-gain 0 --path-exec; the second look is --rule-every 5.
Header of the parent:
v8f_t1_eval_reader_rule_rzf.py -- written by make_rulefast10k.py from v8f_t1_eval_reader_rule_rz.py for card PREREG_V8F_PROOF_ADD62 (RULEFAST10K): one more parameter,
--rule-every h (executed steps). The lead rule acts at the tick as before and then again for every later step q = h, 2h, ... below the replan interval, on the part of the plan
that is still to be driven, with the rule inputs a tick on frame j + q would take (label read or reader read of the logged frame j + q, logged speed and arc at that step,
the car progress at knot q, the car speed = the chord speed of the last step before q). One commit of the replan interval as before. Rule forms RLb, RLc and RR only. The later
looks go to ticks/rule_log_mid_<run>.npz (the tick log keeps one row per window and tick); the plan after the rule at the first tick goes to ticks/p1_first_<run>.npz.
Default h = 10 (or any h at or above the replan interval): nothing new runs and nothing new is written. Text of the source below.
v8f_t1_eval_reader_rule_rz.py -- written by make_accwin10k.py from v8f_t1_eval_reader_rule_ro.py for card PREREG_V8F_PROOF_ADD55 (ACCWIN10K): two more readouts.
Both keep the positions and headings of the pinned frames function and the causal speed at a knot (the step speed into it); they differ from `causal` in the half-step
speed, which sets the one-frame acceleration the model reads at every executed step:
  causal_w  acceleration = (s[i] - s[i-w]) / (0.2 s x w), w = min(i, 5): the executed speed change over the last 1.0 s of the block (i = step in the block, s[0] = the
            step into the block)
  causal_z  acceleration = 0 at every executed step (the model reads its executed speed and no acceleration from its own rollout)
`causal`, `backward`, `pinned` are unchanged (bit for bit). Header of the parent:
v8f_t1_eval_reader_rule_ro.py -- written by make_handover10k.py from v8f_t1_eval_reader_rule_a.py for card PREREG_V8F_PROOF_ADD54 (HANDOVER10K): one more parameter,
--readout, which changes what the rebuilt ego state holds as speed (and with it the one-frame acceleration) after executed steps. The pinned tool fills the 10 Hz frames
of the executed steps by a spline with central tangents, so the speed at the last executed knot is the mean of the last executed step and the next step of the plan,
which is never executed.  pinned   = the tool as it is (the frames function is passed through and only counted);
                          backward = the same function given the executed knots only (a backward difference at the last executed knot);
                          causal   = positions and headings as pinned, the speed column from executed steps only: at a knot the step speed into it, at the half
                                     step the mean of that step and the step before it (so the one-frame acceleration at a tick is the executed one).
It acts on every committed plan (model, kinematic prior alone, rule forms). A study arm; the pinned file is not edited. Header of the parent:
v8f_t1_eval_reader_rule_a.py -- written by make_feas10k.py from v8f_t1_eval_reader_rule_t.py for card PREREG_V8F_PROOF_ADD53 (FEAS10K): two more parameters,
--a-max and --b-lim (m/s2). With them every plan of the model is retimed along its own path before anything else: the step speeds may rise by at most a-max * 0.2 s
and fall by at most b-lim * 0.2 s per step, starting from the ego speed at the tick (the speed feature the model itself reads). It is a study arm, no vehicle model:
no jerk limit, no lateral limit. Works with every form, R0 included; a lead rule, if on, acts on the limited plan. The rollout of the kinematic prior alone (the
floor arrays of the dump) is limited the same way and has its own log columns. Without the two parameters nothing changes.
Header of the parent:
v8f_t1_eval_reader_rule_t.py -- written by make_taucurve10k.py from v8f_t1_eval_reader_rule_d.py for card PREREG_V8F_PROOF_ADD51 (TAUCURVE10K): one more parameter,
--tau SECONDS, the time gap of the distance rule (dist_rule.distance_rule(tau=...)); without it the rule keeps its banked 1.2 s and every form is unchanged.
The scoring of the unsafe rate does not read this parameter. Header of the parent:
v8f_t1_eval_reader_rule_d.py -- written by make_gaprate10k.py from v8f_t1_eval_reader_rule_h.py for card PREREG_V8F_PROOF_ADD50 (GAPRATE10K): one more form,
RRd (reader presence and gap through the map; closing speed = slope times the change of the reader's own gap read over the last K cache steps, where the reader is
present at both ends, else exactly 0; the earlier read uses the logged camera tokens of the cache steps j-K-4 .. j-K of the window's own clip, frames before the
tick only; no model weight, no label). The eight forms of the parent are unchanged. Header of the parent:
v8f_t1_eval_reader_rule_h.py -- written by make_closer10k.py from v8f_t1_eval_reader_rule_c.py for card PREREG_V8F_PROOF_ADD49 (CLOSER10K): one more form,
RRh (reader presence and gap through the map, closing speed = the third output of the same token reader times 10 m/s; tokens only, no model weight, no label).
The seven forms of the parent are unchanged. Header of the parent:
v8f_t1_eval_reader_rule_c.py -- written by make_close10k.py from v8f_t1_eval_reader_rule.py for card PREREG_V8F_PROOF_ADD48 (CLOSE10K): two more forms,
RLc (labels: presence, gap and closing speed) and RRc (reader presence and gap through the map, closing speed from the labels where the label has one, else 0;
a diagnostic, not admissible at inference). The five forms of card ADD46 are unchanged. Header of the original:
v8f_t1_eval_reader_rule.py -- card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE), part 1: the study wrapper around the pinned T1-kin tool v8f_t1_eval.py of the tree of commit cdbff971.

  python v8f_t1_eval_reader_rule.py --tree TREE --form {R0,RLb,RR,RRid,RR0} [--reader CKPT --qmap JSON] [--rule-log NPZ] run <the argv of the pinned tool's `run`>

Forms of the card (the rule is the banked distance rule of cards ADD26/ADD33 in its form WITHOUT the closing speed, floor form cap-only as the arms of ADD33/ADD42, parameters unchanged):
  R0    nothing installed: the pinned tool's own run.
  RLb   the rule fed by labels (presence, gap from the label join; closing speed 0).
  RR    the rule fed by the reader: presence = reader probability >= 0.5 and a full 5-frame history; gap g_used = min(g_reader, Q(g_reader)), Q the monotone map (--qmap).
  RRid  as RR with the reader's gap replaced by 1e9 m (the rule is inactive above 60 m: the plan is returned unchanged).
  RR0   as RR with the reader's gap replaced by 0 m.
The reader reads batch["feats"][:, -5:], the cached encoder tokens of cache steps j-4 .. j at the read tick j (frames at or before the tick only), and nothing else; no model weight is read.
The pinned tool is imported from --tree by file path and is NOT edited; the rule code (dist_rule.py, v8f_t1_eval_rule.py, v8f_t1_eval_rule_mix.py of the distrule folder) is imported, not edited.
A side file `<rule-log minus .npz>_rd.npz` records the reader's presence, gap, gap used and the history flag row for row with the rule log.
Tier T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.
STRATEGIC: not read by this wrapper (see note N1 of the card: T0 at the first tick, unchanged by construction for an inference-time rule).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FORMS = ("R0", "RLb", "RR", "RRid", "RR0", "RLc", "RRc", "RRh", "RRd", "RRe")
DT_S = 0.2                                          # seconds per cache step (the pinned tool: 30 steps = 6 s)
CLOS_SCALE_MS = 10.0                                # = CLOS_SCALE_MS of tok_probe_lr.py (the probe was trained on closing / 10 m/s)
P_MIN = 0.5
GAP_PLANT = {"RRid": 1e9, "RR0": 0.0}
PINNED_SHA256 = "61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397"


def find_distrule():
    cands = [os.environ.get("DISTRULE_DIR", ""), os.path.join(HERE, "distrule"), os.path.join(HERE, "..", "distrule"), HERE]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "v8f_t1_eval_rule_mix.py")):
            return os.path.normpath(c)
    raise SystemExit("REFUSED: the distrule folder (v8f_t1_eval_rule_mix.py, dist_rule.py, v8f_t1_eval_rule.py) was not found; set DISTRULE_DIR")


for _p in (find_distrule(), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import qcal_map as QM  # noqa: E402
import v8f_t1_eval_rule_mix as X  # noqa: E402

W = X.W
R = W.R
_RULE_PLANNER = W.RulePlanner          # the original class, kept before main() points W.RulePlanner at ReaderPlanner
STATE = {"form": "R0", "reader": None, "xs": None, "ys": None, "slope": None, "k": 0, "prev": None, "zero": None, "alim": None, "m": 10, "alog": {}, "readout": "pinned", "gain": None, "path_exec": False, "hf": {"calls": 0, "max_dv": 0.0, "max_da": 0.0, "sum_dv": 0.0, "sum_da": 0.0},
         "cols": {k: [] for k in ("p", "g_read", "g_used", "ok", "c_read")}}


# ---------------------------------------------------------------- the pure feed (tested without the tool)
def q_raw(g, xs, ys):
    """Q(g) of the monotone map without the min with g (the form of qcal_map.apply_map before its min); NaN stays NaN."""
    g = np.asarray(g, np.float64)
    out = np.array(g, copy=True)
    fin = np.isfinite(g)
    gg = g[fin]
    q = np.interp(gg, xs, ys)
    lo = gg < xs[0]
    q[lo] = ys[0] * gg[lo] / xs[0]
    hi = gg > xs[-1]
    q[hi] = gg[hi] - (xs[-1] - ys[-1])
    out[fin] = q
    return out


def feed(form, p, g, ok, xs=None, ys=None, p_min=P_MIN):
    """-> (present [B] bool, g_used [B] float64) of the reader forms. g is the reader's gap read; ok the full-history flag."""
    p = np.asarray(p, np.float64)
    g = np.asarray(g, np.float64)
    present = (p >= p_min) & np.asarray(ok, bool)
    if form in GAP_PLANT:
        used = np.full(g.shape, GAP_PLANT[form], np.float64)
    elif form in ("RR", "RRc", "RRh", "RRd", "RRe"):
        used = np.minimum(g, q_raw(g, xs, ys))                    # g_used = min(g_reader, Q(g_reader)); NaN stays NaN
    else:
        raise ValueError("no reader feed for form " + form)
    return present, used


def reader_closing(rd, tokens):
    """The third output of the token reader: closing speed (lead speed minus ego speed) in m/s, [B]. Reads nothing but the tokens."""
    import tok_reader as TR
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().float().cpu().numpy()
    E = rd.frame_embed(np.asarray(tokens, np.float32))
    x = E.reshape(E.shape[0], -1)
    h = TR._gelu(x @ rd.w["head.mlp.0.weight"].T + rd.w["head.mlp.0.bias"])
    o = h @ rd.w["head.mlp.2.weight"].T + rd.w["head.mlp.2.bias"]
    assert o.shape[1] == 3, o.shape
    return (o[:, 2].astype(np.float64) * CLOS_SCALE_MS).astype(np.float32)


def prev_read(states, rd, K):
    """The reader at the earlier cache step j-K of each window: the tokens of the cache steps j-K-4 .. j-K of the window's own clip (the logged array the batch tokens
    are gathered from; frames at or before the tick only). -> p [B], g [B] (NaN where the 5-frame history before j-K is not complete), ok [B]."""
    B = len(states)
    p, g, ok = np.full(B, np.nan, np.float32), np.full(B, np.nan, np.float32), np.zeros(B, bool)
    rows, toks = [], []
    for i, s in enumerate(states):
        e = int(s.j) - int(K)
        if e - (rd.hist - 1) >= 0 and e < len(s.d["feats"]):
            rows.append(i)
            toks.append(np.asarray(s.d["feats"][e - rd.hist + 1:e + 1], np.float32))
    if rows:
        pp, gg = rd.read(np.stack(toks))
        p[rows], g[rows], ok[rows] = pp, gg, True
    return p, g, ok


def gap_rate_closing(present, g, p_prev, g_prev, ok_prev, slope, K, p_min=P_MIN):
    """-> (have [B] bool, c_fed [B] float32, m/s). have: the reader is present now and at the cache step j-K (full history there, probability >= p_min).
    c_fed = slope * (g - g_prev) / (DT_S * K) where have (the gap rate = lead speed minus ego speed), else exactly 0."""
    g, gp = np.asarray(g, np.float64), np.asarray(g_prev, np.float64)
    with np.errstate(invalid="ignore"):
        have = np.asarray(present, bool) & np.asarray(ok_prev, bool) & (np.asarray(p_prev, np.float64) >= p_min) & np.isfinite(g) & np.isfinite(gp)
        raw = (g - gp) / (DT_S * float(K))
    return have, np.where(have, float(slope) * raw, 0.0).astype(np.float32)


MID_EST = ("c_est", "est_ok")                       # the two keys the later-look log gets in form RRe (estclose)


def _find_closest():
    for c in (os.environ.get("CLOSEST_DIR", ""), HERE, os.path.join(HERE, "..", "closeread"), os.path.join(HERE, "closeread")):
        if c and os.path.isfile(os.path.join(c, "closest.py")):
            return os.path.normpath(c)
    raise SystemExit("REFUSED: closest.py (the closing estimator format of card ADD79) was not found; set CLOSEST_DIR")


def load_estimator(path, rd, rule_log=""):
    """Loads the estimator file of form RRe (closest.ClosingEstimator) and opens the caches and the log columns of the form. Refuses a reader history other than the one the file format has."""
    if not path or not os.path.isfile(path):
        raise SystemExit("REFUSED: --closing-est file not found: %s" % path)
    d = _find_closest()
    if d not in sys.path:
        sys.path.insert(0, d)
    import closest as CL
    est = CL.ClosingEstimator.load(path)
    if int(rd.hist) != int(CL.READER_HIST):
        raise SystemExit("REFUSED: the reader history is %d frames, the estimator format has %d" % (int(rd.hist), int(CL.READER_HIST)))
    STATE["est"] = {"est": est, "path": path, "sha256": W.sha256_file(path), "emb": {}, "reads": {}, "cache": True, "embed_calls": 0, "rule_log": rule_log}
    for k in MID_EST:
        STATE["cols"][k] = []
        STATE["mid"]["rows"][k] = []
    return est


def _head3(rd, E):
    """The head of the token reader on frame embeddings E [n, hist, 64]: (gap_m before any map, presence p, closing_ms), each [n] float32, exactly the quantities of closeread_dump.py
    (TokReader.head_from_embeds for p and the gap, reader_closing for the closing output; CLOS_SCALE_MS applied here)."""
    import tok_reader as TR
    x = np.asarray(E).reshape(len(E), -1)
    h = TR._gelu(x @ rd.w["head.mlp.0.weight"].T + rd.w["head.mlp.0.bias"])
    o = h @ rd.w["head.mlp.2.weight"].T + rd.w["head.mlp.2.bias"]
    assert o.shape[1] == 3, o.shape
    p = (1.0 / (1.0 + np.exp(-o[:, 0].astype(np.float64)))).astype(np.float32)
    g = (o[:, 1] * TR.GAP_SCALE_M).astype(np.float32)
    c = (o[:, 2].astype(np.float64) * CLOS_SCALE_MS).astype(np.float32)
    return g, p, c


def _est_fill(rd, wanted):
    """wanted: {(clip, frame): feats array of that clip}. Fills STATE["est"]["reads"][(clip, frame)] = (gap, p, closing) of the five-frame window ending at that frame. The frame embeddings
    (the per-frame part of the reader) are cached per (clip, frame), so windows that overlap share them. With STATE["est"]["cache"] False both caches are emptied first."""
    S = STATE["est"]
    if not S["cache"]:
        S["emb"].clear()
        S["reads"].clear()
    need_e = {}
    for (ci, f), feats in wanted.items():
        if (ci, f) in S["reads"]:
            continue
        for ff in range(f - rd.hist + 1, f + 1):
            if (ci, ff) not in S["emb"]:
                need_e[(ci, ff)] = feats
    keys = list(need_e)
    for i in range(0, len(keys), 64):
        part = keys[i:i + 64]
        E = rd.frame_embed(np.stack([np.asarray(need_e[k][k[1]], np.float32) for k in part]))
        S["embed_calls"] += 1
        for k, e_ in zip(part, E):
            S["emb"][k] = np.asarray(e_)
    todo = [k for k in wanted if k not in S["reads"]]
    if todo:
        win = np.stack([np.stack([S["emb"][(ci, f - rd.hist + 1 + t)] for t in range(rd.hist)]) for (ci, f) in todo])
        g, p, c = _head3(rd, win)
        for k, a, b, d in zip(todo, g, p, c):
            S["reads"][k] = (float(a), float(b), float(d))


TABLE_STRIDE = 1 << 20                              # key of the table = clip index * TABLE_STRIDE + frame (frames of a clip are far below the stride)


def load_table(path, rule_log=""):
    """Loads the table of form RRe with --closing-table (arrays clip_index, frame, est) and opens the log columns of the form. Refuses a file without the arrays, rows of different
    lengths, a frame outside [0, TABLE_STRIDE), a negative clip index or two rows with the same (clip index, frame)."""
    if not path or not os.path.isfile(path):
        raise SystemExit("REFUSED: --closing-table file not found: %s" % path)
    with np.load(path, allow_pickle=False) as z:
        for k in ("clip_index", "frame", "est"):
            if k not in z.files:
                raise SystemExit("REFUSED: --closing-table file lacks the array %s" % k)
        ci, fr, v = np.asarray(z["clip_index"]).astype(np.int64), np.asarray(z["frame"]).astype(np.int64), np.asarray(z["est"]).astype(np.float64)
    if not (ci.ndim == fr.ndim == v.ndim == 1 and len(ci) == len(fr) == len(v)):
        raise SystemExit("REFUSED: --closing-table arrays are not one-dimensional of one length")
    if len(ci) and (ci.min() < 0 or fr.min() < 0 or fr.max() >= TABLE_STRIDE):
        raise SystemExit("REFUSED: --closing-table has a negative clip index or a frame outside [0, %d)" % TABLE_STRIDE)
    key = ci * TABLE_STRIDE + fr
    order = np.argsort(key, kind="stable")
    key, v = key[order], v[order]
    if len(key) > 1 and bool(np.any(key[1:] == key[:-1])):
        raise SystemExit("REFUSED: --closing-table has two rows with the same clip index and frame")
    STATE["est"] = {"est": None, "table": {"key": key, "val": v}, "path": path, "sha256": W.sha256_file(path), "emb": {}, "reads": {}, "cache": True, "embed_calls": 0, "rule_log": rule_log}
    for k in MID_EST:
        STATE["cols"][k] = []
        STATE["mid"]["rows"][k] = []
    return STATE["est"]["table"]


def _table_at(states, e_off):
    """The closing input of form RRe taken from the table at the look on frame e = j + e_off of each window: the value at (clip index, e). -> c_est [B] float64 (NaN where the table has no
    row or a value that is not finite) and est_ok [B] bool."""
    T = STATE["est"]["table"]
    B = len(states)
    c_est, est_ok = np.full(B, np.nan), np.zeros(B, bool)
    if B and len(T["key"]):
        ci = np.array([int(s.ci) for s in states], np.int64)
        e = np.array([int(s.j) + int(e_off) for s in states], np.int64)
        inside = (ci >= 0) & (e >= 0) & (e < TABLE_STRIDE)
        key = np.where(inside, ci * TABLE_STRIDE + e, -1)
        pos = np.searchsorted(T["key"], key)
        posc = np.minimum(pos, len(T["key"]) - 1)
        hit = inside & (pos < len(T["key"])) & (T["key"][posc] == key)
        v = np.where(hit, T["val"][posc], np.nan)
        est_ok = hit & np.isfinite(v)
        c_est = np.where(est_ok, v, np.nan)
    return c_est, est_ok


def _est_at(states, rd, e_off):
    """The closing estimate (m/s) at the look on frame e = j + e_off of each window: X [H, 3] = the reads (gap, presence, closing) of the windows ending at frames e-H+1 .. e, oldest first,
    through the estimator. -> c_est [B] float64 (NaN where the frames e - need_frames + 1 .. e are not all in the clip) and est_ok [B] bool (full history and a finite estimate)."""
    S = STATE["est"]
    if S.get("table") is not None:                       # --closing-table: the value of the table at (clip index, frame of the look)
        return _table_at(states, e_off)
    est = S["est"]
    H, need = int(est.H), int(est.need_frames(rd.hist))
    B = len(states)
    c_est, est_ok = np.full(B, np.nan), np.zeros(B, bool)
    rows, wanted = [], {}
    for i, s in enumerate(states):
        e = int(s.j) + int(e_off)
        if e - need + 1 >= 0 and e < len(s.d["feats"]):
            rows.append((i, int(s.ci), e))
            for f in range(e - H + 1, e + 1):
                wanted[(int(s.ci), f)] = s.d["feats"]
    if rows:
        _est_fill(rd, wanted)
        X = np.stack([np.array([S["reads"][(ci, f)] for f in range(e - H + 1, e + 1)], np.float64) for _, ci, e in rows])
        v = est.apply(X)
        idx = [i for i, _, _ in rows]
        c_est[idx] = v
        est_ok[idx] = np.isfinite(v)
    return c_est, est_ok


def est_fed(c_est, est_ok):
    """the closing input of the rule: the estimate where there is one, exactly 0.0 where not (the closing input of form RR)."""
    return np.where(est_ok, c_est, 0.0)


def _est_tick(n):
    """the closing input at the tick-level look (frame j of each window); logs the estimate and its flag, one entry per call, row for row with the rule log."""
    states = STATE.get("est_states")
    if states is None or len(states) != n:
        raise RuntimeError("the closing estimate needs the states of this batch (ReaderPlanner not bound)")
    c_est, ok = _est_at(states, STATE["reader"], 0)
    STATE["cols"]["c_est"].append(c_est)
    STATE["cols"]["est_ok"].append(ok)
    return est_fed(c_est, ok)


def est_tick_arrays(log):
    """c_est and est_ok of the tick-level looks of the main phase, as arrays of the length of the rule log."""
    c = np.concatenate(STATE["cols"]["c_est"]) if STATE["cols"]["c_est"] else np.zeros(0)
    ok = np.concatenate(STATE["cols"]["est_ok"]) if STATE["cols"]["est_ok"] else np.zeros(0, bool)
    n = len(log.arrays()["ci"])
    if len(c) != n or len(ok) != n:
        raise RuntimeError("estimate columns have %d / %d rows, the rule log %d" % (len(c), len(ok), n))
    return {"c_est": c, "est_ok": ok}


def _count_ok(parts):
    full = short = 0
    for lst in parts:
        for a in lst:
            a = np.asarray(a, bool)
            full += int(a.sum())
            short += int(a.size - a.sum())
    return full, short


def est_line():
    """the text line of the run file: counts of the looks with a full history (est_ok) and without, scored (tick and later looks of the main phase) and pre-roll; None outside form RRe."""
    S = STATE.get("est")
    if S is None or not S.get("rule_log"):
        return None
    sf, ss = _count_ok([STATE["cols"]["est_ok"], STATE["mid"]["rows"]["est_ok"]])
    runner = STATE.get("runner")
    pf, ps = _count_ok([part for kind, post in (runner.posts if runner is not None else []) if kind == "model" for part in (post["cols"]["est_ok"], post["mid"]["rows"]["est_ok"])])
    return "est looks scored full %d short %d preroll full %d short %d file %s" % (sf, ss, pf, ps, S["sha256"])


def write_est_line():
    """appends est_line() to run_<run>.txt, the file beside the rule log (the queue writes its own lines there); nothing in any other form."""
    line = est_line()
    if line is None:
        return
    d, base = os.path.split(os.path.abspath(STATE["est"]["rule_log"]))
    run = base[len("rule_log_"):-4] if base.startswith("rule_log_") and base.endswith(".npz") else os.path.splitext(base)[0]
    with open(os.path.join(d, "run_" + run + ".txt"), "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def head_values_reader(model, feats, frame_ok=None, hist=None):
    """Replaces W.head_values: p [B], gap [B], closing [B] (zero: the rule form without the closing speed), ok [B]. Takes tokens only; `model` is not read."""
    rd = STATE["reader"]
    f = feats[:, -rd.hist:]
    p, g = rd.read(f)
    ok = np.ones(len(p), bool) if frame_ok is None else frame_ok[:, -rd.hist:].all(-1).cpu().numpy()
    p_rd = p                                                    # the reader's own probability (logged); presence is tested on p2 (study preswrap)
    if STATE.get("pres") is not None:
        p = pres_tick(p)
    present, used = feed(STATE["form"], p, g, ok, STATE["xs"], STATE["ys"])
    vis_c = None
    if STATE.get("vis") is not None:                            # the variant of card ADD102: the file's gap and the closing input of the rule
        used, vis_c = vis_tick(p, g, used, ok)
    c = reader_closing(rd, f)                                   # logged in every reader form; fed into the rule only in form RRh
    for k, v in (("p", p_rd), ("g_read", g), ("g_used", used), ("ok", ok), ("c_read", c)):
        STATE["cols"][k].append(np.asarray(v))
    if STATE.get("pres") is not None:
        STATE["cols"].setdefault("p_used", []).append(np.asarray(p))
    c_fed = np.zeros(len(p), np.float32)
    c_estv = _est_tick(len(p)) if STATE["form"] == "RRe" else None      # form RRe: the closing estimate of the tick-level look
    if STATE["form"] == "RRd":
        if STATE["prev"] is None or STATE["zero"] is None or len(STATE["prev"][0]) != len(p) or len(STATE["zero"][0]) != len(p):
            raise RuntimeError("the earlier read does not belong to this batch")
        pp, gp, okp = STATE["prev"]
        p0, g0, ok0 = STATE["zero"]
        have, c_fed = gap_rate_closing(present, g, pp, gp, okp, STATE["slope"], STATE["k"])
        okb = np.asarray(ok, bool)
        with np.errstate(invalid="ignore"):                         # the gather at K = 0 must give the read of the batch tokens: logged per tick (control d4)
            d0 = np.maximum(np.abs(np.asarray(g, np.float64) - g0), np.abs(np.asarray(p, np.float64) - p0))
        id0 = np.where(okb & ok0, d0, np.where(okb == ok0, 0.0, np.inf))
        for k, v in (("p_prev", pp), ("g_prev", gp), ("ok_prev", okp), ("have", have), ("c_fed", c_fed), ("id0", id0)):
            STATE["cols"].setdefault(k, []).append(np.asarray(v))
        STATE["prev"] = STATE["zero"] = None
    # presence is encoded through p: the mix planner tests p >= W.HEAD_P_MIN and ok; a reader below 0.5 stays below
    return p, used.astype(np.float32), (c if STATE["form"] == "RRh" else (vis_c if vis_c is not None else (c_estv if c_estv is not None else c_fed))), ok


def head_values_none(model, feats, frame_ok=None, hist=None):
    n = len(feats)
    return np.zeros(n, np.float32), np.full(n, np.nan, np.float32), np.zeros(n, np.float32), np.ones(n, bool)


def accel_limit(plan, v0, a_max, b_lim, m, dt=DT_S):
    """Retimes each plan along its own path (the lag form of the distance rule): the arc advance a'(i) of step i is clipped to
    [a'(i-1) + max(0, v'(i-1) - b_lim dt) dt, a'(i-1) + (v'(i-1) + a_max dt) dt] with v'(0) = v0; a step that is not clipped keeps the plan's own advance.
    -> (plan', info); info per row: n_up / n_dn = executed steps (i <= m) clipped by a_max / b_lim, req = the largest acceleration the plan asked for on an executed step."""
    p = np.asarray(plan)
    B, N = p.shape[0], p.shape[1]
    p64 = p.astype(np.float64)
    v0 = np.asarray(v0, np.float64)
    ok = np.isfinite(v0)
    aP = np.zeros((B, N + 1))
    PP, EXT = [], []
    for b in range(B):
        P, L, ext = R.plan_path(p64[b])
        aP[b] = L
        PP.append(P)
        EXT.append(ext)
    lag, a_prev, v_prev = np.zeros(B), np.zeros(B), np.where(ok, v0, 0.0)
    a_new = np.zeros((B, N))
    n_up, n_dn, req = np.zeros(B, np.int64), np.zeros(B, np.int64), np.full(B, -np.inf)
    for i in range(1, N + 1):
        a_w = aP[:, i] - lag
        hi = a_prev + (v_prev + a_max * dt) * dt
        lo = a_prev + np.maximum(0.0, v_prev - b_lim * dt) * dt
        a_i = np.where(ok, np.minimum(np.maximum(a_w, lo), hi), aP[:, i])
        if i <= m:
            n_up += (ok & (a_w > hi)).astype(np.int64)
            n_dn += (ok & (a_w < lo)).astype(np.int64)
            req = np.maximum(req, np.where(ok, ((a_w - a_prev) / dt - v_prev) / dt, -np.inf))
        lag = aP[:, i] - a_i
        v_prev = (a_i - a_prev) / dt
        a_prev = a_i
        a_new[:, i - 1] = a_i
    out = p.copy()
    changed = np.zeros(B, bool)
    for b in range(B):
        keep = a_new[b] == aP[b, 1:]                       # an unchanged step is the original knot (bit for bit)
        if keep.all():
            continue
        pts = R.sample_path(PP[b], aP[b], EXT[b], a_new[b])
        pts[keep] = p64[b][keep]
        new = pts.astype(p.dtype)
        changed[b] = bool((new != p[b]).any())
        out[b] = new
    v_exec = np.diff(np.concatenate([np.zeros((B, 1)), a_new[:, :m]], 1), axis=1) / dt     # arc speeds of the executed steps of the limited plan
    return (out if changed.any() else plan), {"n_up": n_up, "n_dn": n_dn, "req": req, "changed": changed, "v0": v0, "v_exec": v_exec, "_a_new": a_new}


def backward_frames(orig, k_prev, knots, m, psi0):
    """the pinned frames function on the executed knots only: its last tangent is then the backward difference."""
    return orig(k_prev, np.asarray(knots, np.float64)[:m + 1], m, psi0)


def causal_frames(orig, k_prev, knots, m, psi0, acc="step"):
    """positions and headings of the pinned frames function; the speed column from executed steps only. s[0] = the step into the current knot (the last executed step
    before this block, or the logged step before the window), s[i] = executed step i; frame 2i-1 (half step) = (s[i-1] + s[i]) / 2, frame 2i (knot) = s[i]."""
    xy, psi, v = orig(k_prev, knots, m, psi0)
    K = np.vstack([np.asarray(k_prev, np.float64)[None], np.asarray(knots, np.float64)[:m + 1]])
    s = np.linalg.norm(np.diff(K, axis=0), axis=-1) / DT_S
    v = np.asarray(v, np.float64).copy()
    for i in range(1, m + 1):
        if acc == "zero":
            v[2 * (i - 1)] = s[i]
        elif acc == "window":
            n_w = min(i, 5)
            v[2 * (i - 1)] = s[i] - 0.5 * DT_S * (s[i] - s[i - n_w]) / (DT_S * n_w)
        else:
            v[2 * (i - 1)] = 0.5 * (s[i - 1] + s[i])
        v[2 * (i - 1) + 1] = s[i]
    return xy, psi, v


def causal_g_frames(orig, k_prev, knots, m, psi0, gain, path_exec=False, force_general=False):
    """the readout causal_g. Speed column: at the knot of executed step i the step speed s[i], at the half step before it s[i] - 0.5 * gain * (s[i] - s[i-1]); s as in
    causal_frames. gain == 1.0 -> the formula of causal, gain == 0.0 -> the formula of causal_z (both unchanged, unless force_general, used by the tool check only);
    other gains -> the general formula. path_exec: positions and headings from the pinned frames function given the executed knots only (backward_frames)."""
    base = orig
    if path_exec:
        def base(kp, kn, mm, p0):
            return backward_frames(orig, kp, kn, mm, p0)
    g = float(gain)
    if not force_general and g == 1.0:
        return causal_frames(base, k_prev, knots, m, psi0, "step")
    if not force_general and g == 0.0:
        return causal_frames(base, k_prev, knots, m, psi0, "zero")
    xy, psi, v = base(k_prev, knots, m, psi0)
    K = np.vstack([np.asarray(k_prev, np.float64)[None], np.asarray(knots, np.float64)[:m + 1]])
    s = np.linalg.norm(np.diff(K, axis=0), axis=-1) / DT_S
    v = np.asarray(v, np.float64).copy()
    for i in range(1, m + 1):
        v[2 * (i - 1)] = s[i] - 0.5 * g * (s[i] - s[i - 1])
        v[2 * (i - 1) + 1] = s[i]
    return xy, psi, v


import math  # noqa: E402

STATE.setdefault("kappa", None)
STATE.setdefault("kb", None)
KB_BINS = (("lt2", 0.0, 2.0), ("2to8", 2.0, 8.0), ("gt8", 8.0, float("inf")))


def wrap_pi(a):
    """angle into [-pi, pi)"""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def kb_new():
    """the counters of the heading bound (readout log fields)"""
    return {"frames_total": 0, "frames_bound_acted": 0, "max_excess_rad": None, "max_abs_diff_where_not_acted": 0.0, "max_abs_diff_xy": 0.0, "max_abs_diff_v": 0.0,
            "by_speed": {n: {"frames_total": 0, "frames_bound_acted": 0} for n, _, _ in KB_BINS}}


def bound_heading(xy, psi, v, xy0, psi0, kappa, rec=None):
    """the heading bound: frame by frame from psi0 and the position xy0 of the current knot; a heading step is at most kappa x the distance between the two frame positions.
    The output heading is psi[f] itself when its step from the previous output heading is within the bound, else the previous output heading turned by exactly the bound towards psi[f].
    Returns the new heading column; counters into rec (a dict of kb_new) when given."""
    xy = np.asarray(xy, np.float64)
    psi = np.asarray(psi, np.float64)
    vv = np.asarray(v, np.float64)
    out = np.empty(len(psi), np.float64)
    prev, pxy = float(psi0), np.asarray(xy0, np.float64)
    for f in range(len(psi)):
        lim = float(kappa) * float(np.hypot(xy[f, 0] - pxy[0], xy[f, 1] - pxy[1]))
        d = wrap_pi(float(psi[f]) - prev)
        acted = not (abs(d) <= lim)
        o = wrap_pi(prev + (1.0 if d > 0 else -1.0) * lim) if acted else float(psi[f])
        if rec is not None:
            ex = abs(wrap_pi(o - prev)) - lim
            rec["max_excess_rad"] = ex if rec["max_excess_rad"] is None else max(rec["max_excess_rad"], ex)
            rec["frames_total"] += 1
            rec["frames_bound_acted"] += int(acted)
            if not acted:
                rec["max_abs_diff_where_not_acted"] = max(rec["max_abs_diff_where_not_acted"], abs(o - float(psi[f])))
            for n, lo, hi in KB_BINS:
                if lo <= float(vv[f]) < hi:
                    rec["by_speed"][n]["frames_total"] += 1
                    rec["by_speed"][n]["frames_bound_acted"] += int(acted)
        out[f] = o
        prev, pxy = o, xy[f]
    return out


def counted_frames(orig, mode, gain=None, path_exec=False, kappa=None):
    """the frames function of the chosen readout; counts its calls and, per call, how far the speed and the one-frame acceleration at the last executed knot lie from
    the last executed step speed and the executed acceleration (s[m] - s[m-1]) / 0.2 s."""
    def frames(k_prev, knots, m, psi0):
        if mode == "causal_g":
            xy, psi, v = causal_g_frames(orig, k_prev, knots, m, psi0, gain, path_exec)
        elif mode in ("causal", "causal_w", "causal_z"):
            xy, psi, v = causal_frames(orig, k_prev, knots, m, psi0, {"causal": "step", "causal_w": "window", "causal_z": "zero"}[mode])
        elif mode == "backward":
            xy, psi, v = backward_frames(orig, k_prev, knots, m, psi0)
        else:
            xy, psi, v = orig(k_prev, knots, m, psi0)
        if kappa is not None:                                         # the heading bound of card ADD67; positions and speeds pass through
            xy_r, v_r = np.array(xy, copy=True), np.array(v, copy=True)
            kb = STATE["kb"]
            psi = bound_heading(xy, psi, v, np.asarray(knots, np.float64)[0], psi0, kappa, kb)
            kb["max_abs_diff_xy"] = max(kb["max_abs_diff_xy"], float(np.max(np.abs(np.asarray(xy) - xy_r))))
            kb["max_abs_diff_v"] = max(kb["max_abs_diff_v"], float(np.max(np.abs(np.asarray(v) - v_r))))
        K = np.vstack([np.asarray(k_prev, np.float64)[None], np.asarray(knots, np.float64)[:m + 1]])
        s = np.linalg.norm(np.diff(K, axis=0), axis=-1) / DT_S
        h = STATE["hf"]
        dv, da = abs(float(v[-1]) - float(s[m])), abs((float(v[-1]) - float(v[-2])) / (DT_S / 2.0) - (float(s[m]) - float(s[m - 1])) / DT_S)
        a0r, n_w = (float(v[-1]) - float(v[-2])) / (DT_S / 2.0), min(m, 5)
        dga = abs(a0r - float(gain) * (float(s[m]) - float(s[m - 1])) / DT_S) if (mode == "causal_g" and gain is not None) else None
        daw = abs(a0r - (float(s[m]) - float(s[m - n_w])) / (DT_S * n_w))
        if np.isfinite(dv) and np.isfinite(da):
            h["calls"] += 1
            h["max_dv"], h["max_da"], h["sum_dv"], h["sum_da"] = max(h["max_dv"], dv), max(h["max_da"], da), h["sum_dv"] + dv, h["sum_da"] + da
            if dga is not None and np.isfinite(dga):
                h["max_dga"] = max(h.get("max_dga", 0.0), dga)
            if np.isfinite(a0r) and np.isfinite(daw):
                h["max_a0"], h["max_daw"], h["sum_a0"] = max(h.get("max_a0", 0.0), abs(a0r)), max(h.get("max_daw", 0.0), daw), h.get("sum_a0", 0.0) + abs(a0r)
        return xy, psi, v
    return frames


def save_rlog(path, form, replan_steps):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    h = STATE["hf"]
    rec = {"readout": STATE["readout"], "form": form, "replan_steps": int(replan_steps), "calls": int(h["calls"]), "max_abs_speed_minus_last_step": h["max_dv"],
           "max_abs_acceleration_minus_executed": h["max_da"], "mean_abs_speed_minus_last_step": h["sum_dv"] / max(h["calls"], 1),
           "mean_abs_acceleration_minus_executed": h["sum_da"] / max(h["calls"], 1), "max_abs_acceleration_read": h.get("max_a0", 0.0),
           "mean_abs_acceleration_read": h.get("sum_a0", 0.0) / max(h["calls"], 1), "max_abs_acceleration_minus_window": h.get("max_daw", 0.0),
           "acc_gain": STATE["gain"], "path_exec": bool(STATE["path_exec"]), "max_abs_acceleration_minus_gain_executed": h.get("max_dga"),
           "kappa_max": STATE["kappa"]}
    if STATE["kb"] is not None:
        rec.update(frames_total=STATE["kb"]["frames_total"], frames_bound_acted=STATE["kb"]["frames_bound_acted"], max_excess_rad=STATE["kb"]["max_excess_rad"],
                   max_abs_diff_where_not_acted=STATE["kb"]["max_abs_diff_where_not_acted"], max_abs_diff_xy=STATE["kb"]["max_abs_diff_xy"], max_abs_diff_v=STATE["kb"]["max_abs_diff_v"],
                   by_speed_bin=STATE["kb"]["by_speed"])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", path, flush=True)


class AccelLimited:
    """The model planner followed by the acceleration limit. Reads the plan and the ego speed at the tick, nothing else; keeps keys_read, aux, heads of the base."""

    def __init__(self, base, tag="m"):
        self.base, self.tag = base, tag

    keys_read = property(lambda self: self.base.keys_read)
    aux = property(lambda self: self.base.aux)
    heads = property(lambda self: self.base.heads)

    def __call__(self, batch):
        plan = self.base(batch)
        v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()
        out, info = accel_limit(plan, v0, STATE["alim"][0], STATE["alim"][1], STATE["m"])
        info["nb"] = np.asarray([len(v0)])                       # one value per call: the rows of this call
        for k, v in info.items():
            if not k.startswith("_"):
                STATE["alog"].setdefault(self.tag + "_" + k, []).append(np.asarray(v))
        return out


def save_alog(path, form, replan_steps):
    if not path or STATE["alim"] is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = {k: np.concatenate(v) for k, v in STATE["alog"].items()}
    with open(path, "wb") as fh:
        np.savez(fh, **cols, a_max=np.asarray(STATE["alim"][0]), b_lim=np.asarray(STATE["alim"][1]), form=np.asarray(form), replan_steps=np.asarray(int(replan_steps)))
    print("wrote", path, flush=True)

# ---------------------------------------------------------------- the later looks of the rule inside the block (card PREREG_V8F_PROOF_ADD62, RULEFAST10K)
MID_FIELDS = tuple(W.LOG_FIELDS) + ("q", "excess", "nchg", "c_read")      # c_read: the reader closing read at the later look (midlook-closing)
STATE["every"] = 10                                    # --rule-every; 10 (or any value at or above the replan interval) = the rule at the tick only
STATE["T"] = None                                      # the pinned tool module (plan_to_world, envelope), set by main()
STATE["mid"] = {"rows": {f: [] for f in MID_FIELDS}, "rd": {k: [] for k in ("p", "g_read", "g_used", "ok")}, "p1": {"ci": [], "t0": [], "plan": []}, "env": {"plans": 0, "hits": 0}}


def _frame_ok(s, e, hist):
    """frame_ok of the last `hist` frames e-hist+1 .. e as the item builder gives it at a tick on frame e: real frames only (idx >= 0) and, for the rows already in the window
    state (idx <= s.j), a finite p10f row (or idx < 1). Rows after s.j are the executed rows a real tick on frame e would have; they are finite by construction."""
    idx = np.arange(e - hist + 1, e + 1)
    past = idx <= int(s.j)
    idp = np.clip(idx, 0, len(s.d["p10f"]) - 1)
    fin = np.isfinite(np.asarray(s.d["p10f"])[idp]).all(axis=1) | (idx < 1)
    return bool(((idx >= 0) & (~past | fin)).all())


def _reader_at(states, rd, e_off):
    """The token reader at the logged frame j + e_off of each window (cache steps e-hist+1 .. e of the window's own clip, the array the batch tokens are gathered from; frames
    at or before e only). -> p [B], g [B] (float64, NaN where there is no complete history in the clip), ok [B] (full history of real frames)."""
    B = len(states)
    p, g, ok = np.full(B, np.nan), np.full(B, np.nan), np.zeros(B, bool)
    rows, toks = [], []
    for i, s in enumerate(states):
        e = int(s.j) + int(e_off)
        if e - (rd.hist - 1) >= 0 and e < len(s.d["feats"]):
            rows.append(i)
            toks.append(np.asarray(s.d["feats"][e - rd.hist + 1:e + 1], np.float32))
            ok[i] = _frame_ok(s, e, rd.hist)
    if rows:
        pp, gg = rd.read(np.stack(toks))
        p[rows], g[rows] = pp, gg
    return p, g, ok


def _reader_closing_at(states, rd, e_off):
    """The reader's closing output (m/s, [B] float64; NaN where there is no complete history in the clip) on the logged tokens of frame j + e_off of each window: the same token window
    (cache steps e-hist+1 .. e of the window's own clip, frames at or before e only) and the same row rule as the gap read of _reader_at; reader_closing applies CLOS_SCALE_MS. It is
    the quantity the tick-level look of form RRh feeds on frame e (head_values_reader). Nothing but tokens is read."""
    B = len(states)
    c = np.full(B, np.nan)
    rows, toks = [], []
    for i, s in enumerate(states):
        e = int(s.j) + int(e_off)
        if e - (rd.hist - 1) >= 0 and e < len(s.d["feats"]):
            rows.append(i)
            toks.append(np.asarray(s.d["feats"][e - rd.hist + 1:e + 1], np.float32))
    if rows:
        c[rows] = reader_closing(rd, np.stack(toks))
    return c


def mid_looks(pl, plan):
    """The rule again, for every later step q = h, 2h, ... (< m), on the part of the plan that is still to be driven. `plan` is the plan after the rule at the tick (the base planner
    followed by X.MixPlanner); `pl` is the ReaderPlanner (states, m, kw). The car is at knot q of the plan built so far (the executor follows the plan exactly in this tier); the
    rule inputs are those a tick on frame j + q takes (same code as X.MixPlanner: label read at j + q, or the reader on the logged tokens of j + q; logged speed and logged arc at
    that step; the car's progress at knot q; the car's speed = the chord speed of the last step before q); the remaining knots are taken relative to knot q; the executed-step
    count is m - q. Steps up to q are never changed; an unchanged later knot stays the original float32 value. Returns the new plan (a copy)."""
    st, m, h = pl.states, int(pl.m), int(STATE["every"])
    T = STATE["T"]
    B = len(st)
    out = np.array(plan, copy=True)
    ci = np.array([s.ci for s in st], np.int64)
    t0 = np.array([s.t0 for s in st], np.int64)
    j = np.array([s.j for s in st], np.int64)
    k = j - t0
    ar = np.arange(B)
    form, mix, label, rd, M = STATE["form"], X.MIX, X.STATE["label"], STATE["reader"], STATE["mid"]
    first = k == 0
    if first.any():                                    # the plan after the rule at the first tick: the reference path of control k4 (second part)
        M["p1"]["ci"].append(ci[first].copy())
        M["p1"]["t0"].append(t0[first].copy())
        M["p1"]["plan"].append(np.array(plan, copy=True)[first])
    g_log = np.stack([np.asarray(s.d["gt"][s.t0], np.float64) for s in st])
    for q in range(h, m, h):
        P = out.astype(np.float64)
        kq, jq = k + q, j + q
        pos = np.empty((B, 1, 2))                      # the car at knot q in the t0 ego frame: the arithmetic of floors_kin.gt_waypoints on the world position of the knot
        for b, s in enumerate(st):
            wxy = T.plan_to_world(P[b, q - 1:q], np.asarray(s.d["pose"][int(s.j)], np.float64))[0]
            p0 = np.asarray(s.track[2 * int(s.t0)] if getattr(s, "frame_pose5", None) is None else s.frame_pose5, np.float64)
            d = wxy - p0[:2]
            c_, s_ = np.cos(-p0[2]), np.sin(-p0[2])
            pos[b, 0] = (d[0] * c_ - d[1] * s_, d[0] * s_ + d[1] * c_)
        a_m = R.progress(g_log, pos)[:, 0]             # k + q > 0 always
        a_l = R.logged_arc(g_log)[ar, np.clip(kq - 1, 0, 29)]
        v_l = np.array([s.logged_v[int(x)] if int(x) < len(s.logged_v) else np.nan for s, x in zip(st, jq)], np.float64)
        prev = P[:, q - 2] if q >= 2 else np.zeros((B, 2))
        v0 = np.linalg.norm(P[:, q - 1] - prev, axis=-1) / DT_S
        rdl = [label.read(int(c), int(x)) for c, x in zip(ci, jq)]
        lp = np.array([r[0] for r in rdl], bool)
        lg = np.array([r[1] for r in rdl], np.float64)
        lc = np.array([r[2] for r in rdl], np.float64)
        if form in ("RR", "RRc", "RRh", "RRe"):
            p, g, okr = _reader_at(st, rd, q)
            p_rd = p
            if STATE.get("pres") is not None:            # study preswrap: presence at the later look is tested on p2
                p = pres_look(st, q, p)
            c_read = _reader_closing_at(st, rd, q)       # logged in every reader form; fed only in RRh
            _, used = feed(form, p, g, okr, STATE["xs"], STATE["ys"])
            vis_hc = None
            if STATE.get("vis") is not None:             # the variant of card ADD102 at the later look
                used, vis_hc = vis_look(st, rd, q, p, g, used, okr)
            c_est, est_ok = _est_at(st, rd, q) if form == "RRe" else (None, None)       # form RRe: the estimate at frame j + q
            hg, hc = used.astype(np.float32).astype(np.float64), (c_read if form == "RRh" else (est_fed(c_est, est_ok) if form == "RRe" else (np.zeros(B, np.float64) if vis_hc is None else vis_hc.astype(np.float64))))
            hp = (p >= W.HEAD_P_MIN) & okr
            p_src = p
            for kk, vv in (("p", p_rd), ("g_read", g), ("g_used", used), ("ok", okr)):
                M["rd"][kk].append(np.asarray(vv))
            if STATE.get("pres") is not None:
                M["rd"].setdefault("p_used", []).append(np.asarray(p))
        else:                                          # head_values_none of the wrapper: the label forms read no token
            p_src, hg, hc = np.zeros(B, np.float64), np.full(B, np.nan), np.zeros(B, np.float64)
            c_read = np.full(B, np.nan)
            hp = np.zeros(B, bool)
        present = lp.copy() if mix["presence"] == "label" else hp.copy()
        lab_g, lab_c, head_g, head_c = np.isfinite(lg), np.isfinite(lc), np.isfinite(hg), np.isfinite(hc)
        if mix["gap"] == "label":
            g_src = np.where(present & ~lab_g & head_g, hg, lg)
        else:
            g_src = np.where(present & ~head_g & lab_g, lg, hg)
        if mix["closing"] == "label":
            c_src = np.where(present & ~lab_c & head_c, hc, lc)
        elif mix["closing"] == "head":
            c_src = np.where(present & ~head_c & lab_c, lc, hc)
        else:
            c_src = np.zeros(B, np.float64)
        with np.errstate(invalid="ignore"):
            src_bad = present & ~(np.isfinite(g_src) & np.isfinite(c_src) & np.isfinite(v_l))
            present = present & ~src_bad
            v_L = np.maximum(0.0, v_l + c_src)
            g0 = g_src + a_l - a_m
        rel = P[:, q:] - P[:, q - 1:q]                 # the remaining knots relative to knot q (the rule works on arc lengths along the path: a translation changes nothing)
        res = R.distance_rule(rel, v0, present, g_src, g0, v_L, m=m - q, **pl.kw)
        excess, nchg = np.zeros(B), np.zeros(B, np.int64)
        if res.changed.any():
            new_rel = np.asarray(res.plan, np.float64)
            ch = np.nonzero(res.changed)[0]
            prog = R.progress(rel[ch], new_rel[ch])    # arc coordinate of the new knots on the path before the look (float64, before the committed plan is rounded to float32)
            for n_, b in enumerate(ch):
                excess[b] = float(np.max(prog[n_] - R.plan_path(rel[b])[1][1:]))
                same = (new_rel[b] == rel[b]).all(axis=-1)
                nchg[b] = int((~same).sum())
                out[b, q:] = np.where(same[:, None], out[b, q:], (new_rel[b] + P[b, q - 1]).astype(out.dtype))
        row = dict(ci=ci, t0=t0, j=jq, k=kq, tick=k // max(m, 1), active=res.active, present=present, src_bad=src_bad, sat=res.n_sat, sat_exec=res.n_sat_exec, lift=res.n_lift,
                   changed=res.changed, g_src=g_src, c_src=c_src, c_read=c_read, p_src=p_src, v0=v0, v_l=v_l, v_L=v_L, a_l=a_l, a_m=a_m, g0=g0, q=np.full(B, q, np.int64), excess=excess, nchg=nchg)
        if form == "RRe":
            row["c_est"], row["est_ok"] = c_est, est_ok
        for f in MID_FIELDS + (MID_EST if form == "RRe" else ()):
            M["rows"][f].append(np.asarray(row[f]))
    for b in range(B):                                 # how often the pinned tool's step envelope would reshape the committed plan (reported, not acted on)
        M["env"]["plans"] += 1
        M["env"]["hits"] += int(T.envelope(out[b].astype(np.float64))[1])
    return out


def save_mid(rule_log, meta, m):
    """Writes ticks/rule_log_mid_<run>.npz (one row per window, tick and later look: the columns of the tick log plus q, excess, nchg), ticks/rule_log_mid_<run>_rd.npz (reader
    reads of the later looks, reader forms) and ticks/p1_first_<run>.npz (the plan after the rule at the first tick). Nothing is written when the rule acts at the tick only."""
    if int(STATE["every"]) >= int(m):
        return
    d, base = os.path.split(os.path.abspath(rule_log))
    assert base.startswith("rule_log_") and base.endswith(".npz"), base
    run = base[len("rule_log_"):]
    M = STATE["mid"]
    cols = {f: (np.concatenate(M["rows"][f]) if M["rows"][f] else np.zeros(0)) for f in MID_FIELDS + (MID_EST if STATE["form"] == "RRe" else ())}
    extra = {"rule_every": np.asarray(int(STATE["every"])), "mid_envelope_plans": np.asarray(M["env"]["plans"]), "mid_envelope_hits": np.asarray(M["env"]["hits"])}
    mid = os.path.join(d, "rule_log_mid_" + run)
    with open(mid, "wb") as fh:
        np.savez(fh, **cols, **{k: np.asarray(v) for k, v in meta.items()}, **extra)
    print("wrote", mid, flush=True)
    if M["rd"]["p"]:
        side = mid[:-4] + "_rd.npz"
        with open(side, "wb") as fh:
            np.savez(fh, **{k: np.concatenate(v) for k, v in M["rd"].items()}, form=np.asarray(STATE["form"]))
        print("wrote", side, flush=True)
    p1 = os.path.join(d, "p1_first_" + run)
    with open(p1, "wb") as fh:
        np.savez(fh, ci=np.concatenate(M["p1"]["ci"]), t0=np.concatenate(M["p1"]["t0"]), plan=np.concatenate(M["p1"]["plan"]))
    print("wrote", p1, flush=True)



class ReaderPlanner(X.MixPlanner):
    """MixPlanner without the lead-head requirement: the reader uses no model weight."""

    def __init__(self, base, model, mode, m, fk, label=None, log=None, floor="card"):
        _RULE_PLANNER.__init__(self, base, model, mode, m, fk, label=label, log=log, floor=floor)
        if STATE.get("preroll", 0) > 0:
            self.fk = FrameFK(self.fk, self)
        if self.source == "mix":
            self.label = X.STATE["label"]
        if STATE["alim"] is not None:
            self.base = AccelLimited(self.base)        # the lead rule then acts on the limited plan

    def __call__(self, batch):
        if STATE["form"] == "RRd":
            if self.states is None:
                raise RuntimeError("ReaderPlanner not bound to the states of this batch")
            STATE["prev"] = prev_read(self.states, STATE["reader"], STATE["k"])
            STATE["zero"] = prev_read(self.states, STATE["reader"], 0)
        pre = STATE.get("preroll", 0) > 0
        if pre:
            STATE["g0_corr"] = a_m0_first_tick(R, self.fk.gt_waypoints, self.states or ())     # R1: consumed by the next R.distance_rule call (the tick-level one) only
        if STATE["form"] == "RRe":
            STATE["est_states"] = self.states               # the estimate reads the logged tokens of the earlier frames of these windows
        if STATE.get("vis") is not None or STATE.get("pres") is not None:
            STATE["vis_states"] = self.states               # the variant reads the logged tokens of the earlier frames of these windows
        plan = X.MixPlanner.__call__(self, batch)
        if pre:
            STATE["g0_corr"] = None
        if int(STATE["every"]) < int(self.m):
            plan = mid_looks(self, plan)
        return plan


# ==== PREROLL PURE BEGIN (study longroll) =============================================================================================================
import copy  # noqa: E402


def to_frame(xy, pose):
    """world xy [N, 2] -> the frame of pose = (x, y, yaw, ...): the arithmetic of floors_kin.gt_waypoints (rotation by -yaw), operation for operation."""
    xy = np.asarray(xy, np.float64)
    pose = np.asarray(pose, np.float64)
    d = xy - pose[None, :2]
    c, s = np.cos(-pose[2]), np.sin(-pose[2])
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], axis=-1)


def exec_logged_frame(gtw, track, t0, logged_pose, h):
    """The executed knots t0+1 .. t0+h of the own native track (row 2*j per cache step j) in the frame of the LOGGED pose at t0. gtw = floors_kin.gt_waypoints: it is given the 5 Hz
    rows with row t0 replaced by the logged pose, so the convention is the pinned exec_in_ego's own."""
    p5 = np.array(np.asarray(track, np.float64)[0::2], copy=True)
    p5[int(t0)] = np.asarray(logged_pose, np.float64)
    return gtw(p5, np.array([int(t0)]), np.arange(1, int(h) + 1))[0]


def map_plan0(p2w, plan0, own_poses, logged_poses):
    """plan0 [B, N, 2] in the frame of each window's own pose at t0 -> the frame of its logged pose at t0 (p2w = the pinned plan_to_world)."""
    plan0 = np.asarray(plan0)
    out = np.empty(plan0.shape, np.float64)
    for b in range(len(plan0)):
        out[b] = to_frame(p2w(np.asarray(plan0[b], np.float64), own_poses[b]), logged_poses[b])
    return out


def pre_offsets(own_pose, logged_pose, own_v=None):
    """position offset norm (m), absolute heading offset (deg, wrapped), own speed minus logged speed (m/s)."""
    own_pose, logged_pose = np.asarray(own_pose, np.float64), np.asarray(logged_pose, np.float64)
    dp = float(np.hypot(*(own_pose[:2] - logged_pose[:2])))
    dh = (own_pose[2] - logged_pose[2] + math.pi) % (2.0 * math.pi) - math.pi
    v = float(own_pose[3] if own_v is None else own_v)
    return dp, float(abs(math.degrees(dh))), v - float(logged_pose[3])


def stat4(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return None
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(x.max()), "n": int(x.size)}


def merge_counts(items):
    """sum of the counters of several dicts (a key starting with max takes the maximum; None is skipped; nested dicts and lists of numbers are merged the same way)."""
    out = None
    for it in items:
        out = copy.deepcopy(it) if out is None else _merge2(out, it)
    return out


def _merge2(a, b, key=""):
    if isinstance(a, dict) and isinstance(b, dict):
        r = dict(a)
        for k, v in b.items():
            r[k] = _merge2(a[k], v, str(k)) if k in a else copy.deepcopy(v)
        return r
    if isinstance(a, list) and isinstance(b, list):
        if len(a) == len(b) and all(isinstance(x, (int, float, list)) for x in a):
            return [_merge2(x, y, key) for x, y in zip(a, b)]
        return a + b
    if a is None:
        return copy.deepcopy(b)
    if b is None:
        return a
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return max(a, b) if key.startswith("max") else a + b
    return a


def _zero(o):
    """the counters of o at their start: numbers 0, nested dicts and lists of numbers zeroed, row stores (lists of arrays) emptied, None kept."""
    if isinstance(o, dict):
        return {k: _zero(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_zero(x) for x in o] if o and all(isinstance(x, (int, float, list)) for x in o) else []
    if isinstance(o, bool):
        return o
    if isinstance(o, int):
        return 0
    if isinstance(o, float):
        return 0.0
    return o


def _restore(dst, src):
    """put the content of src into dst in place (dst keeps its identity, and so do its dict / list children where the types agree)."""
    if isinstance(dst, dict):
        for k in [k for k in dst if k not in src]:
            del dst[k]
        for k, v in src.items():
            if k in dst and isinstance(v, (dict, list)) and type(dst[k]) is type(v):
                _restore(dst[k], v)
            else:
                dst[k] = copy.deepcopy(v)
    else:
        dst[:] = copy.deepcopy(src)


class FrameFK:
    """Stands in for the tool's fk inside the lead rule's planner. gt_waypoints(poses, t0, frames) for the 5 Hz own track of a window in its main phase (frame_pose5 set) reads the
    own track in the frame of the LOGGED pose at t0; every other call (pre-roll, other arrays) passes through unchanged."""

    def __init__(self, fk, planner):
        self._fk, self._pl = fk, planner

    def __getattr__(self, name):
        return getattr(self._fk, name)

    def gt_waypoints(self, poses, t0, frames):
        for s in (self._pl.states or ()):
            fp = getattr(s, "frame_pose5", None)
            if fp is not None and getattr(poses, "base", None) is s.track and len(np.asarray(t0)) == 1 and int(np.asarray(t0)[0]) == int(s.t0):
                p5 = np.array(poses, copy=True)
                p5[int(s.t0)] = fp
                return self._fk.gt_waypoints(p5, t0, frames)
        return self._fk.gt_waypoints(poses, t0, frames)


SEG = 30                                           # the rule's arc tables are those of the 30-step logged path from st.t0: one pre-roll segment = one table


def a_m0_first_tick(R_, gtw, states):
    """R1: a_m0 [B] = the arc coordinate, on the logged 30-step path of the window (d["gt"][t0], the frame of the logged pose at t0), of the own position at t0, for every state with j == t0
    and a logged frame pose; exactly 0.0 for every other state and for a state whose own x, y equal the logged x, y. R_ = the rule module (progress), gtw = floors_kin.gt_waypoints."""
    out = np.zeros(len(states), np.float64)
    keep, pos, glog = [], [], []
    for b, s in enumerate(states):
        fp = getattr(s, "frame_pose5", None)
        if fp is None or int(s.j) != int(s.t0):
            continue
        fp = np.asarray(fp, np.float64)
        own = np.asarray(s.track[2 * int(s.t0)], np.float64)
        if own[0] == fp[0] and own[1] == fp[1]:
            continue
        pos.append(gtw(np.stack([fp, own]), np.array([0]), np.array([1]))[0, 0])
        glog.append(np.asarray(s.d["gt"][int(s.t0)], np.float64))
        keep.append(b)
    if keep:
        out[keep] = R_.progress(np.stack(glog), np.stack(pos)[:, None, :])[:, 0]
    return out


def make_g0_shim(orig, state):
    """R1: the function that replaces R.distance_rule while P is above 0. state["g0_corr"] ([B] or None) is subtracted from g0 in the NEXT call only, then cleared."""
    def distance_rule(plan, v0, present, g_src, g0, v_L, *args, **kw):
        c = state.get("g0_corr")
        if c is not None:
            state["g0_corr"] = None
            c = np.asarray(c, np.float64)
            g0 = np.asarray(g0, np.float64)
            if c.shape != g0.shape:
                raise RuntimeError("g0 adjustment shape %s differs from g0 %s" % (c.shape, g0.shape))
            g0 = g0 - c
            st = state.setdefault("g0_stats", {"calls": 0, "nonzero": 0, "max_abs": 0.0})
            st["calls"] += 1
            st["nonzero"] += int((c != 0.0).sum())
            st["max_abs"] = max(st["max_abs"], float(np.abs(c).max()) if c.size else 0.0)
        return orig(plan, v0, present, g_src, g0, v_L, *args, **kw)
    return distance_rule


class HookFrame:
    """R3: the hook the main-phase rollout gets. Runs the caller's hook (the pinned TickRecorder), then puts the logged pose at t0 into the pose_w0 rows that the call just appended."""

    def __init__(self, hook, runner):
        self._h, self._r = hook, runner

    def __call__(self, n, batch, plan, states):
        out = self._h(n, batch, plan, states)
        rows = getattr(self._h, "rows", None)
        B = len(states)
        if isinstance(rows, dict) and "pose_w0" in rows and len(rows["pose_w0"]) >= B:
            for b, st in enumerate(states):
                fp = getattr(st, "frame_pose5", None)
                if fp is None:
                    continue
                new = np.array(np.asarray(fp, np.float64)[:3], copy=True)
                shift = float(np.hypot(*(new[:2] - np.asarray(rows["pose_w0"][-B + b], np.float64)[:2])))
                rows["pose_w0"][-B + b] = new
                self._r.hook_rows += 1
                self._r.hook_max_shift = max(self._r.hook_max_shift, shift)
        return out


class PrerollRun:
    """One call of the pinned rollout becomes: P / 30 pre-roll segments (each H = 30, hook None, a re-base after each) + the unmodified main rollout (hook through HookFrame). T = the pinned
    module (H, plan_to_world, fk), R = the rule module. tracked() returns [(name, obj)] of the mutable stores that must hold main-phase data only; the content of each segment is kept in
    self.posts (post_seg = the segment index of each entry)."""

    def __init__(self, T, P, pre_exec, tracked, R_):
        self.T, self.P, self.pre_exec, self.tracked, self.R = T, int(P), pre_exec, tracked, R_
        if self.P % SEG:
            raise ValueError("--preroll-steps %d must be a multiple of %d" % (self.P, SEG))
        self.calls, self.posts, self.post_seg = [], [], []
        self.off = {"kind": [], "ci": [], "t0": [], "dpos": [], "dpsi_deg": [], "dv": [], "a_m0": []}
        self.seg = {"kind": [], "ci": [], "t0": [], "index": [], "dpos": [], "dpsi_deg": [], "dv": [], "a_m0": []}
        self.clipped_pre = {"model": 0, "floor": 0}
        self.hook_rows, self.hook_max_shift = 0, 0.0

    def rebase(self, kind, states, t_news, index, final):
        """advance st.t0 of every state to its new step (the state must stand there), set the logged frame pose, read the offsets and a_m0."""
        T = self.T
        for st, t_new in zip(states, t_news):
            t_new = int(t_new)
            if int(st.j) != t_new:
                raise RuntimeError("pre-roll segment ended at step %d, not at %d" % (int(st.j), t_new))
            g = st.d["gt"] if "gt" in st.d else None
            if g is not None and not np.isfinite(np.asarray(g[t_new], np.float64)).all():
                raise RuntimeError("no finite logged 30-step path at re-base step %d of window (%d, %d)" % (t_new, int(st.ci), int(st.lr_t0)))
            self.clipped_pre[kind] += int(st.clipped)
            st.clipped = 0
            st.t0 = t_new
            st.frame_pose5 = np.asarray(st.lr_track[2 * t_new], np.float64).copy()
        am = a_m0_first_tick(self.R, T.fk.gt_waypoints, states)
        for st, t_new, a in zip(states, t_news, am):
            t_new = int(t_new)
            op = np.asarray(st.d["pose"][t_new], np.float64)
            lp = np.asarray(st.frame_pose5, np.float64)
            dp, dh, dv = pre_offsets(op, lp, own_v=float(st.d["v"][t_new]))
            if final:
                rec, extra = self.off, (("t0", t_new),)
            else:
                rec, extra = self.seg, (("t0", t_new), ("index", int(index)))
            for k, v in (("kind", kind), ("ci", int(st.ci)), ("dpos", dp), ("dpsi_deg", dh), ("dv", dv), ("a_m0", float(a))) + extra:
                rec[k].append(v)

    def __call__(self, orig, planner, states, args, kw):
        T, P = self.T, self.P
        m = int(kw.get("m", 1))
        if m <= 0 or P % m or SEG % m:
            raise ValueError("--preroll-steps %d and 30 must be multiples of --replan-steps %d" % (P, m))
        if int(T.H) != SEG:
            raise RuntimeError("the pinned H is %d, not %d: the segments assume 30" % (int(T.H), SEG))
        kind = "model" if hasattr(planner, "keys_read") else "floor"
        snap = [(n, copy.deepcopy(o)) for n, o in self.tracked()]
        kw_pre = dict(kw)
        kw_pre["hook"] = None
        kw_pre["executor"] = self.pre_exec
        if self.pre_exec == "logged":
            kw_pre["logged_tracks"] = {st.ci: st.lr_track for st in states}
        old_h = T.H
        nseg = P // SEG
        try:
            for sg in range(nseg):
                for n, o in self.tracked():
                    _restore(o, _zero(o))                 # each segment counts into zeroed stores; the main phase gets its own content back below
                T.H = SEG
                try:
                    orig(planner, states, *args, **kw_pre)
                finally:
                    T.H = old_h
                    post = {n: copy.deepcopy(o) for n, o in self.tracked()}
                self.posts.append((kind, post))
                self.post_seg.append(sg)
                self.calls.append({"kind": kind, "windows": len(states), "ticks_per_window": SEG // m, "segment": sg})
                self.rebase(kind, states, [int(st.lr_t0) - P + SEG * (sg + 1) for st in states], sg + 1, sg + 1 == nseg)
        finally:
            for (n, o), (_, s0) in zip(self.tracked(), snap):
                _restore(o, s0)
        own, lg = [], []
        for st in states:
            t0 = int(st.lr_t0)
            lp = np.asarray(st.frame_pose5, np.float64)
            st.executed_ego = (lambda st=st, lp=lp: exec_logged_frame(T.fk.gt_waypoints, st.track, st.lr_t0, lp, T.H))
            own.append(np.asarray(st.d["pose"][t0], np.float64))
            lg.append(lp)
        kw_main = dict(kw)
        if kw.get("hook") is not None:
            kw_main["hook"] = HookFrame(kw["hook"], self)
        plan0, exe = orig(planner, states, *args, **kw_main)
        return map_plan0(T.plan_to_world, plan0, own, lg).astype(np.asarray(plan0).dtype), exe

    def offset_arrays(self):
        o, s = self.off, self.seg
        f8, i8 = np.float64, np.int64
        return dict(kind=np.asarray(o["kind"]), ci=np.asarray(o["ci"], i8), t0=np.asarray(o["t0"], i8), dpos=np.asarray(o["dpos"], f8), dpsi_deg=np.asarray(o["dpsi_deg"], f8),
                    dv=np.asarray(o["dv"], f8), a_m0_first_tick=np.asarray(o["a_m0"], f8),
                    seg_kind=np.asarray(s["kind"]), seg_ci=np.asarray(s["ci"], i8), seg_t0=np.asarray(s["t0"], i8), seg_index=np.asarray(s["index"], i8), seg_dpos=np.asarray(s["dpos"], f8),
                    seg_dpsi_deg=np.asarray(s["dpsi_deg"], f8), seg_dv=np.asarray(s["dv"], f8), seg_a_m0_first_tick=np.asarray(s["a_m0"], f8))

    def summary(self):
        kinds = sorted(set(self.off["kind"]))
        km = np.asarray(self.off["kind"])
        am0 = np.asarray(self.off["a_m0"], np.float64)
        per, am = {}, {}
        for k in kinds:
            sel = km == k
            per[k] = {f: stat4(np.asarray(self.off[f], np.float64)[sel]) for f in ("dpos", "dpsi_deg", "dv")}
            am[k] = {"main_first_tick_abs": stat4(np.abs(am0[sel])), "main_exact_zero": int((am0[sel] == 0.0).sum()), "main_max_abs": float(np.abs(am0[sel]).max()) if sel.any() else 0.0}
        sk = np.asarray(self.seg["kind"])
        sa = np.asarray(self.seg["a_m0"], np.float64)
        for k in sorted(set(self.seg["kind"])):
            am.setdefault(k, {})["segments_max_abs"] = float(np.abs(sa[sk == k]).max())
        first = [c for c in self.calls if c["segment"] == 0]
        return {"windows": {k: int(sum(c["windows"] for c in first if c["kind"] == k)) for k in kinds},
                "ticks_pre_rolled": {k: int(sum(c["windows"] * c["ticks_per_window"] for c in self.calls if c["kind"] == k)) for k in kinds},
                "segments_per_window": self.P // SEG, "rollout_calls": len(self.calls), "offsets_at_t0": per, "a_m0": am, "clipped_windows_in_preroll": dict(self.clipped_pre),
                "tick_recorder_pose_w0_mapped": {"rows": int(self.hook_rows), "max_shift_m": float(self.hook_max_shift)}}


def build_rollout(orig, runner):
    """the rollout the wrapper installs; P = 0 (runner None) installs nothing: the original function itself."""
    if runner is None:
        return orig

    def rollout(planner, states, *args, **kw):
        return runner(orig, planner, states, args, kw)
    return rollout
# ==== PREROLL PURE END ===============================================================================================================================


def _tracked():
    out = [("hf", STATE["hf"]), ("cols", STATE["cols"]), ("mid", STATE["mid"]), ("alog", STATE["alog"]), ("mix_count", X.COUNT), ("causal_stats", STATE["T"].CAUSAL_STATS)]
    if STATE.get("kb") is not None:
        out.append(("kb", STATE["kb"]))
    if STATE.get("rulelog") is not None:
        out.append(("rule_rows", STATE["rulelog"].rows))
    return out


def install_preroll(T, mine, a):
    P = int(mine.preroll_steps)
    STATE["preroll"] = P
    STATE["runner"] = None
    if P < 0:
        raise SystemExit("REFUSED: --preroll-steps must be 0 or more")
    if P == 0:
        if mine.preroll_exec != "plan":
            raise SystemExit("REFUSED: --preroll-exec goes with --preroll-steps above 0")
        return
    if P % int(a.replan_steps):
        raise SystemExit("REFUSED: --preroll-steps %d is not a multiple of --replan-steps %d" % (P, int(a.replan_steps)))
    if P % SEG or SEG % int(a.replan_steps):
        raise SystemExit("REFUSED: --preroll-steps %d is not a multiple of %d, or --replan-steps %d does not divide %d (one pre-roll segment = the 30-step logged path of the rule's arc tables)" % (P, SEG, int(a.replan_steps), SEG))
    runner = PrerollRun(T, P, mine.preroll_exec, _tracked, R)
    orig_ms, orig_ro = T.make_states, T.rollout

    def make_states(store, tracks, J, wins, nav_mode):
        wp = []
        for w in wins:
            if int(w[1]) - P < 3:
                raise SystemExit("REFUSED: window t0 %d with --preroll-steps %d starts the pre-roll at %d, below 3 (restrict the grid with make_longroll_grid.py)" % (int(w[1]), P, int(w[1]) - P))
            wp.append((w[0], int(w[1]) - P) + tuple(w[2:]))
        out = orig_ms(store, tracks, J, wp, nav_mode)
        for st, w in zip(out, wins):
            st.lr_t0 = int(w[1])
            st.lr_track = np.asarray(tracks[str(J["clip"][st.ci])], np.float64)
        return out

    T.make_states = make_states
    T.rollout = build_rollout(orig_ro, runner)
    STATE["pre_shim"] = (R, R.distance_rule)
    R.distance_rule = make_g0_shim(R.distance_rule, STATE)
    STATE["runner"], STATE["pre_orig"], STATE["pre_mine"], STATE["pre_a"] = runner, (T, orig_ms, orig_ro), mine, a


def _log_name(mine, a):
    rl, rd = mine.rule_log, mine.readout_log
    if rl:
        base = os.path.basename(rl)
        return os.path.dirname(os.path.abspath(rl)), (base[len("rule_log_"):-4] if base.startswith("rule_log_") and base.endswith(".npz") else os.path.splitext(base)[0])
    if rd:
        base = os.path.splitext(os.path.basename(rd))[0]
        return os.path.dirname(os.path.abspath(rd)), (base[len("readout_log_"):] if base.startswith("readout_log_") else base)
    return os.path.abspath(a.dump_ticks or a.out_dir), a.tag


def finish_preroll():
    runner = STATE.get("runner")
    if runner is None:
        return
    T, orig_ms, orig_ro = STATE["pre_orig"]
    T.make_states, T.rollout = orig_ms, orig_ro
    Rm, orig_dr = STATE["pre_shim"]
    Rm.distance_rule = orig_dr
    mine, a = STATE["pre_mine"], STATE["pre_a"]
    folder, run = _log_name(mine, a)
    os.makedirs(folder, exist_ok=True)
    if mine.rule_log and STATE.get("rulelog") is not None:
        rows = {k: [] for k in W.LOG_FIELDS}
        segcol = []
        for (kind, post), sg in zip(runner.posts, runner.post_seg):
            if kind == "model" and "rule_rows" in post:
                for k in W.LOG_FIELDS:
                    rows[k] += post["rule_rows"][k]
                segcol.append(np.full(int(sum(len(x) for x in post["rule_rows"]["ci"])), sg, np.int64))
        meta = {}
        if os.path.isfile(mine.rule_log):
            with np.load(mine.rule_log, allow_pickle=False) as z:
                meta = {k: z[k] for k in z.files if k not in W.LOG_FIELDS}
        meta["preroll_steps"] = np.asarray(runner.P)
        cols = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rows.items()}
        cols["pre_segment"] = np.concatenate(segcol) if segcol else np.zeros(0, np.int64)
        if STATE["form"] == "RRe":                     # the pre-roll rows get their own c_est / est_ok (the main-phase arrays of the rule log file are not theirs)
            meta.pop("c_est", None)
            meta.pop("est_ok", None)
            for kk in MID_EST:
                parts = [x for kind, post in runner.posts if kind == "model" for x in post["cols"][kk]]
                cols[kk] = np.concatenate([np.asarray(x) for x in parts]) if parts else np.zeros(0)
                assert len(cols[kk]) == len(cols["ci"]), (kk, len(cols[kk]), len(cols["ci"]))
        pre = os.path.join(folder, "rule_log_pre_" + run + ".npz")
        with open(pre, "wb") as fh:
            np.savez(fh, **cols, **meta)
        print("wrote", pre, flush=True)
    off = os.path.join(folder, "preroll_offsets_" + run + ".npz")
    with open(off, "wb") as fh:
        np.savez(fh, **runner.offset_arrays())
    print("wrote", off, flush=True)
    counters = {}
    for k in ("model", "floor"):
        items = [p for kd, p in runner.posts if kd == k]
        if items:
            counters[k] = {n: merge_counts([p[n] for p in items if n in p and n != "rule_rows" and n != "alog" and n != "mid"]) for n in sorted({n for p in items for n in p})
                           if n not in ("rule_rows", "alog", "mid", "cols")}
            mids = [p["mid"] for p in items if "mid" in p]
            counters[k]["mid_rows"] = int(sum(sum(len(x) for x in mm["rows"]["ci"]) for mm in mids)) if mids else 0
    rec = {"preroll_steps": runner.P, "preroll_exec": runner.pre_exec, "replan_steps": int(a.replan_steps), "nav": a.nav,
           "nav_note": "dead-reckoned nav held from the pre-roll start t0 - P, route progress from own motion",
           "form": mine.form, "readout": mine.readout, "kappa_max": mine.kappa_max, "rule_every": int(mine.rule_every), "tag": a.tag,
           "readout_counters_pre_roll": counters, "offsets_file": os.path.basename(off)}
    rec.update(runner.summary())
    rec["a_m0_used_in_rule"] = STATE.get("g0_stats") or {}
    rec["a_m0_note"] = "rule-log columns a_m and g0 of first-tick rows show the definition values; the rule used g0 - a_m0 (a_m0_first_tick in the offsets file)"
    path = os.path.join(folder, "preroll_log_" + run + ".json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", path, flush=True)


def arg_value(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


VIS_STRIDE = 1 << 20
VIS_GAP_MAX_M, VIS_STEP_MAX_M, VIS_RUN_MAX, VIS_RUN_MIN = 60.0, 3.0, 5, 3        # the constants of LabelSource (distrule/v8f_t1_eval_rule.py:37-39)
STATE["vis"] = None                                    # set by main() when --vis-gap-file or --vis-closing is given
STATE["vis_states"] = None                             # the states of the batch of the tick (ReaderPlanner.__call__)
STATE["vis_counts"] = {"present": 0, "from_head": 0, "no_head_output": 0, "key_missing": 0, "closing_nonzero": 0, "closing_rows": 0}


def load_vis(path, closing):
    """-> the state of the variant: the gap table of --vis-gap-file (arrays clip_index, frame, gap, has_out; one row per (clip, frame), no key twice) and the closing switch. Refuses a missing or malformed file."""
    tab = None
    if path:
        if not os.path.isfile(path):
            raise SystemExit("REFUSED: --vis-gap-file not found: %s" % path)
        with np.load(path, allow_pickle=False) as z:
            for k in ("clip_index", "frame", "gap", "has_out"):
                if k not in z.files:
                    raise SystemExit("REFUSED: --vis-gap-file lacks the array %s" % k)
            ci, fr = np.asarray(z["clip_index"], np.int64), np.asarray(z["frame"], np.int64)
            gap, has = np.asarray(z["gap"], np.float64), np.asarray(z["has_out"], bool)
        if not (ci.ndim == fr.ndim == gap.ndim == has.ndim == 1 and len(ci) == len(fr) == len(gap) == len(has) and len(ci) > 0):
            raise SystemExit("REFUSED: --vis-gap-file arrays are not one-dimensional of one length")
        if (ci < 0).any() or (fr < 0).any() or (fr >= VIS_STRIDE).any():
            raise SystemExit("REFUSED: --vis-gap-file has a negative clip index or a frame outside [0, %d)" % VIS_STRIDE)
        if (has & ~np.isfinite(gap)).any():
            raise SystemExit("REFUSED: --vis-gap-file marks an output with a gap that is not finite")
        key = ci * VIS_STRIDE + fr
        o = np.argsort(key, kind="stable")
        key = key[o]
        if (np.diff(key) == 0).any():
            raise SystemExit("REFUSED: --vis-gap-file has two rows with the same clip index and frame")
        tab = {"key": key, "gap": gap[o], "has": has[o]}
    return {"tab": tab, "close": bool(closing)}


def vis_lookup(tab, ci, fr):
    """-> (found [B] bool, has [B] bool, gap [B] float64, NaN where there is no output) at (clip index, frame)."""
    ci, fr = np.asarray(ci, np.int64), np.asarray(fr, np.int64)
    key = ci * VIS_STRIDE + fr
    pos = np.searchsorted(tab["key"], key)
    pc = np.minimum(pos, len(tab["key"]) - 1)
    found = (pos < len(tab["key"])) & (tab["key"][pc] == key) & (fr >= 0) & (fr < VIS_STRIDE)
    has = found & tab["has"][pc]
    return found, has, np.where(has, tab["gap"][pc], np.nan)


def vis_replace(tab, ci, fr, present, used, count=None):
    """the gap the rule receives: `used` (the reader's own gap after its map and minimum) where the reader marks no lead or the file has no output, else the file's gap as dumped. -> float64 copy."""
    used = np.array(used, np.float64, copy=True)
    if tab is None:
        if count is not None:
            count["present"] += int(np.asarray(present, bool).sum())
        return used
    found, has, g = vis_lookup(tab, ci, fr)
    present = np.asarray(present, bool)
    sel = present & has
    used[sel] = g[sel]
    if count is not None:
        count["present"] += int(present.sum())
        count["from_head"] += int(sel.sum())
        count["no_head_output"] += int((present & found & ~has).sum())
        count["key_missing"] += int((present & ~found).sum())
    return used


def vis_ok(present, gap):
    """a frame is in the run when the reader marks a lead on it and the gap the rule receives is finite and in 0 to 60 m (LabelSource._ok)."""
    gap = np.asarray(gap, np.float64)
    with np.errstate(invalid="ignore"):
        return np.asarray(present, bool) & np.isfinite(gap) & (gap >= 0.0) & (gap <= VIS_GAP_MAX_M)


def vis_run_closing(g, ok, dt=DT_S):
    """g, ok [B, L] (oldest first, the last column is the frame of the look): the closing input by the run rule of LabelSource.read (distrule/v8f_t1_eval_rule.py:92-112): 0 where the frame of the
    look is not in the run; else the least squares slope over the trailing run of at most VIS_RUN_MAX frames, a step of more than VIS_STEP_MAX_M between neighbours ends the run, a run below
    VIS_RUN_MIN frames gives 0. -> float32 [B]."""
    g, ok = np.asarray(g, np.float64), np.asarray(ok, bool)
    B, L = g.shape
    out = np.zeros(B, np.float32)
    for b in range(B):
        if not ok[b, L - 1]:
            continue
        idx = [L - 1]
        f = L - 2
        while f >= 0 and len(idx) < VIS_RUN_MAX and ok[b, f] and abs(g[b, f + 1] - g[b, f]) <= VIS_STEP_MAX_M:
            idx.append(f)
            f -= 1
        if len(idx) < VIS_RUN_MIN:
            continue
        idx = idx[::-1]
        G = g[b, idx]
        t = np.arange(len(idx), dtype=np.float64) * dt
        tb = t.mean()
        out[b] = float(((t - tb) * (G - G.mean())).sum() / ((t - tb) ** 2).sum())
    return out


def vis_look(st, rd, e_off, p, g, used, ok):
    """The look on frame e = j + e_off of each window of `st`. p, g, ok, used: what the reader read and fed at that look (used = after feed()). -> (used with the file's gap where the reader marks a lead and
    the file has an output [B] float64, closing [B] float32 or None when --vis-closing is off). The earlier frames of the run are read again from the logged tokens (_reader_at), fed the same way."""
    S = STATE["vis"]
    ci = np.array([s.ci for s in st], np.int64)
    fr = np.array([int(s.j) + int(e_off) for s in st], np.int64)
    with np.errstate(invalid="ignore"):
        present = (np.asarray(p, np.float64) >= P_MIN) & np.asarray(ok, bool)
    used2 = vis_replace(S["tab"], ci, fr, present, used, STATE["vis_counts"])
    clos = None
    if S["close"]:
        cg, co = [], []
        for d in range(VIS_RUN_MAX - 1, 0, -1):         # the four earlier frames, oldest first
            pe, ge, oke = _reader_at(st, rd, int(e_off) - d)
            if STATE.get("pres") is not None:           # study preswrap: the earlier frames of the run see presence on p2 too (not counted again)
                pe = pres_look(st, int(e_off) - d, pe, False)
            prs, ue = feed(STATE["form"], pe, ge, oke, STATE["xs"], STATE["ys"])
            ue = vis_replace(S["tab"], ci, fr - d, prs, ue).astype(np.float32).astype(np.float64)
            cg.append(ue)
            co.append(vis_ok(prs, ue))
        uc = used2.astype(np.float32).astype(np.float64)
        cg.append(uc)
        co.append(vis_ok(present, uc))
        clos = vis_run_closing(np.stack(cg, 1), np.stack(co, 1))
        STATE["vis_counts"]["closing_rows"] += int(len(clos))
        STATE["vis_counts"]["closing_nonzero"] += int((clos != 0).sum())
    return used2, clos


def vis_tick(p, g, used, ok):
    """the tick-level look (frame j of each window of the batch)."""
    st = STATE["vis_states"]
    if st is None or len(st) != len(p):
        raise RuntimeError("the variant needs the states of this batch (ReaderPlanner not bound)")
    return vis_look(st, STATE["reader"], 0, p, g, used, ok)


PRES_MODES = ("head", "or")
STATE["pres"] = None                                   # set by main() when --vis-pres-file is given
STATE["pres_counts"] = {"rows": 0, "with_file": 0, "no_file": 0, "raised": 0, "lowered": 0}


def load_pres(path, mode, join):
    """-> the state of the presence file (arrays p, has_out, clips, off, n of pres_dump.py) checked against `join` (the evaluation join the run uses: clip, off, n). Refuses a missing or malformed file."""
    if mode not in PRES_MODES:
        raise SystemExit("REFUSED: --vis-pres-mode must be head or or, got %r" % (mode,))
    if not path or not os.path.isfile(path):
        raise SystemExit("REFUSED: --vis-pres-file not found: %s" % path)
    if join is None:
        raise SystemExit("REFUSED: --vis-pres-file needs the evaluation join to check against")
    with np.load(path, allow_pickle=False) as z:
        for k in ("p", "has_out", "clips", "off", "n"):
            if k not in z.files:
                raise SystemExit("REFUSED: --vis-pres-file lacks the array %s" % k)
        p, has = np.asarray(z["p"], np.float64), np.asarray(z["has_out"], bool)
        clips, off, n = np.asarray(z["clips"]), np.asarray(z["off"], np.int64), np.asarray(z["n"], np.int64)
    if not (p.ndim == has.ndim == 1 and len(p) == len(has) and len(p) > 0):
        raise SystemExit("REFUSED: --vis-pres-file arrays p and has_out are not one-dimensional of one length")
    if not (clips.ndim == off.ndim == n.ndim == 1 and len(clips) == len(off) == len(n)):
        raise SystemExit("REFUSED: --vis-pres-file arrays clips, off, n are not one-dimensional of one length")
    jc = [str(c) for c in np.asarray(join["clip"]).ravel()]
    if not np.array_equal(off, np.asarray(join["off"], np.int64)) or not np.array_equal(n, np.asarray(join["n"], np.int64)):
        raise SystemExit("REFUSED: --vis-pres-file off/n differ from those of the evaluation join")
    if [str(c) for c in clips] != jc:
        raise SystemExit("REFUSED: --vis-pres-file clip names differ from those of the evaluation join")
    if len(p) != int((off + n).max()):
        raise SystemExit("REFUSED: --vis-pres-file has %d rows, the join has %d" % (len(p), int((off + n).max())))
    fin = np.isfinite(p)
    if (has & ~fin).any():
        raise SystemExit("REFUSED: --vis-pres-file marks an output where p is not finite")
    if (p[fin] < 0.0).any() or (p[fin] > 1.0).any():
        raise SystemExit("REFUSED: --vis-pres-file has a p outside [0, 1]")
    return {"p": p, "use": has & fin, "off": off, "n": n, "mode": mode, "path": path}


def pres_values(S, ci, fr):
    """-> (pf [B] float64, NaN where there is no usable file value, have [B] bool) at (clip index, frame); a clip index or frame outside the join has none."""
    ci, fr = np.asarray(ci, np.int64), np.asarray(fr, np.int64)
    nc = len(S["off"])
    inside = (ci >= 0) & (ci < nc)
    cc = np.where(inside, ci, 0)
    inside = inside & (fr >= 0) & (fr < S["n"][cc])
    row = np.where(inside, S["off"][cc] + fr, 0)
    have = inside & S["use"][row]
    return np.where(have, S["p"][row], np.nan), have


def pres_replace(S, ci, fr, p, count=None):
    """p2 [B] float64: p where S is None or the file has no value; mode head: the file's value; mode or: the larger of p and the file's value (a p that is not a number takes the file's value)."""
    p = np.asarray(p, np.float64)
    if S is None:
        return p.copy()
    pf, have = pres_values(S, ci, fr)
    p2 = np.where(have, pf if S["mode"] == "head" else np.fmax(p, pf), p)
    if count is not None:
        with np.errstate(invalid="ignore"):
            count["rows"] += int(len(p))
            count["with_file"] += int(have.sum())
            count["no_file"] += int((~have).sum())
            count["raised"] += int(((p < P_MIN) & (p2 >= P_MIN)).sum())
            count["lowered"] += int(((p >= P_MIN) & (p2 < P_MIN)).sum())
    return p2


def pres_look(st, e_off, p, count=True):
    """p2 for frame e = j + e_off of each window of `st` (states with .ci and .j)."""
    ci = np.array([s.ci for s in st], np.int64)
    fr = np.array([int(s.j) + int(e_off) for s in st], np.int64)
    return pres_replace(STATE["pres"], ci, fr, p, STATE["pres_counts"] if count else None)


def pres_tick(p):
    """p2 for the tick-level look (frame j of each window of the batch)."""
    st = STATE["vis_states"]
    if st is None or len(st) != len(p):
        raise RuntimeError("the presence file needs the states of this batch (ReaderPlanner not bound)")
    return pres_look(st, 0, p)


def pres_fields(path, sha, mode):
    """the text the first line of the run log gains when the flag is given."""
    return " vis-pres-file %s vis pres sha256 %s vis-pres-mode %s" % (path, sha, mode)


def pres_meta(path, mode, sha):
    """the run record fields; nothing without the flag."""
    if not path:
        return {}
    return {"vis_pres_file": path, "vis_pres_file_sha256": sha, "vis_pres_mode": mode, "vis_pres_counts": json.dumps(STATE["pres_counts"], sort_keys=True)}


def build_rule_label(a, rule_join, load_join):
    """-> LabelSource of the lead rule: from the rule join when one is given (checked against the evaluation join), else from the evaluation join (the parent's behaviour)."""
    J = load_join(a.eval_join)
    if not rule_join:
        return W.LabelSource(J)
    if not os.path.isfile(rule_join):
        raise SystemExit("REFUSED: --rule-join file not found: %s" % rule_join)
    JR = load_join(rule_join)
    for k in ("clip", "off", "n", "lead_valid"):
        if k not in J or k not in JR or not np.array_equal(np.asarray(J[k]), np.asarray(JR[k])):
            raise SystemExit("REFUSED: --rule-join differs from the evaluation join in %s" % k)
    print("rule join %s sha256 %s (the label source of the rule only; scoring keeps %s)" % (rule_join, W.sha256_file(rule_join), a.eval_join), flush=True)
    return W.LabelSource(JR)


def _main_inner(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--form", choices=FORMS, default=os.environ.get("INTEG_FORM", "R0"))
    ap.add_argument("--reader", default=os.environ.get("INTEG_READER", ""))
    ap.add_argument("--qmap", default=os.environ.get("INTEG_QMAP", ""))
    ap.add_argument("--rule-log", default="")
    ap.add_argument("--slope", type=float, default=None)
    ap.add_argument("--k-diff", type=int, default=0)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--a-max", type=float, default=None)
    ap.add_argument("--b-lim", type=float, default=None)
    ap.add_argument("--alim-log", default="")
    ap.add_argument("--readout", choices=("pinned", "backward", "causal", "causal_w", "causal_z", "causal_g"), default=None)
    ap.add_argument("--acc-gain", type=float, default=None)
    ap.add_argument("--path-exec", action="store_true")
    ap.add_argument("--kappa-max", type=float, default=None)
    ap.add_argument("--readout-log", default="")
    ap.add_argument("--rule-every", type=int, default=10)
    ap.add_argument("--preroll-steps", type=int, default=0)
    ap.add_argument("--preroll-exec", choices=("plan", "logged"), default="plan")
    ap.add_argument("--closing-est", default="")
    ap.add_argument("--closing-table", default="")
    ap.add_argument("--rule-join", default="")
    ap.add_argument("--vis-gap-file", default="")
    ap.add_argument("--vis-closing", action="store_true")
    ap.add_argument("--vis-pres-file", default="")
    ap.add_argument("--vis-pres-mode", choices=("head", "or"), default="")
    mine, rest = ap.parse_known_args(argv)
    if mine.closing_est and mine.closing_table:
        raise SystemExit("REFUSED: --closing-est and --closing-table together (one source of the closing input only)")
    if mine.form == "RRe" and not (mine.closing_est or mine.closing_table):
        raise SystemExit("REFUSED: form RRe needs --closing-est NPZ (an estimator file of closest.py) or --closing-table NPZ")
    if mine.form != "RRe" and mine.closing_est:
        raise SystemExit("REFUSED: --closing-est belongs to form RRe only")
    if mine.form != "RRe" and mine.closing_table:
        raise SystemExit("REFUSED: --closing-table belongs to form RRe only")
    if (mine.vis_gap_file or mine.vis_closing) and mine.form != "RR":
        raise SystemExit("REFUSED: --vis-gap-file and --vis-closing belong to form RR only")
    if (mine.vis_pres_file or mine.vis_pres_mode) and mine.form != "RR":
        raise SystemExit("REFUSED: --vis-pres-file and --vis-pres-mode belong to form RR only")
    if mine.vis_pres_file and not mine.vis_pres_mode:
        raise SystemExit("REFUSED: --vis-pres-file needs --vis-pres-mode head or or")
    if mine.vis_pres_mode and not mine.vis_pres_file:
        raise SystemExit("REFUSED: --vis-pres-mode needs --vis-pres-file")
    if mine.kappa_max is not None and (not math.isfinite(mine.kappa_max) or not mine.kappa_max > 0):
        raise SystemExit("REFUSED: --kappa-max must be finite and above 0")
    if mine.kappa_max is not None and mine.readout is None:
        raise SystemExit("REFUSED: --kappa-max goes with --readout (pinned is allowed): the readout log carries the fields of the bound")
    T, path = W.load_pinned(mine.tree)
    STATE["T"] = T
    a = T.build_parser().parse_args(rest)
    if a.mode != "run":
        raise SystemExit("REFUSED: this wrapper applies to the `run` mode of the pinned tool only")
    pin = W.sha256_file(path)
    if pin != PINNED_SHA256:
        raise SystemExit("REFUSED: pinned tool sha256 %s != %s" % (pin, PINNED_SHA256))
    f = mine.form
    if f != "RRd" and (mine.slope is not None or mine.k_diff):
        raise SystemExit("REFUSED: --slope / --k-diff belong to form RRd only")
    if mine.tau is not None and (f == "R0" or not (0.2 <= mine.tau <= 4.0)):
        raise SystemExit("REFUSED: --tau needs a rule form and a value in [0.2, 4.0] s")
    rule_kw = {} if mine.tau is None else {"tau": float(mine.tau)}
    if (mine.a_max is None) != (mine.b_lim is None) or (mine.a_max is not None and (not mine.a_max > 0 or not mine.b_lim > 0 or not mine.alim_log)):
        raise SystemExit("REFUSED: --a-max and --b-lim come together, both above 0, with --alim-log")
    if mine.a_max is None and mine.alim_log:
        raise SystemExit("REFUSED: --alim-log without the two limits would write nothing")
    STATE["alim"] = None if mine.a_max is None else (float(mine.a_max), float(mine.b_lim))
    STATE["every"] = int(mine.rule_every)
    if STATE["every"] < 1:
        raise SystemExit("REFUSED: --rule-every needs a value of at least 1 step")
    if STATE["every"] < int(a.replan_steps):
        if f == "RRd":
            raise SystemExit("REFUSED: --rule-every below the replan interval is not available for the rule form RRd (its closing speed is a gap rate over an earlier read; the later looks carry none)")
        if f not in ("RLb", "RLc", "RR", "RRc", "RRh", "RRe"):
            raise SystemExit("REFUSED: --rule-every below the replan interval needs the rule form RLb, RLc, RR, RRc, RRh or RRe")
        if int(a.replan_steps) % STATE["every"]:
            raise SystemExit("REFUSED: --rule-every must divide the replan interval")
        if not mine.rule_log:
            raise SystemExit("REFUSED: --rule-every below the replan interval needs --rule-log (the later looks are written beside it)")
        if STATE["alim"] is not None:
            raise SystemExit("REFUSED: --rule-every below the replan interval with an acceleration limit is not a tested combination")
    STATE["m"] = int(a.replan_steps)
    install_preroll(T, mine, a)
    orig_fp = T.floor_planner
    if STATE["alim"] is not None:                      # the rollout of the kinematic prior alone is limited the same way (its own log columns f_*)
        def limited_floor(model, device):
            return AccelLimited(orig_fp(model, device), "f")

        T.floor_planner = limited_floor
    if (mine.readout is None) != (not mine.readout_log):
        raise SystemExit("REFUSED: --readout and --readout-log come together")
    if (mine.readout == "causal_g") != (mine.acc_gain is not None):
        raise SystemExit("REFUSED: --acc-gain goes with --readout causal_g and only with it")
    if mine.path_exec and mine.readout != "causal_g":
        raise SystemExit("REFUSED: --path-exec goes with --readout causal_g and only with it")
    if mine.acc_gain is not None and not np.isfinite(mine.acc_gain):
        raise SystemExit("REFUSED: --acc-gain must be finite")
    orig_hf = T.hermite_frames
    if mine.readout is not None:
        STATE["readout"] = mine.readout
        STATE["gain"] = None if mine.acc_gain is None else float(mine.acc_gain)
        STATE["path_exec"] = bool(mine.path_exec)
        STATE["kappa"] = None if mine.kappa_max is None else float(mine.kappa_max)
        STATE["kb"] = None if mine.kappa_max is None else kb_new()
        T.hermite_frames = counted_frames(orig_hf, mine.readout, STATE["gain"], STATE["path_exec"], STATE["kappa"])
    if f == "R0":
        if mine.rule_log or mine.reader or mine.qmap:
            raise SystemExit("REFUSED: R0 installs nothing; --rule-log/--reader/--qmap would write or read nothing")
        if STATE["alim"] is None:
            try:
                return T.cmd_run(a)
            finally:
                T.hermite_frames = orig_hf
                save_rlog(mine.readout_log, f, a.replan_steps)
        orig_mp = T.model_planner

        def limited_planner(model, device, keep_aux=False, keep_heads=False):
            return AccelLimited(orig_mp(model, device, keep_aux=keep_aux, keep_heads=keep_heads))

        T.model_planner = limited_planner
        try:
            return T.cmd_run(a)
        finally:
            T.model_planner = orig_mp
            T.floor_planner = orig_fp
            T.hermite_frames = orig_hf
            save_alog(mine.alim_log, f, a.replan_steps)
            save_rlog(mine.readout_log, f, a.replan_steps)
    if f in GAP_PLANT or f in ("RR", "RRc", "RRh", "RRd", "RRe"):
        if not mine.reader or not os.path.isfile(mine.reader):
            raise SystemExit("REFUSED: form %s needs --reader CKPT" % f)
        if f in ("RR", "RRc", "RRh", "RRd", "RRe"):
            if not mine.qmap or not os.path.isfile(mine.qmap):
                raise SystemExit("REFUSED: form RR needs --qmap JSON")
            STATE["xs"], STATE["ys"] = QM.load_map(mine.qmap)
            u = QM.unit_check(STATE["xs"], STATE["ys"])
            if not u["pass"]:
                raise SystemExit("REFUSED: the map fails its unit check")
        import tok_reader as TR
        STATE["reader"] = TR.TokReader.from_ckpt(mine.reader)
        if f == "RRe":
            if mine.closing_table:
                load_table(mine.closing_table, mine.rule_log)
            else:
                load_estimator(mine.closing_est, STATE["reader"], mine.rule_log)
        if f == "RRd":
            if mine.slope is None or not (0.0 < mine.slope <= 2.0) or mine.k_diff < 1:
                raise SystemExit("REFUSED: form RRd needs --slope in (0, 2] and --k-diff >= 1")
            STATE["slope"], STATE["k"] = float(mine.slope), int(mine.k_diff)
        STATE["vis"] = load_vis(mine.vis_gap_file, mine.vis_closing) if (mine.vis_gap_file or mine.vis_closing) else None
        STATE["pres"] = load_pres(mine.vis_pres_file, mine.vis_pres_mode, T.D.load_join(a.eval_join)) if mine.vis_pres_file else None
        if STATE["pres"] is not None:
            print("pres run-line fields:" + pres_fields(mine.vis_pres_file, W.sha256_file(mine.vis_pres_file), mine.vis_pres_mode), flush=True)
        W.head_values = head_values_reader
        mix = ("head", "head", {"RRc": "label", "RRh": "head", "RRd": "head", "RRe": "head"}.get(f, "zero"))
        if mine.vis_closing:                               # form RR only (refused above): the closing input is the one the variant computes, read by the rule from the head slot
            mix = (mix[0], mix[1], "head")
    elif f in ("RLb", "RLc"):
        W.head_values = head_values_none
        mix = ("label", "label", "label" if f == "RLc" else "zero")
    STATE["form"] = f
    W.HEAD_P_MIN = P_MIN
    X.MIX.update(presence=mix[0], gap=mix[1], closing=mix[2])
    W.MODE_CFG["mix"] = ("mix", dict(rule_kw))
    W.RulePlanner = ReaderPlanner
    X.STATE["label"] = build_rule_label(a, mine.rule_join, T.D.load_join)
    log = W.RuleLog()
    STATE["rulelog"] = log
    made, restore = W.install(T, "mix", a, None, log, "cap-only")
    try:
        rc = T.cmd_run(a)
    finally:
        restore()
        T.floor_planner = orig_fp
        T.hermite_frames = orig_hf
        save_alog(mine.alim_log, f, a.replan_steps)
        save_rlog(mine.readout_log, f, a.replan_steps)
        if mine.rule_log:
            meta = {"mode": "mix", "source": "mix", "form": f, "mix": json.dumps(X.MIX, sort_keys=True), "mix_counts": json.dumps(X.COUNT, sort_keys=True), "floor": "cap-only",
                    "D": R.D_M, "TAU": float(rule_kw.get("tau", R.TAU_S)), "B_MAX": R.B_MAX, "G_MAX": R.G_MAX_M, "replan_steps": a.replan_steps, "pinned_path": path, "pinned_sha256": pin, "ckpt": a.ckpt, "tag": a.tag,
                    "eval_join": a.eval_join, "join_sha256": W.sha256_file(a.eval_join) if os.path.isfile(a.eval_join) else "", "rule_join": mine.rule_join,
                    "rule_join_sha256": W.sha256_file(mine.rule_join) if mine.rule_join and os.path.isfile(mine.rule_join) else "", "vis_gap_file": mine.vis_gap_file,
                    "vis_gap_file_sha256": W.sha256_file(mine.vis_gap_file) if mine.vis_gap_file and os.path.isfile(mine.vis_gap_file) else "", "vis_closing": int(bool(mine.vis_closing)),
                    "vis_counts": json.dumps(STATE["vis_counts"], sort_keys=True),
                    **pres_meta(mine.vis_pres_file, mine.vis_pres_mode, W.sha256_file(mine.vis_pres_file) if mine.vis_pres_file and os.path.isfile(mine.vis_pres_file) else ""), "reader": mine.reader,
                    "reader_sha256": W.sha256_file(mine.reader) if mine.reader else "", "qmap": mine.qmap, "qmap_sha256": W.sha256_file(mine.qmap) if mine.qmap else "", "mode_kwargs": json.dumps(rule_kw, sort_keys=True), "k_diff": int(STATE["k"] or 0), "slope": float(STATE["slope"] or 0.0), "dt_s": DT_S, "a_max": float(STATE["alim"][0]) if STATE["alim"] else float("nan"), "b_lim": float(STATE["alim"][1]) if STATE["alim"] else float("nan")}
            log.save(mine.rule_log, dict(meta, **est_tick_arrays(log)) if f == "RRe" else meta)
            print("wrote", mine.rule_log, flush=True)
            save_mid(mine.rule_log, meta, a.replan_steps)
            if STATE["cols"]["p"]:
                side = (mine.rule_log[:-4] if mine.rule_log.endswith(".npz") else mine.rule_log) + "_rd.npz"
                cols = {k: np.concatenate(v) for k, v in STATE["cols"].items()}
                with open(side, "wb") as fh:
                    np.savez(fh, **cols, form=np.asarray(f))
                print("wrote", side, flush=True)
    return rc


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        return _main_inner(argv)
    finally:
        finish_preroll()
        write_est_line()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""v8f_t1_eval_reader_rule_rzgfhqs.py -- written by make_handover_src.py from v8f_t1_eval_reader_rule_rzgfhq.py (study "handover-src"): one more parameter.
  --t0-state own|logged|speed_logged|path_logged (default own; accepted only with --preroll-steps above 0, else REFUSED)
  After a pre-roll the model reads, at the FIRST tick of the scored window, the own executed history. This parameter replaces, at that tick only and only in the batch the planner is
  handed, one group of that state by what a fresh window at the same t0 reads (the logged cache). The pre-roll itself, the executor (it continues from the OWN pose), the own state
  that later ticks read, the rule and its geometry (own position, a_m0), and the way the scored track is written (the frame of the logged pose at t0) are not changed.
    own           nothing is replaced. No new statement changes a value: the dumps equal those of the parent bit for bit (the build_item patch is not installed).
    logged        everything the model reads at that tick (the ego history rows of the batch: p10f, base, pose_now / pose_t / pose_s, the route channel nav, and the speed-bucket
                  input d["v"]) is the logged cache of the window: the item is the one a fresh window at the same t0 builds.
    speed_logged  the speed group from the logged cache (v, a = dv/dt, the speed-bucket input d["v"], pose column 3); positions and headings (pose x, y, yaw), the yaw-rate channel
                  and the route channel as own; kappa = clamp(omega / max(v, 2), +-0.3) is recomputed from the own omega and the logged v; p10f and base follow from that kin3.
    path_logged   positions and headings (pose x, y, yaw) and the yaw-rate channel from the logged cache; the speed group and the route channel as own; kappa from the logged
                  omega and the own v; p10f and base follow.
  Rows replaced: the ego history rows the batch reads at that tick (the last hmax rows, up to t0) and, for the speed-bucket input, every row up to t0. The tick-0 ages are 0.
  Mechanism: the pinned build_item is wrapped (T.build_item, restored in finish_preroll) for the main-phase call only (PrerollRun.main_kind); the item is built from a view of the
  state whose arrays are the chosen mixture (HandoverView, handover_shadow). The own item is built once more for the comparison (no counters touched) and the largest absolute
  difference per column group is kept per planner kind: positions (pose x, y), headings (pose yaw, wrapped), speed (p10f column 0), and p10f other columns, base, nav, limit.
  Log: one line per planner kind, "ZZ T0-STATE mode=... kind=...", and the keys t0_state / t0_state_first_tick_max_abs_diff / t0_state_note in preroll_log_<run>.json and t0_state in
  the meta of rule_log_pre_<run>.npz. The tick file is written by the pinned tool and carries no wrapper meta; its tick-0 rows (v0, s_pose, t_pose, plan) show what was handed over.
  The pre-roll offsets file (preroll_offsets_<run>.npz) is identical across the four modes: nothing before the first scored tick is touched.
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
FORMS = ("R0", "RLb", "RR", "RRid", "RR0", "RLc", "RRc", "RRh", "RRd")
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
    elif form in ("RR", "RRc", "RRh", "RRd"):
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


def head_values_reader(model, feats, frame_ok=None, hist=None):
    """Replaces W.head_values: p [B], gap [B], closing [B] (zero: the rule form without the closing speed), ok [B]. Takes tokens only; `model` is not read."""
    rd = STATE["reader"]
    f = feats[:, -rd.hist:]
    p, g = rd.read(f)
    ok = np.ones(len(p), bool) if frame_ok is None else frame_ok[:, -rd.hist:].all(-1).cpu().numpy()
    present, used = feed(STATE["form"], p, g, ok, STATE["xs"], STATE["ys"])
    c = reader_closing(rd, f)                                   # logged in every reader form; fed into the rule only in form RRh
    for k, v in (("p", p), ("g_read", g), ("g_used", used), ("ok", ok), ("c_read", c)):
        STATE["cols"][k].append(np.asarray(v))
    c_fed = np.zeros(len(p), np.float32)
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
    return p, used.astype(np.float32), (c if STATE["form"] == "RRh" else c_fed), ok


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
MID_FIELDS = tuple(W.LOG_FIELDS) + ("q", "excess", "nchg")
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
        if form == "RR":
            p, g, okr = _reader_at(st, rd, q)
            _, used = feed(form, p, g, okr, STATE["xs"], STATE["ys"])
            hg, hc = used.astype(np.float32).astype(np.float64), np.zeros(B, np.float64)
            hp = (p >= W.HEAD_P_MIN) & okr
            p_src = p
            for kk, vv in (("p", p), ("g_read", g), ("g_used", used), ("ok", okr)):
                M["rd"][kk].append(np.asarray(vv))
        else:                                          # head_values_none of the wrapper: the label forms read no token
            p_src, hg, hc = np.zeros(B, np.float64), np.full(B, np.nan), np.zeros(B, np.float64)
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
                   changed=res.changed, g_src=g_src, c_src=c_src, p_src=p_src, v0=v0, v_l=v_l, v_L=v_L, a_l=a_l, a_m=a_m, g0=g0, q=np.full(B, q, np.int64), excess=excess, nchg=nchg)
        for f in MID_FIELDS:
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
    cols = {f: (np.concatenate(M["rows"][f]) if M["rows"][f] else np.zeros(0)) for f in MID_FIELDS}
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
        self.main_kind = None                          # handover-src: the planner kind while the main-phase call runs, else None

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
        self.main_kind = kind
        try:
            plan0, exe = orig(planner, states, *args, **kw_main)
        finally:
            self.main_kind = None
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


# ==== HANDOVER BEGIN (study handover-src) ==========================================================================================================
T0_STATES = ("own", "logged", "speed_logged", "path_logged")
HO_EGO = ("pose", "kin3", "p10f", "base", "a", "ch1", "v")      # the ego arrays of the pinned WindowState (WindowState.EGO)


class HandoverView:
    """What the pinned build_item reads of a window state: ci, j, T, d, nav, scene0 (and nothing else)."""

    def __init__(self, st, d, nav):
        self.ci, self.j, self.T, self.d, self.nav, self.scene0 = st.ci, st.j, st.T, d, nav, st.scene0


def _nanmax_abs(x):
    x = np.abs(np.asarray(x, np.float64)).reshape(-1)
    x = x[np.isfinite(x)]
    return float(x.max()) if x.size else 0.0


def _wrap_pi(x):
    return (np.asarray(x, np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def handover_shadow(mode, own_d, lg, own_nav, lg_nav, t0, hmax, kappa, kin_feat, kdx):
    """-> (d, nav, info): the ego arrays and the route array the first scored tick of an aged window reads under `mode` (not own). own_d = the state's d (own executed rows up to t0), lg = the logged
    cache dict of the clip (unmodified arrays), own_nav / lg_nav = the state's nav and the logged nav rows of the clip. Arrays are copies; own_d and lg are not changed.
    kappa(omega, v) = the programme's _pose_kappa0 (clamp(omega / max(v, 2), +-0.3)); kin_feat(kin3 [n, 3]) -> [n, 8] (floors_kin.kin_features); kdx(kin3 [n, 3]) -> [n, K] (kdx_base, yaw units).
    info: for the two partial modes the largest |kappa(own omega, own v) - own kin3 kappa| and the same for the logged arrays over the history rows (the fidelity of the recomposition)."""
    if mode not in T0_STATES[1:]:
        raise ValueError("t0 state mode %r" % (mode,))
    t0 = int(t0)
    d = dict(own_d)
    for k in HO_EGO:
        d[k] = np.array(own_d[k], copy=True)
    nav = np.array(own_nav, copy=True)
    sel = slice(0, t0 + 1)
    if mode == "logged":
        for k in HO_EGO:
            d[k][sel] = np.asarray(lg[k])[sel]
        nav[sel] = np.asarray(lg_nav)[sel]
        return d, nav, {}
    hist = np.arange(max(1, t0 - int(hmax) + 1), t0 + 1)
    info = {}
    if mode == "speed_logged":
        d["v"][sel] = np.asarray(lg["v"])[sel]
        d["a"][sel] = np.asarray(lg["a"])[sel]
        d["kin3"][sel, 0:2] = np.asarray(lg["kin3"])[sel, 0:2]
        d["pose"][sel, 3] = np.asarray(lg["pose"])[sel, 3]
        omega = np.asarray(own_d["ch1"], np.float64)
    else:
        d["pose"][sel, 0:3] = np.asarray(lg["pose"])[sel, 0:3]
        d["ch1"][sel] = np.asarray(lg["ch1"])[sel]
        omega = np.asarray(lg["ch1"], np.float64)
    info["kappa_recompose_max_abs_own"] = _nanmax_abs(kappa(np.asarray(own_d["ch1"], np.float64)[hist], np.asarray(own_d["kin3"], np.float64)[hist, 0]) - np.asarray(own_d["kin3"], np.float64)[hist, 2])
    info["kappa_recompose_max_abs_logged"] = _nanmax_abs(kappa(np.asarray(lg["ch1"], np.float64)[hist], np.asarray(lg["kin3"], np.float64)[hist, 0]) - np.asarray(lg["kin3"], np.float64)[hist, 2])
    k3 = np.asarray(d["kin3"], np.float64)
    d["kin3"][hist, 2] = kappa(omega[hist], k3[hist, 0])
    d["p10f"][hist] = np.asarray(kin_feat(np.asarray(d["kin3"], np.float64)[hist])).astype(d["p10f"].dtype)
    d["base"][t0] = np.asarray(kdx(np.asarray(d["kin3"], np.float64)[t0:t0 + 1]))[0].astype(d["base"].dtype)
    return d, nav, info


def item_diff(a, b):
    """the largest absolute difference between two items of the pinned build_item, per column group: positions (pose x, y: the history rows and pose_now / pose_t / pose_s), headings
    (pose yaw, wrapped), speed (p10f column 0), and the others (p10f columns 1.., base, nav, limit, frame_ok flags: the number of rows that differ)."""
    pos, head = 0.0, 0.0
    for key in ("pose", "pose_now", "pose_t", "pose_s"):
        pa, pb = np.atleast_2d(np.asarray(a[key], np.float64)), np.atleast_2d(np.asarray(b[key], np.float64))
        pos = max(pos, _nanmax_abs(pa[:, :2] - pb[:, :2]))
        head = max(head, _nanmax_abs(_wrap_pi(pa[:, 2] - pb[:, 2])))
    fa, fb = np.asarray(a["p10f"], np.float64), np.asarray(b["p10f"], np.float64)
    return {"positions_m": pos, "headings_rad": head, "speed_ms": _nanmax_abs(fa[:, 0] - fb[:, 0]), "p10f_other": _nanmax_abs(fa[:, 1:] - fb[:, 1:]),
            "base": _nanmax_abs(np.asarray(a["base"], np.float64) - np.asarray(b["base"], np.float64)), "nav": _nanmax_abs(np.asarray(a["nav"], np.float64) - np.asarray(b["nav"], np.float64)),
            "limit": _nanmax_abs(np.asarray(a["limit"], np.float64) - np.asarray(b["limit"], np.float64)),
            "frame_ok_rows": float(np.sum(np.asarray(a["frame_ok"], bool) != np.asarray(b["frame_ok"], bool)))}


def ho_stats_add(stats, kind, diff, info):
    s = stats.setdefault(kind, {"states": 0, "states_item_differs": 0, "max_abs": {k: 0.0 for k in diff}, "kappa_recompose_max_abs_own": 0.0, "kappa_recompose_max_abs_logged": 0.0})
    s["states"] += 1
    s["states_item_differs"] += int(any(v > 0.0 for v in diff.values()))
    for k, v in diff.items():
        s["max_abs"][k] = max(s["max_abs"][k], float(v))
    for k in ("kappa_recompose_max_abs_own", "kappa_recompose_max_abs_logged"):
        if k in info:
            s[k] = max(s[k], float(info[k]))


def make_handover_build_item(orig_bi, T, runner, mode, stats):
    """The build_item the pinned rollout calls while --t0-state is not own. Only the first tick of the MAIN phase of a re-based state is touched (runner.main_kind is set by PrerollRun while its
    main call runs); every other call (the pre-roll segments, later ticks) goes to orig_bi unchanged."""
    cache = {}

    def kappa(om, v):
        if "f" not in cache:
            import torch
            from tanitad.models.kinematic_prior import _pose_kappa0
            cache["f"] = (torch, _pose_kappa0)
        torch_, pk = cache["f"]
        return pk(torch_.from_numpy(np.array(om, np.float64)), torch_.from_numpy(np.array(v, np.float64))).numpy()

    def kin_feat(k3):
        return T.fk.kin_features(k3)

    def kdx(k3):
        return T.D.kdx_base(k3, T.D.KIN_UNITS["yaw"])

    def build_item(ds, proxy, st, tick, a_t, a_s, scene, limit, stats_=None):
        kind = runner.main_kind
        if kind is None or int(tick) != 0 or getattr(st, "frame_pose5", None) is None or not hasattr(st, "lr_base") or int(st.j) != int(st.lr_t0):
            return orig_bi(ds, proxy, st, tick, a_t, a_s, scene, limit, stats_)
        d, nav, info = handover_shadow(mode, st.d, st.lr_base, st.nav, st.lr_nav, st.lr_t0, ds.hmax, kappa, kin_feat, kdx)
        view = HandoverView(st, d, nav)
        it = orig_bi(ds, proxy, view, tick, a_t, a_s, scene, limit, stats_)
        st.scene0 = view.scene0
        ref = orig_bi(ds, proxy, HandoverView(st, st.d, st.nav), tick, a_t, a_s, scene, limit, None)     # the own item, for the comparison only (no counters)
        ho_stats_add(stats, kind, item_diff(it, ref), info)
        return it
    return build_item
# ==== HANDOVER END ===================================================================================================================================


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
    STATE["t0state"] = str(getattr(mine, "t0_state", "own"))
    STATE["ho_stats"], STATE["ho_orig_bi"] = {}, None
    if P < 0:
        raise SystemExit("REFUSED: --preroll-steps must be 0 or more")
    if P == 0:
        if mine.preroll_exec != "plan":
            raise SystemExit("REFUSED: --preroll-exec goes with --preroll-steps above 0")
        if STATE["t0state"] != "own":
            raise SystemExit("REFUSED: --t0-state %s goes with --preroll-steps above 0 (there is no state hand-over without a pre-roll)" % STATE["t0state"])
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
            if STATE["t0state"] != "own":              # handover-src: the logged cache of the clip and its nav rows (read-only references / a copy)
                st.lr_base = store.get(str(J["clip"][st.ci]))
                st.lr_nav = np.array(J["nav"][int(J["off"][st.ci]):int(J["off"][st.ci]) + int(J["n"][st.ci])], np.float32)
        return out

    T.make_states = make_states
    T.rollout = build_rollout(orig_ro, runner)
    STATE["pre_shim"] = (R, R.distance_rule)
    R.distance_rule = make_g0_shim(R.distance_rule, STATE)
    if STATE["t0state"] != "own":                      # handover-src: not installed for own, so own is the parent
        STATE["ho_orig_bi"] = T.build_item
        T.build_item = make_handover_build_item(T.build_item, T, runner, STATE["t0state"], STATE["ho_stats"])
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
    if STATE.get("ho_orig_bi") is not None:
        T.build_item = STATE["ho_orig_bi"]
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
        meta["t0_state"] = np.asarray(STATE.get("t0state", "own"))
        cols = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rows.items()}
        cols["pre_segment"] = np.concatenate(segcol) if segcol else np.zeros(0, np.int64)
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
    t0m, hs = STATE.get("t0state", "own"), STATE.get("ho_stats") or {}
    rec["t0_state"] = t0m
    rec["t0_state_first_tick_max_abs_diff"] = hs
    rec["t0_state_note"] = "largest absolute difference, over the states of the run and the history rows, between the batch handed to the model at the first scored tick and the own batch; empty for own (nothing replaced)"
    if t0m == "own":
        print("ZZ T0-STATE mode=own nothing replaced: the frames handed to the model are the own frames (all differences 0.0 by construction)", flush=True)
    for kk in sorted(hs):
        mx = hs[kk]["max_abs"]
        print("ZZ T0-STATE mode=%s kind=%s states=%d states_item_differs=%d max_abs_diff positions_m=%.9g headings_rad=%.9g speed_ms=%.9g | p10f_other=%.9g base=%.9g nav=%.9g limit=%.9g frame_ok_rows=%d | kappa_recompose own=%.3g logged=%.3g"
              % (t0m, kk, hs[kk]["states"], hs[kk]["states_item_differs"], mx["positions_m"], mx["headings_rad"], mx["speed_ms"], mx["p10f_other"], mx["base"], mx["nav"], mx["limit"], int(mx["frame_ok_rows"]),
                 hs[kk]["kappa_recompose_max_abs_own"], hs[kk]["kappa_recompose_max_abs_logged"]), flush=True)
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
    ap.add_argument("--t0-state", choices=T0_STATES, default="own")
    mine, rest = ap.parse_known_args(argv)
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
        if f not in ("RLb", "RLc", "RR"):
            raise SystemExit("REFUSED: --rule-every below the replan interval needs the rule form RLb, RLc or RR")
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
    if f in GAP_PLANT or f in ("RR", "RRc", "RRh", "RRd"):
        if not mine.reader or not os.path.isfile(mine.reader):
            raise SystemExit("REFUSED: form %s needs --reader CKPT" % f)
        if f in ("RR", "RRc", "RRh", "RRd"):
            if not mine.qmap or not os.path.isfile(mine.qmap):
                raise SystemExit("REFUSED: form RR needs --qmap JSON")
            STATE["xs"], STATE["ys"] = QM.load_map(mine.qmap)
            u = QM.unit_check(STATE["xs"], STATE["ys"])
            if not u["pass"]:
                raise SystemExit("REFUSED: the map fails its unit check")
        import tok_reader as TR
        STATE["reader"] = TR.TokReader.from_ckpt(mine.reader)
        if f == "RRd":
            if mine.slope is None or not (0.0 < mine.slope <= 2.0) or mine.k_diff < 1:
                raise SystemExit("REFUSED: form RRd needs --slope in (0, 2] and --k-diff >= 1")
            STATE["slope"], STATE["k"] = float(mine.slope), int(mine.k_diff)
        W.head_values = head_values_reader
        mix = ("head", "head", {"RRc": "label", "RRh": "head", "RRd": "head"}.get(f, "zero"))
    elif f in ("RLb", "RLc"):
        W.head_values = head_values_none
        mix = ("label", "label", "label" if f == "RLc" else "zero")
    STATE["form"] = f
    W.HEAD_P_MIN = P_MIN
    X.MIX.update(presence=mix[0], gap=mix[1], closing=mix[2])
    W.MODE_CFG["mix"] = ("mix", dict(rule_kw))
    W.RulePlanner = ReaderPlanner
    X.STATE["label"] = W.LabelSource(T.D.load_join(a.eval_join))
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
                    "eval_join": a.eval_join, "join_sha256": W.sha256_file(a.eval_join) if os.path.isfile(a.eval_join) else "", "reader": mine.reader,
                    "reader_sha256": W.sha256_file(mine.reader) if mine.reader else "", "qmap": mine.qmap, "qmap_sha256": W.sha256_file(mine.qmap) if mine.qmap else "", "mode_kwargs": json.dumps(rule_kw, sort_keys=True), "k_diff": int(STATE["k"] or 0), "slope": float(STATE["slope"] or 0.0), "dt_s": DT_S, "a_max": float(STATE["alim"][0]) if STATE["alim"] else float("nan"), "b_lim": float(STATE["alim"][1]) if STATE["alim"] else float("nan")}
            log.save(mine.rule_log, meta)
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


if __name__ == "__main__":
    sys.exit(main())

"""make_longroll10k_b.py -- writes v8f_t1_eval_reader_rule_rzgfhq.py from v8f_t1_eval_reader_rule_rzgfhp.py by asserted text edits (each old text exactly once).
Second make step of the study "longroll": three repairs of the first build. The first build, the pinned tool v8f_t1_eval.py and the distrule files are not changed. Standard library only.
Usage: python make_longroll10k_b.py (in this folder). Refuses to overwrite the output."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhp.py")
OUT = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhq.py")

HEADER = '''"""v8f_t1_eval_reader_rule_rzgfhq.py -- written by make_longroll10k_b.py from v8f_t1_eval_reader_rule_rzgfhp.py (the first build of the study "longroll"): three repairs, all for P above 0.
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
'''

OLD_FIRST = '"""v8f_t1_eval_reader_rule_rzgfhp.py -- written by make_longroll10k.py from v8f_t1_eval_reader_rule_rzgfh.py for the study "longroll" (pre-roll before every window): two more parameters.\n'

RUNNER = '''SEG = 30                                           # the rule's arc tables are those of the 30-step logged path from st.t0: one pre-roll segment = one table


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


'''

PLANNER_OLD = '''        plan = X.MixPlanner.__call__(self, batch)
        if int(STATE["every"]) < int(self.m):
'''
PLANNER_NEW = '''        pre = STATE.get("preroll", 0) > 0
        if pre:
            STATE["g0_corr"] = a_m0_first_tick(R, self.fk.gt_waypoints, self.states or ())     # R1: consumed by the next R.distance_rule call (the tick-level one) only
        plan = X.MixPlanner.__call__(self, batch)
        if pre:
            STATE["g0_corr"] = None
        if int(STATE["every"]) < int(self.m):
'''


def once(text, old, new):
    n = text.count(old)
    assert n == 1, "expected exactly 1 match, got %d for: %r" % (n, old[:90])
    return text.replace(old, new)


def between(text, a, b, new):
    """replace the text from a (kept out) up to b (kept) by new; a and b each exactly once, a before b."""
    assert text.count(a) == 1 and text.count(b) == 1, (a[:40], b[:40])
    i, j = text.index(a), text.index(b)
    assert i < j
    return text[:i] + new + text[j:]


def main():
    if os.path.exists(OUT) and os.environ.get("LR_OVERWRITE_OWN") != "1":
        sys.exit("REFUSED: %s exists" % OUT)
    t = open(SRC, encoding="utf-8", newline="").read()
    assert "\r" not in t
    t = once(t, OLD_FIRST, HEADER)
    t = between(t, "class PrerollRun:\n", "def build_rollout(orig, runner):\n", RUNNER)
    t = once(t, PLANNER_OLD, PLANNER_NEW)
    # install: P must be a multiple of 30; runner with the rule module; the g0 shim on R.distance_rule
    t = once(t, "    runner = PrerollRun(T, P, mine.preroll_exec, _tracked)\n",
             "    if P % SEG or SEG % int(a.replan_steps):\n        raise SystemExit(\"REFUSED: --preroll-steps %d is not a multiple of %d, or --replan-steps %d does not divide %d (one pre-roll segment = the 30-step logged path of the rule's arc tables)\" % (P, SEG, int(a.replan_steps), SEG))\n"
             "    runner = PrerollRun(T, P, mine.preroll_exec, _tracked, R)\n")
    t = once(t, "    T.rollout = build_rollout(orig_ro, runner)\n",
             "    T.rollout = build_rollout(orig_ro, runner)\n    STATE[\"pre_shim\"] = (R, R.distance_rule)\n    R.distance_rule = make_g0_shim(R.distance_rule, STATE)\n")
    # finish: restore the rule function, segment column, offsets, a_m0 log
    t = once(t, "    T.make_states, T.rollout = orig_ms, orig_ro\n    mine, a = STATE[\"pre_mine\"], STATE[\"pre_a\"]\n",
             "    T.make_states, T.rollout = orig_ms, orig_ro\n    Rm, orig_dr = STATE[\"pre_shim\"]\n    Rm.distance_rule = orig_dr\n    mine, a = STATE[\"pre_mine\"], STATE[\"pre_a\"]\n")
    t = once(t, "        for kind, post in runner.posts:\n            if kind == \"model\" and \"rule_rows\" in post:\n                for k in W.LOG_FIELDS:\n                    rows[k] += post[\"rule_rows\"][k]\n",
             "        segcol = []\n        for (kind, post), sg in zip(runner.posts, runner.post_seg):\n            if kind == \"model\" and \"rule_rows\" in post:\n                for k in W.LOG_FIELDS:\n                    rows[k] += post[\"rule_rows\"][k]\n"
             "                segcol.append(np.full(int(sum(len(x) for x in post[\"rule_rows\"][\"ci\"])), sg, np.int64))\n")
    t = once(t, "        cols = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rows.items()}\n",
             "        cols = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rows.items()}\n        cols[\"pre_segment\"] = np.concatenate(segcol) if segcol else np.zeros(0, np.int64)\n")
    t = once(t, "        np.savez(fh, kind=np.asarray(runner.off[\"kind\"]), ci=np.asarray(runner.off[\"ci\"], np.int64), t0=np.asarray(runner.off[\"t0\"], np.int64),\n"
                "                 dpos=np.asarray(runner.off[\"dpos\"], np.float64), dpsi_deg=np.asarray(runner.off[\"dpsi_deg\"], np.float64), dv=np.asarray(runner.off[\"dv\"], np.float64))\n",
             "        np.savez(fh, **runner.offset_arrays())\n")
    t = once(t, "    rec.update(runner.summary())\n",
             "    rec.update(runner.summary())\n    rec[\"a_m0_used_in_rule\"] = STATE.get(\"g0_stats\") or {}\n"
             "    rec[\"a_m0_note\"] = \"rule-log columns a_m and g0 of first-tick rows show the definition values; the rule used g0 - a_m0 (a_m0_first_tick in the offsets file)\"\n")
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    print("wrote", OUT, hashlib.sha256(t.encode("utf-8")).hexdigest()[:16], t.count("\n"), "lines")


if __name__ == "__main__":
    main()

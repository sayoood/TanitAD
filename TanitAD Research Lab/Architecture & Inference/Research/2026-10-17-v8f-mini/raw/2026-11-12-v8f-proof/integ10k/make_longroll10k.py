"""make_longroll10k.py -- writes v8f_t1_eval_reader_rule_rzgfhp.py from v8f_t1_eval_reader_rule_rzgfh.py by asserted text edits (each old text exactly once).
Study wrapper "longroll": every evaluation window starts after a pre-roll of P cache steps on the model's own actions, so errors can be read at rollout age 6-12 s and 12-18 s.
The parent and the pinned tool v8f_t1_eval.py (sha256 61e140edb9da01ac...) are not changed. Pure text generator (standard library only).
Usage: python make_longroll10k.py (in this folder). Refuses to overwrite the output."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfh.py")
OUT = os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhp.py")

HEADER = '''"""v8f_t1_eval_reader_rule_rzgfhp.py -- written by make_longroll10k.py from v8f_t1_eval_reader_rule_rzgfh.py for the study "longroll" (pre-roll before every window): two more parameters.
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
'''
PARENT_HEAD = '"""v8f_t1_eval_reader_rule_rzgfh.py -- written by make_headbound10k.py from'

PURE = r'''# ==== PREROLL PURE BEGIN (study longroll) =============================================================================================================
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


class PrerollRun:
    """One call of the pinned rollout becomes: pre-roll (P ticks, H = P, hook None) + the unmodified main rollout. T = the pinned module (H, plan_to_world, fk). tracked() returns
    [(name, obj)] of the mutable stores that must hold main-phase data only; their pre-roll content is kept in self.posts."""

    def __init__(self, T, P, pre_exec, tracked):
        self.T, self.P, self.pre_exec, self.tracked = T, int(P), pre_exec, tracked
        self.calls, self.posts = [], []
        self.off = {"kind": [], "ci": [], "t0": [], "dpos": [], "dpsi_deg": [], "dv": []}
        self.clipped_pre = {"model": 0, "floor": 0}

    def __call__(self, orig, planner, states, args, kw):
        T, P = self.T, self.P
        m = int(kw.get("m", 1))
        if P % m:
            raise ValueError("--preroll-steps %d must be a multiple of --replan-steps %d" % (P, m))
        kind = "model" if hasattr(planner, "keys_read") else "floor"
        snap = [(n, copy.deepcopy(o)) for n, o in self.tracked()]
        for n, o in self.tracked():
            _restore(o, _zero(o))                     # the pre-roll counts into zeroed stores; the main phase gets its own content back below
        kw_pre = dict(kw)
        kw_pre["hook"] = None
        kw_pre["executor"] = self.pre_exec
        if self.pre_exec == "logged":
            kw_pre["logged_tracks"] = {st.ci: st.lr_track for st in states}
        old_h = T.H
        T.H = P
        try:
            orig(planner, states, *args, **kw_pre)
        finally:
            T.H = old_h
            post = {}
            for (n, o), (_, s0) in zip(self.tracked(), snap):
                post[n] = copy.deepcopy(o)
                _restore(o, s0)
            self.posts.append((kind, post))
        own, lg = [], []
        for st in states:
            t0 = int(st.lr_t0)
            if int(st.j) != t0:
                raise RuntimeError("pre-roll ended at step %d, not at t0 = %d" % (int(st.j), t0))
            op = np.asarray(st.d["pose"][t0], np.float64)
            lp = np.asarray(st.lr_track[2 * t0], np.float64)
            dp, dh, dv = pre_offsets(op, lp, own_v=float(st.d["v"][t0]))
            for k, v in (("kind", kind), ("ci", int(st.ci)), ("t0", t0), ("dpos", dp), ("dpsi_deg", dh), ("dv", dv)):
                self.off[k].append(v)
            self.clipped_pre[kind] += int(st.clipped)
            st.clipped = 0
            st.t0 = t0
            st.frame_pose5 = lp.copy()
            st.executed_ego = (lambda st=st, lp=lp: exec_logged_frame(T.fk.gt_waypoints, st.track, st.lr_t0, lp, T.H))
            own.append(op)
            lg.append(lp)
        self.calls.append({"kind": kind, "windows": len(states), "ticks_per_window": P // m})
        plan0, exe = orig(planner, states, *args, **kw)
        return map_plan0(T.plan_to_world, plan0, own, lg).astype(np.asarray(plan0).dtype), exe

    def summary(self):
        kinds = sorted(set(self.off["kind"]))
        km = np.asarray(self.off["kind"])
        per = {}
        for k in kinds:
            sel = km == k
            per[k] = {f: stat4(np.asarray(self.off[f], np.float64)[sel]) for f in ("dpos", "dpsi_deg", "dv")}
        return {"windows": {k: int(sum(c["windows"] for c in self.calls if c["kind"] == k)) for k in kinds},
                "ticks_pre_rolled": {k: int(sum(c["windows"] * c["ticks_per_window"] for c in self.calls if c["kind"] == k)) for k in kinds},
                "rollout_calls": len(self.calls), "offsets_at_t0": per, "clipped_windows_in_preroll": dict(self.clipped_pre)}


def build_rollout(orig, runner):
    """the rollout the wrapper installs; P = 0 (runner None) installs nothing: the original function itself."""
    if runner is None:
        return orig

    def rollout(planner, states, *args, **kw):
        return runner(orig, planner, states, args, kw)
    return rollout
# ==== PREROLL PURE END ===============================================================================================================================
'''

GLUE = r'''

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
    runner = PrerollRun(T, P, mine.preroll_exec, _tracked)
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
    mine, a = STATE["pre_mine"], STATE["pre_a"]
    folder, run = _log_name(mine, a)
    os.makedirs(folder, exist_ok=True)
    if mine.rule_log and STATE.get("rulelog") is not None:
        rows = {k: [] for k in W.LOG_FIELDS}
        for kind, post in runner.posts:
            if kind == "model" and "rule_rows" in post:
                for k in W.LOG_FIELDS:
                    rows[k] += post["rule_rows"][k]
        meta = {}
        if os.path.isfile(mine.rule_log):
            with np.load(mine.rule_log, allow_pickle=False) as z:
                meta = {k: z[k] for k in z.files if k not in W.LOG_FIELDS}
        meta["preroll_steps"] = np.asarray(runner.P)
        cols = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rows.items()}
        pre = os.path.join(folder, "rule_log_pre_" + run + ".npz")
        with open(pre, "wb") as fh:
            np.savez(fh, **cols, **meta)
        print("wrote", pre, flush=True)
    off = os.path.join(folder, "preroll_offsets_" + run + ".npz")
    with open(off, "wb") as fh:
        np.savez(fh, kind=np.asarray(runner.off["kind"]), ci=np.asarray(runner.off["ci"], np.int64), t0=np.asarray(runner.off["t0"], np.int64),
                 dpos=np.asarray(runner.off["dpos"], np.float64), dpsi_deg=np.asarray(runner.off["dpsi_deg"], np.float64), dv=np.asarray(runner.off["dv"], np.float64))
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
    path = os.path.join(folder, "preroll_log_" + run + ".json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", path, flush=True)


'''

MAIN = r'''def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        return _main_inner(argv)
    finally:
        finish_preroll()


if __name__ == "__main__":
    sys.exit(main())
'''


def once(text, old, new):
    n = text.count(old)
    assert n == 1, "expected exactly 1 match, got %d for: %r" % (n, old[:90])
    return text.replace(old, new)


def main():
    if os.path.exists(OUT):
        sys.exit("REFUSED: %s exists" % OUT)
    t = open(SRC, encoding="utf-8", newline="").read()
    assert "\r" not in t
    t = once(t, PARENT_HEAD, HEADER + "Header of the parent:\n" + PARENT_HEAD[3:])
    t = once(t, "def arg_value(argv, flag):\n", PURE + GLUE + "def arg_value(argv, flag):\n")
    t = once(t, "            p0 = np.asarray(s.track[2 * int(s.t0)], np.float64)\n",
             "            p0 = np.asarray(s.track[2 * int(s.t0)] if getattr(s, \"frame_pose5\", None) is None else s.frame_pose5, np.float64)\n")
    t = once(t, "        _RULE_PLANNER.__init__(self, base, model, mode, m, fk, label=label, log=log, floor=floor)\n",
             "        _RULE_PLANNER.__init__(self, base, model, mode, m, fk, label=label, log=log, floor=floor)\n"
             "        if STATE.get(\"preroll\", 0) > 0:\n            self.fk = FrameFK(self.fk, self)\n")
    t = once(t, "    ap.add_argument(\"--rule-every\", type=int, default=10)\n",
             "    ap.add_argument(\"--rule-every\", type=int, default=10)\n"
             "    ap.add_argument(\"--preroll-steps\", type=int, default=0)\n"
             "    ap.add_argument(\"--preroll-exec\", choices=(\"plan\", \"logged\"), default=\"plan\")\n")
    t = once(t, "    STATE[\"m\"] = int(a.replan_steps)\n", "    STATE[\"m\"] = int(a.replan_steps)\n    install_preroll(T, mine, a)\n")
    t = once(t, "    log = W.RuleLog()\n", "    log = W.RuleLog()\n    STATE[\"rulelog\"] = log\n")
    t = once(t, "def main(argv=None):\n    argv = list(sys.argv[1:] if argv is None else argv)\n", "def _main_inner(argv=None):\n    argv = list(sys.argv[1:] if argv is None else argv)\n")
    t = once(t, "\n\nif __name__ == \"__main__\":\n    sys.exit(main())\n", "\n\n" + MAIN)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(t)
    print("wrote", OUT, hashlib.sha256(t.encode("utf-8")).hexdigest()[:16], t.count("\n"), "lines")


if __name__ == "__main__":
    main()

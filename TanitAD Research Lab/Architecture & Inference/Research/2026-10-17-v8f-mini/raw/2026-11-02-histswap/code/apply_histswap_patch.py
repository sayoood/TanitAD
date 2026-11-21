"""Apply the HISTSWAP patch (--histswap {off,logged}) to the FEAS2-T1 tool (sha256 4b66cc86...). One-shot; each replacement must match exactly once.
Usage: python -I apply_histswap_patch.py <feas2_t1 v8f_t1_eval.py> <out v8f_t1_eval_histswap.py>
``--histswap logged`` (H-HISTSWAP-1): inside the T1-kin rollout, the model's ego-history INPUT at each tick (every key built from the ego rows <= j: p10f, base, kin3, pose, pose_now / pose_t / pose_s, and the
causal speed bucket) is read from the LOGGED clip arrays at that tick instead of the executed ones. The plan is still committed to the EXECUTED state; nav, scene, ages, the smoother S and the P10 floor are
untouched. The tick file's ``v0`` stays the EXECUTED measured speed (the quantity every reader defines it as); the speed the network read is ``v0_in``. ORACLE diagnostic, not deployable (the logged history after
t0 is the human's). ``off`` (default) is bit-identical to the FEAS2 tool; the flag refuses to combine with ``--feasible`` (not part of the card)."""
import hashlib
import sys

SRC_SHA = "4b66cc86cc5afe185736a65000c7f8cb1aa0c89d22bcd32770c99ad7971596df"
src, dst = sys.argv[1], sys.argv[2]
b = open(src, "rb").read()
assert hashlib.sha256(b).hexdigest() == SRC_SHA, "REFUSED: the source is not the FEAS2-T1 tool 4b66cc86"
s = b.decode("utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


# 1. docstring
rep('''Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""''', '''``--histswap {off,logged}`` (HISTSWAP, H-HISTSWAP-1; this file is ``v8f_t1_eval_histswap.py``): ``logged`` hands the model, at EVERY tick, its ego-history input from the LOGGED clip arrays (rows <= the tick's j;
every key ``ds.item`` builds from the ego rows -- ``p10f``, ``base``, ``kin3``, ``pose``, ``pose_now`` / ``pose_t`` / ``pose_s`` -- and the causal speed bucket) instead of from the executed ones; the plan is
still committed to the EXECUTED state, and ``nav`` (dead-reckoned from the executed motion), the camera stream, the ages, ``--smooth`` and the P10 floor are untouched. It is an ORACLE diagnostic (the logged
history after t0 is the human's, not deployable). Under it the ticks file's ``v0`` stays the EXECUTED measured speed (the quantity ``feas2`` / ``kappa`` readers define) and ``v0_in`` is the speed the network read;
the S head's ``s_pose`` / ``t_pose`` are the logged poses it read. ``off`` (default) and the flag absent are bit-identical to the FEAS2 tool; it refuses to combine with ``--feasible`` (``tests/test_v8f_t1_histswap.py``).

Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""''')

# 2. constants
rep('''SG_WINDOW, SG_ORDER = 7, 2
''', '''SG_WINDOW, SG_ORDER = 7, 2
#: HISTSWAP (PREREG_HISTSWAP_FROZEN.md): ``logged`` = the model's ego-history input is the LOGGED history at the tick; ``off`` = the executed one (FEAS2's behaviour, bit for bit).
HISTSWAP_MODES = ("off", "logged")
''')

# 3. WindowState: the logged view the model reads under the swap
rep('''    def __init__(self, base: dict, track, nav_rows, ci: int, t0: int, nav_mode: str = "deadreckon"):
        if nav_mode not in NAV_MODES:''', '''    def __init__(self, base: dict, track, nav_rows, ci: int, t0: int, nav_mode: str = "deadreckon", hist_swap: bool = False):
        if nav_mode not in NAV_MODES:''')
rep('''        self.scene0 = None
        self.clipped = 0                  # ticks whose plan had a step beyond the envelope (or a non-finite value)
''', '''        self.scene0 = None
        self.clipped = 0                  # ticks whose plan had a step beyond the envelope (or a non-finite value)
        self.hist_swap = bool(hist_swap)
        self.d_in = base if self.hist_swap else d     # the dict the ITEM BUILDER reads: the executed view ``d`` (default) or the LOGGED clip ``base`` (HISTSWAP; read-only, never written)
''')

# 4. build_item reads st.d_in; the causal bucket too; swap statistics
rep('''    proxy.d = st.d
    it = ds.item(st.ci, st.j, a_t, a_s)''', '''    proxy.d = st.d_in
    it = ds.item(st.ci, st.j, a_t, a_s)''')
rep('''        it["limit"], it["limit_ok"] = causal_bucket(st.d["v"], st.j, CAUSAL_STEPS if limit == "causal" else None)
        if stats is not None:''', '''        it["limit"], it["limit_ok"] = causal_bucket(st.d_in["v"], st.j, CAUSAL_STEPS if limit == "causal" else None)
        if stats is not None and st.d_in is not st.d:                    # HISTSWAP only: how often the swapped bucket differs from the executed-history bucket
            ex, ex_ok = causal_bucket(st.d["v"], st.j, CAUSAL_STEPS if limit == "causal" else None)
            stats["swap_ticks"] = stats.get("swap_ticks", 0) + 1
            stats["swap_bucket_differs"] = stats.get("swap_bucket_differs", 0) + int(bool(ex_ok) != bool(it["limit_ok"]) or (bool(ex_ok) and int(ex.argmax()) != int(it["limit"].argmax())))
        if stats is not None:''')

# 5. make_states
rep('''def make_states(store, tracks, J, wins, nav_mode):
    out = []''', '''def make_states(store, tracks, J, wins, nav_mode, hist_swap: bool = False):
    out = []''')
rep('''        out.append(WindowState(base, tracks[clip], J["nav"][o:o + n], ci, t0, nav_mode))''', '''        out.append(WindowState(base, tracks[clip], J["nav"][o:o + n], ci, t0, nav_mode, hist_swap))''')

# 6. TickRecorder: v0 stays EXECUTED under the swap; v0_in is what the network read
rep('''    def __init__(self, planner, store, J, heads: bool = False, smooth: str = "off", feasible: str = "off"):
        self.planner, self.store, self.J, self.heads, self.smooth, self.feasible = planner, store, J, bool(heads), smooth, feasible
        self.rows = {k: [] for k in self.FIELDS + (self.RAW_HEAD_FIELDS if self.heads else ()) + (("plan_raw",) if (smooth != "off" or feasible != "off") else ())
                     + (("plan_s",) if feasible != "off" else ())}''', '''    def __init__(self, planner, store, J, heads: bool = False, smooth: str = "off", feasible: str = "off", histswap: str = "off"):
        self.planner, self.store, self.J, self.heads, self.smooth, self.feasible, self.histswap = planner, store, J, bool(heads), smooth, feasible, histswap
        self.rows = {k: [] for k in self.FIELDS + (self.RAW_HEAD_FIELDS if self.heads else ()) + (("plan_raw",) if (smooth != "off" or feasible != "off") else ())
                     + (("plan_s",) if feasible != "off" else ()) + (("v0_in",) if histswap != "off" else ())}''')
rep('''        v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()             # the measured ego speed the prior (and the gate) read: exactly p10f[:, -1, 0]
''', '''        v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()             # the measured ego speed the prior (and the gate) read: exactly p10f[:, -1, 0]
        v0_in = v0
        if self.histswap != "off":                                       # HISTSWAP: v0 stays the EXECUTED measured speed (what Y's v0 is); v0_in is the speed the network read
            v0 = np.array([float(np.nan_to_num(st.d["p10f"][st.j][0], nan=0.0)) for st in states], np.float64)
''')
rep('''            r["gt"].append(gt); r["gt_ok"].append(ok); r["v0"].append(v0[b]); r["pose"].append(pose_now[:4])''', '''            r["gt"].append(gt); r["gt_ok"].append(ok); r["v0"].append(v0[b]); r["pose"].append(pose_now[:4])
            if self.histswap != "off":
                r["v0_in"].append(v0_in[b])''')

# 7. parser
rep('''    p.add_argument("--dump-heads", action="store_true",''', '''    p.add_argument("--histswap", choices=HISTSWAP_MODES, default="off", help="HISTSWAP (H-HISTSWAP-1): logged = the model's ego-history input at every tick (p10f, base, kin3, pose*, the causal speed bucket) is read from the LOGGED clip arrays instead of the executed ones; the plan is still committed to the executed state; nav, scene, ages, --smooth and the P10 floor untouched. ORACLE diagnostic, not deployable. off (default) = unchanged bit for bit; refuses to combine with --feasible")
    p.add_argument("--dump-heads", action="store_true",''')

# 8. cmd_run
rep('''    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])''', '''    if a.histswap != "off" and a.feasible != "off":
        raise SystemExit("REFUSED: --histswap logged is not defined together with --feasible (abox / kabox read the tick's input speed; the card has no such arm)")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])''')
rep('''        st_m = make_states(store, tracks, J, chunk, a.nav)
        if a.dump_ticks:
            rec = rec or TickRecorder(mp, store, J, heads=a.dump_heads, smooth=a.smooth, feasible=a.feasible)''', '''        st_m = make_states(store, tracks, J, chunk, a.nav, hist_swap=(a.histswap == "logged"))     # the P10 floor's states below are never swapped
        if a.dump_ticks:
            rec = rec or TickRecorder(mp, store, J, heads=a.dump_heads, smooth=a.smooth, feasible=a.feasible, histswap=a.histswap)''')
rep('''    if a.dump_ticks:                                                     # only then: flags absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        tick_path''', '''    if a.histswap != "off":                                              # only then: --histswap off / absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        header["histswap"] = {"mode": a.histswap, "applied": "the model's ego-history input at every tick comes from the LOGGED clip arrays; the plan is committed to the EXECUTED state; the P10 floor, nav, scene, ages and --smooth are untouched",
                              "oracle": "yes: the logged history after t0 is the human's; NOT deployable", "prereg": "dagland/histswap/PREREG_HISTSWAP_FROZEN.md (H-HISTSWAP-1)",
                              "bucket_stats_model_ticks": {k: CAUSAL_STATS.get(k, 0) for k in ("swap_ticks", "swap_bucket_differs")}}
    if a.dump_ticks:                                                     # only then: flags absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        tick_path''')
rep('''**({"feasible": a.feasible} if a.feasible != "off" else {})})''', '''**({"feasible": a.feasible} if a.feasible != "off" else {}),
                             **({"histswap": a.histswap} if a.histswap != "off" else {})})''')

open(dst, "wb").write(s.encode("utf-8"))
print("wrote", dst, hashlib.sha256(s.encode("utf-8")).hexdigest())

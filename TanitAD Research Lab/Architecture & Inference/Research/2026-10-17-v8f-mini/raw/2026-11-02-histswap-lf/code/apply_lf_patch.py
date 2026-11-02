"""Apply the HISTSWAP-LF patch (--histswap {off,logged,cv_exec,cv_logged}) to the HISTSWAP-T1 tool (sha256 7751673d...). One-shot; each replacement must match exactly once.
Usage: python -I apply_lf_patch.py <histswap_t1 v8f_t1_eval_histswap.py> <out v8f_t1_eval_histswap_lf.py>
``off`` and ``logged`` are the HISTSWAP tool's behaviour bit for bit. The two new values replace the model's ego-history input at EVERY tick by a CONSTANT-TWIST history (speed ``v`` and yaw rate
``omega`` held over every earlier row; zero acceleration; the executed pose at the tick is the anchor) built from the programme's one spelling (``yaw_kin_from_track`` / ``ego_channels_native`` /
``kin_features`` / ``kdx_base`` on a synthetic native track):
  cv_exec   (HCV,  deployable)  v = the EXECUTED measured speed at the tick;  omega = the EXECUTED yaw rate at the tick.
  cv_logged (HLVL, ORACLE)      v = the LOGGED  measured speed at the tick;  omega = the EXECUTED yaw rate at the tick  (the speed LEVEL is the only human information)."""
import hashlib
import sys

SRC_SHA = "7751673d17fcff763a940c1e49ba50038d216d0049482582cca2aec7c28af63f"
src, dst = sys.argv[1], sys.argv[2]
b = open(src, "rb").read()
assert hashlib.sha256(b).hexdigest() == SRC_SHA, "REFUSED: the source is not the HISTSWAP-T1 tool 7751673d"
s = b.decode("utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


# 1. docstring
rep('''
Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""''', '''
``--histswap {cv_exec,cv_logged}`` (HISTSWAP-LF, H-HISTSWAP-LF-1; this file is ``v8f_t1_eval_histswap_lf.py``; ``off`` and ``logged`` are the HISTSWAP tool's behaviour bit for bit): the model's ego-history
input at EVERY tick is a CONSTANT-TWIST history -- speed ``v`` and yaw rate ``omega`` held over every earlier row (zero acceleration, no trend), anchored at the EXECUTED pose of the tick -- built through
the programme's one spelling (``yaw_kin_from_track`` / ``ego_channels_native`` / ``kin_features`` / ``kdx_base`` on a synthetic native track) for the same keys ``logged`` swaps (``p10f``, ``base``, ``kin3``,
``pose``, ``pose_now`` / ``pose_t`` / ``pose_s``, the causal speed bucket); ``nav`` (dead-reckoned), the camera stream, the ages, ``--smooth`` and the P10 floor are untouched, and the plan is committed to the
EXECUTED state. ``cv_exec`` (HCV, deployable): ``v`` = the EXECUTED measured speed at the tick. ``cv_logged`` (HLVL, ORACLE, level only): ``v`` = the LOGGED measured speed at the tick (the human's level, no
trend). ``omega`` is the EXECUTED yaw rate in both. Needs ``--limit causal``; refuses ``--feasible``. The ticks file's ``v0`` stays the EXECUTED speed, ``v0_in`` is the speed the network read.

Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""''')

# 2. constants + mode parser
rep('''HISTSWAP_MODES = ("off", "logged")
''', '''HISTSWAP_MODES = ("off", "logged", "cv_exec", "cv_logged")
#: HISTSWAP-LF (PREREG_HISTSWAP_LF_FROZEN.md): a constant-twist history at the executed (HCV) or the logged (HLVL) current speed.
CV_MODES = ("cv_exec", "cv_logged")
#: rows [j - CV_ROWS, j] of the item builder's view are rebuilt: ``ds.item`` reads j - (hmax - 1) .. j, j - a_s (a_s <= 6) and the causal bucket j - CAUSAL_STEPS (20) .. j; ``cv_view`` asserts the bound.
CV_ROWS = 32


def hist_mode_of(x) -> str:
    """The history mode of a ``hist_swap`` argument: False / None / "off" -> off; True / "logged" -> logged (HISTSWAP's bool contract); "cv_exec" / "cv_logged" -> itself; anything else raises."""
    if x is None or x is False or x == "off":
        return "off"
    if x is True or x == "logged":
        return "logged"
    if x in CV_MODES:
        return str(x)
    raise ValueError(f"hist_swap must be one of {HISTSWAP_MODES} (or a bool), got {x!r}")
''')

# 3. module-level helpers (before WindowState)
rep('''class WindowState:
''', '''def cv_motion_track(pose_j, v: float, omega: float, n_back: int) -> np.ndarray:
    """``[2 * n_back + 1, 4]`` native (0.1 s) rows (x, y, yaw, v) of the CONSTANT-TWIST history that ENDS at ``pose_j`` = (x, y, yaw): speed ``v`` and yaw rate ``omega`` (rad/s) held on every earlier
    row (the exact circular arc; a straight line when |omega| < 1e-9). The last row is ``pose_j`` with ``v``; the earlier rows are the arc travelled backwards."""
    n = 2 * int(n_back) + 1
    x0, y0, p0 = float(pose_j[0]), float(pose_j[1]), float(pose_j[2])
    tau = (np.arange(n, dtype=np.float64) - (n - 1)) * D.DT_NATIVE                  # seconds before the tick, <= 0
    psi = p0 + omega * tau
    if abs(omega) < 1e-9:
        x, y = x0 + v * tau * math.cos(p0), y0 + v * tau * math.sin(p0)
    else:
        x, y = x0 + (v / omega) * (np.sin(psi) - math.sin(p0)), y0 + (v / omega) * (math.cos(p0) - np.cos(psi))
    return np.stack([x, y, psi, np.full(n, float(v))], 1)


def channels_from_track(sl) -> dict:
    """The rows ``WindowState._finish`` writes for steps 1 .. n of a native slice ``sl`` of ``2 n + 1`` rows, by the SAME calls: ``kin3``, ``ch1`` (omega), ``p10f``, ``base``, ``pose`` (4 columns), ``a``, ``v``.
    (Re-spelled here so that ``_finish`` stays byte-identical to the HISTSWAP tool; ``tests/test_v8f_t1_histswap_lf.py`` pins the two to equal exactly on a real executed slice.)"""
    sl = np.asarray(sl, np.float64)
    kin, om = D.yaw_kin_from_track(sl)
    ch = D.ego_channels_native(sl)[0::2]
    n = len(kin) - 1
    memo = {}
    base = []
    for s in range(1, n + 1):
        key = kin[s].tobytes()
        if key not in memo:
            memo[key] = D.kdx_base(kin[s:s + 1], D.KIN_UNITS["yaw"])[0]
        base.append(memo[key])
    return {"kin3": kin[1:], "ch1": np.nan_to_num(om[1:], nan=0.0), "p10f": np.stack([fk.kin_features(kin[s:s + 1])[0].astype(np.float32) for s in range(1, n + 1)]),
            "base": np.stack(base), "pose": sl[0::2][1:], "a": ch[1:, 1], "v": sl[0::2][1:, 3]}


class WindowState:
''')

# 4. WindowState: mode, logged reference
rep('''nav_mode: str = "deadreckon", hist_swap: bool = False):''', '''nav_mode: str = "deadreckon", hist_swap=False):''')
rep('''        self.hist_swap = bool(hist_swap)
        self.d_in = base if self.hist_swap else d     # the dict the ITEM BUILDER reads: the executed view ``d`` (default) or the LOGGED clip ``base`` (HISTSWAP; read-only, never written)
''', '''        self.hist_mode = hist_mode_of(hist_swap)
        self.hist_swap = self.hist_mode != "off"
        self.base_log = base                                              # the LOGGED clip (read-only, never written): the source of the logged speed level (cv_logged) and of ``logged``
        self.d_in = base if self.hist_mode == "logged" else d             # the dict the ITEM BUILDER reads: the executed view ``d`` (off), the LOGGED clip ``base`` (logged), or, under cv_*, the per-tick :meth:`cv_view`
''')

# 5. cv_view method
rep('''    def executed_ego(self) -> np.ndarray:''', '''    def cv_view(self, limit: str = "causal") -> dict:
        """HISTSWAP-LF: the dict the item builder reads at this tick under ``cv_exec`` / ``cv_logged`` -- a copy of the EXECUTED view ``d`` whose rows [j - CV_ROWS, j] of every ego key are the CONSTANT-TWIST
        history at speed ``v`` (cv_exec: the executed ``p10f[j][0]``; cv_logged: the LOGGED ``p10f[j][0]``) and yaw rate ``omega`` = the executed ``ch1[j]``, anchored at the executed pose, built by
        :func:`channels_from_track` on :func:`cv_motion_track`. Nothing is written to ``d`` or to the logged clip."""
        if self.hist_mode not in CV_MODES:
            raise RuntimeError("cv_view is defined for cv_exec / cv_logged only")
        if limit == "causal_all":
            raise ValueError("REFUSED: cv_exec / cv_logged are defined with --limit causal (the bucket reads rows [j - 20, j])")
        j, d = self.j, self.d
        src = d if self.hist_mode == "cv_exec" else self.base_log
        v = float(src["p10f"][j][0])
        om = float(np.nan_to_num(d["ch1"][j], nan=0.0))
        pj = np.asarray(d["pose"][j], np.float64)
        if not (math.isfinite(v) and bool(np.isfinite(pj[:3]).all())):
            raise ValueError(f"{self.hist_mode}: the current speed / pose is not finite at step {j} (v={v!r})")
        lo = max(1, j - CV_ROWS)
        assert CV_ROWS >= max(CAUSAL_STEPS, 9, 6), "CV_ROWS must cover every row the item builder reads"
        ch = channels_from_track(cv_motion_track(pj, v, om, j - (lo - 1)))
        out = dict(d)
        for k in self.EGO:
            out[k] = np.array(d[k], copy=True)
        out["kin3"][lo:j + 1], out["ch1"][lo:j + 1], out["p10f"][lo:j + 1], out["base"][lo:j + 1] = ch["kin3"], ch["ch1"], ch["p10f"], ch["base"]
        out["pose"][lo:j + 1], out["a"][lo:j + 1] = ch["pose"], ch["a"]
        out["pose"][j] = d["pose"][j]                                     # the anchor IS the executed pose (the cameras, the nav and every target live in this frame)
        out["v"][max(0, j - CV_ROWS):j + 1] = v                           # the causal bucket reads d["v"][max(0, j - 20): j + 1]
        return out

    def executed_ego(self) -> np.ndarray:''')

# 6. build_item: the per-tick view
rep('''    proxy.d = st.d_in
    it = ds.item(st.ci, st.j, a_t, a_s)''', '''    if st.hist_mode in CV_MODES:                                         # HISTSWAP-LF: the view is rebuilt at every tick from the tick's executed (cv_exec) / logged (cv_logged) speed
        st.d_in = st.cv_view(limit)
    proxy.d = st.d_in
    it = ds.item(st.ci, st.j, a_t, a_s)''')

# 7. make_states signature
rep('''def make_states(store, tracks, J, wins, nav_mode, hist_swap: bool = False):''', '''def make_states(store, tracks, J, wins, nav_mode, hist_swap=False):''')

# 8. parser help
rep('''default="off", help="HISTSWAP (H-HISTSWAP-1): logged = ''', '''default="off", help="HISTSWAP-LF: cv_exec = a constant-twist (constant-velocity) history at the EXECUTED current speed (deployable); cv_logged = the same at the LOGGED current speed (ORACLE, level only); both need --limit causal. HISTSWAP (H-HISTSWAP-1): logged = ''')

# 9. cmd_run: refusals and the state builder
rep('''        raise SystemExit("REFUSED: --histswap logged is not defined together with --feasible (abox / kabox read the tick's input speed; the card has no such arm)")
''', '''        raise SystemExit("REFUSED: --histswap logged is not defined together with --feasible (abox / kabox read the tick's input speed; the card has no such arm)")
    if a.histswap in CV_MODES and a.limit != "causal":
        raise SystemExit(f"REFUSED: --histswap {a.histswap} is defined with --limit causal only (got {a.limit})")
''')
rep('''hist_swap=(a.histswap == "logged"))     # the P10 floor's states below are never swapped''', '''hist_swap=a.histswap)     # the P10 floor's states below are never swapped''')

# 10. header
rep('''    if a.dump_ticks:                                                     # only then: flags absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        tick_path''', '''    if a.histswap in CV_MODES:                                           # LF: the header says what the history IS
        header["histswap"].update({"applied": "the model's ego-history input at every tick is a CONSTANT-TWIST history (speed and yaw rate held, zero acceleration) anchored at the EXECUTED pose; the plan is committed to the EXECUTED state; "
                                              "the P10 floor, nav, scene, ages and --smooth are untouched",
                                   "speed_source": "the EXECUTED measured speed at the tick (deployable)" if a.histswap == "cv_exec" else "the LOGGED measured speed at the tick (ORACLE, level only: the human's speed, no trend)",
                                   "oracle": "no" if a.histswap == "cv_exec" else "yes: the logged speed after t0 is the human's; NOT deployable",
                                   "prereg": "dagland/histswap_lf/PREREG_HISTSWAP_LF_FROZEN.md (H-HISTSWAP-LF-1)"})
    if a.dump_ticks:                                                     # only then: flags absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        tick_path''')

open(dst, "wb").write(s.encode("utf-8"))
print("wrote", dst, hashlib.sha256(s.encode("utf-8")).hexdigest())

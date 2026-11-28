"""Apply the LFO patch (``--floor-vhat-source {heads,logged}``, PREREG_LEVELFLOOR_A4 sha256 8bfbc386...) to the LEVEL-FLOOR tool (sha256 6166178a..., the LF tool = RI tool 0417c838 + apply_lf_patch.py). One-shot; each
replacement must match exactly once.
Usage: python -I apply_lfo_patch.py <LF v8f_t1_eval_ri.py 6166178a> <out v8f_t1_eval_ri.py>     (the patched tool keeps the name v8f_t1_eval_ri.py inside the read tree)
``--floor-vhat-source heads`` (the default under ``--level-floor on``) is the LF tool's behaviour bit for bit: every new code path is behind ``vhat_source == "logged"``.
``--floor-vhat-source logged`` (a CEILING diagnostic, inadmissible at inference): the floor's level input v_hat is the LOGGED route speed of the tick, computed in the tool at the replan BEFORE F from the same ``gt_in_frame`` the
tick dump uses; no complete logged route (gt_ok False) gives no floor. The patched tool's sha256 is regenerable: run this script on the pinned LF tool."""
import hashlib
import sys

SRC_SHA = "6166178a604edc5343faa4fc6f139ec83dc65d00b91155a77b3f33dda65f4d70"
src, dst = sys.argv[1], sys.argv[2]
b = open(src, "rb").read()
assert hashlib.sha256(b).hexdigest() == SRC_SHA, "REFUSED: the source is not the LF tool 6166178a"
assert b"\r" not in b, "the source tool must be LF"
s = b.decode("utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


# 1. docstring ---------------------------------------------------------------------------------------------------------
rep('''(F's v0 would be the logged speed).

Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""''', '''(F's v0 would be the logged speed).

``--floor-vhat-source {heads,logged}`` (LFO, PREREG_LEVELFLOOR_A4 sha256 8bfbc386..., only with ``--level-floor on``; default ``heads`` = the LF tool bit for bit): ``logged`` replaces the floor's level input v_hat with the LOGGED route
speed v_log of the tick, a CEILING diagnostic that is NOT admissible at inference (a privileged channel). v_log = the path length of knots 1..6 of the tick's ``gt`` (the logged route in the executed ego frame, ``gt_in_frame``) / 1.0 s,
the reader's own ``logged_speed`` (``lf_read.py``); it is computed at the replan BEFORE F from the same ``gt_in_frame`` the tick dump uses (``logged_route_speed`` / ``make_logged_oracle``). A tick with no complete logged route
(``gt_ok`` False) has v_hat NaN and therefore NO floor. The presence and gap heads still come from the floor model (the lead cap is unchanged). ``logged`` is refused without ``--level-floor on``, and an explicit ``heads`` is refused with
the floor off. The result JSON header records ``floor_vhat_source`` and, for ``logged``, ``"label": "CEILING"``; the ticks file gains ``lf_vhat_heads`` (what the heads would have said on the same tick; ``lf_vhat`` is then v_log).

Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""''')

# 2. flag refusals + the oracle ------------------------------------------------------------------------------------------
rep('''    elif a.floor_kappa is not None or a.floor_ckpt:
        raise SystemExit("REFUSED: --floor-kappa / --floor-ckpt belong to --level-floor on; with the floor off they would silently do nothing")
''', '''    elif a.floor_kappa is not None or a.floor_ckpt:
        raise SystemExit("REFUSED: --floor-kappa / --floor-ckpt belong to --level-floor on; with the floor off they would silently do nothing")
    src_ = getattr(a, "floor_vhat_source", None)
    if a.level_floor != "on":
        if src_ is not None:
            raise SystemExit(f"REFUSED: --floor-vhat-source {src_} belongs to --level-floor on (the logged-speed oracle exists only inside the floor; with the floor off the flag would silently do nothing)")
    elif src_ is None:
        a.floor_vhat_source = "heads"


LOGGED_KNOTS = 6                                                                              # LFO: knots 1..6 of the tick's logged route = five 0.2 s segments = the route's first second


def logged_route_speed(gt, gt_ok):
    """The logged route's own speed at the tick over its first second: the path length of knots 1..6 of ``gt`` [N, 30, 2] (the logged route in the executed ego frame) / 1.0 s. The same arithmetic as the reader's
    ``logged_speed`` (motion_c/lf_read/code/lf_read.py). -> v [N] float64, NaN where the tick has no complete logged route (``gt_ok`` False or a non-finite value)."""
    gt = np.asarray(gt, np.float64)
    v = np.linalg.norm(np.diff(gt[:, :LOGGED_KNOTS], axis=1), axis=-1).sum(1) / ((LOGGED_KNOTS - 1) * DT)
    ok = np.asarray(gt_ok, bool) & np.isfinite(v)
    return np.where(ok, v, np.nan)


def make_logged_oracle(store, J):
    """LFO: ``oracle(states)`` -> v_log [B] float64 (NaN = no complete logged route), from the SAME ``gt_in_frame(store.get(clip)["pose"], pose_now, st.j)`` the TickRecorder dumps (``pose_now`` = the window's executed pose at the tick).
    CEILING: this reads the human's logged route, a privileged channel; it exists only behind ``--floor-vhat-source logged``."""
    def oracle(states):
        gts, oks = [], []
        for st in states:
            clip = str(J["clip"][st.ci])
            pose_now = np.array(st.d["pose"][st.j], np.float64)
            gt, ok = gt_in_frame(store.get(clip)["pose"], pose_now, st.j)
            gts.append(gt)
            oks.append(ok)
        return logged_route_speed(np.stack(gts), np.asarray(oks, bool))
    return oracle
''')

# 3. LevelFloor ----------------------------------------------------------------------------------------------------------
rep('''    def __init__(self, fm, kappa: float, ckpt_path: str, ckpt_sha256: str):
        self.fm, self.kappa, self.ckpt_path, self.ckpt_sha256 = fm, float(kappa), ckpt_path, ckpt_sha256
        self.op = _lf()
''', '''    def __init__(self, fm, kappa: float, ckpt_path: str, ckpt_sha256: str, vhat_source: str = "heads", oracle=None):
        self.fm, self.kappa, self.ckpt_path, self.ckpt_sha256 = fm, float(kappa), ckpt_path, ckpt_sha256
        if vhat_source not in ("heads", "logged"):
            raise ValueError(f"vhat_source must be 'heads' or 'logged', got {vhat_source!r}")
        self.vhat_source, self.oracle, self.states = vhat_source, oracle, None                      # LFO: the oracle and the tick's window states (set by model_planner before each call)
        self.op = _lf()
''')
rep('''"short_last_seg_binds": 0, "lift_sum_ms": 0.0}
''', '''"short_last_seg_binds": 0, "lift_sum_ms": 0.0}
        if self.vhat_source == "logged":
            self.stats["oracle_no_route"] = 0                                                       # ticks with no complete logged route = no floor
''')
rep('''    def __call__(self, batch, plan, v0):
        v_hat, logit, gap_hat = self.heads(batch)
        out, info = self.op.apply_floor_batch(plan, v0, v_hat, self.op.sigmoid(logit), gap_hat, self.kappa)
        st = self.stats
''', '''    def oracle_vhat(self, n):
        """LFO: v_log [n] for the ticks of the current ``states``."""
        if self.oracle is None or self.states is None or len(self.states) != n:
            raise RuntimeError(f"--floor-vhat-source logged: no oracle or no window states for this call (oracle {self.oracle is not None}, states {None if self.states is None else len(self.states)}, batch {n})")
        v_log = np.asarray(self.oracle(self.states), np.float64)
        if v_log.shape != (n,):
            raise RuntimeError(f"the logged-speed oracle returned {v_log.shape}, expected ({n},)")
        return v_log

    def __call__(self, batch, plan, v0):
        v_hat, logit, gap_hat = self.heads(batch)
        if self.vhat_source == "logged":                                                             # LFO, CEILING: the level input is the logged route speed
            v_heads = v_hat
            v_hat = self.oracle_vhat(len(v_heads))
        out, info = self.op.apply_floor_batch(plan, v0, v_hat, self.op.sigmoid(logit), gap_hat, self.kappa)
        if self.vhat_source == "logged":
            info["vhat_heads"] = np.asarray(v_heads, np.float64)
            self.stats["oracle_no_route"] += int((~np.isfinite(v_hat)).sum())
        st = self.stats
''')
rep('''        return {"level_floor": "on", "floor_kappa": self.kappa, "floor_ckpt_sha256": self.ckpt_sha256}
''', '''        return {"level_floor": "on", "floor_kappa": self.kappa, "floor_ckpt_sha256": self.ckpt_sha256, "floor_vhat_source": self.vhat_source, **({"label": "CEILING"} if self.vhat_source == "logged" else {})}
''')
rep('''        return {"operator": self.op.OPERATOR_TEXT, "constants": self.op.CONSTANTS,''', '''        d = {"operator": self.op.OPERATOR_TEXT, "constants": self.op.CONSTANTS,''')
rep('''"heads_parity_vs_forward_infer": self.parity, "counts": st}
''', '''"heads_parity_vs_forward_infer": self.parity, "counts": st}
        d["floor_vhat_source"] = self.vhat_source
        if self.vhat_source == "logged":
            d["label"] = "CEILING"
            d["prereg"] = "dagland/motion_c/a4_prereg/PREREG_LEVELFLOOR_A4.md (sha256 8bfbc386...), H-LEVELFLOOR-1 (A4)"
            d["vision_only"] = ("NO. CEILING diagnostic, inadmissible at inference: v_hat := the LOGGED route speed of the tick (path length of knots 1..6 of the tick's logged route in the executed ego frame / 1.0 s, a privileged channel); "
                                "presence and gap_hat still come from the floor heads; a tick with no complete logged route has no floor")
            d["level_input"] = "logged_route_speed(gt_in_frame(...)) computed at the replan before F (floor_vhat_source logged); lf_vhat_heads in the ticks file holds what the heads said on the same tick"
        return d
''')
rep('''    return LevelFloor(fm, a.floor_kappa, a.floor_ckpt, h.hexdigest())
''', '''    return LevelFloor(fm, a.floor_kappa, a.floor_ckpt, h.hexdigest(), vhat_source=getattr(a, "floor_vhat_source", None) or "heads")
''')

# 4. model_planner -------------------------------------------------------------------------------------------------------
rep('''            plan_f, plan.floor_info = floor(batch, plan_s, v0)
''', '''            if getattr(floor, "vhat_source", "heads") == "logged":       # LFO: the oracle reads the tick's window states (set by rollout)
                floor.states = plan.states
            plan_f, plan.floor_info = floor(batch, plan_s, v0)
''')
rep('''    plan.floor_info = None
    return plan
''', '''    plan.floor_info = None
    plan.states = None
    return plan
''')

# 5. rollout -------------------------------------------------------------------------------------------------------------
rep('''        plan = planner(batch)
        if plan0 is None:
''', '''        if hasattr(planner, "states"):                                   # LFO: the logged-speed oracle reads these window states (CEILING; unused unless --floor-vhat-source logged)
            planner.states = states
        plan = planner(batch)
        if plan0 is None:
''')

# 6. TickRecorder ---------------------------------------------------------------------------------------------------------
rep('''    HEAD_FIELDS = RAW_HEAD_FIELDS + ("s_ckpt_now", "s_ckpt_w0")''', '''    LFO_FIELDS = ("lf_vhat_heads",)                                                                                                 # LFO (floor_vhat_source logged): what the floor heads said on the same tick
    HEAD_FIELDS = RAW_HEAD_FIELDS + ("s_ckpt_now", "s_ckpt_w0")''')
rep('histswap: str = "off", level_floor: bool = False):\n', 'histswap: str = "off", level_floor: bool = False, floor_oracle: bool = False):\n')
rep('        self.level_floor = bool(level_floor)\n', '        self.level_floor = bool(level_floor)\n        self.floor_oracle = bool(floor_oracle) and self.level_floor\n')
rep('+ (self.LF_FIELDS if level_floor else ())}', '+ (self.LF_FIELDS if level_floor else ()) + (self.LFO_FIELDS if (level_floor and floor_oracle) else ())}')
rep('''                    r["lf_" + k_].append(np.float32(fi[k_][b]))
''', '''                    r["lf_" + k_].append(np.float32(fi[k_][b]))
                if self.floor_oracle:
                    r["lf_vhat_heads"].append(np.float32(fi["vhat_heads"][b]))
''')

# 7. argparse -------------------------------------------------------------------------------------------------------------
rep('    p.add_argument("--dump-heads", action="store_true", help="T1-HEADS', '''    p.add_argument("--floor-vhat-source", choices=("heads", "logged"), default=None, help="LFO (PREREG_LEVELFLOOR_A4): the floor's level input v_hat. heads (the default with --level-floor on) = the floor heads' estimate; logged = the LOGGED route speed of the tick (a CEILING diagnostic, NOT admissible at inference); refused without --level-floor on")
    p.add_argument("--dump-heads", action="store_true", help="T1-HEADS''')

# 8. cmd_run --------------------------------------------------------------------------------------------------------------
rep('''    store = D.ClipStore(a.eval_cache, "eval", lru=a.lru, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
''', '''    store = D.ClipStore(a.eval_cache, "eval", lru=a.lru, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    if floor is not None and floor.vhat_source == "logged":              # LFO: the oracle reads the same store the tick dump reads
        floor.oracle = make_logged_oracle(store, J)
''')
rep('histswap=a.histswap, level_floor=floor is not None)', 'histswap=a.histswap, level_floor=floor is not None, floor_oracle=floor is not None and floor.vhat_source == "logged")')
rep('''        header["level_floor_detail"] = floor.detail()
''', '''        header["level_floor_detail"] = floor.detail()
        header["floor_vhat_source"] = floor.vhat_source
        if floor.vhat_source == "logged":
            header["label"] = "CEILING"
            header["open_channels"].append("LEVEL INPUT (CEILING, inadmissible at inference): the floor's v_hat is the LOGGED route speed of the tick (a privileged channel); no complete logged route = no floor")
''')

assert b"\r" not in s.encode("utf-8")
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("patched", dst, hashlib.sha256(s.encode("utf-8")).hexdigest())

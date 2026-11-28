"""Apply the KAPPA-T1 patch (--feasible {off,box}) to the SMOOTH-T1 tool (sha256 caa71055...). One-shot; each replacement must match exactly once.
Usage: python -I apply_feasible_patch.py <smooth_t1 v8f_t1_eval.py> <out v8f_t1_eval.py>"""
import hashlib
import sys

SRC_SHA = "caa71055f50f8d1bc300c329374af71abdd68ee7c41bef840cb60fdbe599b183"
src, dst = sys.argv[1], sys.argv[2]
b = open(src, "rb").read()
assert hashlib.sha256(b).hexdigest() == SRC_SHA, "REFUSED: the source is not the SMOOTH-T1 tool caa71055"
s = b.decode("utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


# 1. docstring
rep('''The P10 floor is never smoothed. Absent the flag (or ``off``) the output is unchanged bit for bit (``tests/test_v8f_t1_smooth.py``).
''', '''The P10 floor is never smoothed. Absent the flag (or ``off``) the output is unchanged bit for bit (``tests/test_v8f_t1_smooth.py``).

``--feasible {off,box}`` (KAPPA-T1, H-KAPPA-T1-1, default off): ``box`` projects the plan, AFTER ``--smooth``, onto the envelope of
``tanitad.refs.feasible_decode.project_feasible`` verbatim (box only: ``mu=None``, ``a_max`` = ``A_MAX_MPS2`` 4.0, ``kappa_max`` = ``KAPPA_MAX_1PM`` 0.2, the module's
default margin and stop epsilons, ``clamp_entry=False``) on the origin-prepended plan at the plan's own grid ``dt`` = ``DT`` (0.2 s), at EVERY replan, before the executor.
The projected plan is what the executor commits, what ``ol`` holds and what ``--dump-ticks`` calls ``plan``; the ticks file also carries ``plan_raw`` (the model's plan) and
``plan_s`` (the plan after ``--smooth``, before the projection). The P10 floor is never projected. Absent the flag (or ``off``) the output is unchanged bit for bit
(``tests/test_v8f_t1_feasible.py``).
''')

# 2. constants
rep('''SMOOTH_MODES = ("off", "sg72")
''', '''SMOOTH_MODES = ("off", "sg72")
#: KAPPA-T1 (PREREG H-KAPPA-T1-1, gate G-KAPPA outcome (a)): the envelope is feasible_decode's, NOT tuned here.
FEASIBLE_MODES = ("off", "box")
''')

# 3. the operator, after smooth_plan
rep('''    return (p64 + savgol_filter(plan64 - p64, SG_WINDOW, SG_ORDER, axis=1, mode="interp")).astype(np.asarray(plan).dtype)
''', '''    return (p64 + savgol_filter(plan64 - p64, SG_WINDOW, SG_ORDER, axis=1, mode="interp")).astype(np.asarray(plan).dtype)


def feasible_plan(plan, mode: str = "off"):
    """KAPPA-T1's curvature-bounded decode on a batch of plans [B, T, 2] (ego frame, knots 1..T; the origin is implied): ``box`` = ``project_feasible`` of
    ``tanitad.refs.feasible_decode`` on the origin-prepended path, float64, ``dt=DT``, the box envelope verbatim (``mu=None``, ``clamp_entry=False``), returned in the
    plan's dtype without the origin. ``off`` returns ``plan`` itself (the same object)."""
    if mode == "off":
        return plan
    if mode != "box":
        raise ValueError(f"--feasible must be one of {FEASIBLE_MODES}, got {mode!r}")
    import torch
    from tanitad.refs import feasible_decode as FD
    p64 = np.asarray(plan, np.float64)
    if p64.ndim != 3 or p64.shape[-1] != 2 or p64.shape[1] < 3:
        raise ValueError(f"feasible_plan: plan {p64.shape} must be [B, T>=3, 2]")
    full = torch.from_numpy(np.concatenate([np.zeros_like(p64[:, :1]), p64], 1))
    out = FD.project_feasible(full, dt=DT, a_max=FD.A_MAX_MPS2, kappa_max=FD.KAPPA_MAX_1PM, mu=None, clamp_entry=False)
    return out[:, 1:].numpy().astype(np.asarray(plan).dtype)
''')

# 4. the planner
rep('''def model_planner(model, device, keep_aux: bool = False, keep_heads: bool = False, smooth: str = "off"):''',
    '''def model_planner(model, device, keep_aux: bool = False, keep_heads: bool = False, smooth: str = "off", feasible: str = "off"):''')
rep('''        raw = out["traj30"].float().cpu().numpy()
        if smooth == "off":
            return raw
        plan.plan_raw = raw
''', '''        raw = out["traj30"].float().cpu().numpy()
        if smooth == "off" and feasible == "off":
            return raw
        plan.plan_raw = raw
        if feasible != "off":                                            # KAPPA-T1: the projection AFTER S (S is the identity when --smooth off)
            plan.plan_s = smooth_plan(raw, out["p10"].float().cpu().numpy(), smooth)
            return feasible_plan(plan.plan_s, feasible)
''')
rep('''    plan.plan_raw = None
    return plan
''', '''    plan.plan_raw = None
    plan.plan_s = None
    return plan
''')

# 5. the tick recorder
rep('''    def __init__(self, planner, store, J, heads: bool = False, smooth: str = "off"):
        self.planner, self.store, self.J, self.heads, self.smooth = planner, store, J, bool(heads), smooth
        self.rows = {k: [] for k in self.FIELDS + (self.RAW_HEAD_FIELDS if self.heads else ()) + (("plan_raw",) if smooth != "off" else ())}
''', '''    def __init__(self, planner, store, J, heads: bool = False, smooth: str = "off", feasible: str = "off"):
        self.planner, self.store, self.J, self.heads, self.smooth, self.feasible = planner, store, J, bool(heads), smooth, feasible
        self.rows = {k: [] for k in self.FIELDS + (self.RAW_HEAD_FIELDS if self.heads else ()) + (("plan_raw",) if (smooth != "off" or feasible != "off") else ())
                     + (("plan_s",) if feasible != "off" else ())}
''')
rep('''            if self.smooth != "off":                                     # only then: the pre-S plan beside the issued (smoothed) one
                r["plan_raw"].append(np.asarray(self.planner.plan_raw[b], np.float64))
''', '''            if self.smooth != "off" or self.feasible != "off":          # only then: the pre-S plan beside the issued (smoothed) one
                r["plan_raw"].append(np.asarray(self.planner.plan_raw[b], np.float64))
            if self.feasible != "off":                                   # KAPPA-T1: the plan after S, before the projection
                r["plan_s"].append(np.asarray(self.planner.plan_s[b], np.float64))
''')

# 6. the flag
rep('''    p.add_argument("--smooth", choices=SMOOTH_MODES, default="off",''',
    '''    p.add_argument("--feasible", choices=FEASIBLE_MODES, default="off", help="KAPPA-T1: box = feasible_decode.project_feasible (box envelope A_MAX 4.0, KAPPA_MAX 0.2, mu=None, clamp_entry=False, dt=DT) on the plan AFTER --smooth, at every replan, before the executor; off (default) = unchanged bit for bit")
    p.add_argument("--smooth", choices=SMOOTH_MODES, default="off",''')

# 7. the run
rep('''        mp = model_planner(model, dev, keep_aux=bool(a.dump_ticks), keep_heads=bool(a.dump_heads), smooth=a.smooth)''',
    '''        mp = model_planner(model, dev, keep_aux=bool(a.dump_ticks), keep_heads=bool(a.dump_heads), smooth=a.smooth, feasible=a.feasible)''')
rep('''            rec = rec or TickRecorder(mp, store, J, heads=a.dump_heads, smooth=a.smooth)''',
    '''            rec = rec or TickRecorder(mp, store, J, heads=a.dump_heads, smooth=a.smooth, feasible=a.feasible)''')
rep('''    if a.dump_ticks:                                                     # only then: flags absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        tick_path''', '''    if a.feasible != "off":                                              # only then: --feasible off / absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        header["feasible"] = {"mode": a.feasible, "operator": "tanitad.refs.feasible_decode.project_feasible(origin + plan, dt=DT, a_max=A_MAX_MPS2, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False)",
                              "prereg": "dagland/kappa_t1/PREREG_FROZEN.md (H-KAPPA-T1-1)", "applied": "to the plan after --smooth at every replan, before the executor; the P10 floor is not projected"}
    if a.dump_ticks:                                                     # only then: flags absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        tick_path''')
rep('''                             **({"smooth": a.smooth} if a.smooth != "off" else {})})''',
    '''                             **({"smooth": a.smooth} if a.smooth != "off" else {}), **({"feasible": a.feasible} if a.feasible != "off" else {})})''')
open(dst, "wb").write(s.encode("utf-8"))
print("wrote", dst, hashlib.sha256(s.encode("utf-8")).hexdigest())

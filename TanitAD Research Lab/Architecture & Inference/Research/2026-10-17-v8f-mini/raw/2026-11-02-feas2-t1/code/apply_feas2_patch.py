"""Apply the FEAS2-T1 patch (--feasible {off,box,kbox,abox,kabox}) to the KAPPA-T1 tool (sha256 260bd175...). One-shot; each replacement must match exactly once.
Usage: python -I apply_feas2_patch.py <kappa_t1 v8f_t1_eval.py> <out v8f_t1_eval.py>
``box`` is untouched (bit-identical to the KAPPA-T1 tool; tests/test_v8f_t1_feas2.py). The three new modes are the card's (PREREG_FROZEN.md):
  kbox   project_feasible(a_max=inf, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False)
  abox   project_feasible(kappa_max=inf, a_max=A_MAX_MPS2, mu=None, clamp_entry=True, v0 = the tick's measured ego speed batch["p10f"][:, -1, 0])
  kabox  kbox, then abox
all on the origin-prepended plan after --smooth, at the plan's own grid dt = DT, at every replan, before the executor."""
import hashlib
import sys

SRC_SHA = "260bd17505726a297afe4691fafe55b06f20af7c41bface7dacf947f5279de93"
src, dst = sys.argv[1], sys.argv[2]
b = open(src, "rb").read()
assert hashlib.sha256(b).hexdigest() == SRC_SHA, "REFUSED: the source is not the KAPPA-T1 tool 260bd175"
s = b.decode("utf-8")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


# 1. docstring
rep('''(``tests/test_v8f_t1_feasible.py``).
''', '''(``tests/test_v8f_t1_feasible.py``).

``--feasible {kbox,abox,kabox}`` (FEAS2-T1, H-FEAS2-T1-1) split ``box`` into its two clamps, same place (after ``--smooth``, every replan, before the executor), same ``dt``, ``mu=None``:
``kbox`` = ``project_feasible(a_max=inf, kappa_max=KAPPA_MAX_1PM, clamp_entry=False)`` (the curvature clamp alone); ``abox`` = ``project_feasible(kappa_max=inf, a_max=A_MAX_MPS2,
clamp_entry=True, v0=the tick's measured ego speed batch["p10f"][:, -1, 0])`` (the accel clamp and the entry-speed bound alone); ``kabox`` = ``kbox`` then ``abox``. ``box`` is unchanged
bit for bit from KAPPA-T1; ``off`` is unchanged bit for bit (``tests/test_v8f_t1_feas2.py``).
''')

# 2. constants
rep('''FEASIBLE_MODES = ("off", "box")
''', '''FEASIBLE_MODES = ("off", "box", "kbox", "abox", "kabox")
#: FEAS2-T1: the header's operator text per mode (``box`` is KAPPA-T1's, verbatim).
FEASIBLE_OPERATORS = {
    "box": "tanitad.refs.feasible_decode.project_feasible(origin + plan, dt=DT, a_max=A_MAX_MPS2, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False)",
    "kbox": "tanitad.refs.feasible_decode.project_feasible(origin + plan, dt=DT, a_max=inf, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False)",
    "abox": "tanitad.refs.feasible_decode.project_feasible(origin + plan, v0=p10f[:, -1, 0], dt=DT, a_max=A_MAX_MPS2, kappa_max=inf, mu=None, clamp_entry=True)",
    "kabox": "abox(kbox(plan)): kbox = project_feasible(a_max=inf, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False), then abox = project_feasible(v0=p10f[:, -1, 0], a_max=A_MAX_MPS2, kappa_max=inf, mu=None, clamp_entry=True)",
}
''')

# 3. the operator
i0 = s.index("def feasible_plan(plan, mode: str = \"off\"):")
i1 = s.index("#: the S / T head outputs ``--dump-heads`` keeps")
new_fn = '''def feasible_plan(plan, mode: str = "off", v0=None):
    """The feasible-decode projection on a batch of plans [B, T, 2] (ego frame, knots 1..T; the origin is implied): ``project_feasible`` of ``tanitad.refs.feasible_decode`` on the
    origin-prepended path, float64, ``dt=DT``, ``mu=None``, returned in the plan's dtype without the origin. ``box`` = both clamps (KAPPA-T1), ``clamp_entry=False``; ``kbox`` = the
    curvature clamp alone (``a_max=inf``); ``abox`` = the accel clamp alone (``kappa_max=inf``) WITH the entry-speed bound ``clamp_entry=True`` (needs ``v0`` [B], the tick's measured ego
    speed); ``kabox`` = ``kbox`` then ``abox``. ``off`` returns ``plan`` itself (the same object)."""
    if mode == "off":
        return plan
    if mode not in FEASIBLE_MODES:
        raise ValueError(f"--feasible must be one of {FEASIBLE_MODES}, got {mode!r}")
    import torch
    from tanitad.refs import feasible_decode as FD
    p64 = np.asarray(plan, np.float64)
    if p64.ndim != 3 or p64.shape[-1] != 2 or p64.shape[1] < 3:
        raise ValueError(f"feasible_plan: plan {p64.shape} must be [B, T>=3, 2]")
    inf = float("inf")
    full = torch.from_numpy(np.concatenate([np.zeros_like(p64[:, :1]), p64], 1))
    if mode in ("abox", "kabox"):
        if v0 is None or np.asarray(v0).shape != (p64.shape[0],):
            raise ValueError(f"feasible_plan: mode {mode!r} needs v0 of shape [B]={p64.shape[0]}, got {None if v0 is None else np.asarray(v0).shape}")
        v0t = torch.from_numpy(np.asarray(v0, np.float64))
    if mode == "box":
        out = FD.project_feasible(full, dt=DT, a_max=FD.A_MAX_MPS2, kappa_max=FD.KAPPA_MAX_1PM, mu=None, clamp_entry=False)
    elif mode in ("kbox", "kabox"):
        out = FD.project_feasible(full, dt=DT, a_max=inf, kappa_max=FD.KAPPA_MAX_1PM, mu=None, clamp_entry=False)
    if mode in ("abox", "kabox"):
        out = FD.project_feasible(full if mode == "abox" else out, v0=v0t, dt=DT, a_max=FD.A_MAX_MPS2, kappa_max=inf, mu=None, clamp_entry=True)
    return out[:, 1:].numpy().astype(np.asarray(plan).dtype)


'''
s = s[:i0] + new_fn + s[i1:]

# 4. the planner passes the tick's measured ego speed
rep('''            plan.plan_s = smooth_plan(raw, out["p10"].float().cpu().numpy(), smooth)
            return feasible_plan(plan.plan_s, feasible)
''', '''            plan.plan_s = smooth_plan(raw, out["p10"].float().cpu().numpy(), smooth)
            v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()          # FEAS2-T1: the measured ego speed the prior reads (== TickRecorder's ``v0``); used by abox / kabox only
            return feasible_plan(plan.plan_s, feasible, v0=v0)
''')

# 5. the argparse help
rep('''help="KAPPA-T1: box = feasible_decode.project_feasible (box envelope A_MAX 4.0, KAPPA_MAX 0.2, mu=None, clamp_entry=False, dt=DT) on the plan AFTER --smooth, at every replan, before the executor; off (default) = unchanged bit for bit")''',
    '''help="KAPPA-T1 / FEAS2-T1: feasible_decode.project_feasible on the plan AFTER --smooth, at every replan, before the executor (dt=DT, mu=None): box = A_MAX 4.0 and KAPPA_MAX 0.2, clamp_entry=False; kbox = KAPPA_MAX 0.2 only (a_max=inf); abox = A_MAX 4.0 only (kappa_max=inf) with clamp_entry=True and v0 = the tick's measured ego speed; kabox = kbox then abox; off (default) = unchanged bit for bit")''')

# 6. the header
rep('''        header["feasible"] = {"mode": a.feasible, "operator": "tanitad.refs.feasible_decode.project_feasible(origin + plan, dt=DT, a_max=A_MAX_MPS2, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False)",
                              "prereg": "dagland/kappa_t1/PREREG_FROZEN.md (H-KAPPA-T1-1)", "applied": "to the plan after --smooth at every replan, before the executor; the P10 floor is not projected"}
''', '''        header["feasible"] = {"mode": a.feasible, "operator": FEASIBLE_OPERATORS[a.feasible],
                              "prereg": "dagland/kappa_t1/PREREG_FROZEN.md (H-KAPPA-T1-1)" if a.feasible == "box" else "dagland/feas2_t1/PREREG_FROZEN.md (H-FEAS2-T1-1)",
                              "applied": "to the plan after --smooth at every replan, before the executor; the P10 floor is not projected"}
''')
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("wrote", dst, hashlib.sha256(s.encode("utf-8")).hexdigest())

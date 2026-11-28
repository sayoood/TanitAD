"""Apply the FEAS3 patch (--feasible {kbox2, lonfloor, kbox2lf}) to the FEAS2-T1 tool (sha256 4b66cc86...). One-shot; each replacement must match exactly once.
Usage: python -I apply_feas3_patch.py <feas2_t1 v8f_t1_eval.py> <out v8f_t1_eval_feas3.py>
Every FEAS2 mode (off, box, kbox, abox, kabox) is untouched (bit-identical to the FEAS2 tool; tests/test_v8f_t1_feas3.py). The three new modes are the card's (PREREG_FEAS3_FROZEN.md),
all applied to the plan after --smooth, at every replan, before the executor, on the plan's own grid dt = DT, float64:
  kbox2     project_feasible(a_max=inf, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False, min_speed=2.0)   (kbox with the curvature denominator floor lifted from 0.5 to 2.0 m/s)
  lonfloor  executed step speeds floored at (the P10 prior's step speed - 2 m/s), headings kept, re-integrated from the origin
  kbox2lf   kbox2, then lonfloor
The P10 prior is the planner's own ``out["p10"]`` (a function of the measured ego history; the same array the smoother reads)."""
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
rep('''bit for bit from KAPPA-T1; ``off`` is unchanged bit for bit (``tests/test_v8f_t1_feas2.py``).
''', '''bit for bit from KAPPA-T1; ``off`` is unchanged bit for bit (``tests/test_v8f_t1_feas2.py``).

``--feasible {kbox2,lonfloor,kbox2lf}`` (FEAS3, H-FEAS3-T1-1; this file is ``v8f_t1_eval_feas3.py``) add three operators at the same place (after ``--smooth``, every replan, before the executor), float64, ``dt``=``DT``:
``kbox2`` = ``project_feasible(a_max=inf, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False, min_speed=KBOX2_MIN_SPEED_MPS)`` (``kbox`` with the curvature denominator floor at 2.0 m/s instead of 0.5);
``lonfloor`` = :func:`lon_floor`: each executed step's speed is floored at (the P10 prior's step speed - ``LONFLOOR_MARGIN_MPS``), its heading kept, the path re-integrated from the origin
(a plan whose steps all sit above the floor is returned unchanged, bit for bit); ``kbox2lf`` = ``kbox2``, then ``lonfloor``. The prior is the planner's own ``out["p10"]``. The four FEAS2 modes (box, kbox, abox, kabox) and
``off`` are unchanged bit for bit (``tests/test_v8f_t1_feas3.py``).
''')

# 2. constants
rep('''FEASIBLE_MODES = ("off", "box", "kbox", "abox", "kabox")
''', '''FEASIBLE_MODES = ("off", "box", "kbox", "abox", "kabox", "kbox2", "lonfloor", "kbox2lf")
#: FEAS3 (PREREG_FEAS3_FROZEN.md): the two constants of the new operators, NOT tuned here.
KBOX2_MIN_SPEED_MPS = 2.0       # kbox2: ``project_feasible(min_speed=...)``, the curvature denominator floor (module default 0.5)
LONFLOOR_MARGIN_MPS = 2.0       # lonfloor: step speed floor = prior step speed - this
LONFLOOR_EPS_M = 1e-9           # a plan step shorter than this has no heading of its own; the prior's step direction is used (the floor is then > 0, so the prior step is not zero)
''')
rep('''    "kabox": "abox(kbox(plan)): kbox = project_feasible(a_max=inf, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False), then abox = project_feasible(v0=p10f[:, -1, 0], a_max=A_MAX_MPS2, kappa_max=inf, mu=None, clamp_entry=True)",
}
''', '''    "kabox": "abox(kbox(plan)): kbox = project_feasible(a_max=inf, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False), then abox = project_feasible(v0=p10f[:, -1, 0], a_max=A_MAX_MPS2, kappa_max=inf, mu=None, clamp_entry=True)",
    "kbox2": "tanitad.refs.feasible_decode.project_feasible(origin + plan, dt=DT, a_max=inf, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False, min_speed=2.0)",
    "lonfloor": "lon_floor(plan, prior=out['p10'], margin=2.0): step speed = max(plan step speed, prior step speed - 2.0 m/s, 0), plan step heading kept, re-integrated from the origin",
    "kbox2lf": "lon_floor(kbox2(plan), prior=out['p10'], margin=2.0): kbox2 = project_feasible(a_max=inf, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False, min_speed=2.0), then the lonfloor operator",
}
''')

# 3. the operators
i0 = s.index("def feasible_plan(plan, mode: str = \"off\", v0=None):")
i1 = s.index("#: the S / T head outputs ``--dump-heads`` keeps")
body = s[i0:i1]
body = body.replace('def feasible_plan(plan, mode: str = "off", v0=None):', 'def feasible_plan(plan, mode: str = "off", v0=None, prior=None):', 1)
assert body.count('speed); ``kabox`` = ``kbox`` then ``abox``. ``off`` returns ``plan`` itself (the same object)."""') == 1
body = body.replace('speed); ``kabox`` = ``kbox`` then ``abox``. ``off`` returns ``plan`` itself (the same object)."""',
                    'speed); ``kabox`` = ``kbox`` then ``abox``. FEAS3: ``kbox2`` = ``kbox`` with ``min_speed=KBOX2_MIN_SPEED_MPS``; ``lonfloor`` = :func:`lon_floor` over ``prior`` [B, T, 2] (the planner\'s ``out["p10"]``);\n'
                    '    ``kbox2lf`` = ``kbox2`` then ``lonfloor``. ``off`` returns ``plan`` itself (the same object)."""', 1)
assert body.count("    inf = float(\"inf\")\n") == 1
body = body.replace("    inf = float(\"inf\")\n", '''    inf = float("inf")
    if mode in ("lonfloor", "kbox2lf") and (prior is None or np.asarray(prior).shape != p64.shape):
        raise ValueError(f"feasible_plan: mode {mode!r} needs prior of shape {p64.shape}, got {None if prior is None else np.asarray(prior).shape}")
''', 1)
old_ret = "    return out[:, 1:].numpy().astype(np.asarray(plan).dtype)\n"
assert body.count(old_ret) == 1
new_tail = '''    if mode in ("kbox2", "kbox2lf"):
        out = FD.project_feasible(full, dt=DT, a_max=inf, kappa_max=FD.KAPPA_MAX_1PM, mu=None, clamp_entry=False, min_speed=KBOX2_MIN_SPEED_MPS)
    if mode == "lonfloor":
        return lon_floor(p64, prior, LONFLOOR_MARGIN_MPS).astype(np.asarray(plan).dtype)
    if mode == "kbox2lf":
        return lon_floor(out[:, 1:].numpy(), prior, LONFLOOR_MARGIN_MPS).astype(np.asarray(plan).dtype)
    return out[:, 1:].numpy().astype(np.asarray(plan).dtype)


def lon_floor(plan, prior, margin: float = LONFLOOR_MARGIN_MPS):
    """FEAS3 ``lonfloor``: on a batch of plans [B, T, 2] and their prior [B, T, 2] (ego frame, knots 1..T, the origin implied; float64 returned), the step speed of every step is floored at
    ``max(prior step speed - margin, 0)``, the step's own heading is kept, and the path is re-integrated from the origin. A step already at or above its floor is not touched; a plan with no step
    below its floor is returned unchanged (bit for bit, the same values). A zero-length plan step has no heading: the prior's step direction is used there (the floor is > 0 then)."""
    x, p = np.asarray(plan, np.float64), np.asarray(prior, np.float64)
    if x.shape != p.shape or x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError(f"lon_floor: plan {x.shape} and prior {p.shape} must match and be [B, T, 2]")
    z = np.zeros_like(x[:, :1])
    dx, dp = np.diff(np.concatenate([z, x], 1), axis=1), np.diff(np.concatenate([z, p], 1), axis=1)
    sx, sp = np.linalg.norm(dx, axis=-1) / DT, np.linalg.norm(dp, axis=-1) / DT
    floor = np.maximum(sp - float(margin), 0.0)
    active = sx < floor
    ux = dx / np.maximum(np.linalg.norm(dx, axis=-1, keepdims=True), 1e-300)
    up = dp / np.maximum(np.linalg.norm(dp, axis=-1, keepdims=True), 1e-300)
    u = np.where((np.linalg.norm(dx, axis=-1) > LONFLOOR_EPS_M)[..., None], ux, up)
    new = np.where(active[..., None], u * (floor * DT)[..., None], dx)
    return np.where(active.any(axis=1)[:, None, None], np.cumsum(new, axis=1), x)
'''
body = body.replace(old_ret, new_tail, 1)
s = s[:i0] + body + s[i1:]

# 4. the planner passes the prior (the same float32 array the smoother reads)
rep('''            return feasible_plan(plan.plan_s, feasible, v0=v0)
''', '''            return feasible_plan(plan.plan_s, feasible, v0=v0, prior=out["p10"].float().cpu().numpy())   # FEAS3: lonfloor / kbox2lf floor against the prior's step speed
''')

# 5. the header
rep('''"prereg": "dagland/kappa_t1/PREREG_FROZEN.md (H-KAPPA-T1-1)" if a.feasible == "box" else "dagland/feas2_t1/PREREG_FROZEN.md (H-FEAS2-T1-1)",''',
    '''"prereg": ("dagland/kappa_t1/PREREG_FROZEN.md (H-KAPPA-T1-1)" if a.feasible == "box" else "dagland/feas2_t1/PREREG_FEAS3_FROZEN.md (H-FEAS3-T1-1)" if a.feasible in ("kbox2", "lonfloor", "kbox2lf")
                                else "dagland/feas2_t1/PREREG_FROZEN.md (H-FEAS2-T1-1)"),''')

# 6. the argparse help
rep('''kabox = kbox then abox; off (default) = unchanged bit for bit")''',
    '''kabox = kbox then abox; FEAS3: kbox2 = kbox with min_speed 2.0; lonfloor = step speeds floored at the P10 prior's step speed - 2 m/s, headings kept; kbox2lf = kbox2 then lonfloor; off (default) = unchanged bit for bit")''')
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("wrote", dst, hashlib.sha256(s.encode("utf-8")).hexdigest())

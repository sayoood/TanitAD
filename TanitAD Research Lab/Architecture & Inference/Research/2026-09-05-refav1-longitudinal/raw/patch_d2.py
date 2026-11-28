# -*- coding: utf-8 -*-
"""D2 -- `a_shift`: the tokens name a speed change relative to WHERE YOU ARE
GOING, not relative to where you are. Anchored, idempotent, aborts without
writing on any anchor miss."""
import io
import sys
import time

PATH = sys.argv[1]
MARK = "a_shift"


def rd(p, tries=30):
    for _ in range(tries):
        try:
            t = io.open(p, encoding="utf-8", newline="").read()
            if t:
                return t
        except Exception:
            pass
        time.sleep(4)
    raise SystemExit("INCONCLUSIVE read %s" % p)


def wr(p, t, tries=30):
    for _ in range(tries):
        try:
            with io.open(p, "w", encoding="utf-8", newline="") as f:
                f.write(t)
            if len(rd(p)) == len(t):
                return
        except Exception:
            pass
        time.sleep(4)
    raise SystemExit("INCONCLUSIVE write %s" % p)


t = rd(PATH)
if MARK in t:
    print("ALREADY PATCHED")
    raise SystemExit(0)
EOL = "\r\n" if "\r\n" in t else "\n"
t = t.replace("\r\n", "\n")
n0 = len(t)
S, N, W = u"⭐", u"⛔", u"⚠"


def sub(old, new, label, want=1):
    global t
    c = t.count(old)
    if c != want:
        raise SystemExit("ANCHOR %s occurs %d (need %d) -- ABORT" % (label, c, want))
    t = t.replace(old, new, want)
    print("  patched %s" % label)


# ------------------------------------------------------- 1. the signature -- #
sub("""                       kappa_turn: float | None = None,
                       a_sustain: float | None = None) -> Tensor:""",
    """                       kappa_turn: float | None = None,
                       a_sustain: float | None = None,
                       a_shift: float | None = None) -> Tensor:""",
    "canonical_controls signature")

# ------------------------------------------------------- 2. the shift ------ #
SHIFT = u"""    if a_shift is not None and _target_is_relative(lon, v0):
        # {S}{S} `a_shift` -- D2, AND IT DOMINATES `a_sustain` ON EVERY COLUMN.
        # THE RULE IN ONE SENTENCE: *the LON tokens name a speed change
        # relative to WHERE YOU ARE GOING, not relative to where you are.*
        # `v_t` moves by the a0 extrapolation over the token's own reach time,
        # so on the MAINTAIN branch `a[0]` is exactly a0 (i.e. `a_sustain` is
        # this design's special case at the first step) and on every OTHER
        # token the named `dv` is applied ON TOP of the motion already under
        # way instead of pretending the car is coasting.
        # MEASURED, mean paired difference against the `ha0_ext` floor over the
        # same 40 windows (`raw/lon_designs.txt`; the `cl` row of that table
        # reproduces the episode-cluster bootstrap's +0.4862 / +0.4117 /
        # +0.0162 to four decimals, which is what makes the comparison
        # admissible):
        #     SHIPPED canonical      LON speed +0.4551   ADE +0.1492
        #     `a_sustain` (D1)       LON speed +0.1752   ADE +0.0059
        #     `a_shift`   (D2)       LON speed +0.1184   ADE **-0.0389**
        # i.e. D2 closes **76 %** of the shipped vocabulary's longitudinal
        # deficit and is the only design that goes NEGATIVE on ADE -- better
        # than the floor -- at the expressivity level.
        # {W} EXPRESSIVITY, NOT A PLANNER RESULT. Those rows roll a control
        # profile through the unicycle integrator; whether the SEARCH emits
        # them is the arm's question, and the reason to believe it might is
        # `D-REFAV1-LON-PLAN-COPIES-VOCAB` (the emitted `a[0]` IS the decoded
        # token's canonical rung on 27/40 windows).
        # {N} ABSOLUTE TARGETS ARE NOT SHIFTED. `HOLD` (stop) and `CREEP`
        # (1.5 m/s) name a speed IN THE WORLD, and so does
        # `ADAPT_SPEED_FOR_CURVE` once `v0 > GOAL_CURVE_VMAX_MPS` (brake to
        # 8 m/s). Shifting those would change what the token MEANS rather than
        # what it is measured against. On the 40-window grid this predicate is
        # numerically inert -- `HOLD`/`CREEP` decode 0/40 times -- so it costs
        # nothing here and prevents a real error elsewhere.
        # {S} The hint is the MEASURED a0 (backward difference of past speeds
        # closing at t0), so there is no chooser to train and no free
        # parameter fitted on the scored split.
        # {N} `None` is the SHIPPED path: bit-identical to every pre-2026-09-05
        # arm, and mutually exclusive with `a_sustain` (they are two spellings
        # of one lever and combining them would make an arm non-attributable).
        v_t = v_t + float(a_shift) * GOAL_REACH_S
    if a_sustain is not None and abs(v_t - float(v0)) < _MAINTAIN_EPS:""".format(
    S=S, N=N, W=W)
sub("    if a_sustain is not None and abs(v_t - float(v0)) < _MAINTAIN_EPS:",
    SHIFT, "a_shift branch")

# --------------------------------------- 3. the predicate + the guard ------ #
PRED = u'''#: {N} WHICH LON TOKENS NAME A *RELATIVE* TARGET. `a_shift` may only move a
#: target that is already expressed against the measured v0; `HOLD` (0 m/s),
#: `CREEP` (GOAL_CREEP_MPS) and `ADAPT_SPEED_FOR_CURVE` above
#: `GOAL_CURVE_VMAX_MPS` name a speed IN THE WORLD and are left alone.
def _target_is_relative(lon: str, v0: float) -> bool:
    if lon in GOAL_LON_DV_MPS:
        return True
    return (lon == "ADAPT_SPEED_FOR_CURVE"
            and float(v0) <= GOAL_CURVE_VMAX_MPS)


_MAINTAIN_EPS = 1e-9'''.format(N=N)
sub("_MAINTAIN_EPS = 1e-9", PRED, "_target_is_relative")

# ------------------------------------------- 4. mutual-exclusion guard ----- #
sub("""    a = torch.zeros(op_steps)
    k = torch.zeros(op_steps)""",
    u"""    if a_sustain is not None and a_shift is not None:
        raise ValueError(
            "a_sustain and a_shift are two spellings of ONE lever (a_shift "
            "reduces to a_sustain's a[0] on the maintain branch); passing both "
            "would make an arm non-attributable. Pass exactly one.")
    a = torch.zeros(op_steps)
    k = torch.zeros(op_steps)""",
    "mutual-exclusion guard")

# ------------------------------------------- 5. plumbing through the goal -- #
sub("""                               goal_kappa_hint=None,
                               a_sustain=None
                               ) -> tuple[Tensor, dict]:""",
    """                               goal_kappa_hint=None,
                               a_sustain=None,
                               a_shift=None
                               ) -> tuple[Tensor, dict]:""",
    "_imagine_tactical_goal signature")

sub("""        if a_sustain is None:
            asu = [None] * len(lat_i)
        else:
            asu = torch.as_tensor(a_sustain, dtype=torch.float32
                                  ).reshape(-1).tolist()
            if len(asu) == 1:
                asu = asu * len(lat_i)
            if len(asu) != len(lat_i):
                raise ValueError(f"a_sustain carries {len(asu)} rows for a "
                                 f"batch of {len(lat_i)}")""",
    """        def _per_row(x, name):
            if x is None:
                return [None] * len(lat_i)
            v = torch.as_tensor(x, dtype=torch.float32).reshape(-1).tolist()
            if len(v) == 1:
                v = v * len(lat_i)
            if len(v) != len(lat_i):
                raise ValueError(f"{name} carries {len(v)} rows for a batch "
                                 f"of {len(lat_i)}")
            return v

        asu = _per_row(a_sustain, "a_sustain")
        ash = _per_row(a_shift, "a_shift")""",
    "per-row helper")

sub("""            canonical_controls(lat_v[i], lon_v[j], v, cfg.op_steps, cfg.op_dt,
                               kappa_turn=k, a_sustain=s)
            for i, j, v, k, s in zip(lat_i, lon_i, v0.tolist(), kt,
                                     asu)]).to(last)""",
    """            canonical_controls(lat_v[i], lon_v[j], v, cfg.op_steps, cfg.op_dt,
                               kappa_turn=k, a_sustain=s, a_shift=q)
            for i, j, v, k, s, q in zip(lat_i, lon_i, v0.tolist(), kt,
                                        asu, ash)]).to(last)""",
    "ctrl stack")

sub("""                      "a_sustain_used": [None if s is None else float(s)
                                         for s in asu]}""",
    """                      "a_sustain_used": [None if s is None else float(s)
                                         for s in asu],
                      "a_shift_used": [None if q is None else float(q)
                                       for q in ash]}""",
    "goal_info stamp")

# ------------------------------------------------- 6. plan() plumbing ------ #
sub("""             a_sustain=None,
             jerk_seam_a0: float | None = None,""",
    """             a_sustain=None,
             a_shift=None,
             jerk_seam_a0: float | None = None,""",
    "plan signature")

sub("""                goal_kappa_hint=goal_kappa_hint,
                a_sustain=a_sustain)""",
    """                goal_kappa_hint=goal_kappa_hint,
                a_sustain=a_sustain, a_shift=a_shift)""",
    "plan -> goal (2 sites)", want=2)

sub("""        res.a_sustain = (None if a_sustain is None else
                         [float(x) for x in
                          torch.as_tensor(a_sustain).flatten()])""",
    """        res.a_sustain = (None if a_sustain is None else
                         [float(x) for x in
                          torch.as_tensor(a_sustain).flatten()])
        res.a_shift = (None if a_shift is None else
                       [float(x) for x in torch.as_tensor(a_shift).flatten()])""",
    "result stamp")

if EOL != "\n":
    t = t.replace("\n", EOL)
wr(PATH, t)
print("OK %s  %d -> %d chars" % (PATH, n0, len(t)))

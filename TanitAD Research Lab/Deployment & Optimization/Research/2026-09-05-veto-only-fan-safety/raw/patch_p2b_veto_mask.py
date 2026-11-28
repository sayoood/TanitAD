#!/usr/bin/env python3
"""P2 (part b) - insert `veto_mask()` and finish the comment, exact-byte anchored.

Part (a) replaced the two key-membership lines with a call to `veto_mask`; this
inserts the function itself and updates the comment above the call site. Split
because the comment anchor in part (a) used a hyphen where the file has an EM
DASH, and an anchor that does not match reads exactly like "already patched" -
the reason `rw` raises on a zero count instead of skipping.
"""
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
CR = chr(13)
LF = chr(10)
CRLF = CR + LF
EMDASH = chr(8212)
NOENTRY = chr(9940)
STAR = chr(11088)


def rw(path, old, new, *, count=1):
    p = os.path.join(REPO, path)
    raw = open(p, "rb").read()
    is_crlf = CRLF.encode() in raw
    s = raw.decode("utf-8").replace(CRLF, LF)
    if new in s and old not in s:
        print("  [skip] %s: already patched" % path)
        return
    n = s.count(old)
    if n != count:
        raise SystemExit("[patch] %s: expected %d occurrence(s), found %d" % (path, count, n))
    s = s.replace(old, new)
    open(p, "wb").write((s.replace(LF, CRLF) if is_crlf else s).encode("utf-8"))
    print("  [ok]   %s (%s)" % (path, "CRLF" if is_crlf else "LF"))


# --- the comment above the call site --------------------------------------- #
old_c = ("    # " + NOENTRY + " THE VETO CHANNEL " + EMDASH + " constraints, not ranking terms. Collision OR\n"
         "    # TTC-imminent. Applied outside the group-relative centring so a vetoed\n"
         "    # candidate is PINNED, never merely ranked lower.\n")
new_c = ("    # " + NOENTRY + " THE VETO CHANNEL " + EMDASH + " constraints, not ranking terms. Collision OR\n"
         "    # TTC-imminent. Applied outside the group-relative centring so a vetoed\n"
         "    # candidate is PINNED, never merely ranked lower. " + STAR + " Keyed on the\n"
         "    # CONFIG (`veto_enabled`/`veto_collision`/`veto_ttc`), NEVER on\n"
         "    # `spec.weights` " + EMDASH + " see `veto_mask` and config.py's `veto_enabled` note.\n")
rw("stack/tanitad/rl/posttrain.py", old_c, new_c)

# --- the function itself ---------------------------------------------------- #
FN = '''def veto_mask(traj: Tensor, ctx: dict, cfg: PostTrainConfig) -> Tensor:
    """The hard-constraint mask over ``traj [..., S, 2]`` -> bool, EXPLICITLY keyed.

    {NE}{NE} WHY THIS FUNCTION EXISTS, AND WHY THE OLD TWO LINES WERE A DEFECT.
    The veto used to be composed as::

        if "collision" in spec.weights:            # KEY membership: TRUE at 0.0
            veto = COMPONENTS["collision"](traj, ctx) < 0
        ttc = ttc_violation(...)                   # read no config at all
        veto = ttc if veto is None else (veto | ttc)

    so a "constant reward" control with every weight at 0.0 still carried the FULL
    veto. Because `advantage.truncated_inter_anchor_advantage` pins vetoed
    candidates at `veto_value` OUTSIDE the group-relative centring, a constant
    reward yields a VETO-ONLY advantage at full strength rather than a zero one.
    MEASURED on `ctrl_const` (2026-09-05): `veto_rate_mean` **0.0897**,
    `final_loss` **-1.863**, 14 of 57 fan-safety metrics moved {ED} a control that
    could not be a null, and a result that was read as a harness fault before the
    mechanism was found.

    => The channels are now three booleans somebody has to TYPE, and `to_dict()`
    records them, so a run's own record says whether the constraint was in force.
    Returns an all-False mask (never ``None``) when the veto is off, so the caller
    always has a `veto_rate` to log {ED} a channel that reports 0.0 is evidence; a
    channel that reports nothing is not.

    {ST} This also makes VETO-ONLY a first-class arm rather than an accident: zero
    reward weights with `veto_enabled=True` is DDv2's constraint mechanism with no
    ranking term at all, which is the only half of the stage that improved
    refcv3's fan feasibility.
    """
    off = torch.zeros(traj.shape[:-2], dtype=torch.bool, device=traj.device)
    if not cfg.veto_enabled:
        return off
    veto = off
    if cfg.veto_collision:
        veto = veto | (R.COMPONENTS["collision"](traj, ctx) < 0)
    if cfg.veto_ttc:
        veto = veto | R.ttc_violation(traj, {{**ctx, "ttc_min_s": cfg.ttc_min_s}})
    return veto


'''.format(NE=NOENTRY, ED=EMDASH, ST=STAR)

rw("stack/tanitad/rl/posttrain.py",
   "def rl_objective(traj: Tensor, logp: Tensor, ctx: dict, cfg: PostTrainConfig,\n",
   FN + "def rl_objective(traj: Tensor, logp: Tensor, ctx: dict, cfg: PostTrainConfig,\n")

print("[patch] P2b applied")

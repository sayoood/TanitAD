#!/usr/bin/env python3
"""P2 - make the RL safety VETO an EXPLICIT, RECORDED configuration field.

THE DEFECT (MEASURED 2026-09-05, `.../2026-09-05-refc-rl-readiness/RESULT.md` §12.1):
`posttrain.py` keyed the collision veto on ``"collision" in spec.weights`` - KEY
MEMBERSHIP, which is TRUE at weight 0.0 - and the TTC veto on nothing at all. So a
"constant reward" control (every weight 0.0) was NOT a null: `advantage.py` pins vetoed
candidates at -1 OUTSIDE the group-relative centring, so a constant reward yields a
VETO-ONLY advantage AT FULL STRENGTH. `ctrl_const` measured `veto_rate_mean` 0.0897 and
`final_loss` -1.863 with every weight at zero, and moved 14 of 57 fan-safety metrics.

=> After this patch the veto is composed from `cfg.veto_enabled` / `cfg.veto_collision` /
`cfg.veto_ttc` and NEVER from the reward's key set. A zero-weight spec with
`veto_enabled=False` is an ACTUAL null (veto_rate exactly 0.0); a zero-weight spec with
`veto_enabled=True` is a deliberate VETO-ONLY arm, which is what we ship as a product.

REPRODUCIBILITY OF THE BANKED ARMS. Under the old keying, `reg_echo`
(weights={"gt_similarity": 1.0}) had the collision veto OFF and TTC ON. Its ARMS entry is
given `veto_collision=False` so it still reproduces bit-for-bit. `rl` / `ctrl0` /
`ctrl_const` all carried a "collision" key, so both channels were on: the default.
"""
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
CR = chr(13)
LF = chr(10)
CRLF = CR + LF


def rw(path, old, new, *, count=1):
    """Anchor-matched replace that PRESERVES the file's own newline convention.

    The repo is mixed CRLF/LF (config.py CRLF, rl_refcv3_min.py LF). Matching an
    LF anchor against CRLF bytes silently finds ZERO occurrences, which reads
    exactly like "already patched" if the guard is sloppy -- so read with
    universal newlines, patch in LF, and write back in the file's own ending.
    """
    p = os.path.join(REPO, path)
    raw = open(p, "rb").read()
    is_crlf = CRLF.encode() in raw
    s = raw.decode("utf-8").replace(CRLF, LF)
    if new in s and old not in s:
        print("  [skip] %s: already patched" % path)
        return
    n = s.count(old)
    if n != count:
        raise SystemExit("[patch] %s: expected %d occurrence(s) of the anchor, found %d"
                         % (path, count, n))
    s = s.replace(old, new)
    out = s.replace(LF, CRLF) if is_crlf else s
    open(p, "wb").write(out.encode("utf-8"))
    print("  [ok]   %s (%s)" % (path, "CRLF" if is_crlf else "LF"))


# --------------------------------------------------------------------------- #
# 1. config.py - three explicit fields beside the veto's other constants        #
# --------------------------------------------------------------------------- #
rw("stack/tanitad/rl/config.py",
   """    ttc_min_s: float = 1.5
    veto_value: float = -1.0
""",
   """    ttc_min_s: float = 1.5
    veto_value: float = -1.0

    #: THE VETO IS KEYED HERE, EXPLICITLY - NEVER ON THE REWARD'S KEY SET.
    #: MEASURED 2026-09-05 (`.../2026-09-05-refc-rl-readiness/RESULT.md` §12.1):
    #: `posttrain.rl_objective` used to read ``"collision" in spec.weights``, i.e.
    #: KEY MEMBERSHIP, which is TRUE at weight 0.0, while the TTC channel read no
    #: config at all. `advantage.truncated_inter_anchor_advantage` pins a vetoed
    #: candidate at `veto_value` OUTSIDE the group-relative centring, so a
    #: CONSTANT reward did not give a zero advantage - it gave a VETO-ONLY
    #: advantage at FULL STRENGTH. The `ctrl_const` arm (every weight 0.0)
    #: measured `veto_rate_mean` **0.0897**, `final_loss` **-1.863**, and moved
    #: 14 of 57 fan-safety metrics. A zero-weight control that is not a null is
    #: the false-green class wearing a control's clothes.
    #: => `veto_enabled=False` makes a zero-weight spec an ACTUAL null (veto_rate
    #: exactly 0.0); `veto_enabled=True` with zero weights is a deliberate
    #: VETO-ONLY arm. Both are now things somebody TYPED, and `to_dict()` writes
    #: them into the run record.
    #: Pinned by `stack/tests/test_rl_veto_explicit.py`.
    veto_enabled: bool = True
    veto_collision: bool = True
    veto_ttc: bool = True
""")

# --------------------------------------------------------------------------- #
# 2. posttrain.py - compose the veto from the config, never from spec.weights   #
# --------------------------------------------------------------------------- #
rw("stack/tanitad/rl/posttrain.py",
   """    veto = None
    if "collision" in spec.weights:
        veto = R.COMPONENTS["collision"](traj, ctx) < 0
    ttc = R.ttc_violation(traj, {**ctx, "ttc_min_s": cfg.ttc_min_s})
    veto = ttc if veto is None else (veto | ttc)
""",
   """    veto = veto_mask(traj, ctx, cfg)
""")

rw("stack/tanitad/rl/posttrain.py",
   """    # THE VETO CHANNEL - constraints, not ranking terms. Collision OR
    # TTC-imminent. Applied outside the group-relative centring so a vetoed
    # candidate is PINNED, never merely ranked lower.
""".replace("# THE VETO", "# ⛔ THE VETO"),
   """    # ⛔ THE VETO CHANNEL - constraints, not ranking terms. Collision OR
    # TTC-imminent. Applied outside the group-relative centring so a vetoed
    # candidate is PINNED, never merely ranked lower. ⭐ Keyed on the CONFIG
    # (`veto_enabled`/`veto_collision`/`veto_ttc`), never on `spec.weights` -
    # see `veto_mask` and config.py's `veto_enabled` note.
""")

# insert veto_mask() above rl_objective
rw("stack/tanitad/rl/posttrain.py",
   """def rl_objective(traj: Tensor, logp: Tensor, ctx: dict, cfg: PostTrainConfig,
""",
   '''def veto_mask(traj: Tensor, ctx: dict, cfg: PostTrainConfig) -> Tensor:
    """The hard-constraint mask over ``traj [..., S, 2]`` -> bool, EXPLICITLY keyed.

    WHY THIS FUNCTION EXISTS, AND WHY THE OLD TWO LINES WERE A DEFECT.
    The veto used to be composed as::

        if "collision" in spec.weights:            # KEY membership: TRUE at 0.0
            veto = COMPONENTS["collision"](traj, ctx) < 0
        ttc = ttc_violation(...)                   # read no config at all
        veto = ttc if veto is None else (veto | ttc)

    so a "constant reward" control with every weight at 0.0 still carried the FULL
    veto. Because `advantage.truncated_inter_anchor_advantage` pins vetoed
    candidates at `veto_value` OUTSIDE the centring, a constant reward yields a
    VETO-ONLY advantage at full strength rather than a zero one. MEASURED on
    `ctrl_const`: `veto_rate_mean` **0.0897**, `final_loss` **-1.863**, 14 of 57
    fan-safety metrics moved - a control that could not be a null.

    => The channels are now three booleans somebody has to type, and `to_dict()`
    records them, so a run's own record says whether the constraint was in force.
    Returns an all-False mask (never ``None``) when the veto is off, so the caller
    always has a `veto_rate` to log - a channel that reports 0.0 is evidence; a
    channel that reports nothing is not.
    """
    off = torch.zeros(traj.shape[:-2], dtype=torch.bool, device=traj.device)
    if not cfg.veto_enabled:
        return off
    veto = off
    if cfg.veto_collision:
        veto = veto | (R.COMPONENTS["collision"](traj, ctx) < 0)
    if cfg.veto_ttc:
        veto = veto | R.ttc_violation(traj, {**ctx, "ttc_min_s": cfg.ttc_min_s})
    return veto


def rl_objective(traj: Tensor, logp: Tensor, ctx: dict, cfg: PostTrainConfig,
''')

print("[patch] P2 applied")

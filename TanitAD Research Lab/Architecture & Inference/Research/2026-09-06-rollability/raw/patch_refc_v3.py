"""Apply the D-ROLL-1 construction fix to refc_v3.py. ASCII-only prints."""
import hashlib
import io
import sys

SRC = r"C:\Users\Admin\_roll\refc_v3.py.orig"
DST = r"C:\Users\Admin\_roll\refc_v3.py.new"

with io.open(SRC, encoding="utf-8", newline="") as fh:
    t = fh.read()

# ---------------------------------------------------------------- field ----
FIELD_ANCHOR = """    admission_sigma_m: float = 0.8

    # --- E13: nav to the tactical and strategic layers (PI 2026-09-02) ------"""

FIELD_NEW = """    admission_sigma_m: float = 0.8

    # --- D-ROLL-1: the tactical-goal TOKEN head is OPT-IN (2026-09-06) ------
    # \u2b50\u2b50 THE FIELD EXISTS SO A BANKED CHECKPOINT REBUILDS WITH THE PARAMETER
    # SET IT WAS TRAINED WITH. D-TACGOAL-1 built `tac_goal_tok_head`
    # UNCONDITIONALLY under any non-`kin3` vocabulary. Every v7.0 checkpoint in
    # the programme was trained BEFORE that head existed, so a rebuild carried
    # 11,286 params (22 tokens x (d_tac + 1)) that the recorded
    # `param_breakdown` does not name -- and `refcv3_arm.cross_check_config`
    # correctly REFUSED, which made refcv4b@40284, all three local refcv3
    # checkpoints AND the LIVE refcv5 run on the A40 unrollable and
    # unresumable. MEASURED 2026-09-06: none of those five records carries a
    # `tac_goal_tok_head` ledger line; refcv4b/refcv5 `total` are 107,058,488
    # and 108,246,216 with the line ABSENT.
    #
    # \u26a0\ufe0f Default OFF, for the reason this file already states twice --
    # `nav_args_inject` ("no banked arm's recipe changes") and the v4 pins
    # ("a default that moved would silently change a training in flight").
    # A new module that changes an existing arm's parameter set is a silent
    # capacity confound; opting in is a launch decision, recorded in `argv`.
    #
    # \u26d4 THE VOCABULARY REMAINS A NECESSARY CONDITION, NOT A SUFFICIENT ONE.
    # `kin3` has no tactical goal vocabulary at all, so the head can never be
    # built there whatever this flag says -- 22 logits that can never be
    # supervised are the defect `effective_mask` exists to prevent.
    #
    # \u2b50 ONE PREDICATE, ONE CONSUMER. `param_breakdown_v3` reports this line
    # by reading the BUILT OBJECT (`model.tac_goal_tok_head is not None`), never
    # by re-deriving this condition, so the ledger cannot go stale against the
    # constructor the way `refcv3_arm`'s `a_star` comment went stale against the
    # trainer ("exactly as the trainer computes it" -- true until 2026-09-04).
    # `test_refc_v3_rollability.py::test_ledger_line_tracks_the_built_object`
    # pins that there is no second copy of the condition to drift.
    tac_goal_tok_head: bool = False

    # --- E13: nav to the tactical and strategic layers (PI 2026-09-02) ------"""

assert t.count(FIELD_ANCHOR) == 1, "field anchor not unique: %d" % t.count(FIELD_ANCHOR)
t = t.replace(FIELD_ANCHOR, FIELD_NEW)

# ------------------------------------------------------------ construction --
CTOR_ANCHOR = """        # \u26a0\ufe0f It exists ONLY under a v7 vocabulary. `kin3` has no tactical
        # goal vocabulary at all, so building it there would be 22 dead
        # logits that can never be supervised \u2014 the same defect
        # `effective_mask` exists to prevent one layer down.
        self.tac_goal_tok_head = None
        if _vv != "kin3":
"""

CTOR_NEW = """        # \u26a0\ufe0f It exists ONLY under a v7 vocabulary. `kin3` has no tactical
        # goal vocabulary at all, so building it there would be 22 dead
        # logits that can never be supervised \u2014 the same defect
        # `effective_mask` exists to prevent one layer down.
        #
        # \u26d4\u26d4 D-ROLL-1 (2026-09-06) \u2014 AND IT IS OPT-IN, because a v7
        # vocabulary is NECESSARY BUT NOT SUFFICIENT. Built on the vocabulary
        # alone, this head added 11,286 params to every rebuild of a checkpoint
        # trained before it existed, so the recorded `param_breakdown` no longer
        # described the weights and `refcv3_arm.cross_check_config` REFUSED \u2014
        # refcv4b@40284, three local refcv3 checkpoints and the LIVE refcv5 A40
        # run all became unrollable, and refcv5 unresumable, in one commit.
        # The guard was right; the CONSTRUCTION was wrong. See
        # `RefCV3Config.tac_goal_tok_head` for why the default is OFF.
        self.tac_goal_tok_head = None
        if _vv != "kin3" and bool(getattr(cfg, "tac_goal_tok_head", False)):
"""

assert t.count(CTOR_ANCHOR) == 1, "ctor anchor not unique: %d" % t.count(CTOR_ANCHOR)
t = t.replace(CTOR_ANCHOR, CTOR_NEW)

with io.open(DST, "w", encoding="utf-8", newline="") as fh:
    fh.write(t)

h = hashlib.md5(io.open(DST, "rb").read()).hexdigest()
print("WROTE %s md5=%s bytes=%d" % (DST, h, len(t.encode("utf-8"))))
print("MARKER-field  :", t.count("tac_goal_tok_head: bool = False"))
print("MARKER-ctor   :", t.count('if _vv != "kin3" and bool(getattr(cfg, "tac_goal_tok_head", False)):'))
print("MARKER-D-ROLL :", t.count("D-ROLL-1"))
print("CONTROL-nonzero(class RefCV3Model):", t.count("class RefCV3Model"))
sys.exit(0)

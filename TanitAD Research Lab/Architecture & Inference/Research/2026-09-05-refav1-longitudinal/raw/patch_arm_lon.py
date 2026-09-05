# -*- coding: utf-8 -*-
"""Wire the three LONGITUDINAL levers into taniteval/tools/refav1_arm.py.
Anchored, idempotent, aborts on any anchor miss without writing."""
import io
import sys
import time

PATH = sys.argv[1]
MARK = "--a-sustain-mode"


def rd(p, tries=25):
    for _ in range(tries):
        try:
            t = io.open(p, encoding="utf-8", newline="").read()
            if t:
                return t
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("INCONCLUSIVE: could not read %s" % p)


def wr(p, t, tries=25):
    for _ in range(tries):
        try:
            with io.open(p, "w", encoding="utf-8", newline="") as f:
                f.write(t)
            if len(rd(p)) == len(t):
                return
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("INCONCLUSIVE: could not write %s" % p)


t = rd(PATH)
if MARK in t:
    print("ALREADY PATCHED")
    raise SystemExit(0)
n0 = len(t)
EOL = "\r\n" if "\r\n" in t else "\n"
print("  line ending: %r" % EOL)
t = t.replace("\r\n", "\n")

STAR, NO, WARN = u"⭐", u"⛔", u"⚠"


def sub(old, new, label, want=1):
    global t
    c = t.count(old)
    if c != want:
        raise SystemExit("ANCHOR %s occurs %d (need %d) -- ABORT" % (label, c, want))
    t = t.replace(old, new, want)
    print("  patched %s" % label)


# ------------------------------------------------------------------ 1. flags #
FLAGS = u'''    ap.add_argument("--a-sustain-mode", choices=("none", "a0"), default="none",
                    help="THE LONGITUDINAL VOCABULARY LEVER (D-REFAV1-LON-VOCAB). "
                         "'none' is the shipped path, BIT-IDENTICAL to every arm "
                         "banked before 2026-09-05. 'a0' gives the goal's MAINTAIN "
                         "branch (v_t == v0: CRUISE always, ADAPT_SPEED_FOR_CURVE "
                         "below GOAL_CURVE_VMAX_MPS) a CONSTANT acceleration equal "
                         "to the MEASURED a0 at t0 -- the same backward difference "
                         "of past speeds ha0_ext holds, no future. MEASURED reason "
                         "(raw/lon_branch.txt, n = 40 windows, ckpt 21109): the "
                         "maintain branch is 31/40 windows (77.5 %%) and 20/24 "
                         "(83.3 %%) of the GT-LON stratum, and on it the shipped "
                         "goal commands a == 0 EXACTLY; the vocabulary's reachable "
                         "dv over 2 s is [-2.85, +0.98] m/s against a corpus p90 of "
                         "+2.34, so 14 of the 17 accelerating windows (82.4 %%) are "
                         "outside it entirely. THIS IS A VOCABULARY CHANGE: it "
                         "moves the goal field the plan is scored against, so it is "
                         "NOT window-comparable with a shipped-vocabulary arm on "
                         "the goal term -- it is comparable on the four families")
    ap.add_argument("--target-speed-mode", choices=("none", "a0ext"), default="none",
                    help="ARM W_VEND, THE DEAD THIRD WEIGHT (D-REFAV1-LON-COST). "
                         "MEASURED: plan()'s target_speed defaults to None and this "
                         "tool's single .plan(...) call has never passed it, so "
                         "`w_vend * (v_end - target_speed)^2` has contributed ZERO "
                         "cost to EVERY banked refav1 window -- the whole cost "
                         "triple's third entry is a no-op, and the 643x W_VEND "
                         "difference between the shipped and A/B triples is a "
                         "difference in a number that is never read. 'a0ext' sets "
                         "target_speed = max(0, v0 + a0*plan_horizon_s) from the "
                         "MEASURED t0 state (no future). 'none' keeps the shipped "
                         "dead term")
    ap.add_argument("--jerk-seam", choices=("off", "a0"), default="off",
                    help="PRICE THE JERK SEAM (D-REFAV1-LON-COST). The shipped "
                         "jerk term diffs the plan's OWN actions only, so the step "
                         "from the car's measured a0 to controls[0] is FREE: "
                         "dropping instantly from a0 = -2.3 m/s^2 to a = 0 costs "
                         "the same as continuing smoothly, and the all-zero plan is "
                         "the joint minimiser of both regularisers. 'a0' prepends "
                         "the measured a0 as the (-1)-th action so the seam is "
                         "priced with the SAME w_jerk -- a repair of an incomplete "
                         "term, not a new weight. 'off' is bit-identical to every "
                         "pre-2026-09-05 arm")
'''
sub(u'    ap.add_argument("--seed-kappa-ladder", default=None,',
    FLAGS + u'    ap.add_argument("--seed-kappa-ladder", default=None,',
    "argparse flags")

# --------------------------------------------------- 2. the per-window wiring #
WIRE = u'''                    tp = time.time()
                    # {S}{S} THE THREE LONGITUDINAL LEVERS, all OFF by default
                    # and all fed from `ext` -- the SAME measured (a0, kappa0)
                    # `ha0_ext` holds, i.e. a backward difference of past
                    # speeds closing at t0 with NOTHING from the future
                    # (`hold_ext_controls`). Admissible at T1 under the PI
                    # ruling of 2026-09-02 on the measured state at cycle time;
                    # strictly LESS information than `ha`, which holds the last
                    # observed ACTION.
                    _a0 = float(ext[0])
                    a_sustain = _a0 if a.a_sustain_mode == "a0" else None
                    jerk_seam = _a0 if a.jerk_seam == "a0" else None
                    tgt_speed = (max(0.0, v0 + _a0 * (pc.horizon * pc.dt))
                                 if a.target_speed_mode == "a0ext" else None)
'''.format(S=STAR)
sub(u"""                    tp = time.time()
                    # ⭐ the planner->model crossing travels with the call:""",
    WIRE + u"                    # {S} the planner->model crossing travels with the call:".format(S=STAR),
    "per-window lever values")

sub(u"""                                     goal_kappa_hint=gkh,
                                     goal_keeps_seed=gks)""",
    u"""                                     goal_kappa_hint=gkh,
                                     a_sustain=a_sustain,
                                     jerk_seam_a0=jerk_seam,
                                     target_speed=tgt_speed,
                                     goal_keeps_seed=gks)""",
    "plan call kwargs")

# ------------------------------------------------- 3. the REACHED-IT guard(s) #
GUARD = u'''                        # {N} AND THE THREE LONGITUDINAL LEVERS MUST HAVE
                        # REACHED plan() TOO. Same trap as the cost flags
                        # above: a stack that silently ignores a kwarg would
                        # bank a SHIPPED-vocabulary arm under a lever's name,
                        # which is the one failure this whole package exists to
                        # avoid. `w_vend_armed` is checked explicitly because
                        # its absence is exactly what made the third weight a
                        # dead term unnoticed for the whole programme.
                        if a.a_sustain_mode != "none":
                            got_s = getattr(res, "a_sustain", "__absent__")
                            if got_s in ("__absent__", None):
                                raise RuntimeError(
                                    "plan() returned a_sustain=%r for "
                                    "--a-sustain-mode %s: the flag did not "
                                    "reach it (stale stack)"
                                    % (got_s, a.a_sustain_mode))
                        if a.jerk_seam != "off":
                            got_j = getattr(res, "jerk_seam_a0", "__absent__")
                            if got_j in ("__absent__", None):
                                raise RuntimeError(
                                    "plan() returned jerk_seam_a0=%r for "
                                    "--jerk-seam %s: the flag did not reach it"
                                    % (got_j, a.jerk_seam))
                        if a.target_speed_mode != "none":
                            if not getattr(res, "w_vend_armed", False):
                                raise RuntimeError(
                                    "plan() reports w_vend_armed=False for "
                                    "--target-speed-mode %s: the third cost "
                                    "weight is STILL a dead term"
                                    % a.target_speed_mode)
                        elif getattr(res, "w_vend_armed", None) is True:
                            raise RuntimeError(
                                "plan() reports w_vend_armed=True with "
                                "--target-speed-mode none")
'''.format(N=NO)
sub(u"""                        got_b = getattr(res, "lat_logit_bias", "__absent__")""",
    GUARD + u'                        got_b = getattr(res, "lat_logit_bias", "__absent__")',
    "reached-it guards")

# --------------------------------------------------------- 4. manifest stamp #
sub(u'''            "kamm_mu": getattr(a, "kamm_mu", None),''',
    u'''            "kamm_mu": getattr(a, "kamm_mu", None),
            # the longitudinal levers, so a dump says which vocabulary and
            # which cost geometry produced it
            "a_sustain_mode": getattr(a, "a_sustain_mode", "none"),
            "target_speed_mode": getattr(a, "target_speed_mode", "none"),
            "jerk_seam": getattr(a, "jerk_seam", "off"),
            "w_vend_armed": getattr(a, "target_speed_mode", "none") != "none",''',
    "manifest stamp")

if EOL != "\n":
    t = t.replace("\n", EOL)
wr(PATH, t)
print("OK %s  %d -> %d chars" % (PATH, n0, len(t)))

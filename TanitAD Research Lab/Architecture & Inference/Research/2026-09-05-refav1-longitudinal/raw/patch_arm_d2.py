# -*- coding: utf-8 -*-
"""Add `a0_shift` (D2) to refav1_arm's --a-sustain-mode. Anchored, idempotent."""
import io
import sys
import time

PATH = sys.argv[1]


def rd(p, tries=30):
    for _ in range(tries):
        try:
            t = io.open(p, encoding="utf-8", newline="").read()
            if t:
                return t
        except Exception:
            pass
        time.sleep(4)
    raise SystemExit("INCONCLUSIVE read")


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
    raise SystemExit("INCONCLUSIVE write")


t = rd(PATH)
if "a0_shift" in t:
    print("ALREADY PATCHED")
    raise SystemExit(0)
EOL = "\r\n" if "\r\n" in t else "\n"
t = t.replace("\r\n", "\n")
n0 = len(t)


def sub(old, new, label, want=1):
    global t
    c = t.count(old)
    if c != want:
        raise SystemExit("ANCHOR %s occurs %d (need %d) -- ABORT" % (label, c, want))
    t = t.replace(old, new, want)
    print("  patched %s" % label)


sub('''    ap.add_argument("--a-sustain-mode", choices=("none", "a0"), default="none",''',
    '''    ap.add_argument("--a-sustain-mode", choices=("none", "a0", "a0_shift"),
                    default="none",''',
    "choices")

sub('''                         "NOT window-comparable with a shipped-vocabulary arm on "
                         "the goal term -- it is comparable on the four families")''',
    '''                         "NOT window-comparable with a shipped-vocabulary arm on "
                         "the goal term -- it is comparable on the four families. "
                         "'a0_shift' is D2 and DOMINATES 'a0' on every measured "
                         "column: instead of touching only the maintain branch it "
                         "shifts EVERY relative target by the a0 extrapolation "
                         "(v_t' = v_t + a0*GOAL_REACH_S), i.e. the tokens name a "
                         "speed change relative to WHERE YOU ARE GOING rather than "
                         "to where you are; it reduces to 'a0' at the first step on "
                         "the maintain branch. MEASURED mean paired difference "
                         "against the ha0_ext floor (raw/lon_designs.txt): shipped "
                         "canonical LON speed +0.4551 / ADE +0.1492; 'a0' +0.1752 / "
                         "+0.0059; 'a0_shift' +0.1184 / -0.0389 -- 76 %% of the "
                         "deficit closed and the only design that goes NEGATIVE on "
                         "ADE. Absolute targets (HOLD, CREEP, ADAPT above "
                         "GOAL_CURVE_VMAX_MPS) are never shifted")''',
    "help text")

sub('''                    a_sustain = (_a0 if getattr(a, "a_sustain_mode",
                                              "none") == "a0" else None)''',
    '''                    _lm = getattr(a, "a_sustain_mode", "none")
                    a_sustain = _a0 if _lm == "a0" else None
                    a_shift = _a0 if _lm == "a0_shift" else None''',
    "per-window values")

sub("""                                     a_sustain=a_sustain,
                                     jerk_seam_a0=jerk_seam,""",
    """                                     a_sustain=a_sustain,
                                     a_shift=a_shift,
                                     jerk_seam_a0=jerk_seam,""",
    "plan kwargs")

sub('''                        if getattr(a, "a_sustain_mode", "none") != "none":
                            got_s = getattr(res, "a_sustain", "__absent__")
                            if got_s in ("__absent__", None):''',
    '''                        if getattr(a, "a_sustain_mode", "none") != "none":
                            got_s = getattr(
                                res, "a_shift" if _lm == "a0_shift"
                                else "a_sustain", "__absent__")
                            if got_s in ("__absent__", None):''',
    "reached-it guard")

if EOL != "\n":
    t = t.replace("\n", EOL)
wr(PATH, t)
print("OK %s  %d -> %d chars" % (PATH, n0, len(t)))

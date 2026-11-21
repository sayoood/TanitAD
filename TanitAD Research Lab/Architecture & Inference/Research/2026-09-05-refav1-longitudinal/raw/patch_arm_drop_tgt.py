# -*- coding: utf-8 -*-
"""REMOVE the `--target-speed-mode` lever from refav1_arm.py.

WHY: `stack/tests/test_steer_conversion_complete.py::test_C1_no_production_
plan_call_site_passes_target_speed` is a DELIBERATE PIN whose own docstring
says arming T4 "is a PI decision, and it is ESCALATED, not taken here". An
agent does not amend a pin that names a PI decision. The measurement that
motivated the lever stands and is escalated in RESULT.md; the two levers the
pin does NOT cover (`a_sustain`, `jerk_seam_a0`) are unaffected.
"""
import io
import sys
import time

PATH = sys.argv[1]


def rd(p, tries=25):
    for _ in range(tries):
        try:
            t = io.open(p, encoding="utf-8", newline="").read()
            if t:
                return t
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("INCONCLUSIVE read")


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
    raise SystemExit("INCONCLUSIVE write")


t = rd(PATH)
if "--target-speed-mode" not in t:
    print("ALREADY REMOVED")
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


# 1. the argparse block (from its own add_argument to the next one)
i = t.index('    ap.add_argument("--target-speed-mode"')
j = t.index('    ap.add_argument("--jerk-seam"')
assert i < j
removed = t[i:j]
t = t[:i] + t[j:]
print("  removed argparse --target-speed-mode (%d chars)" % len(removed))

# 2. the per-window value
sub("""                    tgt_speed = (max(0.0, v0 + _a0 * (pc.horizon * pc.dt))
                                 if a.target_speed_mode == "a0ext" else None)
""", "", "per-window tgt_speed")

# 3. the call kwarg
sub("""                                     target_speed=tgt_speed,
""", "", "plan call kwarg")

# 4. the reached-it guard, replaced by a PIN-AWARE assertion
sub("""                        if a.target_speed_mode != "none":
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
""",
    u"""                        # ⛔ W_VEND MUST STAY DEAD IN THIS TOOL. `target_speed`
                        # is not passed by design: `tests/test_steer_conversion
                        # _complete.py::test_C1_no_production_plan_call_site_
                        # passes_target_speed` pins the CALL SITE, and its own
                        # docstring reserves arming T4 as a PI decision. So the
                        # third entry of every `--cost-weights` triple this tool
                        # has ever banked contributed EXACTLY ZERO cost -- which
                        # also means the 643x W_VEND difference between the
                        # shipped and A/B triples is a difference in a number
                        # that is never read, not a confound. Asserted here so a
                        # future edit that arms it cannot pass unnoticed.
                        if getattr(res, "w_vend_armed", None) is True:
                            raise RuntimeError(
                                "plan() reports w_vend_armed=True: this tool "
                                "must never arm T4 (test_C1 pins the call "
                                "site; arming it is a PI decision)")
""", "reached-it guard -> pin-aware assertion")

# 5. the manifest stamp
sub("""            "target_speed_mode": getattr(a, "target_speed_mode", "none"),
            "jerk_seam": getattr(a, "jerk_seam", "off"),
            "w_vend_armed": getattr(a, "target_speed_mode", "none") != "none",""",
    u"""            "jerk_seam": getattr(a, "jerk_seam", "off"),""" + """
""" + u"""            # W_VEND is a DEAD TERM in this tool by design (test_C1 pins the
            # call site; arming it is a PI decision). Recorded so a reader of a
            # banked cost triple knows its third entry never bound.
            "w_vend_armed": False,""",
    "manifest stamp")

if EOL != "\n":
    t = t.replace("\n", EOL)
wr(PATH, t)
print("OK %s  %d -> %d chars" % (PATH, n0, len(t)))

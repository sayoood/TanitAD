import io
import os
import sys
import time

F = os.path.join(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD",
                 "Project Steering", "GOALS_AND_CLAIMS.md")
PKG = "TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-rl-generator-collisions"

ROW = (
    "| D-RL-CTRL0-SEPFLOOR-1 | \u2b50\u2b50 **A FROZEN-WEIGHT ARM SEPARATES 4 METRICS, AND THE "
    "RIG'S SEPARATION COUNTS ARE INFLATED BY A TOOL QUIRK \u2014 THE HONEST ZERO-INFORMATION "
    "FIGURE IS 24 OF 57, NOT 35.** MEASURED 2026-09-05 on `ctrl0` (lr = 0, `weights_changed` "
    "**False**, i.e. bitwise frozen): **21 of 57** metrics move at all \u2014 max |\u0394| "
    "**1.16e-07**, GPU float non-determinism in the readout, not a lever \u2014 and the "
    "episode-cluster bootstrap calls **4 of them SEPARATED** (`mass_conf_unsafe` \u22121.025e-08, "
    "CI [\u22122.162e-08, \u22121.235e-09]; `mass_conf_ttc_below`, `mass_rank_unsafe`, "
    "`mass_rank_ttc_below`). \u21d2 **`H-ESTIM-SEED-1` in its purest form and STRONGER than the "
    "`A0b_replicate` case that established it**: there a real run with a real seed separated 3 of "
    "18; here **the weights did not change at all** and 4 still separate. `sep` is a statement "
    "about an INTERVAL and carries **no magnitude information whatever**. \u26d4 **AND A TOOL "
    "QUIRK INFLATES EVERY SEPARATION COUNT ON THIS RIG**: a delta of **exactly 0.0 with CI "
    "[0, 0]** is reported `sep=True`, though [0,0] plainly does not exclude 0 \u2014 **36 of "
    "`ctrl0`'s 40** and **11 of `ctrl_null`'s 35** are this artefact. \u26a0 **This SHARPENS a "
    "programme claim rather than overturning it**: *\"a zero-information arm separates 35 of 57\"* "
    "should read **24 of 57**, and the substance STANDS \u2014 those 24 are real and "
    "`fan_peak_g_mean` drifts **+0.134314** on an arm carrying no reward at all. \u21d2 (1) never "
    "count `sep` without checking the delta \u2014 which is also why `top32_contact`'s "
    "\"+0.000000, CI [0,0], sep=True\" (`D-RL-TOP32-NOMOVE-1`) is a NON-RESULT and was quoted "
    "there by delta and interval, never by the flag; (2) `analyze_veto.py`'s `sep` should require "
    "a non-zero delta, and until it does every separation count from that instrument needs this "
    "split | **SUPPORTED (MEASURED, frozen-weight arm)** | `"
    + PKG + "/RESULT.md` \u00a76.9; `\u2026/veto_run/run/s0/ctrl0/arm_summary.json`, "
    "`\u2026/s0/ctrl_null/arm_summary.json` |"
)

ANCH = "| D-RL-TIMING-SURFACE-1 |"

src = None
for _ in range(14):
    try:
        with io.open(F, encoding="utf-8") as fh:
            src = fh.read()
        break
    except OSError:
        time.sleep(4)
if src is None:
    raise SystemExit("INCONCLUSIVE: could not read GOALS_AND_CLAIMS.md (mount)")

if "D-RL-CTRL0-SEPFLOOR-1" in src:
    print("ALREADY APPLIED")
    sys.exit(0)

i = src.find(ANCH)
if i < 0:
    raise SystemExit("REFUSED: anchor %r not found" % ANCH)
j = src.find("\n", i)
out = src[:j] + "\n" + ROW + src[j:]

for _ in range(14):
    try:
        with io.open(F, "w", encoding="utf-8", newline="") as fh:
            fh.write(out)
        break
    except OSError:
        time.sleep(4)

with io.open(F, encoding="utf-8") as fh:
    back = fh.read()
n = back.count("| D-RL-CTRL0-SEPFLOOR-1 |")
c = back.count("| D-RL-TIMING-SURFACE-1 |")
print("D-RL-CTRL0-SEPFLOOR-1 occurrences=%d (want 1)" % n)
print("D-RL-TIMING-SURFACE-1 CONTROL occurrences=%d (want 1; 0 means unreadable)" % c)
print("INSERT=%s" % ("PASS" if n == 1 and c == 1 else "FAIL"))
if not (n == 1 and c == 1):
    raise SystemExit(1)

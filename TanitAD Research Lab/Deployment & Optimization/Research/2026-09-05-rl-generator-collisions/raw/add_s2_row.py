import io
import os
import sys
import time

F = os.path.join(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD",
                 "Project Steering", "GOALS_AND_CLAIMS.md")
PKG = "TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-rl-generator-collisions"

ROW = (
    "| D-RL-GRADED-TERM-1 | \u2b50 **A GRADED COLLISION TERM IS WORTH BUILDING \u2014 AND THE "
    "BRIEF'S PROPOSED GRADING (TIME-TO-CONTACT) IS THE WRONG ONE.** `rewards._collision` returns "
    "\u22121 or 0, so inside a GRPO group **every colliding candidate gets the SAME value**: the "
    "term can say *\"these are bad\"* but never *\"this one is worse\"*, and supplies **no "
    "direction within the colliding set** (control G1: within-colliders std **0.000e+00**, single "
    "distinct value \u22121.0 \u2014 measured, not assumed). Severity, however, VARIES: penetration "
    "depth into the 2 m disc spans **p10 0.190 m \u2192 p90 1.536 m = 8.09x**, and the "
    "**within-window** spread \u2014 the only thing a group-relative advantage can use \u2014 is "
    "**non-degenerate in 19 / 19 mixed windows** (mean range **1.47 m**, mean std 0.43 m). \u26d4 "
    "**But `min_ttc_s` is saturated at the 0.5 s grid floor for \u226575 % of colliders** (p0 = p25 "
    "= p50 = 0.5000 s), so TTC carries almost no information and a TTC-weighted term would be "
    "nearly as flat as the binary. \u21d2 **grade by PENETRATION DEPTH, not by TTC.** \u2b50 "
    "Internal cross-check that the two derivations agree exactly: **289 of 1,053 colliders (27.4 %) "
    "are SWEPT-ONLY** (no sampled point inside the disc \u2014 they cross between samples), and "
    "1,053 \u2212 764 = **289** is precisely the swept-vs-point gap `D-RL-CONTACT-DEFN-1` measured "
    "by a different route | **SUPPORTED (MEASURED, G1 control passes exactly)** | `"
    + PKG + "/raw/p6_graded_term_headroom.json`; `" + PKG + "/RESULT.md` \u00a75 |"
)

ANCH = "| D-RL-COLL-PRICE-1 |"

src = None
for _ in range(12):
    try:
        with io.open(F, encoding="utf-8") as fh:
            src = fh.read()
        break
    except OSError:
        time.sleep(3)
if src is None:
    raise SystemExit("INCONCLUSIVE: could not read GOALS_AND_CLAIMS.md (mount)")

if "D-RL-GRADED-TERM-1" in src:
    print("ALREADY APPLIED")
    sys.exit(0)

i = src.find(ANCH)
if i < 0:
    raise SystemExit("REFUSED: anchor %r not found" % ANCH)
j = src.find("\n", i)
out = src[:j] + "\n" + ROW + src[j:]

for _ in range(12):
    try:
        with io.open(F, "w", encoding="utf-8", newline="") as fh:
            fh.write(out)
        break
    except OSError:
        time.sleep(3)

with io.open(F, encoding="utf-8") as fh:
    back = fh.read()
n = back.count("| D-RL-GRADED-TERM-1 |")
c = back.count("| D-RL-COLL-PRICE-1 |")
print("D-RL-GRADED-TERM-1 occurrences=%d (want 1)" % n)
print("D-RL-COLL-PRICE-1 CONTROL occurrences=%d (want 1; 0 means unreadable)" % c)
print("INSERT=%s" % ("PASS" if n == 1 and c == 1 else "FAIL"))
if not (n == 1 and c == 1):
    raise SystemExit(1)

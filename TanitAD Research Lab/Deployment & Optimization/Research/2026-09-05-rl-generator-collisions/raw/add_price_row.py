import io
import os
import sys
import time

F = os.path.join(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD",
                 "Project Steering", "GOALS_AND_CLAIMS.md")
PKG = "TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-rl-generator-collisions"

ROW = (
    "| D-RL-COLL-PRICE-1 | \u2b50\u2b50 **THE PRICE OF A COLLISION-FREE FAN, MEASURED WITHOUT "
    "TRAINING ANYTHING: 0.196 m OF AMORTISED FAN DISPLACEMENT AND ZERO COST ON THE DRIVEN PATH.** "
    "`fan_contact` is a property of the candidate SET, not the ranking, so no re-weighting changes "
    "it \u2014 the generator must MOVE the paths. Scaling each path toward the ego origin (shape "
    "preserved, distance reduced) until it clears the 2 m swept disc: **all 1,053 colliders are "
    "fixable by braking alone (100 %)**, needing a mean **5.73 m** displacement (median 4.84, p90 "
    "11.69) = a **42 %** mean retreat; **amortised over every emitted candidate that is 0.196 m "
    "[0.077, 0.346]** and takes `fan_contact` **0.034277 \u2192 0.000000**. \u2b50 **The DRIVEN "
    "path costs 0.000 m** \u2014 the selector already picks a non-collider in **240/240** windows, "
    "so a perfect generator fix moves the selected trajectory in ZERO windows. Controls: Q1 s=1.0 "
    "reproduces the fan exactly (0 disagreements), Q2 s=0.0 clears every candidate (no window "
    "unfixable), Q3 9 non-monotone re-entries of 1,053\u00d7101 reported. \u26a0 0.196 m is "
    "FAN-WIDE mean displacement, NOT a T1 ADE delta \u2014 different objects, never compared "
    "directly. \u26a0 Braking does not repair feasibility (`fan_infeasible` 0.891960 \u2192 "
    "0.891960): collisions and feasibility are ORTHOGONAL defects, the same conclusion "
    "`D-RL-FEASDECODE-CONTACT-1` reaches from the other side | **SUPPORTED (MEASURED, Q1/Q2 "
    "controls pass)** | `" + PKG + "/raw/p5_collision_price.json`; `" + PKG + "/RESULT.md` \u00a75 |"
)

ANCH = "| H-RL-COLL-1 |"

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

if "D-RL-COLL-PRICE-1" in src:
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
n = back.count("| D-RL-COLL-PRICE-1 |")
c = back.count("| H-RL-COLL-1 |")
print("D-RL-COLL-PRICE-1 occurrences=%d (want 1)" % n)
print("H-RL-COLL-1 CONTROL occurrences=%d (want 1; 0 means unreadable)" % c)
print("INSERT=%s" % ("PASS" if n == 1 and c == 1 else "FAIL"))
if not (n == 1 and c == 1):
    raise SystemExit(1)

import io
import os
import sys
import time

F = os.path.join(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD",
                 "Project Steering", "GOALS_AND_CLAIMS.md")
PKG = "TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-rl-generator-collisions"

ROWS = [
    ("D-RL-FIRSTSEG-1",
     "\u2b50 **THE FIRST-SEGMENT BLIND SPOT IN `rewards._collision` IS REAL AND, ON THIS CORPUS, "
     "IMMATERIAL \u2014 +0.19 %.** M39 (`c9ab82c`) names the defect (`rel = lead[...,1:,:] - "
     "traj[...,1:,:]` drops index 0, so the **t0\u2192t1 segment is never swept** and a plan "
     "driving through a car in the first step can read CLEAR) but does not size it, and the size "
     "is what decides whether a structural zero on this metric is a zero on the road. MEASURED, "
     "0 GPU: stock **1,053** colliders \u2192 **1,055** with the first segment swept = **+2 = "
     "+0.19 %**, **0** of 19 collider-bearing windows gains anything, and **0 / 65** leads are "
     "within 2 m at t0. Controls: **P2** the reimplementation restricted to segments 1\u20133 "
     "reproduces the stock flag with **0 disagreements**, so the difference is attributable to "
     "the first segment and nothing else; **P1** no candidate loses its flag under a wider sweep; "
     "**P3** the t0 condition never varies by candidate within a window. \u21d2 **it does NOT "
     "undermine M39's structural zero** \u2014 reporting the defect without its size would have "
     "been alarming and wrong. \u26a0 Immaterial **only here**: this is a lead-following corpus "
     "with no lead within 2 m at t0; on a corpus with close cut-ins or a parked obstacle "
     "(M39's own example) it is not, so the fix is still worth making \u2014 as a SEPARATE lever, "
     "because folding it into the graded term moved the term's SUPPORT as well as its GRADING and "
     "broke `one_variable` (caught by controls S1/S3 failing)",
     "**SUPPORTED (MEASURED, 3 controls pass)**",
     "`" + PKG + "/raw/p7_first_segment_blindspot.json`; `" + PKG + "/RESULT.md` \u00a76.6"),

    ("D-RL-TIMING-SURFACE-1",
     "\u2b50 **THE NEW RL ENDPOINT'S SURFACE, MEASURED ON THE RAW FAN: A 0.5 s LEAD-TRACK ERROR "
     "IN THE RISK DIRECTION MULTIPLIES CONTACT BY 1.16x, AND THERE IS NO SAFE THRESHOLD.** M39 "
     "retires `fan_contact` (solved by construction) and names robustness to agent-MOTION "
     "PREDICTION error as the corrected objective, quoting **2.86 %** at a 0.5 s shift on the "
     "PROJECTED fan. This is the **RAW-fan baseline that number must be read against** \u2014 a "
     "different object, deliberately. MEASURED, 0 GPU, swept definition: `fan_contact` "
     "**0.034277** at dt = 0 \u2192 **0.039811 (1.16x, +0.55 pp)** at dt = \u22120.5 s \u2192 "
     "**0.066243 (1.93x)** at dt = \u22121.0 s, and **0.031413 (0.92x)** at dt = +0.5 s. The "
     "surface is **MONOTONE across the whole \u22121.0\u2026+2.0 s sweep and roughly linear**, so "
     "**there is no threshold to sit safely below** \u2014 which is exactly the shape that makes "
     "it a learnable objective rather than a constraint. \u26a0 **SIGN SEMANTICS STATED AS "
     "GEOMETRY, not as a label**, because \"early\" is ambiguous about which way risk runs: "
     "dt < 0 places the lead EARLIER along its own path, i.e. CLOSER to a following ego, and "
     "that is the risk direction. \u2b50 And `D-RL-FEASDECODE-CONTACT-1` HOLDS UNDER "
     "PERTURBATION: the friction projection is worse than raw at every shift (0.041699 vs "
     "0.039811 at the risk end). Controls: **T1** dt = 0 reproduces the base exactly; **T2** "
     "dt = +0.5 s from the interpolator equals a pure index shift computed without it (0 "
     "disagreements); **T3** extrapolated grid points disclosed per row",
     "**SUPPORTED (MEASURED, T1/T2 controls pass)**",
     "`" + PKG + "/raw/p8_timing_error_surface.json`; `" + PKG + "/RESULT.md` \u00a76.8"),
]

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

changed = False

# ---- 1. resolve H-RL-COLL-1 to its committed FAILURE ----
OLD_STATUS = ("| **OPEN \u2014 pre-registered 2026-09-05 before any arm produced a number; "
              "arms running on the dev-box 4060** |")
NEW_STATUS = ("| **RESOLVED \u2014 FAILURE against the committed SUCCESS text (2026-09-05).** "
              "SUCCESS required `fan_contact` to decrease **separated at BOTH seeds**. It is "
              "separated on **s0 only** (\u22120.001693 sep / \u22120.001693 ns, same sign), and "
              "`top32_contact` and `fan_unsafe` **disagree in sign** across seeds. \u2b50 **The "
              "gain is not replicated; the COSTS are** \u2014 `fan_peak_g_mean` (+0.0253 / "
              "+0.0407) and `fan_infeasible` (+0.0022 / +0.0025) are separated at both seeds, "
              "same sign, and LARGER on the second. This is the veto arm's trade for the third "
              "time. \u26a0 Both seeds report an episode-clustered delta of \u22120.001692708333 "
              "to twelve decimals while their AFTER states genuinely differ (611/4608 vs "
              "613/4608) and their full-population changes are \u22120.001519 vs \u22120.001085 "
              "\u2014 a cluster-weighting artefact, **not** evidence of a stable effect. "
              "\u21d2 endpoint retired by M39/escalation `24bd5e8`; successors S2 (built, five "
              "controls pass) and `D-RL-TIMING-SURFACE-1` carry the work forward |")
if OLD_STATUS in src:
    src = src.replace(OLD_STATUS, NEW_STATUS, 1)
    changed = True
    print("H-RL-COLL-1 status: UPDATED to RESOLVED-FAILURE")
elif "RESOLVED \u2014 FAILURE against the committed SUCCESS text (2026-09-05).** SUCCESS required `fan_contact`" in src:
    print("H-RL-COLL-1 status: already resolved")
else:
    print("H-RL-COLL-1 status: ANCHOR NOT FOUND -- left unchanged, reported not silently skipped")

# ---- 2. append the two new rows ----
ANCH = "| D-RL-GRADED-TERM-1 |"
to_add = [r for r in ROWS if ("| %s |" % r[0]) not in src]
if to_add:
    i = src.find(ANCH)
    if i < 0:
        raise SystemExit("REFUSED: anchor %r not found" % ANCH)
    j = src.find("\n", i)
    block = "".join("\n| %s | %s | %s | %s |" % r for r in to_add)
    src = src[:j] + block + src[j:]
    changed = True
else:
    print("rows already present")

if changed:
    for _ in range(14):
        try:
            with io.open(F, "w", encoding="utf-8", newline="") as fh:
                fh.write(src)
            break
        except OSError:
            time.sleep(4)

with io.open(F, encoding="utf-8") as fh:
    back = fh.read()
ok = True
for rid, _, _, _ in ROWS:
    n = back.count("| %s |" % rid)
    print("  %-24s occurrences=%d %s" % (rid, n, "OK" if n == 1 else "FAIL"))
    ok = ok and n == 1
ctrl = back.count("| D-RL-GRADED-TERM-1 |")
res = back.count("RESOLVED \u2014 FAILURE against the committed SUCCESS text (2026-09-05).** SUCCESS required `fan_contact`")
print("  %-24s occurrences=%d  <- CONTROL, must be 1" % ("D-RL-GRADED-TERM-1", ctrl))
print("  H-RL-COLL-1 resolution present=%d (want 1)" % res)
print("INSERT=%s" % ("PASS" if (ok and ctrl == 1 and res == 1) else "FAIL"))
if not (ok and ctrl == 1 and res == 1):
    raise SystemExit(1)

#!/usr/bin/env python3
"""REPAIR `D-RL-VETO-T1-1`: bash ate every backticked metric name out of the status cell.

WHAT HAPPENED. The update was applied with `python -c "...`ade_m`..."` inside a DOUBLE-quoted
bash string, so every `backtick` pair was COMMAND SUBSTITUTION: bash tried to run `ade_m`,
`LON_accel_mae`, ... printed "command not found" to stderr, and substituted the empty string.
The row committed with its numbers intact and **every metric name deleted** -- e.g.
" **+0.0362 / +0.0475 against a 0.0113 seed-replicate floor**", which is precisely the
"a number carries its arm or it is not quotable" failure the programme's own rules exist to
prevent. The python printed its success line, so nothing looked wrong except eight stderr
lines that were easy to skim past.

DURABLE RULE: never put markdown backticks inside a double-quoted bash string. Write the edit
to a SCRIPT FILE (as this one is) or use a single-quoted heredoc.

This also fixes two smaller defects introduced by the same generation pass: the literal
`{{json,md}}` (an unsubstituted format-brace escape) and artifact paths naming only s0 when
the row now reports both seeds.
"""
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
PATH = os.path.join(REPO, "Project Steering", "GOALS_AND_CLAIMS.md")
CR, LF = chr(13), chr(10)
CRLF = CR + LF
ST, ED, ELL = chr(11088), chr(8212), chr(8230)
BT = chr(96)


def q(name):
    return BT + name + BT


BROKEN = ("s1 landed 17:46 Z:  **+0.0362 / +0.0475 against a 0.0113 seed-replicate floor "
          + ED + " QUOTABLE, the ADE guard fails on BOTH seeds**;  +0.3349/+0.2157 (floor "
          "0.1192),  +0.0779/+0.0628 (0.0151),  +0.0290/+0.0312 (0.0022),  +0.2912/+0.4517 "
          "(0.1605),  -0.0889/-0.0651 (0.0238) all QUOTABLE. " + ST + " And the replicate cut "
          "both ways:  (+0.0154/+0.0322, floor 0.0168) and  (+0.0191/+0.0478, floor 0.0287) "
          "fall to **WITHIN-NOISE** having looked separated on seed 0 alone.")

FIXED = ("s1 landed 17:46 Z: " + q("ade_m") + " **+0.0362 / +0.0475 against a 0.0113 "
         "seed-replicate floor " + ED + " QUOTABLE, the ADE guard fails on BOTH seeds**; "
         + q("fde_m") + " +0.0623/+0.0677 (floor 0.0054), "
         + q("LON_accel_mae_mps2") + " +0.3349/+0.2157 (0.1192), "
         + q("LON_speed_mae_mps") + " +0.0779/+0.0628 (0.0151), "
         + q("LON_along_mae_m") + " +0.0290/+0.0312 (0.0022), "
         + q("LAT_heading_mae_deg") + " +0.2912/+0.4517 (0.1605), "
         + q("TAC_traj_lon_correct") + " -0.0889/-0.0651 (0.0238) all QUOTABLE. " + ST
         + " And the replicate cut both ways: " + q("LAT_cross_mae_m")
         + " (+0.0154/+0.0322, floor 0.0168) and " + q("LAT_yaw_rate_mae_radps")
         + " (+0.0191/+0.0478, floor 0.0287) fall to **WITHIN-NOISE** having looked separated "
           "on seed 0 alone.")

raw = open(PATH, "rb").read()
is_crlf = CRLF.encode() in raw
s = raw.decode("utf-8").replace(CRLF, LF)
# GUARD ON THE BROKEN TEXT, NOT ON THE FIXED TEXT. The first version keyed "already
# repaired" on the presence of "`ade_m` **+0.0362", which ALSO appears in this row's
# DESCRIPTION cell (inserted by a script file, so its backticks survived) -- so it reported
# success without touching the damaged STATUS cell. A guard must be specific to the thing it
# is guarding; a substring shared with healthy text is not one.
n = s.count(BROKEN)
if n == 0:
    raise SystemExit("[repair] the broken status text is not present - nothing to repair")
if n != 1:
    raise SystemExit("[repair] broken text found %d times - refusing" % n)
s = s.replace(BROKEN, FIXED)

# the unsubstituted format-brace escapes, and s0-only artifact paths in this row
i = s.find("| D-RL-VETO-T1-1 |")
j = s.find(LF, i)
row = s[i:j]
row2 = (row.replace("{{json,md}}", "{json,md}")
           .replace("`raw/run/paired_s0-veto200_vs_base.{json,md}`",
                    "`raw/run/paired_s{0,1}-veto200_vs_base.{json,md}`")
           .replace("`raw/run/eval/refcv3-40284-s0-veto200.{json,md}`",
                    "`raw/run/eval/refcv3-40284-s{0,1}-veto200.json`, "
                    "`raw/run/eval/criteria_s{0,1}-veto200.json`"))
s = s[:i] + row2 + s[j:]
open(PATH, "wb").write((s.replace(LF, CRLF) if is_crlf else s).encode("utf-8"))
print("[repair] D-RL-VETO-T1-1 metric names restored (%s)" % ("CRLF" if is_crlf else "LF"))

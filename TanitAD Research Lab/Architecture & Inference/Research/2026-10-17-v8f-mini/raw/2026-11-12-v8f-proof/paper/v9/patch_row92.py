# patch_row92.py -- lead's patch of paper v9 after register row 92 was written (ROW92.md). Runs after patch_tex_v9.py.
# Four replacements in tanitad_4b_paper.tex, each asserted to match exactly once. Reads and writes text only.
import io
import sys

P = "tanitad_4b_paper.tex"
s = io.open(P, encoding="utf-8", newline="").read()
EDITS = [
    ("register rows 84 and 86 to 91; row 92 in preparation",
     "register rows 84 and 86 to 92"),
    ("register row 92 in preparation",
     "row 92"),
    ("The register row for this read is in preparation (register row 92); the numbers below are read from the three name files of the level study",
     "The read is register row 92; the numbers below are those of the three name files of the level study"),
    ("The family and headway reads of these arms are not part of this version.",
     "The family and headway reads of these arms (\\tier{T1-kin}; each arm against the control and against the low weight, bases 0 and 1) "
     "name NO-COST in all six pairs, with no cell outside the six-seed spread of the baseline; the model-induced unsafe-headway rate against "
     "the control differs by $+0.0044$ $[-0.0133,+0.0212]$ (RLVa), $+0.0000$ $[-0.0086,+0.0098]$ (RLVb) and $+0.0027$ $[-0.0126,+0.0172]$ (RLAh), "
     "every interval containing 0. The control that the terms are alive passes on all six runs (every term an arm carries is finite on 120 of 120 logged steps)."),
]
n = 0
for a, b in EDITS:
    c = s.count(a)
    if c != 1:
        print("ZZ PATCH-ROW92 STOP: expected one match, found", c, "for:", a[:60])
        sys.exit(2)
    s = s.replace(a, b)
    n += 1
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("ok edits", n)

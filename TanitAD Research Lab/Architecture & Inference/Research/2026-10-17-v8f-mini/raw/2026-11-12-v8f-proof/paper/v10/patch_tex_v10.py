# patch_tex_v10.py -- paper version 10: register rows 93 to 96 and 98 (and 97, 99 when their fragment files exist) on top of version 9.
# Reads ../v9/tanitad_4b_paper.tex (not edited), writes tanitad_4b_paper.tex here. Every edit is asserted to match exactly once.
# Optional fragments: frag_row99.tex, frag_row97.tex (inserted at their markers of frag_new.tex; a missing file leaves no text),
# frag_status_ii.tex is item (ii) of the running-or-queued paragraph. Usage: python patch_tex_v10.py "<rows text for the date line>"
import io
import os
import sys

SRC = "../v9/tanitad_4b_paper.tex"
DST = "tanitad_4b_paper.tex"
rows_text = sys.argv[1]
s = io.open(SRC, encoding="utf-8", newline="").read()
n = [0]


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def once(a, b):
    global s
    c = s.count(a)
    if c != 1:
        print("ZZ PATCH-V10 STOP: expected one match, found %d for: %s" % (c, a[:90]))
        sys.exit(2)
    s = s.replace(a, b)
    n[0] += 1


def par(start, new):
    """replace the whole one-line paragraph that starts with <start>"""
    global s
    lines = s.split("\n")
    hits = [i for i, ln in enumerate(lines) if ln.startswith(start)]
    if len(hits) != 1:
        print("ZZ PATCH-V10 STOP: expected one paragraph, found %d for: %s" % (len(hits), start[:90]))
        sys.exit(2)
    lines[hits[0]] = new.rstrip("\n")
    s = "\n".join(lines)
    n[0] += 1


BS = chr(92)
# E1 date line
once(BS + "date{November 2026 (version 9, 2026-11-19; register rows 84 and 86 to 92)}",
     BS + "date{November 2026 (version 10, 2026-11-19; register rows 84 and " + rows_text + ")}")
# E2 level section: the open paragraph becomes the read of row 96
par(BS + "paragraph{What follows, and what is not known.} Note ADD37", rd("frag_level.tex"))
# E3 the p5b paragraph: the read on the responding arm exists now
once("so no planner read on it is taken as a result, and DOSE-RESPONDS-ON-OWN-PLANS is not given.",
     "so a planner read on it carries that limit (row 95, " + BS + "cref{sec:fullimag}), and DOSE-RESPONDS-ON-OWN-PLANS is not given.")
# E4 the closing paragraph of the replan subsection
once("The planner on the imagined gap is read next on an arm that both responds on its own plans and keeps the level (" + BS + "cref{sec:level}); none exists yet.",
     "An arm that both responds on its own plans and keeps the level does not exist (" + BS + "cref{sec:level}); the planner on the full imagined rollout was therefore read on the arm that responds, as the card writes it (" + BS + "cref{sec:fullimag}).")
# E5 new subsections before the data-efficiency subsection
new = rd("frag_new.tex")
for tag, f in (("%%ROW99%%", "frag_row99.tex"), ("%%ROW97%%", "frag_row97.tex")):
    if new.count(tag) != 1:
        print("ZZ PATCH-V10 STOP: marker", tag)
        sys.exit(2)
    new = new.replace(tag + "\n\n", (rd(f).rstrip("\n") + "\n\n") if os.path.exists(f) else "")
anchor = BS + "subsection{Data efficiency and the four families}" + BS + "label{sec:dataeff}"
once(anchor, new + anchor)
# E6 status: not met
once("The level term loses at both frozen weights, the level at 0.001 and the response at 0.01, and the third dose point of the consistency weight breaks the level (" + BS + "cref{sec:level}).",
     "The level term works at none of four weights (LEVEL-WEIGHT-NO-WINDOW, rows 92 and 96), a calibration outside the predictor cannot repair the level, and the third dose point of the consistency weight breaks the level (" + BS + "cref{sec:level}). The planner on the full imagined rollout equals extrapolation and does not reach the label-fed planner (row 95, " + BS + "cref{sec:fullimag}). The planner with the trained gap head has a confirmed price in two families and reaches the label-fed planner on one arm type only (row 99). The replicate of the finer token grid is not named: its train-log control failed a third time, and its positive name is out of reach on the per-arm values (row 97).")
# E7 status: running or queued, item (ii)
st = rd("frag_status_ii.tex").rstrip("\n")
once("(ii) The two interior weights of the level term (" + BS + "cref{sec:level}), in training. A pass at bases 0 and 1 counts as a candidate until two new bases confirm it.", st)
# E8 status: established
once("confirmed on two new seeds (rows 80, 88).",
     "confirmed on two new seeds (rows 80, 88); descriptive, at 2.0" + BS + ",s the read of the lead ranks before the forecast of its motion, and inside the read the gap before the closing speed (row 94). At " + BS + "tier{T1-kin}: the planner with the trained gap head cuts the unsafe rate below the plain constant-speed planner at all four checkpoints of two arm types (QPLAN-CUTS-MORE-CONFIRMED, row 99).")
# E10 status: running or queued, item (i) has a result now
once("(i) Card ADD43 with note N1 (pre-registered, no result): a probe-only replicate of the finer token grid with eight new seeds and a two-level read, with the shuffled-label check defined on the intact-label part, under coordination decision D1 on a slot the PI already approved.",
     "(i) Card ADD43 with note N1 (a probe-only replicate of the finer token grid with eight new seeds and a two-level read) was run and is not named (row 97, " + BS + "cref{sec:fullimag}).")
# E9 status: handed to the integrated tree
once("gives statuses only; no number is added here.",
     "gives statuses only; no number is added here. Each lever is handed over with a counted mutation list (row 93, " + BS + "cref{sec:handover}). The two sessions' code lines split on 2026-11-08; by a coordination decision of 2026-11-19 the integrated candidate at 10k clips is built on the line of the other session in two stages: first that tree's own lead head with the distance rule, the trained gap head and the planner as tools outside the trainer, then, only if the pre-test of card ADD44 shows that it matters, the consistency term with its rollout prerequisites.")
io.open(DST, "w", encoding="utf-8", newline="").write(s)
print("ok edits %d" % n[0])

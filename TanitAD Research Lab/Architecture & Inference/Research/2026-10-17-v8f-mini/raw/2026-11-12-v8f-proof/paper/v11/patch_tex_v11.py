# patch_tex_v11.py -- paper version 11: register rows 100 to 125 on top of version 10.
# Reads ../v10/tanitad_4b_paper.tex (not edited), writes tanitad_4b_paper.tex here. Every edit is asserted to match exactly once.
# Optional fragments: frag_row126.tex, frag_row127.tex (inserted at their markers of frag_strat.tex; a missing file leaves no text).
# Usage: python patch_tex_v11.py
import io
import os
import sys

SRC = "../v10/tanitad_4b_paper.tex"
DST = "tanitad_4b_paper.tex"
s = io.open(SRC, encoding="utf-8", newline="").read()
n = [0]
BS = chr(92)


def rd(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def once(a, b):
    global s
    c = s.count(a)
    if c != 1:
        print("ZZ PATCH-V11 STOP: expected one match, found %d for: %s" % (c, a[:90]))
        sys.exit(2)
    s = s.replace(a, b)
    n[0] += 1


def par(start, new):
    """replace the whole one-line paragraph that starts with <start>"""
    global s
    lines = s.split("\n")
    hits = [i for i, ln in enumerate(lines) if ln.startswith(start)]
    if len(hits) != 1:
        print("ZZ PATCH-V11 STOP: expected one paragraph, found %d for: %s" % (len(hits), start[:90]))
        sys.exit(2)
    lines[hits[0]] = new.rstrip("\n")
    s = "\n".join(lines)
    n[0] += 1


# E1 date line
once(BS + "date{November 2026 (version 10, 2026-11-19; register rows 84 and 86 to 99)}",
     BS + "date{November 2026 (version 11, 2026-11-20; register rows 84 and 86 to 125)}")

# E2 new subsections before the data-efficiency subsection
strat = rd("frag_strat.tex")
for tag, f in (("%%ROW126%%", "frag_row126.tex"), ("%%ROW127%%", "frag_row127.tex")):
    if strat.count(tag) != 1:
        print("ZZ PATCH-V11 STOP: marker", tag)
        sys.exit(2)
    strat = strat.replace(tag + "\n\n", (rd(f).rstrip("\n") + "\n\n") if os.path.exists(f) else "")
new = rd("frag_integ.tex") + "\n" + rd("frag_cycle.tex") + "\n" + strat + rd("frag_scale.tex") + "\n"
anchor = BS + "subsection{Data efficiency and the four families}" + BS + "label{sec:dataeff}"
once(anchor, new + anchor)

# E3 four-family table: four more rows after the last row
last = "Future-token target PF (P) & NO-COST (speed error INSIDE-SPREAD, headway U open) & NO-COST & GAIN-TACTICAL: joint error rate BETTER & NOT COMPUTED (n = 0) " + BS + BS + "\n"
once(last, last + rd("frag_fam7rows.tex"))

# E4 strategic family paragraph
once("STRATEGIC is paused (before the ruling) or not computed (after it).",
     "STRATEGIC is paused (before the ruling) or not computed (after it). For the reads of rows 120 to 122 the family has an instrument along the rollout (" + BS + "cref{sec:strat}): the strategic head's route checkpoints at 30, 50 and 80" + BS + ",m, scored at the third replan tick against the programme's route label, a study instrument that by the decision of the PI of 2026-11-20 (row 125) is the strategic read at " + BS + "tier{T1-kin} beside the first-tick read. A decision between routes is still not scored (n = 0), and the earlier reads keep their entry as written.")

# E5 headline table: one compact row
tail = "(IMAG-LEVEL-WORSE-THAN-EXTRAPOLATION) & T1-kin " + BS + BS + "\n" + BS + "bottomrule\n"
once(tail, "(IMAG-LEVEL-WORSE-THAN-EXTRAPOLATION) & T1-kin " + BS + BS + "\n" + rd("frag_t1row.tex") + BS + "bottomrule\n")

# E6 closing paragraph before the status section
once("The replicate of the finer token grid (card ADD43, probe only, eight new seeds) is pre-registered and has no result. What is running and what is open is in " + BS + "cref{sec:status}.",
     "The replicate of the finer token grid (card ADD43, probe only, eight new seeds) was run and is not named. Since version 10, " + BS + "cref{sec:integ10k,sec:cycle,sec:strat,sec:fracscale} read the lead rule on the frozen 10k checkpoints, the replan cycle and the acceleration the plan reads from its own rollout, the strategic read along the rollout, a heading bound on the executor and the inference-side candidate, and the lead read against the number of clips; by the decision of the PI of 2026-11-20 the bounded executor is the tier of record from the overlap read on. What is running and what is open is in " + BS + "cref{sec:status}.")

# E7 status: established
once("(QPLAN-CUTS-MORE-CONFIRMED, row 99).",
     "(QPLAN-CUTS-MORE-CONFIRMED, row 99). On the two frozen 10k checkpoints, under the unbounded executor: the lead rule fed by a token reader cuts the unsafe rate from 0.146 and 0.114 to 0.084 and 0.052 (INTEG10K-CUTS-UNSAFE, rows 101 and 102), and the rule fed by labels with the closing speed cuts it again (CLOSE10K-CLOSING-CUTS-UNSAFE-AT-NO-PRICE, row 104); the acceleration the plan reads from its own rollout raises the errors of three families with its dose (row 119); a heading bound on the executor lowers the errors of all four families and the unsafe rate in the convention (HEADBOUND10K-PINNED-U-LOWER, row 122); the inference-side candidate reads better than the convention with the rule on all ten family cells and on the route checkpoints, with the unsafe rate not separated (row 121). The strategic head's route goal degrades along the model's own rollout (STRATROLL10K-STRATEGIC-DEGRADES-ON-ROLLOUT, row 120). At " + BS + "tier{T0}: twice the clips of the same collection do not lift the lead gap read from the pooled tokens at matched steps (HALFSCALE-LEAD-READ-FLAT-AT-SCALE, row 117).")

# E8 status: not met
once("and the failed controls p2b, f2, k5, q3 and d4.",
     "and the failed controls p2b, f2, k5, q3 and d4. The reader-fed rule keeps its price: falling behind and the 6" + BS + ",s position error (INTEG10K-PRELIMINARY, row 101), and its unsafe rate is above the label-fed rule in the candidate: $+0.0374$ $[+0.0210,+0.0564]$ (row 121). Neither way of feeding the reader's own closing speed passed its control (rows 105 and 106, the third read that misses the same bar). A longer or shorter time gap and a higher braking limit bring no cut of the unsafe rate at label quality (rows 107 and 113). The 1.0" + BS + ",s cycle raises the unsafe rate for every form of the rule under the pinned readout (row 108) and under the backward readout (row 114), and no estimate of the executed acceleration steadies the loop (rows 111 to 116). The quarter and the eighth run of the lead probe name nothing, so the scale curve of card ADD64 has one named fraction and no curve name (rows 123 and 124). The whole candidate does not lower the unsafe rate against the convention with the rule, in either form (row 121).")

# E9 status: running or queued
par(BS + "paragraph{Running or queued.}", rd("frag_status_run.tex"))

# E10 status: handed over
once("Row ids refer to the register.",
     "By the decision of the PI of 2026-11-20 (row 125, " + BS + "cref{sec:strat}) the integrated candidate is specified with the lead rule and its second look in the label-fed and the reader-fed form, reported under the convention and under readout K side by side; readout K stays a study device. Row ids refer to the register.")

# E11 float placement of the headline table: [!tp] lets it take a float page (with [!t] it stayed in the queue behind its height and
# every later single-column float followed it past the references)
HEAD = BS + "begin{table}[!%s]" + chr(10) + BS + "centering" + chr(10) + BS + "caption{" + BS + "textbf{Headline results}"
once(HEAD % "t", HEAD % "tp")

io.open(DST, "w", encoding="utf-8", newline="").write(s)
print("ok edits %d" % n[0])

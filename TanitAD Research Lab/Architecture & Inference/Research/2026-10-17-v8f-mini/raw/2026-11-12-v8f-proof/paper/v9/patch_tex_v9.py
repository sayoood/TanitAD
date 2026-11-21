import os
D = 'F:/TanitAD-scratch/v8proof/paper/'
src = open(D + 'v8/tanitad_4b_paper.tex', encoding='utf8', newline='').read()
s = src.replace('\r\n', '\n')
n_edits = 0


def rd(name):
    return open(D + 'v9/' + name, encoding='utf8').read()


def rep(old, new):
    global s, n_edits
    c = s.count(old)
    assert c == 1, (c, old[:90])
    s = s.replace(old, new)
    n_edits += 1


# ---- version line
rep(r"\date{November 2026}",
    r"\date{November 2026 (version 9, 2026-11-19; register rows 84 and 86 to 91; row 92 in preparation)}")

# ---- abstract: from the consistency term to the end of the 'in training' sentence
a0 = "A kinematic consistency term turns the response (ACT-CONS-RESPONDS);"
a1 = "because a launcher did not start. "
assert s.count(a0) == 1 and s.count(a1) == 1
i = s.index(a0)
j = s.index(a1) + len(a1)
assert i < j and j - i < 4000
new_abs = (
r"A kinematic consistency term turns the response (ACT-CONS-RESPONDS); on bases trained with the rollout loss its named weight breaks the imagination (ACT-CONS-FLAT-RL), and a weight 10,000 times lower responds (slope $+0.811$ $[+0.759,+0.866]$) and keeps the imagination read; on two bases of new seeds it responds at $+0.820$ $[+0.766,+0.878]$ with the level kept and no cost in the families (LOW-WEIGHT-CONFIRMED). A dose ladder finds no weight that answers the action at the kinematic size and keeps the level: 0.001 responds at $+0.971$ $[+0.907,+1.047]$ and misses the level bar, as does 0.01 (DOSE-BREAKS-AT-0.001, DOSE-BREAKS-AT-0.01). A level term inside the loss loses at both frozen weights, the level at 0.001 and the response at 0.01 (LEVEL-TERM-LOSES-LEVEL-0.001, LEVEL-TERM-LOSES-RESPONSE-0.01), and a third dose point of the consistency weight alone responds and breaks the level (DOSE-BREAKS-AT-0.0003). "
r"A head trained on the state with a low-quantile loss, used in the distance rule, is equal to the hand-calibrated margin on the unsafe rate and equal or cheaper on 6\,s error (QHEAD-U-EQUAL-QF, QHEAD-CHEAPER-QF, two checkpoints per type, \tier{T1-kin}); the cost of the rule against no rule stays (QHEAD-COSTS) and its contact rate lies above the label-fed rule. A planner that checks kinematic candidates against the imagined gap cuts the unsafe rate at the bases of two new seeds ($-0.0347$ $[-0.0582,-0.0146]$, IMAG-LEVEL-CUTS-UNSAFE-CONFIRMED) and is not better than plain extrapolation of the lead ($+0.0169$ $[+0.0035,+0.0316]$, IMAG-LEVEL-WORSE-THAN-EXTRAPOLATION); its response on the model's own plans is met only at the weight 0.001, whose level is not kept; the explicit future-token target is flat (TARGET-FLAT); a pilot of a finer token grid is not named because a control failed. "
r"Every number of this paragraph is a diagnostic on logged frames, none is driving performance; the STRATEGIC family is paused in the reads dated before the PI ruling of 2026-11-16 and not computed (n = 0) in the later ones. Pre-registered and running without a result: two interior weights of the level term and a replicate of the finer grid with eight seeds (card ADD43). "
)
s = s[:i] + new_abs + s[j:]
n_edits += 1

# ---- headline table
rep(r"response control not met, $+0.495$ and $+0.479$ against 0.5 & T1-kin \\" + "\n" + r"\bottomrule",
r"response control not met, $+0.495$ and $+0.479$ against 0.5 & T1-kin \\" + "\n"
r"Trained gap head in the distance rule (LSD, all training clips): U against the calibrated map $-0.0098$ $[-0.0302,+0.0103]$, 6\,s error $-0.199$\,m $[-0.391,-0.017]$ (QHEAD-U-EQUAL-QF, QHEAD-CHEAPER-QF); a cost against no rule stays (QHEAD-COSTS) & T1-kin \\" + "\n"
r"Consistency weight 0.0001 on two new bases: response $+0.820$ $[+0.766,+0.878]$ against $-0.746$ $[-0.823,-0.676]$, level ratios 1.141 and 1.059 against the bar 1.15, no cost (LOW-WEIGHT-CONFIRMED); weight 0.001: $+0.971$ $[+0.907,+1.047]$, level ratios 1.710 and 1.252 (DOSE-BREAKS-AT-0.001) & T0, T1-kin \\" + "\n"
r"Level term, weights 0.001 and 0.01: response $+0.9034$ $[+0.8554,+0.9535]$ with level ratios 1.1060 and 1.1977, and $-0.4032$ $[-0.5492,-0.2681]$ with 1.0395 and 1.0525 (LEVEL-TERM-LOSES-LEVEL-0.001, LEVEL-TERM-LOSES-RESPONSE-0.01; register row 92 in preparation) & T0 \\" + "\n"
r"Planner on imagined rollouts at two new bases: U $-0.0347$ $[-0.0582,-0.0146]$ against off (IMAG-LEVEL-CUTS-UNSAFE-CONFIRMED); against extrapolation of the lead $+0.0169$ $[+0.0035,+0.0316]$ (IMAG-LEVEL-WORSE-THAN-EXTRAPOLATION) & T1-kin \\" + "\n"
r"\bottomrule")

# ---- new results subsections
rep(r"\subsection{Data efficiency and the four families}\label{sec:dataeff}",
    rd('frag_qhead.tex') + "\n" + rd('frag_dose.tex') + "\n" + rd('frag_level.tex') + "\n" + rd('frag_replan.tex') + "\n"
    + r"\subsection{Data efficiency and the four families}\label{sec:dataeff}")

# ---- strategic paragraph: work item of the readers
rep(r"The work item is open: the reader writes the reason and the n in place of a fixed stamp, and the planner",
    r"The readers of this study now write the reason and the n in place of a fixed stamp (row 84: 2,401 values of the banked read of row 80 compared, 0 differing), and the planner")

# ---- roadmap
rep(r"the rule in the training target and on the model's own plan does not give it for free \\",
    r"the rule in the training target and on the model's own plan does not give it for free; a head trained with a low-quantile loss stands in for the map (equal on the unsafe rate, equal or cheaper on 6\,s error, a cost against no rule stays; two checkpoints per type) \\")
rep(r"a consistency term at weight 0.0001 responds and keeps the read on two bases (a dose arm; confirmation on new seeds and a dose ladder running); planning with kinematic candidates cuts the unsafe rate and equals extrapolation \\",
    r"a consistency term at weight 0.0001 responds and keeps the read, confirmed on two new seeds; heavier weights and the level term at both frozen weights lose the level or the response (two interior level weights in training); planning with kinematic candidates cuts the unsafe rate on four bases and is not better than extrapolation \\")

# ---- outlook paragraph
rep(r"What is running and what is open is in \cref{sec:status}.",
    r"Since version 8, \cref{sec:qhead,sec:dose,sec:level,sec:replan} read a trained gap head in the rule, a dose ladder of the consistency weight with the low weight confirmed on two new seeds, the level term and the planner at two new bases; the misses are stated there. Limits that bind all of them: one training run per base and arm and two checkpoints per head type, so a separated pair answers whether another draw of episodes would say the same and not whether another training run would; the rollout reads are kinematic on logged frames with a lead that does not react (T1-kin); the imagination reads are diagnostics (T0); and the interior weights of the level term were chosen after the first two reads were seen, so a pass of theirs is a candidate until two new bases confirm it. The replicate of the finer token grid (card ADD43, probe only, eight new seeds) is pre-registered and has no result. What is running and what is open is in \cref{sec:status}.")

# ---- status subsection
k0 = r"\subsection{Status of the study}\label{sec:status}"
k1 = r"\section{Conclusion}"
assert s.count(k0) == 1 and s.count(k1) == 1
i = s.index(k0)
j = s.index(k1)
assert i < j
s = s[:i] + rd('frag_status.tex') + s[j:]
n_edits += 1

# ---- conclusion
rep(r"brings the vision-fed rule to the label-fed one at a measured price (\tier{T1-kin}).",
    r"brings the vision-fed rule to the label-fed one at a measured price, which a trained low-quantile head matches while a cost against no rule stays (\tier{T1-kin}).")
rep(r"that term works at a small weight on two bases and is not yet confirmed on new seeds, and the planner built on it equals extrapolation of the lead at this precision",
    r"that term works at a small weight and is confirmed on two new seeds, the weights that answer the action at its full size lose the level, and the planner built on it cuts the unsafe rate and is not better than extrapolation of the lead")

open(D + 'v9/tanitad_4b_paper.tex', 'w', encoding='utf8', newline='').write(s)
print('ok edits', n_edits)

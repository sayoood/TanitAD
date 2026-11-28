import sys, hashlib, re
D = 'F:/TanitAD-scratch/v8proof/paper/'
src = open(D + 'v7/tanitad_4b_paper.tex', encoding='utf8', newline='').read()
s = src.replace('\r\n', '\n')
n_edits = 0

def rd(name):
    return open(D + 'v8/' + name, encoding='utf8').read()

def rep(old, new):
    global s, n_edits
    c = s.count(old)
    assert c == 1, (c, old[:80])
    s = s.replace(old, new)
    n_edits += 1

# ---- abstract
rep("for training from scratch the comparison is not established. All of it rests on an evaluation",
"for training from scratch the comparison is not established. "
"Since then the distance question has been taken apart (\\tier{T1-kin}). Of the shortfall of the vision-fed rule the gap read carries most (0.68 to 1.00 of it on its own), the closing speed next and presence little (GAP-LIMITS-RULE, CLOSING-LIMITS-RULE); a margin on the gap read calibrated on training clips brings the vision-fed rule to the label-fed rule on unsafe steps in both head arm types and both seeds (0.1246 to 0.0534 and 0.1343 to 0.0516, MARGIN-REACHES-ORACLE), at a price named as written (6\\,s position error $+0.945$\\,m and $+0.858$\\,m against $+0.18$\\,m for the label-fed rule, MARGIN-COSTS); a margin that grows with the read does the same (QCAL-REACHES-ORACLE) and the head's closing speed is not needed (ZERO-CLOSING-EQUAL). Putting the rule into the training target moves the plan and not the unsafe rate (RULE-TRAINED-FLAT at 2,000 and 6,000 steps and with a weight of 4); applying it to the model's own plan lowers the unsafe rate only by slowing the model everywhere (OWN-RULE-FLAT with OWN-RULE-COSTS, $+2.67$\\,m; the bar OWN-RULE-L1-BEATS-MARGIN is not met), and the lead read as an input of the plan head is not used (LEAD-INPUT-SCRATCH-UNUSED). On imagination (\\tier{T0}): the lead-gap gain of the predictor RR on two seeds is not confirmed on two new seeds ($+0.025$ $[-0.000,+0.057]$ on four, MOTION-ADDS-NOTHING-4), while its forecast of the future camera tokens replicates (RR-FORECASTS-TOKENS-BETTER-4); a head reads the lead gap from imagined states (IMAG-HEAD-ADDS), but the imagined gap responds to the ego's own acceleration with the wrong sign (slope $-0.468$ $[-0.530,-0.415]$, IMAG-DEAF), the sign of the association in the logged data itself ($-1.21$ $[-1.94,-0.30]$, LOG-ASSOCIATION-REVERSED). A kinematic consistency term turns the response (ACT-CONS-RESPONDS); on bases trained with the rollout loss its named weight breaks the imagination (ACT-CONS-FLAT-RL) and a weight 10,000 times lower responds (slope $+0.811$ $[+0.759,+0.866]$) and keeps the imagination read with no cost, as a dose arm that names nothing. A planner that checks kinematic candidates against the imagined gap cuts the unsafe rate (IMAG-LEVEL-CUTS-UNSAFE) and equals extrapolation of the lead (IMAG-LEVEL-EQUALS-EXTRAPOLATION); its response control is not met ($+0.495$ and $+0.479$ against 0.5) and the explicit future-token target is flat (TARGET-FLAT). Every number of this paragraph is a diagnostic on logged frames, none is driving performance; the STRATEGIC family is paused in the reads dated before the PI ruling of 2026-11-16 and not computed (n = 0) in the later ones. In training or queued now: the confirmation of the low-weight arm on new seeds and a dose ladder (card ADD37, notes N4 and N5) and a pre-registered pilot of a finer token grid (card ADD41); the pod sat without a job for about 65 hours because a launcher did not start. "
"All of it rests on an evaluation")

# ---- contributions
rep("  \\item \\textbf{Results under a decision-grade doctrine}",
"  \\item \\textbf{The rule in the loop and the imagination of the ego's action} (\\cref{sec:dist,sec:ruletrain,sec:imag}): the shortfall of the vision-fed rule is the gap read, a calibrated margin closes it at a measured price, the rule in the training target and on the model's own plan does not give the distance for free, and the world model's imagined gap answers the ego's action with the sign of the logs until a kinematic term is injected, which works only at a small weight on one base type and is not yet confirmed on new seeds.\n"
"  \\item \\textbf{Results under a decision-grade doctrine}")

# ---- status paragraph of the architecture section
rep("holding the action. No driving result exists yet for this design.",
"holding the action. No driving result exists yet for this design; the reads of \\cref{sec:results} are diagnostics on logged frames, and \\cref{sec:status} states what has been read since and what is running.")

# ---- premises intro: strategic stamp
rep("frames; the strategic family stays paused. Reads that do not yet exist",
"frames; the strategic family is paused for the reads dated before the PI ruling of 2026-11-16 and not computed for the later ones (\\cref{sec:fam7}). Reads that do not yet exist")

# ---- headline table
rep("quarter $+0.1108$\\,m $[-0.1505,+0.3822]$, open & T1-kin \\\\\n\\bottomrule",
"quarter $+0.1108$\\,m $[-0.1505,+0.3822]$, open & T1-kin \\\\\n"
"Distance rule, vision-fed with a margin on the gap read calibrated on training clips: U 0.1246 to 0.0534 (LS) and 0.1343 to 0.0516 (LSD), reaches the label-fed rule (MARGIN-REACHES-ORACLE); price: 6\\,s position error $+0.945$\\,m and $+0.858$\\,m (MARGIN-COSTS) & T1-kin \\\\\n"
"Rule in the training target: flat at 6,000 steps, $-0.0053$ $[-0.0281,+0.0119]$ (RULE-TRAINED-FLAT-6K); rule on the model's own plan: $-0.0276$ $[-0.0647,+0.0089]$ at $+2.67$\\,m (OWN-RULE-FLAT, OWN-RULE-COSTS) & T1-kin \\\\\n"
"Imagined lead gap, response to the ego's acceleration: $-0.468$ $[-0.530,-0.415]$ (IMAG-DEAF; the logs give $-1.21$ $[-1.94,-0.30]$); consistency term at weight 0.0001 on rollout-loss bases: $+0.811$ $[+0.759,+0.866]$, imagination read kept (a dose arm, names nothing) & T0 \\\\\n"
"RR rollout, lead information beyond the present state on four seeds: $+0.025$ $[-0.000,+0.057]$ (MOTION-ADDS-NOTHING-4); forecast of future tokens, RR minus G: $+0.52$ $[+0.41,+0.63]$ (RR-FORECASTS-TOKENS-BETTER-4) & T0 \\\\\n"
"Planning by imagination, kinematic candidates: unsafe rate $-0.043$ $[-0.082,-0.013]$ against no tool, equal to extrapolation of the lead (IMAG-LEVEL-EQUALS-EXTRAPOLATION); response control not met, $+0.495$ and $+0.479$ against 0.5 & T1-kin \\\\\n"
"\\bottomrule")

# ---- token probe
rep("No probe arm has run on real tokens; the job is queued on the pod. If the pooled tokens are the limit, a finer grid or a crop of the lead region needs the camera frames re-encoded, a data and compute step the PI decides (named as a blocker with its cost, not run).",
"The probe arms have since been run on real tokens; the read follows this paragraph. A finer grid or a crop of the lead region at full resolution would need the camera frames re-encoded, a data and compute step the PI decides.")
rep("\\subsection{Distance keeping, and a rule on the plan}\\label{sec:dist}", rd('frag_tokprobe.tex') + "\\subsection{Distance keeping, and a rule on the plan}\\label{sec:dist}")

# ---- head-fed paragraph tail
rep("card ADD29 (RULEATTR, frozen before its numbers, rollouts running) swaps presence, gap and closing speed one at a time from label to head to find which part holds the vision-fed rule back, and then comes the training card with the rule in the loop.",
"card ADD29 (RULEATTR, frozen before its numbers) swapped presence, gap and closing speed one at a time from label to head to find which part holds the vision-fed rule back; the reads that followed are given below.")

# ---- new blocks before data efficiency
rt = rd('frag_ruletrain.tex').replace("with the corrected control q3b of note N1", "with the control q3b of note N1")
rt = rt.replace("0.240/0.246 to 0.139/0.153 in the linear form", "0.2404/0.2455 to 0.1388/0.1530 in the linear form")
rep("\\subsection{Data efficiency and the four families}\\label{sec:dataeff}",
    rd('frag_dist.tex') + rt + rd('frag_imag.tex') + "\\subsection{Data efficiency and the four families}\\label{sec:dataeff}")

# ---- families paragraph + new table
rep("The strategic family is reported as paused by the PI ruling (item 25), never as a pass. The \\tier{T0} reads of this section measure no\nfamily: their four-family statement is that LONGITUDINAL, LATERAL and TACTICAL are not measured by them and STRATEGIC is paused.",
rd('frag_fam7.tex'))

# ---- roadmap
rep("about 13 to 32 percent with the model's own head; next, attribution of the shortfall (card ADD29, running) \\\\",
    "about 13 to 32 percent with the model's own head; the shortfall is the gap read, and a calibrated margin reaches the label-fed rule at a price of $+0.86$ to $+0.95$\\,m of 6\\,s position error (T1-kin); the rule in the training target and on the model's own plan does not give it for free \\\\\n"
    "Imagination and planning \\textit{(measured, T0 and T1-kin)} & the imagined gap answers the ego's action with the sign of the logs; a consistency term at weight 0.0001 responds and keeps the read on two bases (a dose arm; confirmation on new seeds and a dose ladder running); planning with kinematic candidates cuts the unsafe rate and equals extrapolation \\\\")
rep("lead supervision reads gap and presence, not closing speed (token probe queued) \\\\",
    "lead supervision reads gap and presence, not closing speed (the token probe reads no better than the grid state; its closing-speed read is not decidable) \\\\")

# ---- outlook paragraph
i = s.index("The next reads are committed in the cards: the attribution")
j = s.index("The deciding six-seed replicate of the hierarchy against its flat control is read")
assert s.count("The next reads are committed in the cards: the attribution") == 1
s = s[:i] + "The reads since then are in \\cref{sec:dist,sec:ruletrain,sec:imag}: the vision-fed rule is limited by the gap read and a calibrated margin closes the gap at a price; the rule as a training signal moves the plan but not the unsafe steps; and the imagination carries the scene layout and the lead's gap but answers the ego's action with the sign of the logs, a rule that has to be injected. What is running and what is open is in \\cref{sec:status}. " + s[j:]
n_edits += 1

# ---- status subsection
rep("\\section{Conclusion}", rd('frag_status.tex') + "\\section{Conclusion}")

# ---- conclusion
rep("what remains open is a read of the lead's closing speed and a rollout that carries its motion. A fixed distance rule on the plan removes about 60 percent of the model-induced unsafe steps when the lead comes from labels and part of them when it comes from the model's own vision head (\\tier{T1-kin}). The roadmap",
"what remains open is a read of the lead's closing speed and a rollout that carries its motion. A fixed distance rule on the plan removes about 60 percent of the model-induced unsafe steps when the lead comes from labels and part of them when it comes from the model's own vision head, and a calibrated margin on the gap read brings the vision-fed rule to the label-fed one at a measured price (\\tier{T1-kin}). The world model's imagined gap answers the ego's own action with the sign of the logged data until a kinematic term is injected; that term works at a small weight on two bases and is not yet confirmed on new seeds, and the planner built on it equals extrapolation of the lead at this precision (\\tier{T0}, \\tier{T1-kin}; none of it is driving performance). The roadmap")

open(D + 'v8/tanitad_4b_paper.tex', 'w', encoding='utf8', newline='').write(s)
print('ok edits', n_edits)

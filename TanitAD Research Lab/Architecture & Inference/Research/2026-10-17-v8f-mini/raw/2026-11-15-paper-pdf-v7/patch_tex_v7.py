"""patch_tex_v7.py -- the TanitAD-4B paper, version 7 (2026-11-15): v6's tex plus the register rows H-V8F-PROOF-42 to 54.
No measurement is made here: every number is copied from those rows (extracted to rows.txt from the mirror tip a5013b63).
Every anchor is asserted to match exactly once; the input is pinned by sha256 printed at run time."""
import hashlib

SRC = "F:/TanitAD-scratch/v8proof/paper/v6/tanitad_4b_paper.tex"
DST = "F:/TanitAD-scratch/v8proof/paper/v7/tanitad_4b_paper.tex"
raw = open(SRC, "rb").read()
print("v6 input sha256", hashlib.sha256(raw).hexdigest())
t = raw.decode("utf8")
n_edits = 0


def rep(old, new):
    global t, n_edits
    c = t.count(old)
    assert c == 1, "anchor matched %d times: %r" % (c, old[:80])
    t = t.replace(old, new)
    n_edits += 1


def srep(start, end, new, keep_end=True):
    """replace t[start:end) where both markers occur exactly once (end searched after start)"""
    global t, n_edits
    assert t.count(start) == 1, "start marker count %d: %r" % (t.count(start), start[:60])
    i = t.index(start)
    j = t.index(end, i + len(start))
    t = t[:i] + new + t[j:]
    n_edits += 1


# ------------------------------------------------------------------ abstract
rep(r"""two pre-registered levers read the same at seed 0 (one seed each, so no name). What is still open: the model does not yet imagine the motion of the lead vehicle 2\,s ahead better than a copy. Training on a quarter of the clips shows no measured cost in a preliminary read (\tier{T1-kin}, four of six baseline seeds).""",
    r"""the two pre-registered levers (a rollout loss, and the ego speed as an action channel) read flat at both seeds (\tier{T0}). With six paired seeds the grid state costs nothing in the longitudinal, lateral and tactical families (strategic: paused; 6\,s error $-0.0855$\,m $[-0.1705,+0.0044]$ against the mean state, GRID-NO-COST, \tier{T1-kin}). What is still open is the motion of the lead vehicle 2\,s ahead. A head supervised on the lead during training reads its gap ($R^2$ 0.544 to 0.737) and its presence (AUROC 0.904 to 0.922) at no cost in the rollout, and not its closing speed (every interval of six includes zero); the control k5 of its card fails for all six arms as written, so the names are read under two later notes (\tier{T0}). Distance keeping is measured at \tier{T1-kin}: hierarchy and flat control are equal (DISTANCE-EQUAL), and the rollout adds an unsafe step (gap below 2.0\,m or time gap below 1.0\,s) on 0.1275 of the lead windows in which the driver has none. A fixed distance rule on the plan removes about 60 percent of these in six of six seeds for both arm types when the lead comes from the labels (0.1275 to 0.0489 and 0.1296 to 0.0498, an upper bound: RULE-CUTS-UNSAFE), and about 13 to 32 percent when it comes from the model's own vision head (HEAD-RULE-CUTS-UNSAFE for three head arm types, two seeds each). Training on a quarter of the clips shows no measured cost ($+0.1108$\,m $[-0.1505,+0.3822]$, open) and an eighth does ($+0.4311$\,m $[+0.1627,+0.6845]$, DATA-SATURATED, KNEE-1/8; \tier{T1-kin}, six baseline seeds).""")

# ------------------------------------------------------------------ contributions
rep(r"""no rollout yet carries the lead's motion 2\,s ahead.
  \item \textbf{Results under a decision-grade doctrine}""",
    r"""no rollout yet carries the lead's motion 2\,s ahead.
  \item \textbf{The lead vehicle and distance keeping} (\cref{sec:lead,sec:dist}): the lead's gap and presence are readable from
        the state once a head asks for them during training and its closing speed is not; distance keeping is measured, and a fixed rule
        on the plan removes about 60 percent of the model-induced unsafe steps when the lead comes from labels and part of them when it
        comes from the model's own vision head.
  \item \textbf{Results under a decision-grade doctrine}""")

# ------------------------------------------------------------------ headline table
rep(r"""Ground-truth tactical code: 6\,s ADE from $\approx$2.1 to $\approx$1.0\,m (interface headroom) & T0 \\""",
    r"""Ground-truth tactical code: 6\,s ADE from $\approx$2.1 to $\approx$1.0\,m (interface headroom) & T0 \\
Distance rule on the plan, lead from labels (privileged): unsafe rate 0.1275 to 0.0489 (hierarchy), 0.1296 to 0.0498 (grid state), six of six seeds; from the model's own vision head about 13 to 32 percent & T1-kin \\
Grid state, six paired seeds: 6\,s ADE $-0.0855$\,m $[-0.1705,+0.0044]$ (GRID-NO-COST); lead supervision: gap $R^2$ 0.544 to 0.737, closing speed flat (SUP-NO-COST) & T1-kin, T0 \\
Eighth of the clips against the full set: $+0.4311$\,m $[+0.1627,+0.6845]$ (DATA-SATURATED, KNEE-1/8); quarter $+0.1108$\,m $[-0.1505,+0.3822]$, open & T1-kin \\""")

# ------------------------------------------------------------------ premises section
rep(r"""Experiments that are still training are named as such, and their reads are not in this version.""",
    r"""Reads that do not yet exist are named as such, and their numbers are not in this version.""")
rep("""headway and the stop-goal
decision are not in this read, and the planner is deterministic""",
    """the stop-goal
decision is not in this read (headway is read in \\cref{sec:dist}), and the planner is deterministic""")

# ------------------------------------------------------------------ cell predictor: family read now final
rep("""The family read (\\cref{tab:fam6}) is preliminary \\textsc{no-cost}: it
needs all six baseline seeds and four are read.""",
    r"""The family read with all six baseline seeds names NO-COST for the cell predictor against the grid state and against the baseline (joint mode error rate against the baseline $+0.0663$ $[+0.0346,+0.1007]$, inside the spread).""")

# ------------------------------------------------------------------ levers + grid six seeds + new subsections
LEVERS = r"""\paragraph{The two levers at both seeds, and the arms that exist.} Two levers were pre-registered before any of their numbers (cards ADD18 and ADD19). MOTION trains the predictor on its own 2.0\,s rollouts (arm PR on the cell predictor), has it output the state plus a change (RES), or both (RR). TARGET gives the predictor the measured ego speed as a third action channel (arm PE; by a note frozen before the gate its inverse-dynamics head is also trained to read the speed) or trains a linear head per cell on the frozen token grid of the future frame (arm PF). PR and PE now have both seeds, read against the cell predictor P (\cref{tab:levers}; step, lead, change and position reads are \tier{T0} diagnostics on logged frames, 441 windows with a lead at both times; the family read is \tier{T1-kin}; controls first). The names at both seeds: PR reads STEP-BELOW-COPY, MOTION-FLAT, MOTION-STILL, KEEPS-POSITION and NO-COST; PE reads STEP-SAME, TARGET-FLAT, TARGET-STILL, KEEPS-POSITION and NO-COST. The one-step error ratio over the copy is 0.844 $[0.831,0.857]$ and 0.873 $[0.864,0.883]$ for PR, above that of P by $+0.056$ and $+0.077$; the change skill of PR is against a ceiling of $+0.159$ and $+0.190$. Stated plainly beside NO-COST: PR has higher lateral errors than HIER at both seeds (heading $+1.273$\,deg, curvature $+0.0473$\,1/m, yaw rate $+3.68$\,deg/s), each inside the six-seed spread. RES, RR and PF have one seed and name nothing; RR at seed 0 alone reads MOTION-IMAGINES (rollout minus the ego floor $+0.306$ $[+0.087,+0.619]$), and it cannot be attributed to RES or to the rollout loss, since RES at seed 0 is flat and the ROLL arms are not trained. Control x1 is not evaluated for the new arms (the pixel job was not run), and the per-arm gate tokens were not re-checked in the collation. As the cards commit, PR leads to an explicit target of object motion (the PF lever, one seed so far) and PE to the object-level slot, since the hypothesis that the speed channel helps is not supported.

\begin{table*}[!tp]
\centering
\caption{\textbf{The two levers against the cell predictor P} (\tier{T0}; seeds 0 and 1 where both exist; PE position is without interval in the source). Rollout minus the copy, rollout minus the rollout of P, change skill of the rollout over persistence, and the position read as the arm minus G (PR) or as banked (others). A dash is a cell the source does not give. RES, RR and PF have one seed and name nothing.}
\label{tab:levers}
\scriptsize
\begin{tabularx}{\textwidth}{@{}l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}}
\toprule
Arm, seed & Rollout minus copy & Rollout minus rollout of P & Change skill & Position \\
\midrule
PR, 0 & $+0.002$ $[-0.062,+0.065]$ & $-0.059$ $[-0.134,+0.013]$ & $+0.014$ $[-0.092,+0.130]$ & $-0.0005$ $[-0.019,+0.018]$ \\
PR, 1 & $+0.025$ $[-0.038,+0.084]$ & $+0.032$ $[-0.043,+0.105]$ & $+0.091$ $[-0.015,+0.212]$ & $-0.024$ $[-0.043,-0.005]$, below the spread 0.0450 \\
PE, 0 & $+0.021$ $[-0.028,+0.073]$ & $+0.021$ $[-0.041,+0.080]$ & $-0.064$ $[-0.179,+0.039]$ & $+0.007$ \\
PE, 1 & $+0.017$ $[-0.044,+0.079]$ & $-0.011$ $[-0.100,+0.065]$ & $+0.042$ $[-0.059,+0.142]$ & $+0.009$ \\
RES, 0 & $+0.017$ $[-0.021,+0.052]$ & -- & $+0.029$ & $-0.021$ $[-0.038,-0.004]$ \\
RR, 0 & $+0.050$ $[+0.003,+0.098]$ & -- & $+0.091$ $[-0.008,+0.212]$ & $+0.029$ $[+0.012,+0.047]$ \\
PF, 0 & $-0.082$ $[-0.206,+0.018]$ & -- & $-0.009$ & -- \\
\bottomrule
\end{tabularx}
\end{table*}

\paragraph{The grid state costs nothing on six paired seeds (SPATIALREP).} Four more seeds of the grid arm (G2 to G5) were trained on the pod under a gate PASS, 19,164 steps each, so that the grid state is read against the baseline with six paired seeds (\tier{T1-kin}, 141 episodes, 2,399 windows, two-level bootstrap over seeds and then episodes, $B=2000$; the reader controls all pass). G minus HIER on the 6\,s ADE of all windows (the primary cell) reads mean $-0.0855$\,m $[-0.1705,+0.0044]$ against a six-seed spread of the baseline of 0.2845\,m (\cref{tab:perseed}). Four of six differences are negative and the interval includes 0: GRID-NO-COST. No cell of the four families is worse (STRATEGIC: PAUSED, PI item 25). The first two seeds alone read $-0.1198$\,m $[-0.1982,-0.0349]$; four more seeds brought two positive differences, so that read was inside the run-to-run variation of training. Common-case windows $-0.1522$\,m $[-0.2623,-0.0312]$, speed error and cross-track error (\cref{tab:fam6}) are five of six negative and inside the spread, and name nothing. The final read of the state arms at six baseline seeds and two seeds per arm (card ADD12, \tier{T1-kin}) agrees: grid, wide mean state and cell predictor against the baseline, grid against the wide mean state and cell predictor against the grid all name NO-COST (cross-track of the grid $-0.1071$\,m, inside the spread). The committed step: the grid state stays the state of the world-model path (it holds the lead position, at one-frame probe $R^2$ 0.552/0.541) and is the base of the lead-supervision arms of \cref{sec:lead}.

\begin{table*}[!tp]
\centering
\caption{\textbf{Per-seed differences of the six-seed reads} (\tier{T1-kin}; seeds 0 to 5; the mean carries the two-level interval; spread = largest difference between baseline seeds). The first row is in metres of 6\,s ADE on all windows; the others are the unsafe rate U (share of lead windows with an unsafe step of the model among the windows where the driver has none).}
\label{tab:perseed}
\scriptsize
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{36mm}cccccc>{\raggedright\arraybackslash}X@{}}
\toprule
Difference & s0 & s1 & s2 & s3 & s4 & s5 & Mean $[$interval$]$, name \\
\midrule
G minus HIER, ADE 6\,s (spread 0.2845) & $-0.1011$ & $-0.1384$ & $+0.0421$ & $-0.1174$ & $+0.0298$ & $-0.2281$ & $-0.0855$ $[-0.1705,+0.0044]$, GRID-NO-COST \\
HIER minus FLAT, U (spread 0.0463) & $-0.0267$ & $+0.0160$ & $+0.0214$ & $-0.0214$ & $-0.0320$ & $+0.0089$ & $-0.0056$ $[-0.0269,+0.0140]$, DISTANCE-EQUAL \\
Rule minus no rule, HIER, label-fed, U & $-0.0623$ & $-0.0890$ & $-0.1050$ & $-0.0747$ & $-0.0641$ & $-0.0765$ & $-0.0786$ $[-0.1015,-0.0557]$, RULE-CUTS-UNSAFE \\
Rule minus no rule, G, label-fed, U & $-0.0765$ & $-0.0730$ & $-0.0996$ & $-0.0658$ & $-0.0747$ & $-0.0890$ & $-0.0798$ $[-0.1024,-0.0596]$, RULE-CUTS-UNSAFE \\
\bottomrule
\end{tabularx}
\end{table*}

\subsection{What the state holds of the lead's motion}\label{sec:lead}
The open problem of \cref{sec:state} is the lead gap 2.0\,s ahead. Every read below is a \tier{T0} diagnostic on logged frames, scored on 429 windows of 48 evaluation episodes (413 with a closing-speed label), episode-cluster bootstrap ($B=2000$), every control at its known value first. The three families other than STRATEGIC (PAUSED, PI item 25) are not measured by these reads.

\paragraph{A kinematic slot reads the gap when its inputs are right (OBJSLOT).} Two numbers per lead vehicle (gap, closing speed) and one kinematic identity, with zero training, on 1,948 windows of 139 episodes (lead at the start and at the target in 471, 42 dropped by the continuity rule). With the logged gap as input (a privileged ceiling), constant lead speed with the known ego motion (rule R2) reads skill $+0.750$ $[+0.664,+0.811]$ over the copy and constant closing speed (R1) $+0.692$ $[+0.588,+0.766]$; the level $R^2$ is 0.952 against 0.810 for the copy and the RMSE 2.18 against 4.36\,m: KIN-CEILING-HOLDS (the lead recomputed the label numbers with independent rule code: 0.7499, 0.6916, 0.9524, 0.8097). With the gap read from the grid state (vision only), R1 and R2 are below the copy in every arm (R2 skill $-0.236$, $-0.329$, $-0.407$, $-0.304$ for P and G at seeds 0 and 1) and the rule with a cross-fitted residual (R3, primary) reads $+0.011$, $+0.003$, $-0.007$ and $+0.027$, every interval including 0: SLOT-FLAT for the cell-predictor and the grid-state arms. The transition is therefore not the missing part, since the fixed rule halves the error once its two inputs are right; the missing part is a read of the gap and of the closing speed at the needed precision (the level $R^2$ of the vision copy is 0.54 to 0.56 against 0.81 for the label copy).

\paragraph{A head on five frames of the state, and the same head on 17 times more rows (LEADHEAD).} Stage 0 fits a head supervised on gap and closing speed over the five-frame history of the grid state, cross-fitted by episode on the evaluation bank: 6,997 to 7,064 fit rows per fold, of which 1,689 to 1,926 carry a closing-speed label, for 3,200 inputs (26 of 26 controls pass). The held-out closing-speed $R^2$ of the temporal ridge is negative in every arm (\cref{tab:leadA}), RMSE about 2.4\,m/s, and the small network is lower ($-0.275$ to $-0.288$): HEAD-CLOSING-FLAT; the slot read (R3 over the head copy $-0.045$, $-0.018$, $-0.002$, $+0.083$, every interval including 0) is HEAD-SLOT-FLAT; the direct supervised read of the change over 2.0\,s is below its copy in every arm: STATE-LACKS-FUTURE. The card names the limit beside them: too few rows for the input width. Stage 0b fits the same heads on 1,200 training clips disjoint from the evaluation episodes (32,025 gap, 30,865 closing-speed and 24,533 direct rows; controls j1 to j8 pass). One of four closing-speed intervals excludes 0, so by the rule of the card the names are HEAD-CLOSING-FLAT-B and HEAD-SLOT-FLAT-B (slot $+0.095$, $+0.099$, $+0.100$, $+0.104$, every interval including 0) for both arm types; the direct read of the change over 2.0\,s is above its head copy in all four arms with lower ends above 0 (+0.023, +0.043, +0.029, +0.018): STATE-HOLDS-FUTURE-B. The learning curve rises at every step and has not flattened (no exponent is fitted or quoted), so, as the card commits for a rising curve, these names are not read as a property of the state until the fit set is extended; the extension is the LEADSUP arms below.

\begin{table*}[!tp]
\centering
\caption{\textbf{Lead heads on the state of P and G, two reads} (\tier{T0}; P and G at seeds 0 and 1; $R^2$ unless marked). Stage 0 has 1,689 to 1,926 closing-speed rows, stage 0b all rows of the train probe set; intervals are given where the source gives them.}
\label{tab:leadA}
\scriptsize
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{50mm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}}
\toprule
Read & P, 0 & P, 1 & G, 0 & G, 1 \\
\midrule
Stage 0, closing speed, ridge & $-0.094$ $[-0.238,+0.012]$ & $-0.108$ & $-0.098$ & $-0.060$ \\
Stage 0b, closing speed, ridge, all rows & $+0.126$ $[-0.048,+0.227]$ & $+0.145$ $[-0.020,+0.245]$ & $+0.101$ $[-0.083,+0.206]$ & $+0.176$ $[+0.012,+0.278]$ \\
Stage 0b, direct change skill over the head copy & $+0.100$ $[+0.023,+0.170]$ & $+0.119$ & $+0.115$ & $+0.090$ \\
Learning curve, closing speed, 2,000 rows & $-0.155$ & $-0.046$ & $-0.132$ & $-0.155$ \\
\hspace{3mm}8,000 rows & $+0.011$ & $+0.031$ & $+0.023$ & $+0.041$ \\
\hspace{3mm}16,000 rows & $+0.062$ & $+0.060$ & $+0.048$ & $+0.101$ \\
\hspace{3mm}all rows & $+0.126$ & $+0.145$ & $+0.101$ & $+0.176$ \\
\bottomrule
\end{tabularx}
\end{table*}

\paragraph{Asking during training (LEADSUP, card ADD24).} The term is a supervised lead head on the level-1 grid state with a label rule on gap, closing speed and presence. The build is checked and gated before any arm trained: with the flag off the patched trainer equals the base bit for bit on three argument sets; the new term alone reaches the head (gradient 64.1) and the adapter (30.1) and reads exactly 0.0 on the level-1 predictor and the driving heads, while the total reaches those (control); three mutations go red; the arm adds 82,499 parameters. Three arm types, two seeds each, 19,164 steps under a gate PASS: LS (weight 1), LSH (weight 10) and LSD (weight 1, head on the detached state). \emph{By the card as written no outcome is named:} control k5 (the mean of the lead term over the last 1,000 steps below its mean over the first 1,000) FAILS for all six arms (LS 0.3027 to 0.3741 and 0.2867 to 0.3540; LSD 0.3798 to 0.4337 and 0.3710 to 0.4066; LSH 0.2119 to 0.3190 and 0.2088 to 0.2897), and the card says an arm that fails k5 is not read. The slot part of control k3 in the card form passes for LS ($+0.019$, $+0.016$) and fails for LSH ($+0.095$, $+0.028$) and LSD ($+0.055$, $+0.062$) against the bar $+0.02$. Measured on the logs: the lead term falls within the first 300 steps, its presence part then rises and stays higher, the gap part falls over the run in every arm, and the closing-speed part in none. \emph{Read under notes N1 and N2} (N1 defines an attributable slot increment, frozen before seed 1 existed; N2 replaces k5 by k5-N2, frozen after the logs and the seed-0 reads were seen and before any lead-read number of a seed-1 arm; its log parts were written after the logs and add little, the held-out part carries it; k5-N2 passes for all six arms): the closing-speed $R^2$ has every interval including 0, SUP-CLOSING-FLAT in all three arm types (\cref{tab:leadB}); the gap $R^2$ of the head is 0.544 to 0.737 (RMSE 5.3 to 6.9\,m) and the presence AUROC 0.904 to 0.922; the slot increment DELTA reads SUP-SLOT-FLAT-N1 in all three; the gradient question, paired, reads SUP-GRADIENT-EQUAL twice (LS minus LSD $-0.024$ $[-0.164,+0.132]$ and $-0.075$ $[-0.328,+0.156]$; floor 0.089). The cost read (\tier{T1-kin}, 2,399 windows of 141 episodes, spread of G on six seeds 0.4002 on the primary cell) names SUP-NO-COST for all three: no cell worse under the rule in the longitudinal, lateral or tactical family; the primary cell is $-0.1150$\,m $[-0.1995,-0.0274]$ for LS minus G (inside the spread, not an effect), $-0.0179$ $[-0.1239,+0.0909]$ for LSH and $+0.0746$ $[-0.0707,+0.2154]$ for LSD; LS minus LSD has one cell better (joint class error $-0.0604$ $[-0.0953,-0.0248]$, spread 0.0492). Asking for the lead during training therefore gives a gap read and a presence read at no cost in the rollout, and no closing speed. A limit of that reading was measured after the card was frozen (control p2b of card ADD27, synthetic): a head of this form trained by gradient steps does not take a rate from five frames of a planted gap, so the head is a second candidate for the limit beside the cue.

\begin{table*}[!tp]
\centering
\caption{\textbf{LEADSUP and GAPFILTER, six arms} (\tier{T0}; 413 scored windows; read under notes N1 and N2 for LEADSUP, card form of GAPFILTER; seed 0 / seed 1 inside each cell). The filter rows are the numbers of the GAPFILTER card run, whose control f2 fails as written (no outcome named).}
\label{tab:leadB}
\scriptsize
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{34mm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}}
\toprule
Quantity & LS & LSH & LSD \\
\midrule
Closing speed $R^2$ of the head & $+0.155$ $[-0.043,+0.283]$ / $+0.014$ $[-0.278,+0.192]$ & $+0.092$ $[-0.138,+0.231]$ / $+0.026$ $[-0.192,+0.163]$ & $+0.178$ $[-0.059,+0.335]$ / $+0.089$ $[-0.127,+0.234]$ \\
Gap $R^2$ of the head & 0.724 / 0.580 & 0.595 / 0.737 & 0.626 / 0.544 \\
Slot increment DELTA (note N1) & $-0.004$ $[-0.026,+0.016]$ / $-0.004$ $[-0.059,+0.051]$ & $+0.019$ $[-0.027,+0.067]$ / $+0.001$ $[-0.085,+0.060]$ & $+0.001$ $[-0.044,+0.045]$ / $-0.032$ $[-0.066,+0.004]$ \\
Closing speed $R^2$ of the alpha-beta filter & $+0.130$ $[-0.058,+0.256]$ / $+0.177$ $[-0.019,+0.290]$ & $+0.201$ $[-0.005,+0.322]$ / $+0.158$ $[-0.064,+0.282]$ & $+0.149$ $[+0.008,+0.229]$ / $+0.102$ $[-0.062,+0.206]$ \\
Filter minus the head's own output & $-0.025$ $[-0.163,+0.113]$ / $+0.164$ $[-0.035,+0.419]$ & $+0.110$ $[-0.090,+0.312]$ / $+0.132$ $[-0.079,+0.317]$ & $-0.029$ $[-0.178,+0.145]$ / $+0.013$ $[-0.141,+0.185]$ \\
\bottomrule
\end{tabularx}
\end{table*}

\paragraph{Taking the rate outside the network, and a probe of the tokens (GAPFILTER, TOKPROBE).} Card ADD28 asks whether a causal filter over the head's own per-frame gap reads the closing speed on a longer baseline, with zero training. \emph{By the card as written control f2 fails and the card names no outcome:} the slope estimator on the label gap must read the label closing speed at $R^2$ of at least 0.8 and reads $+0.367$; 410 of 413 windows reproduce the label exactly, and the 3 windows with a change of lead inside the last five frames are the ones where the estimator as the card words it fits across the step. Controls f1, f3, f4, f5 and f6 pass for all six arms. The numbers of the card run are in \cref{tab:leadB}: every arm selects the alpha-beta filter (alpha 0.7; beta 0.1, for LSD at seed 1 0.05), every interval of the filter's gain over the head includes 0, and so does every slot increment with the filter rate ($-0.003$ to $+0.029$); no arm type would have met the rule of FILTER-READS-CLOSING had f2 passed. A variant with a same-lead rule (a step above 3.0\,m restarts the run), written after f2 was seen and naming nothing, passes all its controls (f2 $R^2$ $+1.000$) and is flat too (filter $R^2$ $+0.09$ to $+0.24$). Why it is flat, measured (report f7): the head's gap error on evaluation rows is 7.0 to 8.8\,m RMSE and its change between consecutive rows (0.2\,s) 2.2 to 2.3\,m RMSE, with lag-1 autocorrelation 0.95 to 0.96, a slow drift a filter cannot remove; a rate needs a read about 0.5\,m coarse from frame to frame (report f8: closing-speed $R^2$ $+0.90$ at 0.5\,m of white noise on the label gap, $+0.43$ at 1\,m, below 0 at 2\,m). The token probe (card ADD27) has its instrument checked first: its gap-only control p2b, in the wording of the card, FAILS on synthetic data (held-out gap $R^2$ 0.90 to 0.95, closing-speed $R^2$ at most $+0.091$ where a slope on the known directions reads $+0.936$). Note ADD27\_N1, frozen before this result, fixes the consequence: the card can name TOKENS-HOLD-CLOSING and cannot name TOKENS-FLAT. No probe arm has run on real tokens; the job is queued on the pod. If the pooled tokens are the limit, a finer grid or a crop of the lead region needs the camera frames re-encoded, a data and compute step the PI decides (named as a blocker with its cost, not run). Limits: two seeds per arm type, one corpus, and p2b is a small synthetic world.

\subsection{Distance keeping, and a rule on the plan}\label{sec:dist}
\paragraph{Distance keeping is now measured (HEADWAY).} Every \tier{T1-kin} read before this one reported the longitudinal family without headway and time gap. Read from the banked dumps and the lead labels of the evaluation episodes (no new rollout; 642 of the 2,399 windows have a lead, in 56 of 141 episodes): the lead keeps its logged track, the model gap is the logged gap plus the logged progress minus the rolled progress, and an unsafe step is a gap below 2.0\,m or a time gap below 1.0\,s. U is the share of lead windows with an unsafe step of the model among the windows where the driver has none. Controls h1 to h6 pass, each with its regression form failing as required (a planted advance of 0.5\,m/s read back as $-1.0$\,m at 2.0\,s and $-3.0$\,m at 6.0\,s and named worse; a 5-frame shift of the label mapping fails on 470 windows); an independent recomputation of U for HIER at seed 0 agrees (0.106761565836, difference 0.0). Over six seeds (two-level bootstrap, rule of card ADD11): HIER minus FLAT is $-0.0056$ $[-0.0269,+0.0140]$ against a spread of 0.0463 (\cref{tab:perseed}), DISTANCE-EQUAL, and G minus HIER is $+0.0021$ $[-0.0136,+0.0175]$, GRID-DISTANCE-EQUAL. Levels: U is 0.1275 $[0.0970,0.1581]$ for HIER, 0.1332 for FLAT and 0.1296 for G; the contact rate (the rolled ego reaches the logged position of the lead) is 0.0460 $[0.0304,0.0618]$, 0.0439 and 0.0452; the gap bias at 6.0\,s is $+6.15$\,m $[4.89,7.41]$, $+3.91$ and $+5.63$\,m (the rollout is slower than the driver on average). The logged driver by the same rule has an unsafe-step share of 0.1246 over the 642 lead windows and no contact. Beside the names: the gap error at 2.0\,s of HIER minus FLAT is $+0.039$\,m $[+0.017,+0.061]$ in six of six seeds, outside the spread (0.032), worse by 4\,cm; every other cell is open, inside the spread or of mixed sign. The gap comes from the label join of the 141 evaluation episodes (a deviation recorded before any arm comparison: the bank of card ADD20 holds leads within 40\,m only). Limits: the lead does not react, one evaluation set, a deterministic planner.

\paragraph{A fixed rule on the plan (DISTRULE, card ADD26).} Frozen before any number: keep 3.0\,m plus 1.2\,s behind an imagined constant-speed lead, brake at most 6.0\,m/s$^2$, never move a plan ahead; the constants are fixed and not tuned (the metric thresholds are 2.0\,m and 1.0\,s). Arms A use the label source on HIER and G, six seeds each (privileged, an upper bound, never a deployable result); arms B use the head source and the label source on the six arms of the lead-supervision card. Reads use the distance reader above and the four-family reader, with controls r1 to r7, all PASS: rule off through the wrapper equals the banked dump (141 files, 0 differing); the whole wrapper path with no cap equals rule off (in the form of note ADD26\_N1, frozen before any number of the card, because the projection as written advanced plans and the rule therefore runs as cap-only); no plan is moved ahead in any of 25 rule logs (r2b); the deliberate push is named worse on U ($+0.0569$ $[+0.0222,+0.0927]$, r3); and on the real checkpoints the head source of the rollouts equals the banked head read within 0.007\,m, 0.001\,m/s and 0.0002 in presence on all 7,056 ticks of three arms. Decision M3-BC, recorded before any arm number: a random head with an offset of $-22.8$\,m read slot skill 0.82 over the raw copy from the intercept of rule R3 alone, so the gated slot skill is taken over the bias-corrected copy ($-0.009$ for the same head).

\paragraph{Fed by labels: about 60 percent, six of six seeds.} The rule against no rule of the same checkpoint, paired on windows (642 lead windows of 56 episodes for the distance cells, 2,399 windows for the cost cells; inference is deterministic, so a difference is the rule alone). U falls from 0.1275 to 0.0489 for HIER ($-0.0786$ $[-0.1015,-0.0557]$) and from 0.1296 to 0.0498 for G ($-0.0798$ $[-0.1024,-0.0596]$), negative in every seed (\cref{tab:perseed}): RULE-CUTS-UNSAFE for both arm types. The contact rate falls from 0.0460 to 0.0013 ($-0.0447$ $[-0.0601,-0.0293]$) and from 0.0452 to 0.0008 ($-0.0444$ $[-0.0612,-0.0285]$), six of six seeds each: RULE-CUTS-CONTACT. In the four families no cell is worse under the rule of card ADD11 (STRATEGIC: PAUSED, PI item 25): RULE-NO-COST. What the rule does cost, stated beside the name: the 6\,s position error on all windows rises by $+0.179$\,m $[+0.095,+0.280]$ (HIER) and $+0.180$\,m $[+0.098,+0.282]$ (G), the same sign in all six seeds and inside the six-seed training spread of 0.2845\,m, which is why the card's rule does not name it; the model stays further back than the driver (gap bias at 2\,s $+0.44$\,m, named worse by the distance reader).

\paragraph{Fed by the model's own vision head: part of it.} The same checkpoints with a lead head (arms B, two seeds each; \cref{tab:headrule}). The head-fed rule lowers U against no rule in all three head arm types by the rule as written, HEAD-RULE-CUTS-UNSAFE, the first distance-keeping result of the programme with a vision-only lead state; for LSH the interval ends just below 0 and seed 0 is small. The label-fed rule on the same checkpoints lowers U by $-0.0792$, $-0.0863$ and $-0.0872$ and brings the contact rate to 0.0000. The share of the label-fed fall that the head-fed rule reaches is below the bar of one half in at least one seed for every arm type (\cref{tab:headrule}), so HEAD-RULE-REACHES-ORACLE is not named for any. Cost of the head-fed rule: on LS one cell is worse (longitudinal, along-track error $+0.417$\,m $[+0.193,+0.676]$, spread 0.353); LSH and LSD have no cell worse in the name, and the position error on bulk windows is $+0.43$ to $+0.55$\,m in all three, reported beside. Lead supervision alone leaves distance keeping equal: U of LS, LSH and LSD against G with no rule is $-0.0027$ $[-0.0193,+0.0135]$, $+0.0062$ $[-0.0103,+0.0234]$ and $+0.0071$ $[-0.0099,+0.0266]$, SUP-DISTANCE-EQUAL three times; reading the lead does not by itself change the plan. The next step is committed by the cards: the label-fed rule stands as the oracle form of the safety check, and the distance from head-fed to label-fed is the measure of the lead state; card ADD29 (RULEATTR, frozen before its numbers, rollouts running) swaps presence, gap and closing speed one at a time from label to head to find which part holds the vision-fed rule back, and then comes the training card with the rule in the loop. Limits: a lead that does not react; the logged driver has an unsafe-step share of 0.1246 by the same rule and U counts only windows where the driver has none; two seeds for the head arms; the constants were set before the data and not tuned; one corpus; a deterministic planner.

\begin{table*}[!tp]
\centering
\caption{\textbf{Head-fed distance rule, arms B} (\tier{T1-kin}; 642 lead windows; U against no rule, seed 0 / seed 1, two-level interval; the share is the part of the label-fed fall reached, bar one half in both seeds; all three arm types name HEAD-RULE-CUTS-UNSAFE).}
\label{tab:headrule}
\scriptsize
\begin{tabularx}{\textwidth}{@{}lcX@{\hspace{4mm}}cc@{\hspace{4mm}}c@{}}
\toprule
Arm & U, no rule to head-fed & Difference, seed 0 / seed 1; mean $[$interval$]$ & Contact, no rule to head-fed & Share reached & Head-fed minus label-fed, U \\
\midrule
LS & 0.1246 to 0.0845 & $-0.0409$ / $-0.0391$; $-0.0400$ $[-0.0686,-0.0127]$ & 0.0444 to 0.0234 & 0.55 / 0.47 & $+0.0391$ $[+0.0197,+0.0623]$ \\
LSH & 0.1335 to 0.0979 & $-0.0071$ / $-0.0641$; $-0.0356$ $[-0.0858,-0.0006]$ & 0.0452 to 0.0249 & 0.10 / 0.65 & $+0.0507$ $[+0.0227,+0.0848]$ \\
LSD & 0.1343 to 0.1174 & $-0.0196$ / $-0.0142$; $-0.0169$ $[-0.0324,-0.0035]$ & 0.0452 to 0.0335 & 0.22 / 0.17 & $+0.0703$ $[+0.0426,+0.0996]$ \\
\bottomrule
\end{tabularx}
\end{table*}

\subsection{Data efficiency and the four families}\label{sec:dataeff}"""
srep(r"\paragraph{Two levers, seed 0 only.}", r"\subsection{Data efficiency and the four families}\label{sec:dataeff}", LEVERS)
# the srep above kept the old header line after the slice; remove the duplicate header
rep("""\\subsection{Data efficiency and the four families}\\label{sec:dataeff}\\subsection{Data efficiency and the four families}\\label{sec:dataeff}""",
    """\\subsection{Data efficiency and the four families}\\label{sec:dataeff}""")

# ------------------------------------------------------------------ data efficiency paragraph (final read)
DATA = r"""\paragraph{Data efficiency (final read, six baseline seeds).} The same model was trained at a fixed 19,164 steps on a random half, quarter and eighth of the 4,512 clips, two seeds per reduced arm (\tier{T1-kin}, 2,399 windows of 141 episodes, 1,305 selection-critical and 1,094 common-case; the five reader controls pass in every read, a planted regression is read back as 0.899\,m/s). The rule of the card: both seeds the same sign, a two-level interval that excludes 0, and outside the six-seed spread of the baseline (0.4093\,m on the primary cell, the 6\,s ADE on selection-critical windows). Arm minus full on the primary cell: half $-0.1082$\,m $[-0.2279,+0.0159]$, open; quarter $+0.1108$\,m $[-0.1505,+0.3822]$, open (seeds $-0.042$ and $+0.264$); eighth $+0.4311$\,m $[+0.1627,+0.6845]$ with both seeds above 0, worse: DATA-SATURATED with KNEE-1/8. The margin of the eighth over the spread is small (0.431 against 0.409); on all windows the eighth reads $+0.2800$\,m $[+0.1036,+0.4477]$, inside the spread, and its tactical-band ADE is worse ($+0.4204$\,m $[+0.1617,+0.6664]$). Hierarchy minus flat on the primary cell is $-0.0389$\,m at the full set and $-0.1590$\,m $[-0.3452,+0.0551]$ at a quarter; the difference of the two differs in sign between the seeds ($-0.354$ and $+0.114$): HIER-GAP-EQUAL, and the dominance claim rests on the six-seed replicate alone, as the card commits. At the eighth, with both seeds, the gap is $-0.1717$\,m $[-0.3466,+0.0027]$ and the difference from the full-set gap is $-0.3169$ and $+0.0514$ per seed, signs differ: HIER-GAP-EQUAL at one eighth as at one quarter. Clips chosen by their share of selection-critical windows against random clips, at a quarter: the primary cell is $+0.0347$\,m $[-0.1750,+0.2608]$, open; common-case windows $+0.7552$\,m $[+0.4619,+1.0573]$, worse; speed error $+0.3332$\,m/s $[+0.1999,+0.4691]$, worse; tactical-band ADE $+0.5431$\,m $[+0.2561,+0.8412]$, worse: SELECT-EQUAL, and the choice by criticality costs on common-case driving. At the eighth, both seeds: selection-critical windows $-0.1722$\,m $[-0.3719,+0.0245]$, open, common-case windows $+0.6681$\,m $[+0.4101,+0.9193]$, worse: SELECT-EQUAL at one eighth as at one quarter. What follows, as the card commits: more clips of this kind are not the lever at this model size; the scaling step is the state, the cue and the model width, and the next choice rule for clips is model-based. Limits: two runs per reduced arm, so the intervals are wide by construction; fixed steps mean more passes over fewer clips; one corpus and one evaluation set; a deterministic planner; the stop-goal decision is not measured.

"""
srep(r"\paragraph{Data efficiency (preliminary).}", r"\begin{table*}[t]" + "\n" + r"\centering" + "\n" + r"\caption{\textbf{Four metric families for the new", DATA)

# ------------------------------------------------------------------ Table fam6: caption, plus the rows of the six-seed and distance reads
rep("""Headway and stop-goal cells are not in
these reads. The first five rows are preliminary (spread from three or four of six baseline seeds); the last row is an emulation on banked rollouts, not a rollout.""",
    r"""Headway and stop-goal cells are not in the first four rows (headway is read in \cref{sec:dist}); the fourth row is an emulation on banked rollouts, not a rollout; the last six rows are the six-seed and distance reads (seeds in brackets).""")

srep("G minus baseline (0, 1) & speed", "Hold rule minus arm at rest", r"""State arms (grid, wide mean state, cell predictor), six baseline seeds (0, 1) & NO-COST & NO-COST; grid cross-track $-0.1071$\,m, inside the spread & NO-COST; cell predictor minus baseline joint mode error rate $+0.0663$ $[+0.0346,+0.1007]$, inside the spread & PAUSED \\
Half, quarter, eighth of clips minus full (0, 1) & open for all three & open or inside the spread & eighth worse on tactical-band ADE, $+0.4204$\,m $[+0.1617,+0.6664]$ & PAUSED \\
Chosen quarter minus random quarter (0, 1) & speed error $+0.3332$\,m/s $[+0.1999,+0.4691]$, worse & open or inside the spread & tactical-band ADE $+0.5431$\,m $[+0.2561,+0.8412]$, worse & PAUSED \\
""")

FAM7ROWS = r"""\midrule
G minus HIER (0 to 5) & speed error $-0.0643$\,m/s $[-0.1295,-0.0012]$, five of six, inside the spread & cross-track $-0.0757$\,m $[-0.1555,-0.0031]$, five of six, inside the spread & no cell worse & PAUSED \\
HIER minus FLAT, distance (0 to 5) & U $-0.0056$ $[-0.0269,+0.0140]$; gap error at 2.0\,s $+0.039$\,m $[+0.017,+0.061]$, six of six, outside the spread 0.032 & not measured by this read & not measured by this read & PAUSED \\
LS, LSH, LSD minus G (0, 1) & no cell worse under the rule & no cell worse under the rule & no cell worse under the rule; LS minus LSD joint class error $-0.0604$ $[-0.0953,-0.0248]$, spread 0.0492 & PAUSED \\
PR, PE minus P, G (PR) and HIER (0, 1) & NO-COST & NO-COST; PR above HIER in heading $+1.273$\,deg, curvature $+0.0473$\,1/m, yaw rate $+3.68$\,deg/s, each inside the spread & NO-COST & PAUSED \\
Rule minus no rule, label-fed, HIER and G (0 to 5) & U $-0.0786$ $[-0.1015,-0.0557]$ and $-0.0798$ $[-0.1024,-0.0596]$; 6\,s position error $+0.179$\,m $[+0.095,+0.280]$ and $+0.180$\,m $[+0.098,+0.282]$, inside the spread 0.2845; gap bias at 2\,s $+0.44$\,m, named worse & no cell worse & no cell worse & PAUSED \\
Rule minus no rule, head-fed, LS, LSH, LSD (0, 1) & LS along-track error $+0.417$\,m $[+0.193,+0.676]$, worse (spread 0.353); LSH and LSD no cell worse in the name; bulk-window position error $+0.43$ to $+0.55$\,m in all three, reported beside & not named worse & not named worse & PAUSED \\
"""
ANCH = "$+0.156$ $[+0.056,+0.294]$ & PAUSED " + chr(92)*2 + chr(10) + chr(92) + "bottomrule"
rep(ANCH, "$+0.156$ $[+0.056,+0.294]$ & PAUSED " + chr(92)*2 + chr(10) + FAM7ROWS.lstrip(chr(10)) + chr(92) + "bottomrule")

# ------------------------------------------------------------------ roadmap
rep(r"""World-model state with the cell layout \textit{(trained, two seeds)} & reads where the lead is (the layout, not the width); the cell predictor fixes the one-step defect; no rollout yet carries the lead's motion 2\,s ahead, and the next lever is an object-level slot \\""",
    r"""World-model state with the cell layout \textit{(trained, six paired seeds)} & reads where the lead is (the layout, not the width) and costs nothing on six seeds; the cell predictor fixes the one-step defect; neither a rollout loss nor the ego speed as an action channel makes the rollout carry the lead's motion 2\,s ahead; lead supervision reads gap and presence, not closing speed (token probe queued) \\
Distance rule on the plan \textit{(measured, T1-kin)} & about 60 percent fewer model-induced unsafe steps with labels, about 13 to 32 percent with the model's own head; next, attribution of the shortfall (card ADD29, running) \\""")

# ------------------------------------------------------------------ outlook paragraph
OUT = r"""The theory of \cref{sec:theory} orders the next steps, and \cref{sec:state,sec:lead,sec:dist} say where the work stands. The state that keeps the cell layout holds where the lead vehicle is, as \cref{prop:pool} says it must, costs nothing on six paired seeds, and its cell predictor fixes the step it is trained on. Neither a rollout loss nor the ego speed as an action channel makes the rollout imagine the lead's motion, and a supervised head reads its gap and presence and not its closing speed (its gap read drifts about 2.3\,m from frame to frame where a rate needs about 0.5\,m), so by T7 imagination is not yet admissible for objects. The next reads are committed in the cards: the attribution of the head-fed shortfall of the distance rule to presence, gap or closing speed (card ADD29, rollouts running), the token probe on the pod (it can name TOKENS-HOLD-CLOSING and cannot name TOKENS-FLAT), and then the rule in the training loop; a finer grid or a crop of the lead region would need the camera frames re-encoded, a data and compute step that is the PI's decision. The deciding six-seed replicate of the hierarchy against its flat control is read: the hierarchy dominates under the primary
protocol and the comparison from scratch is not established. Two further reads are committed: the
dominance premise on those six seeds, and a go-or-stay cue at rest from perception. Data efficiency is read at six baseline seeds (\cref{sec:dataeff}): the scaling step is the state, the cue and the model width."""
srep(r"The theory of \cref{sec:theory} orders the next steps", "\n\n% ====", OUT)

# ------------------------------------------------------------------ conclusion
rep(r"""A state that keeps the cell layout now holds the lead; what remains open is a rollout that carries the lead's motion.""",
    r"""A state that keeps the cell layout now holds the lead and costs nothing on six seeds; what remains open is a read of the lead's closing speed and a rollout that carries its motion. A fixed distance rule on the plan removes about 60 percent of the model-induced unsafe steps when the lead comes from labels and part of them when it comes from the model's own vision head (\tier{T1-kin}).""")

t = t.replace(r"\begin{table*}[t]", r"\begin{table*}[!tp]")
rep(r"""\begin{table}[t]
\centering
\caption{\textbf{Headline results}""", r"""\begin{table}[!t]
\centering
\caption{\textbf{Headline results}""")
rep(r"""\begin{table}[t]
\centering
\caption{\textbf{Roadmap}""", r"""\begin{table}[!t]
\centering
\caption{\textbf{Roadmap}""")
# ------------------------------------------------------------------ wording: the phrase about performance is replaced where v6 used it (four sites), and Table 8 column widths
rep(r"""diagnostic on logged frames, never driving performance (seed 0 / seed 1""", r"""diagnostic on logged frames (seed 0 / seed 1""")
rep(r"""A diagnostic, never driving performance.""", r"""A diagnostic on logged frames.""")
rep(r"""world-model diagnostics on logged frames, never driving performance, with every""", r"""world-model diagnostics on logged frames, with every""")
rep(r"""a diagnostic, never driving performance). Differences A minus B""", r"""a diagnostic on logged frames). Differences A minus B""")
rep(r"""\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{31mm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{19mm}@{}}""",
    r"""\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{31mm}>{\raggedright\arraybackslash\hsize=1.2\hsize}X>{\raggedright\arraybackslash\hsize=0.9\hsize}X>{\raggedright\arraybackslash\hsize=0.9\hsize}X>{\raggedright\arraybackslash}p{19mm}@{}}""")
# ------------------------------------------------------------------ layout: bibliography in scriptsize
rep(r"""\begin{thebibliography}{99}\footnotesize""", r"""\begin{thebibliography}{99}\scriptsize""")
# ------------------------------------------------------------------ layout: slightly larger text block, roadmap table in footnotesize
rep("top=20mm,bottom=22mm", "top=18mm,bottom=19mm")
rep(r"""\label{tab:roadmap}
\small""", r"""\label{tab:roadmap}
\footnotesize""")

open(DST, "w", encoding="utf8", newline="\n").write(t)
print("ok edits", n_edits, "v7 sha256", hashlib.sha256(t.encode("utf8")).hexdigest())

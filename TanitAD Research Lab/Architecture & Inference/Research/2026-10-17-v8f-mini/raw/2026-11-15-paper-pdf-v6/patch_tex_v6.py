"""patch_tex_v6.py -- the TanitAD-4B paper, version 6 (2026-11-15): v5's tex plus the results of the proof programme
of 13 to 14 Nov (72 Lean theorems, the spatial state, the cell predictor, two levers at seed 0, data efficiency, DOMPREM, HOLDREAD).
No measurement is made here: every number is copied from the register rows GAC_ROWS_20..32 or the two seed-0 verdicts named in BUILD.md.
Every replacement is asserted to match exactly once; the input is pinned by sha256.
"""
import hashlib, sys
SRC = "F:/TanitAD-scratch/v8proof/paper/v5/tanitad_4b_paper.tex"
DST = "F:/TanitAD-scratch/v8proof/paper/v6/tanitad_4b_paper.tex"
SHA = "e4617dcc004f8e6a3123a85285885d7686b56fcf592017e7e2f5a2c849627dba"

raw = open(SRC, "rb").read()
assert hashlib.sha256(raw).hexdigest() == SHA, "v5 input changed"
t = raw.decode("utf8")
n_edits = 0


def rep(old, new):
    global t, n_edits
    c = t.count(old)
    assert c == 1, "anchor matched %d times: %r" % (c, old[:70])
    t = t.replace(old, new)
    n_edits += 1


# ------------------------------------------------------------------ abstract
rep(r"(56 solver conditions and 54 Lean~4 theorems)",
    r"(56 solver conditions and 72 Lean~4 theorems, the rule, imagination and averaging statements included)")

rep(r"nothing about how its gap changes; the hierarchy is ahead of a flat control on selection-critical windows, and the comparison on all windows is not yet established.",
    r"""nothing about how its gap changes. A state that keeps the cell layout reads the lead gap out of sample at $R^2$ 0.55/0.54 against 0.24/0.21 for the mean state, and a mean state of equal width does not (0.36/0.31): the gain is the layout, not the width (two seeds, \tier{T0}). A predictor that attends across the cells fits the step it is trained on (0.79/0.80 of the error of a copy, against 1.23/1.25 for the flat predictor) and keeps the position, but its 2\,s rollout does not yet read the lead better than a copy of the present state; two pre-registered levers read the same at seed 0 (one seed each, so no name). What is still open: the model does not yet imagine the motion of the lead vehicle 2\,s ahead better than a copy. Training on a quarter of the clips shows no measured cost in a preliminary read (\tier{T1-kin}, four of six baseline seeds). The hierarchy is ahead of a flat control on selection-critical windows and not established on all windows; the six-seed replicate that decides it is trained and its read is pending.""")

# ------------------------------------------------------------------ contributions
rep(r"""        with a premise that is then measured.
""",
    r"""        with a premise that is then measured. The Lean~4 total is 72 theorems.
  \item \textbf{A world-model state that keeps the layout} (\cref{sec:state}): a trained state that keeps the image cells reads
        where a lead vehicle is (the gain is the layout, not the width), a cell-attending predictor fixes the one-step defect of
        the flat one, and the open problem is placed: no rollout yet carries the lead's motion 2\,s ahead.
""")

# ------------------------------------------------------------------ section 6
rep("(54 theorems, no \\texttt{sorry}, standard\naxioms only)",
    "(72 theorems in three files, no \\texttt{sorry}, standard\naxioms only)")

rep("& Z3 &\nwith the logged rest distance", "& Z3, Lean &\nwith the logged rest distance")
rep("& Z3 &\nnot yet admissible for objects: the state decodes to the scene gist, and its learned rollout is no closer to the future than a copy",
    "& Z3, Lean &\nnot yet admissible for objects: the mean state decodes to the scene gist; the grid state holds the lead, but no learned rollout yet reads it better than a copy of the present state at 2\\,s")
rep("& Z3 &\na two-member weight average realises", "& Z3, Lean &\na two-member weight average realises")
rep("(probe level); trained arms are running",
    "(probe level); trained arms: a quarter of the clips shows no measured cost (preliminary)")
rep("change of the gap: nothing beyond ego motion",
    "change of the gap: nothing beyond ego motion; a state that keeps the cell layout reads the lead gap at $R^2$ 0.55/0.54 against 0.24/0.21")

rep(r"""        is linear in its weights.
\end{itemize}
""",
    r"""        is linear in its weights.
\end{itemize}

\paragraph{Checked by the kernel (T3, T7, T9).} The third Lean file (18 theorems, no \texttt{sorry}, 18 of 18 on the standard
axioms) states the three groups in general form; it adds no solver condition. A shield whose braking law obeys the recursion above
keeps the braking distance inside the remaining distance, the speed non-negative and the line unpassed at every step of every
sequence of proposals, whatever network makes them; a safe proposal passes unchanged; and if the shield never passes the
\emph{predicted} line, the true line is passed at most as often as the predicted line lies beyond it, so the residual risk of a
shield is the miss rate of its cue. For imagination, with imagined and true values within $G$, a gain above $2G$ is real, the best
imagined candidate is within $2G$ of every candidate, $2G$ is attained, and a gain above $G$ alone can be a real loss; for a step
cost that is $K$-Lipschitz in the state, $G$ is at most $K$ times the summed composed bound of \cref{prop:jump}, so the admissible
horizon of imagination follows from the measured step error. For averaging, the summed squared error of $K$ members equals their
scatter around the average plus $K$ times the squared error of the average, in any real inner-product space. All three are
conditional: every premise is a quantity to measure, and nothing here says that v8F-mini meets it (the shield lacks a cue,
\cref{sec:premises}; the model gap is too large for imagined gains on objects, \cref{sec:state}). The braking law is a premise,
and its closed form in the code is the solver's condition, not re-proved in Lean. Measured premises of the dominance statement are
in \cref{sec:premises}.
""")

# ------------------------------------------------------------------ results 8.1
rep("Three experiments are still training, and their reads are not in this version.",
    "Experiments that are still training are named as such, and their reads are not in this version.")

rep(r"""\paragraph{A class weight is one number (T4).}""",
    r"""\paragraph{The premise of the dominance theorem (T2), read.} For each context the selection premise asks whether the hierarchy
lands in the logged 12-cell mode more often than the largest logged cell holds (the margin, seed 0/1; \tier{T1-kin}, 2,399
windows, 141 episodes, zero training, controls pass). On all windows the margin is $+0.018$ $[-0.045,+0.079]$ and $-0.000$
$[-0.061,+0.059]$: undecided. At $v_0\in[8,15)$\,m/s (923 windows of 85 episodes) it is $+0.153$ $[+0.084,+0.229]$ and $+0.118$
$[+0.022,+0.215]$, so the premise holds at both seeds, and the flat arm has a margin of the same size there ($+0.118$/$+0.125$). Below
2\,m/s it fails at both seeds ($-0.171$ $[-0.326,-0.066]$ and $-0.242$ $[-0.401,-0.138]$), carried by the lateral mode (accuracy
0.218/0.237 against a largest class near 0.80, in both arms). The geometry premise fails in every context: the within-cell spread of
the logged 6\,s endpoints (13 to 81\,m) exceeds the distance between the two largest cells (separation index 0.149 $[0.047,0.232]$
pooled, at most 0.355 in a context), and among windows in the right mode the 90th percentile of the endpoint error is 8.7 to
15.4\,m against a tolerance of 2.0\,m. The 12 cells are therefore not modes in the sense of \cref{prop:dom}, and the theorem does
not imply dominance for v8F-mini in any context. Card outcomes with on-policy data in both arms: DOM-PREMISE-PARTIAL and
HIT-UNDECIDED; the hit rate at 2.0\,m, hierarchy minus flat, is $+0.032$ $[+0.011,+0.053]$ on selection-critical windows at seed 0
and $+0.012$ $[-0.007,+0.032]$ at seed 1. The committed next step is the same read on the six seeds of the replicate, with seeds as
the unit.

\paragraph{A located defect: at rest (HOLDREAD).} On the 211 windows that start below 2\,m/s (29 episodes) one fixed plan hits
31.3\,\% of the logged endpoints, the hierarchy 7.1/5.2\,\% and the flat arm 1.9/2.8\,\% (hit rate, hierarchy minus the best plan:
$-0.242$ $[-0.384,-0.103]$ and $-0.261$ $[-0.405,-0.132]$). In 53 of these windows the logged ego stays (14 episodes) and in 158 it goes.
Where it stays, the hierarchy travels a median of 8.68\,m $[6.87,10.48]$ and 11.83\,m $[8.53,14.45]$ and the flat arm 12.84/11.33\,m
(outcome CREEP). The plan at the window start separates go from stay at AUROC 0.803 $[0.711,0.894]$ and 0.807 $[0.706,0.907]$,
with lower bounds under the bar of 0.80 (CUE-WEAK). A hold rule keyed on that plan, emulated on banked rollouts and cross-fitted,
changes 6\,s ADE at rest by $-0.239$\,m $[-0.595,+0.073]$ and $-0.407$\,m $[-0.772,+0.016]$: both intervals include zero
(HOLD-UNDECIDED). With the logged stay label (a privileged ceiling) it would change it by $-0.542$ $[-0.855,-0.270]$ and $-0.515$
$[-0.826,-0.254]$. At rest the logged future has two modes and both arms drive a blend of them; a decided hold is worth about
0.5\,m when its cue is right, and the plan is not a sharp enough cue. The committed next step is a go-or-stay cue from perception,
which needs a probe or a training arm on training clips and was blocked on pod compute when the row was written.

\paragraph{A class weight is one number (T4).}""")

rep(r"""The two training runs differ by 0.010 in gap $R^2$.""",
    r"""The two training runs differ by 0.010 in gap $R^2$ (a third baseline seed widens the range to 0.045).""")

rep(r"""By T7, imagination is
admissible for the gist and not for objects. Four arms are training: a state that keeps the cell layout, and a mean state of equal
width as its control, at two seeds each.""",
    r"""By T7, imagination is
admissible for the gist and not for objects. The reads of a state that keeps the cell layout, and of a mean state of equal
width as its control, are in \cref{sec:state}.""")

rep(r"""The deciding experiment is training: 24 arms at six seeds, hierarchy against flat under both training protocols,
with seeds as the unit. A window class reads for the hierarchy only when all six seed differences agree in sign and the two-level
interval excludes zero.""",
    r"""The deciding experiment is 24 arms at six seeds, hierarchy against flat under both training protocols,
with seeds as the unit; all 24 arms were trained on 14 Nov and the read is pending, so there is no verdict here. A window class
reads for the hierarchy only when all six seed differences agree in sign and the two-level interval excludes zero.""")

# ------------------------------------------------------------------ results 8.2 and 8.3 (replace the open data paragraph)
OLD_DATA = r"""lands inside the range of 20 random draws, so no selection claim is made for this representation. A data-efficiency experiment is
training: the same model on 1/2, 1/4 and 1/8 of the clips at fixed steps, flat against hierarchy at 1/4, and clips chosen by their
share of selection-critical windows against random clips.
"""
NEW_DATA = r"""lands inside the range of 20 random draws, so no selection claim is made for this representation. The trained arms are in
\cref{sec:dataeff}.

\subsection{A world-model state that keeps the layout}\label{sec:state}
\Cref{prop:pool} says what a mean over cells cannot hold; the least change is to keep the cells. Arm G keeps the layout (40 cells of
16 numbers, 9,091,503 trainable parameters), arm W is a mean state of equal width (640, 9,251,871 parameters) and the baseline is
the mean state of 128 numbers (7,214,495 parameters); each trained 19,164 steps, G and W at two seeds, the baseline at three. All
reads below are \tier{T0} world-model diagnostics on logged frames, never driving performance, with every control passing first.

\paragraph{The gain is the layout, not the width.} Out of sample (139 episodes, lead gap on 2,485 frames), the lead gap is read at
$R^2$ 0.552 $[0.419,0.651]$ and 0.541 $[0.414,0.633]$ from G, 0.361 $[0.146,0.518]$ and 0.309 $[0.115,0.455]$ from W, 0.237, 0.212
and 0.143 from the three baseline seeds, and 0.133 from ego motion; the state minus ego motion is $+0.419$ $[+0.212,+0.708]$ and
$+0.409$ $[+0.192,+0.710]$ for G (values of different arms are side by side, not paired). On 1,200 training clips, paired over
clips, G minus W is $+0.104$ $[+0.074,+0.134]$ and $+0.116$ $[+0.087,+0.146]$, and W minus the baseline is $-0.008$ $[-0.027,+0.010]$
and $+0.009$ $[-0.010,+0.029]$, against a baseline range of 0.045 over three seeds: the gain is assigned to \emph{location} in the
sense of the card. Lead presence reads at AUROC 0.860 $[0.802,0.906]$ and 0.862 $[0.807,0.908]$ for G, 0.819 and 0.789 for W,
0.807/0.788/0.792 for the baseline and 0.596 for ego motion. Decoded to pictures, the held-out skill is 0.605 $[0.567,0.641]$ and
0.555 $[0.510,0.597]$ for G against 0.777 $[0.752,0.801]$ for the token grid (ratio 0.78/0.72, the bar for the strongest name is
0.90, so the outcome is Z-ABOVE-MEAN), $+0.244$ $[+0.210,+0.276]$ and $+0.202$ $[+0.167,+0.237]$ above the baseline of the same
seed, while W reads 0.338 $[0.272,0.405]$ and 0.345 $[0.283,0.410]$ (Z-FLAT at both seeds); G minus W is $+0.267$ $[+0.229,+0.304]$
and $+0.210$ $[+0.173,+0.246]$. What the layout does not give is the \emph{change}: over 2\,s, the state with history minus the
ego block is $-0.050$ $[-0.094,-0.004]$ and $-0.057$ $[-0.104,-0.009]$ for G and $-0.032$ $[-0.070,+0.008]$ and $-0.017$ $[-0.053,+0.019]$
for W; no state adds to ego motion. The four-family read finds no cell worse (\cref{tab:fam6}).

\paragraph{Why the rollout was flat.} Rolled out 2.0\,s by the predictor of its own arm, the grid state does not read the lead
better than a copy of the present state: rolled-out, present and true future state read 0.442, 0.496 and 0.552 for G at seed 0
(rollout minus copy $-0.054$ $[-0.154,+0.033]$ and $-0.036$ $[-0.153,+0.064]$ at seeds 0 and 1; W $-0.079$ $[-0.226,+0.052]$ and
$-0.093$ $[-0.326,+0.119]$), the outcome IMAGINE-FLAT, and the same name in pixels (rollout minus copy at 2.0\,s $-0.00006$
$[-0.0005,+0.0004]$ and $+0.00065$ $[-0.0001,+0.0014]$ for G). Two reads without any training locate the cause. Seeding the
rollout with the true history changes nothing (the rule of the card names it CTX-FLAT: rollout minus copy $-0.034$ and $-0.004$,
and against the one-state rollout $+0.010$ $[-0.005,+0.030]$ and $+0.028$ $[+0.008,+0.055]$). And one step (0.2\,s) ahead the
error of the flat predictor is 1.233 $[1.180,1.295]$ and 1.254 $[1.197,1.317]$ times the error of copying the state on held-out
windows (W at seed 0: 1.026 $[1.013,1.042]$): it has not learned the change at the step it is trained on.

\paragraph{A predictor that attends across the cells.} The cell predictor has 873,488 parameters against 2,074,496 for the flat
one. One step ahead its error is 0.788 $[0.770,0.807]$ and 0.796 $[0.779,0.814]$ of the copy's. On the lead read at $+2.0$\,s (441
windows with a lead at both times), rolled-out, present and true future state read 0.501/0.487/0.532 at seed 0 and
0.506/0.483/0.551 at seed 1; rollout minus copy is $+0.013$ $[-0.037,+0.057]$ and $+0.023$ $[-0.032,+0.081]$, rollout minus the
ego floor $+0.322$ $[+0.116,+0.614]$ and $+0.327$ $[+0.101,+0.655]$, and the rollout of this predictor minus that of the flat one
$+0.058$ $[-0.030,+0.163]$ and $+0.069$ $[-0.017,+0.174]$. No interval of the two deciding lines excludes zero at either seed, so
the outcome is CELL-FLAT as written, with the wanted sign on every line. The position is kept: on the training clips the lead
gap $R^2$ differs from G by $+0.010$ $[-0.011,+0.031]$ and $-0.033$ $[-0.053,-0.014]$, the second below the baseline range of
0.045 (KEEPS-POSITION), and from the baseline by $+0.107$ and $+0.092$. In pixels the learned rollout is below the copy at 0.8\,s
($-0.00058$ and $-0.00072$) and flat at 2.0\,s ($+0.00044$ $[-0.0003,+0.0012]$ and $-0.00004$ $[-0.0005,+0.0004]$). The change
of the gap over 2\,s is the missing part: the true future state holds skill $+0.190$ and $+0.205$ over persistence, the rollout
$+0.007$ $[-0.103,+0.122]$ and $+0.056$ $[-0.037,+0.162]$. The family read (\cref{tab:fam6}) is preliminary \textsc{no-cost}: it
needs all six baseline seeds and four are read.

\paragraph{Two levers, seed 0 only.} Two further levers were pre-registered before any of their numbers. MOTION trains the
predictor on its own 2.0\,s rollouts (arm PR on the cell predictor) or has it output the state plus a change. TARGET gives the
predictor the measured ego speed as a third action channel (arm PE; by a note frozen before the gate, the inverse-dynamics head
of the level is then also trained to read the speed, so PE tests ``given the speed'' and ``asked for the speed'' together) or
trains a linear head per cell on the frozen token grid of the future frame (arm PF). At seed 0 the two arms read so far give the
lead gap at $+2.0$\,s from the rolled-out, present and true future state 0.442/0.440/0.498 (PR) and 0.521/0.500/0.572 (PE);
rollout minus copy is $+0.002$ $[-0.062,+0.065]$ and $+0.021$ $[-0.028,+0.073]$, and the change skill of the rollout $+0.014$
$[-0.092,+0.130]$ and $-0.064$ $[-0.179,+0.039]$ against ceilings of $+0.159$ and $+0.206$. Both read IMAGINE-FLAT at seed 0, and
neither name is assigned, because a name needs both seeds. The second seeds and the arms PF, RES and RR are trained and being
read; no number for them is given here. The next step is committed for each outcome: a rollout that imagines goes to planning by
imagination; a fixed step with a still lead, or a flat future-token arm, goes to an object-level slot in the state (the lead gap
and its closing speed from the history of the lead head, with a kinematic transition as a fixed rule and a learned residual).

\subsection{Data efficiency and the four families}\label{sec:dataeff}
\paragraph{Data efficiency (preliminary).} The same model was trained at a fixed 19,164 steps on a random half, quarter and eighth
of the 4,512 clips (\tier{T1-kin}, 2,399 windows, 141 episodes; controls pass; a spread is the largest difference between baseline
seeds present). At seed 0, 6\,s ADE on all / selection-critical windows is 3.193/3.703\,m for the full set, 3.144/3.566 for a half,
3.152/3.661 for a quarter and 3.405/4.005 for an eighth. With both seeds of a quarter, the difference from the full set is $+0.112$\,m
$[-0.118,+0.353]$ on all windows (per seed $-0.041$ and $+0.265$; spread 0.273) and $+0.111$ $[-0.151,+0.382]$ on selection-critical
ones, with speed error $-0.027$\,m/s $[-0.268,+0.221]$, and each pair reads \textsc{no-cost}. The baseline seeds 0 to 3 read 3.193,
2.977, 3.161 and 2.920\,m, the quarter 3.152 and 3.242\,m. At one eighth, seed 0, the arm is $+0.253$\,m $[+0.075,+0.451]$ above a
quarter on all windows and $+0.344$ $[+0.094,+0.603]$ on critical ones, both outside the spread (0.216/0.278), and $+0.212$
$[-0.009,+0.442]$ above the full set. The reading: at this model size and step count a quarter of the clips costs nothing that is
measured, and the first cost appears between a quarter and an eighth. The limits are the \textsc{no-cost} rule's: it needs six
baseline seeds and four are read, one seed carries the half and the eighth, a separated interval from one seed is necessary and not
sufficient, and equal steps mean that a quarter of the clips is seen four times as often. Two further reads at a quarter, seed 0:
the hierarchy minus the flat control is $-0.246$\,m $[-0.449,-0.034]$ on selection-critical windows (inside the spread 0.278),
against $+0.109$ $[-0.075,+0.289]$ at the full set, so scarce clips move the gap toward the hierarchy by 0.36\,m; and clips chosen
by their share of selection-critical windows against random clips gain $+0.023$ $[-0.243,+0.287]$ on those windows and lose
$+0.803$ $[+0.482,+1.165]$ on the rest (spread 0.348). If seed 1 and the six baseline seeds confirm, the scaling step is a better
cue and state, not more clips.

\begin{table*}[t]
\centering
\caption{\textbf{Four metric families for the new \tier{T1-kin} reads} (kinematic rollout on the model's own actions, logged frames;
a diagnostic, never driving performance). Differences A minus B, interval in brackets; \emph{in} marks an interval
that excludes 0 and lies inside the spread of the baseline seeds, so it is not an effect. Headway and stop-goal cells are not in
these reads. The first five rows are preliminary (spread from three or four of six baseline seeds); the last row is an emulation on banked rollouts, not a rollout.}
\label{tab:fam6}
\small
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{31mm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{19mm}@{}}
\toprule
Pair (seeds) & Longitudinal & Lateral & Tactical & Strategic \\
\midrule
G minus baseline (0, 1) & speed $-0.066$\,m/s $[-0.151,+0.007]$ & cross-track $-0.107$\,m $[-0.196,-0.017]$ \emph{in}; curvature $+0.014$\,1/m $[-0.014,+0.049]$ & joint class error $+0.008$ $[-0.043,+0.059]$ & PAUSED \\
P minus G (0, 1) & speed $+0.133$\,m/s $[-0.042,+0.304]$ & cross-track $+0.093$\,m $[+0.015,+0.171]$ \emph{in} & no cell worse outside the spread & PAUSED \\
Quarter of clips minus full (0, 1) & speed $-0.027$\,m/s $[-0.268,+0.221]$ & cross-track $-0.028$\,m $[-0.147,+0.088]$; yaw rate $-4.98$\,deg/s $[-9.58,-0.79]$ \emph{in} & lateral class error $-0.062$ $[-0.124,-0.007]$ \emph{in} & PAUSED \\
Half of clips minus full (0) & speed $-0.146$\,m/s $[-0.275,-0.013]$ \emph{in} & cross-track $+0.057$\,m $[-0.016,+0.132]$ & joint class error $+0.041$ $[+0.005,+0.078]$ \emph{in} & PAUSED \\
Eighth of clips minus full (0) & speed $-0.099$\,m/s $[-0.251,+0.047]$ & cross-track $+0.242$\,m $[+0.126,+0.362]$, outside the spread 0.212 & joint class error $-0.059$ $[-0.101,-0.020]$ \emph{in} & PAUSED \\
Hold rule minus arm at rest, hierarchy (0, 1; emulation) & speed error $-0.428$ $[-0.892,-0.082]$ and $-1.268$ $[-1.834,-0.814]$\,m/s & cross-track at 6\,s $-1.944$ $[-2.798,-1.156]$ and $-4.454$ $[-5.801,-3.219]$\,m & joint-cell agreement $+0.123$ $[+0.026,+0.249]$ and $+0.156$ $[+0.056,+0.294]$ & PAUSED \\
\bottomrule
\end{tabularx}
\end{table*}

The strategic family is reported as paused by the PI ruling (item 25), never as a pass. The \tier{T0} reads of this section measure no
family: their four-family statement is that LONGITUDINAL, LATERAL and TACTICAL are not measured by them and STRATEGIC is paused.
"""
rep(OLD_DATA, NEW_DATA)

# ------------------------------------------------------------------ roadmap, outlook, conclusion
rep(r"""Band retraining \textit{(read, four seeds)}""",
    r"""World-model state with the cell layout \textit{(trained, two seeds)} & reads where the lead is (the layout, not the width); the cell predictor fixes the one-step defect; no rollout yet carries the lead's motion 2\,s ahead, and the next lever is an object-level slot \\
Band retraining \textit{(read, four seeds)}""")

rep(r"""The theory of \cref{sec:theory} orders the next steps. Three experiments are training on one GPU: the six-seed replicate that
decides the hierarchy against its flat control; a world-model state that keeps the cell layout, which by \cref{prop:pool} is the
least change that lets the state hold where an object is and how it moves; and the data-efficiency curve that says whether the
next limit is data. The premise of \cref{prop:dom} for v8F-mini is the next read: selection accuracy against the share of the
larger mode per context.""",
    r"""The theory of \cref{sec:theory} orders the next steps, and \cref{sec:state} says where the work stands. The state that keeps the
cell layout holds where the lead vehicle is, as \cref{prop:pool} says it must, and the cell predictor fixes the step it is trained
on; what is open is that the model does not yet imagine the motion of the lead vehicle 2\,s ahead better than a copy, so by T7
imagination is not yet admissible for objects. The pre-registered levers (a rollout loss, a residual form, the ego speed as an
action channel, the frozen future-token target) are read at seed 0 or in training; if they stay flat, the committed step is an
object-level slot with a kinematic transition. Three further reads are committed: the deciding six-seed replicate of the hierarchy
against its flat control (all 24 arms were trained on 14 Nov; the read is pending and this paper states no verdict), the
dominance premise on those six seeds, and a go-or-stay cue at rest from perception. Data efficiency is preliminary until the
second seeds and all six baseline seeds are read.""")

rep(r"""ahead
where modes separate, not yet established overall, and limited by a pooled state that keeps the scene and loses its objects.""",
    r"""ahead
where modes separate, not yet established overall, and limited first by a pooled state that kept the scene and lost its objects.
A state that keeps the cell layout now holds the lead; what remains open is a rollout that carries the lead's motion.""")

open(DST, "w", encoding="utf8", newline="\n").write(t)
print("ok edits", n_edits, "sha256", hashlib.sha256(t.encode("utf8")).hexdigest())

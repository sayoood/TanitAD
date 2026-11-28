"""patch_tex_v5.py -- the TanitAD-4B paper, version 5 (2026-11-13): v4's tex plus the theory and proof programme of v8:
a new section (nine statements, machine-checked: 56 solver conditions, 54 Lean 4 theorems), a new results subsection that tests their premises,
one abstract passage, one contribution item, one outlook paragraph, one conclusion sentence, three references. No number of v4 is changed.
Every edit must match exactly once.
Usage: python -I patch_tex_v5.py <v4.tex> <out.tex>
Sources of every new number: folder raw/2026-11-12-v8f-proof/ of the v8f-mini programme (V8_THEORY.md and the verdict files it names); see BUILD.md."""
import hashlib
import sys

SRC, OUT = sys.argv[1], sys.argv[2]
b = open(SRC, "rb").read()
assert hashlib.sha256(b).hexdigest() == "e51ef9c21ca13c1143b931f2533676d125e69aeddc90840c92807a856faf53e6", "base is not v4's tex"
s = b.decode("utf-8")
assert "\r" not in s
EDITS = []

# ---------------------------------------------------------------------------------------------------------------- 1. abstract
EDITS.append((
    r"so the next lever is stop perception. All of it rests on an evaluation",
    r"so the next lever is stop perception. We also state the design as nine theorems with measurable premises and check them by machine "
    r"(56 solver conditions and 54 Lean~4 theorems). They give the exact conditions under which a multi-level predictor and a decision-first "
    r"planner dominate their flat counterparts, and a pooling theorem: a world-model state that averages over image cells keeps counts and "
    r"totals and loses position and motion. Measured on the trained model, that state holds the presence of a lead vehicle beyond ego motion "
    r"(AUROC 0.81/0.79 against 0.60) and nothing about how its gap changes; the hierarchy is ahead of a flat control on selection-critical "
    r"windows, and the comparison on all windows is not yet established. All of it rests on an evaluation"))

# ---------------------------------------------------------------------------------------------------------------- 2. contributions
EDITS.append((
    "        partial codes unsupported, and a three-variance account of evaluation intervals.\n",
    "        partial codes unsupported, and a three-variance account of evaluation intervals.\n"
    r"  \item \textbf{A machine-checked theory} (\cref{sec:theory,sec:premises}): nine statements on when a hierarchy, a pooled state, a" "\n"
    r"        rule shield, imagination and weight averaging can and cannot help, checked by an SMT solver and the Lean~4 kernel, each" "\n"
    r"        with a premise that is then measured." "\n"))

# ---------------------------------------------------------------------------------------------------------------- 3. theory section
THEORY = r"""\section{A machine-checked theory of the hierarchy}\label{sec:theory}
The design claims of \cref{sec:4b} can be stated as theorems about model classes. Each has a premise that must be \emph{measured}
before it says anything about a model. We wrote nine such statements (T1--T9, \cref{tab:theory}) and froze each before its proof.
All were checked by machine: 56 verification conditions with the SMT solver Z3, each family with mutated twins that the solver must
refute (30 are), and the central ones in general form with the Lean~4 kernel and Mathlib (54 theorems, no \texttt{sorry}, standard
axioms only). Two groups were written after reads they explain; for those reads they are explanatory, not predictive. The premises
are tested in \cref{sec:premises}.

\begin{table*}[t]
\centering
\caption{\textbf{Nine statements, their machine check, and their measured premise for v8F-mini.} Z3: verification conditions proved by
an SMT solver, with mutated twins refuted. Lean: checked by the Lean~4 kernel in general form. Every measured value is a \tier{T0} or
\tier{T1-kin} diagnostic on logged frames, never driving performance (seed 0 / seed 1 where two values are given).}
\label{tab:theory}
\small
\begin{tabularx}{\textwidth}{@{}l>{\raggedright\arraybackslash}p{60mm}l>{\raggedright\arraybackslash}X@{}}
\toprule
 & Statement & Check & Measured premise or consequence for v8F-mini \\
\midrule
T1 & A level that jumps $k$ steps has the better worst-case bound at every horizon iff $\varepsilon_k<\varepsilon\sum_{i<k}L^i$ & Z3, Lean &
holds in the linear class: the direct 2\,s error is 0.42/0.43 of ten one-step errors; the learned ten-step rollout fails (skill $-0.15$/$-0.17$ at 2\,s) \\
T2 & Under squared error no plan beats the mean; under a hit-or-miss criterion a decided plan beats every single plan iff selection accuracy exceeds the share of the larger mode & Z3, Lean &
open: top-1 selection 57--59\,\%; the hierarchy is ahead of a flat control on selection-critical windows ($-0.48$/$-0.16$\,m) and not established on all windows \\
T3 & A shield keeps a rule for every network; a penalty of weight $\lambda$ only bounds the violation by $(J_0-J_\infty)/\lambda$ & Z3 &
with the logged rest distance: overrun exactly 0 and stop-window 2--6\,s ADE $-3.51$/$-3.37$\,m; with the model's own stop cue: no gain \\
T4 & Class weighting is a threshold move, $\tau=r/(1+r)$ & Z3 &
one added bias reproduces the stop decision of a class-weighted training run in 4 of 4 cells \\
T5 & A state pooled over cells keeps counts and totals and loses position and motion & Z3, Lean &
planted object: AUROC 1.00 from a grid state, 0.50 from the mean state; trained state: lead presence $+0.21$/$+0.19$ AUROC beyond ego motion, change of the gap: nothing beyond ego motion \\
T6 & Skill over persistence is the predictable fraction of the change & Z3 &
the learned world model is below a closed-form ridge at 0.2\,s and at 2\,s; a closed-form stack of both is above both at every horizon \\
T7 & An imagined gain is a real gain only above twice the model gap & Z3 &
not yet admissible for objects: the state decodes to the scene gist, and its learned rollout is no closer to the future than a copy \\
T8 & Sampling by gradient norm minimises the second moment of the weighted gradient & Z3 &
selection by coverage or by leverage lands inside the range of random draws (probe level); trained arms are running \\
T9 & A member average is never worse than the member mean; a rank-one edit is the least change that moves one key & Z3 &
a two-member weight average realises 0.74 of the ensemble gain in one forward pass \\
\bottomrule
\end{tabularx}
\end{table*}

\subsection{When multi-level prediction dominates}
Let a one-step predictor have error at most $\varepsilon$ and let the dynamics expand an error by at most the factor $L\ge0$ per
step, so that $e_{t+1}\le L\,e_t+\varepsilon$ with $e_0=0$.
\begin{proposition}[Composed bound, jump bound and criterion]\label{prop:jump}
After $H$ steps $e_H\le\varepsilon\sum_{i<H}L^{i}$, and the bound is attained. A level that predicts $k$ steps at once with error
$\varepsilon_k$ has error at most $\varepsilon_k\sum_{j<m}L^{kj}$ after $m$ jumps. At every horizon $H=mk$ with $m\ge1$, the jump bound
is below the composed bound if and only if
\begin{equation}
\varepsilon_k<\varepsilon\sum_{i<k}L^{i}\qquad(\text{for }L=1:\ \varepsilon_k<k\,\varepsilon).\label{eq:jump}
\end{equation}
\end{proposition}
\noindent The horizon drops out. A slow level earns its place exactly when its direct $k$-step error is below $k$ composed fast
steps. This is the formal content of the clocks of \cref{sec:4b}, and its premise is a ratio one can measure.

\subsection{When a decision before the trajectory dominates}
\begin{proposition}[Risk split and non-dominance]\label{prop:split}
Let outcomes $y_i$ carry weights $w_i\ge0$ and modes $g(i)$ with weighted means $\mu_g$. For any plan $c$ that is constant on each mode,
\begin{equation}
\begin{aligned}
\sum_i w_i\|y_i-c_{g(i)}\|^2={}&\sum_i w_i\|y_i-\mu_{g(i)}\|^2\\
&+\sum_i w_i\|\mu_{g(i)}-c_{g(i)}\|^2 .
\end{aligned}
\end{equation}
Under squared error no single plan beats the mean, so a selector that reads only what the flat model reads cannot gain.
\end{proposition}
\noindent A hierarchy therefore pays through three channels only: (a) supervised intermediate labels, which a flat regressor never
sees; (b) an acceptance region that is not convex; (c) inputs that enter at the upper level, such as navigation. Channel (b) has an
exact form.
\begin{proposition}[Dominance under a hit-or-miss criterion]\label{prop:dom}
Let the outcomes of one context fall into two modes with centres $a$ and $b$, each outcome within $r$ of its centre, and let a plan hit
an outcome when it lies within $\mathrm{tol}\ge r$ of it. If $\|a-b\|>2\,\mathrm{tol}+2r$, all outcomes that any single plan hits lie in
one mode. A plan decided by a selector hits every outcome whose mode was selected correctly. Hence, if the selector is right on more
outcomes than the larger mode holds, the decided plan hits strictly more outcomes than every single plan.
\end{proposition}
\noindent This is the dominance claim of the programme in its exact, conditional form. Its premise is a selection accuracy above the
share of the larger mode, at a mode separation above $2\,\mathrm{tol}+2r$. Where the premise fails the theorem is silent, and a flat
plan can be ahead. The literature gives the same caution from the empirical side: most of the benefit of hierarchy in reinforcement
learning has been attributed to exploration~\cite{nachum19}, which offline imitation does not have, and acting through a goal
representation costs a bounded amount that grows with the error the representation induces~\cite{nachum18}. A flat control at matched
budget is therefore not optional.

A commit rule that hands control to the hierarchy only on flagged contexts has an exact break-even. Let $p$ be the share of
selection-critical contexts, $d_c<0<d_b$ the class means of hierarchy minus flat, and $t$ and $f$ the flag rates on the two classes.
With $s=(1-p)\,d_b/(p\,|d_c|)$, the composite beats the flat plan if and only if $t>s f$, and beats the hierarchy if and only if
$1-t<s\,(1-f)$. A flag without information cannot beat both.

\subsection{What a pooled state can hold}
The level-1 state of v8F-mini's world model is a mean over the $4\times10$ cells of adapted camera features.
\begin{proposition}[Pooling]\label{prop:pool}
Let $z=w\sum_{i\in K}(Ax_i+b_i)$ pool per-cell features $x_i$ over a set of cells $K$, with $A$ linear and $b_i$ a constant per cell.
Moving an object of feature $s$ between two cells of $K$ leaves $z$ unchanged, for every background. Moving it across the border of
$K$ changes $z$ by $wAs$. Consequently, for an object equally likely in each cell of $K$, every decoder of $z$ predicts every function
of the object's position with squared error at least the variance of that function: its skill is at most zero. With $m$ objects of
equal strength, $z$ depends on $m$ only. For any per-cell map, linear or not, a permutation of the cells leaves $z$ unchanged.
\end{proposition}
\noindent A mean over cells therefore keeps counts and totals (how many objects, the light level, how much of the frame is occupied)
and loses position and motion. An object that moves between cells changes nothing in the state, a copy predicts the next state
exactly, and a learned predictor of that state has nothing to imagine about the object. A state that keeps the cell layout reads the
position linearly, with a margin that does not depend on the background. Pooling at an intermediate scale loses position inside a
block and keeps it across blocks.

\subsection{Rules, class weights, imagination and weight space}
\begin{itemize}
  \item \emph{Rules (T3).} With a braking distance $B$ that obeys $B(v)=v\,\Delta t+B(\max(v-q,0))$, a stop-line shield preserves
        $B(v)\le d$ for \emph{any} proposed speed, its fallback is always feasible, and the distance to the line never goes below
        zero. A masked selection over candidates is admissible whenever any candidate is. A penalty of weight $\lambda$ only bounds
        the violation by $(J_0-J_\infty)/\lambda$, so a loss term guarantees nothing. What remains is perception: a shield on a
        predicted constraint violates the true rule at most as often as the constraint's envelope misses.
  \item \emph{Class weights (T4).} Under a pure prior change, deciding at 0.5 after weighting equals deciding at
        $\tau=r/(1+r)$ before weighting, with $r$ the odds ratio of the two priors. A class weight moves a threshold along one ROC
        curve and adds no discrimination.
  \item \emph{Yardstick (T6).} Skill over persistence equals the predictable fraction of the change. A feature with lag correlation
        $\rho$ gives its own past a skill of $(1-\rho)/2$, so a slowly varying pooled target leaves little to predict at short
        horizons.
  \item \emph{Imagination (T7).} If imagined and true returns differ by at most $G$, an imagined gain above $2G$ is a real gain, and
        choosing by imagined cost loses at most $2G$. This is the deterministic finite-horizon analogue of the model-based bound
        of~\cite{mbpo}; $G$ follows from \cref{prop:jump}.
  \item \emph{Data and weights (T8, T9).} Sampling in proportion to the per-sample gradient norm minimises the second moment of
        the importance-weighted gradient. The mean of $K$ members has an error equal to the mean member error minus the mean member
        scatter, so it is never worse than the member mean, and a weight average equals that output average exactly for a head that
        is linear in its weights.
\end{itemize}

% =====================================================================================================================
"""
EDITS.append((r"\section{Evaluation doctrine}\label{sec:eval}", THEORY + r"\section{Evaluation doctrine}\label{sec:eval}"))

# ---------------------------------------------------------------------------------------------------------------- 4. results: the premises
PREMISES = r"""\subsection{Testing the premises of the theory}\label{sec:premises}
Each read below was pre-registered with its controls and both outcomes, and is a \tier{T0} or \tier{T1-kin} diagnostic on logged
frames; the strategic family stays paused. Three experiments are still training, and their reads are not in this version.

\paragraph{Hierarchy against a flat control (T2).} A flat control at matched budget cuts the interface between the tactical and the
operative level. On all windows, with on-policy (DAgger) data in both arms, the comparison is \emph{not established}: 6\,s ADE,
hierarchy minus flat, is $-0.345$\,m $[-0.442,-0.251]$ at seed 0 and $-0.054$\,m $[-0.161,+0.055]$ at seed 1 (\tier{T1-kin}, one run
per arm per seed). Split by window class, with the mode thresholds fixed before the read (1,305 selection-critical and 1,094
common-case windows), the hierarchy is ahead on selection-critical windows by 0.48/0.16\,m, separated at both seeds; without
on-policy data it is ahead by 0.18/0.12\,m, separated at one seed. On common-case windows it is ahead by 0.18\,m at seed 0 and behind
by 0.07\,m at seed 1 with on-policy data, and behind by 0.32/0.49\,m without. The difference between the two classes is negative and
separated in all four cells ($-0.29$, $-0.23$, $-0.50$, $-0.61$\,m). This is the pattern that \cref{prop:dom} permits: a gain where
modes separate, and none promised elsewhere. It is not yet the mechanism: an exact decomposition does not place the margin in the
selection term, and both arms miss the logged 12-cell mode on 53 to 71\,\% of windows. A commit rule does not bank the gain either.
Letting the hierarchy drive only where its tactical output is confident beats both arms in no cell, because that confidence
separates the two window classes at AUROC 0.64 only; by the break-even condition of \cref{sec:theory}, a gate at that AUROC banks an
estimated 0.3 to 7.4\,\% of what a gate with the logged class would. The lever is the accuracy of the decision, not a gate on it.
With one run per arm all of this is necessary evidence and not sufficient. The deciding experiment is training: 24 arms at six seeds, hierarchy against flat under both training protocols,
with seeds as the unit. A window class reads for the hierarchy only when all six seed differences agree in sign and the two-level
interval excludes zero.

\paragraph{A yardstick for the world model (T1, T6).} The latent features are slow: their lag correlation is 0.975 at 0.2\,s and 0.883
at 2\,s. Against a forecast of no change, the learned predictor's skill is $+0.001$ at 0.2\,s, where a cross-fitted closed-form ridge
reads $+0.022$, and $-0.15$/$-0.17$ at 2\,s, where the ridge reads $+0.12$/$+0.13$ (\tier{T0}). A closed-form stack of the ridge and
the learned prediction reads $+0.040$, $+0.06$, $+0.15$ and $+0.17$ at 0.2, 0.4, 0.8 and 2.0\,s, above both parts at every horizon and
both seeds: weights written without a gradient step improve the world model. The premise of \cref{prop:jump} holds in the linear
class and strengthens with the horizon. The direct error is 0.74, 0.57, 0.52 and 0.42 of the composed one-step bound at 0.4, 0.8, 1.0
and 2.0\,s (0.416 $[0.389,0.449]$ at 2\,s). Multi-level prediction is therefore the right structure for these latents, and the learned
ten-step rollout is the part that fails. Even the stack explains only 4 to 17\,\% of the change, so the target space limits
imagination as well.

\paragraph{Rules by construction (T3).} Unshielded plans pass the logged rest point in 91.6/84.7\,\% of the 131 stop windows. The
verified shield, given the logged rest distance (a privileged ceiling), brings the overrun to exactly zero on every feasible window
and lowers stop-window 2--6\,s ADE by 3.51/3.37\,m with no cost in any cell (\tier{T1-kin}). Given the stop position that the model
itself emits (median error 11.8/9.6\,m, a flag on 10/18 of the 131 windows) it gains nothing. A split-conformal margin behaves as
proved, with held-out coverage 0.908 for a nominal 0.9, at the price of a margin of about 7.4\,m. Probes on the frozen tokens do not
supply the cue: ego kinematics alone rank stops at AUROC 0.810 $[0.744,0.868]$, and adding the $8\times20$ tokens changes that by
$-0.011$ $[-0.061,+0.043]$ (the stop label is derived from logged ego motion, so the ego baseline partly reads the source of its
own label). Rule adherence comes from the construction; usefulness comes from the sharpness of the cue.

\paragraph{A class weight is one number (T4).} Adding one scalar to the stop-goal bias of the unweighted model reproduces what a full
training run with a stop class weight does to the stop decision. At the offset that matches the trained false-positive rate, the hit rate
is within the replicate gap of the trained arm in all four cells (differences 0.009--0.026 against a gap of 0.068), and the stop
AUROC stays at 0.819/0.850 (\tier{T1-kin}, first tick). A dose is an edit that costs seconds; a better stop decision needs a better cue.

\paragraph{What the camera interface carries (T5).} The frozen tokens carry non-ego content beyond ego motion on four targets
(\tier{T0}, 139 episodes): lead gap $+0.29$ $R^2$ $[+0.11,+0.55]$, nearest agent $+0.15$, agent count $+0.16$ and lead presence $+0.18$
AUROC $[+0.09,+0.27]$. A prediction of ours failed here. From the camera geometry we had expected that no reader of the $4\times10$
tokens recovers far range; measured, they carry range beyond ego motion in the far stratum, and under a rank-256 probe an $8\times20$
grid adds nothing (lead gap $+0.011$ $R^2$ $[-0.023,+0.048]$). The probe rank hid part of the answer: with a full-kernel probe the
finer grid is ahead by $+0.076$ $R^2$ $[+0.001,+0.170]$ on lead gap and $+0.043$ AUROC $[+0.022,+0.066]$ on lead presence. Read
directly with an attentive probe, un-pooled $16\times40$ tokens give lead gap $R^2$ 0.46, against 0.40 for the $4\times10$ grid and 0.13
for ego motion alone; the best fine set gains $+0.055$ $[+0.000,+0.111]$, under its bar of 0.10, and lead presence rises from AUROC
0.77 to 0.84 ($+0.066$ $[+0.033,+0.103]$). The probes are still rising at 111 training episodes, so these reads are lower bounds.
A finer interface is a live but thin lever; the larger gap lies between what the tokens carry and what the model's own state exposes.

\paragraph{The world-model state, untrained and trained (T5, T7).} \Cref{prop:pool} was written before its controls. On real tokens a
planted object's position is read at AUROC 1.000 from a state that keeps the cell layout and 0.495 from the mean state, and a planted
motion at 1.000 from two frames of the grid state and 0.493 from two frames of the mean state. Without training, the lead gap is read
at $R^2$ 0.40 from the grid state, 0.26 from the mean state and 0.13 from ego motion alone; grid minus ego is $+0.27$ $[+0.08,+0.54]$,
and grid minus mean, $+0.14$ $[-0.004,+0.29]$, does not exclude zero on 139 episodes. On the \emph{trained} model the mean state was read at
two training seeds. Out of sample it tells whether a lead vehicle is present at AUROC 0.807/0.788, against 0.596 from ego motion
(differences $+0.211$ $[+0.105,+0.314]$ and $+0.192$ $[+0.101,+0.286]$). On 1,200 training clips it reads the lead gap at $R^2$ 0.456/0.445
against 0.187. For the \emph{change} of that gap over 2\,s, ego motion alone has a skill of $+0.291$ over persistence, and adding the
state adds nothing: $-0.028$ $[-0.065,+0.008]$ and $-0.025$ $[-0.059,+0.009]$. The two training runs differ by 0.010 in gap $R^2$. The
trained state knows where the lead is, and nothing about where it is going beyond ego motion: \cref{prop:pool} on a trained model.
Decoded back to pictures, the state shows the scene gist (sky, road, light level) and no object, at a skill of 0.36/0.35 against 0.78
for the token grid, and its learned 2\,s rollout is no closer to the true future frame than the present frame. By T7, imagination is
admissible for the gist and not for objects. Four arms are training: a state that keeps the cell layout, and a mean state of equal
width as its control, at two seeds each.

\paragraph{Weight space (T9).} The 50/50 weight average of two same-initialisation replicates lowers tactical ADE by 0.075\,m
$[0.047,0.103]$ against the member mean and realises $\kappa=0.74$ $[0.48,0.99]$ of the gain of an output ensemble, in one forward pass
and with no separated cost in any family (\tier{T1-kin}). It is not ahead of the better member; what it removes is the risk of the
worse one. Read in weight space (one replicate pair, no interval), a stop class weight moves the model less than a change of seed does in every module group, and
clearly exceeds seed noise only in two head biases, which is what T4 says a class weight is.

\paragraph{Data (T8).} At the probe level, choosing 25 or 50\,\% of the training episodes by coverage or by leverage in token space
lands inside the range of 20 random draws, so no selection claim is made for this representation. A data-efficiency experiment is
training: the same model on 1/2, 1/4 and 1/8 of the clips at fixed steps, flat against hierarchy at 1/4, and clips chosen by their
share of selection-critical windows against random clips.

"""
EDITS.append((r"\section{The frontier: assertive launch, and re-rendered simulation}\label{sec:frontier}",
              PREMISES + r"\section{The frontier: assertive launch, and re-rendered simulation}\label{sec:frontier}"))

# ---------------------------------------------------------------------------------------------------------------- 5. outlook
EDITS.append((
    r"investment yields a measured answer rather than an anecdote.",
    r"investment yields a measured answer rather than an anecdote." "\n\n"
    r"The theory of \cref{sec:theory} orders the next steps. Three experiments are training on one GPU: the six-seed replicate that" "\n"
    r"decides the hierarchy against its flat control; a world-model state that keeps the cell layout, which by \cref{prop:pool} is the" "\n"
    r"least change that lets the state hold where an object is and how it moves; and the data-efficiency curve that says whether the" "\n"
    r"next limit is data. The premise of \cref{prop:dom} for v8F-mini is the next read: selection accuracy against the share of the" "\n"
    r"larger mode per context."))

# ---------------------------------------------------------------------------------------------------------------- 6. conclusion
EDITS.append((
    r"tactical precision, with convex returns, and in the evaluation tier.",
    r"tactical precision, with convex returns, and in the evaluation tier. A machine-checked theory states when the hierarchy can" "\n"
    r"dominate and what its present world-model state cannot hold, and the measurements place v8F-mini against each premise: ahead" "\n"
    r"where modes separate, not yet established overall, and limited by a pooled state that keeps the scene and loses its objects."))

# ---------------------------------------------------------------------------------------------------------------- 7. references
EDITS.append((
    "\n\\end{thebibliography}",
    "\n"
    r"\bibitem{nachum18} O.~Nachum, S.~Gu, H.~Lee and S.~Levine. Near-optimal representation learning for hierarchical reinforcement learning. arXiv:1810.01257, 2018." "\n"
    r"\bibitem{nachum19} O.~Nachum et al. Why does hierarchy (sometimes) work so well in reinforcement learning? arXiv:1909.10618, 2019." "\n"
    r"\bibitem{mbpo} M.~Janner, J.~Fu, M.~Zhang and S.~Levine. When to trust your model: Model-based policy optimization. arXiv:1906.08253, 2019." "\n"
    r"\end{thebibliography}"))

for old, new in EDITS:
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)
open(OUT, "wb").write(s.encode("utf-8"))
print("edits", len(EDITS), "sha256", hashlib.sha256(s.encode("utf-8")).hexdigest())

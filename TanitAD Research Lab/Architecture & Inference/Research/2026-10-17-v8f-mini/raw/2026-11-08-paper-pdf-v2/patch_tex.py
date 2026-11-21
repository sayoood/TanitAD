"""Applies the PI's rulings of 2026-11-08 and the mathematics fixes to the TanitAD-4B paper. Every edit must match exactly once."""
W = "C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/p4b/"
s = open(W + "tanitad_4b_paper.tex", encoding="utf-8", newline="").read()
assert "\r\n" not in s


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


# ---------------------------------------------------------------------- abstract
rep(r"""brains (operative, tactical, strategic and fallback), each with its own clock, horizon, state space and predictor. Goals condition
strictly downward and latents flow upward through stop-gradient barriers.""",
    r"""brains (operative, tactical, strategic and fallback), each with its own clock, horizon, state space and predictor. The operative
level owns up to 2\,s, the tactical level up to 6\,s and the strategic level 6 to 30\,s. Every level reads the current measured ego state.
The tactical level emits goals and constraints that act as the action conditioning the operative level, and the output is one
trajectory or several, up to 6\,s. Goals condition strictly downward and latents flow upward through stop-gradient barriers.""")
rep(r"""least one row in each of the four metric families: longitudinal, lateral, tactical and strategic. All of it""",
    r"""least one row in each of the four metric families: longitudinal, lateral, tactical and strategic (a measurement; the strategic bars are paused by decision). All of it""")
rep(r"""launch from standstill), the re-rendered simulation tier in preparation to address it, and a research roadmap toward a 300\,M-parameter
4B driver deployable within 100\,ms on an edge computer.""",
    r"""launch from standstill), the re-rendered simulation tier built to address it, and a research roadmap toward a 4B driver within a
budget of at most 800\,M parameters, deployable within 100\,ms on an edge computer.""")

# ---------------------------------------------------------------------- introduction
rep(r"""  \item an \textbf{operative} brain for continuous control;
  \item a \textbf{tactical} brain for manoeuvre selection;
  \item a \textbf{strategic} brain for routing;""",
    r"""  \item an \textbf{operative} brain for continuous control and the trajectory, up to 6\,s;
  \item a \textbf{tactical} brain that emits goals and constraints up to 6\,s;
  \item a \textbf{strategic} brain for route checkpoints, detected context and the goals derived from them, over 6 to 30\,s;""")
rep(r"""and scaling toward the 300\,M-parameter 4B budget.""", r"""and scaling toward the 4B budget of at most 800\,M parameters.""")

# ---------------------------------------------------------------------- figure 1
rep(r"""{measured ego motion\\$(v,a,\dot\psi)$, nav};""", r"""{measured ego motion\\$(v,a,\dot\psi)$, nav\\(read by every level)};""")
rep(r"""$\hat z_O(t{+}j)=P_O\big(z_O,(a,\kappa)\,\big|\,g_{\mathrm{tac}}\big)$\\[0.5mm] continuous control $\cdot$ 0.1--2\,s};""",
    r"""$\hat z_O(t{+}j)=P_O\big(z_O,(a,\kappa)\big)$\\[0.5mm] control and trajectory $\cdot$ band up to 2\,s};""")
rep(r"""$\hat z_T(t{+}k)=P_T\big(z_T,a_T\,\big|\,g_{\mathrm{str}}\big)$\\[0.5mm] imagine-and-select manoeuvres $\cdot$ 2--6\,s};""",
    r"""$\hat z_T(t{+}k)=P_T\big(z_T,a_T\big)$\\[0.5mm] goals and constraints $\cdot$ up to 6\,s};""")
rep(r"""$\hat z_S(t{+}K)=P_S\big(z_S,a_S\big)$\\[0.5mm] routing over a latent transition graph $\cdot$ 6--30\,s};""",
    r"""$\hat z_S(t{+}K)=P_S\big(z_S,a_S\big)$\\[0.5mm] checkpoints, context, derived goals $\cdot$ 6--30\,s};""")
rep(r"""node[lbl,left]{goal $g_{\mathrm{tac}}$}""", r"""node[lbl,left]{action $u_T$: goals, constraints}""")
rep(r"""{6\,s trajectory $\cdot$ controls $\cdot$ decisions exposed per level};""", r"""{one trajectory or several, up to 6\,s $\cdot$ decisions exposed per level};""")
rep(r"""are forced to abstract. Goals condition strictly downward. Latents pass upward only through stop-gradient adapters, and the""",
    r"""are forced to abstract. Goals condition strictly downward, through the readers and the planner cost; the predictors themselves are action-conditioned and goal-free. Latents pass upward only through stop-gradient adapters, and the""")

# ---------------------------------------------------------------------- 3.1: the predictors are goal-free
rep(r"""\emph{stopped} lower latents, $z_T=\phi_T(\mathrm{sg}[z_O])$ and $z_S=\phi_S(\mathrm{sg}[z_T])$. The dynamics are conditioned strictly
downward:
\begin{equation}
\begin{aligned}
\hat z_S(t{+}K) &= P_S\big(z_S(t),a_S(t)\big),\\
\hat z_T(t{+}k) &= P_T\big(z_T(t),a_T(t)\,\big|\,g_{\mathrm{str}}\big),\\
\hat z_O(t{+}j) &= P_O\big(z_O(t),(a,\kappa)_t\,\big|\,g_{\mathrm{tac}}\big).
\end{aligned}
\end{equation}
Two structural""",
    r"""\emph{stopped} lower latents, $z_T=\phi_T(\mathrm{sg}[z_O])$ and $z_S=\phi_S(\mathrm{sg}[z_T])$. Each level's dynamics are conditioned on
that level's own action:
\begin{equation}
\begin{aligned}
\hat z_S(t{+}K) &= P_S\big(z_S(t),a_S(t)\big),\\
\hat z_T(t{+}k) &= P_T\big(z_T(t),a_T(t)\big),\\
\hat z_O(t{+}j) &= P_O\big(z_O(t),(a,\kappa)_t\big).
\end{aligned}
\end{equation}
Goals do not enter the predictors. They condition the lower levels through the readers and the planner cost (\cref{sec:bands}).
Two structural""")
rep(r"""represent lower-level detail even if its objective would reward it. A 15\,s strategic roll is three steps of a 256-dimensional model,
so only the operative path runs at the full rate.""",
    r"""represent lower-level detail even if its objective would reward it. At 0.2\,Hz a 15\,s strategic roll is three steps of a 256-dimensional
model (about seven at 0.5\,Hz), so only the operative path runs at the full rate.""")

# ---------------------------------------------------------------------- the new subsection
rep(r"""Every build checks it by asserting zero gradient on forbidden paths, with a deliberate-regression arm that must fail.
""", r"""Every build checks it by asserting zero gradient on forbidden paths, with a deliberate-regression arm that must fail.

\subsection{Time bands, the current ego state and the tactical action}\label{sec:bands}
Let $\Delta=0.2$\,s be the grid step and $t$ the present. The levels own the bands
\begin{equation}
B_O=(0,2]\,\mathrm{s},\quad B_T=(2,6]\,\mathrm{s},\quad B_S=(6,30]\,\mathrm{s}.
\end{equation}
The output of the stack is one trajectory or several, up to 6\,s,
\begin{equation}
X_t=\big(x_{t+j\Delta}\big)_{j=1}^{30}\in\Rset^{30\times2},
\end{equation}
so knots 1--10 lie in $B_O$ and knots 11--30 in $B_T$. A horizon of 0.8\,s is not a long-term abstract level: a level's reach has to
match its band.

\paragraph{The cascade.} Let $e_t$ be the current measured ego state, including yaw rate, and $V_t$ the vision tokens of the current and
past frames, with the navigation command and speed limit. Every level reads $e_t$, and no level reads a state planned by another level:
\begin{equation}\label{eq:cascade}
\begin{aligned}
s_t&=f_S(e_t,V_t),\\
u_t&=f_T(e_t,V_t,\hat s),\\
X_t&=f_O(e_t,V_t,\hat s,\hat u),
\end{aligned}
\end{equation}
where a hat marks a held output re-expressed in the current ego frame by measured odometry. The tactical output is a \emph{tactical
action},
\begin{equation}
u_t=(g_t,\lambda^{\mathrm{lat}}_t,\lambda^{\mathrm{lon}}_t,\gamma_t),
\end{equation}
the goals, the lateral and longitudinal classes and the constraints, over a horizon of up to 6\,s. T has no trajectory head. $u_t$
conditions O as its action, so the plan is a function of the context \emph{given} $u_t$. A single decoder produces all 30 knots, which
makes the plan continuous across 2\,s by construction, and the two bands differ only in the loss:
\begin{equation}
\mathcal L_O=w_{\mathrm{op}}\,\ell\big(X^{1:10},X^{\ast\,1:10}\big)+w_{\mathrm{tac}}\,\ell\big(X^{11:30},X^{\ast\,11:30}\big).
\end{equation}
For $M$ hypotheses, $X^{(m)}_t=f_O(\,\cdot\mid\hat u^{(m)})$ with $\hat u^{(m)}$ the $M$ best tactical actions; $M=1$ is the arm of record.

\paragraph{The strategic level.} Its output is $\sigma_t=(C_t,p_t,\gamma^S_t)$: checkpoints $C_t$ at $\tau\in\{6,9,\dots,30\}$\,s; the
posteriors $p_t$ of the detected context (country, period of day, illumination, weather, visibility and road surface), read from the
current and past images only; and the goals $\gamma^S_t$ derived from them. $\sigma_t$ enters $f_T$ as a detached input, so strategic goals
reach O through $u_t$. No output of a situation classifier enters any goal path.

\paragraph{World-model planning.} The levels $l=1,2,3$ step at $\tau_l=0.2,0.6,3.0$\,s and are trained to reaches of 2, 6 and 18\,s. Following
H-JEPA~\cite{hjepa}, with $g$ a goal latent, $\tilde z^{\,l+1}_i$ the rollout of level $l$ lifted to the level above, and $l=2,1$,
\begin{equation}
\begin{aligned}
a^{3,\ast}&=\arg\min_{a^3}\big\|\hat z^{\,3}_{T_3}-g\big\|^2,\\
a^{l,\ast}&=\arg\min_{a^l}\big\|\tilde z^{\,l+1}_{K}-\hat z^{\,l+1,\ast}_{K}\big\|^2\\
&\qquad+\lambda\sum_{i<K}\big\|\tilde z^{\,l+1}_{i}-\hat z^{\,l+1,\ast}_{i}\big\|^2 .
\end{aligned}
\end{equation}
Each level plans from its own encoded current state, and the only downward signal is the predicted state of the level above. This
planner is the baseline against which other approaches are compared.

\paragraph{Status.} \Cref{eq:cascade} is trained on v8F-mini as described in \cref{sec:v8f}, with equal knot weights and tactical labels
over 2--8\,s. The band losses, the 2--6\,s labels, the strategic extension and the world-model planner are built and tested, and the
world models for these bands are in training. A first planner read on the earlier models, with a true-future goal (T0), did not beat
holding the action. No driving result exists yet for this design.
""")

# ---------------------------------------------------------------------- anti-collapse
rep(r"""By the Cram\'er--Wold theorem, normal one-dimensional projections in all directions imply a joint isotropic Gaussian, the embedding
law LeJEPA shows to be optimal, and the gradients stay uniformly bounded.""",
    r"""By the Cram\'er--Wold theorem, standard normal one-dimensional projections in all directions imply a joint standard Gaussian, the
embedding law LeJEPA shows to be optimal; $M$ random directions test this only approximately. The gradients stay uniformly bounded~\cite{lejepa}.""")

# ---------------------------------------------------------------------- why a hierarchy
rep(r"""Two arguments support the factorisation~\cite{hplwm,ssshashua}.""", r"""Two arguments motivate the factorisation~\cite{hplwm,ssshashua}.""")
rep(r"""  \item \textbf{Horizon.} Planning regret in learned models grows with the horizon, so splitting one long horizon into per-level short
        horizons gives a smaller bound at each level.""",
    r"""  \item \textbf{Horizon.} Planning regret in learned models grows with the horizon, so a short per-level horizon has a smaller bound at
        that level. Summed over the levels, equal per-step errors give the undivided bound back: the gain requires the coarser levels
        to carry a smaller per-step error, which is an empirical question.""")
rep(r"""control profile, a flat learner must cover $|\mathcal P|\cdot|\mathcal M|$ combinations. A hierarchy that learns each level
        separately covers roughly $|\mathcal P|+|\mathcal M|$.""",
    r"""control profile, a flat learner must cover $|\mathcal P|\cdot|\mathcal M|$ combinations. A hierarchy that learns each level
        separately covers roughly $|\mathcal P|+|\mathcal M|$, provided places and manoeuvres combine independently.""")

# ---------------------------------------------------------------------- v8F-mini
rep(r"""The output is a 6\,s plan $\tau\in\Rset^{60\times2}$ in the ego frame, scored on its 30 knots at 0.2\,s.""",
    r"""The output is a 6\,s plan $\tau\in\Rset^{60\times2}$ in the ego frame, scored on its 30 knots at 0.2\,s. Several plans are possible,
one per top tactical hypothesis; the arm reported here emits one.""")
rep(r"""through shared hidden state. Each $f_\ell$ is a \emph{Reader}:""",
    r"""through shared hidden state. Each level reads the measured ego state of its own tick, and no level reads a state planned by another
level. Each $f_\ell$ is a \emph{Reader}:""")
rep(r"""The 26 constraint cells cover turn timing and geometry, longitudinal target speeds and reach times, the stop point, the lead gap and
time gap, and the speed band.""",
    r"""The 26 constraint cells cover turn timing and geometry, longitudinal target speeds and reach times, the stop point, the lead gap and
time gap, and the speed band. The code is the tactical action of \cref{sec:bands}: T emits goals and constraints, has no trajectory
head, and $\hat c$ conditions O. The route checkpoints of this arm sit at 30, 50 and 80\,m, with a median arrival of 3, 5 and 8\,s, so
they do not yet cover the strategic band in seconds; \cref{sec:bands} gives the extension.""")
rep(r"""  \item \textbf{Band limit.} With $k\le11$ the residual contains at most eleven half-cycles over 6\,s, below 1\,Hz: a smooth adjustment,
        while the prior carries the kinematics.""",
    r"""  \item \textbf{Band limit.} With $k\le11$ the cosine factor has at most eleven half-cycles over 6\,s, below 1\,Hz, and the linear ramp
        modulates it smoothly: a smooth adjustment, while the prior carries the kinematics.""")

# ---------------------------------------------------------------------- DAgger bound
rep(r"""on-policy aggregation (DAgger~\cite{dagger}) reduces it to order $\varepsilon T$.""",
    r"""on-policy aggregation (DAgger~\cite{dagger}) reduces it to order $\varepsilon T$ under that reduction's no-regret assumptions.""")

# ---------------------------------------------------------------------- evaluation
rep(r"""  \item \tier{T2}: cameras re-rendered at the car's own pose and time. In preparation, \cref{sec:frontier}.""",
    r"""  \item \tier{T2}: cameras re-rendered at the car's own pose and time, with replayed traffic, \cref{sec:frontier}.""")
rep(r"""The families are reported per family and never pooled.""",
    r"""The families are reported per family and never pooled. The strategic bars are paused by decision until the tactical and operative
layers are proven, so strategic rows here are measurements and not a pass.""")

# ---------------------------------------------------------------------- frontier
rep(r"""\paragraph{T2 re-perception (in preparation).} We are building a re-rendered evaluation tier on neural reconstructions of real drives:""",
    r"""\paragraph{T2 re-perception.} The re-rendered evaluation tier runs on neural reconstructions of real drives:""")
rep(r"""render domain gap, and both outcomes and their next levers are committed in advance.""",
    r"""render domain gap, and both outcomes and their next levers are committed in advance. First reads now exist on 48 scenes with replayed
traffic. An encoder adapted to the render gap improves speed error at both seeds, does not separate 6\,s error from the unadapted model at
either seed, and is not better than the floor on every row, so the launch row stays open.""")
rep(r"""T2 re-perception \textit{(in preparation)} & a decision-grade read of the launch row on re-rendered scenes; the longitudinal tier for every later arm \\""",
    r"""T2 re-perception \textit{(first reads, 48 scenes)} & a decision-grade read of the launch row on re-rendered scenes; the longitudinal tier for every later arm \\
World-model planner \textit{(built)} & top-down planning over three latent levels, the baseline for comparing other approaches \\
Band retraining \textit{(in training)} & world models, operative and tactical levels on the 2\,s, 6\,s and 30\,s bands \\
Strategic context labels \textit{(in production)} & per-frame country, period of day, illumination, weather, visibility, road surface \\""")
rep(r"""Scale size & from 7\,M toward the 300\,M 4B budget, with each level's predictor in its own latent \\""",
    r"""Scale size & from 7\,M toward the 4B budget of at most 800\,M, with each level's predictor in its own latent \\""")
rep(r"""  \item a \textbf{re-rendered simulation tier}, already in preparation;""", r"""  \item a \textbf{re-rendered simulation tier}, with first reads in hand;""")

open(W + "tanitad_4b_paper_v2.tex", "w", encoding="utf-8", newline="").write(s)
print("ok", len(s))

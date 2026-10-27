# v8F-mini: a three-level hierarchy over frozen visual features with a typed tactical interface, and a rigorous account of what does and does not move six-second driving error

**Status:** research report, v31 of 2026-10-27, results through repository tip `569a9d9b` (v30: `93aa1956`; branch `agent/arch-inf-20260803`; the code line references in §10 are at `a99ca5d6`). Open-loop results only (T0, T1 and the self-action T1-kin); **no closed-loop (T2) number exists** and no published baseline exists on this split. The PI's bar of "dominant driving results" is **not met** (§8.3).

**Evidence classes used throughout.** **[M]** MEASURED (ours, artifact named) · **[P]** PUBLISHED (library key; the PDF is banked, but descriptions here follow the library index entry, UNVERIFIED beyond it) · **[I]** INHERITED (another document, not re-verified by me) · **[E]** ESTIMATED · **[H]** HYPOTHESIS · **[U]** UNVERIFIED. Every result carries its **tier** (T0 teacher-forced / oracle; T1 self-action open loop; T2 closed loop, unprovisioned) and its **estimator** (§5.3). Paths starting `R/` mean `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/`.

---

## 1. Abstract

We study **v8F-mini**, a hierarchical driving model that factorises a 6 s plan into a **strategic** layer S (route checkpoints, about 1 Hz), a **tactical** layer T (a typed 60-dimensional code of goals, manoeuvre classes and 26 continuous constraints, about 2 Hz) and an **operative** layer O (the trajectory, 10 Hz), all reading frozen DINOv3-L features. O emits its trajectory as a *residual on a frozen learned kinematic prior* ("P10") in a ramped 12-term cosine basis, so a zero-initialised head reproduces the prior bit for bit. Layers are trained with *whole-code scheduled teacher forcing* ($p_{GT}$ from 1.0 to 0.2); an optional H-JEPA world-model auxiliary shapes the shared adapter. The model has 5.2 M (M1) or 7.2 M (M2b, with the auxiliary) trainable parameters and trains in about 4 h on one 8 GB GPU.

On a 141-episode, 2,399-window evaluation split (open loop, own held outputs), M2b improves 6 s ADE over the learned kinematic floor by 0.312 / 0.432 m at two training seeds (paired episode-cluster bootstrap, both separated, both beyond the pre-registered $2F = 0.2415$ m) **[M]**. The same arms are *worse* than that floor at 2 s (ADE 0.28–0.32 vs 0.222 m) and in acceleration MAE, and the strategic layer does not beat a nav-only kinematic baseline at 50 m on the banked numbers (§7.8). The world-model auxiliary has no effect (M2b − M1 = −0.071 / +0.030 m, not separated), and none of five pre-registered one-lever arms (loss weight, scheduled sampling, spatial tokens, history, data volume) clears the bar.

**A zero-training post-map closes the gap to the floor [M].** O's plan is blended with its own prior through 30 × 2 per-step weights, fitted on training windows with an acceleration term and no monotone constraint (HACC). The result is separated-better than the kinematic floor P10 on **every** row of a pre-registered dominance row at **both** seeds. That covers ADE2 (−0.011 / −0.008 m), ADE6 (−0.50 / −0.47 m), acceleration MAE (−0.022 / −0.040 m/s²), curvature, speed, heading, yaw rate, decision agreement and consistency; distance keeping is not separated, and nothing is separated-worse (open loop, own held outputs; §7.12). The gap it closes was localised first. H's acceleration error sat where the monotone blend weight ramped (2.2–2.6 s) and where O's own tail is rough.

The contribution we defend is **diagnostic**. Feeding O the ground-truth tactical code cuts 6 s ADE from about 2.1 to about 1.0 m (T0 oracle), so the interface is used and carries about 1.1 m of headroom, but (i) the headroom sits in the 26 continuous constraint cells *as a set*, (ii) the response of O to closing T's error is **convex** (a quarter of the way buys 8–16 % of the gain), (iii) T is **fit-limited**, not data-limited (halving the data moves it 6 %, not separated) and 88 % of its squared error is shared across seven trained arms, and (iv) O **cannot use a partly right code**: single-kind oracles make driving worse, and additivity is −0.30 / −0.55 against a 0.8 bar. We trace (iv) to the whole-code teacher forcing and pre-register the O-side repair (per-kind code mixing, `PREREG_V8F_MIXO`). Throughout, we report refutations, five retracted readings of our own (R63, R66, R69, R70, R71) and the variance structure that makes single-seed lever claims unreliable. We claim neither state of the art nor closed-loop driving. The dominance above is over the programme's own kinematic floor, in open loop. In a self-action open loop it becomes a margin, not dominance (next paragraph); T2, the closed-loop tier, is not provisioned.

**The margin survives the model's own motion [M].** In T1-kin, the ego state the model reads is rebuilt from its own executed motion at every 2 s replan, while the cameras stay logged. There, HACC beats the floor driven the same way by **0.83 / 0.79 m** of 6 s ADE at both seeds, and the margin is 0.79 / 0.71 m with the cameras frozen at the first frame. It is not dominance: curvature is worse at both seeds. About a third of that excess is millimetre noise near standstill. The rest is not a lateral defect but a launch crawl: HACC's own car is still slow where the human pulls away, and at equal own speed its curvature error equals the floor's (§7.15). On the executed tracks, the tactical manoeuvre agrees with the human's more often than the floor's, and the headway behind a lead is closer to the human's (both families NO WORSE, under a disclosed post-data amendment of one control; §7.15). And it does not need the future-derived speed bucket the corpus supplies. Rebuilt from the last 4 s of measured speed alone, the bucket keeps a margin of **0.77 / 0.73 m** at both seeds, and costs 0.06 m against the oracle one. With both open channels closed at once (the first frame held for 6 s, plus the past-only bucket), the margin is still **0.73 / 0.64 m** (§7.15).

**But the hierarchy is not yet shown to be what earns it [M].** Against its own capacity-matched FLAT control (O reads constants in place of S's checkpoints and T's code, keeps the route, and everything else is matched; `PREREG_V8F_FLAT`), M2b's ADE6 is better by only **0.036 / 0.003 m**, not separated, against a committed bar of 2F = 0.24 m. The outcome is **NOT ESTABLISHED**, and v8F is not validated against flat at 2F. The same interface carries 1.2 m when the code is the ground truth's (oracle 0.97 / 1.00 m against 2.18 / 2.05 m), so the channel works and the code O receives at inference does not help it. Descriptively, the hierarchy helps at junctions (−0.20 / −0.32 m) and costs on curves (+0.16 / +0.36 m). The margin over the kinematic floor above is therefore a claim against a prior, not yet against a learned flat planner. The committed next arm trains O on a mix of predicted and ground-truth codes (MIXO) and repeats this read (§7.15).

**On the model's own motion, the levels split [M].** In the self-action open loop, each head's output is scored against the human's labels at the nearest logged point of the route (`PREREG_V8F_T1_HEADS`).
* S's route checkpoints still beat a kinematic arc at 30 / 50 / 80 m, by 1.2 / 3.3 / 7.8 m and 1.3 / 3.5 / 8.8 m, separated at both seeds (S ADDS ROUTE).
* T's decisions DEGRADE. Lateral macro-F1 falls from 0.85 to 0.61 / 0.63, and longitudinal from 0.46 to 0.34 / 0.36. The turn classes lose half their F1.
* The drop holds where the cameras T saw are the very row it is scored on. So it is T reacting to its own ego state, not the replayed cameras.

Together with FLAT, this places the hierarchy's defect at the T level and its interface to O, not at S. The pre-registered fix was O-RECOVER (`PREREG_V8F_ORECOVER`): a fine-tune on ego states perturbed off the logged trajectory, against the same fine-tune unperturbed. It read **HARMS** at both seeds (§7.18), so the committed next was on-policy states in DAgger form at p = 0.1 (`PREREG_V8F_DAGGER`). That arm read **ROBUST** at both seeds (§7.19): T's lateral and longitudinal decisions on the model's own motion improve past every pre-registered bar, and the executed 6 s ADE falls by 0.54 / 0.40 m.

**Two further levers were read on 2026-10-26, and neither is dominant [M].**
* **OACC**, an acceleration-matching loss on O (seed 0): the acceleration error falls by 0.566 m/s², separated. But the ADE6 upper bound (+0.126 m) exceeds the +F/2 bar, and curvature is separated-worse than P10. The committed outcome is COSTS POSITION (§7.16).
* **HACC-MC**, a constant along weight (zero training): the launch crawl falls, and a 5× steeper ramp raises it, at both seeds. So the mechanism is real. But curvature at speed stays separated-worse than P10 at seed 0, and ADE6 rises by 0.14 / 0.21 m. The committed outcome is MECHANISM-NOT-ENOUGH (§7.17), whose committed next is O-RECOVER.
* **O-RECOVER**, a 3,000-step fine-tune on perturbed ego states against the same fine-tune unperturbed (two seeds): the committed outcome is HARMS (§7.18). T0 ADE6 rises by +0.110 / +0.085 m, above the +F/2 bar at both seeds, and the T1-kin ADE6 is separated-worse at seed 0. The arm does buy lateral skill on the model's own motion: heading, curvature and yaw-rate errors are separated-better at both seeds at T1-kin. The cost is T0 longitudinal error, separated-worse at both seeds. ORC is not adopted; the committed next is the DAgger arm at p = 0.1.
* **DAgger** (p = 0.1, two seeds): 10 % of training samples are replaced by the model's OWN rolled-out states, labelled from the human future, against the same 3,000-step fine-tune with p = 0. The committed outcome is **ROBUST** (§7.19), with all 176 controls passing. At T1-kin, T's lateral macro-F1 rises by +0.108 / +0.118 and its longitudinal macro-F1 by +0.055 / +0.081, separated and past bars that include the run-to-run replicate floor. The executed ADE6 falls by 0.54 / 0.40 m (2.28 against 2.82 m, and 2.46 against 2.86 m), and S's checkpoints beat a kinematic arc at every horizon. T0 ADE6 is unchanged at seed 0 and +0.048 m at seed 1, inside the +F/2 guard. It is the first v8F training lever to clear a T1 bar on both heads. DAG is adopted as the base, and its dominance over P10 and its FLAT repeat are pre-registered and running (`PREREG_V8F_DAG_REREAD`).
* **On its own motion, DAG beats the kinematic floor by 1.17 / 0.99 m of 6 s ADE (8–10 F), and is NOT dominant (§7.20).** Every family has a separated-better row, but accel, curvature, L2 and the launch crawl M1 are separated-worse against P10. DAG's tactical agreement is +0.19 / +0.19 over P10, and its S checkpoints halve their error against DAG-0. The cost DAgger adds is distance keeping: the lead-violation share doubles against its control (T1-kin, both seeds).
* **Against its capacity-matched FLAT control** at T1-kin, DAG is better by 0.60 / 0.35 m. That is a pass under a disclosed post-data control amendment, and the margin rides on DAgger (DAG-0 − FLAT is not separated), so FLAT-DAG is the attribution read (§7.20).
* **HACC re-fitted on DAG (zero training)** clears every longitudinal row against P10, leaving curvature, L2 and M1. It reads COSTS POSITION, because 6 s ADE rises by 0.13 m at seed 0, while it falls by 0.10 m at seed 1 (§7.21).
* **FLAT-DAG, the attribution read, does not establish the hierarchy's lead (§7.22).** The flat control is given the same DAgger recipe and read at T1-kin (self-action open loop; paired episode-cluster bootstrap, B = 2,000, 141 episodes). DAG's ADE6 margin over it falls from 0.596 / 0.351 m, against FLAT (B-2, §7.20), to 0.345 / 0.054 m. The PRIMARY bar (≤ −0.2415 m, separated, at both seeds) is met at seed 0 and missed at seed 1, and family rows are separated-worse at both seeds (DK-TG and DK-VIOL at seed 0, accel MAE at seed 1). The outcome is NOT ESTABLISHED. Own-state supervision helps the flat model too (FLAT-DAG − FLAT-DAG-0 ADE6 −0.152 / −0.191 m, separated), so most of B-2's margin was DAgger's.
* **DAG-OACC** (OACC's acceleration term inside DAG's fine-tune recipe, `--w-acc 49.6969`) **reads LONGITUDINAL FIXED and is the adopted base (§7.23).** Against DAG at T1-kin (same tier and estimator), accel MAE falls by 56 % / 58 % (2.90 → 1.27 and 3.53 → 1.49 m/s²; −1.6243 [−1.9766, −1.3236] and −2.0450 [−2.5628, −1.6227], separated), and ADE6 is kept (−0.0356 [−0.1258, +0.0423] and −0.1363 [−0.2057, −0.0731] m, against a bar of +0.0604 m on the upper end). Every longitudinal row is clean against P10 at both seeds, and ADE6 against P10 is −1.2012 / −1.1231 m. It is not DOMINANT: yaw-rate, curvature, L2 and the launch-crawl share M1 stay separated-worse against P10 at both seeds.
* **What is next.** The open gap is the crawl set. DAG-ORC re-based on DAG-OACC (`PREREG_V8F_DAG_ORC` Amendment 1, `6cd741b6`) is aimed at it, and SPEEDREAD (`PREREG_V8F_SPEEDREAD`) asks whether the speed information O needs is present in T's own input. Both are pre-registered and not yet read, and no number for either is reported here.

---

## 2. Introduction and problem statement

**Problem.** Given a short camera history, a navigation command with distance to the next manoeuvre, measured ego motion and a speed limit, predict the ego trajectory over the next 6 s so that it is close to what a human drove and consistent with the scene's constraints (speed limit, stop points, lead vehicles, turns). The programme (TanitAD) pursues a sub-300 M-parameter hierarchical latent world-model driver; v8F-mini is its smallest instance, trained on **cached frozen** features so that one lever can be moved per run on a single consumer GPU.

**Why a hierarchy, and what we test.** Driving decisions live at different time scales: a route (tens of seconds), a manoeuvre with its parameters (seconds), a control profile (tenths of a second). A model that exposes these as separate, *typed* intermediate variables can be supervised at each level, inspected, and intervened on. The question is not only "does it drive better" but **"does the lower level actually use the upper level's output, how much is left on the table, and where does the loss of information occur?"** Intervention on the interface (replace T's code by ground truth, by another window's code, or by a mixture) answers this directly; we call this *oracle-by-interface* analysis.

**Positioning.** The result is a measurement study of an interface in a small model, with an unusually strict evidence apparatus (pre-registered bars, seed floors, paired episode-cluster intervals, retraction log). It is **not** a leaderboard entry. The honest summary of the driving outcome is in §8.

**Contributions.**
1. A formal specification of the S/T/O factorisation, its conditioning sets, and a trajectory parameterisation that makes "zero residual = frozen kinematic prior" an exact identity (§4).
2. Oracle-by-interface experiments (own / GT / donor / per-group / per-kind / dose code) with a clear tier and estimator for each (§7.3–7.6).
3. A mechanism for the headline negative: whole-code teacher forcing makes the operative layer brittle to partly correct codes (§5.2, §7.6), with a pre-registered repair.
4. A variance accounting (episodes / training seeds / inference) that explains why five levers each moving 6 s ADE by less than 0.08 m cannot be told apart from noise on this rig (§5.3, §7.7).
5. An account of our own errors: five readings we published and then retracted, each with its root-cause class (§9).

---

## 3. Related work

*Descriptions follow the banked library index (`TanitAD Research Lab/Library/LIBRARY.md`); I did not re-read the PDFs for this report, so details beyond the index entry are UNVERIFIED.*

**Hierarchical latent world models.** H-JEPA (2610.06805) learns a hierarchy of action-conditioned JEPA world models, each level predicting further ahead in its own latent, with top-down subgoal planning; its library entry records a 3-level Visual AntMaze improvement from 18 % to 73 % success with less planner compute **[P]**. Hierarchical Planning with Latent World Models (2604.03208) reports two-level latent subgoals (70 % vs 0 % flat; a capacity-controlled table, flat 98 M at 35 % / 15 % vs hierarchical 94 M at 78 % / 61 % on Push-T) **[P]**. In driving, Orbis 2 (2607.15898) and Drive-HWM (2609.03572; slow and fast world models with ablations on NAVSIM) apply hierarchical world models, and S2Planner (2609.29813) is a multi-scale semantic planner on NAVSIM **[P]**. v8F-mini differs in using the hierarchy as a *supervised typed interface* (labels from the ego log), with the H-JEPA stack only as an auxiliary on a shared adapter.

**Why modularity can help.** Shalev-Shwartz and Shashua (1604.06915) show a sample-complexity separation favouring semantic-abstraction training over end-to-end, but, per the library note, as a *validation bound under an assumed approximate independence of sub-module failures*, not a constructed learning task **[P]**. Our results bear on exactly that assumption: T's and O's failures are *not* independent here (§7.6).

**Exposure bias and interface mismatch.** Scheduled sampling (1506.03099), DAgger (1011.0686), DART noise injection (1703.09327), causal confusion (1905.11979) and the limits of behaviour cloning (1904.08980) frame the train/inference mismatch between a consumer trained on ground truth and one fed a predecessor's output **[P]**. Our whole-code switch is a coarse form of scheduled sampling, and §5.2 shows its specific failure mode.

**Residual and command-conditioned planners.** ResAD (2510.08562) models trajectories as normalised residuals; conditional imitation learning (1710.02410) conditions on a navigation command; TransFuser (2205.15997), VAD (2303.12077), DiffusionDrive (2411.15139), Hydra-MDP (2406.06978) and ChauffeurNet (1812.03079) are representative end-to-end planners, evaluated on NAVSIM (2406.15349) or nuPlan (2403.04133) **[P]**. We do not compare to them: no published baseline exists on our split, and our evaluation is open loop.

**Representations and adaptation.** DINOv3 (2508.10104) supplies the frozen features; V-JEPA 2 (2506.09985) and LeJEPA (2511.08544, source of the SIGReg regulariser used inside H-JEPA) motivate the auxiliary; LoRA (2106.09685) is the adaptation tested in `PREREG_V8F_LORA` **[P]**.

---

## 4. Method

### 4.1 Inputs and notation

Time is discretised on a 0.2 s cache grid, $t \in \mathbb{Z}$. For a window ending at "now" $t$:

* **Scene** $z_{t'} \in \mathbb{R}^{40\times1024}$: frozen DINOv3-L/16 features of the 256×640 cylindrical front-wide frame, average-pooled to a 4×10 grid and stored in fp8; $h=4$ frames are read per layer (`hist 4`).
* **Ego** $f_t \in \mathbb{R}^8$: $(v,\,a,\,\kappa,\,v^2,\,va,\,v\kappa,\,v^2\kappa,\,a\kappa)$ with $\kappa$ the curvature from the *measured yaw rate* (no steering angle; §4.8).
* **Nav** $n_t$: a 3-way command one-hot, distance to the next manoeuvre ($d/100$, clipped to $[0,3]$) and a validity flag.
* **Limit** $\ell$: a per-clip max-speed one-hot over $\{30,50,100,120\}$ km/h plus an ok flag (a declared sidecar input, §9).
* **Measured odometry** $\mathrm{pose}_{t'}$ between tick frames (an ego-dynamics input).

The output is a 6 s plan $\tau \in \mathbb{R}^{60\times 2}$ on a 0.1 s grid in the ego frame (scored on its 30 knots at 0.2 s, hence "ADE@6 s").

### 4.2 The S / T / O decomposition and tick rates

Let $k \in \mathbb{R}^{3\times3}$ be **route checkpoints** $(x,y,\psi)$ at 30, 50 and 80 m of arc, $c\in\mathbb{R}^{60}$ the **tactical code** (§4.3), and $\tau$ the plan. By design S ticks at about 1 Hz, T at about 2 Hz, O at 10 Hz. On the 0.2 s cache grid the periods are not representable; the implementation uses **sample-and-hold with quantised ages**: T last ticked $a_T\in\{0,1,2\}$ steps ago ($<0.5$ s), S ticked $u\in\{0,\dots,4\}$ steps before T, so $a_S = a_T + u \le 6$ ($\le 1.2$ s). Evaluation uses `--ages cycle` (deterministic coverage of the cycle); `fresh` (all ages 0) is the optimistic variant and has **not** been banked **[U]**. With $t_T=t-a_T$ and $t_S=t-a_S$:

$$
\hat k = f_S\big(z_{t_S},\,f_{t_S},\,n_{t_S},\,\ell\big),\qquad
\hat c = f_T\big(z_{t_T},\,f_{t_T},\,n_{t_T},\,\ell,\;\Pi_{t_S\to t_T}\hat k\big),
$$
$$
\hat\tau = f_O\big(z_{t},\,f_{t},\,n_{t},\,\ell,\;\Pi_{t_S\to t}\hat k,\;\hat c,\;(a_T,a_S)\big),
$$

where $\Pi_{a\to b}$ re-expresses held checkpoints from frame $a$ to frame $b$ with the measured odometry (`transform_ckpt`). **Conditioning sets are strictly nested**: $\mathrm{in}(S)\subset\mathrm{in}(T)\subset\mathrm{in}(O)$; information flows upward only through *held outputs*, never through shared hidden state. In the `navonly` control S and T lose the scene path; O always keeps it.

Each $f_\ell$ is a **Reader**: $n_q=4$ learned queries (width 256, 4 heads) cross-attend to the memory $[\text{conditioning tokens};\text{scene tokens}]$ through a pre-norm transformer decoder of depth 1 (S), 1 (T) and 2 (O), followed by an MLP head. Trainable parameters (M2b, `summary.json`): adapter 275,712; conditioning 25,088; S 1,321,737; T 1,463,868; O 2,120,216; world-model auxiliary 2,007,874; **total 7,214,495** (M1: 5,206,621) **[M]**.

![Figure 1. Schematic of v8F-mini (no data): inputs, the three layers, the typed interfaces, the frozen P10 prior, the training-only label side and the H-JEPA auxiliary.](fig_architecture.png)

### 4.3 The tactical code as an interface

$$
c \;=\; \big[\;\underbrace{\sigma(g)}_{22\ \text{goals}}\;\big|\;\underbrace{\mathrm{softmax}(\lambda_{lat})}_{5}\;\big|\;\underbrace{\mathrm{softmax}(\lambda_{lon})}_{7}\;\big|\;\underbrace{\tilde\gamma}_{26\ \text{constraints}}\;\big]\in\mathbb{R}^{60},
\qquad \tilde\gamma = (\gamma-\mu)/\sigma_{\mathrm{cons}} .
$$

The 26 constraint cells (turn timing, geometry and angles, longitudinal target speeds and reach times, stop point, lead gap and time gap, speed band) are regressed in standardised units. The ground-truth code $c^\ast$ uses the v9 labels (built offline from the ego log and 3-D agent tracks): goal targets with ignore-weights set to 0.5, lateral/longitudinal *allowed-set* indicators normalised to a distribution, and constraint cells with absent entries written as 0 (`_code_gt`). T's loss masks absent cells, so under teacher forcing O sees $0$ in an absent slot, while at inference it sees T's unsupervised value there (§7.5).

**Information-theoretic reading, with its limits.** Write $Y$ for the human future, $X$ for the inputs, $C=f_T(X,\hat K)$. The information-bottleneck Lagrangian $\min_{p(c|x)} I(X;C)-\beta\,I(C;Y)$ describes the trade a *minimal sufficient* interface would make. Our T is not trained on it: it minimises a supervised surrogate against $C^\ast=g(Y)$ (a deterministic function of the future built from labels), and since $C^\ast$ is a function of $Y$, data processing gives $I(C;Y)\ge I(C;C^\ast)$, so maximising agreement with $C^\ast$ is a lower-bound proxy for maximising $I(C;Y)$. Compression is only *implicit* (60 bounded dimensions). Because O also reads the scene, the code is **not** a bottleneck in the strict sense; it is a *typed side channel* whose usage we can measure (§7.3). The oracle experiments measure the **sufficiency gap**: with $C^\ast$ substituted, 6 s ADE is still about 1.0 m, so the code is informative but not sufficient.

### 4.4 Trajectory parameterisation

Let $P=P_{10}(f_t,b)\in\mathbb{R}^{30\times2}$ be the **frozen** P10 prior: $P_{10}(f,b)=b+\mathrm{MLP}\big((f-x_m)/x_s\big)+y_m$, where $b$ is the `kd_x` base (a damped path re-timed by constant-acceleration distance) and the 8-feature MLP is fitted once on 4,512 clips (`requires_grad=False`). Let $\mathrm{up}:\mathbb{R}^{30\times2}\to\mathbb{R}^{60\times2}$ interpolate to 0.1 s: odd indices are the knots, even indices their midpoints (`upsample_prior`). The plan is

$$
\boxed{\;\tau \;=\; \mathrm{up}\!\big(P_{10}(f_t,b)\big)\;+\;B\,c_{\mathrm{res}},\qquad
B_{ik}=\frac{i+1}{N}\cos\!\Big(\frac{k\pi\,(i+\tfrac12)}{N}\Big),\quad N=60,\; i=0..59,\; k=0..11,\;}
$$

with $c_{\mathrm{res}} = s\cdot \mathrm{MLP}(h_O)\in\mathbb{R}^{12\times2}$, $s=2$ m, and the **last layer of the MLP zero-initialised**. Properties (each pinned by a unit test with a mutation check **[M]**):

1. **Identity at initialisation.** $c_{\mathrm{res}}=0\Rightarrow\tau=\mathrm{up}(P_{10})$ exactly; the 6 s ADE of the untrained model *is* the prior's.
2. **Ramp.** $|B_{ik}|\le (i+1)/N$, so the residual vanishes linearly as $\tau\to0$ and the plan starts at the prior's origin.
3. **Band limit.** With $k\le11$ the basis contains at most 11 half-cycles over 6 s ($\approx0.92$ Hz), a smooth correction; the prior carries the kinematics.

This is a "residual on a floor" design: whatever O learns is *measured against* the floor it must beat, which is what the K-1 test does (§7.1).

### 4.5 Losses

With $\mathcal{L}_{ck}$ the L1 loss on checkpoints (positions scaled by 10 m, heading by 0.5 rad, validity-masked), $\mathcal{L}_{goal}$ a masked BCE, $\mathcal{L}_{lat},\mathcal{L}_{lon}$ the **partial-label** cross-entropies $-\log\sum_{c\in\mathcal{A}}p_c$ (rows with an empty allowed set are masked), $\mathcal{L}_{cons}$ a Huber loss ($\delta=1$ SD) over *present* cells, and $\mathcal{L}_{ADE},\mathcal{L}_{FDE}$ on the 30 knots:

$$
\mathcal{L}= w_{ck}\mathcal{L}_{ck}+w_{tac}\Big(\mathcal{L}_{goal}+\tfrac12(\mathcal{L}_{lat}+\mathcal{L}_{lon})+w_{cons}\mathcal{L}_{cons}\Big)+w_{traj}\Big(\tfrac{\mathcal{L}_{ADE}}{3}+\tfrac{\mathcal{L}_{FDE}}{12}\Big)
+w_{pen}\!\sum_{r}\rho_r+w'_{pen}\!\sum_r\rho'_r+w_{wm}\mathcal{L}_{wm}.
$$

Defaults: $w_{ck}=w_{tac}=w_{traj}=1$, $w_{cons}=0.5$, $w_{pen}=0.5$, $w'_{pen}=0.25$, $w_{wm}=0.1$ (M2 only). The penalties $\rho_r$, $r\in\{\mathrm{SPEED},\mathrm{STOP},\mathrm{TURN},\mathrm{LEAD}\}$ are differentiable hinges in the rule's own units against the GT constraints; $\rho'_r$ are the same against T's own detached predictions. A hinge is positive **iff** the frozen numpy rule of `taniteval/v8f_consistency.py` reports a violation (400 random boundary-heavy cases, four mutations RED **[M]**). Training: batch 64, 19,164 steps (four passes over 306,566 windows), `--seed` the only difference between replicates.

### 4.6 Whole-code scheduled teacher forcing

Let $p(s)=1-0.8\,\min(s/S,1)$ for step $s$ of $S=19{,}164$. Per training sample draw independent $\xi_k,\xi_c\sim\mathrm{Bernoulli}(p(s))$:

$$
\tilde k=\xi_k\,k^\ast+(1-\xi_k)\,\hat k\ \ (\text{where the label is valid}),\qquad
\tilde c=\xi_c\,c^\ast+(1-\xi_c)\,\hat c ,
$$

and T reads $\tilde k$, O reads $(\tilde k,\tilde c)$. The switch is **per sample over the whole code**: one scalar $\xi_c$ for all 60 cells (`v8f_mini.py:637` at the tip; `:627` for the checkpoints). By default the predicted branch is **not** detached, so O's gradient reaches T and S when they are read (`detach_upper=False`). At inference $\xi\equiv0$ and the label dictionary is never touched.

**Rationale (exposure bias).** At $p=1$ O trains on $\delta_{c^\ast(x)}$, but at test time it is fed $\hat c(x)\neq c^\ast(x)$: a covariate shift whose size is exactly T's error. Scheduled sampling (1506.03099) anneals toward the inference distribution. The *granularity* of the switch, whole code rather than per cell, is the design choice §5.2 shows to be the weak point.

### 4.7 The H-JEPA auxiliary

The auxiliary (`WMAux`, arms M2a/M2b) takes the mean-pooled adapter token per cached frame, projects it to $D=128$ as the level-1 latent $z^1$, and builds levels 2 and 3 at strides $(2,2)$, i.e. 0.2 / 0.4 / 0.8 s, with causal predictors with adaptive-LayerNorm action conditioning on the (scaled) acceleration and yaw-rate channels. Following H-JEPA (2610.06805, Eqs. 3 and 8) the loss is

$$
\mathcal{L}_{wm}=\sum_{\ell=1}^{3}\Big[\mathcal{L}^{\ell}_{pred}+\beta\,\mathrm{SIGReg}(Z^{\ell})+\gamma\,\mathcal{L}^{\ell}_{idm}\Big],\qquad (\beta,\gamma)=(0.08,\,100)\ \text{in M2b},\ \gamma=0\ \text{in M2a},
$$

with no stop-gradient into the adapter, so the auxiliary *shapes the shared representation* but has no inference path. We note a magnitude fact **[M]** (`R/RESULT_V8F_CZ.md`): at M2b's last step the world-model term is 87 % of the total loss value ($0.1\times162.36$ of 18.60) while T's constraint term is 0.4 %.

### 4.8 Binding design constraints and the key guard

The PI's rulings bind the design, and each is enforced by something checkable:

| constraint | statement | enforcement |
|---|---|---|
| **Labels may use the ego future; inference may not** | PI 2026-08-03 | `INFERENCE_INPUT_KEYS`, a 14-key allow-list: `feats, frame_ok, p10f, act, nav, pose, limit, limit_ok, base, a_t, a_s, pose_now, pose_t, pose_s` |
| **No situation-classifier output in the goal path** (goal ⟂ classifier) | PI 2026-08-03 | `assert_no_situation_classifier_input`: an AST scan of the module's *code* for the tokens `situation, sitclf, scenario_cls, head_img`; `forward_infer` takes exactly `(self, batch)`; keys actually read ⊆ allow-list and none starts `lab` |
| **Nav is an input at training and inference** | PI, repeated | `nav` in the allow-list; the nav-shuffle control (§7.8) shows it is used |
| **No steering angle at any model input** | `D-PI-YAWRATE-1` (PI 2026-10-17) | `--kin-mode yaw`: κ from measured yaw rate; `check_p10_units` refuses a steer-unit prior under yaw features |

The **key guard** (`_KeyGuard`) is a `dict` subclass recording every `__getitem__` and `__contains__` during `forward_infer`; the test then asserts *read ⊆ allowed*, and a deliberately peeking model is caught. Two caveats we state ourselves: it intercepts only those two methods (I found no `.get/.items/.values` on the batch in the forward path by grep, but the guard is a proxy, not a proof), and it checks **which keys** are read, not **what the values were built from** (e.g. `limit` is a per-clip sidecar; `nav` is derived from the ego's own recorded future on this corpus, §9). "Vision-only inference" in the PI's 2026-08-03 sense governed the *scenario classifier*; for the planner, measured ego motion and nav are admitted inputs by explicit ruling.

---

## 5. Mathematical background

### 5.1 Why a hierarchy factorises the trajectory posterior

By the chain rule, with route variable $K$ and tactical variable $C$,
$$
p(y\mid x)=\sum_{k,c}p(k\mid x)\,p(c\mid x,k)\,p(y\mid x,k,c).
$$
This is an identity, so factorisation alone buys nothing. The benefit is **supervision and conditioning structure**: the labels $(k^\ast,c^\ast)$ are derived offline from the log, so the factorised model $q_S(k|x)\,q_T(c|x,k)\,q_O(y|x,k,c)$ is fitted by maximum likelihood on the *augmented* tuples $(x,k^\ast,c^\ast,y)$, each factor against its own target. Inference is a **plug-in** composition $\hat y=\mu_O(x,\hat k,\hat c)$, $\hat k=\mu_S(x)$, $\hat c=\mu_T(x,\hat k)$ (no marginalisation, no sampling). If $\mu_O$ is $L_k$- and $L_c$-Lipschitz in its code arguments, the triangle inequality gives

$$
\|\hat y-y\|\;\le\;\underbrace{\|\mu_O(x,k^\ast,c^\ast)-y\|}_{\text{O under the GT code (oracle)}}\;+\;L_c\,\|\hat c-c^\ast\|\;+\;L_k\,\|\hat k-k^\ast\| ,
$$

so the oracle ADE is the first term and the measured gap (own code minus GT code, 1.20 / 1.05 m at M2b) is the propagated interface error. A **linear-response** model predicts that moving the code a fraction $\alpha$ of the way to the GT buys a fraction $\alpha$ of the gap. Define, on the 26 constraint cells only, $\hat c_\alpha=\hat c+\alpha(c^\ast-\hat c)$ and

$$
r(\alpha)=\frac{\mathrm{ADE}(\hat c)-\mathrm{ADE}(\hat c_\alpha)}{\mathrm{ADE}(\hat c)-\mathrm{ADE}(\hat c_1)} .
$$

Linear response is $r(\alpha)=\alpha$. We measure $r(0.25)=0.08$–$0.16$, $r(0.5)=0.20$–$0.36$, $r(0.75)=0.48$–$0.73$ (§7.5): **convex**, so $L_c$ is not a uniform constant; O is nearly flat near T's own code and steep near the GT code. This is the empirical content behind "the gap is convex in T's accuracy".

Why modularity could also help sample complexity is the subject of 1604.06915, whose separation requires the independence of module failures **[P]**. We have **no flat control** (an O-only model with the same data and budget), so "the hierarchy is what helps" is **[H]**, not tested (§9).

### 5.2 Interface mismatch and exposure bias, as exposed by KIND

For a deterministic T, at each window $x$ the training distribution of O's code input is the two-point mixture $p\,\delta_{c^\ast(x)}+(1-p)\,\delta_{\hat c(x)}$ on **whole** vectors. Take any strict subset $u\subset\{1..60\}$ and the partial code $c_u=(c^\ast_u,\hat c_{\bar u})$. Unless T's code already equals the GT on $u$ or on $\bar u$, $c_u$ is at positive distance from both support points, so a network that fits the two points is **unconstrained** at $c_u$: the training loss says nothing about its output. Any "single-kind oracle" ($c_u$ with $u$ = one kind of cell) therefore measures **extrapolation error**, not the marginal value of $u$; its gain is a *lower bound*, and it cannot rank T-side levers. In symbols, with $\mathrm{gain}_k=\mathrm{ADE}_6(\hat c)-\mathrm{ADE}_6(c_{u_k})$ and additivity $A=\sum_k\mathrm{gain}_k/\mathrm{gain}_{\cup}$: a response that is additive in kinds gives $A\approx1$; measured $A=-0.30/-0.55$ (bar 0.8) means the response is not even monotone in the partial dose.

The remedy tested next (`PREREG_V8F_MIXO`) is to widen the training support: draw an independent $\xi_u$ per **unit** $u\in\{\text{6 kinds, decisions}\}$ at the same marginal $p(s)$,
$$
\tilde c_u=\xi_u\,c^\ast_u+(1-\xi_u)\,\hat c_u,\qquad \xi_u\stackrel{iid}{\sim}\mathrm{Bernoulli}(p(s)),
$$
so each cell's marginal GT fraction equals the baseline's and only the joint changes ($2^{7}$ corners per window instead of 2). Analytic identities, required before launch: at $p=1$ it equals the whole-code switch; at $p=0$ both equal $\hat c$; at $p=0.5$ the per-sample GT fraction must be strictly partial (a mask giving only all-or-nothing "is a switch, not a mix").

A corollary recorded as R63: a **frozen-O intervention cannot size a training change**. Retraining T moves the code distribution and O co-adapts to it; only whole-code oracles are clean, because O was trained at both endpoints.

### 5.3 Estimator theory: three variances, one interval

**Setup.** Windows $w$ are grouped into $M=141$ episodes. For arms $a,b$ let $d_a(w)$ be the 6 s ADE of window $w$; the point estimate is the **full-set** mean difference $\hat\Delta=\frac1N\sum_w\big[d_a(w)-d_b(w)\big]$. Windows within an episode are strongly dependent, so the independent unit is the episode.

**Paired episode-cluster bootstrap** (`taniteval/taniteval/ci.py`, `paired_episode_cluster_bootstrap`). For $j=1..B$ ($B=2000$) draw $M$ episodes with replacement, recompute $\hat\Delta^{(j)}$ on the union of their windows (the *same* draw for both arms), and report the 2.5 / 97.5 percentiles. `separated` holds iff 0 lies outside. Pairing cancels the shared per-window difficulty: $\mathrm{Var}(d_a-d_b)=\mathrm{Var}\,d_a+\mathrm{Var}\,d_b-2\,\mathrm{Cov}$, which a quadrature of two single-arm intervals wrongly sets to zero. **[M]** Single-arm ADE6 half-widths are about 0.17–0.23 m (M2b s0: 2.1753 [2.0071, 2.3568]; P10: 2.487 [2.261, 2.714]) whereas the paired lever-arm half-widths are 0.08–0.12 m. The programme's older "8-split jackknife" was neither a jackknife nor a valid SE and also biased the point estimate (`overlapping_holdout_se`); no number in this report depends on it **[I]**.

**What the interval answers.** Let $\theta$ be the training randomness (seed: initialisation, batch order, teacher-forcing draws), $\omega$ inference randomness and $E$ the evaluation-episode draw. By the law of total variance,
$$
\mathrm{Var}(\hat\Delta)=\underbrace{\mathrm{Var}_\theta\,\mathbb{E}[\hat\Delta\mid\theta]}_{V_\theta:\ \text{training seeds}}
+\underbrace{\mathbb{E}_\theta\mathrm{Var}_\omega\,\mathbb{E}_E[\hat\Delta\mid\theta,\omega]}_{V_\omega:\ \text{inference}}
+\underbrace{\mathbb{E}_{\theta,\omega}\mathrm{Var}_E[\hat\Delta\mid\theta,\omega]}_{V_E:\ \text{episodes}} .
$$
The bootstrap estimates **only $V_E$**, conditional on the two trained models. A "separated" interval is therefore necessary, not sufficient, for a *lever* claim. Three questions, three sources:

| question | variance | estimated by | v8F-mini value |
|---|---|---|---|
| would another draw of **episodes** say this? | $V_E$ | the paired bootstrap | paired half-width 0.08–0.12 m **[M]** |
| would another **training run** say this? | $V_\theta$ | $F=\lvert\Delta_{s_0}-\Delta_{s_1}\rvert$ of one arm at two seeds | $F_{M2b}=0.1207$, $F_{M1}=0.2222$ m **[M]** |
| would another **inference run** say this? | $V_\omega$ | re-running one checkpoint | **0**: the forward pass is deterministic; four checkpoints re-evaluate to the banked eval with max per-window deviation 0.0 m **[M]** |

For a sampling planner the third term is not zero (the programme's refav1 iCEM rig has a seed floor of about 0.30 m ADE, with inference and training seeds not separated in the source **[I]**); v8F-mini has no such term. `R/raw/2026-10-20-tac/tac_read.json` (`checks.*.repro_max_dev_m`).

**The floor $F$ and the bar $2F$.** With two seeds, $\hat s=F/\sqrt2$ and $\hat s^{2}/\sigma^{2}\sim\chi^2_{1}$, so the 95 % interval for the seed standard deviation is $[\hat s/\sqrt{5.024},\ \hat s/\sqrt{0.000982}]\approx[0.45\,\hat s,\ 32\,\hat s]$ (a factor of about 70 **[E]**, analytic). With three seeds it narrows to $[0.52\,\hat s,\ 6.3\,\hat s]$. So **$2F\approx2.8\hat s$ is a pre-registered guard-rail, not a calibrated test**: the pre-registrations require *both* separation *and* $|\Delta|>2F$ at *both* seeds, and the cheapest next experiment for any lever is seeds, not windows. The correct floor depends on the comparison (R66): for a lever arm compared with one baseline draw (M2b seed 0) the relevant scale is the raw seed gap $F$; for a *seed-paired* contrast (M2b − M1 at the same seed) it is the **swing** of that contrast between seeds, 0.1015 m here (`R/raw/2026-10-25-ensemble/paired_swing.json`). A replicate arm on a tiny rig produced "separated" differences on 9.5 % of cells with zero levers moved, about twice the nominal 5 % **[I]** (`H-ESTIM-SEED-1`, corrected 2026-09-26), which is why a single separated interval from a one-seed arm is never treated as proof.

---

## 6. Experimental protocol

**Corpus and parity.** Data are PhysicalAI-Autonomous-Vehicles clips (ego log and front-wide 120° camera). **Train:** 4,512 clips (the 4,572-clip v9-labelled release, md5 `83e30bd3…`, minus a 60-clip holdout `--exclude 540:600`), 306,566 windows, train sha256 `37f4f9ee468e0872…`. **Eval-141:** 141 held-out v7.2 episodes, **2,399 windows**, disjoint from training by clip id and by recording session, evaluated with `kin_mode yaw`. Two eval episodes (34 windows) have no v9 row; their tactical/strategic cells are masked and the oracle falls back to T's own code there. ⚠️ This is box 2's `cache_full`, **not** the programme's parity corpus (`physicalai-train-e438721ae894`), so v8F-mini is not comparable with the older parity arms **[M]** (`MODEL_REGISTRY.md` §17).

**Tiers.** `EVAL_DOCTRINE.md` (amended 2026-09-02) makes T0 and T1 both *open loop*: T0 consumes recorded or oracle future quantities; T1 consumes the model's own outputs with the ego data still arriving from the recording; **T2, the only closed loop, is not provisioned, so no closed-loop number exists**. The v8F-mini own-code eval stamps `T0-open-loop-own-held-outputs`; the verdict header calls it "T1-class open loop". I follow the registry's stamp: *own-code numbers are open-loop with the model's own held outputs; every oracle is T0 by construction and is "never driving performance"*. `D-V8F-T1KIN-1` (a T1-kin replan-and-execute loop) is built and CPU-tested but **not run on real cameras** **[M]**.

**The four metric families** (binding, 2026-08-02; reported per family, never pooled): LONGITUDINAL (speed, along-track, acceleration, distance keeping), LATERAL (heading, yaw rate, curvature, cross-track), TACTICAL (decision accuracy and macro-F1), STRATEGIC (checkpoint error; un-paused for v8F-mini by PI ruling 2026-10-17), plus CONSISTENCY with the frozen SPEED/STOP/TURN/LEAD rules (the GT-plan control violates 7.9 % of windows, mostly the clip-level speed limit, so that is the instrument floor). Distance keeping was first reported UNAVAILABLE and was later read post hoc (`R/RESULT_V8F_DK.md`).

**Pre-registration discipline.** Each arm has a `PREREG_V8F_*.md` committed before its data, with *both outcomes* mapped to a next step, a primary bar ($\Delta\mathrm{ADE}_6<-2F$, separated, at both seeds), a mechanism read, and refusals (no launch without a launch-gate PASS bound to commit, argv and data hashes; identity and reachability checks, each with a deliberate-regression arm that must fail). Seed 0 decides whether seed 1 trains (a failing seed 0 stops the arm). Rule Zero (a refutation is a waypoint, not a deliverable) governs the sequencing.

---

## 7. Results

All numbers are ADE@6 s in metres unless stated; "sep." = paired interval excludes 0; $F_{M2b}=0.1207$, $2F_{M2b}=0.2415$; $F_{M1}=0.2222$, $2F_{M1}=0.4444$. Raw data for the figures are copied under `data/` with checksums.

### 7.1 H-MINI-K-1 and the floors (M1, M2b × 2 seeds)

**[M]; own code, open loop; paired episode-cluster bootstrap (B=2000, 141 episodes); source `R/raw/full/eval/verdict.json`, `floors_report.json`, `RESULT_FULL.md`.**

| run | ADE6 | ADE6 − P10 (paired) | ADE2 |
|---|---|---|---|
| constant velocity | 4.128 | | 0.565 |
| `damp50` / `kd_x` (yaw) | 2.936 / 2.944 | | 0.366 / 0.299 |
| **P10-yaw-4512 (frozen prior)** | **2.487** [2.261, 2.714] | — | **0.222** [0.201, 0.249] |
| M1 seed 0 / 1 | 2.247 / 2.024 | −0.240 [−0.420, −0.058] / −0.463 [−0.594, −0.337] | 0.319 / 0.281 |
| M2b seed 0 / 1 | 2.175 / 2.055 | −0.312 [−0.464, −0.166] / −0.432 [−0.565, −0.296] | 0.316 / 0.291 |

![Figure 2. ADE@6 s by arm under three codes (own, GT = T0 oracle, another window's code), and the K-1 paired deltas against the 2F bars.](fig_oracle_headroom.png)

**Verdicts as committed.** **K-1: PASS on M2b** (both seeds sep. and beyond $2F_{M2b}$; a 12.5–17.4 % lower 6 s error than the learned kinematic floor). **M1 fails its own floor** (seed 0: $-0.240$ vs $2F_{M1}=0.4444$). PASS iff at least one of two arms passes (stated multiplicity, no selection on eval-141). Two caveats belong next to the pass: the reference is a *learned kinematic prior*, not a published planner, and **at 2 s all four runs are worse than P10's point estimate** (ADE2 0.281–0.319 vs 0.222; single-arm intervals overlap only marginally for M1 seed 1; a paired ADE2 was not banked **[U]**).

### 7.2 H-MINI-WM-1: the world-model auxiliary is not a lever

M2b − M1 at 6 s is **−0.0713** [−0.1548, +0.0100] (seed 0) and **+0.0302** [−0.0244, +0.0846] (seed 1): neither separated and seed 1 has the wrong sign, so WM-1 **FAILS under any floor** (`verdict.json` `WM1`). Rung β (the same world model as a *planner* over a 117-candidate fan) **failed** too: β − M2 = +2.084 [+1.648, +2.553] / +2.225 m, much worse, and even the oracle-goal ranking is 1.19 m worse than M2 **[I]** (`H-V8F-BETA-1`, `R/…/2026-10-18-v8f-beta/`, not re-read). Context (§4.7): the auxiliary dominates the loss value by two orders of magnitude without helping the plan.

### 7.3 Oracle headroom: O uses T's code, and a perfect code is worth about 1.1 m

**[M]; T0 for the GT arm, shuffle is own-code-with-donor; paired bootstrap; `R/raw/2026-10-20-tac/tac_read.json`.** Manipulation checks passed: each own-code rerun reproduces the banked eval with max per-window deviation 0.0 m; the shuffle changed T's code on 100 % of windows.

| checkpoint | own code | donor code (shuffle) | GT code (T0) | donor − own | GT − own (T0) |
|---|---|---|---|---|---|
| M1 s0 | 2.2466 | 8.6507 | 0.9506 | +6.404 [+5.214, +7.725] | −1.296 [−1.451, −1.155] |
| M1 s1 | 2.0244 | 7.3285 | 0.9614 | +5.304 [+4.190, +6.473] | −1.063 [−1.179, −0.949] |
| M2b s0 | 2.1753 | 7.7118 | 0.9709 | +5.537 [+4.340, +6.856] | −1.205 [−1.335, −1.080] |
| M2b s1 | 2.0546 | 6.4743 | 1.0037 | +4.420 [+3.364, +5.550] | −1.051 [−1.166, −0.939] |

**TAC-USED: USED** (all four): the operative layer is steered by the tactical layer; the hierarchy functions end to end. **TAC-HEADROOM: LARGE** (about 1.05–1.30 m against $2F$ of 0.24 / 0.44 m). The headroom is mostly longitudinal (along-track −1.0 to −1.2 m, speed MAE −0.43 to −0.58 m/s at M2b). *This is a diagnostic, never a driving result:* the GT code contains quantities computed from the future (target speed, end speed, reach distance, lead gap), so 1.0–1.3 m is an **upper bound** on what any tactical head could buy.

### 7.4 Where the headroom lives: the constraint set, jointly

**[M]; T0; per-group oracle (GT in one group, T's own elsewhere); `R/raw/2026-10-21-tacgrp/tac_groups_read.json`.**

![Figure 3. Per-group oracle at M2b (ADE6 with GT in this group minus own code; negative is better). The decisions together buy 0.08–0.14 m; the 26 constraint cells together buy 0.92–1.08 m; two of the three constraint groups make driving worse when made GT in isolation.](fig_group_oracle.png)

* The **categorical decisions** (goals, lateral and longitudinal classes, 34 dims) buy only 0.14 / 0.08 m (7–12 % of the headroom): **the "better tactical classifier" lever is eliminated** as the main lever.
* The **26 constraint cells** carry 88–90 % (−1.076 / −0.923).
* But `latcons` (+0.32) and `stoplead` (+0.18 to +0.28) *hurt* alone, although the set they belong to carries 0.89 of the gain: the single-group rows are interactions, not contributions (§5.2). Leave-one-out: dropping `lonspeed`'s GT costs +1.99 / +1.74 m, **worse than T alone** (necessity above 1.6): the full oracle with T's own speed cells is a code O never trained on (`R/raw/2026-10-21-tacgrp2/`).

### 7.5 The response is convex; a refuted "mismatch" reading

**[M]; T0; `R/raw/2026-10-21-tacgrp2/tac_groups2_read.json`.** The dose experiment moves the constraint cells a fraction $\alpha$ of the way to the GT (§5.1):

![Figure 4. Share r(alpha) of the constraints-oracle gain realised when T's constraint cells are moved a fraction alpha toward the GT. The response is below the linear diagonal at every checkpoint.](fig_dose_response.png)

$r(0.25/0.50/0.75)=0.16/0.36/0.73$ (M2b s0), $0.08/0.21/0.48$ (M2b s1). The first quarter of the way buys 8–16 % of the gain; a **uniform 33–55 % cut of T's constraint RMSE** would be needed to clear $2F$ by accuracy alone **[E]** (`R/raw/2026-10-25-t-attrib/T_ATTRIB.md`).

**Retracted reading (R63).** The same files showed 92–95 % of T's squared deviation from the label's "0 = absent" convention sits in absent slots, which was read as a train/inference *mismatch error*. `PREREG_V8F_CZ`'s own zero-training gate (write 0 in absent slots) made driving **worse by +1.04 / +1.02 m** at M2b (sep.; **GATE FAIL**; `R/raw/2026-10-22-cz/cz_gate.json`). O had co-adapted to T's values in those slots; they are O's *input*, not error. *Class: an error metric judged against a convention its consumer does not use.*

### 7.6 KIND: O cannot use a partly right code

**[M]; T0; paired bootstrap, 141 episodes; `R/raw/2026-10-25-kind/run/kind_read.json` (30 evals; identity controls read exactly +0 [0, 0]).** `gain_k` = ADE6(own) − ADE6(GT in kind $k$ only); `necessity_k` = ADE6(GT except $k$) − ADE6(full GT); bar $2F=0.2415$.

![Figure 5. Per-kind oracle at M2b, both seeds. Left: gain from GT in one kind alone (time-to-event and geometry make driving worse). Right: necessity (leaving only the speed cells out of the full oracle costs 1.8–2.0 m).](fig_kind.png)

| kind | gain s0 | gain s1 | necessity s0 | necessity s1 |
|---|---|---|---|---|
| speed | **+0.270** [+0.220, +0.323] | +0.185 [+0.145, +0.226] | +1.985 | +1.765 |
| angle | +0.075 | +0.043 | +0.124 | +0.154 |
| time-to-event | **−0.237** [−0.340, −0.140] | **−0.315** [−0.434, −0.210] | +0.457 | +0.460 |
| geometry | **−0.434** [−0.580, −0.305] | **−0.421** [−0.577, −0.286] | +0.444 | +0.611 |
| time gap / lead gap | ≈ 0 (ns) | ≈ 0 (ns) | ≤ 0.017 | ≤ 0.017 |
| **union (26 cells)** | **+1.076** [+0.966, +1.195] | **+0.923** [+0.811, +1.035] | | |

**Outcome as committed: INTERACTIVE.** No kind clears $2F$ at both seeds (speed clears it at seed 0 only, which Amendment 2 says is not a carrier); additivity $A=-0.30$ [−0.48, −0.14] and $-0.55$ [−0.77, −0.36] against 0.8. Own-code ADE6 2.175 / 2.055; union 1.100 / 1.131; full GT 0.971 / 1.004. GT in the time-to-event cells *alone* makes O worse by 0.24–0.32 m; GT geometry alone worse by 0.42–0.43 m, almost all of it in the no-turn stratum (−0.54 / −0.51). Leaving only the speed cells out of the full GT code puts ADE6 at about 2.96 m, worse than T's own code. **O reads the cells jointly**; a code with some cells true and some T's is a distribution it never trained on (`v8f_mini.py:637`; earlier documents cite `:620-622` at older tips). This is the strongest evidence yet for the whole-code confound the PREREG named *before* the data (§5.2). One lead for later (stratum read, EXPLORATORY, 17 episodes): GT angle cells are worth +0.77 / +0.55 m inside the 140 bend windows.

### 7.7 One-lever training arms, all seed 0 against M2b seed 0 (all null)

**[M]; own code, open loop; paired bootstrap; each `R/RESULT_V8F_*.md` and its `*_read*.json`; bar: $\Delta\mathrm{ADE}_6<-0.2415$, separated, at both seeds. Seed 1 was never trained for any of these (seed 0 failed).**

![Figure 6. Every pre-registered one-lever arm and the world-model contrast against the 2F bar. The diamond is the same recipe at a different seed, with no lever moved.](fig_lever_arms.png)

| arm (ONE lever) | $\Delta$ADE6 [95 % CI] | mechanism read | label as committed |
|---|---|---|---|
| **WC** `--w-cons 4.0` (T's constraint loss ×8) | −0.0035 [−0.0795, +0.0850] | T present-slot RMSE −2.4 % (a separated but trivial move; the PREREG had no size bar, a flaw we recorded) | FAIL-MECHANISM-OK |
| **PGT** scheduled sampling 0.5 → 0.0 | +0.0294 [−0.0628, +0.149] | leak Λ 0.268 → 0.046 (**met**, sep.); tail did not follow | FAIL-NO-MECHANISM |
| **GRID** 8×20 scene tokens (4× tokens) | −0.0760 [−0.1746, +0.0158] | latcons RMSE **+1.2 % worse** (bar: fall ≥ 12.5 %) | FAIL-NO-MECHANISM |
| **HIST** 10 history frames (1.8 s) | −0.0774 [−0.1543, +0.0082] | latcons RMSE −0.4 % | FAIL-NO-MECHANISM |
| **DSCALE** half the clips (control) | +0.0389 [−0.0799, +0.1578] | T latcons RMSE +6.25 %, MSE Δ +0.0248 [−0.0053, +0.0569] ns | **NOT DATA-LIMITED** |
| **CZ** zero absent slots (gate only) | +1.04 / +1.02 (worse) | gate FAIL; no training | GATE FAIL |

*Side reads.* PGT removed the leak by making O lean *less* on T's code: the GT-code oracle ADE6 rose from 0.971 to 1.293 (+0.322 [+0.271, +0.374]), so headroom through T shrank. DSCALE: halving the data moves T by about 6 %, so doubling is predicted to move it about 6 % (log-linear, ≤ 2× extrapolation **[E]**), far from the roughly 25 % a $2F$ gain needs. **T-fit analysis (T_ATTRIB)** **[M]**: T's eval Huber is 0.1355 [0.1178, 0.1548] against a train Huber 0.1264 (ratio 1.07, the interval contains the train value), so **T is fit-limited, not generalisation-limited** (the first draft read "generalisation-limited" from a 4.3 % train-loss difference worth about one standard error; retracted as R70); 88 % of its squared error is common to seven trained arms (per-window Pearson 0.93–0.98), a 7-model ensemble cuts the block RMSE by only 5.4 %, the error concentrates in future-timing cells (time-to-event plus lead time-gap: 43 % of squared error on 20 % of cells; the worst 10 % of windows carry 55 %), and the ADE gap is diffuse (the worst-error decile holds 15–20 % of it). Every measured lever moved T's error by 0–6 %. **Seed noise dominates**: all lever points are inside 0.08 m of zero while the same recipe differs by 0.121 m between seeds (diamond), and an earlier "GRID and HIST both lower accel MAE" pattern was M2b_s0's own outlier draw (every seed-0 arm lowers it, including half the data: −0.183; R69).

![Figure 7. Three variances on one axis: paired episode-CI half-widths of the lever arms (blue), training-seed floors and the seed-paired swing (orange), the 2F bar, and inference variance (zero: deterministic).](fig_variances.png)

### 7.8 The four families (descriptive; no bar was registered on them)

**[M]; own code, open loop; `R/RESULT_FULL.md` (paired against P10 per `ra1_hjepa_eval.paired_family_block`; single-arm values below).**

| family | metric | P10 | M1 s0 / s1 | M2b s0 / s1 |
|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 0.980 | 1.123 / 0.925 | 1.067 / 0.911 |
| | along-track MAE (m) | 1.822 | 1.887 / 1.616 | 1.810 / 1.608 |
| | **acceleration MAE (m/s²)** | **0.638** | 1.247 / 0.952 | 1.206 / 0.931 |
| LATERAL | heading MAE (°) | 5.475 | 4.013 / 3.765 | 3.977 / 3.715 |
| | cross-track MAE (m) | 1.186 | 0.759 / 0.822 | 0.766 / 0.864 |
| | yaw-rate MAE (°/s) | 5.230 | 6.829 / 5.475 | 6.160 / 5.357 |
| TACTICAL | lateral decision accuracy (from path) | 0.773 | 0.848 / 0.830 | 0.846 / 0.829 |
| | longitudinal decision accuracy | 0.534 | 0.560 / 0.579 | 0.579 / 0.592 |
| | head macro-F1 lat / lon (majority 0.317 / 0.071) | — | 0.841 / 0.473; 0.832 / 0.532 | 0.816 / 0.499; 0.829 / 0.510 |
| STRATEGIC | checkpoint position error @50 m (m) | — | 8.095 / 8.094 | 8.012 / 7.938 |
| | @80 m (m) | — | 11.377 / 11.288 | 11.206 / 11.214 |
| CONSISTENCY | any-rule violation rate | 0.311 | 0.155 / 0.205 | 0.160 / 0.210 |

* **Longitudinal.** Acceleration MAE is worse than P10 in all four runs (+0.29 to +0.61 m/s², sep.): the one family defect that replicates. Distance keeping (`R/RESULT_V8F_DK.md`, descriptive): at 6 s M2b keeps +0.53 / +0.42 m more headway than P10 (sep.); at 2 s the learned arms close on the lead faster (min-TTC −1.76 / −1.19 s vs P10, sep.), which an exploratory T0 read traces to T's *lateral* constraint group leaking into O's longitudinal profile. An oracle per-window speed scale is worth 0.20–0.39 m while one global scale is worse: the remaining error is a per-window speed *decision* (`H-V8F-LON-1`, partly replicated).
* **Lateral.** Heading and cross-track beat P10 in all four runs (sep.); yaw-rate is worse at seed 0 only. A fixed Savitzky–Golay smoother on the residual removes the yaw-rate excess at ADE cost ≤ 0.0006 m (SMOOTH-PASS) but not the acceleration defect.
* **Tactical.** The lateral head reads the route: with nav shuffled, lateral macro-F1 collapses to 0.30–0.31. The longitudinal family is the weak one (kappa 0.33–0.37).
* **Strategic: the banked negative was the metric (`PREREG_V8F_SFRESH`, STALENESS ARTEFACT, 2026-10-25).** The nav-use control holds (nav-shuffled − M1: +4.25 m at 50 m and +9.86 m at 80 m, sep.). On the deployed cycle ages, learned S − the nav-only kinematic baseline reads +3.28 / +3.21 m at 50 m (M2b, S worse, sep.), because S's checkpoints come from a tick up to 1.2 s old and are scored against checkpoints defined from "now". Scored at its own tick (`--ages fresh`, a diagnostic, not the deployed schedule), **S beats the baseline: −2.43 [−3.22, −1.74] / −2.53 [−3.33, −1.79] m at 50 m and −5.11 / −5.10 m at 80 m, separated at both seeds** (fresh S error 2.30 / 2.20 m against the baseline's 4.73 m). Fresh minus cycle is −5.72 / −5.73 m at 50 m; the gap grows with the ego's travel since S ticked (0 to 13.8 m) while the fresh error stays at 2.0–2.6 m. Controls: the CPU cycle re-run reproduces the banked S error to ≤ 1e-4 m, and the 291 windows whose cycle ages are (0, 0) are bit-identical. ⇒ `H-MINI-STRAT-1` clause 1 passes on fresh ages. What the stale tick costs the *plan* is small: with S and T both ticking at "now", O's ADE6 improves by only −0.052 [−0.078, −0.029] / −0.065 [−0.090, −0.043] m (FDE6 −0.17 / −0.23 m; ADE2 unchanged), separated but below the seed floor $F$ = 0.12 m (`raw/2026-10-25-sfresh/sfresh_ade_paired.json`). This is the upper bound of the PREREG's committed lever, the deployed schedule (a faster S tick or a checkpoint re-anchored to "now"), so that lever ranks below the T-side and smoothness levers. MEASURED, `raw/2026-10-25-sfresh/READ.json` (`f547c11b`).

![Figure 8. Learned S minus the nav-only kinematic baseline at 50 and 80 m (positive = S worse), both seeds, M1 and M2b, and the nav-shuffled control, on the deployed cycle ages. Scored at S's own tick, S is better than the baseline (§7.8, SFRESH).](fig_strategic.png)

### 7.9 LoRA on the encoder: AR-2 infrastructure PASS (identity bit-for-bit), arm not trained

**[M]** (`R/raw/2026-10-25-lora-ar2/run3/`, box 2, RTX 4060). `PREREG_V8F_LORA` replaces the cached features by the frozen DINOv3-L/16 run on the fly with rank-32 LoRA ($W'=W+\tfrac{\alpha}{r}BA$, $\alpha=r=32$, on $q$ and $v$ of 24 blocks: $32\times(1024+1024)\times2\times24=3{,}145{,}728$ trainable parameters, $B$ zero-initialised). Its anchor is an **identity control**: with $B=0$ the on-the-fly encoder must reproduce the cache. Result: **rel-L2 exactly 0.0 and 100 % of tokens bitwise equal on all six clips; a one-constant pixel-scale mutant (`255→256`) reads 0.028 and is caught.** The 200-step smoke exposed a genuine defect that the identity test missed: run 2's *training path* probe read rel-L2 **2.41e-2** (75.6 % bitwise equal, max |Δ| 0.25, one fp8 e4m3 step), attributed to encoding nine frames in one call where the builder encodes eight; with builder-shaped calls (Amendment 3) run 3 reads **exactly 0.0**. Smoke: 1.96 s/step at batch 1, 1.44 GiB peak; **470.2 h per seed on box 2 [E]**, so the venue rule routes the arm to the A40 (the PI's provisioning). **No LoRA arm has been trained; no LoRA driving result exists.** The pre-registered prior for a primary pass is **LOW [E]** (§8.4).

### 7.10 MIXO: O learns to use a partly right code, but the plan does not improve

`PREREG_V8F_MIXO` (committed before any instrument or data) is the committed next step from KIND = INTERACTIVE. Arm X = M2b + `--code-mix kind` (§5.2), one lever; primary own-code ADE6 $X_s-\mathrm{M2b}_s\le-2F$, separated, at seed 0 (seed 1 only after a seed-0 pass); mechanism read: the dose response $r(0.5)$ of Figure 4 measured on X against B's 0.36; harm read: the GT-code oracle on X.

**Result, seed 0 [M] (T1-class open loop for the primary, T0 for the mechanism and harm reads; paired episode-cluster bootstrap, B = 2,000, 141 episodes): FAIL, O LEARNED.**

| read | X − B | bar | met |
|---|---|---|---|
| PRIMARY, own-code ADE6 | +0.022 [−0.059, +0.101] m | ≤ −0.2415, separated | **no** |
| D-M, r(0.5) | 0.644 against B's 0.365 (a rise of +0.28) | rise ≥ 0.10 names the branch | O LEARNED |
| HARM, GT-code oracle ADE6 | +0.400 [+0.347, +0.449] m | beside | — |

* **What it shows.** Trained on codes mixed per kind, O now uses a half-right code: moving the constraints half-way to the truth recovers 64 % of the full-truth gain, against 36 % for B. Its own-code plan is no better, and the full-truth oracle gets worse by 0.40 m, because O now sees fewer all-true codes in training.
* **The verdict was re-derived by hand**, from the per-window arrays and with my own bootstrap, because the reader's −2F boundary and its O-LEARNED branch had survived the mutation proof. The numbers agree with the reader to the bootstrap draw.
* **Seed 1 was not trained**, as committed. The committed next step is T-side: "T is too wrong where O reads it". It is pre-registered as `PREREG_V8F_KINDX`, a seed-0 screen of KIND on X that ranks which kind of T's code is now worth fixing.
* **KINDX (s0 screen, T0) [M]: CANDIDATE, speed.** On X, true speed cells alone (T's code elsewhere) cut ADE6 by +0.716 [+0.623, +0.818] m, against +0.270 on B; angle, time-to-event and geometry are each worth about 0.1 m, the lead cells nothing. On B the kinds interfered (additivity −0.30); on X they overlap (1.28), and geometry turned from harmful (−0.43 m) to useful (+0.11 m). This is what MIXO's mechanism predicted: once O consumes partial codes, the T-side lever becomes rankable, and speed is first by a factor of six. One seed makes no lever claim; the committed confirmation is X at seed 1 on box 2.
* **KINDX seed 1 (T0) [M]: CARRIER, speed.** gain_speed(X_s1) = +0.668 [+0.575, +0.761] m, against +0.716 at s0; no other kind clears 2F. B_s1's tte and geometry gains are negative (−0.315 / −0.421 m) where X's are positive at both seeds, so the second seed reproduces MIXO's mechanism: O trained on per-kind mixed codes can use a partly right code.
* **SPEEDAVG (T1-class) [M]: MIXED.** Averaging the two seeds' speed cells moves ADE6 by −0.041 m (s0, separated) and +0.019 m (s1), far from the −F bar. The reason is in the descriptives: the two independently trained T's make the **same** speed errors (error correlation 0.71–0.85 over the cells), so averaging removes only 3–12 % of the RMSE, as the pure-variance formula predicts for that correlation. T's speed error is systematic. Whether it is a head bias (recoverable by a train-fitted calibration) or information T's inputs do not carry is `PREREG_V8F_SPEEDCAL`'s question; its eval-fitted CEILING decides between the two "no" readings.
* **SPEEDCAL (T1-class; CEILING T0) [M]: INFORMATION-LIMITED.** An affine map per speed cell, fitted on training windows, moves ADE6 by +0.037 [−0.010, +0.086] m (s0) and −0.006 [−0.016, +0.004] m (s1), separated at neither seed. The decisive row is the CEILING: the same map fitted on the eval GT itself reaches only +0.030 / −0.008 m, so no affine correction of T's speed output exists that would pay. Yet the GT speed cells cut ADE6 by 0.72 / 0.67 m (T0). The information O needs is not in T's speed head in a recoverable form; it has to enter through what T sees. The read was re-derived independently (own ADE6 from the plans, own bootstrap) to the draw.

### 7.11 What each level attends to: the cross-attention of the S, T and O Readers

**[M], T0 open loop, M2b s0 and s1, all 2,399 windows; episode-cluster bootstrap (episode draws only); `VIZ_REPORT.md`, `fig_attention_stats.json`, `controls.json`.**

**Method.** Each level's Reader is a pre-norm Transformer decoder whose learned queries cross-attend to $[\text{cond tokens};\ \text{scene tokens}]$. The scene tokens are 4 history frames × 40 pooled DINOv3 tokens, a 4 × 10 grid. We capture every layer's and every head's cross-attention. We plot the mean over heads, queries and layers, and we draw each level on the frame it last ticked at.

**Controls, all passed, before any map was drawn:**
* the instrumented forward is bit-identical to the plain one (max|Δ traj60| = 0.0, both seeds) and reproduces the banked eval's ADE6 (2.1753 / 2.0546 m; per-window max|Δ| ≤ 2.8e-5 m, CPU against GPU);
* every overlay sits on the frame the tokens came from: re-encoding all 404 rendered steps gives cosine 0.9993–0.9995 to the cached tokens against 0.96–0.98 for neighbouring steps, the cached step wins in 404 of 404 frames, and PSNR to the banked frame is ≥ 48.6 dB;
* model-randomisation (Adebayo et al., 2018): randomising one level's Reader decorrelates that level's maps (Spearman ρ: S +0.08, T +0.03, O −0.09) and leaves the other levels' maps unchanged (max|Δ| = 0.0).

![Figure 9. Per-level S / T / O cross-attention on the four pre-declared windows: the lower-median own-code ADE6 window of each stratum (JUNCTION, CURVE, NO-TURN) and the single worst window. Each level is drawn on its own tick frame. Right: GT path (black), O's plan from its own code (orange), S's checkpoints (pink).](fig_attention_levels.png)

![Figure 10. Aggregate attention mass per level: cond vs scene, per conditioning token, per history frame and per token row; seeds 0 and 1, episode-cluster bootstrap.](fig_attention_stats.png)

**Findings.**
1. **The levels do read different abstractions, and the separation is in WHAT they read, not WHERE in the image.** S and T put 97–100 % of their attention on scene tokens. O puts 76 % (s0) / 93 % (s1) on its conditioning tokens: the upstream plan, the ages, the limit and ego. Above the planner, the hierarchy is perception-driven. The planner itself is mostly plan-driven, which matches TAC-USED (§7.3) from a second, independent instrument.
2. **Attention is sharp.** The effective number of tokens attended, out of 160, is about 6 for S, 8 for T and 15 for O. Per head it is close to one-hot.
3. **Spatially, all three levels look at the centre-ahead horizon and the road ahead; the near rows get about 1 %.** In the NO-TURN window, T places 24.8 % on the lead vehicle.
4. **What does NOT replicate across seeds:**
   * which conditioning token O relies on (s0: age and limit; s1: S's checkpoints, 59 %);
   * the far-row share;
   * any level-specific spatial map. Cross-level map correlation is as high as cross-seed correlation.
   
   So **no level-specific spatial "focus" is claimed**. The seed-robust finding is the cond-vs-scene split of (1).
5. **The worst window is a strategic error propagating down the hierarchy.**
   * S's checkpoints point left while the logged path goes straight on, so T predicts TURN_L against a GT of LANE_KEEP, and O follows. ADE6 is 15.95 m.
   * Feeding O the GT code brings it to 6.41 m. The rest of the error stays with S.

### 7.12 Prediction strategy and quality by horizon

**How a plan is made.** Every 0.2 s ("now"), O emits a 6 s plan of 60 points at 0.1 s:
$$\tau = \mathrm{up}(P10) + \mathbf{B}c$$
* $P10$ is a learned kinematic prior that extrapolates the measured ego motion (speed, yaw rate, history).
* $\mathbf{B}c$ is O's learned correction on a 12-term cosine basis. A ramp makes it vanish as $t \to 0$.

O conditions on T's code (held from T's last tick, at most 0.4 s old), on S's checkpoints (held, at most about 1.2 s old), on nav, the limit, ego and 4 frames of scene tokens. The eval scores the 30 points at 0.2 s against the logged future, open loop.

**HBLEND** (`PREREG_V8F_HBLEND`, PASSED at both seeds) adds a fixed, train-fitted shrinkage of the correction toward the prior: $\tau_H = \mathrm{up}(P10) + w(t)\odot\mathbf{B}c$. The fitted weights are themselves a finding:
* along-track, $w \approx 0$ up to 1 s and $\approx 0.1$ up to 2.4 s, then 0.43–0.64;
* cross-track, 0.66–0.89.

So O's *speed* correction is over-confident at short horizons, and its *lateral* correction is mostly trusted.

**Quality by horizon [M]** (ADE in m, eval-141, T1-class open loop; `raw/2026-10-25-hproj/score_M2b_s{0,1}.json`; 95 % episode-cluster intervals are in the source):

| horizon | B s0 | B s1 | H s0 | H s1 | P10 |
|---|---|---|---|---|---|
| 1 s | 0.082 | 0.071 | 0.060 | 0.060 | 0.063 |
| 2 s | 0.316 | 0.291 | 0.212 | 0.214 | 0.222 |
| 3 s | 0.577 | 0.553 | 0.473 | 0.481 | 0.534 |
| 4 s | 0.984 | 0.936 | 0.854 | 0.866 | 1.013 |
| 5 s | 1.518 | 1.440 | 1.353 | 1.369 | 1.663 |
| 6 s | 2.175 | 2.055 | 1.969 | 1.985 | 2.487 |
| FDE 6 s | 6.281 | 5.858 | 5.819 | 5.817 | 7.667 |

**Reading.**
* B is worse than the kinematic prior up to 3 s and better from 4 s on: the crossover sits between 3 and 4 s.
* H is at or below P10 at every horizon, and beats it by about 0.5 m at 6 s (separated). Its gain over B is separated at 2 s and 6 s at both seeds.
* H is not deployable yet. Its plans violate a GT-derived constraint more often (+0.10 / +0.075 of windows, mostly keeping speed where the human slowed). The posted-limit projection did not repair that (`PREREG_V8F_HPROJ`, FAILED). The T-gated form (`PREREG_V8F_HGATE`) also FAILED its consistency bar: it removes 80 % / 62 % of the cost but keeps only 18 % / 38 % of the 6 s gain, because T's own speed-cap prediction cannot tell where the human slowed. So the remaining lever is T itself, not post-processing.
* H's acceleration error is still above P10's. Shrinking the correction per cosine term as well (HSPEC) brings curvature level with P10 at both seeds and acceleration level at seed 1, but leaves seed 0 at +0.026 m/s² (separated). No shrinkage of the cosine coefficients reaches P10's acceleration, even fitted on the eval GT (HSPEC's T0 ceiling), so the committed next lever is O's training loss (`PREREG_V8F_OACC`).
* **Where the acceleration gap sits [M], descriptive, T1-class.** A per-step profile of H's accel error reproduces the banked family metric to float32 epsilon. It shows that the gap to P10 is not spread over the horizon. It sits in three bands: 2.2–2.6 s, where HBLEND's along-track weight ramps from 0.12 to 0.50 and H is worse than **both** B and P10 (a steep $w(t)$ adds $w''r + 2w'r'$ to the blend's second derivative); 5.2–5.8 s, where B's own tail is rough and the monotone weight is at its maximum; and 0.8–1.6 s. From 2.8 s to 5.0 s, H already beats P10 on acceleration at both seeds. This motivates `PREREG_V8F_HACC` (row 1m): a weight fitted on training windows with the second-difference error in its objective and no monotone constraint, at zero training.
* **HACC [M], T1-class open loop, paired episode-cluster bootstrap, B = 2,000, 141 episodes: DOMINATES P10 at both seeds.** HACC − P10: ADE2 −0.0106 [−0.0150, −0.0056] / −0.0083 [−0.0133, −0.0026] m; ADE6 −0.503 / −0.471 m; accel MAE −0.0215 [−0.0345, −0.0087] / −0.0402 [−0.0509, −0.0293] m/s²; curvature MAE −0.0056 / −0.0051 1/m. Speed, along, cross-track, heading, yaw rate, both plan-derived decision agreements and CONSISTENCY are separated-better as well. Distance keeping is not separated, and LEAD-VIOL is better at s0 and not separated at s1. λ is chosen on training windows only (m = 4 / 16). Against H, accel falls by 0.125 / 0.106 m/s² and ADE6 rises by +0.015 / +0.031 m, within the pre-registered F/2: the smoothness is bought with a little displacement. The verdict was re-derived independently, with a different solver and our own bootstrap, to the bootstrap draw.

### 7.13 Who listens to what: a causal grid of the three layers

**Method** (`PREREG_V8F_LISTEN`, `a1dbdc2b`). For each layer and each of its inputs, the layer reads that one input from a window of a *different episode*, while every other input and every other layer read the window's own values. The cell is how much that layer's own output gets worse:
* S: checkpoint error at 50 m;
* T: lateral and longitudinal macro-F1;
* O: ADE6.

O's kinematic prior keeps its own ego input, so the O × ego cell measures O's reader and not the prior. Colour is the cell divided by the swap of *all* that layer's inputs.

![Figure 11. Who listens to what. Rows: S, T-lateral, T-longitudinal, O; columns: the inputs. Cell text: raw degradation with its 95 % paired episode-cluster interval; * = separated at both seeds and above the donor-seed noise. Right: the 6 s ADE cost of a donor nav injected at S only, T only, O only, or all three.](fig_listen_grid.png)

**What each layer depends on** (MEASURED, T1-class open loop, diagnostic; seed 0 / seed 1):

| layer | depends most on | then | barely or not at all |
|---|---|---|---|
| **S** (route) | nav +4.20 / +3.99 m | scene +3.21 / +3.63 m, ego | speed limit |
| **T, lateral** | **S's checkpoints**: F1 drops 0.39 / 0.41 | nav: F1 drops 0.17 / 0.22 | **scene (not separated)**, limit |
| **T, longitudinal** | ego: F1 drops 0.25 / 0.26 | scene: F1 drops 0.18 / 0.19, limit | S's checkpoints, nav |
| **O** (trajectory) | its ego token +5.70 / +4.72 m and **T's code** +5.40 / +4.13 m | S's checkpoints +0.87 / +1.89 m, limit | nav +0.015 / +0.011 m; scene +0.31 / +0.01 m |

**Reading.**
* **The pre-registered hypotheses failed** (outcome NOT SHOWN). I had predicted from the attention maps that T-lateral would depend on the scene and that O would depend most on the layers above. In fact T-lateral depends on S, and O depends as much on its own ego motion as on T.
* **The dependence runs down the chain:** nav → S → T-lateral → O. This is the hierarchy working as an information path, and it answers the PI's question directly. **Nav reaches the trajectory through S and T, not around them** (Q-NAV, THROUGH T):
  * a donor nav given to O alone moves ADE6 by +0.015 / +0.011 m;
  * given to T alone, +0.48 / +0.52 m;
  * given to S alone, +0.73 / +0.82 m.
* **Attention is not dependence.** T puts 97–100 % of its attention on scene tokens, yet swapping T's scene leaves its lateral decision unchanged. The longitudinal decision, by contrast, does use the scene.
* **Limits.** The cells are interventions at open loop, not driving. For O, a single swap hurts more than swapping all inputs (n > 1): a coherent donor world is less damaging than a contradiction between inputs. Three cheap follow-ups are named in the result (a coherence test, mean-token replacement, and an echo test of T-lateral on S's nav read), each needing its own pre-registration.

*Attention is not explanation: Figures 9–10 show where the decoder's softmax mass goes, not a causal attribution. Each Reader also adds $\overline{\text{cond}}$ to every query, a path the attention maps do not show. A video of the four whole clips (`v8f_hierarchy_attention.mp4`, 80 s) accompanies this report.*

### 7.14 The nav command, layer by layer (evidence video 2)

The second video (`v8f_evidence_v5.mp4`, 536 s, below; v5 adds chapter 8, **DAgger: trained on its own states**: DAG against DAG-0 on their own motion, with S's checkpoints, T's decisions against the label and O's executed path at every replan tick, on seven rule-selected windows with the worst kept. v4 was `v8f_evidence_v4.mp4`, 381 s; earlier: v3 added HACC to every plan panel and a chapter on dominance over P10, row 1m; v4 adds chapter 7, the T1-kin executed tracks of §7.15 on seven rule-selected windows with failures kept, the per-window distribution, and the causal, strict and lateral-gap cards) shows §7.12–7.13 on real frames and adds one measurement, a **nav counterfactual**. For every eval window, a constant command (FOLLOW, TURN LEFT or TURN RIGHT, next turn at 30 m) is written into the nav input of the named layer(s). Everything else stays the window's own, and the model runs its own S → T → O chain. **D6** is the mean distance at 6 s between the TURN LEFT / TURN RIGHT plans and the FOLLOW plan.

**How far the command moves the 6 s plan** (MEASURED, T1-class open loop, an intervention on the input, not driving; 2,399 windows, episode-cluster bootstrap, B = 2,000, 141 episodes):

| command written into | D6 seed 0 (m) | D6 seed 1 (m) | T-lateral argmax = command, FOLLOW / TURN L / TURN R (s0; s1) |
|---|---|---|---|
| all three layers | 11.16 [9.77, 12.62] | 10.76 [9.71, 11.85] | 94 / 52 / 62 %; 90 / 54 / 73 % |
| S only | 9.75 [8.48, 11.09] | 11.12 [9.77, 12.53] | 95 / 6 / 16 %; 92 / 7 / 14 % |
| T only | 4.24 [3.56, 4.98] | 4.04 [3.32, 4.84] | 95 / 13 / 19 %; 94 / 15 / 34 % |
| O only | 1.06 [0.98, 1.14] | 1.22 [1.12, 1.31] | 92 / 2 / 6 %; 91 / 3 / 7 % |

**Reading.**
* **The command steers the plan, and it does so mainly through S.** S alone moves the plan almost as far as all three layers; O's own nav token is nearly inert. This is the same ordering as the donor-swap grid of §7.13, now under a different intervention.
* **The plan and T's decision move by different routes** (descriptive, not pre-registered). With the command at S only, the plan moves 10–11 m, yet T's lateral decision follows a turn command in only 6–16 % of windows. The plan therefore moves through S's checkpoints into O, not through T's decision. T's decision follows a turn in 52–73 % of windows only when all three layers receive the command.
* **Does the route reach T's decision through S? Yes, for real routes** (`PREREG_V8F_LISTEN2`, E1, MEASURED, T1-class open loop, paired episode-cluster bootstrap). A donor window's nav injected at S only, with T and O keeping their own nav, drops T-lateral's macro-F1 by +0.442 [+0.353, +0.507] (s0) and +0.419 [+0.349, +0.483] (s1). That is about five times the donor-seed floor, and T's and O's own tokens are bit-identical to clean on every window. Outcome **ROUTE FLOWS DOWN**: nav → S → T-lateral is a working chain. Together with the table above: a real route, arriving through S's checkpoints, changes T's lateral decision; an artificial constant command at S alone mostly does not. LISTEN2 also found O hurt more by a contradiction between its inputs than by a coherent foreign state (E3, −3.8 / −3.2 m), and that O's ego-vs-T-code order is not robust to the intervention used (E2, PARTLY ROBUST).
* **Limits.** A constant command over the whole nav history is out of distribution for many windows (a TURN LEFT where no left turn exists), so these rates do not measure how well T follows a valid route. Controls: the window's own nav reproduces the deployed forward pass to 0.0; a model with its nav embedding zeroed moves exactly 0.0 over 160 cells.

### 7.15 T1-kin: HACC on its own motion (self-action open loop)

`PREREG_V8F_T1KIN_HACC` (with three amendments, each made before the decision rows were read; R74) asks whether HACC's open-loop margin survives when the model drives on its own outputs. In **T1-kin**, every replan (m = 10 steps, 2 s) rebuilds the ego state the model reads from its own executed motion: speed, acceleration, yaw rate and the dead-reckoned nav. The camera frames stay the human's (T2, re-perception, is not provisioned), so ⛔ this is an open loop, not closed-loop driving. The P10 floor is driven the same way, in its own loop. Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes, 2,399 windows; the two training seeds are the replicate.

| HACC − P10, T1-kin m = 10 (lower is better) | s0 | s1 |
|---|---|---|
| **ADE6 (m), PRIMARY** | **−0.829 [−0.971, −0.689]** | **−0.795 [−0.938, −0.658]** |
| ADE2 (m) | −0.011, better | −0.009, better |
| speed / along-track MAE (LONG) | −0.39 m/s, −0.54 m, better | −0.43 m/s, −0.58 m, better |
| accel MAE (m/s², LONG) | +0.045, n.s. | −0.229, better |
| heading / cross-track MAE (LAT) | −0.43 deg n.s., −0.57 m better | −0.77 deg, −0.43 m, better |
| **curvature MAE (1/m, LAT)** | **+0.040, worse** | **+0.016, worse** |
| **yaw-rate MAE (deg/s, LAT)** | **+4.20, worse** | +1.53, n.s. |

**Outcome: T1 BEATS THE FLOOR** (the pre-registered table, verbatim). PRIMARY holds at both seeds; DOMINANCE fails on the lateral rows. The tactical and strategic families are not computed at T1 in this read (the dumps carry no per-tick T code; a work item). The tactical family from the executed track and distance keeping were read later on the same dumps (below). Every control passed, and the read was re-derived independently to the draw.

* **The loop widens the margin.** One shot (m = 30) gives −0.53 m. At m = 10 the floor degrades from 2.49 to 3.44 m in its own loop, while HACC degrades from 1.96 to 2.61 / 2.65 m. At a 1 s replan (m = 5) both loops run away (floor 5.41 m, HACC 5.26 / 4.94 m); HACC − P10 is −0.15 m (not separated) / −0.47 m (separated).
* **The post-map is what keeps the loop smooth.** Plain M2b at m = 10 also beats the floor on ADE6 (−0.41 / −0.65 m) but is **+1.22 / +1.08 m/s² worse in acceleration**. HACC removes that roughness.
* **Brackets** (which open channel carries the margin): with the cameras frozen at the first frame, the margin is still −0.79 / −0.71 m, so the logged cameras flatter little. ⚠ **With the speed bucket removed, it is gone: +0.24 / +0.08 m, not separated**, and HACC's accel turns worse than the floor's. The bucket is a PI-declared input (max target speed), but on PhysicalAI it is the max of the ego's realised future speed. So the claim is conditional on a declared, optimistic input, and it is filed for the PI as `D-V8F-SPEEDBUCKET-T1`. This is SPEEDCAL's finding from the other side: speed is the largest lever, and here the model takes it from the bucket, not from what it sees. ⚠ "Off" is also outside the training distribution, so the 1 m mixes the bucket's information with the model's response to an absent input.
* ⭐ **The causal bucket [M]** (`PREREG_V8F_T1KIN_CAUSALBUCKET`, landed before data; `RESULT_V8F_T1KIN_CAUSALBUCKET.md`, `76c7d98e`).
  * The bucket the model reads at each replan is the sidecar's own containing-window rule applied to the max MEASURED ego speed of the last 4 s (its own executed speed after t0). No future row enters it.
  * **Outcome CAUSAL KEEPS THE MARGIN:** HACC − P10 ADE6 **−0.768 [−0.918, −0.626] / −0.734 [−0.878, −0.598] m**, separated at both seeds. Against the oracle bucket it costs +0.061 / +0.061 m, and the two buckets agree on 79 % of ticks.
  * ⇒ **The `--limit off` collapse was the model's response to an ABSENT input (it never saw one in training), not the oracle's information.** v8F's T1 result stands on an input built from the past alone.
  * The lateral rows are unchanged, which is HACC-LAT's target.
  * Every control passed: logged and off byte-identical, 4 / 4 leak mutants red, the tick-0 bucket equal to a hand computation, and the floor exactly unchanged.
* **Where the lateral excess lives** (descriptive). It is not the replan seams and not the camera viewpoint (it does not grow with the pose offset). It is speed: below 5.2 m/s the curvature excess is +0.40 / +0.25 1/m, and above 11.8 m/s it is about zero. Near standstill a millimetre wiggle is a large curvature, and HACC's metre-scale objective cannot see it. In the first 2 s (logged state) HACC is better than the floor; the excess starts with the first plan issued from its own slow state.
* ⭐ **Strict [M]** (`PREREG_V8F_T1KIN_STRICT`, before data): the cameras frozen at the first frame AND the causal speed bucket.
  * **Outcome STRICT MARGIN:** HACC − P10 ADE6 **−0.728 [−0.868, −0.594] / −0.641 [−0.770, −0.519] m**, separated at both seeds.
  * The model then reads one frame, its own motion, a speed bucket from the past and the dead-reckoned nav, and still beats the floor by 0.73 / 0.64 m. Closing the camera channel on top of the causal bucket costs +0.04 / +0.09 m.
* **The lateral deficit, examined in three pre-registered steps [M].**
  1. **HACC-LAT** (`PREREG_V8F_HACC_LAT`, row 1n): two zero-training arms fitted on T1-kin ticks of training windows. **NOT FIXED.** HACC's weights re-fitted on the ticks are far worse; a speed gate on the cross axis keeps the margin but not the curvature.
  2. **The instrument** (no model arm): 5–10 mm of position noise on the GT, scored against itself, gives 0.021–0.038 1/m of curvature MAE and 2.4–4.7 deg/s of yaw-rate MAE. That is the size of the deficit, and almost all of it sits below 5 m/s. Curvature divides by speed, so the canonical curvature and yaw-rate rows are millimetre-noise-dominated near standstill.
  3. **Well-conditioned rows** (`PREREG_V8F_T1_LATCOND`, rows fixed from the instrument alone, before any arm was read on them).
     * Lateral-acceleration MAE is **not** worse (−0.11 / −0.12 m/s²).
     * Curvature MAE at GT speed ≥ 5.2 m/s **is** worse: +0.021 [+0.010, +0.035] / +0.008 [+0.003, +0.015] 1/m. Outcome **CONDITIONED LATERAL DEFICIT**.
     * ⇒ About a third of the excess was the instrument's. ~~The rest is a real curvature-profile mismatch at speed~~ (retracted, `R76`, step 4), while heading, cross-track and lateral acceleration are better or equal.
  4. **The root cause [M]** (zero-GPU diagnosis on the banked dumps, `raw/2026-10-26-lcdiag/DIAGNOSIS.md`).
     * 96 % / 88 % of the L2 excess sits on pairs where the GT moves ≥ 5.2 m/s but **HACC's own car is below 5.2 m/s**. These are low-speed launches (332 / 220 windows): the human accelerates from 5.7 to 6.8 m/s, while HACC's loop decelerates to 4.8 m/s.
     * **At equal own speed, HACC's curvature error equals the floor's.** With both cars' own speeds ≥ 5.2 m/s, the excess is +0.0018 / +0.0015 1/m, n.s. at s1.
     * ⇒ The "lateral deficit at speed" is a **longitudinal crawl** that a curvature row masked on the GT's speed charges to lateral control: the arm-side twin of the yaw-mask retraction. This also explains why HACC-LAT, which acted on the cross axis, changed nothing.
     * **The leading mechanism** (measured, not yet tested by intervention): HACC's along weight ramps from 0.14 to 0.32 across knots 8–11, the knots the replan state reads. Its step variation adds a term (dw/dk)·(O − P) to the plan's velocity that is in neither plan. That term is 79 % / 74 % of HACC's tick-1 acceleration gap to the floor in the launch windows.
     * **Next (pre-registered, `PREREG_V8F_HACC_MC`, before any arm):** MC-H holds the along weight constant over the executed knots 1–12 (zero training). A 5× steeper ramp is the deliberate-regression arm. Both outcomes are committed, including the risk that the ramp also offsets the floor's own acceleration overshoot, and with it part of the margin.
* ⭐ **The FLAT control [M]** (`PREREG_V8F_FLAT`; `RESULT_V8F_FLAT.md`; T1-class open loop, eval-141).
  * FLAT = M2b with `--o-interface cut`: O reads constants for S's checkpoints and T's code. The launch gate proved that structurally (a zero gradient into the code embedding, and O invariant to S and T), with its own regression arm.
  * **Outcome NOT ESTABLISHED:** ADE6 M2b − FLAT −0.036 [−0.113, +0.035] / −0.003 [−0.081, +0.076] m. FLAT's GT-code oracle equals its own output exactly, as the identity control requires.
  * ⇒ The hierarchy, as trained, adds no measurable value over a route-aware flat planner of the same capacity. Its GT-code oracle (0.97 / 1.00 m) shows the interface can carry 1.2 m, so the defect is the code at inference. That is the PREREG's committed reading, and the MIXO arm is its lever.
* ⭐ **The tactical and distance-keeping families at T1 [M]** (`PREREG_V8F_T1_FAMILIES`; `RESULT_V8F_T1_FAMILIES.md`).
  * The canonical factored labeller is applied to the executed tracks. `lead_metrics` scores them against the logged (non-reactive) lead of the banked lead block: 743 LEAD windows.
  * **As written, the read was REFUSED:** its deliberate-regression control (a one-window roll) was a near-copy, against a comparator it could not beat (`R75`).
  * **Amendment 1, written after the rows were seen and so reported as POST-DATA,** replaced it with speed-matched tracks from another episode. That arm reads separated-worse on all five rows at both seeds.
  * **Outcome under it: TACTICAL NO WORSE, DISTANCE KEEPING NO WORSE.**
    * Lateral manoeuvre agreement with the human: +0.136 / +0.092 (0.73 / 0.69 vs 0.59).
    * Longitudinal manoeuvre agreement: +0.131 / +0.132.
    * The time gap to the lead is closer to the human's by 0.10 / 0.07 s.
    * All of these are separated-better. LEAD-VIOL is n.s.
  * These describe what the car DID. The next item reads what the heads DECIDED.
* ⭐ **The heads at T1 [M]** (`PREREG_V8F_T1_HEADS` with Amendment 1; `RESULT_V8F_T1_HEADS.md`; HACC + causal bucket, m = 10).
  * **How it is scored:** every tick's S and T outputs are dumped. Each is scored against the human's labels at the nearest logged point of the route. Ticks ≥ 1 are the model's own state; tick 0 is the logged state.
  * **Controls K1–K7 all pass.** Among them, tick 0 reproduces the banked T0 eval to ≤ 4.4e-5 m and Δ F1 = 0.0. A mirrored-S and a cross-episode-T regression arm both read separated-worse.
  * **STRATEGIC: S ADDS ROUTE.** The S − CTRV checkpoint error is −1.18 / −3.27 / −7.79 m (s0) and −1.26 / −3.47 / −8.81 m (s1) at 30 / 50 / 80 m, separated at every arc. It is the first strategic-family read at T1.
  * **TACTICAL: DEGRADES.** Lateral macro-F1 Δ (ticks ≥ 1 − tick 0) is −0.235 / −0.227, and longitudinal −0.125 / −0.131, both separated and beyond 2F. TURN_L / TURN_R fall from 0.81 / 0.75 to 0.38 / 0.50 at s0, while LANE_KEEP holds (0.98 → 0.96).
  * **Descriptive, decides nothing (post-data):** at zero row offset between the label and the cameras T saw, lateral F1 is still 0.606 / 0.647 against 0.828 / 0.857 at tick 0. ⇒ The drop is T's sensitivity to its own executed state.
  * **Next (pre-registered, `PREREG_V8F_ORECOVER`, before its code and any run):**
    * ORC = M2b fine-tuned for 3,000 steps with 30 % of samples on a perturbed ego state: v0 × U(0.5, 1.0) and a0 − U(0, 1.5) m/s² below 8 m/s, plus yaw rate + N(0, 2 deg/s). Targets and labels are unchanged.
    * The control is ORC-0, the same fine-tune unperturbed.
    * **ROBUST** needs both F1 rows ≥ 2F better, separated at both seeds, with ADE guarded.


### 7.16 OACC: an acceleration-matching loss buys smoothness and costs position

* **Setup** (`PREREG_V8F_OACC` with Amendments 1 and 2). X = M2b fine-tuned with an added loss on the second difference of O's 0.2 s plan
  against the GT's. Its weight, w_acc = 42.9311, was fixed by a train-only rule before any data. Seed 1 runs only if seed 0 passes
  PRIMARY-ACC and PRIMARY-KEEP.
* **Result [M]** (`RESULT_V8F_OACC.md`, `oacc_read_s0.json`). T1-class open loop over 2,399 windows / 141 episodes; paired
  episode-cluster bootstrap, B = 2,000; X − B, or HX − P10 for the deployed blend:

| read | point [95 %] | bar | result |
|---|---|---|---|
| PRIMARY-ACC: accel MAE X − B | −0.5663 [−0.6724, −0.4711] m/s², separated | < 0, separated | passed |
| PRIMARY-KEEP: ADE6 X − B | +0.0451 [−0.0322, +0.1262] m | upper bound ≤ +F/2 = +0.0604 m | **failed** |
| DOMINANCE: HX − P10 | 14 of 15 rows met; curvature MAE +0.0084 [+0.0039, +0.0139], separated-worse | every row | **failed** |

* **Outcome, as committed: COSTS POSITION.** Seed 1 is not trained, and the weight rule is not re-tuned, both as committed.
* It is one training seed. The intervals answer "would another draw of episodes say this?", not "would another training run?".

### 7.17 HACC-MC: a constant along weight shrinks the launch crawl, but not enough

* **Setup** (`PREREG_V8F_HACC_MC` with Amendments 1 and 2; zero training). MC-H holds HACC's along weight constant over the executed knots 1–12.
  The constant c is HACC's own banked w_along at knot 10, 0.2002 / 0.2198. STEEP, a 5× steeper ramp, is the deliberate-regression arm.
* **Result [M]** (`RESULT_V8F_HACC_MC.md`). T1-kin at m = 10, a self-action open loop; paired episode-cluster bootstrap, B = 2,000, 141 episodes:

| | seed 0 | seed 1 |
|---|---|---|
| launch crawl M1: MC-H / HACC / STEEP | 0.0184 / 0.0536 / 0.0850 | 0.0151 / 0.0277 / 0.0504 |
| L2 (curvature at GT ≥ 5.2 m/s): MC-H − P10 | +0.0028 [+0.0004, +0.0055], separated-worse | +0.0013 [−0.0014, +0.0039], not separated |
| ADE6: MC-H − HACC | +0.1383 m | +0.2060 m |

* The crawl falls with the constant weight and rises with the steep one at both seeds, so the step-varying weight is a real cause.
* It is not the whole cause: L2 stays separated-worse at seed 0, and ADE6 misses KEEP. O's own slow launch intent carries the rest.
* **Controls.** All nine pass under Amendment 2, which is post-data and control-only. The Amendment 1 read REFUSED on control C-1: its
  float bar was blind to cancellation at a tiny |plan|. Amendment 2 measures the bar in ulps of the largest operand, and its exact timeline
  is disclosed.
* **Outcome, as committed: MECHANISM-NOT-ENOUGH.** The committed next is the box-2 training lever, O-RECOVER.


### 7.18 O-RECOVER: perturbed-state training buys lateral skill on the model's own motion and costs position

* **Setup** (`PREREG_V8F_ORECOVER` with Amendments 1–4, all before any run). ORC = M2b fine-tuned 3,000 steps with 30 % of eligible samples
  moved off the logged trajectory and their route rebuilt; ORC-0 = the same fine-tune unperturbed. Both seeds, trained on box 2 through the
  launch gate; the T1-kin read on the dev-box CPU. The reader is `orc_read.py` (sha256 `91a9e971…`).
* **Result [M]** (`RESULT_V8F_ORECOVER.md`, `ORC_READ.json` sha256 `a50dbba9…`). Paired episode-cluster bootstrap, B = 2,000, 141 episodes,
  2,399 windows; ORC − ORC-0:

| row | tier | seed 0 | seed 1 | bar | result |
|---|---|---|---|---|---|
| R-TLAT: lat macro-F1, ticks ≥ 1 | T1-kin | +0.0331 [−0.0079, +0.0730] | +0.0351 [−0.0004, +0.0671] | separated-better and ≥ 2F (0.0248) | not met at either seed |
| R-TLON: lon macro-F1, ticks ≥ 1 | T1-kin | −0.0214 [−0.0497, +0.0131] | +0.0312 [+0.0063, +0.0542] | separated-better and ≥ 2F (0.0146) | met at seed 1 only |
| R-ADE: ADE6, m | T1-kin | **+0.1100 [+0.0083, +0.2283]**, separated-worse | +0.0355 [−0.0363, +0.1047] | not separated-worse | **failed at seed 0** |
| R-T0: ADE6, m | T0 | **+0.1103 [+0.0443, +0.1856]** | **+0.0849 [+0.0378, +0.1324]** | point ≤ +F/2 = 0.0604 | **failed at both seeds** |

* **The four families** (ORC − ORC-0, same estimator):
  * **LATERAL, T1-kin:** separated-better at both seeds. Heading −1.31 / −1.00 deg, curvature −0.054 / −0.022 1/m, yaw rate −6.59 / −3.16 deg/s.
    Cross-track is not separated.
  * **LONGITUDINAL, T0:** separated-worse at both seeds. Speed +0.059 / +0.021 m/s, accel +0.131 / +0.055 m/s², along-track +0.113 / +0.075 m.
    At T1-kin only accel at seed 0 is separated (worse). The programme's anti-echo block admits no longitudinal claim for any of the four runs,
    so these are contrasts, not capability statements.
  * **TACTICAL:** the plan-derived lateral decision agreement is separated-better at T1-kin at both seeds (+0.022 / +0.038). The T0 goal-head
    F1 is separated-worse at seed 0 (−0.030).
  * **STRATEGIC** (not paused for v8F, R73; decides nothing): S's checkpoint errors at 30 / 50 / 80 m are not separated at T0 at either seed.
* **Controls.** All 104 pass, including perturbation reach (ORC shares 0.2998 / 0.3022 against 0.3; exactly 0 for ORC-0), K4 against the
  arm's own fresh-age T0 eval, and K7 (the executed track rebuilt from the dumped plans).
* **Reading.** The perturbation teaches recovery from the model's own lateral drift, which is what §7.15 asked for. It costs T0 position,
  mainly along-track, by 0.70–0.91 F. F is the only run-to-run measure on this rig, so a single replicate could not split that cost from
  training noise. Either way, the cost breaks the pre-registered guard.
* **Outcome, as committed: HARMS.** ORC is not adopted. The committed next is the NO EFFECT row's arm at p = 0.1: on-policy own states from
  frozen M2b at T1-kin ticks 1–2, TRAIN windows only (`PREREG_V8F_DAGGER` with Amendment 1, `2a78f059`).


### 7.19 DAgger: supervising T on the model's own states repairs its decisions on its own motion

* **Setup** (`PREREG_V8F_DAGGER` + Amendment 1, `2a78f059`, before any bank or run).
  * **The bank.** Frozen M2b is rolled out at T1-kin (m = 10, causal bucket, dead-reckoned nav, PLAIN) over the 4,512 TRAIN clips. Its own
    states at ticks 1–2 are kept with their own 16-row ego history and re-anchored poses, and labelled from the human future at the
    matched route point.
  * **COVERAGE:** 0.756 / 0.769 of the windows are eligible, against a bar of 0.5.
  * **DAG:** M2b fine-tuned 3,000 steps with p = 0.1 of eligible samples swapped for a bank state. The swap share reached 0.1013 / 0.1008.
  * **DAG-0:** the same argv with p = 0 (swap share exactly 0).
  * Both seeds were trained on box 2 through the launch gate, and read on the dev-box CPU.
* **The amendment.** The first read REFUSED at the bank stage, before any row: the reader read the generator's `limit_clips` 0 (no limit) as
  a limit. Amendment 2 (`7d29172b`) is post-data and control-only. It aligned that one clause with the generator, and the reader was
  re-proved (160 / 160 mutants) and run once.
* **Result [M]** (`raw/2026-10-27-dagger-run/read/DAG_READ.json`, sha256 `3393fdd5…`). Paired episode-cluster bootstrap, B = 2,000, 141
  episodes, 2,399 windows, DAG − DAG-0. A head clears at a seed iff it is separated-better with point ≥ max(2F, 2·RF). RF is the replicate
  floor: DAG-0 against ORC-0, a same-argv replicate trained by another build.

| row | tier | seed 0 | seed 1 | bar s0 / s1 | result |
|---|---|---|---|---|---|
| R-TLAT: lat macro-F1, ticks ≥ 1 | T1-kin | **+0.1082 [+0.0491, +0.1572]** (0.603 vs 0.495) | **+0.1184 [+0.0755, +0.1585]** (0.639 vs 0.520) | 0.0248 / 0.0248 | **clears at both** |
| R-TLON: lon macro-F1, ticks ≥ 1 | T1-kin | **+0.0554 [+0.0300, +0.0847]** (0.321 vs 0.266) | **+0.0806 [+0.0560, +0.1043]** (0.331 vs 0.251) | 0.0502 / 0.0278 (2·RF) | **clears at both** |
| R-ADE: ADE6, m | T1-kin | **−0.5435 [−0.7564, −0.3690]** (2.278 vs 2.821) | **−0.4046 [−0.5496, −0.2782]** (2.456 vs 2.861) | not separated-worse | held, separated-better |
| R-T0: ADE6, m | T0 | +0.0019 [−0.0230, +0.0266] | +0.0483 [+0.0178, +0.0807] | point ≤ +F/2 = 0.0604 | held |

* **Size against the noise.** R-TLAT is 4.4× / 4.8× 2F. R-TLON is 1.10× / 2.90× its replicate-floor bar, so the s0 margin is the thinnest in
  the table. R-ADE is 4.5 F / 3.4 F.
* **STRATEGIC, beside** (unpaired, within each run, not paused for v8F). S-POS against CTRV at 30 / 50 / 80 m:
  * DAG: −1.72 / −4.48 / −9.69 m (s0) and −1.87 / −4.90 / −10.33 m (s1), all separated.
  * DAG-0: separated only at 80 m at s0 (−3.89 m). At s1 it is separated-worse at 30 m (+0.81 m).
* **Reading.** DAgger does what §7.15 diagnosed was missing. T was trained only on the human's states, so on its own drifted states its
  decisions degraded; supervising it on those states repairs them. It does not buy logged-state accuracy: tick-0 F1 and T0 ADE are flat or
  slightly worse. The gain is specific to the model's own state distribution, which is the one it is deployed on.
  * T1 lateral F1 (0.60 / 0.64) is still below tick 0's (0.85). The gap narrows; it is not closed.
* **Outcome, as committed: ROBUST.** DAG is adopted as the base. The committed next:
  * T1 dominance over P10 is re-read on DAG with a registered bar;
  * FLAT's repeat is run, at its own tier and at T1-kin;
  * if the crawl rows still fail, DAgger iteration 2 follows.
  `PREREG_V8F_DAG_REREAD` (`bdf8d561`) registered these before any of their rows.


### 7.20 DAG against the kinematic floor and the flat control, on its own motion

* **Setup** (`PREREG_V8F_DAG_REREAD`, `bdf8d561`, written before any of its rows). Everything is read at T1-kin: m = 10, causal bucket,
  dead-reckoned nav, logged scene, self-action open loop. Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes.
  * **Read A, dominance:** ADE6 separated-better by ≥ 2F (D1), no row of the four families separated-worse (D2), and a separated-better row
    in every family (D3), all at both seeds.
  * **Read B:** DAG − FLAT, at FLAT's own open-loop tier (B-1) and at T1-kin (B-2).
* **Read A [M]** (`RESULT_V8F_DAG_REREAD.md`). DAG − P10:

| row | seed 0 | seed 1 |
|---|---|---|
| ADE6 (m) | **−1.166 [−1.344, −0.996]** | **−0.987 [−1.178, −0.806]** |
| speed / along-track / heading / cross-track MAE | all separated-better | all separated-better |
| TAC-LAT / TAC-LON | **+0.193 / +0.129** | **+0.185 / +0.114** |
| accel MAE (m/s²) | **+0.807 WORSE** | **+1.444 WORSE** |
| curvature MAE / L2 (1/m) | **+0.061 / +0.029 WORSE** | **+0.113 / +0.060 WORSE** |
| M1 launch-crawl share | **+0.047 WORSE** | **+0.061 WORSE** |
| DK-VIOL / yaw-rate | **+0.035 WORSE** / n.s. | n.s. / **+8.66 WORSE** |

  The outcome is **MARGIN, NOT DOMINANT**. The failing set is mixed:
  * the crawl rows go to DAgger iteration 2, as committed by both PREREGs;
  * the longitudinal rows go to the longitudinal family's lever (§7.21).
* **The four families, DAG − DAG-0** (`RESULT_V8F_DAGGER.md` §3; 14 controls PASS):
  * at T1-kin, speed, along-track, heading, cross-track, plan-derived decisions and turn F1 are separated-better at both seeds;
  * S's checkpoint error is about halved: −2.5 / −3.4 m at 30 m, and −7.6 / −10.4 m at 80 m, paired;
  * **distance keeping is separated-worse at both seeds.** The lead-violation share behind a lead goes 0.082 against 0.040, and 0.052
    against 0.022. The min-TTC error rises by 0.87 / 0.60 s.
* **Read B [M]:**
  * **B-1, open loop:** NOT ESTABLISHED (−0.238, separated / +0.041 m). The s0 gain is the fine-tune's, not DAgger's: DAG-0 − FLAT is −0.240.
  * **B-2, T1-kin:** DAG − FLAT is **−0.596 [−0.738, −0.457] / −0.351 [−0.470, −0.229] m**, and also better on curvature, yaw-rate, L2 and
    M1.
    * Its control C5 failed as written: two output-path flags differ, and an inspection key could not show the cut. The read as written is
      therefore REFUSED. It passes under Amendment 1, which is post-data, control-only and disclosed: the cut is shown by the checkpoint
      config and by FLAT's exact-zero oracle identity.
    * DAG-0 − FLAT is not separated at either seed, so the margin comes with DAgger. Whether it needs the hierarchy is FLAT-DAG's question.

### 7.21 HACC on DAG: the post-map clears the longitudinal rows and costs position at one seed

* **Setup** (`PREREG_V8F_DAG_HACC`, `2a416ca7`, before any fit). HACC's rule is re-fitted verbatim on DAG's own T0 plans over HBLEND's TRAIN
  fit set (141 clips, 9,582 windows), giving λ* 35309 / 33857, and baked as 30 × 2 weights. It is read on Read A's rows, with a KEEP guard:
  DAG-HACC − DAG ADE6, upper bound ≤ +F/2.
* **Controls:**
  * the recipe reproduces HACC's banked weights exactly;
  * a bake with w ≡ 1 is the identity (0.0 m);
  * the bake adds only `hacc_w`.
* **Result [M]** (`RESULT_V8F_DAG_HACC.md`):
  * Against P10, accel reads **−0.644 / −0.847** (plain DAG: +0.807 / +1.444). No longitudinal row is separated-worse at either seed, and
    s1's yaw-rate row is gone.
  * The curvature / L2 / M1 excess roughly halves (curvature +0.017 / +0.047, L2 +0.011 / +0.028, M1 +0.024 / +0.039), but stays
    separated-worse.
  * ADE6 against P10 is −1.039 / −1.092 m.
  * **KEEP fails at seed 0:** +0.126 [+0.052, +0.200] m. It holds at seed 1: −0.105 m.
* **Outcome, as committed: COSTS POSITION.** The post-map is not adopted. Its committed next is the trained form of the same lever: OACC's
  acceleration term in DAG's fine-tune (box 2). The trade has now appeared on two routes (OACC on M2b, §7.16; HACC on DAG). On this rig,
  smoothing the longitudinal profile costs about 0.1 m of 6 s position at one seed. The two seeds disagree in sign, at about ±1 F.


### 7.22 FLAT-DAG: the flat control given the same DAgger recipe

* **Setup** (`PREREG_V8F_FLATDAG`, promoted at `9b91b654` before any FLAT-DAG bank, run or row). The question: does the hierarchy beat flat once
  BOTH get own-state supervision? Everything is read at T1-kin: m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, plain arms,
  self-action OPEN loop; the lead follows its logged future. R-T0 is at T0 (box 2's standard eval).
  * **Estimator:** paired episode-cluster bootstrap, B = 2,000, 141 episodes / 2,399 windows, per seed. It answers *"would another draw of
    EPISODES say this?"*. No replicate arm was trained, so a separated row is a result about one training run per seed (`H-ESTIM-SEED-1`).
    The planner is deterministic, so there is no inference-seed term.
  * **Arms** (box 2, all four through their launch gates, token commit `3101dbf1`). FLAT-DAG = FLAT_s fine-tuned 3,000 steps with DAG's recipe
    at `--dagger-p 0.1` on FLAT's own bank, `--o-interface cut` kept. FLAT-DAG-0 = the same argv at `--dagger-p 0`. Each bank holds 613,132
    records over 4,512 clips, COVERAGE 0.751 / 0.751. The swap share reached in training is 0.1011 (s0) and 0.1005 (s1), and 0 for both
    FLAT-DAG-0. DAG's T1-kin rollouts are the banked ones (`raw/2026-10-26-dagger-t1`).
  * **Queue order, disclosed.** FLAT-DAG ran second, after DAG-OACC's first attempt stopped at its gate asserter before anything trained.
    No row, bar or arm depends on the order.
* **Outcome [M]** (`RESULT_V8F_FLATDAG.md`; reader `raw/2026-10-27-flatdag/t1read/flatdag_read.py`; all 91 reader controls PASS; output
  `raw/2026-10-27-flatdag/read/FLATDAG_READ.json`, sha256 `7274b532…`):

| | s0 | s1 |
|---|---|---|
| PRIMARY H-T1 (ADE6 DAG − FLAT-DAG ≤ −0.2415 m, separated) | **MET: −0.3451 [−0.4421, −0.2506]** | not met: −0.0544 [−0.1605, +0.0547] |
| FLAT BETTER (ADE6 separated-worse) | no | no |
| family rows separated-worse, DAG − FLAT-DAG | **DK-TG +0.0734 [+0.013, +0.150], DK-VIOL +0.0370 [+0.015, +0.062]** | **accel MAE +0.8252 [+0.521, +1.165]** |
| STRATEGIC, paired | not computed: no paired DAG − FLAT-DAG instrument (never counts as passing) | not computed |

  The outcome is **NOT ESTABLISHED** (REFUSED, BLOCKED, FLAT BETTER and HIERARCHY BEATS FLAT do not apply). The PRIMARY fails at s1, and
  family rows are separated-worse at both seeds. Committed next, as written: *"B-2's margin was DAgger's, not the hierarchy's. The interface
  stays the defect; next is MIXO's committed T-side arm, under its own PREREG"*.
* **Controls** (C-F1 to C-F4, 91 checks, all PASS):
  * both banks pass K-B0…K-B5 with headers naming that seed's FLAT checkpoint, and the pair rule holds (argv identical but `--out`,
    `--dagger-p` and the bank flags);
  * **the cut survives the fine-tune:** the checkpoint config reads `o_interface: cut`, and oracle − own is exactly 0 [0, 0] on 2,399 windows
    at both seeds (max |ΔADE6| 0.0);
  * P10's floor reads exactly 0.0, and the reader reproduces the banked B-2 table (19 rows, max |diff| 0.0; DAG − FLAT −0.5957 / −0.3505).
* **The four families [M]** (T1-kin; point [95 % CI]; B / W = separated-better / -worse):

| family | row | DAG − FLAT-DAG s0 | s1 | FLAT-DAG − FLAT-DAG-0 s0 | s1 |
|---|---|---|---|---|---|
| PRIMARY | ADE6 (m) | **−0.3451 [−0.442, −0.251] B** | −0.0544 [−0.161, +0.055] – | −0.1522 [−0.218, −0.088] B | −0.1905 [−0.277, −0.120] B |
| LONGITUDINAL | speed MAE | −0.3241 [−0.443, −0.213] B | +0.0451 [−0.066, +0.151] – | +0.0737 W | +0.0600 – |
| | along-track MAE | −0.1621 [−0.258, −0.066] B | +0.0107 [−0.094, +0.126] – | −0.0427 – | −0.0893 B |
| | accel MAE | −0.1192 [−0.454, +0.213] – | **+0.8252 [+0.521, +1.165] W** | +0.3842 W | +0.2798 W |
| | DK-VIOL | **+0.0370 [+0.015, +0.062] W** | −0.0084 [−0.033, +0.013] – | +0.0168 – | +0.0319 W |
| | DK-HW | +0.0320 [−0.265, +0.265] – | +0.0028 [−0.218, +0.223] – | +0.1371 – | −0.1032 – |
| | DK-TG | **+0.0734 [+0.013, +0.150] W** | +0.0005 [−0.032, +0.031] – | −0.0074 – | −0.0103 – |
| LATERAL | heading MAE | −3.1694 [−4.257, −2.257] B | −1.3925 [−2.036, −0.869] B | −1.4594 B | −1.8141 B |
| | yaw-rate MAE | −11.6272 [−16.844, −7.298] B | −0.6107 [−3.019, +2.104] – | +0.4058 – | −2.0253 B |
| | curvature MAE | −0.0461 [−0.070, −0.025] B | −0.0037 [−0.019, +0.012] – | −0.0088 – | −0.0393 B |
| | cross-track MAE | −0.2989 [−0.372, −0.231] B | −0.0791 [−0.147, −0.015] B | −0.1897 B | −0.1998 B |
| | L1 lateral-accel MAE | −2.6848 [−3.923, −1.625] B | −0.1380 [−0.682, +0.419] – | +0.5423 W | +0.0526 – |
| | L2 curvature MAE (GT ≥ 5.2) | −0.0393 [−0.070, −0.015] B | −0.0068 [−0.025, +0.013] – | +0.0040 – | −0.0110 – |
| TACTICAL | TAC-LAT | +0.0600 [+0.033, +0.088] B | +0.0167 [−0.005, +0.040] – | +0.1263 B | +0.0863 B |
| | TAC-LON | +0.0554 [+0.025, +0.088] B | +0.0088 [−0.030, +0.047] – | +0.0092 – | +0.0042 – |
| | lat head F1 (H-T1-HEADS) | −0.0304 [−0.092, +0.028] – | +0.0070 [−0.041, +0.062] – | +0.1104 B | +0.0957 B |
| | lon head F1 (H-T1-HEADS) | +0.0107 [−0.020, +0.041] – | −0.0078 [−0.046, +0.033] – | +0.0340 B | +0.0691 B |
| STRATEGIC | S-POS@30 / @50 / @80 vs CTRV (m), per arm | DAG −1.72 / −4.48 / −9.69 B; FLAT-DAG −1.65 / −4.67 / −9.83 B | DAG −1.87 / −4.90 / −10.33 B; FLAT-DAG −1.63 / −4.62 / −10.29 B | per arm, not a contrast: FLAT-DAG-0 −1.04 / −3.58 / −7.87 B | per arm: FLAT-DAG-0 +0.21 – / −1.11 B / −5.03 B |
| crawl | M1 crawl share | −0.0115 [−0.022, −0.001] B | −0.0133 [−0.023, −0.004] B | −0.0082 B | −0.0058 B |

  ADE6 arm means (m, 2,399 windows / 141 episodes): DAG 2.2777 / 2.4565, FLAT-DAG 2.6228 / 2.5109, FLAT-DAG-0 2.7750 / 2.7014. D-FLAT's guards
  held: R-T0 FLAT-DAG − FLAT-DAG-0 +0.0254 [−0.0039, +0.0575] / +0.0082 [−0.0205, +0.0357] m, point ≤ +0.0604 at both seeds. The CIs of the
  D-FLAT columns other than ADE6 and the head rows are in `read/FLATDAG_READ.json` (`beside.per_seed.*.D-FLAT`). STRATEGIC is reported, not
  paused, for v8F.
* **What the numbers say** (beside the outcome; none of this moves it):
  1. **Most of B-2's margin was DAgger's.** DAG − FLAT was −0.596 / −0.351 m (B-2). Against a flat model given the same DAgger recipe it is
     −0.345 / −0.054 m, so the hierarchy keeps 58 % of its margin at s0 and 15 % at s1. The s0 contrast still clears 2F; the s1 one is not
     separated.
  2. **DAgger helps flat too, with the same guards held.** FLAT-DAG − FLAT-DAG-0 ADE6 −0.152 / −0.191 m, lat head F1 +0.110 / +0.096, lon
     head F1 +0.034 / +0.069, all separated; R-T0 held. Own-state supervision is a recipe win on both architectures, not a hierarchy
     property.
  3. **The swap's acceleration cost is the recipe's, on both architectures.** FLAT-DAG − FLAT-DAG-0 accel MAE +0.384 / +0.280 (W at both
     seeds); DAG2 − DAG2-0 (DAgger iteration 2 against its control) was +0.58 / +0.60, separated at both seeds (`RESULT_V8F_DAGGER2.md` §2). The s1 row that blocks the claim, DAG − FLAT-DAG accel +0.825, sits on DAG's worse seed (DAG −
     P10 accel +0.81 / +1.44, `RESULT_V8F_DAGGER2` §2). DAG-OACC is the arm aimed at this row (§7.23).
  4. **Where the hierarchy still leads at both seeds, the lead is in O's use of the code, not in T's decisions.** Heading MAE (−3.17 / −1.39),
     cross-track MAE (−0.30 / −0.08) and M1 crawl share (−0.012 / −0.013) are separated-better at both seeds. T's own heads do not differ (lat
     F1 −0.030 / +0.007, lon F1 +0.011 / −0.008, none separated). FLAT's O does not read T's code, so the most direct reading (interpretation,
     no bar) is that the interface contributes lateral precision. At s0 it comes with closer following (DK-TG +0.073, DK-VIOL +0.037, W).
  5. **The contrast moves by 0.29 m between seeds**, more than 2F. DAG's mean worsens by 0.18 m from s0 to s1 while FLAT-DAG's improves by
     0.11 m. With one training run per seed the interval cannot say which seed is typical. That is the training-run question
     (`H-ESTIM-SEED-1`), which the episode bootstrap does not answer.
* **Next (Rule Zero).**
  * The committed next resolves, from the register, to the T-side speed line. KINDX: speed is the carrier at both seeds (gain_speed +0.716 /
    +0.668 m, T0). SPEEDAVG: **MIXED**, the seeds' speed errors correlate 0.71–0.85, so the error is systematic, not variance. SPEEDCAL:
    **INFORMATION-LIMITED**, an affine map moves `v_a_ms` RMSE 1.92 → 1.33 m/s and ADE6 not at all, while the GT speed cells would cut ADE6 by
    0.72 / 0.67 m (T0). SPEEDCAL's committed next is perception (`PREREG_V8F_LORA.md`, projected at 470–494 h per seed on box 2), so its venue
    rule names the A40.
  * **Blocker, named.** The allocation is box 2 and the dev-box CPU. The A40 form of the perception arm waits on `D-V8F-SPEED-PERCEPTION`.
  * **What runs instead.** (a) A box-2-feasible form of the T-side speed lever, inside LORA's own venue rule, under its own PREREG:
    `PREREG_V8F_SPEEDREAD` commissions the cheapest experiment first (is the speed information in T's own input?), and it is pre-registered
    and not yet read. (b) DAG-OACC on box 2 (§7.23). (c) DAG-ORC, built and gated at the time of this read and waiting for the PI's go; it has
    since been re-based on DAG-OACC (§7.23).
  * **Instrument gaps this read exposed:** a paired DAG − FLAT-DAG S-POS row (STRATEGIC is per arm only), and one same-argv replicate per
    architecture, so that a 0.29 m seed swing can be read against a measured run-to-run floor.
* **Provenance.** Box 2: chain `chain_v8f_flatdag.sh` at tip `3101dbf1`, marker `ZZV8FMINI-DONE`; each run took about 37.6 min for 3,000 steps.
  Dev-box CPU: 4 / 4 T1-kin rollouts (258–261 s each) and the reader. Before launch the self-test, mutation proof and a dry run on the real
  config ran. Per-episode dumps, tick arrays, the four checkpoints and the bank shards stay on E: (`v8f-flatdag-t1/`, `v8f-flatdag/`), single
  copy, not in the repo.

### 7.23 DAG-OACC: the acceleration term inside DAgger's recipe fixes the longitudinal family and keeps position

* **Setup** (`PREREG_V8F_DAG_OACC` with **Amendment 1**, `A1-2026-10-27`, landed `db6a1a43` before any DAG-OACC T1-kin rollout or value).
  * **Arms** (box 2, r2 chain at tip `aeb54468`, both gates PASS). DAG-OACC_s = DAG_s's argv verbatim plus `--w-acc 49.6969`. W was computed
    once, by the `waccoacc` stage on DAG_s0, and C-O3 matched the offline recomputation. The swap share reached 0.1013 (s0) and 0.1008 (s1).
    Control = DAG_s, banked (a one-flag pair, C-O1). The first attempt stopped at its gate asserter before anything trained.
  * **Tier and estimator:** T1-kin (self-action OPEN loop; m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, plain arms; the lead
    follows its logged future). R-T0 is at T0. Paired episode-cluster bootstrap, B = 2,000, 141 episodes / 2,399 windows, per seed. The control
    is a different training run (DAG), and the training-run question is carried by the two seeds and F = 0.1207 m (`H-ESTIM-SEED-1`). The
    planner is deterministic, so there is no inference-seed term.
* **Outcome [M]** (`RESULT_V8F_DAG_OACC.md`; reader `raw/2026-10-27-dag-oacc/t1read/dagoacc_read.py`, coding the amendment; enumeration 0 unplaced
  cells; all 60 reader controls PASS; output `raw/2026-10-27-dag-oacc/read/DAGOACC_READ.json`, sha256 `9e6bd4f6…`):

| | s0 | s1 |
|---|---|---|
| PRIMARY-ACC (accel MAE, DAG-OACC − DAG, < 0 and separated) | **PASS: −1.6243 [−1.9766, −1.3236]** (means 1.2727 vs 2.8970 m/s²) | **PASS: −2.0450 [−2.5628, −1.6227]** (1.4885 vs 3.5335) |
| KEEP (ADE6, DAG-OACC − DAG, upper bound ≤ +0.0604 m) | PASS: −0.0356 [−0.1258, +0.0423] | PASS: **−0.1363 [−0.2057, −0.0731]**, separated-better |
| LON-CLEAR (no longitudinal row of DAG-OACC − P10 separated-worse) | clear | clear |
| Read A D1 (ADE6 vs P10 separated-better, ≤ −2F) | yes: −1.2012 | yes: −1.1231 |
| Read A D2 (no row separated-worse vs P10) | **fails**: yaw-rate, curvature, L2 | **fails**: yaw-rate, curvature, L2 |
| Read A D3 (every family has a separated-better row) | yes | yes |
| crawl rows (A1: DOMINANT also needs them to pass) | **fail**: M1 +0.0568 [+0.038, +0.080] | **fail**: M1 +0.0604 [+0.039, +0.086] |
| R-T0 (beside: T0 ADE6, DAG-OACC − DAG) | −0.0100 [−0.0329, +0.0130] | −0.0214 [−0.0536, +0.0077] |

  The outcome is **LONGITUDINAL FIXED** (PRIMARY-ACC and KEEP pass; LON-CLEAR at both seeds; not DOMINANT, because D2 and the crawl rows fail at
  both seeds). The unamended table gives the same word (`outcome_unamended`, descriptive), so the amendment moved no outcome here. Committed
  next, as amended: *the term is adopted and DAG-OACC becomes the base; next is the lever of the row still open*. The open rows are crawl rows,
  so the lever is **DAG-ORC re-based on DAG-OACC**, whose Amendment 1 is landed at `6cd741b6`, before any DAG-ORC run. Under Read A's own table
  (`PREREG_V8F_DAG_REREAD` §5) the same rows read MARGIN, NOT DOMINANT.
* **The four families [M]** (T1-kin; point [95 % CI]; B / W = separated-better / -worse):

| family | row | DAG-OACC − P10 s0 | s1 | DAG-OACC − DAG s0 | s1 |
|---|---|---|---|---|---|
| PRIMARY | ADE6 (m) | −1.2012 [−1.395, −1.019] B | −1.1231 [−1.330, −0.916] B | −0.0356 [−0.126, +0.042] – | −0.1363 [−0.206, −0.073] B |
| LONGITUDINAL | speed MAE | −0.7761 [−0.981, −0.600] B | −0.7030 [−0.909, −0.523] B | −0.1996 [−0.281, −0.124] B | −0.4010 [−0.530, −0.295] B |
| | along-track MAE | −0.6809 [−0.844, −0.521] B | −0.5876 [−0.770, −0.412] B | −0.0527 [−0.153, +0.030] – | −0.1304 [−0.254, −0.044] B |
| | accel MAE | −0.8172 [−1.177, −0.500] B | −0.6014 [−0.906, −0.335] B | **−1.6243 [−1.977, −1.324] B** | **−2.0450 [−2.563, −1.623] B** |
| | DK-VIOL | +0.0017 [−0.023, +0.027] – | −0.0034 [−0.024, +0.018] – | −0.0336 [−0.059, −0.012] B | −0.0084 [−0.030, +0.009] – |
| | DK-HW | −0.3318 [−0.694, −0.024] B | −0.4466 [−0.820, −0.169] B | −0.2475 [−0.458, −0.041] B | −0.2080 [−0.439, +0.016] – |
| | DK-TG | −0.0568 [−0.123, +0.016] – | −0.0724 [−0.138, −0.009] B | −0.0725 [−0.145, −0.012] B | +0.0210 [−0.020, +0.086] – |
| LATERAL | heading MAE | −2.5920 [−3.239, −1.975] B | −2.0539 [−2.709, −1.371] B | **+0.7674 [+0.348, +1.265] W** | −0.4548 [−1.093, +0.101] – |
| | yaw-rate MAE | **+3.4726 [+0.660, +6.331] W** | **+8.6844 [+5.169, +12.574] W** | **+3.5126 [+1.429, +5.878] W** | +0.0292 [−1.762, +1.879] – |
| | curvature MAE | **+0.0854 [+0.058, +0.117] W** | **+0.1352 [+0.089, +0.192] W** | **+0.0241 [+0.011, +0.039] W** | +0.0218 [−0.003, +0.053] – |
| | cross-track MAE | −0.9620 [−1.108, −0.832] B | −0.9742 [−1.115, −0.841] B | +0.0173 [−0.015, +0.051] – | −0.0473 [−0.095, +0.020] – |
| | L1 lateral-accel MAE | −2.4618 [−3.810, −1.292] B | −1.8473 [−2.962, −0.850] B | −0.0296 [−0.308, +0.258] – | −0.7120 [−1.204, −0.234] B |
| | L2 curvature MAE (GT ≥ 5.2) | **+0.0443 [+0.023, +0.074] W** | **+0.0529 [+0.028, +0.085] W** | +0.0158 [−0.001, +0.041] – | −0.0075 [−0.030, +0.014] – |
| TACTICAL | TAC-LAT | +0.1905 [+0.151, +0.232] B | +0.1972 [+0.160, +0.236] B | −0.0021 [−0.019, +0.015] – | +0.0121 [−0.010, +0.033] – |
| | TAC-LON | +0.1396 [+0.077, +0.200] B | +0.1371 [+0.075, +0.200] B | +0.0108 [−0.017, +0.039] – | +0.0229 [−0.004, +0.051] – |
| STRATEGIC | S-POS@30 / @50 / @80 vs CTRV (m), per arm | DAG-OACC −1.74 / −4.53 / −9.56 B (DAG −1.72 / −4.48 / −9.69) | DAG-OACC −1.83 / −4.61 / −9.92 B (DAG −1.87 / −4.90 / −10.33) | not computed: no paired S-POS instrument | |
| crawl | M1 crawl share | **+0.0568 [+0.038, +0.080] W** | **+0.0604 [+0.039, +0.086] W** | **+0.0102 [+0.003, +0.017] W** | −0.0011 [−0.007, +0.005] – |

  STRATEGIC is reported, not paused, for v8F. The T0 four-family leaves (box 2's standard eval, DAG-OACC against DAG) are in `DAGOACC_READ.json`
  (`beside.per_seed.*.T0.four_families_T0`). They are descriptive only: no paired T0 four-family instrument exists. K4 is NOT RUN, because the
  PREREG scheduled no logged-bucket rollout or fresh-age T0 eval.
* **What the numbers say** (beside the outcome; none of this moves it):
  1. **The acceleration error is fixed, and position comes along.** Accel MAE falls by 56 % / 58 % (2.90 → 1.27 and 3.53 → 1.49 m/s²). Against
     the kinematic floor P10 it flips from separated-worse for DAG (+0.81 / +1.44, `RESULT_V8F_DAGGER2` §2) to separated-better (−0.82 /
     −0.60). ADE6 is kept at s0 and improves at s1 (−0.136, separated). The effect is 2.6–3.2 times the cross-seed spread of DAG's own accel row
     (0.63 m/s²). That comparison is context, with no bar.
  2. **Every longitudinal row is clean against P10 at both seeds.** Speed, along-track and accel MAE and headway are separated-better; the time
     gap is better or not separated; DK-VIOL is not separated. Against DAG, speed MAE is separated-better at both seeds (−0.20 / −0.40), so the
     term fixed target speed as well as smoothness.
  3. **What stays open is the crawl set, and at s0 the term costs some lateral.** Against P10, yaw-rate, curvature, L2 and M1 are separated-worse
     at both seeds. Against DAG, s0 reads heading +0.77, yaw-rate +3.51, curvature +0.024 and M1 +0.010, all separated-worse; s1 reads none of
     them worse. The acceleration term did not repair the crawl, so NEXT-CRAWL's COMPOSE condition (DAG-OACC − DAG separated-better on the crawl
     rows) is not met.
  4. **The hierarchy's decisions are untouched.** TAC-LAT and TAC-LON against DAG are not separated at either seed. The lever acts on O's
     execution, as designed.
* **Next (Rule Zero).**
  * **Adopted.** DAG-OACC is the base.
  * **The crawl rows go to DAG-ORC, re-based** (`PREREG_V8F_DAG_ORC` Amendment 1, `6cd741b6`). Form: COMPOSE. A = DAG-OACC's argv plus O-RECOVER's
    perturbation, behind a new default-off composition flag. CTRL = DAG-OACC, banked. REP is new. Launch blocker: the composition flag and its
    reachability proofs (in build at the read). It is pre-registered and not yet read.
  * **DAgger iteration 2**, the other lever v30 named for the crawl rows, read HARMS (`RESULT_V8F_DAGGER2.md` §1: the crawl rows stay separated-worse against P10 at
    both seeds, and yaw-rate MAE turns newly separated-worse at s0), so it was not adopted.
  * **Evidence videos.** The PREREG re-renders them only on DOMINANT, so they are not re-rendered on this read.
* **Provenance.** Box 2: chain `chain_v8f_dagoacc_r2.sh` ran 20:21–21:51, marker `ZZV8FMINI-DONE`; each run took 41 and 42 min with its T0 eval. The
  r2 asserter is banked at `raw/2026-10-27-oaccflat-queue/r2/`. Dev-box CPU: 2 / 2 T1-kin rollouts and the reader; DAG's and P10's runs are the
  banked ones. Reader checks before launch: self-test 287 / 287; mutation proof 127 / 127 killable mutants killed, with 2 real-run-only branches
  UNVERIFIED; real-config dry run `ZZDAGOACCT1-DRY-OK`. Per-episode dumps, tick arrays and checkpoints stay on E: (`v8f-dagoacc-t1/`,
  `v8f-dagoacc/`), single copy, not in the repo.

---

## 8. Discussion

### 8.1 What is established

1. **The hierarchy runs end to end.** O is steered by T's code (donor code costs 4.4–6.4 m; **[M]**, 4 of 4 checkpoints), T's lateral head reads the route (shuffle 0.82–0.84 → 0.30 F1), and S's checkpoints fall by 4.2 m (50 m) when the nav is shuffled. The typed interface is *used*.
2. **A learned-kinematic floor can be beaten at 6 s**, open loop, with a 7.2 M-parameter head on frozen features (K-1 PASS on M2b, 13–17 %).
3. **Where the remaining 6 s headroom sits is localised:** in the *joint* accuracy of 26 continuous constraint cells (not the decisions), convex in that accuracy, and not reachable by any single input, loss or data lever tested (§7.7).
4. **The main obstacle is on O's side of the interface,** not only T's: O has only seen all-GT or all-own codes (§5.2, §7.6).
5. The variance structure of this rig is quantified, and it dictates experimental design (§5.3): replicates and seeds, not windows.
6. **The margin over the floor survives the model's own motion** (T1-kin, §7.15): 0.83 / 0.79 m of 6 s ADE at both seeds, and 0.79 / 0.71 m with the cameras frozen at the first frame. It holds with a speed bucket built from the past alone (0.77 / 0.73 m), and it is not dominance.
7. **The strategic level works on the model's own motion** (§7.15): S's checkpoints beat a kinematic arc at 30 / 50 / 80 m, separated at both seeds.
8. **Own-state supervision repairs T on the model's own motion** (§7.19): DAgger at p = 0.1 lifts T's lateral and longitudinal macro-F1 at T1-kin past bars that include the run-to-run replicate floor, at both seeds, and cuts the executed 6 s ADE by 0.54 / 0.40 m against its own fine-tune control.
9. **On its own motion the model beats the kinematic floor by 8–10 F of 6 s ADE** (§7.20). It does so with better tactical agreement and halved strategic checkpoint error, and with better speed, heading and cross-track. It also beats its flat control at T1-kin with DAgger. It is not dominant: curvature, L2 and the launch crawl remain worse than the floor.
10. **Given the same DAgger recipe, the hierarchy's lead over flat is not established** (§7.22; T1-kin, paired episode-cluster bootstrap). Own-state supervision helps the flat control as well (ADE6 −0.152 / −0.191 m, separated). DAG's margin over it is −0.3451 m at seed 0 (past the bar) and −0.0544 m at seed 1 (not separated), with family rows separated-worse at both seeds, so item 9's comparison with FLAT is confounded by DAgger. The hierarchy keeps 58 % of its B-2 margin at seed 0 and 15 % at seed 1.
11. **The acceleration term fixes the longitudinal family on DAG's recipe, and position comes along** (§7.23; T1-kin, same estimator). DAG-OACC is the adopted base: accel MAE falls by 56 % / 58 % against DAG (separated), ADE6 is kept, and no longitudinal row is separated-worse than P10 at either seed. The open gap is the crawl set (yaw-rate, curvature, L2 and M1 against P10); the arms aimed at it are pre-registered and not yet read.

### 8.2 What is not established

* That the **hierarchy** (rather than the frozen prior, the features or the labels) is responsible for the K-1 gain. The capacity-matched FLAT control (`PREREG_V8F_FLAT`) reads **NOT ESTABLISHED**: M2b − FLAT −0.036 / −0.003 m ADE6, n.s. (§7.15).
* That the **hierarchy, rather than the DAgger recipe, earns DAG's margin over flat**. Given the same recipe, DAG − FLAT-DAG ADE6 is −0.3451 [−0.4421, −0.2506] m at seed 0 (past the bar) and −0.0544 [−0.1605, +0.0547] m at seed 1 (not separated), and family rows are separated-worse at both seeds. The outcome is NOT ESTABLISHED (§7.22; T1-kin), and one training run per seed cannot say which seed is typical.
* That **T's decisions hold on the model's own motion**: they do not. Lateral F1 falls from 0.85 to 0.61 / 0.63 at T1 (§7.15). O-RECOVER moved the lateral errors on the model's own motion but read HARMS on position (§7.18). DAgger (§7.19) lifts lat F1 at ticks ≥ 1 to 0.60 / 0.64 (DAG-0 0.49 / 0.52), separated and past its bar. The gap to tick 0's 0.85 narrows; it is not closed.
* That the **strategic layer** adds anything over a nav-only kinematic extrapolation *on the deployed schedule*: at its own tick it does (−2.4 to −2.5 m at 50 m, §7.8); on the deployed schedule its checkpoint metric includes the travel since its tick, and whether that staleness costs the plan is unmeasured.
* Any **driving** claim: all numbers are open loop (T0/T1), and no T2 closed-loop run exists. The T1-kin read (§7.15) keeps the human's cameras. Its speed input can be built from the past alone without losing the margin, but frozen cameras together with that causal bucket are not measured. Without the post-map, the raw arms are worse than the floor at 2 s and in acceleration.
* That any T-side lever can reach the headroom. T_ATTRIB's ceilings are ≤ 0.17 m for every stratum considered **[E]**, below $2F$.
* Transfer: one corpus, one recording-session split, no out-of-distribution evaluation.

### 8.3 The gap to "dominant driving results"

**Update (v31).** On the model's own motion (T1-kin), the adopted base is now DAG-OACC, and its open rows are lateral and crawl rows:
* every longitudinal row is clean against the floor P10 at both seeds, both tactical rows are separated-better, S's checkpoints beat a kinematic arc at 30 / 50 / 80 m, and ADE6 is kept (§7.23);
* yaw-rate, curvature, L2 and the launch crawl M1 remain separated-worse at both seeds, so the arm is not DOMINANT;
* the arm aimed at that crawl set is DAG-ORC re-based on DAG-OACC (`PREREG_V8F_DAG_ORC` Amendment 1, `6cd741b6`); SPEEDREAD (`PREREG_V8F_SPEEDREAD`) asks whether the speed information O needs is in T's own input; both are pre-registered and not yet read;
* DAgger iteration 2, the other lever v30 named, read HARMS (`RESULT_V8F_DAGGER2.md` §1), so it was not adopted.

Attribution to the hierarchy is not established: against a flat model given the same DAgger recipe, DAG's ADE6 margin clears its bar at seed 0 and not at seed 1 (§7.22).

**Update (v30).** On the model's own motion (T1-kin), DAG with HACC's post-map is two row-families away from dominance over the floor:
* every longitudinal, tactical and strategic row is better or not separated;
* curvature / L2 and the launch crawl M1 remain separated-worse;
* its position guard fails at one seed (§7.21).

At v30, the levers aimed at those rows were DAgger iteration 2 and DAG-OACC, both on box 2, and attribution to the hierarchy waited on FLAT-DAG; v31 reads DAG-OACC and FLAT-DAG (above). The PI's bar is **met on one axis only**. In open loop, the deployed form of the model now dominates the programme's own kinematic prior on every row, at both seeds: HACC, §7.12, row 1m. The rest of the bar is not met: (i) that prior is the only benchmark comparison, and **no published baseline exists on our split**, so we make **no state-of-the-art claim**; (ii) the raw learned arms still lose to the prior at 2 s and in acceleration, and dominance needs the train-fitted post-map. HACC turns the acceleration loss into a gain (−0.022 / −0.040 m/s²) and the consistency cost into a gain (−0.075 / −0.046), at +0.015 / +0.031 m ADE6 against H; (iii) the headroom measured by the oracle (about 1.1 m) is a diagnostic upper bound, and the realised arms moved 6 s ADE by at most 0.43 m against the floor and by under 0.08 m for every single-lever refinement; (iv) there is no closed-loop evidence: in the self-action open loop (T1-kin), the model keeps a 0.8 m margin but loses dominance on low-speed curvature; the margin holds with a speed input built from the past alone (§7.15). What the report does contribute is a **reproducible diagnostic method** (interface oracles, dose response, per-kind partition, variance accounting) and a precise localisation of what blocks the next gain. Closing the gap needs, at minimum, a closed-loop (T2) or published-protocol (NAVSIM-style) evaluation, and gains on all four families at both horizons.

### 8.4 Next levers, each with its pre-registered (or proposed) bar

| # | lever | bar and outcomes | status |
|---|---|---|---|
| 1 | **MIXO** (O trained on per-kind mixed codes) | PRIMARY $\le-2F=-0.2415$ m, sep., seed 0 then 1. PASS ⇒ X becomes the base and KIND is re-read on X. FAIL, O LEARNED ($r(0.5)$ rises ≥ 0.10 over B's 0.36 / 0.21) ⇒ T-side arm on the best kind. FAIL, O DID NOT LEARN ⇒ O-capacity or O-input arm (an architecture change: PI go) | **FAIL, O LEARNED** (s0; §7.10): ADE6 X − B +0.022 [−0.059, +0.101] m; r(0.5) 0.644 against 0.365; next: `PREREG_V8F_KINDX` (KIND re-read on X, s0 screen) |
| 1b | **FLAT control** (O reads constants in place of S's checkpoints and T's code; S and T still trained; O keeps nav) | PRIMARY M2b − FLAT $\le-2F$, sep., at BOTH seeds. Outcomes: HIERARCHY BEATS FLAT / NOT ESTABLISHED / FLAT BETTER / SEEDS DISAGREE | **NOT ESTABLISHED** (`RESULT_V8F_FLAT.md`, `f5c9711d`): ADE6 M2b − FLAT −0.036 [−0.113, +0.035] / −0.003 [−0.081, +0.076] m. The GT-code oracle carries 1.2 m, so the code at inference is the defect; next is the MIXO repeat of the read |
| 1c | **HBLEND**: per-step, per-axis shrinkage of O's residual toward its own prior, $\tau_H=\mathrm{up}(P10)+w(t)\odot(\tau-\mathrm{up}(P10))$, with $w$ fitted in closed form on TRAINING windows, clipped to [0, 1] and isotonic. Zero training, deployable | PRIMARY-S ADE2 H − B $<0$, sep., both seeds; PRIMARY-N ADE6 upper bound $\le+F/2$; SECONDARY accel MAE $<0$, sep. | **PASSED at both seeds** (`RESULT_V8F_HBLEND.md`, `c5ac9a9f`): ADE2 H − B −0.103 / −0.078 m, ADE6 −0.207 / −0.070 m, accel −0.464 / −0.228 m/s². It COSTS CONSISTENCY(a) 'any' +0.101 / +0.075 (§7.12) |
| 1d | **HPROJ**: H + the posted-limit hard projection | PRIMARY-C 'any' HP − B upper bound $\le+0.02$ AND ADE6 / ADE2 kept | **FAILED** (`RESULT_V8F_HPROJ.md`, `305386b4`): 'any' +0.054 / +0.031. The speed cap that binds is the human's own v_hi, which a posted-limit projection cannot see |
| 1e | **HGATE**: H re-fitted on unconstrained training windows, used only where T predicts no FOLLOW / STOP / TURN and no exceedance of its own v_hi | the same two bars, both seeds | **FAILED** (`RESULT_V8F_HGATE.md`, `583cfc02`): 'any' HG − B +0.020 [+0.008, +0.034] / +0.029 [+0.018, +0.040] against an upper-bound bar of +0.02; ADE6 −0.037 / −0.027 m and ADE2 −0.020 / −0.021 m, separated. T's predicted v_hi is the weak trigger (precision 0.06 / 0.14). The post-processing line ends; the lever moves to T |
| 1f | **HSPEC**: H's residual also shrunk per cosine coefficient (stage 2: a second-difference objective, weight by a train-only rule) | dominance over P10 on every family at both seeds; ADE6 kept within $F/2$ of H | **SMOOTHER, NOT DOMINANT** (`RESULT_V8F_HSPEC.md`, `17fc9abb`): HS − P10 ADE6 −0.511 / −0.490 m and ADE2 −0.005 / −0.005 m, separated; accel +0.026 (s0, separated-worse) / +0.006 (ns); curvature not separated; every other family row not worse. Weights fitted on the eval itself still cannot reach P10's accel at s0: O's residual has no acceleration skill |
| 1g | **OACC**: O trained with an acceleration-matching term (second difference of its plan against the GT's; weight fixed by a pre-launch rule) | accel MAE X − B < 0, separated; ADE6 within $F/2$; the H-blended X dominates P10 on HSPEC's row | **read 2026-10-26: COSTS POSITION** (seed 0; ACC passed, KEEP and DOMINANCE failed, §7.16); seed 1 not trained, as committed; `RESULT_V8F_OACC.md`; pre-registered `PREREG_V8F_OACC` (`17fc9abb`) with Amendment 1 (`ea67b7a0`), launch gate PASS (`bf43fd58`), training began 2026-10-26 06:13 |
| 1h | **KHYP**: 7 one-hot longitudinal codes through O, then a selector on model outputs | SEL-LEARN − B ≤ −F, separated, both seeds | **HEADROOM, NO SELECTOR** (`RESULT_V8F_KHYP.md`, `4a59a711`): the oracle best of 7 is −0.42 / −0.36 m ADE6 (T0 headroom), but the deployable selector gives −0.098 (s0) / +0.020 (s1). At s0 it equals "always class 5", so the gain is a bias correction of T's longitudinal decision, not selection |
| 1i | **KSCORE**: a scorer trained on the full training set (B frozen) reading O's 7 hypotheses, T's softmax, the scene and ego | ≤ −F vs B AND better than the best train-chosen constant class, both separated, both seeds | **NOT PREDICTABLE** (`RESULT_V8F_KSCORE.md`, `de270934`): KSCORE − B −0.100 [−0.146, −0.058] (s0, separated, 0.021 m short of −F) / +0.004 [−0.027, +0.032] (s1); KSCORE − SEL-CONST −0.0005 / −0.0072 m, not separated. On the full training set (306,566 windows), the s0 gain is still the constant class 5, picked on 86.6 % of windows, and at s0 it is separated-worse on CONSISTENCY(a) and LEAD-VIOL. The oracle's 0.36–0.42 m are not reachable by choosing among O's plans from B's frozen features. The longitudinal lever moves to O's own speed profile (row 1g) |
| 1j | **KINDX**: KIND's per-kind oracle re-read on X (MIXO's s0 checkpoint), now that O consumes partial codes | CANDIDATE: some kind's gain ≥ 2F, separated at s0 ⇒ train X at s1 and re-read; NO CANDIDATE ⇒ longitudinal line stays top | **CANDIDATE: SPEED** (`4070e29a`, T0, s0): gain_speed(X) +0.716 [+0.623, +0.818] m (B: +0.270); every other kind ≤ 0.11 m; **seed 1: CARRIER (speed)** (`02e13a10`): gain_speed(X_s1) +0.668 [+0.575, +0.761] m (B_s1 +0.185); no other kind clears 2F at either seed. B's tte / geom gains are negative at s1 (−0.315 / −0.421), X's positive at both: MIXO's mechanism replicates |
| 1k | **SPEEDAVG**: T's five speed cells averaged across the two seeds' T at inference (everything else each seed's own); zero training | PRIMARY ADE6 ≤ −F, separated, at BOTH seeds; speed / accel / distance keeping / LEAD-VIOL not separated-worse | **MIXED** (`RESULT_V8F_SPEEDAVG.md`, `41bfca58`, T1-class): −0.041 [−0.073, −0.011] (s0, separated) / +0.019 [−0.012, +0.048] (s1); s1's accel MAE is separated-worse (+0.017). The two seeds' speed-cell **errors correlate 0.71–0.85**, so averaging cuts RMSE by only 3–12 %: T's speed error is systematic, not variance |
| 1l | **SPEEDCAL**: an affine map per speed cell (10 numbers per seed), fitted on HBLEND's TRAINING windows and applied to T's own output | PRIMARY ≤ −F, separated, both seeds; CEILING (the eval-fitted map, T0) separates *head bias* from *information-limited* | **INFORMATION-LIMITED** (`RESULT_V8F_SPEEDCAL.md`, `bb85eff7`, T1-class; CEILING T0): +0.037 [−0.010, +0.086] (s0) / −0.006 [−0.016, +0.004] (s1), separated at neither seed; even the eval-fitted map stays above −F (+0.030 / −0.008). The map does cut the speed-cell RMSE (`v_a_ms` 1.92 → 1.33 m/s at s0) while ADE6 does not move. The speed error O responds to is per-window, not a head bias: the lever moves to perception (row 2). Independently re-read |
| 1m | **HACC**: H's per-step weight re-fitted on TRAINING windows with a second-difference (accel) term and no monotone constraint; λ by a train-only rule; zero training | HSPEC's dominance row over P10 at BOTH seeds; ADE6 within F/2 of H | **DOMINATES P10** (`RESULT_V8F_HACC.md`, `c8ea1623`, T1-class): every row separated-better or not separated at both seeds, nothing separated-worse. ADE6 −0.503 / −0.471 m, accel MAE −0.022 / −0.040 m/s², curvature −0.0056 / −0.0051. KEEP vs H passes, but s1 is thin (+0.0586 against +0.0604). Independently re-derived. Folded into `forward_infer` as an optional buffer, absent by default and bit-identical (`ba27d62d`); the videos are re-rendered on HACC (v3). At T1-kin: **T1 BEATS THE FLOOR** (row 6, §7.15) |
| 1n | **HACC-LAT** (zero training): arm A, HACC's weights and λ rule re-fitted on T1-kin ticks of TRAINING windows; arm B, the cross axis falls back to P10 below a train-chosen speed gate | T1 DOMINANCE over P10 at both seeds, and ADE6 within F/2 of HACC | **NOT FIXED** (`RESULT_V8F_HACC_LAT.md`, T1): arm A −0.22 / +0.05 m ADE6 against P10, far worse than HACC; arm B −0.88 / −0.76 m, but curvature is still worse. Followed by the noise floor and `PREREG_V8F_T1_LATCOND` (CONDITIONED LATERAL DEFICIT, §7.15) |
| 1o | **HACC-MC** (zero training): HACC's along weight held constant over the executed knots 1–12; a 5× steeper ramp is the deliberate-regression arm | FIXED-KEEP: L2 (MC-H − P10) not separated-worse, MC-H − HACC ADE6 upper bound ≤ +F/2 and the crawl fraction M1 below HACC's, all at both seeds | **read 2026-10-26: MECHANISM-NOT-ENOUGH** (§7.17; all controls PASS after Amendment 2, which is post-data and control-only; pre-registered with Amendment 1 at `19d7e069`); committed next: O-RECOVER |
| 1p | **T1 heads**: S and T outputs on the model's own motion, scored at the matched route point | S − CTRV ≤ 0 separated at every arc; T F1 Δ within 2F | **S ADDS ROUTE / T DEGRADES** (`RESULT_V8F_T1_HEADS.md`, `c96f3b69`, T1): S − CTRV −1.18 / −3.27 / −7.79 m (s0); lat F1 0.848 → 0.614 / 0.858 → 0.632 |
| 1q | **O-RECOVER**: M2b fine-tuned 3,000 steps on perturbed ego states (ORC) against the same fine-tune unperturbed (ORC-0) | ROBUST: T1 lat AND lon F1 ORC − ORC-0 ≥ 2F, separated, both seeds; T1 ADE6 not separated-worse; T0 ADE6 ≤ +F/2 | **read 2026-10-26: HARMS** (§7.18; `RESULT_V8F_ORECOVER.md`, `8c10066a`; 104 / 104 controls PASS; pre-registered at `28e28c88` with Amendments 1–4). R-T0 +0.110 / +0.085 m against +F/2; R-ADE separated-worse at s0. T1-kin LATERAL separated-better at both seeds. Not adopted |
| 1r | **DAgger** (p = 0.1): M2b fine-tuned with 10 % of samples swapped for M2b's own T1-kin states at ticks 1–2 (TRAIN only, route-paired), against the same fine-tune with p = 0 (DAG-0) | the PREREG's outcome table; bar max(2F, 2·RF), with RF read from DAG-0 against ORC-0 | **read 2026-10-27: ROBUST** (§7.19; `DAG_READ.json`, 176 / 176 controls PASS; Amendment 2, post-data and control-only, before the one re-read). R-TLAT +0.108 / +0.118, R-TLON +0.055 / +0.081, both clearing at both seeds; T1-kin ADE6 −0.54 / −0.40 m. Adopted as the base |
| 1s | **DAG re-reads** (zero training): (A) DAG − P10 at T1-kin on all four families; (B) DAG − FLAT at FLAT's tier and at T1-kin | A DOMINANT: ADE6 separated-better by ≥ 2F, no family row separated-worse, and a separated-better row in every family, at both seeds. B ESTABLISHED: ADE6 ≤ −2F, separated at both seeds; B at T1-kin is confounded by DAgger and is attributed only by FLAT-DAG | **A: MARGIN, NOT DOMINANT; B-1: NOT ESTABLISHED; B-2: ESTABLISHED under a post-data control amendment** (§7.20; `RESULT_V8F_DAG_REREAD.md`) |
| 1t | **DAG-HACC** (zero training): HACC's rule re-fitted on DAG, read on Read A with KEEP | DOMINANT / LONGITUDINAL FIXED / NOT FIXED / COSTS POSITION (KEEP: ADE6 upper bound ≤ +F/2 vs DAG) | **COSTS POSITION** (§7.21; `RESULT_V8F_DAG_HACC.md`): every longitudinal row cleared, KEEP +0.126 m at s0. Next: DAG-OACC, DAgger iteration 2 and FLAT-DAG on box 2 (PI go); DAG-OACC and FLAT-DAG are read in rows 1u and 1v |
| 1u | **FLAT-DAG**: FLAT given DAG's recipe (3,000 steps, `--dagger-p 0.1` on FLAT's own bank, `--o-interface cut` kept), against FLAT-DAG-0 (`--dagger-p 0`) and DAG | PRIMARY H-T1: ADE6 DAG − FLAT-DAG $\le-2F=-0.2415$ m, separated, at both seeds. HIERARCHY BEATS FLAT also needs no family row separated-worse; the order is REFUSED, BLOCKED, FLAT BETTER, HIERARCHY BEATS FLAT, NOT ESTABLISHED | **NOT ESTABLISHED** (§7.22; `RESULT_V8F_FLATDAG.md`, 91 / 91 controls PASS, T1-kin): −0.3451 [−0.4421, −0.2506] m (s0, met) / −0.0544 [−0.1605, +0.0547] m (s1, not met); separated-worse family rows: DK-TG +0.0734 and DK-VIOL +0.0370 at s0, accel MAE +0.8252 at s1. FLAT-DAG − FLAT-DAG-0 ADE6 −0.152 / −0.191 m. Next: SPEEDREAD (row 1x) |
| 1v | **DAG-OACC**: DAG's fine-tune argv verbatim plus `--w-acc 49.6969` (OACC's acceleration term), against DAG (banked) | PRIMARY-ACC accel MAE DAG-OACC − DAG < 0, separated; KEEP ADE6 upper bound ≤ +0.0604 m (+F/2); LON-CLEAR no longitudinal row of DAG-OACC − P10 separated-worse; all at both seeds. DOMINANT also needs Read A's D1, D2 and D3 and the crawl rows (Amendment 1, `A1-2026-10-27`) | **LONGITUDINAL FIXED** (§7.23; `RESULT_V8F_DAG_OACC.md`, 60 / 60 controls PASS, T1-kin): accel MAE −1.6243 [−1.9766, −1.3236] / −2.0450 [−2.5628, −1.6227]; KEEP PASS (−0.0356 / −0.1363 m); LON-CLEAR clear at both seeds; not DOMINANT (D2 and the crawl rows fail at both seeds). Adopted as the base. Next: DAG-ORC re-based on DAG-OACC (row 1w) |
| 1w | **DAG-ORC, re-based on DAG-OACC** (COMPOSE: DAG-OACC's argv plus O-RECOVER's perturbation behind a new default-off composition flag; CTRL = DAG-OACC, banked; REP new) | the PREREG's outcome table (`PREREG_V8F_DAG_ORC` Amendment 1, `6cd741b6`), aimed at the crawl rows: yaw-rate, curvature, L2 and M1 | **pre-registered, not yet read**. Launch blocker: the composition flag and its reachability proofs |
| 1x | **SPEEDREAD**: is the per-window speed information O needs present in T's own input? A read-out of the 5 GT speed cells, computed from T's input and fitted on training clips only, replaces T's speed cells in the code O reads; no training of the driving model; dev-box CPU | the PREREG's decision table: INFO IN INPUT / PARTIAL / NOT IN INPUT / MIXED (`PREREG_V8F_SPEEDREAD`, claim row `H-V8F-SPEEDREAD-1`) | **pre-registered, not yet read** (Amendment 1, `A1-2026-10-27`, before any SPEEDREAD number). Commissioned by FLAT-DAG's committed next (`RESULT_V8F_FLATDAG.md` §4 (a)); it runs first because it is the cheapest experiment that separates the speed being in T's input from its not being there |
| 2 | **LoRA on DINOv3** | PRIMARY $<-2F$, sep., both seeds; MECHANISM latcons RMSE fall ≥ 12.5 %. Prior **LOW [E]**: a pass needs T about 25 % of the way to the GT (≈ 0.19 m) while no lever has moved T by more than 6 % | pre-registered; AR-2 passed; 470 h/seed on box 2 ⇒ A40. Of the A40 readiness gate, AR-1 (the mini beats P10 at two seeds) now holds through HACC and AR-2 passed; AR-3 (the launch gate) is being dry-run on CPU. SPEEDCAL makes this the speed lever; the A40 go is asked for |
| 3 | **Strategic read under `--ages fresh`** (zero training) | Q-LAYER fresh S − baseline at 50 m; Q-STALE fresh − cycle | **STALENESS ARTEFACT** (`RESULT_V8F_SFRESH.md`, `f547c11b`): Q-LAYER −2.43 / −2.53 m, Q-STALE −5.72 / −5.73 m, separated at both seeds. S adds over nav-only kinematics; the deployed tick schedule is the strategic lever |
| 4 | **Seed replicates** of any arm that moves a headline | the cheapest discriminating experiment on this rig (§5.3) | standing |
| 5 | Longitudinal speed decision (multimodal / winner-take-all output) | HYPOTHESIS (`H-V8F-LON-1`: oracle per-window speed scale is worth 0.20–0.39 m) | needs PREREG and a PI go |
| 6 | **T1-kin** (self-action open loop) read of HACC; T2 (closed loop) | `PREREG_V8F_T1KIN_HACC`: PRIMARY HACC − P10 ADE6 ≤ −2F, separated, both seeds; DOMINANCE adds no four-family row worse | **T1 BEATS THE FLOOR** (`RESULT_V8F_T1KIN_HACC.md`, `96604184`, T1): ADE6 −0.829 / −0.795 m, separated; curvature and s0 yaw rate separated-worse (low speed). With the speed bucket built from the past 4 s only (`PREREG_V8F_T1KIN_CAUSALBUCKET`, `76c7d98e`): **CAUSAL KEEPS THE MARGIN**, −0.768 / −0.734 m, separated. With the cameras frozen as well (`PREREG_V8F_T1KIN_STRICT`): **STRICT MARGIN**, −0.728 / −0.641 m. T2 is not provisioned |

Deprioritised with reasons: D1 (event-stratified batches; mechanism close to WC's), stacking GRID + HIST (their ADE gains sit inside the seed floor, R66), a 2× data arm (predicted ≈ 6 %), T-side single-kind arms (KIND cannot rank them). The D3 data extension has a scoping note only (`R/raw/2026-10-25-d3-scoping/D3_SCOPING.md`) and no pre-registration.

---

## 9. Limitations and threats to validity

1. **Tier.** Everything is open loop. The own-code evaluation has the ego state arriving from the recording; the oracles read the future. A driving claim needs T2.
2. **Estimator scope.** Intervals answer "another draw of *episodes*". Training-seed variance enters only through $F$, estimated from **two** seeds (95 % interval on the seed SD spans ×70). Most arms have one seed (seed 0 gates seed 1). The 9.5 % replicate false-positive rate is inherited from a smaller rig **[I]**.
3. **Eval size.** 141 episodes, 17 windows each; several strata have tens of windows (e.g. 45 left-turn windows), and some conclusions rest on 7–14 episodes (T-error concentration).
4. **Optimistic inputs.** The nav token and the per-clip speed limit are *supplied*; on this corpus the nav is derived from the ego's own recorded future path, so strategic and tactical numbers are optimistic by construction; a deployed system needs a *predicted* goal. The key guard checks which keys are read, not their provenance.
5. **No flat control, no external baseline.** The hierarchy's necessity is untested; no published baseline exists on our split.
6. **Oracles are not driving.** They read quantities computed from the future; frozen-O partial oracles cannot size a training change (R63) and are lower bounds (§5.2).
7. **Short-horizon and acceleration defects** are unexplained: B's ADE2 is 27–44 % above P10 (H removes this, §7.12, at a consistency cost), acceleration MAE is worse in all runs, and a Savitzky–Golay filter does not repair it. The cosine basis is band-limited (≈ 0.92 Hz), so the roughness's origin is **[U]**.
8. **The instrument floor** is not zero: the GT plan violates 7.9 % of windows under the consistency rules (the speed rule uses one limit per clip).
9. **Retracted readings, so far (root-cause class in brackets).** R63 *an error metric judged against a convention its consumer does not use*; R66 *a one-seed gain read against the wrong floor*; R69 *a gain measured against one baseline draw*; R70 *a train-loss difference read as a mechanism without its sampling error*; R71 *a contrast between strata that a gate selected, where the gate also drives the outcome* (the "route-silent turn" proof of existence shrank from 0.95 vs 0.40 to a like-for-like 0.65 vs 0.41 SD on 37 windows, because the nav speaks only on slow junction turns). Source texts in `R/` still contain some retracted statements with strike-through notes; where they conflict, this report follows the corrections.
10. **Source conflicts found while writing (registry/raw JSON wins).** (a) Tier: registry T0 vs verdict "T1-class". (b) `RESULT_V8F_TAC_A2.md` "mismatch" vs R63. (c) "T is generalisation-limited" (WC, HIST, DSCALE) vs R70. (d) Registry §17 "distance keeping UNAVAILABLE" vs the DK erratum. (e) Line numbers `:601-602`/`:620-622` vs `:627`/`:637` at the tip. (f) The PREREG cites the banked P10 (6 s 2.9257) while K-1 uses P10-yaw-4512 (2.4870). (g) The register lists STRAT-1 as OPEN while the banked paired numbers already fail clause 1.
11. **Not verified by me.** Related-work descriptions beyond the library index; the contents of the rung-β and SMOOTH/LON read files (cited from the register); MIXO's training state; the figures' underlying `.out` text files (T_ATTRIB tables are quoted from `T_ATTRIB.md`, not recomputed). I cross-checked against raw JSON: K-1 and WM-1 (`verdict.json`), the oracle table (`tac_read.json`), group and dose tables, the KIND tables and additivity, the five lever-arm deltas and their mechanism cells, the STRAT paired numbers and the P10 floors.

---

## 10. Reproducibility appendix

**Code (tip `a99ca5d6`, branch `agent/arch-inf-20260803`).** Model `stack/tanitad/models/v8f_mini.py` (config `:122`, key allow-list `:116`, P10 prior `:200`, up-sampling `:269`, whole-code switches `:627` and `:637`, losses `:~690-725`, `p_gt_schedule` `:784`, admissibility `:804-841`); world model `stack/tanitad/models/hjepa.py`; LoRA `stack/tanitad/models/v8f_lora_encoder.py`; trainer `stack/scripts/v8f_mini_train.py` (launch gate: declared-vs-built, live check); eval `stack/scripts/v8f_mini_eval.py`; estimator `taniteval/taniteval/ci.py`; floors `taniteval/taniteval/floors_kin.py`; rules `taniteval/taniteval/v8f_consistency.py`.

**Pre-registrations and results.** `Project Steering/PREREG_V8F_{MINI,TAC,CZ,WC,PGT,GRID,HIST,DSCALE,KIND,LORA,MIXO,D1,BETA}.md`; results `R/RESULT.md`, `RESULT_FULL.md`, `RESULT_V8F_{TAC,TAC_A1,TAC_A2,CZ,WC,PGT,GRID,HIST,DSCALE,KIND,DK}.md`; attribution `R/raw/2026-10-25-t-attrib/T_ATTRIB.md`; retractions `Project Steering/RETRACTION_LOG.md` R63, R66, R69–R71; claims `Project Steering/GOALS_AND_CLAIMS.md` (`H-MINI-K-1`, `H-MINI-WM-1`, `H-V8F-SEED-1`, `H-V8F-DSCALE-1`, `D-V8F-T-ATTRIB-1`, `H-V8F-KIND-1`, `H-V8F-MIXO-1`, `H-V8F-LORA-1`).

**Chain tips (box 2) of the arms read here.** TAC `c56add43`; TAC A1 `742fca0e`; TAC A2 `f2f604f5`; CZ `10078b5e`; PGT `14a27de`; GRID `cf5956f2` (first run at `a8f2c89`); HIST `6b860b95`; DSCALE `2d85add1`; KIND eval tip `009d1cba`; LoRA AR-2 run 3 `c0769eaa` (banked in `a99ca5d6`).

**Hashes.** Train sha256 `37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1`; excluded-holdout `f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9`; v9 train release md5 `83e30bd3791f0767707eade5f4671759`; eval inputs `eval_grid_windows.json` `659c3b65…`, `eval141_kin.npz` `be57e283…`, `join_eval141.npz` `f0ea722f…`; P10-yaw-4512 predictions `9657c0fd…`; DSCALE checkpoint `242c92cb…`; `argv` sha of the DSCALE run `d77fa00c8b3d9525`. Checkpoints are not in git (artifacts drive `…/TanitAD-artifacts/`).

**Figures.** `make_figures.py` in this folder regenerates `fig_*.png` on CPU (`CUDA_VISIBLE_DEVICES=-1`) from the banked JSON in `data/` (sha256 in `data/SHA256SUMS.txt`; source paths below); `fig_architecture.png` is a schematic.

| figure | banked source (`R/raw/…`) |
|---|---|
| 2 | `full/eval/verdict.json`, `2026-10-20-tac/tac_read.json` |
| 3 | `2026-10-21-tacgrp/tac_groups_read.json` |
| 4 | `2026-10-21-tacgrp2/tac_groups2_read.json` |
| 5 | `2026-10-25-kind/run/kind_read.json` |
| 6, 7 | `2026-10-22-{wc,pgt,grid}/run/*_read_s0.json`, `2026-10-23-hist/run/hist_read_s0.json`, `2026-10-25-dscale/run/dscale_read.json`, `2026-10-25-ensemble/paired_swing.json` |
| 8 | `full/eval/compare.json`, `full/eval/floors_report.json` |
| 9, 10 | PENDING (parallel agent) |

**Re-running a read.** Each `RESULT_*` names its reader (`code/v8f_*_read.py`) and its box-2 chain (`code/box2_ops/chain_v8f_*.sh`); the evals are eval-only and take minutes once checkpoints (sha256 in `run/ckpt_sha256.json`) are present.

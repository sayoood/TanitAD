# v5 — what Bar A actually means, and the two directions that follow

**Written 2026-07-26 on the PI's instruction**, immediately after Bar A returned `REFUTE`.
> *"give up and do nothing is not an option… The two main directions remain and must be:
> (1) Combine our both most encouraging results: v1 WM results and the REF-C good results of the
> planner. (2) Leverage performance and efficiency dominance based hierarchical planning,
> prediction, thinking and goal setting, e.g. in picking the best trajectory or trajectory plan."*

**The PI is right, and Bar A supports both directions rather than blocking them.** This document
states precisely what was refuted, what was not, and the experiment ladder that follows.

---

## 1. What Bar A killed — stated narrowly, because the narrowness is the point

MEASURED, `…/2026-07-26-bar-a-selector/raw/bar_a_produced.json`, pre-registered before the run:

**Re-scoring the frozen v4 fan with a learned head over the existing latent features cannot reach
v1.** Even fitting *and* scoring on the **same windows** — maximum possible overfit, not a
generalization number — the ceiling is **0.4907 (CE) / 0.5224 (regret)** against v1's **0.4271**.
Out-of-fold, both arms were *worse* than as-trained (recovered **−4.2 %** and **−11.0 %**).

**So one thing is settled: no loss function over these features fixes the picker.** The waste is
real and it is **not recoverable by re-scoring**. *(That corrects my own earlier framing to the PI,
which implied the 0.4093 / 0.6058 m headroom was recoverable.)*

## 2. What Bar A did **not** kill — four openings, each untested

| # | not refuted | why it matters |
|---|---|---|
| **1** | **The fan is good.** `oracle_in_fan` = **0.2505** deployable, vs v1's **0.4271** | the proposals already beat our best deployed model by **41 %**. The generator is not the problem. |
| **2** | A **different selector INPUT** — only the existing feature set was tested | the information may exist; it may just not be in what the head reads |
| **3** | **Simulative** selection — Bar A tested a **discriminative** scorer only | ⭐ the world model can *roll each candidate forward and score what happens*. This is the direct answer. |
| **4** | **Hierarchical** selection — a flat 1-of-256 pick was tested, with no strategic/tactical structure | ⭐ we claim a hierarchy and then select flat. That claim has never been used where it would pay. |

**Openings 3 and 4 are exactly the PI's two directions.** They are not a consolation prize; they are
the hypotheses Bar A was never designed to test.

## 3. The reframe: the question moved from **ranking** to **representation**

The fan contains a 0.2505 m trajectory. A learned head over the latent cannot find it **even when
allowed to memorise the answer.** So the information that identifies the good candidate is **not
exposed in the features the selector reads**.

Two ways to get at information a feature vector does not expose:
- **simulate it** — roll the candidate forward and observe the consequence (direction 2);
- **condition on structure it lacks** — a goal and a manoeuvre class the flat scorer never saw
  (direction 2, hierarchical form);
- **swap in a better representation** — a different world model, or a different planner (direction 1).

---

## 4. Direction 1 — v1's world model + REF-C's planner

**The asymmetry that motivates it, MEASURED:** v4's own `wm_canary_ade_2s` is **1.1409** (bar ≤0.55)
while the v1 line is **0.4271** — ⭐ **now ESTABLISHED, not inherited** (§8; the **~0.452** this line
previously carried was the deprecated `heldout` split-mean, and the independent re-derivation landed
on the registry's `full_set` value to four decimals). **v1 imagines far better than v4. v4 proposes
far better than v1.** Nothing has combined them.

✅ **EXECUTED 2026-07-27 — see §8.** Combining them is **FEASIBLE and general**; the simulator swap is
worth **9.2×** on the imagination rule; and the combination still **REFUTES** the pre-registered bar,
which is what relocated the lever to the fan.

⚠️ **And a scoring rule is only as good as its simulator.** If we score candidates by imagining
consequences, we should imagine with **the best world model we have**, which is not v4's. Whether v1's
WM can score v4's fan is an **architectural-feasibility question that must be established, not
assumed** — and a clean negative there is itself a real finding about arm compatibility.

⚠️ **The REF-C confound that must not be inherited:** REF-C evaluates with **`nav_cmd=None`**, so its
decoder never had a working route input. This is the documented reason the *"strategic choice is a
~2 % lever"* reading was confounded. **A planner with a working goal input has never been compared to
one without.** That comparison is an arm, not a footnote.

## 5. Direction 2 — hierarchical, imagination-based selection

**The mechanism:** for each candidate, roll the world model forward under its action sequence and
score the **imagined outcome** — corridor departure, along/cross-track deviation from goal, kinematic
plausibility, terminal state. Then select. Structure the selection strategically → tactically →
operatively rather than flat.

**The instrument already exists and was verified in source**: `metric_dynamics.rollout_decode`
advances its window by appending the model's **own predicted latent** — no frame is ever re-encoded,
and `k` is free. It is a per-candidate imagination roll-out, ready to use.

**Bars, pre-registered:**
- **CONFIRM** — beats **0.4907**, the in-sample ceiling of *any* re-scoring of this fan ⇒ imagination
  does what discrimination provably cannot.
- ⛔ **STRONG — WITHDRAWN 2026-07-27, THE BAR WAS INCOHERENT.** It read *"additionally beats v1's
  0.4271"*, but `taniteval/rollout.py:170` sets **`actions_source="expert_future"`** and `:174` names
  the metric **`wm_fidelity_ade_2s`** ⇒ **0.4271 is what v1's world model scores when HANDED THE
  EXPERT'S TRUE FUTURE ACTIONS.** Asking a *selector* to beat a model that was *given* the answer is
  not a stretch goal, it is a category error — and it is part of why nothing has cleared it.
  **The legitimate same-surface bar is the in-sample re-scoring ceiling 0.4907, which stands.**
- **REFUTE** — fails to beat 0.4907 ⇒ the limit is the fan's own action-sequence information, and we
  say so.

**Stratify by simulator quality.** Bar B measured that **22.7 % of windows are already under the
canary bar** with a heavy tail (p50 0.9788, p95 2.6356) — **the WM failure is concentrated, not
uniform.** If imagination-scoring works where the WM is good, that is a **deployable gate**, not a
failure — and it is the same shape as the tactical brain deciding when to think harder.

## 6. Why this is also the efficiency argument

Imagination-scoring costs `n_candidates × k` predictor steps and **zero camera passes**. That makes
the efficiency claim structural rather than rhetorical: *think more, look less*.

It also joins the blind-imagination stream: ⛔ **`T_blind`'s gloss here OVERCLAIMED and is WITHDRAWN (corrected 2026-07-27).** It is measured
**against a FROZEN-LATENT comparator**, so it means *"how long imagination beats holding the last
percept"* — **not** *"how long the model can drive"*. MEASURED: the latent's contribution **peaks at
1 s (+3.61), is +0.14 at 9 s, and is NEGATIVE at 18.5 s**, so **the 11.5 s headline sits exactly where
the latent contributes NOTHING and both arms are ~75 m off track.** V5's imagination-scoring trust
horizon must be re-derived at **≤ 6 s**.

⭐ **What survives, and it is the useful half:** the two streams still share one measurement — *the horizon over which the imagined latent carries anything at all*. It is simply **shorter than the
headline suggested: ≤ 6 s, peaking at 1 s.** Imagination-scoring must be trusted only inside it.

⚠️ **§7.3 binding:** state what value of the efficiency metric would be **disappointing** before
quoting it. H2 MEASURED that the naive framing is information-free — never-escalate saves 85.7 %, a
perfect oracle 85.6 %, the whole span 0.1 pp. Build an axis with real dynamic range or report that
you could not.

---

## 7. The ladder, in order, with what each costs

| step | question | cost | bar |
|---|---|---|---|
| **E-V5-1** | does imagination-scoring beat the re-scoring ceiling? | **hours** — reuses Bar A's cache and frozen fan | 0.4907, then 0.4271 |
| **E-V5-1b** | can v1's WM score v4's fan? | hours (feasibility first) | architectural verdict |
| **E-V5-2** | does hierarchical selection beat flat, with a **working** goal input? | hours | flat pick on the same fan |
| **E-V5-3** | cost-vs-quality of candidates × rollout depth | hours | an axis with dynamic range |
| **v5 train** | only if the above CONFIRM | GPU-week | pre-registered from the results |

**No GPU-week is committed until the hours-scale ladder returns.** That discipline is what turned a
59-hour v4 restart into a 2-hour refutation — and it is why declining *that* restart is not
inaction: **it is what makes these four experiments affordable.**

## 8. Status — E-V5-1 and E-V5-1b RETURNED (2026-07-27). **The lever moved off the scorer.**

Artifact: `…/incoming/2026-07-26-v5-imagination-selection/V5_IMAGINATION_SELECTION.md`. Estimator:
paired episode-cluster bootstrap, B = 2000, unit = episode, 881 windows / 40 clusters.

**E-V5-1b — Direction 1 executed. FEASIBLE, and it is a GENERAL capability of this architecture.**
The fan is **metric trajectories in the ego frame, not latents**, so a foreign world model needs only
to consume our frames and emit metres — **`state_dim` need not match, because the two arms never
exchange a latent; they meet at the metric interface.** ⇒ *any* arm with a grounded step-readout can
score *any* other arm's proposals. (MEASURED · CONFIRMED.)

The simulator gap is large: **v1's WM is 2.67× better than v4's on identical windows (0.4271 vs
1.1381)**, and swapping it in improved the per-candidate imagination rule **9.2×** with nothing else
changed. ⚠️ **The v1 line is 0.4271 (`full_set`), not the ~0.452 this document previously carried —
that was the `heldout` split-mean.** Independently re-derived to four decimals; it also confirms that
v1's canary and v1's headline `ade_0_2s` are the same quantity.

**E-V5-1 — REFUTE, as pre-registered, and not re-scoped.** Best imagination arm **0.5645** vs
CONFIRM **< 0.4907** — a **1.15×** miss on v1's WM (**1.57×** on v4's). The stratified "deployable
gate" escape hatch is **CLOSED**: WM quality grades the damage monotonically and never rescues the
rule.

⭐ **THE MECHANISM, and it is the finding — the world model does not VETO an implausible plan, it
obediently SIMULATES it.** Within one window the 256 candidates span **−15.47 m … +100.57 m** of 2 s
along-track displacement (**mean per-window span 108.7 m**) against a ground truth of **25.40 m** — a
2 s candidate travelling 100 m is a **181 km/h plan, and it is in the fan.** Asked to execute it, the
roll-out also travels ≈100 m, so the candidate is **maximally self-consistent** and
imagination-*consistency* ranks it first (**+19.66 m** along-track bias).

⇒ **The advance commitment in §0.9 is discharged verbatim: the limitation is the fan's own
action-sequence information, not the scorer.** `oracle_in_fan = 0.2505` is a statement about the
**marginal coverage of a 256-anchor vocabulary that also contains 181 km/h plans**.

⛔ **BUT THE REDIRECT I DREW FROM IT — *"THE v5 LEVER MOVES TO THE FAN"* — IS REFUTED, BY TWO
INDEPENDENT STREAMS, AND IT WAS MINE.** (`…/incoming/2026-07-27-fan-conditioning/FAN_CONDITIONING.md`,
DECISION-GRADE, replicated on 4 fans; and `…/incoming/2026-07-27-fan-clip-local/`, independently, the
night before.) **The premise is true and the inference from it is false:**

- **TRUE** — the fan is not `v0`-conditioned. Candidate speed tracks ego speed at slope **−0.129**;
  ground truth tracks it at **+1.0003**. The same ~17 m/s distribution appears in every window, on
  REF-C-xl/base/small **and on v4's own fan**.
- **FALSE** — that this costs anything. ⭐ **100.0 % of windows ALREADY contain a candidate within
  0.5 m/s of the exact speed the car took** (mean gap **0.0525 m/s**), and restricting the oracle to
  *only* speed-matched candidates changes it by **+0.0000 m [0.0000, 0.0000]**. **The best-in-fan
  candidate is already speed-matched.**

⇒ **THE FAN IS WIDE, NOT MIS-PLACED. Marginal over-dispersion and exact conditional coverage are the
same fact seen from two directions** — stream 1 measured the *span*, stream 2 the *unreachable
anchors*, and **neither measured the conditional**, which is the only quantity that could have
licensed the inference. *(New retraction class: **MARGINAL MISTAKEN FOR CONDITIONAL**. Sibling class
**CORRELATION-WITHOUT-SLOPE** — here the correlation is **−0.974** and is misleading; the slope
carries the physics.)*

**Both pre-registered halves fail.** Every `v0`-conditioning transform of the real fan is
separated-*worse* (up to **+1.62 m**); the realised pick tops out at **0.7968** vs CONFIRM 0.4907 — a
**1.62×** miss. On *static* anchor sets conditioning does move the ceiling, but the equal-storage
control shows it is **anchor COUNT, not conditioning**: `A_cond(16,16)` is separated-worse than
`A_fixed(256)` (**+0.0555 [+0.0308, +0.0809]**). The instrument was shown able to return the opposite
verdict — on a deliberately starved fan `F_narrow`, conditioning helps monotonically (−0.51 → −7.59,
all separated).

⭐⭐ **THE SHARPEST NUMBER IN THE STREAM, AND IT REDIRECTS THE PROGRAM: an in-sample anchor set with a
PERFECT ceiling (0.0000) still realises only 0.4167. 100 % OF THE RESIDUAL IS SELECTION.** The 2 s
proposal surface is exhausted; **do not fund CoverNet-style anchor sets, longitudinal admissibility
filtering, or any further re-scoring of the frozen 2 s fan.**

### ⭐ What DOES move it: the goal input — 88.0 % of the fan's headroom

Giving the selector the **true 2 s goal position** — **2 of 8 target scalars** — moves the realised
pick **0.4714 → 0.2009 [0.1689, 0.2351]**, paired **−0.2705, separated**: **88.0 % of the fan's
headroom, clearing BOTH pre-registered bars.** *The first thing in this program to do so.*

**It is a pure INTERACTION, replicated ×3:** along-track alone **−31.4 %**, cross-track alone
**−5.8 %**, both together **+88.0 %**. Neither coordinate is the lever; the *position* is.

⛔ **TESTED 2026-07-27 AND REFUTED — THE 88 % IS AN ORACLE ARTIFACT.**
(`…/incoming/2026-07-27-goal-input/GOAL_INPUT.md`.) **The tautology test came back a NULL, and it is a
null and not an underpowered measurement:** on one axis (2 s endpoint L2, metres) the as-trained
selector's own pick is **1.0061 [0.827, 1.193]** and the best out-of-fold learned goal head is
**1.0028 [0.846, 1.169]** — paired **−0.0034 [−0.1169, +0.1232], not separated. The head is 0.33 %
better than the thing it was meant to inform, and separating that would need ≈ 4.99 × 10⁴ episodes.**
⇒ **PREDICTING THE 2-D ENDPOINT *IS* THE PICKING PROBLEM.** The selector already predicts the 2 s
endpoint to 1.0 m against a mean displacement of 25.5 m.

**The operative test agrees.** Break-even is **σ₀ = 0.955 m** radial RMS goal error (**0.606 m** for
half the prize; **0.721 m** for a *biased regressor*, the realistic family, 25 % stricter). **Achieved
1.330 m ⇒ recovery −10.4 %.** The fan-independent **latent-only** head — what a strategic brain would
actually have — is **separated-WORSE (+0.0464 [+0.0164, +0.0792])**, replicated on all three REF-C
fans; **on none is a realisable goal separated-better.** Damage reaches **+8.37 m**.
*(New classes: **BOUND-QUOTED-AS-CAPABILITY**; **ORACLE-SHAPED-AS-EGO-STATE** — `head_deg` is *future*
net heading change and sits beside `v0` in every fan dump, caught pre-fit; **RMS-PLACED-ON-A-NOISE-CURVE**,
which over-predicted damage 5.7×.)*

⚠️ **Scope it honestly: this refutes ONE argument for the hierarchy — the cheapest one — not the
hierarchy.** Goal conditioning **does** work below 0.955 m. What is refuted is that *our features
reach it*, and that *a map would help*.

⭐⭐ **AND THE REDIRECT IS THE MORE VALUABLE HALF — WE WERE AIMING AT THE WRONG AXIS.** Holding the
other axis at its **learned** value: oracle **along-track recovers +83.7 % (separated)**; oracle
**cross-track recovers +2.9 % (NOT separated)**. **Even a *perfect* cross-track buys 2.9 %.**
⇒ **The 88 % lives on HOW FAR THE CAR WILL TRAVEL IN 2 s — not on where the road goes.**

⛔ **THEREFORE THE "WE NEED ALPASIM OR AN EXTERNAL MAPPED CORPUS" LINE TARGETS THE +2.9 % AXIS AND IS
THE WRONG BLOCKER.** A map / route / lane-graph supplies the axis worth 2.9 %.

### ⭐⭐ E-GOAL-1 RETURNED 2026-07-27 — THE LONGITUDINAL AXIS IS REALISABLE, AND THE SUPPLIER IS **EGO SPEED HISTORY**, NOT AGENT TRACKS

⛔ **CORRECTION, AND IT WAS MINE: the text above named `obstacle.offline` / "53.6 agents/window" as the
supplier. That is WRONG, and a v5 decision taken on it would have funded an agent-track ingest instead
of a speed-history feature.** `53.6 agents/window` is also the wrong statistic for this hypothesis —
the same-family census is **2.01 vehicles ahead within 50 m** and **0.40 lead vehicles**.

⛔ **The lead-vehicle hypothesis REFUTES (CONFIRMED tier), and it never could have worked: 41.65 % of
windows have no vehicle ahead within 50 m at all.** Adding `gap`/`closing`/`TTC` moves along-track RMS
**0.9305 → 0.8983 m** (+0.0322 [−0.0040, +0.0969], **not separated** on the primary axis; separated on
the higher-power MAE read at −2.29 %). Worth **+2.3 recovery points**. Not re-scoped.

⭐⭐ **But run through the ACTUAL REF-C rule rather than read off a curve, a head with the measured
error structure recovers +23.6 % of the fan's headroom FROM EGO KINEMATICS ALONE**
(−0.0638 [−0.1271, −0.0008], **separated**), **+25.9 %** with lead added — while **the parent's own head,
pushed through the IDENTICAL construction, is separated-WORSE at −33.1 %.**
⇒ **THE DIFFERENCE IS SPEED HISTORY. `dv_*` / `v_lag_*` alone are worth +0.1428 m [+0.0686, +0.2516] —
4.4× the entire lead block. The parent's head had `v0` and NO HISTORY.** The earlier −10.4 % REFUTE was
a missing-feature result, not a ceiling.
✅ **TIER: CONFIRMED — E-GOAL-2 re-ran it at n = 600 (2026-07-28) and the effect SURVIVES.**
(`…/incoming/2026-07-28-egoal-2-power/`.) Recovery **+25.4 % [−0.0960, −0.0606]** on the conservative
carrier, and ⭐ **`by_speed` — the resampler that would NOT separate at n = 40 — now separates at
+26.2 % [−0.0987, −0.0631]**, on both resamplers and all three cross-track backgrounds. **The −0.0008
upper bound was UNPOWERED, NOT REFUTED.** CI half-widths shrank **×3.20–3.89, median ×3.76** over 18
cells — an independent replication of `MODEL_REGISTRY §1.2a`.
⇒ **The speed-history feature ENTERS the v5 selector. Size it against +25.4 %, NOT the +40.8 % the
friendlier backgrounds return.**

⛔ **AND E-GOAL-1's NUMBER IS RE-SCOPED — the registered bridge FAILED, which is the more important
finding.** E-GOAL-1's cross-track background **cannot be rebuilt at 600** (it needs v4 latents that
exist only on the 881 windows), and the registered substitute deviated **+5.6 recovery points and
FLIPPED `by_speed` to separated at n = 40 — using it would have MANUFACTURED a CONFIRM.**
⭐⭐ **At fixed n and fixed along-track error, recovery spans +13.3 % … +29.2 % PURELY ON THE
BACKGROUND, and separation flips inside that range — a 15.9-point swing.** ⇒ **E-GOAL-1's +23.6 % was
conditional on a background it never named, and SO IS EVERY OTHER RECOVERY NUMBER IN THIS PROGRAM.
From now on a recovery figure without its cross-track background is inadmissible.**

⛔ **A second self-caught defect, in the stream's OWN pre-registered primary: the predicate STOPS
DISCRIMINATING at n = 600 — a deliberately information-free arm separates too (+9.1 %).** The claim
survives on a **direct contrast** instead: history vs noise-in-the-same-columns **−0.0504
[−0.0519, −0.0490]**, while dropped-history vs fake-history is **−0.0001 [−0.0006, +0.0004]**, a tight
null. **64 % of the recovery is speed history; the lead block is 7.9× smaller — E-GOAL-1's lead
refutation REPLICATES.** *(New class: **PREDICATE-STOPS-DISCRIMINATING-AT-HIGH-n** — the inverse of
under-powering, and it would have passed unnoticed as a "stronger" result.)*

⚠️ **Family matters, demonstrated: the family-matched σ₀ at n = 600 is 1.2195 m and the head's
0.9305 m clears it by 1.31× — while FAILING the inherited `ISO` 0.813 bar by 1.14×.** Same number,
opposite verdicts, decided by which error family the bar was computed on (class C24).

### ⭐⭐ E-GOAL-3 (2026-07-28) — THE TRAINED HEAD LANDED, AND IT BEATS THE ESTIMATE THAT PREDICTED IT

`…/incoming/2026-07-28-egoal-3-trained-head/`. **CONFIRM, by 2.4× over its pre-registered ≥ +19.1 % bar.**

| arm (background NAMED: `parent_resampled`) | recovery | realised `ade_0_2s` | paired |
|---|---:|---:|---|
| `H_ego` out-of-fold | **+46.3 %** | **0.3589 [0.3487, 0.3701]** | **−0.1426 [−0.1573, −0.1273]** ✅ sep-better |
| deployable (fitted on the 2376-ep parity train, 0/600 leak) | **+50.7 %** | — | **−0.1564 [−0.1719, −0.1408]** ✅ |

⭐ **0.3589 CLEARS 0.4907 — the in-sample re-scoring ceiling that four consecutive streams failed to
beat.** It can, because this is not a re-scoring: it injects information the fan never had.

⭐ **The resampled estimate TRANSFERRED AND UNDER-STATED THE LEVER (1.82×).** Decomposed at matched
RMS (0.9311 vs E-GOAL-2's 0.9305): a **correlated** head is **+3.9 points** better than the
decorrelated resampler, and the remaining **+17.0 points is pure accuracy** (0.7449 vs 0.9305 m). Head
RMS/MAE **0.7449 / 0.4819** — clears the family-matched σ₀ by **1.64×** and, for the first time in this
program, the inherited `ISO` 0.813 bar as well. *(New class: **RESAMPLED-RESIDUAL-UNDER-STATES-A-TRAINED-HEAD**.)*

⛔⛔ **AND IT REFUTES THE MECHANISM IT INHERITED — the feature list above is WRONG.** The lever is
**not "1 s of speed history". It is ONE 0.1 s speed difference.** `v` alone: **−19.4 %, separated-WORSE**.
**`v + ax` (finite-difference): +46.3 %, a tight null against the full 10-column head
(+0.0002 [−0.0023, +0.0027]).** **The `dv_*` / `v_lag_*` block E-GOAL-2 credited with 64 % is worth
0.9 of 46.3 points — 2.0 %.**

⭐ **Root cause found AND replicated on E-GOAL-2's own corpus with its own fitter, folds and seed**
(both their anchors 0.9305 / 1.0733 reproduce exactly): **`egomotion`'s NATIVE `ax` channel is a poor
derivative of the speed the target integrates — they correlate only 0.759.** `v + ax_fd` (a 0.1 s
backward difference) reaches **0.9270 m**, a null against the whole 10-column block at 0.9305, while
native `v + ax` reaches only **1.1808 m — 0.2539 m worse for one column choice.** ⇒ **the lag block was
a PROXY for a derivative the native channel failed to supply. The two streams agree once the column is
fixed.** *(New class: **ABLATION-CREDITED-TO-THE-WRONG-COLUMN**.)*
⇒ ⛔ **E-GOAL-2's "64 % of the recovery is speed history" is WITHDRAWN. Its statistical result stands;
its causal attribution does not.** ⇒ **A v5 built on the published column list would ship a 1-second
history buffer to buy 2 % of the effect.** ✅ Fixed at the source: `lead_state_gate.py` now emits
**`ax_fd`** with `EGO_COLS_FD = ["v", "ax_fd"]` alongside the unchanged `EGO_COLS` — **`ax` is NOT
redefined**, because committed artifacts carry that name and changing a column's meaning under a
stable name is its own failure mode.

**Controls, all exercised rather than asserted:** REFUTE demonstrated **four** ways
(`CV_head`, `H_v0`, `H_inst`, `N_SHUF` all separated-WORSE); PARTIAL demonstrated (k=1.5 → +11.6 %,
separated); three tight nulls. **C30** — background named in advance and held fixed; span **15.8 pts**,
replicating E-GOAL-2's 15.9, and **separation does NOT flip: 6/6 cells separated-better**. **C31** — the
predicate is confirmed non-discriminating at this n (a noise-history arm separates at +45.4 %), so the
mechanism rests on **direct contrasts**. **C23** — `future_blind` over all **13,198 windows, max
|Δ feature| = 0.0**, with power demonstrated ⚠️ **and the negative-index trap FIRED: 600 windows (the
first of every episode) would have read `poses[-3]` — the FUTURE — and were clamped.** **C24** — family
measured independently (`EMPIRICAL`, α 0.9983): σ₀ **1.2276 m** against E-GOAL-2's 1.2195, **0.7 %
apart on a different error family.** Fidelity: F-1 reproduces E-GOAL-2's n=600 cell to **0.003 recovery
points**, so +46.3 % vs +25.4 % is **treatment, not code**; F-2 is a per-row exact identity over 13,198
rows with F-3 failing hard as required. Leak **0/600 by pose content against 600/600 by filename**,
cross-validated against E-GOAL-2's independent script.

### ✅ E-GOAL-4 (2026-07-28) — JOINT TRAINING CONFIRMS IT, AND RE-SCOPES IT THREE WAYS

`…/incoming/2026-07-28-egoal-4-joint/`. **Coupling named:** a per-candidate learned selector over the
frozen 256-anchor REF-C-XL fan — **13,198 windows × 256 = 3,378,688 rows**, label = each candidate's
own realised `ade_0_2s`, pick = `argmin`, **5 episode-disjoint folds, out-of-fold**, goal head fed as
an input. The fixed rule's own statistic is a fed column, **so the learner CAN express `argmin d_rule`
exactly** — it is not handicapped.

**The recovery survives and GROWS: `S_goal` +62.09 % (`parent_resampled`) / +64.08 % (`sel`,
future-blind)** vs the fixed rule's +46.34 % / +61.46 %, **beating the fixed rule with the same goal on
both** (−0.0479 [−0.0530, −0.0431]; −0.0081 [−0.0112, −0.0051]). **None of the three registered failure
modes occurred** — co-adaptation is a null on both backgrounds, the in-sample↔OOF gap is 2.4–2.5
recovery points, and the selector does not ignore the goal.

⛔ **1. THE +46.3 % OVER-CREDITS THE GOAL BY 1.76×, AND THIS IS THE NUMBER TO PLAN WITH.** It was
measured against the **as-trained** selector. **A trained selector with NO goal already recovers
+35.62 %.** The goal's **capacity-matched marginal is +26.31 recovery points** — and it is *identical*
on both backgrounds (**−0.0811 [−0.0904, −0.0720]** and **[−0.0888, −0.0732]**) **even though their
totals differ by 15 points.** ⭐ That invariance is the useful part: **the capacity-matched contrast is
background-independent where the totals are not** — a partial answer to C30. ⇒ **PLAN v5 WITH +26,
NOT +46.** *(New class: **C34 — a lever measured against the wrong counterfactual**.)*

⭐⭐⭐ **2. GOAL-HEAD ACCURACY IS NEARLY WORTHLESS END-TO-END, AND THIS RETIRES A WHOLE LINE OF WORK.**
A naive **`2·v0`** goal — along-track RMS **1.449 m**, which the *fixed rule* turns into **−18.55 %,
separated-WORSE** — delivers **+62.07 %** through the trained selector. The learned `v + ax_fd` head
buys only **+2.01 / +3.89** points on top, against **+64.89 / +67.57** through the fixed rule: a
**16.7×–33.6× collapse.** The mechanism is visible in the picks: `S_goal_cv` targets to **0.806 m** from
a goal handed to it at **1.449 m**. ⇒ **THE FIXED RULE OBEYS THE GOAL; THE TRAINED SELECTOR TREATS IT
AS EVIDENCE.** **Do not fund goal-head accuracy.** *(σ₀, the requirement curves, the feature hunt — all
of it buys 2–4 points once the consumer is trained.)*

⛔ **3. E-GOAL-3's REQUIREMENT CURVE DOES NOT TRANSFER — σ₀ IS A PROPERTY OF THE CONSUMER, NOT THE
GOAL.** On the identical degradation ladder the fixed rule goes destructive at **1.128 m** and reaches
**−111.78 % at 2.256 m**, where the trained selector is **+16.73 %, separated-BETTER, and never crosses
zero.** *(New class: **C35 — a requirement curve is a property of the consumer**.)*

⭐⭐ **AND THE GOAL CARRIES NO NEW INFORMATION AT ALL.** `g_along` = GBM(`v`, `ax_fd`) at
**R² = 0.999894** (exact, gate G-6) — **and the no-goal arm is fed both columns.** ⇒ **the goal is an
INDUCTIVE BIAS worth +26.3 separated points, not a channel.** *(New class: **C36 — an input can be
worth points while carrying no information**.)* ⇒ 🔴 **FUNDING A STRATEGIC *SUPPLIER* IS THE WRONG
LEVER AT THIS FEATURE LIST** — AlpaSim, an external mapped corpus, or any goal-signal acquisition buys
the +2–4 accuracy points, not the +26.

**Audit and controls.** Every fed column future-blind at **max |Δ| = 0.0 over all 3.38 M rows**, with
power shown three ways — ⭐ **the instrument FIRES on `parent_resampled`'s cross (5.2 × 10⁴,
future-derived by construction) and returns exactly 0.0 on `sel`'s**, which is why every number is
reported on both. Fold disjointness by **pose sha256: 0 shared across all 10 fold pairs**. **G-1
reproduces E-GOAL-3 to 0.004 recovery points.** Two self-caught defects reported rather than worked
around: the registered "MUST BE NULL" control **separates on `sel`** (+0.0066) because a shuffled goal
still yields a real geometric feature — **which is exactly why the headline uses the capacity-matched
contrast that subtracts it** — and the registered C23 power control is itself a null, so power rests on
the oracle arm instead. All four verdict branches were reachable in this run's own data.

⚠️ **Consequent fix owed:** `MODEL_REGISTRY`'s **0.4907** needs a deployment tag (**881-window / 40-ep,
`a0` 0.4714**) — it is being quoted against 600-episode numbers.

⚠️ **AND THE BARS THIS DOCUMENT PUBLISHED WERE COMPUTED ON THE WRONG FAMILY.** The measured heads are
**near-unbiased (α ≈ 0.996 — not `SHRINK`)** and **heavy-tailed (RMS/MAE 1.867 — not `ISO`)**. Sweeping
this stream's own residual pool through the rule gives **σ₀ = 1.1434 m** (half-prize **0.5907 m**)
against the inherited **0.813 / 0.439 m** — **1.41× more forgiving**, and the lead head clears it.
⇒ **Reading the `ISO` bar literally returns REFUTE; running the rule returns +25.9 %.** *(New class
**BAR-INHERITED-FROM-THE-WRONG-FAMILY**.)* **Every bar in this section must now carry its family.**

⭐ **A control that voids an axis rather than a result:** the positive control passes decisively on MAE
(−61.2 %) and **FAILS TO SEPARATE ON RMS** ⇒ **at n = 612 clips the along-track RMS axis cannot separate
even a near-perfect oracle, so NO non-separation on it is evidence of absence.** *(A guard that cannot
fail — class C13 — found in this stream's own pre-registered primary.)* The decorrelation control shows
the injection is **conservative, not optimistic**; the shuffled-lead negative control is separated-worse.

⚠️ **A live defect in a committed artifact:** `stack/scripts/lead_state_gate.py` (the `obstacle.offline`
reader, which **already existed** — imported, not forked, and proved bit-faithful over 104,652 rows)
**never checks that `obstacle.offline` covers its grid. 10.59 % of clips and 4.51 % of windows have no
obstacle data and SILENTLY BECOME `lead_present = 0`**, deflating the committed rate from a true
**0.4046** to **0.3851**. Span guard written; the committed result needs a footnote.

🟠 **Provenance defect, recorded rather than buried:** `fanc_goal_decomp.json` — the source of the
widely-quoted *"−31.4 % / −5.8 % / +88.0 % pure interaction"* — **has no producing script staged
anywhere** (probed twice), so that decomposition is **currently unreproducible from the repo** and it
had already propagated into this document. **The substance survives independent re-derivation**
(+83.7 % / +2.9 % above, from a separate implementation), **but the original artifact must be
reproducible or withdrawn.**

⭐ **And this vindicates a suspicion already on the record in `CLAUDE.md`:** the *"strategic choice is
a ~2 % lever"* reading was flagged as **confounded** because REF-C evaluates with **`nav_cmd=None`**,
so a decoder that never had a working route input learned the marginal. **That confound is now
measured: with a working goal the lever is worth 88.0 % of the headroom.** The three-planner
hierarchy is not over-design — *we had simply never wired the strategic input.*

### What is genuinely positive, with its honesty conditions attached

- ⭐ **C2 (one WM roll as a reference trajectory) with v1's WM beats the as-trained selector,
  separated: −0.2918 [−0.4233, −0.1598], with ZERO training, UNGATED, on 100 % of windows.**
  ⚠️ **The stratum figure — *"0.7085 → 0.3330, a 53 % cut"* on the 22.7 % of windows where the WM is
  good — MUST NEVER TRAVEL WITHOUT ITS 22.7 % FIRING RATE. Deployed as a policy it is worth
  −0.0852 [−0.1190, −0.0548]; quoted bare it overstates the deployable win by 4.4×.**
  *(0.227 × 0.3754 = 0.0852. New retraction class: **a stratum win is not a deployable win**.)*
- **A4 moved the LONGITUDINAL axis: −0.0898 [−0.1708, −0.0160], p = 0.008** — the axis v4's 15k→30k
  regression was **100 %** concentrated on, and the axis Bar A's **trained** regret objective could
  not move at all (+0.0038, p = 0.63, flat). **A training-free scoring rule moved what a trained
  objective could not.** Its `ade_0_2s` (−0.0857, p = 0.033) is **UNPOWERED, NOT REFUTED** at n = 40
  and would separate at n = 600 — the cheapest open question in the stream.
- With a good simulator, per-candidate imagination beats the **WM-free** control, separated
  (**−0.5364 [−1.2234, −0.0842]**) — the first genuine imagination result in the program. **But it
  stays separated-worse than rolling ONCE** ⇒ the failure is the **per-candidate structure**, not the
  world model and not imagination.
- ⛔ **A4's winning weights put their mass on the two CONTROLS** (WM-free bicycle + single reference
  roll); the per-candidate imagination term carries the smallest non-zero weight. **A4 is a
  longitudinal-plausibility result that uses a world model as a reference trajectory** — a different
  and much cheaper mechanism than imagination-scored selection. Read it that way.
- ✅ **RESOLVED 2026-07-27 — the missing instrument EXISTS, and building it proved we should not use
  it.** (`…/incoming/2026-07-27-canary-proxy/CANARY_PROXY.md`, out-of-fold, episode-disjoint.)
  `wm_canary_ade_2s` **is** predictable from deploy-time observables — **R² 0.5526 / Pearson 0.7518**
  (R² 0.4203 with a single world model) against **R² 0.070** for `v0` — and it converts: a learned
  **utility** gate recovers **164 % of the oracle** (−0.1397 [−0.2289, −0.0634], separated, stable
  across 5 fold seeds).
  ⛔ **But the gate is DOMINATED BY ITS OWN PREREQUISITE and should not ship.** The robustly separated
  gate needs **2-world-model ensemble features** — and with a second world model you can simply score
  **ungated** for **−0.2918**, **2.1× more**. Even an *unattainable perfect* per-window gate on v4's
  WM (−0.2353) loses to ungated C2. Single-WM gates are −0.047…−0.060, **UNPOWERED NOT REFUTED**
  (they separate at n ≈ 61–129).
  ⭐ **DECISION AVAILABLE TODAY: apply C2 with v1's WM, UNGATED. Best measured deployable policy
  ⛔ **RETRACTED 2026-07-28, AND THE ERROR WAS MINE, IN RELAY.** `−0.3366 / 0.5196–0.5221` is **NOT the ungated rule** — it is `learned_gate_ALL_ridge_tau0`, **a FITTED RIDGE GATE firing on 66.97 % of windows** over a 73-feature bank **including the 2-WM ensemble family**, i.e. precisely the gate this section already calls *dominated by its own prerequisite*. **The source document was internally correct** (its §1.2 publishes −0.2918 and its §5.2 recommends exactly that); **I conflated the two when relaying it**, and repeated it to the PI twice. ✅ **THE SHIPPED, RE-VERIFIED UNGATED VALUE: 0.8563 → 0.5645, paired −0.2918 [−0.4233, −0.1598], separated, `selected_frac` 1.000** — reproduced to 4 dp by an independent implementation, and at the strongest layer **re-derived from raw geometry (`fan [881,256,20,2]` + `imag_ref`) to 881/881 IDENTICAL PICKS.** My figure overstated the shipped win by **0.0448 m (1.154×)**, and the cut is **34.1 %, not the 39 %** I quoted. *(An agent that had simply hit my stated bar would have shipped a gate.)*
  ⛔ **AND C2 IS NOT UNIVERSALLY GOOD: v4 scoring its OWN fan is separated-WORSE, +0.2090 [+0.0550, +0.3642].** Of the two arms ever measured, one helps and one harms ⇒ **the shipped default is OFF, and there is no default scorer** (`--select-rule c2-wm-ref` without `--c2-scorer` exits on the arguments). ⚠️ **Cost is ≈62 ms/window, not "one extra roll-out"** — 6.7 ms for the roll and **55.6 ms for the second frame encode a FOREIGN scorer requires**; the encode dominates. ⚠️ It remains a **1.15× miss on 0.4907**, leaves **0.3140 [0.2632, 0.3703] m** against `oracle_in_fan` 0.2505, and is **54.9 % better / 32.2 % worse / 12.8 % identical** per window. ⚠️ **−0.2918 is a fact about v4's FAN — v5 must MEASURE it, not inherit it.** *(Superseded text kept for provenance:)* a 39 %
  cut with NO ORACLE ANYWHERE, zero training, one extra roll-out per window.**
  **Three findings that close threads:** `v0` used as a gate is **separated-WORSE than doing nothing**
  (+0.0411 [+0.0133, +0.0752]) — that thread is closed, not merely weak. **Aiming at the canary costs
  3.6×**: gating on predicted **C2-vs-A0 utility** beats gating on the predicted canary
  (−0.1397 vs −0.0383, same features, same folds) ⇒ *optimise the objective you are paid for, not its
  legible correlate*. And **roll-out drift — the most theoretically attractive proxy family — was the
  WEAKEST** (+0.0108 on v4), independently confirming §2.2's mechanism: a simulator with no
  plausibility prior has uninformative self-consistency.
  🚨 **Near-miss worth recording: TWO PURE-NOISE GATES WOULD HAVE BEEN WRITTEN UP AS SEPARATED WINS**
  (−0.2918 and −0.2761 vs A0) **had the pre-registration not required a `selected_frac` and
  incremental columns.** The degeneracy check caught both. 6 pre-registered S-tests + 6 downstream
  negative controls all pass; the harness **refuses to adjudicate if any S-test fails**.

E-V5-2 and E-V5-3 remain on the eval pod. The blind-imagination sweep supplying `T_blind` is on pod2.
**v4's restart budget stays 0/2 — unspent, not forfeited.**

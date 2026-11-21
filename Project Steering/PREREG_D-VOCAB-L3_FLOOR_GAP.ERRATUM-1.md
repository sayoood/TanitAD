# PREREG_D-VOCAB-L3_FLOOR_GAP — ERRATUM 1

**Issued 2026-09-05, before the panel runs.** ⛔ **A SEPARATELY STAGED DOCUMENT, not an edit** —
the pre-registration's falsifiable object is its git blob at staging time, and silently improving a
prereg after new evidence arrives is how a prediction stops being one.

The parent's hypothesis and its three committed outcomes are **UNCHANGED**. This adds a required
control, corrects a mis-cited mechanism, and registers a competing hypothesis that arrived after
staging.

---

## E1. ⛔ THE REPLICATE CONTROL NAMED THE WRONG MECHANISM, AND THE RIGHT ONE IS WORSE

The parent's Control 2 required a *"zero-lever replicate — the L=1 arm run again, same flags"*, and
cited **`H-ESTIM-SEED-1`**. That citation is wrong in a way that matters.

`H-ESTIM-SEED-1` is about **TRAINING variance**: the episode-cluster bootstrap resamples episodes
with the models held fixed, so it cannot see whether another *training run* would agree.

⛔ **refav1 has a SECOND, INDEPENDENT source of run-to-run variance that `H-ESTIM-SEED-1` does not
cover: the planner is STOCHASTIC AT INFERENCE.** iCEM samples. So **the same checkpoint, evaluated
twice, does not give the same answer** — no retraining required. MEASURED: the seed floor is
**≈0.30 m ADE**, *"and the programme has been comparing single-seed arms"*
(`D-REFAV1-SEED-GOAL-MISMATCH`).

⇒ **Correction: the replicate must vary the INFERENCE seed**, and it is required *in addition to*
any training-seed consideration, not instead of it. A vocabulary change is a change to the action
space the sampler draws from, so both arms must be read against a floor built the same way.

⭐ **The general form, which is the part worth keeping:** the paired episode-cluster bootstrap
answers *"would another draw of EPISODES say this?"*. It is blind to *"would another TRAINING RUN
say this?"* (`H-ESTIM-SEED-1`) **and, on a stochastic planner, to "would another INFERENCE RUN say
this?"** — three different questions, one interval. Name which one a separated CI has answered.

## E2. A POWER CHECK IS NOW MANDATORY *BEFORE* THE PANEL RUNS

With a floor of ≈0.30 m ADE, a committed criterion of *"the deficit shrinks by ≥ 50 %"* is only
meaningful if **half the L=1 deficit exceeds that floor on the windows being scored**.

⇒ **Added to the parent's protocol as a blocking step:** compute the L=1 turning-window deficit to
`ha0_ext` and its 50 % target **first**; if that target does not clear the ≈0.30 m inference-seed
floor, the panel is **UNDERPOWERED BY CONSTRUCTION** and must be redesigned — more windows, more
inference seeds, or both — **before** any GPU is spent. ⛔ A confident-looking null from an
underpowered panel is worse than no panel, because it will be quoted.

⚠️ This is the same discipline the refav1 stream applied to itself when it **killed its own A/B**
after a power check showed it would move 3 of 60 windows, and redesigned for 6× the power at 1/5
the cost. A prereg that does not demand of itself what an agent demanded of itself is not holding
the line.

## E3. A COMPETING HYPOTHESIS IS NOW ON THE TABLE, AND THE PANEL MUST DISCRIMINATE

MEASURED after the parent was staged: **a *perfect* goal is worse still** — the oracle-goal arm
reads **ADE 7.47**, saturating **`kappa_max` and `a_max` on ~100 % of windows**. The stream's
reading: *"the binding constraint is the goal→plan pathway, not the goal head"* — give the planner
no signal and it does nothing, more signal and it degrades, a perfect signal and it saturates.

⛔ **That arm is CONFOUNDED and does not refute the parent** — supplying `goal_field` skips the seed
entirely (`supplied` 142/142 vs `tactical_imagined` 142/142), so it is a different pipeline. ⚠️ But
it is a live alternative explanation for the floor gap, and the parent's panel as written **cannot
tell the two apart**: a vocabulary fix and a pathway fix would both show up as a shrunken deficit.

⇒ **Added measurement, in both arms, on the same windows:** the **saturation rate of `kappa_max`
and `a_max`**, reported beside the deficit.

| pattern | reading |
|---|---|
| deficit shrinks **and** saturation rate falls | the vocabulary was the mechanism — **parent SUPPORTED** |
| deficit shrinks **and** saturation stays pinned | something other than expressibility did the work; the parent's claim is **not established** even on a passing number, and the pathway hypothesis takes precedence |
| deficit does not shrink | **parent REFUTED** as already committed; the search moves to the cost metric and the seed pool |

⭐ **This is the discrimination the parent lacked.** A prediction whose SUPPORTED branch could be
produced by two different mechanisms is not yet a test, and adding the discriminating readout
*before* the panel runs is cheaper than arguing about attribution afterwards.

## E4. What is unchanged

The hypothesis, the three committed outcomes and their thresholds, the ⛔ requirement that the
turning/straight split use the **MEASURED** crossover (|gt_κ| > 4e-2) and never the inherited 1e-3,
Controls 1, 3 and 4, and both non-claims (**this does not claim refav1 will drive**; the prize
remains **UNBOUNDED** until the oracle arm is de-confounded by supplying the goal *through* the
seed path).

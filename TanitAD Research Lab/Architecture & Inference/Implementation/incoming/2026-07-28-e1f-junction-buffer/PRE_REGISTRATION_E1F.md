# PRE-REGISTRATION — E1f: is the TARGET wrong, rather than its weight?

**Written and committed BEFORE the run existed.** pod3 (free; pod1 and pod2 unreachable and
untouched). Standing directive **D-A**.

## 1. Why this experiment, and why it is not another knob

Three one-dimensional levers have now returned **BOUND**, each for a *different structural reason*:

| lever | experiment | why it cannot be pushed further |
|---|---|---|
| training time | E1c | open-loop cost **plateaus** from step 2250 (8 points inside ±0.02) |
| weight space | E1d | interpolation is **separated-WORSE at 5 consecutive interior α** — endpoints not linearly mode-connected (**C52**) |
| loss weighting | E1e-A/B | λ sets the **asymptote**, preserves P1, but Ga's lower bound flattens at **+0.023** |

All three moved *how much* or *how hard* the same target is supervised. **E1f changes the target
itself** — the remaining hypothesis, and the one E1d's own decomposition names.

**The measurement that motivates it (E1d, MEASURED):** `dep_junction` is better at **every** α and
**never separated-worse**, separated already at α=0.20 for **+0.0308** open-loop cost; `dep_overall`
is **separated-worse through the middle** and only turns good past α≈0.72.
⇒ **Junction recovery is cheap and monotone; overall-corridor recovery is expensive and
barrier-crossing.** The buffer supervises *all* recoverable pre-failure states — including the
expensive half.

## 2. Design — exactly one thing changes

**E1f = E1e-A with the buffer restricted to junction states.** `lam_replay = 3.0` (E1e-A's setting,
which has a full frontier to compare against), 4000 steps, lr 2e-5, warmup 100, cl-batch 16,
replay-batch 16, seed 0, encoder frozen — **byte-identical to `run_e1e_A.sh` apart from `--buffer`.**

**The buffer (MEASURED, built by `make_junction_buffer.py`):**
- parent `mined_buffer.pt`, md5 `a32cfe9bfea4b1b5c196d3bb7f71fa5f`, **verified at build time**
- filter `|dpsi| >= radians(10.0)` — **the evaluator's OWN `--junction-deg`**, not a threshold
  invented for this arm
- **733 of 3,537 records (20.7 %), across 102 of 362 episodes**
- output md5 `35fe24a2787c2afbf72888aeb23c525f`, asserted by the trainer at run time

Evaluated by **`e1c_eval.py`, unmodified**, at steps 1000/2000/3000/4000 — same paired
episode-cluster bootstrap (`taniteval/ci.py`, B=2000, 44 held-out episodes), same
`evaluate_point`/`render_verdict`, same six conditions as E1c, E1d, E1e.

## 3. ⚠️ A correction to this experiment's own feasibility note, made before it runs

`E1F_FEASIBILITY.md` warned that the 4.8× smaller buffer (~87× reuse per record vs E1c's ~18×) meant
**"a win could be memorisation rather than transfer"**, and proposed shortening the run to match reuse.

**That was wrong, and the run length is therefore NOT changed.** The trainer's leak-guard verifies on
every run that the buffer's episodes and the held-out 44 have **intersection 0** (byte level).
Memorising buffer records therefore **cannot inflate held-out metrics**. Over-reuse would manifest as
**under**performance on held-out data, not as a false success.

⇒ E1f stays **step-matched** to E1c/E1e-A/E1e-B, which is what makes the comparison clean. The real
residual risk is not contamination but **diversity**: **102 episodes**, and that is stated as a bound
below rather than engineered away.

## 4. Both outcomes, committed in advance

**OUTCOME A — a SUCCESS POINT exists.** Some checkpoint satisfies **P1 ∧ P2 ∧ Ga ∧ Gb1 ∧ Gb2 ∧ Gc**.
⇒ **The target was the problem.** Restricting supervision to the cheap, monotone half buys the
closed-loop gain without the open-loop regression that the expensive half was forcing. That
checkpoint is the **D-A deliverable**, registered with its buffer md5, and owes a from-scratch
confirmation before any headline.

**OUTCOME B — BOUND again.** No checkpoint satisfies all six.
⇒ **A real result that closes the target hypothesis too.** Four levers — training time, weight space,
loss weighting, and *what is supervised* — will all have failed to reach Ga. The conflict would then
not be an artifact of how CL-SFT is configured at all, and the next question becomes whether **Ga
itself is the right guardrail** (a strict non-regression on open-loop ADE may be incompatible with any
closed-loop fine-tune of a frozen-encoder arm) — **a question for the PI, not another arm.**

**OUTCOME C — NAMED IN ADVANCE because it is likely and would otherwise be read as a muddle:**
**P2 (junction) improves while P1 (overall) degrades.** The arm is trained only on junctions, so this
is the *expected* shape if the restriction works as intended but narrows competence.
⇒ Read as: **the junction advantage is real and transferable, but junction-only supervision is not
sufficient for the overall corridor primary.** The follow-up would then be a *mixed* buffer with a
junction over-weight — and that is a **new pre-registration**, not a continuation of this one.

## 5. Honest bounds

- **102 episodes** in the junction subset vs 362 in the parent. Diversity, not contamination, is the
  binding constraint; a weak result may reflect it and must not be over-read as "junctions don't help".
- **6 junction windows** at K=185 in the evaluation — `dep_junction` CIs are wide and must not be
  quoted as precise effect sizes.
- `lam_replay = 3.0` is inherited from E1e-A and is **not re-tuned here**; a buffer change may shift
  the optimal λ, and this arm cannot see that.
- A fine-tune of one base checkpoint. Any winner owes a from-scratch confirmation.
- **No finer λ grid and no second junction threshold** will be run off the back of this arm.

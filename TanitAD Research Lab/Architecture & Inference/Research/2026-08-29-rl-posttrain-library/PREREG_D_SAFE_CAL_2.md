# PREREG — D-SAFE-CAL-2: the corrected instrument, on data I have not seen

`PRE-REGISTRATION, 2026-08-30, TanitAD_TrainingFlyWheel. Committed BEFORE any arm
of this run exists. Successor to D-SAFE-CAL, which returned VOID because I
mis-specified two committed artifacts (TRAIN-C12). PI approved the QUESTION;
this re-asks it with a corrected instrument, which is inside that approval and
not a new scope.`

---

## 0. ⛔ WHY THIS RE-TRAINS INSTEAD OF RE-ANALYSING — the load-bearing decision

Both corrections are **analysis-side**: exit 5's condition and the decomposition's
reward spec. I could apply them to the five banked arms for **zero GPU**.

⛔ **That would make this pre-registration worthless.** I have already seen those
arms' ΔR1, ΔR2, ΔR3, R5 and the post-hoc proximity decomposition. A "committed"
exit written against numbers I know is not a commitment — it is a rationalisation
with a timestamp, and it is the precise failure the VOID was reported to avoid.

⇒ **This runs on SEED 1** (the banked campaign is seed 0), so every number the
exits are read against is one I have not seen. ~40 min on the 4060 is the price of
the pre-registration meaning anything. ⚠️ **The banked seed-0 arms are NOT
re-analysed and NOT pooled with these** — that would smuggle the seen data back in.
They stay as a descriptive record only.

## 1. The two corrections

### (a) Exit 5 becomes a REPRODUCTION test — the sign was inverted

| | |
|---|---|
| **was** | *"if arm D (proximity weight 0) moves anything ⇒ instrument failure"* |
| ⛔ **why wrong** | `--reward proximity0` sets `proximity = 0.0` on top of `DEFAULT_WEIGHTS`, **which never contained `proximity`** ⇒ arm D is bit-identical to the plain default-reward arm. It is a full RL run; moving its own objective is *expected*. The exit could only ever fire, and it fired on a **perfect reproduction** (seed 0: ΔR1 −0.0660 vs banked −0.0660; collision −0.0017 vs −0.0017). |
| ✅ **now** | *"arm D must MATCH the banked no-proximity arm within the paired CI. A **DIVERGENCE** is the instrument failure."* |

⭐ A control asserted to be *inert* assumes the thing nobody is checking. A control
asserted to *reproduce a known value* is falsifiable and self-calibrating.

### (b) The decomposition gets a spec containing every varied term

`R4_component_means` is scored with the DEFAULT spec for cross-arm comparability —
and the default spec has **no `proximity`**, so the "PRIMARY diagnostic" was blind
to the only term the experiment varies. **Both are now emitted**: `R4_default`
(comparable across the whole campaign) and `R4_full` (contains every term any arm
carries). ⇒ generalised: **audit the DIAGNOSTICS against the output schema, not
only the exits.**

### (c) Adopted from the Master Mind: the harness EMITS the in-force weights

Each arm prints and records its resolved `reward_weights` dict **before training**,
so an arm that cannot vary its own objective is visible *before* it runs. ⭐ Both
this failure and MM-C5 are **a flag name trusted without measuring what it does**;
operating-standard rule 1 says a claim that decides a GPU-day must be MEASURED, and
the cheapest measurement is printing the resolved value.

## 2. Arms — REG is in the table this time

| arm | `w_anchor` | `proximity_safe_m` | `w_prox` | purpose |
|---|---|---|---|---|
| **A** | 1.0 | **2.0** | 0.5 | the test |
| **B** | 10.0 | **2.0** | 0.5 | does it survive a tight trust region |
| **C** | 1.0 | 5.0 | 0.5 | paired control at the incumbent threshold |
| **D** | 1.0 | 2.0 | **0.0** | reproduction control (see §1a) |
| **REG** | 1.0 | 2.0 | hackable | deliberate regression — the licence to read any null |

⛔ `d_safe = 2.0` FIXED, not swept. Seed **1**. 2,000 steps, `decoder_steps=2`.

## 3. ⛔ BOTH OUTCOMES, COMMITTED BEFORE ANY NUMBER EXISTS

Read in this order; the first that matches wins, and the two VOID conditions
override everything.

| # | if | then |
|---|---|---|
| **V1** | **arm D DIVERGES from the banked no-proximity arm** beyond the paired CI | ⛔ **VOID** — instrument failure. The harness does not reproduce a known value; nothing else in the table is admissible. |
| **V2** | **REG does not degrade** with paired separation on R1 or R3 | ⛔ **VOID** — the readout cannot see failure, so no null is a measurement. |
| **1** | ΔR2 falls with paired separation in **A**, and **C** reproduces the banked null | ⭐ **The threshold WAS the blocker.** Recalibrate the spec (a PI decision), re-open the RL line. |
| **2** | A ≈ C, neither separates on R2, **and `R4_full` proximity moves < 0.01 in A** | ⛔ **The threshold was NOT the blocker; the RL line CLOSES on this base model.** |
| **3** | ΔR2 falls in A but **R5 rises on the frozen 5.0 m ruler** | ⚠️ **Reward hacking** — the scored fan improved while the driven path got worse. Reject. |
| **4** | ΔR3 degrades with separation in A while ΔR2 is flat | The opposition is not threshold-specific ⇒ closes as outcome 2, noting the trust-region conflict persists. |
| **5** | none of the above | **No committed branch matches — report as-is and decide nothing.** ⛔ Do NOT retro-fit. |

## 4. What is NOT claimed, whatever happens

⛔ Nothing about refcv3. ⛔ Nothing about driving — **T0**; a capability claim needs
T1. ⛔ NON-PARITY corpus, 15 val clusters; no number enters cross-arm tables.
⛔ **`proximity_safe_m`'s DEFAULT stays 5.0 in code regardless of the outcome** —
`d_safe=2.0` is an arm parameter. The experiment informs the specification change;
it is not the change, and that decision is the PI's.

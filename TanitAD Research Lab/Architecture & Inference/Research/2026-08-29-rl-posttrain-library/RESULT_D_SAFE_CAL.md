# RESULT — D-SAFE-CAL: ⛔ VOID. The experiment ran; MY PRE-REGISTRATION FAILED IT.

**Date:** 2026-08-30 · **Owner:** TanitAD_TrainingFlyWheel · **Tier: T0 training-side**
**Pre-registration:** `PREREG_D_SAFE_CAL.md`, committed before any arm existed.
**Evidence class:** MEASURED (ours) · ⚠️ NON-PARITY corpus · paired episode-cluster
bootstrap, 4000 reps, eval-mode deterministic readout.
**PI approval:** D-SAFE-CAL approved (relayed via Master Mind; ⚠️ see TRAIN-C11 —
the relay's index was ambiguous and the referent was confirmed before acting).

---

## 0. Verdict: VOID (exit 5), selected mechanically — and the fault is mine

The exit was chosen by `dsafe_analyze.py` from the committed table, not by me
reading the numbers. It returned **exit 5 — instrument failure**, because arm D
(the weight-0 floor) moved ΔR1 **−0.0660** and ΔR3 **+0.1115**, both with paired
separation, and exit 5 says *"if D moves anything at all ⇒ instrument failure."*

⛔ **The harness is fine. THE EXIT CONDITION WAS WRONG, AND I WROTE IT.** See §3.

⭐ **I am reporting the VOID rather than overriding it.** The committed rule fired;
a pre-registration whose author may set aside a branch he no longer likes, after
seeing the data, is not a pre-registration. The honest result of this campaign is
*"no valid verdict was reached, because the exit conditions were mis-specified"* —
and that is a result about my instrument design, not about `d_safe`.

## 1. The arms (all ran clean; 5/5 produced readouts and checkpoints)

| arm | ΔR1 | ΔR2 collision (pp) | ΔR3 sel-ADE (m) |
|---|---|---|---|
| **A** w=1 `d_safe`=2 | −0.0006 [−0.017, +0.016] | −0.0195 [−0.163, +0.111] | **+0.1689** [+0.005, +0.324] **SEP** |
| **B** w=10 `d_safe`=2 | **−0.0126** [−0.022, −0.003] **SEP** | −0.1107 [−0.306, +0.052] | +0.0178 [−0.064, +0.083] |
| **C** w=1 `d_safe`=5 | **−0.0901** [−0.102, −0.078] **SEP** | +0.0130 [−0.215, +0.273] | **+0.0947** [+0.033, +0.144] **SEP** |
| **D** floor, w_prox=0 | **−0.0660** [−0.077, −0.055] **SEP** | +0.1693 [−0.117, +0.475] | **+0.1115** [+0.003, +0.207] **SEP** |
| **REG** deliberate-regression | **−0.0331** [−0.054, −0.012] **SEP** | +0.1107 [−0.195, +0.417] | **+0.0397** [+0.009, +0.076] **SEP** |

✅ **The readout is VALID.** REG degrades with paired separation on both R1 and R3
⇒ the instrument can see failure, so the nulls above are measurements rather than
blindness. That licence holds regardless of the VOID.

⚠️ **REG was NOT in the pre-registered arm table — I added it after the four arms
had run**, because the Master Mind's handover named the regression control as the
licence to read any null and my §2 had omitted it. Stated plainly because a
post-hoc arm normally deserves suspicion: this one can only ever **invalidate** my
result (its committed prediction is that it degrades), so adding it raises the bar
rather than lowering it. It is still a pre-registration defect that it was missing.

## 2. R5 — the driven path, on the frozen 5.0 m ruler

| arm | ΔR5 @5.0 m | level after |
|---|---|---|
| A | +0.0250 [0.000, +0.067] | 0.3667 |
| B / C / D | +0.0083 [0.000, +0.025] | 0.3500 |
| REG | +0.0000 **SEP** | 0.3417 |

None of A–D separates. ⚠️ **`mean_at_arm_threshold` equals the fixed-ruler value in
every row** — the dual-ruler machinery is live but the two rulers happen to
coincide on these windows. It cost nothing and it is the right design; it simply
did not discriminate here.

## 3. ⛔ DEFECT 1 — EXIT 5'S CONDITION DOES NOT DESCRIBE ARM D

`--reward proximity0` sets `weights["proximity"] = 0.0` on top of
`DEFAULT_WEIGHTS`. But **`DEFAULT_WEIGHTS` never contained `proximity`** — so arm
D is *bit-identical to the plain default-reward arm*. It is a **full RL run**, and
a full RL run moving its own objective is **expected behaviour, not a fault**.

⭐ **And it reproduces the banked arm EXACTLY.** Against the previous campaign's
`prev w=1 no-prox` row: ΔR1 **−0.0660 vs −0.0660**, collision **−0.0017 vs
−0.0017**. Four decimals, a different session, a modified script. **Arm D is the
healthiest object in this table** — the exact opposite of what the exit concluded
about it.

⇒ The condition I *meant* was **"D must reproduce the banked no-proximity arm"**,
which D **passes**. The condition I *wrote* was "D moves anything", which D cannot
possibly satisfy. ⚠️ Same family as TRAIN-C6 — a committed exit that does not name
what the instrument actually produces — with the object swapped from the *metric*
to the *arm*.

## 4. ⛔ DEFECT 2 — TRAIN-C6 RECURRED, IN THE EXPERIMENT WRITTEN TO AVOID IT

`R4_component_means` is scored with the **DEFAULT spec** so arms stay comparable —
and the default spec **does not contain `proximity`**. The decomposition that the
pre-registration calls *"the PRIMARY diagnostic"* is therefore **structurally blind
to the only term this experiment varies**.

⚠️ **I fixed exit 3's measurability by adding R5 and never checked the
decomposition for the identical defect.** TRAIN-C6's rule says *"check every
pre-registered exit against the readout's output schema"*; I checked the exits and
not the diagnostic. The rule was too narrow and I have widened it (TRAIN-C12).

**Recovered post-hoc from the saved checkpoints, zero GPU** (the TRAIN-C5
checkpoint fix has now rescued three separate analyses):

| arm | proximity before | after | Δ | weighted Δ |
|---|---|---|---|---|
| A `d_safe`=2 | −0.2502 | −0.2512 | −0.0010 | −0.0005 |
| B `d_safe`=2 | −0.2502 | −0.2495 | +0.0007 | +0.0004 |
| C `d_safe`=5 | −0.2502 | −0.2509 | −0.0007 | −0.0004 |
| D floor | −0.2502 | −0.2521 | −0.0019 | −0.0010 |

⛔ **DESCRIPTIVE ONLY — this may NOT be used to select a branch.** A decomposition
computed after the outcome is known cannot license an exit the committed rules did
not choose.

## 5. What is and is not established

✅ **Established:** the readout is valid (REG separates on both axes); arm D
reproduces the banked control to four decimals; all five arms ran clean with
checkpoints; the dual-ruler R5 machinery works.

⛔ **NOT established — and explicitly NOT claimed:** anything about whether
recalibrating `d_safe` helps. The committed exit was VOID; the descriptive numbers
in §1 and §4 point the same way the campaign always has, and **that is not a
finding, it is an observation awaiting a valid experiment.**

⛔ **`proximity_safe_m`'s DEFAULT IS UNCHANGED IN CODE (still 5.0).** `d_safe=2.0`
was an arm parameter only. The experiment informs the specification change; it is
not the change, and that decision is the PI's whatever any arm says. Nobody should
read a favourable arm here as a shipped default — and no arm here was favourable.

## 6. The successor — NOT run, needs approval

A corrected pre-registration must fix both defects **before** any arm runs:

1. **Exit 5 becomes a REPRODUCTION test:** *"arm D must match the banked
   no-proximity arm within the paired CI; a DIVERGENCE is the instrument
   failure."* ⭐ Note the sign flips — the current text treats motion as the
   alarm; the correct alarm is *failure to reproduce*.
2. **The decomposition is scored with a spec that CONTAINS every term any arm
   varies**, reported alongside the default-spec numbers so comparability survives.
   ⇒ generalised rule: **audit the DIAGNOSTICS against the output schema, not only
   the exits.**
3. Keep REG in the registered arm table rather than adding it afterwards.

⚠️ **Cost check before anyone approves it:** five arms cost ~40 min on the 4060,
so a corrected re-run is cheap. That is not a reason to skip the correction —
cheap re-runs are exactly how an uncorrected instrument gets used twice.

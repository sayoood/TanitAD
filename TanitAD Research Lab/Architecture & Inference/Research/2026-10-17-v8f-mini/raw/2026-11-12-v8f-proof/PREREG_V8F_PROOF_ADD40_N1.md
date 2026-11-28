# PREREG_V8F_PROOF_ADD40_N1 : IMAGPLAN, what the build of the tool fixed, before any rollout on a checkpoint with a lead head

Note to card `PREREG_V8F_PROOF_ADD40.md`. Frozen after the tool was built and unit-tested on the dev box (CPU, tiny models; 25 tests passed in the builder's run and in the lead's own run, 9 of 9 one-line mutations red, `imagplan/IMAGPLAN_DEFS.md`) and before the tool has run on any checkpoint with a lead head: no `cand-cv` or `cand-imag` value of any real arm exists, and no value of an arm of note ADD37_N3 has been read. The only run on real windows so far is the builder's plumbing smoke (16 windows of P_s0, label source, no unsafe rate read). The card's statistics, names and bars stay as written. Tier T1-kin. STRATEGIC: PAUSED (PI item 25).

## A1. The braking floor (card section 2: a_c(i) = max(c * a_P(i), a_B(i)))

As written, candidate c = 1.0 is ahead of the model's plan wherever the plan itself brakes harder than 6.0 m/s2 or starts below the measured speed (a_B above a_P): the same finding as note ADD26_N1 for the projection rule (DISTRULE_DEFS E3b). **Every run of the card uses `--floor cap-only`: a_c(i) = min(a_P(i), max(c * a_P(i), a_B(i))).** No candidate is ahead of the plan the model issued, c = 1.0 is the plan itself bit for bit, and the braking limit bounds only the braking the rule adds. This is the form of the banked runs of row H-V8F-PROOF-54 (amendment A1 of note ADD26_N1), so `head`, `label` and the candidate modes share it.

## A2. Control p1 (identity), the form in which it is read

* With the candidate set {1.0} alone the card rule "none safe, then c = 0.0" still brakes on a tick where the plan is unsafe, so that run cannot equal `off`. p1 first part is read as: `--cand-set 1.0 --fallback none --floor cap-only` (nothing safe leaves the plan untouched) equals `off` bit for bit, with the rule path active (source, gate, candidates, check, selection, plan replacement).
* A source that returns 100 m would be inactive under the 60 m gate and would not exercise the rule. p1 second part is read as: `--const-gap 100` replaces every forecast gap of every candidate at both horizons after the gate (the head still decides presence and the gate), with the full candidate set and the card's fallback; the dump equals `off` bit for bit.

## A3. The actions of a candidate (card section 2: "the same definition that gives the logged actions of the corpus")

MEASURED by the build (12 eval clips): the corpus acceleration channel is the forward difference of the speed at the cache step, a(f) = (v(f+1) - v(f)) / 0.2 s (max difference 1.2e-7); the yaw channel is the one-frame backward yaw difference of the native track at the cache rows. The pinned tool, when it executes a plan, writes a second spelling of the acceleration (the 0.1 s backward difference of the track), about 0.11 m/s2 away from the first on the median. **The card form is the first (`--action-def logged`): the acceleration of a candidate is the forward difference of the speed of the candidate's own track (built by the pinned tool's own interpolation of the knots), the yaw channel the backward yaw difference of that track.** It is the definition the world model was trained on. The second spelling is not used by any run of the card.

## A4. Controls p4 and p5: stay as written; what is known before the measurement

* p4: the builder's plumbing smoke on 16 windows read a correlation of 0.982 on the acceleration channel (below the bar 0.99), 0.9956 on the yaw channel, and a median absolute acceleration difference of 0.040 m/s2 (inside the bar 0.1). p4 is measured on the full window set before any mode is read and reported against the bars as written. If a bar is not met, that is stated beside every name of the card, and a form in which the control is read is given only by a further note with the measured cause, written before any unsafe rate of a card arm is read.
* p5: the banked read of `imag_dump.py` uses the stored state file (float16) and the tool recomputes the tick state from the tokens (float32). The difference is measured and stated; the bar of the card (1e-4 m, or the difference of a twice-run compare) is the bar.

## A5. Order of the runs

1. Plumbing on arms that exist (LFT, LA; they name nothing): p4, p5, p7 from the log of one `cand-imag` run per arm, then p1 and p3. No unsafe rate of these runs is read before p1, p3, p4, p5 and p7 are on file.
2. Tool checks on LFT and LA in all modes with p6 (reported, they name nothing).
3. The card arms (X and RLFT) once note ADD37_N3 has named its arm; if it names none, section 3 of the card applies.

## A6. Options that exist in the tool and are not used by the card

`--floor card`, `--fallback none` outside p1, `--action-def executed`, `--cand-set` other than the 11 values outside p1. A run with one of them is a tool check and names nothing.

# PREREG_V8F_PROOF_ADD39 : LEADSCR, the lead read as an input of the plan head, trained from the start

Card of the v8F proof programme, rule line. Frozen before the arms file of these arms exists on the pod and before any gate, step or read of them. It is the step that section 8 of card `PREREG_V8F_PROOF_ADD35.md` commits for the case that the input is not used after a fine-tune. Tier of every read: T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react). No number of this card is a driving-performance number. STRATEGIC: PAUSED (PI item 25).

## 1. What is known (MEASURED, row H-V8F-PROOF-71)

* Fine-tuned for 6,000 steps from a zero start, the input layer stays near zero (largest weight 0.015 to 0.026). With the layer zeroed afterwards the rollouts change by millimetres to centimetres of gap and not in the unsafe rate U. The input alone moves U no more than a rerun of the control does; with the own-plan term it does not place the braking (`LEAD-INPUT-NO-PLACEMENT`).
* Two readings are open: the lead read is not precise enough to be of use to the plan head, or a short fine-tune from a zero start does not teach a trained plan head to use a new input. This card separates them at the cost of two arms.

## 2. Arms

From the start, 19,164 steps, the argv of arm LS (lead head with gradient) plus `--lead-cond on`, in the tree of card ADD35 (`tree_leadcond`, in which the flag at its default is bit for bit the tree of LS): LCS_s0 and LCS_s1 (seeds 0 and 1). Compared with LS_s0 and LS_s1 (same seeds, same argv apart from the flag, trained earlier). Not trained: LCSZ_s0 and LCSZ_s1, the checkpoints of LCS with the two tensors of the input layer set to zero, rolled with the same pinned tool.

## 3. Reads

Reader, windows and cells as row H-V8F-PROOF-71 (642 lead windows of 56 episodes for U, contact and gap bias; 2,399 windows of 141 episodes for the position error over 6 s and the cost cells of the four families); paired on windows; two-level bootstrap (seeds, then episodes), B 2,000; spread of the six seeds of arm G beside. Pairs: LCS against LS; LCS against LCSZ. Beside: the size of the learned layer; the train-log sums of the lead head terms of LCS against LS (does the input change the lead read).

## 4. Outcomes (both seeds needed)

* Use of the input, LCS against LCSZ: `LEAD-INPUT-SCRATCH-USED` if, with the same sign at both seeds and a two-level interval excluding 0, U differs by at least 0.010 or the gap bias at 2.0 s differs by at least 0.10 m in size at both seeds. Else `LEAD-INPUT-SCRATCH-UNUSED`. (A size bar is set this time: in row H-V8F-PROOF-71 a difference of 4 mm counted as used.)
* Effect, LCS against LS: `LEAD-INPUT-SCRATCH-CUTS-UNSAFE` if U of LCS minus U of LS is below 0 at both seeds, the two-level interval excludes 0, and the reader names it outside the spread of the six G seeds; `-WORSE` the same above 0; else `LEAD-INPUT-SCRATCH-FLAT`. Cost by the rule of card ADD11: `LEAD-INPUT-SCRATCH-NO-COST` or `-COSTS` with the cells.
* Two arms trained from the start differ also by training variance; a named effect says the pair lies outside the spread of six seeds of the base arm, not more.

## 5. Controls (first; a failed control names nothing)

* s1: pod gate of each arm: declared equals built for every lever string (with `levers.lead_cond=on`), live step finite, 19,164 steps; the live probe of the lever: gradient on the layer above 0 and its zero-input control exactly 0.
* s2: the argv of LCS equals the argv of LS apart from the flag, the tree and the out folder (dry run of the chain script).
* s3: reader controls ALL_PASS in every read.
* s4 (known zero): the zeroed checkpoint of an arm rolled twice gives equal dumps (largest absolute difference 0.0), or the difference is stated and the bar of section 4 is read above it.
* s5 (known positive of the reader): the pair LFTRL against LFT of row H-V8F-PROOF-71 is in the same read and is named better on U.

## 6. What follows (committed before the data)

* `-USED` and `-CUTS-UNSAFE` with no cost: the plan reacts to a lead it is told about, with no rule anywhere; next the own-plan term on the LCS base against the term on the LS base (the placement test of card ADD35 on a model that uses the input), two bases.
* `-USED`, not `-CUTS-UNSAFE`: the input is read and does not make the plan safer by itself; next the same term pair on the LCS base, since placement under the term is the claim.
* `-UNUSED`: adding two numbers to the ego token through a layer that starts at zero is not a way the plan head takes up; next another place of entry for the lead read (its own token for the plan heads), built and checked before any arm. The finer lead read stays the open decision of the PI named in row H-V8F-PROOF-56.
* In every case the rule at inference stays the checked way to keep the distance.

## 7. Limits

Two seeds; training variance between two from-start arms; label lead for the lead head at training, vision only at inference; logged frames; one corpus; one place of entry for the input.

## 8. Cost (ESTIMATED)

Two arms of 19,164 steps at about 2 h each beside two other jobs, two short rollouts of the zeroed checkpoints: about 4 to 5 L40-h on the pod, never more than three GPU jobs at once.

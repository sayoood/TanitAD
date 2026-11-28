# PREREG_V8F_PROOF_ADD42_N3 : the LS type is named; the dose step that card section 8 commits for its outcome, run once

Note to card `PREREG_V8F_PROOF_ADD42.md` and notes `_N1`, `_N2`. Frozen after the read of the LS type under notes N1 and N2 (`qhead_n1/read/READ_QHEAD_N1.json`, read 2026-11-19 at 05:17Z) and before any state, fit or rollout of the LS dose arms. The rollouts of the LSD dose arms of note N2 section 3 are in the pod queue; none of their outcomes is known. Tier: T0 for the fits (heads on frozen states of training clips; no model is trained), T1-kin for the rollouts (kinematic rollout on the model's own actions, logged frames, the lead does not react); never driving performance. STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument).

## 1. The read of the LS type (MEASURED, T1-kin, 2,399 windows of 141 episodes, two checkpoints)

Controls h1 to h8: all pass (h1 carried from the first run, rollout tool byte-identical; h3 exact in the rollout and within 3.9 mm across machines against the tolerance 0.02 m, planted difference caught; h5 on the check clips 0.179 and 0.189).

* `QHEAD-REACHES-ORACLE-QE`: model-induced unsafe rate per checkpoint, no rule 0.1139 and 0.1352, trained output 0.0391 and 0.0783, label-fed rule 0.0391 and 0.0516; share of the label-fed fall reached 1.000 and 0.681 (bar 0.5 at each); trained output minus no rule -0.0658 [-0.1055, -0.0269].
* `QHEAD-U-EQUAL-QE`: trained output minus calibrated map -0.0062 [-0.0251, +0.0167] (map 0.0552 and 0.0747).
* `QHEAD-COST-EQUAL-QE`: position error over 6 s, trained output minus map +0.074 m [-0.074, +0.233].
* `QHEAD-COSTS-QE` against no rule (cost in the LONGITUDINAL and the TACTICAL family), as for every form of the rule.
* Beside: against the head-fed rule -0.0258 [-0.0462, -0.0081]; against the label-fed rule +0.0133 [-0.0103, +0.0411]; contact rate against the label-fed rule +0.0117 [+0.0029, +0.0230] (the label-fed rule has none); coverage on the evaluation windows 0.145 and 0.116.

The LSD type is not named under note N1 (h5 at `LSD_s0`); its dose arms are those of note N2 section 3.

## 2. Why the LS type is fitted once more, stated

Card section 8, second bullet, frozen before any data: for `QHEAD-U-EQUAL` with `QHEAD-COST-EQUAL` "the dose step is run once; if it is equal again, the limit is in what the tokens hold". Note N2 section 4 says that card section 8 applies to the LS type once it is named; note N2 section 3 says the LS type is not fitted again in the dose step that the failed coverage of the LSD type triggered. The two lines are in tension. The card decides: its dose step for this outcome is run, once, with its own arm names and for its own question. The read of section 1 stays the read of record of the LS type at 800 fitted clips; the two reads are not pooled, and both are reported whatever the second says.

## 3. Arms, controls, names

* Arms: `LSQF_s0` (threshold 0.4), `LSQF_s1` (threshold 0.5): the fit of note N2 section 3 on the states of all 4,572 training clips (fit 0 to 3,047, select 3,048 to 3,809, check 3,810 to 4,571) of the checkpoints `LS_s0`, `LS_s1`.
* Volume, named before it runs (MEASURED on the LSD checkpoints): 1.19 GB of states on the pod volume, CPU only (about 6 min per checkpoint), fits about 1 min each on the CPU, rollouts about 4 min each on one GPU slot. Never a fourth GPU job.
* Controls: those of note N2 section 3 (h1 to h8 with h5 on the new check clips; d1 on the label pack, which must be the same file by sha256 as the one built for the LSD type; d2 for the two LS checkpoints, with the planted shift that must fail).
* Names: those of card section 7 with the suffix `-QF`, for the LS type.

## 4. What follows (committed before the data)

* `QHEAD-U-EQUAL-QF` with `QHEAD-COST-EQUAL-QF` (equal again): by the card, the limit is in what the tokens hold at this grid. The pilot of card ADD41 (finer grid) ended without a name, so that question stays open; the probe-only repeat with more seeds is the PI decision already listed. The trained output and the calibrated map are then two equal forms of the read; the map stays the form carried forward (it needs no fit).
* `QHEAD-CUTS-MORE-QF`, or `QHEAD-U-EQUAL-QF` with `QHEAD-CHEAPER-QF`: the trained output at the dose fit replaces the map for the LS type; next it is the lead read of the planning tool of card ADD40.
* `QHEAD-CUTS-LESS-QF` or `QHEAD-DEARER-QF`: the map stays; reported as written.
* h5 fails for an LS checkpoint at the dose fit: the LS dose arms are not named; the read of section 1 stands by itself.

## 5. Limits

One fit per checkpoint and seed; two checkpoints per arm type, one training run each, so a separated interval says what another draw of episodes would say; the rollouts are deterministic given the head; the second fit of the LS type is a second look at the same checkpoints and is reported beside the first, never in its place.

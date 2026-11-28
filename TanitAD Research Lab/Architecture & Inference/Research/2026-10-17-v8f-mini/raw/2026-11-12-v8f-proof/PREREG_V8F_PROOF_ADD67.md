# Card PREREG_V8F_PROOF_ADD67 (HEADBOUND10K): a curvature bound on the heading of the kinematic executor, under the pinned readout, under readout K and in the candidate; no training

Card of the v8F proof study, written by the coordinator fork after the read of card ADD66 with its note N1 (row 120) and before the read of card ADD65, before any tool of this card exists and before any of its runs (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; logged camera frames; the lead does not react); never driving performance.

## 1. What is known (MEASURED)

* The pinned tool takes the heading of every executed frame from the tangent of the executed path and holds the previous heading only while that tangent is shorter than 1 mm per 0.2 s (`EPS_M`). There is no vehicle model.
* Note N1 of card ADD66 (row 120, descriptive): in the programme's convention (`R0`), 3.6 s into the rollout, the executed heading is off the logged heading by 52.9 degrees on average where the speed at the tick is below 2 m/s (median 39.8 degrees, n 310), by 13.3 degrees at 2 to 8 m/s (median 4.8 degrees, n 1,874) and by 3.4 degrees above 8 m/s (median 1.9 degrees, n 2,614); above 20 degrees in 12 and 13 % of all rows. The rise of the strategic checkpoint error grows with that offset (+3.4 m at 30 m up to 2 degrees, +25.5 m above 20 degrees).
* The heading sets the frame the next plan is laid in and the yaw history the model reads. A heading that follows millimetres of plan jitter near a standstill is one no car produces. How much of the T1-kin errors of the four families comes from it is not known.

## 2. Question

Does a bound that a car obeys (the heading turns by at most a maximum curvature times the distance travelled) lower the T1-kin errors, family by family, and does it change the unsafe rate: (a) in the programme's convention, (b) under readout K, (c) in the inference-side candidate of card ADD65?

## 3. The mechanism (study wrapper; the pinned tool is not edited)

New wrapper `integ10k/v8f_t1_eval_reader_rule_rzgfh.py`: the wrapper of card ADD65 (`v8f_t1_eval_reader_rule_rzgf.py`, sha256 969936595512ef80) with one more parameter `--kappa-max K` (1/m), applied by asserted text edits. With the parameter, after the chosen readout has filled the frames of the executed steps (positions, headings, speeds), the heading column is passed through the bound, frame by frame at 10 Hz, starting from the heading before the block: the heading of a frame is the readout's heading if it differs from the previous bounded heading by at most K times the distance between the two frame positions, else the previous bounded heading turned by exactly that much towards it. Positions and speeds are not changed. Without the parameter the wrapper gives the parent's result bit for bit. K = 0.2 1/m (a turning radius of 5 m) is the arm; K = 0.1 is a dose beside. The bound is a study device: it is one constraint of a vehicle model, not a vehicle model (the positions still follow the plan's steps exactly).

## 4. Arms

Models `UE_s0`, `UE_s1`, reader arms, maps, rule parameters, evaluation cache, window grid and pinned tool as cards ADD46 to ADD65; 2.0 s cycle; no limit on the plan.

| run | rule | readout | bound | role |
|---|---|---|---|---|
| `R0Hb20` | none | pinned | 0.2 | (a) the bound in the convention |
| `R0KHb20` | none | K | 0.2 | (b) the bound under readout K |
| `R0KHb10` | none | K | 0.1 | dose, names nothing |
| `RLcKF5Hb20` | label-fed, every 5 steps | K | 0.2 | (c) the candidate at label quality with the bound |
| `RQ50KF5Hb20` | reader-fed, every 5 steps | K | 0.2 | (c) the candidate in its deployable form with the bound |
| `R0HbID`, model seed 0 | none | pinned | 1e9 | control h1: must equal the banked `R0` |
| `R0KHbID`, model seed 0 | none | K | 1e9 | control h2: must equal the banked `R0G0X` |

Each run except the controls at both model seeds: ten rollouts. Reused without a rerun: `R0`, `R0G0X`, `RLc`, `RQ50`, and from card ADD65 `RLcKF5`, `RQ50KF5` (admitted only if the controls of card ADD65 pass; else the pairs of (c) are not named and the reason is stated).

## 5. Reads

Headway reader and family reader of card ADD46 (56 lead episodes, 642 lead windows, 562 scored for U; 141 episodes, 2,399 windows), and for STRATEGIC the instrument of card ADD66 with Amendment 1 (`pos` at 30, 50, 80 m at the third tick). Pooled over the two model seeds, paired episode-cluster bootstrap on the same windows; the two model seeds are the only replicate for training variance; the rollout does not sample.

* HP = `R0Hb20` minus `R0`; HK = `R0KHb20` minus `R0G0X`; HC per form = the candidate with the bound minus the candidate (`RLcKF5Hb20` minus `RLcKF5`, `RQ50KF5Hb20` minus `RQ50KF5`). For each: U, C, F and every family cell.
* Beside, naming nothing: `R0KHb10` minus `R0KHb20` (the dose); `R0KHb20` minus `R0` (readout K with the bound against the convention); the candidate with the bound against the convention with the rule (`RLc`, `RQ50`).
* Reported without a bar: per run the share of executed frames at which the bound acted, by the speed bins of note N1 of card ADD66; the heading offset from the logged heading at the third tick by the same speed bins (the quantity of note N1) for each run and its pair partner.

Family rule: named cells LONGITUDINAL speed error and along-track error, LATERAL heading error and yaw-rate error, TACTICAL tactical ADE, STRATEGIC `pos` at 30, 50 and 80 m; a family reads BETTER for a pair if every named cell reads BETTER, WORSE if every named cell reads WORSE, NOT-SEPARATED if none reads either, else MIXED. Never pooled over families.

## 6. Names (fixed here)

For each pair P in PINNED (HP), K (HK), LABEL-CAND, READER-CAND (HC):

* `HEADBOUND10K-<P>-<FAMILY>-BETTER`, `-WORSE`, `-MIXED` or `-NOT-SEPARATED` for LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC.
* `HEADBOUND10K-<P>-U-LOWER`, `-U-HIGHER` or `-U-NOT-SEPARATED` (one sign at both model seeds and the pooled interval on that side of 0).

What follows, fixed now:

* A pair reads BETTER on at least one family, WORSE on none, and U not higher: the unbounded heading of the kinematic executor is a measured term of the T1-kin errors in that setting. It goes to the PI and the other session with the decision card as a proposal for the definition of the tier (a bound a car obeys in the executor). Adopting it changes the evaluation convention; that is their decision, and this card changes nothing by itself. Until then the rows of this line carry the limit that the executor has no heading bound.
* All four families NOT-SEPARATED for a pair: the heading offsets near a standstill do not move the reads in that setting; reported; the limit stays as a note on the tier.
* A family WORSE or U higher: reported as written with the table and the share of frames at which the bound acted; nothing is proposed for that setting.
* The dose at 0.1 reads worse than 0.2 on a family: said as the point where the bound starts to cut turns a car makes.
* A control fails: nothing is named; the tool is repaired and the runs are repeated under new run names, logged in `TIMES_NOTE.md`.

## 7. Controls (all must pass; else nothing is named)

* e1: the pod read reproduces the banked dev-box read of card ADD46 leaf for leaf.
* h1, h2 (wrapper edit, known values): the dumps of `R0HbID` and `R0KHbID` equal the banked dumps of `R0` and `R0G0X` at model seed 0 array for array (141 files, 0 differing each), with marker names no earlier card used (`hb1cmp`, `hb2cmp`; the tool check `hb6check`). The run names carry `Hb` because a run `R0HID` of an earlier card exists on the pod.
* h3 (the bound holds and does nothing else): in every run with a bound, from its log, the largest excess of a heading step over K times the distance travelled is at most 1e-9 rad; at every frame where the bound did not act the heading equals the readout's heading exactly; positions and speeds of every frame equal those of the readout (largest difference 0.0).
* h4 (tool check before any run, on the pod CPU, synthetic plans): a plan that jitters by millimetres around a standing car keeps its heading within the bound; a plan along an arc of 10 m radius at K 0.2 keeps the readout's heading at every frame; the same arc at K 0.05 is bounded at every frame by exactly K times the distance; a huge K gives the parent's frames bit for bit. Planted faults that must fail: the bound not applied, and the bound applied with the turn of the wrong sign.
* h5 (the first block, known value): the first 10 executed positions of every run equal those of the banked run with the same rule form and model seed (largest difference 0.0): the bound changes headings only.
* h6 (counts), h7 (reused dumps unchanged), h8 (reader controls of the headway and family readers and controls s1 to s9 of card ADD66 on the runs read), h9 (checkpoint sha256; the wrapper of the runs has the sha256 the tool check ran with).

## 8. Limits

T1-kin: no vehicle model beyond this one bound; positions follow the plan's steps even where the bounded heading does not point along them; the lead does not react; the camera frames are the logged ones. One value of the bound and one dose. Two model seeds; the episodes are read again with no adjustment for the repeated reads. The STRATEGIC cell is the study instrument of card ADD66. The card was written knowing rows 110 to 120 and note N1 of card ADD66, which is where the bound comes from.

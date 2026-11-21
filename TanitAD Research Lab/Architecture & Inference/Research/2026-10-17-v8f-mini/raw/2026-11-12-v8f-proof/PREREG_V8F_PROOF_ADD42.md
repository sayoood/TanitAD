# PREREG_V8F_PROOF_ADD42 (QHEAD): a low-quantile gap output trained on the model's states, in place of the calibrated map of card ADD31

Frozen 2026-11-18 before its tool exists and before any number of it. Written while the pilot of card ADD41 waits for its slot on the pod (no pilot number exists). Tier of every read: T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react); none is driving performance. Family served: LONGITUDINAL (distance keeping). Evidence class of every number cited here: MEASURED, row H-V8F-PROOF-59 unless said otherwise.

## 1. Why

Card ADD31 section 8 committed this card: its outcome was `QCAL-REACHES-ORACLE` for both head arm types with `QCAL-CHEAPER` on LS (row 59), so "the next card trains it into the head (a low-quantile gap output on the model's states, all training clips) and reads the same cells". The map of card ADD31 sees only the head's own gap read; two frames with the same read and different scenes get the same margin. A head on the state can make the margin depend on what the state holds beyond the read. The label-fed rule is still ahead of the map on LS (U map minus label +0.0196 [+0.0012, +0.0391]) and the map makes more contact than the fixed margin (LS +0.0125 [+0.0049, +0.0213]): there is room to gain. The card is inside the present authority (a head on frozen states of banked checkpoints; no model is trained and no frame is re-encoded).

## 2. The lever

The vision-fed distance rule exactly as in arm QC of card ADD31 (presence threshold `P*` of card ADD30, the head's closing-speed read as in QC), with the gap

`g_used = min(g_head, q_head)`

where `q_head` is a second gap output read from the same state input as the banked lead head of the checkpoint: a head of the same shape as the gap branch of that head, freshly initialised with the seed of the base, trained with the pinball loss at the quantile 0.20 on the label gap (the coverage of cards ADD30 and ADD31: the gap the rule uses lies above the label gap in about one row of five). The banked lead head and the model are not changed. The rule never gets a gap larger than the head read.

## 3. The fit (training clips only; no evaluation file is read)

Rows: the banked states of the checkpoint on the 1,200 train probe clips (clips 600:1800), rows with a full history and a label lead with a finite gap (any distance): the rows of the map of card ADD31 (32,025 per checkpoint there). The last 200 of those clips in manifest order are held out of the fit and used only for control h5; the fit uses the other 1,000. Optimiser, steps, batch and input normalisation: those of the banked lead-head fit, unchanged; the loss line is the only difference (recorded with file and line in the definitions file of the tool). The fit writes the head, the row counts, the coverage on the fitted and on the held-out clips, and the sha256 of every file it read. This first step uses the probe clips so that it differs from the map of card ADD31 in the input of the margin alone (state against read), on the same rows; the fit on all training clips named in card ADD31 is the dose step of section 8.

## 4. Arms

`<BASE>QH_s<seed>` on LS_s0, LS_s1, LSD_s0, LSD_s1: four rollouts on the pod, one at a time, never as a fourth GPU job, memory check before each (28 GB of the container limit free). Control rollout `LSQHI_s0`: the same tool with the quantile output replaced by the head read itself (`q_head = g_head`), threshold 0.5. Budget ESTIMATED: fits under 0.2 pod-h, rollouts about 2 pod-h.

## 5. Read

As card ADD31 section 5, with the reader versions r2 of row H-V8F-PROOF-84: QH against no rule, against the head-fed rule (RH), the label-fed rule (RL), the fixed margin (MC) and the map (QC, the primary partner); unsafe rate U, contact rate and headway cells on the 642 lead windows of 56 episodes, position error over 6 s on all 2,399 windows of 141 episodes; paired on windows, two-level bootstrap (seeds, then episodes), B 2,000, rng 0. Inference is deterministic, so a difference between QH and QC of one checkpoint is the gap source alone. Share of the label-fed fall reached as in card ADD30. Families: LONGITUDINAL (U, contact, headway, speed cells of the family reader), LATERAL and TACTICAL from the family reader, STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument).

## 6. Controls (a failed control is reported first and the outcomes it carries are not named)

* h1: the control rollout `LSQHI_s0` gives arrays equal to the banked head-fed dump LSRH_s0 (0 files differing).
* h2: no plan is moved ahead in any rule log of this card (lift 0).
* h3: at the first tick, on ticks the head calls present, the gap the rule used in each QH run equals `min(g_head, q_head)` recomputed outside the rollout from the banked states of the evaluation windows, within 1e-4 m.
* h4: each head file lists only training files, with their sha256; the fit refuses a path of the evaluation split.
* h5: coverage on the 200 held-out probe clips: the share of rows with `q_head` above the label gap lies in [0.15, 0.25] for every checkpoint. Outside it the quantile was not learned; the arm of that checkpoint is not named.
* h6: the fit reaches its parameters and learns a quantile: non-zero gradient on every parameter of the new head, exactly 0.0 on a control parameter outside the graph and on every parameter of the banked head and the model; on planted states with a known gap and noise that grows with the gap, the fitted coverage is 0.20 within 0.03; a mutation that swaps the loss for the absolute error (quantile 0.5) makes a test fail; a mutation that feeds the head read instead of the state makes a test fail.
* h7: the reader controls of every read; control g5 of card ADD30 (the deliberate regression is read as worse) carries the comparison outcomes of this card too.

## 7. Outcomes (per arm type LS and LSD, both seeds needed)

* `QHEAD-REACHES-ORACLE`: the share reached is at least 0.5 in both seeds and the two-level interval of U(QH) minus U(no rule) lies below 0. Otherwise not named.
* Against the map, U: `QHEAD-CUTS-MORE` if U(QH) minus U(QC) is below 0 in both seeds with a two-level interval excluding 0; `QHEAD-CUTS-LESS` if above 0 in both seeds with an interval excluding 0; `QHEAD-U-EQUAL` otherwise.
* Against the map, cost: position error over 6 s on all windows, QH minus QC, paired: `QHEAD-CHEAPER` if below 0 in both seeds with a two-level interval excluding 0; `QHEAD-DEARER` if above 0 likewise; `QHEAD-COST-EQUAL` otherwise.
* Cost against no rule by the rule of card ADD11: `QHEAD-NO-COST` or `QHEAD-COSTS` with the cells.
* Beside, naming nothing: QH against the label-fed rule and against the fixed margin on U and on contact, the mean amount taken off the read at the first tick, and the coverage on the evaluation windows.

## 8. What follows (committed before the data)

* `QHEAD-CUTS-MORE`, or `QHEAD-U-EQUAL` with `QHEAD-CHEAPER`: the trained output replaces the map. Next: the dose step (the same fit on all training clips; needs the states of all 4,572 training clips on the pod, a pod job named with its volume before it is run) and the trained output as the lead read of the planning tool of card ADD40.
* `QHEAD-U-EQUAL` with `QHEAD-COST-EQUAL`: the state adds nothing to the read at this number of rows. The dose step is run once; if it is equal again, the limit is in what the tokens hold, and the outcome of card ADD41 says whether a finer grid lifts it.
* `QHEAD-CUTS-LESS` or `QHEAD-DEARER`: the map stays the form carried forward; reported as written, no retuning of the quantile after the fact.

## 9. Limits

Two bases per arm type, one corpus; a separated two-level interval from two bases is necessary and not sufficient for a claim about the lever (rule H-ESTIM-SEED-1); the lead does not react (T1-kin); the fit sees 1,000 clips; the quantile 0.20 is taken from cards ADD30 and ADD31 and is not tuned here; the head's closing-speed read is kept as in QC although row H-V8F-PROOF-62 found it without effect, so that QH and QC differ in one thing.

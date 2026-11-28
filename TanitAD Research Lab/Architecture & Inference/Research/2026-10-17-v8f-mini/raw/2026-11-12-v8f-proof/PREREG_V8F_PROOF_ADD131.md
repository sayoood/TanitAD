# PREREG_V8F_PROOF_ADD131 — BOXFUSE10K: the joined gap read in the lead rule of the candidate

Two tiers, each number stamped: T0 (recorded frames; a diagnostic of a perception read, not driving performance) for the gap error of the joined gap file, and **T1-kin/hb0.2** (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m) for the rule, models `UE_s0` and `UE_s1`. Written after rows 198 and 199 (cards ADD129, ADD130: the box read joined with the token head reads the gap about ten per cent better at T0, under three fits of the locator) and after the go of the PI of 2026-11-23, verbatim "Yes, card first", to the question "Run the joined gap read in the lead rule at T1-kin (detector pass on rollout frames, GPU)?". Frozen before any tool of this card and before any value of it exists. This card goes back to the PI with its cost before any GPU pass. No training, no fit, no encode, no download.

## 1. Question

Rows 172 to 178 found that a more exact camera gap from more clips cut the T0 error and moved no driving measure, and row 183 found that a margin on the gap closes the unsafe rate at readout K at a price in falling behind, and does not close it at the convention. Does the joined gap, which is more exact by about ten per cent against the token head, move contact, the unsafe rate or falling behind of the candidate's lead rule?

Stated guess before data: the gain at T0 is of the size that moved no driving measure in row 172; the expected reading is not separated at the convention, with a possible small cut of the unsafe rate at readout K.

## 2. Gap files

* Base file: `VGAPLE.npz` of card ADD106 (sha256 begins 55de2f7c), the gap file of the candidate form GEc (14,237 rows of the 141 evaluation clips).
* Detector pass: the detector and weights of card ADD127 (used as they are), the wide window of 32 degrees at 768 pixels, on EVERY cache frame of the 141 evaluation clips. The pass reads no label: it is called without the selection of lead frames and without the label box folder. The tele read is not used (row 197: it does not add).
* Box gap per row: the pick by the direction of the locator `LOC_s0` (vision only, banked predictions on the evaluation frames), the size-to-gap map of row 197 (fitted on the 500 train clips with the label pick; labels are allowed in fitting).
* Joined file `VGAPJE.npz`: on every row where the base file has an output and the box read has a pick, when the box gap and the base gap differ by at most a factor of 1.5, the geometric mean of the two; on every other row the base gap. A row with no output in the base file has none in the joined file, so the rows the rule treats as a lead are those of the base file. Every other array of the file is unchanged.
* Joined file with the margin `VGAPJE90.npz`: 0.90 times the joined gap (the margin that the PI adopted for readout K, row 183).

## 3. Arms and runs

The candidate form GEc (reader's own presence, `--vis-gap-file`, `--vis-closing`, no label at inference), wrapper `v8f_t1_eval_reader_rule_rzgfhqetjv.py` (28bc31f21ecb4ccb) unchanged, each new run called exactly as the banked GEc run of its setting with the gap file exchanged.

| setting | new runs | banked cells |
|---|---|---|
| K0 (readout K, no pre-roll, pinned grid, rule every 5 steps) | `JEc` (joined file) and `JA90` (joined file with the margin), both model seeds | GEc, A90, RLc, RR |
| CONV (pinned readout, no pre-roll, pinned grid) | `JEc`, both model seeds | GEc, RLc, RR |

Six rollouts. A changed gap input of the rule is a design decision: as a study arm it is measured here; its adoption, and whether a detector enters the perception path, stay with the PI and the other session.

## 4. Measures, estimator, names

All controls of section 5 have to pass before any name or value is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

T0, on the 3,841 rows with a label gap: the mean absolute gap error of the joined file divided by that of the base file; cluster bootstrap over the 141 clips, 2,000 draws, seed 131. `BOXFUSE10K-T0-JOINED-BETTER` (interval below 1), `-T0-JOINED-WORSE` (interval above 1), `-T0-JOINED-NOT-SEPARATED` else.

T1-kin, per setting on its windows with a lead (642 per model seed): contact C, unsafe U, falling behind by more than 10 m F; levels pooled and per model seed; pairs on the same windows, two-level bootstrap over model seeds and evaluation episodes, 2,000 draws: `JEc` minus GEc at K0 and at CONV, `JA90` minus A90 at K0, and each new arm minus RLc (the label form).

* Per pair P of (`K0-JEc`, `CONV-JEc`, `K0-JA90`) and measure M of UNSAFE, CONTACT, FALLING-BEHIND: `BOXFUSE10K-<P>-LOWERS-<M>` / `-RAISES-<M>` / `-<M>-NOT-SEPARATED` (interval of the pair below 0 / above 0 / else).
* One name per pair P: `BOXFUSE10K-<P>-JOINED-GAP-HELPS` (lowers at least one of the three measures and raises none), `-JOINED-GAP-COSTS` (raises at least one and lowers none), `-JOINED-GAP-TRADES` (lowers one and raises another), `-JOINED-GAP-NOT-SEPARATED` (none separated).

Four families per setting as in card ADD117: LONGITUDINAL (the measures above and the engine read of each pair), LATERAL and TACTICAL (engine family reads of each pair), STRATEGIC (the strategic grid at T1-kin for each pair, position error at the marks of 30, 50 and 80 m). The interval answers what another draw of episodes and of these two model seeds would say, not what another training run, another detector or another fit of the locator would say.

## 5. Controls

* j0 detector pass: weights hash as card ADD127; the window is the card's; the detector was run on every decodable cache frame of each of the 141 clips; the pass was called with no label folder and no selection of lead frames (record of the call).
* j1 gap files: the joined file differs from the base file in `gap` only; it has an output exactly where the base has one; on a gated row it equals the geometric mean within 1e-9 m and elsewhere the base gap exactly; the tool with a gate factor of 1.0 gives the base arrays exactly; the margin file is 0.90 times the joined file on every row with an output; the record holds the hashes and the counts of rows with an output, with a pick, and gated.
* j2 composer identity at T0: with the banked detector files of card ADD127 and the three fits of arm D the composer reproduces the errors of `FG` of row 198 within 1e-6 m.
* j3 the two passes agree: on the frames that both passes read (frames with a labelled lead of the 141 clips) the pick by the locator exists in both or in neither on at least 0.99 of the frames, and where it exists in both the box gaps differ by at most 5 per cent on at least 0.99 of them (two runs of one detector on the same pixels in other batches).
* j4 the gate is live: with the base gaps permuted across the rows with an output (seed 131) the share of picked rows that pass the gate is at most 0.75 of the share in place; the gate passes on at least 0.30 of the rows with an output and a pick.
* j5 wrapper and arguments: the wrapper has its hash; for each new run the first line of its run log equals that of the banked GEc run of its setting in every field both carry.
* j6 identity of the banked cells: the reader reproduces within 1e-4 the banked levels of GEc, RR and RLc at K0 and at CONV and of A90 at K0 (rows 179, 181, 183).
* j7 tests: the kits' tests pass in the lead's runs before the queue start and before the controls stage; outputs outside the kit folders.
* j8 engine: the engine's own controls of each grid read and of the strategic grid pass.
* j9 counts and rule logs: every new run has its done marker and 141 dump files; its rule log names the hash of its gap file and the closing switch, and the gap the rule received at steps with an output equals the file's gap within 1e-3 m.

The first controls run on the pod is the run of record. A clause that fails for a reason of construction goes into a note before any value is opened; a missed bar of j3 or j4 fails the card as written.

## 6. Cost and stop rule

ESTIMATED from measured rates: the detector pass reads about 14,240 frames at the measured 36 frames per second of detector time, 0.11 GPU-h, with about 12 minutes of wall time for the video decode; six rollouts at the measured 0.085 pod-h each (card ADD115: four rollouts in 0.34 pod-h), 0.5 pod-h, CPU-bound; the reads are CPU minutes. In all about 0.75 pod-hours on the fork pod, of which 0.11 GPU-h. Stop rule: 0.3 GPU-h of detector time or 1.5 pod-hours in all; above either the queue stops and the card goes back to the PI. The fork pod is shared: the state file is set to busy for the pass and the rollouts. About 30 MB of new files on the pod.

## 7. Committed readings

* `JOINED-GAP-HELPS` at K0 or at CONV: the joined gap enters the candidate spec as a study-arm option with its measured values at both settings and goes to the PI and the other session as a design decision (a detector in the perception path). Next card: the replicate on the second pair of models (`UE_s2`, `UE_s3`) and the aged setting (pre-roll of 30 steps, 6 s).
* `JOINED-GAP-NOT-SEPARATED` at both settings: a gap that is ten per cent more exact at T0 does not move the rule's measures, in line with row 172; the gap line of the candidate is then limited by something other than the mean error of the gap on recorded frames, and the next arm is read from the beside values (the error on the rule's own steps, the far leads, the share of steps where the two files differ by more than a metre). No detector is proposed for the perception path on this evidence.
* `JOINED-GAP-COSTS` or `-TRADES`: as written, with the measure that pays named.

## 8. Limits

Two model seeds; one detector with no training on our data; one fit of the locator in the rollouts (row 199 read three at T0); T1-kin reads recorded frames at the rollout's steps, there is no re-perception of what the rolled-out car would see; the rows the rule treats as a lead are those of the base file, so the presence input is unchanged by construction.

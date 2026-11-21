# Cost card (DRAFT): the closing read of the lead rule, 10k candidate

State: the fork's card for the PI and the other session, 2026-11-21. Rows 149 to 158 are landed (holds 108 to 110); rows 159 and 160 (cards ADD97 and ADD98) land with hold 111. The cost figures of section 6 are the other session's estimate. Nothing in this card is adopted by the fork; the decision is the PI's, with the other session.

## 0. The decision asked

* Whether to run a pilot of a new read of the lead from camera frames (option D, section 6). It needs camera frames on a pod again and a new encode; both are the PI's decisions (spend, data placement, GPU time).
* With no new encode the line is read out (sections 3 to 5, rows 139 to 160): the candidate keeps the form without a closing input, with a contact rate of 0.0954 at T1-kin/hb0.2 and the pre-roll of 30 steps (6 s), against 0.1877 with no rule.
* What the pilot would be asked to show is known from measurements: a gap that is steady on one lead over seconds (row 160), with the present reader's error cut by 2 (row 158); an absolute range accuracy of 1 m is not needed (rows 153, 157).
* The fork's recommendation, for the PI to weigh: the smallest pilot first, the 141 evaluation clips plus 500 train clips with a crop that follows the lead (ESTIMATED 0.35 to 1.46 h wall on one GPU, 13.4 GB of cache at the full grid), with PASS and FAIL as committed in section 6. Whether 500 clips are enough to fit a reader on the crop is not known (HYPOTHESIS); a result between PASS and FAIL comes back to the PI before any rollout.
* Host: the other session's pod (the measured rates are from it; its queue is in use) or the fork's pod, which is idle (two GPUs, 395 GB of disk free, MEASURED on 2026-11-21; encode rates there are not measured, and it would need the stage and encode scripts of the other session's tree).

## 1. What the closing input is worth

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m), pre-roll of 30 steps (6 s), 1,271 windows, 325 with a lead in 46 episodes, models UE_s0 and UE_s1. MEASURED, rows 128, 129, 145, 146.

| form of the lead rule | contact | falling behind (> 10 m) |
|---|---|---|
| no rule | 0.1877 | (rows 128, 129) |
| reader-fed, no closing input (the form in the candidate spec) | 0.0954 | 0.4954 |
| reader-fed, closing speed from the label (a bound, not deployable) | 0.0138 | 0.4738 |

The label closing speed removes most of the remaining contact and has no price in falling behind. The gap between the two forms is 0.0816 in contact (difference of the two table values).

## 2. What a read must reach

MEASURED at T0, row 147 (card ADD85): a closing read with a smooth error needs r 0.9 against the label closing speed; that buys 0.736 of the contact gap. In units of the error: a standard deviation of 1.24 m/s where the label closing speed has 2.85 m/s.

## 3. Where the reads on record sit

MEASURED at T0, rows 139 to 148: r 0.38 to 0.49. The full fit of the token head on the pooled 4 by 10 tokens reads r +0.491 [+0.320, +0.645] over 25 frames (row 148) and is not better than the ridge over the reader's outputs. No form with one of these reads is proposed for the candidate (rows 144 to 147).

## 4. Why: the gap read is the limit (MEASURED at T0, rows 151 and 152)

The reader's gap error against the label gap, row 152 (card ADD91, 5 of 5 controls passed), standard deviation with its interval:

| range of the label gap | gap error (sd) | mean | frames |
|---|---|---|---|
| all scored frames | 8.64 m [7.37, 9.91] | -2.38 m | 5,144 |
| below 20 m | 5.10 m [3.21, 7.88] | +1.11 m | 2,264 |
| 20 to 40 m | 7.90 m [6.51, 9.13] | -1.72 m | 1,854 |
| from 40 m | 9.91 m [7.74, 11.62] | -11.25 m | 1,026 |

* What a slope tolerates, row 151 (card ADD90, 4 of 4 control lines passed): on the label's own runs a slope of the exact label gap over up to 10 frames (2 s) reads the label closing speed at r +0.981 [+0.972, +0.987]. This is a reachability bound on the label's own source, close to holding by construction, and no capability of a reader. With planted gap error of the reader's time course it holds r 0.9 up to 1 m (+0.963 [+0.945, +0.974]); at 2 m the lower end is under 0.9 (+0.908 [+0.866, +0.934]); at 8 m it reads +0.482 [+0.381, +0.564], inside the range of section 3.
* The form that needs no label, row 152: a slope of the reader's own gap with the label's run rule reads the label closing speed at r +0.261 [+0.152, +0.362] over all scored frames and at +0.492 [+0.357, +0.619] below 20 m. Both are below the requirement.

So the requirement of section 2 turns into a target on the gap: about 1 m for the part of the error that changes within a run, to be read as what a read with the exact runs of the label needs and nothing more; row 153 (section 6) shows that the part of the error that stays with one lead is not part of this target. The reader is above 2 m at every range, also below 20 m where contact happens. Reading of the lead, not a name: the reads on record are about what this gap error allows, and a better closing head on the same gap is not the lever.

Limits of this section: T0 only; the range bins use the label gap; the run boundaries in the ladder of row 151 come from the clean label gap; the planted error is Gaussian with one draw; the label closing speed is itself a slope of the label gap; two seeds of one reader, which differ below 20 m (6.20 against 3.69 m). Row 150 (card ADD89) carries post-hoc values only and is not used here.

## 5. Options

| option | what it needs | what is known | whose decision |
|---|---|---|---|
| A. Keep the form without a closing input | nothing | contact 0.0954 at the pre-roll of 30 steps (6 s); this is the candidate as specified | stands |
| B. A constant closing input | a rule parameter | a constant of -1 m/s matches the estimates on record (row 145); it has a price in falling behind (row 145); it reads nothing from the scene | PI and the other session (a rule parameter is a design decision); not recommended by the fork |
| C. The finer token grid both caches hold (8 by 20) | no new encode; T0 probe heads only | measured, not the lever. Closing head, row 156 (card ADD87, 10 of 10 controls passed): r +0.495 [+0.362, +0.624] over 25 frames, against +0.491 [+0.320, +0.645] on the pooled grid (row 148); the difference of the two grids is not separated (mean absolute error -0.071 m/s [-0.179, +0.047]) and both upper ends are below 0.9. Gap read: row 97 (card ADD43) gave no name; printed there with no name, the gap error (RMSE, mean over eight probe seeds) is 5.42 m on the finer grid against 5.54 m on the pooled grid, far above the target of section 4 | T0 probes: the fork, done. A model arm on that grid: PI |
| D. A new gap read from camera frames: a crop of the lead region, or the token grid before pooling | camera frames back on a pod and a new encode (rows 135, 143) | not measured. Row 135: more clips of the present kind do not lift the gap read through pooled tokens | PI (spend, data placement) |
| E. Process the present reader's gap: the mean of the two reader seeds, a causal smoothing | no new encode; a changed input of the rule, measured as a study arm with the label marking the lead | measured, not the lever. Row 159 (card ADD97, 6 of 6 controls passed): contact 0.0369 with the seed mean and 0.0400 to 0.0769 with smoothing, against 0.0477 for the reader's gap as it is; no arm is within half the gap to the label, and three of the four fall behind more than the form in the spec. The errors of the two reader seeds correlate at about 0.76 (ESTIMATED), so more reader seeds do not reach the cut by 2 either | the fork, done |

## 6. Proposal for option D, if the PI wants it: a pilot with both outcomes committed

* Pilot: encode the 141 evaluation clips and a train subset with the chosen form, fit a gap reader, read at T0 against the label gap on the scored frames of card ADD81.
* PASS, read directly as row 152 reads the present reader: the slope of the pilot's own gap with the run rule over up to 10 frames reads the label closing speed with the lower end of r at or above 0.9. Then the rule form with that slope is read in T1-kin/hb0.2 against the two forms of section 1.
* FAIL: the upper end of that r is below 0.9. Then the line ends here and the candidate keeps option A; the contact of 0.0954 is reported as the candidate's value with its cause named (the gap read).
* Between: not decided; the PI gets the number before any rollout.
* What the pilot has to deliver, from row 153 (card ADD92, 5 of 5 controls passed, planted error, the read's own runs): a gap that is steady from frame to frame. A white frame-to-frame error is carried up to 0.5 m (r +0.936 [+0.910, +0.956]; at 1 m +0.690); an error with the slow time course of the present reader's up to 1 m (+0.946 [+0.926, +0.960]) or 2.5 percent of the gap; a scale error that stays with one lead up to the largest level planted, 20 percent or 5.5 m (+0.948 [+0.903, +0.972]). The pilot does not have to reach an absolute range accuracy of 1 m. It needs a lead identity over time beside the gap, since finding the changes of the lead from the noisy gap has a price (row 153).
* The same bar inside the rule, row 157 (card ADD95, T1-kin/hb0.2, pre-roll of 30 steps (6 s), 6 of 6 controls passed; error planted on the label gap, which is then the rule's only source of the lead; the card is the form of card ADD94 with one control clause changed after that card failed on it before any rollout, row 155): contact 0.0169 with a slow error of 1 m and 0.0062 with a white error of 0.5 m, against 0.0954 for the form in the candidate spec and 0.0000 for the label gap. Both cut contact with a separated interval (-0.0785 [-0.1339, -0.0293] and -0.0892 [-0.1518, -0.0352]) and lie within half the gap to the label. With a slow error of 2 m contact is 0.0569. A scale error per lead of 10 and of 20 percent gives 0.0015 and 0.0000, with no separated price in falling behind (at 20 percent the rule's own largest gap of 60 m truncates the planted error on 118 of 3,986 frames with a valid lead gap, all with a label gap of 48.80 m or more; below 40 m it is delivered in full). So on a planted gap a pass at T0 at these levels is also a pass in the rule, and the pilot does not have to reach the absolute scale.
* How far the present reader is from that, measured in the rule, row 158 (card ADD96 with note N1, same tier and pre-roll, 6 of 6 controls passed at the first read): the rule reads a mix of the label gap and the reader's gap as the rule reads it, on the frames the label marks, and the share of the reader's error is cut step by step. Contact 0.0477 with the reader's gap itself (the rule then falls behind more, +0.0677 [+0.0359, +0.1053]); 0.0092 with the error cut by 2, within half the gap to the label (+0.0092 [+0.0016, +0.0199]) and with no separated price in falling behind; 0.0031 cut by 4; 0.0000 cut by 8. The error at the cut by 2 still has a standard deviation of 4.43 m, 3.41 m about the clip mean and 1.47 m between neighbouring frames. So the target of the pilot, in the rule's own plane, is the present reader's gap error cut by 2, with its kind of error kept; the bars above are for additive error and are stricter than the rule needs for an error of this reader's kind. A mix with the label is no reader, and the label marks the frames with a lead in every arm.
* What was tried with no new encode, row 159 (card ADD97, same tier and pre-roll, 6 of 6 controls passed at the first read; the label still marks the frames with a lead): the mean of the two reader seeds gives contact 0.0369, a causal smoothing 0.0400 (a = 0.5), 0.0692 (a = 0.25) and 0.0769 (a = 0.25 on the seed mean), against 0.0477 for the reader's gap as it is. Every arm cuts contact against the form in the spec, none is within half the gap to the label, none is separated from the reader's gap as it is, and three of the four fall behind more than the form in the spec. The errors of the two reader seeds correlate at about 0.76 (ESTIMATED from the statistics of the joins: standard deviation 8.87 and 8.86 m for one seed, 8.32 m for the mean), so a mean over more reader seeds keeps about 7.7 m of that error. Processing the present read does not reach the cut by 2.
* What a new read has to fix, row 160 (card ADD98, same tier and pre-roll, 6 of 6 controls passed at the first read): the reader's error is taken apart with the label and the rule reads the label gap plus one part. Contact 0.0169 with the offset that stays with a clip (not shown above 0: +0.0169 [+0.0000, +0.0388]), 0.0662 with the part that moves within a clip, 0.0431 with its slow drift alone and 0.0185 with its change from frame to frame alone, against 0.0477 for the whole error. The slow drift alone has about the contact of the whole error; the fast part has little contact and makes the rule fall behind more (+0.0262 [+0.0018, +0.0503] against the label gap). Sizes in the present reader: drift 6.31 m, fast part 1.65 m, offset per clip 5.68 m (standard deviations, reader of the first model seed). So the first target of a new read is a gap that is steady on one lead over seconds, then the change from frame to frame; the offset per clip comes last. The fork's reading (HYPOTHESIS, not measured): this speaks for a crop that follows the lead and for a read over time on it. The parts are built with the label, so none is something a reader delivers, and contact does not add up over them.
* Baseline of the present reader in the same terms, row 154 (card ADD93, 5 of 5 controls passed, 3,834 frames per reader seed in 62 clips): the part of its gap error about the clip mean is 6.87 m [6.01, 7.74], 8.3 times the largest slow error carried; the error changes by 2.96 m [2.36, 3.56] from frame to frame, 4.2 times the largest white error carried; its scale error per clip is 0.335 [0.220, 0.452], above the 20 percent that row 153 shows to be carried (the largest level planted, not a measured limit). So the pilot is asked to show a cut of the frame-to-frame part by a factor of about 4 to 8 and a scale error per lead at or below 20 percent; these are diagnostics beside the pass criterion above.
* Cost, the other session's estimate of 2026-11-21 (`F:/TanitAD-scratch/v8proof_encode_cost/ENCODE_COST_ESTIMATE.md`; read only, nothing was run for it). MEASURED on the other session's pod (RTX 4000 Ada, DINOv3-L at 256x640, about 101 frames per clip): encode 2.61 s per clip, frame build 1.98 s per clip at 13 workers, 1,279 clips per hour end to end over the run of 10,000 clips, 20.88 MB of kept cache per clip. ESTIMATED from the token count of the encoder: a crop at native pixels (256x256) about 1.0 s per clip, the frame build then being the slower step; a crop upsampled to 416x1024 about 8.2 s per clip. Wall time on one GPU and cache, for the 141 evaluation clips plus a train subset (ESTIMATED, clips times rate):

| clips | native crop | upsampled crop | cache at 20.88 MB per clip |
|---|---|---|---|
| 141 (evaluation only) | 0.08 h | 0.32 h | 2.9 GB |
| 641 | 0.35 h | 1.46 h | 13.4 GB |
| 1,141 | 0.63 h | 2.60 h | 23.8 GB |
| 2,517 | 1.38 h | 5.73 h | 52.6 GB |

* Where the camera frames are (the other session's listings): no pod holds camera frames. The dev box holds 139 of the 141 evaluation clips at 416x1024 (11.5 GB; two clips are missing). Train frames have to be downloaded again from the NVIDIA PhysicalAI-AV source through the pod's stage step (12.76 MB per clip MEASURED, about 32 GB for 2,517 clips ESTIMATED).
* Open choices for the PI, each of which changes the cost: the crop (native pixels or upsampled; one fixed crop per frame, or one that follows the lead, which needs the agent tracks joined per window); the train subset (the 2,376 clips of the parity corpus, or a part of them); the pooled grid of the output (about 4.4 or 20.9 MB per clip); the two missing evaluation clips; the host. GPU time on the other session's pod needs the PI's go and a slot in its queue after the runs it holds now. The measured rate has no replicate.

## 7. What this card does not claim

No statement on driving in a re-perception tier (T2 is not provisioned). LATERAL, TACTICAL and STRATEGIC are not touched by the closing read; the numbers of section 1 are LONGITUDINAL (distance-keeping) at T1-kin/hb0.2. Two model seeds of one line. Row 157 shows the T0 bar to hold inside the rule for error planted on the label gap; for a real read of the lead the rule test of section 6 still decides, and T1-kin has no re-perception: the lead moves as logged whatever the ego does.

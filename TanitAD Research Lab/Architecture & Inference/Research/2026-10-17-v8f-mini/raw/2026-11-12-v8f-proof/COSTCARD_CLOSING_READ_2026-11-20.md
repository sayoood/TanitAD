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

## 8. Addendum: where the crop comes from (2026-11-21, added before the card was sent)

* Binding rule of the programme: labels may use ego and other agents, inference is vision only. So in sections 0 and 6 a crop that follows the lead means a crop whose position at inference comes from the camera image alone: a fixed crop on the road ahead, or one steered by the coarse read of the present reader (its presence and gap from the pooled tokens).
* A crop placed from the agent tracks of the label is admissible as a bound arm and for training targets only. As a bound it answers the cheaper question first (is the gap readable from a crop on the lead at all), and the arm with the position from the image then says what is deliverable.
* The pilot therefore has two arms on the same clips: crop from the label tracks (bound) and crop from the image (deliverable); PASS and FAIL of section 6 are read on the deliverable arm. This doubles the encode time of the table in section 6 and not the frame build.

## 9. Result of the pilot (row 161, card ADD99, 2026-11-21)

* The PI gave the go on 2026-11-21 ("Follow your recommendation"). The pilot ran on the fork pod on the 141 evaluation clips and 500 train clips, two arms as in section 8, about 1 GPU-h, 33 GB of tokens. Tier T0 (a probe of a reader on logged frames). 8 of 8 controls passed at the first read.
* By the criterion of section 6 the pilot FAILS: on the deliverable arm (fixed windows on the road ahead) the slope of the read's own gap reads the label closing speed at r +0.564 [+0.419, +0.675], against +0.261 [+0.152, +0.362] for the present reader; the upper end is below 0.9. Its gap error is 6.91 m [4.98, 8.98] against 8.64 m [7.37, 9.91], a ratio of 0.800 [0.622, 0.952]. By section 6 the candidate keeps option A.
* The bound arm (windows centred on the lead from the label): gap error 2.91 m [2.14, 3.92], 1.14 m [0.92, 1.33] below 20 m, a ratio of 0.337 [0.251, 0.440] to the present reader's, past the cut by 2 of row 158; r +0.694 [+0.568, +0.776], also below 0.9. So the gap is readable from a crop on the lead with a head fit on 500 clips; what is missing is finding the lead in the image. A part of the bound's cut can come from the label-exact centring of its windows.
* Next, frozen before any tool and inside the pilot's go (same clips, same staged video, the fork pod): card ADD100, the windows steered by a localiser that reads the fixed windows (T0), and card ADD101, the rule reading the pilot's gaps on the frames the label marks (T1-kin/hb0.2, pre-roll of 30 steps (6 s)). A larger train subset stays the PI's call.

## 10. The rule reading the pilot's gap (row 162, card ADD101, 2026-11-21)

* Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1, eight rollouts on the fork pod (16 minutes), inside the pilot's go. 6 of 6 controls passed at the first read.
* With the label marking the frames with a lead, the rule reading the pilot's gap has a contact rate of 0.0169 with the gap from the fixed windows and 0.0077 with the gap from the windows on the lead, against 0.0477 with the present reader's gap, 0.0954 for the reader-fed form of the candidate spec and 0.0000 with the label gap. Both lie within half the gap (0.0477) to the label gap; the fixed-window gap has less contact than the present reader's gap, -0.0308 [-0.0751, -0.0032].
* Price: the fixed-window arm falls behind by more than 10 m more often than the label gap, +0.0508 [+0.0181, +0.0875], and reads LONGITUDINAL WORSE and TACTICAL WORSE against the reader-fed form; the arms with the windows on the lead are not separated in the three families. STRATEGIC is not separated for all eight pairs.
* What it means for this card: the bar of section 6 (r 0.9 of the closing read at T0) was not reached, and the rule still gains from the pilot's gap. The gap error at T0 was a poor guide to the rule here (ratio 0.800 at T0, about a third of the contact in the rule). No arm is a candidate form, because the label marks the frames with a lead.
* Next, frozen before any tool: card ADD102, the same rule with the frames with a lead from the present reader's presence output and the gap from the fixed-window head (no label at inference), six rollouts on the fork pod. Adoption of a changed input of the rule and a larger train subset stay the PI's call.

## 11. Steering the crop from the image (row 163, card ADD100, 2026-11-21)

* Tier T0, on the pilot's clips and staged video, about 1 GPU-h on the fork pod. 8 of 8 controls passed at the first read.
* The windows steered by a localiser (image only at inference) read the gap no better than the fixed windows: 6.51 m [5.22, 7.80] against 6.91 m (ratio 0.942 [0.789, 1.174], not separated), 8.64 m for the present reader (ratio 0.753 [0.628, 0.869]) and 2.91 m for the windows on the lead from the label. The localiser puts the centre of the lead's box inside the narrow window on 0.914 of the frames. By the card's committed reading the cut of the bound comes from its label-exact centring; the image-only candidates for the gap are the fixed windows.

## 12. The rule with no label at inference (row 164, card ADD102, 2026-11-21)

* Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1, six rollouts on the fork pod (37 minutes), no new encode and no new fit. 6 of 6 controls passed at the first read.
* The form: the frames with a lead from the present reader's presence output (as in the candidate spec), the gap from the fixed-window head of the pilot, the closing input the slope of that gap. Contact 0.0169 against 0.0954 for the reader-fed form of the candidate spec (-0.0785 [-0.1351, -0.0288]) and 0.0000 with the label gap (+0.0169 [+0.0013, +0.0381], within half the gap of 0.0477). Unsafe rate 0.0786 against 0.1339. LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC against the reader-fed form: NOT-SEPARATED.
* Price: falling behind by more than 10 m against the label gap form +0.0415 [+0.0095, +0.0783]; against the reader-fed form +0.0369 [-0.0164, +0.0895], not separated. Without the closing input the contact is 0.0769.
* What this means for the decision of this card: the bar of section 6 (r 0.9 at T0) was not reached by any crop read (rows 161 and 163), and the pilot's fixed-window gap with its own slope still brings the rule within half the gap with no label at inference. The new read of the lead is therefore worth more in the rule than its T0 numbers said. What it rests on: one fit of the head per model seed, a head fit on 500 clips, a tier with no re-perception (the crop does not move with the rolled-out ego).
* Next, frozen before any tool and inside the pilot's go (no new encode, no new fit): card ADD103, the other fit seed of the head, the slope on the present reader's own gap (a form that would need no new encode at all) and the head on one frame. Adoption of a changed input of the rule, a larger train subset and the encode of the full corpus stay the PI's call.

## 13. The replicate, and what carries the cut (row 165, card ADD103, 2026-11-21)

* Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1, six rollouts on the fork pod (46 minutes), no new encode and no new fit. 6 of 6 controls passed at the first read.
* The replicate is not decided by the card's rule. With the other fit seed of the head the contact is 0.0200 against 0.0169; the cut against the reader-fed form repeats (-0.0754 [-0.1328, -0.0254]) and the family reads against that form are NOT-SEPARATED; the interval against the label gap, +0.0200 [+0.0015, +0.0498], ends above half the gap of 0.0477. 0.0200 is 13 of 650 windows.
* What carries the cut: the slope closing input. With the crop gap it cuts contact by -0.0600 [-0.1086, -0.0185] and costs falling behind by more than 10 m +0.0477 [+0.0112, +0.0899].
* The form with no new encode at all, the slope on the present reader's own gap: contact 0.0431 against 0.0954 (-0.0523 [-0.1087, -0.0096]), but LONGITUDINAL WORSE, TACTICAL WORSE and STRATEGIC MIXED against the reader-fed form, falling behind +0.0677 [+0.0358, +0.1056], and 0.0185 and 0.0677 on the two model seeds. The crop gap is not separated from the reader's gap in contact (-0.0262 [-0.0724, +0.0017]); its family reads stay NOT-SEPARATED from the reader-fed form.
* What this means for the decision of this card: the larger train subset for the head was approved by the PI for the case that the replicate holds; the card's name is not that case, so that go is not used and the question goes back to the PI with these numbers. Inside the pilot's go (no new encode): card ADD104, frozen before any tool, three new fits of the head on the same 500 clips with the form run for each.

## 14. Five fits, the mean gap, and the larger fit (rows 166 to 168, cards ADD104 and ADD105, 2026-11-21)

* Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1. Cost on the fork pod: three fits of the head (13 minutes, no new encode) and eight rollouts (about 56 minutes with three jobs side by side). All controls of both cards passed at the first read (7 of 7 and 6 of 6).
* Card ADD104: `VISSEED10K-HOLDS`. Three new fits each cut the contact of the reader-fed form and each lie within half the gap of 0.0477 (upper ends +0.0351, +0.0460, +0.0287). Over five fits the contact lies between 0.0123 and 0.0215 against 0.0954. The family reads are NOT-SEPARATED; the strategic read has no arc reading WORSE.
* Card ADD105: `VISENS10K-WITHIN-HALF-THE-GAP`. The mean gap of the five fits: contact 0.0138, not separated from a single fit (-0.0031 [-0.0107, +0.0040]).
* Row 166, a count on the rollouts of row 164: of 62 contacts of the reader-fed form, 52 do not occur under the form with no label and 1 is new.
* The price is unchanged: against the label gap form the single fits fall behind more (+0.0277 to +0.0492 for the three new fits, lower ends above 0).
* What this means for the decision of this card: the question that sections 12 and 13 left open, whether the result rests on one fit of the head, is answered. The PI's go of 2026-11-21 for the head on all 1,679 train clips (given for the case that the three new fits hold) applies. Card ADD106 is frozen for it: encode of the eligible train clips outside the pilot, three fits, ten rollouts, ESTIMATED 3 to 4 pod-hours and about 50 GB, with a stop rule at 5 pod-hours or 60 GB. The encode of the full corpus and the adoption of a changed input of the rule stay the PI's call.

## 15. Where the price comes from: two counts (row 169, 2026-11-22)

* Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1. Cost: no rollout and no GPU; two counts on banked rollouts, seconds on the CPU of the fork pod, run while the encode and the fits of card ADD106 used the GPU.
* The price of the form with no label is more falling behind than the label gap form (sections 12 to 14). The counts find it in 151 arm-windows over five arms (63 distinct windows of 650 with a lead).
* Not the cause: the reader marking a lead where the label has none (0 of 193 cut looks in those windows).
* Ranked candidate: the fed gap lying more than 5 m below the label gap (37 of 193 cut looks, against 109 of 2,030 where neither falls behind and 0 of 106 where only the label gap form falls behind). The extra cut steps are in the pre-roll (+1.09 on average against the label gap form); inside the window there are fewer (-2.06).
* The first of the two counts ranked no lever, by a threshold the lead had set for a log with a row per step; it is reported as written in row 169.
* What this means for the decision of this card: no new item. The candidate is the gap error of the head, which card ADD106 (the head fit on 1,679 clips, under the PI's go of 2026-11-21) measures in its scale name for falling behind. The counts prove no cause; they carry no interval.

## 16. Where the gap error sits (row 170, 2026-11-22)

* Tier T0 (recorded frames, no rollout; a diagnostic of the head, not driving performance). Cost: seconds on the CPU of the fork pod, on banked gap files, while the rollouts of card ADD106 used the GPU.
* The head reads far leads too short: 6.35 m below the label on average for labelled gaps of 40 to 60 m, more than 5 m short in 0.52 of the frames; +0.04 m and 0.00 below 10 m. The mean of five fits reads the same (-6.23 m, 0.47).
* Ruled out with no rollout: a filter over frames (the error rises from 4.112 m to 4.246, 4.417, 4.766 m) and the mean of fits (same reading). A monotone map of the output has 1 to 2 m to gain (label minus fed +0.50 to +1.85 m by bin of the fed gap).
* What this means for the decision of this card: the price of the form with no label is tied to how well the camera reads a far lead. More fit data is measured by card ADD106 under the PI's go of 2026-11-21. More resolution on a far lead would be a new encode and is the PI's call; nothing of it is started.

## 17. Where the head looks (row 171, card ADD107 frozen, 2026-11-22)

* Tier T0 (recorded frames, no rollout; a diagnostic of the head, not driving performance). Cost: seconds on the CPU of the fork pod, on banked data of the pilot.
* With the two windows placed on the lead by the label (the bound arm of the pilot) the same head has about half the gap error: 2.137 and 1.910 m against 3.960 and 3.913 m. At 40 to 60 m it reads 2.99 and 2.18 m short against 5.90 and 6.50 m.
* In 486 of 3,841 frames the lead is more than 8 degrees off the camera axis, outside the fixed narrow window. There the fixed windows have an error of 11.34 m (bound 3.73 m); those frames carry 0.36 of the summed absolute error.
* The bound is not admissible at inference. Card ADD107 (frozen) asks the admissible question at T0 with no new encode: can a head on the tokens of the fixed windows find the direction of the lead. ESTIMATED four fits of about 13 minutes on the fork pod, stop rule 1.5 pod-hours.
* A decision that will come to the PI if that card reads that the head locates the lead: the encode of two windows at the predicted direction on the 641 pilot clips (ESTIMATED by the pilot's own figures 0.35 to 1.46 GPU-hours and about 17 GB), then the gap head on those tokens and the lead rule at T1-kin. Nothing of it is started.

## 18. More fit clips for the camera head (row 172, card ADD106 read, 2026-11-22)

* Spent under the PI's go for the head on all 1,679 eligible train clips: 3 h 17 min on the fork pod from the start of the encode to the names (stop rule 5 pod-hours) and 50 GB in the card's root (stop rule 60 GB). 11 of 11 controls passed at the first read.
* Tier T0 (recorded frames, no rollout; a property of the head, not driving performance): gap error 3.371, 3.295 and 3.344 m on 1,629 fit clips against 4.112, 4.079 and 4.056 m on 450, ratio 0.8173 [0.7258, 0.9101]. The gain sits between 10 and 40 m; leads at 40 to 60 m stay read short (5.40 m against 6.35 m).
* Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s): the form with no label at inference holds with each of the three large fits, contact 0.0062, 0.0077 and 0.0046 against 0.0954 for the reader-fed form. Against the 500-clip fits of the same fit seeds (0.0138, 0.0185, 0.0123) the difference is not separated, and falling behind against the label gap form stays at +0.0308, +0.0185 and +0.0215.
* What the spend bought: a separated gain of the gap reading and three more fits under which the form holds. What it did not buy: a separated gain at T1-kin over the smaller fits, or a lower price in falling behind. A larger encode of the same kind is therefore not proposed.
* No new decision comes from this row. The decision named in section 17 (windows at the predicted direction of the lead) depends on the read of card ADD107.

## 19. Decision for the PI: windows at the head's direction (row 173, card ADD107 read, 2026-11-22)

* What is known. Row 171 (tier T0): with the two windows placed on the lead by the label, a bound that is not admissible at inference, the gap error is about half (2.137 and 1.910 m against 3.960 and 3.913 m). Row 172: more clips of the same kind do not lower the price of the form in falling behind. Row 173 (tier T0): a head finds the direction of the lead from the camera alone, with tokens that exist (the lead in the central half of a narrow window in 0.875, 0.877 and 0.876 of the frames against 0.725; inside the window in 0.584, 0.539 and 0.582 of the 486 frames with the lead off the axis).
* What is asked. A go for one encode on the fork pod: two windows at the head's predicted direction on the 641 pilot clips (500 train, 141 evaluation). ESTIMATED from the pilot's own figures: 0.35 to 1.46 GPU-hours and about 17 GB. With it, on those tokens: fits of the gap head (about 13 minutes each) and a T0 read against the fixed windows and against the bound. Only if that T0 read shows a cut of the gap error: ten rollouts at T1-kin/hb0.2 as in card ADD106 (about 1.5 pod-hours), to read the price in falling behind.
* What it can and cannot give. It can give at most what the bound gives at T0; the admissible form is expected to give less, since the head's direction has its own error. It does not change the convention, the readout or the candidate spec; adoption of any input stays a separate decision.
* Order of work if the go is given: a card frozen with its hash file before any tool of it; directions for the train clips from fits that did not see them; stop rule set in the card from the figures above.
* If no go: the candidates left with no new encode are processed forms of the present gap, which rows 169 and 170 rank low.

## 20. Read after the go of section 19 (rows 174 and 175, cards ADD108 and ADD109, 2026-11-22): no decision asked

* The go: the PI approved the encode on 2026-11-22 ("Yes do it"). Cost: 0.37 of the 2 GPU-hours and 17.23 of the 20 GB of the stop rule; the four fits and the read stayed inside their stop rule.
* The read (tier T0, not driving performance). Row 174: windows at the head's direction give a gap error of 3.895, 3.699 and 3.689 m against 3.960, 3.913 and 4.112 m; ratio of the means 0.9413 [0.8038, 1.0939]; not separated. Row 175: a mix by the head's own direction gives 0.9772 [0.8575, 1.0949]; not separated.
* What is not proposed: an encode of more clips at the head's direction. The condition of section 19 for the ten rollouts (a cut of the gap error at T0) is not met, so they are not run.
* What runs with no decision: card ADD110, eight rollouts at T1-kin/hb0.2 on existing models with gap files composed from the label gap and the camera gap, to read where in the gap range the price in falling behind sits (ESTIMATED 1.2 to 1.6 pod-hours on the fork pod, stop rule 3 pod-hours).

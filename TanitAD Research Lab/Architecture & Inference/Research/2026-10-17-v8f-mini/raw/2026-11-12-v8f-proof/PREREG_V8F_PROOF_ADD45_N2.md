# Note N2 to card PREREG_V8F_PROOF_ADD45 (DSCALE-LEAD): where the probe's labels come from, the port, and the controls of the inputs

Note of the v8F proof study, written by the coordinator fork. Frozen before any run of the card on the union (time of the freeze: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). The tools of the card exist and were tested on stand-in data only (section 5); no number of the card has been produced. The PI's go for the card is recorded in `TIMES_NOTE.md` (2026-11-19T18:52:35Z). Tier: T0; never driving performance.

## 1. What was learned after note N1 was frozen (MEASURED by reading the study tree's source; file `stack/tanitad/data/v8f_mini_data.py` of the study tree, sha256 3b5851379d1c8fa6, lines 312 to 354)

The probe never read its labels from a stored closing-speed field. Its labels come from the join file: presence = `lead_valid` equal to 1; gap = `lead_gap` where presence holds, the value is finite and lies in 0 to 60 m; closing speed = the least-squares slope of that gap over a trailing run of 3 to 5 labelled frames 0.2 s apart (a step above 3.0 m breaks the run). Note N1 took over from the other session that the lead file stores no closing speed, and concluded that a closing speed cannot be derived. For the probe's own label that conclusion does not hold: the slope is defined on any clip whose join file carries the flag and the gap.

## 2. Decision on the arms (unchanged from note N1, now with its reasons)

The arms stay as note N1 fixed them: presence and gap in the loss, the closing part with weight 0 in every arm, M1 not read, no closing name, the control arms with permuted gap labels. Reasons: the card asks about the gap, which row 94 measured as the first limit of the distance line; the levers of the integrated candidate read presence and gap only; and the slope label is taken without an identity of the lead, so its noise can differ between the corpus labels and the 10k labels, which would be a second difference between the two arm types beside the clip set. A closing-speed read at scale is a possible later card; it is not run here and nothing of this card speaks about it.

## 3. The port (facts of the kit, folder `dscale_lead/`)

* Tool `tok_probe_u.py`: generated from the tool of card ADD43 (`finegrid_r/tok_probe_gr.py`) by 12 asserted text edits; without its new flags the training path is the source tool's (tested bit for bit). New flags: `--clip-list` (the corpus arm), `--eval-guard`, `--w-clos 0.0`, `--shuffle-gap-labels`.
* The probe runs in the study tree (the tree of card ADD43), shipped to the other pod as a folder and pinned by sha256 (three tree files and six kit files); the union files of the other session go into the trainer argv. The other session's own tree is not used and not changed.
* Arm folders: `runC` (corpus list, seeds 0 to 3), `runU` (no list, seeds 0 to 3), `runU2` (seed 0, 38,328 steps), `runCSH` and `runUSH` (permuted gap labels, seed 0). Every other arm runs exactly 19,164 steps, given as a parameter, so the step count and the learning-rate schedule are equal between the arm types. No exclusion range in the argv: the union already lacks the 60 holdout clips.
* Stages of the kit: precheck (CPU), cpucheck (CPU: two training steps on the union and two with the corpus list, before any GPU slot), smoke (200 steps; it stops the run if the projection of the five folders exceeds 6 pod-h, and the run then goes back to the PI), run, extract, checklog.

## 4. Controls of the inputs (all gate: a failed or absent one means no name and withheld differences)

* d1 as the card: no evaluation clip id among the training clips of the union (count exactly 0), a planted shared id must fail; in addition every training call refuses to start if an evaluation id is among the clips it built.
* d2 as note N1, first half: for 200 corpus clips drawn at random (seed 45) the lead flag and the gap of the union lead file equal those of the corpus lead file at the same frame index (tolerance 1e-4, no row in one file only).
* d2j: on those clips the join file equals the lead file at raw frame k = 2 j (the join is the probe's label source).
* d2n: the same on 200 clips that are not in the corpus list (seed 46).
* d2g: in the join file both groups (corpus, new) carry presence frames and gap-label frames (counts above 0; the counts and shares are printed).
* d2c, in place of the same-clock result note N1 cited (the other session reported that no such check of the lead file exists): the lead file stores the ego speed per raw frame k, the token cache stores the ego speed per token frame j. Over 200 clips per group (seed 47) the mean absolute difference between the two is read at the pairings k = 2 (j + s), s = -5, -2, -1, 0, 1, 2, 5. Pass, in both groups: the smallest value sits at s = 0 and is at most 0.5 times the smaller of the values at s = -1 and s = +1. Planted: the lead rows moved by one token frame must fail with the smallest value at s = -1.
* d2e: the evaluation cache used for the extract holds the study's 141 evaluation clips in the study's order (the reader pairs states and the evaluation pack of card ADD43 by position). Planted: two swapped ids must fail.
* d3, d4, d5, d6, d7 as note N1.
* Reported, gating nothing (d3t): for how many corpus clips and evaluation clips the content hash of the cached tokens on the other pod equals that of the study's cache. Equal hashes mean the corpus arm reads the study's tokens; if they differ, the row says so as a limit of the link to row 97. The named difference (union minus corpus) is taken inside the other pod's cache either way.

## 5. Tool checks before the run (MEASURED, on the pod of this fork, CPU only; none of it is a result of the card)

* Stand-in data: the corpus cache as the union, the first half of its manifest as the corpus list, the dev box's corpus lead file. Precheck: all checks pass and the planted id fails d1. Clock check: s = 0 reads 0.0046 and 0.0043 m/s against 0.104 and 0.107 m/s at the nearer neighbour shift (ratios 0.045, 0.040; bar 0.5); the planted shifts of +1 and -2 token frames fail with the smallest value at -1 and +2. Evaluation-set check: passes; two swapped ids fail.
* Training path: 20 steps of the port on corpus tokens; the closing part of the loss is exactly 0.0; the refusals for a hit of the evaluation guard, a listed id that was not built, a missing step count and another closing weight all fire.
* Tests: the three test files of the kit and the mutation list, run under a stand-in test runner (the pod has no pytest and nothing was installed into its venv); the counts are in `TIMES_NOTE.md` with this note. They are to be run again under pytest on the dev box when its hold lifts; a difference stops the run.
* Not exercised before the run on the other pod: the union files themselves, the union lead file's format on the new clips, the cache layout of the other pod, and the reader end to end. The precheck and cpucheck stages are there for the first three; they cost no GPU time.

## 6. What does not change

The evaluation set, the pairing, the bootstrap, the bars of the three names on the gap, the budget rule (3 to 5 pod-h estimated, smoke first, above 6 back to the PI) and what follows from each outcome (card section 6).

# Note N3 to card PREREG_V8F_PROOF_ADD45 (DSCALE-LEAD): the evaluation view, and the two encodes

Note of the v8F proof study, written by the coordinator fork. Frozen before any GPU stage of the card and before the second run of its precheck (time of the freeze: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). No number of the card has been produced. Tier: T0 (a probe on cached tokens of logged frames); never driving performance. The arms, the reads, the names and the bars of the card with notes N1 and N2 stay as frozen.

## 1. What the first precheck on the other pod found (MEASURED there by the tree owner session; its files came back as `dscale_lead/back/add45_precheck_out1.tgz`, sha256 40fff474f0eed012, and were read here)

* Passed: the counts (14,512 union clips, 4,512 corpus clips), d1 with its planted id, d2, d2j, d2n, d2g, and the clock check d2c with its planted shift (lowest difference at shift 0 in both groups, 200 clips each; ratio to the nearer neighbour 0.039 on the corpus clips against the bar 0.5).
* Stopped at d2e at 2026-11-19T19:51:14Z, before any other stage: that pod's evaluation manifest holds the study's 141 clips (same id set) in another order (sha256 of the id list cd3d8efe there, 4173cf10 in the study).
* Report d3t: 0 of 141 evaluation clips and 0 of 4,512 corpus clips carry the content md5 of the study's caches; all 4,512 corpus ids are present on both sides. That pod's caches are its own encode of the same clips (same pipeline tag by its own statement, INHERITED).
* INHERITED from the tree owner session, not re-read here: its tracks file (14,653 keys) covers all 14,512 union clip ids.

## 2. The order of the evaluation set (MEASURED by reading the tool: `tok_probe_u.py`, sha256 61073f5f7753c8fe, lines 578 to 580)

* The tool extracts the evaluation states in the order of `sha12` of the clip id (`sorted(..., key=D.sha12)[lo:hi]`), not in manifest order. With the same id set the extracted states come in the same order on both pods, and the reader pairs them with the evaluation pack of card ADD43 by position as before. The sentence of note N2 and of the check's header, "another order would misalign every window", says more than the tool does: the manifest order does not enter the extraction.
* Control d2e stays as note N2 froze it: the ids and their order, a gate, with a planted swap that must fail. It is not relaxed after its first reading. The kit instead puts the evaluation manifest into the study's order (section 3), so the control is met as written and every other reader of that manifest sees the study's order too.

## 3. The evaluation view (a change of the kit, made before any GPU stage)

* `dscale_evalview.py` (sha256 5d50a2d07411cee2): writes `$OUT/eval_view/eval/manifest.jsonl` with the lines of the pod's own evaluation manifest, byte for byte, in the order of `STUDY_EVAL_MD5.tsv`, and links every other entry of the pod's evaluation folder (the clips) into the view. Nothing is copied and no line is edited. It refuses when the id sets differ, an id occurs twice, a count is not 141, the view is the source folder, an existing view manifest differs from the one it would write, or an existing view entry is not its link.
* The kit script (sha256 90f4cdc236840c56) builds or re-checks the view at every stage and reads the evaluation set from it (precheck, evaluation guard, extraction). d2e and its planted swap are checked on the view. The pins file (sha256 b43e3673a75c35e7) holds 10 pins: the three tree files and seven kit files. No other kit file changed: the tool, the three checks and the reader carry the hashes of note N2.
* Checked on this fork's pod, CPU only, in a folder of its own, under the stand-in test runner (file `dscale_lead/N3TRIAL_pod.txt`, sha256 0fae03c4f538dea7): 11 tests pass; 4 planted faults in the view script each turn a test red (the view keeps the source order; the id sets are not compared; a changed existing view is accepted; a real folder is accepted where the link must be); on an evaluation cache whose manifest is this pod's manifest reversed, the check of the kit as landed fails (exit 2, as on the other pod), the kit with the view passes the stages precheck and cpucheck, the view manifest equals the study's manifest byte for byte (140 of 141 positions differed in the source), the second stage finds the view equal and writes nothing, and the extraction under `DRY=1` reads from the view.
* Not exercised: the other pod's folder layout, and the GPU stages reading clip files through the view's link (the training cache view of the kit reads through links the same way). A fault there stops the kit.

## 4. The two encodes

* The names of the card rest on union minus corpus, paired on the same windows, both arms on the other pod's encode. Report d3t does not gate and does not change them (note N2, section on d3t).
* The values of card ADD43 (row 97) and of card ADD60 come from this fork's encode. They are printed beside the values of this card and are not differenced with them for any name.
* New report d3x (gates nothing), tool `dscale_xenc.py` (sha256 61fe14ff8f5943d6): the cosine per token between the two encodes on a fixed sample (the first 20 evaluation clips of the study's list, token frames 0, 10, ..., 90, cache key pool4x10; 8,000 tokens), beside the cosine between tokens 2.0 s apart inside one encode as the scale.
  * The sample of this fork's encode: `dscale_lead/xenc/xenc_sample_v8proof.npz`, sha256 d3d3997086c915c9, 8.6 MB.
  * Self-check of the tool on this pod (MEASURED, `dscale_lead/xenc/XENC_selfcheck_v8proof.json`): against its own cache it reads cosine 1.000000, share of equal elements 1.0, 20 of 20 clips with the same content md5. Scale: tokens 2.0 s apart inside this encode have cosine mean 0.9085, 1st percentile 0.4747 (7,200 token pairs).
  * How its number is read (fixed before it exists): mean cosine between the encodes at least 0.99 = the same tokens up to arithmetic and storage rounding; below 0.9085 (the 2.0 s scale) = two different inputs, stated as such in every row that puts values of the two encodes side by side, and the cause goes to the tree owner session before that cache serves as the base of an integrated candidate; in between = reported as written with the 1st percentile.
  * The same number bounds a limit of the rollout rows of this fork: the 10k models were trained on the other pod's encode and are rolled out here on this fork's encode of the evaluation clips.

## 5. What follows (committed before the data)

* The precheck is run again with the kit of this note and a new output folder; the folder of the first run stays as the record of the stop.
* The precheck passes: cpucheck follows; the GPU stages (smoke, run, extract) start only in the slot the PI set ("After Ruled stage 1", PI to the tree owner session, as that session recorded it).
* The view is refused, or d2e fails on the view: nothing runs and the record comes back.
* Card ADD60 on this fork's pod goes on with the kit as landed with commit 06f205e6: its evaluation manifest is the study's, for which the view writes the same bytes. A fault of the kit or of the reader found there goes into the kit before the GPU stages of this card, as card ADD60 says.

## 6. Limits

The kit's tests ran under a stand-in test runner, not pytest. The view was tried on this fork's pod only. Report d3x reads 20 clips, 10 frames each, of one cache key; the 8x20 key is not compared. The tracks coverage is the other session's statement.

# Card PREREG_V8F_PROOF_ADD64 (FRACSCALE-LEAD): the lead probe on a quarter and on an eighth of the corpus against the whole corpus, with the kit of card ADD60 unchanged

Card of the v8F proof study, written by the coordinator fork. Frozen before any of its runs and before card ADD60 has ended or been read (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Probe only, no model is trained, no spend beyond the pod of this fork. Tier: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance.

## 1. Why

* Card ADD60 (running, not read) reads the lead probe on half of the corpus against the whole corpus; card ADD45 (other session's pod, waiting for its slot) reads the corpus against the union with the 10k clips. Two differences do not say where the read of the lead gap starts to flatten, and that is the question behind data efficiency and behind readiness for scale: how many clips this read needs, and whether it still rises at the largest corpus we hold.
* Two more points with the same kit cost only pod time of this fork, which is free once the rollout queue has ended: a quarter and an eighth of the corpus. With cards ADD60 and ADD45 they make a five-point curve over 571, 1,143, 2,286, 4,572 and 14,512 clips.
* Each run of the kit trains the WHOLE arm again with the same seeds. Three runs of it (cards ADD60 and this one) are a same-seed replicate: the run-to-run floor of this probe, which no earlier card of this line measured.

## 2. Arms

The kit of card ADD45 exactly as card ADD60 runs it (`/workspace/v8proof/dscale_lead/` on the pod of this fork, files as landed with commit 06f205e6; kit script, tool, checks and reader unchanged; inputs, tree, lead file, join, tracks, prior, thread count and step counts as card ADD60), with one input changed per run:

* QUARTER: the "corpus list" of the kit = the first 1,143 clip ids of the training manifest in manifest order (`quarter.txt`, sha256 0915ce4493c38e1efbe9fbb4e2620841735a6cc6e76880111cd3ad32b9952971); reader with the corpus count 1,143.
* EIGHTH: the first 571 clip ids (`eighth.txt`, sha256 b1270c6698eeb427961e6358d8f77c7d500a4506a899f44c18126094eff6671c); reader with the corpus count 571.

The lists are nested: eighth in quarter in half (card ADD60, `half.txt` sha256 1115929b15ea80e9...) in the whole corpus. Arms by the kit's folders in each run: `runC` = the fraction (seeds 0 to 3), `runU` = WHOLE (4,572 clips, seeds 0 to 3), `runU2` = WHOLE with twice the steps (seed 0), `runCSH`, `runUSH` = permuted gap labels (seed 0). 19,164 steps (38,328 for `runU2`).

Launcher `dscale_lead/pod_fracscale_v8proof.sh` (the launcher of card ADD60 with the fraction as a parameter; output folders outside the kit folder, `/workspace/v8proof/dscale_lead_frac/<run>/`). Order: QUARTER starts when the folder of card ADD60 has its end marker, EIGHTH when QUARTER has ended, so one probe of this fork trains at a time beside the rollout queue and each smoke measures its cost under the load card ADD60's smoke had. If card ADD60 or QUARTER ends with a stop marker, the run after it is not started.

Budget: ESTIMATED 3 to 4 GPU-h per run (card ADD60's smoke projected 3.70 h as an upper bound). The kit's smoke measures it; a projection above 6 h stops the kit. Memory rules as card ADD60 (20 GB free before every stage; lowest free value during the smoke above 30 GB).

## 3. Reads and names

Per run, the rules of card ADD45 with note N1 as the reader computes them, on WHOLE minus the fraction: M2 (gap R2 and gap error), M4 (presence AUROC); S per seed paired on the same 429 windows of the 48 evaluation episodes; two-level bootstrap (seeds, then episodes), B 2,000.

* `QUARTERSCALE-LEAD-READ-SCALES-GAP` / `-WORSE-AT-SCALE` / `-FLAT-AT-SCALE` and `EIGHTHSCALE-LEAD-READ-SCALES-GAP` / `-WORSE-AT-SCALE` / `-FLAT-AT-SCALE`, by the rule of card ADD60 section 3 (interval of S on M2 above 0, at least three of four seeds above 0, mean gap error of WHOLE at most 0.90 times that of the fraction; interval below 0; else flat). The reader prints `LEAD-READ-...`; under this card each stands for the name with the prefix of its run.
* The curve, from the three fraction reads (EIGHTH, QUARTER of this card and HALF of card ADD60, each as its reader printed it):
  * `FRACSCALE-LEAD-CURVE-RISES-THROUGHOUT` if all three read SCALES-GAP and the mean M2 over the four seeds is ordered EIGHTH below QUARTER below HALF below WHOLE (WHOLE = the mean over the three runs of it).
  * `FRACSCALE-LEAD-CURVE-FLAT-FROM-HALF`, `-FLAT-FROM-QUARTER` or `-FLAT-FROM-EIGHTH` if that fraction and every larger one read FLAT-AT-SCALE and every smaller one reads SCALES-GAP.
  * else `FRACSCALE-LEAD-CURVE-MIXED`, with the table.
* Reported as a table, no bar: M2, gap error and M4 per arm at 571, 1,143, 2,286 and 4,572 clips, mean over the four seeds with the per-seed values. No exponent is quoted from four points. The steps between neighbouring fractions (HALF minus QUARTER, QUARTER minus EIGHTH) are not tested by a paired read; each fraction is tested against WHOLE only.
* The same-seed replicate (reported, no bar): per seed, M2 and gap error of the WHOLE arm in the three runs (cards ADD60, QUARTER, EIGHTH) side by side and their largest spread. A difference of this card or of card ADD60 that is smaller than that spread is said to be inside the run-to-run floor wherever it is quoted. If the three runs are equal to the last digit the floor is reported as 0 for the same seed and stated as such (it then says nothing about another seed; the four seeds do).
* LONGITUDINAL (distance keeping) is the family this lever serves and it is not measured by this read; LATERAL, TACTICAL: not measured by this read. STRATEGIC: NOT COMPUTED (n = 0; a token probe reads no strategic quantity).

## 4. Controls (all must pass per run; else no name from that run, and its differences stay withheld)

Those of card ADD60 as the kit and the reader compute them (d1 with its planted id, d2j, d2n, d2g, d2c with its planted shift, d2e with its planted swap, d3, d4, d5, d7, the record and pack controls), with the same two remarks (d2 compares the lead file with itself; the token report d3t reads equal hashes by construction). In addition: the list of each run has the sha256 written above (the launcher refuses otherwise), and the kit script, tool and reader have the sha256 the run of card ADD60 logged.

## 5. What follows (committed before the data)

* `CURVE-RISES-THROUGHOUT`: the gap read rises with the number of clips over the whole range inside the corpus at matched compute. Data scale is then a measured lever of the lead read from 571 to 4,572 clips, and card ADD45 says whether it continues to 14,512 clips of another collection. Reported as the curve; the proposal that goes to the other session and the PI with it is to train the lead supervision of the integrated candidate on the largest corpus held. No model arm is changed by this card.
* `CURVE-FLAT-FROM-<fraction>`: the probe has what it can use at that many clips, at this number of steps and on the pooled tokens. More clips of the same collection are then not the lever of the lead read; what remains is the steps (the arm with twice the steps), clips of another collection (card ADD45) and the token grid (the unpooled grid is a PI decision and is not started by this card).
* `CURVE-MIXED`, or a name `WORSE-AT-SCALE`: reported as written with the table and the replicate floor.
* A control fails, or the kit or the reader stops with a fault: nothing is named from that run; the fault is logged in `TIMES_NOTE.md` before its repair. A fault of the kit or of the reader is one card ADD60 meets first; its repair goes into the kit of card ADD45 before that card's GPU stages on the other pod, and the stage of this card that failed is repeated in a new output folder. Probes already trained are not trained again when the fault lies in a later stage and the repair does not touch the training tool.

## 6. Limits

One frozen encoder; pooled tokens; each fraction is the first part of the manifest (one split, no repeat over splits; manifest order is not random, so a fraction may differ from the whole in more than its size); four seeds bound the run-to-run variance of the probe across seeds, the replicate bounds it within a seed; matched steps mean the smaller arm sees each clip more often; a probe is not a model; the evaluation windows come from the same collection as the training clips. The tests of the kit ran under a stand-in test runner on this pod. The card was frozen before card ADD60 was read, so its kit and reader had not yet run end to end when it started.

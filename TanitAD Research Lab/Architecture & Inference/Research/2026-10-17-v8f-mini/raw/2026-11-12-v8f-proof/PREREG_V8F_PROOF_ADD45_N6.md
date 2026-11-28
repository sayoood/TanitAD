# Note N6 of card PREREG_V8F_PROOF_ADD45 (DSCALE-LEAD): the fall clause of control d4 on a run that sees each clip about once

Written by the coordinator fork after the read of record of the union run (notes N4 and N5; folder `dscale_lead_add45/read_n5` on the pod of this fork, result sha256 cab980237d12126d) ended with the reader's exit code 2, and before any outcome of that run was opened (no S, no interval, no name, no level of any arm on the evaluation set). The stop was logged in `TIMES_NOTE.md` before its cause was known. Time of the freeze: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file.

## 1. What failed

Control d4 (`p5_u`: "the lead term of the last 1,000 steps is below that of the first 1,000 steps") failed on the five probes trained on the union list and on no other: `P16U_s0` to `P16U_s3` and `P16USH_s0`. Every other control of the read passed (the precheck controls, the evaluation set, the estimator controls, k2 under note N4, the gap clause of d4 under note N5 with no short record, the header checks, the runs complete at step 19,164). The four corpus probes, the corpus probe with permuted gap labels and the union probe with twice the steps pass d4 as frozen. By the card nothing is named from that read and its differences stay withheld; it stays on record as it is.

## 2. Cause (MEASURED from the train logs of the run on the pod of this fork; training losses only, no evaluation value)

* The lead term at step 1 (the untrained probe) is 0.74 to 0.75 on the union probes. By step 100 it is about 0.40. From there to the end it moves between about 0.26 and 0.55 with the pool of clips the sampler is in, without a trend. The mean of the first 1,000 steps (0.392 to 0.394) is therefore already the level after the fall, and the mean of the last 1,000 steps (0.394 to 0.414) is another stretch of the same level on other clips.
* The union list has 14,512 clips and 15,392 steps per pass; at the matched 19,164 steps a union probe sees each clip about 1.25 times. A corpus probe (4,512 clips, 4,791 steps per pass) sees each clip four times, and its training loss keeps falling over the passes (last 1,000 steps 0.241 to 0.256). The clause as frozen thus reads the fall over repeated passes; on a run with about one pass there is none to read, whether or not the probe has learned.
* That the union probes have learned is seen in the same logs: the lead term of the last 1,000 steps is 0.52 to 0.55 of its value at step 1 (corpus probes 0.33 to 0.35; the union probe with permuted gap labels 0.57 on its presence part; the union probe with twice the steps 0.50), and the gap part of the last 1,000 steps is 0.0084 to 0.0090 on the four union probes against 0.0209 on the union probe with permuted gap labels (corpus: 0.0077 to 0.0079 against 0.0210).

The fault is in the form of the clause, which was written on corpus-size runs: it was the card's own design (matched steps on a list three times as large) that put the union probes at about one pass.

## 3. The rule from this note on

The read of record of the union run of card ADD45 goes through the wrapper `dscale_lead/dscale_lead_read_n6.py` around the unchanged reader and the wrappers of notes N4 and N5. Control d4 of a probe passes if it passes under note N5, or if all of the following hold:

* every other clause of d4 holds (the header checks, the run complete at its last step, the gap clause as frozen or under note N5);
* the log holds the record of step 1, and the lead term of the last 1,000 steps (in the form of the probe: presence and gap parts, or the presence part alone for a probe with permuted gap labels) is at most 0.80 times the lead term at step 1.

The bar 0.80: an untrained probe is at 1.00 by construction; a probe that has only learned the share of frames with a lead reads about 0.94 on the presence part (the entropy of a share of 0.64 against 0.69); the probes of this run read 0.33 to 0.57. A probe that passes under note N5 passes under this note. The result file carries, per probe, the verdict under note N5 (`pass_under_note_n5`), the lead term at step 1, the ratio, and the bar.

Reported beside and not judged: the gap part of the last 1,000 steps of each probe.

## 4. Checks of the wrapper before the read (pod CPU)

* Synthetic logs: a log that passes as frozen passes; a log that falls within the first 100 steps and then stays level fails as frozen and passes under the note; planted faults that must fail under the note: a log that stays at its start value, a log that ends at 0.87 of its start value, a level log whose last record is not the last step, a level log with a full-batch record without a gap label, a log without the record of step 1; the same for the presence-only form.
* Known value: on the HALF run of card ADD60 (read of record under note N4, all controls passed; every probe passes d4 as frozen) the read through the new wrapper gives the same result file leaf for leaf, apart from the fields the wrapper adds.

## 5. What this note does not change

No bar, tolerance or name of card ADD45 other than the clause above; d3 (the corpus arm is a live gap probe) and d5 (the permuted-label probes) stay as frozen and are read as they come. The training tool and the kit are not touched; no probe is trained again. The first read stays in its folder.

## 6. Limits

The note was written knowing the control details of the first read (the list of failed controls, the d4 details of the eleven probes with their training-loss means) and nothing of the evaluation set. The bar 0.80 was set after seeing that the ratios of this run lie between 0.33 and 0.57; it is placed by what an untrained probe and a probe that learned only the lead share would read, not by those ratios. If the PI or the other session reads control d4 as needing the fall over the first and last 1,000 steps as written, card ADD45 stays without a name and its result is reported as a read whose control did not pass.

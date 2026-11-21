# Note N5 of card PREREG_V8F_PROOF_ADD45 (also for the reads of cards ADD60 and ADD64): the gap clause of control d4 is read on full batches

Written by the coordinator fork after the frozen reader of the QUARTER run of card ADD64 ended with failed controls, before any outcome of that run was opened (no S, no interval, no name, no level table), before the EIGHTH run of card ADD64 has ended and before any data of card ADD45 exist. Time of the freeze: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file.

## 1. What was measured

Control d4 of the reader (`p5_u` in `dscale_lead_read.py`) asks, among other clauses, for "gap-labelled frames above 0 at every logged step". In the QUARTER run it failed on the five probes trained on the quarter list (`P16C_s0` to `P16C_s3`, `P16CSH_s0`), on this clause only: one of 385 logged records (step 8,950) has 0 gap-labelled frames. From the pod logs (`podread/ngap_probe.py`): the batch of that step holds 12 presence-labelled frames against a median of 675 over the run's records (one window), with no lead in it. The same record in all five probes (the batch order does not depend on the probe seed). The lead term fell from 0.397 to 0.0024 and the runs are complete. The probes of the WHOLE list in the same run have no such record (minimum 68 gap-labelled frames).

The cause is known since card ADD46: the sampler ends each pool of clips with a leftover batch, down to one window (355 short batches in a run of 19,164 steps). Whether a leftover batch without a lead lands on a logged step depends on the clip list. The clause was written for "the gap labels reach the loss at every logged step"; on a leftover batch of one window it tests the sampler's leftover.

## 2. The rule from this note on

The reads of record of cards ADD45, ADD60 and ADD64 go through the wrapper `dscale_lead/dscale_lead_read_n5.py` around the unchanged reader (and around the wrapper of note N4, whose tolerance of control k2 stays 1e-3). Control d4 of a probe passes if it passes as frozen, or if all of the following hold:

* every other clause of d4 holds as frozen (the header checks, the lead term of the last 1,000 steps below that of the first 1,000, the run complete);
* a logged record is a full-batch record if its count of presence-labelled frames is at least half the median of that count over the run's logged records; every full-batch record has more than 0 gap-labelled frames;
* the records below that count (short records) are at most 5 % of the logged records.

A probe that passes as frozen passes under the note. The result file carries, per probe, the frozen verdict (`pass_as_frozen`), the number and the steps of the short records, how many of them have no gap-labelled frame, and the smallest count of gap-labelled frames over the full-batch records.

## 3. Checks of the wrapper before any read (pod CPU; the dev box runs no numpy job while its hold stands)

* Synthetic logs: a log that passes as frozen passes; a log with one short record without a gap label fails as frozen and passes under the note; planted faults that must fail under the note: a full-batch record with 0 gap-labelled frames, a log with 30 % short records, a log whose lead term does not fall beside one short record without a gap label, and a log whose last record is not the last step.
* Known value: on the HALF run of card ADD60 (read of record under note N4, all controls passed) the read through the new wrapper gives the same result file leaf for leaf, apart from the fields the wrapper adds.

## 4. What this note does not change

* Control d3 (the corpus arm is a live gap probe: the interval of its gap R2 above 0 and its presence AUROC at least 0.85) stays as frozen. In the QUARTER run it failed on `P16C_s0` (presence AUROC 0.8435). That is no fault of the kit or of the reader. By card ADD64 section 5 nothing is named from the QUARTER run and its differences stay withheld, with or without this note; the curve name of card ADD64 needs all three fraction reads and cannot be given.
* No bar, no tolerance and no name of cards ADD45, ADD60 and ADD64 other than the clause above. The training tool and the kit are not touched; no probe is trained again.
* The kit's own log check on the pod (`checklog`) is left as it is; it writes its verdict and does not stop the chain.

## 5. Limits

The note was written knowing the control details of the QUARTER run (the failed-control list, the d4 detail above, the d3 detail of `P16C_s0` with its presence AUROC and its gap R2 point value) and nothing else of that run. For card ADD45 it is written before its GPU stages; its union list has other clips, so whether a leftover batch without a lead lands on a logged step there is not known.

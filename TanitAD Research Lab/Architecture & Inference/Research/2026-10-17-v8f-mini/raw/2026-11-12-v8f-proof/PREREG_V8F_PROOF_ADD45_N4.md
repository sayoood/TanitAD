# Note N4 to card PREREG_V8F_PROOF_ADD45 (DSCALE-LEAD), binding for cards ADD60 and ADD64 as well: the tolerance of reader control k2

Written by the coordinator fork after the first end-to-end run of the kit (card ADD60, folder `half1`) ended with the reader's exit code 2, and before any repaired read (time of the freeze: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). The fault was logged in `TIMES_NOTE.md` (line of 2026-11-19T23:03:11Z) before its cause was known. No outcome number of that run (gap R2, gap error, presence AUROC, any difference between arms) has been read by the coordinator or by the agent that made the measurement below: the files that hold them were not opened.

## 1. What failed

Reader control k2 ("the head applied by the reader to the pulled states equals the pod's outputs", largest absolute difference over all rows with a history, tolerance 1e-4 for the gap in m and the closing speed in m/s; `TOL_K2` in the reader of the lead-supervision card, used unchanged by the readers of cards ADD43 and ADD45) failed on the two permuted-label arms of the run and on no other: `P16CSH_s0` 1.255e-4 m, `P16USH_s0` 1.018e-4 m. The nine other arms read 4.4e-5 to 9.2e-5 m and pass. The closing speed is inside the tolerance in every arm (at most 5.5e-5 m/s). By card ADD60 nothing is named from that read; it stays on record as it is (`half1/read`, exit code 2).

## 2. Cause (MEASURED on the pod CPU; `dscale_lead/halfscale/half1/K2_DIAG.json`, sha256 1ff8ae992bca23df, tool `dscale_lead/k2_diag.py`, sha256 ba2a3c5fa1b13124; only reader-against-pod output differences were computed)

* The reader runs the head forward in float32 on the CPU; the pod's outputs were written in float32 by the GPU extract stage. 13,673 rows per arm, gap values up to 120 m.
* The largest absolute gap difference per arm is 4.4e-5 to 1.26e-4 m; the 99.9th percentile is 3.1e-5 to 6.1e-5 m in every arm; the difference does not grow with the size of the value (Pearson correlation of the absolute value with the absolute difference at most 0.26).
* The same head forward in float64 (a supplementary forward written for this check, same weights) lies up to 1.16e-4 m from the reader's own float32 forward on `P16CSH_s0`. So the float32 forward alone is about one tolerance away from the exact value, and two float32 forwards on two devices cannot be expected to agree to 1e-4 m on values of this size.
* What a real mismatch looks like (planted: the reader's outputs for one arm against the pod's outputs of another arm on the same rows): 36.1, 60.4, 64.8 and 95.8 m.

Reading: the tolerance 1e-4 sits inside the float32 rounding of the comparison (nine arms at 0.44 to 0.92 of it), and the two arms that crossed it did so by rounding, not by a mismatch of head or states. The reading would be contradicted by a difference that is systematic or grows with the value; neither is seen.

## 3. The change

* For the reads of cards ADD45, ADD60 and ADD64 the tolerance of control k2 is 1e-3 (gap in m, closing speed in m/s): eight times the largest rounding difference measured, and more than four orders of magnitude below the smallest planted mismatch (36.1 m). Every other part of control k2 stays (same rows with a history, at least 64 windows), and every other control, read, rule and name of the three cards stays as frozen.
* It is applied through the wrapper `dscale_lead/dscale_lead_read_n4.py` (sha256 d3925e1312e59b00; `--k2-tol 1e-3`), which sets the constant, calls the unchanged reader and records the value used in the result. With its default (1e-4) the wrapper reproduces the frozen read (checked on `half1`: exit code 2, the same two failing entries). No existing file is edited.
* The read of record of each run (`half1`, `quarter1`, `eighth1`, and the union run of card ADD45) is the read through this wrapper into a new output folder `read_n4`. Where the reader with the frozen tolerance passes on its own, its result is kept beside it and must hold the same numbers.
* Card ADD60 says the card "is repeated in a new output folder" after a repair. The coordinator's reading, stated here before the repaired read: the fault lies in the tolerance of a reader control and the repair touches neither the training tool nor the extract stage, so the stage that is repeated is the read, on the same trained probes and the same extracted states. If the PI or the other session reads the sentence as a full repeat, the run is repeated; the WHOLE arm is trained again with the same seeds in the two runs of card ADD64 in any case.

## 4. Why this is admissible

Control k2 checks that the tool is consistent with itself; it is not a bar on an outcome. Its tolerance was set on small earlier runs without a measurement of the rounding of a float32 forward on values of this size. The change is made on a measurement that involves no label and no outcome, before any outcome number of the run was read, and the control keeps its power: a wrong head or wrong states read tens of metres, the tolerance is a millimetre. The failed read with the frozen tolerance is reported as it is in the row of card ADD60.

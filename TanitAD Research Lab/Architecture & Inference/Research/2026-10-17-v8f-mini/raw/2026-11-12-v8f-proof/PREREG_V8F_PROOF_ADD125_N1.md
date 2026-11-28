# PREREG_V8F_PROOF_ADD125_N1 — note to card ADD125 (DSCALE-LEAD-EQP): open definitions found while the reader was built

Written after the reader kit was built and tested on synthetic data, while the run of the card is waiting on the pod of the other session, and before any file of the new run exists on this side. The card (9dfbc7f38e665b06) is unchanged. No bar and no name is touched.

## 1. Control e0, the hash of the run script

The card asks that the hashes of the run script and of the saving tool in the new done file equal the landed ones. The done file that the run script writes carries the hashes of the probe tool, of the saving tool, of the precheck tool and of the clock check tool; it carries no hash of the run script itself. As written, that half of the clause cannot be read from the done file.

Reading fixed here:

* The saving tool: its hash in the new done file and in `runQ/CK_RECORD.json` equals the landed one (5888527b5c37003b). MEASURED by the reader.
* The run script: the other session is asked to put a file `KIT_SHA.txt` into the output folder before it makes the tar, with the sha256 of the three kit files as they stand on its pod at that time. The lead compares its line for `pod_dscale_lead_2x.sh` with the landed hash (a01c5ca8cb43a2c5) by hand before the names are opened and logs the comparison in `TIMES_NOTE.md` with the controls. That file is written on the pod by the session that ran the kit, so the evidence class of this half is INHERITED from that session; its two messages before the start (dry check and chain start) name the same hash. Without that file the half is logged as INHERITED from those messages alone. A hash in that file that differs from the landed one fails e0.
* The reader also hashes the landed copy of the run script and prints it; that line proves the landed file and nothing about the pod.

## 2. Control e1, the batch sequence beside

`runQ/r1_batches.json` carries the total step count of its run, so the two files of the runs with 38,328 and 61,568 steps cannot be equal as files. Beside the control the reader prints how many of the shared entries of the batch hashes agree. As in the card, this is printed and not judged. The judged part of e1 is unchanged: the record of step 1 of each of the four arms agrees with that of the same arm of the run of card ADD123 within 1e-4 on every required loss term.

## 3. Control e4, the states

The reader reads the last state of each arm of the new run (step 61,568), as the card says. The states at 19,164 and 28,746 steps that the script also writes are not read.

## 4. Control e2, which probes have to be live

As written in the card: the eight probes of E (`P16E_s0` to `P16E_s3` and `P16C_s0` to `P16C_s3`). The probes `P16Q` and `P16U` enter only values printed beside; they were read as live in row 192.

## 5. Rehearsal

Before the tar arrives the controls stage is run once on the fork pod with the run of card ADD123 standing in for the new run, into a rehearsal folder. Expected by construction: e0 and e4 fail on the step count (38,328 in place of 61,568), e1 passes trivially (a run against itself). The rehearsal checks that the reader reads the real file layouts; it computes no measure of E.

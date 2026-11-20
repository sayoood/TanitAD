# Note N5 to card PREREG_V8F_PROOF_ADD70 (LONGROLL10K): control n2 of note N4 stopped the re-read; what it showed and its form of record

Written by the coordinator fork after the re-read under note N4 ended with control n2 not passed and with no name, no level and no difference written (`integ10k/read_longroll_n4/pod/READ_LONGROLL10K_N4.json`: names empty), and before any value of a re-read is opened (time: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). As in note N4, the values of the read under note N3 are known. This note applies to the reader of card ADD71 as well (its note N1, item 2).

## 1. What the re-read under note N4 reported (controls only)

* n1 passed: every run without a pre-roll starts at the origin (largest |p0| 0.0 in 12 runs).
* n3 passed: the planted start offset reads a speed error of 0.433 m/s and every window unsafe with the engines as they were, and speed, heading and yaw-rate error 0, along-track error 5.0 m, no window unsafe and the same gaps with the repaired reader; with a zero offset the repaired reader equals the engines exactly.
* n2 passed for the twelve pre-rolled runs under readout K (share of windows whose first track point lies near p0: 1.0 in each) and failed for the four pre-rolled runs of the convention: 0.9536 and 0.9709 at 30 steps, 0.8204 and 0.8768 at 60 steps, against the bar of 0.99.

## 2. Why n2 failed there (probe on the pod, controls only, no outcome cell: `podread/probe/probe_n2.py`)

n2 allowed the first step "0.5 m plus one step of travel", with the travel taken from the own speed at t0. In the convention's pre-rolled windows that fail it, the step the plan drives is not the state speed times the step time: first step from p0 between 1.4 and 12.0 m with a second step of the same size (1.2 to 12.0 m; 12.0 m per step is the cap of the pinned tool), while the state speed times 0.2 s is 0.5 to 11 m. Those tracks jump between steps (|step 1 minus step 2|: 99th percentile 2.69 m in the convention, 0.11 m under readout K). The bound of n2 was wrong for such tracks; the start point was not: in all four probed runs the norm of p0 equals the position offset the wrapper itself wrote for the window (`preroll_offsets`, `dpos`) to 1e-13 m.

## 3. Control n2, form of record

* Every pre-rolled run: each dump row has its tick row, and |p0| equals the wrapper's `dpos` of that window to 1e-6 m. This ties the start point to the wrapper's own record of the carried offset.
* Runs under readout K, whose tracks are smooth: in addition the first track point lies within 0.5 m plus one step of travel of p0 in at least 0.99 of the windows, the step of travel being the larger of the own speed at t0 times 0.2 s and the length of the second step. This checks the direction of p0 in the frame of the dump; the frame mapping of the wrapper is the same code for every readout.
* For the other pre-rolled runs that share is reported and not required (probe, with the larger-of form: 0.9835, 0.9976 at 30 steps and 0.9507 at 60 steps).

The file `integ10k/read_longroll10k_n4.py` is changed in this control only (hash before and after in `TIMES_NOTE.md`); four planted cases of the control pass locally (a K run below the bar fails, a norm that differs from the wrapper's fails, a fresh run away from the origin fails, the form of record passes). The re-read runs into the new folder `out/read_longroll_n5/` with `integ10k/podread_longroll_n5.sh`; the stopped re-read stays on record.

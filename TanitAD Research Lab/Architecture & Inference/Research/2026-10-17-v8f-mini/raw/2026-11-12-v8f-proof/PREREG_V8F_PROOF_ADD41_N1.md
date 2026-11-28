# PREREG_V8F_PROOF_ADD41, note N1: the reader runs on the pod; the pod form of control g7 (TK side); two readings fixed

Frozen 2026-11-18 before any number of the pilot exists (the pilot has not begun its training: its start wrapper waits on the pod for the two confirmation chains to be started). The card is not edited. Everything the card says and this note does not name stays as frozen. Tier T0 (a diagnostic on logged frames, never driving performance).

## 1. Why

The reader of the card loads the state files of every arm (ESTIMATED 7.65 GB for the F arms). The dev box has no room for them (MEASURED 2026-11-18 23:09Z: C: 6.3 GB, F: 18 GB free, F: below the 20 GB floor another session needs; deletes are refused for this session). The states stay on the pod and the reader is run there, unchanged, from a folder that mirrors its fixed paths (`/workspace/v8proof/finegrid/rk/`; the files it opens come from a trace of the reader on the dev box, `finegrid/podkit/`, 38 files in the kit; the states of the TK arms are the pod's own copies in `probe/states`, checked by the reader against their records).

## 2. What was measured before this note (a calibration of the instrument, no pilot number)

* Dev box, `finegrid_read.py --tk-only` (sha256 eae818c1...): the M1, M2, M4 of the four TK arms equal the banked values of row H-V8F-PROOF-56, maximum absolute difference 0.0 (build run and the trace run of 23:15Z). This is control g7, TK side, as the card states it, and it is met.
* Pod, the same file, the same inputs (numpy 2.5.2 and Python 3.12 there, numpy 2.5.1 and Python 3.13 on the dev box, another CPU): maximum absolute difference 2.6e-07, 1.2e-06, 9.3e-07, 1.5e-06 for TK16_s0, TK16_s1, TK64_s0, TK64_s1. The reader therefore reports the control as not met on the pod, in the seventh decimal.

## 3. The pod form of control g7, TK side

On the pod the TK reproduction passes if the maximum absolute difference of every TK arm is at most 1e-5 (about seven times the largest calibrated value). The exact form stands as measured on the dev box in section 2. Reader: `finegrid/finegrid_read_n1.py`, made from `finegrid_read.py` by `finegrid/make_read_n1.py` (an argument `--g7-tol`, default 0.0 = the first version; four edits; nothing else differs). Checks of the changed control, run on the pod before the pilot ends: (a) default tolerance: not met (as in section 2); (b) tolerance 1e-5: met; (c) tolerance 1e-5 with one banked value moved by 1e-3: not met. If (c) is met the tolerance form is not used and the names wait for a dev-box run.

## 4. Guard for a name decided in the last digits

A name of section 5 of the card is given from the pod run only if every comparison that decides it is farther than 1e-4 from its bar: each interval end from 0, each mean from the floor, the gap MAE ratio from 0.90. A name closer than that is reported as "not named yet" and waits for a run of the reader on the dev box.

## 5. Two readings fixed here (review items of the tool build)

* "Each mean is larger than the floor" (section 5 of the card): the mean over the two seeds of G16 on M2, and separately the mean over the two seeds of G16w on M2, each strictly above the floor.
* Token order of control g2: row-major (token index = row times 20 plus column for the 8 by 20 grid). The control itself tests this reading.

## 6. Said plainly

* A check of the tool on the pod's CPU before the start (20 clips, 200 frames, into a separate folder `runs/finegrid_precheck`) found control g2 crashing on the real cache (a tensor index on the stored float8 type); one line of `finegrid/tok_probe_g.py` was changed (cast before index; sha256 d8daaf00... after, 7fe80165... before). The tests of the build were not run again after that change. The precheck then read the g2 correlation as 0.99954 (bar 0.99; the column-major reading gives 0.431). This is a value of a control and was seen before the run; no read of an arm was seen.
* The same precheck ran 20 training steps on the CPU (the train path reads the real cache); its control p5 is not met at 20 steps and says nothing about the run.
* The pilot runner is started through a wrapper (`finegrid/pod_finegrid_start.sh`) that holds it back until the launcher of the confirmation chains has started both, so that two slot tests cannot pass at the same moment.

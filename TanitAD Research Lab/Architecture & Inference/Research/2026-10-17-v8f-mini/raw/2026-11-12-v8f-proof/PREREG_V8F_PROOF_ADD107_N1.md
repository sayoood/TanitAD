# Note N1 to card ADD107 (LOOKAT-T0): open definitions and limits by construction, fixed before any run

Card: `PREREG_V8F_PROOF_ADD107.md` (sha256 begins 97468f1fb405e88b). This note is frozen with its own hash file after the kit of the card was built and tested on synthetic data and before anything of the card has run on the pod (no identity run, no fit, no read). It changes no control, no bar and no name of the card. Kit: `lookat/` (`locator_fit.py` b0276f2f1dff8358, `read_lookat.py` 8ed6c6256fd5149b, `pod_lookat_queue.sh` e3cd11251e6c6c1d, `podread_lookat.sh` d004bc8ff46e3e57; definitions as coded in `LOOKAT_DEFS.md` 2251cabc6ab80024).

## 1. Limits by construction (stated, not repaired)

1. Control k3 depends on the subject: it needs fit seed 0 to locate better than the fit with permuted targets. If the locator of seed 0 is weak, the card fails at k3 and is reported as failed with no name; the names `LOOKAT-T0-DOES-NOT-LOCATE` and `LOOKAT-T0-NOT-SEPARATED` can then only be reached through fit seeds 1 or 2. The lead set k3 this way before the freeze and did not see it; the card runs as frozen. A control that does not depend on the subject would be a new card.
2. Control k5 (identity within 1e-4 on a loss of about 5.17) is expected to hold on the pod's GPU and torch, where the recorded fit was made; it is not expected to hold on another device. The queue runs it first and stops when it fails.
3. The targets are clipped at 32 degrees of azimuth. A row with the lead more than 40 degrees off the axis cannot count for `S8_out` unless the output goes beyond every training target. A lead more than 16 degrees off the axis is outside both fixed windows; the head can only infer it from the rest of the scene. The names log prints the row counts of both kinds.

## 2. Open definitions, as coded

* Frames of the fit: `has_lead` and finite azimuth and elevation in the box file, with a full history of five frames, built as the fit tool of card ADD99 builds its samples of head H5.
* k0, "list hash as in the fit records of card ADD99": the sha256 of the pilot's train list and evaluation list equal the two list hashes in the record of the fit D-H5 seed 0. The frame count of a token file is read from its header.
* k1: against the manifest of arm D of the pilot's encode (641 entries with frame count, byte size and sha256); all 641 token files are hashed.
* k2, "the settings of section 2": learning rate 1e-3, weight decay 0.05, batch 256, 40 epochs, one warm-up epoch, smooth L1 with beta 0.25, token dropout 0.1, width 256, 4 heads; 40 entries of history; the last layer starts from the default initialisation of the library (the card names none).
* k4: also requires a finite output of all four fits on every row of the row set; when rows with the elevation more than 8 degrees off the window centre exist, one count line is printed after the control lines (a count of labels, not a value of the read).
* k6: reads the lead's test output pushed with the kit (28 passed in the lead's run).
* Stage `controls` prints the seven control lines and no value; they are logged in `TIMES_NOTE.md` before stage `names`, which opens nothing unless k0 to k6 are all present and PASS.

## 3. Launch settings (no file of the kit changed)

The queue waits for a free slot (fewer than 3 GPU processes, more than 28 GB free) for 1,800 s by default and then stops. The jobs of card ADD106 hold the GPU slots at the time of the launch, so the lead launches with `LK_WAIT_S=7200`. The stop rule of the card (1.5 pod-hours of this card's jobs) counts the time in which a job of this card runs, not the wait for a slot.

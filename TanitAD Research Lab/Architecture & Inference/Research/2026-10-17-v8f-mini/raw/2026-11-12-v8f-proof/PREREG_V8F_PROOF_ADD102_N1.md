# PREREG_V8F_PROOF_ADD102_N1: definitions the card leaves open, fixed before any rollout

Fork v8proof, 2026-11-21. Note to card `PREREG_V8F_PROOF_ADD102.md` (VISRULE10K, 6e61e621b54667e8). Written after the helper's hand-back of the kit `visrule/` (its list of clauses that could fail by construction and of items the card leaves open) and after the lead's review of the generated wrapper by diff; frozen with its hash file before the kit is pushed, before any gap file is built and before any rollout of the card. It changes no control, no bar and no name of the card. No value of real data of this card exists at this point.

## 1. Control v1: which rates

The card: "contact, unsafe rate and falling behind each within 2 windows of the 650 with a lead". The 650 windows are the two model seeds together (325 per model seed). So each rate is the pooled rate, the mean of the two per-seed rates as the reader forms its pooled levels, and the tolerance is 2/650 as a rate for each of the three. The differences per model seed are printed beside it and carry no part of the control. The helper's first build applied the tolerance to each model seed's own rate; the lead changed `ctl_v1` of `read_visrule10k.py` and its test to the reading above before any run.

Stated in advance: v1 compares a new rollout of the reader-fed form, made by the variant with its new arguments off, with the banked rollouts of RR, which an earlier generation of the wrapper made. The pod check of the variant shows that it equals its source on a few clips; it does not show that a new rollout equals a banked one (GPU arithmetic, the generations between). The tolerance stays as frozen. A failed v1 fails the card with no name.

## 2. Control v2: the reader's own gap as RR feeds it

Read from the rule logs of the control run V0 (the variant with its arguments off, the form RR) at the same clip index and frame, within 1e-4 m. The reader's gap on a logged frame does not depend on the rollout, so V0 carries what RR feeds. A step of VD5c or VD5 with a lead and no head output that has no V0 row with a lead at that clip and frame fails v2. In practice no such step is expected: the reader needs five frames, as the head does, so it marks no lead on frames 0 to 3.

## 3. The head has an output

A frame where the dumped gap of head D-H5 is finite. The dump's own validity field (the present reader's) is not used.

## 4. The run of the closing input in VD5c

The card gives the run rule of the label source. On the reader-fed path a frame is in the run when the reader marks a lead on it (its presence output at or above its threshold with a full history, as RR marks a lead) and the gap the rule receives on it is finite and within 0 to 60 m, the label source's own test. The frame of the look not in the run gives 0. The four earlier frames are read again from the logged tokens and fed the same way (the reader's presence; the head's gap where it has an output, else the reader's own gap). The slope is the least squares slope against time in steps of 0.2 s; positive means the gap opens, as in the label source.

## 5. Where the change sits

After the reader's own map and minimum: on a frame where the reader marks a lead and the head has an output, the gap the rule receives is the head's gap as dumped, with no map and no minimum; on every other frame the reader's own gap after its map and minimum stays. The reader's presence output is not changed, so the frames with a lead are those of RR.

## 6. Control v0: two additions

Beside the card's clause the reader's v0 also asks that the dump's label gap equals the join's lead gap within 1e-4 m on every label frame, and that the saved gap file equals a second routine's value on every row. Both make v0 stricter.

## 7. Half the gap in the names

H is half of the point value of the contact of RR minus that of RLc, as the reader recomputes them (the rule of card ADD101); the card prints it as 0.0477.

## 8. Control v5: what it reads

The engine's own controls of the grid read (the same windows in every dump, its headway controls, its family controls) and of the strategic grid. They read the dumps of the rollouts and not the closing column of the rule logs (checked in code by the lead: `integ10k/read_longroll10k.py`, the line that sets `ALL_PASS`), so the non-zero closing input of VD5c is not a reason for v5 to fail by construction.

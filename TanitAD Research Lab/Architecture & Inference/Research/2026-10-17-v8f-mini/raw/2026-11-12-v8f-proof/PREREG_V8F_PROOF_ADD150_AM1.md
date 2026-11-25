# PREREG_V8F_PROOF_ADD150_AM1 — LEADSPEEDALL10K, amendment 1: two training seeds

Frozen before card ADD150 (abd9d990) has run, before any of its numbers exist, on the other session's read of row 215: card ADD149 line 32 commits a replicate when a result is near a bar, and row 215's held-out S (0.288) is 0.012 under its bar; one training seed cannot read a result within a few hundredths of 0.30.

## The change

ADD150 trains **two training seeds, 148 and 149** (`--train-seed`, `rulefc/leadspeed/train_full_box2.py`: the seed of the head's initialisation and of the batch draws). The clip split stays on seed 148 for both (the same 3,984 fit, 443 held-out and 141 eval clips). Each seed is one run of card ADD150 section 1, with its own output folder (`D:/tanitad/leadspeed/all` for 148, `D:/tanitad/leadspeed/all149` for 149), controls c1 to c5, reads, and the paired read beside against row 215's head.

## Names (committed; replaces card ADD150 section 4's first sentence)

Each seed gets its name by card ADD150 section 4 (A >= 0.30: CARRIED; A < 0.30 and S >= 0.30: SPEED-ONLY; otherwise NOT-CARRIED, on its held-out clips). The arm's name is that name when both seeds give it, and `LEADSPEEDALL10K-SEED-SPLIT` when they differ. Reported beside: the two seeds' S and A, their differences (the training-run spread on this rig), and the mean of the two.

## Consequence

As card ADD150 section 4 for CARRIED, SPEED-ONLY and NOT-CARRIED, with the eval-141 condition required of both seeds. SEED-SPLIT acts on nothing: the next lever is a third seed (its own card).

## Unchanged

Everything else in card ADD150.

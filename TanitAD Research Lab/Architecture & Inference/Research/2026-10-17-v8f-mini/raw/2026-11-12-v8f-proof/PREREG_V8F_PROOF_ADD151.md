# PREREG_V8F_PROOF_ADD151 — LEADSPEEDREP10K: the replicate of card ADD149 that its line 32 commits

Frozen before its run and before any of its numbers exist. Open at this time: row 215 (card ADD149 with amendments 1 to 3, `LEADSPEEDFULL10K-NOT-CARRIED`, held-out S 0.288 and A 0.191 from training seed 148). Card ADD149 line 32: "One training seed: a replicate is a successor if the result is near a bar"; S is 0.012 under its bar of 0.30. Tier **T0** (a head on cached tokens of logged frames); never driving performance. The run waits for the PI's go (box 2 time).

## 1. The run

Card ADD149 with amendments 1 to 3, unchanged, with one change: **training seed 149** (`--train-seed 149`, `rulefc/leadspeed/train_full_box2.py`; the head's initialisation and the batch draws). The split stays on seed 148: the same 1,500 fit, 443 held-out and 141 eval clips. Output `D:/tanitad/leadspeed/rep149`. Controls c1 to c5 as card ADD149 with amendment 3.

## 2. Reads

S and A on the held-out clips (primary) and on the eval-141 clips, as card ADD149 section 2. Beside, names nothing: the paired read against row 215's 30,000-step head (`D:/tanitad/leadspeed/full/head_30000.pt`, sha256 7048714a checked before it is read; also banked at `leadspeed/add149_full/` in the private HF dataset), clip-cluster bootstrap over the 443 held-out clips, 2,000 resamples, seed 150. Since the two runs differ only in the training seed, that difference is the rig's training-run spread at 1,500 clips.

## 3. Names (committed)

The replicate gets its own name by card ADD149 section 4 (`LEADSPEEDREP10K-CARRIED` / `-SPEED-ONLY` / `-NOT-CARRIED`). Row 215's name is `REPLICATED` if the replicate's name is NOT-CARRIED too, and `SEED-SPLIT` otherwise. Reported beside: the two seeds' S and A, their differences and their mean.

## 4. Consequence

REPLICATED: row 215's NOT-CARRIED stands at two seeds, and the data lever (card ADD150 with its amendment 1, two seeds on 3,984 clips) is the next arm, unchanged. SEED-SPLIT: the 1,500-clip head is at the bar within the rig's training spread; card ADD150's two-seed read is still the next arm, and row 215's name is quoted only with the replicate beside it.

## 5. Cost

As row 215: about 3 min of loading and about 27 min of training on box 2, plus the reads (minutes).

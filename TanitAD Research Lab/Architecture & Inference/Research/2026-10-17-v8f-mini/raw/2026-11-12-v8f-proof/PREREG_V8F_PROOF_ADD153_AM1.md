# PREREG_V8F_PROOF_ADD153_AM1 — stage 3 of ACCHIST10K: the two-seed ensemble of the selected arm, a read with no training

Frozen after the six runs of card ADD153 were read and before any ensemble output exists. Tier **T0**; never driving performance. Card ADD153 (sha256 16c605f6) stands unchanged for everything it names.

## 0. What is known at this freeze (card ADD153, MEASURED, `rulefc/leadacc/box2_acc_*/ACC.json`)

The selected arm `K25D` reads `ACCHIST10K-CARRIED` on the held-out clips at both training seeds (A 0.359 / 0.353; dA_floor +0.081 [+0.044, +0.118] / +0.075 [+0.037, +0.117]), so the card's name is `ACCHIST10K-CARRIED`. Its committed consequence also needs A >= 0.30 on eval-141 at both seeds, and that part is not met: eval-141 A 0.260 / 0.289. The two seeds differ by 0.029 in eval-141 A, so part of each head's error is its own training run.

## 1. The read (Rule Zero: the cheapest next lever, no training)

`ENS2`: the mean of the outputs (presence probability, gap, closing, d) of the two `K25D` heads at 30,000 steps (seed 148 `D:/tanitad/leadacc/K25D/head_30000.pt`, sha256 prefix from its ACC.json; seed 149 `D:/tanitad/leadacc/K25D_s149/head_30000.pt`, likewise), read on the same held-out and eval-141 clips and frames, with the reads of card ADD153 section 3 (S, A record, the beside reads, dA_floor with its paired bootstrap). Beside, naming nothing: `ENS3` = `ENS2` plus the smoother-label head (`D:/tanitad/leadacc/K25DS/head_30000.pt`), and the paired difference of `ENS2` against the seed-148 head alone (clip-cluster bootstrap, 2,000 resamples, seed 152).

## 2. Controls

* e1: the one-head "ensemble" of the seed-148 head reproduces that run's ACC.json held-out S and A to 1e-6.
* e2: every checkpoint's sha256 equals the one its ACC.json recorded at save.
* e3: the planted read (label closing and label d in place of the ensemble) reads S = 1 and A = 1.

## 3. Name and consequence (committed)

`ENS2` is named by card ADD153 section 5 (`ACCHIST10K-ENS2-CARRIED` / `-SPEED-ONLY` / `-NOT-CARRIED`, held-out). If `ENS2` is CARRIED on the held-out clips **and** reads A >= 0.30 on eval-141 with its eval-141 dA_floor lower bound > 0, then ENS2's acceleration (and closing) is the input of card ADD144's constant-acceleration forecast in a reader-form rule arm (a successor card: build, rollouts on box 2, the PI's go). An ensemble of two training runs has no replicate within this amendment; the two members' own spread (above) is quoted beside it. Otherwise the eval-141 shortfall is named with its value, and the next levers are named from the measured terms (more training seeds in the ensemble; more fit clips; lever (c), which needs a disk).

## 4. Cost

Three heads read on 443 + 141 clips, about 3 x 3 min on box 2; one job via `chain_rhacc.sh`.

# PREREG_V8F_PROOF_ADD153_AM2 — stage 4 of ACCHIST10K: two more training runs and the four-head ensemble

Frozen after amendment 1's ensemble read was seen and before any run of this amendment exists. Tier **T0**; never driving performance. Card ADD153 (sha256 16c605f6) and amendment 1 (sha256 2474cfeb) stand unchanged for everything they name.

## 0. What is known at this freeze (MEASURED, `rulefc/leadacc/box2_acc_ens/ENS.json`)

`ENS2` (the two `K25D` seeds): held-out A 0.378, `ACCHIST10K-ENS2-CARRIED`; eval-141 A 0.286 (dA_floor +0.075 [+0.015, +0.140]), short of 0.30. `ENS3` (plus the smoother-label head): eval-141 A 0.294. The single heads read 0.260 and 0.289 on eval-141, so each added head has raised the eval-141 read (by +0.026 [−0.002, +0.051] from one head to two).

## 1. The runs (box 2, after card ADD154's job)

Card ADD153's trainer `train_acc_box2.py` (sha256 6f5b0bb4), unchanged:
* `K25D_s150`: arm `K25D`, training seed 150, label run 5.
* `K25DS_s149`: arm `K25D`, training seed 149, label run 10 (the replicate of the smoother-label head).
Each passes card ADD153's controls c1 to c7 and gets its own name by card ADD153 section 5 (beside).

## 2. The read

`ENS4` (of record): the mean of the outputs of `K25D` seeds 148 and 149 and `K25DS` seeds 148 and 149 (both label forms at both seeds). Beside, naming nothing: `ENS5` = `ENS4` plus `K25D_s150`; `ENS2` re-read (it must reproduce amendment 1's held-out S and A to 1e-6, control e1); the paired difference `ENS4` minus `ENS2` (clip-cluster bootstrap, 2,000 resamples, seed 152). The reads, frames and controls e1 to e3 are those of amendment 1, with every checkpoint's sha256 checked against its ACC.json.

## 3. Name and consequence (committed)

`ENS4` is named by card ADD153 section 5 (`ACCHIST10K-ENS4-CARRIED` / `-SPEED-ONLY` / `-NOT-CARRIED`, held-out). If `ENS4` is CARRIED on the held-out clips and reads A >= 0.30 on eval-141 with its eval-141 dA_floor lower bound > 0, then `ENS4`'s acceleration and closing are the inputs of card ADD144's constant-acceleration forecast in a reader-form rule arm (a successor card: build, rollouts on box 2, the PI's go). Otherwise the eval-141 shortfall is named with its value, and the acceleration lane's remaining levers are blocked as named in card ADD153 section 2: more fit clips (none left in this corpus split) and the 8x20 grid (needs a disk for a token store; box 2's D: has 1.93 GB free).

## 4. Cost

Two runs at about 25 min each (`K25D` took 0.04 s per step, 30,000 steps) plus a read of five heads (about 3 min); about 1 h of box 2 in all, one job at a time via `chain_rhacc.sh`.

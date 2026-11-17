# AMEND_HWMPLAN_6 (PRE-REGISTRATION, written 2026-11-17 before any clip is pulled): RA6-L1, the RA readout fit on more TRAIN clips, the AMEND_5 pass otherwise unchanged

Tier T0 (oracle goal): a world-model diagnostic, never driving performance. Evidence classes are stated per number. Dev-box CPU for every compute step; box 2 is read through sftp and runs nothing.
Parents: `AMEND_HWMPLAN_5.md` (sha256 `de6f7477…`) and its read `amend5/readout/RESULT_AMEND5.md`; `amend5/QUESTIONS_AMEND5.md` item P3 (PI approval 2026-11-17).

## 0 What changes (the whole amendment)

| thing | AMEND_5 | AMEND_6 |
|---|---|---|
| RA's fit set | 141 TRAIN clips, 8,306 pairs (`latents_train_R35_s{k}.pt`) | those 141 clips **plus 859 new TRAIN clips** (N = 1,000 clips), fit by the same procedure |
| the readout file | `readout_K10_R35_s{k}.pt`, sha256 `edab7dce…` / `a6d252aa…` | `amend6/readout_in/readout_K10_R35_s{k}.pt`, sha256 recorded by the fit stage in `RA6_FIT.json` and checked by every run |
| everything else | | unchanged: the pass (`ra_l1.py`), the two R35 chassis, the 994 SCORE windows in 71 episodes, A0's banked plans as base (B3), the TRAIN priors (still the 141-clip latents), the targets, HOLD_c, λ = 0, the estimator, the label rules, the families, the inference-seed floors, and the controls of AMEND_5 section 8 |

## 1 Why, from measurements

* **AMEND_5 read (MEASURED, `amend5/readout/RESULT_AMEND5.json`):** RAL1-NARROWS, qualifier MEAN-ONLY, READ-LIMITED.
  * 0-2 s `acc_mae`, RA-L1 − A0: −0.517 [−0.577, −0.456] (s0), −0.375 [−0.434, −0.317] (s1).
  * RA-L1 − HOLD: +0.448 [+0.393, +0.508] / +0.494 [+0.433, +0.554]. RA-L1 − MEAN-L1: −0.0136 / +0.0085, not separated.
  * CEIL-ORACLE failed at both chassis: even the TRUE frame-10 latent, read through RA and inverted through R35, does not beat HOLD.
* **P3's hypothesis:** RA's precision (r 0.47-0.48 against a0's 0.78 on TRAIN, `amend5/ra5_ceiling_s{0,1}.json`) is limited by its 141-clip fit set. HYPOTHESIS.
* **What the 141-clip data say about that, measured for this card (MEASURED, `amend6/work/lcurve141.py`, output `amend6/work/lcurve141_out.txt`; clip-grouped 5-fold, 3 fold seeds, best λ of the grid 1.0 at every size):**

| clips in the fit | held-out R² s0 | held-out R² s1 |
|---|---|---|
| 14 | 0.124 | 0.134 |
| 28 | 0.172 | 0.176 |
| 56 | 0.204 | 0.197 |
| ~113 (4/5 of 141) | 0.205 | 0.201 |

  * The curve is flat from 56 to ~113 clips. This is a plateau at about 0.20 over the measured range. It is not an extrapolation: the planned fit set is 8.8× the largest size measured, which is beyond the 2× rule for extrapolating, so no gain is quoted for N = 1,000.
  * **ESTIMATED expectation: RA6 reads like RA** (label RAL1-NARROWS or NULL, READ-LIMITED, Δ_R not separated). The pull is approved by the PI and costs CPU and LAN time only. Its value is to settle the lever with a measurement, and the card commits what runs next in either case (section 6).

## 2 The fit set and its size

* **Selection rule (committed here, run by `build/select_clips6.py`, output `clips6.json`, sha256 of the clip list `27d2bb2f45d49611aef2ba526bc0937c4d28855f565c16aa8abeaa4010d91649`):**
  * universe: box 2's `cache_full/train/manifest.jsonl` (4,572 clips), sorted by sha12; positions 540:600 are excluded (the 60 RA-1 selection hold-out clips, excluded-list sha12 `f6a1dee0`), which leaves the 4,512 clips M2b trained on;
  * minus the 141 clips of `clips141.json` (the current fit set; sha256 `e9e0e2c4…`);
  * minus every clip of box 2's EVAL manifest (141 episodes, which contains the 71 SCORE episodes) and every eid of the SCORE and TRAIN-selection windows of both chassis. 4,371 clips are eligible (the eval ids removed none: the two caches are disjoint by id);
  * order: `np.random.RandomState(20261117).permutation` of the eligible clip ids sorted ascending; the first 859 are the new fit clips. Every prefix is a seeded random sample.
* **N = 859 new clips, 1,000 in all.** Sizing (ESTIMATED):
  * pairs: 8,306 pairs / 141 clips = 58.9 pairs per clip, so about 50,600 pairs at N = 1,000, a 6.1× larger fit set; 128 dimensions;
  * time: the 141-clip pull took 405 s in total (2.9 s per clip; the last 17 clips ran at 3-18 s per clip, `hwm_plan/collect_R35.log`), so 859 clips take about 0.7 h at the mean and about 3 h at 13 s per clip;
  * reads from box 2: 17.95 GB streamed one clip at a time (20.9 MB per clip), nothing kept on the dev box except the latents (about 0.3 MB per clip for both chassis);
  * 1,000 is the largest size whose worst-case pull stays near 3 h. A larger pull is a named next lever (section 6), not a part of this read.
* **Nested checkpoints (committed):** after 141, 423 and 859 new clips (totals 282, 564, 1,000). The pull writes chunks of 50 new clips, each with its sha256. If fewer than 859 verified clips arrive, the fit set is the largest completed checkpoint of {141, 423} new clips, and every row is labelled with its N. No other N is used and no clip is dropped after a number is seen.

## 3 The pull (study-local; reads box 2, runs nothing there)

* `build/pull_latents6.py`: for each new clip in order, sftp reads `D:/tanitad/cache_full/train/clips/<clip>.pt` into ONE dev-box scratch file, verifies its md5 against box 2's manifest (up to 3 attempts), builds the same 17 windows per clip as `collect_latents_train_R35.py` (the trainer's valid starts, 17 evenly spaced) through the same `wmact_lib.collect_latents`, keeps latents and raw actions only, and writes chunks `lat6/chunk_NNN.pt` plus `lat6/CHUNKS.json` with each chunk's sha256 and clip list. Resumable. Both R35 chassis in one pass.
* Control 4 of the original pull (direct versus stack-loss parity of the model on the first 64 windows) runs once at the start and must read 0.0 at both chassis.

## 4 The fit

* `build/fit_ra6.py` reproduces `amend1_kstep.py`'s frozen-readout procedure exactly: frame table from the windows (`t − 15 + j`, finite-action frames), every 10-step stretch with 11 consecutive frames and finite actions, D = z₁₀ − z₀, y = the mean scaled acceleration over the 10 transitions; fold map `(RandomState(0).permutation(n_clips) % 5)[clip index]` with clips sorted by id; λ chosen on the clip-grouped folds from {1e-3, 1e-2, 1e-1, 1, 10, 100}; ridge on standardised D (sd + 1e-6) with the penalty λ·n; the file layout of `readout_K10_R35_s{k}.pt` (keys tag, K, lambda, RA{mu, sd, W, ym}; the RY entries are not carried because the pass reads RA only).
* **K-control (deliberate regression arm included):** run on the original 141 clips, the procedure must reproduce AMEND_5's RA for both chassis: the same λ (1.0 s0, 0.1 s1), n_pairs 8,306, and (mu, sd, W, ym) within 1e-8. Mutation: the same fit on 140 clips (one clip dropped) must read RED.
* **Beside row FIT-CURVE (never a result, no bar):** clip-grouped held-out R² and Pearson r of the RA fit at the nested totals 141, 282, 564, 1,000, by 5-fold cross-fitting (a held-out fold is never in its fit), per chassis, with clip-cluster bootstrap B 1,000 seed 0.

## 5 Read (AMEND_5 sections 4-5 on the SCORE windows, with RA6 in place of RA)

* Arms, A0 as base, inference seeds 0 and 1 where AMEND_5 had them: **RA6-L1** (committed lever), MEAN-L1, RA6-NEG, RA6-SHUF, RA6-ORACLE, RA-OFF. The refit RA enters ORACLE as AMEND_5's RA did, so the ceiling row is the CEIL-ORACLE row with the refit RA. P's beside RA-L2 is not rerun.
* References read from files that already exist: HOLD (windows), A0 (`score/A0_s{0,1}`), RA-L1 of AMEND_5 (`amend5/score/RAL1_A0_s{0,1}`, sha256 checked).
* Contrasts per band and family, never pooled: RA6-L1 − HOLD, − A0, − MEAN-L1, **− RA-L1 (AMEND_5)**, RA6-SHUF − RA6-L1, RA6-ORACLE − HOLD. Estimator: the paired episode-cluster bootstrap, B 2,000 seed 0, over 994 windows in 71 episodes (it answers another draw of EPISODES). Floors F_I,k as AMEND_5 defines them.
* All four families are reported (STRATEGIC in full; the pause ended 2026-11-16). No outcome is decided on a STRATEGIC or TACTICAL cell.
* **Definitions as AMEND_5 section 6.** Δ_H = RA6-L1 − HOLD, Δ_B = RA6-L1 − A0, Δ_M = RA6-L1 − MEAN-L1, G_x = RA6-L1 − A0 on `xtrack`, D(X) as there. New: **Δ_R = RA6-L1 − RA-L1(AMEND_5)** on 0-2 s `acc_mae`, floor F_I,k = max(|RA6 seed 1 − seed 0|, |RA5 seed 1 − seed 0|). "Separated-better beyond floor" and "WORSE" are AMEND_5's definitions.

## 6 Committed outcomes (both chassis required; one chassis only is SEEDS-DISAGREE and the less favourable label's next step applies)

| outcome | condition | next lever (Rule Zero; started in the same work package unless blocked) |
|---|---|---|
| **R6-CEILING** | Δ_H separated-better beyond floor, G_x not WORSE, D ≥ 0.10 | AMEND_5's CEILING row: RA6-L1 is the candidate level-1 longitudinal cost; the 2-6 s band needs a K = 30 readout, whose windows this pull does not carry (16-frame windows), so it takes its own pull; the T1-kin read is blocked on a goal head (Q2a) |
| **R6-PRECISION** | not CEILING; Δ_R separated-better beyond floor, Δ_B separated-better, G_x not WORSE | fit-set size was a lever. Next: the same pull on the remaining 3,512 eligible clips (about 4-12 h of box-2 reads, one drive, no box-2 process) and AMEND_7's nonlinear readout on whatever N is in hand; CEIL-ORACLE6 decides which of the two the MM runs first |
| **R6-SAME** | Δ_R not separated beyond floor, not WORSE, RA6-L1 not CEILING | fit-set size ELIMINATED as a lever at N = 1,000 (a result, logged). Next, **not blocked, started at once:** **AMEND_7, a nonlinear 2 s acceleration readout** (MLP on [z₀, z₁₀], the shape of the RY head, differentiable so the same pass inverts it) fit on the same N-clip latents, CPU only, zero training of the world model; its FIT-CURVE row is read first and the SCORE read follows only if its held-out R² beats the linear RA's by a separated margin on the same folds. In parallel: R35's response along RA's direction (ACTRSP Option B / action-sensitivity training), **blocked** on box 2 and the PI (Q2b), asked with this read's numbers |
| **R6-WORSE** | Δ_R WORSE at either chassis | C-RA-FIT6, the K-control and λ selection are rerun first (wiring first), then the R6-SAME row |
| **HARM / INVALID** | as AMEND_5 section 6 (Δ_B WORSE or G_x WORSE; any control RED) | as AMEND_5 |

* **CEIL-ORACLE6** (RA6-ORACLE − HOLD separated-better at both chassis) decides the next lever as in AMEND_5: if it holds, the source's 2 s intent is the limit; if it fails, the read or the world model's response is the limit. Expected (ESTIMATED from AMEND_5): fails.
* **A failed bar is reported as FAILED, then its next lever runs.** No arm, clip, constant, λ grid, fold map or criterion changes after any SCORE number is seen.
* **Done** means: a row of this table reached, with its next lever started or, where blocked, named with its blocker and what unblocks it.

## 7 Controls (the read is INVALID if one fails; each has a deliberate-regression arm that must read RED)

* **K-control** (section 4). **C-FIT6:** the refit from the merged latents (141 + new chunks) reproduces the shipped RA6 within 1e-8; n_pairs and n_clips equal `RA6_FIT.json`; 0 fit clips are SCORE episodes. Mutations: one SCORE window's displacement added to the fit goes RED; a SCORE eid planted in the latents goes RED.
* **C-SEL6:** the selection is a prefix-nested seeded sample, disjoint from clips141, the excluded 60, every eval id and every window eid. Mutation: a selection with one eval id planted refuses.
* **C-CHUNK6:** every chunk's sha256 equals `CHUNKS.json`; every new clip's md5 matched box 2's manifest; no clip appears twice. Mutation: a flipped byte in a chunk refuses.
* **AMEND_5's controls** C-RA-SHA (now on RA6's recorded sha256), C-OFF5, C-YAW-ID / C-BAND-ID, C-GRAD, C-PLANT, C-SHUF, C-MEAN, C-L4-RA, C-IN, C-READBACK, C-STRAT, C-TRAIN-ISO re-run on the amend6 folders; AMEND_5's C-RA-FIT is replaced by C-FIT6.
* Tests of `build/test_amend6.py` pin the selection, the chunk merge, the fit (K-control) and each mutation.

## 8 Cost and approvals (ESTIMATED, ±30 %)

| step | where | wall |
|---|---|---|
| pull and encode 859 clips | dev-box CPU (2 threads), sftp reads from box 2 | 0.7-3 h |
| fit (both chassis), K-control, FIT-CURVE | dev-box CPU | under 10 min |
| section 7 controls (A0 only), TRAIN stage, SCORE arms (RA6-L1, MEAN-L1, NEG, SHUF, ORACLE, OFF) | dev-box CPU | about 0.7 h |
| contrasts (B 2,000), families, readout | dev-box CPU | about 1.1 h |

* Basis: AMEND_5's own stage times (`amend5/read.log`): controls 29 min on three bases, TRAIN arms 14 min, SCORE arms 20 min, readout 68 min; this read runs one base and skips AMEND_5's five test suites. About 1.8 h in all after the pull.
* Disk: latents about 0.3 GB on F:, one 21 MB scratch clip. No box-2 process, no GPU, no training, no spend. The PI approved P3 on 2026-11-17. Box 2's `D:` is a bus-powered USB drive that blue-screened under load on 2026-10-17; the pull is a sequential read of one 21 MB file at a time, as the 141-clip pull was.

## 9 Unchanged

Everything not named in section 0.

## Log

* 2026-11-17: written by the Arch+Inference build agent for the MM. `clips6.json` and its sha256 were produced from the two box-2 manifests before this card was hashed; no clip file has been read.

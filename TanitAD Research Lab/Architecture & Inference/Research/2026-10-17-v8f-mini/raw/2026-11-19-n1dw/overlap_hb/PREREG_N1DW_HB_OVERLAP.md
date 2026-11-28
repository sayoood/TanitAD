# PREREG_N1DW_HB_OVERLAP — the overlap read of the T1-kin executor for the v8f-mini line (N1DW D1-HO)

**Frozen 2026-11-20, before any hb0.2 dump of these arms exists.** Basis: the PI ruling on decision card D16, recommendation D
(*"Approve all four"*, confirmed to the MM first-hand: *"Yes, all four"*). D asks for **one overlap read with the old and the new
executor**, after which the bounded executor is the tier of record. The coordinator's card ADD68 is the overlap read for the 10k line.
This card is the one for the v8f-mini line.

## 1. What changes, and what does not

* **Executor:** `stack/scripts/v8f_t1_eval_hb.py --kappa-max 0.2` (landed at bcf267ce) wrapped around each arm's own banked dump
  tool: `v8f_t1_eval_rb` for N1R and `v8f_t1_eval_rbd` for N1D and N1DW. Each tool comes from its own pod tree, pinned tool sha256 180044eb.
* **Everything else is the banked dump's own argument list:** grid `eval_grid_x_feasible.json` (sha256 ca289d0f), join, tracks, text-nav,
  nav-rb tables, m = 10 causal bucket, CPU with 2 threads, `/root/venv`. `hb_dump.sh` carries the three family scripts with only the
  wrapper prefix, the output root and the N1DW checkpoint path changed. The N1DW checkpoints are the dev-box copies, sha256-checked
  against the pod's own record.
* **Reader:** the banked `n1dw_ho_read.py` (D1-HO, 5,736 windows, 497 held-out episodes, paired episode-cluster bootstrap B 2,000,
  seed 0, answering "another draw of EPISODES" only), imported unchanged by `n1dw_ho_read_hb.py`. Only three things change: the dump paths,
  K-CLIP (training logs do not depend on the executor, so they are not re-read) and the output folder.

## 2. Arms

14 dumps at kappa 0.2: N1R_s0, N1R_s1, N1D_s0, N1D_s1 and their zero twins, N1DW2_s0, N1DW2_s1, N1DW4_s0, N1DW4_s1 and their zero
twins. That is the full arm set of the D1-HO read.

## 3. Controls (each must read its value, or the hb read is not reported as a result)

* **K-HB0:** N1R_s0 run through the wrapper with `--kappa-max none` must equal the banked T1-kin/old dump **bit for bit**: the same
  ep*.npz files, the same keys, and every array equal. This proves that the wrapper path and the environment reproduce the old executor.
  On a FAIL the chain stops and no hb read is written.
* **K1** (self-pair exactly 0) and **K-CONST** (0.338215), from the reader as banked.
* **K-RE is expected to differ:** it compares the reader with RESULT_NAVRBD.json, which was read under T1-kin/old. Under hb0.2 that
  equality cannot hold by construction. It is reported with this note and does not count as a control problem. K-HB0 checks instead the
  equality the executor change keeps.
* **Dump count:** 14 DONE and 0 FAIL.

## 4. What is reported, per family, side by side with the banked T1-kin/old read

* **TACTICAL:** the deciding cells N1DW − N1R TAC-LAT at both weights and both seeds, with bars, plus beside N1DW − N1D, TAC-LON
  and the D2 confusion.
* **LONGITUDINAL and LATERAL:** every R1 cell (0-2 s and 2-6 s bands; speed, along-track, acceleration, cross-track, heading, yaw rate,
  curvature). Distance-keeping is NOT-COMPUTABLE with n = 0, the same as in the old read: no lead block exists for the held-out clips.
* **STRATEGIC:** NOT-COMPUTABLE in this read, because the D1-HO read carries no strategic cells. STRATEGIC is read on eval-141.
  The eval-141 guard stays on its pre-registered executor, T1-kin/old.
* **Z-USE**, each arm's TAC-LAT level under both executors, and the outcome labels of the D1-HO read under both executors.

## 5. Outcomes, committed in advance

The verdict of record of card PREREG_V8F_N1DW stays on **T1-kin/old**, the executor it was frozen with. This read is reported beside it.

* **HB-GAIN-ROBUST:** every arm whose T1-kin/old deciding row reads `gain` also reads `gain` under hb0.2, and no arm reads `cost`.
  ⇒ the D1-HO reading holds under both executors.
* **HB-GAIN-EXECUTOR-DEPENDENT:** at least one T1-kin/old gain arm does not read `gain` under hb0.2. ⇒ The verdict of record is reported
  unchanged, with "executor-dependent at <arm>" beside it. The PI ask about s10/s11 carries that statement, and the next card at that
  weight is frozen under T1-kin/hb0.2.
* **HB-NEW-GAIN:** an arm that is not a gain under T1-kin/old reads `gain` under hb0.2. ⇒ It is reported beside and does not change the
  verdict of record. The next card is frozen under the tier of record.
* An arm that reads `cost` under hb0.2 is reported in every case, as a cost under the tier of record.

## 6. Compute and place

The 14 dumps take about 10-15 CPU-min each, 3 at a time, so about 1 h in all, plus about 2 min for the read. They run on the LoRA pod
after the ADD45 GPU chain has ended, **under the pod's GPU queue lock**, so no training runs on the pod during the read. No GPU use,
no training.

## 7. Frozen kit (sha256 in `KIT_SHA256.txt` beside this card)

`hb_dump.sh`, `khb0_compare.py`, `n1dw_ho_read_hb.py`, `mm_hb_overlap.sh`; the wrapper `stack/scripts/v8f_t1_eval_hb.py` at bcf267ce.

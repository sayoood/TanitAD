# PREREG_NATIVE40: T0 read of the native lead head (LLR_s0, LLR_s1), frozen before any data of this read

**Status:** frozen by the Master Mind (MM) on 2026-11-19, before the wrapper exists and before any number of this read is seen.
**Decision rule source:** the v8f coordinator (v8proof fork), in a cross-session message on 2026-11-19, under the PI's delegation of v8f coordination (ruling 2026-11-19). It is direction only and approves no spend. The read is CPU only and trains nothing.

## 1 Why
* The integrated candidate, stage 1 (base: the 648d4ca3 line), uses the native lead head of `stack/tanitad/models/v8f_mini.py` (`_lead`, :917-925) if it reads the lead well enough. Otherwise it uses the coordinator's 5-frame lead-supervised head (LEADSUP), which would need a port.
* The native head's gap error was never read. The LLR read (H-LEVELHEAD-2, `raw/2026-11-05-levelhead-llr-result/llr_read/`) reports only output means (`presence_prob_mean` 0.3046, `gap_hat_mean` 0.6084, `LLR_READOUT_s0.json`).

## 2 What is read
* **Arms:** LLR_s0 and LLR_s1, the native lead head on the corpus (`--w-lead 3.362 --aux-cond token`), at their banked checkpoints, loaded with their code of record.
* **Head outputs** (648d4ca3, `v8f_mini.py:1034-1035`):
  - presence `p = sigmoid(lead_logit)`;
  - gap in metres `g = lead_gap * 40` (`LEAD_RANGE_M = 40.0`, `v8f_mini.py:148`).
  - Both are [B], read at the window's frame `last`.
  - They are added to the dump by a study wrapper around the sha-pinned `v8f_t1_eval.py`. The pinned file is not edited.
* **Tier T0:** the head reads logged frames; nothing is rolled out. This is a head diagnostic, not a driving claim.
* **Gap windows:** the coordinator's 534 reader windows of card ADD36 (`LS_REF40_WINDOWS.json`: episode id, raw start frame, tick, label gap; its sha256 is recorded when read).
  - All 534 carry a label lead within 40 m.
  - If our 141 evaluation episodes are not the same set, the read says so and reports on the matching windows only, with the n.
* **Presence windows:** every evaluation window of the same 141 episodes and the same tick convention that has a lead label (`has`).
  - Positive: a label lead within 40 m (`lead == 1`, finite gap, gap <= 40).
  - Negative: every other window with a label.
  - The n of each class is reported.

## 3 Quantities, per seed
* Gap: MAE in metres, with the head read clipped to [0, 40] m; R2 against the label gap; n.
* Presence: AUROC of `p`, positive against negative as above; n per class.
* **Interval:** paired episode-cluster bootstrap over episodes, B 2,000, seed 0. It answers "another draw of EPISODES" only, not another training run or another inference run.

## 4 Rule (fixed before data)
* **Reference** (INHERITED from the coordinator's lineage, not re-verified by the MM; used only as the bar): mean gap MAE of eight lead-head dumps on the 534 windows = 5.04 m (`LS_REF40.json`, sha256 baf652a5..., T0, clipped at 40 m; per arm 4.85 to 5.59 m; R2 0.50 to 0.58).
* **NATIVE-OK:** (i) the mean over the two seeds of the native head's gap MAE is at most **5.80 m** (1.15 x 5.04), AND (ii) the presence AUROC is at least **0.85 at both seeds**. Then the native head is the stage-1 head.
* **NATIVE-GAP-FAIL:** (i) fails. **NATIVE-PRES-FAIL:** (ii) fails at either seed. On either, the 5-frame LEADSUP head is ported before any 10k arm with a lead head.
* **NOT-COMPUTABLE:** the head outputs are absent, or the windows cannot be matched. The reason and the n are stated.
* Bars, windows and labels do not change after any number of this read is seen.

## 5 Controls (each must read as stated, or the read is not admissible)
* **Shuffle arm:** the head's gap read permuted across windows within a seed. Its MAE must exceed the unshuffled MAE, and its R2 must be at most 0. The presence logits permuted the same way must give an AUROC within 0.45 to 0.55.
* **Units arm:** the gap read scaled by 100 instead of 40 must change the MAE. A normalisation slip of exactly this kind happened in an MM message on 2026-11-19.
* **Label arm:** the label gap fed in as the "head read" must give MAE 0 and R2 1 on the gap windows.

## 6 Families
This read decides a head choice only. It is not a driving evaluation, so no LONGITUDINAL, LATERAL, TACTICAL or STRATEGIC result is claimed from it. The four families are read on the integrated candidate's own arms.

## 7 Cost
CPU only: two checkpoints, teacher-forced head read on the evaluation clips. ESTIMATED at under 2 h of dev-box CPU. No GPU, no training, no spend.

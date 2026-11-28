# PREREG_V8F_PROOF_ADD35_N1 : LEADCOND, what the dry gates found before any arm trained

Note to card `PREREG_V8F_PROOF_ADD35.md` (sha256 92ed08fc13998d541f719c4e481e9fade8a3e9e109953c32a84055547a593720). Written after the dry gates of LFT_s0 and LC_s0 and before any training step, rollout or read of an arm of the card. The card stays as written; this note records two findings of the gates and what is read because of them.

## 1. The gate refused LC_s0 once (MEASURED, `leadcond/ops/pod_leadcond_gate.out`)

* 13:02Z: the selfcheck of LC_s0 stopped with a device error inside the new live probe of the lever (the probe copied the model on the GPU and passed it a batch on the CPU; the build had been tested on the CPU only). LFT_s0 passed. Nothing trained.
* Change: one line in `lead_cond_probe` (the copy of the model goes to the CPU, as in the other probes of the trainer). Trainer `v8f_mini_train.py` sha256 d45621201f4a5bc95df0d5ef2be7c07587d208a752e81f9abb1e935a84e22f9e (the build named in section 2 of the card) to 85035ca4a3e9545659e1e6ccaa24c74d3386b9b83a889bb7df490373b721d519. Model file and test file unchanged. `PINS.txt`, `ARMS_LEADCOND.txt` and `LEADCOND.patch` made again (the earlier ones kept as `*_before_fix1*`), patch replay equal on the three files, test list again 159 passed (`leadcond/tests_fix1.log`).
* Pod tree rebuilt 13:05Z: 18 pins equal, arms file of chain A sha256 77aa05d19bbced352cbfed0be651db800ba1253ffab02c0e93c53d717e8ee469. Dry gates 13:05Z to 13:08Z: LFT_s0 PASS, LC_s0 PASS, token kind HIERFT6 accepted and kind HIERFT refused (`leadcond/ops/pod_leadcond_gate_fix1.out`). Live probe of LC_s0 on one real batch: 63 of 64 rows with a full lead window, presence mean 0.44, gap output mean 0.77 (in units of 50 m), gradient on the layer 2.45, zero-u control exactly 0.0, gradient into the lead head through u exactly 0.0.

## 2. Control q3 as written is NOT MET, and why (MEASURED, `leadcond/ops/pod_leadcond_q3.out`)

* q3 of the card: "the live total of the dry gate of LC_s0 equals that of LFT_s0 (relative difference at most 1e-6)". Read: 14.7784796 (LFT_s0) against 14.78508 (LC_s0), relative difference 4.47e-4. Not met.
* Floor: a second dry gate of LFT_s0 with the same argv gives the same live record in all 35 numeric fields (largest difference 0.0). The record is repeatable, so the difference is real.
* Cause, from the source (`live_check`, trainer lines 1380 to 1397): the live record holds the loss terms of the THIRD of three optimiser steps at learning rate 1e-3 on a copy of the model, not of step 0. The new layer has a gradient above 0 at the first step (control q2 asks for exactly that), so by the third step it is not zero any more and the terms differ. The card named the wrong field: a third-step value cannot show an identity at step 0. This is an error in the card's control, found before any arm trained; it is reported as not met in every row of this card.
* The one field of the same record that is a step-0 value, the gradient sum on the operative head at the first step (`head_o_grad_step0`), is equal in both selfchecks to every printed digit: 277.45855885744095 and 277.45855885744095.

## 3. What is read for the identity at step 0 on real data (q3b, fixed here)

* q3b-i (read above): `head_o_grad_step0` of the dry gates of LFT_s0 and LC_s0 exactly equal.
* q3b-ii (before the read of any arm): the checkpoint LS_s0 with the layer built and both of its tensors zero is rolled with the pinned T1-kin tool of the tree, and so is the unchanged checkpoint LS_s0, in the same pod script; every array of the two dumps is compared. Bar: largest absolute difference 0.0. If two dumps of the unchanged checkpoint made the same way differ from each other, that difference is stated and the bar is "not above it".
* The names of the card are given only if q1, q2, q3b and q4 to q7 hold; q3 as written is stated as not met, with this cause, beside every name.

## 4. Unchanged

Arms, the rule of section 4, reads, outcome names, bars, what follows and limits of the card. Chain A (LFT, LC, LFTb) starts after this note is hashed.

# PREREG_V8F_PROOF addendum 4: BIASEDIT, the stop-weighting lever written as a one-scalar weight edit (frozen before its data; true freeze time = this file's modification time, see TIMES_NOTE.md)

Trigger, MEASURED (`wsp/TASKVEC.md`, weight-space read of banked checkpoints): TASKVEC P2a read DIFFUSE; every group's lever displacement is below the seed displacement (all trainable 0.423), and the only tensors where the dose-2 lever clearly exceeds seed noise are two head biases: `h_lon.bias` 3.81 and `h_goal.bias` 1.55. Theorem T4 (VC-E3) says class weighting is a prior change, and a prior change is a logit offset. So the lever should be reproducible by adding the offset to the bias, with no training.

## The edit (weight-space programming: one scalar, no gradient step)
* GOALEDIT(b): theta = BAL-NULL_s{0,1} with b added to the STOP_POINT component of `h_goal.bias`; every other tensor bit-identical. b1 = ln(W1), b2 = ln(W2), where W1 and W2 are the stop goal positive weights of the trained arms BAL-G and BAL-G2, read from their banked argv (quote them).
* Controls that must read known values: (e1) the edited checkpoint differs from its parent in exactly one scalar; (e2) the stop-goal AUROC of the edited model on the STOPDEC windows equals the parent's to 1e-9 (a constant offset keeps the ranking); (e3) b = 0 reproduces the parent dump bit-for-bit.

## Reads (same readers, windows and thread count as the WSP job; tier T1-kin first tick, a diagnostic)
* STOPDEC hit rate and false-positive rate at p >= 0.5 for GOALEDIT(b1), GOALEDIT(b2), beside the trained BAL-G and BAL-G2 and the parent, at both seeds.
* T1-kin four families for GOALEDIT(b2) against its parent and against trained BAL-G2 (paired episode-cluster bootstrap, B 2,000): stop-window 2-6 s ADE, 6 s ADE, speed error; lateral; tactical lon macro-F1; STRATEGIC PAUSED (PI item 25).
* Also report b_match: the offset at which the edited parent reads the trained arm's false-positive rate, and the hit rate there.

## Prediction and outcomes
* T4 predicts hit and false-positive rate of GOALEDIT(b_j) equal to the trained dose j.
* BIAS-EQUALS-DOSE: at both doses and both seeds, hit and false-positive rate of the edit are within the replicate gap |NULL_rep - NULL_s0| of the trained arm on that metric.
* BIAS-MATCHES-ROC: not equal at b_j, but at b_match the hit rate is within the replicate gap of the trained arm (same ROC, a different offset than ln W).
* BIAS-DIFFERS: at b_match the trained arm's hit rate is outside the replicate gap (training with weights moved the ROC, not only the threshold).
* Next lever, committed: EQUALS or MATCHES -> the weighting line is a threshold move in fact; a dose is a scalar edit, and the remaining lever is the cue (theorem T5). DIFFERS -> read which tensors beyond the bias carry the difference (TASKVEC per tensor) and test the lon head the same way.
* Coordination: the threshold card named STOPTAU belongs to the box-2 session. This card edits a weight and reads the plan as well; it does not read a threshold sweep on reads.

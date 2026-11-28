# Note N1 to PREREG_V8F_PROOF_ADD11 (HIERREP): three choices fixed before any arm is trained

Written 2026-11-13 after the pod data stage stopped at its first gate and before the token cache exists; file time and sha256 in `TIMES_NOTE.md`. The card itself is unchanged (sha256 94647519...). No bar, arm, seed, metric or outcome of the card moves.

## N1a. The file-hash gate of the frame build is waived and replaced by a measured control (r6)
* Fact on this pod (MEASURED, `gate_result.json`): the three gate clips (8f387977, bf5ec76e, dca157a0) built on the pod do not hash-equal the banked frame files (0 of 3).
* Fact on the earlier pod (INHERITED for this pod, `raw/2026-11-06-lora-pod-amend4/HANDOFF_LORAPOD0.md` line 18, same build route): frames differ in 1.377 % of values by exactly +/- 1 level, in all 201 frames; actions, poses and frame geometry equal. A banked 256x640 frame file of these clips is not on HF and was not found on the dev box, so the pixel comparison could not be repeated here.
* The card does not rest on frame identity: it states that the pod cache is a new, internally matched set. The waiver is recorded on the pod (`gate_waiver.json`) with this note as its reason.
* Replacement, measured on this pod before any arm trains, control **r6**: the pod-built eval token cache against the box-2-equal eval cache already on the pod, on all 141 eval clips. Bars: clip sets equal; on every clip the kinematic fields (speed, acceleration, yaw rate, pose grid, kin3) equal within 1e-4; on every clip the pooled 4x10 tokens within rel-L2 5e-2 (the earlier pod read about 2.4e-2). A miss stops the chain before the first gate. A wrong clip, a wrong timing or a wrong rectification would show in r6; a +/- 1 level difference would not, and is accepted.

## N1b. Every arm of the card is evaluated on the pod-built eval cache
The arms train on pod-built tokens, so their T1-kin dumps read the pod-built eval tokens (same pipeline on both sides). Control r1 (harness reproduction of a box-2-trained checkpoint) keeps the box-2-equal eval cache. Windows, labels, join and grid are the same files in both cases.

## N1c. Three label files are named in the argv and are not on the pod
`v9_labels_train.npz`, `lead_train4572.npz` and `refcv6_speed_max_v8_train.jsonl` exist only on box 2. The trainer does not open them while `--join-train` exists (the earlier pod ran the same argv that way; the banked join sha256 978f41bf is shipped and checked). The chain records them as named only. If a selfcheck gate shows the trainer does open one, the chain stops there and the files are requested from the session that owns box 2; nothing is substituted.

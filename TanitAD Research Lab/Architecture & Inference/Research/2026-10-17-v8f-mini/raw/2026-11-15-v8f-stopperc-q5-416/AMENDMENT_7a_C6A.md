# AMENDMENT 7a to STOPPERC Amendment 7 (Master Mind, 2026-11-15, written after the 3-clip smoke and before any 416x1024 token is encoded for the probe or scored)

**What changes.** Control C6a only. Amendment 7 set: the pod-built 416 eval v2ep against E:'s `v2ep-eval139-416x1024cyl`, decoded uint8, all 139 clips, max |diff| 0.

**New C6a.** On the same comparison, over all 139 clips:
* max |diff| ≤ 1 count, and the fraction of values that differ is reported;
* n_frames, poses, actions and the frame record are bitwise equal (as C6c).
A difference above 1 count on any value refuses the read.

**Why (MEASURED, builder smoke, `build/smoke_c6a.json` and the hand-back):**
* On 3 eval clips (01be5919, 060531e1, 0fa2b231; frames 0 / 2 / 100 / 200 inspected on 01be5919), every checked frame differs from E:'s copy.
* The max |diff| is exactly 1 count, on about 1 % of values.
* poses, actions, n_frames and the frame record are bitwise equal. The mp4 source (same HF revision) and the builder are the same.
* HYPOTHESIS, not probed: float rounding of `cylindrical_rectify` between the machine that built E:'s copy and the pod.

**Why this does not touch the read:**
* E:'s eval139 copy is an input to no arm: Amendment 7 §2.1 builds every 416 frame, train and eval, on the pod (one route).
* The 256p comparator frames are also the pod's.
* C6a was a cross-check that the pod's route sees the same video as the existing copy, and a 1-count rounding term on 1 % of values answers that.
* No stop-cue number, probe fit or 416 token exists at the time of this amendment.

**Unchanged:** every other control (C6b ≤ 1.0/255 MAE: 0.931 on the 3 clips, C6c bitwise, C7, C8 bitwise, C0-C5, C9, C10), the outcomes, the samples and the estimator.

**Beside:** the builder runs S1 ∪ S2's clips in one pass (4,458 clips; S1 4,265, sha 843e4017 as frozen; S2 4,256, sha 32ad96be), so S2 needs no second frame build. The samples are unchanged.

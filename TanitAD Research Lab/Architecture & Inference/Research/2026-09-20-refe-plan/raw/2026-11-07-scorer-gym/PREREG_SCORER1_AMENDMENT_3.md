# AMENDMENT 3 to PREREG_SCORER1 (2026-11-08 19:45Z, Master Mind): the PI's go for S1L, and the GPU's name

**Written before any arm trained and before Stage 0 was read. No bar, arm definition, estimator or endpoint changes.**

## 1. The PI's decision on S1L
Amendment 2 says S1L needs the PI's go. The PI's words, 2026-11-08, in reply to the recommendation below: *"Follow your recommendation, think of l40 not a40 in our pod"*.

The recommendation was: approve the extra compute on condition that the Linux relabel passes when P7 ends; if it fails, the arm does not run.

So S1L starts only if all of these hold:
1. The lane-truth control is relabelled on the Linux machine class after P7 ends (about 15 CPU-min), and **G-L1 passes as written** (99 % of proposals within 1 mm).
2. That same pipeline supplies the E2 lane truth (Amendment 2, condition E2).
3. It is established that u400 was trained on the banked SFT-4 lane labels (Amendment 2, condition Labels).
4. The S1L code and gate work in Amendment 2 is done, and the launch gate passes.

If condition 1 fails, S1L does not run. The reason is reported, and the read stays CONDITIONAL. S1 runs regardless, as already granted.

## 2. The GPU is an L40
The pod's GPU is an **NVIDIA L40** (MEASURED `nvidia-smi`, 2026-11-08: 46,068 MiB, 300 W). The compute figures in this pre-registration are written in A40-hours because their dose came from RL-S1b on the earlier A40 pod.

The frozen text is left as written. Read those figures as ESTIMATED doses:

| | A40-h, as written | L40-h, ESTIMATED |
|---|---|---|
| one arm | 3.5 | about 3.0 |
| Stage 1, 3 arms | 10.5 | about 9.1 |
| with S1L | 14.0 | about 12.1 |

The conversion uses P6's measured rate: 50.5 s/step on the L40 (n = 8 steps) against 58.5 s/step on the A40, so 13.7 % less time. That is a P6-training measurement, and whether it carries over to the scorer fine-tune is not measured. The first SCORER1 run's own step time replaces this estimate.

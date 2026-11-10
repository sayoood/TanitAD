# VERDICT: PREREAD_V8F_STOPCUE = SEEDS-DISAGREE (s0 PARTIAL, s1 CUE-PRESENT) (Master Mind, 2026-11-10)

Card frozen 16:36:54 (card 847bc5e4, script 8f223a68; pre-registered in 3d199138 before the run), read 16:37:55-16:38:17, `out/STOPCUE.json`. Inputs: H-V8F-STOPDEC-1's npz (sha256 04872720... s0, 3db98159... s1). Tier: T1-kin first tick, eval-141; a decision-head diagnostic, **never driving performance**. Estimator: episode-cluster bootstrap, B 2,000, seed 0, 139 episodes ("another draw of episodes" only). STRATEGIC: **PAUSED** (PI item 25). All numbers MEASURED.

Controls: self-test AUROC 1.0 (planted perfect), 0.5 (constant), 0.0 (inverted); goal_prob in [0, 1]; `has` 2,365 per seed (= STOPDEC); positives 131 per seed (>= 50; GT goal includes STOP_POINT, with or without a traffic-light goal; prevalence 0.055).

| seed | AUROC p(STOP_POINT) (PRIMARY) | class | average precision | recall at an eval-chosen 5 % false-positive rate | AUROC lon STOP vs regime STOP | AUROC lon slow vs regime STOP or DECEL |
|---|---|---|---|---|---|---|
| s0 | **0.836** [0.784, 0.886] | PARTIAL (< 0.85) | 0.295 [0.161, 0.432] | 0.321 (threshold 0.142) | 0.836 [0.768, 0.897] | 0.713 [0.655, 0.771] |
| s1 | **0.857** [0.809, 0.903] | CUE-PRESENT | 0.298 [0.166, 0.448] | 0.321 (threshold 0.214) | 0.809 [0.738, 0.876] | 0.710 [0.650, 0.769] |

**Outcome: SEEDS-DISAGREE** (the rule needs both seeds in one class). It is not reported as CUE-PRESENT.
* Neither seed is CUE-ABSENT (both far above 0.70): T's features rank stop-point windows above the rest. Its 0.5 operating point is far too conservative for a 5.5 % class (recall 0.10 / 0.13 at 0.5, STOPDEC), yet even an eval-chosen threshold at 5 % false positives recovers only 0.32 of stops, and the average precision is 0.30. The cue is present but weak.
* The two seed classes' triggers share their first card: PARTIAL -> "both, the rebalancing first"; CUE-PRESENT -> rebalancing. The Master Mind takes the rebalancing card next on that overlap (stated here as a reading of the two triggers, not as the read's outcome), and keeps the perception lever (LoRA arm L's read adds this AUROC) behind it.

## Next lever and its blockers (Rule Zero)
* **STOPBAL (training card):** a positive weight on T's STOP_POINT goal BCE (and, as a separate arm, a class weight on the lon STOP class), DAG-OACC recipe otherwise, seeds 0 and 1, read with the STOPDEC / STOPCUE instruments as non-regression controls plus the four families. No such lever exists in the trainer (`v8f_mini_train.py`, `v8f_mini.py` searched; the nearest pattern is `launch_intent`, `v8f_mini.py:460`).
* **Blocked on:** (1) landing: the lever lives in `stack/tanitad/models/v8f_mini.py`, and a peer session holds all `stack/tanitad` lands on this branch until its K2M1N20 launch; (2) GPU: box 2 runs EGOWM then OBD until about 06:00 (PI go 2026-11-10), the pod runs LoRA L_s0 for about 74 h more. The next box-2 slot is the PI's call.
* **Done now:** the lever and its tests are built in an export tree, off the shared checkout, unlanded, with the card drafted for the Master Mind to freeze.

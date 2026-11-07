# THEORY_CRAWL, Amendment 2: P1′ was NOT EVALUABLE; it is re-read UNCHANGED on new held-out windows (Master Mind, 2026-11-07, written before any window of the new set is scored)

## 1. What happened ([M], `wsp_read/P1P_RESULT.md`, spec `73640c67`, stamped before any eval posterior)
* **NOT EVALUABLE as pre-registered.** After the filter there are 168 eval-141 windows from 45 episodes. CONFIDENT (posterior ≥ 0.8) holds 24 windows at both seeds, below the floor of 30.
* **Descriptive only:** Δ median ρ is +0.365 / +0.437, in the predicted direction. These 24 windows are SEEN, so no criterion is changed on them.
* **The probe is not at chance on eval-141:** P-MOVE V+E, AUC 0.750 [0.673, 0.825].

## 2. P1′b (pre-registered now)
* **Exactly P1′'s rule** (`THEORY_CRAWL_AMEND_1.md` §4, frozen `9d39ac53`): the same filter, ρ, thresholds (0.8; [0.35, 0.65]), floor 30, estimator and verdict words, with the probe refit of `P1P_PROBE_MODELS.npz`.
* **The new window set** is the RA-1 holdout clips: the clips withheld from the base's training. Every window passing the filter is read: tick 0, C0 s0 and s1, a CPU forward.
* **Admissibility of the set, checked BEFORE any score, else STOP:**
  * no holdout clip is in C0's training clip list;
  * no holdout clip is in the go/stay probe's training windows.
* **The verdict is read on the holdout set alone.**
* **Reported beside, deciding nothing:** the pooled read (eval-141 ∪ holdout), since eval-141's windows are seen.
* **NOT EVALUABLE again** (a group below 30) closes P1′ as untestable at this data scale. The next test of T1′ is then P4 on the LoRA arm.

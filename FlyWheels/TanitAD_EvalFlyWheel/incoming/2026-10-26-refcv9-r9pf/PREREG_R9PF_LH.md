# PREREG_R9PF_LH -- R9-PF's speed profiles driven by LONHEAD-C's clean-split decision (Master Mind, sealed 2026-10-26 16:02:32 Berlin)

## 0. Why (disclosed; MEASURED on eval139, T1 self-action open loop, paired episode-cluster bootstrap over EPISODES)
* **LONHEAD-C (FAIL_F, `refcv9_lonhead/result/LONHEAD_VERDICT.json`):** the clean-split head LH makes better decel DECISIONS (T0 decel AUC
  0.7305 [0.6749, 0.7855] vs the native head's 0.6851; decel label on GT-decel windows 0.496 vs 0.416, on GT-cruise 0.129 vs 0.136), but the
  SELFSHARP label->plan transfer leaks: LHS_FACT8 FD +0.103 above H2CN (separated), F1 / F2 FAIL.
* **R9-PF (provisional, `refcv9_r9pf/`, K-OFF / K-EQ pending):** its own decision fails (PA-1a / 1b / G / R), but its PROFILES transfer a
  decision to a plan well: EXPLORATORY, with the GT regime and R9's own profiles, 0 rows worse than H2CN_DEPLOY8 and S6 ADE -0.40 m; a
  SOFT (posterior-weighted) decode cut the PA-G rows 13 -> 1 but kept FD +0.026 above H2CN (separated).
* ⇒ The two failures are complementary: a better DECISION (LH) feeding the better TRANSFER (R9 profiles). Both parts are already FROZEN
  and REGISTERED (`REGISTERED_LH.sha256` 14:00:37; `REGISTERED_R9PF_PA.sha256` 14:08:29). Nothing is refitted.
* **Forking-paths disclosure.** This arm was designed AFTER reading R9-PF's provisional bars and its EXPLORATORY ORACLE / SOFT reads, and
  LONHEAD-C's verdict, all on eval139. It is a further eval139 look in the progress-decision family. The multiplicity is paid below (s3);
  the SOFT arm in particular is motivated by an exploratory eval read.

## 1. Arms (offline composition on the BANKED fb47a58 paths, exactly as PREREG_REFCV9_INMODEL_PA + AMENDMENT1 s4)
| arm | decision | profiles | role |
|---|---|---|---|
| **R9LH-ARG** | LH LON6 argmax, grouped to regimes by the IDENTICAL map R9PF-TL uses for the trunk lon head | registered R9PF heads, fit seeds 0 / 1 / 2 | primary |
| **R9LH-SOFT** | LH LON6 softmax summed per regime (same map) -> posterior-weighted profile, as R9-PF's exploratory SOFT decode | same | primary |
| R9DER-ARG | DER (LONHEAD-C's deranged control head) argmax, same map | same | **K-DER**: must NOT pass PA-1a (attribution: the decision information, not the profiles, moves MD) |
| beside | R9PF (own decision), R9PF-TL, ORACLE (GT regime), H2CN_DEPLOY8, FACTORED_DEPLOY8, RETIME_DEPLOY8, LHS_FACT8 | | references |

LH reads z_tac + v0 at t0 (vision-only + admissible ego; z_tac is inference-seed-independent, MEASURED bit-equal); it is the tactical
decision, not the situation classifier, and it feeds the speed profile only (the path stays the planner's selected candidate).

## 2. Controls (a failed control voids the bars)
* Every pin in `REGISTERED_LH.sha256` and `REGISTERED_R9PF_PA.sha256` + `FIX_R9_RNG.sha256` re-hashes at start.
* **K-MAP:** the regime map is read from the R9PF-TL code path, not re-typed; R9LH-ARG with LH replaced by the trunk lon head reproduces
  R9PF-TL's composed plans bit for bit (both inference seeds).
* **K-SOFT1:** R9LH-SOFT with a one-hot posterior reproduces R9LH-ARG bit for bit.
* **K-DER** (above); and R9-PF's own K-EQ / K-OFF must have PASSED in its final verdict, or this arm is VOID.

## 3. Bars (S6 compliant unless stated; paired vs the reference on the SAME draw; B 2000 seed 0; both inference seeds)
PREREG_REFCV9_INMODEL_PA s4 **PA-1a, PA-1b, PA-2, PA-G** unchanged, and **PA-R** = PA-1a / 1b / 2 hold at R9 profile fit seeds 0, 1 AND 2.
**Multiplicity:** two primary arms on a further look -> PA-1a and PA-1b separations are read on the **97.5 %** interval; "not separated
above" clauses on the 95 % interval (the stricter reading of each). An arm PASSES iff all five bars hold at both inference seeds.

## 4. Tier / estimator
T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); paired episode-cluster bootstrap over EPISODES (blind to
training and inference variance); one refcv8 training seed; one LH fit; R9 profile fit seeds 0 / 1 / 2 are the training-variance read;
inference seeds never pooled. Four families reported per family.

## 5. Committed outcomes
* **PASS (either arm)** -> the first post-hoc composition beyond the MD / FD frontier: (i) refcv8 candidate -- a SHIP-R8 amendment needs a
  PI go; (ii) refcv9 design = R9 profiles + a clean-split decision head (the Phase-B request carries this number).
* **PA-1a PASS, PA-1b FAIL** (or the reverse) -> a new operating point, not a frontier move; next = a decision head with richer supervision
  (the regime itself as the target on the fresh split, LH-architecture), pre-registered first.
* **Both FAIL** -> the decision information reachable from frozen z_tac (AUC ~0.73) is not enough for this transfer; the lever is training
  the decision INTO the trunk (Phase B, PI compute, D-R9-INMODEL Q3).
* **K-DER FAIL** (the deranged decision passes PA-1a) -> the bar is passable without decision information: VOID.
No further relabel / decode variant of this family is admissible on eval139 after this read.

# QUESTIONS_416: what only the PI can decide for Q5 (416x1024)

Context: `DRAFT_AMENDMENT_7_Q5_416.md` (probe first) and `SCOPE_416.md` (sources, interface, costs), both in `F:/TanitAD-scratch/q5_416/`. The PI's *"Approve reencoding"* covers the frames and DINOv3-L tokens at 416x1024 on the pod and/or box 2 (`stopperc/PI_Q5_APPROVAL_2026-11-15.md`). Recommendations are the drafter's; the decisions are the PI's.

## Q1. If the probe reads RES416-ADDS-40, approve the training arm's box-2 time now, conditionally?
* **What it is:** Q5_s0 and Q5_s1, 3,000 steps each from M2b. The only change is `--cache-dir` pointing at the 416 cache: no new flag and no architecture change. The stage-1 NULLs and floors are reused.
* **Cost, ESTIMATED from MEASURED rates:**
  * box 2: 2 x ~40 min plus T1-kin dumps and gates ≈ 2 h, plus shipping ~20 GB (4x10 + mean + kinematics; box 2's free disk is UNVERIFIED);
  * the pod's dense cache (~11 h wall) sits inside the re-encode approval.
* **Not included:** RES416-ADDS-160, which needs the R arm's architecture change (`n_tok` 160 from M2b). That case comes back to the PI with its own cost.
* **Recommendation: yes, conditionally.** The arm runs only on RES416-ADDS-40 at both training samples, under a launch-gate PASS. Any overrun above 4 box-2 h, or any change beyond `--cache-dir`, comes back to the PI. This avoids a stall between the verdict and the next arm. The prior for ADDS-40 is LOW (HYPOTHESIS), so the expected spend is small.

## Q2. If the probe reads RES416-FLAT, end the stop-perception line on PhysicalAI-AV and take the stop lever to a map-carried stop-line input on the nuPlan / NAVSIM rig?
* **Why:** four perception levers would each have failed a committed read: the 256 grids, a trained adapter with a stop auxiliary, a trained LoRA encoder, and 2.6x the pixels.
* **What the card's own measurements say:**
  * the cue a readout reaches is already in T's non-vision inputs (S0-P: E alone 0.852 AUROC);
  * weighting only moves the operating point along an unchanged ROC.
* **What is left:** an input that carries the stop location. PhysicalAI-AV has none at any layer (card §3). A map stop line exists on nuPlan / NAVSIM (REFe `AMENDMENT_6_STOP_LINE.md`, cited by card §3).
* **What it involves:** another corpus and rig, its own card, and the goal-input admissibility check (the input must not be computable from the situation classifier).
* **Recommendation: yes.** Log the FLAT row as an elimination on this card, and open the map stop-line card on the REFe rig. Two things stay outside this decision and need their own card if wanted: an unpooled fine-token reader (FULLRES's committed lever), and a stop location predicted from vision as a goal point.

## Q3. Train frames: rebuild on the pod, or read Thor's banked 416 cache?
* **Thor's cache:** 4,713 v2ep, 393 GB as of the 2026-10-04 inventory; present today is UNVERIFIED. It would save the pod's 416 frame build (~4.3-5.4 h CPU per pass, ESTIMATED).
* **What Thor would need:** either a ~380 GB transfer over its wifi link, at an unmeasured rate, or a read-only extraction script on Thor for the probe's NOW frames (~7 GB of PNG bytes). Thor is not a host the re-encode approval names, and it carries other streams.
* **Recommendation: rebuild on the pod from HF at the pinned revision.** It is inside the approval. It uses the pod's otherwise idle CPU (the GPU read 0 % and 2 MiB used on 2026-11-15). It keeps one frame route for train and eval, and C6 cross-checks it against E:'s eval139 copy.

# AMENDMENT 2 to PREREG_SCORER1.md (sha256 1f839af2...): the deciding mechanism test reads the heads that carry the margin

**Date:** 2026-11-08 ~17:05Z, Master Mind.
**Written before** any arm trained, before Stage 0 was read, before any E2 token was read, and before the lane-truth read (`PREREG_LANE_SPEED.md`) produced a valid result. At this moment that read has no result: its definition control G-L1 is failing and its diagnostic variants are still running.

**Why.** `D-REFE-MARGIN-HEADS-1` (exact Shapley over the 6 rule heads of the pick's margin over the best holding plan):
- T0, real frames, 74 decaying CRUISE picks: largest term lane 45, DAC 22, TTC 4, NC 3. Median share lane 0.47, DAC 0.28, TTC 0.09, NC 0.03.
- T2-class, 42 decaying cruise calls (INHERITED from the sim session): DAC 36, lane 5, EP 1. Median share DAC 0.67, lane 0.22, TTC 0.02.

M1 as frozen reads TTC and NC only, about 12 % of the T0 margin. It could pass while the heads that reject the holding plan stay unchanged. The evidence for this change comes from the C3 CRUISE tokens (E1, "never decisive alone") and the sim session's replay. E2 and Stage 0 are untouched.

## Changes

1. **M1 head set becomes {TTC, NC, DAC}**, equal-weight mean, same estimator (within-token Spearman, average ranks, a token enters head h only if its true labels vary, n per head reported). The bar form does not change: S1 − C0 ≥ +0.5·|Δρ_C0| **and** CI-lo > 0 **and** |S1 − C0| > 2·|C0r − C0|, now on the three-head mean.
2. **Stage 0 Z1/Z2 read the same three-head mean.** The thresholds do not change (Z1: mean Δρ ≤ −0.20 with CI-hi < 0; Z2: CI includes 0, or |mean Δρ| < 0.10).
3. **Labelled regression row.** The original two-head M1 (TTC, NC) is still computed and printed beside the amended one, on M1 and on Stage 0. It cannot decide. Each head's Δρ is also printed on its own.
4. **New reported row, not deciding:** the Shapley margin split of `D-REFE-MARGIN-HEADS-1`, on E2's decaying CRUISE picks that have a holder, for u400, C0, C0r and S1. The fan is frozen, so this needs only each arm's logits. It shows whether the share carried by each head moved.
5. **§6 closed-loop read adds** the per-head margin split at every call, not only the first. The first-call split (n 3) says nothing about later calls.

## The lane head: one rule per outcome of `PREREG_LANE_SPEED.md`

S1's head set stays NC, DAC, TTC in every branch. The code and the launch gate pin exactly that (`scorer_finetune.py:149` `PAIR_HEADS`, `launch_gate.py:464`, v3 package).

- **Lane O1 (reads speed as danger, gates G-L1..G-L4 pass):** add one arm, **S1L** = C0 + λ_p·L_pair over {NC, DAC, TTC, lane}, seed 0, 400 updates, λ_p measured by the same rule with the four-head loss.
  - It is read against the same C0 and C0r with the same bars, its M1 being the mean over {TTC, NC, DAC, lane}.
  - If S1 and S1L both clear, neither is called better unless the paired S1L − S1 on M1 and B1 exceeds 2·|C0r − C0|. Otherwise the arm with fewer heads is the default.
  - **Authorisation:** S1L is about +3.5 A40-h (ESTIMATED) beyond the granted 10.5. It needs the PI's go. Without it S1 runs alone and S1L stays the next lever of outcome O2.
- **Lane O2, O3 or O4 (no valid slope defect):** no lane arm for the speed defect. The lane head's margin share then comes from its level rather than a speed slope. The next lever is a zero-GPU selection test of the lane term's weight in the rule, under its own pre-registration, after the P6 read.
- **Lane read INVALID (no variant reproduces the banked labels, G-L1 fails):** no statement about the lane head's speed slope can be made. S1L is blocked, and the named blocker is the unexplained relabel mismatch (`lane_truth/AMENDMENT_1.md`). S1 runs as amended.

## Conditions on S1L (all before any launch)

- **Pair definition.** The lane target is a soft value, `t = clip(1 − (d − 0.5)/1.5, 0, 1)`. A lane pair needs t_i − t_j ≥ 0.1. This is above the 0.08 m (about 0.05 in t) banked near-tie flip. Proposals with a masked target (below 0) are excluded. F/S balancing and the speed-contrast rule are unchanged.
- **Labels.** Training uses the banked SFT-4 lane side files. Whether u400 was trained on those exact labels must be answered first. If it was not, S1L does not start from a comparable recipe, and this is a launch blocker.
- **E2.** The E2 table needs lane truth for its 2,000 tokens from a pipeline that passes G-L1.
- **Code and gate.** New `PAIR_HEADS` entry (column 6), a `refe_scorer1` profile change, G-DVB and G-LIVE entries, and the two mutation arms of section 2. G-LIVE reports the share of sets with at least one kept lane pair; a share of 0 is a launch blocker. A launch-gate pin edit moves profile fingerprints, so every test reading `launch_gate` is run before landing.

## Unchanged

B1, G1, G2, G3, the estimator, the replicate floor, the seeds, E1/E2/E3, outcomes O1 to O5, and sections 7 and 8 of the card. Items 1 to 7 of Amendment 1 stand. Any tool that reads M1 implements the head set of item 1, and this is checked before its first read.

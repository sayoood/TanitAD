# P1 -- why REFe u400 picks a slow-down on CRUISE: a descriptive diagnosis (stream YSEL, 2026-11-07)

**Scope and evidence class.**
- **Tier T0** (open loop, one step). Model u400, sha256 `8b4dba48...`. All numbers are MEASURED.
- **Estimator:** paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile. It answers *"another draw of logs"* only.
- One checkpoint and a deterministic planner: no claim about training or inference variance.
- Nothing is fitted here and no rule is scored.
- Sources:
  - `table.npz`, the C3 1,058 tokens with TRUE per-proposal PDMS;
  - `refe_ego_probe_20261106/full/results.npz`, the raw 20 x 0.2 s proposals and GT speeds.
- Speeds use the ego probe's own estimator: the one-step 0.2 s chord at t = 4 s, with v0 = `ego_real[:, 6]`.
- **G0 gate PASS** (`g0_gate.json`). The offline re-pick reproduces the banked pick on 1,058/1,058 tokens.
  - The rule is `refe/planner.py:551-556`: `p0*p1*(5p2+5p3+2p4)/12*p6` over sigmoid logits, then argmax at `:583`.
  - The RED arms fail as required: dropping the lane factor gives 635/1058, the old logit-sum rule 334/1058, argmin 0/1058.

**Order of reading.**
1. P1 was computed first on the **TRAIN logs only** (`p1_diag_train.json`, 61 logs). The rule family was chosen from it.
2. After the frozen P2 read, the **ALL-logs** run (`p1_diag_all.json`, 154 CRUISE, 63 slow picks, 35 logs) was added as an appendix.
3. `p1_diag_heldout.json` was also added after the read.

The table below quotes ALL. TRAIN agrees in direction on every row.

## The four questions (ALL logs; population = 63 CRUISE tokens whose pick plans a > 15 % slow-down, 55 of which have a fast proposal)

| question | answer (MEASURED, ALL logs) |
|---|---|
| **Q1.** Does a valid proposal exist with v4/v0 >= 0.9 and true PDMS >= the pick's? | **Yes, on 85.7 % [70.6, 97.4] of slow picks.** On 82.5 % [67.1, 94.7] it is strictly better. The median slow-pick token holds **47** fast proposals. |
| **Q2.** Which head makes the slow pick win? | Against the scorer's own best fast proposal (`f_sc`), the slow pick's aggregate is higher by d log agg = **0.162 [0.118, 0.221]**, about 15 %. The **lane head is the largest term on 31/55 tokens**, DAC on 17, TTC on 4, NC on 3. Mean log terms: lane 0.061, DAC 0.043, TTC 0.027, NC 0.021, EP 0.009. Against the oracle fast proposal, lane leads 26/55 and DAC 23/55. |
| **Q3.** Is the slow pick truly better by PDMS? Are humans "wrong"? | **No, on both counts.** The slow pick's true PDMS is **0.349**, and **58.7 % are zeroed**. Those tokens are DAC failures: true DAC at the pick is 0.509. The oracle fast proposal scores **0.931**, and its EP is 0.941 against 0.313. The proposal nearest the human future (ADE 0.51 m, v4/v0 0.915) scores **0.620**, +0.252 [0.112, 0.397] over the pick. Yet the scorer ranks that human-like proposal **26th** of 64 (median). |
| **Q4.** Predicted vs true sub-scores | **The scorer reads speed as danger, and the truth does not.** Within-token Spearman against v4/v0, CRUISE: **pred TTC -0.479 vs true -0.049; pred NC -0.443 vs -0.039; pred DAC -0.290 vs -0.039; pred DDC -0.492 vs -0.038; lane -0.329** (no truth in the table); **pred EP +0.086 vs true +0.408**; the aggregate -0.252 vs true PDMS **+0.370**. On the slow pick against the oracle fast proposal, the scorer predicts DAC 0.930 vs 0.801 and EP 0.760 vs 0.653. The truths are **0.509 vs 0.964** and **0.313 vs 0.941**. |

**Reading.**
- The slow-down is a **selection defect, not a generation defect**. A faster, better proposal is in the set on about 86 % of slow picks.
- The scorer is **miscalibrated against speed on every safety head at once**, and its EP head barely rewards progress.
- The single largest margin term is the **lane head**, whose truth is not in the table, so its calibration cannot be checked here.
- **TRAIN only, pre-freeze:** 23 slow picks out of 86 CRUISE tokens, 74 % zeroed. Lane was the dominant term on 13 of 19. The within-token Spearman signs were the same.

## The sim token `41d7b533797c5209` (TRAIN log: a spec check, not evidence)
- The banked pick is **37**, with v4/v0 **0.43** and true PDMS 0.911. The brief's sim reading (A0 pick 37, v4/v0 0.43) reproduces.
- Four proposals hold speed (v4/v0 0.91 to 1.07), and **all four have true PDMS 1.000**.
- The scorer puts them at 36 % to 65 % of the pick's aggregate. It predicts their **TTC 0.16 to 0.66 lower and NC 0.03 to 0.32 lower** than the pick's, and their lane 0.72 to 0.86 against 0.953.
- With a lead car 17 to 19 m ahead, the **TTC and NC heads, not the lane head**, carry the slow-down on this token.

## Caveat that changes how the brief's "41 %" should be read
- The C3 sample over-represents the zeroed stratum: 529 of 1,058 tokens, against 529 of 12,146 in navtest.
- **91 of the 154 CRUISE tokens are zeroed.** The 41 % slow-down share is therefore not a navtest-population rate.
- **Population re-weighting:** slow-down share TRAIN random 0.200 vs zeroed 0.304. The two rates are of similar order, so the slow tendency is real on the random stratum too, but the headline share is inflated.
- **Corroboration:** the ego probe's non-zeroed CRUISE bias was −0.746 m/s.

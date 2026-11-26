# GOALS_AND_CLAIMS row draft: `H-REFE-SAFEHEAD-1` (for the MM to land; update IN THE SAME TURN as the verdict)

| id | statement | status | evidence |
|---|---|---|---|
| `H-REFE-SAFEHEAD-1` | Re-fitting ONLY C1's four safety read-outs (score_head rows 0, 1, 3, 5 = NC, DAC, TTC, DDC; 1,028 params; init C1; quality-focal BCE γ 2 on TRACKED navtrain safety labels; EP/C rows and every other tensor frozen) raises C1's navtest T1 PDMS (bars P1 CI-lo > 0, P2 Δ ≥ 2·max(F, 0.1158), guard G on NC/DAC/TTC/EP/C, replicate R seed 1) | **PRE-REGISTERED** (card frozen 2026-11-26T22:10Z, sha256 in `PREREG_SAFEHEAD.sha256`); no SH number exists | `PK/raw/2026-11-27-safehead/PREREG_SAFEHEAD.md`; code `CODE.sha256`; synthetic tests 14/14 + 6/6; launch-gate profiles `refe_safehead_{s0,s1,red}` |

Notes for the verdict update:
- **Tier and estimator:** T1, the anchor table `p5c_next/propscore/full_C1_s*of2.jsonl` (C1's own, 92.5548 reproduced). Paired log-cluster bootstrap, 10,000 draws, seed 20260927, 136 logs.
- **The outcome** is one of ADOPT / SEPARATED-SMALL / NULL / ADVERSE / NULL-HELDOUT / INVALID. Quote it from `VERDICT_SAFEHEAD.json`, never from the provisional read line.
- **Every non-ADOPT names its next arm:** SH-MLP (8.1) or SH-TOPK (8.2).
- **Caveats to carry:**
  - DDC is selection- and score-inert (`planner.py:558-561`; the tracked PDMS formula excludes it), so the PLAN's 0.15-point DDC regret is out of this lever's reach.
  - The held-out rule uses safety labels only, because the cache's EP and PDMS are a rebuilt metric cache.

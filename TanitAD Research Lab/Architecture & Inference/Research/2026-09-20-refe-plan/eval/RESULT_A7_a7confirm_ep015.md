# RESULT -- SPEC_NAVTEST AMENDMENT 7 -- `a7confirm_ep015` (the last-pose heading repair)

*Every number is read from the unchanged harness's per-token scores; none is typed. Written by `eval/a7_confirm.py`; the repair, the gates, the statistic and the decision rule were fixed in Amendment 7 (registered ~13:50 Berlin, blob `71fc61b5`) before any confirmation-token result of the repair existed.*

**Confirmation tokens:** 923 from 43 navtest logs (Amendment 5's set, none in W3's 200; md5 `3098d1784a9b99ba9d5a1a0a672a7ed6`). **Snapshot:** after epoch 15 (md5 `d7c59f4f2fbcbde3e2dec8f67d63a7e7`). **Repair:** on the planner's native [20, 3] output, heading[19] := heading[18] for the executed plan, before the NAVSIM conversion; nothing else changes.

## Verdict: **ADOPT**

| seam | PDMS | vs shipped, PDMS points [95 %] | better / worse / tied |
|---|---|---|---|
| shipped (the pipeline's) | 60.76 | -- | -- |
| **repaired (the measure under test)** | **76.96** | **+16.20 [+12.83, +19.52]** | 511 / 99 / 313 |
| path-tangent heading instead (reported, not gating) | 76.07 | +15.31 [+12.12, +18.51] | -- |

Decision rule committed in Amendment 7: ADOPT iff the lower bound is above 0; REFUTED iff the upper bound is below 0; otherwise NOT PROVEN. Estimator: paired log-cluster bootstrap over the 43 logs, 10000 resamples, percentile 95 %, seed 20260927 (eval/snapshot_pair_under_rule.py's estimator). It answers "another draw of episodes" only.

Registration check: SPEC_NAVTEST.md git blob `71fc61b5` (registered `71fc61b5`: match), file time 2026-09-27T13:46:24; the native record this confirmation read was written 2026-09-27T14:16:50.

## Validity gates (all must pass before the statistic is read)

| gate | result |
|---|---|
| (a) repaired x, y bit-identical to the shipped seam; only the t = 4.0 s heading differs | PASS: x/y all 8 poses True, headings 0.5-3.5 s True, 4.0 s heading == native heading[18] True; changed on 923/923 tokens |
| (b) the shipped seam reproduces the pipeline's pick and score | PASS: seam == to_navsim(recorded native) True; dump pick == recorded pick 923/923; == v1 argmax of its logits 923/923; proposal at pick == seam True; scored PASS |
| (c) every harness run PASS, every token valid | PASS: shipped PASS 923/923; repaired PASS 923/923; tangent PASS 923/923 |

## Reported, not gating

NAVSIM sub-scores x100 (repaired - shipped): NC +3.90, DAC +9.97, EP +12.05, TTC +9.10, C +32.39, DDC -0.16.

The defect on these tokens (executed plan): |heading[19]| > pi on 58.7 %; median |heading - path tangent| 1.0567 rad at t = 4.0 s vs 0.0376 rad at 3.8 s.

STOP floor on these tokens (context): 62.17.

Same repair on the latest snapshot's 200 tokens (the exploration): 61.18 -> 77.94, +16.76 [+11.72, +22.13] (snapshot 015, W3's 200 tokens; the exploration that motivated the amendment).

**The four metric families are identical by construction** -- the NAVSIM family adapter reads positions only (it drops the heading column; its heading_mae_deg is the path tangent) -- and running `families6.py` on both seams confirms it: headline numbers identical = True; full per-family blocks identical apart from run labels = True. Strategic: unavailable in NAVSIM.

## Not in scope

The model-side cause of the corrupted heading is a SEPARATE question (Amendment 7): any training change for it needs its own proof. Adoption in `refe/planner.py` is the coordinator's step; this script touches no shipped path.

Artifacts and checksums: `eval/raw/a7_confirm/MANIFEST.md`; full record `eval/raw/a7_confirm/a7_confirm_ep015.json`.

# TanitAD program report — 2026-08-29 evening (D-025)

*Master Mind · ~18:30 Europe/Berlin · branch `agent/arch-inf-20260803` · every number
carries arm + tier + evidence class; sources = registry, raw JSON, named commits.*

## 0. Headlines

1. ⭐⭐ **MM-E1 stage 2 (c71627604): EMA-OUT on drift — and E-DEC-69, the largest
   prediction gain ever measured on the trainable line.** Drift 0.6952 vs 0.6709
   (+3.6 % — the 2k gain was a τ-warmup transient, called in advance by the
   pre-committed frame); cos_ctr **0.6043 → 0.7524 (+24.5 %)**, nrmse −9.9 %,
   absorption HOLDS. ⇒ the drift attractor stays open (next lever: frozen-teacher
   feature target, banked); **ship-v7f-with-EMA is the PI's decision** (τ-ramp arm
   required first if yes; EMA×O14 interaction unseparated — committed limit).
   [MEASURED, T0-DIAGNOSTIC, config-diff CLEAN]
2. ⭐⭐ **The B1 core release is VERIFIED on HF** (16:28, re-download sha match):
   `Sayood/tanitad-v7-training-corpus` private, corpus `a48251e89c7a8603`, 4,719
   clips/26.2 h. Camera phase running (~464/4,719 at last report) behind the
   PI-mandated validation gate; trainer intake starts from the VALIDATED manifest.
3. ⭐ **The RL post-training library is DELIVERED and closed pre-fan** (PI directive
   D-RL-REFCV3, same day): 6 modules, 115 tests, A0 coverage PASS after two textbook
   self-retractions (TRAIN-C1 selector-inversion near-miss; TRAIN-C2 pooled-median-
   over-undefined retraction) + the fan-collapse objective removed before it could
   destroy raw-fan quality. **A1 blocked only on the refcv3 IL cold start = the B1
   refcv3 training run.** RL compute measured ~free (0.054 s/step, 4060).
4. **Benchmark portfolio APPROVED wholesale** (PI) + NavSim pipeline: devkit runs,
   grouping key measured (log_name, Outcome A, 100 %/0-leakage; Eval verification of
   the artifact pending), warmup-never-carries-a-CI rule, EPDMS≠four-families
   finding, the false-green smoke caught by counters.
5. **Media hub delivered** (PI directive): offline clickable UI over 491 assets,
   114 missing videos materialized, verify PASS; 6/6 showcase provenance resolved
   mechanically.
6. **Lab health**: run 003 complete (133 library entries, verify clean); the
   steering audit landed (39-row LAB_BACKLOG; top contradictions queued);
   daily-trigger hardened (cron + drumbeat; session-cron fragility stated).

## 1. Fleet (MEASURED ~18:25)

Thor **idle** (emao14_30k done:true 30k, md5 3a030ba2; awaiting the validated
manifest → epcache build) · dev box idle post-probes · DataFlyWheel active (camera
pull/validation) · EvalFlyWheel active (devkit metadata re-pull) · TrainingFlyWheel
session ended cleanly after its deliveries · 20 commits today on the agent branch.

## 2. The PI decision queue (defaults stated)

| # | decision | default/recommendation |
|---|---|---|
| 1 | **v7f ships with EMA?** (E-DEC-69) | **RECOMMEND YES** — prediction is what the fan rolls on; τ-ramp tiny arm first (~35 min Thor), then launch |
| 2 | **v7f scaled-run GO** | ready on the validated manifest + your word; recipe otherwise final |
| 3 | 132 media mp4s into git? | default: manifest-audited, untracked |
| 4 | D-GATE-D1-EST (biased D1 adjudicator) | fix-and-readjudicate vs freeze-and-document |
| 5 | Parked (recoverable, your call to unpark): lane arm, offset calibration, MM-E3 F1–F4 | stays parked |

## 3. Retraction-log day (the mechanism working)

TRAIN-C1 (mock-validates-plumbing near-miss), TRAIN-C2 (a statistic over a
population where the quantity is undefined is a DIFFERENT measurement — the scope
family reaching statistics; caught RECURRING same-day in the progress pooled mean
and stopped), DE-C151 addendum (suppression supersession with the named cost),
plus my own two named corrections (a commit-message overclaim caught by the
DataFlyWheel's blob check; a design ruling issued on a premise later retracted —
ruling kept on independent merit only). Every sweep named; zero silent.

## 4. Overnight plan

Validated manifest lands → epcache build on Thor (per-clip cy mandatory, range-
reader camera, manifest-derived skip-list) → if the PI answers #1: the τ-ramp arm
(~35 min) → v7f launch readiness at morning with the final recipe. The four-
consumer training order after v7f: refcv3-IL (unblocks A1) → refav1 → refd.

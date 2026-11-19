# HANDOFF_CEIL (PREREG_N1D_CEIL, card sha256 092bb25c), written 2026-11-19 ~16:03Z

## State
- Pipeline built and SMOKED (200 windows): extraction, probe, reader rc 0, all card controls read on smoke (K-CONST / K-PARITY not asserted on smoke). Smoke is a code check, never a result.
- F-EGO = the f8 passed to model._ego in the O call (capture by wrapping model._ego, last of 3 calls). Columns from source (p10_prior.kin_features): v, a, k, v^2, v*a, v*k, v^2*k, a*k.
- FULL RUN launched 16:01Z on the pod; driver pid file /root/strat1/n1dwceil/driver_full.pid, log /root/strat1/n1dwceil/n1dwceil_full.log. One process at a time, nice 10, OMP 1, probe workers 1.
  Stage 1: 4 extractions (s0/s1 x ego1/ego2), ~1.5-5 min each; stage 2: 80 probe fits (tree + logistic); stage 3: reader.
- Sentinel (last log line): `ZZ N1DWCEIL DONE ZZ` or `ZZ N1DWCEIL FAIL ZZ <reason>`.

## Waiter
`bash F:/TanitAD-scratch/n1dw/ceil/wait_ceil.sh` (polls every 5 min, 12 h cap, exits on DONE / FAIL / dead driver).

## Outputs when DONE (pod)
/root/strat1/n1dwceil/out_full/result/{RESULT_CEIL.json, READ_CEIL.txt, oof_correct.npz}; features out_full/ego1, ego2.
Code: pod /root/strat1/n1dwceil/code/ == local F:/TanitAD-scratch/n1dw/ceil/code/.

## Next step after DONE
Pull the three result files with pscp, sha256 them. Read READ_CEIL.txt: HEADLINE (tree decides), per-seed lines, CONTROL STATUS (K-CONST, K-PARITY, K-EGO-DET, IDENTITY-ARM, REGRESSION-ARM), beside lines. Land code + results under raw/2026-11-19-n1dw/ceil/ (MM only). If FAIL: tail of the log names the stage; extractions skip on rerun if their .json exists; rerun `nohup setsid bash code/run_n1dwceil.sh full` from /root/strat1/n1dwceil only after confirming the old driver is dead.
Not committed, not pushed, nothing banked. Deleted: pod out_smoke (DELETIONS_CEIL.log on the pod).

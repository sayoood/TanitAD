# HANDOFF_TCODE (PREREG_N1D_TCODE, card sha256 4cec3917), written 2026-11-19 ~15:25Z

## State
- Pipeline built and SMOKED (200 windows): all captures work, all five card controls read (K-CONST not asserted on smoke), reader rc 0. Smoke is a code check, never a result.
- Smoke facts: F-CODE-LAT columns derived from source = lateral softmax 22..26, turn constraints 35..43 (turn_t_start_s .. turn_exit_psi_deg), CODE_DIM 60. K-PARITY max abs diff 0.0 for F-TEXT and F-PLAN at OMP 1 thread; K-DET bit-identical (all keys incl. F-CODE, F-TTRUNK).
- FULL RUN launched 15:18Z on the pod, driver PID file /root/strat1/n1dwtcode/driver_full.pid, log /root/strat1/n1dwtcode/n1dwtcode_full.log. One process at a time, nice 10, OMP 1, probe workers 1.
  - Stage 1: 4 extractions (N1D_s0/s1 x feat1/feat2), est. ~4-8 min each; stage 2: 160 probe fits (tree + logistic, SHUF beside), est. 1-2 h under load; stage 3: reader.
- Sentinel (last line of log): `ZZ N1DWTCODE DONE ZZ` or `ZZ N1DWTCODE FAIL ZZ <reason>`.

## Waiter
`bash F:/TanitAD-scratch/n1dw/tcode/wait_tcode.sh` (polls every 5 min, 12 h cap, exits on DONE / FAIL / dead driver). The MM runs it.

## Outputs when DONE (pod)
/root/strat1/n1dwtcode/out_full/result/{RESULT_TCODE.json, READ_TCODE.txt, oof_correct.npz}; features /root/strat1/n1dwtcode/out_full/feat1, feat2 (feat_N1D_s{0,1}.npz + .json). Pull RESULT_TCODE.json and READ_TCODE.txt with pscp; sha256 them.
Code: pod /root/strat1/n1dwtcode/code/ and local F:/TanitAD-scratch/n1dw/tcode/code/ (identical).

## Next step after DONE
Read READ_TCODE.txt: HEADLINE per the card sec 4 (CODE-CARRIES / T-HAS-CODE-LACKS / T-LACKS / NEITHER), CONTROL STATUS line, F-PLAN-F-CODE and F-PLAN-F-TTRUNK beside. Land code + RESULT_TCODE.json + READ_TCODE.txt under raw/2026-11-19-n1dw/tcode/ (MM; agents never land). If FAIL: tail of the log names the stage; extractions are skipped on rerun if their .json exists, so rerun `nohup setsid bash code/run_n1dwtcode.sh full` from /root/strat1/n1dwtcode only after confirming the old driver is dead.
Not committed, not pushed, nothing banked to HF. Deleted: pod out_smoke (see DELETIONS_TCODE.log).

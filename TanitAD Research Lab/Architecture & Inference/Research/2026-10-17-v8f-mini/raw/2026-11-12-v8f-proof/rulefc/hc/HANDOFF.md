# HANDOFF — card ADD152 (HEADCLOSE10K), 2026-11-28T08:45Z

State: card frozen (631ef3e1fe8bc2b0); table job done, t1 PASS (rulefc/hc/table/TABLE.json; HEAD_TABLE 4b4a7856, ZERO_TABLE 448fea48);
queue `box2_headclose_queue.py` (8e8ee771) running on box 2 since 08:41Z via `rhacc10k/chain/chain_rhacc.sh headclose_qall rhacc10k`
(chain out `rhacc10k/chain/chain_headclose_qall.out`, box-2 log `D:\tanitad\logs\headclose_qall.log`). Busy marker held by the chain.

Next, when the chain prints ZZRHACC-END:
1. Read the box-2 log tail; a STOP (e.g. CONTROL-ZERO-FAILED) stops the card: freeze a successor amendment.
2. Box 2: sha lists of H_RQ50KE1_s* and H_RQ50KE1_P30_s* dump files -> rulefc/hc/PARTNERS_RQ50KE1_BOX2_SHA.txt, ..._P30_...
3. Pull V1 runs + CONTROL_ZERO.json of both zero runs into rulefc/hc/pull; V1_P30 runs into rulefc/hc/pullp30 with copies of
   e1a/pull H_RQ50KE1_P30_s* and H_RLcA_P0_s*; grid partner copies of e1r/pull H_RQ50KE1_s* into rulefc/hc/pull.
4. Reads: read_headclose10k.py --mode main; read_headclose10k_p30.py; combine_headclose.py --free-gb <box2 D: free>.
5. If run_ceiling: `bash chain_rhacc.sh headclose_ceiling rhacc10k`, then read --mode ceiling, combine with --ceiling.
6. Delete the zero-control run folders on box 2 after their control passed (re-creatable); delete local pulled copies after landing.
7. Row 218 (GAC_ROWS_155.md) with the beside clause; hold 155 (mk_mk_list.py 155 ...; mk_hold_v8proof3.py; scan_hold.py; land_v8proof.sh);
   ls-remote; FYI to local_7718c5d9-da1d-4e01-bfe3-fb70778185a6.

## 2026-11-28T11:30Z — done
Queue complete (t2, t3 PASS); reads passed every control on the second pass (instrument fix logged in TIMES_NOTE); names HEADCLOSE10K-U-LOWER and
HEADCLOSE10K-P30-U-NOT-SEPARATED; combined: ADOPT (HEADCLOSE10K_RESULT.json); ceiling arm not run. Row 218 in GAC_ROWS_155.md, hold 155.
Temporary data to delete after the land is verified: rulefc/hc/pull and rulefc/hc/pullp30 run copies (sources on box 2 and in e1r/e1a pull).

# MM rulings on the CKPT120 read build, 2026-11-17 (before any CKPT120 number exists)

The arms are queued on the pod and none has trained. These rulings set how the reader applies the FROZEN card `PREREG_V8F_CKPT120.md` where `BUILD_CKPT120_READ.md` lists an interpretation (its section "Deviations for the MM", items 1-16). No rule of the card changes.

| # | ruling |
|---|---|
| 1 | **The stricter reading.** Card section 8: "every effect also > F120". This binds the raw point too. FOLLOWS needs, at both seeds: real − donor and balanced − 0.5 separated above 0 and above F120, and the raw point > 0 and > F120. The raw floor is \|H1_rep − H1_s0\| on the raw statistic. Applied in `c120_read.py` (`route_label`, `F120["raw"]`; backup `c120_read.py.before_mm_ruling`). Test `test_route_raw_point_must_clear_f120_mm_ruling`: 28 passed. |
| 2 | **Accepted.** F120 per contrast, on that contrast's own statistic ("F120 = \|H1_rep − H1_s0\| on that cell"). |
| 3 | **Accepted as a beside row.** The echo share is printed under its stated definition, marked UNVERIFIED (hits on tactical-token windows / all hits). No outcome is decided on it. |
| 4 | **Accepted.** The train-side K-LABEL ran at the label build: 0 validity mismatches on 456,946 train steps of release 83e30bd3 (`BUILD_CKPT120_P2A.md`). The eval side reproduces the card: 341 / 27, left 79 / right 262. |
| 5-11 | **Accepted** as built. Each follows the card's words or STRAT1's banked reading choices. |
| 12 | **Accepted for this read, and a WORK ITEM opens.** TTC / DK-VIOL and the tactical confusion matrix with goal selection have no instrument in the STRAT1-family readers. Each prints NOT COMPUTABLE with its reason and n. Under the binding four-family rule, a missing metric is a work item, not an excuse: `W-FAM-STRAT1-TTC-TACCONF` is registered, to be built into the STRAT1-family reader and run on the banked dumps. |
| 13 | **Accepted.** The pod chain continues on INCONCLUSIVE. `resolve_reach.py` resolves it off-GPU with the built model's names. A verdict still INCONCLUSIVE after that stops `after_ckpt120.sh`. HEAD-DEAD makes the arm INVALID. |
| 14, 16 | **Accepted.** A failed step does not stop the others. The state reads PROVISIONAL until R1, R2 and H1_rep have read, and BROKEN READ if R1 or R2 fails. |
| 15 | **Checked by the MM.** The pod chain `/root/strat1/strat1_ckpt120.sh` writes `$RUNS/$tag/train/ckpt_last.pt` with RUNS=/root/strat1/runs and logs to `/root/strat1/logs/ckpt120.log`, the S1d layout. |

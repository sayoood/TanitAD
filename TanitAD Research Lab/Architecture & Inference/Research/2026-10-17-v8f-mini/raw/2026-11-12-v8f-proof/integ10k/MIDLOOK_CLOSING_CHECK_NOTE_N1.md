# Note N1 to the pod check of the second-look generation `rzgfhqc` (study "midlook-closing"): two sub-checks ended FAIL on a tolerance that the check had set tighter than float32 allows

Written by the coordinator fork after the pod check ended and before any repair (time: the line of this note in `TIMES_NOTE.md`, after the line that logs the FAIL; hash in the sidecar file). No card uses the wrapper `integ10k/v8f_t1_eval_reader_rule_rzgfhqc.py` before the check ends PASS in the form of record below.

## 1. What the check reported

`CHECK_MIDLOOK_CLOSING_chk1.txt` on the pod: 22 PASS lines, 2 FAIL lines, last line `ZZ CHECK-MIDLOOK-CLOSING FAIL`.

* FAIL C2 (form RRh, later looks): the closing fed to the rule equals the reader closing recomputed from the cache tokens at frame j + q on 32 present rows with largest difference 4.93e-06 m/s against the bar 1e-06; it is not 0 on 32 of them; it equals the logged `c_read` with difference 0.
* FAIL C4: the logged `c_read` of the forms RR, RRc and RRh equals the recomputed reader closing with largest difference 4.93e-06 m/s (the same value for the three forms, 32 present rows each) against the bar 1e-06.
* Everything else passed, among it: with `--rule-every 10` and with `--rule-every 5` for the forms the parent admits, every dump, tick, rule-log and mid-look-log array of the new wrapper equals the parent's with difference 0; RRc later looks equal the label closing with difference 0; RR later looks feed exactly 0; the form RRd is refused; the planted wrong arms are caught (the RR log read as if it were RRh: largest difference 1.18 m/s; the helper read one frame early; the helper value scaled by ten).

## 2. Cause

The wrapper evaluates the token reader in float32 on the windows of a batch; the check recomputes it one row at a time. The closing is the third output of the reader times 10 m/s. A difference of 4.93e-06 m/s on values of the order of 1 to 10 m/s is 5e-07 to 5e-06 relative, the size of float32 rounding when the same network is evaluated on batches of different shape (float32 resolves about 1.2e-07 relative per operation). The same largest difference on the same rows for the three forms, and the exact equality of the fed closing with the logged one, say the same: one recompute path that differs from the rollout by rounding. The bar 1e-06 was a fault of the check: it stands below what float32 can hold for this comparison. The builder of the check had named this risk in its hand-back before the run.

## 3. Form of record

* C2 and C4 with the tolerance 1e-04 m/s (`--tol-reader 1e-4`): two orders above the rounding seen, four orders below the difference of the planted wrong arm (1.18 m/s), and far below any value the rule acts on (its closing term enters as metres per second against a gap in metres).
* The planted arm of part C has to stay caught under this tolerance (the check fails if it is not), and every other sub-check keeps its bar.
* The analysis is repeated on the run folders already on the pod (`--analyze-only`, folder `midlook_test/chk1`; no rollout is repeated), output `CHECK_MIDLOOK_CLOSING_chk1n1.txt`, runner `integ10k/pod_check_midlook_n1.sh`. The first output stays as written.
* If the form of record ends FAIL, the wrapper is not used and the cause is looked for in the wrapper.

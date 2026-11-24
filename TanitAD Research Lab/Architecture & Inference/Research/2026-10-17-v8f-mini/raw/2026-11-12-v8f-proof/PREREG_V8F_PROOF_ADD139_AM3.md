# PREREG_V8F_PROOF_ADD139_AM3 — amendment 3 of card ADD139 (RHACC10K): the wrapper of the pre-rolled rows g and h

Frozen after the queue stopped at its first run of row g (`H_RLcA_P30_s0`, exit code 2 before any window: the wrapper of section 3, `v8f_t1_eval_reader_rule_rzgfh.py`, has no parameter `--preroll-steps`, so its argument parser refused the call) and before any value of card ADD139 has been opened. Card ADD139 (e7420aaa6947bfd0) and amendments 1 (17e53f758786417a) and 2 stay as written except for this point.

Section 3 of the card names one study wrapper for all rows, while row g and check C5 name the pre-roll of 30 steps of card ADD70 and its reader of record. The pre-roll exists only in the wrapper of record of card ADD70's pre-rolled runs, `integ10k/v8f_t1_eval_reader_rule_rzgfhq.py` (sha256 prefix 92a8034cc0f7b0d2; `integ10k/pod_longroll10k_queue.sh:26`, and line 253, the run `I10RLcKF5P30Hb20` behind the C5 reference of the specification), which with `--preroll-steps 0` is its parent bit for bit (its own header).

* Rows g and h run with that wrapper, g with `--preroll-steps 30`, h with `--preroll-steps 0`, both on the grid of record of card ADD70 (1,271 windows), with the default pre-roll execution (plan), as line 253.
* Rows a to f stay on `v8f_t1_eval_reader_rule_rzgfh.py`.
* The stopped run folder `H_RLcA_P30_s0` (no window written) is moved aside on box 2, not read and not deleted; its run is made again.
* Control k3 checks the wrapper of each row as well as its flags.

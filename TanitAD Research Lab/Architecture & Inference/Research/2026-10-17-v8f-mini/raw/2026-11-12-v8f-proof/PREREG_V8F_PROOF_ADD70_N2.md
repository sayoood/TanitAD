# Note N2 to card PREREG_V8F_PROOF_ADD70 (LONGROLL10K): the read tools, their fixed conventions, and two changes of tool text made before any run

Written by the coordinator fork after the read tools were handed back and before the queue of the card was started (time: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). No run of the card exists except the smoke on two clips (known values and shapes only). Arms, pairs, estimators, names, consequences and limits of the card and of note N1 are unchanged.

## 1. Read tools (written without data, self-test on synthetic arrays)

`integ10k/read_longroll10k.py`, `integ10k/LONGROLL10K_REUSED.json`, `integ10k/podread_longroll.sh`, `integ10k/wait_podread_longroll.sh`; their hashes are in the line of this note in `TIMES_NOTE.md`. The reader restricts every run in memory to the (clip, t0) windows of the grid and drives the engines of the earlier readers in process (headway reader, family reader, strategic instrument of card ADD66). Its self-test passes 32 of 32 items with planted wrong arms. It has not run on a real run folder: the first read is its first run on data, and a fault of plumbing found there is logged before it is repaired, with the names withheld until then.

## 2. Conventions of the reader that the card left open (fixed here, before any data)

* In the consequence rules of section 6 a family that reads MIXED counts as neither BETTER nor WORSE.
* STRATEGIC uses the verdict of the instrument of card ADD66 per checkpoint (interval excludes 0), with no spread rule, as in cards ADD67 to ADD69.
* A U name LOWER or HIGHER needs the pooled interval to exclude 0 and both model seeds on that side, as in the earlier cards.
* Controls of the strategic instrument that compare the first tick with the logged state (s2, s5) run on the P = 0 runs only: the first tick of a pre-rolled window is not the logged state by design. The other strategic controls run on all runs.
* l6 compares with the banked result of card ADD67 after rounding to the four decimals that file holds, and to 1e-9 against the unrounded headway and family files of that card where they are on the pod.
* l7 compares the hashes of the wrapper and of the checkpoints with the run records and records the hashes of the pinned tool, the grids and the map files; the queue itself stops at its start when the pinned tool's hash differs.
* The check of the rule interval in a run record applies to the runs with a rule.

## 3. Two changes of tool text before any run

* Wording only: one word in comments and messages of four files was replaced by another (`v8f_t1_eval_reader_rule_rzgfhq.py`, `check_preroll_b.py`, `make_longroll10k_b.py`, `pod_lr_smoke_b.sh`); no statement of code changed, the byte counts are equal. The maker regenerates the wrapper with the new hash, and `check_preroll_b.py` passes on it locally. Hashes before and after are in `TIMES_NOTE.md`. The smoke of note N1 ran with the earlier bytes; the queue repeats its three checks on the pod with the new bytes before any outcome run (tool check, P = 0 identity on the full grid, state carry on the P = 30 grid), so the wrapper of record is verified again where it runs.
* `podread_longroll.sh`: the default folder of the tool-check output is the one the queue writes (`lr_test` under the work folder). No reader logic is touched.

## 4. Order

The queue starts behind the queue of card ADD69. Its outcome rollouts begin only after its own three checks pass. The pod read is started by the waiter after the end marker of the queue.

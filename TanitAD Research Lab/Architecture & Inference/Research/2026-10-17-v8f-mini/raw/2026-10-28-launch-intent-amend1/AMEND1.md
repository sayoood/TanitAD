
## Amendment 1 (A1-2026-10-28): K-BUILD's data-module clause, before any LAUNCH-INTENT row is read

**Written 2026-10-28 ~15:10 by the Master Mind.** The four box-2 runs are training or trained. No LAUNCH-INTENT row, T0 number or T1-kin number has been read. Seen before writing, all of them controls:
* the T1-kin chain's dry run, where the reader's K-BUILD control REFUSES;
* A_0's run record and K-SHARE (both PASS);
* the four launch-gate lines.

**What is wrong.** §9's K-BUILD row states the rule: against `aeb54468`, the commit DAG-OACC trained at, each of the three files is byte-identical, or every changed hunk sits behind a default-off flag whose bit-identity test passes. The row then adds a fact: *"`v8f_mini_data.py` is byte-identical"*. That fact was measured against `d7208328`, the base the flag was built on, and not against `aeb54468`. The reader codes the added fact as a rule, so it refuses the launch commit on the data module.

**MEASURED (git objects, 2026-10-28):**
* `stack/tanitad/data/v8f_mini_data.py` at the launch commit `d8c9f7bf` has sha256 `c117e1cf…751752`.
* This equals the data module of DAG-ORC r2's build `8c24823f`, the commit REP trained at.
* It differs from `aeb54468`'s (`0e4a8cea…`) by DAG-ORC's compose hunks (`e7ea20d`, `8c24823f`): twelve hunks, one adding the module constant `EGO_PERTURB_COMPOSED_STREAM` and eleven inside `EgoPerturb`. They are the same twelve hunk headers DAG-ORC r2's record lists.
* With `--ego-perturb-with-dagger` off and `--ego-perturb-p 0` (DAG-OACC's and every LAUNCH-INTENT run's flags), `apply` returns the batch itself before any draw, and `eligible_free(batch, None)` is `eligible(batch)`.
* DAG-ORC r2's accepted K-BUILD record lists these hunks, each behind `--ego-perturb-with-dagger`, default off, with bit-identity test PASS (`dagorc_t1/kbuild_r2_launch_8c24823f.json`).

**The amendment (one control; nothing else changes).**
* K-BUILD is read by its registered rule. The sentence *"`v8f_mini_data.py` is byte-identical"* is replaced by: **where `v8f_mini_data.py` differs from `aeb54468`, its launch sha256 equals `c117e1cf…751752` (DAG-ORC r2's build), and each of its hunks is listed behind a default-off flag with a PASSing bit-identity test.**
* The reader changes in that clause alone: `BYTE_IDENTICAL_FILES` becomes the pin `LAUNCH_PINNED_FILES`. Its tests and the full mutation pass are re-run (`raw/2026-10-28-launch-intent-amend1/`).

**What it does to the pairs.**
* The replicate pair is unaffected: A_rep and REP have the same data module.
* The main pair carries DAG-ORC's flag-gated data hunks on the A side, as DAG-ORC's own main pair (A − DAG-OACC) did under `PREREG_V8F_DAG_ORC` Amendment 1.
* No bar, row, outcome, predicate or other control changes.

**Binding.**
* The launch-gate tokens bind the PREREG as promoted (sha256 `469bac1f…`) and the reader as landed (`5e950dcf…`).
* The T1-kin chain's own read runs that reader and is expected to REFUSE on K-BUILD alone. Its refusal is banked.
* The read of record runs the amended reader on the same manifest, with this text's sha256 pin. Both readers' sha256 go into the RESULT.

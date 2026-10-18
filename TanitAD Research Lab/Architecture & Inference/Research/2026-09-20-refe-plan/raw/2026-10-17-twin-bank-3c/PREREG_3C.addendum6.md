# PREREG_3C — ADDENDUM 6 (diagnosis agent for the Master Mind, 2026-10-18): G-CLOCK's step-0 bank size, re-derived for the 3-C bank. Written BEFORE any P6 training data exists; no bar changed.

`PREREG_3C.md` (sha256 `e8365f34...`) and addenda 1-5 are NOT edited.

**Why this addendum exists.** The 3C-PASS outcome says P6 "re-gates on it (new data-manifest sha => new launch token)". It did not name the one gate value
that is a function of the bank: G-CLOCK's registered step-0 tuple count. That value lived in `refe/launch_gate.py` (profile `refe_p6`, `expect_schedule.n_tuples`) as
**189,888**, which is the PRE-3C bank `train_grow_cmd` (SPEC_P6_PAPER_FAITHFUL.md section 0.2 l.63 and D8 l.149: 103,037 rank-0 + 86,851 rank-1 rows). The P6
launch-gate dry run at commit `a5a4ee8` on the 3-C bank therefore FAILED with `G-CLOCK FAIL step-0 bank has 167300 tuples != registered 189888`
(pod `lg/evidence_G-CLOCK.json`, 2026-10-18 06:20Z). This is TRAPS "a derived constant silently changes the experiment when its input changes": the input changed and the
constant did not.

**The registration (for the bank `train_grow_cmd_3c` only).** G-CLOCK's step-0 bank = **167,300 tuples in 103,037 scenes**. Epochs 25, steps 10,075, epoch length 103,037
are unchanged: they count scenes, and the 3-C bank has the same scenes.

**Derivation, from the merge's own artifacts (MEASURED, pod, 2026-10-18).**
* rank 0: `targets_rank0.jsonl` copied byte-identical: **103,037** lines, sha256 `daa23b962e95...` = the input's (`merge_report.json` `copied_unchanged`).
* rank 1: input `train_grow_cmd/targets_rank1.jsonl` **86,851** lines (sha256 `2f3e461391d7...`) minus **22,588** dropped lane-rank twins = **64,263** lines
  (output sha256 `2a63acc2f634...`).
  * 22,588 = 44,409 lane-rank twins − 21,821 kept (K1 18,489 / K2 2,781 / K3 551).
  * By reason: `not_committed` 13,887 + 5,171 + 2,795 = 21,853, and `scenario_too_short_10s` 553 + 138 + 44 = 735.
* total: 103,037 + 64,263 = **167,300**.

**Two further readings that do not share the arithmetic, and agree:**
1. `wc -l` of the bank files on the pod: 103,037 and 64,263.
2. The trainer's OWN `bank` event, in the G-LIVE-MEM instrument run on this bank (`mem_metrics.jsonl`, 2026-10-18 06:09Z): `n_tuples` 167,300, `n_scenes` 103,037,
   `per_rank` {0: 103,037, 1: 64,263}. The same 167,300 / 103,037 is in the gate's BUILD capture (`evidence_G-CLOCK.json` `details.bank`).

⇒ 167,300 is the correct count for the 3-C bank. It is not a build defect.

**What "before any P6 training data" means here, disclosed.** Two things have run on this bank, and both are launch-gate instruments:
* the gate's BUILD / SMOKE runs;
* the G-LIVE-MEM run: 4 optimiser steps into a scratch directory that the chain deletes, and that is never a training result.

The P6 trainer has not started. No P6 training metric, checkpoint or evaluation on this bank exists.

**How it is registered in code: per bank, the old number untouched.**
* `refe/launch_gate.py` gains `P6_BANK_TUPLES = {"train_grow_cmd": 189888, "train_grow_cmd_3c": 167300}`.
* The `refe_p6` profile registers `n_tuples_by_bank` in place of a single `n_tuples`.
* `gate_checks.check_clock` resolves the count from the directory name of the argv's `--targets`. A bank with no entry FAILS, so one bank's number never vouches for
  another bank.
* 189,888 stays the pre-3C bank's registered number. `p6_pod.sh`'s default `CBANK` is still that bank.
* Pinned by `refe/test_p6_levers.py` (`profile_p6`: each entry re-derived from its source, and the resolution of five `--targets` paths). Its two new mutations must go
  RED: the 3-C entry carrying 189888, and a lookup that ignores the bank.

**Unchanged:** every PREREG_3C bar and outcome, the bank itself, SPEC_P6's schedule, and every other gate check and threshold.

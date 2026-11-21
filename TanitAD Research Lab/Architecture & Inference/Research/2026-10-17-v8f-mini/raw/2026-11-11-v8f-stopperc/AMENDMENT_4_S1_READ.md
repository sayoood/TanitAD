# STOPPERC amendment 4: the stage-1 read, fixed by the Master Mind before any stage-1 data (2026-11-12 ~21:10)

**Status.** Written 2026-11-12 ~21:10, after the PI's stage-1 go ("Yes, run arm A", ~20:30) and BEFORE any stage-1 run, dump or number exists. Box 2 was still training SLOWWM's last run. The card (`PREREG_V8F_STOPPERC.md`, sha256 `70c75176…8877`) is unchanged. This file fixes how its §5 rules are read, as STOPBAL's r1-r6 did for that card.

## 1. The package
* `stage1_A/read/` (built by a Sonnet agent; pins in `READ_PINS_S1.sha256`): `stopperc_s1_read.py` (sha256 `9da56c03…7531`), `make_tree_s1.py`, `job_dumps_s1.cmd`, `job_stopdec_s1.cmd`, `launch_wmi_s1.ps1`, `README_READ_S1.md` (run steps, the §5 predicates and outcome table copied verbatim, §8's floor).
* It imports the first dose's read (`theory/stopbal/read/`) and dose 2's (`theory/stopbal/dose2/`) by path, unedited.
* **Dump tree: `3e264bbf`** (arm A's commit, child of `715ef8fc`). Arm A's cfg and state-dict keys do not load on `715ef8fc`.
  * MEASURED: the OFF checkpoint BAL-NULL_s0 dumps **141/141 files bit-identical** on both trees, max |diff| 0.0 (`read/identity_BAL-NULL_s0.json`).
* CPU only, so no GPU lock.

## 2. The MM's checks (MEASURED, 2026-11-12 ~21:05)
* `test_stopperc_s1_read.py`: **23 passed** on the MM's own run. Every file matches `READ_PINS_S1.sha256`.
* **Floor mutation** (`floor_max` returns the NULL_rep gap alone, run through `STOPPERC_READ_SRC`): **5 failed, 18 passed**. The five failures are the floor rule and the three between-the-floors cases (delta, AUROC, fpr), plus the missing-floor case.
* The agent's stand-in smoke read: 110 deltas, all exactly 0.0, with 0 control problems. Its P floor reads 0.702720009 m, equal to STOPBAL's.

## 3. Readings s1-s8: accepted as written (`read/README_READ_S1.md`, "Reading choices")
* **s1:** "STOPCUE AUROC" uses STOPCUE's definition: 2,365 has-label windows, 131 `stop_hit` positives. These are the same windows and positives that S0-P scored, so stage 0 and stage 1 read one AUROC. The first dose's M-gate set (117 positives) is read beside; it does not gate.
* **s2:** The card's "the lon-STOP AUROC is not a cost" is part of G-AUROC, so a lon-STOP cost fails the gate. The AUROC interval comes from the paired cluster bootstrap of the statistic. Its floor is the larger of the two replicate gaps.
* **s3:** G-FP is the share of the 2,234 non-stop windows with p(STOP_POINT) >= 0.5. A cost at either seed fails it, since the card's "not a cost" must hold at both seeds. The hit rate is read beside.
* **s4:** Precedence is REGRESSES > ADOPT > P-OTHER-PATH > FP-COSTS > AUROC-ONLY > NULL. Every predicate that holds is printed with its trigger.
  * FP-COSTS heads AUROC-ONLY when both hold literally. It is the more specific row, and its next step (STOPTAU's threshold under its own card) is the one the evidence asks for.
  * A per-seed mix that meets no predicate is SEEDS-DISAGREE, the card's own rule (a third seed, `--seed 10/11`).
* **s5:** The floors are read at seed 0 and applied to both seeds, as in STOPBAL r2. Both replicates (seeds 2 and 3) are seed-0-init runs.
* **s6:** The P ceiling fraction is printed beside the verdict and does not gate.
* **s7:** Each seed has STOPBAL's 48 cells. REGRESSES reads NR only, as the card says.
* **s8:** The controls fail closed, and a failed or unreadable control makes the outcome INVALID. The chain's own gate covers G-SUITE, so the read does not repeat it. STRATEGIC is always printed PAUSED (PI item 25).

## 4. Not verified before the run
* No real stage-1 bank, ON checkpoint or `NULL_REUSE.json` has been read. Their key layouts are inferred from the STOPBAL banks and `stopperc_check.py`. The read's controls 1, 4 and 6 refuse a layout they do not recognise.
* An ON model's strict load and `forward_topk` have not been run on the read tree.
* The ~5 min dump per arm is ESTIMATED (301.5 s MEASURED for BAL-G2_s0).

# PREREG_RA1_FULL Amendment 2: the instruments' proofs (2026-10-20, before any γ = 200 run)

These files prove the instruments. They hold **no result**: no γ = 200 run existed when they were written.

| file | what it shows |
|---|---|
| `mutate_amend2.py` / `mutate_amend2.out` | `code/ra1_full_amend2.py`: **8 / 8 mutations RED** against `stack/tests/test_ra1_full_amend2.py`. The reader is byte-identical after restoration. |
| `mutate_queue_full5.sh` / `mut_q.py` / `mutate_queue_full5.out` | The queue's `full5` stage: **7 / 7 RED** against its dry-run test, including the tip's queue without the stage. The queue is byte-identical after restoration. |

**The 8 reader mutations:**
1. separation not required;
2. floor taken from A1γ200 alone;
3. one seed suffices;
4. built γ not checked;
5. missing run not refused;
6. A1γ200 paired with A1γ50 instead of A0;
7. families paired with themselves;
8. the bar's sign dropped.
   * This one first SURVIVED. The test then gained a separated-WORSE γ 200 world.

**The 7 queue mutations:**
1. the tip's queue;
2. γ hardwired to 50;
3. no missing-γ refusal;
4. wrong run directory;
5. a verdict added;
6. wrong β;
7. one seed.

**Pipeline smoke (NOT a result, not banked as data):**
* The reader ran end to end on REAL arrays: A0 ×2 and A1 (γ 50) ×2 from Amendment 1, with **A2's two runs relabelled as γ 200**, which are the only runs carrying per-window family arrays.
* It produced a reading and a family block with status OK and all 7 LON / LAT components.
* That proves the loaders, the built-γ check, the pairing and the family import on real files. The numbers it printed are A2's, not γ 200's, and decide nothing.

Suites, CPU (`CUDA_VISIBLE_DEVICES=-1`), on a tip-`beba0241` export plus these changes:
* `test_ra1_hjepa.py`, `test_ra1_full_amend2.py` and `test_ra1_full_report.py`: **90 passed**.

## The temperature guard switched itself off, and was re-armed (2026-10-20 22:43, during the γ = 200 runs)

* **What happened:**
  * The guard was started by `chain_ra1_full5.sh` at 22:43. In the same minute it logged *"runs chain ended; guard off (max USB 0 C)"*.
  * Its end test was an UNANCHORED `grep -E '…|ZZRA1A1-DONE|…'` over the chain's log.
  * The chain's first line, *"waiting for ZZRA1A1-DONE in …/chain_ra1_full4.out"*, contains that marker.
  * This is the TRAPS.md class: a polling filter that contains the pattern it searches for matches itself.
  * Amendment 1's chain did not trip it: its waiting line named ZZBETA-DONE, which the guard does not watch.
* **Fix, within about a minute:**
  * The end test is now anchored, `^(ZZRA1FULLRUNS|ZZRA1A1|ZZRA1A2)-(DONE|STOP)`.
  * The guard was re-armed by hand on the same chain log, and logged *"new max USB temp 48 C"*.
  * Training ran unguarded for about one minute. The rule needs 4 of 20 samples (15 s apart) at ≥ 61 °C, or one at ≥ 63 °C, and the box read 48 °C.
* **Pinned:** `stack/tests/test_ra1_box2_guard.py`, 3 tests.
  * A marker matches only at the start of a line.
  * No watched chain's timestamped progress line matches, including its waiting line.
  * Every chain that starts the guard has its own markers in the regex.
  * Both mutations go RED: the deployed unanchored regex, and the A2 markers dropped.
* **`chain_ra1_full5.sh`'s pre-launch check** now looks for the anchored marker group, so a re-run against this guard does not stop itself. The running chain was not edited mid-run.

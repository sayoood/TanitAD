# PREREG_V8F_PROOF_ADD149_AM1 — LEADSPEEDFULL10K, amendment 1: the pre-load memory guard counts the process once

Frozen after the first launch of card ADD149 stopped at its own pre-load guard and before any training step, outcome number or label read of the head exists. The card's other sections are unchanged.

## What happened (MEASURED, `rulefc/leadspeed/box2_full/FULL.json`)

The first launch (2026-11-25T19:03Z, `train_full_box2.py` sha256 71a11908) stopped as `LEADSPEEDFULL10K-STOPPED-MEMORY` before loading any clip: inside the process, commit available read 16.00 GB against the 16.14 GB the card asks for (5.5 GB + 8.14 GB for the 1,943 clips + 2.5 GB for the process). Measured from outside the process at the card's freeze: 17.2 GB; at launch: 17.33 GB. The check runs inside the process after the library imports and the lead file load, so the process's own commit charge (about 1.3 GB by the difference) is counted both in the measured value and in the card's 2.5 GB term. Control c2 passed (0 held-out and 0 eval clips in the fit set). No training step ran.

## The change (control form: the pre-load guard only)

Before loading, commit available must be at least 5.5 GB + 8.14 GB + max(0, 2.5 GB − own), where own is the process's own commit charge at the check (`PROCESS_MEMORY_COUNTERS_EX.PrivateUsage`, `rulefc/leadspeed/memstat.py` `own_commit_gb`). So the process is budgeted 2.5 GB in total, counted once. The per-clip guard (at least 5.5 GB plus the bytes still to load, before every clip), the 5.5 GB floor, the stop name, and the rule that the fit set is never shrunk during a run are unchanged. The values (commit available, own, need) are recorded in the result.

Check of the probe (dev box): a planted 0.47 GB (5e8 bytes) allocation reads +0.467 GB.

## Unchanged

The head, labels, split, 1,500 fit and 443 held-out clips, 30,000 steps, schedule, checkpoints, reads S and A, controls c1 to c4, the bars (0.30), the names and the consequence.

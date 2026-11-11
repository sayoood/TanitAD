# Test results, STOPBAL second-dose package (2026-11-11; code of record 715ef8fc; amendment sha256 460fd548... re-hashed, equal)

| test | command | result |
|---|---|---|
| bash syntax | `bash -n` on `dose2_chain.sh`, `stages/*.sh`, `tests/test_dose2_dryrun.sh` | all ok |
| unit | `python -I tests/test_dose2_unit.py <scratch>` | `UNIT2 ok=194 bad=0` |
| dry run (stubs, no box call) | `bash tests/test_dose2_dryrun.sh` | `DRYRUN2 ok=102 failed=0` |
| read additions (planted fake banks) | `python -I read/test_dose2_read.py <scratch>` | `READ2 ok=114 bad=0` |

Found by the dry run and fixed before the last pass: the first dose's checker read the data paths from a `BAL-NULL_s0` token that this table does not carry (stage crash on the box at the pin step); `stopbal_check.py` now patches `data_paths`, and the unit test pins it.
Covered by the dry run: normal flow (ship, pilot, 12 h permit, four runs, banks, cleanups, isolation of the SLOWWM folder and the shared repo); SLOWWM gate waiting (no marker, other chain's marker, marker without prefix, STOP with head rc=12 with and without detail, busy marker held) and released (DONE, STOP with another head, DONE written while the chain polls);
state-dir collision guard (three dirs); box-side pin (changed seed, first dose's 4.6991 in the L2 argv, changed argv file, changed bank); red arms (wrong weight at the pilot gate for G2 and for L2, wrong weight in the trained checkpoint, wrong commit at ship incl. the real mirror's parent 5514a684 refused and 715ef8fc accepted,
missing / changed NULL bank in five ways); 12 h rule; disk gate; transport fail-closed; first-dose and NULL tags refused.
Not run (by instruction): any contact with box 2, any real training. The dry run's trainer is a stand-in; the real trainer, GPU and box-2 disk behaviour are as measured at the first dose.

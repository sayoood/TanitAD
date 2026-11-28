# AMEND_5 read: the MM's launcher fix (2026-11-17)

The first launch (14:52Z) stopped in the test step before any work: `test_plantest2_controls.py` asserts that `tanitad` comes from PLANTEST's pinned export (`plantest/plantest_real.py:13`). The builder had added that test to `run_amend5.sh` and never ran it ("not re-run by me", BUILD_AMEND5.md). That export (`F:/TanitAD-scratch/plantest_tree`, `git archive 26fb0bc1 stack/tanitad taniteval/taniteval`) was deleted by PLANTEST at its end, as its handoff planned.

* The MM re-exported it from git (38 MB; `stack/tanitad/__init__.py` blob 002bcd76 equal to 26fb0bc1's). The test passed there: 14 passed in 607 s.
* `build_mmfix/run_amend5.sh` runs that one test in its own tree, behind a blob check. Every other test and stage runs as built (ladder_tree, the code of record). The diff against the builder's launcher is the one block in the test loop.
* The relaunch (~16:35Z) ended `ZZ AMEND5 READ DONE ZZ`. Nothing in planning, scoring or the readout changed.

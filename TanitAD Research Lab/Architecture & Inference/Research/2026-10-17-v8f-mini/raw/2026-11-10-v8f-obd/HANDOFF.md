# HANDOFF: OBD launch package (built 2026-11-10, no box-2 contact)

State: package complete and tested over stubs; **cannot launch CTL / BAND arms at `202849da`** (purity probe does not read `--w-acc`; `PURITY_ACC_FINDING.md`). Nothing was started, stopped or touched on any process; nothing in git, `stack/`, `theory/otband/`, the card, `egowm_state`, `box2_ship`, `qland` or the mirror was changed.

## Where things are (all under `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/obd/`)
* `argvs_obd.json|txt|meta.json`, `make_argvs_obd.py` (needs an export of 202849da at `OBD_TREE`; asserts trainer sha256 `1ddf491d...`)
* `PURITY_ACC_FINDING.md`, `purity_acc/{PURITY_ACC_PROPOSED.diff, test_v8f_oband_acc.py, measure_acc_purity.py}` (proposal only)
* `box2/` the chain package: `obd_chain.sh`, `obd_check.py`, `obd_ship_tree.py`, `obd_refresh.py`, `stages/`, `jobs/`, `frozen/argvs_obd.json`, `tests/`, `README.md`, `PACKAGE_MANIFEST.sha256`
* `IMPL_CHOICES_OBD.md` (+ `.sha256`): every choice the card does not fix, and the proposals for the card (NULL parity basis; name the probe commit).

## Exact next steps
1. The held `stack/` session lands `PURITY_ACC_PROPOSED.diff` + `test_v8f_oband_acc.py` (a new commit; run the stack tests). The Master Mind names that commit in the card and freezes it.
2. Re-run, with `OBD_COMMIT=<new sha>`: the unit test and dry run (README, "Tests"); the package needles are in `obd_check.NEEDLES`. If the new commit changes `v8f_mini_train.py`, re-check `make_argvs_obd.py` (it asserts the old trainer sha: update `TRAINER_SHA256_AT_COMMIT` and re-run; the argvs should not change).
3. Launch line: README, "What the Master Mind does before launch". The chain waits for `ZZ EGOWM-DONE` / `ZZ EGOWM-STOP` and a free `BOX2_BUSY.txt` by itself (up to 96 h of polling).
4. After OBD-NULL_s0 banks: read `C:/Users/Admin/obd_state/NULL_PARITY.json`. Expect `file_sha_equal: false` (the checkpoint embeds `--out` via `argv_sha`; IMPL_CHOICES_OBD 6), hence OBD-NULL_s1 is appended; `tensor_equal` says whether the weights are identical.

## How to check a running chain (script, not turns)
`C:/Users/Admin/obd_state/{chain.log, STOP.txt, DONE.txt, ARM_REFUSED_*.txt, NULL_PARITY.json, pilot/PILOT.verified.json}`; sentinels are in their own files only.

## Temporary data
Scratch under `F:/TanitAD-scratch/obd_build/` (tree export of 202849da, unit and dry-run scratch) is deleted at the end; deletions are listed in the hand-back.

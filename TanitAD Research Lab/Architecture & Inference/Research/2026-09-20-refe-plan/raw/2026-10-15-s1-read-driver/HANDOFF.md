# HANDOFF (2026-10-15): S1 read driver built; steps 1-3 done; step 4 (real H2-s0 plumbing) NOT run

## State
- `s1_read.py` + `test_s1_read.py` (`ZZTEST_PASS 94`) are in this folder (E: working tree, unstaged by instruction; the Master Mind lands through the mirror, copies destined for `eval/`).
- Dry run on N-s0 / N-s1 / N-s2 done (`dryrun/`): INCOMPLETE, 6 arms missing, 27 files / 44.66 MB verified, 72 s. The head path was proved on an untrained head (`dryrun/headpath_standin/CONTROL.md`).
- `arms/H2-s0/` was **not on HF** at 06:24Z (HF listing: only N-s0, N-s1, N-s2). Expected about 07:25Z (training began 05:24:53Z, about 2 h) plus the push.

## Next step (a fresh agent, about 12 min, background it)
1. The driver lists HF itself: an arm that is not (completely) on HF shows in `S1_READ.json -> arms.partial_on_hf` and a `--plumbing-check` on it returns SKIPPED. So just run step 2; if `partial_on_hf` names H2-s0, wait and rerun (a PARTIAL push has some files but not `run_check.json`).
2. From a tree that holds the landed or copied `eval/s1_read.py`:
   ```
   cd "<package root>"
   HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9]+' E:/Projects/TanitAD/Keys.txt | head -1)" PYTHONIOENCODING=utf-8 \
     C:/Users/Admin/venvs/tanitad/Scripts/python.exe eval/s1_read.py read --arms N-s0 N-s1 N-s2 H2-s0 \
     --plumbing-check H2-s0 --cache D:/Projects/TanitAD-artifacts/s1_read_cache --out raw/2026-10-15-s1-read-driver/dryrun_h2s0
   ```
   (the temp working copy is `C:/Users/Admin/AppData/Local/Temp/s1rd/.../eval/`, identical to this folder's files; `S1_EVAL_DIR` points the driver at an `eval/` dir.)
3. Report ONLY plumbing facts: did the H2-s0 evidence pass `check_arm_evidence` (a REFUSED here is the finding: read the code), did `forward` rc 0 with identity PASS, did `perception_navtest` rc 0. ⛔ Do not open `instrument/plumbing/H2-s0/perception_navtest.json` for values and do not write an H-vs-N statement: the set is incomplete and the prereg's first read is after all nine arms (A5).
4. If the evidence checks REFUSE a genuine H2-s0 on a field name (UNVERIFIED in `NOTES.md` 7), fix the name in `s1_read.py`, add the case to `test_s1_read.py`, rerun the suite (about 2 min: `PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe test_s1_read.py`).

## Open items for the Master Mind
- Land the two scripts; decide AMB-1..4 (`NOTES.md` 6) before the full read; optionally land the 3-line `rl_val_eval.load_scores` fix (`NOTES.md` 1.2).
- The full nine-arm read is ESTIMATED at 25 min; its command is in `NOTES.md` 8.

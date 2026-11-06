# Brief: finish the stage-1 box-2 package (AGENT E; build and rehearse on CPU; do NOT launch)

**Rules.** Everything in `BRIEF_T2V8F_E3E6.md` and `BRIEF_T2V8F_STAGE1.md` (same folder) binds:
* the preamble;
* CPU only (`CUDA_VISIBLE_DEVICES=-1`), no ssh to box 2;
* never commit or stage, never delete (move aside), LF line endings, banned words;
* no downloads and no new packages;
* ≤ 150k context: at about 130k, write `HANDOFF_E.md` and end.

**Your plan is `HANDOFF_D.md` (same folder).** Read its DONE / WRITTEN / LEFT sections first. You own every file in `t2v8f/`; no other
agent edits them. The Master Mind (MM) is running the full suite now; its log is `logs/pytest_mm_post_d.log`. Read the log before you
edit anything that the suite covers: if a test in gates / e4 / e5 / e5b is red, fixing it is item 0.

## Items, in order
0. **Code written but never run.** Run `python -I code/t2v8f_nocv2.py`, then compile-check `code/t2v8f_stage1.py`, and fix their
   first-run errors.
1. **The package** (`HANDOFF_D.md` LEFT 1). Write `code/make_stage1.py`, then build ONE tar at `t2v8f/stage1/t2v8f_stage1.tar` with a
   `.sha256` beside it.
   * Assert the BLIND-mean sha256 named there; the zeros stand-in must never ship.
   * Report the tar byte size and the allocated footprint on box 2's 1 MiB-cluster D:. The footprint must leave ≥ 10 GB free (45.4 GB
     free at the last probe).
2. **MM ruling, the planted no-lead (card section 6b, 12:3x entry).** The deciding planted read is condition `empty` rendered at the
   SAME pose and tick as the `objects` read. The 30 s-past-the-end clock read is reported beside and never decides. The thresholds stay
   as committed in `t2v8f_stage1.py`'s docstring.
3. **Rehearsal** (`HANDOFF_D.md` LEFT 2): `tests/test_t2v8f_stage1.py` on a SHADOW kit; never write into the real kit at
   `C:/Users/Admin/t2a_scratch/box2_kit`. Every gate must read correctly on the stub, and every arm below must go RED:
   * the planted missing `ckpt/DAGOACC_s1.pt` (`STAGE1.fail`, MISSING, preflight only);
   * a changed byte (HASH_MISMATCH);
   * a shifted ROI (actor FAIL);
   * an `empty` frame that draws the lead (actor FAIL);
   * `check_score_e2e` with its two mutations.
4. **`STAGE1_LAUNCH.md`** (`HANDOFF_D.md` LEFT 3):
   * exact commands, the marker meanings, ESTIMATED wall time, and the pull and dev-box score commands;
   * a note that box 2's venv imports (gsplat, msgpack, transformers, PIL, decoder) are first checked by the preflight.
5. **Close-out:** run the whole suite once at the end (`pytest tests -q`), refresh `PINS_E.txt` (sha256 of every file written or edited),
   and hand back.

## Hand-back (≤ 300 words)
Tests (n passed) and mutations, the rehearsal result, the tar path, size, footprint and sha256, the exact launch steps, open items,
blockers, and a deliverable manifest.

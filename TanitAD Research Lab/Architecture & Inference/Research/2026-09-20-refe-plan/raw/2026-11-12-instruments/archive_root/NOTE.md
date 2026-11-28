# teacher_navtest_seam.py: the navsim archive is found by the file, not by the folder (2026-11-12, MM)

## What happened
- `eval/teacher_navtest_seam.py:48` hardcoded `EXPORT = E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz`. The archive moved to F: on 2026-10-25.
- To let TF-N2 roll, the MM copied that one file to the same path on E: (8,188,529 B, sha256 `b0dc05f3…`, equal to the F: source).
- The pinned harness (`eval/harness/w3e/code/run_v1.py:42-44`) takes the FIRST EXISTING root folder among E:, F:, D:. The copy made the E: root exist, so the harness looked for `navsim-3e8291…/…/navtest.yaml` on E:, where it is absent.
- Effect: TF-N2 A1 `score_T` failed, 2 rounds of 6 attempts, rc 1, no CSV. No N2 arm was scored.
- Rolls and build were not affected; C-R0 and C-BUILD passed.
- Repair on the dev box: the E: copy and its folders were deleted. The chain was resumed at `score_T` with the same command, and the harness root is now F:, as it was in TF-N1 A1. C-ID-H is the control that would catch any root effect.

## The fix landed here
- `resolve_archive_file(rel)` returns `$TANITAD_NAVSIM_ROOT/<rel>` if that variable is set, else the first `<root>/<rel>` among E:, F:, D: that exists as a FILE.
- If no root holds the file, it returns the E: path, so the open fails and names the path.
- It changes paths only; the file's bytes are unchanged.
- On the dev box `EXPORT` now resolves to `F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz`.

## Tests
`eval/test_teacher_seam_archive_root.py`: 6 tests OK on the dev box (driverl-eval venv, `-W error::ResourceWarning`). They include:
- a partial root that holds the folders but not the file must not shadow the root that holds the file;
- a mutation arm that swaps the file test for the old folder test and must reproduce the shadowing (it does).

## Not changed
- About 20 files under `eval/harness/w3e/` hardcode `E:/Archive` or resolve by folder, among them the pinned `run_v1.py`. They wait for a harness re-pin.
- Until then, never create anything under `E:/Archive/devbox-C/navsim` on the dev box.
- Run trees pinned before this commit keep their own copies; a new card re-pins `w/pk/eval/teacher_navtest_seam.py`.

# BRIEF: register the C1 seed-1 replicate `refe_p5c_c1r` (arm "C1r") in the REFe launch gate, with its pod argv and tests. Agent: Sonnet. MM, 2026-11-21

Read `Project Steering/AGENT_OPERATING_STANDARD.md` first (from the mirror: `git -C C:/Users/Admin/tanitad-push show 5fe83fe4:"Project Steering/AGENT_OPERATING_STANDARD.md"`). **Binding:** never commit, push or `git add`; the MM lands. "Stage" here = write the file whole into the hold below. End with a deliverable manifest. Hand-back <= 300 words + paths. Stay under ~150k context. Never wait alive on a job > 4 min: write `HANDOFF.md` in the hold and end.

`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. The tip is `5fe83fe4` (its PK tree is byte-identical to `ad30f516`).

## Why
PI go 2026-11-20 (*"Follow your recommendations"*): a **C1 seed-1 replicate**, run on the pod after the field chain. P5-C's C1-vs-C0 delta (+1.17 PDMS [0.67, 1.65], T1) is one training seed. The episode bootstrap cannot see training variance (`H-ESTIM-SEED-1`), so C1r is the arm that reads that delta against the rig's own seed noise. C1r = **C1 exactly, plus `--seed 1` and its own `--out`**. This mirrors how `refe_p5c_c0r` relates to `refe_p5c_c0`.

## Where to work
- **Tree:** `C:/Users/Admin/swp/q`, the MM's test tree, whose PK equals the tip. Before you edit any file, copy it to `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/c1r/orig/<repo path>`.
- **Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/c1r/land/<repo path>`. Put every modified or new file there whole, with LF bytes.
- **Scratch:** `C:/Users/Admin/swp/c1r_scratch/` only. ⚠️ C: has about 6 GB free. Before each test, require C: free >= 4 GB; if it is lower, stop and report.
- **Test temp:** run each test with `TMP`, `TEMP` and `TMPDIR` pointed at `C:/Users/Admin/swp/c1r_scratch/tmp_<test>`, and delete that directory after the test, by its literal path.
- **Test runs:**
  - Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`. `PYTHONPATH` is `;`-separated Windows paths (`cygpath -w`) for `<tree>/stack`, `<tree>/PK/refe` and `<tree>`. Set `CUDA_VISIBLE_DEVICES=-1`, and run from `<tree>/PK/refe`.
  - One test at a time, under the shared mutex: `mkdir E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/pytest.lockdir` before the test, `rmdir` it after. Before that, wait until `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain/tools/commit_guard.py --min 6` exits 0.
  - Template: `C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/e07efd49-d158-4484-85f3-d8888196bc46/scratchpad/sweep2.sh`, which you may read.
  - Many REFe tests are scripts: if pytest exits with rc 5, run `python <file> [--fast]`.
- Do not touch the pod, HF, git, or any other directory.

## Deliverables (priority order; bank each into the hold as it is done)
1. **`PK/refe/launch_gate.py`** (P5-C block, about lines 431-566):
   - Add `"C1r": 1` to `P5C_ARM_SEED` and `"C1r": "refe_p5c_c1r"` to `P5C_PROFILE_NAME`.
   - Introduce one named set of table-carrying arms, `{"C1", "C1r"}`. Use it at every place that now tests `arm == "C1"`: `p5c_argv`, `p5c_profile`, `p5c_refusals`, and anywhere else you find.
   - Register the profile in the `for _arm in (...)` loop.
   - Extend the block comment so it reads `argv(C1r) - argv(C1) = {--seed, --out}`, and cite the PI go.
   - ⛔ **The 37 existing profiles must not change.** Compute `sha256(repr(sorted(PROFILES[n].items())))[:16]` for every profile on the unmodified tree, then on yours. All 37 must be equal, and exactly one new name may appear. Bank both lists as `fingerprints_before.txt` and `fingerprints_after.txt`.
2. **`PK/refe/gate_checks.py`** and any other implementation file that special-cases C1 or enumerates the P5-C arms: give C1r C1's treatment. For example, the label-table G-DVB rows and `p5c_live_rows` (h). `grep -n '"C1"\|p5c_arm\|P5C_PROFILE_NAME\|refe_p5c' PK/refe/*.py` gives you the census; bank the census too.
3. **Tests.**
   - `test_p5c_profile.py`: add C1r and these deliberate-regression arms, each of which must be REFUSED:
     - C1r with `--seed 0`;
     - C1r with no `--traj-label-table`;
     - C1r with one extra lever;
     - C1's argv under the C1r profile;
     - C1r's argv under C1's profile.
     The real C1r argv must PASS `p5c_refusals`.
   - Pinned-set tests: every test that pins the profile set or count (37 -> 38). Find them with `grep -ln "len(L.PROFILES)\|refe_scorerft_c_s1\|PINNED_FP\|TIP_FP" PK/refe/test_*.py`. The last such edit is in `.../scratchpad/fix_pins.py`: follow its exact-once replacement pattern.
4. **The pod argv.**
   - Inputs, as pulled by the MM: `.../c1r/pod_inputs/ARGS_C1.txt` (sha256 ca370d07...), which is C1's REAL pod argv, and `c0r_mkargv.py`, which is how C0r was derived from C0.
   - Write `c1r_mkargv.py` in the same style. It reads `ARGS_C1.txt`, changes exactly `--seed 0` -> `1` and `--out .../p5c_c1` -> `.../p5c_c1r`, asserts that the token diff is exactly those two values, and prints `ZZC1R_ARGV_OK <sha256>`.
   - Run it locally on the pulled `ARGS_C1.txt`. Then check that the output passes `p5c_refusals(PROFILES["refe_p5c_c1r"], argv)`, apart from the fresh-`--out` and file-existence checks, which are pod-only; state which checks were skipped. Bank `ARGS_C1R.txt`.
5. **Run the targeted tests** on your tree, whole files: `test_p5c_profile.py`, `test_launch_gate.py`, `test_p5c_live.py`, `test_p5c_postgate.py`, `test_p5c_chain.py`, and every pin test you edited.
   - Record each as `<file>\t<mode>\t<rc>\t<last line>` in `c1r/targeted_results.tsv`.
   - A failure that also fails on the unmodified tree is pre-existing: prove that by running it on the `orig/` bytes, and mark it as pre-existing.
   - ⛔ Do NOT run the full ~55-file REFe sweep. The MM runs it from your hold.
6. **`REPORT_C1R.md`:** the diff summary, the census, the fingerprint check, the tests, and anything you could not do.

## Rules
- Mutation discipline: each new check must go RED when the line it guards is deleted. Show one such mutation for the C1r seed check.
- Do not edit `PREREG_*`, `GOALS_AND_CLAIMS.md`, the registry, or any banked `raw/` artifact.
- If C1 is special-cased somewhere you cannot safely generalise, stop and report it; do not narrow the scope silently.

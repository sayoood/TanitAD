# REPORT_C1R (REFe agent, 2026-11-21)
Goal: register launch-gate profile `refe_p5c_c1r` (arm C1r = C1 + `--seed 1` + own `--out`, PI go 2026-11-20). Nothing committed or `git add`ed.

## Diff summary (hold: `land/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/`; originals in `orig/`)
- `launch_gate.py`: `P5C_ARM_SEED["C1r"]=1`, `P5C_PROFILE_NAME["C1r"]="refe_p5c_c1r"`, new `P5C_TABLE_ARMS = frozenset({"C1","C1r"})`; all 8 `arm == "C1"` / `!= "C1"` sites in `p5c_argv`, `p5c_profile`, `p5c_refusals` now test it; C1r registered in the `for _arm` loop; block comment extended (`argv(C1r) - argv(C1) = {--seed, --out}`, PI go cited). `--out` is `p5c_c1r` via the existing `arm.lower()`.
- `gate_checks.py`: only C1 special case is G-LIVE (h) `needs_table` (line 2605): `p5c_arm in ("C1","C1r")` (gate_checks does not import launch_gate, so the set is repeated literally and cross-referenced).
- `test_p5c_profile.py`: C1r in NAMES / levers / fresh-out / pins loops; argv delta C1r-C1 = {--seed,--out}; real C1r argv PASSES `p5c_refusals`; REFUSED: C1r seed 0, C1r without table, C1r + extra lever, C1 argv under C1r profile, C1r argv under C1 profile (and under C0/C0r); 3 new mutations (two on the C1r seed, one on P5C_TABLE_ARMS); existing mutation strings updated to the new source text.
- Pin tests 37 -> 38 (exact-once pattern): test_div/dose/vdac/s1zsm/scorer1p6_gate (set + count), test_field_p7_profile, test_rep_gate (count 38; the "not in listed" difference stays 19), test_scorerft_profile (FP_PINS gains `refe_p5c_c1r: 4a76a09574fbc144`; the two source-string pins on the (h) line updated).

## Census (`census_C1_impl.txt`)
Implementation files only: launch_gate.py, gate_checks.py:2605/2608, p5c_pod.sh:109. The `RLS1*_ARM_*` "C1" entries are a different arm namespace (RL-S1), untouched. `scorerft_argv/profile` build from `p5c_argv("C1")`/`PROFILES["refe_p5c"]`, which are unchanged.
**NOT generalised (reported, not silent): `p5c_pod.sh`** (`ARMS="C1 C0"`, `arm_profile()`, the C0r gating at line 339) and `test_p5c_chain.py`. C1r is not in the pod chain; launch it via `launch_gate.py run --profile refe_p5c_c1r` with `ARGS_C1R.txt`, or extend the chain (MM decision; needs its own chain tests).

## Fingerprints
`fingerprints_before.txt` (37) vs `fingerprints_after.txt` (38): all 37 equal; the only new line is `refe_p5c_c1r 4a76a09574fbc144`.

## Pod argv
`c1r_mkargv.py` reads ARGS_C1.txt (sha256 ca370d07...), builds `p5c_argv("C1r", paths=<C1's own paths>)`, asserts the diff is exactly `--seed 0->1` and `--out .../p5c_c1 -> .../p5c_c1r`. Run locally (C1R_PKG_REFE/C1R_DIR/C1R_OUT env; pod defaults otherwise): `ZZC1R_ARGV_OK 2ae47b6e9bdf462ea6e086f4b6aa176a050f8850b53fd4844e77a1ee076d3eb3`; `ARGS_C1R.txt` banked. `p5c_refusals(PROFILES["refe_p5c_c1r"], argv)` = [] and `profile_refusals` = [] with the real pins (base 2613fc17, table 3b198314, rank0). Pod-only, NOT verified here: the fresh-`--out` (no ckpt_last.pt) check passes only vacuously because the /workspace path does not exist locally, and path-file existence / data fingerprints.

## Tests (`targeted_results.tsv`)
All pass except `test_launch_gate.py`: PRE-EXISTING, identical `FileNotFoundError D:/Projects/TanitAD/data/refe_consistent/targets_rank1.jsonl` (gate_fixture.CONSISTENT) on a copy with the orig launch_gate/gate_checks (`orig_launch_gate_result.tsv`), so that file's coverage is not exercised on this box. test_rep_gate first failed from my wrong pin (20); fixed, rerun green. Full mutation harness: test_p5c_profile ZZP5CPROF_PASS 334 (16 mutations RED), test_scorerft_profile ZZSCORERFTPROF_PASS 272.
Mutation of the C1r seed check: `P5C_ARM_SEED ... "C1r": 1 -> 0` makes the `argv_delta` check go RED (log: "mutation RED as required ... -> argv_delta", twice: C0r and C1r variants).

## Not done / notes
p5c_pod.sh chain extension (above). Not run: the full ~55-file sweep (MM). All files LF. Scratch (`C:/Users/Admin/swp/c1r_scratch`) deleted except my small scripts; per-test TMP dirs and the orig-root copy (~30 MB) deleted.

# Red-suite triage for the DAgger integration proof (2026-10-27, Master Mind)

**Question.** The proof of record (`../proofs/FINAL.txt`) ends `NOT ALL OK`: six suites of `p3` read FAIL. Which of them does the DAgger change
cause? `FINAL.txt` is banked unchanged. This file explains its FAIL rows; it does not replace them.

**Why `p3` could not answer it.** `di/tree` was a PARTIAL export: 1 of the tip's 96 `Project Steering/PREREG_*.md` files, no
`2026-10-17-refav1-hj-mini/code`, and (in one suite) `tanitad` imported from the venv's G: install instead of the tree's `stack/`.

**Method [MEASURED].** Two trees, both `git archive` of live tip `4993ae0b` holding everything these suites read (4,200 files, blob ids
verified): `pristine/`, and `dagger/` = pristine + the 5 DAgger code files from `di/tree` (each `cmp`'d). Each of the six suites ran on both
trees, with `tanitad.__file__` probed from inside the run to show it resolves to the tree's own `stack/` (all 12 runs).
Outputs: `pristine__*.out`, `dagger__*.out`, `*.probe.json`; the table: `COMPARE.txt`.

| suite | partial tree (`p3`) | pristine p / f / e | dagger p / f / e | cause of the `p3` red |
|---|---|---|---|---|
| test_v8f_oacc | 5 failed | 133 / 0 / 0 | 133 / 0 / 0 | `PREREG_V8F_OACC.md` absent from the partial tree |
| test_v8f_lora | 15 failed | 38 / 2 / 0 | 38 / 2 / 0 | 13: `refav1-hj-mini/code` absent; 2: already red on the tip |
| test_v8f_lora_ar2 | 2 failed, 14 errors | 64 / 0 / 0 | 64 / 0 / 0 | `refav1-hj-mini/code` absent |
| test_v8f_speedavg | 1 failed | 53 / 0 / 0 | 53 / 0 / 0 | `PREREG_V8F_SPEEDAVG.md` absent |
| test_v8f_kindx_s1 | 41 failed, 3 errors | 187 / 0 / 0 (+1 skipped) | 187 / 0 / 0 (+1 skipped) | `tanitad` imported from G:, which has no `v8f_mini` |
| test_v8f_kscore | 2 failed | 123 / 0 / 0 | 123 / 0 / 0 | `PREREG_V8F_KSCORE.md` absent (also `kscore_full_tip_plus_dagger.out`: 123 passed) |

⇒ **No red is caused by the DAgger change.** Exactly one test differs between the trees, and only in its message:
`test_v8f_lora::test_the_encoder_build_never_moves_the_heads_initialisation` fails on BOTH with `spy() takes from 5 to 6 positional arguments`
(8 given on pristine, 9 with DAgger).

**The two reds already on the tip (both in `test_v8f_lora`, neither DAgger's):**
1. **The stale spy.** The test's stub has the signature `declared_vs_built` had before O-RECOVER (`9871325`) added `ego` and `sched`; DAgger
   adds `dag`. Fixed in this landing: the spy forwards `*rest, **kw` to the original. With the fix, on the DAgger tree:
   `dagger_spyfix__test_v8f_lora.out` (39 passed, 1 failed).
2. **`test_end_to_end_identity_the_on_the_fly_feats_are_the_caches_feats_exactly_and_a_one_step_shift_is_caught`** reads
   `max_abs 0.015625` on both trees. Not DAgger's, not fixed here; a follow-up on the LoRA arm.

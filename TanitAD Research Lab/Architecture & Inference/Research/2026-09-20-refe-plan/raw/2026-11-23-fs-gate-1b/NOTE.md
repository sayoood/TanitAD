# Fs hybrid gate 1b (MM 2026-11-23)
H = 10848403249a0c3d9beabbba70ac56d86bac78e6's tree + 2 files (PK/refe/): gate_checks.py, test_field_p7_profile.py.  Nothing else touched; no commit/branch/push; pod not touched.

- gate_checks.py: `field_reach` + `judge_field_reach` bodies replaced verbatim by 95b3d174's.  gate_checks.diff = 2 hunks (@@ -1953 and @@ -1984) only, both inside those two functions.  ast.parse OK, LF only (0 CR).
- test_field_p7_profile.py: + imports (types, SimpleNamespace, gate_checks as GC), + `make_model` and `chk_field_reach_active_sample` (verbatim from test_scorerft_profile.py), registered as CHECKS["field_reach_active"], + ONE MUTATION (always probe idx[0]) targeting gate_checks.py.  Only existing line changed: the CHECKS tail line (registration).  Mutation goes RED (full run prints "mutation RED as required ... field_reach_active").
- Fs's G-SUITE (13 entries, MEASURED from 10848403 PROFILES['refe_field_p7_s1']['suite']) + test_field_p7_profile.py FULL, serial, TMP on E:, >= 5.5 GB commit wait: tests.tsv.  12 of 13 suite runs + FULL pass.
- ONE environmental failure: test_p2r_chain.py -> OSError WinError 87 at `os.utime(ckpt, ns=(1, 1.7e18))`: E: is exFAT and cannot store a 1970 atime.  Independent of both overlays (neither touches that test).  Supplementary rerun with os.utime atime clamped to 1980 (utime_shim.py): ZZTEST_PASS 139 (tests_supplementary.tsv).  On the pod's real filesystem this test needs no shim; the MM may want one native-FS confirmation.
- First harness attempt exported only refe/ (test_p2r_chain and test_launch_gate read PKG/eval, code, splits, SHIP_MANIFEST): aborted, rerun on the whole PK minus raw/ + media/ (prep_fs.py).

## MM addendum (2026-11-23)
The hybrid tree is commit `86d174bbd0eeffee83700b5b0f99332087d40771` on branch `agent/fs-replicate-gate1b` (parent 10848403; tree aabec6be; only the two files above change; blob sha256 == this hold). Fs ships from it as package E (`code_86d174bb`). test_p2r_chain.py is not touched by either file; G-SUITE on the pod (native filesystem) runs it for real at the Fs gate.

"""Source-level DELIBERATE REGRESSIONS on the tree: each mutation re-introduces a defect the round-3 code guards against;
the named tests must go RED (pytest exit 1), then the file is restored byte for byte."""
import json, os, subprocess, sys
from pathlib import Path
TREE = Path("F:/refcv9_code3/tree/stack")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
env = dict(os.environ, PYTHONPATH="F:/refcv9_code3/tree/stack;F:/refcv9_code3/tree/taniteval",
           TMP="F:/c_overflow_2026-10-26/tmp", TEMP="F:/c_overflow_2026-10-26/tmp")
T = "tests/test_r9_train_only.py"
MUTS = [
 ("M1-round2-planner-freeze", "tanitad/train/map_only_ft.py", "        want = is_trained(n, mods, dec)\n",
  "        want = is_trainable_under(n, mods)\n",
  [T + "::test_R8_RIG_the_trainable_frozen_and_declared_counts_of_each_set",
   T + "::test_REAL_BUILD_the_trainable_frozen_and_declared_counts_of_each_set",
   T + "::test_DECLARED_the_built_declarations_inside_the_sets_stay_frozen_and_every_check_is_clean"]),
 ("M2-emit-start-0-admitted-for-all", "scripts/refc_v3_train.py",
  "    _es0_r9 = _es is not None and int(_es) == 0 and _mof.is_r9(args)",
  "    _es0_r9 = _es is not None and int(_es) == 0",
  [T + "::test_R9W_EMIT_START_zero_is_admitted_under_the_r9_freeze_only"]),
 ("M3-emit-start-0-refused-under-r9", "scripts/refc_v3_train.py",
  "        if int(_es) < 1 and not _es0_r9:", "        if int(_es) < 1:",
  [T + "::test_R9W_EMIT_START_zero_is_admitted_under_the_r9_freeze_only",
   T + "::test_GATE_the_warm_start_launch_argv_has_no_refusal"]),
 ("M4-schedule-end-is-steps", "scripts/launch_gate.py",
  "        end = int(n[0]) // int(ru[\"end_div\"])", "        end = int(n[0])",
  [T + "::test_GATE_R9W_SCHEDULE_the_end_step_is_floor_steps_over_two",
   T + "::test_GATE_the_warm_start_launch_argv_has_no_refusal"]),
 ("M5-refcv8-family-admits-r9", "scripts/launch_gate.py",
  "((\"--r9-train-only\",), \"the refcv9 warm-start freeze", "((\"--r9-train-only-NOT\",), \"the refcv9 warm-start freeze",
  [T + "::test_the_refcv8_family_profiles_refuse_the_r9_freeze"]),
 ("M6-gate-ignores-declared-literal", "scripts/launch_gate.py",
  "    if built_decl != sorted(decl):", "    if False and built_decl != sorted(decl):",
  [T + "::test_GATE_G_DVB_holds_the_built_declarations_to_the_profiles_literal_on_the_refcv8_rig"]),
 ("M7-gate-glive-declared-trainable", "scripts/launch_gate.py",
  "        return not _under_any(n_, mo_mods) or _under_any(n_, mo_decl)", "        return not _under_any(n_, mo_mods)",
  [T + "::test_the_gate_reads_declared_subtrees_as_frozen_in_G_LIVE"]),
 ("M8-AT-ignores-also-frozen", "scripts/assert_map_only_finetune.py",
  "        return _under(k, mods) and not _under(k, also)", "        return _under(k, mods)",
  [T + "::test_ASSERT_TOOL_r9_also_frozen_holds_a_declared_subtree_bit_equal"]),
 ("M9-AT-require-moved-inert", "scripts/assert_map_only_finetune.py",
  "    if require_moved and unmoved:", "    if False and require_moved and unmoved:",
  [T + "::test_ASSERT_TOOL_r9_require_each_module_moved"]),
 ("M11-B5-shared-env-pin", "scripts/launch_gate.py",
  "registered skip is registered here as well\n    suite_env_failures=(",
  "registered skip is registered here as well\n    _unused_env_failures=(",
  [T + "::test_B5_the_profile_resolves_to_refcv8_mapfts_env_failure_pin"]),
 ("M10-guard-ignores-declared", "tanitad/train/map_only_ft.py",
  "_mof_PLACEHOLDER", "", []),
]
out = []
for name, rel, old, new, tests in MUTS:
    if not tests:
        continue
    p = TREE / rel
    orig = p.read_bytes()
    crlf = b"\r\n" in orig
    o = old.replace("\n", "\r\n") if crlf else old
    nw = new.replace("\n", "\r\n") if crlf else new
    s = orig.decode("utf-8")
    n = s.count(o)
    if n != 1:
        out.append({"mutation": name, "error": f"anchor matched {n}"}); print(name, "ANCHOR", n); continue
    p.write_bytes(s.replace(o, nw).encode("utf-8"))
    try:
        log = Path(f"F:/refcv9_code3/evidence/mutations/{name}.log")
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run([PY, "-X", "utf8", "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests,
                                 f"--basetemp=F:/c_overflow_2026-10-26/tmp/mut_{name}"], cwd=TREE, env=env,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        tail = log.read_text(encoding="utf-8").strip().splitlines()[-1]
    finally:
        p.write_bytes(orig)
    assert p.read_bytes() == orig
    out.append({"mutation": name, "file": rel, "pytest_exit": rc, "summary": tail, "RED": rc == 1})
    print(name, rc, tail)
Path("F:/refcv9_code3/evidence/mutations/mutations.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

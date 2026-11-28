"""Meta-test: mutate srttc_r1_gate.py and require the named tests to go RED (a test file that cannot fail proves nothing)."""
import os, shutil, subprocess, sys, json
SRC = "C:/Users/Admin/srttc_gate/ex"
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
M = [
 ("M1 DVB lever compare removed", [("            if built[k] != v:", "            if False:")], ["TestDVB.test_each_wrong_lever_fails"]),
 ("M2 BANK unit sha compare removed", [("            if sha != table[n]:\n                bad.append(n)", "            if False:\n                bad.append(n)")], ["TestBank.test_one_corrupted_unit_byte_fails"]),
 ("M3 LIVE exact-zero check on other rows removed", [("            if r[\"frozen_w\"] != 0.0 or r[\"frozen_b\"] != 0.0:", "            if False:")], ["TestLive.test_a_loss_that_leaks_into_another_row_fails"]),
 ("M4 SUITE 'OK' status check removed", [("        if status != \"OK\":", "        if False:")], ["TestSuiteAndCommit.test_pass_then_each_regression_arm_fails"]),
 ("M5 VERIFY argv compare removed", [("    if b.get(\"argv_sha256\") != a_sha or b.get(\"argv\") != argv:", "    if False:")], ["TestMintAndVerify.test_token_for_argv_x_used_with_argv_y_fails_verify"]),
 ("M6 MINT verdict forced PASS", [("    verdict = \"PASS\" if all(checks[c][\"status\"] == \"PASS\" for c in CHECKS) else \"FAIL\"", "    verdict = \"PASS\"")], ["TestMintAndVerify.test_every_check_has_a_regression_arm_that_blocks_the_token"]),
 ("M7 PATH guard-not-refusing probes removed (read AND write)", [("                reasons.append(f\"the guard did NOT refuse a read of {rp!r} (token {tok!r})\")", "                pass"), ("                reasons.append(f\"the guard did NOT refuse a write of a path naming {tok!r}\")", "                pass")], ["TestPath.test_a_disabled_guard_fails"]),
 ("M8 COMMIT blob compare removed", [("            elif sha256_bytes(blob) != disk:", "            elif False:")], ["TestSuiteAndCommit.test_commit_check"]),
 ("M9 LIVE one-step-moved check removed", [("        if not (np.isfinite(moved) and moved > 0.0):", "        if False:")], ["TestLive.test_optimiser_step_cut_fails"]),
 ("M10 BANK 5 % rule removed", [("                if frac > REG_MAX_EMPTY_LOG_FRACTION:", "                if False:")], ["TestBank.test_logs_without_any_token_threshold_is_five_percent"]),
 ("M11 DVB grid checks removed (both the count and the order/value compare)", [("        if len(cfgs) != 9:", "        if False:"), ("        if got_grid != want_grid:", "        if False:")], ["TestDVB.test_grid_with_one_extra_config_fails"]),
 ("M12 VERIFY HMAC check removed", [("    if not signature_ok(tok, key):\n        reasons.append(f\"HMAC does not verify: edited", "    if False:\n        reasons.append(f\"HMAC does not verify: edited")], ["TestMintAndVerify.test_edited_token_wrong_key_and_wrong_profile_fail"]),
 ("M13 DVB 'lever must be NAMED' removed", [("            if f not in names:\n", "            if False:\n")], ["TestDVB.test_a_lever_must_be_named"]),
 ("M14 BANK R1-reader sub-check removed", [("            mod.load_manifest(guard, opts.manifest)\n            det[\"r1_reads_manifest\"] = True", "            det[\"r1_reads_manifest\"] = True")], ["TestBank.test_a_manifest_that_r1s_own_reader_rejects_fails_the_gate"]),
 ("M15 LIVE positive (leak) control removed", [("        if not (leak[\"frozen_w\"] > 0.0):", "        if False:")], ["TestLive.test_a_blind_probe_fails_its_own_positive_control"]),
 ("M16 PATH argv scan removed", [("            hit = mod.PathGuard.forbidden(p)\n            if hit:\n                reasons.append(f\"a path in this launch names", "            hit = None\n            if hit:\n                reasons.append(f\"a path in this launch names")], ["TestPath.test_a_forbidden_token_only_in_the_argv_fails"]),
]
base = open(f"{SRC}/srttc_r1_gate.py", encoding="utf-8", newline="").read()
out = []
for name, subs, tests in M:
    s = base
    for old, new in subs:
        assert s.count(old) == 1, (name, old[:60], s.count(old))
        s = s.replace(old, new)
    d = "C:/Users/Admin/srttc_gate/work/ex_mut"
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC, d)
    open(f"{d}/srttc_r1_gate.py", "w", encoding="utf-8", newline="").write(s)
    if tests is None:
        out.append((name, "n/a (control of a control; see RESULT)")); continue
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    r = subprocess.run([PY, "-I", "test_srttc_r1_gate.py"] + tests, cwd=d, env=env, capture_output=True)
    tail = r.stderr.decode("utf-8", "replace").strip().splitlines()[-1]
    out.append((name, "RED (as required)" if r.returncode != 0 else "STILL GREEN  <-- the test does not detect this mutation", tail))
for row in out:
    print(" | ".join(row))

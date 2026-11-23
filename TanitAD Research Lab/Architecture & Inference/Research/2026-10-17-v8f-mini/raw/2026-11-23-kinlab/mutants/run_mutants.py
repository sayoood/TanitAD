"""Mutation runner (KINLAB build): copies the build tree's stack/ and taniteval/taniteval/ once (no results / experiments), applies ONE mutation at a time, runs stack/tests/test_v8f_kinlab.py whole-file (-x), requires a FAILURE (rc != 0), restores the file.
usage: python run_mutants.py [M1 M2 ...]   (no args = all)"""
import json
import os
import shutil
import subprocess
import sys
import time

K = "F:/TanitAD-scratch/kinlab"
SRC = K + "/tree"
DST = K + "/mutants/tree"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
if not os.path.isdir(DST + "/stack"):
    shutil.copytree(SRC + "/stack", DST + "/stack", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "experiments"))
    shutil.copytree(SRC + "/taniteval/taniteval", DST + "/taniteval/taniteval", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
D = DST + "/stack/tanitad/data/v8f_dagger.py"
R = DST + "/stack/tanitad/data/v8f_recovery.py"
T = DST + "/stack/scripts/v8f_mini_train.py"
M = {
    "M0_control_no_mutation": (R, "T_B = 1.0               # s, the blend horizon", "T_B = 1.0               # s, the blend horizon"),
    "M1_T_B_zero": (R, "T_B = 1.0               # s, the blend horizon", "T_B = 1e-12             # s, the blend horizon"),
    "M2_w_ignored": (R, "        w = min(1.0, t / T_B)", "        w = 0.0"),
    "M3_kappa_o_ignored": (R, "kap = min(max((1.0 - w) * k_o + w * kpp,", "kap = min(max((1.0 - w) * 0.0 + w * kpp,"),
    "M4_a_o_ignored": (R, "a = min(max((1.0 - w) * a_o + w * a_rule,", "a = min(max((1.0 - w) * 0.0 + w * a_rule,"),
    "M5_merge_never_happens": (R, "            if (n + 1) * H_SUB >= T_B and dist < MERGE_LAT_M and abs(_wrap(th - hd)) < MERGE_HEAD_RAD and abs(v - float(uh_k[kk])) <= MERGE_SPEED_MS:", "            if False:"),
    "M6_refusal_falls_back_to_human": (R, "        elif res[\"speed_label\"] is not None:                                          # the fallback", "        elif False:                                          # the fallback"),
    "M7_identity_rule_off": (R, "    if abs(k_o - k_ref0) <= IDENT_TOL and abs(a_o - a_ref0) <= IDENT_TOL:", "    if False:"),
    "M8_friction_clip_off": (R, "        if math.hypot(a, v * v * kap) > F_LIM:", "        if False:"),
    "M9_flag_parsed_not_passed_to_bank": (T, "bank.attach_recovery(res[\"slot\"], res[\"label\"], res[\"status_counts\"], kin=res[\"is_kin\"])", "bank.attach_recovery(res[\"slot\"], res[\"label\"], res[\"status_counts\"])"),
    "M10_trainer_never_runs_the_kin_labeller": (T, "        if a.dagger_label == \"kin_v1\":                                                           # KINLAB", "        if False:                                                           # KINLAB"),
    "M11_gather_drops_rl_kin": (D, "                rec[\"rl_kin\"] = self.rl[\"kin\"][idx]", "                pass"),
    "M12_kin_counter_line_deleted": (D, "            new[\"_rl_kin_n\"] = int(np.asarray(rec[\"rl_kin\"], bool).sum())", "            new[\"_rl_kin_n\"] = 0"),
    "M13_feasibility_check_skipped": (R, "    if check:\n        vv = violations(pts, v_o, H)", "    if False:\n        vv = violations(pts, v_o, H)"),
    "M14_pursuit_sign_flipped": (R, "            kpp = 2.0 * math.sin(", "            kpp = -2.0 * math.sin("),
    "M15_a_limit_clip_off": (R, "        a = min(max((1.0 - w) * a_o + w * a_rule, -A_LIM), A_LIM)", "        a = (1.0 - w) * a_o + w * a_rule"),
    "M1b_T_B_zero_constants_test_deselected": (R, "T_B = 1.0               # s, the blend horizon", "T_B = 1e-12             # s, the blend horizon", ["-k", "not test_constants_are_the_briefs"]),
    "M17_merge_ignores_the_speed_condition": (R, " and abs(v - float(uh_k[kk])) <= MERGE_SPEED_MS:", ":"),
    "M18_longitudinal_uses_the_per_step_accelerations_again": (R, "        a_rule = min(max((u_h_t - v) / DT, -A_LABEL), A_LABEL)", "        a_rule = float((ul[k + 1] - ul[k]) / DT)"),
    "M16_probe_drops_kin_label_kind": (T, "dag.bank.label_kind in (\"speed_v1\", \"kin_v1\") else ()", "dag.bank.label_kind in (\"speed_v1\",) else ()"),
}
only = set(sys.argv[1:])
resf = K + "/mutants/MUTANTS_RESULT.json"
out = json.load(open(resf)) if (only and os.path.exists(resf)) else {}
for name, spec in M.items():
    path, old, new = spec[:3]
    extra = spec[3] if len(spec) > 3 else []
    if only and name.split("_")[0] not in only:
        continue
    orig = open(path, encoding="utf-8", newline="").read()
    assert orig.count(old) == 1, (name, orig.count(old))
    open(path, "w", encoding="utf-8", newline="").write(orig.replace(old, new))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TMP=K + "/tmp", TEMP=K + "/tmp", PYTHONPATH=DST + "/stack")
    t0 = time.time()
    try:
        subprocess.run([PY, "-I", K.replace("kinlab", "strat1") + "/read/seeds/wait_commit.py", "5.5", "60"], capture_output=True, timeout=700)
        p = subprocess.run([PY, "-m", "pytest", "-q", "-x", *extra, "stack/tests/test_v8f_kinlab.py", "--basetemp=" + K + "/tmp/ptm", "-p", "no:cacheprovider"],
                           cwd=DST, env=env, capture_output=True, text=True, timeout=1500)
        tail = [ln for ln in p.stdout.splitlines() if ln.startswith(("FAILED", "ERROR")) or " passed" in ln or " failed" in ln][-3:]
        rc = p.returncode
    finally:
        open(path, "w", encoding="utf-8", newline="").write(orig)
        shutil.rmtree(K + "/tmp/ptm", ignore_errors=True)
    if name.startswith("M0_"):
        print(name, "GREEN (control, as required)" if rc == 0 else "!! CONTROL RED !!", tail[:2], flush=True)
    out[name] = {"rc": rc, "RED": rc != 0, "seconds": round(time.time() - t0), "tail": tail}
    print(name, "RED" if rc != 0 else "!! STILL GREEN !!", tail[:2], flush=True)
    json.dump(out, open(resf, "w"), indent=1)

"""Mutation runner (RL build): copies the build tree once, applies ONE mutation at a time, runs the two RL test files (recovery + wiring), requires a FAILURE (rc != 0), restores the file. Prints one line per mutant."""
import json, shutil, subprocess, sys, os, time
SRC = "F:/TanitAD-scratch/reclab/tree97"
DST = "F:/TanitAD-scratch/reclab/mutants/tree"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
if not os.path.isdir(DST + "/stack"):
    shutil.copytree(SRC + "/stack", DST + "/stack", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(SRC + "/taniteval", DST + "/taniteval", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
D = DST + "/stack/tanitad/data/v8f_dagger.py"
R = DST + "/stack/tanitad/data/v8f_recovery.py"
T = DST + "/stack/scripts/v8f_mini_train.py"
M = {
 "M1_delete_gt_install_line": (D, "        gt[rows[hm]] = torch.from_numpy(lab).to(dev).to(gt.dtype)", "        pass"),
 "M2_delete_s_h_rederivation": (D, "        sh[rows[hm]] = torch.from_numpy(RV.arc_from_label(lab)).to(dev).to(sh.dtype)", "        pass"),
 "M3_human_speed_in_place_of_own_speed": (R, "        v_own, r = float(p10f[i, -1, 0]), int(rr[i])", "        r = int(rr[i])\n        v_own = float(np.hypot(*(grid[r + 1, :2] - grid[r, :2])) / DT) if r + 1 < len(grid) else 0.0"),
 "M4_skip_the_feasibility_refusal": (R, "        if v[\"any\"]:\n            out[\"status\"] = \"refused_infeasible\"", "        if False and v[\"any\"]:\n            out[\"status\"] = \"refused_infeasible\""),
 "M5_speed_jump_clip_false": (R, "    ul = label_steps(float(v_own), uh, a_rate=a_rate, clip=clip)", "    ul = label_steps(float(v_own), uh, a_rate=a_rate, clip=False)"),
 "M6_trainer_never_attaches_the_labels": (T, "        bank.attach_recovery(res[\"slot\"], res[\"label\"], res[\"status_counts\"])", "        pass"),
 "M7_flag_default_is_speed_v1": (T, "choices=DR.DAGGER_LABELS, default=\"human\"", "choices=DR.DAGGER_LABELS, default=\"speed_v1\""),
 "M8_gather_drops_the_label_keys": (D, "            rec[\"rl_has\"], rec[\"rl_label\"] = has, lab", "            pass"),
 "M9_the_deficit_is_ignored": (R, "    L = np.concatenate([[0.0], np.cumsum(ul[1:]) * DT])", "    L = s[:H + 1].copy()"),
 "M10_skip_the_path_end_refusal": (R, "    if float(L[-1]) > float(s[-1]) + 1e-9:", "    if False:"),
 "M12_label_rate_A_4_0_no_margin": (R, "A_LABEL = 3.0 ", "A_LABEL = 4.0 "),
 "M11_off_route_made_eligible": (D, "        self.rl = {\"slot\": slot,", "        self._index(self.reason != self.reason + 99)\n        self.rl = {\"slot\": slot,"),
}
only = set(sys.argv[1:])
out = {}
for name, (path, old, new) in M.items():
    if only and name.split("_")[0] not in only:
        continue
    orig = open(path, encoding="utf-8", newline="").read()
    if old is None:
        mut = orig + new
    else:
        assert orig.count(old) == 1, (name, orig.count(old))
        mut = orig.replace(old, new)
    open(path, "w", encoding="utf-8", newline="").write(mut)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TMP="F:/TanitAD-scratch/reclab/tmp", TEMP="F:/TanitAD-scratch/reclab/tmp", PYTHONPATH=DST + "/stack;" + DST)
    t0 = time.time()
    try:
        p = subprocess.run([PY, "-I", "-m", "pytest", "-q", "-x", "stack/tests/test_v8f_recovery.py", "stack/tests/test_v8f_recovery_wiring.py", "--basetemp=F:/TanitAD-scratch/reclab/tmp/ptm", "-p", "no:cacheprovider"],
                           cwd=DST, env=env, capture_output=True, text=True, timeout=900)
        tail = [l for l in p.stdout.splitlines() if l.startswith(("FAILED", "ERROR")) or " passed" in l or " failed" in l][-3:]
        rc = p.returncode
    finally:
        open(path, "w", encoding="utf-8", newline="").write(orig)
    out[name] = {"rc": rc, "RED": rc != 0, "seconds": round(time.time() - t0), "tail": tail}
    print(name, "RED" if rc != 0 else "!! STILL GREEN !!", tail[:2], flush=True)
json.dump(out, open("F:/TanitAD-scratch/reclab/mutants/MUTANTS_RESULT.json", "a" if only else "w"), indent=1)

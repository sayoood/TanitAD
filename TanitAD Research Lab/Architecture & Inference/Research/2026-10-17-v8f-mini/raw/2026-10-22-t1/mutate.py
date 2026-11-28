import subprocess, shutil, sys, re, json
SRC = "tree/stack/scripts/v8f_t1_eval.py"
orig = open(SRC, encoding="utf-8").read()
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MUTS = {
 "M1_keep_logged_future_in_view": ("            arr[t0 + 1:] = np.nan\n            d[k] = arr", "            d[k] = arr"),
 "M2_ignore_the_plan_execute_logged_motion": ('                st.commit_plan(pl.astype(np.float64), m)', '                st.commit_logged(logged_tracks[st.ci] if logged_tracks else __import__("builtins").NotImplemented, m) if False else st.commit_plan(np.zeros((30, 2)) + np.arange(1, 31)[:, None] * 2.0, m)'),
 "M3_no_nav_patch": ('    it["nav"] = st.nav[idc].astype(np.float32)\n', '    pass\n'),
 "M4_allowlist_includes_act": ('NO_READ = frozenset({"act", "pose"})', 'NO_READ = frozenset()'),
 "M5_nav_not_decremented": ("                    row[3] = np.clip(row[3] - self.arc[j] / 100.0, 0.0, 3.0)", "                    row[3] = row[3]"),
 "M6_state_not_rebuilt": ('            d["p10f"][j] = fk.kin_features(kin[s:s + 1])[0].astype(np.float32)', '            d["p10f"][j] = d["p10f"][self.t0]'),
}
res = {}
for name, (a, b) in MUTS.items():
    assert a in orig, name
    open(SRC, "w", encoding="utf-8").write(orig.replace(a, b, 1))
    try:
        r = subprocess.run([PY, "-m", "pytest", "-q", "tree/stack/tests/test_v8f_t1.py", "-p", "no:cacheprovider", "--tb=no", "-q", "-W", "ignore"], capture_output=True, text=True, timeout=900)
        failed = sorted(set(re.findall(r"FAILED \S+::(\S+)", r.stdout)))
        res[name] = {"failed": failed, "tail": r.stdout.strip().splitlines()[-1]}
    finally:
        open(SRC, "w", encoding="utf-8").write(orig)
json.dump(res, open("mutation_results.json", "w"), indent=1)
for k, v in res.items():
    print(k, "->", v["tail"], v["failed"])

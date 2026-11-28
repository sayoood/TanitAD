"""Regression vs aedf8716: every test file that imports (or names) a changed module, plus every launch_gate reader, on the TFAR tree, serially (wait_commit before each run); every failing test id is re-run on a pristine aedf8716 export.
Writes tmp/regress_summary.json. A regression = fails on the new tree and passes on the base."""
import json, os, re, subprocess, sys, glob

N = "F:/TanitAD-scratch/strat1/tfar"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
NEW = N + "/tree"
BASE = N + "/tmp/reg_base"
TMP = N + "/tmp/t"
os.makedirs(TMP, exist_ok=True)
env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TMP=TMP, TEMP=TMP, PYTHONDONTWRITEBYTECODE="1")
NEWFILES = {"test_v8f_tfar.py", "test_tfar_check.py", "test_tfar_reach.py", "test_tfar_eval.py"}      # run by their own passes; listed in the summary too


def wait():
    subprocess.run([PY, "F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py", "6", "120"], check=False)


def pytest(root, files, tag):
    wait()
    env = dict(env0, PYTHONPATH=os.path.abspath(root + "/stack") + ";" + os.path.abspath(root))
    xml = f"{N}/tmp/junit_{tag}.xml"
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", f"{N}/tmp/bt_{tag}", f"--junitxml={xml}"] + files
    r = subprocess.run(cmd, cwd=root + "/stack", env=env, stdout=open(f"{N}/tmp/regress_{tag}.log", "wb"), stderr=subprocess.STDOUT)
    import xml.etree.ElementTree as ET
    res = {}
    try:
        for tc in ET.parse(xml).getroot().iter("testcase"):
            st = "pass"
            for ch in tc:
                if ch.tag in ("failure", "error"):
                    st = "fail"
                elif ch.tag == "skipped" and st == "pass":
                    st = "skip"
            res[(tc.get("classname") or "") + "::" + (tc.get("name") or "")] = st
    except Exception as e:  # noqa: BLE001
        res["__error__"] = str(e)
    return r.returncode, res


tests = sorted(glob.glob(NEW + "/stack/tests/*.py"))
pat = re.compile(r"v8f_mini|V8FMini|launch_gate|ckpt120|strat1|tfar|v8f_t1_eval|zerograd")
todo = []
for t in tests:
    try:
        if pat.search(open(t, encoding="utf-8", errors="replace").read()):
            todo.append(os.path.basename(t))
    except OSError:
        pass
todo = [n for n in ["test_v8f_tfar.py", "test_tfar_check.py", "test_tfar_reach.py", "test_tfar_eval.py"] if os.path.exists(NEW + "/stack/tests/" + n)] + [n for n in todo if n not in NEWFILES]      # the new files FIRST: a defect in them is found in the first hour
print("files", len(todo), flush=True)
summary = {"files": todo, "new": {}, "failing_files": []}
fails = {}
for n in todo:
    rc, res = pytest(NEW, ["tests/" + n], "new_" + n[:-3])
    bad = sorted(k for k, v in res.items() if v == "fail")
    summary["new"][n] = {"rc": rc, "n": len(res), "fail": len(bad)}
    if bad or rc not in (0, 5):
        fails[n] = bad
        summary["failing_files"].append(n)
    json.dump(summary, open(N + "/tmp/regress_summary.json", "w"), indent=1)
    print(n, summary["new"][n], flush=True)
if fails:
    if not os.path.isdir(BASE):
        os.makedirs(BASE)
        subprocess.run(f'git -C C:/Users/Admin/tanitad-push archive aedf8716 stack/tanitad stack/scripts stack/tests stack/pyproject.toml taniteval | tar -x -C "{BASE}"', shell=True, check=True)
    summary["base"], summary["regressions"] = {}, []
    for n, bad in fails.items():
        if n in NEWFILES:
            continue
        rc, res = pytest(BASE, ["tests/" + n], "base_" + n[:-3])
        summary["base"][n] = {"rc": rc, "n": len(res), "fail": sum(1 for v in res.values() if v == "fail")}
        for k in bad:
            if res.get(k) == "pass":
                summary["regressions"].append(k)
        json.dump(summary, open(N + "/tmp/regress_summary.json", "w"), indent=1)
summary["n_regressions"] = len(summary.get("regressions", []))
summary["new_file_failures"] = {n: fails[n] for n in fails if n in NEWFILES}
json.dump(summary, open(N + "/tmp/regress_summary.json", "w"), indent=1)
print("ZZ REGRESS DONE n_regressions", summary["n_regressions"], "new_file_failures", len(summary["new_file_failures"]), flush=True)

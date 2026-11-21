"""Detached finisher for the HNSCREEN-FP-REQUIRED sibling-test check (one child python at a time, CPU only, RAM-checked).
  1. run every refe/ test file that mentions launch_gate on the FIXED tree (work/refe)            -> suite_after.json
  2. run on the ORIGINAL tree (base/refe) only the files that FAIL on the fixed tree and have no baseline yet (five files already have one), to tell NEW from PRE-EXISTING failures
  3. write suite_compare.md (one row per file: base rc / after rc / verdict) and the marker file suite_finish.DONE
Resumable: a file already present in suite_after.json / suite_base.json is not re-run.  Never touches a lock, the GPU, a pod or par18/{l1run,probe,r0eb}."""
import json
import os
import subprocess
import sys
import time

import psutil

ROOT = r"C:\Users\Admin\par18\hnfp"
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
FILES = ["test_aux_g12_bf16.py", "test_aux_s1_wiring.py", "test_carve_feed_gate.py", "test_gate_gpu_probes.py", "test_gate_suite_prerun.py", "test_hnscreen_chain.py", "test_p2r_chain.py",
         "test_p2rep_chain.py", "test_p6_chain.py", "test_p6_levers.py", "test_rl_pickcon.py", "test_rl_r0c.py", "test_rl_s1b.py", "test_rl_safeobj.py", "test_rls1_chain.py", "test_rls1_queue.py",
         "test_s1aux_chain.py", "test_s1c.py", "test_sft4r_chain.py", "test_sft4r_gate.py", "test_suite_nesting.py"]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")


def load(name):
    p = os.path.join(ROOT, name)
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}


def save(name, d):
    with open(os.path.join(ROOT, name), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(d, fh, indent=1)


def run_one(tree, f):
    waited = 0
    while psutil.virtual_memory().available < 2 * 2**30 and waited < 900:
        time.sleep(15)
        waited += 15
    t0 = time.time()
    try:
        cp = subprocess.run([PY, f], cwd=os.path.join(ROOT, tree, "refe"), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
        rc, so, se = cp.returncode, cp.stdout, cp.stderr
    except subprocess.TimeoutExpired:
        rc, so, se = "TIMEOUT", "", ""
    lines = [ln for ln in (so + "\n" + se).splitlines() if ln.strip()]
    return {"rc": rc, "secs": round(time.time() - t0, 1), "tokens": sorted({t for t in ("ZZTEST_PASS", "ZZLG_STATIC_PASS", "ZZLG_FULL_PASS") if t in so}),
            "skips": sum(1 for ln in lines if ln.startswith("ZZTEST_SKIP")), "tail": [ln[:300] for ln in lines[-4:]]}


after, base = load("suite_after.json"), load("suite_base.json")
for f in FILES:
    if f not in after:
        after[f] = run_one("work", f)
        save("suite_after.json", after)
        print("after", f, after[f]["rc"], after[f]["secs"], flush=True)
for f in FILES:
    if f not in base and after[f]["rc"] != 0:                                  # only a failing file needs the baseline, to tell NEW from PRE-EXISTING
        base[f] = run_one("base", f)
        save("suite_base.json", base)
        print("base ", f, base[f]["rc"], base[f]["secs"], flush=True)
rows = ["| file | base rc | after rc | skips base/after | verdict |", "|---|---|---|---|---|"]
for f in FILES:
    a = after[f]
    b = base.get(f) or {"rc": "(not run)", "tokens": a["tokens"], "skips": "-"}
    verdict = "same" if (b["rc"], b["tokens"]) == (a["rc"], a["tokens"]) else "DIFFERS"
    if a["rc"] != 0:
        verdict += " / after FAILS" + (" (base too: PRE-EXISTING)" if b["rc"] != 0 else " (NEW FAILURE)")
    rows.append(f"| {f} | {b['rc']} | {a['rc']} | {b['skips']}/{a['skips']} | {verdict} |")
open(os.path.join(ROOT, "suite_compare.md"), "w", encoding="utf-8", newline="\n").write("\n".join(rows) + "\n")
open(os.path.join(ROOT, "suite_finish.DONE"), "w").write("done\n")
print("DONE", flush=True)

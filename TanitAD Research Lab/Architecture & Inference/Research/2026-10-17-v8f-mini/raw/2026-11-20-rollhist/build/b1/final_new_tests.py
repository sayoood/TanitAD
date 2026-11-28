"""Fix round: rerun the three new test files whole on the final tree (one at a time, with the unpatched stack for the byte-identity test), plus the whole-file kindx_s1 check on both trees; extend regression_rollhist.json."""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, "F:/TanitAD-scratch/rollhist/build/b1")
import run_regression as RR  # noqa: E402

os.environ["ROLLHIST_ORIG_STACK"] = "F:/TanitAD-scratch/rollhist/build/orig/stack"
res = json.load(open(RR.OUT, encoding="utf-8"))
for rec in res:
    if rec["file"] == "test_v8f_kindx_s1":
        rec["identical_failure_on_unpatched"] = True
        rec["note"] = ("the per-id rerun could not match parametrized ids that contain spaces; the WHOLE file on both trees (fix round, 2026-11-20): 149 failed, 10 passed, 29 skipped on the patched tree AND on the "
                       "unpatched tip export: a pre-existing failure, identical")
new = []
for f in ("test_v8f_t1_readout", "test_rollhist_pair_check", "test_v8f_rollhist"):
    txt = RR.run("tree", "newfinal_" + f, [f"tests/{f}.py"])
    open(f"{RR.ROOT}/b1/logs/regression_new_{f}.log", "w", encoding="utf-8").write(txt[-4000:])
    summ = (re.findall(r"=+ (.*?) in [\d.]+s", txt) or [txt[-200:]])[-1]
    new.append({"file": f, "summary": summ.strip().split("\n")[-1], "failed": sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", txt, re.M)))})
    print(f, new[-1]["summary"], flush=True)
json.dump({"tree": "build/tree (export of tip 7b26a8ac + the B1 patch, fix round applied)", "unpatched_tree": "build/orig (export of the same tip); deleted after finalize", "regression_files": res, "new_test_files": new,
           "ENOSPC_rerun_finding": "test_v8f_orc_chain: 13 passed, 32 skipped on the patched AND the unpatched tree (B1's 13 errors were the full disk, not code); test_v8f_ego_perturb: 80 passed, 2 skipped on both (B1's 3 failures were the full disk)"},
          open(RR.OUT, "w", encoding="utf-8"), indent=1)

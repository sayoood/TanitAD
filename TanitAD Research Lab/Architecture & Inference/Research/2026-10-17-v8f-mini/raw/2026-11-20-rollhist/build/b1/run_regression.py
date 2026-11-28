"""Regression driver, ROLLHIST B1: the existing test files that touch the generator, the swap, the trainer's FLAG_CHECKS / selfcheck, and the hb executor, run ONE AT A TIME on build/tree.
Any failing test id is re-run on build/orig (the unpatched tip export): a pre-existing failure must fail identically there. Writes b1/regression_rollhist.json incrementally; logs in b1/logs/."""
import json
import os
import re
import subprocess
import sys

ROOT = "F:/TanitAD-scratch/rollhist/build"
OUT = ROOT + "/b1/regression_rollhist.json"
BOTH = ("test_v8f_orc_chain", "test_v8f_ego_perturb")            # the two B1 could not settle: run WHOLE on the patched and on the unpatched tree
FILES = ["test_v8f_orc_chain", "test_v8f_ego_perturb", "test_v8f_t1_eval_hb", "test_v8f_dagger", "test_v8f_dagger_train", "test_v8f_ego_dagger_compose", "test_v8f_launch_intent", "test_v8f_recover_wiring", "test_v8f_mini", "test_v8f_mini_full",
         "test_v8f_dscale", "test_v8f_window_start", "test_v8f_oacc", "test_v8f_flat", "test_v8f_mixo", "test_v8f_cz", "test_v8f_hist", "test_v8f_pgt",
         "test_v8f_wc", "test_v8f_kscore", "test_v8f_level_head", "test_v8f_lead_head", "test_v8f_grid", "test_v8f_fast_table", "test_v8f_init_ckpt_o_interface", "test_v8f_kindx_s1",
         "test_v8f_lora", "test_v8f_lora_ar2", "test_v8f_recover_pertick", "test_t1_eval_hb_placeholder"]


import shutil as _sh
import time as _t
DELLOG = "F:/TanitAD-scratch/rollhist/DELETIONS.log"


def free_gb():
    return _sh.disk_usage("F:/").free / 2**30


def wait_disk():
    for _ in range(20):
        if free_gb() >= 4.0:
            return True
        _t.sleep(60)
    raise SystemExit("F: under 4 GB free for 20 min: not running")


def cleanup_tmp(run):
    p = ROOT + "/tmp/" + run
    if os.path.isdir(p):
        sz = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)
        _sh.rmtree(p, ignore_errors=True)
        open(DELLOG, "a", encoding="utf-8").write(f"{_t.strftime('%Y-%m-%dT%H:%MZ', _t.gmtime())} rollhist-fix-round deleted {p} ({sz} bytes) after its pytest run\n")



def run(tree, tag, args):
    wait_disk()
    r = subprocess.run(["C:/Program Files/Git/usr/bin/bash.exe", ROOT + "/runpy.sh", tree, tag, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=line", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    cleanup_tmp(tag)
    return r.stdout + r.stderr


def main():
    res = []
    for f in FILES:
        import os
        if not os.path.exists(f"{ROOT}/tree/stack/tests/{f}.py"):
            continue
        txt = run("tree", "reg_" + f, [f"tests/{f}.py"])
        open(f"{ROOT}/b1/logs/regression_{f}.log", "w", encoding="utf-8").write(txt[-6000:])
        tail = re.findall(r"=+ (.*?) in [\d.]+s", txt)
        failed = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", txt, re.M)))
        rec = {"file": f, "summary": tail[-1] if tail else txt[-200:], "failed": failed}
        if failed:
            ids = [x.split(" - ")[0] for x in failed]
            t2 = run("orig", "regorig_" + f, ids)
            open(f"{ROOT}/b1/logs/regression_orig_{f}.log", "w", encoding="utf-8").write(t2[-6000:])
            f2 = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", t2, re.M)))
            rec["unpatched_summary"] = (re.findall(r"=+ (.*?) in [\d.]+s", t2) or [t2[-200:]])[-1]
            rec["identical_failure_on_unpatched"] = sorted(x.split(" - ")[0] for x in f2) == sorted(ids)
        if f in BOTH:
            t3 = run("orig", "regfull_orig_" + f, [f"tests/{f}.py"])
            open(f"{ROOT}/b1/logs/regression_orig_full_{f}.log", "w", encoding="utf-8").write(t3[-6000:])
            rec["unpatched_full_summary"] = (re.findall(r"=+ (.*?) in [\d.]+s", t3) or [t3[-200:]])[-1]
            rec["unpatched_full_failed"] = sorted(set(x.split(" - ")[0] for x in re.findall(r"^(?:FAILED|ERROR) (\S+)", t3, re.M)))
            rec["same_failing_ids_both_trees"] = rec["unpatched_full_failed"] == sorted(x.split(" - ")[0] for x in failed)
        res.append(rec)
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
        print(f, rec["summary"], rec.get("identical_failure_on_unpatched"), flush=True)


if __name__ == "__main__":
    main()

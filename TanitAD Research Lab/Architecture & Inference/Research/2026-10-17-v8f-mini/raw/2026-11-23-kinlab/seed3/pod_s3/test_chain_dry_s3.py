"""test_chain_dry_s3.py -- dev-box test of the SEED3 chain's pre-lock validations (rh_s3_chain.sh with DRY=1) on a PLANTED pod root: bash + the kit, no pod, no GPU, nothing written outside the scratch root.
The planted root: rh_env.sh (minimal), a stub blobcheck.py, tree_cdb and tree_ki (files named by their blob lists), the s0 argv (REAL, fx/), a G-SUITE result per commit, a pilot record, the shared pod files with their pin files, UE_s2.A's stand-in
(S3_TEST=1 lets the dev-box test name a stand-in file / sha; the pod walk never sets it). The well-formed walk must end `ZZ ROLLHIST-S3 DRY PASS ZZ` and write nothing; each mutation must end `ZZ ROLLHIST-S3 DRY FAIL ZZ` naming its step.
usage: python test_chain_dry_s3.py <kit dir> <scratch dir> <bash exe> <python exe>"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

KIT, ROOT, BASH, PYX = (os.path.abspath(a) for a in sys.argv[1:5])
FX = os.path.join(KIT, "fx")
RHC, KIC = "97fe0a56767320dbc5f4b823d67fb07a3209a745", "4097a89e4b4a6c9bf309cbae9dc6a8dd1e574fcb"
RH_ENV = """#!/bin/bash
R=${RH_ROOT:-/root/rollhist}; TREE=$R/tree_cdb; PY=${RH_PY:-/root/venv/bin/python}; DS=$R/ds; S1=$R/strat1
COMMIT=97fe0a56767320dbc5f4b823d67fb07a3209a745
export_py() { export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; }
"""
ok = True
KIT_RUN = os.path.join(ROOT, "kit").replace("\\", "/")
SHARED_R = ["rh_bank.sh", "rh_pilot_check.py", "rh_selfcheck.py", "rh_evidence.py", "blobcheck.py", "rh_env.sh"]
SHARED_DS = ["argv/B_s0.B.tmpl.json", "argv/UE_s2.A.json", "dry/UE_s2.A.dry.json", "gate/evidence_UE_s2.json", "chainlib.py", "argv/U_s1.B.tmpl.json"]
KITFILES = ["s3_env.sh", "s3_argv.py", "s3_arm.sh", "chk_arm_s3.sh", "rh_s3_chain.sh", "rh_s3_marker_watch.sh", "launch_s3.sh", "s3_bank_check.py", "dscale10k_check_s3.py", "rh_gate_record_s3.py", "ki_gate_record_s3.py",
            "U_s2.B.tmpl.json", "all_blobs_ki.txt"]


def sh(p):
    return p.replace("\\", "/")


def w(p, o, mode="w"):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode, **({} if "b" in mode else {"encoding": "utf-8", "newline": ""})) as f:
        f.write(o if isinstance(o, (str, bytes)) else json.dumps(o))


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def stage_kit(tamper=False):
    if os.path.exists(KIT_RUN):
        shutil.rmtree(KIT_RUN)
    os.makedirs(KIT_RUN)
    for f in KITFILES:
        shutil.copy(os.path.join(KIT, f), KIT_RUN)
    open(f"{KIT_RUN}/SHA256_S3.txt", "w", newline="\n").write("".join(f"{fsha(os.path.join(KIT_RUN, f))} *{f}\n" for f in KITFILES))
    if tamper:
        with open(f"{KIT_RUN}/s3_arm.sh", "a", encoding="utf-8", newline="") as f:
            f.write("\n# tampered\n")


def plant(R, mut=None):
    if os.path.exists(R):
        shutil.rmtree(R)
    os.makedirs(R)
    w(f"{R}/rh_env.sh", RH_ENV)
    for f in SHARED_R:
        if f != "rh_env.sh":
            w(f"{R}/{f}", "stub " + f)
    w(f"{R}/blobcheck.py", "import sys\nsys.exit(0)\n")
    w(f"{R}/RH_READOUT.chosen", "causal_g1\n" if mut == "readout" else "causal_g1x\n")
    rows = ["%064d x/%d.py" % (i, i) for i in range(20)]
    w(f"{R}/all_blobs_97fe0a56.txt", "".join(r + "\n" for r in rows))
    for i, r in enumerate(rows):
        if mut == "tree_cdb" and i == 3:
            continue
        w(f"{R}/tree_cdb/{r.split(' ', 1)[1]}", "")
    for i, l in enumerate(open(os.path.join(KIT, "all_blobs_ki.txt"), encoding="utf-8")):
        if mut == "tree_ki" and i == 5:
            continue
        w(f"{R}/tree_ki/{l.rstrip(chr(10)).split(' ', 1)[1]}", "")
    ds = f"{R}/ds"
    for f in SHARED_DS:
        src = f"{FX}/U_s1.B.tmpl.json" if f.endswith("U_s1.B.tmpl.json") else f"{FX}/B_s0.B.tmpl.json" if f.endswith("B_s0.B.tmpl.json") else None
        if src:
            shutil.copy(src, f"{ds}/{f}") if os.path.isdir(os.path.dirname(f"{ds}/{f}")) else (os.makedirs(os.path.dirname(f"{ds}/{f}")), shutil.copy(src, f"{ds}/{f}"))
        else:
            w(f"{ds}/{f}", "stub " + f)
    if mut == "u_s1_notseed1":
        t = json.load(open(f"{ds}/argv/U_s1.B.tmpl.json"))["argv"]
        t[t.index("--seed") + 1] = "5"
        w(f"{ds}/argv/U_s1.B.tmpl.json", {"argv": t})
    for f in ("study/dscale10k_merge.py", "pilot/PILOT_RH.json"):
        if f.startswith("pilot"):
            w(f"{R}/{f}", {"pass": mut != "pilot", "readout": "causal_g1x"})
        else:
            w(f"{R}/{f}", "stub")
    w(f"{R}/strat1/eval_grid_windows.json", "{}")
    w(f"{R}/strat1/n1dc/kclip.py", "stub")
    w(f"{R}/ue2.bin", b"UE2-stand-in", "wb")
    for t, f in (("B0_s0", "B0_s0_argv.json"), ("RH_s0", "RH_s0_argv.json"), ("KI_s0", "KI_s0_argv.json")):
        if not (mut == "ki_unfinished" and t == "KI_s0"):
            w(f"{R}/runs/{t}/ARM_DONE", "x")
        os.makedirs(f"{R}/runs/{t}", exist_ok=True)
        shutil.copy(f"{FX}/{f}", f"{R}/runs/{t}/argv.json")
    if mut == "b0_changed":
        t = json.load(open(f"{R}/runs/B0_s0/argv.json"))["argv"]
        t[t.index("--lr") + 1] = "2e-4"
        w(f"{R}/runs/B0_s0/argv.json", {"argv": t})
    w(f"{R}/gsuite_result_rh.json", {"new_commit": "cdbff971fa6dad4d20b937fe8abda6cb9f622adb" if mut == "gs_rh_old" else RHC, "n_regressions": 0, "new_test_failures": [], "missing_in_new": []})
    if mut != "gs_ki_absent":
        w(f"{R}/gsuite_result_ki.json", {"new_commit": KIC, "n_regressions": 1 if mut == "gs_ki_regress" else 0, "new_test_failures": [], "missing_in_new": []})
    sr = "".join(f"{fsha(f'{R}/{f}')}  {f}\n" for f in SHARED_R)
    sd = "".join(f"{fsha(f'{ds}/{f}')}  {f}\n" for f in SHARED_DS)
    w(f"{R}/SHARED_S3.sha256", sr)
    w(f"{R}/SHARED_DS_S3.sha256", sd)
    if mut == "shared_r":
        w(f"{R}/rh_bank.sh", "changed after the pin")
    if mut == "shared_ds":
        w(f"{ds}/chainlib.py", "changed after the pin")


def listing(R):
    return sorted((os.path.join(d, f), os.path.getsize(os.path.join(d, f)), os.path.getmtime(os.path.join(d, f))) for d, _, fs in os.walk(R) for f in fs)


def dry(R, ue_sha=None):
    ue = f"{R}/ue2.bin"
    env = dict(os.environ, RH_ROOT=R, RH_PY=sh(PYX), DRY="1", S3_TEST="1", S3_TEST_UE2=ue, S3_TEST_UE2_SHA=ue_sha or fsha(ue), S3_TEST_UE2_SIZE=str(os.path.getsize(ue)))
    p = subprocess.run([BASH, f"{KIT_RUN}/rh_s3_chain.sh"], env=env, capture_output=True, text=True, encoding="utf-8", timeout=900)
    return p.returncode, p.stdout + p.stderr


def report(name, good, extra=""):
    global ok
    ok &= bool(good)
    print(name, "PASS" if good else "FAIL", extra)


R = sh(os.path.join(ROOT, "R"))
os.makedirs(ROOT, exist_ok=True)
stage_kit()
plant(R)
before = listing(R)
rc, out = dry(R)
need = ["ZZ S3-INPUTS-PRECHECK OK B0_s2", "ZZ S3-INPUTS-PRECHECK OK RH_s2", "ZZ S3-INPUTS-PRECHECK OK KI_s2", "ZZ S3-ARGV SHOW PASS", "ZZ S3-ARGV TEMPLATE PASS"]
report("WELL-FORMED DRY walk runs every validator (each tool line present)", all(n in out for n in need), "missing: " + str([n for n in need if n not in out]))
report("WELL-FORMED DRY walk passes and writes nothing", rc == 0 and "ZZ ROLLHIST-S3 DRY PASS ZZ" in out and listing(R) == before and not os.path.exists(f"{R}/rh_s3_chain.log") and not os.path.exists(f"{R}/done_s3"),
       "\n".join(out.strip().splitlines()[-6:]) if rc else "")
for mut, why in (("gs_rh_old", "step gsuite"), ("gs_ki_regress", "step gsuite"), ("gs_ki_absent", "step gsuite"), ("tree_cdb", "step trees"), ("tree_ki", "step trees"), ("shared_r", "step shared"), ("shared_ds", "step shared"),
                 ("b0_changed", "step argv"), ("u_s1_notseed1", "step argv"), ("pilot", "step gate_inputs"), ("ki_unfinished", "step inputs"), ("readout", "RH_READOUT.chosen is not causal_g1x"), ("kit", "step kit"), ("ue2", "step ue2")):
    stage_kit(tamper=(mut == "kit"))
    plant(R, mut)
    before = listing(R)
    rc, out = dry(R, ue_sha="0" * 64 if mut == "ue2" else None)
    report(f"MUTANT {mut} refused before the lock", rc != 0 and "ZZ ROLLHIST-S3 DRY FAIL ZZ" in out and why in out and "DRY PASS" not in out and listing(R) == before, "\n".join(out.strip().splitlines()[-4:]) if not (rc != 0 and why in out) else "")
shutil.rmtree(R)
shutil.rmtree(KIT_RUN)
print("ZZ S3-CHAIN-DRY TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

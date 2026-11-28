"""test_chain_dry_ki.py -- Dev-box test of the CHAIN's pre-lock validations (rh_ki_chain.sh with DRY=1) on a PLANTED pod root: bash + the KI kit, no pod, no GPU, nothing written outside the scratch root.
The planted root has: rh_env.sh (copy of the pod's), a stub blobcheck.py, tree_ki (3,425 empty files named by all_blobs_ki.txt), RH / B0 argv (RH_s0's real argv from test_ki_argv.py, a seed-1 form), banks with their checks, a pilot record,
a G-SUITE result at 4097a89e. The well-formed walk must end `ZZ ROLLHIST-KI DRY PASS ZZ`; each mutation (G-SUITE at the old commit / with a failure / log not DONE / absent, tree file missing, tree absent, the kit tampered, an existing KI argv with a
third change, RH carrying a lever flag, the bank check failing, the pilot failing, RH_READOUT.chosen wrong) must end `ZZ ROLLHIST-KI DRY FAIL ZZ` naming the step, and the walk must write nothing (a tree listing before and after is equal).
The DRY walk waits for NO other chain: the planted root has no ADD123 log at all and the walk still passes.
usage: python -I test_chain_dry_ki.py <kit dir> <scratch dir> <bash exe> <python exe for the pod's $PY>"""
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys

KIT, ROOT, BASH, PYX = (os.path.abspath(a) for a in sys.argv[1:5])
COMMIT = "4097a89e4b4a6c9bf309cbae9dc6a8dd1e574fcb"
OLD = "1a8b11a1abd50c09636da8b7425e97573cdf1d04"
src = ast.parse(open(os.path.join(KIT, "test_ki_argv.py"), encoding="utf-8").read())
vals = {}
for n in src.body:
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("RH", "BSHA"):
        vals[n.targets[0].id] = eval(compile(ast.Expression(n.value), "x", "eval"), {"BSHA": vals.get("BSHA")})
RH0 = vals["RH"]
RH_ENV = """#!/bin/bash
R=${RH_ROOT:-/root/rollhist}; TREE=$R/tree_cdb; PY=${RH_PY:-/root/venv/bin/python}; DS=/root/dscale10k; S1=/root/strat1
COMMIT=97fe0a56767320dbc5f4b823d67fb07a3209a745
export_py() { export PYTHONPATH="$TREE/stack:$TREE/stack/scripts:$TREE" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; }
"""
ok = True


def w(p, o, mode="w"):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode, **({} if "b" in mode else {"encoding": "utf-8", "newline": ""})) as f:
        f.write(o if isinstance(o, (str, bytes)) else json.dumps(o))


def sh(p):                      # forward-slash Windows form: valid for Git bash (which converts path arguments for native programs) and for the Windows python
    return p.replace("\\", "/")


def plant(R, mut=None):
    if os.path.exists(R):
        shutil.rmtree(R)
    os.makedirs(R)
    w(f"{R}/rh_env.sh", RH_ENV)
    w(f"{R}/blobcheck.py", "import sys\nsys.exit(0)\n")
    w(f"{R}/RH_READOUT.chosen", "causal_g1\n" if mut == "readout" else "causal_g1x\n")
    rows = [l.rstrip("\n").split(" ", 1)[1] for l in open(os.path.join(KIT, "all_blobs_ki.txt"), encoding="utf-8")]
    for i, p in enumerate(rows):
        if mut == "tree" and i == 7:
            continue
        if mut == "notree":
            break
        w(f"{R}/tree_ki/{p}", "")
    for s in (0, 1):
        bank = f"{R}/bank/s{s}"
        w(f"{bank}/bank.json", {"bank": s})
        w(f"{bank}/BANK_CHECK.json", {"pass": not (mut == "bankcheck" and s == 1), "readout": "causal_g1x"})
        bsha = hashlib.sha256(open(f"{bank}/bank.json", "rb").read()).hexdigest()
        rh = [t.replace("/root/rollhist", R).replace("UE_s0", f"UE_s{s}").replace("RH_s0", f"RH_s{s}").replace("bank/s0", f"bank/s{s}") if t != vals["BSHA"] else bsha for t in RH0]
        rh[rh.index("--seed") + 1] = str(s)
        rh[rh.index("--dagger-seed") + 1] = str(s)
        if mut == "rhlabel" and s == 0:                                                  # RH already carrying a lever flag
            rh += ["--dagger-label", "kin_v1"]
        w(f"{R}/runs/RH_s{s}/argv.json", {"argv": rh})
        b0 = list(rh)
        b0[b0.index("--dagger-p") + 1] = "0"
        w(f"{R}/runs/B0_s{s}/argv.json", {"argv": b0})
        if mut == "ownki" and s == 0:                                                    # an existing KI argv whose --steps is a third change
            ki = [t.replace("RH_s0", "KI_s0") for t in rh] + ["--dagger-base", "logged"]
            ki[ki.index("--steps") + 1] = "3001"
            w(f"{R}/runs/KI_s0/argv.json", {"argv": ki})
    w(f"{R}/pilot/PILOT_RH.json", {"pass": mut != "pilot", "readout": "causal_g1x"})
    if mut != "nogsuite":
        w(f"{R}/gsuite_result_ki.json", {"new_commit": OLD if mut == "oldcommit" else COMMIT, "n_regressions": 1 if mut == "regress" else 0, "new_test_failures": [], "missing_in_new": []})
    w(f"{KIT_RUN}/prep_logs/ki_gsuite.log", "x\nZZ KI GSUITE FAIL ZZ\n" if mut == "gsnolog" else "x\nZZ KI GSUITE DONE ZZ\n")


def listing(R):
    out = []
    for d, _, fs in os.walk(R):
        for f in fs:
            p = os.path.join(d, f)
            out.append((p, os.path.getsize(p), os.path.getmtime(p)))
    return sorted(out)


KIT_RUN = sh(os.path.join(ROOT, "kit"))                      # the kit copy the walk runs from (its prep_logs is planted; a mutant may tamper with one file)


def stage_kit(tamper=False):
    if os.path.exists(KIT_RUN):
        shutil.rmtree(KIT_RUN)
    os.makedirs(KIT_RUN)
    for f in os.listdir(KIT):
        p = os.path.join(KIT, f)
        if os.path.isfile(p) and not f.startswith("test_") and f.endswith((".sh", ".py", ".txt", ".tar")) and "result" not in f and not f.startswith("test_log"):
            shutil.copy(p, KIT_RUN)
    if tamper:
        with open(os.path.join(KIT_RUN, "ki_arm.sh"), "a", encoding="utf-8", newline="") as f:
            f.write("\n# tampered\n")


def dry(R):
    env = dict(os.environ, RH_ROOT=sh(R), RH_PY=sh(PYX), DRY="1")
    p = subprocess.run([BASH, sh(os.path.join(KIT_RUN, "rh_ki_chain.sh"))], env=env, capture_output=True, text=True, encoding="utf-8", timeout=900)
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
report("WELL-FORMED DRY walk passes (no other chain to wait for) and writes nothing", rc == 0 and "ZZ ROLLHIST-KI DRY PASS ZZ" in out and listing(R) == before and not os.path.exists(f"{R}/rh_ki_chain.log"),
       out.strip().splitlines()[-3:] if rc else "")
for mut, why in (("oldcommit", "step gsuite"), ("regress", "step gsuite"), ("gsnolog", "step gsuite"), ("nogsuite", "step gsuite"), ("tree", "step tree_prep"), ("notree", "step tree_prep"),
                 ("ownki", "step argv"), ("rhlabel", "step argv"), ("bankcheck", "step gate_inputs"), ("pilot", "step gate_inputs"), ("readout", "RH_READOUT.chosen is not causal_g1x"), ("kit", "step kit")):
    stage_kit(tamper=(mut == "kit"))
    plant(R, mut)
    before = listing(R)
    rc, out = dry(R)
    report(f"MUTANT {mut} refused before the lock", rc != 0 and "ZZ ROLLHIST-KI DRY FAIL ZZ" in out and why in out and listing(R) == before, [l for l in out.splitlines() if "PROBLEM" in l or "ZZ" in l][:2] if not (rc != 0 and why in out) else "")
shutil.rmtree(R)
shutil.rmtree(KIT_RUN)
print("ZZ KI-CHAIN-DRY TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

"""test_chain_dry_rk.py -- REVISION 2. Dev-box test of the CHAIN's pre-lock validations (rh_rk_chain.sh with DRY=1) on a PLANTED pod root: bash + the rev2 kit, no pod, no GPU, nothing written outside the scratch root.
The planted root has: rh_env.sh (copy of the pod's), a stub blobcheck.py, tree_rk (3,420 empty files named by all_blobs_rk.txt), RL / B0 argv (RL_s0's real argv from test_rk_argv.py, a seed-1 form), banks, both caches
with their CPU-precompute selfchecks, a G-SUITE result at 1a8b11a1. The well-formed walk must end `ZZ ROLLHIST-RK DRY PASS ZZ`; each mutation (G-SUITE at the old commit / with a failure / log not DONE, tree file missing, a cache absent,
a cache edited after its precompute, the other seed's cache named in an existing RK argv, census below 0.5) must end `ZZ ROLLHIST-RK DRY FAIL ZZ` naming the step, and the walk must write nothing (a tree listing before and after is equal).
usage: python -I test_chain_dry_rk.py <kit dir> <scratch dir> <bash exe> <python exe for the pod's $PY>"""
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys

KIT, ROOT, BASH, PYX = (os.path.abspath(a) for a in sys.argv[1:5])
COMMIT = "1a8b11a1abd50c09636da8b7425e97573cdf1d04"
OLD = "0ca6f49dd076c035c1cd42da5e4e89a6f6961b9a"
src = ast.parse(open(os.path.join(KIT, "test_rk_argv.py"), encoding="utf-8").read())
vals = {}
for n in src.body:
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("RL", "BSHA"):
        vals[n.targets[0].id] = eval(compile(ast.Expression(n.value), 'x', 'eval'), {'BSHA': vals.get('BSHA')})
RL0 = vals["RL"]
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
    w(f"{R}/RH_READOUT.chosen", "causal_g1x\n")
    rows = [l.rstrip("\n").split(" ", 1)[1] for l in open(os.path.join(KIT, "all_blobs_rk.txt"), encoding="utf-8")]
    for i, p in enumerate(rows):
        if mut == "tree" and i == 7:
            continue
        w(f"{R}/tree_rk/{p}", "")
    for s in (0, 1):
        bank = f"{R}/bank/s{s}"
        w(f"{bank}/bank.json", {"bank": s})
        bsha = hashlib.sha256(open(f"{bank}/bank.json", "rb").read()).hexdigest()
        rl = [t.replace("/root/rollhist", R).replace("UE_s0", f"UE_s{s}").replace("RL_s0", f"RL_s{s}").replace("bank/s0", f"bank/s{s}") if t != vals["BSHA"] else bsha for t in RL0]
        rl[rl.index("--seed") + 1] = str(s)
        if mut == "ownrk" and s == 0:                                                    # an existing RK argv whose cache is the OTHER seed's
            rk = [t.replace("RL_s0", "RK_s0") for t in rl]
            rk[rk.index("--dagger-label") + 1] = "kin_v1"
            rk += ["--dagger-label-cache", f"{R}/kin_cache/s1_kin_v1.npz"]
            w(f"{R}/runs/RK_s0/argv.json", {"argv": rk})
        w(f"{R}/runs/RL_s{s}/argv.json", {"argv": rl})
        b0 = list(rl)
        b0[b0.index("--dagger-p") + 1] = "0"
        w(f"{R}/runs/B0_s{s}/argv.json", {"argv": b0})
        cpath = f"{R}/kin_cache/s{s}_kin_v1.npz"
        if not (mut == "nocache" and s == 1):
            w(cpath, b"planted cache %d" % s, "wb")
        csha = hashlib.sha256(b"planted cache %d" % s).hexdigest()
        kin, rest = (40, 60) if (mut == "census" and s == 1) else (63, 37)
        w(f"{R}/kin_cache/s{s}_selfcheck_cpu.json", {"levers": {"dagger": {"dagger_label": "kin_v1", "dagger_label_cache": cpath, "dagger_label_cache_sha256": csha}},
                                                      "dagger_probe": {"kin_status_counts": {"kin_emitted": kin, "refused_x": rest}}})
        if mut == "edited" and s == 0:
            w(cpath, b"edited after the precompute", "wb")
    w(f"{R}/gsuite_result_rk.json", {"new_commit": OLD if mut == "oldcommit" else COMMIT, "n_regressions": 1 if mut == "regress" else 0, "new_test_failures": [], "missing_in_new": []})
    if mut != "gsnolog":
        w(f"{R}/logs/rk_gsuite.log", "x\nZZ RK GSUITE DONE ZZ\n")
    else:
        w(f"{R}/logs/rk_gsuite.log", "x\nZZ RK GSUITE FAIL ZZ\n")


def listing(R):
    out = []
    for d, _, fs in os.walk(R):
        for f in fs:
            p = os.path.join(d, f)
            out.append((p, os.path.getsize(p), os.path.getmtime(p)))
    return sorted(out)


def dry(R):
    env = dict(os.environ, RH_ROOT=sh(R), RH_PY=sh(PYX), DRY="1")
    p = subprocess.run([BASH, sh(os.path.join(KIT, "rh_rk_chain.sh"))], env=env, capture_output=True, text=True, encoding="utf-8", timeout=600)
    return p.returncode, p.stdout + p.stderr


def report(name, good, extra=""):
    global ok
    ok &= bool(good)
    print(name, "PASS" if good else "FAIL", extra)


R = sh(os.path.join(ROOT, "R"))
plant(R)
before = listing(R)
rc, out = dry(R)
report("WELL-FORMED DRY walk passes and writes nothing", rc == 0 and "ZZ ROLLHIST-RK DRY PASS ZZ" in out and listing(R) == before, out.strip().splitlines()[-3:] if rc else "")
for mut, why in (("oldcommit", "step gsuite"), ("regress", "step gsuite"), ("gsnolog", "step gsuite"), ("tree", "step tree_prep"), ("nocache", "step caches"), ("edited", "step caches"), ("census", "step caches"),
                 ("ownrk", "step argv")):
    plant(R, mut)
    before = listing(R)
    rc, out = dry(R)
    report(f"MUTANT {mut} refused before the lock", rc != 0 and "ZZ ROLLHIST-RK DRY FAIL ZZ" in out and why in out and listing(R) == before, [l for l in out.splitlines() if "PROBLEM" in l or "ZZ" in l][:2] if not (rc != 0 and why in out) else "")
shutil.rmtree(R)
print("ZZ RK-CHAIN-DRY TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

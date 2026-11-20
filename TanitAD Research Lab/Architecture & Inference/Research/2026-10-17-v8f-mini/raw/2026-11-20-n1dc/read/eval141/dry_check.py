"""dry_check.py -- resolves every step of run_n1dc141.sh without running a long step: tree problems, checkpoint refusals, every T1 dump argv (rendered through the shim, not run), every T0 argv (input paths exist),
the STRATEGIC view plans (banked sources exist; sources that the chain itself makes are listed as made-by-chain), the strategic / read scripts import and parse, and the chain script is syntax-checked.
Usage: python dry_check.py   (prints one line per check; exit 1 if any check fails; writes DRY_CHECK.json)"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
res, bad = [], []


def chk(name, ok, detail=""):
    res.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})
    print(("ok   " if ok else "FAIL "), name, str(detail)[:200], flush=True)
    if not ok:
        bad.append(name)


def sub(code, env_extra):
    env = dict(os.environ, **env_extra)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=HERE, encoding="utf-8")
    return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:] if (r.stdout + r.stderr).strip() else []


DUMP_CODE = r'''
import sys, os, json
sys.path.insert(0, ".")
import n1dc_dumps as ND
tp = ND.tree_problem()
assert tp is None, tp
out = {}
for t in ND.TAGS:
    assert ND.refusal(t) is None, (t, ND.refusal(t))
    a = ND.extra_args(t)
    assert a[a.index("--nav-rb-variant") + 1] == ("zero" if t.endswith("Z") else "follow")
    ND._Shim.cur = t
    class R:  # capture the argv instead of running it
        pass
    import subprocess as sp
    cap = []
    ND._Shim.run.__func__  # exists
    orig = sp.run
    sp.run = lambda argv, **kw: (cap.append(argv), R())[1]
    try:
        ND._Shim.run([ND.RD.PY, ND.TREE + "/stack/scripts/v8f_t1_eval.py", "run", "--ckpt", "X", "--tag", "T"])
    finally:
        sp.run = orig
    argv = cap[0]
    assert argv[1].endswith("v8f_t1_eval_hb.py") and argv[argv.index("--kappa-max") + 1] == ND.KAPPA and argv[argv.index("--inner") + 1] == "v8f_t1_eval_rbd", argv[:8]
    assert os.path.isfile(argv[1])
out["tags"] = len(ND.TAGS); out["out"] = ND.OUT; out["kappa"] = ND.KAPPA
for k, v in ND.RD.INPUTS.items():
    assert os.path.exists(v), (k, v)
print(json.dumps(out))
'''
for kappa in ("0.2", "none"):
    rc, last = sub(DUMP_CODE, {"N1DC_KAPPA": kappa, "N1DC_D_141": HERE + "/dry_scratch_unused"})
    chk(f"T1 dump argv routing, kappa {kappa} (8 tags, hb wrapper, inner v8f_t1_eval_rbd, inputs exist, ckpt sha refusals none)", rc == 0, last)

T0_CODE = r'''
import sys, os
sys.path.insert(0, ".")
import n1dc_t0 as T
n = 0
for arm in T.D_ARMS:
    for v in ("real", "donor", "zero"):
        a = T.argv_for(arm, v, "OUT")
        for i, x in enumerate(a):
            if x.startswith(("F:/", "E:/", "C:/")) and not x.endswith("OUT"):
                if x.endswith(".py") or x.endswith(".pt") or x.endswith(".npz") or x.endswith(".json") or x.endswith(".jsonl") or "cache" in x:
                    assert os.path.exists(x), x
        assert os.path.isfile(a[1]) and a[a.index("--ckpt") + 1].endswith(f"{arm}/ckpt_last.pt") and os.path.isfile(a[a.index("--ckpt") + 1])
        n += 1
print(n, "T0 argvs")
'''
rc, last = sub(T0_CODE, {})
chk("T0 argv render + inputs exist (2 arms x 3 variants)", rc == 0, last)

VIEW_CODE = r'''
import sys, os
sys.path.insert(0, ".")
import n1dc_view as V, n1dw_view_s1slot as S1
BN = "banked_n1d/x"
# N1-slot view for N1DC: banked C0 sources must exist; the N1DC passes / ticks are made by the chain
p = V.plan()
made = [s for s, _ in p if not os.path.exists(s)]
banked_missing = [s for s in made if "N1DC" not in os.path.basename(s)]
assert not banked_missing, banked_missing
# S1-slot view for the N1D reference: ALL sources exist now
p1 = S1.plan("NAVRBD-N1D", BN + "/out/t0", BN + "/dumps141/ticks")
miss = [s for s, _ in p1 if not os.path.exists(s)]
assert not miss, miss
# S1-slot view for N1DC: only the N1DC files are chain-made
p2 = S1.plan("N1DC-N1DC", "out/t0", "dumps141_hb02/ticks")
m2 = [s for s, _ in p2 if not os.path.exists(s) and "N1DC" not in os.path.basename(s)]
assert not m2, m2
print(len(made), "N1-slot sources made by the chain;", len(p1), "N1D-reference S1 sources present;", len([1 for s, _ in p2 if "N1DC" in os.path.basename(s)]), "N1DC S1 sources made by the chain")
'''
rc, last = sub(VIEW_CODE, {})
chk("STRATEGIC view plans (banked sources present, chain-made ones named)", rc == 0, last)

IMPORT_CODE = r'''
import sys
sys.path.insert(0, ".")
import ast
for f in ("n1dc_strategic.py", "n1dc_read141.py", "k_hb0_141.py", "check_banked_n1d.py"):
    ast.parse(open(f, encoding="utf-8").read())
sys.path[:0] = ["F:/TanitAD-scratch/strat1/navrb2/read", "F:/TanitAD-scratch/strat1/read", "F:/TanitAD-scratch/strat1/read/theory/obd/read", "F:/TanitAD-scratch/strat1/read/theory/otband/read"]
import n1dc_strategic as NS
assert NS.pair_name("N1D") == "N1DC-N1D" and NS.COMPARATORS[0][1] == "NAVRBD-N1D"
import n1dc_read141 as NR
assert NR.arm(0) == "N1DC_s0" and NR.guard_of({0: set(), 1: set()}, False) == "GUARD-HOLDS" and NR.guard_of({0: set(), 1: {"x"}}, False) == "GUARD-COST"
import navrb2_read, strat1_read, obd_read, obd_geom, navrb2_strategic, strat1_strategic, strat1_ctx_read
print("imports ok")
'''
rc, last = sub(IMPORT_CODE, {})
chk("strategic / read scripts parse and import their frozen readers", rc == 0, last)

for f in ("SR", "N2R"):
    pass
need = ["F:/TanitAD-scratch/strat1/read/strat1_strategic.py", "F:/TanitAD-scratch/strat1/read/strat1_ctx_read.py", "F:/TanitAD-scratch/strat1/navrb2/read/navrb2_ctx_nofog.py",
        "F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py", "C:/Users/Admin/venvs/tanitad/Scripts/python.exe", "C:/Program Files/Git/usr/bin/bash.exe",
        "F:/TanitAD-scratch/strat1/navrb2/read/out/RESULT_NAVRB2.json"]
chk("chain-script external files exist", all(os.path.exists(x) for x in need), [x for x in need if not os.path.exists(x)])
r = subprocess.run(["C:/Program Files/Git/usr/bin/bash.exe", "-n", HERE + "/run_n1dc141.sh"], capture_output=True, text=True)
chk("run_n1dc141.sh bash -n", r.returncode == 0, r.stderr)
r = subprocess.run([sys.executable, HERE + "/check_banked_n1d.py", HERE + "/banked_n1d/x"], capture_output=True, text=True)
chk("banked N1D inputs present", r.returncode == 0, r.stdout.strip())
r = subprocess.run(["sha256sum", "-c", "SHA256_PODS.txt"], capture_output=True, text=True, cwd=HERE + "/ckpts")
chk("checkpoint sha256 equals the pod's", r.returncode == 0, r.stdout.replace("\n", " "))
json.dump({"checks": res, "all_ok": not bad}, open(HERE + "/DRY_CHECK.json", "w"), indent=1)
print("DRY CHECK", "PASS" if not bad else "FAIL " + str(bad))
sys.exit(1 if bad else 0)

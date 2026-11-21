"""dry_check.py -- resolves every step of rh_read_chain.sh without a rollout: the 24 rollout argvs rendered with UE_s0.A standing in for each arm (forms, flags, tier-of-record executor, wrapper, paths),
inputs exist, T0 argvs, reader argument acceptance and imports, `bash -n`, the reader's selftest, the tree blobcheck, sha256 of every script.  Runs nothing heavier than imports.  Writes DRY_CHECK.json.
Usage: python dry_check.py   (exit 1 if any check fails)"""
import ast
import glob
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import rh_common as C  # noqa: E402

res, bad = [], []


def chk(name, ok, detail=""):
    res.append({"check": name, "ok": bool(ok), "detail": str(detail)[:400]})
    print(("ok   " if ok else "FAIL "), name, str(detail)[:200], flush=True)
    if not ok:
        bad.append(name)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# 1 tree
r = subprocess.run([C.PY, HERE + "/blobcheck.py", C.TREE, "43319227"], capture_output=True, text=True, encoding="utf-8")
chk("tree blobcheck 3407 of 3407, 0 bad, 0 extra", r.stdout.startswith("checked 3407 of 3407 bad 0 extra []"), r.stdout.splitlines()[0] if r.stdout else r.stderr[-200:])
chk("hb executor sha256 74fd8445", sha(C.TREE + "/stack/scripts/v8f_t1_eval_hb.py").startswith("74fd8445"))
chk("pinned tool v8f_t1_eval.py sha256 61e140ed (cdbff971 lineage)", sha(C.TREE + "/stack/scripts/v8f_t1_eval.py").startswith("61e140ed"))
chk("study wrapper rzgfh sha256 373ffe72f2655c35 (ADD68)", sha(C.CODE + "/integ10k/v8f_t1_eval_reader_rule_rzgfh.py").startswith("373ffe72f2655c35"))
chk("readout wrapper rzg sha256 93879db8c0233867 (ADD63)", sha(C.CODE + "/integ10k/v8f_t1_eval_reader_rule_rzg.py").startswith("93879db8c0233867"))
# 2 inputs
for k, v in C.INPUTS.items():
    chk(f"input {k} exists", os.path.exists(v), v)
for f in ("dist_rule.py", "v8f_t1_eval_rule.py", "v8f_t1_eval_rule_mix.py", "qcal_map.py"):
    chk(f"distrule file {f}", os.path.isfile(C.CODE + "/distrule/" + f))
chk("readout wrapper adapter rzgi present", os.path.isfile(C.CODE + "/integ10k/" + C.WRAPPER + ".py"), C.WRAPPER)
chk("UE_s0.A checkpoint sha256 2fd4b747", os.path.isfile(C.ckpt_of("UEA", 0)) and sha(C.ckpt_of("UEA", 0)).startswith("2fd4b747"))
# 3 the 24 argvs
seen = set()
for k in C.KINDS:
    for s in C.SEEDS:
        for f in C.FORM_ORDER:
            out = C.folder(k, f, s)
            a = C.argv_record(k, f, s, ckpt=C.ckpt_of("UEA", 0))
            j = " ".join(a)
            ok = (a[1].endswith("/stack/scripts/v8f_t1_eval_hb.py") and a[a.index("--kappa-max") + 1] == "0.2" and a[a.index("--inner") + 1] == C.WRAPPER and a[a.index("--form") + 1] == "R0"
                  and a[a.index("run") - 0] == "run" and f"--out-dir {out}" in j and f"--tag {C.tag(k, f, s)}" in j and "--device cpu" in j and "--dump-ticks" in j and "--limit causal" in j)
            fl = C.FORMS[f]
            ok &= all(x in a for x in fl) and (("--readout-log" in a) and a[a.index("--readout") + 1] == fl[1])
            if "--acc-gain" in fl:
                ok &= a[a.index("--acc-gain") + 1] == fl[fl.index("--acc-gain") + 1]
            ok &= ("--path-exec" in a) == ("--path-exec" in fl)
            seen.add(out)
            if not ok:
                chk(f"argv {k} {f} s{s}", False, j[:300])
chk("24 rollout argvs render (kind x seed x form), 24 distinct output folders, tier-of-record executor, six forms", len(seen) == 24 and not any(not r["ok"] for r in res if r["check"].startswith("argv ")), f"{len(seen)} folders")
rs = subprocess.run([C.PY, HERE + "/rh_roll.py", "RH", "R0C10XHb20", "1", "--dry", "--ckpt", C.ckpt_of("UEA", 0)], capture_output=True, text=True, encoding="utf-8")
chk("rh_roll.py --dry renders one argv", rs.returncode == 0 and "v8f_t1_eval_hb.py" in rs.stdout, rs.stdout[:120])
# 4 T0
sys.path.insert(0, HERE)
import rh_t0 as T0  # noqa: E402
a0 = T0.argv_t0("RH_s0", ckpt=C.ckpt_of("UEA", 0))
miss = [x for x in a0 if x.startswith(("F:/", "E:/")) and not x.endswith("ckpt_last.pt") and not os.path.exists(x) and x != T0.T0DIR]
chk("T0 (G-T0) argv inputs exist", not miss and a0[1].endswith("v8f_mini_eval.py") and "--project" in a0 and "--families" in a0, miss)
# 5 reader tools
for f in ("code/q3/q3_read_r3.py", "code/q3/q3_read.py", "code/headway/headway_read_r2.py", "code/integ10k/strat_roll_read.py", "rh_read.py", "rh_strat.py", "rh_t0.py", "rh_roll.py", "rh_replay.py", "c_exec.py", "rh_dryrun.py", "rh_t0_dry.py"):
    try:
        ast.parse(open(HERE + "/" + f, encoding="utf-8").read())
        chk(f"parses {f}", True)
    except SyntaxError as e:
        chk(f"parses {f}", False, e)
for v in (C.W + "/../../../v8proof/pod/stageB/hierrep_read.py", "F:/TanitAD-scratch/v8proof/pod/stageB/hierrep_read.py", "F:/TanitAD-scratch/v8proof/res/cache/frames.npz", "F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py"):
    if "../" not in v:
        chk(f"reader dependency {v}", os.path.exists(v))
q3a = subprocess.run([C.PY, HERE + "/code/q3/q3_read_r3.py", "read", "--help"], capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, HIERREP_STAGEB="F:/TanitAD-scratch/v8proof/pod/stageB", TANITEVAL_ROOT=C.TREE + "/taniteval"), cwd=HERE + "/code/q3")
chk("q3_read_r3.py imports and takes --t1-root --pairs --seeds --spread-arm --same-sign", q3a.returncode == 0 and "--same-sign" in q3a.stdout, q3a.stderr[-200:])
hwa = subprocess.run([C.PY, HERE + "/code/headway/headway_read_r2.py", "read", "--help"], capture_output=True, text=True, encoding="utf-8",
                     env=dict(os.environ, HEADWAY_Q3DIR=HERE + "/code/q3", HIERREP_STAGEB="F:/TanitAD-scratch/v8proof/pod/stageB", TANITEVAL_ROOT=C.TREE + "/taniteval"), cwd=HERE + "/code/headway")
chk("headway_read_r2.py imports and takes --t1-root --pairs --seeds", hwa.returncode == 0 and "--t1-root" in hwa.stdout, hwa.stderr[-200:])
sr = subprocess.run([C.PY, "-c", "import sys; sys.path.insert(0, r'%s'); import rh_strat as R; r=R.S.run_s4(); print(bool(r['pass']))" % HERE], capture_output=True, text=True, encoding="utf-8")
chk("rh_strat imports the frozen ADD66 reader and its s4 synthetic control passes", sr.stdout.strip().endswith("True"), sr.stdout + sr.stderr[-200:])
st = subprocess.run([C.PY, HERE + "/rh_read.py", "--selftest"], capture_output=True, text=True, encoding="utf-8")
chk("rh_read.py selftest (K-SELF, K-PLANT, K-FLOOR, K-SEEDS, K-HARMS, mutation other-seed arm)", st.returncode == 0 and "all hold: True" in st.stdout, st.stdout.strip().splitlines()[-1] if st.stdout else st.stderr[-200:])
sf = subprocess.run([C.PY, HERE + "/rh_strat_floor.py", "--help"], capture_output=True, text=True, encoding="utf-8")
chk("rh_strat_floor.py imports the frozen ADD66 reader and takes --s0 --s1 --out", sf.returncode == 0 and "--s0" in sf.stdout, sf.stderr[-200:])
chk("rh_read_chain.sh has the strat_floor step before read, and rh_read.py takes --strat-floor", open(HERE + "/rh_read_chain.sh", encoding="utf-8").read().index("step strat_floor") < open(HERE + "/rh_read_chain.sh", encoding="utf-8").read().index('step read "') and "--strat-floor" in open(HERE + "/rh_read.py", encoding="utf-8").read())
pr = subprocess.run([C.PY, "-c", "import sys; sys.path.insert(0, r'%s'); import rh_common as C; print(','.join('I10%%s:I10%%s'%%(a,b) for a,b in C.PAIRS.values()).count(':'))" % HERE], capture_output=True, text=True, encoding="utf-8")
chk("pair list renders", pr.stdout.strip() == str(len(C.PAIRS)), pr.stdout)
b = subprocess.run(["C:/Program Files/Git/usr/bin/bash.exe", "-n", HERE + "/rh_read_chain.sh"], capture_output=True, text=True)
chk("bash -n rh_read_chain.sh", b.returncode == 0, b.stderr[:200])
# 6 sha256 of every script
files = sorted(f for f in glob.glob(HERE + "/*") if os.path.isfile(f) and f.endswith((".py", ".sh", ".cmd", ".md", ".json", ".txt")) and not f.endswith(("MANIFEST_SHA256.txt",)))
files += sorted(glob.glob(HERE + "/code/*/*.py") + glob.glob(HERE + "/code/*/*.md"))
man = {os.path.relpath(f, HERE).replace("\\", "/"): sha(f) for f in files}
json.dump(man, open(HERE + "/tmp/dry_sha.json", "w"), indent=0)
chk("sha256 of every script recorded", len(man) > 20, f"{len(man)} files")
json.dump({"ok": not bad, "failed": bad, "checks": res}, open(HERE + "/DRY_CHECK.json", "w", encoding="utf-8"), indent=1)
print("ZZ DRY-CHECK " + ("PASS" if not bad else "FAIL " + ",".join(bad)) + " ZZ")
sys.exit(1 if bad else 0)

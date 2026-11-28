#!/usr/bin/env python3
"""l3_tree_prep.py -- dev-box port of the pod's tree prep (blobcheck.py + the file-count check of rh_gsuite.sh / ki_tree_prep.sh). CPU, seconds, no GPU, no lock.
Verifies the code tree (default C:/Users/Admin/lever3/tree_cdb, `git archive 97fe0a56` of stack + taniteval) BLOB BY BLOB against `git ls-tree -r <commit> stack taniteval` of the mirror
(C:/Users/Admin/tanitad-push/.git), and that it holds exactly as many files (no __pycache__, nothing extra). Writes <kit>/tree_ok.json (done-file, plain write) and prints `ZZ L3 TREE OK ZZ`.
usage: l3_tree_prep.py [--tree C:/Users/Admin/lever3/tree_cdb] [--commit 97fe0a56767320dbc5f4b823d67fb07a3209a745]"""
import argparse, hashlib, json, os, subprocess, sys, time

KIT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ap = argparse.ArgumentParser()
ap.add_argument("--tree", default="C:/Users/Admin/lever3/tree_cdb")
ap.add_argument("--commit", default="97fe0a56767320dbc5f4b823d67fb07a3209a745")
ap.add_argument("--git-dir", default="C:/Users/Admin/tanitad-push/.git")
a = ap.parse_args()
assert len(a.commit) == 40, "commit must be 40 hex"
out = subprocess.run(["git", "--git-dir", a.git_dir, "ls-tree", "-r", a.commit, "stack", "taniteval"], capture_output=True, text=True, encoding="utf-8", check=True).stdout
want = {}
for ln in out.splitlines():
    meta, path = ln.split("\t", 1)
    want[path] = meta.split()[2]
bad = []
for p, h in want.items():
    fp = f"{a.tree}/{p}"
    try:
        d = open(fp, "rb").read()
    except OSError:
        bad.append(("missing", p))
        continue
    if hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest() != h:
        bad.append(("differs", p))
have = [os.path.relpath(os.path.join(r, f), a.tree).replace("\\", "/") for r, ds, fs in os.walk(a.tree) for f in fs if "__pycache__" not in r]
extra = sorted(set(have) - set(want))
ok = not bad and not extra and len(have) == len(want)
rec = {"tree": a.tree, "commit": a.commit, "n_expected": len(want), "n_files": len(have), "bad": bad[:10], "extra": extra[:10], "ok": ok, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
print(json.dumps(rec))
if ok:
    open(KIT + "/tree_ok.json", "w").write(json.dumps(rec))
    print("ZZ L3 TREE OK ZZ")
else:
    print("ZZ L3 TREE FAIL ZZ")
sys.exit(0 if ok else 1)

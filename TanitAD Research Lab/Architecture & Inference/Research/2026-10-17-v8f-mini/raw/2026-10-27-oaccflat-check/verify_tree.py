import subprocess, sys, os
from pathlib import Path
tree = Path(sys.argv[1]); tip = sys.argv[2]
G = ["git", "-c", "safe.directory=*", "-C", "C:/Users/Admin/tanitad-push"]
out = subprocess.run(G + ["ls-tree", "-r", "-z", tip, "stack", "taniteval"], capture_output=True).stdout
exp = {}
for rec in out.split(b"\0"):
    if not rec: continue
    meta, path = rec.split(b"\t", 1)
    exp[path.decode("utf-8")] = meta.split()[2].decode()
files = [p for p in tree.rglob("*") if p.is_file() and p.suffix == ".py" and "experiments" not in p.parts and ("stack" in p.relative_to(tree).parts[:1] or "taniteval" in p.relative_to(tree).parts[:1])]
rel = [p.relative_to(tree).as_posix() for p in files]
inp = "\n".join(str(p) for p in files).encode()
h = subprocess.run(["git", "hash-object", "--no-filters", "--stdin-paths"], input=inp, capture_output=True).stdout.decode().split()
ok = bad = nt = 0
for r, hh in zip(rel, h):
    e = exp.get(r)
    if e is None: nt += 1; print("NOT_IN_TIP", r)
    elif e != hh: bad += 1; print("DIFF", r)
    else: ok += 1
print(f"tip {tip}: ok {ok} diff {bad} notintip {nt} of {len(rel)} .py files")

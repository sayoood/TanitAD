"""blobcheck.py TREE COMMIT -- every file under stack/ and taniteval/ of TREE has the git blob sha of COMMIT; lists extras/missing. Extras allowed: stack/scripts/v8f_t1_eval_hb.py (checked against bcf267ce by its own sha256)."""
import subprocess, sys, os
tree, commit = sys.argv[1], sys.argv[2]
G = ["git", "-C", "C:/Users/Admin/tanitad-push"]
ls = subprocess.run(G + ["ls-tree", "-r", commit, "stack", "taniteval"], capture_output=True, text=True, check=True).stdout.splitlines()
want = {l.split("\t", 1)[1]: l.split()[2] for l in ls}
bad = []; n = 0
for p, sha in want.items():
    f = os.path.join(tree, p)
    if not os.path.isfile(f): bad.append(("missing", p)); continue
    d = open(f, "rb").read()
    import hashlib
    got = hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest()
    n += 1
    if got != sha: bad.append(("differs", p))
extra = []
for root in ("stack", "taniteval"):
    for dp, dn, fn in os.walk(os.path.join(tree, root)):
        for x in fn:
            rel = os.path.relpath(os.path.join(dp, x), tree).replace("\\", "/")
            if rel not in want: extra.append(rel)
print("checked", n, "of", len(want), "bad", len(bad), "extra", extra)
for b in bad[:20]: print(b)
sys.exit(1 if bad else 0)

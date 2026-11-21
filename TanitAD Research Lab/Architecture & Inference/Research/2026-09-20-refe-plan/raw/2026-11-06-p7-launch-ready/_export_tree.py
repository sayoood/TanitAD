"""Raw-blob export (git cat-file --batch) of a path set from the mirror at a fixed ref.
No autocrlf, no checkout. usage: _export_tree.py <git-dir> <ref> <dest> <strip-prefix> <path-prefix> [...]"""
import subprocess, sys, os
gitdir, ref, dest, strip = sys.argv[1:5]
prefixes = sys.argv[5:]
def git(*a, inp=None):
    return subprocess.run(["git", "--git-dir", gitdir, *a], input=inp, capture_output=True, check=True).stdout
ents = []
for p in prefixes:
    out = git("ls-tree", "-r", "-z", "-l", ref, "--", p)
    for rec in out.split(b"\0"):
        if not rec: continue
        meta, path = rec.split(b"\t", 1)
        mode, typ, sha, size = meta.split()
        if typ == b"blob":
            ents.append((path.decode("utf-8"), sha.decode()))
ents = sorted(set(ents))
proc = subprocess.Popen(["git", "--git-dir", gitdir, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
n = 0
for path, sha in ents:
    proc.stdin.write((sha + "\n").encode()); proc.stdin.flush()
    hdr = proc.stdout.readline().split()
    size = int(hdr[2])
    data = proc.stdout.read(size); proc.stdout.read(1)
    rel = path[len(strip):].lstrip("/") if path.startswith(strip) else path
    dst = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f: f.write(data)
    n += 1
proc.stdin.close(); proc.wait()
print("exported", n, "files to", dest)

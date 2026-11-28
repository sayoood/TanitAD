"""Export stack/ and taniteval/ of a commit from the mirror to a directory with raw bytes (git cat-file --batch; autocrlf breaks hashes).
usage: python export_tree.py <commit> <outdir>"""
import subprocess
import sys
from pathlib import Path

MIRROR = "C:/Users/Admin/tanitad-push"
commit, out = sys.argv[1], Path(sys.argv[2])
ls = subprocess.run(["git", "-C", MIRROR, "ls-tree", "-r", "-z", commit, "--", "stack", "taniteval"], capture_output=True, check=True).stdout
ents = []
for rec in ls.split(b"\0"):
    if not rec:
        continue
    meta, path = rec.split(b"\t", 1)
    mode, typ, sha = meta.decode().split()
    if typ == "blob":
        ents.append((mode, sha, path.decode()))
p = subprocess.Popen(["git", "-C", MIRROR, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
n = 0
for mode, sha, path in ents:
    p.stdin.write((sha + "\n").encode())
    p.stdin.flush()
    hdr = p.stdout.readline().split()
    size = int(hdr[2])
    data = p.stdout.read(size)
    p.stdout.read(1)
    assert len(data) == size, path
    dst = out / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    n += 1
p.stdin.close()
p.wait()
print("exported", n, "blobs of", len(ents), "to", out)

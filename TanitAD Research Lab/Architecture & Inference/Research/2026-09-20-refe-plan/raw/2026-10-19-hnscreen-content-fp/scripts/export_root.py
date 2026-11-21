import subprocess, os, sys
REPO = "C:/Users/Admin/tanitad-push"
TIP = sys.argv[1]
dsts = ["C:/Users/Admin/par18/hnfp2/base", "C:/Users/Admin/par18/hnfp2/work"]
ents = []
for sub in sys.argv[2:]:
    out = subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "-z", "-l", TIP, "--", f"{sub}/"], capture_output=True, check=True).stdout
    for rec in out.split(b"\0"):
        if not rec: continue
        meta, path = rec.split(b"\t", 1)
        mode, typ, sha, size = meta.split()
        ents.append((sha.decode(), path.decode("utf-8")))
print(len(ents), "entries")
proc = subprocess.Popen(["git", "-C", REPO, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for sha, path in ents:
    proc.stdin.write((sha + "\n").encode()); proc.stdin.flush()
    hdr = proc.stdout.readline().split()
    size = int(hdr[2]); data = proc.stdout.read(size); proc.stdout.read(1)
    assert len(data) == size
    for dst in dsts:
        p = os.path.join(dst, path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(data)
proc.stdin.close(); proc.wait()
print("done")

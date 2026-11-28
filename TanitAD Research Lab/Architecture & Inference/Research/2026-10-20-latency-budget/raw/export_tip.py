"""Raw export (cat-file --batch, no eol conversion) of tools/ and TanitAD Research Lab/Library/ at a pinned tip."""
import subprocess, sys, os
repo = r"C:/Users/Admin/tanitad-push"
tip = sys.argv[1]
dest = sys.argv[2]
paths = ["tools/", "TanitAD Research Lab/Library/"]
ls = subprocess.run(["git","-C",repo,"ls-tree","-r","-z",tip,"--"]+paths, capture_output=True, check=True).stdout
ents=[]
for rec in ls.split(b"\0"):
    if not rec: continue
    meta, path = rec.split(b"\t",1)
    mode, typ, sha = meta.split()
    ents.append((sha.decode(), path.decode()))
print("entries", len(ents), flush=True)
p = subprocess.Popen(["git","-C",repo,"cat-file","--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
n=0
for sha, path in ents:
    p.stdin.write((sha+"\n").encode()); p.stdin.flush()
    hdr = p.stdout.readline().split()
    size = int(hdr[2])
    out = os.path.join(dest, path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,"wb") as f:
        left=size
        while left:
            b=p.stdout.read(min(left,1<<20)); f.write(b); left-=len(b)
    p.stdout.read(1)
    n+=1
print("done", n, flush=True)
p.stdin.close(); p.wait()

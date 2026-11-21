import subprocess, os, sys
REPO = "C:/Users/Admin/tanitad-push"
TIP = sys.argv[1]
DSTS = ["C:/Users/Admin/par18/hnfp3/base", "C:/Users/Admin/par18/hnfp3/work"]
RP = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
SUBS = [RP + "/refe/", RP + "/eval/", RP + "/code/", RP + "/splits/", RP + "/report/"]
out = subprocess.run(["git","-C",REPO,"ls-tree","-r","-z","-l",TIP,"--",RP+"/","Project Steering/MODEL_REGISTRY.md"],capture_output=True,check=True).stdout
ents=[]
for rec in out.split(b"\0"):
    if not rec: continue
    meta, path = rec.split(b"\t",1)
    mode, typ, sha, size = meta.split()
    p = path.decode("utf-8")
    top = p.startswith(RP + "/") and "/" not in p[len(RP)+1:]
    if p == "Project Steering/MODEL_REGISTRY.md" or top or any(p.startswith(s) for s in SUBS):
        ents.append((sha.decode(), p))
print(len(ents), "entries")
proc = subprocess.Popen(["git","-C",REPO,"cat-file","--batch"],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for sha, path in ents:
    proc.stdin.write((sha+"\n").encode()); proc.stdin.flush()
    hdr = proc.stdout.readline().split()
    size = int(hdr[2]); data = proc.stdout.read(size); proc.stdout.read(1)
    assert len(data)==size
    for DST in DSTS:
        dst = os.path.join(DST, path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst,"wb") as f: f.write(data)
proc.stdin.close(); proc.wait()
print("done")

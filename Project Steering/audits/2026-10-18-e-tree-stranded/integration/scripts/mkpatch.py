import subprocess, json, os, sys, hashlib, shutil
MIRROR="C:/Users/Admin/tanitad-push"; TIP=sys.argv[1]; ER="E:/Projects/TanitAD/"
PK=sys.argv[2]   # package-relative dir containing stack/...
OUTD=sys.argv[3]; NAME=sys.argv[4]
files=sys.argv[5:]  # repo-relative targets
def git(*a, cwd=None):
    return subprocess.run(["git","-C",MIRROR,"-c","core.autocrlf=false","--no-optional-locks",*a],capture_output=True)
os.makedirs(OUTD,exist_ok=True)
W="C:/Users/Admin/integ/work/pw_"+NAME
shutil.rmtree(W,ignore_errors=True)
os.makedirs(W+"/a"); os.makedirs(W+"/b")
meta=[]
for f in files:
    r=git("rev-parse",f"{TIP}:{f}")
    tb=r.stdout.decode().strip() if r.returncode==0 else None
    v=open(ER+PK+"/"+f,"rb").read()
    tdat=git("cat-file","blob",tb).stdout if tb else None
    # keep the tip's EOL convention for the proposed bytes
    if tdat is not None:
        tcrlf=b"\r\n" in tdat
        vl=v.replace(b"\r\n",b"\n")
        v=vl.replace(b"\n",b"\r\n") if tcrlf else vl
        os.makedirs(os.path.dirname(W+"/a/"+f),exist_ok=True)
        open(W+"/a/"+f,"wb").write(tdat)
    os.makedirs(os.path.dirname(W+"/b/"+f),exist_ok=True)
    open(W+"/b/"+f,"wb").write(v)
    meta.append({"path":f,"tip_blob":tb,"new":tdat is None,"variant_sha1":hashlib.sha1(b"blob %d\0"%len(v)+v).hexdigest()})
# diff (run inside W so paths are a/<f> b/<f>)
dr=subprocess.run(["git","-c","core.autocrlf=false","diff","--no-index","--no-color","--src-prefix=a/","--dst-prefix=b/","a","b"],cwd=W,capture_output=True)
patch=dr.stdout
patch=patch.replace(b"a/a/",b"a/").replace(b"b/b/",b"b/").replace(b" a/b/",b" a/")
open(f"{OUTD}/{NAME}.patch","wb").write(patch)
# verify: apply onto a/ copy and compare to b/
V=W+"/verify"; shutil.copytree(W+"/a",V)
ap=subprocess.run(["git","-c","core.autocrlf=false","apply","--check","-p1","--directory=.","--unsafe-paths",os.path.abspath(f"{OUTD}/{NAME}.patch")],cwd=V,capture_output=True)
ok_check=ap.returncode==0
res=None
if ok_check:
    subprocess.run(["git","-c","core.autocrlf=false","apply","-p1",os.path.abspath(f"{OUTD}/{NAME}.patch")],cwd=V,check=True)
    res=all(open(V+"/"+m["path"],"rb").read()==open(W+"/b/"+m["path"],"rb").read() for m in meta)
print(NAME,"patch bytes",len(patch),"apply_check",ok_check,ap.stderr.decode()[:300],"byte_identical_after_apply",res)
json.dump({"name":NAME,"files":meta,"apply_check":ok_check,"byte_identical_after_apply":res},open(f"{OUTD}/{NAME}.json","w"),indent=1)

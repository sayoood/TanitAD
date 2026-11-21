import subprocess, json, os, sys, hashlib, shutil
MIRROR="C:/Users/Admin/tanitad-push"; TIP=sys.argv[1]; ER="E:/Projects/TanitAD/"
X=json.load(open('X.json'))
OUTD="patches_out"
def git(*a, inp=None):
    return subprocess.run(["git","-C",MIRROR,"-c","core.autocrlf=false","--no-optional-locks",*a],capture_output=True,input=inp)
def lf(b): return b.replace(b"\r\n",b"\n")
def build(name, items, W):
    shutil.rmtree(W,ignore_errors=True); os.makedirs(W+"/a"); os.makedirs(W+"/b")
    rep=[]
    for it in items:
        repo=it['repo']; pkg=it.get('pkg_resolved') or it['pkg']
        v=open(ER+pkg,'rb').read()
        tb=git("rev-parse",f"{TIP}:{repo}").stdout.decode().strip() if git("rev-parse","--verify","-q",f"{TIP}:{repo}").returncode==0 else None
        tdat=git("cat-file","blob",tb).stdout if tb else None
        status="NEW" if tdat is None else "?"
        newb=v
        if tdat is not None:
            tcrlf=b"\r\n" in tdat
            vl=lf(v)
            decl_base=it.get('decl_base')
            if decl_base and decl_base!='NEW' and tb!=decl_base and git("cat-file","-e",decl_base).returncode==0:
                base=lf(git("cat-file","blob",decl_base).stdout)
                tl=lf(tdat)
                d=W+"/m"; os.makedirs(d,exist_ok=True)
                open(d+"/base","wb").write(base); open(d+"/ours","wb").write(tl); open(d+"/theirs","wb").write(vl)
                r=subprocess.run(["git","-c","core.autocrlf=false","merge-file","-p",d+"/ours",d+"/base",d+"/theirs"],capture_output=True)
                nconf=r.returncode  # number of conflicts (>0), negative on error
                if nconf==0:
                    vl=r.stdout; status="MERGED-CLEAN(3-way base=declared)"
                else:
                    status=f"CONFLICT({nconf})"; vl=None
            else:
                status="TIP==DECLARED-BASE(direct)" if tb==it.get('decl_base') else "NO-DECLARED-BASE(direct)"
            if vl is not None:
                newb=vl.replace(b"\n",b"\r\n") if tcrlf else vl
            else:
                newb=None
        rep.append({"repo":repo,"status":status,"tip_blob":tb})
        if tdat is not None:
            os.makedirs(os.path.dirname(W+"/a/"+repo),exist_ok=True); open(W+"/a/"+repo,"wb").write(tdat)
        if newb is not None:
            os.makedirs(os.path.dirname(W+"/b/"+repo),exist_ok=True); open(W+"/b/"+repo,"wb").write(newb)
        elif tdat is not None:
            # conflict: leave unchanged in b so the patch omits it
            os.makedirs(os.path.dirname(W+"/b/"+repo),exist_ok=True); open(W+"/b/"+repo,"wb").write(tdat)
    dr=subprocess.run(["git","-c","core.autocrlf=false","diff","--no-index","--no-color","a","b"],cwd=W,capture_output=True)
    patch=dr.stdout.replace(b"a/a/",b"a/").replace(b"b/b/",b"b/").replace(b" a/b/",b" a/")
    open(f"{OUTD}/{name}.patch","wb").write(patch)
    V=W+"/verify"; shutil.copytree(W+"/a",V)
    ok=subprocess.run(["git","-c","core.autocrlf=false","apply","--check","-p1",os.path.abspath(f"{OUTD}/{name}.patch")],cwd=V,capture_output=True).returncode==0
    ident=None
    if ok:
        subprocess.run(["git","-c","core.autocrlf=false","apply","-p1",os.path.abspath(f"{OUTD}/{name}.patch")],cwd=V,check=True)
        ident=all(open(V+"/"+x['repo'],"rb").read()==open(W+"/b/"+x['repo'],"rb").read() for x in rep if os.path.exists(W+"/b/"+x['repo']))
    json.dump({"name":name,"files":rep,"apply_check":ok,"byte_identical_after_apply":ident,"patch_bytes":len(patch)},open(f"{OUTD}/{name}.json","w"),indent=1)
    print(name,"bytes",len(patch),"apply_check",ok,"identical",ident); [print("  ",x['status'],x['repo']) for x in rep]
sel=lambda pred:[r for r in X if r['status'] in ('DIFFERS-NOT-IN-HISTORY','NEW-NOT-ON-TIP') and pred(r)]
pk=lambda r:(r.get('pkg_resolved') or r['pkg'])
build("refav1_R1R6_plus_R1VMAX", sel(lambda r:'refav1' in pk(r) and 'v7f_merge' not in pk(r) and 'v7f_gate' not in pk(r) or pk(r).endswith('refa_v1_plan.py') or pk(r).endswith('refav1_arm.py')) , "C:/Users/Admin/integ/work/pw_refav1")
build("v7f_gate_profile", sel(lambda r:'v7f_gate' in pk(r)), "C:/Users/Admin/integ/work/pw_v7fgate")

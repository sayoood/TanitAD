import os, subprocess, json, sys
import xml.etree.ElementTree as ET
O="F:/TanitAD-scratch/actrsp/optb"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
TMP=O+"/tmp"
files=[l.strip() for l in open(O+"/failing_files.txt") if l.strip()]
out=O+"/regress_pertest.json"
res=json.load(open(out)) if os.path.exists(out) else {}
for f in files:
    for side,root in (("new",O+"/tree"),("base",O+"/basetree")):
        key=side+"|"+f
        if key in res: continue
        subprocess.run([PY,"F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py","6","120"])
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="2",TMP=TMP,TEMP=TMP,PYTHONDONTWRITEBYTECODE="1",
          PYTHONPATH=";".join(os.path.abspath(root+p) for p in ("/stack","/taniteval","/stack/scripts")))
        xml=f"{TMP}/j_{side}.xml"
        if os.path.exists(xml): os.remove(xml)
        bt=f"{TMP}/bt_{side}"
        subprocess.run(["cmd","/c","rmdir","/s","/q",bt.replace("/","\\")],capture_output=True)
        try:
            r=subprocess.run([PY,"-m","pytest","-q","--no-header","-p","no:cacheprovider","--basetemp",bt,f"--junitxml={xml}",f],cwd=root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=2400)
            rc=r.returncode
        except subprocess.TimeoutExpired: rc="timeout"
        d={}
        try:
            for tc in ET.parse(xml).getroot().iter("testcase"):
                st="pass"; msg=""
                for ch in tc:
                    if ch.tag in("failure","error"): st="fail"; msg=(ch.get("message") or "")[:200]
                    elif ch.tag=="skipped" and st=="pass": st="skip"
                d[(tc.get("classname") or "")+"::"+(tc.get("name") or "")]=[st,msg]
        except Exception as e: d["__error__"]=["fail",str(e)]
        res[key]={"rc":rc,"t":d}
        json.dump(res,open(out,"w"))
open(O+"/regress2.done","w").write("done")

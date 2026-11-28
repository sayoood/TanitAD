import json,subprocess,sys,os
sys.path.insert(0,'F:/safehead_cache/code')
import sh_lib as L, sh_env as E
CT=L.wp('ctrl'); job=json.load(open(CT+'/job_ctrl.json')); os.makedirs(CT+'/lab_dump',exist_ok=True)
logs=job['logs'] if len(sys.argv)<2 else job['logs'][:int(sys.argv[1])]
for e in logs:
    rc=subprocess.call([L.PY_N,'F:/safehead_cache/code/sh_label.py','--dump',CT+f"/dump/{e['log']}.dump.npz",'--db',e['db'],'--out',CT+f"/lab_dump/{e['log']}.lab.npz",'--workers','4','--chunk','2'],env=E.nenv())
    print(e['log'],rc,flush=True)

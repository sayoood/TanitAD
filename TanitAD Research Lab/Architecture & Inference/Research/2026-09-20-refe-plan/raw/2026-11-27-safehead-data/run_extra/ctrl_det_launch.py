import subprocess,sys
sys.path.insert(0,'F:/safehead_cache/code')
import sh_lib as L, sh_env as E
CT=L.wp('ctrl')
rc=subprocess.call([L.PY_D,'F:/safehead_cache/code/sh_dump.py','--job',CT+'/job_det.json','--ckpt',L.C1_CKPT,'--expect-sha256',L.C1_SHA,'--out-dir',CT+'/dump2','--jobname','refe-safehead-ctrl-det'],env=E.denv(),stdout=open(L.wp('logs','ctrl_dump2.log'),'a'),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
print('rc',rc)

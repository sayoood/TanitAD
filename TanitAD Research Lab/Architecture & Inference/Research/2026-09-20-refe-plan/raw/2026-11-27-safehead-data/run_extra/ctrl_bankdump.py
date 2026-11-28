import json,os,sys
import numpy as np
sys.path.insert(0,'F:/safehead_cache/code')
import sh_lib as L
CT=L.wp('ctrl'); job=json.load(open(CT+'/job_ctrl.json')); b=np.load(CT+'/bank_ctrl.npz')
bt=[str(t) for t in b['token']]; os.makedirs(CT+'/dump_bank',exist_ok=True)
for e in job['logs']:
    ix=[bt.index(t) for t in e['tokens']]
    L.save_npz_det(CT+f"/dump_bank/{e['log']}.dump.npz",token=np.array(e['tokens']),props8=b['proposals'][ix],pick=b['pick'][ix],logits=b['logits'][ix],s=np.zeros((len(ix),64,256),np.float32))
print('ok')

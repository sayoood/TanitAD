import sys,os,csv,numpy as np
sys.path.insert(0,'F:/safehead_cache/code')
import sh_lib as L, sh_ctrl as C
lab=sys.argv[1]; rows={r['token']:r for r in csv.DictReader(open(C.BANK_CSV))}
dd=C._load_dumps('F:/safehead_cache/ctrl/dump_bank'); n=0; bad=[]
for f in sorted(os.listdir(lab)):
    if f.endswith('.lab.npz'):
        d=np.load(os.path.join(lab,f))
        for i,t in enumerate(d['token']):
            t=str(t);k=int(dd[t]['pick']);n+=1
            e=abs(float(d['tr_ep'][i][k])-float(rows[t]['ego_progress']))
            if e>1e-4: bad.append((t,round(e,5),float(d['tr_ep'][i][k]),float(rows[t]['ego_progress'])))
print(n,len(bad),bad[:8])

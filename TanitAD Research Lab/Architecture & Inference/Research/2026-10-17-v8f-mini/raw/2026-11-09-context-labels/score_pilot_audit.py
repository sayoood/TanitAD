import sys,json,re,base64,glob,os,math,collections
D='E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/strat_labels'
sys.path.insert(0,D)
import numpy as np, pandas as pd
import merge_context as M, extract_context as X
A=json.load(open(D+'/pilot_audit_answers_PI_60.json')); fr=A['frames']
print('incomplete:',[(a['order'],[f for f in ('weather','surface','visibility','illumination','tod') if a[f] is None]) for a in fr if any(a[f] is None for f in ('weather','surface','visibility','illumination','tod'))])
tab=pd.read_parquet(D+'/splits/clips_table.parquet')
clips=sorted(os.path.basename(p)[:-4] for p in glob.glob('F:/strat_out/pilot/ctxB/*.npz'))
data={c:M.load_parts('F:/strat_out/pilot',c) for c in clips}
train=[c for c in tab[tab.split=='train4512'].clip_id if c in data]
print('clips',len(data),'train',len(train)); P=M.fit_probes(data,train)
print({k:(type(v).__name__, list(v[0].keys())[:1] if isinstance(v,tuple) else None) for k,v in P.items()})
html=open('F:/strat_out/pilot/audit60/PILOT_AUDIT.html',encoding='utf8').read()
H=json.loads(base64.b64decode(re.search(r'const HIDDEN_B64 = "([^"]+)"',html).group(1)))
print('audit frames at Qwen 1 Hz:',sum(1 for a in fr if a['frame_idx']%5==0),'of',len(fr))
CL={'weather':X.WEATHER,'surface':X.SURFACE,'visibility':X.VISIB}
def wil(k,n,z=1.96):
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return (100*(c-h)/d,100*(c+h)/d)
for scope,sel in (('random40',lambda a:a['stratum']=='random'),('all60',lambda a:True)):
    for f,cl in CL.items():
        oof=P[f][0]; n=k=q=0
        for a,h in zip(fr,H):
            if not sel(a) or a[f] in (None,'cant_tell') or a['clip_id'] not in oof: continue
            pr=oof[a['clip_id']][a['frame_idx']]; lab=cl[int(np.argmax(pr))]; n+=1; k+=int(lab==a[f]); q+=int(h['src'][f]['qwen']==a[f])
        print(scope,f,'probe(OOF) vs PI',f'{k}/{n}',round(100*k/n,1),[round(x) for x in wil(k,n)],'| Qwen vs PI',f'{q}/{n}',round(100*q/n,1))
# probe vs Qwen at all 1 Hz frames, out of fold
for f,cl in CL.items():
    oof=P[f][0]; key={'weather':'weather','surface':'surface','visibility':'visib'}[f]
    n=k=0
    for c,(a,b) in data.items():
        if c not in oof: continue
        cand=[x for x in b if key in x.lower() and b[x].ndim>=2]
        if not cand: continue
        q=b[cand[0]]; idx=np.arange(0,len(q))*5
        pr=oof[c][idx]; n+=len(q); k+=int((pr.argmax(1)==q.argmax(1)).sum())
    print(f,'OOF probe vs Qwen at 1 Hz frames',k,n,round(100*k/max(n,1),1), cand if n else list(b.keys()))
# illumination with glare merged into day
ill={'dark':'dark','lit_night':'lit_night','twilight':'twilight','day':'day','glare':'day'}
for scope,sel in (('random40',lambda a:a['stratum']=='random'),('all60',lambda a:True)):
    n=k=0
    for a,h in zip(fr,H):
        if not sel(a) or a['illumination'] in (None,'cant_tell'): continue
        n+=1; k+=int(ill[a['illumination']]==ill[h['src']['illumination']['siglip']])
    print(scope,'illumination glare=day',f'{k}/{n}',round(100*k/n,1),[round(x) for x in wil(k,n)])

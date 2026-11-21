import json, subprocess, sys, collections, os
MIRROR="C:/Users/Admin/tanitad-push"; TIP=sys.argv[1]; ER="E:/Projects/TanitAD/"
B=json.load(open('B.json'))
REG=('Project Steering/BACKLOG.md','Project Steering/GOALS_AND_CLAIMS.md','Project Steering/MODEL_REGISTRY.md','Project Steering/PI_DECISION_QUEUE.md','Project Steering/RETRACTION_LOG.md','TanitAD Research Lab/Library/LIBRARY.md','TanitAD Research Lab/Library/library.json','.claude/settings.local.json')
def git(*a, inp=None):
    return subprocess.run(["git","-C",MIRROR,"-c","core.autocrlf=false","--no-optional-locks",*a],capture_output=True,input=inp).stdout
def lines_of(b):
    return [l.rstrip() for l in b.decode('utf8','replace').replace('\r\n','\n').split('\n')]
pat={}  # line -> set(item idx)
items={}
for i,r in enumerate(B):
    p=r['path']
    if p in REG or '/raw/' in p or 'tip_blob' not in r: continue
    el=lines_of(open(ER+p,'rb').read()); tl=lines_of(git('cat-file','blob',r['tip_blob']))
    ce,ct=collections.Counter(el),collections.Counter(tl)
    only=[]; seen=set()
    for l in el:
        s=l.strip()
        if (ce-ct).get(l,0)>0 and len(s)>=28 and s not in seen:
            seen.add(s); only.append(s)
    items[i]=only
    for s in only: pat.setdefault(s,set()).add(i)
print('items',len(items),'patterns',len(pat),flush=True)
pf='C:/Users/Admin/integ/work/pat.txt'
open(pf,'w',encoding='utf8',newline='\n').write('\n'.join(pat.keys())+'\n')
r=subprocess.run(["git","-C",MIRROR,"-c","core.autocrlf=false","--no-optional-locks","grep","-I","-F","-h","-o","-f",pf,TIP,"--",":(exclude)*/raw/*",":(exclude)*.jsonl",":(exclude)*.csv"],capture_output=True)
found=set(x.strip() for x in r.stdout.decode('utf8','replace').split('\n') if x.strip())
print('grep rc',r.returncode,'matched strings',len(found),flush=True)
out={}
for i,only in items.items():
    f=[s for s in only if s in found]
    out[i]={'path':B[i]['path'],'n_E_only_len28':len(only),'n_found_on_tip_elsewhere':len(f),'absent_examples':[s[:110] for s in only if s not in found][:6]}
    print(i,B[i]['path'][-55:],'E-only(>=28ch)',len(only),'found-on-tip',len(f),flush=True)
json.dump(out,open('B3.json','w'),indent=1)

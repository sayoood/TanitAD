import json,gzip,collections as C
L=[json.loads(l) for l in gzip.open('states.jsonl.gz','rt')]
mean=lambda x:sum(x)/len(x) if x else float('nan')
cyc=C.defaultdict(lambda:[0,0]); grp=C.defaultdict(lambda:[0,0]); ch=C.defaultdict(C.Counter)
for s in L:
  for k,a in s['arms'].items():
    mode=k.split(':')[0]; i=int(k.split(':')[1]); g='fan' if i<15 else 'sweep'
    for c in a['logs']:
      f=bool(c['comf']); cyc[c['cycle']][0]+=f; cyc[c['cycle']][1]+=1
      grp[(mode,g)][0]+=f; grp[(mode,g)][1]+=1
      for x in c['comf']: ch[g][x]+=1
print('by cycle',{k:round(v[0]/v[1],3) for k,v in sorted(cyc.items())})
print('by grp',{k:(round(v[0]/v[1],3),v[1]) for k,v in grp.items()})
print(dict(ch))
n=f=0;chb=C.Counter()
for s in L:
  for m in('log','idm'):
    a=s['arms'].get(f"{m}:{s['pdm_best']}")
    if a:
      for c in a['logs']:
        n+=1;f+=bool(c['comf']);chb.update(c['comf'])
print('pdm_best arm',f,n,chb)
for nm in ['slowest:log','fastest:log','stop:log']:
  n=f=0;cc=C.Counter()
  for s in L:
    a=s['reg'].get(nm)
    if a:
      for c in a['logs']: n+=1;f+=bool(c['comf']);cc.update(c['comf'])
  print(nm,f,n,cc)
print(set(k for s in L for k in s['reg']))
lo=[0,0];hi=[0,0]
for s in L:
  for k,a in s['arms'].items():
    for c in a['logs']:
      t=lo if c['v']<2 else hi; t[0]+=bool(c['comf']);t[1]+=1
print('v<2',lo,'v>=2',hi)
ac=[abs(c['a']) for s in L for a in s['arms'].values() for c in a['logs'] if c['comf']]
an=[abs(c['a']) for s in L for a in s['arms'].values() for c in a['logs'] if not c['comf']]
print(mean(ac),mean(an))
# per state comf rate pdm_best
print([ (s['idx'], round(mean([bool(c['comf']) for m in ('log',) for c in s['arms'][f"{m}:{s['pdm_best']}"]['logs']]),2)) for s in L if f"log:{s['pdm_best']}" in s['arms']])

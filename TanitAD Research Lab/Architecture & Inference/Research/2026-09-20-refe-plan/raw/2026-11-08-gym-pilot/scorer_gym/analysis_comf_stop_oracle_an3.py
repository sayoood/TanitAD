import json,gzip,collections as C
L=[json.loads(l) for l in gzip.open('run/states.jsonl.gz','rt')]
mean=lambda x:sum(x)/len(x) if x else float('nan')
tot=0;brk=0;ps=0;pp=0
for s in L:
  st=s['reg'].get('stop:log')
  if not st or s['v0']<=0.5: continue
  for c in st['logs']:
    tot+=c['reward']; 
    if c['v']>=0.05: brk+=c['reward']
  pp+=s['arms'][f"log:{s['pdm_best']}"]['ret']
print('moving states stop total',tot,'braking-phase part',brk,'pdm best-arm total',pp,'ratio',tot/pp)
print('stop hard:',[(s['idx'],s['reg']['stop:log']['logs'][-1]['hard'],len(s['reg']['stop:log']['logs'])) for s in L if 'stop:log' in s['reg'] and s['reg']['stop:log']['logs'][-1]['hard']])
# mean reward per cycle first 5 cycles of stop vs pdm
sc=[];pc=[]
for s in L:
  st=s['reg'].get('stop:log')
  if st and s['v0']>0.5:
    sc.append(mean([c['reward'] for c in st['logs']]))
print('per-state mean reward/cycle stop',[round(x,3) for x in sc])
# stop-phase reward: progress in the stopped cycles
print('q_stall fired on stop arm', sum(1 for s in L if 'stop:log' in s['reg'] for c in s['reg']['stop:log']['logs'] if c['q']['q_stall']<1))
print('stop q_free<1 cycles',sum(1 for s in L if 'stop:log' in s['reg'] for c in s['reg']['stop:log']['logs'] if c['q']['q_free']<1), 'q_gap',sum(1 for s in L if 'stop:log' in s['reg'] for c in s['reg']['stop:log']['logs'] if c['q']['q_gap']<1))
# oracle hard cases
for s in L:
  for m in('log','idm'):
    ks=[k for k in s['arms'] if k.startswith(m+':')]
    best=max(ks,key=lambda k:s['arms'][k]['ret'])
    b=s['arms'][best]['logs'][-1]
    if b['hard']:
      allh=all(s['arms'][k]['logs'][-1]['hard'] for k in ks)
      print('oracle hard',s['idx'],m,b['hard'],'cycle',b['cycle'],'all arms hard',allh,'nhard arms',sum(1 for k in ks if s['arms'][k]['logs'][-1]['hard']),len(ks),'v0',round(s['v0'],1))
# hard by type for all arms
h=C.Counter();
for s in L:
  for k,a in s['arms'].items():
    if a['logs'][-1]['hard']: h[(k.split(':')[0],a['logs'][-1]['hard'])]+=1
print(h)
# pdm arm returns
print('mean pdm best ret',mean([s['arms'][f"log:{s['pdm_best']}"]['ret'] for s in L]))
print('q terms stalled: states with v0<0.5',[s['idx'] for s in L if s['v0']<0.5])

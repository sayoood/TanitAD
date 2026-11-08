import json,gzip,collections as C
L=[json.loads(l) for l in gzip.open('run/states.jsonl.gz','rt')]
mean=lambda x:sum(x)/len(x) if x else float('nan')
# arms with >=1 fire
n=f=0
for s in L:
  for k,a in s['arms'].items():
    n+=1; f+=any(c['comf'] for c in a['logs'])
print('arms with >=1 comf fire',f,n)
# do-nothing detail
for s in L:
  st=s['reg'].get('stop:log'); 
  if not st: continue
  lg=st['logs']; pdm=s['arms'][f"log:{s['pdm_best']}"]
  print(s['idx'],round(s['v0'],1),'stop ret',round(st['ret'],2),'cycles',len(lg),'mean r',round(mean([c['reward'] for c in lg]),3),'pdm ret',round(pdm['ret'],2),'v_end',round(lg[-1]['v'],2),'first stop cyc',next((c['cycle'] for c in lg if c['v']<0.05),None))

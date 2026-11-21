import gzip, json, collections as C
R=[json.loads(l) for l in gzip.open('E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot/raw_states.jsonl.gz','rt')]
print("states v0<0.5:",[(s['idx'],round(s['v0'],2)) for s in R if s['v0']<0.5])
def scan(logs,v0):
    pre=exc=fired=0; vs=[v0]*4
    for c in logs:
        vs.append(c['v']); w=vs[-4:]   # 4 cycles ~ last 2 s (end-of-cycle speeds; start padded with v0)
        if all(x<0.5 for x in w):
            pre+=1; fired+=c['q']['q_stall']<1
            if c['q']['q_stall']==1: exc+= 1
    return pre,fired,exc
tot=C.Counter(); per=C.defaultdict(C.Counter)
for s in R:
    for k,a in s['arms'].items():
        p,f,e=scan(a['logs'],s['v0']); tot['pre']+=p;tot['fired']+=f;tot['excused']+=e
        if p: per[s['idx']]['arm_with_pre']+=1; per[s['idx']]['pre_cycles']+=p
print("ARM SET totals",dict(tot)); print({k:dict(v) for k,v in per.items()})
tot=C.Counter()
for s in R:
    for k,a in s['reg'].items():
        p,f,e=scan(a['logs'],s['v0']); tot[k.split(':')[0]+'_pre']+=p; tot[k.split(':')[0]+'_fired']+=f
print("REG",dict(tot))
# stationary states: what do arms do, lead gap?
for s in R:
    if s['v0']<0.5:
        a=s['arms'][f"log:{s['pdm_best']}"]['logs']
        print(s['idx'],'v0',round(s['v0'],2),'best-arm v',[round(c['v'],1) for c in a][:8],'lead_gap',[None if c['lead_gap'] is None else round(c['lead_gap'],1) for c in a][:6])
print('--- excusal reason for precondition cycles (arm set)')
rc=C.Counter()
for s in R:
    for k,a in s['arms'].items():
        vs=[s['v0']]*4
        for c in a['logs']:
            vs.append(c['v'])
            if all(x<0.5 for x in vs[-4:]):
                r=[]
                if c['conf_ttc'] is not None: r.append('conflict')
                if c['lead_gap'] is not None and c['lead_gap']<=10: r.append('lead<=10m')
                rc[(s['idx'],'+'.join(r) or 'NOT conflict/lead => red-light excuse (inferred)')]+=1
for k,v in sorted(rc.items()): print(k,v)

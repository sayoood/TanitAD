"""Aggregate q2_probe outputs: precision of each q_stall excuse. usage: q2_agg.py DIR > out"""
import sys,json,glob,collections as C
D=sys.argv[1]
rows=[]
for f in sorted(glob.glob(D+'/q2_*.jsonl')):
    rows+= [json.loads(l) for l in open(f)]
st=[r for r in rows if r['stalled']]
fired=[r for r in st if r['q_stall']<1]
ex=[r for r in st if r['q_stall']>=1]
print("files",len(glob.glob(D+'/q2_*.jsonl')),"cycle records",len(rows),"stalled (code precondition)",len(st),"q_stall fired",len(fired),"excused",len(ex))
def flags(r):
    lead=r.get('lead'); L= lead is not None and lead['gap']<=10.0
    Cn= bool(r.get('confs')); R= bool(r.get('reds'))
    return L,Cn,R
def truth(r):
    lead=r.get('lead'); t={}
    # LEAD: object whose box intersects the ego-width swept corridor, not pulling away fast
    t['L_strict_geom']= bool(lead and lead['gap']<=10 and lead['strict'])
    t['L_block']= bool(lead and lead['gap']<=10 and lead['strict'] and lead['vlon']<2.0)
    # CONFLICT: dynamic object ahead (lx>0) whose straight-line path enters the strict ego-width corridor within 4 s
    t['C_strict_ahead']= any(c['t_strict'] is not None and c['lx']>0 for c in r.get('confs',[]))
    t['C_strict_ahead_le3']= any(c['t_strict'] is not None and c['t_strict']<=3.0 and c['lx']>0 for c in r.get('confs',[]))
    t['C_behind_only']= bool(r.get('confs')) and all(c['lx']<=0 for c in r['confs'])
    # RED: red connector that is on the ego route and intersects the strict corridor
    t['R_route_strict']= any(x['in_route'] and x['strict'] for x in r.get('reds',[]))
    t['R_route_any']= any(x['in_route'] for x in r.get('reds',[]))
    return t
def report(name,sub):
    print("\n##",name,"n_excused_cycles=",len(sub))
    cause=C.Counter(); prec=C.defaultdict(lambda:[0,0]); sole=C.defaultdict(lambda:[0,0]); anyvalid=[0,0]
    for r in sub:
        L,Cn,R=flags(r); t=truth(r)
        key=''.join(k for k,f in (('L',L),('C',Cn),('R',R)) if f) or '-'
        cause[key]+=1
        if L:
            prec['LEAD: code says lead<=10m; strict corridor hit'][1]+=1; prec['LEAD: code says lead<=10m; strict corridor hit'][0]+=t['L_strict_geom']
            prec['LEAD: strict hit AND not pulling away (vlon<2)'][1]+=1; prec['LEAD: strict hit AND not pulling away (vlon<2)'][0]+=t['L_block']
        if Cn:
            prec['CONFLICT: code says conf; ahead object enters strict corridor <=4s'][1]+=1; prec['CONFLICT: code says conf; ahead object enters strict corridor <=4s'][0]+=t['C_strict_ahead']
            prec['CONFLICT: ... <=3s'][1]+=1; prec['CONFLICT: ... <=3s'][0]+=t['C_strict_ahead_le3']
            prec['CONFLICT: all conflicting objects are BEHIND ego'][1]+=1; prec['CONFLICT: all conflicting objects are BEHIND ego'][0]+=t['C_behind_only']
        if R:
            prec['RED: code says red; red connector is ON ROUTE and in strict corridor'][1]+=1; prec['RED: code says red; red connector is ON ROUTE and in strict corridor'][0]+=t['R_route_strict']
            prec['RED: on route (any overlap)'][1]+=1; prec['RED: on route (any overlap)'][0]+=t['R_route_any']
        valid= (L and t['L_block']) or (Cn and t['C_strict_ahead']) or (R and t['R_route_strict'])
        anyvalid[1]+=1; anyvalid[0]+=bool(valid)
        # sole-cause precision
        if key=='L': sole['sole LEAD valid'][1]+=1; sole['sole LEAD valid'][0]+=t['L_block']
        if key=='C': sole['sole CONFLICT valid'][1]+=1; sole['sole CONFLICT valid'][0]+=t['C_strict_ahead']
        if key=='R': sole['sole RED valid'][1]+=1; sole['sole RED valid'][0]+=t['R_route_strict']
    print(" cause-combination counts (L=lead<=10, C=conflict, R=red):",dict(cause))
    for k,(a,b) in prec.items(): print("  %-80s %d/%d = %.1f%%"%(k,a,b,100*a/max(b,1)))
    for k,(a,b) in sole.items(): print("  %-80s %d/%d = %.1f%%"%(k,a,b,100*a/max(b,1)))
    print("  EXCUSED cycles with >=1 VALID excuse: %d/%d = %.1f%%  -> falsely excused (q_stall should have fired): %d"%(anyvalid[0],anyvalid[1],100*anyvalid[0]/max(anyvalid[1],1),anyvalid[1]-anyvalid[0]))
report("ALL arms/modes",ex)
report("stop-policy scenes (ego exactly still)",[r for r in ex if r['arm']=='stop'])
report("24-arm-set members (pdm_best + sweep0.5)",[r for r in ex if r['arm']!='stop'])
for m in ('log','idm'): report("mode "+m,[r for r in ex if r['mode']==m])
print("\nper-state stalled/excused/fired:")
c=C.Counter()
for r in st: c[(r['sampler'],r['state'],r['arm']=='stop','F' if r['q_stall']<1 else 'X')]+=1
for k in sorted(c): print(" ",k,c[k])

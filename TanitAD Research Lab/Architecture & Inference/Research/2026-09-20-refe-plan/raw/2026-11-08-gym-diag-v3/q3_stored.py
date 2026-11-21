"""Q3 from stored results (pilot_v2 states.jsonl.gz, log mode): sweep arms 15..23 = v_target 0.5..1.3 x max(v0,1) (capped at limit).
Free-road state = >=90 % of cycle records of the 24 arms have q_free applicable (no lead, no conflict). Exclude states with a hard event in any sweep arm
from the monotonicity count but list them."""
import gzip,json
R='E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v2/'
M=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3]
def loss_by_term(logs):
    L={}
    for c in logs:
        for t,q in c['q'].items():
            if q<1 and c['qprod']>0: L[t]=L.get(t,0)+c['progress']*(c['qprod']/q)*(1-q)
    return {k:round(v,3) for k,v in L.items()}
for run in ('run_main','run_stall'):
    for l in gzip.open(R+run+'/states.jsonl.gz','rt'):
        d=json.loads(l)
        A=[d['arms'][f'log:{k}'] for k in range(24)]
        n=sum(len(a['logs']) for a in A); fr=sum(1 for a in A for c in a['logs'] if c['active']['q_free'])/n
        if fr<0.9 or d['v0']<1.0: continue
        sw=A[15:]; ret=[round(a['ret'],2) for a in sw]; hard=[any(c['hard'] for c in a['logs']) for a in sw]
        slow=max(r for r,m in zip(ret,M) if m<=0.7); hold=ret[M.index(1.0)]
        # does the sweep even change v_target? (limit cap)
        vt=[min(d['limit'],m*max(d['v0'],1.0)) for m in M]
        cap=len(set(round(x,2) for x in vt))
        print(f"{run} state {d['idx']} v0={d['v0']:.1f} lim={d['limit']:.1f} free={fr:.2f} distinct_vtargets={cap}/9 sweep_ret={ret} hard={''.join('H' if h else '.' for h in hard)}")
        print(f"   best slow(m<=0.7)={slow} hold(1.0)={hold} -> slow>=hold: {slow>=hold}; mean v last 8 cycles slow/hold: {sum(c['v'] for c in sw[0]['logs'][-8:])/8:.2f}/{sum(c['v'] for c in sw[5]['logs'][-8:])/8:.2f}")
        print("   loss by term (sum prog*qprod/q*(1-q)) sweep0.5:",loss_by_term(sw[0]['logs'])," sweep1.0:",loss_by_term(sw[5]['logs']))
        print("   first-4-cycle reward sweep0.5:",[round(c['reward'],3) for c in sw[0]['logs'][:4]]," sweep1.0:",[round(c['reward'],3) for c in sw[5]['logs'][:4]],"progress_legacy sweep0.5/1.0:",[round(c['progress_legacy'],2) for c in sw[0]['logs'][:4]],[round(c['progress_legacy'],2) for c in sw[5]['logs'][:4]])
        reg=d['reg'].get('slowest:log'); 
        if reg: print("   always-slowest policy ret=%.2f vs PDM-best ret=%.2f; slowest v by cycle: %s"%(reg['ret'],d['arms'][f"log:{d['pdm_best']}"]['ret'],[round(c['v'],1) for c in reg['logs']][:12]))

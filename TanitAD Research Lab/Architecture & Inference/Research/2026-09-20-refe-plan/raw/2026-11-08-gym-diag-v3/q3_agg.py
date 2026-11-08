"""Aggregate q3_const outputs (F:/tanitad_scratch/mgym3/diag/q3_*.json): constant-target-speed policies on free-road states."""
import json,glob,sys
D=sys.argv[1]
for f in sorted(glob.glob(D+'/q3_*.json')):
    r=json.load(open(f)); print(f"\n== {r['sampler']} state {r['state']} limit={r['limit']:.1f} v0={r['v0']:.2f} PDM-best-arm return={r['pdm_best']:.2f}")
    print(" target        vt   return  hard  mean_v  mean_q_free(applicable)  q_stall_fired  loss_free  loss_other  mean_prog")
    for k,x in r['runs'].items():
        L=x['logs']; n=len(L)
        mv=sum(c['v'] for c in L)/n; app=[c for c in L if c['act']['q_free']]
        qf=sum(c['q']['q_free'] for c in app)/max(len(app),1)
        qs=sum(1 for c in L if c['q']['q_stall']<1)
        lf=sum(c['prog']*(c['qprod']/c['q']['q_free'])*(1-c['q']['q_free']) for c in L if c['q']['q_free']<1 and c['qprod']>0)
        lo=sum(c['prog']*(1-c['qprod']) for c in L)-lf
        h=[c for c in L if c['hard']]
        print(f" {k:9s} {x['vt']:7.2f} {x['ret']:8.3f} {'H' if h else '.':>4} {mv:7.2f} {qf:8.2f} ({len(app)}/{n}) {qs:10d} {lf:10.2f} {lo:10.2f} {sum(c['prog'] for c in L)/n:9.3f}")
